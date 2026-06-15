# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.

本模块是 run_qwen3_8b_fsdp.sh 调用的 Python 入口，负责:
  1. 通过 Hydra 解析命令行中的所有 key=value override（DATA / MODEL / ACTOR / ROLLOUT / TRAINER / EXTRA）
  2. 初始化 Ray 分布式运行时，划分 global_pool（学生）和 teacher_pool（教师）两个 GPU 资源池
  3. 构建 RayPPOTrainer，完成 Actor/Rollout/Ref Worker 以及 Teacher Worker 的初始化
  4. 调用 trainer.fit() 启动迭代训练循环

整体调用链:
  run_qwen3_8b_fsdp.sh
    └─ python3 -m verl.trainer.main_ppo  (本文件 main())
         └─ run_ppo(config)
              └─ TaskRunner.run(config)
                   ├─ RayPPOTrainer.__init__()
                   ├─ RayPPOTrainer.init_workers()  # 在 Ray 集群上拉起各 Worker
                   └─ RayPPOTrainer.fit()           # 训练主循环
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import deprecated, load_class_from_fqn


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """PPO 训练主入口，由 Hydra 框架自动调用。

    Hydra 会先加载 config/ppo_trainer.yaml 作为默认配置，再将 shell 脚本传入的
    key=value override 覆盖对应字段，最终合并为一个 OmegaConf DictConfig 对象传入。

    Args:
        config: Hydra 合并后的配置对象，包含 run_qwen3_8b_fsdp.sh 中所有参数。
    """
    # 检测运行平台：若在昇腾 NPU 上运行，自动将 config.trainer.device 设为 "npu"
    # FSDP 脚本在 NVIDIA GPU 上运行时此调用为 no-op
    auto_set_device(config)
    # 将旧版奖励函数实现迁移到新接口（向后兼容处理）
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    """初始化 Ray 集群并启动分布式 PPO/蒸馏训练。

    run_qwen3_8b_fsdp.sh 通过 main() → run_ppo() 最终到达本函数。
    本函数负责：
      1. 启动（或连接已有的）Ray 集群，设置运行时环境变量
      2. 将 TaskRunner 作为 Ray remote 任务提交执行
      3. 阻塞等待 TaskRunner.run() 完成

    GPU 资源池划分（由 TaskRunner.init_resource_pool_mgr 完成）:
      - global_pool : NGPUS_PER_NODE × NNODES 块 GPU，供学生 Actor/Rollout/Ref 使用
      - teacher_pool: TEACHER_WORLD_SIZE × NNODES 块 GPU，供教师模型 vLLM 推理使用

    Args:
        config: 训练配置对象，包含 run_qwen3_8b_fsdp.sh 中所有参数。
        task_runner_class: 自定义 TaskRunner 子类（recipe 使用），默认为本文件的 TaskRunner。
    """
    # 若 Ray 尚未初始化（单进程启动时），创建本地集群
    if not ray.is_initialized():
        # get_ppo_ray_runtime_env() 返回预设的运行时环境变量，例如：
        #   TOKENIZERS_PARALLELISM=false（避免 HuggingFace 分词器死锁）
        #   NCCL_DEBUG=WARN（降低 NCCL 日志噪音）
        #   VLLM_LOGGING_LEVEL=WARNING
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # TransferQueue 用于异步传输模型权重（学生 → vLLM），启用时注入环境变量
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # 合并默认运行时环境与用户自定义覆盖
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        # 将 TaskRunner 包装成 Ray remote class，分配 1 个 CPU（轻量级调度节点，不占 GPU）
        # 注意：main_task 必须不在 head 节点上运行，以避免资源竞争
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    # 根据是否启用 nsys 性能分析来决定 runner 的启动方式
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        # 启用 nsys 性能分析（run_qwen3_8b_fsdp.sh 默认不开启）
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # 标准路径：创建 TaskRunner remote actor 并阻塞等待训练完成
        runner = task_runner_class.remote()
    # ray.get() 阻塞直到整个训练循环（trainer.fit()）结束
    ray.get(runner.run.remote(config))

    # 可选：将 Ray 分布式 timeline 追踪写入文件，用于性能分析（默认不配置）
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """以 Ray remote actor 形式运行的训练任务执行器。

    TaskRunner 运行在 Ray worker 节点上，负责:
      1. 注册各 Role（ActorRollout、Critic、Teacher 等）的 Worker 类和 GPU 映射
      2. 初始化 ResourcePoolManager，隔离学生/教师 GPU 资源池
      3. 创建数据集、tokenizer，实例化并启动 RayPPOTrainer

    run_qwen3_8b_fsdp.sh 的参数通过 config 传入，最终在 run() 方法中生效:
      - config.actor_rollout_ref.*  → add_actor_rollout_worker()
      - config.distillation.*       → add_teacher_model_resource_pool()
      - config.trainer.*            → RayPPOTrainer 调度

    Attributes:
        role_worker_mapping: Role 枚举 → Ray remote Worker 类的映射字典
        mapping: Role 枚举 → 资源池 ID（"global_pool" 或 "teacher_pool"）的映射字典
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """注册 ActorRollout Worker，对应 run_qwen3_8b_fsdp.sh 中的 actor_rollout_ref.* 配置。

        学生模型（Qwen3-8B）通过 ActorRolloutRefWorker 同时承担三个角色:
          - Actor  : 接收 advantage，计算策略梯度更新模型权重（FSDP 后端）
          - Rollout: 使用 vLLM 对 prompt 进行高效采样生成（rollout_tp=2）
          - Ref    : 计算参考策略的 log-prob（当无 LoRA 时与 Actor 合并为同一 Worker）
        """
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with a dedicated ref model.
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping using the unified model engine implementation."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        # The model-engine TrainingWorker handles all critic backends (fsdp/fsdp2/megatron/...)
        # internally based on ``config.critic.strategy``.
        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """初始化 GPU 资源池管理器，隔离学生模型和教师模型的 GPU 使用。

        资源池布局（对应 run_qwen3_8b_fsdp.sh 单节点配置）:
          global_pool  : [8] GPU（NGPUS_PER_NODE=8，供 Actor/Rollout/Ref 使用）
          teacher_pool : [4] GPU（TEACHER_WORLD_SIZE=4，供 Qwen3-32B vLLM 推理使用）
          共计 12 块 GPU，需确保物理节点有足够 GPU 资源

        ResourcePoolManager 负责在 Ray 集群中按池分配 GPU，防止学生/教师相互抢占显存。
        """

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            # we do not use reward model workers, so we only register reward model in resource pool
            # without continue to register reward model worker in role mapping
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        """将教师模型注册到独立的 teacher_pool GPU 资源池（不创建 Worker 类）。

        对应 run_qwen3_8b_fsdp.sh 中的 EXTRA 配置:
          distillation.enabled=True
          distillation.n_gpus_per_node=TEACHER_WORLD_SIZE（4）
          distillation.teacher_models.teacher_model.model_path=Qwen3-32B

        教师模型（Qwen3-32B）只进行推理，不更新权重，因此只需分配资源池，
        无需向 role_worker_mapping 注册实际的 Worker 类。
        实际推理由 RayPPOTrainer 内部的 TeacherModelWorker 调用 vLLM AsyncLLMEngine 完成。
        """
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            # we do not use teacher model workers, so we only register teacher model in resource pool
            # without registering a teacher model worker in role-worker mapping
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Ref policy is fused into ActorRolloutRefWorker in the unified model engine.

        Kept for backward compatibility with subclasses that still invoke it; the method
        is now a no-op because the reference policy lives on the same worker group as
        the actor/rollout.
        """
        return

    def run(self, config):
        """训练主流程：初始化所有组件并启动 RayPPOTrainer。

        本方法是 run_qwen3_8b_fsdp.sh 所有配置参数的最终执行点，按以下步骤运行:
          1. 打印完整解析后的配置（便于调试 Hydra override 是否生效）
          2. 注册各 Role 的 Worker 类（Actor/Rollout、Critic、Teacher）
          3. 初始化 ResourcePoolManager（划分 global_pool 和 teacher_pool）
          4. 将学生模型权重从 HDFS/本地复制到本机（如有需要）
          5. 加载 tokenizer 和 processor（多模态模型时 processor 非 None）
          6. 构建训练/验证数据集（RLHFDataset，读取 gsm8k + math 的 parquet 文件）
          7. 实例化 RayPPOTrainer 并调用 init_workers() 在 Ray 上拉起所有 Worker
          8. 调用 trainer.fit() 启动迭代训练

        Args:
            config: 训练配置对象，包含 run_qwen3_8b_fsdp.sh 中所有参数。
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # 打印解析后的完整配置，resolve=True 会把 ${xxx} 插值展开
        # 便于确认 run_qwen3_8b_fsdp.sh 中的所有 override 是否正确生效
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 注册 ActorRollout Worker（学生模型：Actor + vLLM Rollout + Ref 三合一）
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # 注册 Critic Worker（GRPO 模式下 Critic 可能为空，仍需注册以满足接口要求）
        self.add_critic_worker(config)

        # 注册奖励模型资源池（run_qwen3_8b_fsdp.sh 未启用 reward_model，跳过）
        self.add_reward_model_resource_pool(config)

        # 将 TeacherModel 注册到 teacher_pool（对应 distillation.enabled=True）
        # 教师模型（Qwen3-32B）在此处只划分 GPU 资源池，不创建 Worker 类
        self.add_teacher_model_resource_pool(config)

        # 当 use_kl_loss=True 或 use_kl_in_reward=True 时才需要独立 Ref Worker
        # run_qwen3_8b_fsdp.sh 中两者均为 False，此处为 no-op
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 校验配置一致性（例如 critic 是否必要、ref_policy 是否启用等）
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # 将学生模型权重从 HDFS 或远端存储下载到本地
        # use_shm=False 时直接写磁盘；use_shm=True 时使用共享内存加速加载
        # 对应 run_qwen3_8b_fsdp.sh: actor_rollout_ref.model.path=Qwen/Qwen3-8B
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        # 加载 HuggingFace tokenizer（用于 prompt/response token 化）
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # 多模态模型（如 VLM）需要 processor；纯文本模型此处返回 None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 划分 GPU 资源池：global_pool（8 GPU） + teacher_pool（4 GPU）
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 构建训练数据集
        # 对应 run_qwen3_8b_fsdp.sh: data.train_files=['gsm8k_train', 'math_train']
        # RLHFDataset 读取 parquet 文件，字段包括 prompt、data_source 等
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        # 构建验证数据集
        # 对应 run_qwen3_8b_fsdp.sh: data.val_files=['gsm8k_test', 'math_test']
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        # 创建采样器
        # data.shuffle=False 时使用 SequentialSampler，保证 checkpoint 可精确恢复
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 默认使用标准 PPO Trainer；Memory-OPD 等特殊数据流可以通过 FQN 选择薄封装，
        # 避免复制或继续修改通用 TaskRunner。
        trainer_class_fqn = config.trainer.get("trainer_class")
        trainer_cls = (
            load_class_from_fqn(trainer_class_fqn, "Ray trainer")
            if trainer_class_fqn
            else RayPPOTrainer
        )
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # 在 Ray 集群上拉起所有 Worker（Actor/Rollout/Ref + Teacher vLLM）
        trainer.init_workers()

        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """根据配置实例化训练或验证数据集。

    对应 run_qwen3_8b_fsdp.sh 中的数据文件配置:
      data.train_files=['$gsm8k_train', '$math_train']  → 训练集
      data.val_files=['$gsm8k_test', '$math_test']      → 验证集

    Parquet 文件每行应包含字段: prompt, data_source, reward_model.ground_truth 等。
    RLHFDataset 类会将 prompt token 化并填充/裁剪到 max_prompt_length=1024。

    Arguments:
        data_paths: 数据文件路径列表，支持单个或多个 parquet 路径。
        data_config: 数据配置，包含 max_prompt_length、max_response_length 等。
        tokenizer: HuggingFace tokenizer，用于 prompt 的 token 化。
        processor: 多模态处理器（纯文本模型为 None）。
        is_train: True 表示训练集，False 表示验证集。
        max_samples: 最大样本数，-1 表示无限制。

    Returns:
        dataset: 实例化的数据集对象（通常为 RLHFDataset）。
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config)

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """为数据集创建采样器，支持随机和顺序两种模式。

    run_qwen3_8b_fsdp.sh 中设置了 data.shuffle=False，因此使用 SequentialSampler:
      - SequentialSampler: 按文件顺序迭代，配合 checkpoint 可精确恢复训练进度
      - RandomSampler: 随机打乱顺序，可通过设置 seed 保证可重现性

    Arguments:
        data_config: 数据配置，包含 shuffle 和 seed 字段。
        dataset: 将要采样的数据集对象。

    Returns:
        sampler: 采样器对象（SequentialSampler 或 RandomSampler）。
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
