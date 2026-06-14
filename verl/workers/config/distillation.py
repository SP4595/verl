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
# =============================================================================
# On-Policy 蒸馏相关配置数据类
#
# 本模块定义三个层次的配置类，直接对应 run_qwen3_8b_fsdp.sh 中 EXTRA 数组
# 的 distillation.* 配置路径:
#
#   shell 参数路径                                    → Python 配置类字段
#   ─────────────────────────────────────────────────────────────────────
#   distillation.enabled=True                        → DistillationConfig.enabled
#   distillation.n_gpus_per_node=4                   → DistillationConfig.n_gpus_per_node
#   distillation.nnodes=1                            → DistillationConfig.nnodes
#   distillation.teacher_models.teacher_model.*      → DistillationTeacherModelConfig.*
#   distillation.distillation_loss.loss_mode=k1      → DistillationLossConfig.loss_mode
#   distillation.distillation_loss.topk=64           → DistillationLossConfig.topk
#   distillation.distillation_loss.use_task_rewards  → DistillationLossConfig.use_task_rewards
#   distillation.distillation_loss.use_policy_gradient → DistillationLossConfig.use_policy_gradient
#   distillation.distillation_loss.loss_max_clamp    → DistillationLossConfig.loss_max_clamp
#   distillation.distillation_loss.log_prob_min_clamp → DistillationLossConfig.log_prob_min_clamp
#
# 配置类实例化发生在 TaskRunner.run() 调用 validate_config() 之前，
# 由 verl/utils/config.py 中的 omega_conf_to_dataclass() 将 OmegaConf
# DictConfig 转换为类型化的 Python dataclass 实例。
# =============================================================================

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    # 对应 run_qwen3_8b_fsdp.sh: DISTILLATION_LOSS_MODE=k1
    # 可选: "kl"/"k1"/"abs"/"mse"/"k2"/"k3"/"low_var_kl"（estimator 类）
    #        或 "forward_kl_topk"（top-k 类，需同时设置 topk 和教师 max_logprobs）
    loss_mode: str = "k3"
    # 对应 run_qwen3_8b_fsdp.sh: DISTILLATION_TOPK=64
    # 仅在 loss_mode=forward_kl_topk 时使用；estimator 模式下（如 k1）忽略
    topk: Optional[int] = 128
    # 对应 run_qwen3_8b_fsdp.sh: use_task_rewards=False
    # False 时: 纯蒸馏模式，不使用任务奖励的策略梯度损失（policy_loss = 0）
    # True  时: 任务奖励与蒸馏损失以 distillation_loss_coef 为权重线性叠加
    use_task_rewards: bool = True
    # 蒸馏损失权重，仅 use_task_rewards=True 时生效
    distillation_loss_coef: float = 1.0
    # 对应 run_qwen3_8b_fsdp.sh: loss_max_clamp=10.0
    # 双向截断蒸馏损失：[-10, 10]，防止梯度爆炸
    # k1 损失可为负（学生比教师更确信该 token），故需双向截断
    loss_max_clamp: Optional[float] = 10.0
    # 对应 run_qwen3_8b_fsdp.sh: log_prob_min_clamp=-10.0
    # 截断 log-prob 下界，避免极小概率 token（log p → -∞）导致数值不稳定
    log_prob_min_clamp: Optional[float] = -10.0

    # 对应 run_qwen3_8b_fsdp.sh: use_policy_gradient=True
    # True : 将 -KL 作为 advantage 送入策略梯度（OPD 方案，推荐与 k1 配合使用）
    # False: 直接对蒸馏损失做监督反向传播（SFT 蒸馏，推荐与 k3/forward_kl_topk 配合）
    use_policy_gradient: bool = True
    # 策略梯度损失函数类型，当前只支持 "vanilla"（标准 PPO clip 损失）
    policy_loss_mode: str = "vanilla"
    # PPO 裁剪比率，仅 use_policy_gradient=True 时生效
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        # 根据 loss_mode 查找注册表，填充运行时设置（use_topk/use_estimator）
        # 这决定了教师模型推理时是否需要请求 top-k logprobs
        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        # 警告: forward_kl_topk 与 use_policy_gradient=True 一起使用效果较差
        # 因为策略梯度只用到采样 token 的 ∇logπ(a)，top-k 的分布信息被浪费
        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        # k1 损失关于模型参数的梯度不依赖教师 log-prob，因此不能直接监督反向传播
        # run_qwen3_8b_fsdp.sh 中 use_policy_gradient=True，故不触发此错误
        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    # 多教师蒸馏时用于路由样本的标识符（对应 data_source 字段）
    # 单教师模式（run_qwen3_8b_fsdp.sh）下由 _resolve_teacher_models() 自动填充
    key: Optional[str] = None
    # 对应 run_qwen3_8b_fsdp.sh: distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-32B
    model_path: Optional[str] = None
    # 教师模型推理配置（vLLM 参数，如 tp size、gpu_memory_utilization、max_model_len）
    # 对应 run_qwen3_8b_fsdp.sh EXTRA 数组中所有 teacher_model.inference.* 字段
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    # 教师模型副本数，与 per_replica_world_size 的乘积必须等于 n_gpus_per_node * nnodes
    # run_qwen3_8b_fsdp.sh: teacher_tp=2, TEACHER_WORLD_SIZE=4 → num_replicas=2, per_replica=2
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    # 对应 run_qwen3_8b_fsdp.sh: distillation.enabled=True
    # False 时所有蒸馏相关逻辑均被跳过
    enabled: bool = False
    # 对应 run_qwen3_8b_fsdp.sh: distillation.n_gpus_per_node=TEACHER_WORLD_SIZE=4
    # 指教师 GPU 资源池每节点的 GPU 数
    n_gpus_per_node: int = 0
    # 对应 run_qwen3_8b_fsdp.sh: distillation.nnodes=NNODES=1
    nnodes: int = 0
    # 对应 run_qwen3_8b_fsdp.sh: distillation.teacher_models.teacher_model.*
    # 多教师模式下可添加多个教师（如 teacher_model1、teacher_model2）
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    # 多教师蒸馏时，用于从样本中读取路由字段名
    teacher_key: str = "data_source"
    # 蒸馏损失配置，对应 run_qwen3_8b_fsdp.sh: distillation.distillation_loss.*
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        # 将 OmegaConf DictConfig 子字典解析为 DistillationTeacherModelConfig 实例列表
        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            # 根据 loss_settings.use_topk 决定教师是否需要返回 top-k logprobs
            # 同时将 inference.prompt_length 调整为 prompt+response，因为教师需要看到完整上下文
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
            teacher_world_size_sum += teacher_model.world_size
        # 校验: 所有教师副本占用的 GPU 总数必须等于 teacher_pool 大小
        # run_qwen3_8b_fsdp.sh: teacher_tp=2, TEACHER_WORLD_SIZE=4 →
        #   num_replicas=2 (4/2), per_replica_world_size=2 (tp=2) → world_size=4 = pool_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
