"""Memory-OPD 使用的纯蒸馏 Ray Trainer 入口。

这里不再复制整份 ``RayPPOTrainer``。Actor、teacher、checkpoint、分布式 worker 和
基础训练循环继续复用 VeRL；本类只声明 Memory-OPD 的约束，并阻止把完整 episode
误当成一个训练 prompt。

数据流必须是：

``LoCoMo episode source -> Agentic Episode Collector -> MemoryOPDStep -> 本 Trainer``

Reward、critic、reference policy 和任务 advantage 不属于当前纯 OPD 阶段。配置应关闭
这些组件，并设置 ``distillation.distillation_loss.use_task_rewards=False``。

这里接收的是已经冻结并展开的 ``memory_step``，不是在线可变的 Memory 环境。Trainer
只关心 token sequence 和蒸馏 loss；跨 step 状态推进必须在 Episode Collector 中完成。
因此当前实现是“离线/缓冲区式 step OPD”，不是 StepGRPO，也不会从一个状态采样多个
动作分支。
"""

from __future__ import annotations

import uuid
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.utils import need_critic, need_reference_policy, need_reward_model
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class RayPrivilegeOPDTrainer(RayPPOTrainer):
    """只接受已经展开的 single-turn Memory-OPD step。

    当前类有意保持为 VeRL Trainer 的薄封装。与 Memory 状态相关的 session/QA 编排、
    query 执行、update 写回和 episode 展开属于 Agentic Collector，而不是 Trainer。

    一次训练迭代只完成：

    ``memory_step -> student rollout -> teacher log-prob -> actor distillation update``。
    """

    def __init__(self, *args, **kwargs):
        # 步骤 1：复用 RayPPOTrainer 初始化 DataLoader、worker、rollout manager 和 checkpoint。
        super().__init__(*args, **kwargs)
        # 步骤 2：在训练启动前验证当前配置确实满足纯 Memory-OPD 算法边界。
        self._validate_memory_opd_config()

    def _validate_memory_opd_config(self) -> None:
        """尽早拒绝会把纯 OPD 重新变成 PPO/RL 或破坏状态语义的配置。

        这些检查不是性能优化，而是算法边界。比如 ``rollout.n > 1`` 会让 VeRL 在不了解
        Memory 环境的情况下隐式复制 step；它无法决定哪个分支应写回长期 memory。
        """

        # 步骤 1：纯 OPD 必须启用 teacher distillation。
        if not is_distillation_enabled(self.config.get("distillation")):
            raise ValueError("RayPrivilegeOPDTrainer 要求 distillation.enabled=True")
        # 步骤 2：禁止把任务 reward 混入当前蒸馏 loss。
        if self.config.distillation.distillation_loss.use_task_rewards:
            raise ValueError(
                "Memory-OPD 当前不使用任务 reward；请设置 "
                "distillation.distillation_loss.use_task_rewards=False"
            )
        # 步骤 3：禁止启动神经 Reward Model、Critic 和 reference-policy KL。
        if need_reward_model(self.config):
            raise ValueError("Memory-OPD 当前不使用 Reward Model；请关闭 reward.reward_model")
        if need_critic(self.config):
            raise ValueError("Memory-OPD 当前不使用 Critic；请使用无需 critic 的配置")
        if need_reference_policy(self.config):
            raise ValueError("Memory-OPD 当前不使用 reference policy 或 reward KL")
        # 步骤 4：禁止 Trainer 隐式复制状态；多分支必须由状态化 Collector 管理。
        if self.config.actor_rollout_ref.rollout.n != 1:
            raise ValueError(
                "状态化 Memory episode 当前要求 rollout.n=1；同状态多分支采样需要由 "
                "Episode Collector 显式管理，不能让 Trainer 隐式 repeat"
            )
        # 步骤 5：禁止依赖任务 reward 的默认 validation 路径。
        test_freq = self.config.trainer.get("test_freq", -1)
        if self.config.trainer.get("val_before_train", False) or (test_freq is not None and test_freq > 0):
            raise ValueError(
                "当前纯 OPD Trainer 不执行基于 reward 的 validation；请设置 "
                "trainer.val_before_train=False 和 trainer.test_freq=-1"
            )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """检查 Trainer 收到的是 step，而不是尚未展开的 episode。

        ``memory_episode`` 必须先由 Collector 顺序执行，因为只有 Collector 拥有活跃
        Memory 状态。``memory_step`` 已经是可被 single-turn AgentLoop 消费的冻结快照。
        """

        # 步骤 1：拒绝尚未由 Collector 展开的完整 episode。
        if "memory_episode" in batch.non_tensor_batch:
            raise ValueError(
                "RayPrivilegeOPDTrainer 不能直接消费 memory_episode。请先由 Agentic "
                "Episode Collector 将 episode 展开为包含 memory_step 的 single-turn 样本。"
            )
        # 步骤 2：确认每条样本携带 single-turn AgentLoop 实际消费的冻结 step。
        if "memory_step" not in batch.non_tensor_batch:
            raise KeyError("Memory-OPD 训练样本必须包含 memory_step")
        # 步骤 3：复用父类逻辑移除 rollout 不需要的 tensor 字段并生成 gen batch。
        return super()._get_gen_batch(batch)

    def fit(self):
        """执行纯 OPD：rollout -> teacher logprob -> actor update。

        与默认 PPO loop 相比，这里有意不计算 reward、task advantage、critic value、
        reference-policy KL 或 reward validation。``use_policy_gradient=True`` 时只额外
        重算一次 student old logprob，供 OPD 自身的 policy-gradient estimator 使用。

        Teacher log-prob 不在此函数显式计算：``async_rollout_manager.generate_sequences``
        调用自定义 AgentLoopWorker，并由 Worker 在 rollout 后处理中附加
        ``teacher_ids``/``teacher_logprobs``。
        """

        # 步骤 1：初始化实验日志；完整解析后的配置一并记录，便于复现实验。
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 步骤 2：恢复 checkpoint，并把当前 actor 权重同步到 rollout replicas。
        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        # 步骤 3：根据恢复后的 global step 计算起始 epoch，并初始化进度条。
        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Memory-OPD")
        self.global_steps += 1

        # 步骤 4：读取 OPD loss 是否需要额外重算 student old log-prob。
        use_policy_gradient = self.config.distillation.distillation_loss.use_policy_gradient
        # 步骤 5：按 epoch 和 DataLoader batch 进入纯 OPD 训练循环。
        
        # Note： 这里面每个 batch可能会被切分成无数个single turn batch。
        
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 步骤 6：计算终止条件，并为当前训练 step 初始化日志指标。
                is_last_step = self.global_steps >= self.total_training_steps
                metrics = {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
                # 步骤 7：把 collate_fn 输出转换为 VeRL DataProto。
                batch = DataProto.from_single_dict(batch_dict)
                # 每个展开后的 step 是一个独立蒸馏样本。这里的 uid 用于 VeRL 内部追踪，
                # 不代表 episode 身份；episode_id 仍保存在 memory_step/extra_info 中。
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                # 步骤 8：把 rollout temperature 写入元数据，供后续 loss/指标使用。
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # 步骤 9：验证并提取 generation batch，标记当前不是 validation。
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info.update(
                    {
                        "global_steps": self.global_steps,
                        "validate": False,
                    }
                )
                # 步骤 10：执行 student rollout；自定义 Worker 同时计算 teacher log-prob。
                gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
                # generate_sequences 已完成 student rollout 和 teacher log-prob 计算。
                # 后续训练循环不再执行 Memory action，也不改变 Collector 环境。
                # 步骤 11：暂停 rollout replicas 释放资源，并合并生成输出与原始 step 数据。
                self.checkpoint_manager.sleep_replicas()
                metrics.update(gen_output.meta_info.pop("timing", {}))
                batch = batch.union(gen_output)

                # 步骤 12：构造 response mask，并按配置平衡各 worker 的有效 token 数。
                if "response_mask" not in batch.batch:
                    # 蒸馏 loss 只应覆盖 student 生成的 response token，不能训练动态 prompt。
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                # 步骤 13：记录每条序列 token 数；纯文本 Memory-OPD 没有图片序列。
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"],
                    dim=-1,
                ).tolist()
                batch.meta_info["images_seqlens"] = []

                # OPD 的 policy-gradient 形式只需要 old_log_probs，不需要 task advantage。
                # 步骤 14：可选计算 OPD policy-gradient estimator 所需 old log-prob。
                if use_policy_gradient:
                    # 这是 OPD loss 自身可选的 estimator 输入，不是 PPO task advantage。
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    old_log_prob.batch.pop("entropys", None)
                    batch = batch.union(old_log_prob)
                    metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

                # 步骤 15：用 teacher log-prob 和 response mask 更新 actor 参数。
                actor_output = self._update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                # 步骤 16：按保存频率持久化 checkpoint，并同步新 actor 权重到 replicas。
                should_save = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                )
                if should_save:
                    self._save_checkpoint()
                self.checkpoint_manager.update_weights(self.global_steps)

                # 步骤 17：写入日志、推进进度，并通知 Dataset 当前 batch 已训练完成。
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
                self.global_steps += 1

                # 步骤 18：达到总训练步数时关闭资源并结束训练。
                if is_last_step:
                    progress_bar.close()
                    self._shutdown_dump_executor()
                    pprint("Memory-OPD training finished.")
                    return

        # 步骤 19：所有 epoch 自然结束时同样关闭进度条和异步 dump executor。
        progress_bar.close()
        self._shutdown_dump_executor()


# 保留旧名称，避免已有入口脚本立即失效。
RayOPDTrainer = RayPrivilegeOPDTrainer


__all__ = ["RayOPDTrainer", "RayPrivilegeOPDTrainer"]
