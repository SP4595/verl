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
        super().__init__(*args, **kwargs)
        self._validate_memory_opd_config()

    def _validate_memory_opd_config(self) -> None:
        """尽早拒绝会把纯 OPD 重新变成 PPO/RL 或破坏状态语义的配置。

        这些检查不是性能优化，而是算法边界。比如 ``rollout.n > 1`` 会让 VeRL 在不了解
        Memory 环境的情况下隐式复制 step；它无法决定哪个分支应写回长期 memory。
        """

        if not is_distillation_enabled(self.config.get("distillation")):
            raise ValueError("RayPrivilegeOPDTrainer 要求 distillation.enabled=True")
        if self.config.distillation.distillation_loss.use_task_rewards:
            raise ValueError(
                "Memory-OPD 当前不使用任务 reward；请设置 "
                "distillation.distillation_loss.use_task_rewards=False"
            )
        if need_reward_model(self.config):
            raise ValueError("Memory-OPD 当前不使用 Reward Model；请关闭 reward.reward_model")
        if need_critic(self.config):
            raise ValueError("Memory-OPD 当前不使用 Critic；请使用无需 critic 的配置")
        if need_reference_policy(self.config):
            raise ValueError("Memory-OPD 当前不使用 reference policy 或 reward KL")
        if self.config.actor_rollout_ref.rollout.n != 1:
            raise ValueError(
                "状态化 Memory episode 当前要求 rollout.n=1；同状态多分支采样需要由 "
                "Episode Collector 显式管理，不能让 Trainer 隐式 repeat"
            )
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

        if "memory_episode" in batch.non_tensor_batch:
            raise ValueError(
                "RayPrivilegeOPDTrainer 不能直接消费 memory_episode。请先由 Agentic "
                "Episode Collector 将 episode 展开为包含 memory_step 的 single-turn 样本。"
            )
        if "memory_step" not in batch.non_tensor_batch:
            raise KeyError("Memory-OPD 训练样本必须包含 memory_step")
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

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Memory-OPD")
        self.global_steps += 1

        use_policy_gradient = self.config.distillation.distillation_loss.use_policy_gradient
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics = {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
                batch = DataProto.from_single_dict(batch_dict)
                # 每个展开后的 step 是一个独立蒸馏样本。这里的 uid 用于 VeRL 内部追踪，
                # 不代表 episode 身份；episode_id 仍保存在 memory_step/extra_info 中。
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info.update(
                    {
                        "global_steps": self.global_steps,
                        "validate": False,
                    }
                )
                gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
                # generate_sequences 已完成 student rollout 和 teacher log-prob 计算。
                # 后续训练循环不再执行 Memory action，也不改变 Collector 环境。
                self.checkpoint_manager.sleep_replicas()
                metrics.update(gen_output.meta_info.pop("timing", {}))
                batch = batch.union(gen_output)

                if "response_mask" not in batch.batch:
                    # 蒸馏 loss 只应覆盖 student 生成的 response token，不能训练动态 prompt。
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"],
                    dim=-1,
                ).tolist()
                batch.meta_info["images_seqlens"] = []

                # OPD 的 policy-gradient 形式只需要 old_log_probs，不需要 task advantage。
                if use_policy_gradient:
                    # 这是 OPD loss 自身可选的 estimator 输入，不是 PPO task advantage。
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    old_log_prob.batch.pop("entropys", None)
                    batch = batch.union(old_log_prob)
                    metrics["perf/mfu/actor_infer"] = old_log_prob_mfu

                actor_output = self._update_actor(batch)
                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                should_save = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                )
                if should_save:
                    self._save_checkpoint()
                self.checkpoint_manager.update_weights(self.global_steps)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    self._shutdown_dump_executor()
                    pprint("Memory-OPD training finished.")
                    return

        progress_bar.close()
        self._shutdown_dump_executor()


# 保留旧名称，避免已有入口脚本立即失效。
RayOPDTrainer = RayPrivilegeOPDTrainer


__all__ = ["RayOPDTrainer", "RayPrivilegeOPDTrainer"]
