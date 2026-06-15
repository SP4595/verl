# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# 蒸馏损失函数注册表与核心计算逻辑
#
# 本模块是 run_qwen3_8b_fsdp.sh 中 distillation.distillation_loss.loss_mode 参数的
# 最终实现所在。脚本设置了 DISTILLATION_LOSS_MODE=k1，对应本模块中注册的
# compute_distillation_loss_reverse_kl_estimator 函数（names=["kl","k1","abs","mse","k2","low_var_kl","k3"]）。
#
# 调用链（从 ActorRolloutRefWorker 的训练步骤出发）:
#   ActorRolloutRefWorker._compute_loss()
#     └─ distillation_ppo_loss()         # 本模块的总入口
#          ├─ compute_topk_loss()        # [logit processor 阶段] 计算 token 级 KL
#          └─ distillation_loss()        # [最终 loss 阶段] 聚合并与策略梯度结合
#               └─ compute_distillation_loss_reverse_kl_estimator()  # k1 模式
#                  或 compute_forward_kl_topk()                       # forward_kl_topk 模式
#
# 蒸馏信号流:
#   教师模型 (Qwen3-32B) 生成每个 token 的 log-prob
#     └─ 存入 TensorDict["teacher_logprobs"]
#          └─ 与学生 log-prob 计算 KL 散度
#               └─ 以负 KL 作为 advantage 送入策略梯度（use_policy_gradient=True 时）
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """检查是否启用蒸馏训练。

    run_qwen3_8b_fsdp.sh 通过 distillation.enabled=True 启用蒸馏。
    在 TaskRunner.init_resource_pool_mgr() 和 add_teacher_model_resource_pool() 中
    均调用此函数决定是否分配 teacher_pool GPU 资源池。
    """
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    蒸馏损失函数注册时所需的元数据设置。

    每个损失函数在注册（@register_distillation_loss）时必须声明自己属于哪种类型:
      - use_topk=True   : 使用 top-k log-prob（如 forward_kl_topk），需要教师返回 top-k token ID 和概率
      - use_estimator=True: 使用单样本 KL 估计器（如 k1/k3），只需教师返回采样 token 的 log-prob

    run_qwen3_8b_fsdp.sh 使用 loss_mode=k1，对应 use_estimator=True 的注册条目。
    这影响教师模型推理时是否需要请求 top-k logprobs（DistillationTeacherModelConfig._validate_topk_logprobs）。

    Args:
        names (str | list[str]): 注册的损失函数名称列表。
        use_topk (bool): 是否使用 top-k log 概率（影响教师推理时 vLLM max_logprobs 参数）。
        use_estimator (bool): 是否使用单样本 KL 估计器（只需教师返回采样 token 的概率）。
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


# 全局损失函数注册表：loss_mode 字符串 → 损失函数
# run_qwen3_8b_fsdp.sh 设置 loss_mode=k1 时，
# get_distillation_loss_fn("k1") 从此注册表取出对应函数
DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
# 全局损失设置注册表：loss_mode 字符串 → DistillationLossSettings
# 用于决定教师推理时是否需要 top-k logprobs
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """将蒸馏损失函数注册到全局注册表的装饰器。

    使用方式示例（已在本文件末尾注册）:
      @register_distillation_loss(DistillationLossSettings(names=["k1", "kl"], use_estimator=True))
      def compute_distillation_loss_reverse_kl_estimator(...):
          ...

    run_qwen3_8b_fsdp.sh 中 loss_mode=k1 对应注册名称中包含 "k1" 的函数。
    """

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """在 logit processor 阶段计算 token 级别的 top-k KL 蒸馏损失。

    此函数在模型前向传播期间被 logit processor 钩子调用，
    此时学生 logits 尚未解码为 token，可以高效地计算与教师分布的 KL 散度。

    根据 actor.strategy 路由到不同后端实现:
      - "fsdp" / "veomni": 调用 verl/trainer/distillation/fsdp/losses.py
      - "megatron"       : 调用 verl/trainer/distillation/megatron/losses.py

    run_qwen3_8b_fsdp.sh 使用 FSDP 策略，因此路由到 fsdp/losses.py 的实现。

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)，每个 token 位置的 KL 散度
    - student_mass: (bsz, seqlen/cp_size)，学生模型在教师 top-k token 上的概率质量
    - teacher_mass: (bsz, seqlen/cp_size)，教师模型在 top-k token 上的概率质量
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """蒸馏训练的统一损失入口，同时服务于两个不同的调用阶段。

    此函数作为 ActorRolloutRefWorker 的 logit_processor 钩子和最终 loss 计算器，
    通过 student_logits 是否为 None 来区分调用阶段:

    阶段 1 —— logit processor（student_logits 非 None）:
      在模型前向传播时被触发，此时 logits 尚未被解码。
      计算每个 token 的 top-k KL 损失并缓存到 model_output 中供阶段 2 使用。
      返回: token 级 KL 张量 (bsz, seqlen/cp_size)

    阶段 2 —— 最终 loss（student_logits 为 None）:
      在前向传播结束后调用，合并 distillation loss 与 PPO policy loss。
      对于 run_qwen3_8b_fsdp.sh（use_task_rewards=False, use_policy_gradient=True）:
        - 不使用 PPO 任务奖励（policy_loss=0）
        - 将 -KL 作为 advantage 送入策略梯度更新学生模型
      返回: (标量 loss, metrics dict)

    整体 loss 计算流程:
      [split sequence across sp/cp groups]
                     |
      [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                     |
      [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                     |
      [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                     |
      [combine topk loss with policy loss]

    Args:
        config: Actor 配置，包含 strategy（"fsdp"）等字段。
        distillation_config: 蒸馏配置，包含 loss_mode=k1、use_policy_gradient=True 等。
        model_output: 模型输出，包含 log_probs、entropy 和缓存的蒸馏损失张量。
        data: Micro-batch TensorDict，包含:
          - teacher_logprobs: (bsz, seqlen, topk) 教师 token log-prob
          - teacher_ids: (bsz, seqlen, topk) 教师 top-k token ID
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size)，仅阶段 1 提供。
        data_format: "thd" 或 "bshd"（Qwen3.5 等不支持 THD 格式的模型使用 "bshd"）

    Returns:
    - 阶段 1: token 级 KL 张量 (bsz, seqlen/cp_size)
    - 阶段 2: (标量 loss, metrics 字典)
    """

    # ---- 阶段 1: logit processor 调用（student_logits 非 None）----
    # 在模型前向传播时被触发，计算并返回 token 级 KL 损失
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # ---- 阶段 2: 最终 loss 计算（student_logits 为 None）----
    distillation_loss_config = distillation_config.distillation_loss
    # 计算聚合蒸馏损失标量
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    # 只有混合任务 reward 时才计算 PPO loss。纯 OPD batch 不包含 advantages，
    # 无条件调用 ppo_loss 不仅浪费计算，还会错误要求 reward/advantage 字段存在。
    if distillation_loss_config.use_task_rewards:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    else:
        policy_loss, policy_metrics = 0.0, {}

    # 合并蒸馏损失与策略损失
    policy_metrics.update(distill_metrics)
    # use_task_rewards=False 时 distillation_loss_coef 强制为 1.0
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    从注册表中取出对应 loss_mode 的损失函数并计算蒸馏损失。

    run_qwen3_8b_fsdp.sh 配置:
      loss_mode=k1, use_policy_gradient=True, use_task_rewards=False

    k1 模式下的损失计算逻辑:
      1. 调用 compute_distillation_loss_reverse_kl_estimator 计算每 token 的 k1 估计值
         k1(t) = log(p_teacher(t) / p_student(t)) = log_p_teacher(t) - log_p_student(t)
         （k1 是 KL 散度的单样本无偏估计，可为负值）
      2. 对损失进行 [-loss_max_clamp, loss_max_clamp] 截断（防止梯度爆炸）
      3. use_policy_gradient=True 时：
         - 将 -k1(t) 作为每 token 的 advantage（负 KL 越小 = 学生越接近教师 = 越好）
         - 通过 policy_loss_fn（vanilla PPO）用策略梯度更新学生
      4. use_policy_gradient=False 时：
         - 直接对 k1 损失做监督反向传播（SFT 蒸馏）

    Returns:
    - distillation_loss: 聚合后的蒸馏损失标量。
    - distillation_metrics: 包含 loss_min/max、abs_loss 等监控指标的字典。
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    # 从注册表中取出 loss_mode=k1 对应的损失函数
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    # 调用损失函数，得到每 token 的损失张量和中间监控指标
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    # 记录损失在响应 token 范围内的最小/最大值（用于监控蒸馏稳定性）
    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # 对应 run_qwen3_8b_fsdp.sh: loss_max_clamp=10.0
        # k1 损失可为负值（学生比教师更确信该 token），因此双向截断
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # ---- use_policy_gradient=True 分支（run_qwen3_8b_fsdp.sh 的选择）----
        # 将负 KL 散度作为 token 级 advantage 送入策略梯度
        # 原理参考: https://thinkingmachines.ai/blog/on-policy-distillation/
        # advantage = -KL(student || teacher)，KL 越小说明学生越接近教师，advantage 越大
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),  # 用 detach 防止梯度通过 advantage 流动
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        # 将指标从 "actor/" 前缀重命名为 "distillation/" 便于 WandB 区分
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # ---- use_policy_gradient=False 分支 ----
        # 直接对蒸馏损失做监督反向传播（类 SFT），参考 https://arxiv.org/abs/2306.13649
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """使用 top-k log-prob 计算前向 KL 蒸馏损失（已在 logit processor 阶段预计算）。

    loss_mode="forward_kl_topk" 时激活此函数（run_qwen3_8b_fsdp.sh 默认不使用，
    默认为 k1；切换时需同时设置 use_policy_gradient=False）。

    前向 KL: KL(teacher || student) = sum_k [ p_teacher(k) * log(p_teacher(k)/p_student(k)) ]
      - 教师分布主导，student 会覆盖教师分布的全部支撑（mean-seeking 行为）
      - 需要教师返回 top-k token 及对应概率，教师推理时 max_logprobs 必须 >= topk

    注意: 实际的 KL 张量已在 logit processor 阶段（compute_forward_kl_topk in fsdp/losses.py）
    计算并缓存到 model_output["distillation_losses"] 中，此函数只做解包和指标计算。

    Returns:
    - distillation_losses: (bsz, resp_len)，每 token 的前向 KL 值（已截断为非负）
    - distillation_metrics: 包含 student_mass、teacher_mass 和 overlap 指标的字典
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    使用单样本 KL 估计器计算蒸馏损失（对应 k1/k3 等 estimator 类损失模式）。

    run_qwen3_8b_fsdp.sh 使用 loss_mode=k1，此函数是最终执行者。

    各 estimator 含义（来自 verl/trainer/ppo/core_algos.py:kl_penalty）:
      - "kl"       : KL(student || teacher) = exp(log_s - log_t) - (log_s - log_t) - 1（标准 KL 估计）
      - "k1"       : log_p_teacher - log_p_student（无偏单样本 KL 估计，可为负）
      - "k3"       : max((log_p_teacher - log_p_student), 0)（截断 k1，始终非负）
      - "abs"      : |log_p_teacher - log_p_student|
      - "mse"      : (log_p_teacher - log_p_student)^2
      - "low_var_kl": k1 + 0.5 * (log_p_teacher - log_p_student)^2（低方差估计）

    k1 的优势: 梯度 ∇_θ k1 = -∇_θ log p_student，等价于以 log_p_teacher 为目标的
    最大似然梯度，当 use_policy_gradient=True 时可直接作为 advantage 使用。

    只需教师返回采样 token 的单点 log-prob（不需要 top-k），教师推理时无需设置
    max_logprobs，通信和显存开销更低。

    Returns:
    - distillation_losses: (bsz, resp_len)，每 token 的 KL 估计值（k1 可为负）
    - distillation_metrics: 包含 abs_loss 的监控指标字典
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
