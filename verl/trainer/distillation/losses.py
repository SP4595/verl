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
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def is_distillation_actor_loss_enabled(config: Optional[DistillationConfig]) -> bool:
    """Whether the actor objective actually contains a distillation term.

    A teacher server may still be enabled for uses outside the actor objective (for
    example, a teacher-forced reward/potential scorer).  When task rewards are on
    and the distillation coefficient is exactly zero, selecting
    ``distillation_ppo_loss`` would only compute a teacher loss and multiply it by
    zero.  In that case the actor should use the ordinary PPO loss directly.
    """

    if not is_distillation_enabled(config):
        return False
    loss_config = config.distillation_loss
    return not (loss_config.use_task_rewards and float(loss_config.distillation_loss_coef) == 0.0)


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
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


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

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
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """

    # NOTE：这里才是真正算 loss 的地方（被第一次调用 = forward 内部的 logits processor 触发）。
    # 按 config.strategy 分发到 fsdp/megatron 后端的 compute_forward_kl_topk，拿原始 student_logits
    # + 老师 top-k 现算。★ 返回的是【每 token 张量的 dict】
    #   {distillation_losses, student_mass, teacher_mass, overlap_count, overlap_token_advantage}，
    #   不是标量 loss；这个 dict 会被上层 prepare_model_outputs 逐 key 存进 model_output，留给第二次调用聚合。

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
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

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
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # ===================== NOTE：本函数怎么被调用的？——被塞进 forward，一共调两次 =====================
    # 这个函数会被绑定成 loss_function 后「塞给 engine 的 forward」（见 engine_workers.py::init_model 里的
    # functools.partial），而 FSDPEngineWithLMHead.forward_step 会在【同一个前向步】里把它用两次：
    #
    #   第一次【在 forward 内部，当 logits processor】（只有 top-k 模式才会发生）：
    #     forward_step → prepare_model_outputs(..., logits_processor_func=本函数) 里，
    #     趁原始 logits (bsz, seqlen, vocab) 还活着，调 logits_processor_func(student_logits=logits, data=...)。
    #     → 命中下面 `student_logits is not None` 分支 → compute_topk_loss → 后端算出【每 token 的 KL 字典】
    #       {distillation_losses, student_mass, teacher_mass, ...}，return 回 prepare_model_outputs，
    #       被逐 key 存进 model_output[k]（见 transformer_impl.py `for k, v in outputs.items()`）。
    #     ★ 注意：这一次 return 的【不是标量 loss】，而是每 token 张量的 dict；它 return 给的是 engine，不是训练循环。
    #
    #   第二次【forward 结束后，当最终 loss】：
    #     forward_step 里 loss_function(model_output=model_output, data=...)（没传 student_logits → 默认 None）。
    #     → 走下面 else 分支 → distillation_loss(...) 把第一次存进 model_output 的每 token 结果读出来、
    #       agg_loss 聚合成【标量 loss】return 出去，这个标量才拿去 backward。
    #
    # 为什么非得勈成两次：原始 logits 太大且被 TP/SP 分片，不能 gather 出 forward 交给外层 loss；
    #   只能趁它在 forward 里、还带 grad_fn 时就地压成每 token loss（第一次），
    #   等 engine 把 model_output 拼好、SP all-gather 完，再聚合成标量（第二次）。
    #   两次都不多余：第一次产「原料」（每 token KL），第二次做「成品」（标量）。而且 engine 钩子契约也只允许
    #   第一次返回「形状==log_probs 的每 token 张量」，不许返回标量，所以聚合必须推迟到第二次。
    # ============================================================================================

    # ===================== NOTE：怎么区分「传统 OPD」和「PG-OPD」？=====================
    # 三个开关都在 distillation_config.distillation_loss 下，组合出不同路线：
    #
    #   (1) use_policy_gradient —— 这是「传统 OPD vs PG-OPD」的【核心分水岭】，判定在下面的 distillation_loss() 里：
    #         · False → 传统 OPD（监督式 GKD）：distillation_losses 直接 agg_loss 当监督 loss 回传（arxiv 2306.13649）。
    #                   走这条时 loss_mode=forward_kl_topk（top-k 前向 KL）；这个 GKD 分支本身只消费
    #                   teacher_logprobs/teacher_ids，不用 old_log_probs/advantages。但外层 VeRL 原版仍会调用
    #                   一次随后清零的 ppo_loss scaffold，因此提交给统一入口的 batch 仍要带这两个接口字段。
    #         · True  → PG-OPD（on-policy 蒸馏）：把 -distillation_losses.detach() 当 advantage/reward，
    #                   丢进 policy_loss_fn 走策略梯度（thinkingmachines on-policy-distillation）。
    #                   走这条时 loss_mode=k1，需要 old_log_probs、response_mask、可选 rollout_is_weights（IS 校正）。
    #
    #   (2) use_task_rewards —— 正交开关，决定要不要在蒸馏目标之外再叠一个「真实任务」的 PPO 目标（就在本函数下面）：
    #         · False → 纯蒸馏：仍按原版执行 ppo_loss，但立刻清零其标量，最终 policy_loss = distill_loss
    #                   （Memory-OPD 就是这条，见 trainer 的校验）；
    #         · True  → 混合：policy_loss = ppo_loss(...) + distill_loss * distillation_loss_coef。
    #
    #   (3) loss_mode —— 选具体蒸馏 loss 实现（forward_kl_topk / k1 ...），与 (1) 配套（见 get_distillation_loss_fn）。
    #
    #   小结：本函数负责「蒸馏 vs 蒸馏+任务奖励」(use_task_rewards)；
    #         「监督式 vs 策略梯度」(use_policy_gradient) 的真正分叉在 distillation_loss() 内部。
    # ================================================================================

    # ---------- 第一次调用：在 forward 内部当 logits processor（仅 top-k 模式才会发生）----------
    # 进来时 student_logits 非 None（engine 趁 logits 还活着把它传进来）。
    # 这里只负责：拿原始 logits 现算「每 token 的 top-k KL 字典」，然后 return 回 engine
    #（prepare_model_outputs 会把它逐 key 存进 model_output，供下面第二次调用聚合用）。
    # ★ 返回的是「每 token 张量的 dict」，不是标量 loss；接收方是 engine，不是训练循环。
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # ---------- 第二次调用：forward 之后当最终 loss ----------
    # 进来时 student_logits 为 None：
    #   · top-k 模式(forward_kl_topk)：model_output 里已经有第一次存好的每 token 结果，distillation_loss 只做「读出+聚合」。
    #   · estimator 模式(k1/kl/...)：根本没有第一次调用，distillation_loss 直接用 log_probs 现算 KL（见其内部）。
    # 总之 distillation_loss(...) 产出【标量 distill_loss】，这才是 backward 用的 loss。
    distillation_loss_config = distillation_config.distillation_loss

    # NOTE（严格保留 VeRL 原版的最终 loss 编排）：下面的先后顺序和职责不能混为一谈。
    #
    #   1. ``distillation_loss(...)`` 始终先计算蒸馏项：
    #      · ``use_policy_gradient=False`` 是传统 GKD-OPD，直接聚合 teacher top-k
    #        forward KL；
    #      · ``use_policy_gradient=True`` 是 PG-OPD，把 teacher/student 差异作为
    #        advantage，走蒸馏配置自己的 policy-gradient loss。
    #   2. ``ppo_loss(...)`` 随后始终执行。这是 VeRL 为“纯蒸馏”和“蒸馏 + task reward”共用
    #      一套 actor loss 接口而保留的标准 PPO scaffold。它会读取 response_mask、old_log_probs、
    #      advantages，并顺带把 data 中的 dp_size、batch_num_tokens、global_batch_size 写入
    #      ``config.global_batch_info``；这些字段也供后续 micro-batch 的 loss 聚合复用。
    #   3. ``use_task_rewards=False`` 时，刚算出的标准 PPO 标量会被显式清零，因此它不参与最终
    #      参数更新；最终只留下 distill_loss。此时 old_log_probs/advantages 仍是调用
    #      ``ppo_loss`` 的接口契约，并不表示传统 GKD-OPD 算法本身依赖 PPO ratio 或 task reward。
    #   4. ``use_task_rewards=True`` 时才保留标准 PPO 标量，并按 distillation_loss_coef 叠加
    #      蒸馏项，形成 PPO/GRPO task objective + OPD 的混合目标。
    #
    # NOTE（原版调用顺序的可见副作用）：FSDP 的 forward_backward_batch 会先在完整 optimizer
    # batch 上计算 data["batch_num_tokens"]，再切 micro-batch；ppo_loss 只是把 data 中已经存在的
    # dp_size/batch_num_tokens/global_batch_size 复制到 ``config.global_batch_info``。但 VeRL 原版在
    # 当前调用中先聚合 distill_loss，之后才调用 ppo_loss，所以当前 distill 聚合不会由“下面这次”
    # ppo_loss 预先初始化；ActorConfig 在调用间复用，刚写入的信息只能供后续调用继续读取。
    # 本项目严格保留这个顺序，不提前复制、不抽取公共初始化，也不把它包装成自己的聚合修复。
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        # NOTE：严格对应 VeRL 原版。这里只清零标准 PPO/task-reward 标量；上面已经得到的
        # distill_loss 不受影响，下一步会以系数 1.0 加回。因此纯传统 OPD 的最终目标仍然只有
        # teacher top-k forward KL，不会混入零 advantage 产生的 PPO 项。
        policy_loss = 0.0

    # NOTE：VeRL 原版规定：纯 OPD 固定以 1.0 使用蒸馏项；只有显式启用 task reward、需要把
    # PPO 与蒸馏混合时，distillation_loss_coef 才控制蒸馏项相对于 PPO 项的权重。
    policy_metrics.update(distill_metrics)
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
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    # NOTE：distillation_loss_fn 是「按 loss_mode 从注册表动态查出来的函数」，所以点不进去。
    #   解析链：get_distillation_loss_fn(loss_mode) → DISTILLATION_LOSS_REGISTRY[loss_mode]，
    #   这张表由 @register_distillation_loss(...) 装饰器在模块导入时填充。loss_mode → 实际源码：
    #     · "forward_kl_topk"                          → compute_forward_kl_topk（本文件下方）；
    #           ⚠ 但它只是「读结果」：真正的逐 token top-k KL 早在 logits processor 阶段就算好了
    #             （compute_topk_loss → 按 strategy 落到 trainer/distillation/{fsdp,megatron}/losses.py 的
    #              compute_forward_kl_topk），这里只从 model_output["distillation_losses"/"student_mass"/...] 取出来。
    #     · "kl"/"k1"/"abs"/"mse"/"k2"/"low_var_kl"/"k3" → compute_distillation_loss_reverse_kl_estimator（本文件下方，
    #           用 core_algos.kl_penalty 的单样本 KL 估计器，直接从 student/teacher log_probs 现算）。
    #   两个被注册的实现都带 @register_distillation_loss，就在本文件后半段，Ctrl+F 函数名即可跳到。
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
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
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # 按 https://arxiv.org/abs/2306.13649 的监督式蒸馏做法，直接反向传播 distillation loss。
        # NOTE（传统 GKD-OPD）：这个分支不使用 old_log_probs、PPO ratio、clip 或 task advantage；
        # 它只把每 token teacher top-k forward KL 按 actor 的 loss_agg_mode 聚合为标量并直接反传。
        #
        # NOTE（为什么这里仍读取 global_batch_info）：VeRL 原版把 dp_size、batch_num_tokens、
        # global_batch_size 的写入放在外层 ``distillation_ppo_loss`` 随后调用的 ``ppo_loss`` 中，
        # 并通过可变的 ActorConfig 在 micro-batch 间复用。这里保持原版读取方式，不把 PPO scaffold
        # 误解成 GKD 算法的一部分，也不在本分支另建一套聚合协议。
        #
        # NOTE（全局均值的目标）：FSDP 会在切分 micro-batch 前计算整批 batch_num_tokens；当
        # global_batch_info 已由原版 PPO scaffold 刷新后，每个 micro-batch 只贡献“本地 token loss
        # 之和 / 全局 token 数”，engine 再累加这些贡献，得到整个 optimizer batch 的 token mean。
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)

        # NOTE：这里不是简单对“样本条数”求平均；当前 loss_agg_mode="token-mean"，所以分母是
        # response_mask 覆盖的全局有效 token 数。只有改成 seq-* 模式时才会采用逐序列语义。
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
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    
    # NOTE： 这是传统 OPD 的 loss function
    #
    # ★★★ 重要（也是一个设计上的坑）★★★
    # 别被这个函数名骗了：真正的 top-k 蒸馏 loss 并不是在这里算的，而是早在
    #     FSDPEngineWithLMHead.prepare_model_outputs（见 verl/workers/engine/fsdp/transformer_impl.py，
    #     那段 `if distillation_use_topk:` → logits_processor_func(student_logits=logits_rmpad, ...)）
    # 里、前向还没结束时就已经算好、并存进了 model_output["distillation_losses"/"student_mass"/"teacher_mass"]。
    # 这个函数只是「收尾」：把那批已经算好的每 token 结果读出来、聚合成一个标量。
    # 所以下面几行 model_output["distillation_losses"] 一进来就是现成的，看着很反直觉。
    #
    # 为什么会变成这样（两次调用的真相）：distillation_ppo_loss 其实被调了两次——
    #   第一次（在 prepare_model_outputs 内部，logits 还活着）：引擎把同一个 loss 函数当作
    #     “logits processor” 钩子传进去，student_logits is not None → compute_topk_loss，
    #     就地把 (total_nnz, vocab) 的巨型 logits + 老师 top-k 压成「每 token 的 KL 字典」（不是标量），
    #     return 回 engine 后被逐 key 存回 model_output，然后巨型 logits 用完即弃。
    #   第二次（前向结束，当最终 loss）：student_logits=None → distillation_loss → 走到这里做聚合。
    # 它这么绕的（唯一）理由：logits 形状 (所有 token × 整个词表) 巨大且被 TP/SP 分片，不能 gather 出来
    #   传给外层 loss；必须趁它还在 forward、还带 grad_fn 时就地压成每 token loss，梯度链才能保持连通
    #   （参数→logits→distillation_losses→这里聚合→标量 loss→backward）。
    #
    # 吐槽：从代码设计角度这确实是灾难——loss 的计算被劈成两半、一半藏在 engine 的 prepare_model_outputs 里，
    #   违背了「loss 就该在 loss function 里算完」的直觉；靠 student_logits 是否为 None 来复用同一函数做两件事，
    #   可读性很差。它是为性能/显存做的妥协，不是好范式，读的时候心里有数即可。
    #
    # ── 真正干活的 loss function 到底在哪？（完整调用链，方便你去看）──
    #   engine 前向内部：FSDPEngineWithLMHead.prepare_model_outputs
    #     （verl/workers/engine/fsdp/transformer_impl.py，`if distillation_use_topk:` 那段）
    #       └─ 调 logits_processor_func(student_logits=logits_rmpad, ...)  # 即本文件的 distillation_ppo_loss
    #            └─ distillation_ppo_loss 里 `student_logits is not None` 分支
    #                 └─ compute_topk_loss(...)              # 本文件，line ~139，按 config.strategy 分发
    #                      ├─ strategy=="fsdp"/"veomni" → verl/trainer/distillation/fsdp/losses.py::compute_forward_kl_topk  ← ★真正算 KL 的地方
    #                      └─ strategy=="megatron"       → verl/trainer/distillation/megatron/losses.py::compute_forward_kl_topk
    #   ★ 那个后端 compute_forward_kl_topk 才是真身：F.log_softmax(student_logits) → 对老师 top-k 位置 gather →
    #     kl_divergence(log_q=student, log_p=teacher)，产出每 token 的 distillation_losses/student_mass/teacher_mass。
    #   注意：本文件这个同名的 compute_forward_kl_topk（下面）只是「聚合器」，别和 fsdp/megatron 后端那个真身混了。
    #
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
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
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
