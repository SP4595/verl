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
# FSDP 后端的前向 KL 蒸馏损失计算（logit processor 阶段）
#
# 本模块由 verl/trainer/distillation/losses.py 中的 compute_topk_loss() 在
# actor.strategy="fsdp" 时调用，对应 run_qwen3_8b_fsdp.sh 所选用的训练后端。
#
# 核心功能:
#   compute_forward_kl_topk:
#     在模型前向传播期间，当 logit processor 钩子被触发时，
#     利用学生 logits 和教师 top-k log-probs 计算 token 级别的 KL 散度。
#     结果缓存到 model_output 中，供后续 distillation_loss() 聚合使用。
#
# 注意: run_qwen3_8b_fsdp.sh 默认使用 loss_mode=k1（单样本估计器），
# 不经过本模块（k1 不需要 top-k logits）。
# 只有切换到 loss_mode=forward_kl_topk 时才会调用此处的 compute_forward_kl_topk。
# =============================================================================

import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """计算两个分布 P 和 Q 之间的 KL 散度： KL(P || Q) = sum_k p_k * log(p_k / q_k)

    此函数在 compute_forward_kl_topk 中被调用，计算教师分布对学生分布的
    前向 KL（前向指教师分布为基准分布），只在 teacher top-k token 上近似计算。

    Args:
        log_q: 学生分布的 log-prob，形状 (..., vocab_size/topk)
        log_p: 教师分布的 log-prob，形状与 log_q 相同

    Returns:
        token 级 KL 值，将最后一个维度（token 维）求和后的形状 (...)
    """
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 logit processor 阶段使用 top-k log-prob 计算前向 KL 蒸馏损失。

    本函数在学生模型前向传播期间被调用，此时有助于:
      1. logits 尚未被特殊处理（如温度/top_p 采样），能精确表征完整分布
      2. 可利用并行算力避免额外一次全量前向

    计算流程:
      1. 按序列并行分组（sp）切分 teacher topk logprobs/ids：
         (bsz, seqlen, topk) → (bsz, seqlen/sp_size, topk)
      2. 对学生 logits 做 log_softmax 得到完整分布的 log-prob
      3. 从学生分布中 gather 教师 top-k token 对应的 log-prob
      4. 计算 teacher top-k token 上的概率质量（mass）和学生质量
      5. 计算 KL(teacher || student) 在 top-k token 上的近似值
      6. 计算 teacher/student top-k token 的 overlap 分析指标

    Args:
        student_logits: 学生模型输出的未归一化 logits，形状 (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: 教师 top-k token log-prob，形状 (bsz, seqlen, topk).
        teacher_topk_ids: 教师 top-k token ID，形状 (bsz, seqlen, topk).
        config: 蒸馏配置，包含 topk、log_prob_min_clamp 等。
        data_format: "thd" 或 "bshd"，Qwen3 使用 "bshd"。

    Returns:
        字典，包含:
        - distillation_losses: (bsz, seqlen/sp_size) token 级 KL 散度
        - student_mass: (bsz, seqlen/sp_size) 学生在教师 top-k token 上的概率质量
        - teacher_mass: (bsz, seqlen/sp_size) 教师在 top-k token 上的概率质量
        - overlap_count: (bsz, seqlen/sp_size) 学生/教师 top-k token 的重叠数
        - overlap_token_advantage: (bsz, seqlen/sp_size) 重叠 token 上的平均教师 KL 贡献
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    # 将 NestedTensor 转换为普通张量，便于后续切片和计算
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. 按序列并行分组切分 (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    #    只有开启了 Ulysses 序列并行时才需要，单 GPU 训练时跳过
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. 对学生 logits 做 log_softmax，得到全词表上的 log 概率分布
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    # 获取学生自己的 top-k token ID（用于 overlap 分析）
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    # 3. 从学生分布中汇聚教师 top-k token 的 log-prob
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    # 4. 计算概率质量（top-k token 上的概率和），监控 teacher/student 分布截断程度
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        # 对应 run_qwen3_8b_fsdp.sh: log_prob_min_clamp=-10.0
        # 防止极小概率 token 的 log-prob 越界（-inf）导致数値不稳定
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    # 5. 计算 token 级的前向 KL 散度： KL(teacher || student) 在 top-k token 上的近似
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # 6. Overlap 分析：跟踪学生/教师 top-k token 的重叠情况
    # 参考论文: "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016)
    # 重叠比较高说明学生和教师对同一 token 有共同关注
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)  # 每个 token 位置的重叠 token 数量
    # 重叠 token 上教师贡献的平均 KL （负号代表可用于优化的比较优势）
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }
