"""可直接交给 VeRL RewardManager 的普通答案 reward function 模板。

当前 ``RayPrivilegeOPDTrainer`` 是纯 OPD，配置会明确禁止 Reward Model 和任务 reward，
因此它不会调用本模块。保留这些函数是为了普通 RL/评估阶段或未来在 answer step 上
启用任务 reward；它们不评价 query/update 对 Memory 状态造成的长期影响。

真正的 Memory-RL reward 需要接收动作执行后的环境状态或未来 QA 结果，不能只依赖这里
的 ``solution_str`` 和 ``ground_truth`` 接口。
"""

import math
import re
from collections.abc import Iterable
from typing import Any


def _normalize_answer(value: Any) -> str:
    """执行保守的 exact-match 文本规范化。"""

    # 步骤 1：统一转换为字符串、去除首尾空白并执行大小写无关规范化。
    text = str(value).strip().casefold()
    # 步骤 2：把任意连续空白压缩为单个空格。
    return re.sub(r"\s+", " ", text)


def _extract_final_answer(solution_str: str) -> tuple[str, bool]:
    """提取最终答案，并返回是否命中推荐格式。

    按明确格式优先级查找，若都未命中则回退到最后一个非空行。回退答案可以参与正确性
    判断，但不会获得 ``format_score``。
    """

    # 步骤 1：清理模型响应首尾空白。
    text = solution_str.strip()
    # 步骤 2：按优先级声明支持的显式答案格式。
    patterns = (
        r"####\s*(.+?)(?:\n|$)",
        r"<answer>\s*(.*?)\s*</answer>",
        r"\\boxed\{([^{}]+)\}",
    )
    # 步骤 3：依次匹配格式；同一格式多次出现时使用最后一个答案。
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return str(matches[-1]).strip(), True

    # 步骤 4：未命中显式格式时，回退到最后一个非空行。
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (non_empty_lines[-1] if non_empty_lines else ""), False


def _iter_ground_truths(ground_truth: Any) -> Iterable[Any]:
    """把单个答案或多个可接受答案统一成可迭代对象。"""

    # 步骤 1：多个可接受答案保持为可迭代对象。
    if isinstance(ground_truth, (list, tuple, set)):
        return ground_truth
    # 步骤 2：单答案包装为单元素 tuple。
    return (ground_truth,)


def _is_correct(prediction: str, ground_truth: Any, numeric_tolerance: float | None) -> bool:
    """执行规范化 exact match，并可选允许绝对数值误差。"""

    # 步骤 1：预先规范化预测文本。
    normalized_prediction = _normalize_answer(prediction)
    # 步骤 2：逐个尝试所有可接受 ground truth。
    for candidate in _iter_ground_truths(ground_truth):
        # 步骤 3：优先执行保守 exact match。
        if normalized_prediction == _normalize_answer(candidate):
            return True

        # 步骤 4：配置数值容差时，再尝试解析数字并比较绝对误差。
        if numeric_tolerance is not None:
            try:
                predicted_number = float(prediction.replace(",", "").strip())
                expected_number = float(str(candidate).replace(",", "").strip())
            except (TypeError, ValueError):
                continue
            if math.isclose(predicted_number, expected_number, abs_tol=numeric_tolerance, rel_tol=0.0):
                return True
    # 步骤 5：所有候选答案均未匹配。
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    *,
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
    format_score: float = 0.0,
    numeric_tolerance: float | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """逐样本 reward function：最终答案 exact-match 加可选格式奖励。

    实际多任务项目通常应按 ``data_source`` 分发到数学、代码、工具调用等任务
    专属 verifier。返回字典至少包含 ``score``。

    参数 ``extra_info`` 可为单条样本覆盖 ``numeric_tolerance``。额外 ``kwargs`` 被
    接受并忽略，以兼容 VeRL RewardManager 在不同版本中传入的扩展参数。
    """

    # 步骤 1：接受但忽略 RewardManager 额外参数，保持版本兼容。
    del kwargs
    # 步骤 2：读取样本级数值容差，优先覆盖函数默认值。
    sample_tolerance = extra_info.get("numeric_tolerance", numeric_tolerance)
    if sample_tolerance is not None:
        sample_tolerance = float(sample_tolerance)

    # 步骤 3：提取最终答案并判断输出格式。
    prediction, has_recommended_format = _extract_final_answer(solution_str)
    # 步骤 4：计算正确性、准确率奖励和可选格式奖励。
    is_correct = _is_correct(prediction, ground_truth, sample_tolerance)
    accuracy_reward = float(correct_score if is_correct else incorrect_score)
    final_format_reward = float(format_score if has_recommended_format else 0.0)
    # 步骤 5：当前基础实现不按 data_source 路由；保留参数供未来多任务扩展。
    _ = data_source

    # 步骤 6：返回 RewardManager 需要的总分和可观测分项。
    return {
        "score": accuracy_reward + final_format_reward,
        "acc": float(is_correct),
        "accuracy_reward": accuracy_reward,
        "format_reward": final_format_reward,
    }


def compute_score_batched(
    data_sources: Iterable[str],
    solution_strs: Iterable[str],
    ground_truths: Iterable[Any],
    extra_infos: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """批量 RewardManager 使用的 reward function。

    ``strict=True`` 会在各字段长度不一致时立即报错，避免 zip 静默截断导致部分样本
    没有 reward。
    """

    # 步骤 1：严格按相同位置组合四个批量字段。
    # 步骤 2：逐样本复用 compute_score；任意字段长度不一致时立即报错。
    return [
        compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
            strict=True,
        )
    ]


__all__ = ["compute_score", "compute_score_batched"]
