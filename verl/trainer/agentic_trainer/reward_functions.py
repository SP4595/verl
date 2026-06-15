"""可直接交给 VeRL RewardManager 的基础 reward function 模板。"""

import math
import re
from collections.abc import Iterable
from typing import Any


def _normalize_answer(value: Any) -> str:
    """执行保守的 exact-match 文本规范化。"""

    text = str(value).strip().casefold()
    return re.sub(r"\s+", " ", text)


def _extract_final_answer(solution_str: str) -> tuple[str, bool]:
    """提取最终答案，并返回是否命中推荐格式。"""

    text = solution_str.strip()
    patterns = (
        r"####\s*(.+?)(?:\n|$)",
        r"<answer>\s*(.*?)\s*</answer>",
        r"\\boxed\{([^{}]+)\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return str(matches[-1]).strip(), True

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (non_empty_lines[-1] if non_empty_lines else ""), False


def _iter_ground_truths(ground_truth: Any) -> Iterable[Any]:
    if isinstance(ground_truth, (list, tuple, set)):
        return ground_truth
    return (ground_truth,)


def _is_correct(prediction: str, ground_truth: Any, numeric_tolerance: float | None) -> bool:
    normalized_prediction = _normalize_answer(prediction)
    for candidate in _iter_ground_truths(ground_truth):
        if normalized_prediction == _normalize_answer(candidate):
            return True

        if numeric_tolerance is not None:
            try:
                predicted_number = float(prediction.replace(",", "").strip())
                expected_number = float(str(candidate).replace(",", "").strip())
            except (TypeError, ValueError):
                continue
            if math.isclose(predicted_number, expected_number, abs_tol=numeric_tolerance, rel_tol=0.0):
                return True
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
    """

    del kwargs
    sample_tolerance = extra_info.get("numeric_tolerance", numeric_tolerance)
    if sample_tolerance is not None:
        sample_tolerance = float(sample_tolerance)

    prediction, has_recommended_format = _extract_final_answer(solution_str)
    is_correct = _is_correct(prediction, ground_truth, sample_tolerance)
    accuracy_reward = float(correct_score if is_correct else incorrect_score)
    final_format_reward = float(format_score if has_recommended_format else 0.0)
    _ = data_source

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
    """批量 RewardManager 使用的 reward function。"""

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
