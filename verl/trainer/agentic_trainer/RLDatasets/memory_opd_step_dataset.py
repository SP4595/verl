"""Agentic Collector 输出到 RayPrivilegeOPDTrainer 的 step buffer Dataset。"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from verl.trainer.agentic_trainer.RLDatasets.common import normalize_data_files
from verl.trainer.agentic_trainer.RLDatasets.schema import validate_sample


class MemoryOPDStepDataset(Dataset):
    """将 collector trace 中的每个动态状态暴露为独立 single-turn 训练样本。

    这个 Dataset 不执行 episode，也不构造 prompt。``MemoryOPDStepAgentLoop`` 会读取
    ``memory_step``，根据其中的 Cache/full-memory 快照动态构造 student/teacher prompt。

    支持的输入 JSON 结构：

    - 单条 step：``{"memory_step": {...}, ...}``
    - step 列表；
    - ``MemoryOPDEpisodeCollector.collect()`` 返回的完整 episode trace。

    在线训练可在 collector 生成新轨迹后调用 :meth:`replace_steps` 更新 driver 侧
    buffer。若 DataLoader 使用多进程 worker，必须重建 DataLoader 或使用共享 buffer，
    否则 worker 仍会持有旧副本。
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        del tokenizer, processor
        self.config = config
        self.default_data_source = config.get("default_data_source", "memory_opd")
        self.default_agent_name = config.get("default_agent_name", "memory_opd_step")
        self.validate_custom_sample = config.get("validate_custom_sample", True)
        self.max_samples = max_samples
        self.rows: list[dict[str, Any]] = []
        self._load(data_files)

    @classmethod
    def _extract_steps(cls, payload: Any) -> list[dict[str, Any]]:
        """递归读取 step、episode trace 或它们的列表。"""

        if isinstance(payload, list):
            return [step for item in payload for step in cls._extract_steps(item)]
        if not isinstance(payload, Mapping):
            raise TypeError(f"Memory-OPD trace 必须是 dict/list，实际为 {type(payload)!r}")
        if "memory_step" in payload:
            return [copy.deepcopy(dict(payload))]
        if "steps" in payload:
            return cls._extract_steps(payload["steps"])
        if "tasks" in payload:
            return cls._extract_steps(payload["tasks"])
        raise KeyError("Memory-OPD trace 中找不到 memory_step、steps 或 tasks")

    @staticmethod
    def _validate_step(step: Mapping[str, Any]) -> None:
        required = ("episode_id", "phase", "task_mode", "current_input", "memory_cache", "full_memory")
        missing = [key for key in required if key not in step]
        if missing:
            raise KeyError(f"memory_step 缺少字段: {missing}")
        if step["task_mode"] not in {"update", "answer"}:
            raise ValueError(f"memory_step.task_mode 非法: {step['task_mode']!r}")

    def _build_row(self, trace_step: Mapping[str, Any], index: int) -> dict[str, Any]:
        memory_step = copy.deepcopy(dict(trace_step["memory_step"]))
        self._validate_step(memory_step)
        raw_prompt = [
            {
                "role": "user",
                "content": (
                    f"Memory-OPD step {memory_step.get('episode_id', '?')}:"
                    f"{memory_step.get('step_index', index)}"
                ),
            }
        ]
        extra_info = {
            "index": index,
            "episode_id": memory_step["episode_id"],
            "phase": memory_step["phase"],
            "task_mode": memory_step["task_mode"],
            "step_index": memory_step.get("step_index", index),
            # collector action 仅用于追踪状态来源；训练时仍由当前 student 在线采样。
            "collected_action": trace_step.get("action"),
            "collected_status": trace_step.get("status"),
        }
        row = {
            "prompt": copy.deepcopy(raw_prompt),
            "raw_prompt": raw_prompt,
            "data_source": self.default_data_source,
            "reward_model": {},
            "extra_info": extra_info,
            "index": index,
            "agent_name": self.default_agent_name,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "memory_step": memory_step,
        }
        if self.validate_custom_sample:
            validate_sample(row, require_ground_truth=False)
        return row

    def replace_steps(self, trace_payload: Any) -> None:
        """用新 collector trace 原子替换当前 driver 侧 step buffer。"""

        trace_steps = self._extract_steps(trace_payload)
        if self.config.get("shuffle", False):
            rng = random.Random(self.config.get("seed", 0) or 0)
            rng.shuffle(trace_steps)
        if 0 < self.max_samples < len(trace_steps):
            trace_steps = trace_steps[: self.max_samples]
        self.rows = [self._build_row(step, index) for index, step in enumerate(trace_steps)]

    def _load(self, data_files: Any) -> None:
        from verl.utils.fs import copy_to_local

        trace_steps = []
        for data_file in normalize_data_files(data_files):
            local_file = copy_to_local(src=data_file, cache_dir=self.config.get("cache_dir"))
            path = Path(local_file)
            if path.suffix == ".jsonl":
                payload = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
            trace_steps.extend(self._extract_steps(payload))
        self.replace_steps(trace_steps)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[item])


__all__ = ["MemoryOPDStepDataset"]
