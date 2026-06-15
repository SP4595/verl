"""LoCoMo 原始长对话到 Memory-OPD episode source 的适配器。

这个 Dataset 不生成模型 prompt，也不执行 tokenization。它只负责把不同来源的
LoCoMo JSON 规范化成 episode 参数。后续 Agentic Collector 按 session 顺序创建
memory，再按 QA 顺序运行回答轨迹，并把每个决策状态展开成独立的 single-turn
``MemoryOPDStep``。

重要边界：

- 一条 Dataset 记录对应一个完整 LoCoMo conversation，而不是一条 QA；
- ``memory_episode`` 是 rollout 输入，不是直接用于 actor forward 的 prompt；
- query/update/answer 的实际 prompt 由 AgentLoop 在每个动态 cache 状态下构造；
- teacher 可见的完整 memory 是 rollout 期间逐步形成的，不能在 Dataset 中预先构造。
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from verl.trainer.meminaction.RLDatasets.common import normalize_data_files
from verl.trainer.meminaction.RLDatasets.schema import validate_memory_episode, validate_sample


class LoCoMoPrivilegeSubsetDataset(Dataset):
    """把 LoCoMo conversation 转换为统一的 Memory-OPD episode API。

    输出中的主要字段：

    - ``memory_episode.sessions``：按时间顺序执行的 memory 创建任务；
    - ``memory_episode.qa``：memory 创建完成后执行的 QA 任务；
    - ``raw_prompt``：仅用于兼容 VeRL Dataset 契约和日志，不是训练 prompt；
    - ``agent_name``：标记该样本必须先由 episode collector 展开。

    ``tokenizer`` 和 ``processor`` 仅为兼容 VeRL ``create_rl_dataset`` 的固定构造
    签名而保留。动态 prompt 只能在 AgentLoop 中根据当前 Memory Cache token 化。

    该 Dataset 是 episode source，不能直接交给 ``RayPrivilegeOPDTrainer``。正确路径是
    先把每条 ``memory_episode`` 交给 ``MemoryOPDEpisodeCollector.collect()``，再将
    collector trace 装入 ``MemoryOPDStepDataset``。
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        # VeRL create_rl_dataset 会无条件传入 tokenizer/processor。这里不能使用它们：
        # Dataset 尚不知道 rollout 中每一步的动态 Cache，因此任何长度过滤都会失真。
        del tokenizer, processor
        self.config = config
        self.default_data_source = config.get("default_data_source", "locomo_memory_opd")
        self.default_agent_name = config.get("default_agent_name", "memory_opd_episode")
        self.skip_category_5 = config.get("locomo_skip_category_5", True)
        self.max_sessions = config.get("locomo_max_sessions")
        self.max_qa = config.get("locomo_max_qa")
        self.validate_custom_sample = config.get("validate_custom_sample", True)

        self.rows: list[dict[str, Any]] = []
        self._load_episodes(data_files)

        if config.get("shuffle", False):
            rng = random.Random(config.get("seed", 0) or 0)
            rng.shuffle(self.rows)
        if 0 < max_samples < len(self.rows):
            self.rows = self.rows[:max_samples]

    @staticmethod
    def _render_turn(turn: Mapping[str, Any]) -> str:
        """渲染单轮对话，并保留 speaker、dia_id 和图片描述。"""

        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "?")
        text = str(turn.get("text") or turn.get("clean_text") or "").strip()
        caption = str(turn.get("blip_caption") or "").strip()
        if caption:
            text = f"{text} [Image: {caption}]".strip()
        return f"{speaker} ({dia_id}): {text}"

    @staticmethod
    def _session_numbers(sample: Mapping[str, Any]) -> list[int]:
        """返回 conversation 中真正的 session 编号，跳过日期字段。"""

        conversation = sample["conversation"]
        return sorted(
            int(match.group(1))
            for key in conversation
            if (match := re.fullmatch(r"session_(\d+)", key)) is not None
        )

    @classmethod
    def _build_session(cls, sample: Mapping[str, Any], session_number: int) -> dict[str, Any]:
        """构造一次 memory 创建任务的输入参数。

        一个 LoCoMo session 会作为一个整体 ``current_input`` 交给 update trajectory。
        Dataset 只做确定性文本规范化；是否 query、怎样 update 由运行时 controller 决定。
        """

        conversation = sample["conversation"]
        date_time = str(conversation.get(f"session_{session_number}_date_time") or "")
        turns = conversation[f"session_{session_number}"]
        rendered_turns = "\n".join(cls._render_turn(turn) for turn in turns)
        chunk = f"Date: {date_time}\n{rendered_turns}".strip()
        return {
            "session_index": session_number,
            "date_time": date_time,
            "input": chunk,
            # 原始 turn 用于追踪 evidence 和后续扩展，不直接放进 prompt。
            "turns": copy.deepcopy(turns),
        }

    def _build_qa(self, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
        """构造 memory 创建完成后依次执行的 QA 任务。

        ``answer`` 是评估或后续 reward 所需标签，不会被放进 student prompt。``evidence``
        同样只作为可追踪元数据保留，不在这里预先注入 Cache。
        """

        rows = []
        for qa_index, qa in enumerate(sample.get("qa", [])):
            category = int(qa.get("category", 0))
            if self.skip_category_5 and category == 5:
                continue
            rows.append(
                {
                    "qa_index": qa_index,
                    "question": str(qa["question"]).strip(),
                    "answer": qa.get("answer"),
                    "category": category,
                    "evidence": copy.deepcopy(qa.get("evidence", [])),
                }
            )
        return rows[: self.max_qa]

    def _build_row(self, sample: Mapping[str, Any], source_file: str) -> dict[str, Any]:
        """把一个完整 LoCoMo conversation 转换为一条 episode source。

        这里保持“一条 conversation 对应一条 Dataset row”，避免提前展平 QA 后丢失
        session 写入顺序和共享长期 memory 的语义。
        """

        row_index = len(self.rows)
        sample_id = str(sample.get("sample_id") or f"locomo-{row_index}")
        session_numbers = self._session_numbers(sample)[: self.max_sessions]
        memory_episode = {
            "schema_version": 1,
            "episode_id": sample_id,
            "source": "locomo",
            "sessions": [self._build_session(sample, number) for number in session_numbers],
            "qa": self._build_qa(sample),
            "metadata": {
                "source_file": source_file,
                "sample_id": sample_id,
            },
        }
        validate_memory_episode(memory_episode)

        # VeRL 当前 Dataset/AgentLoop 管道要求 raw_prompt 非空。这里仅放一个可读的
        # episode 启动标记；真正的每步 prompt 由 MemoryOPDPromptRenderer 生成。
        raw_prompt = [
            {
                "role": "user",
                "content": f"Run Memory-OPD episode {sample_id}.",
            }
        ]
        row = {
            "prompt": copy.deepcopy(raw_prompt),
            "raw_prompt": raw_prompt,
            "data_source": self.default_data_source,
            "reward_model": {},
            # episode source 尚未产生模型 response，纯 OPD 也不在此阶段计算 reward。
            "extra_info": {
                "index": row_index,
                "sample_id": sample_id,
                "num_sessions": len(memory_episode["sessions"]),
                "num_qa": len(memory_episode["qa"]),
            },
            "index": row_index,
            "agent_name": self.default_agent_name,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "memory_episode": memory_episode,
        }
        if self.validate_custom_sample:
            validate_sample(row, require_ground_truth=False)
        return row

    def _load_episodes(self, data_files: Any) -> None:
        """读取 LoCoMo JSON；每个 conversation 保持为一条 episode。

        与 RLHFDataset 不同，这里不会应用 chat template、tokenize 或过滤动态 prompt
        长度。长度限制必须在 ``MemoryOPDStepAgentLoop`` 渲染真实 prompt 后检查。
        """

        from verl.utils.fs import copy_to_local

        for data_file in normalize_data_files(data_files):
            local_file = copy_to_local(src=data_file, cache_dir=self.config.get("cache_dir"))
            payload = json.loads(Path(local_file).read_text(encoding="utf-8"))
            samples = payload if isinstance(payload, list) else [payload]
            for sample in samples:
                self.rows.append(self._build_row(sample, source_file=str(data_file)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[item])


__all__ = ["LoCoMoPrivilegeSubsetDataset"]
