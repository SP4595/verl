"""Agentic Collector trace 到 ``RayPrivilegeOPDTrainer`` 的 step buffer Dataset。

这是 episode source 和 Trainer 之间的第二阶段 Dataset：

``memory_episode -> Collector trace -> MemoryOPDStepDataset -> step Trainer``。

它消费的是 Collector 已经运行过状态机后保存的冻结状态，不持有活跃 Memory，也不会
在 ``__getitem__`` 中执行 query/update。在线重新采集时，应替换整个 step buffer 或
使用显式共享存储，不能假设 DataLoader worker 会自动看到 driver 内存中的新列表。
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from verl.trainer.meminaction.RLDatasets.common import normalize_data_files, to_plain_container
from verl.trainer.meminaction.RLDatasets.schema import validate_sample


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

    ``tokenizer`` 和 ``processor`` 同样只是 VeRL 工厂兼容参数。step 内 Cache 已冻结，
    但真实 prompt 仍需结合全局模板在 AgentLoop 中构造，不能使用这里的占位
    ``raw_prompt`` 做 tokenization。
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        # 真实 prompt 由 AgentLoop 根据 memory_step 动态渲染；这里不提前 tokenize。
        # 步骤 1：丢弃只用于 VeRL 工厂签名兼容的 tokenizer/processor。
        del tokenizer, processor
        # 步骤 2：读取 step buffer 的路由、校验、采样配置。
        self.config = config
        self.default_data_source = config.get("default_data_source", "memory_opd")
        self.default_agent_name = config.get("default_agent_name", "memory_opd_step")
        self.validate_custom_sample = config.get("validate_custom_sample", True)
        self.max_samples = max_samples
        # 步骤 3：从 collector trace 文件构建初始 step buffer。
        self.rows: list[dict[str, Any]] = []
        self._load(data_files)

    @classmethod
    def _extract_steps(cls, payload: Any, config: Any | None = None) -> list[dict[str, Any]]:
        """递归读取单 step、task trace、episode trace 或它们的列表。

        Collector 返回的 episode trace 同时含有嵌套 ``tasks[*].steps`` 和便于消费的顶层
        ``steps``。发现顶层 ``steps`` 后直接使用它，避免同一 step 被重复展开。

        另外支持 oracle snapshot payload：

        ``{"memory_episode": {...}, "oracle_session_snapshots": [...]}``

        这会在 Dataset 加载阶段展开为“一条 session/QA snapshot anchor 对应一条
        ``memory_step``”，便于直接把离线 oracle 快照文件作为 step dataset 输入。
        """

        # 步骤 1：列表输入逐项递归展开。
        if isinstance(payload, list):
            return [step for item in payload for step in cls._extract_steps(item, config=config)]
        # 步骤 2：拒绝无法表达 trace 层级的非 mapping 值。
        if not isinstance(payload, Mapping):
            raise TypeError(f"Memory-OPD trace 必须是 dict/list，实际为 {type(payload)!r}")
        # 步骤 3：发现 memory_step 表示已到达叶子 trace record，深拷贝后返回。
        if "memory_step" in payload:
            return [copy.deepcopy(dict(payload))]
        # 步骤 4：oracle snapshot 输入在 Dataset 中展开为 step trace。
        if "memory_episode" in payload and "oracle_session_snapshots" in payload:
            from mem_in_action.configs import MAgentConfig
            from verl.trainer.meminaction.agentic_loop import build_oracle_memory_opd_trace

            config_data = dict(to_plain_container(config) or {}) if config is not None else {}
            controller_config_data = dict(
                config_data.get("oracle_snapshot_controller")
                or config_data.get("memory_opd_prompt")
                or {}
            )
            trace = build_oracle_memory_opd_trace(
                payload["memory_episode"],
                payload["oracle_session_snapshots"],
                config=MAgentConfig(**controller_config_data),
                answer_memory=payload.get("answer_memory"),
            )
            return cls._extract_steps(trace, config=config)
        # 步骤 5：优先展开顶层 steps；不存在时再展开 tasks。
        if "steps" in payload:
            return cls._extract_steps(payload["steps"], config=config)
        if "tasks" in payload:
            return cls._extract_steps(payload["tasks"], config=config)
        raise KeyError("Memory-OPD trace 中找不到 memory_step、steps 或 tasks")

    @staticmethod
    def _validate_step(step: Mapping[str, Any]) -> None:
        """在构造 VeRL row 前检查 Collector trace 的最低动态状态契约。"""

        # 步骤 1：检查 AgentLoop 渲染和信息隔离所需的最低字段集合。
        required = ("episode_id", "phase", "task_mode", "current_input", "memory_cache", "full_memory")
        missing = [key for key in required if key not in step]
        if missing:
            raise KeyError(f"memory_step 缺少字段: {missing}")
        # 步骤 2：确保 task_mode 可以映射到合法 terminal action。
        if step["task_mode"] not in {"update", "answer"}:
            raise ValueError(f"memory_step.task_mode 非法: {step['task_mode']!r}")

    def _build_row(self, trace_step: Mapping[str, Any], index: int) -> dict[str, Any]:
        """将一个冻结 trace step 包装成 VeRL single-turn Dataset sample。"""

        # 步骤 1：复制并验证冻结 step，避免 row 与输入 trace 共享可变对象。
        memory_step = copy.deepcopy(dict(trace_step["memory_step"]))
        self._validate_step(memory_step)
        step_metadata = copy.deepcopy(dict(memory_step.get("metadata") or {}))
        # 步骤 2：创建 VeRL 契约要求的占位 raw_prompt。
        raw_prompt = [
            {
                "role": "user",
                "content": (
                    f"Memory-OPD step {memory_step.get('episode_id', '?')}:"
                    f"{memory_step.get('step_index', index)}"
                ),
            }
        ]
        # 步骤 3：提取不参与 prompt 的 episode/step 追踪信息。
        extra_info = {
            "index": index,
            "episode_id": memory_step["episode_id"],
            "phase": memory_step["phase"],
            "task_mode": memory_step["task_mode"],
            "step_index": memory_step.get("step_index", index),
            # metadata 是 snapshot anchor 的结构化身份：第几个 session/QA、oracle
            # snapshot 编号、以及当前样本来自 update 还是 answer。
            "memory_step_metadata": step_metadata,
            "session_index": step_metadata.get("session_index"),
            "qa_index": step_metadata.get("qa_index"),
            "date_time": step_metadata.get("date_time"),
            "collection_mode": step_metadata.get("collection_mode"),
            "oracle_snapshot_before_index": step_metadata.get("oracle_snapshot_before_index"),
            "oracle_snapshot_after_index": step_metadata.get("oracle_snapshot_after_index"),
            "oracle_snapshot_index": step_metadata.get("oracle_snapshot_index"),
            # collector action 仅用于追踪状态来源；训练时仍由当前 student 在线采样。
            "collected_action": trace_step.get("action"),
            "collected_status": trace_step.get("status"),
        }
        # 步骤 4：包装为路由到 memory_opd_step AgentLoop 的 VeRL sample。
        row = {
            # prompt/raw_prompt 是 VeRL 公共 Dataset 契约的兼容占位。AgentLoop 读取的是
            # memory_step，并以当前全局模板重新构造真正的 student prompt。
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
        # 步骤 5：校验 wrapper 契约；纯 OPD step 不要求 reward ground truth。
        if self.validate_custom_sample:
            validate_sample(row, require_ground_truth=False)
        # 步骤 6：返回可被 collate_fn 批处理的 single-turn step row。
        return row

    def replace_steps(self, trace_payload: Any) -> None:
        """用新 collector trace 原子替换当前 driver 侧 step buffer。

        先完成提取、shuffle、截断和 row 构造，再整体替换 ``self.rows``，避免 driver
        读取到半更新列表。该原子性不覆盖多进程 DataLoader 的独立 Dataset 副本。
        """

        # 步骤 1：将任意受支持 trace 层级展开成叶子 step records。
        trace_steps = self._extract_steps(trace_payload, config=self.config)
        # 步骤 2：按配置确定性 shuffle。
        if self.config.get("shuffle", False):
            rng = random.Random(self.config.get("seed", 0) or 0)
            rng.shuffle(trace_steps)
        # 步骤 3：在 step 粒度应用 max_samples。
        if 0 < self.max_samples < len(trace_steps):
            trace_steps = trace_steps[: self.max_samples]
        # 步骤 4：先完整构造新 rows，再一次性替换当前 buffer。
        self.rows = [self._build_row(step, index) for index, step in enumerate(trace_steps)]

    def _load(self, data_files: Any) -> None:
        """从 JSON/JSONL collector trace 文件初始化 step buffer。"""

        from verl.utils.fs import copy_to_local

        # 步骤 1：准备汇总所有 trace 文件中的 step。
        trace_steps = []
        for data_file in normalize_data_files(data_files):
            # 步骤 2：复制到本地并根据扩展名解析 JSON 或 JSONL。
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
            # 步骤 3：递归提取当前文件的叶子 step，并追加到总 buffer。
            trace_steps.extend(self._extract_steps(payload, config=self.config))
        # 步骤 4：统一执行 shuffle、截断和 row 包装。
        self.replace_steps(trace_steps)

    def __len__(self) -> int:
        # 步骤 1：Dataset 长度等于当前冻结 step buffer 的大小。
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        # 步骤 1：返回深拷贝，隔离 DataLoader/Trainer 对缓存 row 的修改。
        return copy.deepcopy(self.rows[item])


__all__ = ["MemoryOPDStepDataset"]
