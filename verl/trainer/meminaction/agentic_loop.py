"""Mem-In-Action 的状态化采集、动态 prompt 和 single-turn Memory-OPD rollout。

本模块同时连接 Mem-In-Action 状态机和 VeRL rollout，但需要严格区分三种对象：

1. ``memory_episode``
   Dataset 从 LoCoMo 等原始来源生成的高层任务描述。它只声明按顺序到来的 session
   和最后需要回答的 QA，不包含运行中的 Cache，也不是模型 prompt。
2. :class:`MemoryOPDStep`
   Collector 在一次模型决策前冻结的状态快照。它包含当前输入、student 可见 Cache、
   teacher-only 完整 memory 和允许动作。一个 episode 会展开成许多 step。
3. VeRL single-turn sample
   ``MemoryOPDStepDataset`` 把一个 step 包装成 VeRL Dataset 契约；
   :class:`MemoryOPDStepAgentLoop` 才在 rollout 时将该 step 渲染、tokenize 并生成动作。

完整数据流如下::

    原始长对话
      -> episode Dataset                 # 静态任务参数
      -> MemoryOPDEpisodeCollector       # 维护 Memory/RAG 动态状态
      -> MemoryOPDStep trace             # 每次决策前的冻结快照
      -> MemoryOPDStepDataset
      -> MemoryOPDStepAgentLoop          # 动态生成 student/teacher prompt
      -> PrivilegeOPDAgentLoopWorker     # 计算同一 response 的 teacher log-prob
      -> RayPrivilegeOPDTrainer

信息可见性是本模块最重要的约束：

- student 永远只看 ``current_input`` 和 ``memory_cache``；
- ``full_memory`` 只允许进入 query/update 的 teacher prompt；
- answer teacher 与 student 使用相同 prompt，避免 teacher 利用隐藏 memory 直接回答；
- teacher 评价的是 student 已经采样出的同一串 response token，不重新生成答案。

不能把完整 episode 直接塞进普通 VeRL AgentLoop。默认 AgentLoop 是一条输入对应一条
输出，而一个 LoCoMo episode 会产生几十到上百个具有不同 Cache 状态的训练 step。
因此 Collector 必须位于 episode Dataset 与 step Trainer 之间，并负责推进状态。
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping
from uuid import uuid4

import ray
import torch
from mem_in_action.agents.magentchat import MAgentChat
from mem_in_action.configs import MAgentConfig
from mem_in_action.llms.openvllm import THINK_CLOSE, split_thinking_answer
from mem_in_action.llms.openai_llm import render_prompt_from_template
from mem_in_action.memory.memory import Memory
from mem_in_action.memory.querys import QueryBuffer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    register,
)
from verl.trainer.meminaction.RLDatasets.schema import validate_memory_episode
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

# 一个高层 task 的目标：session 写入最终必须 update，QA 最终必须 answer。
MemoryTaskMode = Literal["update", "answer"]
# Controller 单轮允许生成的三种动作；query 是非终止动作，update/answer 是终止动作。
MemoryAction = Literal["query", "update", "answer"]
# Teacher 对当前 student response 的信息权限：是否允许额外查看完整长期 memory。
DistillationScope = Literal["privileged", "normal"]

# 只接受完整闭合的动作标签；payload 可以跨行。
_ACTION_PATTERN = re.compile(r"<(query|update|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
# 动作解析前删除完整 think 块，防止思考中的示例动作被真实执行。
_THINK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def visible_memory_response_text(text: str) -> str:
    """返回可执行 action parser 应看到的公开输出。

    Qwen/vLLM 类 tokenizer 解码时可能只保留 ``</think>`` closing tag，或在触达
    response length 时留下未闭合 thinking。这里与 Mem-In-Action LLM 适配器一致：
    有 closing tag 时只解析最后一个 ``</think>`` 之后的文本；存在未闭合 opening tag
    时不执行任何 action；普通非 thinking 输出按原样解析。
    """

    if THINK_CLOSE in text:
        split = split_thinking_answer(text, thinking_enabled=True)
        return split.answer if split.parse_success else ""
    if re.search(r"<think\b[^>]*>", text, re.IGNORECASE):
        return ""
    return _THINK_PATTERN.sub("", text).strip()


def parse_memory_action(text: str) -> tuple[str | None, str]:
    """从模型输出中解析第一个可执行 action。

    ``<think>`` 内容在动作解析前被移除，避免模型在思考文本里举例写出的 action tag
    被误执行。返回值为 ``(action, payload)``；无法解析时返回 ``(None, "")``。

    这里只做语法解析，不判断动作是否被当前 step 允许。允许性检查和真实状态变更由
    :meth:`MemoryOPDEpisodeCollector._collect_task` 完成。
    """

    # 步骤 1：移除不可执行的思考内容，只保留模型最终公开输出。
    visible_text = visible_memory_response_text(text)
    # 步骤 2：在公开输出中查找第一个完整闭合的 Memory action tag。
    match = _ACTION_PATTERN.search(visible_text)
    # 步骤 3：规范化 action 名称并返回 payload；未命中时显式返回无效动作。
    return (match.group(1).lower(), match.group(2).strip()) if match else (None, "")


def iter_memory_episode_tasks(memory_episode: Mapping[str, Any]):
    """按 Mem-In-Action 执行顺序遍历一个 episode 的高层任务。

    Collector 对每个 yielded task 启动一条由多个 ``MemoryOPDStep`` 组成的 trajectory：
    先逐 session 执行 memory creation，再基于创建完成的同一份长期 memory 执行 QA。

    注意，此函数不会执行 Memory 操作，也不会产生模型 prompt。它只是把 episode 中的
    ``sessions`` 和 ``qa`` 统一投影成 Collector 可以消费的 task 参数。
    """

    # 步骤 1：先验证 episode 契约，避免执行到一半才发现 session/QA 结构损坏。
    validate_memory_episode(memory_episode)
    episode_id = memory_episode["episode_id"]
    # 步骤 2：先把所有 session 转换为 update task，按时间顺序写入长期 memory。
    for session in memory_episode["sessions"]:
        yield {
            "episode_id": episode_id,
            "phase": "memory_creation",
            "task_mode": "update",
            "current_input": session["input"],
            "metadata": {
                "session_index": session["session_index"],
                "date_time": session.get("date_time", ""),
            },
        }
    # 步骤 3：session 全部完成后，再把 QA 转换为 answer task，共享同一长期 memory。
    for qa in memory_episode["qa"]:
        yield {
            "episode_id": episode_id,
            "phase": "qa",
            "task_mode": "answer",
            "current_input": qa["question"],
            "metadata": copy.deepcopy(dict(qa)),
        }


def _memory_content(row: Any) -> str:
    """从普通字典或 Mem-In-Action MemoryItem 中读取可渲染文本。"""

    # 步骤 1：兼容已经是纯文本的 memory row。
    if isinstance(row, str):
        return row.strip()
    # 步骤 2：兼容经过 Ray/JSON 序列化后的普通字典。
    if isinstance(row, Mapping):
        return str(row.get("content") or row.get("text") or "").strip()
    # 步骤 3：最后兼容 Mem-In-Action 自定义 MemoryItem 对象。
    return str(getattr(row, "content", "") or "").strip()


def _render_cache(rows: list[Any]) -> str:
    """按照 student 可引用的 ``vid`` 渲染当前 Memory Cache。

    Cache 是检索后暂时暴露给 student 的窗口，而不是完整长期 memory。``vid`` 是
    当前 Cache 中可用于 update 操作的可见 ID，不能替换为 teacher-only memory ID。
    """

    # 步骤 1：空 Cache 使用稳定占位文本，避免模板字段为空导致语义不明确。
    if not rows:
        return "(empty memory cache)"
    lines = []
    # 步骤 2：逐条提取 student 可见内容和当前 Cache vid。
    for index, row in enumerate(rows, start=1):
        content = _memory_content(row)
        if not content:
            continue
        vid = row.get("vid", index) if isinstance(row, Mapping) else getattr(row, "vid", index)
        lines.append(f"[{vid}] {content}")
    # 步骤 3：拼接为 prompt 文本；全是空内容时仍返回空 Cache 占位。
    return "\n".join(lines) or "(empty memory cache)"


def _render_full_memory(rows: list[Any]) -> str:
    """渲染仅 teacher 可见的完整长期 memory。

    完整 memory 中的稳定 ID 只用于 teacher 理解全局状态。teacher 仍必须按照 student
    可见的 action 协议输出 query，或使用 Cache 中可见的 vid 执行 update。
    """

    # 步骤 1：完整 memory 为空时向 teacher 明确说明当前全局状态。
    if not rows:
        return "(complete memory is empty)"
    lines = []
    # 步骤 2：逐条提取内容，并优先使用长期 memory 的稳定 ID。
    for index, row in enumerate(rows, start=1):
        content = _memory_content(row)
        if not content:
            continue
        if isinstance(row, Mapping):
            memory_id = row.get("memory_id") or row.get("rid") or f"m{index:06d}"
        else:
            memory_id = getattr(row, "memory_id", None) or getattr(row, "rid", None) or f"m{index:06d}"
        lines.append(f"[{memory_id}] {content}")
    # 步骤 3：生成只允许进入 privileged teacher prompt 的完整 memory 文本。
    return "\n".join(lines) or "(complete memory is empty)"


def _serialize_memory_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """把 MemoryItem/RAG row 转成可跨 Ray 进程传递的普通字典。

    Collector 所持有的 Memory 对象可能包含自定义类实例，不能假定 Ray、JSON trace
    或 DataLoader worker 都能稳定序列化这些实例。因此 step 快照只保存深拷贝后的
    Python 基础容器，不把活跃 Memory 对象本身送入 Dataset。
    """

    # 步骤 1：创建独立列表，保证快照与活跃 Memory 环境解除引用关系。
    serialized = []
    # 步骤 2：逐条将字典或对象转换为仅含基础 Python 类型的记录。
    for row in rows:
        if isinstance(row, Mapping):
            serialized.append(copy.deepcopy(dict(row)))
            continue
        serialized.append(
            {
                "vid": getattr(row, "vid", None),
                "rid": getattr(row, "rid", None),
                "content": str(getattr(row, "content", "") or ""),
                "score": getattr(row, "score", None),
                "metadata": copy.deepcopy(getattr(row, "metadata", {}) or {}),
            }
        )
    # 步骤 3：返回可安全进入 Ray、DataLoader 和 JSON trace 的快照。
    return serialized


def _normalize_oracle_memory_rows(rows: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    """规范化 oracle 生成的 memory snapshot，使其可直接作为 Cache/full-memory。

    Oracle snapshot 通常来自离线 privileged 模型，可能是字符串列表，也可能是
    ``{"content": ...}``/``{"text": ...}`` 字典列表。这里为每条非空 memory 补齐
    student 可见 ``vid`` 和 teacher 可读 ``rid``，从而无需真实 RAG 后端也能构造
    ``MemoryOPDStep``。
    """

    # 步骤 1：空 snapshot 合法，表示当前没有长期 memory。
    if rows is None:
        return []
    if not isinstance(rows, (list, tuple)):
        raise TypeError(f"oracle memory snapshot 必须是 list/tuple，实际为 {type(rows)!r}")

    # 步骤 2：逐条保留已有元数据，同时把 content/text/string 统一成 content 字段。
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            item = copy.deepcopy(dict(row))
        elif isinstance(row, str):
            item = {"content": row}
        else:
            item = {
                "vid": getattr(row, "vid", None),
                "rid": getattr(row, "rid", None),
                "content": str(getattr(row, "content", "") or ""),
                "score": getattr(row, "score", None),
                "metadata": copy.deepcopy(getattr(row, "metadata", {}) or {}),
            }

        content = _memory_content(item)
        if not content:
            continue
        item["content"] = content
        # 步骤 3：补齐 Cache 可见 ID 和完整 memory 稳定 ID。
        if item.get("vid") is None:
            item["vid"] = len(normalized) + 1
        if item.get("rid") is None and item.get("memory_id") is None:
            item["rid"] = f"oracle_{len(normalized) + 1:06d}"
        normalized.append(item)

    # 步骤 4：返回与活跃对象解耦的普通 Python 容器。
    return normalized


def _terminal_force_instruction(config: MAgentConfig, task_mode: MemoryTaskMode) -> str:
    """返回无需继续 query 时写入 prompt 的 terminal 指令。"""

    if task_mode == "update":
        if config.update_protocol == "patch":
            return config.force_patch_update_instruction
        return config.force_update_instruction
    return config.force_answer_instruction


def _oracle_snapshot_pairs(
    oracle_session_snapshots: list[Any] | tuple[Any, ...],
    num_sessions: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """把每个 session 的 oracle snapshot 转成 ``(before, after)`` 序列。

    支持两种输入：

    - 长度等于 session 数：第一个 session 的 before 默认为空；
    - 长度等于 session 数 + 1：第 0 个 snapshot 显式表示初始 memory。
    """

    snapshots = list(oracle_session_snapshots)
    if len(snapshots) == num_sessions:
        normalized = [[]] + [_normalize_oracle_memory_rows(snapshot) for snapshot in snapshots]
    elif len(snapshots) == num_sessions + 1:
        normalized = [_normalize_oracle_memory_rows(snapshot) for snapshot in snapshots]
    else:
        raise ValueError(
            "oracle_session_snapshots 长度必须等于 sessions 数或 sessions 数 + 1；"
            f"实际 snapshots={len(snapshots)}, sessions={num_sessions}"
        )
    return list(zip(normalized[:-1], normalized[1:], strict=True))


def build_oracle_memory_opd_trace(
    memory_episode: Mapping[str, Any],
    oracle_session_snapshots: list[Any] | tuple[Any, ...],
    *,
    config: MAgentConfig | None = None,
    answer_memory: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """用离线 oracle memory 快照直接构造 single-turn Memory-OPD steps。

    这条路径绕过 ``MemoryOPDEpisodeCollector`` 的完整 student trajectory rollout：

    - update step：student 只看上一个 session 的 oracle snapshot 作为 Cache；
      privileged teacher 的 ``full_memory`` 是当前 session 更新后的 oracle target；
    - answer step：student 直接看最终 oracle memory snapshot，然后只允许 ``answer``。

    因此每个 session/QA 都变成一个 terminal single-turn step，不再需要 query/update/
    answer 串行推进完整轨迹。返回结构仍兼容 ``MemoryOPDStepDataset``。
    """

    # 步骤 1：验证 episode，并准备全局配置与 oracle snapshot 对。
    validate_memory_episode(memory_episode)
    controller_config = config or MAgentConfig()
    sessions = memory_episode["sessions"]
    normalized_snapshots = [
        _normalize_oracle_memory_rows(snapshot)
        for snapshot in list(oracle_session_snapshots)
    ]
    snapshot_pairs = _oracle_snapshot_pairs(oracle_session_snapshots, len(sessions))
    final_oracle_memory = (
        _normalize_oracle_memory_rows(answer_memory)
        if answer_memory is not None
        else (
            copy.deepcopy(snapshot_pairs[-1][1])
            if snapshot_pairs
            else copy.deepcopy(normalized_snapshots[-1] if normalized_snapshots else [])
        )
    )

    # 步骤 2：逐 session 构造 update anchor。Cache 是 before，teacher target 是 after。
    tasks: list[dict[str, Any]] = []
    flat_steps: list[dict[str, Any]] = []
    for session_index, (session, (before_memory, after_memory)) in enumerate(
        zip(sessions, snapshot_pairs, strict=True),
        start=1,
    ):
        task = {
            "episode_id": memory_episode["episode_id"],
            "phase": "memory_creation",
            "task_mode": "update",
            "current_input": session["input"],
            "metadata": {
                "phase": "memory_creation",
                "task_mode": "update",
                "session_index": session.get("session_index", session_index),
                "date_time": session.get("date_time", ""),
                "collection_mode": "oracle_snapshot",
                "oracle_snapshot_before_index": session_index - 1,
                "oracle_snapshot_after_index": session_index,
                "oracle_memory_before_size": len(before_memory),
                "oracle_memory_after_size": len(after_memory),
            },
        }
        step = MemoryOPDStep(
            episode_id=task["episode_id"],
            phase="memory_creation",
            task_mode="update",
            current_input=task["current_input"],
            memory_cache=copy.deepcopy(before_memory),
            full_memory=copy.deepcopy(after_memory),
            force_instruction=_terminal_force_instruction(controller_config, "update"),
            allowed_actions=["update"],
            step_index=0,
            metadata=copy.deepcopy(task["metadata"]),
        )
        record = {
            "memory_step": step.as_dict(),
            "response_text": "",
            "action": "update",
            "payload": "",
            "status": "oracle_anchor",
        }
        tasks.append(
            {
                "task": task,
                "steps": [record],
                "result": {
                    "oracle_memory_before_size": len(before_memory),
                    "oracle_memory_after_size": len(after_memory),
                },
                "final_memory": copy.deepcopy(after_memory),
            }
        )
        flat_steps.append(record)

    # 步骤 3：每个 QA 构造 answer anchor。Answer 不需要 privileged hidden memory。
    for qa in memory_episode["qa"]:
        task = {
            "episode_id": memory_episode["episode_id"],
            "phase": "qa",
            "task_mode": "answer",
            "current_input": qa["question"],
            "metadata": {
                **copy.deepcopy(dict(qa)),
                "phase": "qa",
                "task_mode": "answer",
                "collection_mode": "oracle_snapshot",
                "oracle_snapshot_index": len(sessions),
                "oracle_answer_memory_size": len(final_oracle_memory),
            },
        }
        step = MemoryOPDStep(
            episode_id=task["episode_id"],
            phase="qa",
            task_mode="answer",
            current_input=task["current_input"],
            memory_cache=copy.deepcopy(final_oracle_memory),
            full_memory=copy.deepcopy(final_oracle_memory),
            force_instruction=_terminal_force_instruction(controller_config, "answer"),
            allowed_actions=["answer"],
            step_index=0,
            metadata=copy.deepcopy(task["metadata"]),
        )
        record = {
            "memory_step": step.as_dict(),
            "response_text": "",
            "action": "answer",
            "payload": "",
            "status": "oracle_anchor",
        }
        tasks.append(
            {
                "task": task,
                "steps": [record],
                "result": qa.get("answer"),
                "final_memory": copy.deepcopy(final_oracle_memory),
            }
        )
        flat_steps.append(record)

    # 步骤 4：保持 Collector trace 兼容结构，供 MemoryOPDStepDataset 直接展开。
    return {
        "episode_id": memory_episode["episode_id"],
        "collection_mode": "oracle_snapshot",
        "tasks": tasks,
        "steps": flat_steps,
        "final_memory": final_oracle_memory,
    }


@dataclass(slots=True)
class MemoryOPDStep:
    """一个可独立训练的 Memory Controller 决策状态。

    字段按可见性分为三组：

    - 任务身份：``episode_id``、``phase``、``task_mode``、``step_index``；
    - student 可见状态：``current_input``、``memory_cache``、``history``、
      ``query_feedback``、``force_instruction``、``allowed_actions``；
    - teacher-only 状态：``full_memory``。

    ``full_memory`` 绝不能进入 student prompt。它只在 student 已生成 action 后用于构造
    teacher prompt：

    - query/update：teacher 看到 ``full_memory``，执行 privileged OPD；
    - answer：teacher 与 student 看到完全相同的 prompt，执行普通 OPD。

    Step 是状态快照，不是可变环境。执行 query/update 后，Collector 会根据更新后的
    Memory 环境重新创建下一条 Step，不能原地修改旧 Step 来代表新状态。
    """

    # 跨 session、QA 和展开 step 保持不变的 episode 身份。
    episode_id: str
    # 当前 step 属于写入长期 memory 的 session 阶段，还是读取 memory 的 QA 阶段。
    phase: Literal["memory_creation", "qa"]
    # 当前 trajectory 的合法终止动作；update task 不能 answer，反之亦然。
    task_mode: MemoryTaskMode
    # 当前 session 文本或 QA 问题，是本 step 的主要外部输入。
    current_input: str
    # Student 当前可见的临时检索 Cache；其中 vid 可用于 update patch。
    memory_cache: list[Any] = field(default_factory=list)
    # Teacher-only 的完整长期 memory 快照，绝不能进入 student prompt。
    full_memory: list[Any] = field(default_factory=list)
    # 预留给显式对话历史的字段；当前 single-turn Memory-OPD 默认不使用。
    history: str = "(not applicable)"
    # 当前 task 已执行 query 的反馈摘要，帮助 student 判断是否继续检索。
    query_feedback: str = "(no previous active queries)"
    # 预算耗尽时注入的强制 terminal 指令；普通 step 为空。
    force_instruction: str = ""
    # 当前 step 对 student 公开的真实动作空间。
    allowed_actions: list[MemoryAction] = field(default_factory=list)
    # 当前 task 内从 0 开始的决策序号，不是全局训练 step。
    step_index: int = 0
    # 来源 session/QA 的追踪字段，不参与核心 prompt 协议。
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """在 step 离开 Collector 前验证动作空间与任务模式的一致性。"""

        # 步骤 1：验证 task_mode，决定该 trajectory 最终必须 update 还是 answer。
        if self.task_mode not in {"update", "answer"}:
            raise ValueError(f"未知 task_mode: {self.task_mode!r}")
        # 步骤 2：验证高层阶段，区分 memory creation 和 QA。
        if self.phase not in {"memory_creation", "qa"}:
            raise ValueError(f"未知 phase: {self.phase!r}")
        # 步骤 3：拒绝无法构造有效 prompt 的空输入。
        if not self.current_input.strip():
            raise ValueError("MemoryOPDStep.current_input 不能为空")
        # 步骤 4：调用方未指定动作空间时，默认允许 query 后执行当前 terminal action。
        if not self.allowed_actions:
            self.allowed_actions = ["query", self.task_mode]
        # 步骤 5：确保 step 不会暴露属于另一 task_mode 的 terminal action。
        invalid = set(self.allowed_actions) - {"query", self.task_mode}
        if invalid:
            raise ValueError(f"{self.task_mode} mode 不允许 actions: {sorted(invalid)}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MemoryOPDStep:
        """从 Ray/DataLoader 传来的普通字典恢复类型化 step，并隔离后续修改。"""

        # 步骤 1：深拷贝跨进程字典；步骤 2：构造 dataclass 并自动运行 __post_init__。
        return cls(**copy.deepcopy(dict(value)))

    def as_dict(self) -> dict[str, Any]:
        """转换为可序列化快照，供 trace、Dataset 和 Ray worker 传递。"""

        # 步骤 1：递归展开 dataclass；嵌套列表和字典也会被复制为普通容器。
        return asdict(self)


class MemoryOPDPromptRenderer:
    """复用 Mem-In-Action 固定模板，按动态 step 构造 student/teacher prompt。

    Prompt 模板路径和动作协议来自全局 :class:`MAgentConfig`，不会随 Dataset 样本变化；
    真正变化的是每个 step 的输入、Cache、query feedback 和允许动作。因此 renderer
    必须位于 rollout 侧，而不能在 episode Dataset 中提前渲染或 tokenization。
    """

    def __init__(self, config: MAgentConfig | None = None):
        # 步骤 1：使用显式 rollout 配置；未提供时回退到 Mem-In-Action 默认配置。
        self.config = config or MAgentConfig()

    def _prompt_path(self, task_mode: MemoryTaskMode) -> str:
        """根据 controller 兼容模式选择固定 prompt 模板文件。"""

        # 步骤 1：完整 legacy controller 始终使用旧模板。
        if self.config.controller_mode == "legacy":
            return self.config.legacy_prompt_path
        # 步骤 2：新 controller 也可只为 update task 保留旧模板兼容行为。
        if (
            task_mode == "update"
            and self.config.legacy_update_prompt
            and self.config.update_protocol == "replace-cache"
        ):
            return self.config.legacy_prompt_path
        # 步骤 3：其他情况使用当前统一 controller 模板。
        return self.config.prompt_path

    def _action_protocol(self, allowed_actions: list[MemoryAction]) -> str:
        """只向模型描述当前 step 实际允许执行的动作及其输出语法。"""

        # 步骤 1：从空协议开始，仅加入当前 step 允许的动作。
        actions: list[str] = []
        # 步骤 2：允许 query 时描述多行 query 格式及单次条数上限。
        if "query" in allowed_actions:
            actions.append(
                "`<query>query 1\nquery 2\nquery 3</query>` loads related long-term memories into "
                "Memory Cache. Put one query per line, with at most "
                f"{self.config.max_queries_per_action} non-empty lines."
            )
        # 步骤 3：允许 update 时根据配置描述 replace-cache 或 patch 协议。
        if "update" in allowed_actions:
            if self.config.update_protocol == "replace-cache":
                update = (
                    "Replace-cache protocol: inside <update>, output the entire desired cache "
                    "as numbered lines. Every loaded item omitted from the list is permanently deleted."
                )
            else:
                update = (
                    "Stable patch protocol: inside <update>, emit zero or more operations, one per line: "
                    '<replace id="VID">complete replacement text</replace> to update a loaded item in place; '
                    "<add>new self-contained atomic memory</add> to create an unmatched item; "
                    '<delete id="VID"/> only when the input explicitly proves a loaded item is false or redundant. '
                    "Unmentioned loaded items remain unchanged. Use the visible [VID] from Memory Cache. "
                    "An empty <update></update> is a valid no-op."
                )
            actions.append(f"`<update>operations</update>` modifies long-term memory.\n{update}")
        # 步骤 4：允许 answer 时声明最终回答格式。
        if "answer" in allowed_actions:
            actions.append("`<answer>concise answer</answer>` returns the final answer.")
        # 步骤 5：编号后写入 prompt，帮助模型区分多个候选动作。
        return "\n".join(f"{index}. {action}" for index, action in enumerate(actions, start=1))

    def _task_policy(self, task_mode: MemoryTaskMode) -> str:
        """加载 update 或 answer 的固定策略说明。"""

        # 步骤 1：根据 task_mode 选择策略文件。
        path = self.config.update_policy_path if task_mode == "update" else self.config.answer_policy_path
        # 步骤 2：读取固定策略文本；其内容不由 Dataset 样本决定。
        return Path(path).read_text(encoding="utf-8").strip()

    @staticmethod
    def _to_messages(prompt: Any) -> list[dict[str, str]]:
        """将模板渲染结果统一成 tokenizer/AgentLoop 使用的 chat messages。"""

        # 步骤 1：优先保留模板对象已经提供的多角色 message 结构。
        if hasattr(prompt, "to_messages"):
            return prompt.to_messages()
        # 步骤 2：纯文本模板结果统一包装成单条 user message。
        return [{"role": "user", "content": str(prompt)}]

    def render_student_messages(self, step: MemoryOPDStep) -> list[dict[str, str]]:
        """构造严格 cache-only 的 student prompt。

        这里故意不读取 ``step.full_memory``。调用方应把返回的 messages 交给
        AgentLoop 的 tokenizer；Dataset 中的兼容 ``raw_prompt`` 不参与真实 rollout。
        """

        # 步骤 1：根据 task_mode 读取固定 controller 模板。
        template = Path(self._prompt_path(step.task_mode)).read_text(encoding="utf-8")
        # 步骤 2：只使用 student 可见字段渲染当前动态状态。
        prompt = render_prompt_from_template(
            template,
            history=step.history,
            memory=_render_cache(step.memory_cache),
            input=step.current_input,
            query_feedback=step.query_feedback,
            force_instruction=step.force_instruction,
            action_protocol=self._action_protocol(step.allowed_actions),
            task_policy=self._task_policy(step.task_mode),
        )
        # 步骤 3：统一转换为 AgentLoop 可以应用 chat template 的 messages。
        return self._to_messages(prompt)

    @staticmethod
    def distillation_scope(action: str | None) -> DistillationScope:
        """根据 student 已采样动作决定 teacher 的信息权限。

        先生成 student action、再决定 scope 很重要：否则 Dataset 或 prompt 路由可能
        提前泄露 teacher-only 信息。无效动作按普通 OPD 处理，不给予隐藏 memory。
        """

        # 步骤 1：仅检索和写入动作允许 teacher 利用完整 memory 判断动作质量。
        return "privileged" if action in {"query", "update"} else "normal"

    def render_teacher_messages(
        self,
        step: MemoryOPDStep,
        action: str | None,
        *,
        student_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """在 student action 已知后构造对应 teacher prompt。

        answer 不注入完整 memory，避免 teacher 通过 student 永远不可见的信息直接回答。
        query/update 才注入完整 memory，并明确要求输出仍遵循 student action protocol。
        """

        # 步骤 1：从 student prompt 深拷贝，保证 teacher 注入不会污染 student 输入。
        messages = copy.deepcopy(student_messages or self.render_student_messages(step))
        # 步骤 2：answer/无效动作保持与 student 完全相同的可见上下文。
        if self.distillation_scope(action) == "normal":
            return messages

        # 步骤 3：为 query/update 构造 teacher-only 规则和完整长期 memory。
        is_oracle_update = (
            action == "update"
            and isinstance(step.metadata, Mapping)
            and step.metadata.get("collection_mode") == "oracle_snapshot"
        )
        if is_oracle_update:
            memory_label = "Oracle Target Long-Term Memory after this update (teacher only)"
            memory_rule = (
                "- The target memory below is the desired state after applying Current Input to the "
                "visible Cache.\n"
                "- Score updates by whether they move the visible Cache toward this target while using "
                "only information justified by Current Input and visible Cache.\n"
            )
        else:
            memory_label = "Complete Long-Term Memory (teacher only)"
            memory_rule = (
                "- Use complete memory to judge which missing information should be retrieved next and "
                "whether the visible Cache is sufficient for a correct update.\n"
            )
        privileged_block = (
            "\n\nPrivileged OPD teacher view:\n"
            f"- The student cannot see the {memory_label} below.\n"
            "- Evaluate the same sampled response under the same action protocol.\n"
            f"{memory_rule}"
            "- A query should retrieve useful hidden memory into Cache. An update may reference only "
            "visible Cache IDs and must obey the student update protocol.\n"
            "- Do not copy hidden memory directly into an action that the student could not justify "
            "from Current Input and visible Cache.\n\n"
            f"{memory_label}:\n"
            f"{_render_full_memory(step.full_memory)}"
        )
        # 步骤 4：定位已有 system message；不存在时创建一个。
        system_index = next((index for index, message in enumerate(messages) if message["role"] == "system"), None)
        if system_index is None:
            messages.insert(0, {"role": "system", "content": privileged_block.strip()})
        else:
            messages[system_index]["content"] += privileged_block
        # 步骤 5：返回只供 teacher 计算条件概率的 messages。
        return messages


class MemoryOPDEpisodeCollector:
    """把一个长 episode 顺序展开成多个可独立蒸馏的 single-turn step。

    ``generate_step`` 接收当前 ``MemoryOPDStep``，返回 student 的可见文本输出。生产
    环境中该 callback 应调用 VeRL rollout server；单元测试中可以使用确定性 fake。

    Collector 只负责状态机和 Memory 环境，不负责梯度更新。它会在每次模型调用前冻结
    当前 Cache 与完整 memory 快照，因此 query/update 可以使用 privilege teacher；
    answer 的 teacher prompt 仍由 :class:`MemoryOPDPromptRenderer` 保持为 cache-only。

    ``generate_step`` 是 Collector 与 rollout 系统之间唯一的调用边界：

    - 输入是冻结后的 :class:`MemoryOPDStep`；
    - 输出只是模型可见文本；
    - Collector 解析并执行文本动作，然后创建下一状态；
    - callback 不应私自修改 Collector 中的 ``Memory``。
    """

    def __init__(self, memory: Memory, config: MAgentConfig | None = None):
        # 步骤 1：保存唯一的活跃长期 Memory 环境；它会跨当前 episode 的 task 复用。
        self.memory = memory
        # 步骤 2：加载 controller 和预算配置。
        self.config = config or MAgentConfig()
        # 步骤 3：创建只保存当前 task 检索反馈的临时 QueryBuffer。
        self.query_buffer = QueryBuffer(enabled=self.config.query_feedback)

    def _seed_query_top_n(self, task_mode: MemoryTaskMode) -> int:
        """返回 task 开始前用 ``current_input`` 自动检索的条数。"""

        # 步骤 1：answer task 可使用单独的起始检索条数。
        if task_mode == "answer" and self.config.answer_seed_query_top_n is not None:
            return self.config.answer_seed_query_top_n
        # 步骤 2：其他情况优先使用统一 seed 配置，否则回退到普通 query_top_n。
        return self.config.seed_query_top_n if self.config.seed_query_top_n is not None else self.config.query_top_n

    def _full_memory_rows(self) -> list[dict[str, Any]]:
        """读取并序列化 teacher-only 的完整长期 memory。"""

        # 步骤 1：读取 RAG 后端持有的完整长期 memory；步骤 2：序列化为冻结快照。
        return _serialize_memory_rows(list(getattr(self.memory.rag, "memory", [])))

    def _make_step(
        self,
        task: Mapping[str, Any],
        *,
        step_index: int,
        force_terminal: bool,
    ) -> MemoryOPDStep:
        """冻结当前环境，构造下一次模型决策所需的不可变 step 快照。

        达到步数或 query 预算时，只暴露 terminal action，并通过
        ``force_instruction`` 要求模型结束当前 task。
        """

        # 步骤 1：读取当前 task 的 terminal 类型。
        task_mode: MemoryTaskMode = task["task_mode"]
        # 步骤 2：预算耗尽时禁止继续 query，并注入强制结束指令。
        if force_terminal:
            force_instruction = self._force_terminal_instruction(task_mode)
            allowed_actions: list[MemoryAction] = [task_mode]
        else:
            force_instruction = ""
            allowed_actions = ["query", task_mode]
        # 步骤 3：同时冻结 student Cache 和 teacher-only 完整 memory，形成决策前状态。
        return MemoryOPDStep(
            episode_id=task["episode_id"],
            phase=task["phase"],
            task_mode=task_mode,
            current_input=task["current_input"],
            memory_cache=_serialize_memory_rows(self.memory.items),
            full_memory=self._full_memory_rows(),
            query_feedback=self.query_buffer.render(),
            force_instruction=force_instruction,
            allowed_actions=allowed_actions,
            step_index=step_index,
            metadata=copy.deepcopy(dict(task.get("metadata") or {})),
        )

    def _force_terminal_instruction(self, task_mode: MemoryTaskMode) -> str:
        """返回当前 task 的协议感知强制终止提示。"""

        if task_mode == "update":
            if self.config.update_protocol == "patch":
                return self.config.force_patch_update_instruction
            return self.config.force_update_instruction
        return self.config.force_answer_instruction

    def _run_initial_query(self, task: Mapping[str, Any]) -> None:
        """按配置执行 task 起始检索，使第一步可以看到基础 Cache。"""

        # 步骤 1：计算当前 task 的自动起始检索预算。
        top_n = self._seed_query_top_n(task["task_mode"])
        # 步骤 2：预算大于零时用 current_input 填充第一步可见 Cache。
        if top_n > 0:
            self.memory.query(task["current_input"], top_n=top_n)

    def _run_query(self, payload: str, task_mode: MemoryTaskMode, remaining: int) -> int:
        """执行一个 query action 中允许的多行 query，并返回实际执行条数。"""

        # 步骤 1：update/answer 可使用不同的单条 query 检索深度。
        top_n = self.config.update_query_top_n if task_mode == "update" else self.config.query_top_n
        # 步骤 2：清理模型输出中的空行，并受单动作和剩余总预算双重限制。
        queries = [line.strip() for line in payload.splitlines() if line.strip()]
        executed = queries[: min(self.config.max_queries_per_action, remaining)]
        # 步骤 3：依次执行 query，并记录每条 query 新增了哪些 Cache vid。
        for query in executed:
            before_vids = {getattr(item, "vid", None) for item in self.memory.items}
            hits = self.memory.query(query, top_n=top_n)
            new_vids = [
                item.vid
                for item in self.memory.items
                if getattr(item, "vid", None) not in before_vids
            ]
            self.query_buffer.record(
                query,
                retrieved=len(hits),
                new=len(new_vids),
                dup_skip=len(hits) - len(new_vids),
                new_vids=new_vids,
            )
        # 步骤 4：返回真实执行条数，供 trajectory 累计 query 预算。
        return len(executed)

    def _apply_update(self, payload: str) -> dict[str, Any]:
        """复用 MAgentChat 的解析语义，把 terminal update 写回长期 memory。"""

        # 步骤 1：按全局配置选择 replace-cache 或 patch 更新协议。
        if self.config.update_protocol == "replace-cache":
            # 步骤 2A：解析完整目标 Cache；无法解析时返回错误而不修改 memory。
            texts = MAgentChat._extract_update_texts(payload)
            if not texts:
                result = {"protocol": "replace-cache", "texts": []}
                if payload.strip():
                    result["error"] = "no parseable numbered memory items"
                return result
            # 步骤 3A：将解析后的目标 Cache 批量写回长期 memory。
            self.memory.update_batch(texts)
            return {"protocol": "replace-cache", "texts": texts}

        # 步骤 2B：patch 协议分别解析 replace/add/delete 操作。
        replacements, additions, deletions = MAgentChat._extract_update_patch(payload)
        # 步骤 3B：非空 payload 若没有任何可解析操作，则拒绝静默更新。
        if payload.strip() and not (replacements or additions or deletions):
            return {
                "protocol": "patch",
                "replacements": {},
                "additions": [],
                "deletions": [],
                "error": "no parseable patch operations",
            }
        # 步骤 4B：执行 patch；引用非法 vid 等错误转为 trace 可记录结果。
        try:
            result = self.memory.apply_patch(replacements, additions, deletions)
        except (KeyError, ValueError) as exc:
            return {
                "protocol": "patch",
                "replacements": replacements,
                "additions": additions,
                "deletions": deletions,
                "error": str(exc),
            }
        # 步骤 5B：序列化真实变更项，方便后续分析 update 行为。
        return {
            "protocol": "patch",
            "replacements": replacements,
            "additions": additions,
            "deletions": deletions,
            "replaced": _serialize_memory_rows(result["replaced"]),
            "added": _serialize_memory_rows(result["added"]),
            "deleted": _serialize_memory_rows(result["deleted"]),
        }

    async def _collect_task(
        self,
        task: Mapping[str, Any],
        generate_step: Callable[[MemoryOPDStep], Awaitable[str]],
    ) -> dict[str, Any]:
        """执行一次 update 或 answer trajectory，并返回展开后的 step records。

        每轮严格遵循 ``冻结状态 -> 生成动作 -> 校验动作 -> 修改环境``。record 保存的是
        动作执行前的 step，因此之后即使 Memory 继续变化，teacher 仍能复现采样时状态。
        """

        # 不允许上一个 task 的临时 Cache/query feedback 泄漏进当前 task；长期 memory
        # 不清空，因为 session 创建和后续 QA 正是通过它共享跨 task 信息。
        # 步骤 1：清理上一个 task 的临时状态，并执行当前输入的自动起始检索。
        self.memory.clear_cache()
        self.query_buffer.clear()
        self._run_initial_query(task)

        # 步骤 2：初始化 trajectory 预算、step trace 和 terminal 结果。
        query_rounds = 0
        records: list[dict[str, Any]] = []
        terminal_result: Any = None
        # 最后额外留一步强制 terminal，避免 query 预算耗尽后静默丢弃当前任务。
        for step_index in range(self.config.max_steps + 1):
            # 步骤 3：根据总步数/query 预算决定当前轮是否必须 terminal。
            force_terminal = step_index >= self.config.max_steps or query_rounds >= self.config.max_query_rounds
            # 步骤 4：在动作执行前冻结当前环境，作为本轮可复现训练状态。
            step = self._make_step(task, step_index=step_index, force_terminal=force_terminal)
            # 步骤 5：请求 student 对冻结 step 生成一次动作文本。
            response_text = str(await generate_step(step))
            # 步骤 6：解析动作，并读取当前 step 实际允许的动作集合。
            action, payload = parse_memory_action(response_text)
            allowed = set(step.allowed_actions)
            status = "proposed"
            # 步骤 7：按动作类型校验并推进 Memory 环境；无效动作不修改状态。
            if action not in allowed:
                status = "disallowed" if action is not None else "invalid"
            elif action == "query" and payload:
                executed = self._run_query(
                    payload,
                    task["task_mode"],
                    remaining=self.config.max_query_rounds - query_rounds,
                )
                query_rounds += executed
                status = "query_applied" if executed else "query_empty"
            elif action == "update":
                terminal_result = self._apply_update(payload)
                if (
                    self.config.update_protocol == "patch"
                    and isinstance(terminal_result, Mapping)
                    and terminal_result.get("error")
                    and not force_terminal
                ):
                    status = "invalid_update"
                else:
                    status = "terminal"
            elif action == "answer":
                terminal_result = payload
                status = "terminal"

            # 步骤 8：保存动作执行前状态、模型输出和执行结果，形成可训练 trace。
            records.append(
                {
                    "memory_step": step.as_dict(),
                    "response_text": response_text,
                    "action": action,
                    "payload": payload,
                    "status": status,
                }
            )
            # 步骤 9：terminal 动作结束当前 task；query/无效动作继续创建下一 step。
            if status == "terminal":
                break

        # final_memory 用于 trace/debug；下一个 task 会继续使用同一长期 Memory，
        # 但从干净的临时 Cache 和 QueryBuffer 开始。
        # 步骤 10：冻结 task 完成后的长期 memory，并清理临时状态。
        final_memory = self._full_memory_rows()
        self.memory.clear_cache()
        self.query_buffer.clear()
        return {
            "task": copy.deepcopy(dict(task)),
            "steps": records,
            "result": terminal_result,
            "final_memory": final_memory,
        }

    async def collect(
        self,
        memory_episode: Mapping[str, Any],
        generate_step: Callable[[MemoryOPDStep], Awaitable[str]],
    ) -> dict[str, Any]:
        """顺序执行完整 memory creation + QA episode。

        session task 的 terminal update 会永久改变长期 memory；随后的 QA task 在同一
        Memory 实例上运行，因此可以测试之前创建的记忆是否足以支持回答。
        """

        # 步骤 1：准备保存按执行顺序完成的 task trajectory。
        tasks = []
        # 步骤 2：先执行全部 session update，再执行全部 QA answer。
        for task in iter_memory_episode_tasks(memory_episode):
            tasks.append(await self._collect_task(task, generate_step))
        # 步骤 3：同时返回分 task trace、扁平 step buffer 和 episode 最终 memory。
        return {
            "episode_id": memory_episode["episode_id"],
            "tasks": tasks,
            "steps": [step for task in tasks for step in task["steps"]],
            "final_memory": self._full_memory_rows(),
        }


@register("memory_opd_step")
class MemoryOPDStepAgentLoop(AgentLoopBase):
    """只生成一个 Memory Controller action 的 single-turn AgentLoop。

    该类不执行 query/update，也不推进 episode。Episode Collector 在收到输出后解析
    action、修改 Memory 环境，再用新状态调用下一次 single-turn rollout。

    VeRL Dataset 中的 ``raw_prompt`` 仅用于兼容公共管道。这里实际消费的是
    ``kwargs["memory_step"]``，并在当前 worker 内完成渲染和 tokenization。
    """

    def __init__(self, *args, **kwargs):
        # 步骤 1：让 VeRL AgentLoopBase 初始化 tokenizer、rollout server 和配置。
        super().__init__(*args, **kwargs)
        # 步骤 2：缓存动态 prompt 和单次 action response 的 token 长度上限。
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Prompt 路径和动作协议是全局 rollout 配置，不随 Dataset 样本变化。
        # 步骤 3：从 rollout 配置构造全局 PromptRenderer。
        prompt_config = dict(self.rollout_config.agent.get("memory_opd_prompt", {}) or {})
        self.renderer = MemoryOPDPromptRenderer(MAgentConfig(**prompt_config))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """对一个冻结 step 采样一次动作，并附带 teacher prompt 元数据。"""

        # Dataset 传入的是普通 dict。恢复为 MemoryOPDStep 会再次执行边界校验，避免
        # 损坏的 trace 在生成阶段悄悄进入模型。
        # 步骤 1：恢复并验证 Dataset 传入的冻结 memory_step。
        step = MemoryOPDStep.from_mapping(kwargs["memory_step"])
        # 步骤 2：用固定模板和当前动态状态生成 cache-only student messages。
        student_messages = self.renderer.render_student_messages(step)
        # tokenizer 只在 rollout 阶段使用，因为此刻动态 Cache 和最终 prompt 才已确定。
        # 步骤 3：应用 actor chat template，并检查真实动态 prompt 长度。
        prompt_ids = await self.apply_chat_template(student_messages)
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Memory-OPD student prompt 长度 {len(prompt_ids)} 超过 rollout.prompt_length={self.prompt_length}"
            )

        # 步骤 4：调用 rollout server，从当前 student policy 采样一个 action response。
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        # 步骤 5：裁剪响应、解码文本并解析 student 实际选择的 Memory action。
        response_ids = output.token_ids[: self.response_length]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        action, payload = parse_memory_action(response_text)
        # Teacher prompt 必须在知道 student 实际采样了什么动作后构造，才能决定是否允许
        # 注入 full_memory。Teacher 评价同一 response，不在这里重新生成。
        # 步骤 6：按实际 action 构造 normal 或 privileged teacher messages。
        teacher_messages = self.renderer.render_teacher_messages(
            step,
            action,
            student_messages=student_messages,
        )

        # 步骤 7：将 student sequence、生成指标和 teacher 评分元数据封装给 Worker。
        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            num_turns=2,
            metrics=AgentLoopMetrics(
                generate_sequences=metrics.get("generate_sequences", 0.0),
                num_preempted=output.num_preempted if output.num_preempted is not None else -1,
            ),
            extra_fields={
                # Worker 在 rollout 后处理阶段弹出 teacher_prompt 并计算 teacher logprob。
                # 其余字段用于 trace、调试和按 distillation_scope 分析训练数据。
                "teacher_prompt": teacher_messages,
                "memory_opd_step": step.as_dict(),
                "memory_action": action,
                "memory_action_payload": payload,
                "distillation_scope": self.renderer.distillation_scope(action),
                "turn_scores": [],
                "tool_rewards": [],
                **output.extra_fields,
            },
        )
        # 步骤 8：返回 single-turn rollout；本 AgentLoop 不执行 action 或推进 Memory。
        return result


class PrivilegeOPDAgentLoopWorker(AgentLoopWorker):
    """支持 student/teacher 使用不同 prompt 前缀的 AgentLoopWorker。

    Teacher 在 privileged prompt 下计算同一条 student response 的 token log-prob。
    由于 teacher 与 student prompt 长度不同，这里只取 teacher 的 response 区间，
    再对齐回 student sequence 的 response 位置；prompt 区间不会参与蒸馏损失。

    该 Worker 改变的是 teacher 评分上下文，不改变 student rollout。它也不会根据
    teacher prompt 重新采样 response，否则就不再是对 student 行为的 on-policy
    distillation。
    """

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """纯 OPD 不计算 reward；保留签名以跳过默认 RewardLoop 调用。"""

        # 步骤 1：明确跳过 RewardLoop，teacher distillation loss 是当前唯一训练信号。
        return None

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """在可选 privileged prompt 下计算并对齐 teacher token log-prob。

        ``prompt_ids``/``response_ids`` 属于 student sequence。Teacher 使用自己的
        prompt prefix 加同一份 ``response_ids`` 计算条件概率；最终结果再填回与 student
        sequence 等长的 tensor，使 VeRL 原有蒸馏 loss 可以继续按 ``response_mask`` 工作。
        """

        # 步骤 1：未启用蒸馏或处于 validation 时不请求 teacher server。
        if not self.distillation_enabled or validate:
            return

        # 步骤 2：取出 MemoryOPDStepAgentLoop 构造的自定义 teacher prompt。
        teacher_prompt = output.extra_fields.pop("teacher_prompt", None)
        if teacher_prompt is None:
            # 非 Memory-OPD loop 或未提供自定义 teacher prompt 时，保持 VeRL 默认行为。
            # 步骤 3A：回退到 VeRL 默认的同 prompt teacher 评分路径。
            await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
            return

        # 步骤 3B：从样本字段读取可选 teacher routing key，用于多 teacher 路由。
        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

        # 步骤 4：独立 tokenize teacher prompt；它的 prefix 长度可以与 student 不同。
        teacher_prompt_ids = normalize_token_ids(
            apply_chat_template(
                self.tokenizer,
                teacher_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.config.data.get("apply_chat_template_kwargs", {}),
            )
        )
        # 步骤 5：让 teacher 在自己的上下文下评价 student 已采样的同一串 response token。
        teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            # response_ids 必须直接复用 student 采样结果，不能由 teacher 重新 tokenize 文本，
            # 否则两侧 token 边界可能不同，无法逐 token 蒸馏。
            sequence_ids=teacher_prompt_ids + response_ids,
            routing_key=routing_key,
        )

        # 蒸馏 loss 仅使用 response_mask 覆盖的位置。teacher prompt 部分填充占位值，
        # teacher response 部分则与 student response 逐 token 对齐。
        # 步骤 6：创建与 student sequence 等长的占位 tensor。
        student_sequence_length = len(prompt_ids) + len(response_ids)
        teacher_id_shape = (student_sequence_length, *teacher_ids.shape[1:])
        teacher_logprob_shape = (student_sequence_length, *teacher_logprobs.shape[1:])
        aligned_ids = torch.full(teacher_id_shape, self.tokenizer.pad_token_id, dtype=teacher_ids.dtype)
        aligned_logprobs = torch.zeros(teacher_logprob_shape, dtype=teacher_logprobs.dtype)
        # 步骤 7：只把 teacher response 区间对齐到 student response 位置。
        if response_ids:
            aligned_ids[-len(response_ids) :] = teacher_ids[-len(response_ids) :]
            aligned_logprobs[-len(response_ids) :] = teacher_logprobs[-len(response_ids) :]
        # 步骤 8：写回 VeRL 约定字段，供 actor distillation loss 消费。
        output.extra_fields["teacher_ids"] = aligned_ids
        output.extra_fields["teacher_logprobs"] = aligned_logprobs


class PrivilegeOPDAgentLoopManager(AgentLoopManager):
    """使用 :class:`PrivilegeOPDAgentLoopWorker` 的 VeRL manager。

    配置方式：

    ``actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.trainer.meminaction.agentic_loop.PrivilegeOPDAgentLoopManager``

    该 manager 仍然遵守“一条 step 输入对应一条 step 输出”。LoCoMo episode 的动态
    展开由独立 Episode Collector 完成。
    """

    def __init__(self, *args, **kwargs):
        # 步骤 1：在父 Manager 创建 worker group 前指定自定义 privileged OPD Worker。
        self.agent_loop_workers_class = ray.remote(PrivilegeOPDAgentLoopWorker)
        # 步骤 2：复用 VeRL Manager 完成 Ray worker、rollout server 和调度初始化。
        super().__init__(*args, **kwargs)


__all__ = [
    "MemoryOPDPromptRenderer",
    "MemoryOPDEpisodeCollector",
    "MemoryOPDStep",
    "MemoryOPDStepAgentLoop",
    "PrivilegeOPDAgentLoopManager",
    "PrivilegeOPDAgentLoopWorker",
    "build_oracle_memory_opd_trace",
    "iter_memory_episode_tasks",
    "parse_memory_action",
    "visible_memory_response_text",
]
