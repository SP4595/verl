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

MemoryTaskMode = Literal["update", "answer"]
MemoryAction = Literal["query", "update", "answer"]
DistillationScope = Literal["privileged", "normal"]

_ACTION_PATTERN = re.compile(r"<(query|update|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_THINK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def parse_memory_action(text: str) -> tuple[str | None, str]:
    """从模型输出中解析第一个可执行 action。

    ``<think>`` 内容在动作解析前被移除，避免模型在思考文本里举例写出的 action tag
    被误执行。返回值为 ``(action, payload)``；无法解析时返回 ``(None, "")``。

    这里只做语法解析，不判断动作是否被当前 step 允许。允许性检查和真实状态变更由
    :meth:`MemoryOPDEpisodeCollector._collect_task` 完成。
    """

    visible_text = _THINK_PATTERN.sub("", text).strip()
    match = _ACTION_PATTERN.search(visible_text)
    return (match.group(1).lower(), match.group(2).strip()) if match else (None, "")


def iter_memory_episode_tasks(memory_episode: Mapping[str, Any]):
    """按 Mem-In-Action 执行顺序遍历一个 episode 的高层任务。

    Collector 对每个 yielded task 启动一条由多个 ``MemoryOPDStep`` 组成的 trajectory：
    先逐 session 执行 memory creation，再基于创建完成的同一份长期 memory 执行 QA。

    注意，此函数不会执行 Memory 操作，也不会产生模型 prompt。它只是把 episode 中的
    ``sessions`` 和 ``qa`` 统一投影成 Collector 可以消费的 task 参数。
    """

    validate_memory_episode(memory_episode)
    episode_id = memory_episode["episode_id"]
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

    if isinstance(row, str):
        return row.strip()
    if isinstance(row, Mapping):
        return str(row.get("content") or row.get("text") or "").strip()
    return str(getattr(row, "content", "") or "").strip()


def _render_cache(rows: list[Any]) -> str:
    """按照 student 可引用的 ``vid`` 渲染当前 Memory Cache。

    Cache 是检索后暂时暴露给 student 的窗口，而不是完整长期 memory。``vid`` 是
    当前 Cache 中可用于 update 操作的可见 ID，不能替换为 teacher-only memory ID。
    """

    if not rows:
        return "(empty memory cache)"
    lines = []
    for index, row in enumerate(rows, start=1):
        content = _memory_content(row)
        if not content:
            continue
        vid = row.get("vid", index) if isinstance(row, Mapping) else getattr(row, "vid", index)
        lines.append(f"[{vid}] {content}")
    return "\n".join(lines) or "(empty memory cache)"


def _render_full_memory(rows: list[Any]) -> str:
    """渲染仅 teacher 可见的完整长期 memory。

    完整 memory 中的稳定 ID 只用于 teacher 理解全局状态。teacher 仍必须按照 student
    可见的 action 协议输出 query，或使用 Cache 中可见的 vid 执行 update。
    """

    if not rows:
        return "(complete memory is empty)"
    lines = []
    for index, row in enumerate(rows, start=1):
        content = _memory_content(row)
        if not content:
            continue
        if isinstance(row, Mapping):
            memory_id = row.get("memory_id") or row.get("rid") or f"m{index:06d}"
        else:
            memory_id = getattr(row, "memory_id", None) or getattr(row, "rid", None) or f"m{index:06d}"
        lines.append(f"[{memory_id}] {content}")
    return "\n".join(lines) or "(complete memory is empty)"


def _serialize_memory_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """把 MemoryItem/RAG row 转成可跨 Ray 进程传递的普通字典。

    Collector 所持有的 Memory 对象可能包含自定义类实例，不能假定 Ray、JSON trace
    或 DataLoader worker 都能稳定序列化这些实例。因此 step 快照只保存深拷贝后的
    Python 基础容器，不把活跃 Memory 对象本身送入 Dataset。
    """

    serialized = []
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
    return serialized


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

    episode_id: str
    phase: Literal["memory_creation", "qa"]
    task_mode: MemoryTaskMode
    current_input: str
    memory_cache: list[Any] = field(default_factory=list)
    full_memory: list[Any] = field(default_factory=list)
    history: str = "(not applicable)"
    query_feedback: str = "(no previous active queries)"
    force_instruction: str = ""
    allowed_actions: list[MemoryAction] = field(default_factory=list)
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """在 step 离开 Collector 前验证动作空间与任务模式的一致性。"""

        if self.task_mode not in {"update", "answer"}:
            raise ValueError(f"未知 task_mode: {self.task_mode!r}")
        if self.phase not in {"memory_creation", "qa"}:
            raise ValueError(f"未知 phase: {self.phase!r}")
        if not self.current_input.strip():
            raise ValueError("MemoryOPDStep.current_input 不能为空")
        if not self.allowed_actions:
            self.allowed_actions = ["query", self.task_mode]
        invalid = set(self.allowed_actions) - {"query", self.task_mode}
        if invalid:
            raise ValueError(f"{self.task_mode} mode 不允许 actions: {sorted(invalid)}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MemoryOPDStep:
        """从 Ray/DataLoader 传来的普通字典恢复类型化 step，并隔离后续修改。"""

        return cls(**copy.deepcopy(dict(value)))

    def as_dict(self) -> dict[str, Any]:
        """转换为可序列化快照，供 trace、Dataset 和 Ray worker 传递。"""

        return asdict(self)


class MemoryOPDPromptRenderer:
    """复用 Mem-In-Action 固定模板，按动态 step 构造 student/teacher prompt。

    Prompt 模板路径和动作协议来自全局 :class:`MAgentConfig`，不会随 Dataset 样本变化；
    真正变化的是每个 step 的输入、Cache、query feedback 和允许动作。因此 renderer
    必须位于 rollout 侧，而不能在 episode Dataset 中提前渲染或 tokenization。
    """

    def __init__(self, config: MAgentConfig | None = None):
        self.config = config or MAgentConfig()

    def _prompt_path(self, task_mode: MemoryTaskMode) -> str:
        """根据 controller 兼容模式选择固定 prompt 模板文件。"""

        if self.config.controller_mode == "legacy":
            return self.config.legacy_prompt_path
        if task_mode == "update" and self.config.legacy_update_prompt:
            return self.config.legacy_prompt_path
        return self.config.prompt_path

    def _action_protocol(self, allowed_actions: list[MemoryAction]) -> str:
        """只向模型描述当前 step 实际允许执行的动作及其输出语法。"""

        actions: list[str] = []
        if "query" in allowed_actions:
            actions.append(
                "`<query>query 1\\nquery 2\\nquery 3</query>` loads related long-term memories "
                "into Memory Cache. Put one query per line, with at most "
                f"{self.config.max_queries_per_action} non-empty lines."
            )
        if "update" in allowed_actions:
            if self.config.update_protocol == "replace-cache":
                update = (
                    "Inside <update>, output the entire desired cache as numbered lines. Every loaded "
                    "item omitted from the list is permanently deleted."
                )
            else:
                update = (
                    "Inside <update>, use visible Cache IDs with <replace>, <add>, and <delete> "
                    "operations. Unmentioned loaded items remain unchanged."
                )
            actions.append(f"`<update>operations</update>` modifies long-term memory.\n{update}")
        if "answer" in allowed_actions:
            actions.append("`<answer>concise answer</answer>` returns the final answer.")
        return "\n".join(f"{index}. {action}" for index, action in enumerate(actions, start=1))

    def _task_policy(self, task_mode: MemoryTaskMode) -> str:
        """加载 update 或 answer 的固定策略说明。"""

        path = self.config.update_policy_path if task_mode == "update" else self.config.answer_policy_path
        return Path(path).read_text(encoding="utf-8").strip()

    @staticmethod
    def _to_messages(prompt: Any) -> list[dict[str, str]]:
        """将模板渲染结果统一成 tokenizer/AgentLoop 使用的 chat messages。"""

        if hasattr(prompt, "to_messages"):
            return prompt.to_messages()
        return [{"role": "user", "content": str(prompt)}]

    def render_student_messages(self, step: MemoryOPDStep) -> list[dict[str, str]]:
        """构造严格 cache-only 的 student prompt。

        这里故意不读取 ``step.full_memory``。调用方应把返回的 messages 交给
        AgentLoop 的 tokenizer；Dataset 中的兼容 ``raw_prompt`` 不参与真实 rollout。
        """

        template = Path(self._prompt_path(step.task_mode)).read_text(encoding="utf-8")
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
        return self._to_messages(prompt)

    @staticmethod
    def distillation_scope(action: str | None) -> DistillationScope:
        """根据 student 已采样动作决定 teacher 的信息权限。

        先生成 student action、再决定 scope 很重要：否则 Dataset 或 prompt 路由可能
        提前泄露 teacher-only 信息。无效动作按普通 OPD 处理，不给予隐藏 memory。
        """

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

        messages = copy.deepcopy(student_messages or self.render_student_messages(step))
        if self.distillation_scope(action) == "normal":
            return messages

        privileged_block = (
            "\n\nPrivileged OPD teacher view:\n"
            "- The student cannot see the Complete Long-Term Memory below.\n"
            "- Evaluate the same sampled response under the same action protocol.\n"
            "- Use complete memory to judge which missing information should be retrieved next and "
            "whether the visible Cache is sufficient for a correct update.\n"
            "- A query should retrieve useful hidden memory into Cache. An update may reference only "
            "visible Cache IDs and must obey the student update protocol.\n"
            "- Do not copy hidden memory directly into an action that the student could not justify "
            "from Current Input and visible Cache.\n\n"
            "Complete Long-Term Memory (teacher only):\n"
            f"{_render_full_memory(step.full_memory)}"
        )
        system_index = next((index for index, message in enumerate(messages) if message["role"] == "system"), None)
        if system_index is None:
            messages.insert(0, {"role": "system", "content": privileged_block.strip()})
        else:
            messages[system_index]["content"] += privileged_block
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
        self.memory = memory
        self.config = config or MAgentConfig()
        self.query_buffer = QueryBuffer(enabled=self.config.query_feedback)

    def _seed_query_top_n(self, task_mode: MemoryTaskMode) -> int:
        """返回 task 开始前用 ``current_input`` 自动检索的条数。"""

        if task_mode == "answer" and self.config.answer_seed_query_top_n is not None:
            return self.config.answer_seed_query_top_n
        return self.config.seed_query_top_n if self.config.seed_query_top_n is not None else self.config.query_top_n

    def _full_memory_rows(self) -> list[dict[str, Any]]:
        """读取并序列化 teacher-only 的完整长期 memory。"""

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

        task_mode: MemoryTaskMode = task["task_mode"]
        if force_terminal:
            force_instruction = (
                self.config.force_update_instruction
                if task_mode == "update"
                else self.config.force_answer_instruction
            )
            allowed_actions: list[MemoryAction] = [task_mode]
        else:
            force_instruction = ""
            allowed_actions = ["query", task_mode]
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

    def _run_initial_query(self, task: Mapping[str, Any]) -> None:
        """按配置执行 task 起始检索，使第一步可以看到基础 Cache。"""

        top_n = self._seed_query_top_n(task["task_mode"])
        if top_n > 0:
            self.memory.query(task["current_input"], top_n=top_n)

    def _run_query(self, payload: str, task_mode: MemoryTaskMode, remaining: int) -> int:
        """执行一个 query action 中允许的多行 query，并返回实际执行条数。"""

        top_n = self.config.update_query_top_n if task_mode == "update" else self.config.query_top_n
        queries = [line.strip() for line in payload.splitlines() if line.strip()]
        executed = queries[: min(self.config.max_queries_per_action, remaining)]
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
        return len(executed)

    def _apply_update(self, payload: str) -> dict[str, Any]:
        """复用 MAgentChat 的解析语义，把 terminal update 写回长期 memory。"""

        if self.config.update_protocol == "replace-cache":
            texts = MAgentChat._extract_update_texts(payload)
            if not texts:
                result = {"protocol": "replace-cache", "texts": []}
                if payload.strip():
                    result["error"] = "no parseable numbered memory items"
                return result
            self.memory.update_batch(texts)
            return {"protocol": "replace-cache", "texts": texts}

        replacements, additions, deletions = MAgentChat._extract_update_patch(payload)
        if payload.strip() and not (replacements or additions or deletions):
            return {
                "protocol": "patch",
                "replacements": {},
                "additions": [],
                "deletions": [],
                "error": "no parseable patch operations",
            }
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
        self.memory.clear_cache()
        self.query_buffer.clear()
        self._run_initial_query(task)

        query_rounds = 0
        records: list[dict[str, Any]] = []
        terminal_result: Any = None
        # 最后额外留一步强制 terminal，避免 query 预算耗尽后静默丢弃当前任务。
        for step_index in range(self.config.max_steps + 1):
            force_terminal = step_index >= self.config.max_steps or query_rounds >= self.config.max_query_rounds
            step = self._make_step(task, step_index=step_index, force_terminal=force_terminal)
            response_text = str(await generate_step(step))
            action, payload = parse_memory_action(response_text)
            allowed = set(step.allowed_actions)
            status = "proposed"
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
                status = "terminal"
            elif action == "answer":
                terminal_result = payload
                status = "terminal"

            records.append(
                {
                    "memory_step": step.as_dict(),
                    "response_text": response_text,
                    "action": action,
                    "payload": payload,
                    "status": status,
                }
            )
            if status == "terminal":
                break

        # final_memory 用于 trace/debug；下一个 task 会继续使用同一长期 Memory，
        # 但从干净的临时 Cache 和 QueryBuffer 开始。
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

        tasks = []
        for task in iter_memory_episode_tasks(memory_episode):
            tasks.append(await self._collect_task(task, generate_step))
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
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Prompt 路径和动作协议是全局 rollout 配置，不随 Dataset 样本变化。
        prompt_config = dict(self.rollout_config.agent.get("memory_opd_prompt", {}) or {})
        self.renderer = MemoryOPDPromptRenderer(MAgentConfig(**prompt_config))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """对一个冻结 step 采样一次动作，并附带 teacher prompt 元数据。"""

        # Dataset 传入的是普通 dict。恢复为 MemoryOPDStep 会再次执行边界校验，避免
        # 损坏的 trace 在生成阶段悄悄进入模型。
        step = MemoryOPDStep.from_mapping(kwargs["memory_step"])
        student_messages = self.renderer.render_student_messages(step)
        # tokenizer 只在 rollout 阶段使用，因为此刻动态 Cache 和最终 prompt 才已确定。
        prompt_ids = await self.apply_chat_template(student_messages)
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Memory-OPD student prompt 长度 {len(prompt_ids)} 超过 rollout.prompt_length={self.prompt_length}"
            )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        response_ids = output.token_ids[: self.response_length]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        action, payload = parse_memory_action(response_text)
        # Teacher prompt 必须在知道 student 实际采样了什么动作后构造，才能决定是否允许
        # 注入 full_memory。Teacher 评价同一 response，不在这里重新生成。
        teacher_messages = self.renderer.render_teacher_messages(
            step,
            action,
            student_messages=student_messages,
        )

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

        if not self.distillation_enabled or validate:
            return

        teacher_prompt = output.extra_fields.pop("teacher_prompt", None)
        if teacher_prompt is None:
            # 非 Memory-OPD loop 或未提供自定义 teacher prompt 时，保持 VeRL 默认行为。
            await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
            return

        routing_key = None
        if sample_kwargs is not None:
            routing_value = sample_kwargs.get(self.teacher_key)
            if routing_value is not None:
                routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value

        teacher_prompt_ids = normalize_token_ids(
            apply_chat_template(
                self.tokenizer,
                teacher_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.config.data.get("apply_chat_template_kwargs", {}),
            )
        )
        teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
            # response_ids 必须直接复用 student 采样结果，不能由 teacher 重新 tokenize 文本，
            # 否则两侧 token 边界可能不同，无法逐 token 蒸馏。
            sequence_ids=teacher_prompt_ids + response_ids,
            routing_key=routing_key,
        )

        # 蒸馏 loss 仅使用 response_mask 覆盖的位置。teacher prompt 部分填充占位值，
        # teacher response 部分则与 student response 逐 token 对齐。
        student_sequence_length = len(prompt_ids) + len(response_ids)
        teacher_id_shape = (student_sequence_length, *teacher_ids.shape[1:])
        teacher_logprob_shape = (student_sequence_length, *teacher_logprobs.shape[1:])
        aligned_ids = torch.full(teacher_id_shape, self.tokenizer.pad_token_id, dtype=teacher_ids.dtype)
        aligned_logprobs = torch.zeros(teacher_logprob_shape, dtype=teacher_logprobs.dtype)
        if response_ids:
            aligned_ids[-len(response_ids) :] = teacher_ids[-len(response_ids) :]
            aligned_logprobs[-len(response_ids) :] = teacher_logprobs[-len(response_ids) :]
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
        self.agent_loop_workers_class = ray.remote(PrivilegeOPDAgentLoopWorker)
        super().__init__(*args, **kwargs)


__all__ = [
    "MemoryOPDPromptRenderer",
    "MemoryOPDEpisodeCollector",
    "MemoryOPDStep",
    "MemoryOPDStepAgentLoop",
    "PrivilegeOPDAgentLoopManager",
    "PrivilegeOPDAgentLoopWorker",
    "iter_memory_episode_tasks",
    "parse_memory_action",
]
