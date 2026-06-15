"""Mem-In-Action 的 single-turn Memory-OPD rollout 组件。

整体训练管道分为两层：

1. Episode Collector
   读取 ``LoCoMoPrivilegeSubsetDataset.memory_episode``，复用 Mem-In-Action 的
   Memory/RAG 状态机，按顺序执行 session 创建和 QA。每次需要模型决策时，collector
   生成一个 ``memory_step``，调用本模块的 single-turn AgentLoop，然后把动作执行到
   Memory 环境并进入下一状态。
2. OPD Trainer
   消费 collector 展开的 single-turn step。query/update 使用完整 memory 构造
   privileged teacher prompt；answer 使用与 student 相同的 prompt。

不能把完整 episode 直接塞进普通 VeRL AgentLoop：默认 AgentLoop 是一入一出，而一个
LoCoMo episode 会产生几十到上百个独立训练 step。Episode Collector 必须位于
Dataset 与 Trainer 之间，并负责维护跨 step 的 Memory 状态。
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
from verl.trainer.agentic_trainer.RLDatasets.schema import validate_memory_episode
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
    """解析模型输出中的第一个可见 action tag。"""

    visible_text = _THINK_PATTERN.sub("", text).strip()
    match = _ACTION_PATTERN.search(visible_text)
    return (match.group(1).lower(), match.group(2).strip()) if match else (None, "")


def iter_memory_episode_tasks(memory_episode: Mapping[str, Any]):
    """按 Mem-In-Action 执行顺序遍历一个 episode 的高层任务。

    Collector 对每个 yielded task 启动一条由多个 ``MemoryOPDStep`` 组成的 trajectory：
    先逐 session 执行 memory creation，再基于创建完成的同一份长期 memory 执行 QA。
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
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, Mapping):
        return str(row.get("content") or row.get("text") or "").strip()
    return str(getattr(row, "content", "") or "").strip()


def _render_cache(rows: list[Any]) -> str:
    """按照 student 可引用的 vid 渲染当前 Memory Cache。"""

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
    """把 MemoryItem/RAG row 转成可跨 Ray 进程传递的普通字典。"""

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

    ``full_memory`` 绝不能进入 student prompt。它只在 student 已生成 action 后用于构造
    teacher prompt：

    - query/update：teacher 看到 ``full_memory``，执行 privileged OPD；
    - answer：teacher 与 student 看到完全相同的 prompt，执行普通 OPD。
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
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryOPDStep":
        return cls(**copy.deepcopy(dict(value)))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryOPDPromptRenderer:
    """复用 Mem-In-Action chat template 构造 student/teacher prompt。"""

    def __init__(self, config: MAgentConfig | None = None):
        self.config = config or MAgentConfig()

    def _prompt_path(self, task_mode: MemoryTaskMode) -> str:
        if self.config.controller_mode == "legacy":
            return self.config.legacy_prompt_path
        if task_mode == "update" and self.config.legacy_update_prompt:
            return self.config.legacy_prompt_path
        return self.config.prompt_path

    def _action_protocol(self, allowed_actions: list[MemoryAction]) -> str:
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
        path = self.config.update_policy_path if task_mode == "update" else self.config.answer_policy_path
        return Path(path).read_text(encoding="utf-8").strip()

    @staticmethod
    def _to_messages(prompt: Any) -> list[dict[str, str]]:
        if hasattr(prompt, "to_messages"):
            return prompt.to_messages()
        return [{"role": "user", "content": str(prompt)}]

    def render_student_messages(self, step: MemoryOPDStep) -> list[dict[str, str]]:
        """构造严格 cache-only 的 student prompt。"""

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
        """query/update 使用 privilege；answer 和无效输出使用普通 OPD。"""

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
    """

    def __init__(self, memory: Memory, config: MAgentConfig | None = None):
        self.memory = memory
        self.config = config or MAgentConfig()
        self.query_buffer = QueryBuffer(enabled=self.config.query_feedback)

    def _seed_query_top_n(self, task_mode: MemoryTaskMode) -> int:
        if task_mode == "answer" and self.config.answer_seed_query_top_n is not None:
            return self.config.answer_seed_query_top_n
        return self.config.seed_query_top_n if self.config.seed_query_top_n is not None else self.config.query_top_n

    def _full_memory_rows(self) -> list[dict[str, Any]]:
        return _serialize_memory_rows(list(getattr(self.memory.rag, "memory", [])))

    def _make_step(
        self,
        task: Mapping[str, Any],
        *,
        step_index: int,
        force_terminal: bool,
    ) -> MemoryOPDStep:
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
                getattr(item, "vid")
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
        """执行一次 update 或 answer trajectory，并返回展开后的 step records。"""

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
        """执行完整 memory creation + QA episode。"""

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
        step = MemoryOPDStep.from_mapping(kwargs["memory_step"])
        student_messages = self.renderer.render_student_messages(step)
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
        if not self.distillation_enabled or validate:
            return

        teacher_prompt = output.extra_fields.pop("teacher_prompt", None)
        if teacher_prompt is None:
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

    ``actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.trainer.agentic_trainer.agentic_loop.PrivilegeOPDAgentLoopManager``

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
