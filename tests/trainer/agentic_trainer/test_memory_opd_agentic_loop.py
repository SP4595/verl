import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

from mem_in_action.configs import MAgentConfig
from verl.trainer.distillation import losses as distillation_losses
from verl.trainer.meminaction.agentic_loop import (
    MemoryOPDEpisodeCollector,
    MemoryOPDPromptRenderer,
    MemoryOPDStep,
    build_oracle_memory_opd_trace,
    iter_memory_episode_tasks,
    parse_memory_action,
    visible_memory_response_text,
)


def _write_prompt_config(tmp_path: Path) -> MAgentConfig:
    prompt_path = tmp_path / "chat.txt"
    update_policy_path = tmp_path / "update.txt"
    answer_policy_path = tmp_path / "answer.txt"
    prompt_path.write_text(
        """
<system>
Protocol: {% action_protocol %}
Policy: {% task_policy %}
</system>
<user>
Cache:
{% memory %}
Input:
{% input %}
Feedback:
{% query_feedback %}
{% force_instruction %}
</user>
""".strip(),
        encoding="utf-8",
    )
    update_policy_path.write_text("Update policy.", encoding="utf-8")
    answer_policy_path.write_text("Answer policy.", encoding="utf-8")
    return MAgentConfig(
        prompt_path=str(prompt_path),
        legacy_prompt_path=str(prompt_path),
        update_policy_path=str(update_policy_path),
        answer_policy_path=str(answer_policy_path),
    )


def test_privilege_is_selected_from_sampled_action(tmp_path):
    renderer = MemoryOPDPromptRenderer(_write_prompt_config(tmp_path))
    step = MemoryOPDStep(
        episode_id="locomo-1",
        phase="qa",
        task_mode="answer",
        current_input="Where does Alice live?",
        memory_cache=[{"vid": 1, "content": "Alice owns a cat."}],
        full_memory=[
            {"rid": "m1", "content": "Alice owns a cat."},
            {"rid": "m2", "content": "Alice lives in Paris."},
        ],
    )

    student = renderer.render_student_messages(step)
    query_teacher = renderer.render_teacher_messages(step, "query", student_messages=student)
    answer_teacher = renderer.render_teacher_messages(step, "answer", student_messages=student)

    assert "Alice lives in Paris." not in str(student)
    assert "Alice lives in Paris." in str(query_teacher)
    assert answer_teacher == student
    assert renderer.distillation_scope("query") == "privileged"
    assert renderer.distillation_scope("update") == "privileged"
    assert renderer.distillation_scope("answer") == "normal"


def test_prompt_protocol_only_lists_current_step_actions(tmp_path):
    renderer = MemoryOPDPromptRenderer(_write_prompt_config(tmp_path))
    normal = MemoryOPDStep(
        episode_id="locomo-1",
        phase="qa",
        task_mode="answer",
        current_input="Where does Alice live?",
        allowed_actions=["query", "answer"],
    )
    forced = MemoryOPDStep(
        episode_id="locomo-1",
        phase="qa",
        task_mode="answer",
        current_input="Where does Alice live?",
        allowed_actions=["answer"],
    )

    normal_text = str(renderer.render_student_messages(normal))
    forced_text = str(renderer.render_student_messages(forced))

    assert "loads related long-term memories" in normal_text
    assert "<answer>concise answer</answer>" in normal_text
    assert "loads related long-term memories" not in forced_text
    assert "1. `<answer>concise answer</answer>`" in forced_text


def test_parse_memory_action_ignores_thinking_block():
    action, payload = parse_memory_action(
        "<think><answer>wrong hidden answer</answer></think><query>Alice location</query>"
    )

    assert action == "query"
    assert payload == "Alice location"


def test_parse_memory_action_uses_text_after_qwen_think_close():
    raw = "hidden <answer>wrong</answer></think>\n<query>Alice location</query>"

    assert visible_memory_response_text(raw) == "<query>Alice location</query>"
    assert parse_memory_action(raw) == ("query", "Alice location")


def test_parse_memory_action_rejects_unclosed_thinking():
    assert parse_memory_action("<think><query>Alice location</query>") == (None, "")


def test_episode_tasks_create_memory_before_running_qa():
    tasks = list(
        iter_memory_episode_tasks(
            {
                "schema_version": 1,
                "episode_id": "locomo-1",
                "source": "locomo",
                "sessions": [
                    {"session_index": 1, "date_time": "day 1", "input": "session one"},
                    {"session_index": 2, "date_time": "day 2", "input": "session two"},
                ],
                "qa": [
                    {
                        "qa_index": 0,
                        "question": "Where does Alice live?",
                        "answer": "Paris",
                        "category": 1,
                        "evidence": [],
                    }
                ],
                "metadata": {},
            }
        )
    )

    assert [task["task_mode"] for task in tasks] == ["update", "update", "answer"]
    assert [task["phase"] for task in tasks] == ["memory_creation", "memory_creation", "qa"]


class _FakeMemory:
    def __init__(self):
        self.rag = SimpleNamespace(memory=[])
        self._cache = []

    @property
    def items(self):
        return deepcopy(self._cache)

    def clear_cache(self):
        self._cache = []

    def query(self, text, top_n=1):
        del text
        self._cache = [
            SimpleNamespace(vid=index, **row)
            for index, row in enumerate(self.rag.memory[:top_n], start=1)
        ]
        return self.items

    def update_batch(self, texts):
        self.rag.memory = [
            {"rid": f"m{index}", "content": text, "metadata": {}}
            for index, text in enumerate(texts, start=1)
        ]
        self.query("", top_n=len(texts))

    def apply_patch(self, replacements, additions, deletions):
        by_vid = {item.vid: item for item in self._cache}
        if any(vid not in by_vid for vid in replacements) or any(vid not in by_vid for vid in deletions):
            raise KeyError("unknown vid")
        replaced = []
        for vid, text in replacements.items():
            by_vid[vid].content = text
            replaced.append(by_vid[vid])
        deleted = [by_vid[vid] for vid in deletions]
        self._cache = [item for item in self._cache if item.vid not in set(deletions)]
        next_vid = max((item.vid for item in self._cache), default=0) + 1
        added = [SimpleNamespace(vid=next_vid + index, content=text) for index, text in enumerate(additions)]
        self._cache.extend(added)
        self.rag.memory = [
            {"rid": f"m{index}", "content": item.content, "metadata": {}}
            for index, item in enumerate(self._cache, start=1)
        ]
        return {"replaced": replaced, "added": added, "deleted": deleted}


def test_episode_collector_expands_long_flow_into_single_turn_steps():
    memory = _FakeMemory()
    collector = MemoryOPDEpisodeCollector(
        memory,  # type: ignore[arg-type]
        MAgentConfig(
            seed_query_top_n=0,
            answer_seed_query_top_n=0,
            max_query_rounds=3,
            max_steps=4,
        ),
    )
    episode = {
        "schema_version": 1,
        "episode_id": "locomo-1",
        "source": "locomo",
        "sessions": [{"session_index": 1, "date_time": "day 1", "input": "Alice moved."}],
        "qa": [
            {
                "qa_index": 0,
                "question": "Where does Alice live?",
                "answer": "Paris",
                "category": 1,
                "evidence": [],
            }
        ],
        "metadata": {},
    }

    async def generate(step):
        if step.task_mode == "update":
            return "<update>1. Alice lives in Paris.</update>"
        if not step.memory_cache:
            return "<query>Alice location</query>"
        return "<answer>Paris</answer>"

    trace = asyncio.run(collector.collect(episode, generate))

    assert [row["action"] for row in trace["steps"]] == ["update", "query", "answer"]
    assert trace["tasks"][0]["result"]["texts"] == ["Alice lives in Paris."]
    assert trace["tasks"][1]["result"] == "Paris"
    assert trace["steps"][1]["memory_step"]["full_memory"][0]["content"] == "Alice lives in Paris."


def test_episode_collector_retries_malformed_patch_update():
    memory = _FakeMemory()
    collector = MemoryOPDEpisodeCollector(
        memory,  # type: ignore[arg-type]
        MAgentConfig(
            update_protocol="patch",
            seed_query_top_n=0,
            max_query_rounds=1,
            max_steps=3,
        ),
    )
    episode = {
        "schema_version": 1,
        "episode_id": "locomo-1",
        "source": "locomo",
        "sessions": [{"session_index": 1, "date_time": "day 1", "input": "Alice moved."}],
        "qa": [],
        "metadata": {},
    }
    responses = iter(
        [
            "<update>nothing to parse</update>",
            "<update><add>Alice lives in Paris.</add></update>",
        ]
    )

    async def generate(step):
        assert "Use only the stable patch protocol" not in step.force_instruction
        return next(responses)

    trace = asyncio.run(collector.collect(episode, generate))

    assert [row["status"] for row in trace["steps"]] == ["invalid_update", "terminal"]
    assert trace["tasks"][0]["result"]["additions"] == ["Alice lives in Paris."]


def test_oracle_snapshot_builder_creates_independent_terminal_steps():
    episode = {
        "schema_version": 1,
        "episode_id": "locomo-1",
        "source": "locomo",
        "sessions": [
            {"session_index": 1, "date_time": "day 1", "input": "Alice moved to Paris."},
            {"session_index": 2, "date_time": "day 2", "input": "Alice adopted a cat."},
        ],
        "qa": [
            {
                "qa_index": 0,
                "question": "Where does Alice live?",
                "answer": "Paris",
                "category": 1,
                "evidence": [],
            }
        ],
        "metadata": {},
    }

    trace = build_oracle_memory_opd_trace(
        episode,
        oracle_session_snapshots=[
            [{"content": "Alice lives in Paris."}],
            [
                {"content": "Alice lives in Paris."},
                {"content": "Alice has a cat."},
            ],
        ],
        config=MAgentConfig(update_protocol="patch"),
    )

    assert trace["collection_mode"] == "oracle_snapshot"
    assert [row["action"] for row in trace["steps"]] == ["update", "update", "answer"]
    assert [row["memory_step"]["allowed_actions"] for row in trace["steps"]] == [
        ["update"],
        ["update"],
        ["answer"],
    ]
    second_update = trace["steps"][1]["memory_step"]
    assert second_update["metadata"]["task_mode"] == "update"
    assert second_update["metadata"]["phase"] == "memory_creation"
    assert second_update["memory_cache"][0]["content"] == "Alice lives in Paris."
    assert [row["content"] for row in second_update["full_memory"]] == [
        "Alice lives in Paris.",
        "Alice has a cat.",
    ]
    answer_step = trace["steps"][2]["memory_step"]
    assert answer_step["metadata"]["task_mode"] == "answer"
    assert answer_step["metadata"]["phase"] == "qa"
    assert answer_step["memory_cache"] == answer_step["full_memory"]
    assert answer_step["metadata"]["answer"] == "Paris"


def test_pure_opd_loss_does_not_require_task_reward_or_advantage(monkeypatch):
    monkeypatch.setattr(
        distillation_losses,
        "distillation_loss",
        lambda *args, **kwargs: (torch.tensor(2.0), {"distillation/test": 1.0}),
    )

    def fail_ppo_loss(*args, **kwargs):
        raise AssertionError("pure OPD must not call PPO task-reward loss")

    monkeypatch.setattr(distillation_losses, "ppo_loss", fail_ppo_loss)
    config = SimpleNamespace()
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            use_task_rewards=False,
            distillation_loss_coef=1.0,
        )
    )

    loss, metrics = distillation_losses.distillation_ppo_loss(
        config=config,
        distillation_config=distillation_config,
        model_output={},
        data={},
    )

    assert loss.item() == 2.0
    assert metrics["distillation/test"] == 1.0
