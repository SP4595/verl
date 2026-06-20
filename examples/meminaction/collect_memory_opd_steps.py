#!/usr/bin/env python3
"""Collect offline Memory-OPD step traces from LoCoMo episodes.

This is the bridge between the episode-level LoCoMo source and the step-level
``MemoryOPDStepDataset`` used by the current pure OPD trainer.

Two collection modes are supported:

- ``trajectory`` runs the Mem-In-Action state machine once and records every
  frozen decision state.
- ``oracle_snapshot`` asks a privileged model to produce one canonical memory
  snapshot after each session, then creates one terminal update step per session
  and one terminal answer step per QA without rolling out the full trajectory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
VERL_ROOT = PROJECT_ROOT / "src" / "verl"
for path in (str(SRC_ROOT), str(VERL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _max_tokens(value: str | None) -> int | None:
    if value is None:
        return 20_000
    lowered = value.strip().lower()
    if lowered in {"", "0", "none", "null", "unlimited"}:
        return None
    return int(value)


def _messages_to_prompt(messages: list[dict[str, str]]):
    from mem_in_action.llms.openai_llm import RenderedChatPrompt

    system_parts: list[str] = []
    user_parts: list[str] = []
    other_parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        else:
            other_parts.append(f"{role}: {content}")
    if other_parts:
        user_parts.extend(other_parts)
    return RenderedChatPrompt(
        system_prompt="\n\n".join(system_parts) or None,
        user_prompt="\n\n".join(user_parts) or None,
    )


def _first_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("oracle snapshot response did not contain a JSON object or array")


def _parse_oracle_memory_snapshot(text: str) -> list[dict[str, Any]]:
    payload = _first_json_payload(text.strip())
    if isinstance(payload, Mapping):
        for key in ("memories", "memory", "entries", "snapshot"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise TypeError(f"oracle snapshot must be a JSON list, got {type(payload)!r}")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            content = item.strip()
            row = {"content": content}
        elif isinstance(item, Mapping):
            row = dict(item)
            content = str(row.get("content") or row.get("text") or "").strip()
            row["content"] = content
        else:
            raise TypeError(f"oracle snapshot entry {index} must be string or object, got {type(item)!r}")
        if content:
            rows.append(row)
    return rows


def _oracle_snapshot_messages(
    episode: Mapping[str, Any],
    session: Mapping[str, Any],
    previous_snapshot: list[dict[str, Any]],
) -> list[dict[str, str]]:
    previous_json = json.dumps(previous_snapshot, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are an omniscient memory curator for a long-dialogue Memory RAG system. "
                "No Memory Cache or retrieval tool is available. Build the complete canonical "
                "long-term memory state directly from the previous oracle snapshot and the current session."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Episode id: {episode['episode_id']}\n\n"
                "Previous oracle memory snapshot (complete state, not a cache):\n"
                f"{previous_json}\n\n"
                "Current session:\n"
                f"{session['input']}\n\n"
                "Return JSON only. Use this schema exactly:\n"
                '{"memories":[{"content":"self-contained atomic memory"}]}\n\n'
                "Rules:\n"
                "- Preserve still-valid prior memories.\n"
                "- Add new durable facts from the current session.\n"
                "- Merge duplicates and update contradictions when the current session makes them clear.\n"
                "- Do not include commentary, markdown, hidden thoughts, or answer text."
            ),
        },
    ]


def _generate_oracle_session_snapshots(
    episode: Mapping[str, Any],
    llm: Any,
) -> list[list[dict[str, Any]]]:
    snapshots: list[list[dict[str, Any]]] = []
    previous_snapshot: list[dict[str, Any]] = []
    for session in episode["sessions"]:
        messages = _oracle_snapshot_messages(episode, session, previous_snapshot)
        raw = str(llm.complete_chat(_messages_to_prompt(messages)))
        previous_snapshot = _parse_oracle_memory_snapshot(raw)
        snapshots.append(previous_snapshot)
    return snapshots


def _build_episode_dataset(args: argparse.Namespace):
    from verl.trainer.meminaction.RLDatasets import LoCoMoPrivilegeSubsetDataset

    config = OmegaConf.create(
        {
            "default_data_source": "locomo_memory_opd",
            "require_ground_truth": True,
            "locomo_skip_category_5": args.locomo_skip_category_5,
            "locomo_max_sessions": args.max_sessions,
            "locomo_max_qa": args.max_qa,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "validate_custom_sample": True,
            "cache_dir": args.cache_dir,
        }
    )
    return LoCoMoPrivilegeSubsetDataset(
        data_files=[str(args.data_file)],
        tokenizer=None,
        config=config,
        max_samples=args.max_samples,
    )


def _build_memory(args: argparse.Namespace, episode_id: str):
    from mem_in_action.configs import MemoryConfig
    from mem_in_action.memory.memory import Memory
    from mem_in_action.utils.log import register_logger

    logger = register_logger(f"memory_opd_collect_{episode_id}")
    collection_name = f"{args.collection_prefix}_{episode_id.replace('-', '_')}_{uuid.uuid4().hex[:8]}"
    return Memory(
        logger=logger,
        enable_log=not args.quiet,
        config=MemoryConfig(
            collection_name=collection_name,
            persist_path=None,
            output_root=str(args.rag_output_dir.resolve()) if args.rag_output_dir else None,
            output_namespace=f"memory_opd_steps/{episode_id}",
            output_run_timestamp=None,
            embedding_model=_none_if_empty(args.embedding_model),
            embedding_base_url=args.embedding_base_url,
            embedding_api_key=args.embedding_api_key,
            embedding_timeout=args.embedding_timeout,
        ),
    )


def _build_controller_config(args: argparse.Namespace):
    from mem_in_action.configs import MAgentConfig

    return MAgentConfig(
        controller_mode=args.controller_mode,
        update_protocol=args.update_protocol,
        legacy_update_prompt=args.legacy_update_prompt,
        query_feedback=args.query_feedback,
        seed_query_top_n=args.seed_query_top_n,
        answer_seed_query_top_n=args.answer_seed_query_top_n,
        query_top_n=args.query_top_n,
        update_query_top_n=args.update_query_top_n,
        max_queries_per_action=args.max_queries_per_action,
        max_query_rounds=args.max_query_rounds,
        max_steps=args.max_steps,
        max_invalid_action_retries=args.max_invalid_action_retries,
    )


def _build_llm(args: argparse.Namespace):
    from mem_in_action.configs import OpenAILLMConfig
    from mem_in_action.llms import build_chat_llm

    return build_chat_llm(
        config=OpenAILLMConfig(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            max_tokens=_max_tokens(args.max_tokens),
            timeout=args.llm_timeout,
            enable_thinking=args.enable_thinking,
        ),
        enable_log=not args.quiet,
    )


async def _collect(args: argparse.Namespace) -> dict[str, int]:
    dataset = _build_episode_dataset(args)
    if args.dry_run:
        sessions = sum(len(row["memory_episode"]["sessions"]) for row in dataset)
        qa = sum(len(row["memory_episode"]["qa"]) for row in dataset)
        estimated_steps = sessions + qa if args.collection_mode == "oracle_snapshot" else 0
        return {"episodes": len(dataset), "tasks": sessions + qa, "steps": estimated_steps}

    from verl.trainer.meminaction.agentic_loop import (
        MemoryOPDEpisodeCollector,
        MemoryOPDPromptRenderer,
        build_oracle_memory_opd_trace,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    controller_config = _build_controller_config(args)
    llm = _build_llm(args)
    renderer = MemoryOPDPromptRenderer(controller_config) if args.collection_mode == "trajectory" else None

    episodes = 0
    steps = 0
    tasks = 0
    with args.out_file.open("w", encoding="utf-8") as handle:
        for row in dataset:
            episode = row["memory_episode"]
            if args.collection_mode == "trajectory":
                assert renderer is not None
                memory = _build_memory(args, episode["episode_id"])
                collector = MemoryOPDEpisodeCollector(memory=memory, config=controller_config)

                async def generate_step(step):
                    messages = renderer.render_student_messages(step)
                    return str(llm.complete_chat(_messages_to_prompt(messages)))

                trace = await collector.collect(episode, generate_step)
            else:
                oracle_session_snapshots = _generate_oracle_session_snapshots(episode, llm)
                trace = build_oracle_memory_opd_trace(
                    episode,
                    oracle_session_snapshots,
                    config=controller_config,
                )
            episodes += 1
            tasks += len(trace["tasks"])
            for step_record in trace["steps"]:
                handle.write(json.dumps(step_record, ensure_ascii=False) + "\n")
                steps += 1
            handle.flush()
    return {"episodes": episodes, "tasks": tasks, "steps": steps}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True, help="LoCoMo JSON file.")
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/memory_opd/train_steps.jsonl"),
        help="JSONL output consumed by MemoryOPDStepDataset.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional VeRL file cache directory.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Maximum LoCoMo episodes to collect.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Maximum sessions per episode.")
    parser.add_argument("--max-qa", type=int, default=None, help="Maximum QA rows per episode.")
    parser.add_argument("--locomo-skip-category-5", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only load and count episodes; do not call services.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--collection-mode",
        choices=["trajectory", "oracle_snapshot"],
        default="trajectory",
        help="trajectory rolls out the full state machine; oracle_snapshot builds one anchored step per session/QA.",
    )

    parser.add_argument("--llm-model", default=os.getenv("MIA_LLM_MODEL", "qwen3.6:35b"))
    parser.add_argument("--llm-base-url", default=os.getenv("MIA_LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("MIA_LLM_API_KEY", "ollama"))
    parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("MIA_LLM_TIMEOUT", "120")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("MIA_TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("MIA_TOP_P", "0.9")))
    parser.add_argument("--frequency-penalty", type=float, default=float(os.getenv("MIA_FREQUENCY_PENALTY", "0.0")))
    parser.add_argument("--presence-penalty", type=float, default=float(os.getenv("MIA_PRESENCE_PENALTY", "0.0")))
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.getenv("MIA_REPETITION_PENALTY", "1.4")),
    )
    parser.add_argument("--max-tokens", default=os.getenv("MIA_MAX_TOKENS", "20000"))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--embedding-model", default=os.getenv("MIA_EMBEDDING_MODEL", "qwen3-embedding:8b"))
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("MIA_EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1"),
    )
    parser.add_argument("--embedding-api-key", default=os.getenv("MIA_EMBEDDING_API_KEY", "ollama"))
    parser.add_argument("--embedding-timeout", type=float, default=float(os.getenv("MIA_EMBEDDING_TIMEOUT", "120")))
    parser.add_argument("--rag-output-dir", type=Path, default=Path("output"))
    parser.add_argument("--collection-prefix", default="memory_opd")

    parser.add_argument("--controller-mode", choices=["separated", "legacy"], default="separated")
    parser.add_argument("--update-protocol", choices=["patch", "replace-cache"], default="patch")
    parser.add_argument("--legacy-update-prompt", action="store_true")
    parser.add_argument("--query-feedback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-query-top-n", type=int, default=5)
    parser.add_argument("--answer-seed-query-top-n", type=int, default=5)
    parser.add_argument("--query-top-n", type=int, default=5)
    parser.add_argument("--update-query-top-n", type=int, default=5)
    parser.add_argument("--max-queries-per-action", type=int, default=3)
    parser.add_argument("--max-query-rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-invalid-action-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=int(os.getenv("MIA_SEED", "42")))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(_collect(args))
    if args.dry_run:
        print(
            f"dry-run: episodes={summary['episodes']} tasks={summary['tasks']} "
            f"estimated_steps={summary['steps']} mode={args.collection_mode}"
        )
    else:
        print(
            f"wrote {summary['steps']} steps from {summary['tasks']} tasks "
            f"across {summary['episodes']} episodes to {args.out_file} "
            f"(mode={args.collection_mode})"
        )


if __name__ == "__main__":
    main()
