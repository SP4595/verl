"""Mem-In-Action 到 VeRL Ray Trainer 的数据边界与校验逻辑。

所有放入同一个 VeRL batch 的样本必须拥有一致的顶层 key。Tensor 字段会由
``collate_fn`` 堆叠，其他字段会进入 ``DataProto.non_tensor_batch``。

本模块校验两种不同层级的对象：

- ``memory_episode``：Collector 的高层输入，声明 sessions 和 QA；
- VeRL sample：Dataset ``__getitem__`` 输出，包含 AgentLoop 路由和 batch 兼容字段。

``MemoryOPDStep`` 的动态状态校验位于 ``agentic_loop.py``，因为它同时被 Collector、
trace 和 AgentLoop 使用，不能与 episode schema 混为一层。
"""

from collections.abc import Mapping
from typing import Any

import torch

# 每个 VeRL 兼容样本都必须返回这些 key。字段会经过默认 collate_fn 分成两类：
#
# - dummy_tensor -> DataProto.batch，用来让纯非 tensor 样本仍有可推断 batch size；
# - 其余字段 -> DataProto.non_tensor_batch，逐样本传递给 AgentLoop/RewardManager。
#
# key 存在不代表内容在所有训练模式中都必需。例如纯 OPD 可以返回
# ``reward_model={}``，但仍应保留该 key，避免混合 batch 的 schema 不一致。
NORMALIZED_SAMPLE_KEYS = (
    "raw_prompt",
    "data_source",
    "reward_model",
    "extra_info",
    "index",
    "agent_name",
    "tools_kwargs",
    "interaction_kwargs",
    "dummy_tensor",
)

# PrivilegeOPDDataset 在基础 API 上增加的来源追踪字段。
PRIVILEGE_OPD_SAMPLE_KEYS = NORMALIZED_SAMPLE_KEYS + (
    "subset_name",
    "subset_index",
    "source_index",
)


def validate_memory_episode(episode: Mapping[str, Any]) -> None:
    """校验 Agentic Collector 消费的完整 memory episode。

    Dataset 只声明需要执行的 session 和 QA；动态 Memory Cache、完整 memory、
    student prompt 和 teacher prompt 都由 rollout 阶段生成。

    顶层字段语义：

    - ``schema_version``：显式版本号，避免旧 trace 被新 Collector 静默误读；
    - ``episode_id``：跨 session、QA 和 step trace 的稳定身份；
    - ``source``：原始数据来源，不用于保存 prompt；
    - ``sessions``：按顺序执行的 memory creation 输入；
    - ``qa``：全部 session 写入后执行的回答任务；
    - ``metadata``：只用于追踪和扩展，不参与核心状态机。
    """

    required_keys = ("schema_version", "episode_id", "source", "sessions", "qa", "metadata")
    missing_keys = [key for key in required_keys if key not in episode]
    if missing_keys:
        raise KeyError(f"memory_episode 缺少字段: {missing_keys}")
    if episode["schema_version"] != 1:
        raise ValueError(f"不支持的 memory_episode.schema_version: {episode['schema_version']!r}")
    if not isinstance(episode["episode_id"], str) or not episode["episode_id"]:
        raise TypeError("memory_episode.episode_id 必须是非空字符串")
    if not isinstance(episode["sessions"], list) or not isinstance(episode["qa"], list):
        raise TypeError("memory_episode.sessions 和 memory_episode.qa 必须是 list")
    if not isinstance(episode["metadata"], Mapping):
        raise TypeError("memory_episode.metadata 必须是 dict")

    for index, session in enumerate(episode["sessions"]):
        if not isinstance(session, Mapping) or not str(session.get("input") or "").strip():
            raise TypeError(f"memory_episode.sessions[{index}] 必须包含非空 input")
    for index, qa in enumerate(episode["qa"]):
        if not isinstance(qa, Mapping) or not str(qa.get("question") or "").strip():
            raise TypeError(f"memory_episode.qa[{index}] 必须包含非空 question")


def validate_sample(sample: Mapping[str, Any], require_ground_truth: bool = True) -> None:
    """校验 Trainer、AgentLoop 和 RewardManager 依赖的公共 Dataset 契约。

    参数:
        sample: ``Dataset.__getitem__`` 返回的单条字典。
        require_ground_truth: 是否要求 ``reward_model.ground_truth``。纯 OPD 或
            自定义 RewardManager 可以关闭；默认规则 RewardManager 通常必须开启。

    ``raw_prompt`` 在普通静态 Dataset 中是真实 prompt；在 Memory episode/step Dataset
    中只是 VeRL 公共管道要求的非空兼容字段。校验器只验证其结构，无法判断它是否会被
    当前 AgentLoop 用于模型输入。
    """

    missing_keys = [key for key in NORMALIZED_SAMPLE_KEYS if key not in sample]
    if missing_keys:
        raise KeyError(f"Dataset 样本缺少 VeRL 必需字段: {missing_keys}")

    raw_prompt = sample["raw_prompt"]
    if not isinstance(raw_prompt, list) or not raw_prompt:
        raise TypeError("raw_prompt 必须是非空 list[dict]，例如 [{'role': 'user', 'content': '...'}]")
    for message_index, message in enumerate(raw_prompt):
        if not isinstance(message, Mapping):
            raise TypeError(f"raw_prompt[{message_index}] 必须是 dict，实际为 {type(message)!r}")
        if "role" not in message or "content" not in message:
            raise KeyError(f"raw_prompt[{message_index}] 必须同时包含 role 和 content")

    data_source = sample["data_source"]
    if not isinstance(data_source, str) or not data_source:
        raise TypeError("data_source 必须是非空字符串")

    reward_model = sample["reward_model"]
    if not isinstance(reward_model, Mapping):
        raise TypeError(f"reward_model 必须是 dict，实际为 {type(reward_model)!r}")
    if require_ground_truth and reward_model.get("ground_truth") is None:
        raise KeyError("reward_model.ground_truth 缺失；规则奖励默认需要该字段")

    for key in ("extra_info", "tools_kwargs", "interaction_kwargs"):
        if not isinstance(sample[key], Mapping):
            raise TypeError(f"{key} 必须是 dict，实际为 {type(sample[key])!r}")

    if not isinstance(sample["dummy_tensor"], torch.Tensor):
        raise TypeError("dummy_tensor 必须是 torch.Tensor，否则 DataProto.batch 可能为空")


__all__ = [
    "NORMALIZED_SAMPLE_KEYS",
    "PRIVILEGE_OPD_SAMPLE_KEYS",
    "validate_memory_episode",
    "validate_sample",
]
