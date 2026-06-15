"""静态 prompt 任务使用的 VeRL Ray Trainer 基础 Dataset。

原始 JSON/JSONL/Parquet 推荐每行至少包含::

    {
        "prompt": [{"role": "user", "content": "1 + 1 = ?"}],
        "data_source": "my_dataset",
        "reward_model": {"style": "rule", "ground_truth": "2"},
        "extra_info": {"index": 0}
    }

``reward_model`` 是奖励计算元数据，不是神经网络 Reward Model。纯 OPD 或自定义
RewardManager 可以省略其内容，但同一个 batch 中仍应保留 ``reward_model`` key。

此类继承 VeRL ``RLHFDataset``，适合 prompt 在读取数据时已经确定的普通单轮任务。
它会使用传入 tokenizer 做 prompt 长度过滤。Mem-In-Action 的 ``memory_episode`` 和
``memory_step`` prompt 会随 Cache 状态变化，不应继承此类；对应 Dataset 只保留
``tokenizer`` 构造参数以兼容 VeRL 工厂，并在 AgentLoop 中完成真正 tokenization。
"""

from typing import Any

import torch

from verl.trainer.meminaction.RLDatasets.schema import validate_sample
from verl.utils.dataset.rl_dataset import RLHFDataset


class BaseDataset(RLHFDataset):
    """规范化 VeRL Dataset 输出，并提供单条样本转换扩展点。

    构造函数必须保持以下 VeRL Dataset 工厂 API：

    - ``data_files``：训练或验证文件路径；
    - ``tokenizer``：Actor tokenizer；本静态 Dataset 会用它做长度过滤；
    - ``config``：``config.data``；
    - ``processor``：可选多模态 processor；
    - ``max_samples``：最多加载的样本数。

    ``__getitem__`` 返回普通字典，不直接返回 ``DataProto``。主要字段含义：

    - ``raw_prompt``：AgentLoop 消费的原始聊天消息；
    - ``data_source``：任务、reward verifier 或 teacher 路由键；
    - ``reward_model``：奖励元数据，规则奖励通常需要 ``ground_truth``；
    - ``extra_info``：reward、调试和分析使用的附加信息；
    - ``index``：稳定样本 ID；
    - ``agent_name``：处理该样本的 AgentLoop；
    - ``tools_kwargs`` / ``interaction_kwargs``：工具和多轮交互参数；
    - ``dummy_tensor``：保证当前 DataProto 能推断 batch size 的技术字段。

    子类通常只需重写 :meth:`transform_sample`。不要在 Dataset 中生成
    ``responses``、``teacher_logprobs``、``rm_scores`` 或 ``advantages``，
    这些字段由 Trainer 和 AgentLoop 在训练阶段生成。

    注意：此类的 ``raw_prompt`` 是真实静态 prompt。与之不同，
    ``LoCoMoPrivilegeSubsetDataset`` 和 ``MemoryOPDStepDataset`` 中的 ``raw_prompt``
    只是 VeRL 兼容占位，真实 prompt 由 ``MemoryOPDStepAgentLoop`` 动态生成。
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        self.default_data_source = config.get("default_data_source", "custom")
        self.default_agent_name = config.get("default_agent_name", "single_turn_agent")
        self.default_reward_style = config.get("default_reward_style", "rule")
        self.require_ground_truth = config.get("require_ground_truth", True)
        self.validate_custom_sample = config.get("validate_custom_sample", True)

        # 复用 RLHFDataset 完成文件加载、长度过滤、多模态处理和恢复逻辑。
        # 这里 tokenizer 是真实依赖；只有静态 prompt 才能在 Dataset 阶段正确量长度。
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        """返回一条可直接交给 VeRL 默认 ``collate_fn`` 的规范化样本。

        Tensor 字段将被堆叠到 ``DataProto.batch``，普通 Python 字段会进入
        ``DataProto.non_tensor_batch``，之后逐样本转发给对应 AgentLoop。
        """

        # RLHFDataset 在这里生成 raw_prompt、dummy_tensor 等基础字段。
        sample = dict(super().__getitem__(item))

        # 复制字典，避免后续修改污染 Hugging Face Dataset 中的原始对象。
        extra_info = dict(sample.get("extra_info") or {})
        extra_info.setdefault("index", item)
        sample["extra_info"] = extra_info
        sample["index"] = extra_info["index"]

        # data_source 应保持低基数和稳定，用于任务、reward 或 teacher 路由。
        sample["data_source"] = sample.get("data_source") or self.default_data_source

        # 默认 RewardManager 会读取 reward_model["ground_truth"]。如果原始数据
        # 将答案放在 extra_info 中，这里提供兼容补齐。
        reward_model = dict(sample.get("reward_model") or {})
        if "ground_truth" not in reward_model:
            ground_truth = extra_info.get("ground_truth", extra_info.get("answer"))
            if ground_truth is not None:
                reward_model["ground_truth"] = ground_truth
        if reward_model:
            reward_model.setdefault("style", self.default_reward_style)
        sample["reward_model"] = reward_model

        sample["agent_name"] = sample.get("agent_name") or self.default_agent_name
        sample["tools_kwargs"] = dict(sample.get("tools_kwargs") or extra_info.get("tools_kwargs") or {})
        sample["interaction_kwargs"] = dict(
            sample.get("interaction_kwargs") or extra_info.get("interaction_kwargs") or {}
        )

        # 当前 DataProto 要求 Dataset batch 至少包含一个 Tensor 字段。
        sample.setdefault("dummy_tensor", torch.tensor([0], dtype=torch.uint8))

        sample = self.transform_sample(sample=sample, item=item)
        if not isinstance(sample, dict):
            raise TypeError(f"transform_sample 必须返回 dict，实际返回 {type(sample)!r}")

        if self.validate_custom_sample:
            validate_sample(sample, require_ground_truth=self.require_ground_truth)
        return sample

    def transform_sample(self, sample: dict[str, Any], item: int) -> dict[str, Any]:
        """转换单条规范化样本的扩展点。

        ``sample`` 已经包含 ``raw_prompt``，一般不需要提前应用 chat template；
        模板应用和 tokenization 应由 AgentLoop 完成。
        """

        return sample

    def on_batch_end(self, batch) -> None:
        """训练 batch 完成后的可选回调，用于在线数据或 curriculum learning。"""

        return None


__all__ = ["BaseDataset"]
