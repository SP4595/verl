"""Agentic Trainer 的自定义 Dataset 与 Reward Function 基础模板。

这个文件可以同时承担两个职责：

1. 作为 ``data.custom_cls`` 指向的自定义 Dataset。
2. 作为 ``reward.custom_reward_function.path`` 指向的自定义奖励函数文件。

推荐配置示例::

    data.custom_cls.path=/absolute/path/to/BaseDatasets.py
    data.custom_cls.name=BaseDataset

    reward.custom_reward_function.path=/absolute/path/to/BaseDatasets.py
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
    +reward.custom_reward_function.reward_kwargs.format_score=0.1

如果要使用批量奖励函数，则改为::

    reward.custom_reward_function.name=compute_score_batched
    reward.reward_manager.name=batch

``config.data`` 和 ``reward.custom_reward_function`` 通常是结构化配置。新增本文件
定义的配置项时，Hydra 命令行需要使用 ``+``，例如::

    +data.default_data_source=my_dataset
    +data.default_agent_name=single_turn_agent
    +data.require_ground_truth=True

原始数据与 Dataset 返回值不是同一个 schema。建议按下面三层理解：

第一层：磁盘中的原始数据
------------------------

原始 Parquet/JSON/JSONL 每行推荐包含::

    {
        "prompt": [{"role": "user", "content": "1 + 1 = ?"}],
        "data_source": "my_math_dataset",
        "reward_model": {"style": "rule", "ground_truth": "2"},
        "extra_info": {"index": 0}
    }

各字段含义：

- ``prompt``：**必需**。原始聊天消息，类型为 ``list[dict]``。每条消息至少包含
  ``role`` 和 ``content``。Dataset 会把它处理成 AgentLoop 使用的
  ``raw_prompt``。不要预先 tokenize，也不要提前应用 chat template。
- ``data_source``：**条件必需，强烈建议始终提供**。用于区分任务/数据集、选择
  reward verifier、聚合验证指标；多教师 OPD 中还可作为教师路由键。其值应稳定，
  例如 ``"openai/gsm8k"``，不要使用每条样本不同的随机值。
- ``reward_model``：**条件必需**，它只是“奖励计算元数据”，不等于真正的神经网络
  Reward Model。默认 ``naive``、``dapo``、``gdpo`` 等 RewardManager 会读取
  ``reward_model["ground_truth"]``，所以使用这些 Manager 时必须提供。纯 OPD、
  环境直接给 reward、或者自定义 RewardManager 完全不读取它时可以省略。
- ``reward_model.style``：**可选**。描述奖励风格，默认模板中写作 ``"rule"``。
  当前很多 RewardManager 不直接使用它，但保留该字段有助于标识奖励来源。
- ``reward_model.ground_truth``：**条件必需**。规则奖励的参考答案，可以是字符串、
  数字、列表、测试用例或任务自定义结构。其结构必须与 ``compute_score`` 匹配。
- ``extra_info``：**可选但推荐提供**。存放 reward、调试和数据分析需要的任务元数据，
  例如原题、测试用例、难度、split、数值误差阈值。不要放大体积对象或不可序列化对象。
- ``extra_info.index``：**可选**。原始样本 ID，用于 trace 和问题定位；缺失时本类
  使用 Dataset 下标补齐。
- ``agent_name``：**可选**。选择处理该样本的 AgentLoop，例如
  ``"single_turn_agent"`` 或 ``"tool_agent"``。缺失时使用默认 AgentLoop。
- ``extra_info.tools_kwargs``：**工具任务条件必需**。按工具名称保存创建、执行、奖励
  等参数。普通单轮任务省略即可。
- ``extra_info.interaction_kwargs``：**交互任务条件必需**。传递给自定义多轮交互逻辑。
- ``images``、``videos``、``audios``：**多模态任务条件必需**。字段名可由
  ``config.data.image_key/video_key/audio_key`` 修改，prompt 中需包含对应占位符
  或结构化多模态内容。
- 其他自定义字段：可以保留，但默认 collate 后通常会进入
  ``DataProto.non_tensor_batch`` 并可能被传给 AgentLoop。字段应尽量小且可序列化。

第二层：``Dataset.__getitem__`` 返回值
-------------------------------------

Dataset 的 ``__getitem__`` 不应直接返回 ``DataProto``，而应返回一个普通字典：

- ``torch.Tensor`` 字段会被默认 ``collate_fn`` 堆叠到 ``DataProto.batch``。
- 其他字段会被转换为 ``np.ndarray(dtype=object)``，进入
  ``DataProto.non_tensor_batch``。

本类会将每条样本规范化为以下主要字段：

- ``raw_prompt``：**AgentLoop 必需**。由原始 ``prompt`` 构造，SingleTurnAgentLoop
  和 ToolAgentLoop 都直接读取该字段。它仍然是消息列表，不是 token IDs。
- ``data_source``：**基础字段**。缺失时补为 ``default_data_source``。使用多教师
  OPD 时，它的值必须与配置的 teacher routing key 对应。
- ``reward_model``：**稳定 schema 字段，内容条件必需**。本类始终返回一个字典；
  无奖励元数据时允许是 ``{}``。默认规则 RewardManager 仍要求其中存在
  ``ground_truth``。
- ``extra_info``：**基础字段**。始终返回字典，供自定义 reward function 使用。
- ``index``：**基础字段**。从 ``extra_info.index`` 提取，供 rollout trace 使用。
- ``agent_name``：**基础字段**。决定每条样本由哪个 AgentLoop 执行。
- ``tools_kwargs``：**基础字段**。从 ``extra_info.tools_kwargs`` 提取；无工具时为
  ``{}``，以保证 batch 中所有样本 key 一致。
- ``interaction_kwargs``：**基础字段**。从 ``extra_info.interaction_kwargs``
  提取；无交互参数时为 ``{}``。
- ``dummy_tensor``：**当前 DataProto 实现的技术必需字段**。它没有业务含义，只是
  保证 Dataset batch 至少包含一个 Tensor，使 Trainer 能确定 batch size。
- 原始数据其他字段：``RLHFDataset`` 通常会继续保留。只有确实会被 AgentLoop、
  RewardManager 或日志使用的字段才应保留，避免 Ray 传输大量无用数据。

同一个 batch 中的样本必须拥有一致的 key。Tensor 字段还必须具有一致 shape，
否则默认 ``collate_fn`` 无法正确组成 batch。

第三层：Trainer/rollout 自动生成的字段
------------------------------------

以下字段不应由 Dataset 构造，Trainer 和 AgentLoop 会在 rollout 后生成：

- ``uid``：Trainer 为每个 prompt 自动生成的分组 ID；
- ``prompts``、``responses``、``input_ids``、``attention_mask``、
  ``position_ids``、``response_mask``：tokenize、生成和 padding 后的 Tensor；
- ``rollout_log_probs``：开启 rollout log-prob 计算时生成；
- ``teacher_ids``、``teacher_logprobs``：启用 OPD/教师模型时生成；
- ``rm_scores``、``token_level_rewards``、``advantages``、``returns``：奖励与
  算法阶段生成；
- ``multi_modal_inputs``：多模态 AgentLoop 根据 ``raw_prompt`` 和 processor 生成。

常见任务设计
------------

- **规则奖励 RL**：提供 ``data_source`` 和 ``reward_model.ground_truth``，修改
  :func:`compute_score`；使用 ``reward.reward_manager.name=naive`` 或 ``dapo``。
- **纯 OPD，不使用 task reward**：可省略原始 ``reward_model``，配置
  ``+data.require_ground_truth=False``。但当前 ``RayOPDTrainer`` 即使
  ``distillation.distillation_loss.use_task_rewards=False`` 仍会运行 reward
  pipeline，因此还必须使用“不读取 ground_truth 且能返回合法 rm_scores”的自定义
  RewardManager；否则应继续提供占位 ground truth。
- **环境/工具直接产生奖励**：通过 ``agent_name``、``tools_kwargs`` 或自定义
  AgentLoop 产生 reward；如不需要 ground truth，同样关闭 ground truth 校验并使用
  匹配的 RewardManager。
- **多任务/多教师**：保持 ``data_source`` 取值稳定；在 ``compute_score`` 中按
  ``data_source`` 分发 verifier，并确保多教师路由配置能够识别这些值。
- **在线数据或 curriculum learning**：继承 :class:`BaseDataset` 并重写
  :meth:`on_batch_end`，根据训练完成后的 batch 更新 Dataset 自身状态。

返回字段的设计原则
------------------

设计自定义 Dataset 时建议遵守以下边界：

1. **控制流程字段放顶层**：``raw_prompt``、``data_source``、``agent_name``、
   ``tools_kwargs`` 等会被 AgentLoop、Trainer 或路由逻辑直接读取，应放在样本顶层。
2. **奖励参考信息分类存放**：主要参考答案放 ``reward_model.ground_truth``；
   verifier 的附加参数、测试用例和分析标签放 ``extra_info``。
3. **同批次 schema 必须稳定**：可选字段也应为每条样本提供同类型默认值，例如
   ``reward_model={}``、``tools_kwargs={}``，不要只在部分样本中添加 key。
4. **Dataset 不生成训练阶段字段**：不要在 Dataset 中预计算 ``uid``、
   ``responses``、``teacher_logprobs``、``rm_scores`` 或 ``advantages``。
5. **路由字段保持低基数和稳定性**：``data_source``、``agent_name`` 等用于分组和
   路由，不应包含随机 ID；样本唯一 ID 应放在 ``index`` 或 ``extra_info``。
6. **避免传输大对象**：默认非 Tensor 字段会通过 NumPy object array 和 Ray 传输。
   大型测试资源应保存路径/ID，而不是在每条样本中复制完整对象。
7. **不要提前应用 chat template**：原始文件保存 ``prompt``，Dataset 输出
   ``raw_prompt``；真正的模板应用和 tokenization 由 AgentLoop 完成。

本类规范化后的典型返回值如下::

    {
        # 原始列通常仍会保留；prompt 主要用于数据追踪，rollout 实际读取 raw_prompt。
        "prompt": [{"role": "user", "content": "1 + 1 = ?"}],
        "raw_prompt": [{"role": "user", "content": "1 + 1 = ?"}],

        # 路由、奖励和分析字段。
        "data_source": "my_math_dataset",
        "reward_model": {"style": "rule", "ground_truth": "2"},  # 无需 ground truth 时可为 {}
        "extra_info": {"index": 0, "difficulty": "easy"},
        "index": 0,

        # AgentLoop 控制字段。
        "agent_name": "single_turn_agent",
        "tools_kwargs": {},
        "interaction_kwargs": {},

        # 无业务含义，仅保证 DataProto.batch 非空。
        "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
    }

本模板继承 VeRL 的 ``RLHFDataset``，因此继续复用以下能力：

- 加载 Parquet、JSON、JSONL 文件；
- prompt 长度过滤；
- 多模态 prompt 处理；
- checkpoint 恢复；
- 默认 AgentLoop 所需的 ``raw_prompt``、``dummy_tensor`` 等字段生成。

通常只需要继承 :class:`BaseDataset` 并重写 :meth:`transform_sample`，
以及按任务修改 :func:`compute_score` 即可。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset

# BaseDataset 会始终规范化并返回这些 key。
#
# 注意：key 存在不代表其内容在所有训练模式中都必需。例如纯 OPD 可以让
# reward_model={}，但保持 key 存在可以避免同一个 batch 内 schema 不一致。
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


class BaseDataset(RLHFDataset):
    """适用于 RayPPOTrainer/RayOPDTrainer 的可扩展 Dataset 基类。

    ``create_rl_dataset`` 会使用如下参数实例化自定义 Dataset，因此自定义类的
    构造函数必须保持兼容：

    - ``data_files``：训练或验证文件路径；
    - ``tokenizer``：Actor 使用的 tokenizer；
    - ``config``：``config.data``；
    - ``processor``：多模态 processor，纯文本任务通常为 ``None``；
    - ``max_samples``：最多加载多少条样本。

    本类首先调用 ``RLHFDataset.__getitem__`` 构造 VeRL 标准样本，然后补齐
    AgentLoop、RewardManager 和多教师蒸馏常用字段，最后进行尽早校验。

    可在 ``config.data`` 中添加以下自定义配置：

    - ``default_data_source``：原始数据没有 ``data_source`` 时使用的默认值；
    - ``default_agent_name``：默认 AgentLoop 名称；
    - ``default_reward_style``：默认写入 ``reward_model.style`` 的值；
    - ``require_ground_truth``：是否强制要求 ground truth，默认 ``True``；
    - ``validate_custom_sample``：是否在每次取样时校验 schema，默认 ``True``。
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

        # 复用 VeRL 默认实现完成文件加载、长度过滤、多模态处理和恢复逻辑。
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        """返回一条可直接交给 VeRL 默认 ``collate_fn`` 的样本字典。

        返回值中的顶层字段不是随意命名：

        - ``raw_prompt`` 和 ``agent_name`` 决定 AgentLoop 如何执行；
        - ``data_source`` 决定任务/教师/verifier 路由；
        - ``reward_model`` 和 ``extra_info`` 决定 reward 如何计算；
        - ``index`` 用于 trace；
        - ``tools_kwargs`` 和 ``interaction_kwargs`` 控制工具及多轮交互；
        - ``dummy_tensor`` 满足当前 DataProto 的 batch-size 推断约束。

        注意：同一个 batch 中每条样本应具有相同的 key。默认 ``collate_fn``
        会逐 key 收集值；如果某个 key 只存在于部分样本中，构造 ``DataProto``
        时会因为该字段长度与 batch size 不一致而失败。
        """

        # RLHFDataset 在这里生成 raw_prompt、dummy_tensor、tools_kwargs 等字段。
        sample = dict(super().__getitem__(item))

        # extra_info 会传给自定义 reward function。复制字典，避免后续修改污染
        # HuggingFace Dataset 内部保存的原始对象。
        extra_info = dict(sample.get("extra_info") or {})
        extra_info.setdefault("index", item)
        sample["extra_info"] = extra_info
        sample["index"] = extra_info["index"]

        # data_source 有三个常见用途：
        # 1. 选择默认 reward function；
        # 2. 区分验证指标；
        # 3. 多教师 OPD 中作为 distillation.teacher_key 的路由值。
        sample["data_source"] = sample.get("data_source") or self.default_data_source

        # 默认 RewardManager 会读取 reward_model["ground_truth"]。如果原始数据
        # 没有 reward_model，允许从 extra_info 中的 ground_truth/answer 补齐。
        #
        # 对纯 OPD、环境奖励或自定义 RewardManager，reward_model 可以为空字典。
        # 只有确实存在奖励元数据时才补 style，避免把无奖励样本误标成 rule reward。
        reward_model = dict(sample.get("reward_model") or {})
        if "ground_truth" not in reward_model:
            ground_truth = extra_info.get("ground_truth", extra_info.get("answer"))
            if ground_truth is not None:
                reward_model["ground_truth"] = ground_truth
        if reward_model:
            reward_model.setdefault("style", self.default_reward_style)
        sample["reward_model"] = reward_model

        # agent_name 决定该样本由哪个 AgentLoop 执行。不提供时使用配置中的
        # actor_rollout_ref.rollout.agent.default_agent_loop；这里显式补齐可以让
        # 数据内容和实际执行逻辑更容易排查。
        sample["agent_name"] = sample.get("agent_name") or self.default_agent_name

        # ToolAgentLoop 和自定义多轮 AgentLoop 会从这两个字段读取运行参数。
        # 即使当前任务不用工具，也建议每条样本都返回空字典，保证 batch schema
        # 一致。
        sample["tools_kwargs"] = dict(sample.get("tools_kwargs") or extra_info.get("tools_kwargs") or {})
        sample["interaction_kwargs"] = dict(
            sample.get("interaction_kwargs") or extra_info.get("interaction_kwargs") or {}
        )

        # 当前 DataProto 仍要求 Dataset batch 至少包含一个 Tensor 字段。
        # RLHFDataset 已经添加 dummy_tensor，此处防御性补齐，避免子类转换时删除。
        sample.setdefault("dummy_tensor", torch.tensor([0], dtype=torch.uint8))

        # 子类可在这里添加任务专属字段、改写 prompt 或改变 reward metadata。
        sample = self.transform_sample(sample=sample, item=item)
        if not isinstance(sample, dict):
            raise TypeError(f"transform_sample 必须返回 dict，实际返回 {type(sample)!r}")

        if self.validate_custom_sample:
            validate_sample(sample, require_ground_truth=self.require_ground_truth)
        return sample

    def transform_sample(self, sample: dict[str, Any], item: int) -> dict[str, Any]:
        """子类自定义单条样本的主要扩展点。

        默认实现不做修改。典型用法：

        .. code-block:: python

            class MyDataset(BaseDataset):
                def transform_sample(self, sample, item):
                    sample["extra_info"]["difficulty"] = "hard"
                    sample["data_source"] = "my_math_dataset"
                    return sample

        参数 ``sample`` 已经包含 ``raw_prompt``，因此一般不需要再次调用 chat
        template。Chat template 和 tokenization 会由 AgentLoop 在 rollout 时完成。
        """

        return sample

    def on_batch_end(self, batch) -> None:
        """训练 batch 完成后的可选回调。

        Ray Trainer 在每个训练 batch 结束后检测并调用此方法。默认实现为空。
        如果要实现 curriculum learning、在线数据池更新或失败样本重采样，可以在
        子类中重写。传入的 ``batch`` 已包含 rollout、reward 和训练阶段产生的
        字段，修改它不会自动修改 Dataset，子类需要自行维护状态。
        """

        return None


def validate_sample(sample: Mapping[str, Any], require_ground_truth: bool = True) -> None:
    """尽早校验 Dataset 单条输出，提供比训练中途报错更清晰的信息。

    这里只校验 Trainer/AgentLoop/RewardManager 的公共契约。任务专属字段应在
    子类的 ``transform_sample`` 中自行校验。

    ``require_ground_truth=False`` 只关闭 Dataset 层校验，不会改变 RewardManager
    的行为。如果使用默认 ``naive``、``dapo``、``gdpo`` 等 RewardManager，它们
    仍可能直接读取 ``reward_model["ground_truth"]``。
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


def _normalize_answer(value: Any) -> str:
    """将答案转换为适合简单 exact-match 的规范形式。

    这是一个保守的通用实现，只处理大小写、首尾空白和连续空白。数学等价、
    代码执行、JSON/tool-call 比较等任务应替换为任务专属 verifier。
    """

    text = str(value).strip().casefold()
    return re.sub(r"\s+", " ", text)


def _extract_final_answer(solution_str: str) -> tuple[str, bool]:
    """从模型响应中提取最终答案，并返回是否命中推荐格式。

    支持以下常见格式，按优先级匹配：

    - ``#### answer``，GSM8K 常用；
    - ``<answer>answer</answer>``；
    - ``\\boxed{answer}``，这里只支持不嵌套花括号的常见情况；
    - 如果都没有，使用最后一个非空行作为答案。
    """

    text = solution_str.strip()
    patterns = (
        r"####\s*(.+?)(?:\n|$)",
        r"<answer>\s*(.*?)\s*</answer>",
        r"\\boxed\{([^{}]+)\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            return str(matches[-1]).strip(), True

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (non_empty_lines[-1] if non_empty_lines else ""), False


def _iter_ground_truths(ground_truth: Any) -> Iterable[Any]:
    """统一单答案与多答案 ground truth 的遍历方式。"""

    if isinstance(ground_truth, (list, tuple, set)):
        return ground_truth
    return (ground_truth,)


def _is_correct(prediction: str, ground_truth: Any, numeric_tolerance: float | None) -> bool:
    """执行通用 exact-match，并可选支持数值容差。"""

    normalized_prediction = _normalize_answer(prediction)
    for candidate in _iter_ground_truths(ground_truth):
        if normalized_prediction == _normalize_answer(candidate):
            return True

        if numeric_tolerance is not None:
            try:
                predicted_number = float(prediction.replace(",", "").strip())
                expected_number = float(str(candidate).replace(",", "").strip())
            except (TypeError, ValueError):
                continue
            if math.isclose(predicted_number, expected_number, abs_tol=numeric_tolerance, rel_tol=0.0):
                return True
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    *,
    correct_score: float = 1.0,
    incorrect_score: float = 0.0,
    format_score: float = 0.0,
    numeric_tolerance: float | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """VeRL ``naive``/``dapo`` RewardManager 使用的逐样本奖励函数模板。

    RewardManager 会以关键字参数调用此函数：

    - ``data_source``：来自 Dataset，可用于为不同数据源分发不同 verifier；
    - ``solution_str``：模型生成的完整响应文本；
    - ``ground_truth``：来自 ``reward_model.ground_truth``；
    - ``extra_info``：来自 Dataset，可携带测试用例、难度、题目文本等；
    - ``kwargs``：保留以兼容 reward model router 或未来新增参数。

    返回值可以是一个 float，也可以是至少包含 ``score`` 的字典。返回字典时，
    其他数值字段会进入 VeRL 的 reward extra info，便于日志统计、验证分析以及
    GDPO 多奖励组件训练。

    当前默认实现提供“最终答案 exact-match + 格式奖励”。实际项目通常应按照
    ``data_source`` 分发到不同任务 verifier，例如数学等价校验、代码沙箱测试、
    tool-call JSON 校验等。
    """

    del kwargs  # 当前模板不使用额外参数，但保留 **kwargs 以兼容 VeRL 调用协议。

    # 允许每条样本通过 extra_info 覆盖数值容差；适合不同精度要求混合训练。
    sample_tolerance = extra_info.get("numeric_tolerance", numeric_tolerance)
    if sample_tolerance is not None:
        sample_tolerance = float(sample_tolerance)

    prediction, has_recommended_format = _extract_final_answer(solution_str)
    is_correct = _is_correct(prediction, ground_truth, sample_tolerance)

    accuracy_reward = float(correct_score if is_correct else incorrect_score)
    final_format_reward = float(format_score if has_recommended_format else 0.0)
    total_score = accuracy_reward + final_format_reward

    # data_source 当前没有参与默认计算，但故意保留这个分支位置，便于按任务扩展：
    #
    # if data_source == "my_code_dataset":
    #     return run_code_verifier(solution_str, ground_truth, extra_info)
    #
    # 不建议在这里静默接受未知 data_source；生产环境可按需显式抛错。
    _ = data_source

    return {
        "score": total_score,
        "acc": float(is_correct),
        "accuracy_reward": accuracy_reward,
        "format_reward": final_format_reward,
    }


def compute_score_batched(
    data_sources: Iterable[str],
    solution_strs: Iterable[str],
    ground_truths: Iterable[Any],
    extra_infos: Iterable[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    """VeRL ``batch`` RewardManager 使用的批量奖励函数模板。

    使用该函数时配置：

    .. code-block:: bash

        reward.reward_manager.name=batch
        reward.custom_reward_function.name=compute_score_batched

    批量实现适合可以向量化、批量请求远端服务或批量运行 verifier 的任务。
    此处为了保持示例简单，逐条复用 :func:`compute_score`。
    """

    return [
        compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
            strict=True,
        )
    ]


__all__ = [
    "BaseDataset",
    "compute_score",
    "compute_score_batched",
    "validate_sample",
]
