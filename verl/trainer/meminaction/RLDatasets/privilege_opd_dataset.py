"""多个异构 source 组成的统一 VeRL Dataset 组合器。

LoCoMo 等长轨迹来源应输出 ``memory_episode``，再由 Agentic Collector 展开为
single-turn ``memory_step``。不要把 episode source 与已经展开的 step source 放进
同一个 Trainer batch。

该组合器只解决“多个数据来源如何采样并暴露一致顶层 key”，不解决状态推进。组合
episode source 时，消费者必须是 Collector；组合 step source 时，消费者才可以是
``RayPrivilegeOPDTrainer``。配置者必须保证同一个 batch 中所有 subset 位于同一阶段。
"""

import copy
import random
from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.trainer.meminaction.RLDatasets.base_dataset import BaseDataset
from verl.trainer.meminaction.RLDatasets.common import infer_dataset_split, to_plain_container
from verl.trainer.meminaction.RLDatasets.locomo_privilege_subset_dataset import (
    LoCoMoPrivilegeSubsetDataset,
)
from verl.trainer.meminaction.RLDatasets.memory_opd_step_dataset import MemoryOPDStepDataset
from verl.trainer.meminaction.RLDatasets.schema import PRIVILEGE_OPD_SAMPLE_KEYS, validate_sample
from verl.utils.import_utils import load_extern_object

# 这些配置只由组合 Dataset 使用，不应继续传给子 Dataset。
_COMPOSITE_CONFIG_KEYS = {
    "subsets",
    "subset_sampling",
    "shared_fields",
    "shared_field_defaults",
    "preserve_source_fields",
    "allow_empty_subsets",
}

_BUILTIN_DATASET_CLASSES = {
    "BaseDataset": BaseDataset,
    "LoCoMoPrivilegeSubsetDataset": LoCoMoPrivilegeSubsetDataset,
    "MemoryOPDStepDataset": MemoryOPDStepDataset,
}


class PrivilegeOPDDataset(Dataset):
    """对 Trainer 暴露统一 API 的多 subset 组合 Dataset。

    每个 subset 可以来自不同文件格式和 Dataset 类，例如普通长文本数据使用
    :class:`BaseDataset`，LoCoMo JSON 使用
    :class:`LoCoMoPrivilegeSubsetDataset`。每个子 Dataset 必须保持 VeRL 构造 API，
    且每条样本至少满足 ``BaseDataset`` 输出 schema。

    推荐配置::

        data:
          # 组合 Dataset 用这些稳定标记判断当前构造 train 还是 val。
          train_files: ["privilege_opd://train"]
          val_files: ["privilege_opd://val"]

          custom_cls:
            path: /absolute/path/to/RLDatasets/privilege_opd_dataset.py
            name: PrivilegeOPDDataset

          # Episode source 共同暴露的额外字段。prompt 在 AgentLoop 中动态构造。
          shared_fields: [memory_episode]
          shared_field_defaults:
            memory_episode: null

          # False 时丢弃各源私有顶层字段，保证混合 batch 的 key 完全一致。
          preserve_source_fields: false

          subset_sampling:
            train_strategy: weighted
            train_epoch_size: 100000
            val_strategy: concat
            seed: 42

          subsets:
            long_text:
              train_files: [/data/long_text/train.parquet]
              val_files: [/data/long_text/val.parquet]
              weight: 1.0
              dataset_cls:
                name: BaseDataset
              config:
                default_data_source: long_text
                require_ground_truth: false

            locomo:
              train_files: [/data/locomo/train.json]
              val_files: [/data/locomo/val.json]
              weight: 2.0
              dataset_cls:
                name: LoCoMoPrivilegeSubsetDataset
              config:
                default_data_source: locomo
                require_ground_truth: true

    ``dataset_cls.name`` 可以直接使用本包内置的 ``BaseDataset`` 和
    ``LoCoMoPrivilegeSubsetDataset``。其他自定义 Dataset 必须同时配置
    ``dataset_cls.path`` 与 ``dataset_cls.name``。

    组合策略：

    - ``concat``：按 subset 顺序连接；
    - ``round_robin``：各 subset 轮流取样，直到全部耗尽；
    - ``weighted``：按权重有放回采样，长度由 ``epoch_size`` 决定。

    该类只负责组合数据来源。完整 memory、Memory Cache、student prompt 和
    privileged teacher prompt 均由 rollout/AgentLoop 动态构造。

    ``shared_fields`` 是混合异构 subset 时最关键的配置：默认 ``collate_fn`` 按 key
    收集值，如果不同样本顶层 key 不一致，batch 长度和逐样本字段会错位。因此每个
    Trainer batch 使用到的额外字段都应声明为 shared field，例如 step 训练声明
    ``shared_fields: [memory_step]``。
    """

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        # 组合器自身不 tokenize，但必须把同一个 tokenizer/processor 继续传给子类。
        # 静态 BaseDataset 会真实使用 tokenizer；Memory episode/step Dataset 会忽略它。
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.split = infer_dataset_split(data_files, config)

        # 允许纯 OPD 与规则奖励 subset 共存。需要所有样本都有 ground truth 时，
        # 在顶层显式配置 require_ground_truth=True。
        self.require_ground_truth = config.get("require_ground_truth", False)
        self.validate_custom_sample = config.get("validate_custom_sample", True)
        self.preserve_source_fields = config.get("preserve_source_fields", False)
        self.allow_empty_subsets = config.get("allow_empty_subsets", False)

        shared_fields = to_plain_container(config.get("shared_fields", [])) or []
        if isinstance(shared_fields, str):
            shared_fields = [shared_fields]
        self.shared_fields = tuple(dict.fromkeys(str(field) for field in shared_fields))
        self.shared_field_defaults = dict(to_plain_container(config.get("shared_field_defaults", {})) or {})

        self.subset_names: list[str] = []
        self.subsets: list[Dataset] = []
        self.subset_specs: list[dict[str, Any]] = []
        self._build_subsets()
        if not self.subsets:
            raise ValueError("PrivilegeOPDDataset 没有非空 subset，请检查 subsets 和 allow_empty_subsets")

        self._index_map = self._build_index_map(max_samples=self.max_samples)
        if not self._index_map:
            raise ValueError("PrivilegeOPDDataset 没有可用样本，请检查 subsets 和采样配置")

    def _iter_subset_specs(self) -> list[tuple[str, dict[str, Any]]]:
        """把 mapping/list 两种 subsets 配置统一为 ``(name, spec)`` 列表。"""

        raw_subsets = to_plain_container(self.config.get("subsets"))
        if not raw_subsets:
            raise ValueError("PrivilegeOPDDataset 要求 config.data.subsets 至少包含一个 subset")

        if isinstance(raw_subsets, Mapping):
            return [(str(name), dict(spec or {})) for name, spec in raw_subsets.items()]
        if isinstance(raw_subsets, list):
            specs = []
            for spec in raw_subsets:
                spec = dict(spec or {})
                name = spec.pop("name", None)
                if not name:
                    raise ValueError("list 形式的 subsets 中每项必须包含 name")
                specs.append((str(name), spec))
            return specs
        raise TypeError("config.data.subsets 必须是 mapping 或 list")

    def _resolve_subset_dataset_cls(self, subset_name: str, spec: Mapping[str, Any]) -> type[Dataset]:
        """解析当前 subset 使用的 Dataset 类。

        内置类按名称解析；外部类必须提供文件路径和类名。这里仅检查其继承自 PyTorch
        Dataset，具体样本契约会在 ``__getitem__`` 时通过 ``validate_sample`` 校验。
        """

        dataset_cls_config = spec.get("dataset_cls") or {}
        if isinstance(dataset_cls_config, str):
            dataset_cls_config = {"name": dataset_cls_config}
        dataset_cls_config = dict(dataset_cls_config)

        module_path = dataset_cls_config.get("path")
        class_name = dataset_cls_config.get("name")
        if module_path:
            if not class_name:
                raise ValueError(f"subset {subset_name!r} 配置 dataset_cls.path 时必须同时配置 name")
            dataset_cls = load_extern_object(module_path=module_path, object_name=class_name)
        elif class_name:
            # globals() fallback 方便测试或项目代码显式注册本地 class；生产中的外部
            # Dataset 更推荐配置 dataset_cls.path，避免依赖进程内 monkeypatch。
            dataset_cls = _BUILTIN_DATASET_CLASSES.get(class_name) or globals().get(class_name)
            if dataset_cls is None:
                raise ValueError(
                    f"subset {subset_name!r} 找不到内置 Dataset 类 {class_name!r}；"
                    "外部类必须同时配置 dataset_cls.path"
                )
        else:
            dataset_cls = BaseDataset

        if not isinstance(dataset_cls, type) or not issubclass(dataset_cls, Dataset):
            raise TypeError(f"subset {subset_name!r} 的 Dataset 类必须继承 torch.utils.data.Dataset")
        if dataset_cls.__name__ == self.__class__.__name__:
            raise ValueError(f"subset {subset_name!r} 不能再次使用 PrivilegeOPDDataset，避免递归构造")
        return dataset_cls

    def _build_child_config(self, subset_name: str, spec: Mapping[str, Any]):
        """继承顶层 data 配置，并合并当前 subset 的局部覆盖项。

        组合器专属字段必须先移除，尤其是 ``custom_cls``，否则子 Dataset 可能再次解析
        到组合器自身并发生递归构造。
        """

        base_config = dict(to_plain_container(self.config))
        for key in _COMPOSITE_CONFIG_KEYS:
            base_config.pop(key, None)

        # 防止子 Dataset 再次通过顶层 custom_cls 解析到组合 Dataset。
        base_config["custom_cls"] = {"path": None, "name": None}
        child_config = OmegaConf.merge(OmegaConf.create(base_config), spec.get("config", {}))
        if child_config.get("default_data_source") in (None, "custom"):
            child_config["default_data_source"] = spec.get("data_source", subset_name)
        return child_config

    def _select_subset_files(self, subset_name: str, spec: Mapping[str, Any]) -> Any:
        """按当前 train/val split 选择 subset 文件。"""

        split_files_key = f"{self.split}_files"
        subset_files = spec.get(split_files_key, spec.get("data_files"))
        if subset_files is None:
            raise ValueError(
                f"subset {subset_name!r} 缺少 {split_files_key!r} 或 'data_files'；当前 split={self.split!r}"
            )
        return subset_files

    def _build_subsets(self) -> None:
        """实例化所有异构子 Dataset，并保持配置顺序作为稳定 subset 顺序。"""

        for subset_name, spec in self._iter_subset_specs():
            dataset_cls = self._resolve_subset_dataset_cls(subset_name, spec)
            child_config = self._build_child_config(subset_name, spec)
            child_max_samples = spec.get(f"{self.split}_max_samples", spec.get("max_samples", -1))
            subset = dataset_cls(
                data_files=self._select_subset_files(subset_name, spec),
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=child_config,
                max_samples=child_max_samples,
            )
            if len(subset) == 0:
                if self.allow_empty_subsets:
                    continue
                raise ValueError(f"subset {subset_name!r} 为空")

            self.subset_names.append(subset_name)
            self.subsets.append(subset)
            self.subset_specs.append(spec)

    def _build_index_map(self, max_samples: int) -> list[tuple[int, int]]:
        """构建全局 index 到 ``(subset_index, local_index)`` 的确定性映射。

        ``concat``/``round_robin`` 不重复样本；``weighted`` 按权重有放回采样，因此同一
        local sample 可以在一个 epoch 中出现多次。映射在构造或 resume 时一次生成，
        使 ``__getitem__`` 不依赖运行时随机状态。
        """

        sampling_config = dict(to_plain_container(self.config.get("subset_sampling", {})) or {})
        strategy = sampling_config.get(f"{self.split}_strategy", sampling_config.get("strategy", "concat"))
        seed = int(sampling_config.get("seed", self.config.get("seed", 0) or 0))
        rng = random.Random(seed)

        if strategy == "concat":
            index_map = [
                (subset_index, local_index)
                for subset_index, subset in enumerate(self.subsets)
                for local_index in range(len(subset))
            ]
        elif strategy == "round_robin":
            index_map = []
            for local_index in range(max(len(subset) for subset in self.subsets)):
                for subset_index, subset in enumerate(self.subsets):
                    if local_index < len(subset):
                        index_map.append((subset_index, local_index))
        elif strategy == "weighted":
            total_size = sum(len(subset) for subset in self.subsets)
            epoch_size = sampling_config.get(
                f"{self.split}_epoch_size",
                sampling_config.get("epoch_size", total_size),
            )
            epoch_size = int(epoch_size or total_size)
            if epoch_size <= 0:
                raise ValueError("weighted subset_sampling 的 epoch_size 必须大于 0")

            weights = [
                float(spec.get(f"{self.split}_weight", spec.get("weight", 1.0))) for spec in self.subset_specs
            ]
            if any(weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError(f"subset weights 必须非负且总和大于 0，实际为 {weights}")

            sampled_subset_indices = rng.choices(range(len(self.subsets)), weights=weights, k=epoch_size)
            index_map = [
                (subset_index, rng.randrange(len(self.subsets[subset_index])))
                for subset_index in sampled_subset_indices
            ]
        else:
            raise ValueError(f"未知 subset_sampling strategy: {strategy!r}")

        if 0 < max_samples < len(index_map):
            if self.config.get("shuffle", False):
                rng.shuffle(index_map)
            index_map = index_map[:max_samples]
        return index_map

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """读取任意 subset 样本，并投影为统一 VeRL batch API。

        投影只统一顶层 key 和来源索引，不会把 episode 转换成 step，也不会构造
        student/teacher prompt。
        """

        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)

        subset_index, local_index = self._index_map[item]
        subset_name = self.subset_names[subset_index]
        sample = dict(self.subsets[subset_index][local_index])

        # 先校验子 Dataset 的基础 API，再做字段投影，以便准确定位错误来源。
        try:
            validate_sample(sample, require_ground_truth=False)
        except (KeyError, TypeError) as exc:
            raise type(exc)(f"subset {subset_name!r} 的样本 {local_index} 不符合统一 API: {exc}") from exc

        source_index = sample.get("index", local_index)
        extra_info = dict(sample.get("extra_info") or {})
        extra_info.update(
            {
                "subset_name": subset_name,
                "subset_index": local_index,
                "source_index": source_index,
                "global_index": item,
            }
        )
        sample.update(
            {
                "extra_info": extra_info,
                "index": item,
                "subset_name": subset_name,
                "subset_index": local_index,
                "source_index": source_index,
            }
        )

        # 缺失的 shared field 使用统一默认值，保证混合 batch 的 key 一致。
        for field in self.shared_fields:
            if field not in sample:
                sample[field] = copy.deepcopy(self.shared_field_defaults.get(field))

        if not self.preserve_source_fields:
            # 丢弃未声明的源私有字段，确保 batch 中每条样本键集合一致。需要进入
            # AgentLoop/Trainer 的扩展字段必须显式加入 shared_fields。
            output_keys = PRIVILEGE_OPD_SAMPLE_KEYS + self.shared_fields
            sample = {key: sample[key] for key in output_keys}

        if self.validate_custom_sample:
            validate_sample(sample, require_ground_truth=self.require_ground_truth)
        return sample

    def on_batch_end(self, batch) -> None:
        """按 subset 拆分训练完成后的 batch，再转发给对应子 Dataset。"""

        subset_names = getattr(batch, "non_tensor_batch", {}).get("subset_name")
        if subset_names is None:
            return None

        for subset_name, subset in zip(self.subset_names, self.subsets, strict=True):
            callback = getattr(subset, "on_batch_end", None)
            if not callable(callback):
                continue
            positions = [index for index, name in enumerate(subset_names) if name == subset_name]
            if positions:
                callback(batch=batch[positions])
        return None

    def resume_dataset_state(self) -> None:
        """恢复所有子 Dataset，并根据恢复后的长度重建全局索引。"""

        for subset in self.subsets:
            resume = getattr(subset, "resume_dataset_state", None)
            if callable(resume):
                resume()
        self._index_map = self._build_index_map(max_samples=self.max_samples)

    def subset_summary(self) -> dict[str, int]:
        """返回各 subset 的原始长度，便于启动时检查混合数据。"""

        return {name: len(subset) for name, subset in zip(self.subset_names, self.subsets, strict=True)}


__all__ = ["PrivilegeOPDDataset"]
