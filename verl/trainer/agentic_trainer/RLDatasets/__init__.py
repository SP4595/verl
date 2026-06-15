"""Agentic Trainer 使用的唯一 RL Dataset 公共入口。

每个 Dataset class 位于独立模块；这里仅负责导出稳定的公共 API。

Python 代码推荐从本包导入::

    from verl.trainer.agentic_trainer.RLDatasets import PrivilegeOPDDataset

Hydra 的 ``data.custom_cls.path`` 应直接指向具体 class 文件，例如::

    data.custom_cls.path=/absolute/path/to/RLDatasets/privilege_opd_dataset.py
    data.custom_cls.name=PrivilegeOPDDataset
"""

from verl.trainer.agentic_trainer.RLDatasets.base_dataset import BaseDataset
from verl.trainer.agentic_trainer.RLDatasets.locomo_privilege_subset_dataset import (
    LoCoMoPrivilegeSubsetDataset,
)
from verl.trainer.agentic_trainer.RLDatasets.memory_opd_step_dataset import MemoryOPDStepDataset
from verl.trainer.agentic_trainer.RLDatasets.privilege_opd_dataset import PrivilegeOPDDataset
from verl.trainer.agentic_trainer.RLDatasets.schema import (
    NORMALIZED_SAMPLE_KEYS,
    PRIVILEGE_OPD_SAMPLE_KEYS,
    validate_memory_episode,
    validate_sample,
)

__all__ = [
    "BaseDataset",
    "LoCoMoPrivilegeSubsetDataset",
    "MemoryOPDStepDataset",
    "NORMALIZED_SAMPLE_KEYS",
    "PRIVILEGE_OPD_SAMPLE_KEYS",
    "PrivilegeOPDDataset",
    "validate_memory_episode",
    "validate_sample",
]
