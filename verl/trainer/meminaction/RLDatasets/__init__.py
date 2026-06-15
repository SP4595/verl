"""``verl.trainer.meminaction`` 使用的 RL Dataset 公共入口。

每个 Dataset class 位于独立模块；这里仅负责导出稳定的公共 API。

Python 代码推荐从本包导入::

    from verl.trainer.meminaction.RLDatasets import PrivilegeOPDDataset

Hydra 的 ``data.custom_cls.path`` 应直接指向具体 class 文件，例如::

    data.custom_cls.path=/absolute/path/to/RLDatasets/privilege_opd_dataset.py
    data.custom_cls.name=PrivilegeOPDDataset

Dataset 分为两阶段，不能混入同一个 Trainer batch：

- ``LoCoMoPrivilegeSubsetDataset`` 输出完整 ``memory_episode``，供 Collector 消费；
- ``MemoryOPDStepDataset`` 输出冻结 ``memory_step``，供 single-turn OPD Trainer 消费。
"""

from verl.trainer.meminaction.RLDatasets.base_dataset import BaseDataset
from verl.trainer.meminaction.RLDatasets.locomo_privilege_subset_dataset import (
    LoCoMoPrivilegeSubsetDataset,
)
from verl.trainer.meminaction.RLDatasets.memory_opd_step_dataset import MemoryOPDStepDataset
from verl.trainer.meminaction.RLDatasets.privilege_opd_dataset import PrivilegeOPDDataset
from verl.trainer.meminaction.RLDatasets.schema import (
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
