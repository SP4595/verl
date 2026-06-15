"""RLDatasets 内部共享的 OmegaConf、路径和 split 适配函数。

这些函数只处理 Dataset 构造配置，不读取训练内容，也不参与 prompt 渲染。
"""

from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf


def to_plain_container(value: Any) -> Any:
    """把 OmegaConf 容器解析为可复制、可传给普通 Dataset 的 Python 容器。"""

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def normalize_data_files(data_files: Any) -> tuple[str, ...]:
    """把字符串、ListConfig、list 等文件配置统一为可比较的路径 tuple。"""

    if data_files is None:
        return ()
    if isinstance(data_files, str | Path):
        return (str(data_files),)
    if isinstance(data_files, list | tuple | ListConfig):
        return tuple(str(path) for path in data_files)
    raise TypeError(f"data_files 必须是路径或路径列表，实际为 {type(data_files)!r}")


def infer_dataset_split(data_files: Any, config: Any) -> str:
    """根据 Trainer 传入的路径判断当前构造的是 train 还是 val Dataset。

    VeRL 对 train/val 都调用同一个自定义 Dataset 类，但不会显式传入 split 名称。
    组合 Dataset 需要依靠当前 ``data_files`` 与配置中的路径精确比较来选择每个 subset
    的 ``train_files`` 或 ``val_files``。无法判断时返回 ``"unknown"``，后续缺少对应
    文件配置时会给出明确错误。
    """

    current_files = normalize_data_files(data_files)
    if current_files == normalize_data_files(config.get("train_files")):
        return "train"
    if current_files == normalize_data_files(config.get("val_files")):
        return "val"
    return "unknown"
