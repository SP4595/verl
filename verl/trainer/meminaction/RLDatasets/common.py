"""RLDatasets 内部共享的 OmegaConf、路径和 split 适配函数。

这些函数只处理 Dataset 构造配置，不读取训练内容，也不参与 prompt 渲染。
"""

from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf


def to_plain_container(value: Any) -> Any:
    """把 OmegaConf 容器解析为可复制、可传给普通 Dataset 的 Python 容器。"""

    # 步骤 1：OmegaConf 输入先解析插值并转换为普通容器。
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    # 步骤 2：普通 Python 值保持原样。
    return value


def normalize_data_files(data_files: Any) -> tuple[str, ...]:
    """把字符串、ListConfig、list 等文件配置统一为可比较的路径 tuple。"""

    # 步骤 1：空配置规范化为空 tuple。
    if data_files is None:
        return ()
    # 步骤 2：单路径规范化为单元素 tuple。
    if isinstance(data_files, str | Path):
        return (str(data_files),)
    # 步骤 3：路径序列逐项转成字符串 tuple，便于稳定比较。
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

    # 步骤 1：规范化当前工厂调用传入的文件路径。
    current_files = normalize_data_files(data_files)
    # 步骤 2：与 train_files 精确匹配。
    if current_files == normalize_data_files(config.get("train_files")):
        return "train"
    # 步骤 3：与 val_files 精确匹配。
    if current_files == normalize_data_files(config.get("val_files")):
        return "val"
    # 步骤 4：无法匹配时返回 unknown，由调用方给出具体配置错误。
    return "unknown"
