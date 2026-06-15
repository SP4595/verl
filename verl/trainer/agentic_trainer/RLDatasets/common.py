"""RLDatasets 内部共享的配置与路径辅助函数。"""

from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf


def to_plain_container(value: Any) -> Any:
    """把 OmegaConf 容器转换为普通 Python 容器。"""

    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def normalize_data_files(data_files: Any) -> tuple[str, ...]:
    """把字符串、ListConfig、list 等文件配置统一为 tuple。"""

    if data_files is None:
        return ()
    if isinstance(data_files, str | Path):
        return (str(data_files),)
    if isinstance(data_files, list | tuple | ListConfig):
        return tuple(str(path) for path in data_files)
    raise TypeError(f"data_files 必须是路径或路径列表，实际为 {type(data_files)!r}")


def infer_dataset_split(data_files: Any, config: Any) -> str:
    """根据 Trainer 传入的路径判断当前构造的是 train 还是 val Dataset。"""

    current_files = normalize_data_files(data_files)
    if current_files == normalize_data_files(config.get("train_files")):
        return "train"
    if current_files == normalize_data_files(config.get("val_files")):
        return "val"
    return "unknown"
