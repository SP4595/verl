from types import SimpleNamespace

import pytest
import torch.nn as nn
from peft import TaskType
from transformers import BertConfig, BertForTokenClassification

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _ToyValueModel(nn.Module):
    """最小 value model：backbone 模拟冻结基模，score 模拟可训练 critic 头。"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.score = nn.Linear(4, 1)


def _engine(trainable_modules, *, lora_rank=0):
    """绕过分布式初始化，只构造测试参数过滤逻辑所需的 FSDPEngine 外壳。"""

    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = SimpleNamespace(trainable_modules=trainable_modules, lora_rank=lora_rank)
    return engine


def test_trainable_module_filter_keeps_only_value_head_trainable():
    model = _ToyValueModel()

    # head_only 分支必须冻结全部 backbone 参数，只留下 score 的 weight/bias。
    _engine(["score"])._apply_trainable_module_filter(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"score.weight", "score.bias"}


def test_trainable_module_filter_rejects_unknown_module():
    # 未匹配任何 head 时直接失败，避免训练看似运行但 critic 实际没有参数更新。
    with pytest.raises(ValueError, match="No modules matched"):
        _engine(["missing_head"])._apply_trainable_module_filter(_ToyValueModel())


def test_value_model_lora_uses_token_classification_task_type():
    # lora critic 使用真实 HF token-classification 模型，验证 PEFT task type 没有走 actor 分支。
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = SimpleNamespace(
        model_type="value_model",
        lora_adapter_path=None,
        lora_rank=2,
        lora_alpha=4,
        target_modules=["query", "value"],
        target_parameters=None,
        exclude_modules=None,
    )
    model = BertForTokenClassification(
        BertConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_labels=1,
        )
    )

    lora_model = engine._build_lora_module(model)

    assert lora_model.peft_config["default"].task_type == TaskType.TOKEN_CLS


def test_lora_filter_keeps_adapter_and_value_head_trainable():
    """LoRA 模式不能在解冻 value head 时把刚创建的 adapter 再次冻结。"""

    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = SimpleNamespace(
        model_type="value_model",
        lora_adapter_path=None,
        lora_rank=2,
        lora_alpha=4,
        target_modules=["query", "value"],
        target_parameters=None,
        exclude_modules=None,
        trainable_modules=["classifier"],
    )
    model = BertForTokenClassification(
        BertConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_labels=1,
        )
    )

    # 先走真实 PEFT 构造，再走与生产代码相同的可训练参数过滤顺序。
    filtered = engine._apply_trainable_module_filter(engine._build_lora_module(model))
    trainable = {name for name, parameter in filtered.named_parameters() if parameter.requires_grad}

    assert any("lora_" in name for name in trainable)
    assert any("classifier" in name for name in trainable)
    assert all("lora_" in name or "classifier" in name for name in trainable)
    assert not any("original_module" in name for name in trainable)
