# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass
from verl.utils.config import validate_config


@dataclass
class TestDataclass(BaseConfig):
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)


class TestPrintCfgCommand(unittest.TestCase):
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        """Test that the command runs without error when overriding config values."""
        import subprocess

        # Run the command
        result = subprocess.run(
            ["python3", "scripts/print_cfg.py"],
            capture_output=True,
            text=True,
        )

        # Verify the command exited successfully
        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        # Verify the output contains expected config information
        self.assertIn("critic", result.stdout)
        self.assertIn("profiler", result.stdout)


def test_validate_config_can_defer_source_batch_checks_for_post_rollout_rows(monkeypatch):
    """Post-rollout trainers keep generic config checks without inventing a source batch."""

    validated_train_batch_sizes = []

    class StubActorConfig:
        def validate(self, n_gpus, train_batch_size, model_config, *, validate_train_batch_size=True):
            validated_train_batch_sizes.append((train_batch_size, validate_train_batch_size))

    monkeypatch.setattr("verl.utils.config.omega_conf_to_dataclass", lambda *_args, **_kwargs: StubActorConfig())
    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 3, "nnodes": 1},
            "data": {"train_batch_size": 2, "val_batch_size": None},
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "use_dynamic_bsz": False,
                    "strategy": "fsdp",
                    "use_kl_loss": False,
                },
                "model": {"lora": {}, "lora_rank": 0},
                "rollout": {
                    "n": 4,
                    "name": "hf",
                    "temperature": 1.0,
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "val_kwargs": {"do_sample": False},
                },
                "ref": {
                    "log_prob_micro_batch_size": None,
                    "log_prob_micro_batch_size_per_gpu": 1,
                },
            },
        }
    )

    # Native source-batch semantics still reject 2*4 rows on three ranks.
    with pytest.raises(AssertionError, match="real_train_batch_size"):
        validate_config(config, use_reference_policy=False, use_critic=False)

    # A post-rollout trainer owns the eventual row count. VeRL retains the real
    # source batch and explicitly disables only this source-level comparison.
    validate_config(
        config,
        use_reference_policy=False,
        use_critic=False,
        defer_train_batch_size_validation=True,
    )
    assert validated_train_batch_sizes == [(2, False)]


if __name__ == "__main__":
    unittest.main()
