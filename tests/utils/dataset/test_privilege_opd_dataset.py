import json

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl import DataProto
from verl.trainer.agentic_trainer.RLDatasets import (
    MemoryOPDStepDataset,
    PRIVILEGE_OPD_SAMPLE_KEYS,
    PrivilegeOPDDataset,
    privilege_opd_dataset,
)
from verl.utils.dataset.rl_dataset import collate_fn


class StaticPrivilegeSubsetDataset(Dataset):
    """Test-only heterogeneous subset that already follows the BaseDataset API."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        del data_files, tokenizer, processor, max_samples
        self.data_source = config.default_data_source

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return {
            "raw_prompt": [{"role": "user", "content": f"short question {item}"}],
            "data_source": self.data_source,
            "reward_model": {},
            "extra_info": {"index": f"long-{item}"},
            "index": f"long-{item}",
            "agent_name": "single_turn_agent",
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "privileged_context": f"long document {item}",
        }


def _write_locomo_file(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "locomo-1",
                    "conversation": {
                        "session_1": [
                            {"speaker": "Alice", "dia_id": "D1:1", "text": "I moved to Paris."},
                            {"speaker": "Bob", "dia_id": "D1:2", "text": "That sounds exciting."},
                        ],
                        "session_1_date_time": "1:00 pm on 8 May, 2023",
                    },
                    "qa": [
                        {
                            "question": "Where did Alice move?",
                            "answer": "Paris",
                            "category": 1,
                            "evidence": ["D1:1"],
                        },
                        {
                            "question": "What cannot be answered?",
                            "answer": "Unknown",
                            "category": 5,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def _build_config(locomo_path):
    return OmegaConf.create(
        {
            "train_files": ["privilege_opd://train"],
            "val_files": ["privilege_opd://val"],
            "require_ground_truth": False,
            "shared_fields": ["memory_episode"],
            "shared_field_defaults": {
                "memory_episode": None,
            },
            "subset_sampling": {"train_strategy": "concat", "seed": 7},
            "subsets": {
                "long_text": {
                    "train_files": ["unused"],
                    "dataset_cls": {"name": "StaticPrivilegeSubsetDataset"},
                    "config": {
                        "default_data_source": "long_text",
                        "require_ground_truth": False,
                    },
                },
                "locomo": {
                    "train_files": [str(locomo_path)],
                    "dataset_cls": {"name": "LoCoMoPrivilegeSubsetDataset"},
                    "config": {
                        "default_data_source": "locomo",
                        "require_ground_truth": True,
                    },
                },
            },
        }
    )


def test_privilege_opd_dataset_projects_heterogeneous_subsets_to_one_api(tmp_path, monkeypatch):
    monkeypatch.setattr(
        privilege_opd_dataset,
        "StaticPrivilegeSubsetDataset",
        StaticPrivilegeSubsetDataset,
        raising=False,
    )
    config = _build_config(_write_locomo_file(tmp_path))

    dataset = PrivilegeOPDDataset(
        data_files=config.train_files,
        tokenizer=None,
        config=config,
    )

    assert dataset.subset_summary() == {"long_text": 2, "locomo": 1}
    assert len(dataset) == 3

    samples = [dataset[index] for index in range(len(dataset))]
    expected_keys = set(PRIVILEGE_OPD_SAMPLE_KEYS) | {"memory_episode"}
    assert all(set(sample) == expected_keys for sample in samples)
    assert samples[0]["memory_episode"] is None
    assert samples[2]["reward_model"] == {}
    assert samples[2]["memory_episode"]["episode_id"] == "locomo-1"
    assert "Alice (D1:1): I moved to Paris." in samples[2]["memory_episode"]["sessions"][0]["input"]
    assert samples[2]["memory_episode"]["qa"][0]["answer"] == "Paris"
    assert samples[2]["extra_info"]["source_index"] == 0

    data_proto = DataProto.from_single_dict(collate_fn(samples))
    assert len(data_proto) == 3
    assert data_proto.non_tensor_batch["subset_name"].tolist() == ["long_text", "long_text", "locomo"]


def test_privilege_opd_dataset_weighted_sampling_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(
        privilege_opd_dataset,
        "StaticPrivilegeSubsetDataset",
        StaticPrivilegeSubsetDataset,
        raising=False,
    )
    config = _build_config(_write_locomo_file(tmp_path))
    config.subset_sampling = {
        "train_strategy": "weighted",
        "train_epoch_size": 8,
        "seed": 19,
    }
    config.subsets.long_text.train_weight = 0.0
    config.subsets.locomo.train_weight = 1.0

    first = PrivilegeOPDDataset(data_files=config.train_files, tokenizer=None, config=config)
    second = PrivilegeOPDDataset(data_files=config.train_files, tokenizer=None, config=config)

    assert len(first) == 8
    assert first._index_map == second._index_map
    assert {first[index]["subset_name"] for index in range(len(first))} == {"locomo"}


def test_privilege_opd_dataset_supports_base_dataset_subset(tmp_path, monkeypatch):
    import datasets

    hf_cache = str(tmp_path / "hf_cache")
    monkeypatch.setenv("HF_DATASETS_CACHE", hf_cache)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", hf_cache)
    long_text_path = tmp_path / "long_text.json"
    long_text_path.write_text(
        json.dumps(
            [
                {
                    "prompt": [{"role": "user", "content": "Summarize the document."}],
                    "data_source": "long_text",
                    "extra_info": {"index": "doc-1"},
                    "privileged_context": "A very long document.",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "train_files": ["privilege_opd://train"],
            "val_files": ["privilege_opd://val"],
            "filter_overlong_prompts": False,
            "cache_dir": str(tmp_path / "verl_cache"),
            "shared_fields": ["privileged_context"],
            "subsets": {
                "long_text": {
                    "train_files": [str(long_text_path)],
                    "dataset_cls": {"name": "BaseDataset"},
                    "config": {
                        "default_data_source": "long_text",
                        "require_ground_truth": False,
                    },
                }
            },
        }
    )

    dataset = PrivilegeOPDDataset(data_files=config.train_files, tokenizer=None, config=config)
    sample = dataset[0]

    assert sample["subset_name"] == "long_text"
    assert sample["source_index"] == "doc-1"
    assert sample["reward_model"] == {}
    assert sample["privileged_context"] == "A very long document."


def test_memory_opd_step_dataset_flattens_collector_trace(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "episode_id": "locomo-1",
                "steps": [
                    {
                        "memory_step": {
                            "episode_id": "locomo-1",
                            "phase": "memory_creation",
                            "task_mode": "update",
                            "current_input": "new session",
                            "memory_cache": [],
                            "full_memory": [],
                            "step_index": 0,
                        },
                        "action": "update",
                        "status": "terminal",
                    },
                    {
                        "memory_step": {
                            "episode_id": "locomo-1",
                            "phase": "qa",
                            "task_mode": "answer",
                            "current_input": "Where?",
                            "memory_cache": [{"vid": 1, "content": "Alice lives in Paris."}],
                            "full_memory": [{"rid": "m1", "content": "Alice lives in Paris."}],
                            "step_index": 0,
                        },
                        "action": "answer",
                        "status": "terminal",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create({"validate_custom_sample": True})

    dataset = MemoryOPDStepDataset(
        data_files=[str(trace_path)],
        tokenizer=None,
        config=config,
    )

    assert len(dataset) == 2
    assert dataset[0]["agent_name"] == "memory_opd_step"
    assert dataset[1]["memory_step"]["task_mode"] == "answer"
    assert dataset[1]["extra_info"]["collected_action"] == "answer"
