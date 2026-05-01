# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end save/load roundtrip tests for ``eLargeModel``.

The eLargeModel configuration is a deeply nested ``eLMConfig`` covering
every subsystem the engine cares about: model identity, loader dtypes,
sharding/partitioning, platform/backend, quantization, base config
overrides, dataset mixture, eSurge inference, trainer hyperparameters,
and lm-eval defaults. These tests construct an instance with **every
section populated** and then verify that JSON and YAML save/load are
canonically equivalent (i.e., ``make_serializable(normalize(...))`` is
fixed under save → load).
"""

from __future__ import annotations

from typing import Any

import pytest

from easydel.infra.elarge import eLargeModel
from easydel.infra.elarge.processing import (
    load_elm_config,
    make_serializable,
    normalize,
    save_elm_config,
)


def _full_elm_config() -> dict[str, Any]:
    """Return an ELM config dict that exercises every documented section.

    Values are deliberately non-default so we can detect cases where save/load
    silently falls back to defaults instead of preserving user intent.
    """
    return {
        "model": {
            "name_or_path": "meta-llama/Llama-2-7b-hf",
            "tokenizer": "meta-llama/Llama-2-7b-hf",
            "task": "causal-language-model",
            "extra_kwargs": {"trust_remote_code": False, "revision": "main"},
        },
        "teacher_model": {
            "name_or_path": "meta-llama/Llama-2-13b-hf",
            "task": "causal-language-model",
        },
        "reference_model": {
            "name_or_path": "meta-llama/Llama-2-7b-chat-hf",
            "task": "causal-language-model",
        },
        "loader": {
            "device": "tpu",
            "dtype": "bf16",
            "param_dtype": "fp32",
            "precision": "HIGHEST",
            "verbose": False,
            "from_torch": True,
            "trust_remote_code": True,
        },
        "sharding": {
            "axis_dims": [1, 2, 4, 1, -1, 1],
            "dcn_axis_dims": [1, 1, 1, 1, 1, 1],
            "axis_names": ["pp", "dp", "fsdp", "ep", "tp", "sp"],
            "auto_shard_model": False,
            "use_ring_of_experts": True,
            "fsdp_is_ep_bound": False,
            "sp_is_ep_bound": False,
        },
        "platform": {
            "backend": "tpu",
            "platform": "triton",
        },
        "quantization": {
            "platform": "triton",
            "apply_quantization": True,
            "use_qmm_best_config": True,
            "qmm_platform_override": "xla",
            "qmm_tpu_path_override": "packed",
            "model": {
                "dtype": "nf4",
                "group_size": 128,
                "bits": 4,
                "simulate": False,
                "jax_native": False,
                "pattern": ".*linear.*",
            },
            "kv_cache": {
                "dtype": "int8",
                "group_size": 64,
                "bits": 8,
                "simulate": False,
                "jax_native": False,
                "pattern": ".*",
            },
        },
        "base_config": {
            "values": {
                "attn_dtype": "bf16",
                "kvdtype": "bf16",
                "freq_max_position_embeddings": 8192,
                "mask_max_position_embeddings": 8192,
            },
        },
        "mixture": {
            "informs": [
                {
                    "type": "json",
                    "data_files": ["/tmp/train.jsonl"],
                    "split": "train",
                    "content_field": "text",
                    "num_rows": 10000,
                },
                {
                    "type": "huggingface",
                    "data_files": None,
                    "split": "train[:5%]",
                    "content_field": "text",
                },
            ],
            "cache_dir": "/tmp/easydel-cache",
            "streaming": True,
            "text_target_field": "text",
            "image_target_field": "image",
            "batch_size": 4,
            "shuffle_buffer_size": 2048,
            "seed": 1234,
            "pack_tokens": True,
            "tokens_field_name": "input_ids",
            "pack_seq_length": 4096,
            "pack_eos_token_id": 2,
            "pack_shuffle": True,
            "pack_shuffle_buffer_factor": 8,
            "use_sharded_source": True,
            "use_fast_loader": True,
            "num_workers": 8,
            "prefetch_size": 16,
            "enable_caching": True,
            "mixture_weights": {"src_a": 0.7, "src_b": 0.3},
            "stop_strategy": "first_exhausted",
            "block_mixture": True,
            "mixture_block_size": 256,
        },
        "esurge": {
            "max_model_len": 8192,
            "min_input_pad": 32,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "hbm_utilization": 0.85,
            "page_size": 256,
            "use_aot_forward": True,
            "bind_graphstate_for_aot": True,
            "enable_window_aware_runtime_cap": True,
            "enable_prefix_caching": True,
            "auto_shard_model": True,
            "sharding_axis_dims": [1, 1, 1, -1],
            "compile_runner": True,
            "async_scheduling": True,
            "runner_verbose": False,
            "verbose": True,
            "overlap_execution": True,
            "sampler_metrics": True,
            "data_parallelism_axis": "dp",
            "esurge_name": "main-engine",
            "reserve_tokens": 128,
            "auto_truncate_prompt": True,
            "auto_cap_new_tokens": True,
            "strict_context": False,
            "truncate_mode": "left",
            "prefer_preserve_prompt": True,
            "decode_truncated_prompt": False,
            "destroy_pages_on_pause": False,
            "extra_eos_token_ids": [128001, 128009],
            "extra_stops": ["</s>", "<|endoftext|>"],
            "silent_mode": False,
        },
        "trainer": {
            "trainer_type": "sft",
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "total_batch_size": 32,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_length": 4096,
            "save_steps": 500,
            "save_total_limit": 5,
            "save_directory": "/tmp/checkpoints",
            "use_wandb": False,
            "log_steps": 10,
            "do_train": True,
            "do_eval": False,
            "is_fine_tuning": True,
        },
        "eval": {
            "num_fewshot": 5,
            "max_new_tokens": 1024,
            "hard_max_new_tokens": True,
            "enable_thinking": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "batch_size": 16,
            "max_batch_size": 32,
            "limit": 100,
            "log_samples": True,
            "apply_chat_template": True,
            "fewshot_as_multiturn": False,
            "bootstrap_iters": 1000,
            "random_seed": 42,
            "predict_only": False,
            "include_defaults": True,
        },
    }


def _canonical(cfg: dict[str, Any]) -> Any:
    """Return the canonical save form: ``make_serializable(normalize(cfg))``.

    This is what ``save_elm_config`` writes to disk; equality on this form
    is exactly the round-trip contract.
    """
    return make_serializable(normalize(cfg))


@pytest.fixture
def full_config() -> dict[str, Any]:
    return _full_elm_config()


def test_elarge_json_roundtrip_full_config(tmp_path, full_config):
    """All eLMConfig sections survive a ``to_json`` → ``from_json`` cycle."""
    elm = eLargeModel(full_config)
    expected = _canonical(full_config)

    json_path = tmp_path / "config.json"
    elm.to_json(json_path)
    loaded = eLargeModel.from_json(json_path)

    assert _canonical(loaded._config) == expected


def test_elarge_yaml_roundtrip_full_config(tmp_path, full_config):
    """All eLMConfig sections survive a ``to_yaml`` → ``from_yaml`` cycle."""
    pytest.importorskip("yaml")
    elm = eLargeModel(full_config)
    expected = _canonical(full_config)

    yaml_path = tmp_path / "config.yaml"
    elm.to_yaml(yaml_path)
    loaded = eLargeModel.from_yaml(yaml_path)

    assert _canonical(loaded._config) == expected


def test_elarge_save_load_helpers_roundtrip_full_config(tmp_path, full_config):
    """Module-level ``save_elm_config`` / ``load_elm_config`` roundtrip cleanly."""
    expected = _canonical(full_config)

    json_path = tmp_path / "module_helpers.json"
    save_elm_config(full_config, json_path)
    loaded = load_elm_config(json_path)

    assert _canonical(loaded) == expected


def test_elarge_to_dict_returns_loadable_view(tmp_path, full_config):
    """``eLargeModel.to_dict()`` round-trips back into another eLargeModel."""
    elm = eLargeModel(full_config)
    expected = _canonical(elm.to_dict())

    rebuilt = eLargeModel(elm.to_dict())
    assert _canonical(rebuilt.to_dict()) == expected


def test_elarge_roundtrip_preserves_every_top_level_section(tmp_path, full_config):
    """Every populated section in the input config must reappear after round-trip.

    This complements equality checks by guarding against the case where a
    section silently disappears on save/load.
    """
    elm = eLargeModel(full_config)

    json_path = tmp_path / "sections.json"
    elm.to_json(json_path)
    loaded = eLargeModel.from_json(json_path)

    for section in full_config:
        assert section in loaded._config, f"Section {section!r} missing after JSON roundtrip"


def test_elarge_minimal_config_roundtrip(tmp_path):
    """A minimal config (only model.name_or_path) roundtrips and gets normalized defaults."""
    minimal = {"model": {"name_or_path": "gpt2"}}
    elm = eLargeModel(minimal)
    expected = _canonical(minimal)

    json_path = tmp_path / "minimal.json"
    elm.to_json(json_path)
    loaded = eLargeModel.from_json(json_path)

    assert _canonical(loaded._config) == expected
    assert loaded._config["model"]["name_or_path"] == "gpt2"
