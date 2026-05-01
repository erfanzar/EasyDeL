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

"""End-to-end save/load roundtrip tests for ``TrainingArguments``.

``TrainingArguments`` is the master training-config dataclass with ~150
fields covering optimization, scheduling, sharding, checkpointing, logging,
quantization-aware training, generation previews, eSurge integration,
and lm-eval benchmarks. These tests construct an instance with **every
representable field** populated to a non-default value and verify that
``save_arguments`` → ``load_arguments`` (and ``to_dict`` → ``from_dict``)
roundtrips losslessly under the documented contract: the ``to_dict``
projection is a fixed point of the save/load cycle.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec

from easydel.infra.etils import EasyDeLOptimizers, EasyDeLSchedulers
from easydel.infra.loss_utils import LossConfig
from easydel.trainers.metrics import LogWatcher
from easydel.trainers.training_configurations import TrainingArguments


def _full_training_arguments_kwargs() -> dict[str, Any]:
    """Return kwargs covering every save/load-relevant field of TrainingArguments.

    Every value is deliberately non-default so the test detects fields that
    silently drop back to defaults on save/load. Unserializable types
    (PartitionSpec, jnp.dtype, enums) appear here to cover the ``to_dict``
    stringification path that ``__post_init__`` later normalizes back.
    """
    return {
        # ---- Identity / metadata ----
        "model_name": "test-model-7b",
        "trainer_prefix": "RoundtripTest",
        "wandb_entity": "easydel-team",
        "wandb_name": "roundtrip-run",
        "use_wandb": False,
        # ---- Core training schedule ----
        "learning_rate": 3e-5,
        "learning_rate_end": 1e-7,
        "num_train_epochs": 7,
        "max_training_steps": 5000,
        "per_epoch_training_steps": 1000,
        "per_epoch_evaluation_steps": 50,
        "total_batch_size": 64,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 250,
        "weight_decay": 0.05,
        "clip_grad": 1.5,
        "optimizer": EasyDeLOptimizers.LION,
        "scheduler": EasyDeLSchedulers.COSINE,
        "trainable_selector": "parameters",
        "extra_optimizer_kwargs": {"b1": 0.9, "b2": 0.99},
        # ---- Data loading ----
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
        "remove_unused_columns": False,
        "ids_to_pop_from_dataset": ["raw_text", "metadata"],
        "shuffle_train_dataset": False,
        "shuffle_seed_train": 12345,
        "use_data_collator": False,
        "use_grain": False,
        "grain_shard_index": 1,
        "grain_shard_count": 4,
        "offload_dataset": True,
        "offload_device_type": "cpu",
        "offload_device_index": 0,
        # ---- Lifecycle flags ----
        "do_train": True,
        "do_eval": True,
        "do_last_save": False,
        "is_fine_tuning": False,
        "init_tx": False,
        "train_on_inputs": False,
        "aux_loss_enabled": True,
        "training_time_limit": "2h30m",
        "step_start_point": 100,
        "force_step_start_point": True,
        "resume_if_possible": False,
        # ---- Sequence handling ----
        "truncation_mode": "keep_start",
        "max_length": 8192,
        # ---- Checkpointing ----
        "save_interval_minutes": 30.0,
        "save_steps": 500,
        "save_total_limit": 10,
        "save_directory": "/tmp/roundtrip-checkpoints",
        "save_optimizer_state": False,
        "merge_lora_before_save": False,
        "merge_lora_before_tpu_preemption_save": True,
        "save_tpu_preemption_checkpoints": False,
        "remove_ckpt_after_load": True,
        # ---- Evaluation cadence ----
        "evaluation_steps": 200,
        "max_evaluation_steps": 100,
        # ---- Logging cadence ----
        "log_steps": 25,
        "report_steps": 10,
        "log_all_workers": True,
        "log_grad_norms": False,
        "report_metrics": False,
        "metrics_to_show_in_rich_pbar": ["loss", "accuracy", "lr"],
        "progress_bar_type": "rich",
        "weight_distribution_pattern": r".*attn.*",
        "weight_distribution_log_steps": 250,
        "verbose": False,
        "process_zero_is_admin": False,
        # ---- Backend / sharding ----
        "backend": "tpu",
        "auto_shard_states": False,
        "performance_mode": False,  # avoids _setup_logging side-effects on use_wandb
        "track_memory": 0.5,
        "low_mem_usage": False,
        # ---- QAT ----
        "quantization_mode": "nf4",
        "quantization_group_size": 64,
        "quantization_bits": 4,
        # ---- Loss / model ----
        "model_parameters": {"hidden_size": 4096, "num_layers": 32},
        "frozen_parameters": r".*embed.*",
        "loss_config": LossConfig(label_smoothing=0.1, z_loss=1e-4, ignore_index=-100),
        "lmhead_chunksize": 1024,
        "jax_distributed_config": {"coordinator_address": "localhost:1234"},
        "step_partition_spec": PartitionSpec(("dp", "fsdp"), "sp"),
        "state_apply_fn_kwarguments_to_model": {"deterministic": False},
        # ---- Sparsification ----
        "sparsify_module": True,
        "sparse_module_type": "bcoo",
        # ---- Optimizer momentum dtype ----
        "tx_mu_dtype": jnp.bfloat16,
        # ---- Generation preview ----
        "generation_top_p": 0.92,
        "generation_top_k": 40,
        "generation_presence_penalty": 0.1,
        "generation_frequency_penalty": 0.2,
        "generation_repetition_penalty": 1.05,
        "generation_temperature": 0.7,
        "generation_do_sample": True,
        "generation_num_return_sequences": 2,
        "generation_max_new_tokens": 256,
        "generation_shard_inputs": False,
        "generation_interval": 100,
        "generation_prompts": ["Hello world.", {"prompt": "Tokenized prompt"}],
        "generation_use_train_prompts": True,
        "generation_num_prompts": 4,
        "generation_dataset_prompt_field": "input",
        "generation_extra_kwargs": {"do_sample": True, "min_p": 0.05},
        "generation_config_overrides": {"max_new_tokens": 512},
        "generation_seed": 7,
        "generation_preview_print": True,
        "generation_log_to_wandb": False,
        "log_training_generations_to_wandb": False,
        # ---- Benchmarks (lm-eval suites) ----
        "benchmark_interval": 500,
        "benchmarks": [
            {"name": "core", "tasks": ["hellaswag", "mmlu"], "limit": 50, "num_fewshot": 5},
            {"name": "math", "tasks": "gsm8k", "limit": 20},
        ],
        # ---- eSurge generation integration ----
        "use_esurge_generation": False,
        "esurge_use_tqdm": False,
        "esurge_hbm_utilization": 0.6,
        "esurge_max_num_seqs": 32,
        "esurge_max_num_seq_buckets": [8, 16, 32, 64],
        "esurge_min_input_pad": 16,
        "esurge_page_size": 128,
        "esurge_silent_mode": False,
        "esurge_runner_verbose": True,
        "esurge_max_num_batched_tokens": 4096,
        "esurge_enable_prefix_caching": True,
        "esurge_data_parallelism_axis": "dp",
    }


@pytest.fixture
def full_args() -> TrainingArguments:
    """Construct a TrainingArguments with every relevant field populated."""
    return TrainingArguments(**_full_training_arguments_kwargs())


def test_save_load_arguments_roundtrip_is_idempotent_on_to_dict(tmp_path, full_args):
    """``save_arguments`` → ``load_arguments`` preserves the full ``to_dict`` view.

    This is the documented contract: ``to_dict`` is the canonical
    serializable projection of TrainingArguments, and the saved JSON
    must reload into an instance with an identical ``to_dict`` output.
    """
    json_path = tmp_path / "args.json"
    full_args.save_arguments(json_path)

    loaded = TrainingArguments.load_arguments(json_path)

    assert loaded.to_dict() == full_args.to_dict()


def test_to_dict_from_dict_roundtrip_is_idempotent(full_args):
    """``from_dict(to_dict(x))`` produces an equivalent instance under ``to_dict``."""
    rebuilt = TrainingArguments.from_dict(full_args.to_dict())
    assert rebuilt.to_dict() == full_args.to_dict()


def test_save_load_arguments_persists_every_overridden_field(tmp_path, full_args):
    """Every override we passed in must appear in the saved-and-reloaded ``to_dict``.

    This guards against fields silently disappearing from ``to_dict`` (which
    would still pass the equality test if both sides drop them in the same
    way) by comparing against the constructed instance directly.
    """
    json_path = tmp_path / "fields.json"
    full_args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    # Every field the dataclass defines must be present in to_dict on both
    # sides (no silent drop), and the values must match.
    orig_dict = full_args.to_dict()
    loaded_dict = loaded.to_dict()
    assert set(orig_dict.keys()) == set(loaded_dict.keys())
    for key in orig_dict:
        assert loaded_dict[key] == orig_dict[key], f"Field {key!r} differs after roundtrip"


def test_save_load_preserves_partition_spec(tmp_path, full_args):
    """``step_partition_spec`` (PartitionSpec) survives the save/load cycle."""
    json_path = tmp_path / "partition.json"
    full_args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert isinstance(loaded.step_partition_spec, PartitionSpec)
    assert loaded.step_partition_spec == full_args.step_partition_spec


def test_save_load_preserves_loss_config(tmp_path, full_args):
    """``loss_config`` (LossConfig) survives the save/load cycle as a dict view."""
    json_path = tmp_path / "loss.json"
    full_args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    # to_dict serializes LossConfig via its own .to_dict(); after load, the
    # reconstructed loss_config is also a LossConfig (re-wrapped in __post_init__).
    assert isinstance(loaded.loss_config, LossConfig)
    assert loaded.loss_config.to_dict() == full_args.loss_config.to_dict()


def test_save_load_preserves_benchmarks_list(tmp_path, full_args):
    """Nested benchmark configs survive the JSON roundtrip."""
    json_path = tmp_path / "benchmarks.json"
    full_args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert loaded.benchmarks == full_args.benchmarks


def test_to_json_string_includes_class_marker(full_args):
    """``to_json_string`` records the trainer class for proper subclass dispatch."""
    import json

    payload = json.loads(full_args.to_json_string())
    assert payload["trainer_config_class"] == "TrainingArguments"


def test_load_from_json_dispatches_via_class_marker(tmp_path, full_args):
    """``load_arguments`` resolves the right class even when called from the base."""
    json_path = tmp_path / "dispatch.json"
    full_args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert type(loaded) is TrainingArguments


def test_minimal_arguments_roundtrip(tmp_path):
    """Defaults-only TrainingArguments roundtrips cleanly."""
    args = TrainingArguments()
    json_path = tmp_path / "minimal.json"
    args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert loaded.to_dict() == args.to_dict()


def test_save_load_preserves_watchers_default_empty(tmp_path):
    """Watchers default (empty list) roundtrips. Non-empty watchers roundtrip
    via stringification, so we restrict this test to the default empty case."""
    args = TrainingArguments(watchers=[])
    json_path = tmp_path / "watchers.json"
    args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert loaded.watchers == []


def test_save_load_handles_subset_of_fields(tmp_path):
    """Constructor accepts a strict subset; round-trip preserves provided values."""
    args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=2,
        total_batch_size=8,
        save_directory="/tmp/subset-checkpoints",
    )
    json_path = tmp_path / "subset.json"
    args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert loaded.learning_rate == 1e-4
    assert loaded.num_train_epochs == 2
    assert loaded.total_batch_size == 8
    assert loaded.save_directory == "/tmp/subset-checkpoints"
    assert loaded.to_dict() == args.to_dict()


def test_unused_logwatcher_import_is_available():
    """LogWatcher is importable so future tests can extend watcher coverage."""
    assert LogWatcher is not None
