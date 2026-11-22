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

"""Comprehensive regression tests for Core Trainer Infrastructure & Standard Trainers review fixes.

Covers CORE-C1..CORE-C11, CORE-H1..CORE-H16, CORE-M1..CORE-M8, CORE-L1..CORE-L5,
plus RL-H16/RL-H17 (base_trainer.py) and PKD-L2 (utils.py).
"""

from __future__ import annotations

import collections
import contextlib
import copy
import functools
import gc
import inspect
import json
import os
import pickle
import threading
import time
import types
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from datasets import Dataset
from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms.pack import PackedShardedSource
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal, EasyDeLTimerError
from easydel.infra.etils import EasyDeLGradientCheckPointers, EasyDeLSchedulers
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.modules.llama import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification
from easydel.modules.qwen2 import Qwen2Config, Qwen2ForEmbedding
from easydel.trainers import (
    BEMACallback,
    BEMADPOTrainer,
    DPOConfig,
    DPOTrainer,
    EmbeddingConfig,
    EmbeddingTrainer,
    MergeConfig,
    MergeModelCallback,
    PRMConfig,
    PRMPreprocessTransform,
    PRMTrainer,
    RewardConfig,
    RewardTrainer,
    SFTConfig,
    SFTTrainer,
    Trainer,
    TrainingArguments,
    _reward_sideband,
    assistant_only_mask,
    data_lanes,
    packer,
    pose,
    process_encoder,
    prompt_transforms,
    prompt_utils,
    randomized_yarn,
    reward_protocol,
    training_utils,
)
from easydel.trainers import base_trainer as base_trainer_mod
from easydel.trainers import ray_scaler as ray_scaler_pkg
from easydel.trainers import reward_trainer as reward_trainer_pkg
from easydel.trainers import training_configurations as tc_mod
from easydel.trainers.base_trainer import BaseTrainer
from easydel.trainers.batch_placement import batch_leaf_sharding
from easydel.trainers.batch_prefetcher import TrainBatchPrefetcher, _ReplayIterator
from easydel.trainers.buckets import ModBucketRule, TrainingBucket, apply_config_overrides
from easydel.trainers.dp_sharding import (
    _batch_compatible_with_dp_vmap,
    _is_mrope_position_leaf,
    _split_batch_for_dp,
    dp_spmd_step_gradients,
    reduce_dp_weighted,
    resolve_graphstate_specs,
)
from easydel.trainers.embedding_trainer._fn import (
    _l2_normalize,
    embedding_training_step,
    infonce_loss,
    matryoshka_loss,
    triplet_loss,
)
from easydel.trainers.execution_accumulation import ExecutionAccumulation
from easydel.trainers.merge_callback import merge_pytrees
from easydel.trainers.merge_callback.merge_methods import (
    MergeLeafContext,
    _dare_ties_merge,
    _slerp_merge,
    _ties_merge,
    _topk_density_mask,
)
from easydel.trainers.metrics import LogWatcher, MetricsHistogram, MetricsTracker, StepMetrics, compute_weight_stats
from easydel.trainers.model_loading import disable_state_dropout
from easydel.trainers.ray_scaler import distributed_trainer as ray_dist_mod
from easydel.trainers.ray_scaler.distributed_trainer import RayDistributedConfig, RayDistributedTrainer
from easydel.trainers.reward_trainer._fn import (
    _make_reward_scheduled_loss,
    _reward_pair_loss,
)
from easydel.trainers.reward_trainer._fn import (
    evaluation_step as reward_evaluation_step,
)
from easydel.trainers.trainer._fn import _dft_causal_lm_metrics, base_step
from easydel.trainers.trainer_protocol import BaseTrainerProtocol
from easydel.trainers.training_configurations import get_safe_arr
from easydel.trainers.utils import (
    BCODataCollatorGrain,
    BCODataCollatorTFDS,
    CollateMapTransform,
    DataCollatorForCompletionOnlyLM,
    DataCollatorForPreferenceGrain,
    DataCollatorForPreferenceTFDS,
    DPODataCollatorWithPaddingGrain,
    DPODataCollatorWithPaddingTFDS,
    EmbeddingDataCollatorGrain,
    EmbeddingDataCollatorTFDS,
    GRPODataCollatorGrain,
    GRPODataCollatorTFDS,
    HFDataSource,
    RewardDataCollatorWithPaddingGrain,
    RewardDataCollatorWithPaddingTFDS,
    ToNumpy,
    _pad_single_along_axis,
    add_eos_token_if_needed,
    conversations_formatting_function,
    create_constant_length_dataset,
    pad_single,
    pad_to_length,
    shift_and_pad,
)
from easydel.utils.helpers import SummaryWriter
from grain import python as pygrain
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from spectrax import PartitionAxis
from transformers.tokenization_utils_base import BatchEncoding

pytestmark = pytest.mark.cpu_ok

# =============================================================================
# Domain A: Core Trainer Infrastructure & BaseTrainer
# =============================================================================

def _make_tiny_llama_state(*, with_lora: bool = False, disable_dropout: bool = False):
    cfg = ed.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
    )
    cfg.attn_dtype = jnp.float32
    model = ed.LlamaForCausalLM(
        config=cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    if with_lora:
        model = model.apply_lora_to_layers(lora_rank=4, lora_pattern="o_proj|down_proj", verbose=False)
    if disable_dropout:
        model.eval()
    return model.shard_model().to_state()


# ---------------------------------------------------------------------------
# CORE-C1: GCS save_total_limit cleanup (non-root prefix + oldest deletion)
# ---------------------------------------------------------------------------


def test_core_c1_gcs_cleanup_runs_on_non_root_prefix_and_deletes_oldest(monkeypatch):
    """Simulate GCSPath where save_dir.exists() is False, is_dir() is True, and run-*.stat() raises FileNotFoundError."""

    class _FakeGCSMetaFile:
        def __init__(self, mtime: float):
            self._mtime = mtime

        def exists(self) -> bool:
            return True

        def stat(self) -> dict[str, float]:
            return {"mtime": self._mtime}

    class _FakeGCSRunDir:
        def __init__(self, name: str, mtime: float):
            self.name = name
            self._path = f"gs://bucket/exp/{name}"
            self._meta = _FakeGCSMetaFile(mtime)

        def is_dir(self) -> bool:
            return True

        def exists(self) -> bool:
            return False

        def stat(self):
            raise FileNotFoundError(f"virtual prefix {self._path} has no blob")

        def __truediv__(self, other: str):
            assert other == "metadata.json"
            return self._meta

        def __str__(self) -> str:
            return self._path

    # Lexicographic order mimics GCS glob: run-1000 < run-200 < run-50
    run_1000 = _FakeGCSRunDir("run-1000", mtime=300.0)
    run_200 = _FakeGCSRunDir("run-200", mtime=200.0)
    run_50 = _FakeGCSRunDir("run-50", mtime=100.0)

    class _FakeGCSSaveDir:
        def exists(self) -> bool:
            # Non-root GCS prefix after mkdir() creating .keep returns False for exists()
            return False

        def is_dir(self) -> bool:
            return True

        def glob(self, pattern: str):
            assert pattern == "run-*"
            return [run_1000, run_200, run_50]

    meta_by_path = {
        str(run_1000): {"step": 1000, "timestamp": "2026-11-22T03:00:00Z", "is_temporary": False},
        str(run_200): {"step": 200, "timestamp": "2026-11-22T02:00:00Z", "is_temporary": False},
        str(run_50): {"step": 50, "timestamp": "2026-11-22T01:00:00Z", "is_temporary": False},
    }
    monkeypatch.setattr("easydel.trainers.base_trainer.ePath", lambda _: _FakeGCSSaveDir())
    monkeypatch.setattr("easydel.trainers.base_trainer.is_remote_path", lambda _: True)
    monkeypatch.setattr("spectrax.serialization.read_checkpoint_metadata", lambda p: meta_by_path[p])

    queued: list[str] = []
    stub = SimpleNamespace(
        arguments=SimpleNamespace(_get_save_directory=lambda: "gs://bucket/exp", save_total_limit=2),
        checkpointer=SimpleNamespace(_queue_checkpoint_removal=queued.append),
    )

    BaseTrainer._cleanup_old_checkpoints(stub)
    assert queued == [str(run_50)], f"Expected oldest run-50 deleted, got {queued}"


# ---------------------------------------------------------------------------
# CORE-C4: loss_type="dft" without explicit labels & honoring attention_mask
# ---------------------------------------------------------------------------


def test_core_c4_dft_metrics_honors_attention_mask_and_base_step_defaults_labels():
    logits = jnp.array(
        [[[2.0, 0.1, -1.0, 0.0], [0.2, 3.0, -0.5, 0.0], [-2.0, -2.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    input_ids = jnp.array([[1, 0, 2, 3]], dtype=jnp.int32)
    attention_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    labels_masked = jnp.array([[1, 0, -100, -100]], dtype=jnp.int32)

    m_with_attn = _dft_causal_lm_metrics(logits, input_ids, attention_mask=attention_mask)
    m_with_neg100 = _dft_causal_lm_metrics(logits, labels_masked, attention_mask=None)
    m_unmasked = _dft_causal_lm_metrics(logits, input_ids, attention_mask=None)

    assert float(m_with_attn.weight_sum) == 1.0
    assert jnp.allclose(m_with_attn.loss, m_with_neg100.loss, atol=1e-6)
    assert jnp.allclose(m_with_attn.accuracy, m_with_neg100.accuracy, atol=1e-6)
    assert float(m_unmasked.weight_sum) == 3.0

    # End-to-end base_step with loss_type="dft" and NO "labels" key in batch
    state = _make_tiny_llama_state()
    import optax

    state = state.init_tx(optax.adam(1e-3))
    batch = {
        "input_ids": jnp.array([[2, 3, 4, 5, 0, 0], [6, 7, 8, 0, 0, 0]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=jnp.int32),
    }
    eval_metrics = base_step(state=state, batch=batch, is_training=False, loss_type="dft")
    assert np.isfinite(float(eval_metrics.loss))
    # Shifted valid tokens: row 0 has 3 valid shifted positions, row 1 has 2 -> weight_sum == 5
    assert float(eval_metrics.weight_sum) == 5.0

    new_state, train_metrics = base_step(
        state=state,
        batch=batch,
        learning_rate_fn=lambda _: 1e-3,
        is_training=True,
        loss_type="dft",
    )
    assert int(new_state.step) == 1
    assert np.isfinite(float(train_metrics.loss))
    assert float(train_metrics.weight_sum) == 5.0


# ---------------------------------------------------------------------------
# CORE-C8: step-skip, EasyDeLBreakRequest, and _finalize_training run_exception
# ---------------------------------------------------------------------------


def test_core_c8_skipped_steps_avoid_duplicate_save_eval_and_infinite_loop():
    save_calls: list[int] = []
    eval_calls: list[int] = []
    pbar_updates: list[int] = []

    def _exec_skip(state, batch):
        del batch
        return state, LossMetrics(loss=jnp.asarray(float("nan"))), None

    host = SimpleNamespace(
        arguments=SimpleNamespace(
            dataloader_prefetch=False,
            max_length=8,
            learning_rate=1e-3,
            merge_lora_before_save=False,
            log_all_workers=False,
            profiler_include_compile=False,
            max_consecutive_skipped_steps=3,
        ),
        data_collator=None,
        max_training_steps=10,
        _has_buckets=False,
        _bucket_for_step=0,
        _get_next_batch=lambda it, ds: (next(it), it),
        _execute_train_step=_exec_skip,
        _runtime_batch_summary=lambda b: b,
        _runtime_trace=lambda *a, **kw: None,
        _log_mesh_topology=lambda: None,
        _apply_pose_to_batch=lambda b, s: b,
        _apply_randomized_yarn_to_batch=lambda b, s: b,
        _apply_process_encoder_to_batch=lambda b, s: b,
        on_step_start=lambda state, step: state,
        on_step_end=lambda state, metrics, step: (state, metrics),
        apply_training_hooks=lambda metrics, loss_value=None: metrics,
        _metric_scalar_to_float=float,
        _maybe_start_profiler=lambda step: None,
        _maybe_stop_profiler=lambda step: None,
        _should_save_tpu_preemption_checkpoint=lambda step: False,
        _should_run_evaluation=lambda step: True,
        _profiler_should_block_until_ready=lambda: False,
        _save_checkpoint_for_step=lambda state, step, **kw: save_calls.append(step),
        eval=lambda model_state: iter(eval_calls.append(1) or []),
        log_metrics=lambda **kw: pbar_updates.append(kw.get("step")),
        log_weight_distribution=lambda **kw: None,
        log_watchers=lambda **kw: None,
        maybe_generate=lambda **kw: None,
        maybe_benchmark=lambda **kw: None,
        train_tracker=SimpleNamespace(trace_compilation=contextlib.nullcontext),
        scheduler=None,
        _backward_flops_per_token=0.0,
        _extra_backward_flops_per_token=0.0,
        training_batch_size=1,
    )
    pbar = SimpleNamespace(update=lambda n: pbar_updates.append(n), set_postfix=lambda **kw: None)
    tracker = SimpleNamespace(update=lambda **kw: (0.0, 0.0))
    step_metrics = SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {})
    state = SimpleNamespace(step=jnp.asarray(2))
    data = list(range(20))

    _, exc, _ = Trainer._train_epoch(
        host,
        state=state,
        train_dataset=data,
        train_iter=iter(data),
        metrics_tracker=tracker,
        step_metrics=step_metrics,
        pbar=pbar,
        epoch=0,
        epoch_start_step=0,
        epoch_end_step=10,
    )
    assert isinstance(exc, EasyDeLBreakRequest)
    assert save_calls == []
    assert eval_calls == []
    assert pbar_updates == []


def test_core_c8_prepare_training_output_handles_break_request_and_finalize_skips_eval():
    saved_steps: list[int] = []
    stub = SimpleNamespace(
        arguments=SimpleNamespace(
            do_last_save=True,
            merge_lora_before_save=False,
            save_directory="/tmp/ckpt",
        ),
        mesh=None,
        _preemption_checkpoint_path=None,
        _last_saved_checkpoint=None,
        _train_end_callbacks_dispatched=False,
        _is_memory_oom_exception=lambda exc: False,
        _save_checkpoint_for_step=lambda state, step, **kw: (saved_steps.append(step) or f"run-{step}"),
        _dispatch_callbacks=lambda *a, **kw: None,
    )
    state = SimpleNamespace(step=jnp.asarray(7))
    output = BaseTrainer._prepare_training_output(stub, state=state, run_exception=EasyDeLBreakRequest("NaN loss"))
    assert saved_steps == [7]
    assert output.last_save_file_name == "run-7"

    # _finalize_training must skip eval when run_exception is non-None
    eval_called = []
    finish_called = []
    trainer_stub = SimpleNamespace(
        arguments=SimpleNamespace(do_eval=True),
        eval=lambda st: iter(eval_called.append(st) or []),
        finish=lambda: finish_called.append(True),
    )
    for exc in (EasyDeLPreemptionSignal("preempt"), KeyboardInterrupt(), EasyDeLTimerError("timeout"), EasyDeLBreakRequest("nan")):
        Trainer._finalize_training(trainer_stub, output, exc)
    assert eval_called == []
    assert len(finish_called) == 4

    Trainer._finalize_training(trainer_stub, output, None)
    assert len(eval_called) == 1
    assert len(finish_called) == 5


# ---------------------------------------------------------------------------
# CORE-C9: _run_scheduled_value_and_grad zero-weight NaN & bf16 dtype preservation
# ---------------------------------------------------------------------------


def test_core_c9_scheduled_value_and_grad_skips_zero_weight_nan_and_preserves_bf16():
    def vag(tree, mb):
        is_empty = bool(jnp.all(mb["mask"] == 0))
        if is_empty:
            nan_loss = jnp.asarray(jnp.nan, dtype=jnp.float32)
            nan_grads = {"w": jnp.full_like(tree["w"], jnp.nan)}
            nan_aux = {"accuracy": jnp.asarray(jnp.nan, dtype=jnp.float32), "weight_sum": jnp.asarray(0.0, dtype=jnp.float32)}
            return nan_loss, nan_grads, nan_aux
        loss = jnp.asarray(2.5, dtype=jnp.float32)
        grads = {"w": jnp.ones_like(tree["w"]) * jnp.asarray(3.0, dtype=jnp.bfloat16)}
        aux = {"accuracy": jnp.asarray(0.75, dtype=jnp.float32), "weight_sum": jnp.sum(mb["mask"]).astype(jnp.float32)}
        return loss, grads, aux

    graphstate = {"w": jnp.ones((4,), dtype=jnp.bfloat16)}
    batch = {
        "x": jnp.arange(4, dtype=jnp.float32),
        "mask": jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32),
    }
    loss, grads, aux = training_utils._run_scheduled_value_and_grad(
        value_and_grad=vag,
        graphstate=graphstate,
        batch=batch,
        batch_size=4,
        minibatch_size=2,
        microbatch_weight_fn=lambda _gs, mb: jnp.sum(mb["mask"]),
    )
    assert np.isfinite(float(loss)) and float(loss) == pytest.approx(2.5)
    assert grads["w"].dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(grads["w"], dtype=np.float32), np.full((4,), 3.0))
    assert float(aux["accuracy"]) == pytest.approx(0.75)
    assert float(aux["weight_sum"]) == pytest.approx(2.0)

    # All-zero weights with NaN chunks must produce finite zeros
    empty_batch = {"x": batch["x"], "mask": jnp.zeros((4,), dtype=jnp.float32)}
    loss0, grads0, aux0 = training_utils._run_scheduled_value_and_grad(
        value_and_grad=vag,
        graphstate=graphstate,
        batch=empty_batch,
        batch_size=4,
        minibatch_size=2,
        microbatch_weight_fn=lambda _gs, mb: jnp.sum(mb["mask"]),
    )
    assert float(loss0) == 0.0
    assert grads0["w"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(grads0["w"], dtype=np.float32), np.zeros((4,)))
    assert float(aux0["accuracy"]) == 0.0
    assert float(aux0["weight_sum"]) == 0.0


# ---------------------------------------------------------------------------
# CORE-C11: dp_sharding mRoPE (3, B, S), unaligned multimodal, and graphother
# ---------------------------------------------------------------------------


def test_core_c11_dp_sharding_mrope_multimodal_and_graphother():
    mrope_pos = jnp.arange(3 * 4 * 6, dtype=jnp.int32).reshape(3, 4, 6)
    batch_mrope = {"input_ids": jnp.ones((4, 6), dtype=jnp.int32), "position_ids": mrope_pos}
    assert _is_mrope_position_leaf("position_ids", mrope_pos, 4)
    assert _batch_compatible_with_dp_vmap(batch_mrope, 4)

    batch_unaligned_mm = {
        "input_ids": jnp.ones((4, 6), dtype=jnp.int32),
        "pixel_values": jnp.ones((12, 16), dtype=jnp.float32),
    }
    assert not _batch_compatible_with_dp_vmap(batch_unaligned_mm, 4)

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1), ("dp",))
    split = _split_batch_for_dp(batch_mrope, batch_size=4, partition_spec=PartitionSpec("dp"), jax_mesh=mesh)
    assert split["position_ids"].shape == (1, 3, 4, 6)
    np.testing.assert_array_equal(np.asarray(split["position_ids"][0]), np.asarray(mrope_pos))

    # resolve_graphstate_specs on graphother resolves actual NamedSharding specs instead of PartitionSpec()
    sharded_buf = jax.device_put(jnp.ones((4,), dtype=jnp.float32), NamedSharding(mesh, PartitionSpec("dp")))
    fake_state = SimpleNamespace(
        model=None,
        graphstate={"w": jnp.ones((4,), dtype=jnp.float32)},
        graphother={"frozen_base": sharded_buf, "count": 3},
    )
    go_specs = resolve_graphstate_specs(fake_state, mesh, tree=fake_state.graphother)
    assert go_specs["frozen_base"] == PartitionSpec("dp")
    assert go_specs["count"] == PartitionSpec()

    # dp_spmd_step_gradients must return None when graphother mentions dp_axis
    fake_state_with_mesh = SimpleNamespace(
        model=SimpleNamespace(mesh=mesh, resolve_shardings_regex=lambda mesh=None: []),
        graphstate={"w": jnp.ones((4,), dtype=jnp.float32)},
        graphother={"frozen_base": sharded_buf},
    )
    res = dp_spmd_step_gradients(
        student_state=fake_state_with_mesh,
        batch={"input_ids": jnp.ones((4, 6), dtype=jnp.int32)},
        batch_size=4,
        minibatch_size=4,
        partition_spec=PartitionSpec("dp"),
        slice_grad_and_weight_fn=lambda *a: None,
    )
    assert res is None


# ---------------------------------------------------------------------------
# CORE-H1: Grain-lane dataloader with dataset_eval and dataset_train=None
# ---------------------------------------------------------------------------


def test_core_h1_grain_lane_with_eval_dataset_does_not_touch_dataset_train(monkeypatch):
    eval_ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 4, "attention_mask": [[1, 1, 1, 1]] * 4})
    eval_source = BaseTrainer._to_sharded_source(SimpleNamespace(), eval_ds)

    import easydel.trainers.data_lanes as data_lanes

    monkeypatch.setattr(data_lanes, "configure_parquetra_lane", lambda trainer: (["train_batch"], 123))

    metadata_calls: list[tuple[bool, int]] = []
    stub = SimpleNamespace(
        arguments=SimpleNamespace(
            parquetra_train_datasets=["dummy.parquet"],
            do_eval=True,
            use_grain=True,
            max_length=4,
            truncation_mode="keep_end",
            shuffle_seed_train=0,
            grain_shard_index=0,
            grain_shard_count=1,
            max_evaluation_steps=None,
        ),
        dataset_train=None,
        _train_source=None,
        dataset_eval=eval_ds,
        _eval_source=eval_source,
        evaluation_batch_size=2,
        _set_dataset_size_metadata=lambda *, is_train, resolution: metadata_calls.append((is_train, resolution.steps)),
        _resolve_step_count=lambda ds, source, is_train, drop_remainder: SimpleNamespace(steps=2),
        _create_dataloader_from_source=lambda **kw: iter([{"input_ids": np.ones((2, 4), dtype=np.int32)}]),
    )
    stub._configure_eval_dataloader = BaseTrainer._configure_eval_dataloader.__get__(stub, type(stub))
    out = BaseTrainer._configure_grain_lane_dataloader(stub)
    assert out.max_training_steps == 123
    assert out.max_evaluation_steps == 2
    assert out.dataloader_eval is not None
    assert metadata_calls == [(True, 123), (False, 2)]


# ---------------------------------------------------------------------------
# CORE-H2: Raw-text bucket_datasets tokenization + sequence_packing
# ---------------------------------------------------------------------------


def test_core_h2_bucket_preprocess_uses_bucket_source_and_max_length():
    from easydel.data.sources.base import ShardedDataSource
    from easydel.data.transforms.pack import PackedShardedSource

    seen: list[tuple[object, int]] = []
    primary_source = object()
    bucket_source = object()

    stub = SimpleNamespace(
        _train_source=primary_source,
        arguments=SimpleNamespace(max_length=64),
    )

    def _get_preprocess():
        seen.append((stub._train_source, stub.arguments.max_length))
        return "tok_transform"

    stub._get_preprocess_transform = _get_preprocess
    stub._preprocess_transform_for_source = BaseTrainer._preprocess_transform_for_source.__get__(stub, type(stub))

    bucket = TrainingBucket(name="short", max_length=16)
    transform = BaseTrainer._get_bucket_preprocess_transform(stub, bucket_source, bucket)
    assert transform == "tok_transform"
    assert seen == [(bucket_source, 16)]
    assert stub._train_source is primary_source
    assert stub.arguments.max_length == 64

    # max_length=None must still override to bucket.max_length and restore None afterwards
    stub.arguments.max_length = None
    seen.clear()
    BaseTrainer._get_bucket_preprocess_transform(stub, bucket_source, bucket)
    assert seen == [(bucket_source, 16)]
    assert stub.arguments.max_length is None

    # _pack_token_source on streaming ShardedDataSource (no __len__) wrapped in PackedShardedSource
    class _StreamingNoLenSource(ShardedDataSource[dict]):
        @property
        def shard_names(self) -> list[str]:
            return ["s0"]

        def num_shards(self) -> int:
            return 1

        def open_shard(self, shard_name: str):
            del shard_name
            yield {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

    stream_src = _StreamingNoLenSource()
    packed_16 = PackedShardedSource(
        source=stream_src,
        seq_length=16,
        eos_token_id=2,
        pad_token_id=0,
        strategy="greedy",
        include_segment_ids=True,
        shuffle=False,
    )
    pack_stub = SimpleNamespace(
        arguments=SimpleNamespace(max_length=64),
        _tokenizer_int_attr=lambda name, default=None: 2 if name == "eos_token_id" else 0,
        _sequence_packing_extra_fields=lambda src: ({}, {}),
        _sequence_packing_strategy=lambda: "greedy",
    )
    # Matching seq_length returns the exact PackedShardedSource without calling __len__()
    assert BaseTrainer._pack_token_source(pack_stub, packed_16, seq_length=16) is packed_16
    # Different seq_length unwraps _source without calling bool()/__len__() and repacks to 32
    repacked_32 = BaseTrainer._pack_token_source(pack_stub, packed_16, seq_length=32)
    assert repacked_32 is not packed_16
    assert repacked_32._seq_length == 32
    assert repacked_32._source is stream_src


# ---------------------------------------------------------------------------
# CORE-H3: Bucket & eval config override graphdefs preserve LoRA & disable_dropout
# ---------------------------------------------------------------------------


def test_core_h3_config_override_preserves_lora_and_disable_dropout():
    from easydel.utils.traversals import iter_module_search

    state = _make_tiny_llama_state(with_lora=True, disable_dropout=True)
    assert state.model.lora_is_enabled
    # Populate non-zero lora_b so any alpha=0.0 bug zeroes out the LoRA delta visibly.
    orig_model = state.model
    orig_loras = list(iter_module_search(orig_model, spx.nn.LoRA))
    assert orig_loras, "Expected at least one LoRA layer"
    for _, lora_mod in orig_loras:
        lora_mod.lora_b.value = jnp.full_like(lora_mod.lora_b.value, 0.5)
    state = orig_model.to_state()
    orig_model = state.model
    orig_loras = list(iter_module_search(orig_model, spx.nn.LoRA))

    stub = SimpleNamespace(arguments=SimpleNamespace(disable_dropout=True, trainable_selector="parameters"))
    stub._build_variant_module = BaseTrainer._build_variant_module.__get__(stub, type(stub))
    gdef = BaseTrainer._build_config_override_graphdef(
        stub,
        model=orig_model,
        overrides={"attn_mechanism": "sdpa"},
        reference_graphstate=state.graphstate,
        trainable_selector="parameters",
        label="test_lora_override",
        eval_mode=False,
    )
    swapped = state.replace(graphdef=gdef).model
    assert str(swapped.config.attn_mechanism) == "sdpa"
    assert swapped.lora_is_enabled
    assert not swapped.training

    swap_loras = dict(iter_module_search(swapped, spx.nn.LoRA))
    for path, orig_lora in orig_loras:
        swap_lora = swap_loras[path]
        assert swap_lora.has_alpha == orig_lora.has_alpha
        assert orig_lora.has_alpha is False
        x = jnp.ones((1, 4, int(orig_lora.d_in)), dtype=jnp.float32)
        y_orig = orig_lora(x)
        y_swap = swap_lora(x)
        y_base = swap_lora.base_module(x)
        np.testing.assert_allclose(np.asarray(y_swap), np.asarray(y_orig), atol=1e-5, rtol=1e-5)
        assert float(jnp.max(jnp.abs(y_swap - y_base))) > 1e-3, "LoRA delta was zeroed out!"

    # Also verify explicit alpha (has_alpha=True) is preserved accurately
    for _, lora_mod in orig_loras:
        lora_mod.has_alpha = True
        lora_mod.alpha = 8.0
    state_alpha = orig_model.to_state()
    gdef_alpha = BaseTrainer._build_config_override_graphdef(
        stub,
        model=state_alpha.model,
        overrides={"attn_mechanism": "sdpa"},
        reference_graphstate=state_alpha.graphstate,
        trainable_selector="parameters",
        label="test_lora_alpha_override",
        eval_mode=False,
    )
    swapped_alpha = state_alpha.replace(graphdef=gdef_alpha).model
    for path, orig_lora in iter_module_search(state_alpha.model, spx.nn.LoRA):
        swap_lora = dict(iter_module_search(swapped_alpha, spx.nn.LoRA))[path]
        assert swap_lora.has_alpha is True
        assert swap_lora.alpha == pytest.approx(8.0)
        x = jnp.ones((1, 4, int(orig_lora.d_in)), dtype=jnp.float32)
        np.testing.assert_allclose(np.asarray(swap_lora(x)), np.asarray(orig_lora(x)), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# CORE-H4: _leaf_batch_partition_spec for (3, B, S) on batch_ways in {1, 3} + nested constrain
# ---------------------------------------------------------------------------


def test_core_h4_mrope_position_ids_partition_spec_and_nested_constrain(monkeypatch):
    spec = PartitionSpec(("dp", "fsdp"), "sp")
    # Pure SP mesh: batch_ways == 1
    out_sp = training_utils._leaf_batch_partition_spec(
        (3, 4, 16), spec, {"dp": 1, "fsdp": 1, "sp": 4}, key="position_ids"
    )
    assert out_sp == PartitionSpec(None, ("dp", "fsdp"), "sp")

    # dp=3 mesh: batch_ways == 3
    out_dp3 = training_utils._leaf_batch_partition_spec(
        (3, 6, 16), spec, {"dp": 3, "fsdp": 1, "sp": 4}, key="position_ids"
    )
    assert out_dp3 == PartitionSpec(None, ("dp", "fsdp"), "sp")

    # batch_leaf_sharding & place_batch propagate key
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("dp", "sp"))
    sh = batch_leaf_sharding((3, 4, 16), PartitionSpec("dp", "sp"), mesh, key="position_ids")
    assert sh.spec == PartitionSpec(None, "dp", "sp")

    # Nested containers in constrain_batch_sharding receive constraints
    constrained_specs: list[tuple[tuple[int, ...], PartitionSpec]] = []

    def _fake_constraint(leaf, pspec, *, mesh, ignore_mpmd=True):
        constrained_specs.append((tuple(leaf.shape), pspec))
        return leaf

    monkeypatch.setattr(training_utils.spx, "with_sharding_constraint", _fake_constraint)
    batch = {
        "position_ids": jnp.zeros((3, 4, 16), dtype=jnp.int32),
        "deepstack_visual_embeds": [jnp.zeros((4, 16, 8), dtype=jnp.float32), jnp.zeros((4, 16, 8), dtype=jnp.float32)],
    }
    fake_mesh = SimpleNamespace(shape={"dp": 1, "fsdp": 1, "sp": 4})
    training_utils.constrain_batch_sharding(batch, spec, mesh=fake_mesh)
    assert ((3, 4, 16), PartitionSpec(None, ("dp", "fsdp"), "sp")) in constrained_specs
    assert sum(1 for shape, _ in constrained_specs if shape == (4, 16, 8)) == 2


# ---------------------------------------------------------------------------
# CORE-H5: Zero-weight NaN guards, fp32 scan accumulation, and scan vs execution parity
# ---------------------------------------------------------------------------


def test_core_h5_zero_weight_nan_guards_and_bf16_fp32_accumulation():
    # 1. minibatch_call zero-weight NaN guard + bf16 fp32 accumulation
    state = SimpleNamespace(graphstate=jnp.asarray(1.0, dtype=jnp.bfloat16))
    batch = {
        "x": jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        "mask": jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32),
    }

    def nan_on_empty_grad_fn(param, mb):
        w = jnp.sum(mb["mask"])
        val = jnp.where(w > 0, jnp.asarray(2.0, dtype=jnp.float32), jnp.asarray(jnp.nan, dtype=jnp.float32))
        grad = jnp.where(w > 0, jnp.asarray(4.0, dtype=param.dtype), jnp.asarray(jnp.nan, dtype=param.dtype))
        return (val, LossMetrics(loss=val, weight_sum=w)), grad

    g, m = training_utils.minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=2,
        grad_fn=nan_on_empty_grad_fn,
        microbatch_weight_fn=lambda _p, mb: jnp.sum(mb["mask"]),
    )
    assert g.dtype == jnp.bfloat16 and float(g) == pytest.approx(4.0)
    assert float(m.loss) == pytest.approx(2.0)

    # 2. bf16 precision: 64 uniform microbatches of 0.01 would swamp in bf16 if accumulated in bf16
    # First microbatch has grad 1.0, remaining 63 have grad 0.001 (in bf16, 1.0 + 0.001 == 1.0!).
    vals = jnp.concatenate([jnp.array([1.0]), jnp.full((63,), 0.001)]).astype(jnp.bfloat16)
    batch64 = {"g": vals}

    def bf16_grad_fn(param, mb):
        g_val = mb["g"][0].astype(param.dtype)
        return (g_val.astype(jnp.float32), g_val.astype(jnp.float32)), g_val

    g64, _ = training_utils.minibatch_call(
        state=SimpleNamespace(graphstate=jnp.asarray(0.0, dtype=jnp.bfloat16)),
        batch=batch64,
        minibatch_size=1,
        grad_fn=bf16_grad_fn,
    )
    expected_fp32 = float(jnp.mean(vals.astype(jnp.float32)))
    assert abs(float(g64) - expected_fp32) < 5e-4

    # 3. accumulate_microbatch & fold_microbatch_metrics zero-weight NaN guard
    acc = training_utils.zero_gradient_accumulator(state.graphstate)
    acc, sm0, w0 = training_utils.accumulate_microbatch(
        acc, state.graphstate, {"x": batch["x"][:2], "mask": batch["mask"][:2]}, nan_on_empty_grad_fn, lambda _p, mb: jnp.sum(mb["mask"])
    )
    fm = training_utils.fold_microbatch_metrics(None, sm0, w0)
    acc, sm1, w1 = training_utils.accumulate_microbatch(
        acc, state.graphstate, {"x": batch["x"][2:], "mask": batch["mask"][2:]}, nan_on_empty_grad_fn, lambda _p, mb: jnp.sum(mb["mask"])
    )
    fm = training_utils.fold_microbatch_metrics(fm, sm1, w1)
    eg, em = training_utils.finalize_accumulation(acc, fm, state.graphstate)
    assert float(eg) == pytest.approx(4.0)
    assert float(em.loss) == pytest.approx(2.0)

    # 4. reduce_dp_weighted m_red zero-weight NaN guard
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1), ("dp",))
    g_dp = jnp.asarray([jnp.nan], dtype=jnp.float32)
    m_dp = LossMetrics(loss=jnp.asarray([jnp.nan], dtype=jnp.float32), weight_sum=jnp.asarray([0.0], dtype=jnp.float32))
    w_dp = jnp.asarray([0.0], dtype=jnp.float32)
    gr, mr, tw = reduce_dp_weighted(g_dp, m_dp, w_dp, PartitionSpec(), mesh, dp_axis="dp")
    assert float(gr) == 0.0
    assert float(mr.loss) == 0.0
    assert float(tw) == 0.0


# ---------------------------------------------------------------------------
# CORE-H6: Prefetcher leak on unhandled exception + flat replay iterator + parallel deepcopy
# ---------------------------------------------------------------------------


def test_core_h6_prefetcher_flat_replay_and_exception_cleanup():
    # 1. Multi-epoch chaining must stay flat (no nested _ReplayIterator)
    raw = list(range(30))
    it = iter(raw)
    consumed = []
    for _ in range(6):
        pf = TrainBatchPrefetcher(
            fetch_fn=lambda cur_it, dl: (next(cur_it), cur_it),
            data_iter=it,
            dataloader=raw,
            data_collator=lambda x: x,
            worker_count=2,
            buffer_size=3,
        )
        batch, _, _ = pf.next(schedule_next=True)
        consumed.append(batch)
        it = pf.data_iter
        pf.close()
        assert isinstance(it, _ReplayIterator)
        assert it._owner is None
        assert not isinstance(it._tail, _ReplayIterator)
    assert consumed == list(range(6))
    assert list(it) == list(range(6, 30))

    # 2. Unhandled ValueError in _train_epoch closes prefetcher
    created_prefetchers: list[TrainBatchPrefetcher] = []
    orig_init = TrainBatchPrefetcher.__init__

    def _track_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created_prefetchers.append(self)

    host = SimpleNamespace(
        arguments=SimpleNamespace(
            dataloader_prefetch=True,
            dataloader_prefetch_workers=1,
            dataloader_prefetch_buffer_size=1,
            max_length=8,
            learning_rate=1e-3,
            merge_lora_before_save=False,
            profiler_include_compile=False,
        ),
        data_collator=lambda b: b,
        max_training_steps=2,
        _has_buckets=False,
        _bucket_for_step=0,
        _get_next_batch=lambda cur_it, dl: (next(cur_it), cur_it),
        _execute_train_step=lambda state, batch: (_ for _ in ()).throw(ValueError("boom")),
        _runtime_batch_summary=lambda b: b,
        _runtime_trace=lambda *a, **kw: None,
        _log_mesh_topology=lambda: None,
        _apply_pose_to_batch=lambda b, s: b,
        _apply_randomized_yarn_to_batch=lambda b, s: b,
        _apply_process_encoder_to_batch=lambda b, s: b,
        on_step_start=lambda state, step: state,
        train_tracker=SimpleNamespace(trace_compilation=contextlib.nullcontext),
    )
    TrainBatchPrefetcher.__init__ = _track_init
    try:
        with pytest.raises(ValueError, match="boom"):
            Trainer._train_epoch(
                host,
                state=SimpleNamespace(step=0),
                train_dataset=list(range(5)),
                train_iter=iter(range(5)),
                metrics_tracker=SimpleNamespace(),
                step_metrics=SimpleNamespace(start_step=lambda: None),
                pbar=SimpleNamespace(),
                epoch=0,
                epoch_start_step=0,
                epoch_end_step=2,
            )
    finally:
        TrainBatchPrefetcher.__init__ = orig_init

    assert len(created_prefetchers) == 1
    assert created_prefetchers[0]._closed is True
    assert created_prefetchers[0]._executor._shutdown is True


def test_core_h6_prefetcher_deepcopy_does_not_block_next_fetch():
    second_fetch_started = threading.Event()
    allow_first_deepcopy = threading.Event()

    class _SlowCopyBatch:
        def __init__(self, idx: int):
            self.idx = idx

        def __deepcopy__(self, memo):
            if self.idx == 0:
                allow_first_deepcopy.wait(timeout=2.0)
            return _SlowCopyBatch(self.idx)

    items = [_SlowCopyBatch(0), _SlowCopyBatch(1), _SlowCopyBatch(2)]

    def _fetch(cur_it, dl):
        item = next(cur_it)
        if item.idx == 1:
            second_fetch_started.set()
            allow_first_deepcopy.set()
        return item, cur_it

    pf = TrainBatchPrefetcher(
        fetch_fn=_fetch,
        data_iter=iter(items),
        dataloader=items,
        data_collator=lambda x: x.idx,
        worker_count=2,
        buffer_size=2,
    )
    try:
        assert second_fetch_started.wait(timeout=2.0), "Ticket 1 fetch was blocked by ticket 0 deepcopy!"
        b0, _, _ = pf.next(schedule_next=False)
        b1, _, _ = pf.next(schedule_next=False)
        assert (b0, b1) == (0, 1)
    finally:
        allow_first_deepcopy.set()
        pf.close()


# ---------------------------------------------------------------------------
# CORE-H7: Bounded aux cache, key-aware aux concat, position_paths cache key, xla options
# ---------------------------------------------------------------------------


def test_core_h7_aux_cache_concat_position_paths_and_xla_options(monkeypatch):
    # 1. _aux_concat_microbatch_outputs handles (3, chunk, S) position_ids & 0-d scalars
    mb0 = {
        "position_ids": jnp.arange(3 * 2 * 4, dtype=jnp.int32).reshape(3, 2, 4),
        "logits": jnp.ones((2, 4), dtype=jnp.float32),
        "mean_score": jnp.asarray(1.0, dtype=jnp.float32),
    }
    mb1 = {
        "position_ids": jnp.arange(3 * 2 * 4, dtype=jnp.int32).reshape(3, 2, 4) + 100,
        "logits": jnp.ones((2, 4), dtype=jnp.float32) * 2.0,
        "mean_score": jnp.asarray(3.0, dtype=jnp.float32),
    }
    out = training_utils._aux_concat_microbatch_outputs([mb0, mb1], chunk=2)
    assert out["position_ids"].shape == (3, 4, 4)
    assert out["logits"].shape == (4, 4)
    assert float(out["mean_score"]) == pytest.approx(2.0)

    # 2. Bounded auxiliary cache does not grow unboundedly
    cache = training_utils._BoundedScheduledAuxCache(maxsize=4)
    for i in range(10):
        cache[i] = i
    assert len(cache) == 4
    assert list(cache.keys()) == [6, 7, 8, 9]

    # 3. _ScheduledValueAndGradCompiler includes position_paths in cache key & forwards jit_kwargs
    jit_calls: list[dict] = []
    monkeypatch.setattr(training_utils.spx, "jit", lambda fn, **kw: (jit_calls.append(kw) or fn))
    monkeypatch.setattr(
        training_utils.spx,
        "sxvalue_and_grad",
        lambda fn, argnums=0, has_aux=False, microbatch_weight_fn=None: (lambda t, b: (fn(t, b), (t,))),
    )
    adapter = training_utils.ScheduledLossAdapter(
        name="pos_test",
        make_loss=lambda call: (lambda tree, batch: jnp.asarray(0.0)),
        make_cache_key=lambda call: ("same",),
    )
    compiler = training_utils._ScheduledValueAndGradCompiler(
        mesh=SimpleNamespace(is_mpmd=True),
        schedule=object(),
        batch_argnums=(1,),
        adapter=adapter,
        jit_kwargs={"compiler_options": {"xla_flag": "1"}},
    )
    text_call = training_utils.ScheduledStepCall(
        step_fn=lambda: None,
        state=None,
        batch={"input_ids": jnp.ones((2, 4), dtype=jnp.int32)},
        args=(),
        kwargs={},
        bound_arguments={},
        schedule=None,
    )
    vlm_call = training_utils.ScheduledStepCall(
        step_fn=lambda: None,
        state=None,
        batch={
            "input_ids": jnp.ones((2, 4), dtype=jnp.int32),
            "position_ids": jnp.ones((3, 2, 4), dtype=jnp.int32),
        },
        args=(),
        kwargs={},
        bound_arguments={},
        schedule=None,
    )
    compiler.get(text_call)
    compiler.get(vlm_call)
    assert len(jit_calls) == 2
    assert jit_calls[0]["compiler_options"] == {"xla_flag": "1"}


# ---------------------------------------------------------------------------
# CORE-H8: Trainer.create_tfds_collect_function scalars & pixel_values
# ---------------------------------------------------------------------------


def test_core_h8_tfds_collect_preserves_scalars_and_pixel_values():
    trainer_stub = SimpleNamespace(model=SimpleNamespace(lossfn_type="ForCausalLM"))
    collate = Trainer.create_tfds_collect_function(trainer_stub, max_sequence_length=4, truncation_mode="keep_start")
    rows = [
        {"input_ids": list(range(10)), "pixel_values": np.ones((3, 8, 8), dtype=np.float32), "weight": 0.5},
        {"input_ids": list(range(10, 20)), "pixel_values": np.zeros((3, 8, 8), dtype=np.float32), "weight": 1.5},
    ]
    out = collate(rows)
    assert out["input_ids"].shape == (2, 4)
    assert out["pixel_values"].shape == (2, 3, 8, 8)
    assert out["weight"].shape == (2,)


# ---------------------------------------------------------------------------
# CORE-H9: save_information, _setup_pose(None), _prepare_generation_input padding_side
# ---------------------------------------------------------------------------


def test_core_h9_save_information_setup_pose_and_padding_side_restore(tmp_path):
    # 1. save_information writes file (no IsADirectoryError)
    info_path = tmp_path / "nested" / "model_info.md"
    stub = SimpleNamespace(_get_information=lambda: "# Info\nOK\n")
    BaseTrainer.save_information(stub, info_path)
    assert info_path.is_file()
    assert info_path.read_text() == "# Info\nOK\n"

    # 2. _setup_pose with processing_class=None does not crash
    pose_cfg = ed.trainers.PoSEConfig(enabled=True, p_pose=0.5, target_max_pos=64)
    pose_stub = SimpleNamespace(
        arguments=SimpleNamespace(pose=pose_cfg),
        _model=SimpleNamespace(config=SimpleNamespace(granted_freq_max_position_embedding=128)),
        processing_class=None,
    )
    image_id, pad_id = BaseTrainer._setup_pose(pose_stub)
    assert (image_id, pad_id) == (None, None)

    # 3. _prepare_generation_input restores padding_side="right"
    class _Proc:
        padding_side = "right"

        def apply_chat_template(self, messages, **kwargs):
            assert self.padding_side == "left"
            if kwargs.get("tokenize") is False:
                return "rendered"
            return {"input_ids": np.array([[1, 2, 3]], dtype=np.int32), "attention_mask": np.array([[1, 1, 1]], dtype=np.int32)}

    proc = _Proc()
    gen_stub = SimpleNamespace(
        arguments=SimpleNamespace(max_length=16, generation_max_prompt_length=16),
        _get_processing_class=lambda: proc,
        _batch_decode_tokens=lambda ids: ["decoded"],
    )
    res_str = BaseTrainer._prepare_generation_input(gen_stub, "hello")
    assert res_str is not None and proc.padding_side == "right"
    res_chat = BaseTrainer._prepare_generation_input(gen_stub, [{"role": "user", "content": "hi"}])
    assert res_chat is not None and proc.padding_side == "right"


# ---------------------------------------------------------------------------
# CORE-H14 (buckets.py) & CORE-L5 (buckets.py)
# ---------------------------------------------------------------------------


def test_core_h14_and_l5_buckets_recursive_overrides_and_validation():
    from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import Qwen3OmniMoeConfig

    cfg = Qwen3OmniMoeConfig()
    assert cfg.thinker_config.text_config.attn_mechanism == "vanilla"
    overridden = apply_config_overrides(cfg, {"attn_mechanism": "sdpa"}, label="vlm-bucket")
    assert overridden.attn_mechanism == "sdpa"
    assert overridden.thinker_config.text_config.attn_mechanism == "sdpa"
    assert overridden.talker_config.text_config.attn_mechanism == "sdpa"
    # Base config must remain untouched
    assert cfg.thinker_config.text_config.attn_mechanism == "vanilla"

    with pytest.raises(AttributeError):
        apply_config_overrides(cfg, {"nonexistent_field_xyz": 1}, label="vlm-bucket")

    with pytest.raises(ValueError):
        ModBucketRule(mod=0)
    with pytest.raises(ValueError):
        ModBucketRule(mod=2, on_bucket=-1)
    with pytest.raises(ValueError):
        TrainingBucket(name="", max_length=16)
    with pytest.raises(ValueError):
        TrainingBucket(name="b", max_length=0)
    with pytest.raises(ValueError):
        TrainingBucket(name="b", max_length=16, total_batch_size=0)


# ---------------------------------------------------------------------------
# CORE-M1, CORE-M2, CORE-M4, CORE-L1, CORE-L4
# ---------------------------------------------------------------------------


def test_core_m1_lr_and_bucket_throughput_metrics():
    captured_calc: list[dict] = []

    def _exec(state, batch):
        del batch
        return (
            SimpleNamespace(step=jnp.asarray(int(state.step) + 1)),
            LossMetrics(loss=jnp.asarray(1.0), learning_rate=jnp.asarray(0.5)),
            None,
        )

    host = SimpleNamespace(
        arguments=SimpleNamespace(
            dataloader_prefetch=False,
            max_length=128,
            learning_rate=1e-3,
            merge_lora_before_save=False,
            profiler_include_compile=False,
        ),
        data_collator=None,
        max_training_steps=1,
        _has_buckets=True,
        _buckets=[TrainingBucket(name="short", max_length=32, total_batch_size=4)],
        _bucket_loaders={0: [0, 1]},
        _bucket_iters={0: iter([0, 1])},
        _bucket_collators={0: lambda x: x},
        _bucket_rule=SimpleNamespace(select=lambda step: 0),
        _bucket_for_step=0,
        _get_next_batch=lambda it, ds: (next(it), it),
        _execute_train_step=_exec,
        _runtime_batch_summary=lambda b: b,
        _runtime_trace=lambda *a, **kw: None,
        _log_mesh_topology=lambda: None,
        _apply_pose_to_batch=lambda b, s: b,
        _apply_randomized_yarn_to_batch=lambda b, s: b,
        _apply_process_encoder_to_batch=lambda b, s: b,
        on_step_start=lambda state, step: state,
        on_step_end=lambda state, metrics, step: (state, metrics),
        apply_training_hooks=lambda metrics, loss_value=None: metrics,
        _metric_scalar_to_float=float,
        _maybe_start_profiler=lambda step: None,
        _maybe_stop_profiler=lambda step: None,
        _should_save_tpu_preemption_checkpoint=lambda step: False,
        _should_run_evaluation=lambda step: False,
        _profiler_should_block_until_ready=lambda: False,
        _save_checkpoint_for_step=lambda *a, **kw: None,
        log_metrics=lambda **kw: None,
        log_weight_distribution=lambda **kw: None,
        log_watchers=lambda **kw: None,
        maybe_generate=lambda **kw: None,
        maybe_benchmark=lambda **kw: None,
        train_tracker=SimpleNamespace(trace_compilation=contextlib.nullcontext),
        scheduler=lambda step: 0.9 if step == 0 else 0.1,
        _backward_flops_per_token=0.0,
        _extra_backward_flops_per_token=0.0,
        training_batch_size=8,
    )
    Trainer._train_epoch(
        host,
        state=SimpleNamespace(step=jnp.asarray(0)),
        train_dataset=[0, 1],
        train_iter=iter([0, 1]),
        metrics_tracker=SimpleNamespace(update=lambda **kw: (1.0, 1.0)),
        step_metrics=SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: (captured_calc.append(kw) or {})),
        pbar=SimpleNamespace(),
        epoch=0,
        epoch_start_step=0,
        epoch_end_step=1,
    )
    assert len(captured_calc) == 1
    assert captured_calc[0]["learning_rate"] == pytest.approx(0.5)
    assert captured_calc[0]["batch_size"] == 4
    assert captured_calc[0]["seq_length"] == 32


def test_core_m2_base_trainer_utilities_and_logging(monkeypatch):
    # calculate_number_total_flops honors is_training
    params = {"w": np.zeros((10,), dtype=np.float32)}
    assert BaseTrainer.calculate_number_total_flops(SimpleNamespace(), params, is_training=True) == 60
    assert BaseTrainer.calculate_number_total_flops(SimpleNamespace(), params, is_training=False) == 20

    # _pad_token_id & _eos_token_id use _get_processing_class() safely
    tok_stub = SimpleNamespace(
        _get_processing_class=lambda: SimpleNamespace(pad_token_id=7, eos_token_id=9),
        processing_class=None,
        model=SimpleNamespace(generation_config=None),
    )
    assert BaseTrainer._pad_token_id.func(tok_stub) == 7
    assert BaseTrainer._eos_token_id.func(tok_stub) == [9]
    none_stub = SimpleNamespace(
        _get_processing_class=lambda: None,
        processing_class=None,
        model=SimpleNamespace(generation_config=None),
        _eos_token_id=[],
    )
    assert BaseTrainer._eos_token_id.func(none_stub) == []
    with pytest.raises(ValueError, match="pad_token_id"):
        BaseTrainer._pad_token_id.func(none_stub)
    cfg_fallback_stub = SimpleNamespace(
        _get_processing_class=lambda: None,
        processing_class=None,
        config=SimpleNamespace(pad_token_id=11),
        _eos_token_id=[],
    )
    assert BaseTrainer._pad_token_id.func(cfg_fallback_stub) == 11

    # maybe_generate prints ALL preview records, not only the last record
    printed: list[str] = []
    monkeypatch.setattr("easydel.trainers.base_trainer.logger.info", lambda msg, *a: printed.append(msg % a if a else str(msg)))
    mg_stub = SimpleNamespace(
        arguments=SimpleNamespace(
            generation_interval=1,
            generation_preview_print=True,
            generation_log_to_wandb=False,
            generation_shard_inputs=False,
            generation_num_return_sequences=1,
            use_wandb=False,
            can_log_metrics=False,
        ),
        _pad_token_id=0,
        latest_generation_samples=None,
        preview_log_table=None,
        _collect_generation_prompts=lambda: ["p0", "p1"],
        _prepare_generation_input=lambda p: {
            "input_ids": np.array([[1, 2]], dtype=np.int32),
            "attention_mask": np.array([[1, 1]], dtype=np.int32),
            "prompt_text": p,
        },
        rollout=lambda **kw: SimpleNamespace(
            text=["o0", "o1"],
            generation_results=["o0", "o1"],
            reasoning=[None, None],
            tool_calls=[None, None],
            prompt_ids=np.array([[1, 2], [1, 2]], dtype=np.int32),
        ),
        _coerce_generation_texts=BaseTrainer._coerce_generation_texts,
        _coerce_optional_generation_texts=BaseTrainer._coerce_optional_generation_texts,
        _coerce_generation_metadata_list=BaseTrainer._coerce_generation_metadata_list,
    )
    BaseTrainer.maybe_generate(mg_stub, state=SimpleNamespace(), step=1)
    assert any("p0" in line for line in printed) and any("p1" in line for line in printed)
    assert any("o0" in line for line in printed) and any("o1" in line for line in printed)

    # _setup_initial_metrics logs billions under "(Billion)"
    logged: list[dict] = []
    init_stub = SimpleNamespace(
        count_model_parameters=lambda gs: 2_000_000_000,
        _train_dataset_num_examples=None,
        _eval_dataset_num_examples=None,
        arguments=SimpleNamespace(log_metrics=lambda m, **kw: logged.append(m)),
    )
    BaseTrainer._setup_initial_metrics(init_stub, SimpleNamespace(graphstate={}))
    assert logged[0]["Number of Model Parameters (Billion)"] == pytest.approx(2.0)

    # _purify_batch pads variable-length input_ids with pad_token_id
    purify_stub = SimpleNamespace(
        _tokenizer_int_attr=lambda name, default=None: 99 if name == "pad_token_id" else default,
        config=None,
    )
    out = BaseTrainer._purify_batch(purify_stub, [{"input_ids": [1, 2]}, {"input_ids": [3, 4, 5]}])
    np.testing.assert_array_equal(out["input_ids"], np.array([[1, 2, 99], [3, 4, 5]], dtype=np.int32))


def test_core_m8_tfds_fallback_dataloader_is_reiterable(monkeypatch):
    import sys

    fake_tf = SimpleNamespace(
        config=SimpleNamespace(set_visible_devices=lambda *a, **kw: None, get_visible_devices=lambda: []),
        data=SimpleNamespace(AUTOTUNE=1),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr("easydel.trainers.base_trainer.is_package_available", lambda pkg: pkg == "tensorflow")

    class _OneShotIterable:
        def __init__(self, rows):
            self._rows = list(rows)
            self._used = False

        def repeat(self, n):
            return self

        def prefetch(self, n):
            return self

        def as_numpy_iterator(self):
            assert not self._used, "One-shot iterator reused directly without factory"
            self._used = True
            return iter(self._rows)

    class _FakeHFDataset:
        def __init__(self, rows):
            self._rows = rows
            self.calls = 0

        def __len__(self):
            return len(self._rows)

        def to_tf_dataset(self, **kwargs):
            self.calls += 1
            return _OneShotIterable(self._rows)

    train_ds = _FakeHFDataset([{"input_ids": np.array([1, 2])}, {"input_ids": np.array([3, 4])}])
    stub = SimpleNamespace(
        arguments=SimpleNamespace(
            use_grain=False,
            do_eval=False,
            max_length=4,
            truncation_mode="keep_end",
            shuffle_train_dataset=False,
            dataset_num_proc=1,
            dataloader_num_workers=0,
            total_batch_size=2,
            eval_batch_size=2,
            num_train_epochs=1,
            max_training_steps=None,
            max_evaluation_steps=None,
            auto_shard_states=False,
            grain_shard_index=0,
            grain_shard_count=1,
            is_process_zero=True,
        ),
        dataset_train=train_ds,
        dataset_eval=None,
        _train_source=None,
        _eval_source=None,
        training_batch_size=2,
        evaluation_batch_size=2,
        data_collator=None,
        create_tfds_collect_function=lambda **kw: (lambda x: x),
        _resolve_step_count=lambda ds, source, is_train, drop_remainder: SimpleNamespace(steps=1),
        _set_dataset_size_metadata=lambda **kw: None,
        _to_sharded_source=lambda ds: None,
    )
    out = BaseTrainer._configure_tfds_dataloader(stub)
    first_pass = list(out.dataloader_train)
    second_pass = list(out.dataloader_train)
    assert len(first_pass) == 2
    assert len(second_pass) == 2
    assert train_ds.calls >= 2


def test_core_m4_protocol_signatures_match_implementation():
    eval_sig = inspect.signature(BaseTrainerProtocol._run_evaluation)
    assert "start_time" not in eval_sig.parameters
    readme_sig = inspect.signature(BaseTrainerProtocol._save_readme)
    assert "save_directory" in readme_sig.parameters
    assert isinstance(inspect.getattr_static(BaseTrainerProtocol, "finish"), staticmethod)


def test_core_l1_execution_accumulation_streams_microbatches_lazily(monkeypatch):
    placed_calls: list[int] = []
    monkeypatch.setattr(
        "easydel.trainers.execution_accumulation.place_batch",
        lambda part, spec, mesh: (placed_calls.append(int(part["x"][0])) or part),
    )
    ea = object.__new__(ExecutionAccumulation)
    ea.num_steps = 4
    ea._batch_spec = PartitionSpec("dp")
    ea._mesh = object()

    seq = ea.microbatches({"x": np.arange(8, dtype=np.int32)})
    assert len(seq) == 4
    assert placed_calls == []
    first = seq[0]
    assert placed_calls == [0]
    np.testing.assert_array_equal(first["x"], np.array([0, 1], dtype=np.int32))


def test_core_l4_disable_state_dropout_and_trainers_exports():
    import easydel.trainers as trainers_pkg

    for name in ("EmbeddingPreprocessTransform", "PoSEConfig", "RandomizedYaRNConfig", "ProcessEncoderConfig"):
        assert name in trainers_pkg.__all__
        assert getattr(trainers_pkg, name) is not None

    # Real spx.Module TypeError in spx.export must not be swallowed
    class _TinyMod(spx.Module):
        def __init__(self):
            self.x = spx.Parameter(jnp.ones((2,)))

    mod = _TinyMod()
    st = SimpleNamespace(model=mod, replace=lambda **kw: SimpleNamespace(**kw))
    orig_export = spx.export
    spx.export = lambda m: (_ for _ in ()).throw(TypeError("export bug"))
    try:
        with pytest.raises(TypeError, match="export bug"):
            disable_state_dropout(st)
    finally:
        spx.export = orig_export


# ---------------------------------------------------------------------------
# RL-H16 & RL-H17 (base_trainer.py parts)
# ---------------------------------------------------------------------------


def test_rl_h16_and_rl_h17_base_trainer_rollout_and_all_gather(monkeypatch):
    # RL-H16: Real rollout() -> _generate_via_esurge preserves config_overrides
    # (including do_sample=True when args.generation_do_sample=False and non-15-key
    # fields like bad_words) while applying generate_kwargs (seed=42, min_p=0.1).
    from easydel.trainers._shared import esurge_generation

    captured_engine_sampling: list[object] = []

    class _FakeEngine:
        def generate(self, *, prompts, sampling_params, use_tqdm, prompt_token_ids=None):
            del prompts, use_tqdm, prompt_token_ids
            captured_engine_sampling.append(sampling_params)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 6], text="ok", logprobs=None)])]

    fake_engine = _FakeEngine()
    fake_state = SimpleNamespace(
        model=SimpleNamespace(
            mesh=Mesh(np.array(jax.devices()[:1]), ("dp",)),
            get_esurge=lambda **kw: fake_engine,
        )
    )
    e2e_stub = SimpleNamespace(
        arguments=SimpleNamespace(
            use_esurge_generation=True,
            generation_shard_inputs=False,
            generation_temperature=0.7,
            generation_top_p=0.95,
            generation_top_k=64,
            generation_do_sample=False,
            generation_max_new_tokens=16,
            max_completion_length=16,
            max_prompt_length=16,
            max_length=32,
            generation_extra_kwargs=None,
            esurge_use_tqdm=False,
            esurge_max_num_seqs=None,
            esurge_max_num_batched_tokens=None,
            esurge_hbm_utilization=None,
            esurge_page_size=None,
            esurge_enable_prefix_caching=None,
            esurge_data_parallelism_axis=None,
            esurge_runner_verbose=False,
            esurge_silent_mode=True,
            esurge_use_aot_forward=None,
            esurge_min_input_pad=None,
            esurge_max_model_len=None,
            esurge_reserve_tokens=None,
            esurge_auto_truncate_prompt=None,
            esurge_auto_cap_new_tokens=None,
            esurge_strict_context=None,
            esurge_truncate_mode=None,
            esurge_prefer_preserve_prompt=None,
            esurge_decode_truncated_prompt=None,
            esurge_destroy_pages_on_pause=None,
            esurge_disable_compile=None,
            esurge_overlap_execution=None,
            esurge_sampler_metrics=None,
            esurge_tool_parser=None,
            esurge_reasoning_parser=None,
            esurge_enable_window_aware_runtime_cap=None,
            esurge_max_num_seq_buckets=None,
            esurge_extra_kwargs=None,
            esurge_cache_capacity_margin=None,
        ),
        model_state=fake_state,
        _pad_token_id=0,
        _get_processing_class=lambda: None,
        _normalize_supported_generation_model_kwargs=lambda st, kw: ({}, []),
        _decode_prompt_batch=lambda *a, **kw: ["p"],
        maybe_release_generation_runtime=lambda res, **kw: res,
        _all_gather=lambda arr: arr,
    )
    e2e_stub._generate_via_esurge = BaseTrainer._generate_via_esurge.__get__(e2e_stub, type(e2e_stub))
    monkeypatch.setattr(
        esurge_generation,
        "outputs_to_arrays",
        lambda outputs, **kw: SimpleNamespace(
            prompt_ids=jnp.ones((1, 4), dtype=jnp.int32),
            prompt_mask=jnp.ones((1, 4), dtype=jnp.int32),
            sequences=jnp.ones((1, 6), dtype=jnp.int32),
            completion_ids=jnp.ones((1, 2), dtype=jnp.int32),
            completion_mask=jnp.ones((1, 2), dtype=jnp.int32),
            output_records=["ok"],
            reasoning_records=[None],
            tool_call_records=[None],
            raw_output_records=["ok"],
            finish_reason_records=["stop"],
            completion_prompts=["p"],
            sampling_per_token_logps=None,
        ),
    )
    BaseTrainer.rollout(
        e2e_stub,
        input_ids=jnp.ones((1, 4), dtype=jnp.int32),
        attention_mask=jnp.ones((1, 4), dtype=jnp.int32),
        config_overrides={"do_sample": True, "temperature": 0.8, "bad_words": ["SECRET"]},
        seed=42,
        min_p=0.1,
    )
    assert len(captured_engine_sampling) == 1
    sp = captured_engine_sampling[0]
    assert sp.temperature == pytest.approx(0.8)
    assert sp.bad_words == ["SECRET"]
    assert sp.seed == 42
    assert sp.min_p == pytest.approx(0.1)

    # RL-H17: _all_gather handles host numpy, local jax.Array, and multi-host callback path
    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    stub = SimpleNamespace(model=SimpleNamespace(mesh=mesh))

    arr_np = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out_np = BaseTrainer._all_gather(stub, arr_np)
    assert isinstance(out_np, jax.Array)
    np.testing.assert_array_equal(np.asarray(out_np), arr_np)

    # Simulate multi-host non-addressable target sharding -> must route through make_array_from_callback
    callback_calls: list[tuple] = []

    class _FakeMultiHostTarget:
        device_set = frozenset({object(), object()})
        is_fully_addressable = False

    monkeypatch.setattr("easydel.trainers.base_trainer.replicated_named_sharding", lambda m: _FakeMultiHostTarget())
    monkeypatch.setattr(
        jax,
        "make_array_from_callback",
        lambda shape, sharding, cb: (callback_calls.append((shape, cb((slice(None), slice(None))))) or jnp.asarray(cb((slice(None), slice(None))))),
    )
    local_arr = jnp.array([[5, 6]], dtype=jnp.int32)
    gathered = BaseTrainer._all_gather(stub, local_arr)
    assert len(callback_calls) == 1
    np.testing.assert_array_equal(np.asarray(gathered), np.array([[5, 6]], dtype=np.int32))

# =============================================================================
# Domain B: Training Configurations, Metrics & Collators/Utils
# =============================================================================

def _load_grain_batch(examples, collator, batch_size: int = 2):
    loader = pygrain.DataLoader(
        data_source=pygrain.MapDataset.source(examples),
        sampler=pygrain.IndexSampler(
            num_records=len(examples),
            shard_options=pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
            num_epochs=1,
            shuffle=False,
            seed=0,
        ),
        operations=[
            ToNumpy(),
            CollateMapTransform(collate_fn=collator),
            pygrain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=0,
        worker_buffer_size=1,
        read_options=pygrain.ReadOptions(num_threads=1, prefetch_buffer_size=2),
    )
    return next(iter(loader))


# ---------------------------------------------------------------------------
# CORE-C6: TensorBoard histogram downsampling & single flush per call
# ---------------------------------------------------------------------------


def test_core_c6_tensorboard_downsamples_large_histogram_and_flushes_once(tmp_path):
    class RecordingSummaryWriter(SummaryWriter):
        def __init__(self, log_dir):
            super().__init__(log_dir=log_dir, auto_flush=False)
            self.flush_calls = 0
            self.recorded_histograms: dict[str, np.ndarray] = {}
            self.recorded_scalars: dict[str, float] = {}

        def add_scalar(self, tag, value, step):
            self.recorded_scalars[tag] = value
            super().add_scalar(tag, value, step)

        def add_histogram(self, tag, values, step):
            self.recorded_histograms[tag] = np.asarray(values)
            super().add_histogram(tag, values, step)

        def flush(self):
            self.flush_calls += 1
            super().flush()

    writer = RecordingSummaryWriter(tmp_path / "tb")
    try:
        args = TrainingArguments(
            model_name="tb-test",
            save_directory=str(tmp_path),
            use_wandb=False,
            report_metrics=True,
        )
        args.__dict__["_tensorboard_writer"] = writer

        # Simulate a 600M-parameter LM head histogram with an outlier tail bin of count 1.
        bin_counts = np.zeros(64, dtype=np.int64)
        bin_counts[30] = 300_000_000
        bin_counts[31] = 299_999_999
        bin_counts[63] = 1
        bin_edges = np.linspace(-2.0, 2.0, 65, dtype=np.float32)

        small_counts = np.array([2, 3, 5], dtype=np.int64)
        small_edges = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        args._log_to_tensorboard(
            {
                "loss": 1.25,
                "lr": 1e-4,
                "huge_weight": (bin_counts, bin_edges),
                "small_weight": (small_counts, small_edges),
            },
            step=500,
        )

        assert writer.flush_calls == 1
        assert writer.recorded_scalars["loss"] == 1.25
        assert writer.recorded_scalars["lr"] == 1e-4
        huge_samples = writer.recorded_histograms["huge_weight"]
        assert 0 < huge_samples.size <= 4096
        # Tail bin center should be preserved despite tiny relative frequency.
        expected_tail_center = 0.5 * (bin_edges[63] + bin_edges[64])
        assert np.any(np.isclose(huge_samples, expected_tail_center))
        # Small histograms keep exact counts.
        small_samples = writer.recorded_histograms["small_weight"]
        assert small_samples.size == 10
    finally:
        args.close_tensorboard()


# ---------------------------------------------------------------------------
# CORE-C7: O(N) MetricsHistogram without (64, N) broadcast allocation
# ---------------------------------------------------------------------------


def test_core_c7_metrics_histogram_avoids_64xN_broadcast_and_matches_numpy():
    n = 1024
    jaxpr = jax.make_jaxpr(MetricsHistogram._create_histogram_bin_edges)(jnp.zeros((n,), dtype=jnp.float32))
    for eqn in jaxpr.jaxpr.eqns:
        for outvar in eqn.outvars:
            shape = getattr(outvar.aval, "shape", ())
            assert shape != (64, n) and shape != (n, 64), f"Found O(64*N) broadcast shape {shape} in {eqn}"

    rng = np.random.default_rng(42)
    for arr_np in [
        rng.normal(size=512).astype(np.float32),
        np.full(128, 3.5, dtype=np.float32),
        np.array([7.0], dtype=np.float32),
        np.linspace(-10.0, 10.0, 257, dtype=np.float32),
    ]:
        hist = MetricsHistogram.from_array(jnp.asarray(arr_np))
        ref_counts, ref_edges = np.histogram(arr_np, bins=64)
        np.testing.assert_array_equal(np.asarray(hist.bin_counts), ref_counts)
        np.testing.assert_allclose(np.asarray(hist.bin_edges), ref_edges, rtol=1e-5, atol=1e-5)
        assert int(hist.bin_counts.sum()) == arr_np.size
        assert hist.bin_counts.dtype == jnp.int32

    stats = compute_weight_stats({"layer": {"w": jnp.arange(64, dtype=jnp.bfloat16)}}, r".*")
    assert int(stats["layer/w/histogram"].bin_counts.sum()) == 64


# ---------------------------------------------------------------------------
# CORE-H10: TrainingArguments dtype serialization, runtime fields & optimizer
# ---------------------------------------------------------------------------


def test_core_h10_dtype_roundtrip_and_runtime_only_fields_excluded(tmp_path):
    class _DummyPruning:
        def wrap_optax(self, tx):
            return tx

    class _DummySchedule:
        pass

    args = TrainingArguments(
        model_name="h10-model",
        save_directory=str(tmp_path),
        use_wandb=False,
        report_metrics=False,
        tx_mu_dtype=jnp.bfloat16,
        watchers=[LogWatcher("w", lambda x: {"norm": 1.0})],
        custom_scheduler=lambda step: 1e-4,
        pruning_module=_DummyPruning(),
        tensor_straight_through=lambda x: x,
        straight_through_emulator=lambda s: s,
        mpmd_scheduler=_DummySchedule(),
    )

    as_dict = args.to_dict()
    assert as_dict["tx_mu_dtype"] == "bfloat16"
    for field_name in (
        "buckets",
        "bucket_datasets",
        "watchers",
        "custom_scheduler",
        "pruning_module",
        "tensor_straight_through",
        "straight_through_emulator",
        "mpmd_scheduler",
    ):
        assert field_name not in as_dict

    json_path = tmp_path / "args.json"
    args.save_arguments(json_path)
    loaded = TrainingArguments.load_arguments(json_path)

    assert jnp.dtype(loaded.tx_mu_dtype) == jnp.dtype(jnp.bfloat16)
    assert loaded.watchers == []
    assert loaded.custom_scheduler is None
    assert loaded.pruning_module is None
    assert loaded.tensor_straight_through is None
    assert loaded.straight_through_emulator is None
    assert loaded.mpmd_scheduler is None
    assert loaded.to_dict() == as_dict

    # Must construct a working optimizer using the deserialized tx_mu_dtype.
    tx, sched = loaded.get_optimizer_and_scheduler()
    params = {"w": jnp.ones((4,), dtype=jnp.float32)}
    opt_state = tx.init(params)
    updates, _ = tx.update({"w": jnp.ones((4,), dtype=jnp.float32)}, opt_state, params)
    assert updates["w"].shape == (4,)
    assert float(sched(0)) == pytest.approx(loaded.learning_rate)

    # Legacy stringified dtype representation also deserializes cleanly.
    legacy = TrainingArguments.from_dict({"tx_mu_dtype": "<class 'ml_dtypes.bfloat16'>", "use_wandb": False})
    assert jnp.dtype(legacy.tx_mu_dtype) == jnp.dtype(jnp.bfloat16)


def test_core_h10_get_optimizer_and_scheduler_defaults_steps_zero_and_live_fields():
    # Default warmup_steps=0, max_training_steps=None, scheduler=NONE must not crash.
    args = TrainingArguments(use_wandb=False, report_metrics=False)
    tx, sched = args.get_optimizer_and_scheduler()
    assert tx is not None
    assert float(sched(0)) == pytest.approx(args.learning_rate)

    # Explicit steps=0 must be honored (not replaced with max_training_steps).
    args_with_max = TrainingArguments(
        use_wandb=False,
        report_metrics=False,
        max_training_steps=100,
        scheduler=EasyDeLSchedulers.NONE,
        warmup_steps=0,
    )
    tx0, sched0 = args_with_max.get_optimizer_and_scheduler(steps=0)
    assert tx0 is not None
    assert float(sched0(0)) == pytest.approx(args_with_max.learning_rate)

    # Mutating live fields after __post_init__ must be picked up by get_optimizer_and_scheduler().
    args.learning_rate = 2.5e-3
    args.scheduler = EasyDeLSchedulers.COSINE
    args.max_training_steps = 40
    args.warmup_steps = 4
    args.tx_mu_dtype = jnp.bfloat16
    _, live_sched = args.get_optimizer_and_scheduler()
    assert float(live_sched(0)) == pytest.approx(0.0, abs=1e-6)
    assert float(live_sched(4)) == pytest.approx(2.5e-3, rel=1e-3)
    assert float(live_sched(40)) < float(live_sched(4))


# ---------------------------------------------------------------------------
# CORE-H11: np.ndarray in get_safe_arr/logging, key stripping, variance/std
# ---------------------------------------------------------------------------


def test_core_h11_get_safe_arr_and_numpy_histograms_and_weight_dist_keys(monkeypatch, tmp_path):
    assert get_safe_arr(np.asarray(3.25)) == 3.25
    assert isinstance(get_safe_arr(np.asarray(3.25)), float)
    arr_2d = np.asarray([1.0, 2.0])
    assert get_safe_arr(arr_2d) is arr_2d

    class _FakeHistogram:
        def __init__(self, value=None, np_histogram=None):
            self.value = value
            self.np_histogram = np_histogram

    wandb_logged = []
    tb_scalars = {}
    tb_histograms = {}

    class _FakeTB:
        def add_scalar(self, k, v, step):
            tb_scalars[k] = v

        def add_histogram(self, k, v, step):
            tb_histograms[k] = np.asarray(v)

        def flush(self):
            pass

    monkeypatch.setattr(
        tc_mod,
        "wandb",
        SimpleNamespace(Histogram=_FakeHistogram, log=lambda m, step: wandb_logged.append((m, step))),
    )

    def _watch_fn(arr):
        counts, edges = jnp.histogram(arr, bins=4)
        return {
            "scalar": jnp.mean(arr),
            "dist": (counts, edges),
        }

    args = TrainingArguments(
        model_name="h11-test",
        save_directory=str(tmp_path),
        use_wandb=True,
        report_metrics=True,
        weight_distribution_log_steps=1,
        watchers=[LogWatcher("probe", _watch_fn, interval=1, pattern=r".*weight$")],
    )
    monkeypatch.setattr(args, "get_tensorboard", lambda: _FakeTB())

    state = SimpleNamespace(graphstate={"layer": {"weight": jnp.arange(8, dtype=jnp.float32)}})
    args.log_watchers(state, step=1)

    assert len(wandb_logged) == 1
    w_metrics, _ = wandb_logged[0]
    # 0-d numpy scalar became Python float, not Histogram.
    assert isinstance(w_metrics["probe/layer/weight/scalar"], float)
    assert w_metrics["probe/layer/weight/scalar"] == pytest.approx(3.5)
    # (np.ndarray, np.ndarray) tuple was recognized as np_histogram by both wandb and TB.
    assert isinstance(w_metrics["probe/layer/weight/dist"], _FakeHistogram)
    assert w_metrics["probe/layer/weight/dist"].np_histogram is not None
    assert "probe/layer/weight/scalar" in tb_scalars
    assert "probe/layer/weight/dist" in tb_histograms
    assert tb_histograms["probe/layer/weight/dist"].size == 8

    # Weight distribution strips "/histogram" before replacing "/" with ".".
    wandb_logged.clear()
    args.log_weight_distribution(state, step=1)
    assert len(wandb_logged) == 1
    wd_metrics, _ = wandb_logged[0]
    assert "weights-information/layer.weight/mean" in wd_metrics
    assert "weights-information/layer.weight/std" in wd_metrics
    assert "weights-information/layer.weight.histogram/mean" not in wd_metrics


def test_core_h11_metrics_histogram_variance_clamped_and_host_std_uses_numpy(monkeypatch):
    # Constant tensor whose float32 sum_squares/N - mean^2 can dip slightly negative.
    const = jnp.full((10000,), 1.2345679, dtype=jnp.float32)
    hist = MetricsHistogram.from_array(const)
    assert float(hist.variance) >= 0.0
    assert np.isfinite(float(np.asarray(hist.std).item()))

    # Explicitly construct a host MetricsHistogram with negative cancellation before clamp.
    host_hist = MetricsHistogram(
        bin_counts=np.zeros((64,), dtype=np.int32),
        bin_edges=np.linspace(0.0, 1.0, 65, dtype=np.float32),
        size=10,
        min=np.float32(1.0),
        max=np.float32(1.0),
        sum=np.float32(10.0),
        sum_squares=np.float32(9.9999),
    )

    def _forbid_jnp_sqrt(_):
        raise AssertionError("Host MetricsHistogram.std must use np.sqrt, not jnp.sqrt")

    monkeypatch.setattr("easydel.trainers.metrics.jnp.sqrt", _forbid_jnp_sqrt)
    assert float(host_hist.variance) == 0.0
    assert float(host_hist.std.item()) == 0.0
    assert isinstance(host_hist.std, np.ndarray)

    # Restore jnp.sqrt and verify traced JAX execution still works inside jit.
    monkeypatch.undo()
    traced_std = jax.jit(lambda x: MetricsHistogram.from_array(x).std)(jnp.arange(8, dtype=jnp.float32))
    assert float(traced_std.item()) == pytest.approx(float(np.std(np.arange(8, dtype=jnp.float32))), rel=1e-5)


# ---------------------------------------------------------------------------
# CORE-H12: StepMetrics detailed grad norms (no double prefix, batched D2H, float leaves)
# ---------------------------------------------------------------------------


def test_core_h12_step_metrics_detailed_no_double_prefix_single_d2h_and_float_leaves(monkeypatch):
    import easydel.trainers.metrics as metrics_mod

    device_get_calls = 0
    real_device_get = jax.device_get

    def _counting_device_get(tree):
        nonlocal device_get_calls
        device_get_calls += 1
        return real_device_get(tree)

    monkeypatch.setattr(metrics_mod.jax, "device_get", _counting_device_get)

    step_metrics = StepMetrics(arguments=SimpleNamespace(performance_mode=False, log_grad_norms=True))
    grad_norms = {f"layer_{i}": {"weight": jnp.asarray(float(i), dtype=jnp.float32)} for i in range(40)}
    metrics = LossMetrics(
        loss=1.0,
        accuracy=0.5,
        execution_time=1.0,
        max_grad_norm=jnp.asarray(39.0, dtype=jnp.float32),
        mean_grad_norm=jnp.asarray(19.5, dtype=jnp.float32),
        grad_norms=grad_norms,
    )

    res = step_metrics.calculate(
        metrics=metrics,
        current_step=1,
        epoch=0,
        epoch_progress=None,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=1e-4,
        mode="train",
    )

    assert "train/max_grad_norm" in res
    assert "train/mean_grad_norm" in res
    assert "train/train/max_grad_norm" not in res
    assert "train/train/mean_grad_norm" not in res
    assert res["train/max_grad_norm"] == pytest.approx(39.0)
    assert res["train/mean_grad_norm"] == pytest.approx(19.5)
    assert res["train/grad_norm/layer_5.weight"] == pytest.approx(5.0)
    # All 42 grad norm leaves fetched in a single batched jax.device_get call!
    assert device_get_calls == 1

    # Python float leaves must not crash with AttributeError (.tolist).
    float_metrics = LossMetrics(
        loss=1.0,
        execution_time=1.0,
        max_grad_norm=2.5,
        mean_grad_norm=1.25,
        grad_norms=spx.State({"parameters": {"proj.weight": 1.25}}),
    )
    res_float = step_metrics.calculate(
        metrics=float_metrics,
        current_step=2,
        epoch=0,
        epoch_progress=None,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=1e-4,
        mode="train",
    )
    assert res_float["train/max_grad_norm"] == 2.5
    assert res_float["train/mean_grad_norm"] == 1.25
    assert res_float["train/grad_norm/parameters/proj.weight"] == 1.25


# ---------------------------------------------------------------------------
# CORE-H13: utils.py collators & HFDataSource
# ---------------------------------------------------------------------------


class _PublicPadTokenizer:
    """Tokenizer exposing only public pad_token (no private _pad_token)."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


def test_core_h13_completion_only_collator_attention_mask_linear_scan_empty_row_and_public_pad():
    tok = _PublicPadTokenizer()
    tok.padding_side = "left"
    collator = DataCollatorForCompletionOnlyLM(tok, response_template=[7, 8], instruction_template=[6])

    # 1. Empty row [] with left-padding must not crash on -0:, and attention_mask must be emitted.
    with pytest.warns(UserWarning):
        batch_empty = collator([[], [6, 1, 7, 8, 3]])
    assert "attention_mask" in batch_empty
    np.testing.assert_array_equal(np.asarray(batch_empty["input_ids"]), [[0, 0, 0, 0, 0], [6, 1, 7, 8, 3]])
    np.testing.assert_array_equal(np.asarray(batch_empty["attention_mask"]), [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(
        np.asarray(batch_empty["labels"]),
        [[-100, -100, -100, -100, -100], [-100, -100, -100, -100, 3]],
    )

    # 2. Quoted instruction marker inside user prompt & quoted response marker inside assistant turn.
    tok.padding_side = "right"
    quoted_inst = [6, 1, 6, 2, 7, 8, 3, 6, 4, 7, 8, 5]
    quoted_resp = [6, 1, 7, 8, 3, 7, 8, 4, 6, 2, 7, 8, 5]
    batch = collator([quoted_inst, quoted_resp])
    assert "attention_mask" in batch
    np.testing.assert_array_equal(
        np.asarray(batch["attention_mask"]),
        [[1] * 12 + [0], [1] * 13],
    )
    np.testing.assert_array_equal(
        np.asarray(batch["labels"][0]),
        [-100, -100, -100, -100, -100, -100, 3, -100, -100, -100, -100, 5, -100],
    )
    np.testing.assert_array_equal(
        np.asarray(batch["labels"][1]),
        [-100, -100, -100, -100, 3, 7, 8, 4, -100, -100, -100, -100, 5],
    )


def test_core_h13_preference_collators_explicit_prefix_and_vision_tensors():
    tfds = DataCollatorForPreferenceTFDS(max_prompt_length=4, max_completion_length=3, pad_token_id=0)
    grain = DataCollatorForPreferenceGrain(max_prompt_length=4, max_completion_length=3, pad_token_id=0)

    # Completion-only sequence longer than max_completion_length (NO labels, does NOT start with prompt):
    # must NOT strip prompt_length tokens from the front!
    long_comp_feature = {
        "prompt_input_ids": [11, 12],
        "prompt_attention_mask": [1, 1],
        "chosen_input_ids": [21, 22, 23, 24, 25],
        "chosen_attention_mask": [1, 1, 1, 1, 1],
        "rejected_input_ids": [31, 32, 33, 34],
        "rejected_attention_mask": [1, 1, 1, 1],
    }
    b_tfds = tfds([long_comp_feature])
    b_grain = grain(long_comp_feature)
    assert b_tfds["chosen_input_ids"].tolist() == [[21, 22, 23]]
    assert b_grain["chosen_input_ids"].tolist() == [21, 22, 23]

    # Short combined prompt+completion sequence (len <= max_completion_length, NO labels):
    # MUST explicitly strip matching prompt prefix!
    tfds_wide = DataCollatorForPreferenceTFDS(max_prompt_length=4, max_completion_length=6, pad_token_id=0)
    grain_wide = DataCollatorForPreferenceGrain(max_prompt_length=4, max_completion_length=6, pad_token_id=0)
    short_combined = {
        "prompt_input_ids": [11, 12],
        "prompt_attention_mask": [1, 1],
        "chosen_input_ids": [11, 12, 21],
        "chosen_attention_mask": [1, 1, 1],
        "rejected_input_ids": [11, 12, 31, 32],
        "rejected_attention_mask": [1, 1, 1, 1],
    }
    assert tfds_wide([short_combined])["chosen_input_ids"].tolist() == [[21, 0, 0, 0, 0, 0]]
    assert grain_wide(short_combined)["chosen_input_ids"].tolist() == [21, 0, 0, 0, 0, 0]

    # Vision tensors must NOT be truncated or padded to max_prompt_length (4).
    vis_a = {
        **short_combined,
        "pixel_values": np.ones((3, 12, 16), dtype=np.float32),
        "pixel_attention_mask": np.ones((12, 16), dtype=np.int32),
    }
    vis_b = {
        **short_combined,
        "pixel_values": np.ones((3, 12, 20), dtype=np.float32) * 2.0,
        "pixel_attention_mask": np.ones((12, 20), dtype=np.int32),
    }
    vis_tfds = tfds_wide([vis_a, vis_b])
    assert vis_tfds["pixel_values"].shape == (2, 3, 12, 20)
    assert vis_tfds["pixel_attention_mask"].shape == (2, 12, 20)
    vis_grain = grain_wide(vis_a)
    assert vis_grain["pixel_values"].shape == (3, 12, 16)
    assert vis_grain["pixel_attention_mask"].shape == (12, 16)


def test_core_h13_bco_grain_pads_embedding_fields_for_pygrain_batch():
    examples = [
        {
            "prompt_input_ids": [1, 2],
            "prompt_attention_mask": [1, 1],
            "completion_input_ids": [1, 2, 7],
            "completion_attention_mask": [1, 1, 1],
            "completion_labels": [-100, -100, 7],
            "label": True,
            "embedding_input_ids": [10, 11],
            "embedding_attention_mask": [1, 1],
        },
        {
            "prompt_input_ids": [3, 4, 5],
            "prompt_attention_mask": [1, 1, 1],
            "completion_input_ids": [3, 4, 5, 8],
            "completion_attention_mask": [1, 1, 1, 1],
            "completion_labels": [-100, -100, -100, 8],
            "label": False,
            "embedding_input_ids": [20, 21, 22, 23],
            "embedding_attention_mask": [1, 1, 1, 1],
        },
    ]
    batch = _load_grain_batch(
        examples,
        BCODataCollatorGrain(
            max_prompt_length=5,
            max_completion_length=4,
            pad_token_id=0,
            label_pad_token_id=-100,
            is_encoder_decoder=False,
        ),
    )
    assert batch["embedding_input_ids"].shape == (2, 5)
    assert batch["embedding_attention_mask"].shape == (2, 5)
    np.testing.assert_array_equal(batch["embedding_input_ids"], [[10, 11, 0, 0, 0], [20, 21, 22, 23, 0]])
    np.testing.assert_array_equal(batch["embedding_attention_mask"], [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])


def test_core_h13_dpo_collators_enc_dec_input_ids_prepadded_false_and_numeric_dtypes():
    # 1. Encoder-decoder must pad chosen_input_ids / rejected_input_ids with pad_token_id, NOT -100.
    enc_dec_tfds = DPODataCollatorWithPaddingTFDS(
        max_prompt_length=3, max_completion_length=4, pad_token_id=0, is_encoder_decoder=True, prepadded=False
    )
    enc_dec_grain = DPODataCollatorWithPaddingGrain(
        max_prompt_length=3, max_completion_length=4, pad_token_id=0, is_encoder_decoder=True, prepadded=False
    )
    enc_row = {
        "prompt_input_ids": [1, 2],
        "chosen_input_ids": [5, 6],
        "rejected_input_ids": [7],
        "chosen_labels": [5, 6],
    }
    out_ed_tfds = enc_dec_tfds([enc_row])
    out_ed_grain = enc_dec_grain(enc_row)
    assert out_ed_tfds["chosen_input_ids"].tolist() == [[0, 0, 5, 6]]
    assert out_ed_tfds["rejected_input_ids"].tolist() == [[0, 0, 0, 7]]
    assert out_ed_tfds["chosen_labels"].tolist() == [[-100, -100, 5, 6]]
    assert out_ed_grain["chosen_input_ids"].tolist() == [0, 0, 5, 6]
    assert out_ed_grain["rejected_input_ids"].tolist() == [0, 0, 0, 7]
    assert out_ed_grain["chosen_labels"].tolist() == [-100, -100, 5, 6]

    # 2. Decoder-only prepadded=False with full prompt+completion must use combined max_length (3+4=7).
    dec_tfds = DPODataCollatorWithPaddingTFDS(
        max_prompt_length=3,
        max_completion_length=4,
        pad_token_id=0,
        is_encoder_decoder=False,
        prepadded=False,
        output_arrays_only=True,
    )
    dec_grain = DPODataCollatorWithPaddingGrain(
        max_prompt_length=3,
        max_completion_length=4,
        pad_token_id=0,
        is_encoder_decoder=False,
        prepadded=False,
        output_arrays_only=True,
    )
    full_row = {
        "prompt_input_ids": [1, 2, 3],
        "prompt_attention_mask": [1, 1, 1],
        "chosen_input_ids": [1, 2, 3, 10, 11, 12],
        "chosen_attention_mask": [1, 1, 1, 1, 1, 1],
        "chosen_labels": [-100, -100, -100, 10, 11, 12],
        "rejected_input_ids": [1, 2, 3, 20, 21],
        "rejected_attention_mask": [1, 1, 1, 1, 1],
        "rejected_labels": [-100, -100, -100, 20, 21],
        "bf16_logps": np.asarray(-1.5, dtype=jnp.bfloat16),
        "sample_id": np.asarray(42, dtype=np.int64),
        "is_valid": np.asarray(True, dtype=np.bool_),
        "raw_text": "drop me",
    }
    out_dec_tfds = dec_tfds([full_row])
    out_dec_grain = dec_grain(full_row)
    assert out_dec_tfds["chosen_input_ids"].shape == (1, 7)
    assert out_dec_tfds["chosen_input_ids"].tolist() == [[1, 2, 3, 10, 11, 12, 0]]
    assert out_dec_grain["chosen_input_ids"].shape == (7,)
    assert out_dec_grain["chosen_input_ids"].tolist() == [1, 2, 3, 10, 11, 12, 0]

    # 3. output_arrays_only=True preserves bfloat16, int64, bool_ and drops strings.
    for out in (out_dec_tfds, out_dec_grain):
        assert "bf16_logps" in out
        assert "sample_id" in out
        assert "is_valid" in out
        assert "raw_text" not in out


def test_core_h13_hf_data_source_thread_safe_and_distinct_non_numbered_threads(monkeypatch):
    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    ds = Dataset.from_dict({"x": list(range(20))}).to_iterable_dataset(num_shards=4)
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    source = HFDataSource(ds, options, num_threads=2)

    barrier = threading.Barrier(2)
    results: dict[str, list[int]] = {"alpha": [], "beta": []}
    errors: list[Exception] = []

    def _worker(label: str):
        try:
            barrier.wait()
            while True:
                try:
                    results[label].append(source[0]["x"])
                except IndexError:
                    break
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_worker, args=("alpha",), name="reader-alpha")
    t2 = threading.Thread(target=_worker, args=("beta",), name="reader-beta")
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors
    assert results["alpha"] and results["beta"]
    assert set(results["alpha"]).isdisjoint(set(results["beta"]))
    assert sorted(results["alpha"] + results["beta"]) == list(range(20))


# ---------------------------------------------------------------------------
# CORE-M3: TrainingArguments loading, aliases, unions, validation, TB & WandB
# ---------------------------------------------------------------------------


def test_core_m3_load_from_json_custom_and_unknown_class():
    with pytest.raises(ValueError, match="trainer config class"):
        TrainingArguments.load_from_json({"trainer_config_class": "NonExistentConfigXYZ"})

    @dataclass
    class _LocalCustomConfig(TrainingArguments):
        custom_tuple: tuple[int, ...] | None = field(default=None)

    reloaded = _LocalCustomConfig.load_from_json(
        {
            "trainer_config_class": "_LocalCustomConfig",
            "custom_tuple": [1, 2, 3],
            "use_wandb": False,
            "report_metrics": False,
        }
    )
    assert isinstance(reloaded, _LocalCustomConfig)
    assert reloaded.custom_tuple == (1, 2, 3)


def test_core_m3_legacy_max_sequence_length_in_from_dict():
    with pytest.warns(FutureWarning, match="max_sequence_length"):
        args = TrainingArguments.from_dict({"max_sequence_length": 512, "use_wandb": False, "report_metrics": False})
    assert args.max_length == 512


@pytest.mark.parametrize(
    ("kwargs", "exc_type"),
    [
        ({"learning_rate": True}, TypeError),
        ({"weight_decay": False}, TypeError),
        ({"warmup_steps": True}, TypeError),
        ({"gradient_accumulation_steps": True}, TypeError),
        ({"total_batch_size": True}, TypeError),
        ({"eval_batch_size": True}, TypeError),
        ({"max_length": True}, TypeError),
        ({"log_steps": True}, TypeError),
        ({"report_steps": True}, TypeError),
        ({"zero_supervision_report_fraction": True}, ValueError),
        ({"zero_supervision_abort_fraction": float("nan")}, ValueError),
        ({"arrayrecord_read_rate_burst_seconds": True}, ValueError),
        ({"arrayrecord_read_rate_burst_seconds": float("nan")}, ValueError),
        ({"arrayrecord_read_rate_burst_seconds": float("inf")}, ValueError),
        ({"arrayrecord_num_threads": True}, TypeError),
        ({"log_steps": 0}, ValueError),
        ({"report_steps": 0}, ValueError),
        ({"total_batch_size": 0}, ValueError),
        ({"eval_batch_size": 0}, ValueError),
        ({"max_length": 0}, ValueError),
        ({"warmup_steps": -1}, ValueError),
    ],
)
def test_core_m3_validation_rejects_bool_nan_inf_and_nonpositive(kwargs, exc_type):
    with pytest.raises(exc_type):
        TrainingArguments(use_wandb=False, report_metrics=False, **kwargs)


def test_core_m3_tensorboard_skipped_on_non_logging_worker_and_closes_cleanly(monkeypatch, tmp_path):
    # Non-logging worker (process_index=1, log_all_workers=False): must NOT open SummaryWriter or create dir.
    worker_dir = tmp_path / "worker1"
    args_worker = TrainingArguments(
        model_name="m",
        save_directory=str(worker_dir),
        use_wandb=False,
        report_metrics=True,
        log_all_workers=False,
    )
    args_worker.__dict__["is_process_zero"] = False
    assert args_worker.can_log_metrics is False
    assert args_worker.get_tensorboard() is None
    assert not worker_dir.exists()

    # Logging worker: opens SummaryWriter, and close_tensorboard() closes its backend cleanly.
    rank0_dir = tmp_path / "rank0"
    args_rank0 = TrainingArguments(
        model_name="m",
        save_directory=str(rank0_dir),
        use_wandb=False,
        report_metrics=True,
    )
    writer = args_rank0.get_tensorboard()
    assert writer is not None
    assert getattr(writer, "_closed", False) is False
    args_rank0.close_tensorboard()
    assert getattr(writer, "_closed", False) is True
    assert "_tensorboard_writer" not in args_rank0.__dict__
    # Idempotent second close.
    args_rank0.close_tensorboard()


def test_core_m3_get_wandb_init_no_warning_when_use_wandb_false(monkeypatch):
    monkeypatch.setattr(tc_mod, "wandb", None)
    args_off = TrainingArguments(use_wandb=False, report_metrics=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert args_off.get_wandb_init() is None
    assert not caught

    args_on = TrainingArguments(use_wandb=True, report_metrics=True)
    with pytest.warns(UserWarning, match="use_wandb=True"):
        assert args_on.get_wandb_init() is None


# ---------------------------------------------------------------------------
# CORE-M4: MetricsTracker retains mean_accuracy after None / NaN steps
# ---------------------------------------------------------------------------


def test_core_m4_metrics_tracker_retains_mean_accuracy_after_none_and_nonfinite():
    tracker = MetricsTracker()
    loss1, acc1 = tracker.update(loss=1.0, accuracy=0.8, step=1)
    assert loss1 == pytest.approx(1.0)
    assert acc1 == pytest.approx(0.8)

    loss2, acc2 = tracker.update(loss=3.0, accuracy=None, step=2)
    assert loss2 == pytest.approx(2.0)
    assert acc2 == pytest.approx(0.8)

    loss3, acc3 = tracker.update(loss=5.0, accuracy=float("nan"), step=3)
    assert loss3 == pytest.approx(3.0)
    assert acc3 == pytest.approx(0.8)

    loss4, acc4 = tracker.update(loss=3.0, accuracy=float("inf"), step=4)
    assert loss4 == pytest.approx(3.0)
    assert acc4 == pytest.approx(0.8)

    loss5, acc5 = tracker.update(loss=3.0, accuracy=0.4, step=5)
    assert loss5 == pytest.approx(3.0)
    assert acc5 == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CORE-M5: GRPO collator, constant-length dataset, pad helpers, embedding collators
# ---------------------------------------------------------------------------


def test_core_m5_grpo_pixel_attention_mask_zero_pad_and_consistent_multimodal_rank():
    tfds = GRPODataCollatorTFDS(max_prompt_length=4, pad_token_id=99)
    grain = GRPODataCollatorGrain(max_prompt_length=4, pad_token_id=99)

    row_a = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "pixel_attention_mask": [1, 1],
        "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2]], dtype=np.int32),
        "pixel_values": np.ones((4, 8), dtype=np.float32),
    }
    row_b = {
        "input_ids": [3, 4, 5],
        "attention_mask": [1, 1, 1],
        "pixel_attention_mask": [1, 1, 1],
        "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2]], dtype=np.int32),
        "pixel_values": np.ones((4, 8), dtype=np.float32) * 2.0,
    }
    row_c = {
        "input_ids": [6],
        "attention_mask": [1],
        "pixel_attention_mask": [1],
        "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2], [1, 2, 2]], dtype=np.int32),
        "pixel_values": np.ones((6, 8), dtype=np.float32) * 3.0,
    }

    batch_equal = tfds([row_a, row_b])
    batch_varied = tfds([row_a, row_c])
    grain_row = grain(row_a)

    # pixel_attention_mask must be padded with 0, NOT pad_token_id=99!
    assert batch_equal["pixel_attention_mask"].tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
    assert grain_row["pixel_attention_mask"].tolist() == [0, 0, 1, 1]

    # FLATTENABLE_MULTIMODAL_KEYS must maintain consistent rank regardless of equal vs varied item counts.
    assert batch_equal["image_grid_thw"].shape == (4, 3)
    assert batch_varied["image_grid_thw"].shape == (5, 3)
    assert batch_equal["pixel_values"].shape == (8, 8)
    assert batch_varied["pixel_values"].shape == (10, 8)


def test_core_m5_constant_length_dataset_eos_zero_callable_and_empty_string_infinite():
    class _EosZeroTok:
        eos_token_id = 0

        def __call__(self, *, text, **kwargs):
            ids = [[int(t) for t in row.split()] for row in text]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    # 1. eos_token_id == 0 must not fall back to default eos_token_id=99.
    gen = create_constant_length_dataset(
        _EosZeroTok(),
        [{"text": "1 2"}, {"text": "3 4"}],
        dataset_text_field="text",
        seq_length=6,
        eos_token_id=99,
        append_concat_token=True,
        shuffle=False,
    )
    items = list(gen())
    assert len(items) == 1
    np.testing.assert_array_equal(np.asarray(items[0]["input_ids"]), [1, 2, 0, 3, 4, 0])

    # 2. functools.partial and callable object work without AttributeError on __code__.
    def _fmt(ex, field_name):
        return ex[field_name]

    class _CallableFmt:
        def __call__(self, ex):
            return ex["body"]

    for fn in (functools.partial(_fmt, field_name="body"), _CallableFmt()):
        g = create_constant_length_dataset(
            _EosZeroTok(),
            [{"body": "5 6 7"}],
            formatting_func=fn,
            seq_length=4,
            append_concat_token=True,
            shuffle=False,
        )
        out = list(g())
        assert len(out) == 1
        np.testing.assert_array_equal(np.asarray(out[0]["input_ids"]), [5, 6, 7, 0])

    # 3. Infinite empty-string dataset must terminate instead of spinning forever.
    empty_gen = create_constant_length_dataset(
        _EosZeroTok(),
        [{"text": ""}, {"text": ""}],
        dataset_text_field="text",
        seq_length=4,
        infinite=True,
        append_concat_token=False,
        shuffle=False,
    )
    with pytest.warns(UserWarning, match="iterator is reset"):
        assert list(empty_gen()) == []


def test_core_m5_pad_single_max_length_zero_and_pad_to_length_dtype():
    assert pad_single(np.array([1, 2, 3], dtype=np.int32), max_length=0, padding_side="left").shape == (0,)
    assert pad_single(np.array([1, 2, 3], dtype=np.int32), max_length=0, padding_side="right").shape == (0,)
    assert _pad_single_along_axis(
        np.array([[1, 2, 3]], dtype=np.int32),
        max_length=0,
        padding_value=0,
        axis=-1,
        padding_side="left",
    ).shape == (1, 0)

    bool_padded = pad_to_length(jnp.array([True, False], dtype=jnp.bool_), 4, 0)
    assert bool_padded.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(bool_padded), [True, False, False, False])

    int_padded = pad_to_length(jnp.array([1, 2], dtype=jnp.int32), 4, 0.0)
    assert int_padded.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(int_padded), [1, 2, 0, 0])


def test_core_m5_embedding_collators_pad_and_truncate():
    examples = [
        {
            "query_input_ids": [1, 2],
            "query_attention_mask": [1, 1],
            "positive_input_ids": [3, 4, 5, 6, 7],
            "positive_attention_mask": [1, 1, 1, 1, 1],
            "negative_input_ids": [8],
            "negative_attention_mask": [1],
        },
        {
            "query_input_ids": [10, 11, 12, 13, 14],
            "query_attention_mask": [1, 1, 1, 1, 1],
            "positive_input_ids": [20],
            "positive_attention_mask": [1],
            "negative_input_ids": [30, 31, 32],
            "negative_attention_mask": [1, 1, 1],
        },
    ]

    tfds_batch = EmbeddingDataCollatorTFDS(pad_token_id=7, max_length=4, has_negatives=True)(examples)
    grain_batch = _load_grain_batch(
        examples,
        EmbeddingDataCollatorGrain(pad_token_id=7, max_length=4, has_negatives=True),
    )

    for batch in (tfds_batch, grain_batch):
        np.testing.assert_array_equal(np.asarray(batch["query_input_ids"]), [[1, 2, 7, 7], [10, 11, 12, 13]])
        np.testing.assert_array_equal(np.asarray(batch["query_attention_mask"]), [[1, 1, 0, 0], [1, 1, 1, 1]])
        np.testing.assert_array_equal(np.asarray(batch["positive_input_ids"]), [[3, 4, 5, 6], [20, 7, 7, 7]])
        np.testing.assert_array_equal(np.asarray(batch["positive_attention_mask"]), [[1, 1, 1, 1], [1, 0, 0, 0]])
        np.testing.assert_array_equal(np.asarray(batch["negative_input_ids"]), [[8, 7, 7, 7], [30, 31, 32, 7]])
        np.testing.assert_array_equal(np.asarray(batch["negative_attention_mask"]), [[1, 0, 0, 0], [1, 1, 1, 0]])


# ---------------------------------------------------------------------------
# CORE-L2: StepMetrics mode=None & summary reduction alignment
# ---------------------------------------------------------------------------


def test_core_l2_step_metrics_mode_none_and_summary_reduction():
    sm = StepMetrics(arguments=SimpleNamespace(performance_mode=True))
    res = sm.calculate(
        metrics=LossMetrics(loss=0.5, execution_time=1.0),
        current_step=7,
        epoch=0,
        epoch_progress=None,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=1e-4,
        mode=None,
    )
    assert "None_step" not in res
    assert res["step"] == 7
    assert "performance/step/total_flops" in res
    assert StepMetrics._get_summary_metric_reduction("performance/step/total_flops", mode=None) == "sum"
    assert StepMetrics._get_summary_metric_reduction("performance/step/total_tokens", mode=None) == "sum"
    assert StepMetrics._get_summary_metric_reduction("performance/step/total_time", mode=None) == "ignore"
    assert StepMetrics._get_summary_metric_reduction("performance/eval/total_flops", mode="eval") == "sum"
    assert StepMetrics._get_summary_metric_reduction("performance/total_flops", mode="train") == "sum"


# ---------------------------------------------------------------------------
# CORE-L3: Reward collators padding modes, shift_and_pad, formatting, add_eos
# ---------------------------------------------------------------------------


def test_core_l3_reward_collators_padding_modes():
    tok = SimpleNamespace(pad_token_id=0)
    f1 = {
        "input_ids_chosen": [1, 2, 3],
        "attention_mask_chosen": [1, 1, 1],
        "input_ids_rejected": [4, 5],
        "attention_mask_rejected": [1, 1],
    }
    f2 = {
        "input_ids_chosen": [6, 7, 8],
        "attention_mask_chosen": [1, 1, 1],
        "input_ids_rejected": [9, 10],
        "attention_mask_rejected": [1, 1],
    }

    # padding=False / "do_not_pad" must NOT pad to max_length or pad_to_multiple_of.
    for mode in (False, "do_not_pad"):
        b_tfds = RewardDataCollatorWithPaddingTFDS(tok, padding=mode, max_length=8, pad_to_multiple_of=4)([f1, f2])
        b_grain = RewardDataCollatorWithPaddingGrain(tok, padding=mode, max_length=8, pad_to_multiple_of=4)(f1)
        assert b_tfds["input_ids_chosen"].shape == (2, 3)
        assert b_tfds["input_ids_rejected"].shape == (2, 2)
        assert b_grain["input_ids_chosen"].shape == (3,)
        assert b_grain["input_ids_rejected"].shape == (2,)

    # padding=True / "longest" pads to longest in batch/example (capped by max_length), NOT full max_length.
    for mode in (True, "longest"):
        b_tfds = RewardDataCollatorWithPaddingTFDS(tok, padding=mode, max_length=8)([f1, f2])
        b_grain = RewardDataCollatorWithPaddingGrain(tok, padding=mode, max_length=8)(f1)
        assert b_tfds["input_ids_chosen"].shape == (2, 3)
        assert b_tfds["input_ids_rejected"].shape == (2, 3)
        assert b_grain["input_ids_chosen"].shape == (3,)
        assert b_grain["input_ids_rejected"].shape == (3,)


def test_core_l3_shift_and_pad_all_zero_jax_and_3d():
    # NumPy with all-zero row and 3D tensor (B=2, S=4, D=2).
    mask_np = np.array([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=np.int32)
    t3d_np = np.arange(16, dtype=np.int32).reshape(2, 4, 2)
    expected_row0 = np.roll(t3d_np[0], shift=-2, axis=0)
    expected_row1 = t3d_np[1].copy()

    out_mask, out_t3d = shift_and_pad(mask_np, t3d_np)
    np.testing.assert_array_equal(out_mask, [[1, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(out_t3d[0], expected_row0)
    np.testing.assert_array_equal(out_t3d[1], expected_row1)

    # Immutable JAX arrays with all-zero row and 3D tensor.
    mask_jax = jnp.array([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=jnp.int32)
    t3d_jax = jnp.arange(16, dtype=jnp.int32).reshape(2, 4, 2)
    out_mask_j, out_t3d_j = shift_and_pad(mask_jax, t3d_jax)
    assert isinstance(out_mask_j, jax.Array)
    assert isinstance(out_t3d_j, jax.Array)
    np.testing.assert_array_equal(np.asarray(out_mask_j), [[1, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(np.asarray(out_t3d_j[0]), np.roll(np.asarray(t3d_jax[0]), shift=-1, axis=0))
    np.testing.assert_array_equal(np.asarray(out_t3d_j[1]), np.asarray(t3d_jax[1]))


def test_core_l3_conversations_formatting_empty_and_add_eos_none():
    tok = SimpleNamespace(apply_chat_template=lambda msgs, tokenize=False, tools=None: str(msgs))
    formatter = conversations_formatting_function(tok, messages_field="messages")
    assert formatter({"messages": []}) == []

    chosen = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    rejected = {"input_ids": [3], "attention_mask": [1]}
    c_out, r_out = add_eos_token_if_needed(None, chosen, rejected)
    assert c_out["input_ids"] == [1, 2]
    assert r_out["input_ids"] == [3]


# ---------------------------------------------------------------------------
# PKD-L2: Preserve completion_decoder_input_ids in BCO collators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_encoder_decoder", [True, False])
def test_pkd_l2_bco_collators_preserve_completion_decoder_input_ids(is_encoder_decoder):
    row = {
        "prompt_input_ids": [1, 2],
        "prompt_attention_mask": [1, 1],
        "completion_input_ids": [10, 11],
        "completion_attention_mask": [1, 1],
        "completion_labels": [10, 11],
        "completion_decoder_input_ids": [0, 10],
        "label": True,
    }
    expected_len = 4 if is_encoder_decoder else 7
    tfds_out = BCODataCollatorTFDS(
        max_prompt_length=3,
        max_completion_length=4,
        pad_token_id=9,
        label_pad_token_id=-100,
        is_encoder_decoder=is_encoder_decoder,
    )([row])
    grain_out = BCODataCollatorGrain(
        max_prompt_length=3,
        max_completion_length=4,
        pad_token_id=9,
        label_pad_token_id=-100,
        is_encoder_decoder=is_encoder_decoder,
    )(row)

    assert "completion_decoder_input_ids" in tfds_out
    assert "completion_decoder_input_ids" in grain_out
    assert tfds_out["completion_decoder_input_ids"].shape == (1, expected_len)
    assert grain_out["completion_decoder_input_ids"].shape == (expected_len,)
    expected_dec = [0, 10] + [9] * (expected_len - 2)
    assert tfds_out["completion_decoder_input_ids"].tolist() == [expected_dec]
    assert grain_out["completion_decoder_input_ids"].tolist() == expected_dec


# ---------------------------------------------------------------------------
# Additional parity & edge-case coverage across roundtrip / MLM / pickling
# ---------------------------------------------------------------------------


def test_training_arguments_full_roundtrip_and_optimizer_kwargs_parity(tmp_path):
    args = TrainingArguments(
        model_name="full-rt",
        save_directory=str(tmp_path),
        use_wandb=False,
        report_metrics=False,
        tx_mu_dtype=jnp.bfloat16,
        extra_optimizer_kwargs={"mu_dtype": "bfloat16", "b1": 0.85},
        weight_decay=0.02,
        learning_rate=2e-4,
        max_training_steps=50,
        warmup_steps=5,
        scheduler=EasyDeLSchedulers.COSINE,
    )
    assert args.optimizer_kwargs["mu_dtype"] == "bfloat16"
    assert args.optimizer_kwargs["b1"] == 0.85

    path = tmp_path / "full.json"
    args.save_arguments(path)
    loaded = TrainingArguments.load_arguments(path)
    assert loaded.to_dict() == args.to_dict()
    tx, sched = loaded.get_optimizer_and_scheduler()
    assert tx is not None
    assert float(sched(5)) == pytest.approx(2e-4, rel=1e-3)

    # Legacy stringified runtime-only fields in saved JSON are safely skipped on load.
    legacy = TrainingArguments.from_dict(
        {
            "use_wandb": False,
            "report_metrics": False,
            "custom_scheduler": "<function foo at 0x1234>",
            "pruning_module": "<DummyPruning>",
            "watchers": ["LogWatcher(...)"],
        }
    )
    assert legacy.custom_scheduler is None
    assert legacy.pruning_module is None
    assert legacy.watchers == []


def test_completion_only_collator_mlm_emits_attention_mask():
    class _MLMTok(_PublicPadTokenizer):
        pad_token_id = 50
        mask_token_id = 49

        def _convert_id_to_token(self, token):
            return str(token)

        def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
            return [token in (50, 99) for token in ids]

        def __len__(self):
            return 100

    collator = DataCollatorForCompletionOnlyLM(_MLMTok(), response_template=[7, 8], mlm=True)
    with pytest.warns(UserWarning):
        out = collator([[1, 7, 8, 10, 11, 12], [1, 7, 8, 10]])
    assert "attention_mask" in out
    np.testing.assert_array_equal(
        np.asarray(out["attention_mask"]),
        [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]],
    )


def test_hf_data_source_pickle_roundtrip(monkeypatch):
    import pickle

    monkeypatch.setenv("HFDATASOURCE_NONSTOP", "0")
    ds = Dataset.from_dict({"x": [0, 1, 2, 3]}).to_iterable_dataset(num_shards=2)
    options = pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False)
    src = HFDataSource(ds, options, num_threads=2)
    restored = pickle.loads(pickle.dumps(src))
    assert restored[0]["x"] in {0, 1, 2, 3}

# =============================================================================
# Domain C: Data Lanes, Assistant Masking, Prompts, Vision & Packing
# =============================================================================

class _SimpleVocabTokenizer:
    """Minimal deterministic tokenizer supporting chat templating, BatchEncoding, and truncation."""

    def __init__(
        self,
        *,
        eos_token: str | None = "<eos>",
        bos_token: str | None = "<bos>",
        pad_token: str | None = "<pad>",
        return_batch_encoding: bool = False,
        fuse_newline_header: bool = False,
    ) -> None:
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.padding_side = "right"
        self.truncation_side = "right"
        self.chat_template = "custom"
        self.return_batch_encoding = return_batch_encoding
        self.fuse_newline_header = fuse_newline_header
        self.chat_template_calls: list[dict[str, Any]] = []
        self._vocab: dict[str, int] = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<|im_start|>": 3,
            "<|im_end|>": 4,
            "user": 5,
            "assistant": 6,
            "system": 7,
            "\n": 8,
            "<|im_start|>assistant\n": 9,
            "end_a": 200,
            "end_b": 201,
        }
        self._next_id = 20

    @property
    def pad_token_id(self) -> int | None:
        return None if self.pad_token is None else self._vocab.get(self.pad_token, 0)

    @property
    def bos_token_id(self) -> int | None:
        return None if self.bos_token is None else self._vocab.get(self.bos_token, 1)

    @property
    def eos_token_id(self) -> int | None:
        return None if self.eos_token is None else self._vocab.get(self.eos_token, 2)

    def _tok_id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = self._next_id
            self._next_id += 1
        return self._vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special_tokens and self.bos_token_id is not None:
            ids.append(self.bos_token_id)

        remaining = text
        if self.fuse_newline_header and "<|im_start|>assistant\n" in remaining:
            parts = remaining.split("<|im_start|>assistant\n")
            for idx, part in enumerate(parts):
                if idx > 0:
                    ids.append(self._vocab["<|im_start|>assistant\n"])
                ids.extend(self._encode_plain(part))
            return ids

        ids.extend(self._encode_plain(remaining))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in (
            text.replace("<|im_start|>", " <|im_start|> ")
            .replace("<|im_end|>", " <|im_end|> ")
            .replace("\n", " \n ")
            .split(" ")
        ):
            if token == "":
                continue
            ids.append(self._tok_id(token))
        return ids

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        is_batched = isinstance(text, list)
        items = text if is_batched else [text]
        out_ids: list[list[int]] = []
        out_mask: list[list[int]] = []
        for item in items:
            ids = self.encode(item, add_special_tokens=add_special_tokens)
            if truncation and max_length is not None and len(ids) > max_length:
                if self.truncation_side == "left":
                    ids = ids[-max_length:]
                else:
                    ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length" and max_length is not None and len(ids) < max_length:
                pad_n = max_length - len(ids)
                pad_id = 0 if self.pad_token_id is None else self.pad_token_id
                if self.padding_side == "left":
                    ids = [pad_id] * pad_n + ids
                    mask = [0] * pad_n + mask
                else:
                    ids = ids + [pad_id] * pad_n
                    mask = mask + [0] * pad_n
            out_ids.append(ids)
            out_mask.append(mask)
        if not is_batched:
            res: dict[str, Any] = {"input_ids": out_ids[0]}
            if return_attention_mask:
                res["attention_mask"] = out_mask[0]
            return res
        res = {"input_ids": out_ids}
        if return_attention_mask:
            res["attention_mask"] = out_mask
        return res

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        chunks: list[str] = []
        if self.bos_token is not None:
            chunks.append(self.bos_token)
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
        rendered = " ".join(chunks)
        if not tokenize:
            return rendered
        ids = self.encode(rendered, add_special_tokens=False)
        if self.return_batch_encoding:
            return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})
        return ids


# ---------------------------------------------------------------------------
# CORE-C2 & CORE-H14 & CORE-M6 & CORE-L5: assistant_only_mask.py
# ---------------------------------------------------------------------------


class _DummyMaskHost(assistant_only_mask.AssistantOnlyMaskMixin):
    def __init__(self, span_ids: assistant_only_mask.AssistantSpanIds | None) -> None:
        self._span_ids = span_ids
        self.resolve_calls = 0

    def _assistant_span_ids(self) -> assistant_only_mask.AssistantSpanIds:
        self.resolve_calls += 1
        assert self._span_ids is not None
        return self._span_ids


def test_core_c2_assemble_per_dataset_mask_updates_preexisting_labels() -> None:
    span = assistant_only_mask.AssistantSpanIds(header_ids=(3, 6, 8), end_id=4)
    host = _DummyMaskHost(span)

    # Packed row with 2 docs + 2 padding tokens (segment_ids == -1):
    # Doc 0 (seg 0, ao=1 -> mask prompt): tokens [5, 10, 3, 6, 8, 11, 4]
    # Doc 1 (seg 1, ao=0 -> supervise all): tokens [5, 12, 3, 6, 8, 13, 4]
    # Padding (seg -1): tokens [0, 0]
    input_ids = np.array([[5, 10, 3, 6, 8, 11, 4, 5, 12, 3, 6, 8, 13, 4, 0, 0]], dtype=np.int32)
    segment_ids = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1]], dtype=np.int32)
    ao = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1]], dtype=np.int8)
    # Pre-existing labels has prompt masked on doc 1 AND prompt UNMASKED on doc 0
    original_labels = np.array(
        [[5, 10, 3, 6, 8, 11, 4, -100, -100, -100, -100, -100, 13, 4, 0, 0]],
        dtype=np.int32,
    )
    labels_snapshot = original_labels.copy()

    batch = {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "labels": original_labels,
    }
    host._assemble_per_dataset_mask(batch, ao, global_flag=False)

    # Original array must not be mutated in place
    assert np.array_equal(original_labels, labels_snapshot)

    cm = batch["completion_mask"]
    labels = batch["labels"]
    # Doc 0 (ao=1): only response token 11 + end token 4 supervised
    assert cm[0, :7].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert labels[0, :7].tolist() == [-100, -100, -100, -100, -100, 11, 4]
    # Doc 1 (ao=0): interior valid doc tokens restored from -100 to input_ids,
    # while non-leading segment start (index 7) preserves -100 so Doc 0 never predicts Doc 1's first token.
    assert cm[0, 7:14].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert labels[0, 7:14].tolist() == [-100, 12, 3, 6, 8, 13, 4]
    # Padding (seg=-1): masked out in both completion_mask and labels
    assert cm[0, 14:].tolist() == [0, 0]
    assert labels[0, 14:].tolist() == [-100, -100]


def test_core_h14_need_scan_gated_on_uncovered_wanted_positions() -> None:
    span = assistant_only_mask.AssistantSpanIds(header_ids=(3, 6, 8), end_id=4)

    # Case 1: ao==1 everywhere, but completion_mask already covers all positions -> no scan!
    host_covered = _DummyMaskHost(span)
    batch_covered = {
        "input_ids": np.array([[5, 3, 6, 8, 11, 4]], dtype=np.int32),
        "completion_mask": np.array([[0, 0, 0, 0, 1, 1]], dtype=np.int32),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32),
    }
    ao_all_one = np.ones((1, 6), dtype=np.int8)
    host_covered._assemble_per_dataset_mask(batch_covered, ao_all_one, global_flag=False)
    assert host_covered.resolve_calls == 0
    assert batch_covered["completion_mask"][0].tolist() == [0, 0, 0, 0, 1, 1]

    # Case 2: ao==-1 (inherit global_flag=True) with no pre-existing completion_mask -> MUST scan!
    host_inherit = _DummyMaskHost(span)
    batch_inherit = {
        "input_ids": np.array([[5, 3, 6, 8, 11, 4]], dtype=np.int32),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 0]], dtype=np.int32),
    }
    ao_inherit = np.full((1, 6), -1, dtype=np.int8)
    host_inherit._assemble_per_dataset_mask(batch_inherit, ao_inherit, global_flag=True)
    assert host_inherit.resolve_calls == 1
    assert batch_inherit["completion_mask"][0].tolist() == [0, 0, 0, 0, 1, 1]


def test_core_m6_assistant_span_multitoken_end_fused_newline_and_zero_stats() -> None:
    # Multi-token end_ids=(200, 201): bare 200 inside content must NOT end assistant span
    span = assistant_only_mask.AssistantSpanIds(header_ids=(3, 6, 8), end_id=200, end_ids=(200, 201))
    input_ids = np.array([[5, 3, 6, 8, 10, 200, 11, 200, 201, 5, 12]], dtype=np.int32)
    seg = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.int32)
    mask = assistant_only_mask.assistant_mask_from_token_ids(input_ids, span, segment_ids=seg)
    assert mask[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]

    # Empty seqlen guard
    empty_ids = np.zeros((2, 0), dtype=np.int32)
    assert assistant_only_mask.assistant_mask_from_token_ids(empty_ids, span).shape == (2, 0)

    # Fused newline header must NOT add standalone "\n" to newline_family
    tok_fused = _SimpleVocabTokenizer(fuse_newline_header=True)
    derived = assistant_only_mask.derive_assistant_span_ids(
        tok_fused,
        header_text="<|im_start|>assistant\n",
        end_text="end_a end_b",
    )
    assert derived.header_ids == (9,)
    assert derived.newline_family == ()
    assert derived.end_ids == (200, 201)

    # Vectorized zero_supervision_stats matches exact counts across segments & padding
    ids_stats = np.array(
        [
            [5, 10, 3, 6, 8, 11, 4, 5, 12, 13, 0, 0],
            [5, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    cm_stats = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    seg_stats = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1],
            [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int32,
    )
    stats = assistant_only_mask.zero_supervision_stats(
        ids_stats,
        cm_stats,
        segment_ids=seg_stats,
        header_ids=(3, 6, 8),
    )
    assert stats.n_docs == 3
    assert stats.n_zero == 2


def test_core_l5_segment_aware_label_mask_all_negative_segments() -> None:
    labels = np.array([[-100, -100, -100]], dtype=np.int32)
    seg_all_neg = np.array([[-2, -2, -3]], dtype=np.int32)
    res = assistant_only_mask._segment_aware_label_mask(labels, attention=None, segment_ids=seg_all_neg)
    assert res == (None, None, None)

    # Mixed valid (seg=0) and padding (seg=-1) with -100 only on padding -> returns (None, None, None)
    labels_pad_only = np.array([[5, 6, -100, -100]], dtype=np.int32)
    seg_mixed = np.array([[0, 0, -1, -1]], dtype=np.int32)
    res2 = assistant_only_mask._segment_aware_label_mask(labels_pad_only, attention=None, segment_ids=seg_mixed)
    assert res2 == (None, None, None)


# ---------------------------------------------------------------------------
# CORE-C3 & CORE-H14 & CORE-H15 & CORE-M6 & CORE-L5: prompt_transforms.py
# ---------------------------------------------------------------------------


def test_core_c3_sft_prompt_completion_unwraps_batch_encoding_and_gates_scan() -> None:
    tok = _SimpleVocabTokenizer(return_batch_encoding=True)
    # With mask_prompt=True: BatchEncoding must be unwrapped to exact input_ids length (not len(dict)==2!)
    sft_masked = prompt_transforms.SFTPreprocessTransform(tok, max_length=64, mask_prompt=True, padding=False)
    row = {
        "prompt": [{"role": "user", "content": "hello world"}],
        "completion": [{"role": "assistant", "content": "good morning"}],
    }
    out_masked = sft_masked(row)
    expected_prompt_ids = tok.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True
    )["input_ids"]
    cm = out_masked["completion_mask"]
    assert sum(1 for x in cm if x == 0) == len(expected_prompt_ids)
    assert sum(cm) == len(out_masked["input_ids"]) - len(expected_prompt_ids)

    # With mask_prompt=False: apply_chat_template(..., tokenize=True) must NOT be called
    tok_unmasked = _SimpleVocabTokenizer(return_batch_encoding=True)
    sft_unmasked = prompt_transforms.SFTPreprocessTransform(
        tok_unmasked, max_length=64, mask_prompt=False, padding=False
    )
    out_unmasked = sft_unmasked(row)
    assert "input_ids" in out_unmasked
    assert not any(call["tokenize"] for call in tok_unmasked.chat_template_calls)


def test_core_h14_temporary_truncation_side_thread_safe() -> None:
    tok = _SimpleVocabTokenizer()
    tok.truncation_side = "right"
    t_left = prompt_transforms.SFTPreprocessTransform(
        tok, max_length=32, truncation_mode="keep_end", padding=False
    )
    t_right = prompt_transforms.SFTPreprocessTransform(
        tok, max_length=32, truncation_mode="keep_start", padding=False
    )
    errors: list[str] = []

    def worker(idx: int) -> None:
        transform = t_left if idx % 2 == 0 else t_right
        expected = "left" if idx % 2 == 0 else "right"
        for _ in range(30):
            with transform._temporary_truncation_side():
                observed = tok.truncation_side
                if observed != expected:
                    errors.append(f"expected {expected}, got {observed}")
                time.sleep(0.0002)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert not errors
    assert tok.truncation_side == "right"


def test_core_h15_sft_vlm_keys_eos_none_and_left_pad_labels() -> None:
    # _TOKENIZED_FIELDS includes VLM/mRoPE/segment/packed-embed keys
    required_keys = {
        "image_grid_thw",
        "video_grid_thw",
        "token_type_ids",
        "mm_token_type_ids",
        "segment_ids",
        "image_embeds",
        "embed_n_tok",
    }
    assert required_keys.issubset(prompt_transforms._TOKENIZED_FIELDS)

    # Pretokenized SFT preserves VLM keys
    tok_no_eos = _SimpleVocabTokenizer(eos_token=None)
    sft = prompt_transforms.SFTPreprocessTransform(tok_no_eos, max_length=16, padding=False)
    pretok = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "image_grid_thw": [[1, 2, 2]],
        "segment_ids": [0, 0, 0, 0],
    }
    preserved = sft(pretok)
    assert preserved["image_grid_thw"] == [[1, 2, 2]]
    assert preserved["segment_ids"] == [0, 0, 0, 0]

    # eos_token=None does not crash when add_eos=True
    out_text = sft({"text": "hello world"})
    assert len(out_text["input_ids"]) > 0

    # _pad_row_to_multiple respects padding_side="left" and pads labels with -100
    tok_left = _SimpleVocabTokenizer()
    tok_left.padding_side = "left"
    sft_left = prompt_transforms.SFTPreprocessTransform(
        tok_left, max_length=16, padding=False, pad_to_multiple_of=8
    )
    padded = sft_left._pad_row_to_multiple(
        {
            "input_ids": [10, 11, 12],
            "attention_mask": [1, 1, 1],
            "labels": [10, 11, 12],
            "completion_mask": [0, 1, 1],
        }
    )
    assert padded["input_ids"] == [0, 0, 0, 0, 0, 10, 11, 12]
    assert padded["attention_mask"] == [0, 0, 0, 0, 0, 1, 1, 1]
    assert padded["labels"] == [-100, -100, -100, -100, -100, 10, 11, 12]
    assert padded["completion_mask"] == [0, 0, 0, 0, 0, 0, 1, 1]


def test_core_h15_dpo_and_kto_dedup_eos_zero_prompt_and_partial_caps() -> None:
    tok = _SimpleVocabTokenizer()
    eos_id = tok.eos_token_id
    assert eos_id is not None

    # DPO: completion ending with EOS must not get a second EOS; max_prompt_length=0 must yield empty prompt
    dpo_zero = prompt_transforms.DPOPreprocessTransform(tok, max_prompt_length=0, max_completion_length=16)
    res_dpo_zero = dpo_zero(
        {
            "prompt": "alpha beta gamma",
            "chosen": "good <eos>",
            "rejected": "bad <eos>",
        }
    )
    assert res_dpo_zero["prompt_input_ids"] == []
    assert res_dpo_zero["chosen_input_ids"][-2:] != [eos_id, eos_id]
    assert res_dpo_zero["rejected_input_ids"][-2:] != [eos_id, eos_id]

    # DPO partial cap (max_prompt_length set, max_completion_length=None) must NOT pad ragged completions
    dpo_partial = prompt_transforms.DPOPreprocessTransform(tok, max_prompt_length=4, max_completion_length=None)
    res_dpo_partial = dpo_partial(
        {
            "prompt": "alpha beta",
            "chosen": "short",
            "rejected": "a much longer rejected completion text here",
        }
    )
    assert len(res_dpo_partial["chosen_input_ids"]) < len(res_dpo_partial["rejected_input_ids"])

    # KTO: max_prompt_length=0 yields empty prompt and no duplicate EOS
    kto_zero = prompt_transforms.KTOPreprocessTransform(tok, max_prompt_length=0, max_completion_length=16)
    res_kto = kto_zero({"prompt": "alpha beta", "completion": "reply <eos>", "label": True})
    assert res_kto["prompt_input_ids"] == []
    assert res_kto["completion_input_ids"][-2:] != [eos_id, eos_id]


def test_core_m6_grpo_reward_and_sft_conversational_transforms() -> None:
    tok = _SimpleVocabTokenizer()

    # GRPO left-truncates prompts and retains None sideband values
    grpo = prompt_transforms.GRPOPreprocessTransform(
        tok, max_prompt_length=3, skip_apply_chat_template=True
    )
    full_ids = tok("one two three four five")["input_ids"]
    grpo_out = grpo({"prompt": "one two three four five", "answer": None, "meta": "ok"})
    assert grpo_out["input_ids"] == full_ids[-3:]
    assert "answer" in grpo_out and grpo_out["answer"] is None
    assert grpo_out["meta"] == "ok"

    # RewardPreprocessTransform preserves prompt when chosen/rejected don't start with it,
    # avoids duplication when they do, and avoids double BOS on conversational rows.
    rew = prompt_transforms.RewardPreprocessTransform(tok, max_length=64)
    out_plain = rew({"prompt": "Question: ", "chosen": "Yes", "rejected": "No"})
    q_tok = tok.encode("Question:", add_special_tokens=False)[0]
    assert q_tok in out_plain["input_ids_chosen"]
    assert q_tok in out_plain["input_ids_rejected"]

    out_prefixed = rew({"prompt": "Question: ", "chosen": "Question: Yes", "rejected": "Question: No"})
    assert out_prefixed["input_ids_chosen"].count(q_tok) == 1

    out_conv = rew(
        {
            "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "no"}],
        }
    )
    bos_id = tok.bos_token_id
    assert bos_id is not None
    assert out_conv["input_ids_chosen"].count(bos_id) == 1

    # SFT conversational fallback does not double BOS or double EOS when chat template succeeds
    sft = prompt_transforms.SFTPreprocessTransform(tok, max_length=64, mask_prompt=False, padding=False)
    out_sft = sft(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
    )
    assert out_sft["input_ids"].count(bos_id) == 1
    assert out_sft["input_ids"][-1] != tok.eos_token_id


def test_core_l5_embedding_transform_pretokenized_and_none_negative() -> None:
    tok = _SimpleVocabTokenizer()
    emb = prompt_transforms.EmbeddingPreprocessTransform(
        tok, max_length=16, query_field="query", positive_field="positive", negative_field="negative"
    )
    # Pretokenized passthrough
    pre = {"query_input_ids": [1, 2], "positive_input_ids": [3, 4], "negative_input_ids": [5, 6]}
    assert emb(pre) == pre

    # None negative does not crash
    out = emb({"query": "hello", "positive": "world", "negative": None})
    assert "query_input_ids" in out and "positive_input_ids" in out
    assert "negative_input_ids" not in out


# ---------------------------------------------------------------------------
# CORE-H15 & CORE-M6 & CORE-L5: prompt_utils.py
# ---------------------------------------------------------------------------


def test_core_h15_m6_l5_prompt_utils() -> None:
    # CORE-H15: maybe_extract_prompt guards
    already = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a"}],
        "rejected": [{"role": "assistant", "content": "b"}],
    }
    assert prompt_utils.maybe_extract_prompt(already) == already
    empty_chosen = {"prompt": [], "chosen": [], "rejected": []}
    assert prompt_utils.maybe_extract_prompt(empty_chosen) == empty_chosen

    # CORE-H15: normalize_message_payload validates role/content
    bad_payload = [{"role": 123, "content": "hi"}]
    assert prompt_utils.normalize_message_payload(bad_payload, allow_plain_text=False) is None
    fallback = prompt_utils.normalize_message_payload(bad_payload, allow_plain_text=True)
    assert fallback is not None and fallback[0]["role"] == "user"

    missing_body = [{"role": "user"}]
    assert prompt_utils.normalize_message_payload(missing_body, allow_plain_text=False) is None

    # CORE-M6: normalize_tool_payload / resolve_example_tools nullish strings
    for nullish in ("", "   ", "null", "NULL", "none", "None"):
        assert prompt_utils.normalize_tool_payload(nullish) is None
        assert prompt_utils.resolve_example_tools({"tools": nullish}) is None

    # CORE-M6: unpair_preference_dataset preserves extra metadata columns
    ds = Dataset.from_dict(
        {
            "prompt": ["p1", "p2"],
            "chosen": ["c1", "c2"],
            "rejected": ["r1", "r2"],
            "source": ["s1", "s2"],
            "weight": [1.5, 2.5],
        }
    )
    unpaired = prompt_utils.unpair_preference_dataset(ds)
    assert len(unpaired) == 4
    assert unpaired["source"] == ["s1", "s2", "s1", "s2"]
    assert unpaired["weight"] == [1.5, 2.5, 1.5, 2.5]

    # CORE-M6: extract_prompt multi-turn shared prefix ending in assistant turn
    multi = {
        "chosen": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2_good"},
        ],
        "rejected": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
    }
    extracted = prompt_utils.extract_prompt(multi)
    assert len(extracted["prompt"]) == 1 and extracted["prompt"][0]["content"] == "u1"
    assert len(extracted["rejected"]) == 1 and extracted["rejected"][0]["content"] == "a1"

    # CORE-M6: apply_chat_template renders prompt once (not 3x)
    tok = _SimpleVocabTokenizer()
    ex = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    prompt_utils.apply_chat_template(ex, tok)
    prompt_only_calls = [c for c in tok.chat_template_calls if c["messages"] == ex["prompt"]]
    assert len(prompt_only_calls) == 1

    # CORE-L5: reverse_openai_format joins all text parts with "\n"
    rev = prompt_utils.reverse_openai_format(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "image", "image": "ignored"},
                    {"type": "text", "text": "line2"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
    )
    assert isinstance(rev, list) and rev[0]["content"] == "line1\nline2" and rev[1]["content"] == "done"

    # CORE-L5: keep_array_and_primitives keeps numpy numeric/bool scalars, drops strings/objects
    import ml_dtypes

    cleaned = prompt_utils.keep_array_and_primitives(
        {
            "np_int": np.int64(7),
            "np_float": np.float32(1.5),
            "np_bf16": ml_dtypes.bfloat16(1.5),
            "np_bool": np.bool_(True),
            "arr": np.array([1, 2]),
            "text": "drop me",
            "obj": object(),
        }
    )
    assert set(cleaned.keys()) == {"np_int", "np_float", "np_bf16", "np_bool", "arr"}


# ---------------------------------------------------------------------------
# CORE-H14 & CORE-M6 & CORE-L5: data_lanes.py
# ---------------------------------------------------------------------------


_FAST_FACTORY_LOCK = threading.Lock()
_FAST_FACTORY_INITS = 0


def _module_fast_encoder_factory(tok: _SimpleVocabTokenizer):
    global _FAST_FACTORY_INITS
    time.sleep(0.01)
    with _FAST_FACTORY_LOCK:
        _FAST_FACTORY_INITS += 1
    return lambda text: tok.encode(text, add_special_tokens=False)


def test_core_h14_m6_l5_data_lanes_loader_tokenizer_and_validation(tmp_path) -> None:
    # CORE-M6: _PASSTHROUGH_FIELDS includes completion_mask and assistant_masks
    assert "completion_mask" in data_lanes._PASSTHROUGH_FIELDS
    assert "assistant_masks" in data_lanes._PASSTHROUGH_FIELDS

    import grain.python as grain

    # CORE-H14: scalar embed_n_tok works; embed_rows > embed_cap raises; base_shift aligns cursor step
    rows = [
        {
            "input_ids": np.array([1, 2, 3, 4], dtype=np.int32),
            "image_embeds": np.ones((2, 4), dtype=np.float32),
            "embed_n_tok": np.int64(2),
            "__dataset_id__": 0,
            "__assistant_only__": 0,
        },
        {
            "input_ids": np.array([5, 6, 7, 8], dtype=np.int32),
            "image_embeds": np.ones((2, 4), dtype=np.float32),
            "embed_n_tok": 2,
            "__dataset_id__": 0,
            "__assistant_only__": 0,
        },
        {
            "input_ids": np.array([9, 10, 11, 12], dtype=np.int32),
            "image_embeds": np.ones((1, 4), dtype=np.float32),
            "embed_n_tok": 1,
            "__dataset_id__": 0,
            "__assistant_only__": 0,
        },
    ]
    loader = data_lanes.PackedParquetLoader(
        lambda shift: grain.MapDataset.source(rows),
        batch_size=1,
        pack_seq_len=8,
        max_steps=11,
        misfit_patience=1,
        embed_cap=4,
        base_shift=10,
    )
    batches = list(loader)
    assert len(batches) == 1
    assert len(batches[0]) == 2
    assert loader.cursor_at(11) is not None
    assert loader.cursor_at(1) is None

    # Overflow embed_rows > embed_cap raises ValueError
    loader_overflow = data_lanes.PackedParquetLoader(
        lambda shift: grain.MapDataset.source(
            [
                {
                    "input_ids": np.array([1, 2, 3, 4], dtype=np.int32),
                    "image_embeds": np.ones((5, 4), dtype=np.float32),
                    "embed_n_tok": 5,
                }
            ]
        ),
        batch_size=1,
        pack_seq_len=8,
        max_steps=1,
        misfit_patience=4,
        embed_cap=4,
    )
    with pytest.raises(ValueError, match="embed_cap"):
        list(loader_overflow)

    # CORE-H14: _parquetra_tokenizers enforces padding=False & truncation_mode="keep_start" without mutating trainer
    tok = _SimpleVocabTokenizer()
    orig_transform = prompt_transforms.SFTPreprocessTransform(
        tok, max_length=32, mask_prompt=True, padding="max_length", truncation_mode="keep_end"
    )
    dummy_trainer = SimpleNamespace(
        _get_preprocess_transform=lambda: orig_transform,
        _get_processing_class=lambda: tok,
        processing_class=tok,
        arguments=SimpleNamespace(
            max_length=32,
            assistant_only_loss=True,
            dataset_text_field="text",
            truncation_mode="keep_end",
            parquetra_assistant_only={"ds_a": True},
            parquetra_fast_tokenizer=None,
            parquetra_fast_tokenizer_verify_rows=64,
        ),
        _assistant_only_by_source={"ds_a": True},
    )
    toks = data_lanes._parquetra_tokenizers(dummy_trainer, ["ds_a"])
    rebuilt = toks["ds_a"].transform
    assert isinstance(rebuilt, prompt_transforms.SFTPreprocessTransform)
    assert rebuilt._padding is False
    assert rebuilt._truncation_mode == "keep_start"
    # Original transform on trainer was NOT mutated
    assert orig_transform._padding == "max_length"
    assert orig_transform._truncation_mode == "keep_end"

    # CORE-H14: ParquetRowTokenize is thread-safe AND picklable
    global _FAST_FACTORY_INITS
    _FAST_FACTORY_INITS = 0

    sft_nopad = prompt_transforms.SFTPreprocessTransform(tok, max_length=32, padding=False)
    row_tok = data_lanes.ParquetRowTokenize(
        sft_nopad,
        hf_tokenizer=tok,
        fast_encoder_factory=_module_fast_encoder_factory,
        verify_rows=10,
    )
    # Must round-trip through pickle before and after initialization
    restored = pickle.loads(pickle.dumps(row_tok))
    sample_row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: restored(dict(sample_row)), range(20)))
    assert len(results) == 20
    assert _FAST_FACTORY_INITS == 1
    assert restored.fast_verified == 10
    # Still picklable after lock usage
    assert pickle.loads(pickle.dumps(restored)).fast_verified == 10

    # CORE-M6: empty input_ids and all-pad rows rejected
    with pytest.raises(ValueError, match="empty"):
        restored({"input_ids": []})
    with pytest.raises(ValueError, match="no non-pad tokens"):
        restored({"input_ids": [0, 0, 0], "attention_mask": [0, 0, 0]})

    # CORE-M6: arrayrecord_manifest_instructions preserves relative paths
    sub = tmp_path / "code"
    sub.mkdir()
    (sub / "_MANIFEST.tsv").write_text("code/s0.array_record\t512\t5\n", encoding="utf-8")
    (tmp_path / "_MANIFEST.tsv").write_text("s0.array_record\t256\t4\n", encoding="utf-8")

    prev_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        out_rel = data_lanes.arrayrecord_manifest_instructions("code/*.array_record", source_name="code")
        assert out_rel is not None
        assert out_rel[0][0].filename == "code/s0.array_record"
        out_bare = data_lanes.arrayrecord_manifest_instructions("*.array_record", source_name="bare")
        assert out_bare is not None
        assert out_bare[0][0].filename == "s0.array_record"
    finally:
        os.chdir(prev_cwd)

    # CORE-L5: parameter validation
    with pytest.raises(ValueError):
        data_lanes.PackedParquetLoader(
            lambda _: iter(()), batch_size=0, pack_seq_len=8, max_steps=1, misfit_patience=1
        )
    with pytest.raises(ValueError):
        data_lanes.PackedParquetLoader(
            lambda _: iter(()), batch_size=1, pack_seq_len=0, max_steps=1, misfit_patience=1
        )
    with pytest.raises(ValueError):
        data_lanes.PackedParquetLoader(
            lambda _: iter(()), batch_size=1, pack_seq_len=8, max_steps=-1, misfit_patience=1
        )
    with pytest.raises(ValueError):
        data_lanes.PackedParquetLoader(
            lambda _: iter(()), batch_size=1, pack_seq_len=8, max_steps=1, misfit_patience=0
        )
    with pytest.raises(ValueError):
        data_lanes.build_map_lane(
            [],
            [],
            {},
            None,
            batch_size=1,
            max_steps=1,
            num_epochs=None,
            shard_index=2,
            shard_count=2,
            read_threads=1,
            prefetch_buffer=1,
            lane="train",
        )
    with pytest.raises(ValueError):
        data_lanes.build_packed_parquet_stream_factory(
            {},
            {},
            {},
            None,
            shard_index=0,
            shard_count=0,
            seed=0,
            shuffle=False,
            rows_needed=1,
        )


# ---------------------------------------------------------------------------
# CORE-C5 & CORE-H15 & CORE-M6: process_encoder.py
# ---------------------------------------------------------------------------


def test_core_c5_compute_embedding_only_vlms_not_falsely_bound() -> None:
    from easydel.modules.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration, Gemma3Model
    from easydel.modules.llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel
    from easydel.modules.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration, Mistral3Model
    from easydel.modules.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaModel,
    )
    from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

    def _shell(outer_cls, base_cls, tower_attr="vision_tower"):
        outer = object.__new__(outer_cls)
        base = object.__new__(base_cls)
        object.__setattr__(base, tower_attr, object())
        object.__setattr__(outer, "_base_model_name", "model")
        object.__setattr__(outer, "model", base)
        return outer

    # Compute-embedding-only VLMs MUST return None (no false binding!)
    for outer_cls, base_cls in (
        (Mistral3ForConditionalGeneration, Mistral3Model),
        (Gemma3ForConditionalGeneration, Gemma3Model),
        (LlavaForConditionalGeneration, LlavaModel),
    ):
        shell = _shell(outer_cls, base_cls)
        assert process_encoder.find_vision_tower_attr(shell) == "base_model.vision_tower"
        assert callable(getattr(shell, "get_image_features", None))
        assert process_encoder.resolve_encoder_binding(shell) is None, f"{outer_cls.__name__} falsely bound"

    # Real forward-accepting VLMs MUST still bind cleanly
    pg = _shell(PaliGemmaForConditionalGeneration, PaliGemmaModel)
    pg_binding = process_encoder.resolve_encoder_binding(pg)
    assert pg_binding is not None and pg_binding.feature_kwarg == "image_features"

    q2 = _shell(Qwen2VLForConditionalGeneration, Qwen2VLModel, tower_attr="visual")
    q2_binding = process_encoder.resolve_encoder_binding(q2)
    assert q2_binding is not None and q2_binding.feature_kwarg == "image_embeds"

    # Strict outer wrapper without **kwargs or image_features MUST NOT bind even if base_model accepts it
    class _Base:
        vision_tower = object()

        def forward(self, input_ids=None, image_features=None):
            return input_ids

        def compute_embedding(self, input_ids=None, image_features=None):
            return input_ids

    class _StrictOuter:
        def __init__(self) -> None:
            self.base_model = _Base()

        def get_image_features(self, pixel_values):
            return pixel_values

        def forward(self, input_ids=None, pixel_values=None):
            return input_ids

    assert process_encoder.resolve_encoder_binding(_StrictOuter()) is None


def test_core_h15_m6_process_encoder_layout_presence_and_bucket_isolation() -> None:
    # CORE-H15: grid_describes_rows short-circuits to IMAGE_MAJOR even when pixel_values.shape[0] == B
    batch_grid = {
        "input_ids": np.zeros((4, 8), dtype=np.int32),
        "pixel_values": np.ones((4, 16), dtype=np.float32),
        "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
    }
    assert process_encoder.resolve_pixel_layout(batch_grid) is process_encoder.PixelLayout.IMAGE_MAJOR

    # CORE-H15: pops presence_key and strips dummy 0-image ROW_MAJOR pixels + companion keys
    class _FakeVLM:
        def __init__(self) -> None:
            self.vision_tower = object()
            self._multimodal_merge_feature = SimpleNamespace(image_token_id=99)
            self.calls = 0

        def get_image_features(self, pixel_values):
            self.calls += 1
            # Return [N_img, 4, 8]
            return np.ones((pixel_values.shape[0], 4, 8), dtype=np.float32)

        def forward(self, input_ids=None, image_features=None, image_features_valid_length=None):
            return input_ids

    model = _FakeVLM()
    cfg = process_encoder.ProcessEncoderConfig(
        enabled=True,
        row_bucket_multiple=2,
        presence_key="_vision_present",
    )
    proc = process_encoder.EncoderProcessor(model, cfg)

    zero_img_batch = {
        "input_ids": np.zeros((2, 8), dtype=np.int32),
        "pixel_values": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "pixel_attention_mask": np.zeros((2, 4, 4), dtype=np.int32),
        "_vision_present": np.array([False, False]),
    }
    out_zero = proc.process(zero_img_batch)
    assert "_vision_present" not in out_zero
    assert "pixel_values" not in out_zero
    assert "pixel_attention_mask" not in out_zero
    assert model.calls == 0

    # Fallback path (bucket excluded by only_buckets) also strips presence_key
    proc_restricted = process_encoder.EncoderProcessor(
        model,
        process_encoder.ProcessEncoderConfig(
            enabled=True,
            row_bucket_multiple=2,
            presence_key="_vision_present",
            only_buckets=[1],
        ),
    )
    out_skipped = proc_restricted.process(
        {
            "input_ids": np.full((2, 8), 99, dtype=np.int32),
            "pixel_values": np.ones((2, 3, 4, 4), dtype=np.float32),
            "_vision_present": np.array([True, True]),
        },
        bucket_index=0,
    )
    assert "_vision_present" not in out_skipped
    assert "pixel_values" in out_skipped

    # CORE-M6: per-bucket high-water feature target isolation
    big_ids = np.zeros((4, 8), dtype=np.int32)
    big_ids[:, :4] = 99  # 4 images * 4 tokens = 16 placeholder tokens
    big_batch = {
        "input_ids": big_ids,
        "pixel_values": np.ones((4, 3, 4, 4), dtype=np.float32),
        "_vision_present": np.array([True, True, True, True]),
    }
    small_ids = np.zeros((2, 8), dtype=np.int32)
    small_ids[0, :4] = 99  # 1 image * 4 tokens = 4 placeholder tokens
    small_batch = {
        "input_ids": small_ids,
        "pixel_values": np.ones((2, 3, 4, 4), dtype=np.float32),
        "_vision_present": np.array([True, False]),
    }
    out_b1 = proc.process(big_batch, bucket_index=1)
    out_b0 = proc.process(small_batch, bucket_index=0)
    assert "_vision_present" not in out_b1 and "_vision_present" not in out_b0
    assert "pixel_values" not in out_b1 and "pixel_values" not in out_b0
    # Bucket 1 grew to 4 images * 4 tokens = 16; Bucket 0 must stay isolated at 2 images (bucket_multiple=2) * 4 = 8
    assert out_b1["image_features"].shape[0] == 16
    assert out_b0["image_features"].shape[0] == 8


# ---------------------------------------------------------------------------
# CORE-H15 & CORE-M6 & CORE-L5: pose.py & randomized_yarn.py
# ---------------------------------------------------------------------------


def test_core_h15_m6_l5_pose_and_randomized_yarn() -> None:
    # CORE-H15: _draw_gap biased mode clamped to [0, max_gap]
    cfg_biased = pose.PoSEConfig(enabled=True, p_pose=1.0, target_max_pos=128, gap_mode="biased")

    class _EdgeRng:
        def random(self) -> float:
            return 1.0 - 1e-16

    assert pose._draw_gap(cfg_biased, _EdgeRng(), max_gap=5) == 5

    # CORE-H15: negative segment_ids and attention_mask == 0 excluded
    seg = np.array([[0, 0, 0, 1, 1, -1, -1]], dtype=np.int32)
    attn = np.array([[1, 1, 1, 1, 1, 0, 0]], dtype=np.int32)
    assert pose.valid_sequence_end(0, 7, None, None, None, -100, attention_mask=attn, segment_ids=seg) == 5
    seq_lens, seq_lens_num = pose.seq_lens_from_segment_ids(seg)
    assert seq_lens_num.tolist() == [2]
    assert seq_lens[0, :2].tolist() == [3, 2]
    pos = np.array([[0, 1, 2, 0, 1, 0, 0]], dtype=np.int32)
    ranges = pose.document_segment_ranges(0, 5, pos, None, None, None, None, attention_mask=attn, segment_ids=seg)
    assert ranges == [(0, 3), (3, 5)]

    # CORE-H15: 3-D mRoPE shared reset across T/H/W when segment_ids absent
    # Axis 0 resets at index 2 (e.g. spatial/temporal grid), while axes 1 and 2 are non-zero -> NOT a doc split!
    # All 3 axes reset to 0 at index 4 -> IS a doc split!
    mrope_pos = np.array(
        [
            [[0, 1, 0, 1, 0, 1]],
            [[0, 1, 2, 3, 0, 1]],
            [[0, 1, 2, 3, 0, 1]],
        ],
        dtype=np.int32,
    )
    _, meta = pose.extract_position_batch({"position_ids": mrope_pos}, feature="position_ids")
    assert meta["cu_seqlens_num"].tolist() == [3]
    assert meta["cu_seqlens"][0, :3].tolist() == [0, 4, 6]

    # CORE-M6: pose_rng avoids seed+step collisions and handles negative seeds
    r01 = pose.pose_rng(0, 1).integers(0, 1 << 30, size=4).tolist()
    r10 = pose.pose_rng(1, 0).integers(0, 1 << 30, size=4).tolist()
    assert r01 != r10
    assert len(pose.pose_rng(-7, -3).integers(0, 100, size=4)) == 4

    # CORE-L5: randomized_yarn_rng handles negative seed and step cleanly
    ry_vals = randomized_yarn.randomized_yarn_rng(-5, -2).integers(0, 100, size=4)
    assert len(ry_vals) == 4


# ---------------------------------------------------------------------------
# CORE-H15 & CORE-M6: packer.py
# ---------------------------------------------------------------------------


def test_core_h15_m6_packer_segment_ids_labels_and_exact_fit() -> None:
    # Exact fit: 3 + 1 (sep) + 4 = 8 == max_length -> must fit in ONE packed row (no off-by-one flush!)
    ds = Dataset.from_dict(
        {
            "input_ids": [[10, 11, 12], [20, 21, 22, 23]],
            "attention_mask": [[1, 1, 1], [1, 1, 1, 1]],
            "completion_mask": [[0, 1, 1], [1, 1, 1, 1]],
        }
    )
    packed = packer.pack_sequences(ds, max_length=8, pad_token_id=0, reset_position_ids=True)
    assert len(packed) == 1
    row = packed[0]
    assert row["input_ids"] == [10, 11, 12, 0, 20, 21, 22, 23]
    assert row["segment_ids"] == [0, 0, 0, -1, 1, 1, 1, 1]
    # Separator, first token of doc 1 (20), and completion_mask==0 token 10 are all masked to -100
    assert row["labels"] == [-100, 11, 12, -100, -100, 21, 22, 23]
    assert row["completion_mask"] == [0, 1, 1, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# CORE-H15 & CORE-M6: reward_protocol.py & _reward_sideband.py
# ---------------------------------------------------------------------------


class _EchoReward(reward_protocol.RewardProtocol):
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def compute(
        self,
        *,
        prompt: Any = None,
        completion: Any = None,
        weights: Any = None,
        **kwargs: Any,
    ) -> float:
        self.seen.append({"prompt": prompt, "completion": completion, "weights": weights})
        return float(weights) if weights is not None else 1.0


def test_core_h15_m6_reward_protocol_and_sideband() -> None:
    # CORE-H15: _index_batch_field slices numpy array columns and unwraps 0-d scalars
    indexed = reward_protocol.RewardProtocol._index_batch_field(
        {"weights": np.array([1.25, 2.5], dtype=np.float32), "vec": np.array([[1, 2], [3, 4]])},
        i=1,
        n=2,
    )
    assert isinstance(indexed["weights"], float) and indexed["weights"] == pytest.approx(2.5)
    assert np.array_equal(indexed["vec"], np.array([3, 4]))

    # CORE-M6: align_reward_sidechannels rejects completion_count < 1
    with pytest.raises(ValueError):
        reward_protocol.align_reward_sidechannels({"k": ["a"]}, prompt_count=1, completion_count=0)

    # CORE-M6: compute_batch empty n==0, prompt-major expansion, and mismatch error
    rew = _EchoReward()
    assert rew.compute_batch(prompts=["p1"], completions=[]) == []

    scores = rew.compute_batch(
        prompts=["p1", "p2"],
        completions=["c1a", "c1b", "c2a", "c2b"],
        weights=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert scores == [1.0, 2.0, 3.0, 4.0]
    assert [s["prompt"] for s in rew.seen] == ["p1", "p1", "p2", "p2"]

    with pytest.raises(ValueError, match="prompts"):
        rew.compute_batch(prompts=["p1", "p2"], completions=["c1", "c2", "c3"])

    # CORE-M6: text_of handles multimodal content lists and None content
    assert reward_protocol.text_of(
        [{"role": "assistant", "content": [{"type": "text", "text": "hello"}, {"type": "image"}, {"type": "text", "text": "world"}]}]
    ) == "hello\nworld"
    assert reward_protocol.text_of([{"role": "assistant", "content": None}]) == ""

    # CORE-M6: _reward_sideband validates sibling values recursively
    assert _reward_sideband.is_reward_sideband({"answer": "42", "score": 1.0, "nested": [True, None]})
    assert not _reward_sideband.is_reward_sideband({"answer": "42", "bad": object()})
    assert not _reward_sideband.is_reward_sideband(["ok", object()])

# =============================================================================
# Domain D: Standard Trainers (Ray, SFT, Embedding, Reward, PRM, Merge, BEMA)
# =============================================================================

class _SimpleTokenizer:
    """Minimal real tokenizer for CPU/TPU unit tests without network I/O."""

    def __init__(self, *, bos_token_id: int | None = 1, eos_token_id: int = 2, pad_token_id: int | None = 0):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eos_token = "</s>"
        self.pad_token_id = pad_token_id
        self.pad_token = "<pad>" if pad_token_id is not None else None
        self.chat_template = None
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self) -> int:
        return 64

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        if text == "\n":
            return [3]
        return [10 + (ord(ch) % 40) for ch in text]

    def _encode_one(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int | None,
        padding: bool | str,
        add_special_tokens: bool,
    ) -> tuple[list[int], list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None and len(ids) < max_length:
            pad_id = 0 if self.pad_token_id is None else int(self.pad_token_id)
            pad_len = max_length - len(ids)
            if self.padding_side == "left":
                ids = [pad_id] * pad_len + ids
                mask = [0] * pad_len + mask
            else:
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
        return ids, mask

    def __call__(
        self,
        text: str | list[str],
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict[str, list]:
        del kwargs
        if isinstance(text, (list, tuple)):
            pairs = [
                self._encode_one(
                    item,
                    truncation=truncation,
                    max_length=max_length,
                    padding=padding,
                    add_special_tokens=add_special_tokens,
                )
                for item in text
            ]
            out: dict[str, list] = {"input_ids": [p[0] for p in pairs]}
            if return_attention_mask:
                out["attention_mask"] = [p[1] for p in pairs]
            return out
        ids, mask = self._encode_one(
            text,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = mask
        return out


class _ListSource(ShardedDataSource):
    """Lightweight ShardedDataSource wrapping an in-memory row list."""

    def __init__(self, rows: list[dict]):
        self._rows = list(rows)

    @property
    def shard_names(self) -> list[str]:
        return ["shard_0"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str):
        del shard_name
        yield from self._rows

    def __len__(self) -> int:
        return len(self._rows)


def _tiny_llama_causal_lm(*, attn_mechanism: str = "sdpa", vocab_size: int = 64) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        attn_mechanism=attn_mechanism,
        gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
    )
    config.attn_dtype = jnp.float32
    return LlamaForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()


def _tiny_llama_reward_model(*, attn_mechanism: str = "vanilla", attention_dropout: float = 0.0):
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        attn_mechanism=attn_mechanism,
        attention_dropout=attention_dropout,
        gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
    )
    config.num_labels = 1
    config.attn_dtype = jnp.float32
    return LlamaForSequenceClassification(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()


def _tiny_qwen2_embedding_model() -> Qwen2ForEmbedding:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        attn_mechanism="vanilla",
        gradient_checkpointing=EasyDeLGradientCheckPointers.NONE,
    )
    config.attn_dtype = jnp.float32
    return Qwen2ForEmbedding(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    ).shard_model()


def _base_train_kwargs(tmp_path) -> dict:
    return dict(
        save_directory=str(tmp_path),
        max_training_steps=1,
        num_train_epochs=1,
        total_batch_size=2,
        eval_batch_size=2,
        use_wandb=False,
        do_last_save=False,
        save_steps=None,
        save_optimizer_state=False,
        weight_distribution_log_steps=0,
        track_memory=False,
        shuffle_train_dataset=False,
        progress_bar_type="json",
    )


# =============================================================================
# 1. CORE-C10 + CORE-H16 + CORE-L5: RayScaler
# =============================================================================


def test_ray_config_save_load_roundtrip_preserves_types_and_never_mutates_in_place(tmp_path):
    custom_pa = PartitionAxis(data_parallel_axis="dp", batch_axis=("fsdp", "dp"))
    launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny-llama",
        bucket_path="gs://my-bucket/exp/run-100",
        model_class=LlamaForCausalLM,
        offload_backend="cpu",
        config_scaling_variables={"hidden_size": 128},
        config_variables={
            "dtype": jnp.bfloat16,
            "param_dtype": jnp.float32,
            "precision": lax.Precision.HIGHEST,
            "partition_axis": custom_pa,
            "gradient_checkpointing": EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            "sharding_axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
            "sharding_axis_dims": (1, 1, -1, 1, 1, 1),
            "sharding_dcn_axis_dims": (1, 1, 1, 1, 1, 1),
        },
    )

    cfg_path = tmp_path / "ray_dist_config.json"
    launcher.save_config(cfg_path)

    # Original launcher config_variables must never be mutated in-place.
    assert isinstance(launcher.config_variables["partition_axis"], PartitionAxis)
    assert launcher.config_variables["dtype"] is jnp.bfloat16
    assert launcher.config_variables["param_dtype"] is jnp.float32
    assert launcher.config_variables["precision"] is lax.Precision.HIGHEST

    # Saved JSON must round-trip bucket_path, PartitionAxis, Precision, dtypes, and tuples cleanly.
    loaded = RayDistributedTrainer.from_config(cfg_path)
    assert loaded.pretrained_model_name_or_path == "org/tiny-llama"
    assert loaded.bucket_path == "gs://my-bucket/exp/run-100"
    assert loaded.model_type == launcher.model_type
    assert loaded.model_task == launcher.model_task
    assert loaded.config_variables["partition_axis"] == custom_pa
    assert isinstance(loaded.config_variables["partition_axis"].batch_axis, tuple)
    assert loaded.config_variables["precision"] is lax.Precision.HIGHEST
    assert loaded.config_variables["dtype"] is jnp.bfloat16
    assert loaded.config_variables["param_dtype"] is jnp.float32
    assert loaded.config_variables["gradient_checkpointing"] is EasyDeLGradientCheckPointers.NOTHING_SAVEABLE
    assert loaded.config_variables["sharding_axis_names"] == ("pp", "dp", "fsdp", "ep", "tp", "sp")
    assert isinstance(loaded.config_variables["sharding_axis_names"], tuple)
    assert loaded.config_variables["sharding_axis_dims"] == (1, 1, -1, 1, 1, 1)
    assert isinstance(loaded.config_variables["sharding_axis_dims"], tuple)
    assert loaded.config_variables["sharding_dcn_axis_dims"] == (1, 1, 1, 1, 1, 1)
    assert isinstance(loaded.config_variables["sharding_dcn_axis_dims"], tuple)

    # Direct RayDistributedConfig._saving_preprocess must not mutate caller dict either,
    # and must support jnp.dtype objects and 2-tuple lax.Precision.
    caller_vars = {
        "dtype": jnp.dtype("bfloat16"),
        "precision": (lax.Precision.HIGH, lax.Precision.HIGHEST),
        "partition_axis": PartitionAxis(),
    }
    raw_cfg = RayDistributedConfig(
        pretrained_model_name_or_path="org/model",
        bucket_path="gs://bucket/p",
        config_variables=caller_vars,
    )
    raw_cfg._saving_preprocess()
    assert isinstance(caller_vars["partition_axis"], PartitionAxis)
    assert caller_vars["precision"] == (lax.Precision.HIGH, lax.Precision.HIGHEST)
    reloaded_cfg = RayDistributedConfig(**json.loads(raw_cfg.model_dump_json()))
    reloaded_cfg._loading_postprocess()
    assert reloaded_cfg.bucket_path == "gs://bucket/p"
    assert reloaded_cfg.config_variables["dtype"] is jnp.bfloat16
    assert reloaded_cfg.config_variables["precision"] == (lax.Precision.HIGH, lax.Precision.HIGHEST)
    assert reloaded_cfg.config_variables["partition_axis"] == PartitionAxis()


def test_ray_create_trainer_maps_kwargs_across_trainer_subclasses(tmp_path):
    tok = _SimpleTokenizer()
    lm_state = _tiny_llama_causal_lm().to_state()
    sft_ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1]] * 2})

    # Base Trainer: dataset_train / dataset_eval / model_state
    base_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=LlamaForCausalLM,
        trainer_module=Trainer,
    )
    base_trainer = base_launcher.create_trainer(
        arguments=TrainingArguments(max_length=4, **_base_train_kwargs(tmp_path / "base")),
        dataset_train=sft_ds,
        dataset_eval=sft_ds,
        state=lm_state,
    )
    assert isinstance(base_trainer, Trainer)

    # SFTTrainer: train_dataset / eval_dataset / model / processing_class (via cached processor)
    sft_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=LlamaForCausalLM,
        trainer_module=SFTTrainer,
    )
    sft_launcher.processor = tok
    sft_trainer = sft_launcher.create_trainer(
        arguments=SFTConfig(max_length=4, **_base_train_kwargs(tmp_path / "sft")),
        dataset_train=sft_ds,
        dataset_eval=sft_ds,
        state=lm_state,
    )
    assert isinstance(sft_trainer, SFTTrainer)
    assert sft_trainer.processing_class is tok

    # RewardTrainer: train_dataset / eval_dataset / model / processing_class
    reward_state = _tiny_llama_reward_model().to_state()
    reward_ds = Dataset.from_dict(
        {
            "input_ids_chosen": [[1, 2, 3, 4], [2, 3, 4, 5]],
            "attention_mask_chosen": [[1, 1, 1, 1]] * 2,
            "input_ids_rejected": [[1, 2, 0, 0], [2, 3, 0, 0]],
            "attention_mask_rejected": [[1, 1, 0, 0]] * 2,
        }
    )
    reward_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=LlamaForSequenceClassification,
        trainer_module=RewardTrainer,
    )
    reward_trainer = reward_launcher.create_trainer(
        arguments=RewardConfig(max_length=4, **_base_train_kwargs(tmp_path / "reward")),
        dataset_train=reward_ds,
        dataset_eval=reward_ds,
        state=reward_state,
        processing_class=tok,
    )
    assert isinstance(reward_trainer, RewardTrainer)

    # EmbeddingTrainer: train_dataset / eval_dataset / model / processing_class
    embed_state = _tiny_qwen2_embedding_model().to_state()
    embed_ds = Dataset.from_dict(
        {
            "query_input_ids": [[1, 2], [3, 4]],
            "query_attention_mask": [[1, 1]] * 2,
            "positive_input_ids": [[1, 2], [3, 4]],
            "positive_attention_mask": [[1, 1]] * 2,
        }
    )
    embed_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=Qwen2ForEmbedding,
        trainer_module=EmbeddingTrainer,
    )
    embed_trainer = embed_launcher.create_trainer(
        arguments=EmbeddingConfig(max_length=4, **_base_train_kwargs(tmp_path / "embed")),
        dataset_train=embed_ds,
        dataset_eval=embed_ds,
        state=embed_state,
        processing_class=tok,
    )
    assert isinstance(embed_trainer, EmbeddingTrainer)

    # DPOTrainer and BEMADPOTrainer (MRO **kwargs wrapper)
    dpo_ds = Dataset.from_dict(
        {
            "prompt": ["hello", "world"],
            "chosen": ["good", "nice"],
            "rejected": ["bad", "poor"],
        }
    )
    for cls in (DPOTrainer, BEMADPOTrainer):
        dpo_launcher = RayDistributedTrainer(
            pretrained_model_name_or_path="org/tiny",
            model_class=LlamaForCausalLM,
            trainer_module=cls,
        )
        dpo_launcher.processor = tok
        inst = dpo_launcher.create_trainer(
            arguments=DPOConfig(
                max_length=8,
                max_prompt_length=4,
                max_completion_length=4,
                **_base_train_kwargs(tmp_path / cls.__name__),
            ),
            dataset_train=dpo_ds,
            state=lm_state,
        )
        assert isinstance(inst, cls)

    # PRMTrainer signature mapping check (recording kwargs on subclass)
    captured_prm = {}

    class _RecordingPRMTrainer(PRMTrainer):
        def __init__(self, arguments, model, processing_class, train_dataset=None, eval_dataset=None, data_collator=None):
            captured_prm.update(
                arguments=arguments,
                model=model,
                processing_class=processing_class,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

    prm_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=LlamaForCausalLM,
        trainer_module=_RecordingPRMTrainer,
    )
    prm_launcher.processor = tok
    prm_args = PRMConfig(max_length=8, **_base_train_kwargs(tmp_path / "prm"))
    prm_launcher.create_trainer(arguments=prm_args, dataset_train=sft_ds, dataset_eval=sft_ds, state=lm_state)
    assert captured_prm["train_dataset"] is sft_ds
    assert captured_prm["eval_dataset"] is sft_ds
    assert captured_prm["model"] is lm_state
    assert captured_prm["processing_class"] is tok

    # MRO walk resets accepts_var_kw when terminating at a fixed-signature parent lacking data_collator
    from easydel.trainers.grpo_replay_buffer_trainer import GRPOWithReplayBufferTrainer

    replay_params, replay_var_kw = ray_dist_mod._resolve_trainer_init_params(GRPOWithReplayBufferTrainer)
    assert "data_collator" not in replay_params
    assert replay_var_kw is False

    class _NoCollatorParent:
        def __init__(self, arguments, model, processing_class, train_dataset=None, eval_dataset=None):
            self.captured = (arguments, model, processing_class, train_dataset, eval_dataset)

    class _VarKwWrapperOverFixedParent(_NoCollatorParent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    wrap_launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        model_class=LlamaForCausalLM,
        trainer_module=_VarKwWrapperOverFixedParent,
    )
    wrap_launcher.processor = tok
    wrapped_tr = wrap_launcher.create_trainer(
        arguments=prm_args,
        dataset_train=sft_ds,
        dataset_eval=sft_ds,
        state=lm_state,
    )
    assert wrapped_tr.captured[1] is lm_state


def test_ray_convert_model_to_state_before_train_load_processor_load_state_task_and_stage_leaf(tmp_path):
    launcher = RayDistributedTrainer(
        pretrained_model_name_or_path="org/tiny",
        bucket_path="gs://bucket/ckpt",
        model_class=LlamaForCausalLM,
        offload_backend="cpu",
    )

    # 1. convert_model_to_state works before train() sets self.arguments
    model = _tiny_llama_causal_lm()
    state = launcher.convert_model_to_state(model)
    assert isinstance(state, ed.EasyDeLState)

    # 2. load_processor sets BOTH pad_token and pad_token_id
    class _BareTokenizer:
        eos_token = "<eos>"
        eos_token_id = 7
        pad_token = None
        pad_token_id = None

    class _FakeLoader:
        @classmethod
        def from_pretrained(cls, path):
            del path
            return _BareTokenizer()

    launcher._processor_loader_class = _FakeLoader
    loaded_tok = launcher.load_processor()
    assert loaded_tok.pad_token == "<eos>"
    assert loaded_tok.pad_token_id == 7

    # 3. load_state receives model_task and stage_leaf guards ShapeDtypeStruct + non-numeric leaves
    captured_load_kwargs = {}
    captured_inner_train = []

    class _DummyStateClass:
        @classmethod
        def load_state(cls, **kwargs):
            captured_load_kwargs.update(kwargs)
            return {
                "arr": jnp.ones((2,), dtype=jnp.float32),
                "abstract": jax.ShapeDtypeStruct((2,), jnp.float32),
                "meta": "non-array-leaf",
                "obj_arr": np.array(["a", "b"], dtype=object),
                "step": 0,
            }

    class _DummyInnerTrainer:
        def __init__(self, arguments, dataset_train, dataset_eval=None, data_collator=None, model_state=None):
            del arguments, dataset_train, dataset_eval, data_collator
            self.model_state = model_state

        def train(self):
            captured_inner_train.append(self.model_state)
            return "ok"

    launcher.state_class = _DummyStateClass
    launcher.trainer_module = _DummyInnerTrainer
    res = launcher.train(
        scaling_index=1,
        arguments=TrainingArguments(**_base_train_kwargs(tmp_path)),
        dataset_train=[],
    )
    assert res == "ok"
    assert captured_load_kwargs["model_task"] == launcher.model_task
    staged = captured_inner_train[0]
    assert isinstance(staged["abstract"], jax.ShapeDtypeStruct)
    assert staged["meta"] == "non-array-leaf"
    assert staged["obj_arr"].dtype == object
    assert isinstance(staged["arr"], jax.Array)

    # _stage_leaf preserves non-numeric np.generic scalars & datetime64/timedelta64 arrays while staging ml_dtypes.bfloat16
    import ml_dtypes

    cpu_dev = jax.devices("cpu")[0]
    for non_stageable in (
        np.str_("hello"),
        np.bytes_(b"hi"),
        np.void(b"ab"),
        np.datetime64("2020-01-01"),
        np.timedelta64(1, "s"),
        np.array([np.datetime64("2020-01-01")]),
        np.array([np.timedelta64(1, "s")]),
    ):
        out_leaf = RayDistributedTrainer._stage_leaf(non_stageable, cpu_dev)
        assert out_leaf is non_stageable

    bf16_scalar = RayDistributedTrainer._stage_leaf(ml_dtypes.bfloat16(1.5), cpu_dev)
    assert isinstance(bf16_scalar, jax.Array) and bf16_scalar.dtype == jnp.bfloat16
    bf16_arr = RayDistributedTrainer._stage_leaf(np.array([1.5], dtype=jnp.bfloat16), cpu_dev)
    assert isinstance(bf16_arr, jax.Array) and bf16_arr.dtype == jnp.bfloat16


@pytest.mark.cpu_ok
def test_ray_scaler_docstrings_do_not_claim_nonexistent_ray_train_or_tune():
    for doc in (ray_scaler_pkg.__doc__ or "", ray_dist_mod.__doc__ or ""):
        assert "Ray Train" not in doc
        assert "Ray Tune" not in doc
        assert "num_workers=4" not in doc


# =============================================================================
# 2. CORE-H16 + CORE-M7: SFTConfig & SFTTrainer
# =============================================================================


def test_sft_eval_state_shardings_ste_and_bucket_step_args(tmp_path):
    model = _tiny_llama_causal_lm(attn_mechanism="sdpa")
    tok = _SimpleTokenizer()
    batch_size = max(4, len(jax.devices()))
    rows_long = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]] * (batch_size // 2),
            "attention_mask": [[1] * 8] * batch_size,
            "labels": [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]] * (batch_size // 2),
        }
    )
    rows_short = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [2, 3, 4, 5]] * (batch_size // 2),
            "attention_mask": [[1] * 4] * batch_size,
            "labels": [[1, 2, 3, 4], [2, 3, 4, 5]] * (batch_size // 2),
        }
    )

    ste_calls = []

    def ste_fn(tree):
        ste_calls.append(True)
        return tree

    buckets = [
        TrainingBucket(name="long", max_length=8, config=None, total_batch_size=batch_size),
        TrainingBucket(name="short", max_length=4, config={"attn_mechanism": "vanilla"}, total_batch_size=batch_size),
    ]
    base_kw = _base_train_kwargs(tmp_path)
    base_kw["total_batch_size"] = batch_size
    base_kw["eval_batch_size"] = batch_size
    args = SFTConfig(
        max_length=8,
        loss_type="dft",
        straight_through_emulator=ste_fn,
        eval_config_overrides={"attn_mechanism": "vanilla"},
        buckets=buckets,
        bucket_rule=ModBucketRule(mod=2, on_bucket=1, off_bucket=0),
        bucket_datasets=[rows_long, rows_short],
        **base_kw,
    )

    trainer = SFTTrainer(
        arguments=args,
        processing_class=tok,
        model=model,
        train_dataset=rows_long,
        eval_dataset=rows_long,
    )

    # 1. STE stashed and preserved in both primary and bucket static args alongside loss_type="dft".
    assert trainer._straight_through_emulator is ste_fn
    assert trainer._train_shared_fn_static_args[-2] is ste_fn
    assert trainer._train_shared_fn_static_args[-1] == "dft"
    assert trainer._bucket_static_argnums == (2, 3, 4, 5, 6, 7)
    _, bucket_static = trainer._resolve_bucket_step_args(1)
    assert len(bucket_static) == 6
    assert bucket_static[-2] is ste_fn
    assert bucket_static[-1] == "dft"

    # 2. Eval compiles with eval_state_shardings and runs with eval_config_overrides active.
    assert trainer._eval_graphdef is not None
    assert trainer.eval_state_shardings.graphdef is trainer._eval_graphdef
    eval_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    assert eval_state.model.training is False
    eval_batch = {
        "input_ids": jnp.asarray(rows_long["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.asarray(rows_long["attention_mask"], dtype=jnp.int32),
        "labels": jnp.asarray(rows_long["labels"], dtype=jnp.int32),
    }
    eval_metrics = trainer.sharded_evaluation_step_function(
        eval_state, eval_batch, *trainer._eval_shared_fn_extra_args, *trainer._eval_shared_fn_static_args
    )
    assert jnp.isfinite(eval_metrics.loss)

    # 3. Bucket 1 step function runs with its 6 static args (including STE and loss_type="dft").
    short_batch = {
        "input_ids": jnp.asarray(rows_short["input_ids"], dtype=jnp.int32),
        "attention_mask": jnp.asarray(rows_short["attention_mask"], dtype=jnp.int32),
        "labels": jnp.asarray(rows_short["labels"], dtype=jnp.int32),
    }
    bucket_state = trainer.model_state.replace(graphdef=trainer._bucket_graphdefs[1])
    new_state, bucket_metrics = trainer._bucket_step_fns[1](
        bucket_state, short_batch, *trainer._bucket_extra_args[1], *trainer._bucket_static_args[1]
    )
    assert jnp.isfinite(bucket_metrics.loss)
    assert int(jax.device_get(new_state.step)) == 1
    assert ste_calls


@pytest.mark.parametrize(
    ("packing", "sequence_packing"),
    [(False, True), (True, False), (True, True)],
)
def test_sft_eval_packing_false_honored_and_eval_rows_padded(tmp_path, packing, sequence_packing):
    tok = _SimpleTokenizer()
    trainer = object.__new__(SFTTrainer)
    trainer.arguments = SFTConfig(
        save_directory=str(tmp_path),
        max_length=12,
        packing=packing,
        sequence_packing=sequence_packing,
        eval_packing=False,
        resume_if_possible=False,
    )
    trainer.processing_class = tok
    trainer._formatting_func = None
    trainer._dataset_text_field = "text"
    trainer._train_source = _ListSource([{"text": "ab"}, {"text": "cde"}])
    trainer._eval_source = _ListSource([{"text": "ab"}, {"text": "cdefg"}])

    trainer._apply_preprocess_transforms()

    # Train is packed; eval is NOT packed and every eval row is padded to max_length=12.
    assert isinstance(trainer._train_source, PackedShardedSource)
    assert not isinstance(trainer._eval_source, PackedShardedSource)
    eval_rows = list(trainer._eval_source.open_shard(trainer._eval_source.shard_names[0]))
    assert len(eval_rows) == 2
    assert all(len(row["input_ids"]) == 12 for row in eval_rows)
    assert all(len(row["attention_mask"]) == 12 for row in eval_rows)
    assert eval_rows[0]["attention_mask"][-1] == 0


@pytest.mark.cpu_ok
def test_sft_config_does_not_mutate_caller_loss_config_and_validates_fields():
    caller_lc = LossConfig(z_loss=1e-4, chunk_token_size=None)
    cfg = SFTConfig(loss_type="chunked_nll", completion_loss_chunk_size=128, loss_config=caller_lc)
    assert caller_lc.chunk_token_size is None
    assert cfg.loss_config is not caller_lc
    assert cfg.loss_config.chunk_token_size == 128
    assert cfg.loss_config.z_loss == 1e-4

    with pytest.raises(ValueError, match="completion_loss_chunk_size"):
        SFTConfig(completion_loss_chunk_size=0)
    with pytest.raises(ValueError, match="completion_loss_chunk_size"):
        SFTConfig(completion_loss_chunk_size=-8)
    with pytest.raises(ValueError, match="packing_strategy"):
        SFTConfig(packing_strategy="nonexistent_strategy")
    assert SFTConfig(packing_strategy="bfd_split").packing_strategy == "bfd"
    assert SFTConfig(packing_strategy="wrapped").packing_strategy == "wrapped"
    assert SFTConfig(packing_strategy="pool").packing_strategy == "pool"

    with pytest.raises(ValueError, match="model"):
        SFTTrainer(arguments=SFTConfig(), processing_class=_SimpleTokenizer(), model=None)


# =============================================================================
# 3. CORE-H16 + CORE-M7: EmbeddingTrainer
# =============================================================================


def test_embedding_pretokenized_negatives_preserved_and_triplet_nonzero(tmp_path):
    model = _tiny_qwen2_embedding_model()
    tok = _SimpleTokenizer()
    rows = {
        "query_input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "query_attention_mask": [[1, 1, 1, 1]] * 2,
        "positive_input_ids": [[1, 2, 3, 9], [5, 6, 7, 10]],
        "positive_attention_mask": [[1, 1, 1, 1]] * 2,
        "negative_input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]],
        "negative_attention_mask": [[1, 1, 1, 1]] * 2,
    }
    ds = Dataset.from_dict(rows)
    args = EmbeddingConfig(
        loss_type="triplet",
        negative_field=None,
        margin=0.5,
        max_length=4,
        **_base_train_kwargs(tmp_path),
    )
    trainer = EmbeddingTrainer(
        arguments=args,
        model=model,
        train_dataset=ds,
        processing_class=tok,
    )

    grain_collator = trainer.create_grain_collect_function(max_sequence_length=4)
    tfds_collator = trainer.create_tfds_collect_function(max_sequence_length=4)
    assert grain_collator.has_negatives is True
    assert tfds_collator.has_negatives is True

    collated = tfds_collator([ds[0], ds[1]])
    assert "negative_input_ids" in collated
    assert "negative_attention_mask" in collated

    # Negative equals query, positive differs -> pos_dist > neg_dist -> strictly positive loss!
    _, metrics = trainer.sharded_training_step_function(trainer.model_state, collated)
    assert float(metrics.loss) > 0.0


def test_embedding_l2_norm_zero_grad_fp16_stability_and_infonce_fp32_and_triplet_l2():
    # 1. Safe L2 norm: no NaN gradient at exact zero
    q_zero = jnp.zeros((2, 4), dtype=jnp.float32)
    p_unit = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    grad_q = jax.grad(lambda q: infonce_loss(_l2_normalize(q), p_unit, temperature=0.05)[0])(q_zero)
    assert bool(jnp.all(jnp.isfinite(grad_q)))

    # 2. Safe L2 norm in fp16: no overflow on large values (500^2 > 65504) and no underflow on 1e-4
    x_fp16 = jnp.array([[500.0, 500.0], [1e-4, 0.0], [0.0, 0.0]], dtype=jnp.float16)
    normed = _l2_normalize(x_fp16)
    assert bool(jnp.all(jnp.isfinite(normed)))
    np.testing.assert_allclose(np.asarray(normed[0]), [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(normed[1]), [1.0, 0.0], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.asarray(normed[2]), [0.0, 0.0], atol=1e-6)

    # 3. InfoNCE logits computed in fp32 (no bf16/fp16 rounding collapse at small temperature)
    q_bf16 = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.bfloat16)
    p_bf16 = jnp.array([[0.99, 0.14], [0.14, 0.99]], dtype=jnp.bfloat16)
    loss_bf16, _ = infonce_loss(q_bf16, p_bf16, temperature=0.01)
    loss_fp32, _ = infonce_loss(q_bf16.astype(jnp.float32), p_bf16.astype(jnp.float32), temperature=0.01)
    assert loss_bf16.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(loss_bf16), np.asarray(loss_fp32), rtol=1e-5, atol=1e-5)

    # 4. Triplet loss uses Euclidean L2 distance (matching docstring sqrt(2 - 2*cos)) with finite zero-diff grad
    q = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    p = jnp.array([[0.6, 0.8]], dtype=jnp.float32)
    n = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    t_loss, _ = triplet_loss(q, p, n, margin=0.6)
    expected = max(0.0, np.sqrt(0.8) - np.sqrt(2.0) + 0.6)
    np.testing.assert_allclose(np.asarray(t_loss), expected, rtol=1e-5, atol=1e-5)
    grad_same = jax.grad(lambda a: triplet_loss(a, a, n, margin=2.0)[0])(q)
    assert bool(jnp.all(jnp.isfinite(grad_same)))


def test_embedding_matryoshka_normalize_false_empty_guard_ste_and_loss_config(tmp_path):
    q = jnp.array([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype=jnp.float32)
    p = jnp.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]], dtype=jnp.float32)
    loss_unnorm, _ = matryoshka_loss(infonce_loss, q, p, None, [2, 4], normalize=False, temperature=1.0)
    loss_norm, _ = matryoshka_loss(infonce_loss, q, p, None, [2, 4], normalize=True, temperature=1.0)
    expected_unnorm, _ = infonce_loss(q, p, temperature=1.0)
    np.testing.assert_allclose(np.asarray(loss_unnorm), np.asarray(expected_unnorm), rtol=1e-6)
    assert float(loss_unnorm) < float(loss_norm)

    with pytest.raises(ValueError, match="matryoshka_dims"):
        matryoshka_loss(infonce_loss, q, p, None, [])
    with pytest.raises(ValueError, match="matryoshka_dims"):
        EmbeddingConfig(matryoshka_dims=[])
    with pytest.raises(ValueError, match="matryoshka_dims"):
        EmbeddingConfig(matryoshka_dims=[0, 8])

    ste_called = []

    def ste_fn(tree):
        ste_called.append(True)
        return tree

    model = _tiny_qwen2_embedding_model()
    state = model.to_state()
    state = state.init_tx(ed.TrainingArguments(learning_rate=1e-3).get_optimizer_and_scheduler(steps=4)[0])
    batch = {
        "query_input_ids": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "query_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "positive_input_ids": jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        "positive_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
    }
    lc = LossConfig(break_on_nan=False)
    new_state, step_metrics = embedding_training_step(
        state=state,
        batch=batch,
        loss_config=lc,
        straight_through_emulator=ste_fn,
    )
    assert ste_called
    assert jnp.isfinite(step_metrics.loss)
    assert int(jax.device_get(new_state.step)) == 1

    # Processor tokenizer unwrapping + pad_token=eos_token fallback + pooling pad_token_id preservation
    inner_tok = _SimpleTokenizer(pad_token_id=None, eos_token_id=5)
    processor = SimpleNamespace(tokenizer=inner_tok)

    class _DummyPool:
        def __init__(self, *, strategy, pad_token_id):
            self.strategy = strategy
            self.pad_token_id = pad_token_id

    dummy_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None),
        _pooling_feature=_DummyPool(strategy="last", pad_token_id=9),
    )
    EmbeddingTrainer._apply_pooling_strategy(dummy_model, "mean")
    assert dummy_model._pooling_feature.pad_token_id == 9

    dummy_model_no_pad = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None),
        _pooling_feature=_DummyPool(strategy="mean", pad_token_id=None),
    )
    EmbeddingTrainer._apply_pooling_strategy(dummy_model_no_pad, "mean", pad_token_id=5)
    assert dummy_model_no_pad._pooling_feature.pad_token_id == 5

    emb_ds = Dataset.from_dict(
        {
            "query_input_ids": [[1, 2], [3, 4]],
            "query_attention_mask": [[1, 1]] * 2,
            "positive_input_ids": [[1, 2], [3, 4]],
            "positive_attention_mask": [[1, 1]] * 2,
        }
    )
    emb_trainer = EmbeddingTrainer(
        arguments=EmbeddingConfig(
            max_length=4,
            loss_config=lc,
            straight_through_emulator=ste_fn,
            **_base_train_kwargs(tmp_path),
        ),
        model=_tiny_qwen2_embedding_model(),
        train_dataset=emb_ds,
        processing_class=processor,
    )
    assert inner_tok.pad_token == "</s>"
    assert inner_tok.pad_token_id == 5
    assert emb_trainer.padding_value == 5
    assert emb_trainer._straight_through_emulator is ste_fn


# =============================================================================
# 4. CORE-H16 + CORE-M7 + CORE-L5: RewardTrainer
# =============================================================================


def test_reward_margin_2d_no_broadcast_and_scheduled_uneven_lengths():
    r_chosen = jnp.array([2.0, 0.5, 1.5], dtype=jnp.float32)
    r_rejected = jnp.array([0.0, 0.25, 0.5], dtype=jnp.float32)
    margin_1d = jnp.array([1.0, -0.5, 0.25], dtype=jnp.float32)
    margin_2d = margin_1d[:, None]

    loss_1d = _reward_pair_loss(r_chosen, r_rejected, margin_1d, center_rewards_coefficient=0.05)
    loss_2d = _reward_pair_loss(r_chosen, r_rejected, margin_2d, center_rewards_coefficient=0.05)
    np.testing.assert_allclose(np.asarray(loss_2d), np.asarray(loss_1d), rtol=1e-6, atol=1e-6)

    # Scheduled loss handles uneven chosen/rejected sequence lengths cleanly and matches separate forwards
    model = _tiny_llama_reward_model()
    state = model.to_state()
    uneven_batch = {
        "input_ids_chosen": jnp.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=jnp.int32),
        "attention_mask_chosen": jnp.ones((2, 5), dtype=jnp.int32),
        "input_ids_rejected": jnp.asarray([[1, 2, 3], [2, 3, 4]], dtype=jnp.int32),
        "attention_mask_rejected": jnp.ones((2, 3), dtype=jnp.int32),
        "margin": jnp.asarray([[0.2], [0.4]], dtype=jnp.float32),
    }

    class _ScheduledCallStub:
        def __init__(self, st):
            self.state = st
            self.schedule = None

        def get(self, name, default=None):
            return {
                "partition_spec": None,
                "center_rewards_coefficient": 0.1,
                "straight_through_emulator": None,
            }.get(name, default)

    scheduled_loss_fn = _make_reward_scheduled_loss(_ScheduledCallStub(state))
    sched_loss = scheduled_loss_fn(state.graphstate, uneven_batch)

    eval_metrics = reward_evaluation_step(state, uneven_batch, center_rewards_coefficient=0.1)
    assert jnp.isfinite(sched_loss)
    np.testing.assert_allclose(np.asarray(sched_loss), np.asarray(eval_metrics.loss), rtol=1e-4, atol=1e-5)


def test_reward_eval_state_shardings_eval_mode_and_grain_max_length_none(tmp_path):
    tok = _SimpleTokenizer()
    batch_size = max(4, len(jax.devices()))
    ds = Dataset.from_dict(
        {
            "input_ids_chosen": [[1, 2, 3, 4], [2, 3, 4, 5]] * (batch_size // 2),
            "attention_mask_chosen": [[1, 1, 1, 1]] * batch_size,
            "input_ids_rejected": [[1, 2, 0, 0], [2, 3, 0, 0]] * (batch_size // 2),
            "attention_mask_rejected": [[1, 1, 0, 0]] * batch_size,
        }
    )
    model = _tiny_llama_reward_model(attn_mechanism="vanilla", attention_dropout=0.2)
    base_kw = _base_train_kwargs(tmp_path)
    base_kw["total_batch_size"] = batch_size
    base_kw["eval_batch_size"] = batch_size
    args = RewardConfig(
        max_length=4,
        disable_dropout=False,
        eval_config_overrides={"attn_mechanism": "sdpa"},
        **base_kw,
    )
    trainer = RewardTrainer(
        arguments=args,
        processing_class=tok,
        model=model,
        train_dataset=ds,
        eval_dataset=ds,
    )
    eval_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    assert trainer.model_state.model.training is True
    assert eval_state.model.training is False
    batch = {k: jnp.asarray(v, dtype=jnp.int32) for k, v in ds.to_dict().items()}
    metrics = trainer.sharded_evaluation_step_function(
        eval_state, batch, *trainer._eval_shared_fn_extra_args, *trainer._eval_shared_fn_static_args
    )
    assert jnp.isfinite(metrics.loss)

    # Direct evaluation_step also switches a train-mode module to eval()
    eval_called = []

    class _TrainModeModule:
        mesh = None
        training = True

        def eval(self):
            eval_called.append(True)
            self.training = False

        def __call__(self, input_ids, attention_mask):
            del attention_mask
            return SimpleNamespace(logits=jnp.ones((input_ids.shape[0], 1), dtype=jnp.float32))

    stub_mod = _TrainModeModule()
    stub_state = SimpleNamespace(graphstate={}, model=stub_mod, merge=lambda tree: stub_mod)
    reward_evaluation_step(stub_state, batch)
    assert eval_called == [True]
    assert stub_mod.training is False

    # Grain collation when max_length=None handles uneven batches and uneven single rows
    trainer.arguments.max_length = None
    trainer.input_data_collator_grain.max_length = None
    trainer.input_data_collator_tfds.max_length = None
    collator = trainer.create_grain_collect_function(max_sequence_length=None)
    uneven_rows = [
        {
            "input_ids_chosen": [1, 2, 3, 4, 5],
            "attention_mask_chosen": [1, 1, 1, 1, 1],
            "input_ids_rejected": [1, 2],
            "attention_mask_rejected": [1, 1],
        },
        {
            "input_ids_chosen": [3, 4, 5],
            "attention_mask_chosen": [1, 1, 1],
            "input_ids_rejected": [3, 4, 5, 6],
            "attention_mask_rejected": [1, 1, 1, 1],
        },
    ]
    batched = collator(uneven_rows)
    assert batched["input_ids_chosen"].shape == (2, 5)
    assert batched["input_ids_rejected"].shape == (2, 5)
    single = collator(uneven_rows[0])
    assert single["input_ids_chosen"].shape == (5,)
    assert single["input_ids_rejected"].shape == (5,)


@pytest.mark.cpu_ok
def test_reward_guards_and_docstring():
    with pytest.raises(ValueError, match="center_rewards_coefficient"):
        RewardConfig(center_rewards_coefficient=-0.1)
    with pytest.raises(ValueError, match="center_rewards_coefficient"):
        RewardConfig(center_rewards_coefficient=float("nan"))
    assert RewardConfig(center_rewards_coefficient=0.0).center_rewards_coefficient == 0.0

    with pytest.raises(ValueError, match="model"):
        RewardTrainer(arguments=RewardConfig(), processing_class=_SimpleTokenizer(), model=None)

    assert "use_margin_loss" not in (reward_trainer_pkg.__doc__ or "")


# =============================================================================
# 5. CORE-H16 + CORE-M7: PRMTrainer & PRMPreprocessTransform
# =============================================================================


@pytest.mark.cpu_ok
def test_prm_truncate_before_last_step_masking_bos_preservation_and_packing(tmp_path):
    tok = _SimpleTokenizer(bos_token_id=1, eos_token_id=2, pad_token_id=0)

    # 1. Truncate BEFORE masking non-final steps: 3 steps, budget drops step 3 -> step 2 retains label!
    transform = PRMPreprocessTransform(
        tokenizer=tok,
        step_separator="\n",
        max_length=10,
        max_completion_length=6,
        train_on_last_step_only=True,
        is_eval=False,
        padding=False,
    )
    row = {"prompt": "pq", "completions": ["aa", "bb", "cc"], "labels": [0, 1, 0]}
    out = transform(row)
    supervised = [lbl for lbl in out["labels"] if lbl != -100]
    assert supervised == [1], f"Expected last surviving step label [1], got {supervised}"

    # 2. BOS preserved when prompt is truncated
    row_long_prompt = {"prompt": "abcdef", "completions": ["xy"], "labels": [1]}
    out_bos = PRMPreprocessTransform.tokenize_row(
        row_long_prompt,
        tokenizer=tok,
        step_separator="\n",
        max_length=6,
        max_completion_length=None,
        train_on_last_step_only=False,
        is_eval=False,
        pad_token_id=0,
        padding=False,
    )
    prompt_body = tok.encode("abcdef", add_special_tokens=False)
    assert out_bos["input_ids"][0] == 1
    assert out_bos["input_ids"][1:3] == prompt_body[-2:]
    assert len(out_bos["input_ids"]) == 6

    # 3. PRMTrainer.tokenize_row forwards pad_to_multiple_of and padding
    padded_mult = PRMTrainer.tokenize_row(
        {"prompt": "p", "completions": ["a"], "labels": [1]},
        tokenizer=tok,
        max_length=6,
        pad_to_multiple_of=4,
    )
    assert len(padded_mult["input_ids"]) == 8
    assert padded_mult["attention_mask"][-2:] == [0, 0]
    assert padded_mult["labels"][-2:] == [-100, -100]

    unpadded = PRMTrainer.tokenize_row(
        {"prompt": "p", "completions": ["a"], "labels": [1]},
        tokenizer=tok,
        max_length=16,
        padding=False,
    )
    assert len(unpadded["input_ids"]) < 16

    # 4. sequence_packing=True omits unconditional padding so PackedShardedSource packs multiple rows!
    prm_trainer = object.__new__(PRMTrainer)
    prm_trainer.processing_class = tok
    prm_trainer.arguments = PRMConfig(
        save_directory=str(tmp_path),
        max_length=12,
        sequence_packing=True,
        resume_if_possible=False,
    )
    prm_trainer._train_source = _ListSource(
        [
            {"prompt": "p", "completions": ["a"], "labels": [1]},
            {"prompt": "q", "completions": ["b"], "labels": [0]},
        ]
    )
    prm_trainer._eval_source = None
    prm_trainer._apply_preprocess_transforms()
    assert isinstance(prm_trainer._train_source, PackedShardedSource)
    packed_rows = list(prm_trainer._train_source.open_shard(prm_trainer._train_source.shard_names[0]))
    assert len(packed_rows) == 1
    assert [int(v) for v in packed_rows[0]["labels"].tolist() if int(v) != -100] == [1, 0]

    # 5. chat_template_path wired in PRMTrainer.__init__, model=None rejected, center_rewards_coefficient validated
    assert PRMConfig().center_rewards_coefficient is None
    with pytest.raises(ValueError, match="center_rewards_coefficient"):
        PRMConfig(center_rewards_coefficient=0.1)
    with pytest.raises(ValueError, match="model"):
        PRMTrainer(arguments=PRMConfig(), model=None, processing_class=tok)

    template_file = tmp_path / "prm_chat.jinja"
    template_file.write_text("{{ messages[0]['content'] }}", encoding="utf-8")
    orig_init = Trainer.__init__
    try:
        Trainer.__init__ = lambda self, **kwargs: None  # type: ignore[method-assign]
        tok2 = _SimpleTokenizer()
        stub_model = SimpleNamespace(to_state=lambda **kwargs: object())
        PRMTrainer(
            arguments=PRMConfig(chat_template_path=str(template_file), disable_dropout=False),
            model=stub_model,
            processing_class=tok2,
        )
        assert tok2.chat_template == "{{ messages[0]['content'] }}"
    finally:
        Trainer.__init__ = orig_init  # type: ignore[method-assign]


# =============================================================================
# 6. CORE-H16 + CORE-L5: MergeCallback
# =============================================================================


def test_merge_methods_real_arrays_slerp_fp16_zero_bf16_antiparallel_and_ties_dare():
    # 1. Linear merge (normalized)
    p = {"w": jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)}
    t = {"w": jnp.array([5.0, 6.0, 7.0, 8.0], dtype=jnp.float32)}
    lin = merge_pytrees(
        p,
        t,
        MergeConfig(method="linear", policy_model_weight=1.0, target_model_weight=3.0, normalize=1.0),
    )
    np.testing.assert_allclose(np.asarray(lin["w"]), [4.0, 5.0, 6.0, 7.0], rtol=1e-6)

    # 2. SLERP orthogonal midpoint
    ctx_half = MergeLeafContext(0.5, 0.5, 1.0, 0.5, 0, 0)
    orth = _slerp_merge(jnp.array([1.0, 0.0], dtype=jnp.float32), jnp.array([0.0, 1.0], dtype=jnp.float32), ctx_half)
    np.testing.assert_allclose(np.asarray(orth), [np.sqrt(0.5), np.sqrt(0.5)], rtol=1e-6)

    # 3. SLERP fp16 zero vector -> finite linear fallback (no NaN)
    fp16_zero = _slerp_merge(
        jnp.zeros((4,), dtype=jnp.float16),
        jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float16),
        ctx_half,
    )
    assert fp16_zero.dtype == jnp.float16
    assert bool(jnp.all(jnp.isfinite(fp16_zero)))
    np.testing.assert_allclose(np.asarray(fp16_zero, dtype=np.float32), [0.5, 1.0, 1.5, 2.0], rtol=1e-2)

    # 4. SLERP bf16 antiparallel -> finite linear fallback (no ~1033x blowup!)
    bf16_p = jnp.array([1.0, 2.0, -3.0, 4.0], dtype=jnp.bfloat16)
    bf16_anti = _slerp_merge(bf16_p, -bf16_p, ctx_half)
    assert bf16_anti.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(bf16_anti)))
    np.testing.assert_allclose(np.asarray(bf16_anti, dtype=np.float32), np.zeros(4), atol=1e-2)

    # 5. TIES top-k magnitude pruning
    ties_p = jnp.zeros((4,), dtype=jnp.float32)
    ties_t = jnp.array([0.1, -5.0, 0.2, 4.0], dtype=jnp.float32)
    ties_out = _ties_merge(ties_p, ties_t, MergeLeafContext(0.5, 1.0, 0.5, 0.5, 0, 0))
    np.testing.assert_allclose(np.asarray(ties_out), [0.0, -5.0, 0.0, 4.0], rtol=1e-6)

    # 6. DARE-TIES deterministic + edge densities
    dare_zero = _dare_ties_merge(ties_p, ties_t, MergeLeafContext(0.5, 1.0, 0.0, 0.5, 7, 0))
    np.testing.assert_allclose(np.asarray(dare_zero), np.asarray(ties_p))
    dare_full = _dare_ties_merge(ties_p, ties_t, MergeLeafContext(0.5, 0.5, 1.0, 0.5, 7, 0))
    np.testing.assert_allclose(np.asarray(dare_full), 0.5 * np.asarray(ties_t), rtol=1e-6)
    dare_a = _dare_ties_merge(ties_p, ties_t, MergeLeafContext(0.5, 1.0, 0.5, 0.5, 42, 3))
    dare_b = _dare_ties_merge(ties_p, ties_t, MergeLeafContext(0.5, 1.0, 0.5, 0.5, 42, 3))
    np.testing.assert_array_equal(np.asarray(dare_a), np.asarray(dare_b))


def test_merge_pytrees_preserves_non_floating_leaves_and_named_sharding():
    policy = {
        "weight": jnp.array([1.0, 3.0], dtype=jnp.float32),
        "bf16_w": jnp.array([2.0, 4.0], dtype=jnp.bfloat16),
        "step": jnp.array([10, 20], dtype=jnp.int32),
        "mask": jnp.array([True, False], dtype=jnp.bool_),
        "abstract": jax.ShapeDtypeStruct((2,), jnp.float32),
        "meta": "keep-me",
    }
    target = {
        "weight": jnp.array([5.0, 7.0], dtype=jnp.float32),
        "bf16_w": jnp.array([6.0, 8.0], dtype=jnp.bfloat16),
        "step": jnp.array([99, 100], dtype=jnp.int32),
        "mask": jnp.array([False, True], dtype=jnp.bool_),
        "abstract": jax.ShapeDtypeStruct((2,), jnp.float32),
        "meta": "target-meta",
    }
    for method in ("linear", "slerp", "ties", "dare_ties"):
        merged = merge_pytrees(policy, target, MergeConfig(method=method, target_model_density=(0.5,)))
        assert merged["step"].dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(merged["step"]), [10, 20])
        assert merged["mask"].dtype == jnp.bool_
        np.testing.assert_array_equal(np.asarray(merged["mask"]), [True, False])
        assert isinstance(merged["abstract"], jax.ShapeDtypeStruct)
        assert merged["meta"] == "keep-me"

    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("fsdp",))
    sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
    n_rows = max(4, len(devices) * 2)
    raw_p = jnp.arange(n_rows * 4, dtype=jnp.float32).reshape(n_rows, 4)
    raw_t = raw_p * 2.0 + jnp.sin(raw_p)
    sharded_p = jax.device_put(raw_p, sharding)
    sharded_t = jax.device_put(raw_t, sharding)

    mask_sharded = _topk_density_mask(sharded_t - sharded_p, 0.5)
    mask_plain = _topk_density_mask(raw_t - raw_p, 0.5)
    assert mask_sharded.sharding == sharding
    np.testing.assert_array_equal(np.asarray(mask_sharded), np.asarray(mask_plain))

    for method in ("ties", "dare_ties"):
        cfg = MergeConfig(method=method, target_model_density=(0.5,), seed=123)
        out_sharded = merge_pytrees({"w": sharded_p}, {"w": sharded_t}, cfg)["w"]
        out_plain = merge_pytrees({"w": raw_p}, {"w": raw_t}, cfg)["w"]
        assert out_sharded.sharding == sharding
        np.testing.assert_allclose(np.asarray(out_sharded), np.asarray(out_plain), rtol=1e-6, atol=1e-6)


def test_merge_callback_process0_push_to_hub_none_kwargs_and_config_validation(monkeypatch):
    with pytest.raises(ValueError, match="positive"):
        MergeConfig(policy_model_weight=0.0, target_model_weight=0.0)
    with pytest.raises(ValueError, match="finite"):
        MergeConfig(policy_model_weight=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        MergeConfig(t_values=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        MergeConfig(target_model_density=(float("nan"),))
    with pytest.raises(ValueError, match="finite"):
        MergeConfig(target_model_weight=float("inf"))

    pushed = []

    class _FakeModel:
        def push_to_hub(self, **kwargs):
            pushed.append(kwargs)

    class _FakeState:
        def __init__(self, tree):
            self.graphstate = tree

        def merge(self, tree):
            self.merged_tree = tree
            return _FakeModel()

    policy_st = _FakeState({"w": jnp.array([1.0, 2.0], dtype=jnp.float32)})
    target_st = _FakeState({"w": jnp.array([3.0, 4.0], dtype=jnp.float32)})
    cb = MergeModelCallback(
        merge_config=MergeConfig(method="linear"),
        merge_at_every_checkpoint=True,
        push_to_hub=True,
        hub_model_id="org/merged",
        target_state=target_st,
    )

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    cb.on_save(policy_state=policy_st, target_state=None, hub_kwargs=None)
    assert cb.last_merged_state is not None
    assert pushed == []

    monkeypatch.setattr(jax, "process_index", lambda: 0)
    cb.on_save(policy_state=policy_st, target_state=None, hub_kwargs=None)
    assert pushed == [{"repo_id": "org/merged"}]


# =============================================================================
# 7. CORE-H16 + CORE-M7: BEMADPOTrainer & BEMACallback
# =============================================================================


def test_bema_dpo_strips_precomputed_ref_logps_and_validates_clamps_params():
    with pytest.raises(ValueError, match="lag"):
        BEMACallback(lag=-1)
    with pytest.raises(ValueError, match="ema_power"):
        BEMACallback(ema_power=-0.5)
    with pytest.raises(ValueError, match="bias_power"):
        BEMACallback(bias_power=-0.2)
    with pytest.raises(ValueError, match="min_ema_multiplier"):
        BEMACallback(min_ema_multiplier=1.5)
    with pytest.raises(ValueError, match="both be zero"):
        BEMACallback(lag=0, multiplier=0.0)
    with pytest.raises(ValueError, match="ema_power"):
        BEMACallback(ema_power=float("nan"))

    cb_edge = BEMACallback(lag=0, multiplier=0.5, ema_power=0.5, bias_power=0.5, min_ema_multiplier=0.0)
    for s in (0, 1, 2, 10):
        beta = cb_edge._ema_beta(s)
        alpha = cb_edge._bema_alpha(s)
        assert isinstance(beta, float) and 0.0 <= beta <= 1.0
        assert isinstance(alpha, float) and 0.0 <= alpha <= 1.0
    assert cb_edge._ema_beta(0) == 1.0
    assert cb_edge._ema_beta(1) == 1.0

    # Non-floating leaves are NOT promoted to float32; bf16 stays bf16
    cb = BEMACallback(update_after=0, update_freq=1, min_ema_multiplier=0.0)
    gs0 = {
        "w_f32": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "w_bf16": jnp.array([1.0, 2.0], dtype=jnp.bfloat16),
        "idx_i32": jnp.array([3, 4], dtype=jnp.int32),
        "flag_bool": jnp.array([True, False], dtype=jnp.bool_),
        "tag": "meta",
    }
    gs1 = {
        "w_f32": jnp.array([5.0, 6.0], dtype=jnp.float32),
        "w_bf16": jnp.array([5.0, 6.0], dtype=jnp.bfloat16),
        "idx_i32": jnp.array([7, 8], dtype=jnp.int32),
        "flag_bool": jnp.array([False, True], dtype=jnp.bool_),
        "tag": "meta",
    }
    cb.update_graphstate(gs0, step=0)
    out = cb.update_graphstate(gs1, step=1)
    assert out["w_f32"].dtype == jnp.float32
    assert out["w_bf16"].dtype == jnp.bfloat16
    assert out["idx_i32"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out["idx_i32"]), [7, 8])
    assert out["flag_bool"].dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(out["flag_bool"]), [False, True])

    # BEMADPOTrainer._preprocess_batch_input strips cached ref logps when update_ref_model=True
    trainer = object.__new__(BEMADPOTrainer)
    trainer.arguments = DPOConfig(precompute_ref_log_probs=True, sync_ref_model=False)
    trainer.bema_callback = BEMACallback(update_ref_model=True)
    trainer.reference_state = object()
    raw_batch = {
        "prompt_input_ids": jnp.ones((2, 4), dtype=jnp.int32),
        "ref_chosen_logps": jnp.zeros((2,), dtype=jnp.float32),
        "ref_rejected_logps": jnp.zeros((2,), dtype=jnp.float32),
        "reference_chosen_log_probs": jnp.zeros((2,), dtype=jnp.float32),
        "reference_rejected_log_probs": jnp.zeros((2,), dtype=jnp.float32),
    }
    from easydel.trainers.trainer import Trainer as _BaseT

    orig_base = _BaseT._preprocess_batch_input
    try:
        _BaseT._preprocess_batch_input = lambda self, state, batch, is_train: (batch, {})  # type: ignore[method-assign]
        cleaned, _ = trainer._preprocess_batch_input(None, copy.deepcopy(raw_batch), is_train=True)
        for key in (
            "ref_chosen_logps",
            "ref_rejected_logps",
            "reference_chosen_log_probs",
            "reference_rejected_log_probs",
        ):
            assert key not in cleaned
        assert "prompt_input_ids" in cleaned

        trainer.bema_callback = BEMACallback(update_ref_model=False)
        kept, _ = trainer._preprocess_batch_input(None, copy.deepcopy(raw_batch), is_train=True)
        assert "ref_chosen_logps" in kept
        assert "ref_rejected_logps" in kept
    finally:
        _BaseT._preprocess_batch_input = orig_base  # type: ignore[method-assign]


def test_bema_step_indexing_metric_logging_and_multihost_manifest_and_barrier(tmp_path, monkeypatch):
    @dataclass
    class _MiniState:
        graphstate: object
        graphother: object = None
        model: object = None

        def replace(self, **changes):
            return replace(self, **changes)

    trainer = object.__new__(BEMADPOTrainer)
    trainer.arguments = DPOConfig(sync_ref_model=False)
    trainer.bema_callback = BEMACallback(
        update_after=0,
        update_freq=1,
        update_ref_model=True,
        ref_model_update_after=0,
        ref_model_update_freq=4,
    )
    trainer._last_bema_ref_update_step = None
    ref0 = _MiniState(graphstate={"w": jnp.zeros((2,), dtype=jnp.float32)})
    trainer.reference_state = ref0
    pol1 = _MiniState(graphstate={"w": jnp.ones((2,), dtype=jnp.float32)})

    logged_calls = []
    monkeypatch.setattr(
        DPOTrainer,
        "log_metrics",
        lambda self, metrics, pbar, step, mode="train", *a, **kw: logged_calls.append((step, mode, dict(metrics))),
    )

    _, m1 = trainer.on_step_end(pol1, LossMetrics(loss=jnp.float32(1.0)), step=1)
    assert trainer.reference_state is not ref0
    assert int(m1.other_metrics["bema/ref_model_updated"]) == 1

    trainer.log_metrics({"train/loss": 1.0}, pbar=None, step=1, mode="train")
    assert logged_calls[-1][0] == 1
    assert logged_calls[-1][2]["train/bema/ref_model_updated"] == 1

    ref1 = trainer.reference_state
    pol2 = _MiniState(graphstate={"w": jnp.full((2,), 2.0, dtype=jnp.float32)})
    _, m2 = trainer.on_step_end(pol2, LossMetrics(loss=jnp.float32(0.9)), step=2)
    assert trainer.reference_state is ref1
    assert "bema/ref_model_updated" not in (m2.other_metrics or {})
    trainer.log_metrics({"train/loss": 0.9}, pbar=None, step=2, mode="train")
    assert "train/bema/ref_model_updated" not in logged_calls[-1][2]

    trainer.on_step_end(pol2, LossMetrics(loss=jnp.float32(0.8)), step=3)
    trainer.on_step_end(pol2, LossMetrics(loss=jnp.float32(0.7)), step=4)
    assert trainer.reference_state is ref1
    pol5 = _MiniState(graphstate={"w": jnp.full((2,), 5.0, dtype=jnp.float32)})
    _, m5 = trainer.on_step_end(pol5, LossMetrics(loss=jnp.float32(0.6)), step=5)
    assert trainer.reference_state is not ref1
    assert int(m5.other_metrics["bema/ref_model_updated"]) == 1

    from easydel.trainers.bema_trainer import bema_trainer as bema_mod
    from jax.experimental import multihost_utils

    events = []

    class _StubAsyncManager:
        def save_pytree(self, tree, path, *, prefix, write_index=True, **kwargs):
            del tree, path, write_index, kwargs
            events.append(f"save:{prefix}")

    monkeypatch.setattr(DPOTrainer, "_save_state", lambda self, st, save_directory=None, *a, **kw: str(save_directory))
    monkeypatch.setattr(bema_mod, "AsyncCheckpointManager", _StubAsyncManager)
    monkeypatch.setattr(multihost_utils, "sync_global_devices", lambda tag: events.append(f"sync:{tag}"))

    def _make_save_trainer():
        t = object.__new__(BEMADPOTrainer)
        t.bema_callback = BEMACallback(update_freq=1, ref_model_update_freq=2, update_ref_model=True)
        for name in ("theta0", "ema", "bema"):
            setattr(t.bema_callback, f"_{name}_graphstate", spx.State({"params": {"w": 1.0}}))
        t.reference_state = _MiniState(
            graphstate=spx.State({"params": {"w": 1.0}}),
            graphother=spx.State({}),
        )
        return t

    events.clear()
    monkeypatch.setattr(bema_mod.jax, "process_count", lambda: 2)
    monkeypatch.setattr(bema_mod.jax, "process_index", lambda: 1)
    rank1_dir = tmp_path / "ckpt_rank1"
    _make_save_trainer()._save_state(object(), str(rank1_dir))
    assert not (rank1_dir / "bema" / "manifest.json").exists()
    assert "sync:bema.checkpoint.complete" in events
    assert "sync:bema.checkpoint.committed" in events
    assert events.index("sync:bema.checkpoint.complete") < events.index("sync:bema.checkpoint.committed")

    events.clear()
    monkeypatch.setattr(bema_mod.jax, "process_index", lambda: 0)
    rank0_dir = tmp_path / "ckpt_rank0"
    _make_save_trainer()._save_state(object(), str(rank0_dir))
    manifest_data = json.loads((rank0_dir / "bema" / "manifest.json").read_text())
    assert manifest_data["complete"] is True
    assert events[-2:] == ["sync:bema.checkpoint.complete", "sync:bema.checkpoint.committed"]


# ---------------------------------------------------------------------------
# Audit #193 & #194: Core & Standard Trainers Leak & Post-Fix Regression Tests
# ---------------------------------------------------------------------------


def test_audit193_1_scheduled_module_binder_and_core_adapters_do_not_pin_step0_state() -> None:
    """#1 (Core MPMD): make_scheduled_module_binder / state_shell and all 3 core adapters never pin call/state/batch."""
    from easydel.infra.base_state import EasyDeLState
    from easydel.trainers.embedding_trainer import _fn as embedding_fn
    from easydel.trainers.reward_trainer import _fn as reward_fn
    from easydel.trainers.trainer import _fn as trainer_fn
    from easydel.trainers.training_utils import (
        ScheduledStepCall,
        make_scheduled_module_binder,
        make_scheduled_state_shell,
    )

    class _TrackedBox:
        def __init__(self, payload: Any) -> None:
            self.payload = payload

    class _TinyModule(spx.Module):
        def __init__(self, w: jax.Array, mesh: Mesh) -> None:
            self.w = spx.Parameter(w)
            self.mesh = mesh
            self.config = types.SimpleNamespace(pad_token_id=0, pipeline_virtual_stages=1)

        def prepare_inputs_for_call(self, **kwargs):
            return dict(kwargs)

        def compute_loss(self, labels=None, loss_config=None, **kwargs):
            del labels, loss_config
            loss = jnp.sum(self.w.value * kwargs["input_ids"].astype(jnp.float32))
            return types.SimpleNamespace(loss=loss), LossMetrics(loss=loss)

        def __call__(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            x = input_ids.astype(jnp.float32) * self.w.value
            return types.SimpleNamespace(logits=x[:, :1], embeddings=x)

    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))

    def _build_call(batch_payload: dict[str, Any], bound_extra: dict[str, Any]):
        mod = _TinyModule(jnp.asarray(2.0, dtype=jnp.float32), mesh)
        gdef, gstate = spx.export(mod)
        gother = spx.State({})
        opt_sentinel = _TrackedBox(jnp.ones((4,), dtype=jnp.float32))
        batch_box = _TrackedBox(batch_payload)
        state = EasyDeLState(
            step=jnp.asarray(0, dtype=jnp.int32),
            graphdef=gdef,
            graphstate=gstate,
            graphother=gother,
            tx=None,
            opt_state={"m": opt_sentinel},
        )
        call = ScheduledStepCall(
            step_fn=lambda *a, **k: None,
            state=state,
            batch=batch_payload,
            args=(state, batch_payload),
            kwargs={"_box": batch_box},
            bound_arguments={"partition_spec": PartitionSpec(), **bound_extra},
            schedule=types.SimpleNamespace(virtual_stages_per_rank=lambda: 2, stage_layout=None),
        )
        refs = (
            weakref.ref(call),
            weakref.ref(state),
            weakref.ref(gstate),
            weakref.ref(opt_sentinel),
            weakref.ref(batch_box),
        )
        return call, gstate, refs

    # 1. Direct make_scheduled_module_binder & make_scheduled_state_shell checks
    call0, gstate0, refs0 = _build_call({"input_ids": jnp.ones((2, 2), dtype=jnp.int32)}, {})
    binder = make_scheduled_module_binder(call0)
    shell = make_scheduled_state_shell(call0)
    assert shell.graphstate is None
    assert shell.opt_state is None
    assert shell.tx is None
    assert shell.step == 0
    step1_tree = spx.State({"parameters": {"w": jnp.asarray(5.0, dtype=jnp.float32)}})
    del call0, gstate0
    gc.collect()
    assert all(r() is None for r in refs0), f"Pinned objects after make_scheduled_module_binder: {[r() for r in refs0]}"
    rebound = binder(step1_tree)
    assert float(rebound.w.value) == pytest.approx(5.0)
    assert rebound.config.pipeline_virtual_stages == 2

    # 2. Base trainer MPMD adapter
    call_b, gstate_b, refs_b = _build_call({"input_ids": jnp.ones((2, 2), dtype=jnp.int32)}, {})
    base_loss = trainer_fn._make_base_scheduled_loss(call_b)
    del call_b, gstate_b
    gc.collect()
    assert all(r() is None for r in refs_b), f"trainer._fn pinned Step-0 call/state: {[r() for r in refs_b]}"
    loss_val, aux = base_loss(step1_tree, {"input_ids": jnp.ones((2, 2), dtype=jnp.int32)})
    assert float(loss_val) == pytest.approx(20.0)
    assert "loss_detached" in aux

    # 3. Reward trainer MPMD adapter
    reward_batch = {
        "input_ids_chosen": jnp.ones((1, 2), dtype=jnp.int32),
        "attention_mask_chosen": jnp.ones((1, 2), dtype=jnp.int32),
        "input_ids_rejected": jnp.zeros((1, 2), dtype=jnp.int32),
        "attention_mask_rejected": jnp.ones((1, 2), dtype=jnp.int32),
    }
    call_r, gstate_r, refs_r = _build_call(reward_batch, {"center_rewards_coefficient": None})
    rew_loss = reward_fn._make_reward_scheduled_loss(call_r)
    del call_r, gstate_r
    gc.collect()
    assert all(r() is None for r in refs_r), f"reward_trainer._fn pinned Step-0 call/state: {[r() for r in refs_r]}"
    r_val = rew_loss(step1_tree, reward_batch)
    assert np.isfinite(float(r_val))

    # 4. Embedding trainer MPMD adapter
    emb_batch = {
        "query_input_ids": jnp.ones((2, 2), dtype=jnp.int32),
        "query_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "positive_input_ids": jnp.ones((2, 2), dtype=jnp.int32),
        "positive_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
    }
    call_e, gstate_e, refs_e = _build_call(emb_batch, {"loss_type": "infonce", "temperature": 0.1})
    emb_loss = embedding_fn._make_embedding_scheduled_loss(call_e)
    del call_e, gstate_e
    gc.collect()
    assert all(r() is None for r in refs_e), f"embedding_trainer._fn pinned Step-0 call/state: {[r() for r in refs_e]}"
    e_val = emb_loss(step1_tree, emb_batch)
    assert np.isfinite(float(e_val))


def test_audit193_4_global_assistant_only_loss_masks_without_per_dataset_carriers() -> None:
    """#4: global assistant_only_loss=True must mask prompt tokens even when _assistant_only is absent or all -1."""
    span = assistant_only_mask.AssistantSpanIds(header_ids=(3, 6, 8), end_id=4)

    class _GlobalMaskHost(assistant_only_mask.AssistantOnlyMaskMixin):
        def __init__(self) -> None:
            self.arguments = types.SimpleNamespace(
                assistant_only_loss=True,
                zero_supervision_abort_fraction=None,
                zero_supervision_report_fraction=None,
            )
            self._cached_assistant_span_ids = span

    host = _GlobalMaskHost()
    ids = np.array([[5, 10, 3, 6, 8, 11, 12, 4]], dtype=np.int32)

    # Single-dataset batch with NO _assistant_only carrier and NO completion_mask/assistant_masks
    batch_no_carrier = {"input_ids": ids.copy(), "attention_mask": np.ones_like(ids)}
    out1 = host._apply_assistant_supervision(batch_no_carrier, {})
    assert out1["completion_mask"][0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert out1["labels"][0].tolist() == [-100, -100, -100, -100, -100, 11, 12, 4]

    # All -1 (inherit global) carrier must also mask prompt tokens
    batch_all_inherit = {
        "input_ids": ids.copy(),
        "attention_mask": np.ones_like(ids),
        "_assistant_only": np.array([-1], dtype=np.int32),
    }
    out2 = host._apply_assistant_supervision(batch_all_inherit, {})
    assert out2["completion_mask"][0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert out2["labels"][0].tolist() == [-100, -100, -100, -100, -100, 11, 12, 4]


def test_audit193_5_trainer_updates_model_state_and_encoder_processor_does_not_pin_step0() -> None:
    """#5: Trainer.train() updates self.model_state per step, drops Step-0 refs, and EncoderProcessor doesn't pin."""
    from easydel.trainers.metrics import NullProgressBar
    from easydel.trainers.process_encoder import EncoderBinding, EncoderProcessor, ProcessEncoderConfig

    # 1. BaseTrainer._setup_process_encoder creates EncoderProcessor(model=None, ...) without pinning Step-0 model.
    class _FakeVLM:
        def get_image_features(self, pixel_values):
            return pixel_values

    vlm = _FakeVLM()
    vlm_ref = weakref.ref(vlm)
    dummy_trainer = object.__new__(Trainer)
    dummy_trainer.arguments = types.SimpleNamespace(
        process_encoder=ProcessEncoderConfig(enabled=True),
        trainable_selector=None,
        gradient_accumulation_steps=1,
        mpmd_scheduler=None,
    )
    dummy_trainer._buckets = []
    dummy_trainer._has_buckets = False
    dummy_trainer.model_state = types.SimpleNamespace(model=vlm)
    binding = EncoderBinding(
        tower_attr="visual",
        feature_kwarg="image_embeds",
        feature_fn_name="get_image_features",
        accepted_kwargs=frozenset(),
        feature_valid_length_kwarg=None,
    )
    # Monkeypatch validate_process_encoder in base_trainer module
    orig_validate = base_trainer_mod.validate_process_encoder
    try:
        base_trainer_mod.validate_process_encoder = lambda *a, **kw: binding
        proc = dummy_trainer._setup_process_encoder()
    finally:
        base_trainer_mod.validate_process_encoder = orig_validate

    assert isinstance(proc, EncoderProcessor)
    assert proc._model is None
    dummy_trainer.model_state = None
    del vlm
    gc.collect()
    assert vlm_ref() is None, "EncoderProcessor pinned Step-0 model!"

    # 2. Trainer.train() updates self.model_state after every step and passes live state to _apply_process_encoder_to_batch
    class _DonatingState:
        def __init__(self, step_val: int) -> None:
            self.step = jnp.asarray(step_val, dtype=jnp.int32)
            self.donated = False

        @property
        def model(self):
            if self.donated:
                raise RuntimeError("Array has been deleted (Step-0 donated buffer accessed!)")
            return types.SimpleNamespace(step=int(self.step))

    tr = object.__new__(Trainer)
    tr.arguments = TrainingArguments(
        num_train_epochs=1,
        max_length=4,
        total_batch_size=1,
        do_eval=False,
        use_wandb=False,
        report_metrics=False,
        log_all_workers=True,
        dataloader_fast_forward_on_resume=False,
        progress_bar_type="json",
        track_memory=False,
    )
    tr.max_training_steps = 2
    tr.dataloader_train = [{"input_ids": np.ones((1, 4), dtype=np.int32)} for _ in range(3)]
    tr.data_collator = None
    tr._has_buckets = False
    tr._buckets = []
    tr._bucket_iters = {}
    tr._backward_flops_per_token = 0.0
    tr._extra_backward_flops_per_token = 0.0
    tr.scheduler = None
    tr._model = types.SimpleNamespace(mesh=contextlib.nullcontext())
    tr.train_tracker = types.SimpleNamespace(trace_compilation=lambda: contextlib.nullcontext())
    tr.start_training_hook = lambda: None
    tr._setup_initial_metrics = lambda st: None
    tr._runtime_trace = lambda *a, **kw: None
    tr._stop_profiler = lambda: None
    tr._maybe_start_profiler = lambda step: None
    tr._maybe_stop_profiler = lambda step: None
    tr.create_progress_bar = lambda **kw: NullProgressBar()
    tr._get_epoch_step_bounds = lambda epoch: (0, 2)
    tr._apply_pose_to_batch = lambda batch, step: batch
    tr._apply_randomized_yarn_to_batch = lambda batch, step: batch
    tr.on_step_start = lambda state, step: state
    tr.on_step_end = lambda state, metrics, step: (state, metrics)
    tr.apply_training_hooks = lambda metrics, loss_value: metrics
    tr.log_metrics = lambda *a, **kw: None
    tr.maybe_generate = lambda *a, **kw: None
    tr.maybe_benchmark = lambda *a, **kw: None
    tr._should_save_tpu_preemption_checkpoint = lambda step: False
    tr._should_save_checkpoint = lambda step: False
    tr._save_checkpoint_for_step = lambda *a, **kw: None
    tr.log_weight_distribution = lambda *a, **kw: None
    tr.log_watchers = lambda *a, **kw: None
    tr._should_run_evaluation = lambda step: False
    tr._prepare_training_output = lambda state, run_exception: types.SimpleNamespace(state=state)
    tr.finish = lambda: None

    seen_encoder_steps: list[int] = []

    def _check_encoder(batch, step, state=None):
        active = state if state is not None else tr.model_state
        # Accessing .model raises if donated Step-0 state is ever reused on Step 1!
        seen_encoder_steps.append(active.model.step)
        return batch

    tr._apply_process_encoder_to_batch = _check_encoder

    step0_state = _DonatingState(0)
    step0_ref = weakref.ref(step0_state)
    tr.model_state = step0_state
    del step0_state

    step0_collected_during_step1: list[bool] = []

    def _exec_step(state, batch):
        del batch
        cur = int(state.step)
        if cur == 1:
            gc.collect()
            step0_collected_during_step1.append(step0_ref() is None)
        state.donated = True
        nxt = _DonatingState(cur + 1)
        return nxt, LossMetrics(loss=jnp.float32(0.5)), None

    tr._execute_train_step = _exec_step
    out = tr.train()
    assert seen_encoder_steps == [0, 1]
    assert int(tr.model_state.step) == 2
    assert tr.model_state is out.state
    assert step0_collected_during_step1 == [True], "Step-0 state was still pinned during Step 1!"


def test_audit193_6_bema_save_and_load_do_not_spawn_checkpointer_threads(tmp_path, monkeypatch) -> None:
    """#6 (BEMA): Repeated BEMADPOTrainer checkpoint saves/loads spawn zero Checkpointer daemon threads."""
    from easydel.trainers.bema_trainer import bema_trainer as bema_mod

    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))

    @dataclass
    class _BemaMiniState:
        graphstate: Any
        graphother: Any
        model: Any

        def replace(self, **kw):
            return _BemaMiniState(
                graphstate=kw.get("graphstate", self.graphstate),
                graphother=kw.get("graphother", self.graphother),
                model=kw.get("model", self.model),
            )

    fake_model = types.SimpleNamespace(mesh=mesh, resolve_shardings_regex=lambda: ())

    saved_store: dict[tuple[str, str], Any] = {}
    manager_inits = 0

    class _RecordingAsyncManager:
        def __init__(self) -> None:
            nonlocal manager_inits
            manager_inits += 1

        def save_pytree(self, pytree, path, *, prefix, write_index=True, **kwargs):
            del write_index, kwargs
            saved_store[(str(path), prefix)] = pytree

        def load_pytree(self, path, mesh, *, prefix, template=None, sharding_rules=None, **kwargs):
            del mesh, template, sharding_rules, kwargs
            return saved_store[(str(path), prefix)], {}

    monkeypatch.setattr(bema_mod, "AsyncCheckpointManager", _RecordingAsyncManager)
    monkeypatch.setattr(DPOTrainer, "_save_state", lambda self, st, save_directory=None, *a, **kw: str(save_directory))
    monkeypatch.setattr(DPOTrainer, "_load_auxiliary_states", lambda self, path: None)

    trainer = object.__new__(BEMADPOTrainer)
    trainer.bema_callback = BEMACallback(update_freq=1, ref_model_update_freq=2, update_ref_model=True)
    for name in ("theta0", "ema", "bema"):
        setattr(
            trainer.bema_callback,
            f"_{name}_graphstate",
            spx.State({"params": {"w": jnp.ones((2,), dtype=jnp.float32)}}),
        )
    trainer.reference_state = _BemaMiniState(
        graphstate=spx.State({"params": {"w": jnp.ones((2,), dtype=jnp.float32)}}),
        graphother=spx.State({}),
        model=fake_model,
    )

    threads_before = len(threading.enumerate())
    for step in range(4):
        ckpt_dir = tmp_path / f"ckpt_{step}"
        trainer._save_state(object(), str(ckpt_dir))
    trainer._load_auxiliary_states(str(tmp_path / "ckpt_3"))
    threads_after = len(threading.enumerate())

    assert threads_after == threads_before
    assert manager_inits == 1, f"Expected single cached AsyncCheckpointManager, got {manager_inits}"


def test_audit193_7_wandb_tables_rotate_to_bound_host_memory(monkeypatch) -> None:
    """#7: preview_log_table, benchmark_log_table, and training_generation_log_table rotate cleanly."""
    created_tables: list[Any] = []
    logged_payloads: list[dict[str, Any]] = []

    class _FakeWandbTable:
        def __init__(self, columns, log_mode=None) -> None:
            self.columns = list(columns)
            self.log_mode = log_mode
            self.data: list[tuple[Any, ...]] = []
            created_tables.append(self)

        def add_data(self, *row) -> None:
            self.data.append(row)

    fake_wandb = types.SimpleNamespace(
        Table=_FakeWandbTable,
        log=lambda payload, step=None: logged_payloads.append(dict(payload)),
    )
    monkeypatch.setattr(base_trainer_mod, "wandb", fake_wandb)

    trainer = object.__new__(Trainer)
    trainer.arguments = types.SimpleNamespace(
        use_wandb=True,
        can_log_metrics=True,
        generation_log_to_wandb=True,
        log_training_generations_to_wandb=True,
    )
    trainer.preview_log_table = None
    trainer.benchmark_log_table = None
    trainer.training_generation_log_table = None
    trainer.processing_class = None
    trainer._WANDB_TABLE_MAX_ROWS = 4
    trainer._WANDB_TABLE_MAX_INCREMENTS = 3

    # Log 10 benchmark updates of 2 rows each -> must rotate so no table exceeds max_rows (4) or max_increments (3)
    for step in range(10):
        trainer._log_benchmark_results_to_wandb(
            step=step,
            benchmark_results={"suite": {"results": {"t": {"m1": float(step), "m2": float(step + 1)}}}},
        )
        assert len(trainer.benchmark_log_table.data) <= 4
    assert len(created_tables) >= 5
    assert all(t.log_mode == "INCREMENTAL" for t in created_tables)

    # Same bound on training_generation_log_table
    created_tables.clear()
    for step in range(8):
        trainer._log_training_generations_to_wandb(
            state=types.SimpleNamespace(step=jnp.asarray(step)),
            prompts=["p1", "p2"],
            completions=["c1", "c2"],
        )
        assert len(trainer.training_generation_log_table.data) <= 4
    assert len(created_tables) >= 4


def test_audit193_12_cached_scheduled_auxiliary_immutable_mesh_and_closure_gc() -> None:
    """#12: cached_scheduled_auxiliary works on immutable jax.Mesh/SpxMesh without pinning closures globally."""
    training_utils._SCHEDULED_AUXILIARY_CACHE.clear()

    class _TrainerBox:
        def __init__(self) -> None:
            self.payload = jnp.ones((4,), dtype=jnp.float32)

        def bound_fn(self, x):
            return x + self.payload

    for mesh_factory in (
        lambda: Mesh(np.array(jax.devices()[:1]), ("dp",)),
        lambda: spx.create_cpu_mesh((1,), ("dp",)),
    ):
        mesh = mesh_factory()
        box = _TrainerBox()
        box_ref = weakref.ref(box)

        def _make_closure(owner: _TrainerBox):
            def forward(x):
                return x + owner.payload

            return forward

        fn = _make_closure(box)
        fn_ref = weakref.ref(fn)

        c1 = training_utils.cached_scheduled_auxiliary(fn, mesh)
        c2 = training_utils.cached_scheduled_auxiliary(fn, mesh)
        assert c1 is c2, "Expected cache hit on identical (fn, immutable_mesh)"
        assert len(training_utils._SCHEDULED_AUXILIARY_CACHE) == 0, "Should not fall back to global cache!"

        # Invoke once to populate spx.jit caches
        np.testing.assert_allclose(c1(jnp.zeros((4,), dtype=jnp.float32)), np.ones((4,), dtype=np.float32))

        # Bound method should also hit cache on repeated lookups and die with owning box
        bm1 = training_utils.cached_scheduled_auxiliary(box.bound_fn, mesh)
        bm2 = training_utils.cached_scheduled_auxiliary(box.bound_fn, mesh)
        assert bm1 is bm2
        assert len(training_utils._SCHEDULED_AUXILIARY_CACHE) == 0

        # Deleting trainer box + closure must GC both even while `mesh` stays alive!
        del c1, c2, bm1, bm2, fn, box
        gc.collect()
        assert fn_ref() is None, "Closure was pinned in cache while mesh was alive!"
        assert box_ref() is None, "Trainer box was pinned via closure in cache!"
        del mesh
        gc.collect()

    # Module-level function + unique jax.Mesh: deleting mesh evicts entry via weakref.finalize
    def _static_aux_fn(x):
        return x * 2.0

    jmesh = Mesh(np.array(jax.devices()[:1]), ("dp_aux_evict_unique",))
    jmesh_ref = weakref.ref(jmesh)
    compiled = training_utils.cached_scheduled_auxiliary(_static_aux_fn, jmesh)
    _ = compiled(jnp.ones((2,), dtype=jnp.float32))
    assert len(_static_aux_fn._easydel_scheduled_aux_cache) == 1
    del compiled, jmesh
    gc.collect()
    assert jmesh_ref() is None
    assert len(_static_aux_fn._easydel_scheduled_aux_cache) == 0


def test_audit193_15_packed_boundary_and_distinct_marker_template_masking() -> None:
    """#15: (a) packed boundary -100 doesn't skip span scan; (b) ao==0 keeps boundary -100; (c) distinct user markers."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    # (a) & (b): Pack 2 docs with packer.pack_sequences -> Doc 1 starts with boundary -100 only.
    # Doc 0: [5, 10, 3, 6, 8, 11, 4], Doc 1: [5, 12, 3, 6, 8, 13, 4]
    span = assistant_only_mask.AssistantSpanIds(header_ids=(3, 6, 8), end_id=4)
    host = _DummyMaskHost(span)

    ds = Dataset.from_dict(
        {
            "input_ids": [[5, 10, 3, 6, 8, 11, 4], [5, 12, 3, 6, 8, 13, 4]],
            "attention_mask": [[1] * 7, [1] * 7],
        }
    )
    packed = packer.pack_sequences(ds, max_length=16, pad_token_id=0, reset_position_ids=True)[0]
    # Index 0..6 = doc 0, index 7 = pad sep (seg=-1), index 8..14 = doc 1 (labels[8] == -100 boundary!), index 15 = sep
    assert packed["labels"][8] == -100 and all(x != -100 for x in packed["labels"][9:15])

    batch_opt_in = {
        "input_ids": np.asarray([packed["input_ids"]], dtype=np.int32),
        "attention_mask": np.asarray([packed["attention_mask"]], dtype=np.int32),
        "segment_ids": np.asarray([packed["segment_ids"]], dtype=np.int32),
        "labels": np.asarray([packed["labels"]], dtype=np.int32),
    }
    # Both docs opt in (ao == 1): Doc 1's lone boundary -100 MUST NOT skip span scan!
    ao_in = np.ones((1, 16), dtype=np.int32)
    host._assemble_per_dataset_mask(batch_opt_in, ao_in, global_flag=False)
    assert batch_opt_in["completion_mask"][0, 8:15].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert batch_opt_in["labels"][0, 8:15].tolist() == [-100, -100, -100, -100, -100, 13, 4]

    # (b) Doc 1 opts out (ao == 0) with NO pre-existing labels -> synthesized labels MUST mask boundary index 8!
    batch_synth = {
        "input_ids": np.asarray([packed["input_ids"]], dtype=np.int32),
        "attention_mask": np.asarray([packed["attention_mask"]], dtype=np.int32),
        "segment_ids": np.asarray([packed["segment_ids"]], dtype=np.int32),
    }
    ao_out = np.zeros((1, 16), dtype=np.int32)
    host._assemble_per_dataset_mask(batch_synth, ao_out, global_flag=False)
    assert batch_synth["labels"][0, 8] == -100
    assert batch_synth["labels"][0, 9:15].tolist() == [12, 3, 6, 8, 13, 4]

    # (c) Distinct-marker template (Phi-3 / DeepSeek style with BOS): user-only row must fail safe to all zeros!
    words = ["[PAD]", "[UNK]", "<bos>", "<|user|>", "<|assistant|>", "<|end|>", "hello", "world", "plain", "text"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="<bos>",
        eos_token="<|end|>",
    )
    tok.chat_template = (
        "<bos>{% for m in messages %}{% if m['role'] == 'assistant' %}<|assistant|> {{ m['content'] }} <|end|> "
        "{% else %}<|user|> {{ m['content'] }} {% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}"
    )
    derived = assistant_only_mask.derive_assistant_span_ids(tok)
    assert words.index("<|user|>") in derived.structure_ids
    assert words.index("<bos>") not in derived.structure_ids

    # User-only row (<bos> <|user|> hello world) has NO <|assistant|> and NO <|end|>, yet MUST get mask all-0!
    user_only = np.asarray(
        [[words.index("<bos>"), words.index("<|user|>"), words.index("hello"), words.index("world")]],
        dtype=np.int32,
    )
    assert assistant_only_mask.assistant_mask_from_token_ids(user_only, derived)[0].tolist() == [0, 0, 0, 0]

    # Plain-text row (<bos> plain text) has no chat markers -> default fallback supervises it,
    # while explicit ao == 1 fails safe to all zeros!
    plain_ids = np.asarray(
        [[words.index("<bos>"), words.index("plain"), words.index("text")]],
        dtype=np.int32,
    )
    assert assistant_only_mask.assistant_mask_from_token_ids(plain_ids, derived)[0].tolist() == [1, 1, 1]
    strict_host = _DummyMaskHost(derived)
    strict_batch = {"input_ids": plain_ids.copy()}
    strict_host._assemble_per_dataset_mask(strict_batch, np.ones_like(plain_ids), global_flag=False)
    assert strict_batch["completion_mask"][0].tolist() == [0, 0, 0]


def test_audit193_16_sft_conversational_mask_fallback_and_init_warning() -> None:
    """#16: SFTPreprocessTransform warns when template lacks {% generation %}, retries without tools, and fails safe."""
    tok_no_gen = _SimpleVocabTokenizer()
    tok_no_gen.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    with pytest.warns(UserWarning, match="generation"):
        prompt_transforms.SFTPreprocessTransform(tok_no_gen, max_length=32, mask_prompt=True, padding=False)

    class _ToolsFailTokenizer(_SimpleVocabTokenizer):
        def __init__(self, fail_without_tools: bool = False) -> None:
            super().__init__()
            self.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
            self._fail_without_tools = fail_without_tools
            self.attempts: list[dict[str, Any]] = []

        def apply_chat_template(self, messages, **kwargs):
            self.attempts.append(dict(kwargs))
            if kwargs.get("return_assistant_tokens_mask"):
                if kwargs.get("tools"):
                    raise ValueError("template does not support tools")
                if self._fail_without_tools:
                    raise ValueError("masked rendering unavailable")
                return {
                    "input_ids": [1, 2, 3, 4],
                    "attention_mask": [1, 1, 1, 1],
                    "assistant_masks": [0, 0, 1, 1],
                }
            return "user: hi\nassistant: bye"

    # Case 1: tools fails -> retries return_assistant_tokens_mask=True WITHOUT tools and keeps assistant_masks!
    tok1 = _ToolsFailTokenizer(fail_without_tools=False)
    sft1 = prompt_transforms.SFTPreprocessTransform(tok1, max_length=16, mask_prompt=True, padding=False)
    out1 = sft1(
        {
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}],
            "tools": [{"name": "calc"}],
        }
    )
    assert out1["assistant_masks"] == [0, 0, 1, 1]

    # Case 2: even no-tools masked render fails -> emits all-zero assistant_masks + warning (never leaks prompt!).
    tok2 = _ToolsFailTokenizer(fail_without_tools=True)
    sft2 = prompt_transforms.SFTPreprocessTransform(tok2, max_length=16, mask_prompt=True, padding=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out2 = sft2({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]})
    assert "assistant_masks" in out2
    assert len(out2["assistant_masks"]) == len(out2["input_ids"])
    assert sum(out2["assistant_masks"]) == 0
    assert any("all-zero" in str(w.message) for w in caught)


def test_audit193_17_teardown_tensorboard_rich_pbar_iterators_and_packed_parquet() -> None:
    """#17: Teardown closes TensorBoard, stops Rich Progress thread, closes bucket/prefetch/grain iterators."""
    from easydel.trainers.metrics import RichProgressBar

    # 1. RichProgressBar.close() stops Progress when its last task is removed
    stopped = []

    class _FakeProgress:
        def __init__(self) -> None:
            self.tasks: list[int] = [1]

        def remove_task(self, tid) -> None:
            self.tasks.remove(tid)

        def stop(self) -> None:
            stopped.append(True)

    fp = _FakeProgress()
    RichProgressBar(fp, 1).close()
    assert stopped == [True]

    # 2. BaseTrainer.finish() closes TensorBoard, Rich pbar, bucket iters, eSurge, and clears W&B tables
    events: list[str] = []

    class _CloseableIter:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def close(self) -> None:
            events.append(f"close:{self.tag}")

    tr = object.__new__(Trainer)
    tr.arguments = types.SimpleNamespace(close_tensorboard=lambda: events.append("close:tb"))
    tr._hidden_rich_pbar = types.SimpleNamespace(stop=lambda: events.append("stop:rich"))
    tr._bucket_iters = {0: _CloseableIter("bucket0"), 1: _CloseableIter("bucket1")}
    tr.release_generation_runtime = lambda **kw: events.append(f"release:{kw.get('clear_esurge_compiled_cache')}")
    tr.preview_log_table = object()
    tr.benchmark_log_table = object()
    tr.training_generation_log_table = object()
    tr.finish()
    assert "stop:rich" in events
    assert "close:bucket0" in events and "close:bucket1" in events
    assert "close:tb" in events
    assert "release:True" in events
    assert not tr._bucket_iters
    assert tr.preview_log_table is None and not hasattr(tr, "_hidden_rich_pbar")

    # 3. Trainer.train() calls self.finish() even when an unhandled exception is raised
    finish_calls = []
    failing_tr = object.__new__(Trainer)
    failing_tr._runtime_trace = lambda *a, **kw: None
    failing_tr.start_training_hook = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    failing_tr.finish = lambda: finish_calls.append("finished")
    with pytest.raises(RuntimeError, match="boom"):
        failing_tr.train()
    assert finish_calls == ["finished"]

    # 4. PackedParquetLoader._iterate closes underlying row iterator on early close
    row_iter_closed = []

    class _TrackedRowIter:
        def __iter__(self):
            return self

        def __next__(self):
            return {"input_ids": [1, 2, 3]}

        def close(self) -> None:
            row_iter_closed.append(True)

    loader = object.__new__(data_lanes.PackedParquetLoader)
    loader.max_steps = 10
    loader.batch_size = 1
    loader.pack_seq_len = 4
    loader.embed_cap = 0
    loader.misfit_patience = 1
    loader.fingerprint = "fp"
    loader._lock = threading.Lock()
    loader._cursors = collections.OrderedDict()
    loader._cursor_history = 4
    loader._stream = lambda shift: []
    loader._row_iter = lambda stream, start: _TrackedRowIter()

    gen = loader._iterate(step=0, shift=0, cursor=0, carry=())
    first_batch = next(gen)
    assert len(first_batch) == 1
    gen.close()
    assert row_iter_closed == [True]


def test_audit193_18_esurge_error_cleanup_base_exception_and_padding_side_restore() -> None:
    """#18: eSurge setup failure routes via release_generation_runtime; KeyboardInterrupt cleans up; padding_side restored."""
    # (a) Setup failure (get_esurge raises before returning engine) calls release_generation_runtime with clear flag
    released: list[tuple[Any, bool]] = []
    tr = object.__new__(Trainer)
    tr.arguments = TrainingArguments(max_length=16, esurge_use_tqdm=False, esurge_hbm_utilization=0.5)
    tr._normalize_esurge_prompts = lambda prompts, apply_chat_template: list(prompts)
    tr.release_generation_runtime = lambda *, state=None, clear_esurge_compiled_cache=False, **kw: released.append(
        (state, clear_esurge_compiled_cache)
    )

    class _FailingSetupModel:
        def get_esurge(self, **kw):
            raise RuntimeError("init failed")

    st_setup = types.SimpleNamespace(model=_FailingSetupModel())
    with pytest.raises(RuntimeError, match="init failed"):
        tr._generate_via_esurge(
            input_ids=None,
            attention_mask=None,
            prompts=["hi"],
            processor=None,
            pad_token_id=0,
            sampling_params=types.SimpleNamespace(
                max_tokens=4,
                logprobs=None,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                n=1,
            ),
            state=st_setup,
            apply_chat_template=False,
            all_gather=False,
            release_runtime_after_generation=True,
            clear_esurge_compiled_cache_after_generation=True,
        )
    assert released == [(st_setup, True)]

    # (b) KeyboardInterrupt during engine.generate cleans up engine (pause + release_model_state)
    engine_events: list[tuple[str, Any]] = []

    class _InterruptEngine:
        def pause(self) -> None:
            engine_events.append(("pause", None))

        def release_model_state(self, *, clear_compiled_cache: bool = False) -> None:
            engine_events.append(("release", clear_compiled_cache))

    class _InterruptModel:
        def get_esurge(self, **kw):
            return _InterruptEngine()

        def _call_esurge_engine(self, engine, **kw):
            raise KeyboardInterrupt("user interrupt")

    with pytest.raises(KeyboardInterrupt):
        tr._generate_via_esurge(
            input_ids=None,
            attention_mask=None,
            prompts=["hi"],
            processor=None,
            pad_token_id=0,
            sampling_params=types.SimpleNamespace(
                max_tokens=4,
                logprobs=None,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                n=1,
            ),
            state=types.SimpleNamespace(model=_InterruptModel()),
            apply_chat_template=False,
            all_gather=False,
            release_runtime_after_generation=True,
            clear_esurge_compiled_cache_after_generation=True,
        )
    assert engine_events == [("pause", None), ("release", True)]

    # (c) _generate_via_compiled restores processor.padding_side even when tokenization raises
    class _FailingProcessor:
        padding_side = "right"

        def encode(self, *a, **kw):
            return [1]

        def __call__(self, *a, **kw):
            raise RuntimeError("tokenize failed")

    proc = _FailingProcessor()
    with pytest.raises(RuntimeError, match="tokenize failed"):
        tr._generate_via_compiled(
            input_ids=None,
            attention_mask=None,
            prompts=["hello"],
            processor=proc,
            pad_token_id=0,
            sampling_params=None,
            state=None,
            apply_chat_template=False,
            generation_config=None,
            shard_inputs=False,
            config_overrides=None,
            normalized_model_kwargs=None,
            has_model_kwargs=False,
            all_gather=False,
            release_runtime_after_generation=True,
            clear_esurge_compiled_cache_after_generation=False,
            generate_kwargs=None,
        )
    assert proc.padding_side == "right"


def test_audit193_23_release_generation_runtime_and_scope_sharing_for_aux_states() -> None:
    """#23: release_generation_runtime releases ref/teacher states; scope sharing aligns compatible models."""
    paused: list[tuple[str, bool, bool]] = []

    def _make_state(name: str, scope: str, sig: str, mesh: str = "m0", shape: tuple[int, ...] = (2,)):
        class _M:
            def __init__(self) -> None:
                self.mesh = mesh

            def _source_layout_signature_for_esurge_metadata(self) -> str:
                return sig

            def pause_esurge(self, *, release_model_state: bool = False, clear_compiled_cache: bool = False) -> None:
                paused.append((name, release_model_state, clear_compiled_cache))

        @dataclass
        class _S:
            graphdef: Any
            graphstate: Any
            graphother: Any
            esurge_cache_scope_key: str
            model: Any

            def replace(self, **kw):
                return _S(
                    graphdef=kw.get("graphdef", self.graphdef),
                    graphstate=kw.get("graphstate", self.graphstate),
                    graphother=kw.get("graphother", self.graphother),
                    esurge_cache_scope_key=kw.get("esurge_cache_scope_key", self.esurge_cache_scope_key),
                    model=self.model,
                )

        return _S(
            graphdef=object(),
            graphstate={"w": jnp.zeros(shape, dtype=jnp.float32)},
            graphother={},
            esurge_cache_scope_key=scope,
            model=_M(),
        )

    tr = object.__new__(Trainer)
    pol = _make_state("policy", "scope-pol", "sig-llama")
    ref_same = _make_state("ref", "scope-ref", "sig-llama")
    teacher_diff = _make_state("teacher", "scope-teacher", "sig-qwen", shape=(4,))

    tr.model_state = pol
    # Different graphdef objects, same layout signature + mesh + leaf shapes -> shares policy scope!
    aligned_ref = tr._share_esurge_cache_scope_if_compatible(ref_same)
    assert aligned_ref.esurge_cache_scope_key == "scope-pol"
    # Different layout signature -> keeps separate scope!
    unaligned_teacher = tr._share_esurge_cache_scope_if_compatible(teacher_diff)
    assert unaligned_teacher.esurge_cache_scope_key == "scope-teacher"

    tr.ref_state = aligned_ref
    tr.teacher_state = unaligned_teacher
    tr.release_generation_runtime(clear_esurge_compiled_cache=True)
    # Policy and aligned_ref share scope-pol -> deduplicated (1 pause), teacher paused separately (1 pause)!
    assert paused == [("policy", True, True), ("teacher", True, True)]


def test_audit193_25_pose_seq_lens_preserves_interior_and_leading_separators() -> None:
    """#25: seq_lens_from_segment_ids keeps leading/interior <0 runs and extract_position_batch skips redundant compute."""
    seg = np.array([[-1, 0, 0, -1, 1, 1, 1, -1, -1]], dtype=np.int32)
    attn = np.array([[0, 1, 1, 0, 1, 1, 1, 0, 0]], dtype=np.int32)
    pos = np.array([[0, 0, 1, 0, 0, 1, 2, 0, 0]], dtype=np.int32)

    seq_lens, seq_lens_num = pose.seq_lens_from_segment_ids(seg)
    assert seq_lens_num.tolist() == [4]
    assert seq_lens[0, :4].tolist() == [1, 2, 1, 3]

    via_seg = pose.document_segment_ranges(0, 7, pos, None, None, None, None, attention_mask=attn, segment_ids=seg)
    via_lens = pose.document_segment_ranges(
        0, 7, pos, seq_lens, seq_lens_num, None, None, attention_mask=attn, segment_ids=None
    )
    assert via_seg == [(1, 3), (4, 7)]
    assert via_lens == via_seg

    # extract_position_batch skips redundant seq_lens when segment_ids is present
    _, meta = pose.extract_position_batch({"position_ids": pos, "segment_ids": seg}, feature="pose")
    assert meta["segment_ids"] is not None
    assert meta["seq_lens"] is None and meta["seq_lens_num"] is None


def test_audit193_26_packer_masks_first_active_token_after_leading_and_internal_pad() -> None:
    """#26: packer.pack_sequences masks any active token preceded by PAD (leading or internal)."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[0, 0, 10, 11], [20, 21, 0, 22, 23]],
            "attention_mask": [[0, 0, 1, 1], [1, 1, 0, 1, 1]],
            "completion_mask": [[0, 0, 1, 1], [1, 1, 0, 1, 1]],
            "assistant_masks": [[0, 0, 1, 1], [1, 1, 0, 1, 1]],
        }
    )
    packed = packer.pack_sequences(ds, max_length=12, pad_token_id=0, reset_position_ids=True)
    assert len(packed) == 1
    row = packed[0]
    # Doc 0: [0, 0, 10, 11] -> idx 2 (10) follows PAD idx 1 -> MUST be masked! idx 3 (11) kept!
    assert row["labels"][:4] == [-100, -100, -100, 11]
    assert row["completion_mask"][:4] == [0, 0, 0, 1]
    assert row["assistant_masks"][:4] == [0, 0, 0, 1]
    # Sep at idx 4; Doc 1 at idx 5..9: [20, 21, 0, 22, 23] -> idx 5 (after sep) & idx 8 (22, after internal PAD idx 7) masked!
    assert row["labels"][5:10] == [-100, 21, -100, -100, 23]
    assert row["completion_mask"][5:10] == [0, 1, 0, 0, 1]
    assert row["assistant_masks"][5:10] == [0, 1, 0, 0, 1]


def test_audit193_27_extract_prompt_strips_messages_and_bco_avoids_double_templating() -> None:
    """#27: extract_prompt_from_preference strips 'messages'; paired BCOPreprocessTransform doesn't double-template."""
    row_with_msgs = {
        "prompt": "Q?",
        "chosen": "good",
        "rejected": "bad",
        "messages": [{"role": "user", "content": "stale"}],
    }
    extracted = prompt_transforms.extract_prompt_from_preference(row_with_msgs)
    assert "messages" not in extracted
    assert extracted["prompt"] == "Q?"

    # Paired BCO with JSON-like completion string: must NOT re-run maybe_convert_to_chatml/maybe_apply_chat_template
    template_calls = 0

    class _CountingTok(_SimpleVocabTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            nonlocal template_calls
            template_calls += 1
            return super().apply_chat_template(messages, **kwargs)

    tok = _CountingTok()
    bco = prompt_transforms.BCOPreprocessTransform(tok, max_prompt_length=32, max_completion_length=32)
    paired_row = {
        "prompt": [{"role": "user", "content": "return json"}],
        "chosen": [{"role": "assistant", "content": '[{"role": "user", "content": "literal json"}]'}],
        "rejected": [{"role": "assistant", "content": "plain bad"}],
    }
    outputs = list(bco(paired_row))
    assert len(outputs) == 2
    assert [o["label"] for o in outputs] == [True, False]
    # Only the paired step called apply_chat_template (prompt + chosen + rejected), NOT again in unpaired tokenization!
    assert template_calls == 3


def test_audit193_28_merge_methods_hoisted_jit_reuses_single_compilation(monkeypatch) -> None:
    """#28: _topk_density_mask and _dare_ties_merge reuse hoisted module-level JIT helpers without per-leaf recompile."""
    from easydel.trainers.merge_callback import merge_methods

    mesh = Mesh(np.array(jax.devices()[:1]), ("dp",))
    sharding = NamedSharding(mesh, PartitionSpec("dp"))
    p = jax.device_put(jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float32), sharding)
    t = jax.device_put(jnp.linspace(1.0, -1.0, 8, dtype=jnp.float32), sharding)

    # 1. Verify _topk_density_mask_sharded compiles once across repeated leaves & preserves sharding
    topk_traces = 0
    orig_topk_impl = merge_methods._topk_density_mask_impl

    def _counting_topk(values, keep):
        nonlocal topk_traces
        topk_traces += 1
        return orig_topk_impl(values, keep)

    monkeypatch.setattr(merge_methods, "_topk_density_mask_impl", _counting_topk)
    merge_methods._topk_density_mask_sharded.clear_cache()
    m1 = merge_methods._topk_density_mask(t - p, 0.5)
    m2 = merge_methods._topk_density_mask((t - p) * 2.0, 0.25)
    assert topk_traces == 1
    assert m1.sharding == sharding and m2.sharding == sharding

    # 2. Verify _dare_ties_sharded compiles once across different leaf_index/seed values & preserves sharding/dtype
    dare_traces = 0
    orig_dare_apply = merge_methods._dare_ties_apply

    def _counting_dare(p_arg, t_arg, seed_arg, density_arg, weight_arg, policy_sharding):
        nonlocal dare_traces
        dare_traces += 1
        return orig_dare_apply(p_arg, t_arg, seed_arg, density_arg, weight_arg, policy_sharding)

    monkeypatch.setattr(
        merge_methods,
        "_dare_ties_sharded",
        jax.jit(_counting_dare, static_argnames=("policy_sharding",)),
    )
    o1 = merge_methods._dare_ties_merge(
        p,
        t,
        merge_methods.MergeLeafContext(
            policy_weight=0.5, target_weight=0.5, t_value=0.5, density=0.5, seed=7, leaf_index=0
        ),
    )
    o2 = merge_methods._dare_ties_merge(
        p,
        t,
        merge_methods.MergeLeafContext(
            policy_weight=0.5, target_weight=0.5, t_value=0.5, density=0.5, seed=7, leaf_index=1
        ),
    )
    assert dare_traces == 1
    assert o1.sharding == sharding and o2.sharding == sharding
    assert o1.dtype == p.dtype


def test_audit193_30_process_encoder_handles_python_set_image_token_id() -> None:
    """#30: process_encoder converts Python set/frozenset image_token_id before np.isin on host input_ids."""
    from easydel.trainers.process_encoder import EncoderBinding, EncoderProcessor, ProcessEncoderConfig

    class _SetTokenVLM:
        _multimodal_merge_feature = types.SimpleNamespace(image_token_id={99, 100})

        def get_image_features(self, pixel_values):
            # Return 2 genuine image features matching the 2 placeholder tokens in input_ids
            return jnp.ones((pixel_values.shape[0], 2, 4), dtype=jnp.float32)

    binding = EncoderBinding(
        tower_attr="visual",
        feature_kwarg="image_embeds",
        feature_fn_name="get_image_features",
        accepted_kwargs=frozenset(),
        feature_valid_length_kwarg=None,
    )
    proc = EncoderProcessor(None, ProcessEncoderConfig(enabled=True), binding=binding)
    batch = {
        "input_ids": np.array([[1, 99, 100, 2]], dtype=np.int32),
        "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
    }
    out = proc.process(batch, model=_SetTokenVLM())
    assert "image_embeds" in out and "pixel_values" not in out
    assert out["image_embeds"].shape == (2, 4)


def test_audit193_33_sft_formatting_func_string_not_discarded_when_messages_present() -> None:
    """#33: SFTPreprocessTransform uses formatting_func string output even when row also has 'messages'."""
    tok = _SimpleVocabTokenizer()
    sft = prompt_transforms.SFTPreprocessTransform(
        tok,
        max_length=32,
        padding=False,
        formatting_func=lambda ex: f"CUSTOM_MARKER {ex['tag']}",
    )
    row = {
        "tag": "alpha",
        "messages": [{"role": "user", "content": "ignored_prompt"}, {"role": "assistant", "content": "ignored_reply"}],
    }
    out = sft(row)
    plain_sft = prompt_transforms.SFTPreprocessTransform(tok, max_length=32, padding=False)
    expected_from_text = plain_sft({"text": "CUSTOM_MARKER alpha"})
    assert out["input_ids"] == expected_from_text["input_ids"]


def test_audit194_reg2_tokenizer_int_attr_safe_without_processing_class() -> None:
    """Reg #2: BaseTrainer._tokenizer_int_attr returns default when processing_class attribute is absent."""
    stub = object.__new__(Trainer)
    assert not hasattr(stub, "processing_class")
    assert stub._tokenizer_int_attr("pad_token_id", 42) == 42
    assert stub._tokenizer_int_attr("eos_token_id", None) is None


def test_audit228_1_packer_and_global_mask_apply_assistant_masks_to_labels() -> None:
    """#228-1: pack_sequences masks synthesized labels from assistant_masks/completion_mask; global_flag syncs labels."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[10, 11, 12, 13, 14]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "assistant_masks": [[0, 0, 0, 1, 1]],
        }
    )
    packed = packer.pack_sequences(ds, max_length=8, pad_token_id=0, reset_position_ids=True)[0]
    assert packed["assistant_masks"] == [0, 0, 0, 1, 1, 0, 0, 0]
    assert packed["labels"] == [-100, -100, -100, 13, 14, -100, -100, -100]

    # Even if a batch arrives with unmasked labels + assistant_masks under global assistant_only_loss=True,
    # _apply_assistant_supervision must still mask labels where completion_mask == 0.
    class _GlobalOnlyHost(assistant_only_mask.AssistantOnlyMaskMixin):
        def __init__(self) -> None:
            self.arguments = types.SimpleNamespace(
                assistant_only_loss=True,
                zero_supervision_abort_fraction=None,
                zero_supervision_report_fraction=None,
            )

    host = _GlobalOnlyHost()
    unmasked_labels = np.array([[10, 11, 12, 13, 14]], dtype=np.int32)
    snapshot = unmasked_labels.copy()
    batch = {
        "input_ids": np.array([[10, 11, 12, 13, 14]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int32),
        "assistant_masks": np.array([[0, 0, 0, 1, 1]], dtype=np.int32),
        "labels": unmasked_labels,
    }
    out = host._apply_assistant_supervision(batch, {})
    assert np.array_equal(unmasked_labels, snapshot)  # original not mutated
    assert out["completion_mask"][0].tolist() == [0, 0, 0, 1, 1]
    assert out["labels"][0].tolist() == [-100, -100, -100, 13, 14]


def test_audit228_2_eval_wraparound_and_train_epoch_exception_close_replacement_iterators() -> None:
    """#228-2: _run_evaluation/_eval_epoch wraparound and _train_epoch exceptions close replacement iterators."""
    from easydel.trainers.metrics import NullProgressBar

    created: list[int] = []
    closed: list[int] = []

    class _TrackedLoader:
        def __init__(self, batches_per_iter: int = 1) -> None:
            self._batches_per_iter = batches_per_iter
            self._next_id = 0

        def __iter__(self):
            self._next_id += 1
            iter_id = self._next_id
            created.append(iter_id)
            remaining = [iter_id] * self._batches_per_iter

            class _It:
                def __iter__(self_it):
                    return self_it

                def __next__(self_it):
                    if not remaining:
                        raise StopIteration
                    return {"input_ids": np.array([[remaining.pop()]], dtype=np.int32)}

                def close(self_it):
                    closed.append(iter_id)

            return _It()

    # 1. _run_evaluation / _eval_epoch wraparound (1 batch per iter, 2 eval steps -> creates 1 & 2, closes 1 & 2)
    tr = object.__new__(Trainer)
    tr.arguments = TrainingArguments(
        max_length=4,
        total_batch_size=1,
        use_wandb=False,
        report_metrics=False,
        log_all_workers=True,
    )
    tr.max_evaluation_steps = 2
    tr.dataloader_eval = _TrackedLoader(batches_per_iter=1)
    tr.data_collator = None
    tr._model = types.SimpleNamespace(mesh=contextlib.nullcontext())
    tr.evalu_tracker = types.SimpleNamespace(trace_compilation=lambda: contextlib.nullcontext())
    tr._forward_flops_per_token = 0.0
    tr._extra_forward_flops_per_token = 0.0
    tr.create_progress_bar = lambda **kw: NullProgressBar()
    tr._execute_eval_step = lambda st, b: LossMetrics(loss=jnp.float32(0.5))
    tr.log_metrics = lambda *a, **kw: None

    eval_out = list(
        tr._run_evaluation(
            state=types.SimpleNamespace(step=jnp.asarray(1)),
            metrics_tracker=MetricsTracker(),
            step_metrics=StepMetrics(tr.arguments),
        )
    )
    assert len(eval_out) == 2
    assert created == [1, 2]
    assert closed == [1, 2]

    # 2. _train_epoch uncaught exception after wraparound closes replacement train_iter
    created.clear()
    closed.clear()
    loader = _TrackedLoader(batches_per_iter=1)
    tr2 = object.__new__(Trainer)
    tr2.arguments = TrainingArguments(
        max_length=4,
        total_batch_size=1,
        use_wandb=False,
        report_metrics=False,
        log_all_workers=True,
        dataloader_prefetch=False,
    )
    tr2.max_training_steps = 3
    tr2.data_collator = None
    tr2._has_buckets = False
    tr2._buckets = []
    tr2._bucket_iters = {}
    tr2._backward_flops_per_token = 0.0
    tr2._extra_backward_flops_per_token = 0.0
    tr2.scheduler = None
    tr2.train_tracker = types.SimpleNamespace(trace_compilation=lambda: contextlib.nullcontext())
    tr2._runtime_trace = lambda *a, **kw: None
    tr2._log_mesh_topology = lambda: None
    tr2._apply_pose_to_batch = lambda b, s: b
    tr2._apply_randomized_yarn_to_batch = lambda b, s: b
    tr2._apply_process_encoder_to_batch = lambda b, s, state=None: b
    tr2.on_step_start = lambda state, step: state
    tr2.on_step_end = lambda state, metrics, step: (state, metrics)
    tr2.apply_training_hooks = lambda metrics, loss_value=None: metrics
    tr2._maybe_start_profiler = lambda step: None
    tr2._maybe_stop_profiler = lambda step: None
    tr2._should_save_tpu_preemption_checkpoint = lambda step: False
    tr2._should_save_checkpoint = lambda step: False
    tr2._save_checkpoint_for_step = lambda *a, **kw: None
    tr2._should_run_evaluation = lambda step: False
    tr2.log_metrics = lambda *a, **kw: None
    tr2.log_weight_distribution = lambda *a, **kw: None
    tr2.log_watchers = lambda *a, **kw: None
    tr2.maybe_generate = lambda *a, **kw: None
    tr2.maybe_benchmark = lambda *a, **kw: None

    step_calls = 0

    def _boom_on_step2(state, batch):
        nonlocal step_calls
        del batch
        step_calls += 1
        if step_calls == 2:
            raise RuntimeError("uncaught step boom")
        return types.SimpleNamespace(step=jnp.asarray(int(state.step) + 1)), LossMetrics(loss=jnp.float32(0.5)), None

    tr2._execute_train_step = _boom_on_step2
    initial_it = iter(loader)
    with pytest.raises(RuntimeError, match="uncaught step boom"):
        tr2._train_epoch(
            state=types.SimpleNamespace(step=jnp.asarray(0)),
            train_dataset=loader,
            train_iter=initial_it,
            metrics_tracker=MetricsTracker(),
            step_metrics=StepMetrics(tr2.arguments),
            pbar=NullProgressBar(),
            epoch=0,
            epoch_start_step=0,
            epoch_end_step=3,
        )
    # Iter #1 closed during wraparound by _get_next_batch; Iter #2 closed by _train_epoch's except BaseException!
    assert created == [1, 2]
    assert closed == [1, 2]


def test_audit228_3_sft_singular_conversation_key_and_map_batch_mask_fallback() -> None:
    """#228-3: formatting_func pops singular 'conversation'; map_batch emits zero assistant_masks + warning on fallback."""
    # (a) #33 singular 'conversation' key must not override formatting_func string output
    tok = _SimpleVocabTokenizer()
    sft = prompt_transforms.SFTPreprocessTransform(
        tok,
        max_length=32,
        padding=False,
        formatting_func=lambda ex: f"CUSTOM_CONV {ex['tag']}",
    )
    row = {
        "tag": "beta",
        "conversation": [{"from": "human", "value": "ignored"}, {"from": "gpt", "value": "ignored_reply"}],
    }
    plain_sft = prompt_transforms.SFTPreprocessTransform(tok, max_length=32, padding=False)
    assert sft(row)["input_ids"] == plain_sft({"text": "CUSTOM_CONV beta"})["input_ids"]

    # (b) #16 map_batch template fallback with mask_prompt=True must emit all-zero assistant_masks + warning
    class _AlwaysFailTemplateTok(_SimpleVocabTokenizer):
        def __init__(self) -> None:
            super().__init__()
            self.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

        def apply_chat_template(self, messages, **kwargs):
            raise ValueError("broken chat template")

    tok_fail = _AlwaysFailTemplateTok()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sft_masked = prompt_transforms.SFTPreprocessTransform(tok_fail, max_length=32, mask_prompt=True, padding=False)

    example = {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
    with warnings.catch_warnings(record=True) as single_warns:
        warnings.simplefilter("always")
        single_out = sft_masked(example)
    with warnings.catch_warnings(record=True) as batch_warns:
        warnings.simplefilter("always")
        batch_out = sft_masked.map_batch([example, example])

    assert "assistant_masks" in single_out and sum(single_out["assistant_masks"]) == 0
    assert any("all-zero" in str(w.message) for w in single_warns)
    assert len(batch_out) == 2
    for row_out in batch_out:
        assert "assistant_masks" in row_out
        assert len(row_out["assistant_masks"]) == len(row_out["input_ids"])
        assert sum(row_out["assistant_masks"]) == 0
    assert any("all-zero" in str(w.message) for w in batch_warns)
