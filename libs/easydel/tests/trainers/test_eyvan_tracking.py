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

"""CPU-only regression and integration tests for Eyvan trainer tracking & progress reporting."""

from __future__ import annotations

import math
import os
import threading
import types
import warnings
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from easydel.infra.elarge.types import BASE_TRAINER_DEFAULTS, normalize_trainer_config
from easydel.trainers import training_configurations as tc
from easydel.trainers.trainer import Trainer
from easydel.trainers.training_configurations import (
    TrainingArguments,
    _sanitize_eyvan_key,
    _sanitize_eyvan_slug,
    _sanitize_eyvan_tags,
    _to_eyvan_scalar,
    is_eyvan_auto_track_enabled,
    set_eyvan_auto_track_enabled,
)
from eyvan.track._format import decode_segment
from eyvan.track._store import store_for_uri

# Pure-Python + NumPy + local Eyvan store tests; no JAX device array computation runs.
pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def _clean_env_and_no_dist(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    for key in (
        "EYVAN_TRACKING_URI",
        "EYVAN_WORKLOAD_ID",
        "EYVAN_PROGRESS_URL",
        "EYVAN_MANAGED_RESULT_TOKEN",
        "EYVAN_DISTRIBUTED_PROCESS_ID",
        "JAX_PROCESS_ID",
        "EYVAN_ATTEMPT",
    ):
        monkeypatch.delenv(key, raising=False)


def _read_store_stream(store_dir: Path, run_id: str, stream: str) -> list[dict[str, object]]:
    store = store_for_uri(str(store_dir))
    records: list[dict[str, object]] = []
    for key in store.list(f"{run_id}/{stream}"):
        records.extend(decode_segment(store.get(key)))
    return records


def _make_args(**kwargs) -> TrainingArguments:
    defaults = {
        "model_name": "org/tiny-model",
        "use_wandb": False,
        "save_directory": None,
    }
    defaults.update(kwargs)
    return TrainingArguments(**defaults)


def test_eyvan_activation_auto_explicit_and_performance_mode(monkeypatch, tmp_path):
    # Under pytest default isolation, use_eyvan_track=None without URI/env stays hermetically off
    assert is_eyvan_auto_track_enabled() is False
    args_isolated = _make_args()
    assert args_isolated.use_eyvan_track is None
    assert args_isolated.eyvan_tracking_enabled is False
    assert args_isolated.get_eyvan_track() is None

    # Outside pytest isolation (auto-track enabled), use_eyvan_track=None auto-enables whenever eyvan is importable
    with set_eyvan_auto_track_enabled(True):
        assert is_eyvan_auto_track_enabled() is True
        args_auto_on = _make_args()
        assert args_auto_on.use_eyvan_track is None
        assert args_auto_on.eyvan_tracking_enabled is True

        # Explicit False disables cleanly even when auto-track is enabled
        args_auto_off = _make_args(use_eyvan_track=False)
        assert args_auto_off.eyvan_tracking_enabled is False
        assert args_auto_off.get_eyvan_track() is None

        # performance_mode=True and report_metrics=False disable cleanly even when auto-track is enabled
        assert _make_args(performance_mode=True).eyvan_tracking_enabled is False
        assert _make_args(report_metrics=False).eyvan_tracking_enabled is False

        # Missing eyvan dependency disables auto-ON silently without crashing
        with monkeypatch.context() as patch:
            patch.setattr(tc, "_import_eyvan_track", lambda: None)
            assert _make_args().eyvan_tracking_enabled is False
            assert _make_args().get_eyvan_track() is None

    # Imperative toggle + .reset() restores previous ContextVar state cleanly
    assert is_eyvan_auto_track_enabled() is False
    tok = set_eyvan_auto_track_enabled(True)
    try:
        assert is_eyvan_auto_track_enabled() is True
    finally:
        tok.reset()
    assert is_eyvan_auto_track_enabled() is False

    # Auto-enabled via explicit eyvan_track_uri
    args_uri = _make_args(eyvan_track_uri=str(tmp_path / "uri_store"))
    assert args_uri.eyvan_tracking_enabled is True

    # Explicit False wins over eyvan_track_uri and Eyvan env vars
    monkeypatch.setenv("EYVAN_TRACKING_URI", str(tmp_path / "env_store"))
    monkeypatch.setenv("EYVAN_WORKLOAD_ID", "wl-auto-1")
    args_forced_off = _make_args(use_eyvan_track=False, eyvan_track_uri=str(tmp_path / "uri_store"))
    assert args_forced_off.eyvan_tracking_enabled is False
    assert args_forced_off.get_eyvan_track() is None

    # Auto-enabled via EYVAN_TRACKING_URI
    args_env_uri = _make_args()
    assert args_env_uri.eyvan_tracking_enabled is True

    # Auto-enabled via EYVAN_WORKLOAD_ID alone
    monkeypatch.delenv("EYVAN_TRACKING_URI")
    args_env_wl = _make_args()
    assert args_env_wl.eyvan_tracking_enabled is True

    # Explicit True outside Eyvan
    monkeypatch.delenv("EYVAN_WORKLOAD_ID")
    args_forced_on = _make_args(use_eyvan_track=True)
    assert args_forced_on.eyvan_tracking_enabled is True

    # performance_mode=True disables Eyvan tracking (both explicit True and auto)
    monkeypatch.setenv("EYVAN_TRACKING_URI", str(tmp_path / "perf_store"))
    args_perf_true = _make_args(use_eyvan_track=True, performance_mode=True)
    assert args_perf_true.use_eyvan_track is False
    assert args_perf_true.eyvan_tracking_enabled is False
    assert args_perf_true.get_eyvan_track() is None

    args_perf_auto = _make_args(use_eyvan_track=None, performance_mode=True)
    assert args_perf_auto.use_eyvan_track is False
    assert args_perf_auto.eyvan_tracking_enabled is False

    # report_metrics=False disables tracking through can_log_metrics
    args_no_report = _make_args(use_eyvan_track=True, report_metrics=False)
    assert args_no_report.eyvan_tracking_enabled is False


def test_eyvan_missing_dependency_warns_only_when_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(tc, "_import_eyvan_track", lambda: None)

    args_true = _make_args(use_eyvan_track=True)
    with pytest.warns(UserWarning, match="use_eyvan_track=True"):
        assert args_true.get_eyvan_track() is None
    # Subsequent call on the same instance does not spam warnings
    assert args_true.get_eyvan_track() is None
    # log_metrics remains a safe no-op
    args_true.log_metrics({"train/loss": 0.5}, step=1)

    args_false = _make_args(use_eyvan_track=False)
    assert args_false.get_eyvan_track() is None


def test_eyvan_config_validation_and_sanitizers():
    for bad_interval in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eyvan_track_flush_interval_s"):
            _make_args(eyvan_track_flush_interval_s=bad_interval)

    with pytest.raises(TypeError, match="eyvan_track_flush_interval_s"):
        _make_args(eyvan_track_flush_interval_s=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="use_eyvan_track"):
        _make_args(use_eyvan_track="yes")  # type: ignore[arg-type]

    # Slug sanitization
    assert _sanitize_eyvan_slug("EasyDeL-SFT") == "EasyDeL-SFT"
    assert _sanitize_eyvan_slug("  ///my project!  ") == "my-project"
    assert _sanitize_eyvan_slug("...---") == "easydel"
    assert len(_sanitize_eyvan_slug("a" * 200)) == 128

    # Key sanitization preserves valid keys, cleans spaces/parens/commas, deduplicates >128 char truncations
    assert _sanitize_eyvan_key("train/loss-1.0_total") == "train/loss-1.0_total"
    assert _sanitize_eyvan_key("Number of Model Parameters (Billion)") == "Number_of_Model_Parameters_Billion"
    assert _sanitize_eyvan_key("benchmark/core/hellaswag/acc,none") == "benchmark/core/hellaswag/acc_none"
    assert _sanitize_eyvan_key("   ") == "metric"
    long_a = "weights-information/" + ("layer." * 30) + "mean"
    long_b = "weights-information/" + ("layer." * 30) + "std"
    s_a, s_b = _sanitize_eyvan_key(long_a), _sanitize_eyvan_key(long_b)
    assert len(s_a) <= 128 and len(s_b) <= 128
    assert s_a != s_b

    # Tag sanitization: merges defaults, deduplicates, enforces 64-tag / 64-char limits
    tags = _sanitize_eyvan_tags(["EasyDeL", "  custom  ", "", "x" * 80, *[f"t{i}" for i in range(100)]])
    assert tags[:4] == ("EasyDeL", "Jax", "Train", "custom")
    assert len(tags) == 64
    assert all(1 <= len(t) <= 64 for t in tags)
    assert len(set(tags)) == len(tags)


def test_eyvan_end_to_end_scalars_histograms_summary_config_and_close(tmp_path):
    store_dir = tmp_path / "eyvan_store"
    args = _make_args(
        model_name="acme/llm-1b",
        trainer_prefix="SFT",
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_project="My Project!",
        eyvan_track_name="sft-exp-1",
        eyvan_track_group="sweep-a",
        eyvan_track_tags=["custom", "EasyDeL"],
        eyvan_track_flush_interval_s=0.2,
    )

    run = args.get_eyvan_track()
    assert run is not None
    assert args.get_eyvan_track() is run
    run_id = run.id
    assert run.name == "sft-exp-1"
    assert run.group == "sweep-a"
    assert {"easydel", "jax", "sft", "llm-1b", "custom", "EasyDeL"} <= set(run.tags)
    assert len(run.tags) <= 64
    assert len(set(run.tags)) == len(run.tags)

    # Config logging with keys requiring sanitization
    args.log_metrics(
        {
            "Number of Model Parameters (Billion)": 1.25,
            "device_count": np.int64(8),
        },
        step=0,
        log_as="config",
    )

    # Regular step: scalars (including bool, numpy scalar, 0-d array, NaN), valid pre-binned histogram,
    # invalid-edge histogram (fallback), oversized-bin histogram (fallback), and raw 1D array
    valid_counts = [2, 5, 3]
    valid_edges = [0.0, 1.0, 2.0, 3.0]
    flat_edges_counts = [4, 6]
    flat_edges = [1.0, 1.0, 1.0]
    oversized_counts = np.ones(600, dtype=np.int64)
    oversized_edges = np.linspace(-1.0, 1.0, 601)
    raw_arr = np.asarray([-1.5, 0.0, 0.5, 2.5], dtype=np.float32)
    bf16_arr = np.asarray([-0.5, 0.25, 1.5], dtype=tc.jnp.bfloat16)

    args.log_metrics(
        {
            "train/loss": np.float32(0.75),
            "train/grad_norm/layer0": np.asarray(1.25),
            "train/flag": True,
            "train/nan_probe": float("nan"),
            "benchmark/core/hellaswag/acc,none": 0.82,
            "weights-histogram/valid": (valid_counts, valid_edges),
            "weights-histogram/flat_edges": (flat_edges_counts, flat_edges),
            "weights-histogram/oversized": (oversized_counts, oversized_edges),
            "activations/raw": raw_arr,
            "activations/bf16": bf16_arr,
            "ignored_none": None,
        },
        step=10,
    )

    # Summary logging
    args.log_metrics({"best eval/loss": 0.42, "steps": np.int32(10)}, step=10, log_as="summary")

    # Close idempotently & flush all segments to disk without leaking background threads
    args.close_eyvan_track(status="finished")
    assert run.finished is True
    assert "_eyvan_track_run" not in args.__dict__
    assert not any(t.name == f"eyvan-track-{run_id}" and t.is_alive() for t in threading.enumerate())
    # Second close is a clean no-op
    args.close_eyvan_track(status="finished")

    # Verify decoded NDJSON segments on disk
    meta_records = _read_store_stream(store_dir, run_id, "meta")
    meta_ops = [rec["op"] for rec in meta_records]
    assert "start" in meta_ops
    assert "config" in meta_ops
    assert "summary" in meta_ops
    assert "finish" in meta_ops
    finish_rec = [rec for rec in meta_records if rec["op"] == "finish"][-1]
    assert finish_rec["status"] == "finished"

    config_rec = [rec for rec in meta_records if rec["op"] == "config"][-1]
    assert config_rec["config"]["Number_of_Model_Parameters_Billion"] == pytest.approx(1.25)
    assert config_rec["config"]["device_count"] == 8

    summary_rec = [rec for rec in meta_records if rec["op"] == "summary"][-1]
    assert summary_rec["summary"]["best_eval/loss"] == pytest.approx(0.42)
    assert summary_rec["summary"]["steps"] == 10

    scalar_records = _read_store_stream(store_dir, run_id, "scalars")
    assert len(scalar_records) == 1
    assert scalar_records[0]["step"] == 10
    logged_v = scalar_records[0]["v"]
    assert logged_v["train/loss"] == pytest.approx(0.75)
    assert logged_v["grad_norm/layer0"] == pytest.approx(1.25)
    assert logged_v["train/flag"] == 1.0
    assert logged_v["train/nan_probe"] == "nan"
    assert logged_v["benchmark/core/hellaswag/acc_none"] == pytest.approx(0.82)

    hist_records = _read_store_stream(store_dir, run_id, "hist")
    by_key = {rec["key"]: rec for rec in hist_records}
    assert set(by_key) == {
        "weights-histogram/valid",
        "weights-histogram/flat_edges",
        "weights-histogram/oversized",
        "activations/raw",
        "activations/bf16",
    }
    assert by_key["weights-histogram/valid"]["counts"] == [2, 5, 3]
    assert by_key["weights-histogram/valid"]["edges"] == [0.0, 1.0, 2.0, 3.0]
    assert by_key["weights-histogram/flat_edges"]["n"] == 10
    assert by_key["weights-histogram/oversized"]["n"] > 0
    assert by_key["activations/raw"]["n"] == 4
    assert by_key["activations/bf16"]["n"] == 3


def test_eyvan_run_name_fallback_and_multihost_ranks(monkeypatch, tmp_path):
    store_dir = tmp_path / "ranks_store"

    # Name falls back to wandb_name when eyvan_track_name is None
    args_wb = _make_args(
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        wandb_name="from-wandb-name",
    )
    run_wb = args_wb.get_eyvan_track()
    assert run_wb is not None
    assert run_wb.name == "from-wandb-name"
    args_wb.close_eyvan_track()

    # Name falls back to a cached W&B-style auto name when both eyvan_track_name and wandb_name are None
    args_auto = _make_args(
        model_name="Qwen/Qwen3-8B",
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        total_batch_size=16,
    )
    run_auto = args_auto.get_eyvan_track()
    assert run_auto is not None
    assert run_auto.name.startswith("qwen3-8b-")
    assert len(run_auto.name) <= 64
    assert run_auto.name == args_auto.resolve_eyvan_run_name()
    args_auto.close_eyvan_track()

    # Non-zero rank with log_all_workers=False does not start a run
    monkeypatch.setattr(tc.jax, "process_index", lambda: 1)
    args_rank1 = _make_args(use_eyvan_track=True, eyvan_track_uri=str(store_dir), log_all_workers=False)
    assert args_rank1.eyvan_tracking_enabled is False
    assert args_rank1.get_eyvan_track() is None

    # Non-zero rank with log_all_workers=True and NO env rank vars falls back to jax.process_index()
    # without mutating os.environ
    monkeypatch.setattr(tc.jax, "process_index", lambda: 3)
    args_rank3_raw = _make_args(use_eyvan_track=True, eyvan_track_uri=str(store_dir), log_all_workers=True)
    assert args_rank3_raw.eyvan_tracking_enabled is True
    run_r3 = args_rank3_raw.get_eyvan_track()
    assert run_r3 is not None
    assert run_r3.rank == 3
    assert run_r3.active is True
    assert run_r3.episodes_active is True
    assert "JAX_PROCESS_ID" not in os.environ
    args_rank3_raw.close_eyvan_track()

    # Non-zero rank with ranks="zero" (e.g. can_log_metrics overridden on rank 2) passes episode_ranks="zero"
    # so no writer thread is spawned on the inactive nonzero rank
    monkeypatch.setattr(tc.jax, "process_index", lambda: 2)
    args_rank2_override = _make_args(use_eyvan_track=True, eyvan_track_uri=str(store_dir), log_all_workers=False)
    args_rank2_override.can_log_metrics = True
    run_r2_inactive = args_rank2_override.get_eyvan_track()
    assert run_r2_inactive is not None
    assert run_r2_inactive.rank == 2
    assert run_r2_inactive.active is False
    assert run_r2_inactive.episodes_active is False
    assert run_r2_inactive._thread is None
    args_rank2_override.close_eyvan_track()

    # Explicit EYVAN_DISTRIBUTED_PROCESS_ID still takes precedence when set
    monkeypatch.setattr(tc.jax, "process_index", lambda: 1)
    monkeypatch.setenv("EYVAN_DISTRIBUTED_PROCESS_ID", "1")
    args_rank1_all = _make_args(use_eyvan_track=True, eyvan_track_uri=str(store_dir), log_all_workers=True)
    assert args_rank1_all.eyvan_tracking_enabled is True
    run_r1 = args_rank1_all.get_eyvan_track()
    assert run_r1 is not None
    assert run_r1.rank == 1
    assert run_r1.active is True
    args_rank1_all.close_eyvan_track()

    # On multi-host GCE TPU pods (gcetpu), jax.process_index()==0 can reside on a non-zero VM ordinal
    # (e.g. w-1 carrying EYVAN_DISTRIBUTED_PROCESS_ID="1"). The sole can_log_metrics host must still
    # start an active rank-0 Eyvan tracking run without mutating os.environ.
    monkeypatch.setattr(tc.jax, "process_index", lambda: 0)
    monkeypatch.setattr(tc.jax, "process_count", lambda: 16)
    monkeypatch.setenv("EYVAN_DISTRIBUTED_PROCESS_ID", "1")
    monkeypatch.setenv("EYVAN_JAX_CLUSTER_DETECTION", "gcetpu")
    args_gcetpu_rank0 = _make_args(use_eyvan_track=True, eyvan_track_uri=str(store_dir), log_all_workers=False)
    assert args_gcetpu_rank0.eyvan_tracking_enabled is True
    run_gcetpu = args_gcetpu_rank0.get_eyvan_track()
    assert run_gcetpu is not None
    assert run_gcetpu.rank == 0
    assert run_gcetpu.active is True
    assert run_gcetpu.episodes_active is True
    assert os.environ["EYVAN_DISTRIBUTED_PROCESS_ID"] == "1"
    args_gcetpu_rank0.close_eyvan_track()


def test_eyvan_bfloat16_scalar_complex_dtypes_and_special_table_cells(tmp_path):
    # ml_dtypes.bfloat16 scalar (np.generic, not np.floating) extracts cleanly; complex is rejected
    ok_bf16, val_bf16 = _to_eyvan_scalar(ml_dtypes.bfloat16(1.5))
    assert ok_bf16 is True
    assert val_bf16 == pytest.approx(1.5)
    assert _to_eyvan_scalar(np.complex64(1.0 + 2.0j)) == (False, 0.0)
    assert _to_eyvan_scalar(np.asarray(1.0 + 2.0j, dtype=np.complex64)) == (False, 0.0)

    store_dir = tmp_path / "dtype_store"
    args = _make_args(
        use_wandb=False,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_flush_interval_s=0.2,
    )
    run = args.get_eyvan_track()
    assert run is not None
    run_id = run.id

    # Log ml_dtypes.bfloat16 scalar plus complex array and complex pre-binned histogram (must skip cleanly with no warning)
    args.log_metrics(
        {
            "train/bf16_loss": ml_dtypes.bfloat16(1.5),
            "ignored/complex_scalar": np.complex64(1.0 + 2.0j),
            "ignored/complex_array": np.asarray([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64),
            "ignored/complex_hist": (
                np.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
                np.asarray([0.0, 1.0, 2.0]),
            ),
        },
        step=3,
    )

    trainer = object.__new__(Trainer)
    trainer.arguments = args
    trainer.eyvan_run = run
    trainer.processing_class = None
    trainer._log_table_to_eyvan(
        "special_cells",
        columns=["bf16", "bytes_val", "complex_val", "int_val"],
        rows=[(ml_dtypes.bfloat16(2.5), np.bytes_(b"raw"), np.complex64(3.0 + 4.0j), np.int64(9))],
        step=3,
    )
    trainer.finish()

    scalar_records = _read_store_stream(store_dir, run_id, "scalars")
    assert len(scalar_records) == 1
    assert scalar_records[0]["v"] == {"train/bf16_loss": pytest.approx(1.5)}
    assert _read_store_stream(store_dir, run_id, "hist") == []

    table_records = _read_store_stream(store_dir, run_id, "tables")
    assert len(table_records) == 1
    row = table_records[0]["rows"][0]
    assert row[0] == pytest.approx(2.5)
    assert isinstance(row[1], str) and "raw" in row[1]
    assert isinstance(row[2], str) and "3" in row[2]
    assert row[3] == 9


def test_eyvan_report_progress_forwarding_and_step_gt_total_safety(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_report_progress(*, step=None, total=None, message="", metrics=None, environment=None):
        calls.append({"step": step, "total": total, "metrics": dict(metrics or {})})
        return True

    fake_eyvan = types.SimpleNamespace(report_progress=_fake_report_progress)
    monkeypatch.setattr(tc, "_import_eyvan", lambda: fake_eyvan)

    args = _make_args(use_eyvan_track=False, max_training_steps=100, eyvan_report_progress=True)
    # Regular step with finite + non-finite scalars
    args.log_metrics(
        {"train/loss": 0.5, "train/nan": float("nan"), "train/inf": float("inf"), "train/ok": True},
        step=25,
    )
    assert len(calls) == 1
    assert calls[0]["step"] == 25
    assert calls[0]["total"] == 100
    assert calls[0]["metrics"] == {"train/loss": 0.5, "train/ok": 1.0}

    # step > total is clamped safely so report_progress never raises ValueError
    args.log_metrics({"train/loss": 0.4}, step=125)
    assert len(calls) == 2
    assert calls[1]["step"] == 125
    assert calls[1]["total"] == 125

    # Negative step clamped to 0
    args.log_metrics({"train/loss": 0.9}, step=-3)
    assert calls[2]["step"] == 0
    assert calls[2]["total"] == 100

    # Summary/config do not invoke report_progress
    args.log_metrics({"best": 0.4}, step=125, log_as="summary")
    args.log_metrics({"lr": 1e-4}, step=0, log_as="config")
    assert len(calls) == 3

    # Disabled when eyvan_report_progress=False
    args_no_prog = _make_args(use_eyvan_track=False, max_training_steps=50, eyvan_report_progress=False)
    args_no_prog.log_metrics({"train/loss": 0.3}, step=5)
    assert len(calls) == 3

    # Non-zero rank never calls report_progress even with log_all_workers=True
    monkeypatch.setattr(tc.jax, "process_index", lambda: 2)
    args_rank2 = _make_args(use_eyvan_track=False, log_all_workers=True, max_training_steps=50)
    args_rank2.log_metrics({"train/loss": 0.2}, step=5)
    assert len(calls) == 3


def test_real_eyvan_report_progress_step_gt_total_noop_without_env():
    # Uses real eyvan.report_progress (not mocked): must never raise even with step > total and NaN values
    args = _make_args(use_eyvan_track=False, max_training_steps=10, eyvan_report_progress=True)
    args.log_metrics({"train/loss": 0.5, "train/nan": math.nan}, step=25)


def test_eyvan_tracking_errors_never_crash_training(monkeypatch):
    class _BrokenRun:
        finished = False

        def log(self, *a, **kw):
            raise RuntimeError("boom log")

        def histogram(self, *a, **kw):
            raise RuntimeError("boom hist")

        def summary(self, *a, **kw):
            raise RuntimeError("boom summary")

        def config(self, *a, **kw):
            raise RuntimeError("boom config")

    fake_eyvan = types.SimpleNamespace(report_progress=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom prog")))
    monkeypatch.setattr(tc, "_import_eyvan", lambda: fake_eyvan)

    args = _make_args(use_eyvan_track=True, eyvan_report_progress=True)
    args.__dict__["_eyvan_track_run"] = _BrokenRun()
    with pytest.warns(UserWarning):
        args.log_metrics({"loss": 1.0, "arr": np.asarray([1.0, 2.0])}, step=1)
    with pytest.warns(UserWarning):
        args.log_metrics({"best": 1.0}, step=1, log_as="summary")
    with pytest.warns(UserWarning):
        args.log_metrics({"cfg": 1}, step=0, log_as="config")


def test_base_trainer_eyvan_tables_and_teardown_without_wandb(tmp_path):
    store_dir = tmp_path / "trainer_store"
    args = _make_args(
        use_wandb=False,
        generation_log_to_wandb=False,
        log_training_generations_to_wandb=False,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_flush_interval_s=0.2,
    )
    trainer = object.__new__(Trainer)
    trainer.arguments = args
    trainer.eyvan_run = None
    trainer.processing_class = None
    trainer.preview_log_table = None
    trainer.benchmark_log_table = None
    trainer.training_generation_log_table = None

    trainer._initialize_eyvan_track()
    assert trainer.eyvan_run is not None
    run_id = trainer.eyvan_run.id

    # 1. Preview generations table (with use_wandb=False and generation_log_to_wandb=False)
    trainer._log_preview_results(
        step=5,
        records=[
            {
                "prompt": "What is 2+2?",
                "completions": ["4", "Four"],
                "reasoning": ["simple arithmetic", ""],
                "tool_calls": ["No tools were called ...", "No tools were called ..."],
                "step": 5,
            }
        ],
    )

    # 2. Benchmark results table (with use_wandb=False)
    trainer._log_benchmark_results_to_wandb(
        step=10,
        benchmark_results={
            "core": {
                "results": {
                    "hellaswag": {"acc,none": 0.78, "ignored_str": "n/a"},
                }
            }
        },
    )

    # 3. Training generations table (with use_wandb=False and log_training_generations_to_wandb=False)
    trainer._log_training_generations_to_wandb(
        state=types.SimpleNamespace(step=12),
        prompts=["Solve x=1", "Solve y=2"],
        completions=["x=1", "y=2"],
        completion_lengths=np.asarray([3, 3]),
        generation_time=0.4,
        reasoning=["r1", "r2"],
        tool_calls=[{"name": "calc"}, None],
        source="grpo",
    )

    # 4. Bounded row cap (> 1000 rows does not fail Eyvan's 1000-row limit)
    trainer._log_table_to_eyvan(
        "large_table",
        columns=["idx", "val"],
        rows=[(i, float(i)) for i in range(1200)],
        step=15,
    )

    # 5. Teardown flushes & closes run cleanly with no leaked thread
    trainer.finish()
    assert trainer.eyvan_run is None
    assert "_eyvan_track_run" not in args.__dict__
    assert not any(t.name == f"eyvan-track-{run_id}" and t.is_alive() for t in threading.enumerate())

    table_records = _read_store_stream(store_dir, run_id, "tables")
    tables_by_key = {rec["key"]: rec for rec in table_records}
    assert set(tables_by_key) == {
        "preview_generations",
        "benchmark_results",
        "training_generations",
        "large_table",
    }
    assert len(tables_by_key["preview_generations"]["rows"]) == 2
    assert tables_by_key["preview_generations"]["step"] == 5
    assert len(tables_by_key["benchmark_results"]["rows"]) == 1
    assert tables_by_key["benchmark_results"]["rows"][0][1:] == ["core", "hellaswag", "acc,none", 0.78]
    assert len(tables_by_key["training_generations"]["rows"]) == 2
    assert tables_by_key["training_generations"]["step"] == 12
    assert len(tables_by_key["large_table"]["rows"]) == 1000

    meta_records = _read_store_stream(store_dir, run_id, "meta")
    assert any(rec["op"] == "finish" and rec["status"] == "finished" for rec in meta_records)


def test_elarge_trainer_defaults_and_normalization_include_eyvan_fields():
    assert BASE_TRAINER_DEFAULTS["use_eyvan_track"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_uri"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_project"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_name"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_group"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_tags"] is None
    assert BASE_TRAINER_DEFAULTS["eyvan_track_flush_interval_s"] == 5.0
    assert BASE_TRAINER_DEFAULTS["eyvan_report_progress"] is True

    normalized = normalize_trainer_config(
        {
            "trainer_type": "sft",
            "use_eyvan_track": True,
            "eyvan_track_uri": "/tmp/elarge-eyvan",
            "eyvan_track_project": "elarge-proj",
            "eyvan_track_name": "elarge-run",
            "eyvan_track_group": "elarge-grp",
            "eyvan_track_tags": ["elarge"],
            "eyvan_track_flush_interval_s": 3.0,
            "eyvan_report_progress": False,
        }
    )
    assert normalized["use_eyvan_track"] is True
    assert normalized["eyvan_track_uri"] == "/tmp/elarge-eyvan"
    assert normalized["eyvan_track_project"] == "elarge-proj"
    assert normalized["eyvan_track_name"] == "elarge-run"
    assert normalized["eyvan_track_group"] == "elarge-grp"
    assert normalized["eyvan_track_tags"] == ["elarge"]
    assert normalized["eyvan_track_flush_interval_s"] == 3.0
    assert normalized["eyvan_report_progress"] is False


def test_eyvan_guard1_lazy_init_lifecycle_checkpoints_eval_and_exception_statuses(monkeypatch, tmp_path, caplog):
    from easydel.trainers.trainer_protocol import TrainerOutput

    def _build_stub_trainer(store_dir: Path, *, do_eval: bool = False) -> Trainer:
        args = _make_args(
            model_name="Qwen/Qwen3-4B",
            trainer_prefix="SFT",
            use_wandb=False,
            save_directory=str(tmp_path / "ckpts"),
            do_eval=do_eval,
            use_eyvan_track=True,
            eyvan_track_uri=str(store_dir),
            eyvan_track_flush_interval_s=0.1,
            report_steps=1,
            log_steps=1,
        )
        trainer = object.__new__(Trainer)
        trainer.arguments = args
        trainer._model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3"),
            flops_per_token=lambda **kw: 1.0,
        )
        trainer.config = trainer._model.config
        trainer.model_state = types.SimpleNamespace(step=0, graphstate={}, model=trainer._model)
        trainer._initialize_attributes()
        # Stub all non-Eyvan setup steps so initialize_trainer_utils exercises the real init sequence.
        for hook_name in (
            "_initialize_wandb",
            "_initialize_timer",
            "_configure_dataloaders",
            "_configure_model",
            "_configure_state",
            "_configure_buckets",
            "_configure_eval_config_override",
            "_configure_functions",
            "_configure_runtime_trace",
            "_configure_checkpoint_policy",
            "_setup_static_metrics",
            "_setup_initial_metrics",
            "_close_rich_progress_bar",
            "_close_bucket_iterators",
            "release_generation_runtime",
        ):
            setattr(trainer, hook_name, lambda *a, **kw: None)
        return trainer

    # 1. Guard 1: initialize_trainer_utils() (called by BaseTrainer.__init__) creates ZERO files/threads!
    store_ok = tmp_path / "guard1_ok"
    baseline_threads = set(threading.enumerate())
    trainer_ok = _build_stub_trainer(store_ok, do_eval=True)
    trainer_ok.initialize_trainer_utils()
    assert trainer_ok.eyvan_run is None
    assert "_eyvan_track_run" not in trainer_ok.arguments.__dict__
    assert "_eyvan_runtime_context" in trainer_ok.arguments.__dict__
    assert not store_ok.exists()
    assert [t for t in threading.enumerate() if t not in baseline_threads and t.is_alive()] == []

    # 2. Normal train() lazily starts Eyvan, emits rank-0 banner with dashboard hint, records checkpoint + eval
    #    summaries, and closes with status="finished" and zero leaked threads.
    info_messages: list[str] = []
    monkeypatch.setattr(tc.logger, "info", lambda msg, *a, **kw: info_messages.append(str(msg)))

    def _run_loop_ok(*, state, metrics_tracker, step_metrics):
        assert trainer_ok.eyvan_run is not None
        trainer_ok._eyvan_tokens_seen = 2048.0
        trainer_ok._record_eyvan_checkpoint(str(tmp_path / "ckpts" / "checkpoint-10"), step=10)
        final_state = types.SimpleNamespace(step=10, graphstate={})
        out = TrainerOutput(
            state=final_state,
            mesh=None,
            checkpoint_path=str(tmp_path / "ckpts" / "checkpoint-10"),
            last_save_file_name="checkpoint-10",
        )
        return out, None

    def _fake_eval(state):
        trainer_ok._record_eyvan_eval_summary({"eval/loss": 0.31, "eval/accuracy": 0.92})
        yield {"eval/loss": 0.31, "eval/accuracy": 0.92}

    trainer_ok._run_training_loop = _run_loop_ok
    trainer_ok.eval = _fake_eval

    run_id_holder: list[str] = []
    orig_finish = trainer_ok.finish

    def _capture_before_finish(*a, **kw):
        if trainer_ok.eyvan_run is not None:
            run_id_holder.append(trainer_ok.eyvan_run.id)
        return orig_finish(*a, **kw)

    trainer_ok.finish = _capture_before_finish
    output = trainer_ok.train()
    assert output.checkpoint_path.endswith("checkpoint-10")
    assert trainer_ok.eyvan_run is None
    assert len(run_id_holder) == 1
    run_id_ok = run_id_holder[0]
    assert not any(t.name == f"eyvan-track-{run_id_ok}" and t.is_alive() for t in threading.enumerate())
    # Rank-0 startup banner emitted once with run_id and dashboard hint.
    banner_hits = [m for m in info_messages if run_id_ok in m and "eyvan dashboard" in m]
    assert len(banner_hits) == 1

    meta_ok = _read_store_stream(store_ok, run_id_ok, "meta")
    assert [r for r in meta_ok if r["op"] == "finish"][-1]["status"] == "finished"
    merged_summary: dict[str, object] = {}
    for rec in meta_ok:
        if rec["op"] == "summary":
            merged_summary.update(rec["summary"])
    assert merged_summary["checkpoint_path"] == str(tmp_path / "ckpts" / "checkpoint-10")
    assert merged_summary["last_checkpoint_step"] == 10
    assert merged_summary["checkpoints_saved"] == 1
    assert merged_summary["final_step"] == 10
    assert merged_summary["total_tokens_seen"] == pytest.approx(2048.0)
    assert merged_summary["eval/loss"] == pytest.approx(0.31)
    assert merged_summary["eval/accuracy"] == pytest.approx(0.92)

    # 3. Uncaught training exception closes Eyvan cleanly with status="failed" and stop_reason.
    store_fail = tmp_path / "guard1_fail"
    trainer_fail = _build_stub_trainer(store_fail, do_eval=False)
    trainer_fail.initialize_trainer_utils()
    fail_run_ids: list[str] = []

    def _run_loop_crash(*, state, metrics_tracker, step_metrics):
        assert trainer_fail.eyvan_run is not None
        fail_run_ids.append(trainer_fail.eyvan_run.id)
        raise RuntimeError("boom train step")

    trainer_fail._run_training_loop = _run_loop_crash
    with pytest.raises(RuntimeError, match="boom train step"):
        trainer_fail.train()
    assert trainer_fail.eyvan_run is None
    fail_run_id = fail_run_ids[0]
    assert not any(t.name == f"eyvan-track-{fail_run_id}" and t.is_alive() for t in threading.enumerate())
    meta_fail = _read_store_stream(store_fail, fail_run_id, "meta")
    assert [r for r in meta_fail if r["op"] == "finish"][-1]["status"] == "failed"
    fail_summary = [r for r in meta_fail if r["op"] == "summary"][-1]["summary"]
    assert fail_summary["stop_reason"] == "RuntimeError"

    # 4. KeyboardInterrupt closes Eyvan cleanly with status="aborted".
    store_abort = tmp_path / "guard1_abort"
    trainer_abort = _build_stub_trainer(store_abort, do_eval=False)
    trainer_abort.initialize_trainer_utils()
    abort_run_ids: list[str] = []

    def _run_loop_interrupt(*, state, metrics_tracker, step_metrics):
        assert trainer_abort.eyvan_run is not None
        abort_run_ids.append(trainer_abort.eyvan_run.id)
        raise KeyboardInterrupt()

    trainer_abort._run_training_loop = _run_loop_interrupt
    with pytest.raises(KeyboardInterrupt):
        trainer_abort.train()
    assert trainer_abort.eyvan_run is None
    abort_run_id = abort_run_ids[0]
    assert not any(t.name == f"eyvan-track-{abort_run_id}" and t.is_alive() for t in threading.enumerate())
    meta_abort = _read_store_stream(store_abort, abort_run_id, "meta")
    assert [r for r in meta_abort if r["op"] == "finish"][-1]["status"] == "aborted"

    # 5. First log_metrics() call without train() also triggers lazy init and syncs trainer.eyvan_run.
    store_log = tmp_path / "guard1_first_log"
    trainer_log = _build_stub_trainer(store_log, do_eval=False)
    trainer_log.initialize_trainer_utils()
    assert trainer_log.eyvan_run is None
    pbar = types.SimpleNamespace(set_postfix=lambda **kw: None, update=lambda n=1: None)
    trainer_log.log_metrics({"train/loss": 0.42}, pbar, step=1)
    assert trainer_log.eyvan_run is not None
    log_run_id = trainer_log.eyvan_run.id
    trainer_log.finish()
    assert trainer_log.eyvan_run is None
    assert not any(t.name == f"eyvan-track-{log_run_id}" and t.is_alive() for t in threading.enumerate())
    scalars = _read_store_stream(store_log, log_run_id, "scalars")
    assert len(scalars) == 1
    assert scalars[0]["v"]["train/loss"] == pytest.approx(0.42)


def test_eyvan_guard2_unwritable_home_and_start_failure_warn_once_noop(monkeypatch, tmp_path):
    # 1. Mocked eyvan.track.start failure (e.g. PermissionError) warns ONCE, sets _eyvan_init_failed,
    #    and never retries on subsequent get_eyvan_track / log_metrics / log_eyvan_* calls.
    start_calls = 0

    def _failing_start(**_kwargs):
        nonlocal start_calls
        start_calls += 1
        raise PermissionError("read-only tracking directory")

    monkeypatch.setattr(tc, "_import_eyvan_track", lambda: types.SimpleNamespace(start=_failing_start))
    args_fail = _make_args(use_eyvan_track=True, eyvan_track_uri=str(tmp_path / "writable_dir"))
    with pytest.warns(UserWarning, match="PermissionError"):
        assert args_fail.get_eyvan_track() is None
    assert start_calls == 1
    assert args_fail.__dict__.get("_eyvan_init_failed") is True
    assert args_fail.eyvan_tracking_enabled is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert args_fail.get_eyvan_track() is None
        args_fail.log_metrics({"train/loss": 0.5}, step=1)
        assert args_fail.log_eyvan_episode(1.0, 2, step=1) is False
        assert args_fail.log_eyvan_rollout(tracks={"reward": [1.0]}, step=1) is False
        assert args_fail.log_eyvan_completions("p", ["r"], rewards=[1.0], step=1) is False
        assert args_fail.log_eyvan_histogram("rl/h", [1.0, 2.0], step=1) is False
    assert start_calls == 1
    assert caught == []

    # 2. Real read-only HOME directory (non-root): fails fast before spawning any writer thread,
    #    warns ONCE, and cleanly no-ops all subsequent calls.
    monkeypatch.undo()
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        ro_home = tmp_path / "readonly_home"
        ro_home.mkdir()
        ro_home.chmod(0o555)
        try:
            monkeypatch.setenv("HOME", str(ro_home))
            baseline_threads = set(threading.enumerate())
            with set_eyvan_auto_track_enabled(True):
                args_ro = _make_args(eyvan_track_project="ro-proj")
                assert args_ro.eyvan_tracking_enabled is True
                with pytest.warns(UserWarning, match="PermissionError|OSError"):
                    assert args_ro.get_eyvan_track() is None
                assert args_ro.__dict__.get("_eyvan_init_failed") is True
                assert args_ro.eyvan_tracking_enabled is False
                assert [t for t in threading.enumerate() if t not in baseline_threads and t.is_alive()] == []
                with warnings.catch_warnings(record=True) as caught_ro:
                    warnings.simplefilter("always")
                    assert args_ro.get_eyvan_track() is None
                    args_ro.log_metrics({"train/loss": 0.9}, step=2)
                    assert args_ro.log_eyvan_episode(1.0, 3, step=2) is False
                assert caught_ro == []
        finally:
            ro_home.chmod(0o755)
