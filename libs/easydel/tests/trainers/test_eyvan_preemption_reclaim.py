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

"""CPU-only end-to-end tests for Eyvan 10-minute graceful reclaim and checkpoint auto-resume."""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
from easydel.infra.errors import EasyDeLPreemptionSignal
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers import base_trainer as bt
from easydel.trainers.agent_flow_trainer import AgentFlowTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
from easydel.trainers.trainer import Trainer
from easydel.trainers.training_configurations import TrainingArguments
from easydel.utils import eyvan_helpers as eh
from eyvan.track._format import decode_segment
from eyvan.track._store import store_for_uri

# Pure-Python + NumPy + local Eyvan store tests; no JAX device array computation runs.
pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def _clean_preemption_env_and_state(monkeypatch):
    monkeypatch.setattr(TrainingArguments, "_setup_distributed", lambda self: None)
    eh._reset_fallback_preemption_state()
    for key in (
        "EYVAN_RECLAIM_NOTICE_PATH",
        "EYVAN_CHECKPOINT_RECEIPT_PATH",
        "EYVAN_RESUME_CHECKPOINT_URI",
        "EYVAN_WORKLOAD_ID",
        "EYVAN_TRACKING_URI",
        "EYVAN_PROGRESS_URL",
        "EYVAN_MANAGED_RESULT_TOKEN",
        "EYVAN_DISTRIBUTED_PROCESS_ID",
        "JAX_PROCESS_ID",
        "EYVAN_ATTEMPT",
    ):
        monkeypatch.delenv(key, raising=False)
    try:
        yield
    finally:
        eh._reset_fallback_preemption_state()


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _read_store_stream(store_dir: Path, run_id: str, stream: str) -> list[dict[str, object]]:
    store = store_for_uri(str(store_dir))
    records: list[dict[str, object]] = []
    for key in store.list(f"{run_id}/{stream}"):
        records.extend(decode_segment(store.get(key)))
    return records


class _NullMeshCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StateStub:
    def __init__(self, step: int = 0, model: object | None = None):
        self.step = step
        self.graphstate = {}
        self.opt_state = None
        self.tx = None
        self.model = model

    def replace(self, **kwargs):
        out = _StateStub(step=kwargs.get("step", self.step), model=self.model)
        out.graphstate = kwargs.get("graphstate", self.graphstate)
        out.opt_state = kwargs.get("opt_state", self.opt_state)
        out.tx = kwargs.get("tx", self.tx)
        return out


class _DummyPbar:
    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 1. Preemption Bridge (stdlib fallback & Eyvan delegation)
# ---------------------------------------------------------------------------


def test_preemption_bridge_fallback_notice_signal_receipt_and_resume(monkeypatch, tmp_path: Path):
    # Force fallback path (simulating an older eyvan package without preemption exports).
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)

    notice_file = tmp_path / "reclaim_notice.json"
    receipt_file = tmp_path / "checkpoint_receipt.json"
    env = {
        "EYVAN_RECLAIM_NOTICE_PATH": str(notice_file),
        "EYVAN_CHECKPOINT_RECEIPT_PATH": str(receipt_file),
        "EYVAN_WORKLOAD_ID": "wl-reclaim-42",
    }

    # Absent notice file -> None / False
    assert eh.preemption_notice(environment=env) is None
    assert eh.reclaim_requested(environment=env) is False
    assert eh.resume_checkpoint_uri(environment=env) is None

    # Malformed notice file -> safe None (never raises)
    notice_file.write_text("{not valid json", encoding="utf-8")
    assert eh.preemption_notice(environment=env) is None
    assert eh.reclaim_requested(environment=env) is False

    # Valid 600s reclaim notice file -> parsed PreemptionNotice with remaining_seconds
    deadline = time.time() + 600.0
    _write_atomic_json(
        notice_file,
        {
            "reason": "owner_reclaim",
            "requested_by": "bob",
            "deadline_at": "2099-01-01T00:00:00Z",
            "deadline_epoch_s": deadline,
            "grace_seconds": 600,
            "policy": "requeue",
            "checkpoint_uri": "gs://eyvan_us_central2/checkpoints/wl-reclaim-42",
        },
    )
    notice = eh.preemption_notice(environment=env)
    assert isinstance(notice, eh.PreemptionNotice)
    assert eh.reclaim_requested(environment=env) is True
    assert notice.reason == "owner_reclaim"
    assert notice.requested_by == "bob"
    assert notice.grace_seconds == 600
    assert notice.policy == "requeue"
    assert notice.checkpoint_uri == "gs://eyvan_us_central2/checkpoints/wl-reclaim-42"
    assert notice.workload_id == "wl-reclaim-42"
    assert 500.0 <= notice.remaining_seconds <= 601.0

    # acknowledge_checkpoint writes atomic 0o600 receipt JSON and resume_checkpoint_uri reads it
    saved_uri = "gs://eyvan_us_central2/checkpoints/wl-reclaim-42/run-12"
    written = eh.acknowledge_checkpoint(saved_uri, step=12, environment=env)
    assert written == receipt_file
    assert receipt_file.exists()
    receipt_data = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt_data["checkpoint_uri"] == saved_uri
    assert receipt_data["step"] == 12
    assert isinstance(receipt_data["recorded_at"], str) and receipt_data["recorded_at"]
    if os.name == "posix":
        assert stat.S_IMODE(receipt_file.stat().st_mode) == 0o600
    assert eh.resume_checkpoint_uri(environment=env) == saved_uri

    # EYVAN_RESUME_CHECKPOINT_URI takes precedence over receipt file
    env_with_resume = dict(env, EYVAN_RESUME_CHECKPOINT_URI="gs://eyvan_us_central2/checkpoints/override/run-99")
    assert eh.resume_checkpoint_uri(environment=env_with_resume) == "gs://eyvan_us_central2/checkpoints/override/run-99"

    # install_preemption_handler handles SIGUSR1 idempotently without aborting process
    if hasattr(signal, "SIGUSR1"):
        received: list[eh.PreemptionNotice] = []
        eh.install_preemption_handler(callback=received.append)
        eh.install_preemption_handler(callback=received.append)  # idempotent
        os.kill(os.getpid(), signal.SIGUSR1)
        assert len(received) == 1
        assert received[0].reason == "signal:SIGUSR1"
        assert eh.reclaim_requested() is True
        live_notice = eh.preemption_notice()
        assert live_notice is not None and live_notice.reason == "signal:SIGUSR1"


def test_preemption_bridge_delegates_to_eyvan_when_available(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, object]] = []
    fake_notice = eh.PreemptionNotice(reason="owner_reclaim", requested_by="admin", grace_seconds=600)

    fake_eyvan = types.SimpleNamespace(
        preemption_notice=lambda **kw: calls.append(("preemption_notice", kw)) or fake_notice,
        reclaim_requested=lambda **kw: calls.append(("reclaim_requested", kw)) or True,
        install_preemption_handler=lambda **kw: calls.append(("install_preemption_handler", kw)),
        acknowledge_checkpoint=lambda uri, **kw: (
            calls.append(("acknowledge_checkpoint", (uri, kw))) or (tmp_path / "r.json")
        ),
        resume_checkpoint_uri=lambda **kw: calls.append(("resume_checkpoint_uri", kw)) or "gs://bucket/run-7",
    )
    monkeypatch.setitem(sys.modules, "eyvan", fake_eyvan)

    cb = lambda _notice: None  # noqa: E731
    assert eh.preemption_notice() is fake_notice
    assert eh.reclaim_requested() is True
    eh.install_preemption_handler(callback=cb)
    assert eh.acknowledge_checkpoint("gs://bucket/run-7", step=7) == tmp_path / "r.json"
    assert eh.resume_checkpoint_uri() == "gs://bucket/run-7"
    assert [name for name, _ in calls] == [
        "preemption_notice",
        "reclaim_requested",
        "install_preemption_handler",
        "acknowledge_checkpoint",
        "resume_checkpoint_uri",
    ]
    assert calls[2][1] == {"callback": cb}
    assert calls[3][1] == ("gs://bucket/run-7", {"step": 7})


# ---------------------------------------------------------------------------
# 2. Clean Public API Boundary (Compute Dispatch & Eyvan SDK Stay in `eyvan`)
# ---------------------------------------------------------------------------


def test_easydel_public_api_does_not_reexport_eyvan_sdk_or_compute_dispatch():
    import easydel as ed

    for name in (
        "PreemptionNotice",
        "acknowledge_checkpoint",
        "configure_storage",
        "eyvan_execute",
        "eyvan_init",
        "get_monitoring_storage",
        "get_storage",
        "install_preemption_handler",
        "monitoring_path",
        "preemption_notice",
        "reclaim_requested",
        "resume_checkpoint_uri",
        "run_on_tpu",
        "storage_path",
    ):
        assert not hasattr(ed, name), f"easydel.{name} should live only in eyvan, not top-level easydel"

    for name in ("run_on_tpu", "eyvan_execute", "eyvan_init"):
        assert not hasattr(eh, name), f"easydel.utils.eyvan_helpers.{name} should live only in eyvan"


# ---------------------------------------------------------------------------
# 3. End-to-End 600s Graceful Reclaim Checkpoint & Exit in Trainer.train()
# ---------------------------------------------------------------------------


def _build_loop_trainer(
    *,
    tmp_path: Path,
    store_dir: Path,
    save_directory: str | None,
    max_training_steps: int = 5,
    execute_step_fn=None,
) -> Trainer:
    args = TrainingArguments(
        model_name="Qwen/Qwen3-8B",
        trainer_prefix="SFT",
        use_wandb=False,
        save_directory=save_directory,
        num_train_epochs=1,
        max_training_steps=max_training_steps,
        total_batch_size=2,
        max_length=16,
        learning_rate=1e-4,
        do_eval=False,
        do_last_save=True,
        use_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_flush_interval_s=0.1,
        report_steps=1,
        log_steps=1,
        save_steps=None,
        dataloader_prefetch=False,
        progress_bar_type="json",
    )
    trainer = object.__new__(Trainer)
    trainer.arguments = args
    fake_model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="qwen3"),
        flops_per_token=lambda **kw: 1.0,
        param_dtype="bfloat16",
        mesh=_NullMeshCtx(),
    )
    trainer._model = fake_model
    trainer.config = fake_model.config
    trainer.model_state = _StateStub(step=0, model=fake_model)
    trainer._initialize_attributes()
    trainer._has_buckets = False
    trainer.max_training_steps = max_training_steps
    trainer.dataloader_train = [{"input_ids": [i]} for i in range(max_training_steps)]
    trainer.data_collator = None
    trainer.scheduler = None
    trainer._backward_flops_per_token = 0.0
    trainer._extra_backward_flops_per_token = 0.0
    trainer._setup_initial_metrics = lambda state: None
    trainer._log_mesh_topology = lambda: None
    trainer.create_progress_bar = lambda **kw: _DummyPbar()
    trainer.log_weight_distribution = lambda **kw: None
    trainer.log_watchers = lambda **kw: None
    trainer.maybe_generate = lambda **kw: None
    trainer.maybe_benchmark = lambda **kw: None
    trainer._should_save_tpu_preemption_checkpoint = lambda step: False
    trainer._should_run_evaluation = lambda step: False
    trainer._maybe_start_profiler = lambda step: None
    trainer._maybe_stop_profiler = lambda step: None
    trainer._stop_profiler = lambda: None
    trainer._profiler_should_block_until_ready = lambda: False
    trainer._apply_pose_to_batch = lambda batch, step: batch
    trainer._apply_randomized_yarn_to_batch = lambda batch, step: batch
    trainer._apply_process_encoder_to_batch = lambda batch, step, state=None: batch
    trainer.apply_training_hooks = lambda metrics, **kw: metrics
    trainer._dispatch_callbacks = lambda *a, **kw: None
    trainer.release_generation_runtime = lambda **kw: None
    trainer._execute_train_step = execute_step_fn
    return trainer


def test_trainer_graceful_reclaim_mid_training_saves_acknowledges_and_exits(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)

    notice_file = tmp_path / "notice.json"
    receipt_file = tmp_path / "receipt.json"
    store_dir = tmp_path / "eyvan_store"
    ckpt_root = tmp_path / "checkpoints"
    monkeypatch.setenv("EYVAN_RECLAIM_NOTICE_PATH", str(notice_file))
    monkeypatch.setenv("EYVAN_CHECKPOINT_RECEIPT_PATH", str(receipt_file))
    monkeypatch.setenv("EYVAN_WORKLOAD_ID", "wl-600s-1")

    install_calls: list[bool] = []
    orig_install = eh.install_preemption_handler

    def _spy_install(*args, **kwargs):
        install_calls.append(True)
        return orig_install(*args, **kwargs)

    monkeypatch.setattr(eh, "install_preemption_handler", _spy_install)

    executed_steps: list[int] = []
    saved_calls: list[tuple[int, bool]] = []
    warning_messages: list[str] = []
    monkeypatch.setattr(bt.logger, "warning", lambda msg, *a, **kw: warning_messages.append(str(msg)))

    def _exec_step(*, state, batch):
        del batch
        next_step = int(state.step) + 1
        executed_steps.append(next_step)
        if next_step == 2:
            # Eyvan delivers a 10-minute reclaim notice mid-step 2.
            _write_atomic_json(
                notice_file,
                {
                    "reason": "owner_reclaim",
                    "requested_by": "bob",
                    "deadline_epoch_s": time.time() + 600.0,
                    "grace_seconds": 600,
                    "policy": "requeue",
                },
            )
        return (
            _StateStub(step=next_step, model=state.model),
            LossMetrics(loss=0.5 / next_step, accuracy=0.8, execution_time=0.01),
            None,
        )

    trainer = _build_loop_trainer(
        tmp_path=tmp_path,
        store_dir=store_dir,
        save_directory=str(ckpt_root),
        max_training_steps=5,
        execute_step_fn=_exec_step,
    )

    def _fake_save_checkpoint(*, state, step, force=False, merge_lora_before_save=False):
        del state, merge_lora_before_save
        saved_calls.append((int(step), bool(force)))
        if not force:
            return None
        ckpt_dir = ckpt_root / "Qwen" / "Qwen3-8B" / f"run-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metadata.json").write_text(json.dumps({"step": int(step)}), encoding="utf-8")
        trainer._record_eyvan_checkpoint(str(ckpt_dir), step=int(step), is_temporary=False)
        return str(ckpt_dir)

    trainer._save_checkpoint_for_step = _fake_save_checkpoint
    trainer.checkpointer = object()

    captured_run_id: list[str] = []
    orig_finish = trainer.finish

    def _spy_finish(*a, **kw):
        if trainer.eyvan_run is not None:
            captured_run_id.append(trainer.eyvan_run.id)
        return orig_finish(*a, **kw)

    trainer.finish = _spy_finish

    # Capture the exact run_exception returned from _run_training_loop
    loop_exceptions: list[BaseException | None] = []
    orig_run_loop = trainer._run_training_loop

    def _spy_run_loop(*a, **kw):
        out, exc = orig_run_loop(*a, **kw)
        loop_exceptions.append(exc)
        return out, exc

    trainer._run_training_loop = _spy_run_loop

    output = trainer.train()

    # 1. Handler installed at train() entry
    assert len(install_calls) >= 1

    # 2. Step 2 completed, step 3 never ran, EasyDeLPreemptionSignal returned
    assert executed_steps == [1, 2]
    assert len(loop_exceptions) == 1
    assert isinstance(loop_exceptions[0], EasyDeLPreemptionSignal)

    # 3. Crisp warning named requested_by, reason, remaining_seconds, policy
    assert any(
        "bob" in m and "owner_reclaim" in m and "requeue" in m and "remaining_seconds=" in m for m in warning_messages
    ), warning_messages

    # 4. Forced checkpoint saved at step 2 and receipt acknowledged atomically
    assert (2, True) in saved_calls
    expected_ckpt = str(ckpt_root / "Qwen" / "Qwen3-8B" / "run-2")
    assert output.checkpoint_path == expected_ckpt
    assert receipt_file.exists()
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["checkpoint_uri"] == expected_ckpt
    assert receipt["step"] == 2

    # 5. Eyvan tracking recorded reclaim metadata, closed with status="aborted", zero leaked threads
    assert len(captured_run_id) == 1
    run_id = captured_run_id[0]
    assert trainer.eyvan_run is None
    assert not any(t.name == f"eyvan-track-{run_id}" and t.is_alive() for t in threading.enumerate())

    meta_records = _read_store_stream(store_dir, run_id, "meta")
    finish_records = [rec for rec in meta_records if rec["op"] == "finish"]
    assert finish_records[-1]["status"] == "aborted"

    merged_summary: dict[str, object] = {}
    for rec in meta_records:
        if rec["op"] == "summary":
            merged_summary.update(rec["summary"])
    assert merged_summary["reclaim_checkpoint"] == expected_ckpt
    assert merged_summary["reclaimed_by"] == "bob"
    assert merged_summary["reclaim_reason"] == "owner_reclaim"
    assert merged_summary["reclaim_policy"] == "requeue"
    assert merged_summary["reclaim_step"] == 2
    assert merged_summary["stop_reason"] == "EasyDeLPreemptionSignal"


def test_reclaim_checkpoint_target_resolution_and_fallback_save(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)

    def _make_stub_trainer(save_directory: str | None, storage_uri: str | None = None) -> Trainer:
        args = TrainingArguments(
            model_name="Qwen/Qwen3-4B",
            use_wandb=False,
            save_directory=save_directory,
            use_eyvan_track=False,
        )
        if storage_uri is not None:
            args.resolve_eyvan_storage_path = lambda *parts, **kw: f"{storage_uri}/{'/'.join(parts)}"
        else:
            args.resolve_eyvan_storage_path = lambda *parts, **kw: (_ for _ in ()).throw(RuntimeError("no bucket"))
        tr = object.__new__(Trainer)
        tr.arguments = args
        tr._model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3"),
            param_dtype="bfloat16",
            flops_per_token=lambda **kw: 1.0,
            mesh=_NullMeshCtx(),
        )
        tr.model_state = _StateStub(step=4, model=tr._model)
        tr._initialize_attributes()
        tr.checkpointer = None
        return tr

    # Case 1: Durable remote save_directory wins over notice.checkpoint_uri
    tr_remote = _make_stub_trainer("gs://user-durable/exp")
    n_uri = eh.PreemptionNotice(checkpoint_uri="gs://eyvan-fallback/wl-1")
    assert tr_remote._resolve_eyvan_reclaim_checkpoint_target(n_uri) == ("gs://user-durable/exp", False)

    # Case 2: Non-durable local save_directory defers to notice.checkpoint_uri
    tr_local = _make_stub_trainer(str(tmp_path / "local_ephemeral"))
    assert tr_local._resolve_eyvan_reclaim_checkpoint_target(n_uri) == ("gs://eyvan-fallback/wl-1", True)

    # Case 3: Local save_directory with no notice.checkpoint_uri keeps save_directory
    n_empty = eh.PreemptionNotice(checkpoint_uri=None)
    assert tr_local._resolve_eyvan_reclaim_checkpoint_target(n_empty) == (str(tmp_path / "local_ephemeral"), False)

    # Case 4: save_directory=None uses notice.checkpoint_uri
    tr_none = _make_stub_trainer(None)
    assert tr_none._resolve_eyvan_reclaim_checkpoint_target(n_uri) == ("gs://eyvan-fallback/wl-1", True)

    # Case 5: save_directory=None and notice.checkpoint_uri=None falls back to resolve_eyvan_storage_path
    tr_storage = _make_stub_trainer(None, storage_uri="gs://eyvan_us_central2")
    target, is_override = tr_storage._resolve_eyvan_reclaim_checkpoint_target(n_empty)
    assert is_override is True
    assert target is not None and target.startswith("gs://eyvan_us_central2/checkpoints/")

    # Case 6: save_directory=None, no notice URI, failing storage resolver degrades non-fatally to (None, False)
    assert tr_none._resolve_eyvan_reclaim_checkpoint_target(n_empty) == (None, False)
    sig_no_save = tr_none._handle_eyvan_reclaim_preemption(state=tr_none.model_state, step=4, notice=n_empty)
    assert isinstance(sig_no_save, EasyDeLPreemptionSignal)
    assert tr_none._preemption_checkpoint_path is None

    # Case 7: save_directory=None + concrete notice.checkpoint_uri saves cleanly via temporary Checkpointer
    fallback_dir = tmp_path / "reclaim_fallback"
    receipt_file = tmp_path / "fallback_receipt.json"
    monkeypatch.setenv("EYVAN_CHECKPOINT_RECEIPT_PATH", str(receipt_file))
    saved_dirs: list[str] = []

    def _fake_save_state(*, state, save_directory=None, merge_lora_before_save=False):
        del state, merge_lora_before_save
        assert save_directory is not None
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps({"step": 4}), encoding="utf-8")
        saved_dirs.append(str(out_dir))
        return str(out_dir)

    tr_none._save_state = _fake_save_state
    tr_none._save_data_lane_state = lambda *a, **kw: None
    tr_none._dispatch_callbacks = lambda *a, **kw: None

    sig = tr_none._handle_eyvan_reclaim_preemption(
        state=tr_none.model_state,
        step=4,
        notice=eh.PreemptionNotice(
            reason="owner_reclaim",
            requested_by="admin",
            policy="stop",
            checkpoint_uri=str(fallback_dir),
        ),
    )
    assert isinstance(sig, EasyDeLPreemptionSignal)
    assert len(saved_dirs) == 1
    assert saved_dirs[0].startswith(str(fallback_dir))
    assert tr_none.checkpointer is None  # restored cleanly
    assert tr_none._preemption_checkpoint_path == saved_dirs[0]
    assert json.loads(receipt_file.read_text(encoding="utf-8"))["checkpoint_uri"] == saved_dirs[0]


# ---------------------------------------------------------------------------
# 4. Multi-Host Coordination Safety
# ---------------------------------------------------------------------------


def test_multihost_reclaim_coordination_broadcasts_notice_and_stays_cpu_safe(monkeypatch):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)

    tr = object.__new__(Trainer)
    tr.arguments = TrainingArguments(use_wandb=False, save_directory=None, use_eyvan_track=False)
    tr._eyvan_reclaim_multihost_sync_available = None

    # On CPU backend with mocked process_count=4, no device collective is invoked (zero XLA compile).
    monkeypatch.setattr(bt.jax, "process_count", lambda: 4)
    monkeypatch.setattr(bt.jax, "process_index", lambda: 2)
    monkeypatch.setenv("EYVAN_WORKLOAD_ID", "wl-mh-1")
    assert tr._can_use_multihost_collectives() is False
    assert tr._should_save_eyvan_reclaim_checkpoint(step=5) is None

    # Simulate multi-host accelerator where rank 0 saw the notice and rank 2 hasn't seen it yet.
    rank0_notice = eh.PreemptionNotice(
        reason="owner_reclaim",
        requested_by="bob",
        grace_seconds=600,
        policy="requeue",
        checkpoint_uri="gs://eyvan_us_central2/checkpoints/wl-mh-1",
        workload_id="wl-mh-1",
    )
    rank0_bytes = json.dumps(eh._serialize_preemption_notice(rank0_notice), sort_keys=True).encode("utf-8")

    monkeypatch.setattr(tr, "_can_use_multihost_collectives", lambda: True)

    from jax.experimental import multihost_utils

    monkeypatch.setattr(
        multihost_utils,
        "process_allgather",
        lambda arr: np.asarray([1, 0, 0, 0], dtype=np.int32),
    )

    def _fake_broadcast(arr, is_source=None):
        assert is_source is False  # rank 2 is not source (rank 0 is)
        if arr.ndim == 0:
            return np.asarray(len(rank0_bytes), dtype=np.int64)
        return np.frombuffer(rank0_bytes, dtype=np.uint8)

    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", _fake_broadcast)

    synced = tr._should_save_eyvan_reclaim_checkpoint(step=5)
    assert isinstance(synced, eh.PreemptionNotice)
    assert synced.reason == "owner_reclaim"
    assert synced.requested_by == "bob"
    assert synced.policy == "requeue"
    assert synced.checkpoint_uri == "gs://eyvan_us_central2/checkpoints/wl-mh-1"


# ---------------------------------------------------------------------------
# 5. GRPOTrainer (async mode & dynamic sampling) & AgentFlowTrainer Reclaim
# ---------------------------------------------------------------------------


def test_grpo_async_mode_on_step_end_triggers_reclaim_and_raises_signal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)
    notice_file = tmp_path / "grpo_notice.json"
    receipt_file = tmp_path / "grpo_receipt.json"
    _write_atomic_json(
        notice_file,
        {"reason": "owner_reclaim", "requested_by": "bob", "grace_seconds": 600, "policy": "requeue"},
    )
    monkeypatch.setenv("EYVAN_RECLAIM_NOTICE_PATH", str(notice_file))
    monkeypatch.setenv("EYVAN_CHECKPOINT_RECEIPT_PATH", str(receipt_file))

    grpo = object.__new__(GRPOTrainer)
    grpo.arguments = types.SimpleNamespace(
        async_mode=True,
        sync_ref_model=False,
        startup_behavior_k3_max=None,
        save_directory=str(tmp_path / "grpo_ckpts"),
        log_all_workers=False,
        merge_lora_before_save=False,
        _get_save_directory=lambda create=True: tmp_path / "grpo_ckpts",
    )
    grpo.ref_state = None
    grpo._eyvan_reclaim_multihost_sync_available = None
    grpo._preemption_checkpoint_path = None
    grpo.checkpointer = object()
    grpo._save_checkpoint_for_step = lambda *, state, step, force=False, merge_lora_before_save=False: (
        str(tmp_path / "grpo_ckpts" / f"run-{step}") if force else None
    )

    state = _StateStub(step=9)
    with pytest.raises(EasyDeLPreemptionSignal, match="bob"):
        grpo.on_step_end(state=state, metrics=types.SimpleNamespace(other_metrics={}), step=9)

    assert grpo._preemption_checkpoint_path == str(tmp_path / "grpo_ckpts" / "run-9")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert receipt["checkpoint_uri"] == str(tmp_path / "grpo_ckpts" / "run-9")
    assert receipt["step"] == 9


def test_agent_flow_trainer_installs_handler_and_preempts_before_rollout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)
    notice_file = tmp_path / "af_notice.json"
    receipt_file = tmp_path / "af_receipt.json"
    _write_atomic_json(
        notice_file,
        {"reason": "owner_reclaim", "requested_by": "admin", "grace_seconds": 600, "policy": "requeue"},
    )
    monkeypatch.setenv("EYVAN_RECLAIM_NOTICE_PATH", str(notice_file))
    monkeypatch.setenv("EYVAN_CHECKPOINT_RECEIPT_PATH", str(receipt_file))

    installed: list[bool] = []
    monkeypatch.setattr(eh, "install_preemption_handler", lambda *a, **kw: installed.append(True))

    inner = object.__new__(Trainer)
    inner.arguments = TrainingArguments(
        model_name="Qwen/Qwen3-4B",
        use_wandb=False,
        save_directory=str(tmp_path / "af_ckpts"),
        use_eyvan_track=False,
    )
    inner._model = types.SimpleNamespace(
        config=types.SimpleNamespace(model_type="qwen3"),
        param_dtype="bfloat16",
        flops_per_token=lambda **kw: 1.0,
    )
    inner.model_state = _StateStub(step=6, model=inner._model)
    inner._initialize_attributes()
    inner.checkpointer = object()
    inner._save_checkpoint_for_step = lambda *, state, step, force=False, merge_lora_before_save=False: (
        str(tmp_path / "af_ckpts" / f"run-{step}") if force else None
    )
    inner.pruning_module = None

    af = object.__new__(AgentFlowTrainer)
    af.trainer = inner
    af.arguments = types.SimpleNamespace(max_workers=1, async_mode=False)
    af._rollout_call_count = 0
    af._extract_tasks = lambda batch: (_ for _ in ()).throw(AssertionError("rollouts should not launch on reclaim"))
    inner._preprocess_batch_input = af._wrapped_preprocess_batch_input

    # AgentFlowTrainer.train() installs the handler before delegating to inner trainer.
    inner.train = lambda *a, **kw: "delegated"
    assert af.train() == "delegated"
    assert installed == [True]

    # _execute_train_step catches EasyDeLPreemptionSignal cleanly when reclaim is active before rollout.
    state_out, _metrics, exc = inner._execute_train_step(state=inner.model_state, batch={"prompt": "hi"})
    assert state_out.step == 6
    assert isinstance(exc, EasyDeLPreemptionSignal)
    assert inner._preemption_checkpoint_path == str(tmp_path / "af_ckpts" / "run-6")
    assert json.loads(receipt_file.read_text(encoding="utf-8"))["checkpoint_uri"] == str(tmp_path / "af_ckpts" / "run-6")


# ---------------------------------------------------------------------------
# 6. Automatic Resume on Subsequent Rescheduled Attempt
# ---------------------------------------------------------------------------


def test_base_trainer_auto_resumes_from_eyvan_reclaimed_checkpoint_uri(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(eh, "_get_eyvan_preemption_callable", lambda name: None)

    saved_ckpt = tmp_path / "reclaimed_ckpts" / "run-17"
    saved_ckpt.mkdir(parents=True, exist_ok=True)
    (saved_ckpt / "metadata.json").write_text(json.dumps({"step": 17, "timestamp": 100.0}), encoding="utf-8")

    # 1. Direct checkpoint directory via EYVAN_RESUME_CHECKPOINT_URI
    monkeypatch.setenv("EYVAN_RESUME_CHECKPOINT_URI", str(saved_ckpt))

    load_calls: list[str] = []
    info_logs: list[str] = []
    monkeypatch.setattr(bt.logger, "info", lambda msg, *a, **kw: info_logs.append(str(msg)))

    resumed_model = types.SimpleNamespace(
        dtype="bfloat16",
        param_dtype="bfloat16",
        precision="default",
        model_task="causal-lm",
        _model_type="qwen3",
        flops_per_token=lambda **kw: 1.0,
        mesh=_NullMeshCtx(),
        config=types.SimpleNamespace(
            partition_axis=None,
            sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
            sharding_axis_dims=(1, 1, 1, 1, 1),
            sharding_dcn_axis_dims=None,
        ),
    )

    def _fake_load_state(*, load_directory, **kw):
        del kw
        load_calls.append(str(load_directory))
        return _StateStub(step=17, model=resumed_model)

    monkeypatch.setattr(bt.EasyDeLState, "load_state", staticmethod(_fake_load_state))
    monkeypatch.setattr(bt.jax, "eval_shape", lambda fn: resumed_model)

    class _MinimalResumeTrainer(Trainer):
        def initialize_trainer_utils(self):
            pass

        def _in_training_mode(self, state):
            return state

        def _apply_runtime_model_config_overrides(self):
            pass

        def _apply_step_start_point(self):
            pass

        def _setup_pose(self):
            return None, None

        def _setup_randomized_yarn(self):
            pass

    # Even with save_directory=None and resume_if_possible=False, Eyvan's explicit resume URI is honored!
    args = TrainingArguments(
        model_name="Qwen/Qwen3-8B",
        use_wandb=False,
        save_directory=None,
        resume_if_possible=False,
        use_eyvan_track=False,
    )
    args.get_tx_template = lambda *a, **kw: None  # type: ignore[method-assign]

    trainer = _MinimalResumeTrainer(
        arguments=args,
        model_state=_StateStub(step=0, model=resumed_model),
    )
    assert trainer._resumed_from_checkpoint is True
    assert trainer._eyvan_resumed_from_checkpoint == str(saved_ckpt)
    assert trainer.model_state.step == 17
    assert load_calls == [str(saved_ckpt)]
    assert any(f"Resuming from Eyvan reclaimed checkpoint: {saved_ckpt}" in msg for msg in info_logs), info_logs

    # 2. Parent directory via EYVAN_RESUME_CHECKPOINT_URI automatically resolves latest child run-17
    monkeypatch.setenv("EYVAN_RESUME_CHECKPOINT_URI", str(tmp_path / "reclaimed_ckpts"))
    load_calls.clear()
    info_logs.clear()

    trainer_parent = _MinimalResumeTrainer(
        arguments=args,
        model_state=_StateStub(step=0, model=resumed_model),
    )
    assert trainer_parent._resumed_from_checkpoint is True
    assert load_calls == [str(saved_ckpt)]
    assert any(f"Resuming from Eyvan reclaimed checkpoint: {saved_ckpt}" in msg for msg in info_logs), info_logs
