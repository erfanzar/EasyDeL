# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Behaviour of the in-job ``eyvan.track`` client: what it writes, and what it survives."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import eyvan.track as track
from eyvan.track import _format as fmt
from eyvan.track._store import LocalTrackingStore, SegmentConflictError, TrackingStoreError, store_for_uri

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class BrokenStore:
    """A store whose every upload fails, like an unreachable bucket."""

    def __init__(self, root: str = "gs://unreachable/tracking/default/p") -> None:
        self._root = root
        self.attempts = 0

    @property
    def root(self) -> str:
        return self._root

    def put(self, key: str, data: bytes) -> None:
        self.attempts += 1
        raise TrackingStoreError("network unreachable")

    def get(self, key: str) -> bytes:
        raise TrackingStoreError("network unreachable")

    def list(self, prefix: str = "") -> list[str]:
        raise TrackingStoreError("network unreachable")


def _records(store: LocalTrackingStore, run_id: str, stream: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for key in store.list(f"{run_id}/{stream}"):
        out.extend(fmt.decode_segment(store.get(key)))
    return out


def _start(tmp_path: Path, store: object | None = None, **kwargs: object) -> track.Run:
    kwargs.setdefault("environment", {})
    kwargs.setdefault("flush_interval_s", 3600.0)
    kwargs.setdefault("spill_dir", tmp_path / "spill")
    return track.start(store=store or LocalTrackingStore(tmp_path / "store"), **kwargs)  # type: ignore[arg-type]


def test_logged_scalars_and_lifecycle_reach_the_store_as_segments(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    clock = FakeClock()
    run = _start(tmp_path, store, run_id="ppo-1", name="ppo", group="sweep-3", tags=("pong",), clock=clock)
    run.config({"lr": 2.5e-4, "envs": 64})
    run.log({"loss/policy": 0.31, "entropy": 1.2}, step=0, axes={"env_steps": 8192})
    clock.now += 1
    run.log({"loss/policy": float("nan"), "grad": float("inf")})
    run.summary(best_return=21.0)

    assert run.finish() is True

    scalars = _records(store, "ppo-1", fmt.STREAM_SCALARS)
    assert [record["step"] for record in scalars] == [0, 1]
    assert scalars[0]["v"] == {"loss/policy": 0.31, "entropy": 1.2}
    assert scalars[0]["x"] == {"env_steps": 8192.0}
    assert scalars[1]["v"] == {"loss/policy": "nan", "grad": "inf"}
    assert "x" not in scalars[1]

    meta = _records(store, "ppo-1", fmt.STREAM_META)
    ops = [record["op"] for record in meta]
    assert ops[0] == "start" and ops[-1] == "finish"
    start = meta[0]
    assert (start["name"], start["group"], start["tags"], start["attempt"], start["rank"]) == (
        "ppo",
        "sweep-3",
        ["pong"],
        1,
        0,
    )
    assert start["fmt"] == fmt.FORMAT_VERSION
    config = next(record for record in meta if record["op"] == "config")
    assert config["config"] == {"lr": 2.5e-4, "envs": 64}
    stats = next(record for record in meta if record["op"] == "stats")
    assert stats["records"] == 2
    assert stats["dropped"] == {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}
    assert meta[-1]["status"] == "finished"
    for key in store.list("ppo-1"):
        name = fmt.SegmentName.parse(key.rsplit("/", 1)[1])
        assert name is not None and (name.attempt, name.rank) == (1, 0)


def test_a_workload_retry_resumes_the_same_run_under_the_dispatched_scope(tmp_path: Path) -> None:
    scope = tmp_path / "tracking" / "acme" / "rl"
    ids = []
    for attempt in ("1", "2"):
        environment = {
            "EYVAN_WORKLOAD_ID": "wl_7358abc",
            "EYVAN_ATTEMPT": attempt,
            fmt.TRACKING_URI_ENV: scope.as_uri(),
        }
        run = track.start("ppo", environment=environment, flush_interval_s=3600, spill_dir=tmp_path / "spill")
        run.log({"loss": 1.0}, step=5)
        run.finish()
        ids.append(run.id)
        assert run.workload_id == "wl_7358abc"

    assert ids == ["wl_7358abc", "wl_7358abc"]
    attempts = {
        fmt.SegmentName.parse(key.rsplit("/", 1)[1]).attempt  # type: ignore[union-attr]
        for key in LocalTrackingStore(scope).list("wl_7358abc/scalars")
    }
    assert attempts == {1, 2}


def test_nonzero_rank_records_nothing_by_default_but_still_validates(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="r", environment={"JAX_PROCESS_ID": "3"})

    assert (run.rank, run.active) == (3, False)
    run.log({"loss": 1.0}, step=1)
    with pytest.raises(ValueError):
        run.log({"bad key!": 1.0})
    assert run.finish() is True
    assert store.list() == []

    everyone = _start(tmp_path, store, run_id="r2", environment={"JAX_PROCESS_ID": "3"}, ranks="all")
    everyone.log({"loss": 1.0}, step=1)
    everyone.finish()
    assert _records(store, "r2", fmt.STREAM_SCALARS)[0]["v"] == {"loss": 1.0}


def test_storage_failure_never_raises_and_spilled_segments_replay_exactly_once(tmp_path: Path) -> None:
    broken = BrokenStore()
    spill = tmp_path / "spill"
    run = _start(tmp_path, broken, run_id="offline", spill_dir=spill)
    run.log({"loss": 0.5}, step=0)
    run.log({"loss": 0.4}, step=1)

    assert run.finish() is False
    assert broken.attempts > 0
    assert run.stats()["spilled"] >= 2
    spilled = sorted(path.name for path in (spill / "offline").iterdir())
    assert spilled and all(name.endswith((".json", ".seg")) for name in spilled)

    target = tmp_path / "recovered"
    result = track.replay_spill(spill, store_uri=target.as_uri())
    assert (result.replayed, result.failed) == (len(spilled) // 2, 0)
    assert list((spill / "offline").iterdir()) == []
    recovered = LocalTrackingStore(target)
    assert [record["v"] for record in _records(recovered, "offline", "scalars")] == [{"loss": 0.5}, {"loss": 0.4}]
    assert track.replay_spill(spill, store_uri=target.as_uri()).replayed == 0


def test_replay_reports_failures_without_raising(tmp_path: Path) -> None:
    run = _start(tmp_path, BrokenStore("gs://still-broken/x"), run_id="offline")
    run.log({"loss": 0.5}, step=0)
    run.finish()

    result = track.replay_spill(tmp_path / "spill", store_uri=str(tmp_path / "unwritable" / "\x00bad"))

    assert result.replayed == 0 and result.failed > 0 and result.errors


def test_caps_drop_visibly_instead_of_growing_without_bound(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    clock = FakeClock()
    limits = track.TrackLimits(max_keys=2, max_records_per_s=3)
    run = _start(tmp_path, store, run_id="capped", limits=limits, clock=clock)
    run.log({"a": 1, "b": 2, "c": 3}, step=0)  # "c" exceeds max_keys
    run.log({"a": 1}, step=1)
    run.log({"a": 1}, step=2)
    run.log({"a": 1}, step=3)  # fourth call in the same second exceeds the rate
    clock.now += 1
    run.log({"a": 1}, step=4, axes={"env_steps": float("nan")})
    run.finish()

    scalars = _records(store, "capped", fmt.STREAM_SCALARS)
    assert [record["step"] for record in scalars] == [0, 1, 2, 4]
    assert scalars[0]["v"] == {"a": 1.0, "b": 2.0}
    assert "x" not in scalars[-1]
    stats = [record for record in _records(store, "capped", fmt.STREAM_META) if record["op"] == "stats"][-1]
    assert stats["dropped"] == {"keys": 1, "rate": 1, "invalid": 1, "queue": 0}


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda run: run.log([("a", 1)]), TypeError),
        (lambda run: run.log({"a": "x"}), TypeError),
        (lambda run: run.log({"": 1}), ValueError),
        (lambda run: run.log({"a": 1}, step=-1), ValueError),
        (lambda run: run.log({"a": 1}, axes={"bad axis": 1}), ValueError),
        (lambda run: run.finish("done"), ValueError),
    ],
)
def test_bad_arguments_still_raise(tmp_path: Path, call, error) -> None:
    run = _start(tmp_path, run_id="args")
    with pytest.raises(error):
        call(run)
    run.finish()


def test_start_rejects_malformed_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _start(tmp_path, run_id="../escape")
    with pytest.raises(ValueError):
        _start(tmp_path, ranks="some")
    with pytest.raises(ValueError):
        _start(tmp_path, tags=("",))
    with pytest.raises(ValueError):
        track.start(uri="s3://bucket/x", environment={})


def test_context_manager_records_failure_and_finish_is_idempotent(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    with pytest.raises(RuntimeError), _start(tmp_path, store, run_id="boom") as run:
        run.log({"loss": 1.0}, step=0)
        raise RuntimeError("diverged")

    assert run.finished
    assert run.finish() is True
    finishes = [record for record in _records(store, "boom", fmt.STREAM_META) if record["op"] == "finish"]
    assert [record["status"] for record in finishes] == ["failed"]


def test_background_writer_flushes_when_a_batch_fills(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="batch", limits=track.TrackLimits(batch_records=5, max_records_per_s=100))
    for step in range(5):
        run.log({"loss": float(step)}, step=step)

    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not store.list("batch/scalars"):
        time.sleep(0.02)
    assert len(_records(store, "batch", fmt.STREAM_SCALARS)) == 5
    run.finish()


def test_local_store_segments_are_create_if_absent(tmp_path: Path) -> None:
    store = store_for_uri(str(tmp_path / "s"))
    store.put("r/scalars/x.ndjson.gz", b"one")
    store.put("r/scalars/x.ndjson.gz", b"one")
    with pytest.raises(SegmentConflictError):
        store.put("r/scalars/x.ndjson.gz", b"two")
    with pytest.raises(ValueError):
        store.put("../outside", b"x")
    assert store.list() == ["r/scalars/x.ndjson.gz"]


def test_run_ids_from_workloads_are_safe_path_segments() -> None:
    assert fmt.run_id_for_workload("wl_7358…e176") == "wl_7358-e176"
    with pytest.raises(ValueError):
        fmt.run_id_for_workload("///")
    assert fmt.decode_value(fmt.encode_value(float("-inf"))) == float("-inf")


def test_importing_the_client_does_not_pull_server_dependencies() -> None:
    probe = (
        "import sys, eyvan.track as t; t.start; "
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'fastapi','httpx','pydantic','sqlalchemy','starlette','uvicorn'} or m.startswith('eyvan.track.tracker')); "
        "print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
