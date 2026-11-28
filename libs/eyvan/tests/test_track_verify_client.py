# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Executed regressions for the independent tracking-client review findings."""

from __future__ import annotations

import errno
import itertools
import math
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

import eyvan.track as track
from eyvan.track import _format as fmt
from eyvan.track import _rich
from eyvan.track._store import LocalTrackingStore

PNG = b"\x89PNG\r\n\x1a\n" + bytes(32)
MP4 = b"\x00\x00\x00\x18ftypmp42" + bytes(40)


def _start(tmp_path: Path, **kwargs: object) -> track.Run:
    kwargs.setdefault("store", LocalTrackingStore(tmp_path / "store"))
    kwargs.setdefault("run_id", "verified")
    kwargs.setdefault("environment", {})
    kwargs.setdefault("flush_interval_s", 3600)
    kwargs.setdefault("spill_dir", tmp_path / "spill")
    return track.start(**kwargs)  # type: ignore[arg-type]


def _records(run: track.Run, stream: str) -> list[dict[str, object]]:
    store = run._store
    return [record for key in store.list(f"{run.id}/{stream}/") for record in fmt.decode_segment(store.get(key))]


class SelectiveStore(LocalTrackingStore):
    """Persist everything except the selected class of uploads during an outage."""

    fail = ""

    def put(self, key: str, data: bytes) -> None:
        if self.fail and self.fail in key:
            raise OSError(errno.EIO, "injected storage failure")
        super().put(key, data)


@pytest.mark.parametrize("operation", ["flush", "finish"])
def test_04_upload_timeout_bounds_blocking_put(tmp_path: Path, operation: str) -> None:
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    class BlockingStore(LocalTrackingStore):
        def put(self, key: str, data: bytes) -> None:
            entered.set()
            release.wait(5)
            super().put(key, data)

    run = _start(tmp_path, store=BlockingStore(tmp_path / "store"))
    run.log({"loss": 1})
    result: list[bool] = []

    def upload() -> None:
        result.append(getattr(run, operation)(timeout=0.03))
        returned.set()

    caller = threading.Thread(target=upload)
    caller.start()
    try:
        assert entered.wait(1), "upload never started"
        assert returned.wait(0.5), "caller is blocked in store.put past its timeout"
        assert result == [False]
        markers = list((tmp_path / "spill").rglob("*.json"))
        assert markers, "unacknowledged payloads must be spilled before returning"
    finally:
        release.set()
        caller.join(3)
        run.finish(timeout=1)
        if run._thread is not None:
            run._thread.join(3)


def test_04_process_exits_after_a_timed_out_upload(tmp_path: Path) -> None:
    source = """
import sys, time
import eyvan.track as track
class Store:
    root = sys.argv[1]
    def put(self, key, data):
        time.sleep(60)
r = track.start(store=Store(), environment={}, spill_dir=sys.argv[2], flush_interval_s=3600)
r.log({'loss': 1})
print(r.finish(timeout=0.03), flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path / "store"), str(tmp_path / "spill")],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("operation", ["media", "blob", "rollout"])
@pytest.mark.parametrize("error", [OSError(errno.EIO, "disk error"), PermissionError("denied")])
def test_05_media_source_errors_are_counted_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, error: OSError
) -> None:
    path = tmp_path / ("video.mp4" if operation == "rollout" else "frame.png")
    path.write_bytes(MP4 if operation == "rollout" else PNG)
    run = _start(tmp_path)
    original_open = Path.open
    original_read = Path.read_bytes

    def read(self: Path, *args: object, **kwargs: object):
        if self == path:
            raise error
        return original_read(self, *args, **kwargs)

    def open_path(self: Path, *args: object, **kwargs: object):
        if self == path:
            raise error
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(Path, "open", open_path)
    try:
        result = (
            run.rollout(tracks={"reward": [1]}, video=path)
            if operation == "rollout"
            else run.media("frame", path)
            if operation == "media"
            else run.blob(path)
        )
        assert result is (False if operation == "rollout" else None)
        assert run.stats()["dropped"]["invalid"] == 1
        run.finish()
        assert _records(run, "meta")[-2]["dropped"]["invalid"] == 1
    finally:
        run.finish()


@pytest.mark.parametrize("operation", ["media", "blob", "rollout"])
@pytest.mark.parametrize("state", ["inactive", "finished"])
def test_05_inactive_and_finished_media_never_read_files(tmp_path: Path, operation: str, state: str) -> None:
    env = {"JAX_PROCESS_ID": "2"} if state == "inactive" else {}
    run = _start(tmp_path, environment=env, episode_ranks="zero")
    if state == "finished":
        run.finish()
    path = tmp_path / ("absent.mp4" if operation == "rollout" else "absent.png")
    try:
        if operation == "rollout":
            assert run.rollout(tracks={"reward": [1]}, video=path) is False
        elif operation == "media":
            assert run.media("frame", path) is None
        else:
            assert run.blob(path) is None
        assert run.stats()["dropped"]["invalid"] == 0
    finally:
        run.finish()


def test_06_segment_order_is_reserved_when_draining(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _start(tmp_path)
    run.flush()
    drained, resume = threading.Event(), threading.Event()
    original = run._drain

    def drain(*, force_stats: bool):
        result = original(force_stats=force_stats)
        if threading.current_thread().name == "older-flush":
            drained.set()
            assert resume.wait(3)
        return result

    monkeypatch.setattr(run, "_drain", drain)
    run.log({"loss": 1}, step=5)
    run.histogram("h", counts=[1], edges=[0, 1], step=5)
    caller = threading.Thread(target=run.flush, name="older-flush")
    caller.start()
    try:
        assert drained.wait(1)
        run.log({"loss": 2}, step=5)
        run.histogram("h", counts=[2], edges=[0, 1], step=5)
        assert run.flush(timeout=1)
    finally:
        resume.set()
        caller.join(3)
        run.finish()
    for stream in ("scalars", "hist"):
        keys = run._store.list(f"{run.id}/{stream}/")
        latest = max(keys, key=lambda key: fmt.SegmentName.parse(key.rsplit("/", 1)[1]).seq)
        (record,) = fmt.decode_segment(run._store.get(latest))
        assert (record["v"]["loss"] if stream == "scalars" else record["counts"][0]) == 2


CALLS: list[tuple[str, Callable[[track.Run], object]]] = [
    ("log", lambda r: r.log({"loss": 1})),
    ("histogram", lambda r: r.histogram("h", [1])),
    ("episode", lambda r: r.episode(1)),
    ("rollout", lambda r: r.rollout(tracks={"reward": [1]})),
    ("completions", lambda r: r.completions("p", ["r"], rewards=[1])),
    ("table", lambda r: r.table("t", [[1]], columns=["a"])),
    ("media", lambda r: r.media("m", PNG, content_type="image/png")),
    ("blob", lambda r: r.blob(PNG, content_type="image/png")),
    ("config", lambda r: r.config({"lr": 1})),
    ("summary", lambda r: r.summary(best=1)),
]


@pytest.mark.parametrize(("name", "call"), CALLS, ids=[name for name, _ in CALLS])
def test_07_recording_race_cannot_enqueue_after_finish(
    tmp_path: Path, name: str, call: Callable[[track.Run], object]
) -> None:
    run = _start(tmp_path)
    before_lock, resume = threading.Event(), threading.Event()
    original = run._lock

    class GateLock:
        def __enter__(self):
            if threading.current_thread().name == "racing-log":
                before_lock.set()
                assert resume.wait(3)
            original.acquire()

        def __exit__(self, *args: object) -> None:
            original.release()

    run._lock = GateLock()  # type: ignore[assignment]
    caller = threading.Thread(target=lambda: call(run), name="racing-log")
    caller.start()
    try:
        assert before_lock.wait(1), name
        run.finish()
        stats = run.stats()
    finally:
        resume.set()
        caller.join(3)
    assert run.stats() == stats
    assert run._pending == 0 and not any(run._buffers.values()) and not run._blob_queue
    assert not run._config


def test_07_finished_calls_are_noops_even_with_invalid_arguments(tmp_path: Path) -> None:
    run = _start(tmp_path)
    run.finish()
    run.log(None)  # type: ignore[arg-type]
    run.histogram("bad key")
    run.episode(None)
    assert run.rollout() is False
    run.completions(None, [])  # type: ignore[arg-type]
    run.table("bad key", None)
    assert run.media("bad key", None) is None
    assert run.blob(None) is None
    run.config(None)  # type: ignore[arg-type]
    run.summary(best=1)
    assert run.finish("bad status") is True
    assert run.flush(timeout=0) is True


def test_07_flush_racing_shutdown_does_not_wait_on_a_stopped_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _start(tmp_path)
    before_drain, resume, returned = threading.Event(), threading.Event(), threading.Event()
    original = run._drain

    def drain(*, force_stats: bool):
        if threading.current_thread().name == "late-flush":
            before_drain.set()
            assert resume.wait(3)
        return original(force_stats=force_stats)

    monkeypatch.setattr(run, "_drain", drain)

    def flush() -> None:
        run.flush()
        returned.set()

    caller = threading.Thread(target=flush, name="late-flush", daemon=True)
    caller.start()
    try:
        assert before_drain.wait(1)
        run.finish()
        if run._thread is not None:
            run._thread.join(1)
        resume.set()
        assert returned.wait(0.5), "flush queued work after the writer stopped"
        assert not run._batches and not run._inflight
    finally:
        resume.set()
        for batch in list(run._inflight):
            run._expire_batch(batch)
        caller.join(1)


def test_13_oversized_media_is_rejected_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "huge.mp4"
    with path.open("wb") as file:
        file.truncate(_rich.MAX_BLOB_BYTES + 1)
    reads: list[Path] = []
    original = Path.read_bytes

    def read(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", read)
    run = _start(tmp_path)
    try:
        with pytest.raises(ValueError, match="limit"):
            run.media("video", path)
        assert path not in reads, "oversized path was read before the size check"
    finally:
        run.finish()


def test_13_media_read_stays_bounded_if_file_grows_after_stat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "growing.mp4"
    path.write_bytes(MP4)
    reads: list[int] = []
    original = Path.open

    class GrowingSource:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            return b"x" * (_rich.MAX_BLOB_BYTES + 1)

    def open_path(self: Path, *args: object, **kwargs: object):
        return GrowingSource() if self == path else original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    run = _start(tmp_path)
    try:
        with pytest.raises(ValueError, match="limit"):
            run.media("video", path)
        assert reads == [_rich.MAX_BLOB_BYTES + 1]
    finally:
        run.finish()


def test_14_failed_blob_spill_is_counted_and_identical_bytes_can_retry(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block spill")
    store = SelectiveStore(tmp_path / "store")
    store.fail = "/blobs/"
    run = _start(tmp_path, store=store, spill_dir=blocker)
    ref = run.blob(PNG, content_type="image/png")
    assert run.flush(timeout=1) is False
    assert run.stats()["dropped"]["queue"] >= 1, "failed blob spill was invisible"
    assert run._blob_total == 0
    store.fail = ""
    assert run.blob(PNG, content_type="image/png") == ref
    assert run.flush(timeout=1)
    run.finish()
    assert store.get(fmt.blob_key(run.id, ref["sha256"])) == PNG


def test_15_final_stats_include_losses_from_final_upload(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block spill")
    store = SelectiveStore(tmp_path / "store")
    store.fail = "/scalars/"
    run = _start(tmp_path, store=store, spill_dir=blocker)
    run.log({"loss": 1})
    assert run.finish() is False
    stats = [r for r in _records(run, "meta") if r["op"] == "stats"][-1]
    assert stats["dropped"]["queue"] == run.stats()["dropped"]["queue"] == 1


def test_15_finish_waits_for_earlier_flush_outcomes_before_final_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SelectiveStore(tmp_path / "store")
    store.fail = "/scalars/"
    run = _start(tmp_path, store=store)
    run.flush()
    drained, resume = threading.Event(), threading.Event()
    original = run._drain

    def drain(*, force_stats: bool):
        batch = original(force_stats=force_stats)
        if threading.current_thread().name == "paused-flush":
            drained.set()
            assert resume.wait(3)
        return batch

    monkeypatch.setattr(run, "_drain", drain)
    run.log({"loss": 1})
    caller = threading.Thread(target=run.flush, name="paused-flush")
    caller.start()
    try:
        assert drained.wait(1)
        assert run.finish(timeout=0.1) is False
    finally:
        resume.set()
        caller.join(3)
    # The corrected final stats may themselves be spilled after the deadline.
    stats = [r for r in _records(run, "meta") if r["op"] == "stats"]
    for path in (tmp_path / "spill").rglob("*.seg"):
        stats.extend(r for r in fmt.decode_segment(path.read_bytes()) if r.get("op") == "stats")
    assert stats and stats[-1]["spilled"] >= 1
    assert run.stats()["spilled"] >= 1


def test_16_spill_backlog_is_bounded_and_disk_replay_retains_every_record(tmp_path: Path) -> None:
    store = SelectiveStore(tmp_path / "store")
    store.fail = "/"
    run = _start(tmp_path, store=store, limits=track.TrackLimits(max_queued=2))
    try:
        for step in range(25):
            run.log({"loss": step}, step=step)
            assert run.flush(timeout=1) is False
        assert len(run._spill_backlog) <= run._limits.max_queued
        store.fail = ""
        assert run.flush(timeout=1)
    finally:
        run.finish()
    result = track.replay_spill(tmp_path / "spill")
    assert result.failed == 0
    assert sorted(r["step"] for r in _records(run, "scalars")) == list(range(25))


@pytest.mark.parametrize("operation", ["rollout", "media", "video"])
def test_17_queue_rejection_returns_failure_without_orphan_blobs(tmp_path: Path, operation: str) -> None:
    run = _start(tmp_path, limits=track.TrackLimits(max_queued=1))
    try:
        if operation == "media":
            result = run.media("frame", PNG, content_type="image/png")
        else:
            result = run.rollout(tracks={"reward": [1]}, video=MP4 if operation == "video" else None)
        assert result is (None if operation == "media" else False)
        assert run.stats()["records"] == 0
        assert run.stats()["dropped"]["queue"] == 1
        assert run._blob_total == 0 and not run._blob_queue
    finally:
        run.finish()
    assert run._store.list(f"{run.id}/blobs/") == []


def test_18_episode_policy_is_independent_of_scalar_policy(tmp_path: Path) -> None:
    run = _start(tmp_path, environment={"JAX_PROCESS_ID": "3"}, ranks="all", episode_ranks="zero")
    try:
        run.log({"loss": 1})
        run.episode(1)
    finally:
        run.finish()
    assert run.active and not run.episodes_active
    assert len(_records(run, "scalars")) == 1 and _records(run, "episodes") == []


def test_19_inactive_completion_ranks_intentionally_record_nothing(tmp_path: Path) -> None:
    run = _start(tmp_path, environment={"JAX_PROCESS_ID": "2"})
    run.completions("p", ["r"], rewards=[1])
    run.finish()
    assert _records(run, "completions") == []
    assert run.stats() == {"records": 0, "dropped": {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}, "spilled": 0}


@pytest.mark.parametrize("value", [1e20, -1e20, sys.float_info.max, -sys.float_info.max])
def test_20_constant_large_finite_histograms_have_distinct_finite_edges(tmp_path: Path, value: float) -> None:
    run = _start(tmp_path)
    try:
        run.histogram("h", iter([value]), bins=64)
    finally:
        run.finish()
    (record,) = _records(run, "hist")
    edges = record["edges"]
    assert len(edges) == 65 and all(math.isfinite(e) for e in edges)
    assert all(a < b for a, b in itertools.pairwise(edges))
    assert sum(record["counts"]) == 1


@pytest.mark.parametrize("operation", ["completions", "histogram"])
def test_21_large_finite_variances_do_not_overflow(tmp_path: Path, operation: str) -> None:
    run = _start(tmp_path)
    try:
        if operation == "completions":
            run.completions("p", ["a", "b"], rewards=[-1e200, 1e200])
        else:
            run.histogram("h", iter([-1e200, 1e200]), bins=2)
    finally:
        run.finish()
    (record,) = _records(run, "completions" if operation == "completions" else "hist")
    if operation == "completions":
        assert [r["advantage"] for r in record["responses"]] == pytest.approx([-1, 1])
    else:
        assert record["mean"] == 0 and record["std"] == pytest.approx(1e200)


@pytest.mark.parametrize("size", [3, 7, 49])
def test_21_constant_large_rewards_keep_zero_advantages(tmp_path: Path, size: int) -> None:
    run = _start(tmp_path)
    run.completions("p", ["r"] * size, rewards=[1e200] * size)
    run.histogram("h", iter([1e200] * size))
    run.finish()
    (record,) = _records(run, "completions")
    assert [response["advantage"] for response in record["responses"]] == [0] * size
    assert _records(run, "hist")[0]["std"] == 0


def test_22_numpy_and_stdlib_use_the_serialized_edges(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    run = _start(tmp_path)
    run.histogram("np", np.asarray([0.3]), bins=10, range=(0, 1))
    run.histogram("py", iter([0.3]), bins=10, range=(0, 1))
    run.finish()
    records = _records(run, "hist")
    left, right = records
    assert left["edges"] == right["edges"] and left["counts"] == right["counts"]
    edges = left["edges"]
    for i, count in enumerate(left["counts"]):
        if count:
            assert edges[i] <= 0.3 < edges[i + 1]


def test_23_free_text_table_columns_are_accepted_by_design(tmp_path: Path) -> None:
    run = _start(tmp_path)
    run.table("eval", [[1, "a"]], columns=["pass@k", "case name"])
    run.table("inferred", [{"pass@k": 1, "case name": "a"}])
    run.finish()
    assert [r["columns"] for r in _records(run, "tables")] == [["pass@k", "case name"]] * 2
