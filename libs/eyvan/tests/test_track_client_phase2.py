# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Histograms and RL episodes in the in-job ``eyvan.track`` client."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import eyvan.track as track
from eyvan.track import _format as fmt
from eyvan.track._store import LocalTrackingStore, TrackingStoreError

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class BrokenStore:
    @property
    def root(self) -> str:
        return "gs://unreachable/tracking/default/p"

    def put(self, key: str, data: bytes) -> None:
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


def test_histogram_bins_raw_values_without_numpy_and_counts_outliers(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="h")
    run.histogram(
        "adv",
        [[0.0, 0.5], [1.0, 2.0, -1.0], float("nan"), float("inf")],
        step=3,
        bins=2,
        range=(0.0, 1.0),
        axes={"env_steps": 24576},
    )
    run.finish()

    (record,) = _records(store, "h", fmt.STREAM_HIST)
    assert record["key"] == "adv" and record["step"] == 3
    assert record["edges"] == [0.0, 0.5, 1.0]
    # 0.0 -> bin 0; 0.5 and 1.0 (the last edge) -> bin 1.
    assert record["counts"] == [1, 2]
    assert (record["n"], record["under"], record["over"], record["nonfinite"]) == (3, 1, 1, 2)
    assert (record["min"], record["max"]) == (-1.0, 2.0)
    assert record["mean"] == pytest.approx(0.5)
    # Population std over the five finite values: E[x^2] = 6.25 / 5 = 1.25, minus mean^2 0.25.
    assert record["std"] == pytest.approx(1.0)
    assert record["x"] == {"env_steps": 24576.0}


def test_histogram_auto_range_and_degenerate_values(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="h")
    run.histogram("same", [2.0, 2.0, 2.0], step=0, bins=4)
    run.histogram("empty", [float("nan")], step=0, bins=2)
    run.finish()

    same, empty = sorted(_records(store, "h", fmt.STREAM_HIST), key=lambda r: str(r["key"]), reverse=True)
    assert same["edges"][0] == 1.5 and same["edges"][-1] == 2.5
    assert sum(same["counts"]) == 3
    assert empty["n"] == 0 and empty["nonfinite"] == 1 and empty["mean"] is None


def test_histogram_numpy_path_matches_stdlib_path(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="h")
    values = [-3.0, -1.0, 0.0, 0.25, 0.9, 1.0, 7.0]
    run.histogram("np", np.asarray(values), step=0, bins=4, range=(-1.0, 1.0))
    run.histogram("py", tuple(iter(values)).__iter__(), step=0, bins=4, range=(-1.0, 1.0))
    run.finish()

    by_key = {r["key"]: r for r in _records(store, "h", fmt.STREAM_HIST)}
    for field in ("edges", "counts", "n", "under", "over", "nonfinite", "min", "max"):
        assert by_key["np"][field] == by_key["py"][field], field
    assert by_key["np"]["mean"] == pytest.approx(by_key["py"]["mean"])


def test_prebinned_histogram_is_stored_as_given(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="h")
    run.histogram("q", counts=[4, 0, 6], edges=[0, 1, 2, 4], step=9)
    run.finish()

    (record,) = _records(store, "h", fmt.STREAM_HIST)
    assert record["counts"] == [4, 0, 6] and record["edges"] == [0.0, 1.0, 2.0, 4.0]
    assert record["n"] == 10 and record["mean"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": [1.0], "counts": [1], "edges": [0, 1]},
        {},
        {"counts": [1, 2], "edges": [0, 1]},
        {"counts": [1], "edges": [1, 1]},
        {"counts": [-1], "edges": [0, 1]},
        {"values": [1.0], "bins": 0},
        {"values": [1.0], "bins": fmt.MAX_HIST_BINS + 1},
        {"values": [1.0], "range": (1.0, 0.0)},
        {"values": [1.0], "step": -1},
        {"values": ["x"]},
    ],
)
def test_histogram_rejects_bad_arguments(tmp_path: Path, kwargs: dict[str, object]) -> None:
    run = _start(tmp_path, run_id="h")
    with pytest.raises((TypeError, ValueError)):
        run.histogram("k", **kwargs)  # type: ignore[arg-type]
    run.finish()


def test_episode_record_fields_and_default_step(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="e")
    run.log({"loss": 1.0}, step=41)
    run.episode(
        12.5,
        400,
        success=True,
        actor=3,
        env="Pong-v5",
        seed=7,
        policy_version=17,
        learner_version=19,
        checkpoint="ckpt-0040",
        axes={"env_steps": 1e6, "bad": float("nan")},
        extra={"lives_lost": 2},
    )
    run.episode(float("-inf"), kind="eval", step=5)
    run.finish()

    first, second = _records(store, "e", fmt.STREAM_EPISODES)
    assert first["step"] == 41
    assert (first["ret"], first["length"], first["success"], first["kind"]) == (12.5, 400, True, "train")
    assert (first["actor"], first["env"], first["seed"]) == ("3", "Pong-v5", 7)
    assert (first["policy_version"], first["learner_version"], first["checkpoint"]) == (17, 19, "ckpt-0040")
    assert first["x"] == {"env_steps": 1e6} and first["extra"] == {"lives_lost": 2.0}
    assert (second["ret"], second["kind"], second["step"], second["length"]) == ("-inf", "eval", 5, None)
    stats = [r for r in _records(store, "e", fmt.STREAM_META) if r["op"] == "stats"][-1]
    assert stats["dropped"]["invalid"] == 1


def test_episode_without_any_logged_step_has_null_step(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    run = _start(tmp_path, store, run_id="e")
    run.episode(1.0)
    run.finish()
    assert _records(store, "e", fmt.STREAM_EPISODES)[0]["step"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ret": "x"},
        {"ret": 1.0, "kind": "test"},
        {"ret": 1.0, "length": -1},
        {"ret": 1.0, "length": 2.5},
        {"ret": 1.0, "success": 1},
        {"ret": 1.0, "actor": "x" * 65},
        {"ret": 1.0, "seed": "7"},
        {"ret": 1.0, "extra": {f"k{i}": i for i in range(33)}},
        {"ret": 1.0, "extra": {"bad key": 1}},
        {"ret": 1.0, "step": -2},
    ],
)
def test_episode_rejects_bad_arguments(tmp_path: Path, kwargs: dict[str, object]) -> None:
    run = _start(tmp_path, run_id="e")
    with pytest.raises((TypeError, ValueError)):
        run.episode(**kwargs)  # type: ignore[arg-type]
    run.finish()


def test_actor_ranks_write_episodes_only_and_never_own_the_lifecycle(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    actor = _start(tmp_path, store, run_id="r", environment={"JAX_PROCESS_ID": "5"})
    assert (actor.active, actor.episodes_active) == (False, True)
    actor.log({"loss": 1.0}, step=0)
    actor.histogram("adv", [1.0], step=0)
    actor.episode(3.0, actor="a5", step=0)
    actor.finish()

    assert store.list("r/scalars") == [] and store.list("r/hist") == []
    (episode,) = _records(store, "r", fmt.STREAM_EPISODES)
    assert episode["actor"] == "a5"
    ops = [record["op"] for record in _records(store, "r", fmt.STREAM_META)]
    assert ops == ["stats"]
    for key in store.list("r"):
        assert fmt.SegmentName.parse(key.rsplit("/", 1)[1]).rank == 5  # type: ignore[union-attr]

    silent = _start(tmp_path, store, run_id="s", environment={"JAX_PROCESS_ID": "5"}, episode_ranks="zero")
    silent.episode(1.0)
    silent.finish()
    assert store.list("s") == []
    with pytest.raises(ValueError):
        _start(tmp_path, store, run_id="t", episode_ranks="some")


def test_episode_and_histogram_caps_drop_visibly(tmp_path: Path) -> None:
    store = LocalTrackingStore(tmp_path / "store")
    clock = FakeClock()
    limits = track.TrackLimits(max_hist_keys=1, max_episodes_per_s=2)
    run = _start(tmp_path, store, run_id="c", limits=limits, clock=clock)
    run.histogram("a", [1.0], step=0)
    run.histogram("b", [1.0], step=0)  # second key exceeds max_hist_keys
    for _ in range(3):  # third in the same second exceeds the episode rate
        run.episode(1.0)
    run.finish()

    assert [r["key"] for r in _records(store, "c", fmt.STREAM_HIST)] == ["a"]
    assert len(_records(store, "c", fmt.STREAM_EPISODES)) == 2
    stats = [r for r in _records(store, "c", fmt.STREAM_META) if r["op"] == "stats"][-1]
    assert stats["dropped"]["keys"] == 1 and stats["dropped"]["rate"] == 1
    assert stats["records"] == 3


def test_episodes_and_histograms_spill_and_replay_when_the_store_is_down(tmp_path: Path) -> None:
    spill = tmp_path / "spill"
    run = _start(tmp_path, BrokenStore(), run_id="off", spill_dir=spill)
    run.histogram("adv", [0.1, 0.2], step=0)
    run.episode(5.0, step=0)
    assert run.finish() is False

    target = tmp_path / "recovered"
    result = track.replay_spill(spill, store_uri=target.as_uri())
    assert result.failed == 0 and result.replayed >= 3
    recovered = LocalTrackingStore(target)
    assert _records(recovered, "off", fmt.STREAM_HIST)[0]["counts"] == [1] * 1 + [0] * 62 + [1]
    assert _records(recovered, "off", fmt.STREAM_EPISODES)[0]["ret"] == 5.0


def test_histogram_does_not_import_numpy_itself() -> None:
    probe = (
        "import sys, eyvan.track as t, tempfile; "
        "r = t.start(run_id='x', uri=tempfile.mkdtemp(), spill_dir=tempfile.mkdtemp(), environment={}); "
        "r.histogram('k', [1.0, 2.0], step=0); r.episode(1.0); r.finish(); print('numpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
