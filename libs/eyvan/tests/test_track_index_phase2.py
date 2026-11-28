# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Phase-two index contracts exercised through real immutable local segments."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker.index import TrackingIndex


def put(store: LocalTrackingStore, stream: str, records: list[dict[str, Any]], **identity: Any) -> str:
    """Write one real phase-two segment, with optional segment identity fields."""

    name = SegmentName(identity.get("attempt", 1), identity.get("rank", 0), "bbbbbbbbbbbb", identity.get("seq", 0))
    key = f"org/project/run/{stream}/{name.filename}"
    store.put(key, encode_segment(records))
    return key


def hist(step: int = 1, **updates: Any) -> dict[str, Any]:
    """Return a complete histogram record."""

    return {
        "key": "advantages",
        "step": step,
        "t": 100.0 + step,
        "edges": [0.0, 1.0, 2.0],
        "counts": [2, 4],
        "n": 6,
        "under": 0,
        "over": 0,
        "nonfinite": 0,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "x": {},
        **updates,
    }


def episode(step: int | None = 1, ret: Any = 1, **updates: Any) -> dict[str, Any]:
    """Return a complete episode record."""

    return {
        "t": 100.0 + (step or 0),
        "step": step,
        "ret": ret,
        "length": 10,
        "success": None,
        "kind": "train",
        "actor": None,
        "env": None,
        "seed": None,
        "policy_version": None,
        "learner_version": None,
        "checkpoint": None,
        "x": {},
        "extra": {},
        **updates,
    }


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[LocalTrackingStore, TrackingIndex]:
    return LocalTrackingStore(tmp_path / "store"), TrackingIndex(tmp_path / "index.sqlite3")


def snapshot(index: TrackingIndex) -> tuple[Any, ...]:
    """Capture query results whose identity must survive a rebuild."""

    return (
        index.list_runs("org", "project"),
        index.run_detail("org", "project", "run"),
        index.histograms("org", "project", "run", "advantages"),
        index.episodes("org", "project", "run"),
        index.episode_curve("org", "project", "run"),
        index.episode_breakdown("org", "project", "run"),
    )


def test_ingest_rebuild_detail_and_episode_only_attempt(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    now = time.time()
    put(store, "meta", [{"op": "start", "t": 90, "name": "PPO", "workload_id": "wl-1"}])
    put(store, "hist", [hist(8, t=95)])
    put(store, "episodes", [episode(12, t=now, actor="2", env="Pong", x={"frames": 200})], attempt=2, rank=3)
    result = index.scan(store)
    assert (result.ingested_segments, result.errors) == (3, 0)
    detail = index.run_detail("org", "project", "run")
    assert detail["name"] == "PPO" and detail["workload_id"] == "wl-1"
    assert detail["status"] == "running" and detail["last_step"] == 12
    assert detail["updated_at"] == now and detail["started_at"] == 90
    assert detail["hist_keys"] == ["advantages"]
    assert detail["episodes"] == {"train": 1, "eval": 0}
    assert detail["episode_axes"] == ["frames"] and detail["envs"] == ["Pong"] and detail["actors"] == 1
    assert detail["attempt_list"] == [
        {"attempt": 1, "started_at": 90, "first_step": 8, "last_step": 8, "status": None},
        {"attempt": 2, "started_at": now, "first_step": 12, "last_step": 12, "status": None},
    ]
    before = snapshot(index)
    assert index.scan(store).ingested_segments == 0
    assert index.rebuild(store).ingested_segments == 3
    assert snapshot(index) == before
    fresh = TrackingIndex(index.db_path.with_name("fresh.sqlite3"))
    fresh.scan(store)
    assert snapshot(fresh) == before


def test_phase_one_database_upgrade_keeps_old_rows(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "scalars", [{"t": 10, "step": 3, "v": {"loss": 2}}])
    index.scan(store)
    with sqlite3.connect(index.db_path) as db:
        db.execute("DROP TABLE histogram_records")
        db.execute("DROP TABLE episode_records")
    upgraded = TrackingIndex(index.db_path)
    put(store, "hist", [hist(4)])
    upgraded.scan(store)
    assert upgraded.series("org", "project", "run", ["loss"])["series"][0]["total"] == 1
    assert upgraded.histograms("org", "project", "run", "advantages")["total"] == 1


@pytest.mark.parametrize("stream,record", [("hist", hist()), ("episodes", episode())])
def test_corrupt_and_invalid_segments_are_atomic_and_isolated(
    indexed: tuple[LocalTrackingStore, TrackingIndex], stream: str, record: dict[str, Any]
) -> None:
    store, index = indexed
    broken = f"org/project/run/{stream}/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}"
    store.put(broken, b"not gzip")
    put(store, stream, [record, {**record, "t": "bad"}], seq=1)
    put(store, stream, [record], seq=2)
    assert index.scan(store).errors == 2
    assert len(index.run_detail("org", "project", "run")["errors"]) == 2
    result = (
        index.histograms("org", "project", "run", "advantages")
        if stream == "hist"
        else index.episodes("org", "project", "run")
    )
    assert result["total"] == 1
    assert index.scan(store).ingested_segments == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"edges": [0, 0, 2]},
        {"edges": [0, 1]},
        {"counts": [-1, 7]},
        {"counts": [True, 5]},
        {"n": 7},
        {"under": -1},
        {"std": -1},
        {"min": "nan"},
        {"x": {"bad key": 1}},
        {"key": "bad key"},
        {"step": True},
        {"edges": list(range(514)), "counts": [0] * 513, "n": 0},
    ],
)
def test_strict_histogram_validation(indexed: tuple[LocalTrackingStore, TrackingIndex], updates: dict[str, Any]) -> None:
    store, index = indexed
    put(store, "hist", [hist(**updates)])
    assert index.scan(store).errors == 1
    assert index.histograms("org", "project", "run", "advantages")["columns"] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"ret": True},
        {"ret": "NaN"},
        {"length": 1.5},
        {"success": 1},
        {"kind": "other"},
        {"actor": "a" * 65},
        {"env": "e" * 129},
        {"checkpoint": "c" * 257},
        {"seed": False},
        {"policy_version": -1},
        {"learner_version": -1},
        {"extra": {str(i): 0 for i in range(33)}},
        {"extra": {"bad key": 0}},
        {"x": {"steps": "inf"}},
    ],
)
def test_strict_episode_validation(indexed: tuple[LocalTrackingStore, TrackingIndex], updates: dict[str, Any]) -> None:
    store, index = indexed
    put(store, "episodes", [episode(**updates)])
    assert index.scan(store).errors == 1
    assert index.episodes("org", "project", "run")["total"] == 0


def test_histogram_attempt_rank_order_rebin_and_column_selection(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(store, "hist", [hist(0), hist(1), hist(2)], rank=2)
    put(store, "hist", [hist(1, counts=[4, 0], n=4), hist(1, counts=[0, 8], n=8)], attempt=2, rank=2)
    put(store, "hist", [hist(1, edges=[-2, 0, 2], counts=[4, 8], n=12)], attempt=2, rank=2, seq=1)
    put(store, "hist", [hist(1, counts=[90, 0], n=90)], rank=3)
    index.scan(store)
    result = index.histograms("org", "project", "run", "advantages", bins=4)
    assert result["rank"] == 2 and result["edges"] == [-2, -1, 0, 1, 2]
    assert [c["attempt"] for c in result["columns"]] == [1, 2, 1]
    assert result["columns"][1]["counts"] == [2, 2, 4, 4]
    for column in result["columns"]:
        assert sum(column["counts"]) == pytest.approx(column["n"])
    selected = index.histograms("org", "project", "run", "advantages", points=2)
    assert [c["step"] for c in selected["columns"]] == [0, 2]
    assert selected["total"] == 3 and selected["downsampled"]
    assert selected["edges"][0] == 0  # Grid uses selected columns, not the discarded wider one.
    assert index.histograms("org", "project", "run", "advantages", points=1)["columns"][0]["step"] == 2
    assert index.histograms("org", "project", "run", "advantages", rank=3)["columns"][0]["n"] == 90
    unknown = index.histograms("org", "project", "run", "unknown")
    assert unknown["columns"] == [] and unknown["edges"] == [] and unknown["total"] == 0


def test_histogram_axes_and_non_aligned_mass(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "hist", [hist(0, edges=[0, 0.3, 2], counts=[3, 17], n=20, x={"frames": 9}), hist(1)])
    index.scan(store)
    result = index.histograms("org", "project", "run", "advantages", x="frames", bins=7)
    assert result["total"] == 1 and result["columns"][0]["x"] == 9
    assert sum(result["columns"][0]["counts"]) == pytest.approx(20)
    assert index.histograms("org", "project", "run", "advantages", x="time")["columns"][0]["x"] == 100


def test_episode_replay_filters_null_steps_and_rebuild(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "episodes", [episode(100), episode(200), episode(300), episode(None, actor="null-step")])
    # Replay frontier spans ranks and kinds, not just the selected query subset.
    put(store, "episodes", [episode(200, kind="eval", env="new")], attempt=2, rank=3)
    put(store, "episodes", [episode(None, actor="third")], attempt=3)
    index.scan(store)
    listed = index.episodes("org", "project", "run")
    assert listed["total"] == 4 and not any(e["replayed"] for e in listed["episodes"])
    all_episodes = index.episodes("org", "project", "run", include_replayed=True)
    assert all_episodes["total"] == 6
    assert [e["step"] for e in all_episodes["episodes"] if e["replayed"]] == [300, 200]
    assert index.episodes("org", "project", "run", kind="eval", env="new")["total"] == 1
    assert index.episodes("org", "project", "run", actor="third")["total"] == 1
    assert len(index.episodes("org", "project", "run", limit=1, offset=1)["episodes"]) == 1
    assert index.episode_curve("org", "project", "run")["replayed_excluded"] == 2
    assert index.episode_breakdown("org", "project", "run")["replayed_excluded"] == 2
    assert index.run_detail("org", "project", "run")["episodes"] == {"train": 3, "eval": 1}
    before = snapshot(index)
    index.rebuild(store)
    assert snapshot(index) == before


def test_curve_contiguous_bucket_percentiles_axes_and_nonfinite(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(
        store,
        "episodes",
        [
            episode(i, value, t=100 + i, success=success, length=2 * i, x={"frames": 10 * i})
            for i, (value, success) in enumerate(
                [(0, True), (10, False), (20, None), (30, True), ("nan", True), ("inf", None)]
            )
        ],
    )
    index.scan(store)
    curve = index.episode_curve("org", "project", "run", points=2)
    assert curve["total"] == 6 and curve["nonfinite"] == 2
    first, second = curve["buckets"]
    assert first == {
        "x": 1,
        "x_min": 0,
        "x_max": 2,
        "count": 3,
        "mean": 10,
        "min": 0,
        "max": 20,
        "p5": 1,
        "p25": 5,
        "p50": 10,
        "p75": 15,
        "p95": 19,
        "success_rate": 0.5,
    }
    assert second["mean"] == 30 and second["count"] == 3 and second["success_rate"] == 1
    lengths = index.episode_curve("org", "project", "run", metric="length", points=1, x="frames")
    assert lengths["buckets"][0]["mean"] == 5 and lengths["buckets"][0]["x"] == 25
    assert index.episode_curve("org", "project", "run", x="time")["buckets"][0]["x"] == 100
    assert index.episode_curve("org", "project", "run", x="missing")["total"] == 0
    all_bad = index.episode_curve("org", "project", "run", points=6)["buckets"][-1]
    assert all_bad["mean"] is None and all_bad["p50"] is None


def test_breakdown_natural_actors_window_env_and_lag(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(
        store,
        "episodes",
        [
            episode(0, 100, actor="actor 2", env="A", t=100, policy_version=2, learner_version=2),
            episode(1, 4, actor="actor 2", env="A", t=101, success=True, policy_version=2, learner_version=17),
            episode(2, 8, actor="actor 2", env="A", t=102, success=False, policy_version=2, learner_version=18),
            episode(3, 6, actor="actor 10", env="B", t=103, policy_version=2, learner_version=1),
            episode(4, "-inf", t=104),
        ],
    )
    index.scan(store)
    result = index.episode_breakdown("org", "project", "run", last=2)
    assert [a["actor"] for a in result["actors"]] == ["actor 2", "actor 10", None]
    first = result["actors"][0]
    assert first == {
        "actor": "actor 2",
        "count": 3,
        "window": 2,
        "mean_return": 6,
        "p50_return": 6,
        "last_return": 8,
        "mean_length": 10,
        "success_rate": 0.5,
        "mean_lag": 15.5,
        "max_lag": 16,
        "last_t": 102,
    }
    assert result["envs"][0] == {"env": "A", "count": 3, "mean_return": pytest.approx(112 / 3), "success_rate": 0.5}
    lag = result["lag"]
    assert lag["count"] == 4 and lag["counts"][0] == 1 and lag["counts"][15] == 1
    assert sum(lag["counts"]) == 2 and lag["over"] == 1 and lag["negative"] == 1
    assert lag["p50"] == 7.5 and lag["p95"] == pytest.approx(15.85) and lag["max"] == 16


def test_evals_group_precedence_population_std_and_single_best(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(
        store,
        "episodes",
        [
            episode(10, 8, kind="eval", checkpoint="ckpt", policy_version=1, t=100),
            episode(11, 12, kind="eval", checkpoint="ckpt", policy_version=2, t=101),
            episode(20, 10, kind="eval", policy_version=3, t=90),
            episode(21, 10, kind="eval", policy_version=3, t=91),
            episode(30, 5, kind="eval", t=80),
        ],
    )
    index.scan(store)
    evals = index.episode_breakdown("org", "project", "run")["evals"]
    assert [v["count"] for v in evals] == [2, 2, 1]
    assert evals[0]["checkpoint"] == "ckpt" and evals[0]["std_return"] == 2
    assert sum(v["best"] for v in evals) == 1
    assert evals[1]["best"]  # Tied highest mean: earliest first_t, not first in display sort.


def test_all_nonfinite_evals_still_have_one_earliest_best(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "episodes", [episode(2, "nan", kind="eval", t=101), episode(3, "inf", kind="eval", t=100)])
    index.scan(store)
    result = index.episode_breakdown("org", "project", "run")
    assert [v["best"] for v in result["evals"]] == [False, True]
    assert all(v["mean_return"] is None for v in result["evals"])
    assert result["lag"]["p50"] is None


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("histograms", {"key": "advantages", "points": 1001}),
        ("histograms", {"key": "advantages", "bins": 0}),
        ("histograms", {"key": "advantages", "rank": 10000}),
        ("episodes", {"limit": 1001}),
        ("episodes", {"offset": -1}),
        ("episodes", {"kind": "bad"}),
        ("episode_curve", {"points": 2001}),
        ("episode_curve", {"metric": "bad"}),
        ("episode_breakdown", {"last": 5001}),
    ],
)
def test_index_query_validation(
    indexed: tuple[LocalTrackingStore, TrackingIndex], method: str, kwargs: dict[str, Any]
) -> None:
    _, index = indexed
    with pytest.raises(ValueError):
        getattr(index, method)("org", "project", "run", **kwargs)


def test_histogram_writer_may_omit_empty_axes(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    record = hist()
    del record["x"]  # The SDK omits this field when no custom axes were supplied.
    put(store, "hist", [record])
    assert index.scan(store).errors == 0
    assert index.histograms("org", "project", "run", "advantages")["total"] == 1


def test_histogram_even_column_selection_and_missing_winning_axis(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(store, "hist", [hist(i, x={"frames": i * 10}) for i in range(8)])
    put(store, "hist", [hist(2)], attempt=2)
    index.scan(store)
    result = index.histograms("org", "project", "run", "advantages", points=4)
    assert [column["step"] for column in result["columns"]] == [0, 2, 5, 7]
    named = index.histograms("org", "project", "run", "advantages", x="frames")
    assert [column["step"] for column in named["columns"]] == [0, 1, 3, 4, 5, 6, 7]


def test_episode_order_replay_and_uneven_buckets(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "episodes", [episode(100, t=100), episode(250, t=100), episode(400, t=100)])
    put(store, "episodes", [episode(300, t=100)], attempt=2)
    put(store, "episodes", [episode(200, t=100)], attempt=3, rank=2)
    index.scan(store)
    listed = index.episodes("org", "project", "run", include_replayed=True)
    assert [row["attempt"] for row in listed["episodes"]] == [3, 2, 1, 1, 1]
    assert sum(row["replayed"] for row in listed["episodes"]) == 3
    assert index.episode_breakdown("org", "project", "run")["replayed_excluded"] == 3
    # There are two visible episodes; asking for more buckets never creates empties.
    curve = index.episode_curve("org", "project", "run", points=200)
    assert [bucket["x"] for bucket in curve["buckets"]] == [100, 200]


def test_unknown_run_queries_return_none(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    _, index = indexed
    assert index.histograms("org", "project", "missing", "advantages") is None
    assert index.episodes("org", "project", "missing") is None
    assert index.episode_curve("org", "project", "missing") is None
    assert index.episode_breakdown("org", "project", "missing") is None
