# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Phase-three queries over real immutable segments and content-addressed blobs."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pytest

from eyvan.track._format import SegmentName, blob_key, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker.index import TrackingIndex


def put(store: LocalTrackingStore, stream: str, records: list[dict[str, Any]], **identity: Any) -> str:
    """Write one segment with a stable identity in the test scope."""

    name = SegmentName(
        identity.get("attempt", 1),
        identity.get("rank", 0),
        identity.get("writer", "bbbbbbbbbbbb"),
        identity.get("seq", 0),
    )
    key = f"org/project/{identity.get('run', 'run')}/{stream}/{name.filename}"
    store.put(key, encode_segment(records))
    return key


def blob(store: LocalTrackingStore, data: bytes = b"png bytes", *, run: str = "run") -> dict[str, Any]:
    """Store a real immutable blob and return its reference."""

    digest = hashlib.sha256(data).hexdigest()
    store.put(f"org/project/{blob_key(run, digest)}", data)
    return {"sha256": digest, "content_type": "image/png", "bytes": len(data)}


def rollout(step: int | None = 1, **updates: Any) -> dict[str, Any]:
    """Return a complete sampled rollout record."""

    return {
        "t": 100.0 + (step or 0),
        "step": step,
        "episode_id": "episode-1",
        "actor": "actor 2",
        "env": "Pong",
        "kind": "train",
        "policy_version": 3,
        "length": 2,
        "tracks": {"reward": [1, 3], "value": [2, "nan"]},
        "actions": [0, 1],
        "action_names": ["left", "right"],
        "ret": 4,
        "video": None,
        "fps": None,
        "x": {},
        **updates,
    }


def completion(step: int | None = 1, rewards: tuple[Any, ...] = (0, 2), **updates: Any) -> dict[str, Any]:
    """Return a complete completion group, including component rewards."""

    return {
        "t": 100.0 + (step or 0),
        "step": step,
        "group": "group-1",
        "prompt": "What is 1 + 1?",
        "answer": "2",
        "policy_version": 3,
        "responses": [
            {
                "text": str(i),
                "rewards": {"correct": reward, "format": 1},
                "reward": reward,
                "advantage": i * 2 - 1,
                "tokens": 10 * (i + 1),
                "kl": 0.1,
                "truncated": bool(i),
            }
            for i, reward in enumerate(rewards)
        ],
        **updates,
    }


def table(step: int | None = 1, **updates: Any) -> dict[str, Any]:
    """Return a complete table record."""

    return {"t": 100.0 + (step or 0), "step": step, "key": "eval/cases", "columns": ["case"], "rows": [["A"]], **updates}


def media(step: int | None = 1, **updates: Any) -> dict[str, Any]:
    """Return a complete media record with an intentionally missing blob."""

    return {
        "t": 100.0 + (step or 0),
        "step": step,
        "key": "eval/image",
        "blob": {"sha256": "0" * 64, "content_type": "image/png", "bytes": 3},
        "caption": "evaluation",
        **updates,
    }


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[LocalTrackingStore, TrackingIndex]:
    return LocalTrackingStore(tmp_path / "store"), TrackingIndex(tmp_path / "index.sqlite3")


def snapshot(index: TrackingIndex, store: LocalTrackingStore) -> tuple[Any, ...]:
    """Capture all phase-three outputs to check rebuild identity."""

    rollouts = index.rollouts("org", "project", "run", store=store)
    completions = index.completions("org", "project", "run")
    return (
        index.list_runs("org", "project"),
        index.run_detail("org", "project", "run"),
        rollouts,
        index.rollout_detail("org", "project", "run", rollouts["rollouts"][0]["id"], store=store),
        completions,
        index.completion_detail("org", "project", "run", completions["groups"][0]["id"]),
        index.completion_stats("org", "project", "run"),
        index.tables("org", "project", "run", store=store),
        index.tables("org", "project", "run", "eval/cases", store=store),
        index.media("org", "project", "run", store=store),
    )


def test_ingest_rebuild_ids_detail_times_and_blobs_are_not_segments(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    ref = blob(store)
    now = time.time()
    put(store, "rollouts", [rollout(4, t=90, video=ref)])
    put(store, "completions", [completion(5, t=91)])
    put(store, "tables", [table(6, t=92, rows=[[ref]])])
    put(store, "media", [media(9, t=now, blob=ref)], attempt=2, rank=3)
    # Even a segment-looking name under blobs is not a stream segment.
    store.put(f"org/project/run/blobs/{SegmentName(3, 0, 'bbbbbbbbbbbb', 0).filename}", b"not gzip")
    scanned = index.scan(store)
    assert (scanned.ingested_segments, scanned.indexed_segments, scanned.errors) == (4, 4, 0)
    detail = index.run_detail("org", "project", "run")
    assert detail["status"] == "running" and detail["updated_at"] == now
    assert detail["started_at"] == 90 and detail["last_step"] == 9
    assert detail["rollouts"] == 1 and detail["completion_groups"] == 1
    assert detail["table_keys"] == ["eval/cases"] and detail["media_keys"] == ["eval/image"]
    assert detail["attempt_list"] == [
        {"attempt": 1, "started_at": 90, "first_step": 4, "last_step": 6, "status": None},
        {"attempt": 2, "started_at": now, "first_step": 9, "last_step": 9, "status": None},
    ]
    before = snapshot(index, store)
    assert before[2]["rollouts"][0]["id"] == "a1-r0-bbbbbbbbbbbb-0-0"
    assert before[3]["tracks"]["value"] == [2, "nan"]
    assert before[3]["video"] == {**ref, "available": True}
    assert before[8]["rows"] == [[{**ref, "available": True}]]
    assert index.scan(store).ingested_segments == 0
    assert index.rebuild(store).ingested_segments == 4
    assert snapshot(index, store) == before
    fresh = TrackingIndex(index.db_path.with_name("fresh.sqlite3"))
    fresh.scan(store)
    assert snapshot(fresh, store) == before


@pytest.mark.parametrize(
    "stream,record", [("rollouts", rollout()), ("completions", completion()), ("tables", table()), ("media", media())]
)
def test_corrupt_segments_are_atomic_isolated_and_rebuildable(
    indexed: tuple[LocalTrackingStore, TrackingIndex], stream: str, record: dict[str, Any]
) -> None:
    store, index = indexed
    store.put(f"org/project/run/{stream}/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}", b"not gzip")
    put(store, stream, [record, {**record, "t": "bad"}], seq=1)
    put(store, stream, [record], seq=2)
    assert index.scan(store).errors == 2
    detail = index.run_detail("org", "project", "run")
    assert len(detail["errors"]) == 2
    if stream == "tables":
        assert index.tables("org", "project", "run")["keys"][0]["count"] == 1
    else:
        assert getattr(index, stream)("org", "project", "run")["total"] == 1
    assert index.scan(store).ingested_segments == 0
    assert index.rebuild(store).errors == 2
    assert index.run_detail("org", "project", "run") == detail


@pytest.mark.parametrize(
    "stream,record,updates",
    [
        ("rollouts", rollout(), {"length": 0}),
        ("rollouts", rollout(), {"length": 10001}),
        ("rollouts", rollout(), {"tracks": {"reward": [1]}}),
        ("rollouts", rollout(), {"tracks": {"bad name": [1, 2]}}),
        ("rollouts", rollout(), {"tracks": {str(i): [1, 2] for i in range(33)}}),
        ("rollouts", rollout(), {"tracks": {"reward": [True, 2]}}),
        ("rollouts", rollout(), {"actions": [True, 1]}),
        ("rollouts", rollout(), {"actions": [0]}),
        ("rollouts", rollout(), {"action_names": ["a"] * 257}),
        ("rollouts", rollout(), {"episode_id": "a" * 129}),
        ("rollouts", rollout(), {"actor": "a" * 65}),
        ("rollouts", rollout(), {"env": "e" * 129}),
        ("rollouts", rollout(), {"kind": "invalid"}),
        ("rollouts", rollout(), {"fps": 0}),
        ("rollouts", rollout(), {"ret": "nan"}),
        ("rollouts", rollout(), {"video": {"sha256": "a" * 64, "content_type": "text/html", "bytes": 3}}),
        ("completions", completion(), {"responses": []}),
        ("completions", completion(), {"responses": completion()["responses"] * 33}),
        ("completions", completion(), {"prompt": "p" * 32769}),
        ("completions", completion(), {"group": "g" * 129}),
        ("completions", completion(), {"responses": [{**completion()["responses"][0], "text": "r" * 32769}]}),
        ("completions", completion(), {"responses": [{**completion()["responses"][0], "tokens": -1}]}),
        ("completions", completion(), {"responses": [{**completion()["responses"][0], "truncated": 1}]}),
        ("completions", completion(), {"responses": [{**completion()["responses"][0], "advantage": True}]}),
        ("completions", completion(), {"responses": [{**completion()["responses"][0], "rewards": {"bad key": 1}}]}),
        ("tables", table(), {"columns": ["a", "a"], "rows": [[1, 2]]}),
        ("tables", table(), {"columns": []}),
        ("tables", table(), {"columns": [str(i) for i in range(65)]}),
        ("tables", table(), {"rows": [[1, 2]]}),
        ("tables", table(), {"rows": [["a" * 4097]]}),
        ("tables", table(), {"rows": [[1]] * 1001}),
        ("tables", table(), {"rows": [[{"not": "blob"}]]}),
        ("media", media(), {"caption": "c" * 1025}),
        ("media", media(), {"blob": {"sha256": "bad", "content_type": "image/png", "bytes": 3}}),
        ("media", media(), {"blob": {"sha256": "a" * 64, "content_type": "image/png", "bytes": -1}}),
        ("media", media(), {"blob": {"sha256": "a" * 64, "content_type": "image/png", "bytes": True}}),
        ("media", media(), {"blob": {"sha256": "a" * 64, "content_type": "image/png", "bytes": 64 * 1024**2 + 1}}),
    ],
)
def test_strict_phase_three_records(
    indexed: tuple[LocalTrackingStore, TrackingIndex], stream: str, record: dict[str, Any], updates: dict[str, Any]
) -> None:
    store, index = indexed
    put(store, stream, [{**record, **updates}])
    assert index.scan(store).errors == 1


@pytest.mark.parametrize(
    "stream,record", [("rollouts", rollout()), ("completions", completion()), ("tables", table()), ("media", media())]
)
@pytest.mark.parametrize("step", [-1, True, 2**63])
def test_all_phase_three_steps_are_nonnegative_sqlite_integers(
    indexed: tuple[LocalTrackingStore, TrackingIndex], stream: str, record: dict[str, Any], step: Any
) -> None:
    store, index = indexed
    put(store, stream, [{**record, "step": step}])
    assert index.scan(store).errors == 1


def test_shared_frontier_replays_across_streams_but_not_null_steps(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    # Full episode shape, unchanged from phase two.
    episode = {
        "t": 110,
        "step": 10,
        "ret": 1,
        "length": 1,
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
    }
    put(store, "episodes", [episode])
    put(store, "rollouts", [rollout(4), rollout(10), rollout(None)])
    put(store, "completions", [completion(10)])
    put(store, "tables", [table(10)])
    put(store, "media", [media(10)])
    put(store, "completions", [completion(8)], attempt=2, rank=3)
    put(store, "media", [media(None)], attempt=3)
    index.scan(store)
    assert index.episodes("org", "project", "run")["total"] == 0
    assert index.episode_curve("org", "project", "run")["replayed_excluded"] == 1
    assert index.rollouts("org", "project", "run")["total"] == 2
    assert index.rollouts("org", "project", "run", include_replayed=True)["total"] == 3
    assert index.completions("org", "project", "run")["total"] == 1
    assert index.tables("org", "project", "run") == {"keys": []}
    assert index.media("org", "project", "run")["total"] == 1
    assert index.run_detail("org", "project", "run")["table_keys"] == []
    # A later episode also replays records in a different phase-three stream.
    put(store, "episodes", [{**episode, "step": 3}], attempt=4)
    index.scan(store)
    assert index.rollouts("org", "project", "run")["total"] == 1
    assert index.completions("org", "project", "run")["total"] == 0
    assert index.rollouts("org", "project", "run")["rollouts"][0]["step"] is None


def test_rollout_filters_order_pagination_and_detail(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "rollouts", [rollout(1, t=100), rollout(2, t=100), rollout(3, t=101, kind="eval", actor=None)])
    index.scan(store)
    listed = index.rollouts("org", "project", "run")
    assert [r["step"] for r in listed["rollouts"]] == [3, 2, 1]
    assert "tracks" not in listed["rollouts"][0] and "actions" not in listed["rollouts"][0]
    assert listed["rollouts"][0]["track_names"] == ["reward", "value"]
    assert index.rollouts("org", "project", "run", kind="eval")["total"] == 1
    assert index.rollouts("org", "project", "run", actor="actor 2")["total"] == 2
    assert index.rollouts("org", "project", "run", offset=1, limit=1)["rollouts"][0]["step"] == 2
    assert index.rollout_detail("org", "project", "run", "a1-r0-bbbbbbbbbbbb-0-100") is None


def test_completion_summary_sort_finite_only_and_stats(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    put(store, "completions", [completion(1), completion(2, (3, 3)), completion(3, ("nan", "inf"))])
    index.scan(store)
    listed = index.completions("org", "project", "run")
    assert listed["steps"] == [1, 2, 3]
    assert [g["step"] for g in listed["groups"]] == [3, 2, 1]
    group = listed["groups"][-1]
    assert {
        k: group[k] for k in ("k", "mean_reward", "std_reward", "pass_rate", "zero_variance", "mean_tokens", "truncated")
    } == {
        "k": 2,
        "mean_reward": 1,
        "std_reward": 1,
        "pass_rate": 0.5,
        "zero_variance": False,
        "mean_tokens": 15,
        "truncated": 1,
    }
    assert listed["groups"][0]["mean_reward"] is None and listed["groups"][0]["pass_rate"] is None
    assert listed["groups"][0]["zero_variance"] is False  # No finite evidence of zero variance.
    assert index.completions("org", "project", "run", sort="zero_variance")["groups"][0]["step"] == 2
    assert index.completions("org", "project", "run", sort="reward")["groups"][0]["step"] == 2
    assert index.completions("org", "project", "run", step=1)["total"] == 1
    detail = index.completion_detail("org", "project", "run", group["id"])
    assert detail["responses"][1]["reward"] == 2 and detail["std_reward"] == 1
    stats = index.completion_stats("org", "project", "run")["steps"]
    assert stats[0] == {
        "step": 1,
        "groups": 1,
        "mean_reward": 1,
        "zero_variance_frac": 0,
        "mean_tokens": 15,
        "truncated_frac": 0.5,
        "components": {"correct": 1, "format": 1},
    }
    assert stats[1]["zero_variance_frac"] == 1
    assert stats[2]["components"] == {"correct": None, "format": 1}
    # Downsampling buckets distinct steps, rather than dropping groups.
    bucket = index.completion_stats("org", "project", "run", points=1)["steps"][0]
    assert bucket["step"] == 1 and bucket["groups"] == 3
    assert bucket["mean_reward"] == 2 and bucket["zero_variance_frac"] == pytest.approx(1 / 3)


def test_completion_detail_keeps_full_prompt_and_statistics_weight_responses(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(store, "completions", [completion(1, (2,), prompt="p" * 500), completion(1, (0, 0, 0), group="other")])
    index.scan(store)
    listed = index.completions("org", "project", "run")["groups"]
    group = next(g for g in listed if g["group"] == "group-1")
    assert len(group["prompt"]) == 300
    assert len(index.completion_detail("org", "project", "run", group["id"])["prompt"]) == 500
    stats = index.completion_stats("org", "project", "run")["steps"][0]
    assert stats["groups"] == 2 and stats["mean_reward"] == 0.5


def test_table_latest_by_step_keys_and_media_availability(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    ref = blob(store)
    put(store, "tables", [table(None), table(2, rows=[["old"]]), table(5, rows=[[ref]])])
    put(store, "tables", [table(5, rows=[["last"]]), table(1, key="second")], seq=1)
    put(store, "media", [media(1, blob=ref), media(2), media(3, key="other", blob=ref)])
    index.scan(store)
    assert index.tables("org", "project", "run")["keys"] == [
        {"key": "eval/cases", "count": 4, "last_step": 5},
        {"key": "second", "count": 1, "last_step": 1},
    ]
    latest = index.tables("org", "project", "run", "eval/cases", store=store)
    assert latest["rows"] == [["last"]] and latest["steps"] == [2, 5]
    assert index.tables("org", "project", "run", "eval/cases", step=2)["rows"] == [["old"]]
    assert index.tables("org", "project", "run", "absent") is None
    assert index.tables("org", "project", "run", "eval/cases", step=3) is None
    listed = index.media("org", "project", "run", store=store)
    assert [r["step"] for r in listed["media"]] == [3, 2, 1]
    assert [r["blob"]["available"] for r in listed["media"]] == [True, False, True]
    assert (
        index.media("org", "project", "run", key="eval/image", limit=1, offset=1, store=store)["media"][0]["step"] == 1
    )


def test_availability_lists_once_and_never_downloads_blobs(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    ref = blob(store)
    put(store, "media", [media(1, blob=ref), media(2, blob=ref)])
    index.scan(store)

    class ListingStore:
        """Reject body reads during availability checks."""

        def __init__(self) -> None:
            self.prefixes: list[str] = []

        @property
        def root(self) -> str:
            return store.root

        def list(self, prefix: str = "") -> list[str]:
            self.prefixes.append(prefix)
            return store.list(prefix)

        def get(self, key: str) -> bytes:
            raise AssertionError("availability must not fetch blob bytes")

        def put(self, key: str, data: bytes) -> None:
            raise AssertionError("read-only query")

    listing = ListingStore()
    assert all(row["blob"]["available"] for row in index.media("org", "project", "run", store=listing)["media"])
    assert listing.prefixes == ["org/project/run/blobs/"]


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("rollouts", {"kind": "bad"}),
        ("rollouts", {"actor": "a" * 65}),
        ("rollouts", {"limit": 501}),
        ("rollouts", {"offset": -1}),
        ("rollouts", {"include_replayed": 1}),
        ("rollout_detail", {"rollout_id": "bad"}),
        ("completions", {"sort": "bad"}),
        ("completions", {"step": -1}),
        ("completions", {"limit": 501}),
        ("completions", {"include_replayed": 1}),
        ("completion_detail", {"completion_id": "bad"}),
        ("completion_stats", {"points": 2001}),
        ("tables", {"key": "bad key"}),
        ("tables", {"step": -1}),
        ("media", {"limit": 1001}),
        ("media", {"key": "bad key"}),
        ("media", {"offset": -1}),
    ],
)
def test_query_validation(
    indexed: tuple[LocalTrackingStore, TrackingIndex], method: str, kwargs: dict[str, Any]
) -> None:
    _, index = indexed
    with pytest.raises(ValueError):
        getattr(index, method)("org", "project", "run", **kwargs)


@pytest.mark.parametrize("method", ["rollouts", "completions", "completion_stats", "tables", "media"])
def test_unknown_run(indexed: tuple[LocalTrackingStore, TrackingIndex], method: str) -> None:
    _, index = indexed
    assert getattr(index, method)("org", "project", "missing") is None


@pytest.mark.parametrize(
    "stream,record", [("rollouts", rollout()), ("completions", completion()), ("tables", table()), ("media", media())]
)
def test_nullable_steps_unknown_fields_and_missing_required_fields(
    indexed: tuple[LocalTrackingStore, TrackingIndex], stream: str, record: dict[str, Any]
) -> None:
    store, index = indexed
    valid = {**record, "step": None, "future_field": {"arbitrary": True}}
    invalid = dict(valid)
    del invalid["step"]
    put(store, stream, [valid])
    put(store, stream, [invalid], seq=1)
    assert index.scan(store).errors == 1
    detail = index.run_detail("org", "project", "run")
    assert detail["last_step"] is None and detail["updated_at"] == record["t"]
    assert len(detail["errors"]) == 1


def test_nullable_tracks_table_cells_and_finite_extreme_statistics(
    indexed: tuple[LocalTrackingStore, TrackingIndex],
) -> None:
    store, index = indexed
    put(store, "rollouts", [rollout(tracks={"reward": [None, "-inf"]}, ret=None, actions=None, action_names=None)])
    put(store, "tables", [table(columns=["n", "s", "b", "null"], rows=[[1, "nan", True, None]])])
    responses = completion(rewards=(1e308, -1e308))["responses"]
    responses[0].update(tokens=None, truncated=None, kl=None)
    put(store, "completions", [completion(responses=responses)])
    assert index.scan(store).errors == 0
    assert index.tables("org", "project", "run", "eval/cases")["rows"] == [[1, "nan", True, None]]
    summary = index.completions("org", "project", "run")["groups"][0]
    assert summary["mean_reward"] == 0 and summary["std_reward"] == pytest.approx(1e308)
    assert summary["mean_tokens"] == 20 and summary["truncated"] == 1
    detail = index.rollout_detail("org", "project", "run", "a1-r0-bbbbbbbbbbbb-0-0")
    assert detail["tracks"] == {"reward": [None, "-inf"]} and detail["actions"] is None


def test_large_integer_table_cells_remain_exact(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    value = 2**60 + 1
    put(store, "tables", [table(rows=[[value]])])
    assert index.scan(store).errors == 0
    assert index.tables("org", "project", "run", "eval/cases")["rows"] == [[value]]


def test_unrepresentable_token_means_do_not_break_queries(indexed: tuple[LocalTrackingStore, TrackingIndex]) -> None:
    store, index = indexed
    responses = completion()["responses"]
    responses[0]["tokens"] = 10**400
    put(store, "completions", [completion(responses=responses)])
    assert index.scan(store).errors == 0
    assert index.completions("org", "project", "run")["groups"][0]["mean_tokens"] is None
    assert index.completion_stats("org", "project", "run")["steps"][0]["mean_tokens"] is None
