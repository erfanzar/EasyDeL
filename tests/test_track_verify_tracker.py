# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Executed regressions for the independent tracker review and core follow-up."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from eyvan.track._format import SegmentName, encode_segment
from eyvan.track._store import LocalTrackingStore
from eyvan.track.tracker import phase3
from eyvan.track.tracker.index import TrackingIndex
from eyvan.track.tracker.phase3 import BlobIntegrityError
from tests.test_track_index_phase2 import episode, hist
from tests.test_track_index_phase3 import completion, media, rollout, table

_SCOPE = ("org", "project", "run")


def put(store: LocalTrackingStore, stream: str, records: list[dict[str, Any]], seq: int = 0) -> str:
    """Append one real segment in the shared disposable test scope."""

    key = f"org/project/run/{stream}/{SegmentName(1, 0, 'bbbbbbbbbbbb', seq).filename}"
    store.put(key, encode_segment(records))
    return key


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[LocalTrackingStore, TrackingIndex]:
    return LocalTrackingStore(tmp_path / "store"), TrackingIndex(tmp_path / "index.sqlite3")


def test_08_blob_prefers_bounded_read_and_rechecks_declared_size(indexed: Any, monkeypatch: Any) -> None:
    store, index = indexed
    data = b"image"
    digest = hashlib.sha256(data).hexdigest()
    ref = {"sha256": digest, "content_type": "image/png", "bytes": len(data)}
    put(store, "media", [media(blob=ref)])
    index.scan(store)

    class BoundedStore:
        reads = 0

        def get(self, key: str) -> bytes:
            pytest.fail("blob read must use the available bounded operation")

        def get_bounded(self, key: str, max_bytes: int) -> bytes:
            assert max_bytes <= phase3.MAX_BLOB_BYTES + 1
            self.reads += 1
            return data

    bounded = BoundedStore()
    assert index.blob(*_SCOPE, digest, store=bounded) == (data, "image/png")
    assert bounded.reads == 1
    monkeypatch.setattr(phase3, "MAX_BLOB_BYTES", len(data) - 1)
    with pytest.raises(BlobIntegrityError):
        index.blob(*_SCOPE, digest, store=bounded)
    assert bounded.reads == 1, "oversized declared length must be rejected before store I/O"


def test_24_lists_and_detail_counts_do_not_decode_off_page_payloads(indexed: Any, monkeypatch: Any) -> None:
    store, index = indexed
    put(store, "completions", [completion(i, prompt="heavy-prompt", answer="heavy-answer") for i in range(8)])
    put(store, "rollouts", [rollout(i, action_names=["heavy-action"]) for i in range(8)])
    put(store, "tables", [table(i, rows=[["heavy-cell"]]) for i in range(8)])
    put(store, "media", [media(i, caption="heavy-caption") for i in range(8)])
    index.scan(store)
    loads = json.loads
    decoded: list[str] = []

    def measured(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str) and "heavy-" in value:
            decoded.append(value)
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", measured)
    assert index.run_detail(*_SCOPE)["completion_groups"] == 8
    assert decoded == [], "run counts must not decode rich payloads"
    for query in (
        lambda: index.completions(*_SCOPE, limit=1),
        lambda: index.completions(*_SCOPE, limit=1, sort="reward"),
        lambda: index.completions(*_SCOPE, limit=1, sort="zero_variance"),
        lambda: index.rollouts(*_SCOPE, limit=1),
        lambda: index.media(*_SCOPE, limit=1),
        lambda: index.tables(*_SCOPE, key="eval/cases"),
        lambda: index.completion_detail(*_SCOPE, "a1-r0-bbbbbbbbbbbb-0-7"),
        lambda: index.rollout_detail(*_SCOPE, "a1-r0-bbbbbbbbbbbb-0-7"),
    ):
        decoded.clear()
        assert query() is not None
        assert len(decoded) <= 1, "only the selected full record/page may be decoded"


def test_24_completion_stats_do_not_decode_response_text(indexed: Any, monkeypatch: Any) -> None:
    store, index = indexed
    record = completion()
    record["responses"][0]["text"] = "heavy-response-text"
    put(store, "completions", [record])
    index.scan(store)
    loads = json.loads
    decoded: list[str] = []

    def measured(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str) and "heavy-response-text" in value:
            decoded.append(value)
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", measured)
    assert index.completion_stats(*_SCOPE)["steps"][0]["mean_reward"] == 1
    assert decoded == []


@pytest.mark.parametrize("operation", ["blob", "media", "rollout", "table", "scan", "rebuild"])
def test_25_store_io_does_not_block_unrelated_queries(indexed: Any, operation: str) -> None:
    store, index = indexed
    data = b"image"
    digest = hashlib.sha256(data).hexdigest()
    ref = {"sha256": digest, "content_type": "image/png", "bytes": len(data)}
    store.put(f"org/project/run/blobs/{digest}", data)
    put(store, "media", [media(blob=ref)])
    put(store, "rollouts", [rollout(video=ref)])
    put(store, "tables", [table(rows=[[ref]])])
    # A different organization's query must remain independent of blocked I/O.
    store.put(
        f"other/project/run/meta/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}",
        encode_segment([{"op": "start", "t": 1, "name": "other"}]),
    )
    index.scan(store)
    entered, release = threading.Event(), threading.Event()

    class PausedStore:
        root = store.root

        def get(self, key: str) -> bytes:
            if operation == "blob":
                entered.set()
                assert release.wait(5)
            return store.get(key)

        def list(self, prefix: str = "") -> list[str]:
            entered.set()
            assert release.wait(5)
            return store.list(prefix)

    paused = PausedStore()
    queries = {
        "blob": lambda: index.blob(*_SCOPE, digest, store=paused),
        "media": lambda: index.media(*_SCOPE, store=paused),
        "rollout": lambda: index.rollouts(*_SCOPE, store=paused),
        "table": lambda: index.tables(*_SCOPE, key="eval/cases", store=paused),
        "scan": lambda: index.scan(paused),
        "rebuild": lambda: index.rebuild(paused),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(queries[operation])
        try:
            assert entered.wait(2)
            fast = pool.submit(lambda: (index.scan_state(), index.run_detail("other", "project", "run")))
            health, other = fast.result(timeout=1)
            assert health["indexed_segments"] == 4
            assert other["name"] == "other"
        finally:
            release.set()
        slow.result(timeout=3)


def test_26_sdk_table_headings_round_trip(indexed: Any) -> None:
    from eyvan.track._rich import table_body

    store, index = indexed
    columns, rows, _ = table_body([{"Sample text": "hello", "Réponse?!": "oui"}], None)
    put(store, "tables", [table(columns=columns, rows=rows), table(key="other")])
    assert index.scan(store).errors == 0
    assert index.tables(*_SCOPE, key="eval/cases")["columns"] == columns
    assert index.run_detail(*_SCOPE)["table_keys"] == ["eval/cases", "other"]


def test_28_completion_extreme_finite_rewards(indexed: Any) -> None:
    store, index = indexed
    maximum = sys.float_info.max
    put(store, "completions", [completion(rewards=(maximum,) * 3)])
    assert index.scan(store).errors == 0
    group = index.completions(*_SCOPE)["groups"][0]
    assert group["mean_reward"] == maximum and group["std_reward"] == 0
    assert index.completion_detail(*_SCOPE, group["id"])["mean_reward"] == maximum
    assert index.completion_stats(*_SCOPE)["steps"][0]["mean_reward"] == maximum


def test_29_large_table_integers_are_exact_after_rebuild(indexed: Any) -> None:
    store, index = indexed
    values = [[10**400], [-(10**400)]]
    put(store, "tables", [table(rows=values), table(key="other")])
    assert index.scan(store).errors == 0
    assert index.tables(*_SCOPE, key="eval/cases")["rows"] == values
    assert index.rebuild(store).errors == 0
    assert index.tables(*_SCOPE, key="eval/cases")["rows"] == values


@pytest.mark.parametrize("field", ["prompt", "response", "table", "actor", "episode", "metadata"])
def test_30_surrogates_are_isolated_as_corrupt_segments(indexed: Any, field: str) -> None:
    store, index = indexed
    surrogate = chr(0xD800)
    record, stream = completion(prompt=surrogate), "completions"
    if field == "response":
        record = completion()
        record["responses"][0]["text"] = surrogate
    elif field == "table":
        record, stream = table(rows=[[surrogate]]), "tables"
    elif field == "actor":
        record, stream = rollout(actor=surrogate), "rollouts"
    elif field == "episode":
        record, stream = episode(env=surrogate), "episodes"
    elif field == "metadata":
        record, stream = {"t": 1, "op": "config", "config": {"nested": [surrogate]}}, "meta"
    key = f"org/project/run/{stream}/{SegmentName(1, 0, 'bbbbbbbbbbbb', 0).filename}"
    store.put(key, gzip.compress((json.dumps(record, ensure_ascii=True) + "\n").encode("ascii")))
    put(store, "completions", [completion()], seq=1)
    assert index.scan(store).errors == 1
    assert len(index.run_detail(*_SCOPE)["errors"]) == 1
    assert index.completions(*_SCOPE)["total"] == 1
    assert index.rebuild(store).errors == 1


@pytest.mark.parametrize("flags,expected", [([None, None], None), ([True, None], 1.0), ([False, None], 0.0)])
def test_31_truncation_fraction_uses_only_known_flags(indexed: Any, flags: Any, expected: Any) -> None:
    store, index = indexed
    record = completion()
    for response, flag in zip(record["responses"], flags, strict=True):
        response["truncated"] = flag
    put(store, "completions", [record])
    index.scan(store)
    assert index.completion_stats(*_SCOPE)["steps"][0]["truncated_frac"] == expected


def test_core_scalar_bucket_mean_at_float_max(indexed: Any) -> None:
    store, index = indexed
    maximum = sys.float_info.max
    put(store, "scalars", [{"t": 1, "step": i, "v": {"loss": maximum}} for i in range(3)])
    index.scan(store)
    assert index.series(*_SCOPE, ["loss"], points=1)["series"][0]["points"] == [[0, maximum, maximum, maximum]]


def test_core_episode_large_integer_aggregates_do_not_crash(indexed: Any) -> None:
    store, index = indexed
    put(store, "episodes", [episode(length=10**400, policy_version=0, learner_version=10**400)])
    assert index.scan(store).errors == 0
    assert index.episode_curve(*_SCOPE, metric="length")["buckets"][0]["mean"] is None
    lag = index.episode_breakdown(*_SCOPE)["lag"]
    assert lag["count"] == 1 and lag["max"] == 10**400 and lag["p50"] is None


def test_core_episode_pagination_decodes_only_selected_rows(indexed: Any, monkeypatch: Any) -> None:
    store, index = indexed
    put(store, "episodes", [episode(i, checkpoint="heavy-checkpoint") for i in range(8)])
    index.scan(store)
    loads = json.loads
    decoded: list[str] = []

    def measured(value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str) and "heavy-checkpoint" in value:
            decoded.append(value)
        return loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", measured)
    page = index.episodes(*_SCOPE, limit=1, offset=3)
    assert page["total"] == 8 and page["episodes"][0]["step"] == 4
    assert len(decoded) == 1


def test_core_scaled_mean_keeps_small_cancellation_residual(indexed: Any) -> None:
    store, index = indexed
    put(
        store,
        "scalars",
        [
            {"t": 1, "step": i, "v": {"loss": value}}
            for i, value in enumerate([sys.float_info.max, -sys.float_info.max, 3e-100])
        ],
    )
    index.scan(store)
    assert index.series(*_SCOPE, ["loss"], points=1)["series"][0]["points"][0][1] == pytest.approx(1e-100, abs=0)


def test_core_histogram_near_float_max_conserves_counts(indexed: Any) -> None:
    store, index = indexed
    maximum = sys.float_info.max
    put(store, "hist", [hist(edges=[math.nextafter(maximum, 0), maximum], counts=[6], n=6)])
    assert index.scan(store).errors == 0
    for bins in range(1, 65):
        result = index.histograms(*_SCOPE, "advantages", bins=bins)
        json.dumps(result, allow_nan=False)
        assert sum(result["columns"][0]["counts"]) == pytest.approx(6), bins


def test_pagination_keeps_oversized_offsets_and_missing_ids_safe(indexed: Any) -> None:
    store, index = indexed
    put(store, "completions", [completion()])
    put(store, "episodes", [episode()])
    index.scan(store)
    assert index.completions(*_SCOPE, offset=10**100)["groups"] == []
    assert index.episodes(*_SCOPE, offset=10**100)["episodes"] == []
    assert index.completion_detail(*_SCOPE, f"a1-r0-bbbbbbbbbbbb-0-{10**100}") is None


def test_core_finish_order_uses_sequence_before_line(indexed: Any) -> None:
    store, index = indexed
    put(store, "meta", [{"t": 1, "op": "finish", "status": "failed"}], seq=1)
    put(store, "meta", [{"t": 1, "op": "finish", "status": "finished"}], seq=2)
    assert index.scan(store).errors == 0
    assert index.run_detail(*_SCOPE)["status"] == "finished"
    expected = index.run_detail(*_SCOPE)
    index.rebuild(store)
    assert index.run_detail(*_SCOPE) == expected
