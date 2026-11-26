# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The hop that made structured logs absent on every remote fleet.

Both ends already had everything. The worker's systemd runtime has produced
structured pages since Phase 4 -- journald's own JSON, with priority, timestamp,
PID and an opaque cursor -- and ``log_records`` knew how to merge and render
them. There was simply no route between them: the worker exposed only the
plain-text ``logs`` endpoint, so ``eyvan workload log-page`` answered
``unsupported_capability`` for every fleet that runs on a remote worker, which
is every TPU fleet.

Observed running P2: `execution backend 'eyvan-native' does not expose
structured logs`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from eyvan.worker.runtime.log_records import (
    CursorFresh,
    CursorInvalid,
    LogPage,
    LogRecord,
    decode_log_cursor,
    join_log_cursor,
    log_page_from_json,
    split_log_cursor,
)


def _record(message: str, *, rank: int = 0, cursor: str = "c1") -> LogRecord:
    return LogRecord(
        message=message,
        stream="stdout",
        timestamp=datetime(2026, 8, 6, 12, 0, tzinfo=UTC),
        cursor=cursor,
        rank=rank,
    )


# --------------------------------------------------------------------------- the wire shape


def test_a_page_survives_a_round_trip() -> None:
    """`as_json` and `log_page_from_json` are one protocol, not two spellings."""

    page = LogPage(records=(_record("hello"),), next_cursor="c1")
    document = {
        "records": [dict(r.as_json()) for r in page.records],
        "next_cursor": page.next_cursor,
        "unavailable": False,
        "detail": "",
        "truncated_ranks": [],
    }

    restored = log_page_from_json(document)

    assert [r.message for r in restored.records] == ["hello"]
    assert restored.records[0].timestamp == page.records[0].timestamp
    assert restored.next_cursor == "c1"
    # The field this round trip was silently dropping. `stream` is decoded by the
    # sender, and feeding it back through the *priority* mapper turned every
    # 'stdout' and 'stderr' into 'unknown' -- so the one thing structured logs add
    # over plain text was empty on every remote read, and the first version of
    # this test asserted everything except it.
    assert restored.records[0].stream == "stdout"


def test_both_streams_survive_the_wire() -> None:
    """stdout and stderr must stay distinguishable, which is half the point."""

    document = {
        "records": [
            dict(LogRecord(message="out", stream="stdout").as_json()),
            dict(LogRecord(message="err", stream="stderr").as_json()),
        ]
    }

    page = log_page_from_json(document)

    assert [r.stream for r in page.records] == ["stdout", "stderr"]


def test_a_raw_journald_priority_is_still_understood() -> None:
    """A sender forwarding PRIORITY rather than a decoded name still works."""

    page = log_page_from_json({"records": [{"message": "x", "stream": 3}]})

    assert page.records[0].stream == "stderr"


def test_a_malformed_payload_becomes_unavailable_rather_than_raising() -> None:
    """A follower polling every second must not have to catch exceptions.

    "Unavailable" is also the honest description of a host that answered
    nonsense, which an empty page is not.
    """

    assert log_page_from_json("not a document").unavailable is True
    assert log_page_from_json({"records": "not a list"}).records == ()


def test_an_unparseable_timestamp_does_not_lose_the_line() -> None:
    """The message is the point; the timestamp is context."""

    page = log_page_from_json({"records": [{"message": "kept", "timestamp": "not-a-date"}]})

    assert [r.message for r in page.records] == ["kept"]
    assert page.records[0].timestamp is None


# --------------------------------------------------------------------------- per-rank cursors


def test_each_rank_resumes_from_its_own_position() -> None:
    """A cursor is meaningless to a journal that did not issue it."""

    encoded = join_log_cursor(2, {0: "a", 1: "b"})
    attempt, cursors = split_log_cursor(encoded)

    assert attempt == 2
    assert cursors == {0: "a", 1: "b"}


def test_an_inner_cursor_can_contain_the_legacy_delimiters() -> None:
    """Opaque v2 framing must not parse backend cursor contents as structure."""

    inner = "s=journal|position=7"
    encoded = join_log_cursor(2, {0: inner})

    assert "|" not in encoded
    assert split_log_cursor(encoded) == (2, {0: inner})


def test_v2_cursor_encoding_is_deterministic_across_mapping_order() -> None:
    first = join_log_cursor(4, {2: "c2", 0: "c0", 1: "c1"})
    second = join_log_cursor(4, {1: "c1", 2: "c2", 0: "c0"})

    assert first == second
    assert first.startswith("eyvan-log-v2.")


def test_nested_slice_cursors_with_the_largest_manual_host_count_round_trip() -> None:
    """The safety bounds must still admit real two-level fan-out cursors."""

    inner_positions = {rank: f"s=journal-{rank}|position=7" for rank in range(512)}
    inner = join_log_cursor(0, inner_positions)
    outer = join_log_cursor(9, {0: inner, 1: join_log_cursor(0, {0: "single-host"})})

    assert split_log_cursor(outer) == (9, {0: inner, 1: join_log_cursor(0, {0: "single-host"})})
    assert split_log_cursor(split_log_cursor(outer)[1][0]) == (0, inner_positions)


def test_cursor_size_and_entry_bounds_fail_closed_before_decode() -> None:
    """Hostile query cursors cannot trigger unbounded base64/JSON work.

    The bound is still checked before any base64 or JSON allocation. What this
    test no longer says is that breaching it *restarts the read*: the earlier
    version asserted ``split_log_cursor(oversized) == (None, {})``, which is the
    same answer as "no cursor was sent", so the assertion pinned the fail-open
    decoder DR-0002 invariant 4 exists to forbid. A refused cursor now names
    itself, and only empty input decodes to :class:`CursorFresh`.
    """

    import eyvan.worker.runtime.log_records as log_records

    oversized = "x" * (log_records._LOG_CURSOR_MAX_BYTES + 1)

    for value in (oversized, f"eyvan-log-v2.{oversized}"):
        outcome = decode_log_cursor(value)
        assert isinstance(outcome, CursorInvalid)
        assert outcome.reason == log_records.CURSOR_TOO_LARGE
        assert outcome != CursorFresh()

    with pytest.raises(ValueError, match="entry limit"):
        join_log_cursor(1, {rank: "c" for rank in range(log_records._LOG_CURSOR_MAX_ENTRIES + 1)})
    with pytest.raises(ValueError, match="size limit"):
        join_log_cursor(1, {0: "x" * (log_records._LOG_CURSOR_MAX_VALUE_BYTES + 1)})


def test_an_omitted_leaf_is_named_by_the_decoder_rather_than_silently_missing() -> None:
    """An empty cursor is not a position, so the gap has to be reported instead.

    This used to be asserted as a *feature* -- "a rank that reported no cursor is
    omitted" -- and the controller relied on it, keying the envelope by cell index
    and leaving out any cell that had not moved. DR-0002 lines 42-44 and 68-70 say
    why that is wrong: an omitted key is indistinguishable from a rank that is not
    part of the attempt, so the reader filled the gap with "start from the
    beginning" and that rank replayed its whole journal while the page reported a
    clean read.

    The encoder still drops an empty leaf, because an empty string genuinely is
    not a position. What changed is that the omission is no longer silent: decoded
    against the attempt's expected rank set, every rank with no position comes
    back in ``missing_ranks``, so "has no position" and "is not in this attempt"
    are different answers on the wire.
    """

    from eyvan.worker.runtime.log_records import (
        CURSOR_KIND_WORKLOAD_LOG,
        CURSOR_MODE_LIVE,
        CursorContext,
        CursorResume,
        decode_log_cursor,
    )

    encoded = join_log_cursor(1, {0: "a", 1: "", 3: "d"})
    outcome = decode_log_cursor(
        encoded,
        expected=CursorContext(
            kind=CURSOR_KIND_WORKLOAD_LOG,
            mode=CURSOR_MODE_LIVE,
            attempt=1,
            ranks=(0, 1, 2, 3),
        ),
    )

    assert encoded.startswith("eyvan-log-v2.")
    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: "a", 3: "d"}
    assert outcome.missing_ranks == (1, 2)


def test_no_cursors_at_all_encodes_to_empty() -> None:
    """Empty means "wherever you were"; a start cursor means "replay"."""

    assert join_log_cursor(1, {}) == ""


# --------------------------------------------------------------------------- the slice fan-out


class _Worker:
    def __init__(self, page: LogPage | Exception) -> None:
        self._page = page
        self.seen_after: str | None = None

    def log_page(self, _remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        self.seen_after = after
        if isinstance(self._page, Exception):
            raise self._page
        return self._page


def _slice(*workers: _Worker):
    from eyvan.worker.runtime.native_slice import NativeSliceExecutor

    executor = NativeSliceExecutor.__new__(NativeSliceExecutor)
    executor._workers = list(workers)
    return executor


def test_every_host_in_the_slice_is_merged() -> None:
    """A v5p-16 is one cell with two hosts; reporting one is reporting half."""

    page = _slice(
        _Worker(LogPage(records=(_record("from host 0"),), next_cursor="c0")),
        _Worker(LogPage(records=(_record("from host 1"),), next_cursor="c1")),
    ).log_page("wl-1")

    assert sorted(r.message for r in page.records) == ["from host 0", "from host 1"]
    assert {r.rank for r in page.records} == {0, 1}


def test_an_unreachable_host_is_reported_in_place() -> None:
    """The failure this aggregation exists to prevent.

    Silently returning the hosts that answered shows a complete-looking log for a
    half-dead slice.
    """

    page = _slice(
        _Worker(LogPage(records=(_record("alive"),), next_cursor="c0")),
        _Worker(httpx.ConnectError("host is gone")),
    ).log_page("wl-1")

    assert [r.message for r in page.records] == ["alive"]
    assert page.truncated_ranks == (1,)


def test_each_host_is_handed_its_own_cursor() -> None:
    """Handing rank 1 rank 0's cursor asks journald to resume from a foreign id."""

    first, second = _Worker(LogPage()), _Worker(LogPage())

    _slice(first, second).log_page("wl-1", after=join_log_cursor(1, {0: "aaa", 1: "bbb"}))

    assert first.seen_after == "aaa"
    assert second.seen_after == "bbb"


def test_the_slice_cursor_carries_every_host_position() -> None:
    """`merge_pages` returns none on purpose; the slice composes the real one."""

    page = _slice(
        _Worker(LogPage(records=(_record("a"),), next_cursor="c0")),
        _Worker(LogPage(records=(_record("b"),), next_cursor="c1")),
    ).log_page("wl-1", after=join_log_cursor(3, {}))

    _attempt, cursors = split_log_cursor(page.next_cursor)
    assert cursors == {0: "c0", 1: "c1"}


def test_the_control_plane_decoder_keeps_the_timestamp() -> None:
    """The defect that made multi-host ordering meaningless.

    `_log_page_from_document` built `LogRecord` itself and omitted `timestamp`
    entirely, so every record read over the wire arrived without one. Worse than
    a missing column: `merge_pages` orders by `(timestamp or _EPOCH, rank)`, so
    with every timestamp None a multi-host page collapsed to rank order and the
    chronological interleaving that per-rank attribution exists to produce was
    silently gone.
    """

    from eyvan.server.controller.observation import _log_page_from_document

    document = {"records": [dict(_record("line", cursor="c1").as_json())], "next_cursor": "c1"}

    page = _log_page_from_document(document, rank=3)

    assert page.records[0].timestamp == datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    assert page.records[0].stream == "stdout"
    assert page.records[0].rank == 3


def test_records_from_two_hosts_interleave_by_time() -> None:
    """What the lost timestamp cost: ordering across hosts.

    Rank 1's earlier line must sort before rank 0's later one. Without
    timestamps every record ties at the epoch and the merge degenerates to rank
    order, which reads as "host 0's whole log, then host 1's".
    """

    from eyvan.worker.runtime.log_records import merge_pages

    early = LogRecord(message="early", timestamp=datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC), rank=1)
    late = LogRecord(message="late", timestamp=datetime(2026, 8, 6, 12, 0, 5, tzinfo=UTC), rank=0)

    merged = merge_pages({0: LogPage(records=(late,)), 1: LogPage(records=(early,))})

    assert [r.message for r in merged.records] == ["early", "late"]
