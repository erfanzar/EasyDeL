# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The cursor decoder that answered "start from the beginning" to everything.

DR-0002 invariant 4: *corrupt, stale, or incompatible cursors fail explicitly,
not as start or end*. The decoder this file covers used to return
``(None, {})`` for a truncated cursor, an oversized one, one carrying an
unknown version, one minted for another attempt, and one that was simply never
sent -- five different situations with one indistinguishable answer. A client
whose cursor was corrupted in transit re-read the entire journal every poll and
was never told why, and the test suite asserted that as the intended behaviour.

Every case here asserts a *typed outcome*. ``CursorFresh`` is reserved for empty
input; if a malformed cursor can still produce it, the defect is back.
"""

from __future__ import annotations

import base64
import json

import pytest

from eyvan.worker.runtime.log_records import (
    _LOG_CURSOR_MAX_BYTES,
    _LOG_CURSOR_MAX_ENTRIES,
    _LOG_CURSOR_MAX_VALUE_BYTES,
    ARCHIVE_OFFSET_PREFIX,
    ARCHIVE_POSITION_AMBIGUOUS,
    ARCHIVE_POSITION_BEYOND_ARCHIVE,
    ARCHIVE_POSITION_MALFORMED,
    ARCHIVE_POSITION_UNKNOWN,
    CURSOR_ATTEMPT_CHANGED,
    CURSOR_DUPLICATE_KEY,
    CURSOR_DUPLICATE_RANK,
    CURSOR_INVALID_ATTEMPT,
    CURSOR_INVALID_POSITION,
    CURSOR_INVALID_RANK,
    CURSOR_KIND_WORKLOAD_LOG,
    CURSOR_LEAF_TOO_LARGE,
    CURSOR_MALFORMED,
    CURSOR_MISSING_FIELD,
    CURSOR_MODE_ARCHIVE,
    CURSOR_MODE_LIVE,
    CURSOR_MODE_MISMATCH,
    CURSOR_RESET_REQUIRED,
    CURSOR_TOO_LARGE,
    CURSOR_TOO_MANY_POSITIONS,
    CURSOR_TOPOLOGY_CHANGED,
    CURSOR_UNKNOWN_FIELD,
    CURSOR_UNKNOWN_KIND,
    CURSOR_UNKNOWN_MODE,
    CURSOR_UNKNOWN_RANK,
    CURSOR_UNSUPPORTED_VERSION,
    CURSOR_WORKLOAD_MISMATCH,
    INVALID_LOG_RANK,
    LOG_CURSOR_V2_EMISSION_ENV,
    ArchiveGap,
    ArchiveOffset,
    CursorContext,
    CursorFresh,
    CursorIncompatible,
    CursorInvalid,
    CursorPosition,
    CursorResume,
    decode_log_cursor,
    encode_archive_offset,
    encode_log_cursor,
    encode_unplaced_archive_position,
    is_archive_position,
    join_log_cursor,
    log_cursor_v2_emission_enabled,
    log_page_from_json,
    resolve_archive_position,
)

_PREFIX = "eyvan-log-v2."


def _wrap(payload: bytes) -> str:
    """Frame raw bytes as a cursor without going through the encoder."""

    return _PREFIX + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _envelope(document: object) -> str:
    """Frame a JSON document as a cursor, bypassing every emission check."""

    return _wrap(json.dumps(document).encode())


def _bound(**overrides: object) -> dict[str, object]:
    """A complete bound-v2 document, overridable field by field."""

    document: dict[str, object] = {
        "v": 2,
        "kind": CURSOR_KIND_WORKLOAD_LOG,
        "attempt": 3,
        "topology": "sha256:aaaa",
        "mode": CURSOR_MODE_LIVE,
        "workload": "wl-1",
        "positions": [{"rank": 0, "cursor": "c0"}, {"rank": 1, "cursor": "c1"}],
    }
    document.update(overrides)
    return document


def _draft(**overrides: object) -> dict[str, object]:
    """A complete draft-v2 document -- the shape emitted today."""

    document: dict[str, object] = {"version": 2, "attempt": 3, "cursors": [[0, "c0"], [1, "c1"]]}
    document.update(overrides)
    return document


def _reason(value: str, *, expected: CursorContext | None = None) -> str:
    """Decode and return the refusal code, failing loudly on a resume."""

    outcome = decode_log_cursor(value, expected=expected)
    assert isinstance(outcome, (CursorInvalid, CursorIncompatible)), outcome
    return outcome.reason


# --------------------------------------------------------------------------- only empty means start


def test_only_empty_input_decodes_to_fresh() -> None:
    """The whole point. Everything else is a named refusal."""

    assert decode_log_cursor("") == CursorFresh()


@pytest.mark.parametrize(
    "value",
    [
        "eyvan-log-v2.!!!not-base64!!!",
        _wrap(b"{not json"),
        _wrap(b'"a string"'),
        _wrap(b"\xff\xfe\xfd"),
    ],
    ids=["not-base64", "not-json", "json-but-not-an-object", "not-utf8"],
)
def test_a_corrupt_cursor_is_invalid_rather_than_a_silent_restart(value: str) -> None:
    """Each of these used to be indistinguishable from a first read."""

    outcome = decode_log_cursor(value)

    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == CURSOR_MALFORMED
    assert outcome.detail


def test_an_oversized_cursor_is_a_named_refusal_not_a_restart() -> None:
    """A limit breach is an answer, not a reason to replay the journal.

    The bound is still enforced before any base64 or JSON allocation; what
    changed is that breaching it now says so.
    """

    oversized = "x" * (_LOG_CURSOR_MAX_BYTES + 1)

    assert _reason(oversized) == CURSOR_TOO_LARGE
    assert _reason(f"{_PREFIX}{oversized}") == CURSOR_TOO_LARGE


def test_too_many_positions_is_rejected_before_the_fan_out() -> None:
    """The cardinality bound exists so a tiny request cannot cause a huge fan-out."""

    over = [{"rank": rank, "cursor": "c"} for rank in range(_LOG_CURSOR_MAX_ENTRIES + 1)]

    assert _reason(_envelope(_bound(positions=over))) == CURSOR_TOO_MANY_POSITIONS
    assert _reason(_envelope(_draft(cursors=[[rank, "c"] for rank in range(_LOG_CURSOR_MAX_ENTRIES + 1)]))) == (
        CURSOR_TOO_MANY_POSITIONS
    )


def test_an_oversized_leaf_is_rejected_by_name() -> None:
    """One rank's position cannot be an unbounded allocation either."""

    huge = "x" * (_LOG_CURSOR_MAX_VALUE_BYTES + 1)

    assert _reason(_envelope(_bound(positions=[{"rank": 0, "cursor": huge}]))) == CURSOR_LEAF_TOO_LARGE
    assert _reason(_envelope(_draft(cursors=[[0, huge]]))) == CURSOR_LEAF_TOO_LARGE
    assert _reason(f"3:{huge}") == CURSOR_LEAF_TOO_LARGE


# --------------------------------------------------------------------------- the JSON layer


def test_a_repeated_json_key_is_refused_rather_than_last_wins() -> None:
    """``json`` collapses duplicates silently, so a cursor can carry two attempts.

    A proxy, a log line, or an operator reading the decoded document sees the
    first; the decoder used to act on the last.
    """

    payload = b'{"version":2,"attempt":1,"cursors":[[0,"c0"]],"attempt":9}'

    outcome = decode_log_cursor(_wrap(payload))

    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == CURSOR_DUPLICATE_KEY
    assert "attempt" in outcome.detail


def test_a_repeated_key_inside_a_position_is_refused_too() -> None:
    """The same collapse one level down decides which journal a rank resumes."""

    payload = (
        b'{"v":2,"kind":"workload-log","attempt":3,"mode":"live","positions":[{"rank":0,"cursor":"a","cursor":"b"}]}'
    )

    assert _reason(_wrap(payload)) == CURSOR_DUPLICATE_KEY


def test_an_unknown_field_is_refused_because_the_schema_is_closed() -> None:
    """Ignoring an unrecognised field is how two writers disagree in silence."""

    assert _reason(_envelope(_bound(extra="surprise"))) == CURSOR_UNKNOWN_FIELD
    assert _reason(_envelope(_draft(extra="surprise"))) == CURSOR_UNKNOWN_FIELD
    assert _reason(_envelope(_bound(positions=[{"rank": 0, "cursor": "c0", "junk": 1}]))) == CURSOR_UNKNOWN_FIELD


def test_a_missing_required_field_is_refused() -> None:
    document = _bound()
    del document["positions"]

    outcome = decode_log_cursor(_envelope(document))

    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == CURSOR_MISSING_FIELD
    assert "positions" in outcome.detail


def test_two_schema_versions_in_one_document_are_refused() -> None:
    """A document claiming both shapes has no single reading."""

    assert _reason(_envelope({"v": 2, "version": 2, "attempt": 3})) == CURSOR_MALFORMED


@pytest.mark.parametrize("version", [1, 3, "2", 2.0, None, True], ids=["one", "three", "text", "float", "null", "bool"])
def test_an_unknown_version_is_refused(version: object) -> None:
    """DR-0002: unknown version returns an explicit refusal, never start."""

    assert _reason(_envelope(_bound(v=version))) == CURSOR_UNSUPPORTED_VERSION
    assert _reason(_envelope(_draft(version=version))) == CURSOR_UNSUPPORTED_VERSION


def test_a_versionless_document_is_refused() -> None:
    assert _reason(_envelope({"attempt": 3, "cursors": []})) == CURSOR_UNSUPPORTED_VERSION


# --------------------------------------------------------------------------- ranks


@pytest.mark.parametrize(
    ("rank", "reason"),
    [
        (-1, CURSOR_INVALID_RANK),
        (True, CURSOR_INVALID_RANK),
        (False, CURSOR_INVALID_RANK),
        (1.0, CURSOR_INVALID_RANK),
        ("0", CURSOR_INVALID_RANK),
        (None, CURSOR_INVALID_RANK),
    ],
    ids=["negative", "true", "false", "float", "text", "null"],
)
def test_a_rank_that_is_not_a_non_negative_integer_is_refused(rank: object, reason: str) -> None:
    """``True`` indexes as ``1`` and ``1.0`` compares equal to ``1``.

    Either would quietly become a real rank and hand one host another host's
    position.
    """

    assert _reason(_envelope(_bound(positions=[{"rank": rank, "cursor": "c"}]))) == reason
    assert _reason(_envelope(_draft(cursors=[[rank, "c"]]))) == reason


def test_a_duplicate_rank_is_refused_rather_than_overwritten() -> None:
    """v1 let the last spelling of a rank win, so one journal decided both."""

    assert _reason(_envelope(_bound(positions=[{"rank": 0, "cursor": "a"}, {"rank": 0, "cursor": "b"}]))) == (
        CURSOR_DUPLICATE_RANK
    )
    assert _reason(_envelope(_draft(cursors=[[0, "a"], [0, "b"]]))) == CURSOR_DUPLICATE_RANK
    assert _reason("3:0=a|0=b") == CURSOR_DUPLICATE_RANK


def test_a_rank_outside_this_attempt_is_incompatible_not_invalid() -> None:
    """A well-formed cursor for a wider topology names a rank that is not here.

    Incompatible rather than invalid: nothing is corrupt, the read simply moved.
    """

    outcome = decode_log_cursor(_envelope(_bound()), expected=CursorContext(attempt=3, ranks=(0,)))

    assert isinstance(outcome, CursorIncompatible)
    assert outcome.reason == CURSOR_UNKNOWN_RANK
    assert "1" in outcome.detail


def test_a_rank_with_no_position_is_named_rather_than_assumed() -> None:
    """A resume covering three of four ranks is not a whole resume.

    The fourth restarts from the beginning of its journal; saying so is the
    difference between a partial read and a partial read presented as complete.
    """

    outcome = decode_log_cursor(_envelope(_bound()), expected=CursorContext(attempt=3, ranks=(0, 1, 2, 3)))

    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: "c0", 1: "c1"}
    assert outcome.missing_ranks == (2, 3)


def test_a_position_without_a_usable_leaf_is_refused() -> None:
    """An empty leaf means "replay this rank", which is not a position."""

    assert _reason(_envelope(_bound(positions=[{"rank": 0, "cursor": ""}]))) == CURSOR_INVALID_POSITION
    assert _reason(_envelope(_bound(positions=[{"rank": 0, "cursor": 7}]))) == CURSOR_INVALID_POSITION
    assert _reason(_envelope(_bound(positions=["0=c"]))) == CURSOR_INVALID_POSITION
    assert _reason(_envelope(_draft(cursors=[[0, "a", "b"]]))) == CURSOR_INVALID_POSITION


@pytest.mark.parametrize("attempt", [-1, True, "3", 1.5, None], ids=["negative", "bool", "text", "float", "null"])
def test_an_unusable_attempt_is_refused(attempt: object) -> None:
    assert _reason(_envelope(_bound(attempt=attempt))) == CURSOR_INVALID_ATTEMPT
    assert _reason(_envelope(_draft(attempt=attempt))) == CURSOR_INVALID_ATTEMPT


# --------------------------------------------------------------------------- leaves stay opaque


def test_a_leaf_may_contain_the_legacy_delimiters() -> None:
    """The reason the envelope exists: JSON needs no delimiter escaping."""

    leaf = "s=journal|position=7|rank=9"

    outcome = decode_log_cursor(_envelope(_bound(positions=[{"rank": 0, "cursor": leaf}])))

    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: leaf}


def test_a_leaf_may_be_unicode_including_astral_characters() -> None:
    """A backend position is bytes we hand back, not text we interpret."""

    leaf = "curseur-\u00e9\u4e2d\u6587-\U0001f680\u0000tail"

    outcome = decode_log_cursor(_envelope(_bound(positions=[{"rank": 0, "cursor": leaf}])))

    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: leaf}


def test_a_unicode_leaf_survives_emission_and_decoding() -> None:
    """The round trip, not just the decode: the encoder must not mangle it."""

    leaf = "s=\u00e9\U0001f680|7"

    outcome = decode_log_cursor(join_log_cursor(2, {0: leaf}))

    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: leaf}


def test_a_rank_with_an_empty_leaf_is_dropped_at_emission_not_encoded_as_empty() -> None:
    """Encoding an empty leaf would tell that rank to replay everything."""

    outcome = decode_log_cursor(join_log_cursor(1, {0: "a", 1: ""}))

    assert isinstance(outcome, CursorResume)
    assert outcome.cursors == {0: "a"}


def test_no_leaves_at_all_encodes_to_empty_which_is_the_only_fresh_input() -> None:
    """Empty means "wherever you were"; a start cursor means "replay"."""

    assert join_log_cursor(1, {}) == ""
    assert encode_log_cursor({}, context=CursorContext(attempt=1)) == ""
    assert decode_log_cursor("") == CursorFresh()


# --------------------------------------------------------------------------- context binding


def test_a_bound_cursor_carries_kind_mode_workload_topology_and_attempt() -> None:
    """Without a binding, a cursor minted for one journal replays against another."""

    outcome = decode_log_cursor(
        _envelope(_bound()),
        expected=CursorContext(
            kind=CURSOR_KIND_WORKLOAD_LOG,
            mode=CURSOR_MODE_LIVE,
            attempt=3,
            workload="wl-1",
            topology="sha256:aaaa",
        ),
    )

    assert isinstance(outcome, CursorResume)
    assert outcome.bound is True
    assert (outcome.kind, outcome.mode, outcome.workload, outcome.topology) == (
        CURSOR_KIND_WORKLOAD_LOG,
        CURSOR_MODE_LIVE,
        "wl-1",
        "sha256:aaaa",
    )
    assert outcome.cursors == {0: "c0", 1: "c1"}


def test_a_cursor_from_another_attempt_is_incompatible() -> None:
    """A retry runs a different execution; the old positions name dead journals."""

    outcome = decode_log_cursor(_envelope(_bound(attempt=2)), expected=CursorContext(attempt=3))

    assert isinstance(outcome, CursorIncompatible)
    assert outcome.reason == CURSOR_ATTEMPT_CHANGED
    assert "2" in outcome.detail


def test_a_cursor_from_another_topology_is_incompatible() -> None:
    """The rank space is derived from the topology; change it and rank N moved."""

    assert (
        _reason(_envelope(_bound()), expected=CursorContext(attempt=3, topology="sha256:bbbb"))
        == CURSOR_TOPOLOGY_CHANGED
    )


def test_a_cursor_from_another_workload_is_incompatible() -> None:
    """base64 is not authorization and not addressing."""

    assert _reason(_envelope(_bound()), expected=CursorContext(attempt=3, workload="wl-2")) == (CURSOR_WORKLOAD_MISMATCH)


def test_a_live_cursor_replayed_against_the_archive_is_incompatible() -> None:
    """Live positions address backend journals; archive positions address objects."""

    assert _reason(_envelope(_bound()), expected=CursorContext(attempt=3, mode=CURSOR_MODE_ARCHIVE)) == (
        CURSOR_MODE_MISMATCH
    )


def test_an_archive_cursor_decodes_against_an_archive_read() -> None:
    """The other half of the mode check: the matching mode must still work."""

    outcome = decode_log_cursor(
        _envelope(_bound(mode=CURSOR_MODE_ARCHIVE)),
        expected=CursorContext(attempt=3, mode=CURSOR_MODE_ARCHIVE),
    )

    assert isinstance(outcome, CursorResume)
    assert outcome.mode == CURSOR_MODE_ARCHIVE


@pytest.mark.parametrize("kind", ["cell-metrics", "", "workload_log", 7], ids=["other", "empty", "typo", "not-text"])
def test_a_cursor_for_an_unknown_journal_kind_is_refused(kind: object) -> None:
    """Only one kind of journal mints these; anything else addresses nothing here."""

    assert _reason(_envelope(_bound(kind=kind))) == CURSOR_UNKNOWN_KIND


@pytest.mark.parametrize("mode", ["tail", "", "LIVE", 7], ids=["other", "empty", "case", "not-text"])
def test_a_cursor_with_an_unknown_read_mode_is_refused(mode: object) -> None:
    """The mode vocabulary is closed, so an unrecognised one is not a mismatch."""

    assert _reason(_envelope(_bound(mode=mode))) == CURSOR_UNKNOWN_MODE


def test_a_draft_cursor_is_still_decoded_and_reported_as_unbound() -> None:
    """Today's emission must keep working, and must not claim a binding it lacks."""

    outcome = decode_log_cursor(_envelope(_draft()), expected=CursorContext(attempt=3, workload="wl-9"))

    assert isinstance(outcome, CursorResume)
    assert outcome.bound is False
    assert outcome.cursors == {0: "c0", 1: "c1"}
    assert outcome.workload == ""


def test_a_draft_cursor_from_another_attempt_is_still_incompatible() -> None:
    """The one binding the draft envelope does carry is still enforced."""

    assert _reason(_envelope(_draft(attempt=1)), expected=CursorContext(attempt=3)) == CURSOR_ATTEMPT_CHANGED


# --------------------------------------------------------------------------- the legacy grammar


def test_a_simple_legacy_cursor_is_accepted() -> None:
    """``"{attempt}:{cursor}"`` has exactly one reading, delimiters and all."""

    outcome = decode_log_cursor("3:s=abc|journal=cursor")

    assert isinstance(outcome, CursorResume)
    assert outcome.version == 1
    assert outcome.attempt == 3
    assert outcome.cursors == {0: "s=abc|journal=cursor"}


def test_a_legacy_cursor_with_no_position_resumes_the_attempt_only() -> None:
    outcome = decode_log_cursor("7:")

    assert isinstance(outcome, CursorResume)
    assert (outcome.attempt, outcome.cursors) == (7, {})


def test_an_ambiguous_multi_rank_legacy_cursor_is_refused() -> None:
    """``"3:0=a|1=b"`` reads equally as one rank whose cursor is ``a|1=b``.

    DR-0002 requires ``cursor_reset_required`` rather than a guess, because the
    guess hands rank 1 a position from a journal that never issued it.
    """

    outcome = decode_log_cursor("3:0=a|1=b")

    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason == CURSOR_RESET_REQUIRED


def test_a_multi_rank_legacy_cursor_is_accepted_against_a_known_rank_set() -> None:
    """The rank set is the witness that makes the split reversible."""

    outcome = decode_log_cursor("3:0=a|1=b", expected=CursorContext(attempt=3, ranks=(0, 1)))

    assert isinstance(outcome, CursorResume)
    assert outcome.version == 1
    assert outcome.cursors == {0: "a", 1: "b"}


def test_a_legacy_cursor_whose_ranks_do_not_match_the_attempt_is_incompatible() -> None:
    """Missing or extra ranks mean the split was read against the wrong shape."""

    assert _reason("3:0=a|1=b", expected=CursorContext(attempt=3, ranks=(0, 1, 2))) == CURSOR_RESET_REQUIRED
    assert _reason("3:0=a|1=b", expected=CursorContext(attempt=3, ranks=(0,))) == CURSOR_RESET_REQUIRED


def test_a_legacy_piece_that_is_not_a_rank_assignment_is_refused() -> None:
    """The old decoder skipped these, so the surviving ranks looked complete."""

    assert _reason("3:0=a|garbage", expected=CursorContext(ranks=(0, 1))) == CURSOR_RESET_REQUIRED
    assert _reason("3:0=a|1=", expected=CursorContext(ranks=(0, 1))) == CURSOR_RESET_REQUIRED
    assert _reason("3:0=a|+1=b", expected=CursorContext(ranks=(0, 1))) == CURSOR_RESET_REQUIRED


def test_a_legacy_cursor_that_nests_a_cursor_is_refused() -> None:
    """A nested level in a grammar with no escaping cannot be unnested."""

    assert _reason("3:0=a|1=4:0=x", expected=CursorContext(ranks=(0, 1))) == CURSOR_RESET_REQUIRED
    assert _reason(f"3:0=a|1={_PREFIX}abc", expected=CursorContext(ranks=(0, 1))) == CURSOR_RESET_REQUIRED


@pytest.mark.parametrize(
    "value",
    ["abc", "notanint:0=abc", "-1:0=abc", ":0=abc", " 3:0=abc"],
    ids=["no-attempt", "unparseable", "negative", "empty-attempt", "padded"],
)
def test_a_value_that_is_not_a_cursor_at_all_is_invalid(value: str) -> None:
    """ "No opinion" was the old answer to these, which reads as a first poll."""

    outcome = decode_log_cursor(value)

    assert isinstance(outcome, CursorInvalid)
    assert outcome.reason in {CURSOR_MALFORMED, CURSOR_INVALID_ATTEMPT}


# --------------------------------------------------------------------------- emission stays behind the flag


def test_the_bound_envelope_is_not_emitted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage one ships the decoder everywhere; emission waits.

    A rolling upgrade must not meet a schema its rollback image cannot read, so
    the default emission is byte-for-byte the draft shape.
    """

    monkeypatch.delenv(LOG_CURSOR_V2_EMISSION_ENV, raising=False)

    assert log_cursor_v2_emission_enabled({}) is False

    encoded = join_log_cursor(3, {0: "c0", 1: "c1"})
    document = json.loads(base64.urlsafe_b64decode(encoded.removeprefix(_PREFIX) + "=="))

    assert document == {"version": 2, "attempt": 3, "cursors": [[0, "c0"], [1, "c1"]]}


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"], ids=["one", "true", "upper", "yes", "on"])
def test_the_flag_turns_emission_on(raw: str) -> None:
    assert log_cursor_v2_emission_enabled({LOG_CURSOR_V2_EMISSION_ENV: raw}) is True


@pytest.mark.parametrize("raw", ["", "0", "false", "no", "maybe"], ids=["unset", "zero", "false", "no", "other"])
def test_anything_else_leaves_emission_off(raw: str) -> None:
    assert log_cursor_v2_emission_enabled({LOG_CURSOR_V2_EMISSION_ENV: raw}) is False


def test_the_flag_changes_the_bytes_join_log_cursor_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag has to reach the call sites that already exist, not just new ones."""

    monkeypatch.setenv(LOG_CURSOR_V2_EMISSION_ENV, "1")

    encoded = join_log_cursor(3, {0: "c0"})
    document = json.loads(base64.urlsafe_b64decode(encoded.removeprefix(_PREFIX) + "=="))

    assert document["v"] == 2
    assert document["kind"] == CURSOR_KIND_WORKLOAD_LOG
    assert document["positions"] == [{"rank": 0, "cursor": "c0"}]
    assert "version" not in document and "cursors" not in document


def test_the_bound_envelope_records_the_context_it_was_minted_for() -> None:
    """What binding means on the wire, not just in the decoder."""

    encoded = encode_log_cursor(
        [CursorPosition(rank=4, cursor="c4", cell="cell-b", local=1)],
        context=CursorContext(attempt=9, workload="wl-7", topology="sha256:cccc", mode=CURSOR_MODE_ARCHIVE),
        bound=True,
    )
    document = json.loads(base64.urlsafe_b64decode(encoded.removeprefix(_PREFIX) + "=="))

    assert document == {
        "v": 2,
        "kind": CURSOR_KIND_WORKLOAD_LOG,
        "attempt": 9,
        "topology": "sha256:cccc",
        "mode": CURSOR_MODE_ARCHIVE,
        "workload": "wl-7",
        "positions": [{"rank": 4, "cursor": "c4", "cell": "cell-b", "local": 1}],
    }


def test_both_emitted_schemas_decode_to_the_same_positions() -> None:
    """N and N-1 must agree, which is the precondition for flipping the flag."""

    positions = {0: "c0", 3: "s=journal|x=1"}
    draft = encode_log_cursor(positions, context=CursorContext(attempt=5), bound=False)
    bound = encode_log_cursor(positions, context=CursorContext(attempt=5), bound=True)

    assert draft != bound
    for encoded in (draft, bound):
        outcome = decode_log_cursor(encoded, expected=CursorContext(attempt=5))
        assert isinstance(outcome, CursorResume)
        assert outcome.cursors == positions


def test_emission_still_refuses_an_unusable_rank_or_size() -> None:
    """The writer's bounds are unchanged; only the reader's answers are new."""

    with pytest.raises(ValueError, match="entry limit"):
        join_log_cursor(1, {rank: "c" for rank in range(_LOG_CURSOR_MAX_ENTRIES + 1)})
    with pytest.raises(ValueError, match="size limit"):
        join_log_cursor(1, {0: "x" * (_LOG_CURSOR_MAX_VALUE_BYTES + 1)})
    with pytest.raises(ValueError, match="non-negative"):
        join_log_cursor(1, {-1: "c"})
    with pytest.raises(ValueError, match="non-negative"):
        join_log_cursor(-1, {0: "c"})


# --------------------------------------------------------------------------- the page decoder


def test_an_unusable_rank_survives_decoding_instead_of_becoming_rank_zero() -> None:
    """Two bad ranks used to collide onto rank 0 and the page looked complete.

    The controller validates each rank against the cell's durable process width,
    so the sentinel is what lets it see the page is partial.
    """

    page = log_page_from_json(
        {"records": [{"message": "a", "rank": "nope"}, {"message": "b", "rank": -4}, {"message": "c", "rank": True}]}
    )

    assert [record.rank for record in page.records] == [INVALID_LOG_RANK] * 3
    assert [record.message for record in page.records] == ["a", "b", "c"]


def test_an_absent_rank_is_still_the_documented_default() -> None:
    """A missing field is not a declared-but-unusable one."""

    page = log_page_from_json({"records": [{"message": "a"}]})

    assert page.records[0].rank == 0


def test_a_malformed_truncation_entry_is_preserved_rather_than_filtered() -> None:
    """Dropping it turned a truncation notice into "nothing was truncated"."""

    page = log_page_from_json({"records": [], "truncated_ranks": [0, "two", -3, None]})

    assert page.truncated_ranks == (0, INVALID_LOG_RANK, INVALID_LOG_RANK, INVALID_LOG_RANK)


def test_a_truncation_field_that_is_not_a_list_is_not_read_as_nothing_truncated() -> None:
    page = log_page_from_json({"records": [], "truncated_ranks": "0"})

    assert page.truncated_ranks == (INVALID_LOG_RANK,)


def test_a_boolean_pid_is_not_read_as_process_one() -> None:
    """``True`` is an ``int`` in Python and would have been reported as PID 1."""

    page = log_page_from_json({"records": [{"message": "a", "pid": True}]})

    assert page.records[0].pid is None


# --------------------------------------------------------------------------- archive positions


def test_an_empty_position_starts_the_archive_from_its_first_line() -> None:
    """The only input that means "start", here as in the cursor decoder."""

    assert resolve_archive_position("", ["c0", "c1"], rank=2) == ArchiveOffset(rank=2, offset=0)


def test_a_source_position_resumes_at_the_line_after_the_one_it_names() -> None:
    """The handover: every archived line keeps the position its backend issued."""

    assert resolve_archive_position("c1", ["c0", "c1", "c2"], rank=0) == ArchiveOffset(rank=0, offset=2)


def test_a_source_position_no_archived_line_carries_is_a_gap_not_offset_zero() -> None:
    """Offset zero replays the object, and the receiver cannot tell it happened."""

    outcome = resolve_archive_position("c9", ["c0", "c1"], rank=3)

    assert isinstance(outcome, ArchiveGap)
    assert outcome.reason == ARCHIVE_POSITION_UNKNOWN
    assert (outcome.rank, outcome.position) == (3, "c9")


def test_a_position_two_archived_lines_carry_is_a_gap_rather_than_a_choice() -> None:
    """One answer duplicates lines and the other drops them; neither is known."""

    outcome = resolve_archive_position("same", ["same", "other", "same"], rank=0)

    assert isinstance(outcome, ArchiveGap)
    assert outcome.reason == ARCHIVE_POSITION_AMBIGUOUS


def test_an_offset_past_the_end_of_the_object_is_a_gap() -> None:
    """The object cannot have served more lines than it holds."""

    outcome = resolve_archive_position(encode_archive_offset(3), ["c0", "c1"], rank=0)

    assert isinstance(outcome, ArchiveGap)
    assert outcome.reason == ARCHIVE_POSITION_BEYOND_ARCHIVE


def test_an_offset_at_the_end_of_the_object_is_a_caught_up_follower() -> None:
    """ "Nothing new" and "cannot be placed" are different answers."""

    assert resolve_archive_position(encode_archive_offset(2), ["c0", "c1"], rank=0) == ArchiveOffset(rank=0, offset=2)


@pytest.mark.parametrize("text", ["", "-1", "01", " 1", "1_0", "1.0"])
def test_a_malformed_offset_is_a_gap_rather_than_a_guess(text: str) -> None:
    """Two spellings of one offset, or none at all, must not both resolve."""

    outcome = resolve_archive_position(f"{ARCHIVE_OFFSET_PREFIX}{text}", ["c0"], rank=0)

    assert isinstance(outcome, ArchiveGap)
    assert outcome.reason == ARCHIVE_POSITION_MALFORMED


def test_an_archive_position_is_recognisable_without_a_bound_envelope() -> None:
    """The default envelope carries no mode, so the value has to say what it is."""

    assert is_archive_position(encode_archive_offset(4))
    assert is_archive_position(encode_unplaced_archive_position("s=journal;i=9"))
    assert not is_archive_position("s=journal;i=9")
    assert not is_archive_position("4")


def test_an_unplaced_position_is_retained_verbatim_behind_its_tag() -> None:
    """It has to survive the round trip, or the next poll is a first read again."""

    tagged = encode_unplaced_archive_position("s=journal;i=9")

    assert tagged.endswith("s=journal;i=9")
    assert encode_unplaced_archive_position(tagged) == tagged
    assert resolve_archive_position(tagged, ["s=journal;i=9"], rank=0) == ArchiveOffset(rank=0, offset=1)
