# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structured log records, shared by every execution backend.

Sibling of ``callable_protocol.py`` and under the same discipline: **this module
imports no Eyvan package modules**, because the OCI runner stages it beside the
worker rather than installing the package.

**Log lines do not go in ``audit_events``.** Three reasons, each independently
sufficient. ``prune_events`` deserializes every row before testing the cutoff, so
at log volume it starves the governance ledger it exists to trim.
``audit_retention_s`` is 90 days -- right for a compliance ledger, catastrophic
for stdout. And interleaving log lines into the audit ``sequence`` space breaks
the 256/4096 lookback-and-dedupe budget ``stream_events`` sizes deliberately.

**journald stays the body store.** It already provides durable ordering, an
opaque ``__CURSOR``, ``--after-cursor``, ``--lines``, ``--since``, and
``--output=json`` with ``PRIORITY`` and ``_PID``. Reimplementing that in Postgres
would be strictly worse; the control plane keeps an *index* plus an archived blob
for terminal attempts, because releasing a cell otherwise destroys the logs.

**A refused cursor is an answer.** ``decode_log_cursor`` returns a
:data:`CursorOutcome` -- fresh, resume, invalid, or incompatible -- because the
one thing a log cursor decoder must never do is answer "start from the
beginning" to a cursor it could not read. That answer is byte-identical to a
first poll, so a client whose cursor was truncated, oversized, minted for
another attempt, or aimed at another journal re-reads everything on every poll
and is never told why. See DR-0002 (``docs/design/0002-global-log-ranks-and-cursors.md``)
invariant 4.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import typing as tp
from dataclasses import dataclass, field
from datetime import UTC, datetime

#: Which stream a line came from. ``"unknown"`` is a real value, not a default:
#: a missing or unexpected ``PRIORITY`` must not be silently reported as stdout,
#: mirroring the ``pids is None`` vs ``[]`` discipline ``systemd_transient``
#: already documents.
LogStream = tp.Literal["stdout", "stderr", "unknown"]

#: journald priorities that mean "this was written to stderr". Anything outside
#: this set and the stdout set is ``unknown`` rather than guessed.
_STDERR_PRIORITIES = frozenset({"0", "1", "2", "3", "4"})
_STDOUT_PRIORITIES = frozenset({"5", "6", "7"})

#: Marker stored as the first JSON line in rank zero's terminal log archive.
#: The manifest makes sparse/truncated ranks replayable after their cells and
#: durable runtime process-count records are gone.
LOG_ARCHIVE_MANIFEST_KEY = "_eyvan_log_archive"
LOG_ARCHIVE_MANIFEST_VERSION = 1
#: Contiguous rank range probed by readers that predate archive manifests.
#: Writers may place compatibility gap objects only inside this fixed window.
LOG_ARCHIVE_LEGACY_RANK_SCAN_LIMIT = 64

#: Marker stored as the first JSON line of an incremental log chunk. Chunks hold
#: every rank together rather than one object per rank, because they are written
#: while the workload runs: a per-rank split would multiply the object count by
#: the attempt's width on every interval, and the records already name their own
#: rank. Its own key and its own version, distinct from the terminal manifest's,
#: so neither reader can be handed the other's document.
LOG_CHUNK_HEADER_KEY = "_eyvan_log_chunk"
LOG_CHUNK_VERSION = 1

#: Rank recorded when a payload declared a ``rank`` that is not a non-negative
#: integer. Coercing such a rank to ``0`` is worse than useless: two malformed
#: ranks collide onto rank zero and the page then looks complete to the
#: controller, which validates ranks against the cell's durable process width.
#: ``-1`` fails that range check, so invalidity survives decoding instead of
#: being normalised away.
INVALID_LOG_RANK = -1


@dataclass(frozen=True, slots=True)
class LogRecord:
    """One captured output line, with enough context to place it."""

    message: str
    stream: LogStream = "unknown"
    timestamp: datetime | None = None
    #: Backend-opaque resume position. journald's ``__CURSOR``, a byte offset for
    #: local processes, a timestamp for Docker. Never parsed by the control
    #: plane -- only handed back to the backend that issued it.
    cursor: str = ""
    rank: int = 0
    pid: int | None = None
    #: journald's ``_HOSTNAME``: which machine produced the line. None for
    #: backends that do not report one.
    host: str | None = None

    def as_json(self) -> dict[str, object]:
        """Return the stable wire shape."""

        return {
            "message": self.message,
            "stream": self.stream,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "cursor": self.cursor,
            "rank": self.rank,
            "pid": self.pid,
            "host": self.host,
        }


@dataclass(frozen=True, slots=True)
class LogPage:
    """One page of log records plus the cursor to continue from."""

    records: tuple[LogRecord, ...] = ()
    #: Where a follower resumes. Empty when nothing was read, which is distinct
    #: from a cursor pointing at the start: resuming from empty means "wherever
    #: you were", and resuming from a start cursor means "replay everything".
    next_cursor: str = ""
    #: True when the backend reported it could not read logs at all, as opposed
    #: to reading zero lines. A caller that cannot tell those apart will report
    #: "no output" for a job whose host is unreachable.
    unavailable: bool = False
    detail: str = ""
    truncated_ranks: tuple[int, ...] = field(default_factory=tuple)


_LOG_CURSOR_PREFIX = "eyvan-log-v2."
_LOG_CURSOR_VERSION = 2
# A cursor is accepted from an API query and recursively embeds one opaque
# backend position per fan-out level. Bound both its serialized size and its
# cardinality before decoding so an attacker cannot turn a tiny log request into
# an unbounded base64/JSON allocation. The entry ceiling matches the largest
# workload cell group accepted by the public model; 16 MiB still accommodates a
# legitimate 10,000-cell outer cursor whose entries are ordinary journald or
# small nested slice cursors.
_LOG_CURSOR_MAX_BYTES = 16 * 1024 * 1024
_LOG_CURSOR_MAX_ENTRIES = 10_000
_LOG_CURSOR_MAX_VALUE_BYTES = 8 * 1024 * 1024
_LEGACY_RANK_PREFIX = re.compile(r"^[0-9]+=")
#: A legacy rank is a canonical decimal. ``int()`` also accepts ``" +1_0 "``,
#: which would let two spellings of the same rank collide silently.
_CANONICAL_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)$")
#: A legacy leaf that itself looks like a cursor cannot be told apart from a
#: second level of the same delimiter grammar, so it is refused rather than guessed.
_LEGACY_NESTED_LEAF = re.compile(r"^(?:0|[1-9][0-9]*):")

#: The only journal kind this codec mints cursors for. A cursor naming any other
#: kind was minted for a different reader and is not decoded into positions.
CURSOR_KIND_WORKLOAD_LOG = "workload-log"
_KNOWN_CURSOR_KINDS = frozenset({CURSOR_KIND_WORKLOAD_LOG})

#: Live reads address backend journals; archive reads address immutable objects.
#: A live position replayed against the archive addresses nothing.
CURSOR_MODE_LIVE = "live"
CURSOR_MODE_ARCHIVE = "archive"
_KNOWN_CURSOR_MODES = frozenset({CURSOR_MODE_LIVE, CURSOR_MODE_ARCHIVE})

#: Environment flag gating *emission* of the fully bound DR-0002 v2 envelope.
#: Decoding both shapes ships first and unconditionally; emission stays off
#: until every serving and rollback image can decode it, which is the whole
#: point of the two-stage rollout. Turning this on mid-upgrade is safe only once
#: that is true.
LOG_CURSOR_V2_EMISSION_ENV = "EYVAN_LOG_CURSOR_V2_EMISSION_ENABLED"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Closed key sets. An envelope carrying a key outside its schema is rejected
# rather than ignored: silently dropping a field is how a reader ends up
# resuming from a position that meant something else to the writer.
_STRICT_ENVELOPE_KEYS = frozenset({"v", "kind", "attempt", "topology", "mode", "workload", "positions"})
_STRICT_ENVELOPE_REQUIRED = frozenset({"v", "kind", "attempt", "mode", "positions"})
_STRICT_POSITION_KEYS = frozenset({"rank", "cursor", "cell", "local"})
_STRICT_POSITION_REQUIRED = frozenset({"rank", "cursor"})
#: The shape emitted before the bound envelope existed. Kept decodable for the
#: rolling upgrade; discriminated from the bound envelope by disjoint keys, so
#: the two never have to be told apart by guessing.
_DRAFT_ENVELOPE_KEYS = frozenset({"version", "attempt", "cursors"})

# Reason codes. Stable strings rather than prose: a caller routes on these, and
# ``detail`` carries the human sentence.
CURSOR_MALFORMED = "cursor_malformed"
CURSOR_TOO_LARGE = "cursor_too_large"
CURSOR_LEAF_TOO_LARGE = "cursor_leaf_too_large"
CURSOR_TOO_MANY_POSITIONS = "cursor_too_many_positions"
CURSOR_DUPLICATE_KEY = "cursor_duplicate_key"
CURSOR_UNKNOWN_FIELD = "cursor_unknown_field"
CURSOR_MISSING_FIELD = "cursor_missing_field"
CURSOR_UNSUPPORTED_VERSION = "cursor_unsupported_version"
CURSOR_UNKNOWN_KIND = "cursor_unknown_kind"
CURSOR_UNKNOWN_MODE = "cursor_unknown_mode"
CURSOR_INVALID_ATTEMPT = "cursor_invalid_attempt"
CURSOR_INVALID_RANK = "cursor_invalid_rank"
CURSOR_DUPLICATE_RANK = "cursor_duplicate_rank"
CURSOR_INVALID_POSITION = "cursor_invalid_position"
#: DR-0002's name for "this cursor cannot be resumed; ask for a fresh read".
CURSOR_RESET_REQUIRED = "cursor_reset_required"
CURSOR_ATTEMPT_CHANGED = "cursor_attempt_changed"
CURSOR_TOPOLOGY_CHANGED = "cursor_topology_changed"
CURSOR_KIND_MISMATCH = "cursor_kind_mismatch"
CURSOR_MODE_MISMATCH = "cursor_mode_mismatch"
CURSOR_WORKLOAD_MISMATCH = "cursor_workload_mismatch"
CURSOR_UNKNOWN_RANK = "cursor_unknown_rank"


@dataclass(frozen=True, slots=True)
class CursorPosition:
    """One backend position, addressed by its global rank.

    Attributes:
        rank: Global rank inside the attempt's flattened rank space.
        cursor: The issuing backend's opaque position. Never parsed.
        cell: Diagnostic source cell identity; empty when the cursor is unbound.
        local: Diagnostic rank inside that cell; ``None`` when unbound.
    """

    rank: int
    cursor: str
    cell: str = ""
    local: int | None = None


@dataclass(frozen=True, slots=True)
class CursorContext:
    """The read a cursor is being decoded *for*.

    A cursor is opaque but neither authenticated nor confidential, so the read
    it is served against is the only thing that can decide whether it addresses
    this journal. Empty/``None`` fields mean "do not constrain".

    Attributes:
        kind: Journal kind being read.
        mode: ``"live"`` or ``"archive"``.
        attempt: Attempt being served; a cursor from another attempt names lines
            in journals that no longer exist.
        workload: Workload the read belongs to.
        topology: Digest of the attempt topology the rank space was derived from.
        ranks: Expected global rank set. Also the reversibility witness that
            makes a multi-rank legacy cursor decodable at all.
    """

    kind: str = CURSOR_KIND_WORKLOAD_LOG
    mode: str = CURSOR_MODE_LIVE
    attempt: int | None = None
    workload: str = ""
    topology: str = ""
    ranks: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class CursorFresh:
    """Empty input: this read starts at the beginning.

    The *only* outcome that means "start". A malformed cursor decoding to this
    is the fail-open defect DR-0002 exists to close.
    """


@dataclass(frozen=True, slots=True)
class CursorResume:
    """A decoded cursor that addresses the read being served.

    Attributes:
        positions: One backend position per rank the cursor carried.
        attempt: Attempt the positions came from, ``None`` when unstated.
        version: ``1`` for a legacy cursor, ``2`` for either v2 envelope.
        bound: True when the cursor carried its own kind/mode context rather
            than inheriting the reader's assumption.
        kind: Kind declared by the cursor, empty when unbound.
        mode: Mode declared by the cursor, empty when unbound.
        workload: Workload declared by the cursor, empty when unbound.
        topology: Topology digest declared by the cursor, empty when unbound.
        missing_ranks: Expected ranks the cursor carried no position for. Those
            ranks resume from the start of their journal; naming them keeps that
            visible instead of presenting a partial resume as a whole one.
    """

    positions: tuple[CursorPosition, ...] = ()
    attempt: int | None = None
    version: int = _LOG_CURSOR_VERSION
    bound: bool = False
    kind: str = ""
    mode: str = ""
    workload: str = ""
    topology: str = ""
    missing_ranks: tuple[int, ...] = ()

    @property
    def cursors(self) -> dict[int, str]:
        """Return the positions as the ``{rank: backend cursor}`` fan-out map."""

        return {position.rank: position.cursor for position in self.positions}


@dataclass(frozen=True, slots=True)
class CursorInvalid:
    """The bytes are not a well-formed cursor of ours.

    Attributes:
        reason: Stable machine-readable code.
        detail: One human sentence naming what failed.
    """

    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CursorIncompatible:
    """A well-formed cursor that addresses a different read.

    Distinct from :class:`CursorInvalid` because the caller's remedy differs: an
    incompatible cursor means the journal moved on (retry, topology change,
    archive handover), not that the client corrupted anything.

    Attributes:
        reason: Stable machine-readable code.
        detail: One human sentence naming the mismatch.
    """

    reason: str
    detail: str = ""


#: Every decode ends in exactly one of these. There is no fallthrough.
CursorOutcome = CursorFresh | CursorResume | CursorInvalid | CursorIncompatible


class _DuplicateCursorKey(ValueError):
    """Raised by the JSON hook when an object repeats a key."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing the last-wins collapse ``json`` performs.

    ``{"attempt": 1, "attempt": 9}`` is valid JSON and decodes to ``9`` under the
    default hook, so a cursor can carry one attempt past a proxy and a different
    one into the decoder.

    Args:
        pairs: Key/value pairs in document order.

    Returns:
        The object as a mapping.

    Raises:
        _DuplicateCursorKey: If any key appears more than once.
    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateCursorKey(key)
        result[key] = value
    return result


def _is_rank(value: object) -> bool:
    """Return whether ``value`` is a usable rank (``bool`` is not an ``int`` here)."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _leaf_too_large(cursor: str) -> bool:
    """Return whether one backend position exceeds the per-leaf byte ceiling."""

    try:
        return len(cursor.encode("utf-8")) > _LOG_CURSOR_MAX_VALUE_BYTES
    except UnicodeEncodeError:
        return True


def decode_log_cursor(value: str, *, expected: CursorContext | None = None) -> CursorOutcome:
    """Decode a log cursor into an explicit outcome.

    The contract that matters: **only empty input decodes to**
    :class:`CursorFresh`. Every other failure is named. A decoder that answers
    "start from the beginning" for a truncated, oversized, unknown-version or
    foreign cursor is indistinguishable from a first read, so a client whose
    cursor was corrupted silently re-reads the whole journal and never learns
    why.

    Three wire shapes are accepted, discriminated without guessing:

    * the DR-0002 bound envelope (``{"v": 2, "kind": ..., "positions": [...]}``),
    * the draft envelope still emitted today (``{"version": 2, "cursors": [...]}``),
      whose keys are disjoint from the bound one,
    * the pre-envelope delimiter grammar, accepted only where it is reversible.

    Args:
        value: The cursor exactly as the client returned it.
        expected: The read being served. ``None`` decodes structure only and
            performs no binding checks, which is appropriate for a nested
            fan-out level whose parent already validated the context.

    Returns:
        A :data:`CursorOutcome`.
    """

    return _decode_log_cursor(value, expected=expected, require_rank_witness=True)


def _decode_log_cursor(
    value: str,
    *,
    expected: CursorContext | None,
    require_rank_witness: bool,
) -> CursorOutcome:
    """Run the decode, parameterized by how strictly legacy cursors are read."""

    if not isinstance(value, str):
        return CursorInvalid(CURSOR_MALFORMED, "a log cursor must be a string")
    if not value:
        return CursorFresh()
    # Character count first: a hostile non-ASCII value is rejected without
    # allocating up to four encoded bytes per code point.
    if len(value) > _LOG_CURSOR_MAX_BYTES:
        return CursorInvalid(CURSOR_TOO_LARGE, "log cursor exceeds the size limit")
    try:
        if len(value.encode("utf-8")) > _LOG_CURSOR_MAX_BYTES:
            return CursorInvalid(CURSOR_TOO_LARGE, "log cursor exceeds the size limit")
    except UnicodeEncodeError:
        return CursorInvalid(CURSOR_MALFORMED, "log cursor is not encodable as UTF-8")
    if value.startswith(_LOG_CURSOR_PREFIX):
        outcome = _decode_v2_log_cursor(value.removeprefix(_LOG_CURSOR_PREFIX))
    else:
        outcome = _decode_v1_log_cursor(value, expected=expected, require_rank_witness=require_rank_witness)
    if not isinstance(outcome, CursorResume):
        return outcome
    return _bind(outcome, expected)


def _decode_v2_log_cursor(encoded: str) -> CursorOutcome:
    """Decode either base64url JSON envelope into positions."""

    if len(encoded) > _LOG_CURSOR_MAX_BYTES - len(_LOG_CURSOR_PREFIX):
        return CursorInvalid(CURSOR_TOO_LARGE, "log cursor exceeds the size limit")
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor is not valid base64url")
    if len(raw) > _LOG_CURSOR_MAX_BYTES:
        return CursorInvalid(CURSOR_TOO_LARGE, "log cursor exceeds the size limit")
    try:
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateCursorKey as exc:
        return CursorInvalid(CURSOR_DUPLICATE_KEY, f"log cursor repeats the key {exc.args[0]!r}")
    except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor does not contain a JSON document")
    if not isinstance(document, dict):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor document is not an object")
    strict = "v" in document
    draft = "version" in document
    if strict and draft:
        return CursorInvalid(CURSOR_MALFORMED, "log cursor declares two schema versions at once")
    if strict:
        return _decode_bound_envelope(document)
    if draft:
        return _decode_draft_envelope(document)
    return CursorInvalid(CURSOR_UNSUPPORTED_VERSION, "log cursor document declares no version")


def _check_keys(document: dict[str, object], allowed: frozenset[str], required: frozenset[str]) -> CursorInvalid | None:
    """Enforce a closed key set.

    Args:
        document: The decoded JSON object.
        allowed: Every key the schema defines.
        required: The subset that must be present.

    Returns:
        The refusal, or ``None`` when the key set is exact.
    """

    unknown = sorted(set(document) - allowed)
    if unknown:
        return CursorInvalid(CURSOR_UNKNOWN_FIELD, f"log cursor carries unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(document))
    if missing:
        return CursorInvalid(CURSOR_MISSING_FIELD, f"log cursor is missing field(s): {', '.join(missing)}")
    return None


def _decode_bound_envelope(document: dict[str, object]) -> CursorOutcome:
    """Decode the DR-0002 envelope: closed key set, one position per rank."""

    failure = _check_keys(document, _STRICT_ENVELOPE_KEYS, _STRICT_ENVELOPE_REQUIRED)
    if failure is not None:
        return failure
    version = document["v"]
    # ``2.0 == 2`` and ``True == 1`` in Python, so identity of type is part of
    # the version check rather than an afterthought.
    if not isinstance(version, int) or isinstance(version, bool) or version != _LOG_CURSOR_VERSION:
        return CursorInvalid(CURSOR_UNSUPPORTED_VERSION, f"log cursor version {version!r} is not supported")
    kind = document["kind"]
    if not isinstance(kind, str) or kind not in _KNOWN_CURSOR_KINDS:
        return CursorInvalid(CURSOR_UNKNOWN_KIND, f"log cursor kind {kind!r} is not a known journal kind")
    mode = document["mode"]
    if not isinstance(mode, str) or mode not in _KNOWN_CURSOR_MODES:
        return CursorInvalid(CURSOR_UNKNOWN_MODE, f"log cursor mode {mode!r} is not a known read mode")
    attempt = document["attempt"]
    if not _is_rank(attempt):
        return CursorInvalid(CURSOR_INVALID_ATTEMPT, "log cursor attempt must be a non-negative integer")
    topology = document.get("topology", "")
    workload = document.get("workload", "")
    if not isinstance(topology, str) or not isinstance(workload, str):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor topology and workload must be strings")
    entries = document["positions"]
    if not isinstance(entries, list):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor positions must be a list")
    if len(entries) > _LOG_CURSOR_MAX_ENTRIES:
        return CursorInvalid(CURSOR_TOO_MANY_POSITIONS, "log cursor exceeds the entry limit")
    positions: list[CursorPosition] = []
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return CursorInvalid(CURSOR_INVALID_POSITION, "log cursor position is not an object")
        failure = _check_keys(entry, _STRICT_POSITION_KEYS, _STRICT_POSITION_REQUIRED)
        if failure is not None:
            return failure
        rank = entry["rank"]
        if not _is_rank(rank):
            return CursorInvalid(CURSOR_INVALID_RANK, f"log cursor rank {rank!r} is not a non-negative integer")
        if rank in seen:
            return CursorInvalid(CURSOR_DUPLICATE_RANK, f"log cursor names rank {rank} twice")
        cursor = entry["cursor"]
        if not isinstance(cursor, str) or not cursor:
            return CursorInvalid(CURSOR_INVALID_POSITION, f"log cursor for rank {rank} is not a non-empty string")
        if _leaf_too_large(cursor):
            return CursorInvalid(CURSOR_LEAF_TOO_LARGE, f"log cursor for rank {rank} exceeds the leaf size limit")
        cell = entry.get("cell", "")
        local = entry.get("local")
        if not isinstance(cell, str):
            return CursorInvalid(CURSOR_INVALID_POSITION, f"log cursor cell for rank {rank} is not a string")
        if local is not None and not _is_rank(local):
            return CursorInvalid(CURSOR_INVALID_RANK, f"log cursor local rank for rank {rank} is not valid")
        seen.add(rank)
        positions.append(CursorPosition(rank=rank, cursor=cursor, cell=cell, local=local))
    return CursorResume(
        positions=tuple(positions),
        attempt=attempt,
        version=_LOG_CURSOR_VERSION,
        bound=True,
        kind=kind,
        mode=mode,
        workload=workload,
        topology=topology,
    )


def _decode_draft_envelope(document: dict[str, object]) -> CursorOutcome:
    """Decode the ``[[rank, cursor], ...]`` envelope emitted before the binding.

    Structurally as strict as the bound one; it simply carries no context, so
    the resume it produces is reported as unbound rather than pretending the
    read it names was checked.
    """

    failure = _check_keys(document, _DRAFT_ENVELOPE_KEYS, _DRAFT_ENVELOPE_KEYS)
    if failure is not None:
        return failure
    version = document["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != _LOG_CURSOR_VERSION:
        return CursorInvalid(CURSOR_UNSUPPORTED_VERSION, f"log cursor version {version!r} is not supported")
    attempt = document["attempt"]
    if not _is_rank(attempt):
        return CursorInvalid(CURSOR_INVALID_ATTEMPT, "log cursor attempt must be a non-negative integer")
    entries = document["cursors"]
    if not isinstance(entries, list):
        return CursorInvalid(CURSOR_MALFORMED, "log cursor entries must be a list")
    if len(entries) > _LOG_CURSOR_MAX_ENTRIES:
        return CursorInvalid(CURSOR_TOO_MANY_POSITIONS, "log cursor exceeds the entry limit")
    positions: list[CursorPosition] = []
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 2:
            return CursorInvalid(CURSOR_INVALID_POSITION, "log cursor entry is not a [rank, cursor] pair")
        rank, cursor = entry
        if not _is_rank(rank):
            return CursorInvalid(CURSOR_INVALID_RANK, f"log cursor rank {rank!r} is not a non-negative integer")
        if rank in seen:
            return CursorInvalid(CURSOR_DUPLICATE_RANK, f"log cursor names rank {rank} twice")
        if not isinstance(cursor, str) or not cursor:
            return CursorInvalid(CURSOR_INVALID_POSITION, f"log cursor for rank {rank} is not a non-empty string")
        if _leaf_too_large(cursor):
            return CursorInvalid(CURSOR_LEAF_TOO_LARGE, f"log cursor for rank {rank} exceeds the leaf size limit")
        seen.add(rank)
        positions.append(CursorPosition(rank=rank, cursor=cursor))
    return CursorResume(positions=tuple(positions), attempt=attempt, version=_LOG_CURSOR_VERSION, bound=False)


def _decode_v1_log_cursor(
    value: str,
    *,
    expected: CursorContext | None,
    require_rank_witness: bool = True,
) -> CursorOutcome:
    """Decode the pre-envelope grammar, only where it is reversible.

    Two shapes existed. ``"{attempt}:{cursor}"`` has exactly one reading and is
    always accepted -- this is what makes a journald cursor such as ``s=...``
    work. ``"{attempt}:{rank}={cursor}|..."`` nests an unescaped delimiter
    grammar inside opaque leaves, so ``"3:0=a|1=b"`` is equally readable as one
    rank whose cursor is ``a|1=b``. It is decoded only when the caller supplies
    the rank set that makes the split reversible; otherwise DR-0002 requires
    ``cursor_reset_required`` rather than a guess.

    Args:
        value: The cursor as received.
        expected: The read being served; its ``ranks`` are the reversibility
            witness for a multi-rank legacy cursor.
        require_rank_witness: Demand that witness. The pre-outcome
            :func:`split_log_cursor` shim clears it so an upgrade does not turn
            every retained legacy cursor into a silent restart at a call site
            that cannot report the refusal.
    """

    attempt_text, separator, remainder = value.partition(":")
    if not separator:
        return CursorInvalid(CURSOR_MALFORMED, "log cursor carries no attempt")
    if not _CANONICAL_DECIMAL.match(attempt_text):
        return CursorInvalid(CURSOR_INVALID_ATTEMPT, f"log cursor attempt {attempt_text!r} is not a decimal integer")
    try:
        attempt = int(attempt_text)
    except ValueError:
        return CursorInvalid(CURSOR_INVALID_ATTEMPT, "log cursor attempt exceeds the decimal conversion limit")
    if not remainder:
        return CursorResume(attempt=attempt, version=1)
    if not _LEGACY_RANK_PREFIX.match(remainder):
        if _leaf_too_large(remainder):
            return CursorInvalid(CURSOR_LEAF_TOO_LARGE, "log cursor exceeds the leaf size limit")
        return CursorResume(positions=(CursorPosition(rank=0, cursor=remainder),), attempt=attempt, version=1)
    pieces = remainder.split("|")
    if len(pieces) > _LOG_CURSOR_MAX_ENTRIES:
        return CursorInvalid(CURSOR_TOO_MANY_POSITIONS, "log cursor exceeds the entry limit")
    positions: list[CursorPosition] = []
    seen: set[int] = set()
    for piece in pieces:
        rank_text, assigned, cursor = piece.partition("=")
        if not assigned or not cursor or not _CANONICAL_DECIMAL.match(rank_text):
            return CursorInvalid(CURSOR_RESET_REQUIRED, f"legacy log cursor piece {piece!r} is not rank=cursor")
        if _LEGACY_NESTED_LEAF.match(cursor) or cursor.startswith(_LOG_CURSOR_PREFIX):
            return CursorInvalid(CURSOR_RESET_REQUIRED, f"legacy log cursor nests a cursor at rank {rank_text}")
        if _leaf_too_large(cursor):
            return CursorInvalid(CURSOR_LEAF_TOO_LARGE, "log cursor exceeds the leaf size limit")
        try:
            rank = int(rank_text)
        except ValueError:
            return CursorInvalid(CURSOR_INVALID_RANK, "log cursor rank exceeds the decimal conversion limit")
        if rank in seen:
            return CursorInvalid(CURSOR_DUPLICATE_RANK, f"legacy log cursor names rank {rank} twice")
        seen.add(rank)
        positions.append(CursorPosition(rank=rank, cursor=cursor))
    if len(positions) > 1:
        if expected is None or expected.ranks is None:
            if require_rank_witness:
                return CursorInvalid(
                    CURSOR_RESET_REQUIRED,
                    "a multi-rank legacy log cursor is reversible only against a known rank set",
                )
        elif seen != set(expected.ranks):
            return CursorIncompatible(
                CURSOR_RESET_REQUIRED,
                "legacy log cursor rank set does not match this attempt's ranks",
            )
    return CursorResume(positions=tuple(positions), attempt=attempt, version=1)


def _bind(resume: CursorResume, expected: CursorContext | None) -> CursorOutcome:
    """Check a decoded cursor against the read it is being served for.

    Base64 opacity is not authorization and not addressing. A cursor minted for
    another workload, attempt, topology, kind or mode is structurally perfect and
    still names positions in a journal this read is not looking at.
    """

    if expected is None:
        return resume
    if resume.kind and expected.kind and resume.kind != expected.kind:
        return CursorIncompatible(CURSOR_KIND_MISMATCH, f"log cursor was minted for kind {resume.kind!r}")
    if resume.mode and expected.mode and resume.mode != expected.mode:
        return CursorIncompatible(CURSOR_MODE_MISMATCH, f"log cursor was minted for the {resume.mode!r} read mode")
    if resume.workload and expected.workload and resume.workload != expected.workload:
        return CursorIncompatible(CURSOR_WORKLOAD_MISMATCH, f"log cursor was minted for workload {resume.workload!r}")
    if resume.topology and expected.topology and resume.topology != expected.topology:
        return CursorIncompatible(CURSOR_TOPOLOGY_CHANGED, "log cursor was minted for a different attempt topology")
    if resume.attempt is not None and expected.attempt is not None and resume.attempt != expected.attempt:
        return CursorIncompatible(CURSOR_ATTEMPT_CHANGED, f"log cursor belongs to attempt {resume.attempt}")
    if expected.ranks is None:
        return resume
    allowed = set(expected.ranks)
    unknown = sorted({position.rank for position in resume.positions} - allowed)
    if unknown:
        return CursorIncompatible(
            CURSOR_UNKNOWN_RANK,
            f"log cursor names rank(s) outside this attempt: {', '.join(str(rank) for rank in unknown)}",
        )
    missing = tuple(sorted(allowed - {position.rank for position in resume.positions}))
    return CursorResume(
        positions=resume.positions,
        attempt=resume.attempt,
        version=resume.version,
        bound=resume.bound,
        kind=resume.kind,
        mode=resume.mode,
        workload=resume.workload,
        topology=resume.topology,
        missing_ranks=missing,
    )


def split_log_cursor(value: str) -> tuple[int | None, dict[int, str]]:
    """Decode a cursor, collapsing every refusal into "start from the beginning".

    Compatibility shim for call sites that predate :data:`CursorOutcome` and have
    nowhere to put a refusal. It is *fail-open by construction*: an invalid or
    incompatible cursor is reported as ``(None, {})``, which the caller cannot
    tell apart from a first read. That is precisely the behaviour DR-0002 rules
    out, so new code -- and every path that answers a client -- must call
    :func:`decode_log_cursor` and act on the outcome.

    Because it cannot report a refusal, this shim also keeps reading a
    well-formed multi-rank legacy cursor without the rank-set witness
    :func:`decode_log_cursor` demands. Tightening it here would silently restart
    every retained legacy cursor mid-upgrade, which trades a wrong-position risk
    for a guaranteed duplicate-output one.

    Args:
        value: The cursor exactly as the client returned it.

    Returns:
        The attempt (``None`` when unstated or undecodable) and the per-rank
        backend positions.
    """

    outcome = _decode_log_cursor(value, expected=None, require_rank_witness=False)
    if isinstance(outcome, CursorResume):
        return outcome.attempt, outcome.cursors
    return None, {}


def log_cursor_v2_emission_enabled(environ: tp.Mapping[str, str] | None = None) -> bool:
    """Return whether the fully bound v2 envelope may be emitted.

    Off unless explicitly enabled. Stage one of the DR-0002 rollout ships the
    decoder everywhere while emission keeps the draft shape, so no serving or
    rollback image can meet a payload it cannot read.

    Args:
        environ: Environment to read; defaults to the process environment.

    Returns:
        True when emission of the bound envelope is enabled.
    """

    source = os.environ if environ is None else environ
    return source.get(LOG_CURSOR_V2_EMISSION_ENV, "").strip().lower() in _TRUTHY


def _validated_positions(
    positions: tp.Mapping[int, str] | tp.Sequence[CursorPosition],
) -> list[CursorPosition]:
    """Normalize and bound an emission's positions.

    Raises:
        ValueError: If a rank, cursor, or the total size is unusable.
    """

    items = (
        [CursorPosition(rank=rank, cursor=cursor) for rank, cursor in positions.items()]
        if isinstance(positions, tp.Mapping)
        else list(positions)
    )
    live: list[CursorPosition] = []
    total = 0
    for item in items:
        if isinstance(item.rank, bool) or not isinstance(item.rank, int) or item.rank < 0:
            raise ValueError("structured log cursor ranks must be non-negative integers")
        if not isinstance(item.cursor, str):
            raise ValueError("structured log backend cursors must be strings")
        if not item.cursor:
            continue
        try:
            size = len(item.cursor.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("structured log backend cursors must be valid UTF-8") from exc
        if size > _LOG_CURSOR_MAX_VALUE_BYTES:
            raise ValueError("structured log backend cursor exceeds the size limit")
        total += size
        if total > _LOG_CURSOR_MAX_BYTES:
            raise ValueError("structured log cursor exceeds the size limit")
        live.append(item)
    if len({item.rank for item in live}) != len(live):
        raise ValueError("structured log cursor ranks must be unique")
    if len(live) > _LOG_CURSOR_MAX_ENTRIES:
        raise ValueError("structured log cursor exceeds the entry limit")
    live.sort(key=lambda item: item.rank)
    return live


def _encode_envelope(document: dict[str, object]) -> str:
    raw = json.dumps(document, separators=(",", ":"), sort_keys=False).encode()
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    result = f"{_LOG_CURSOR_PREFIX}{encoded}"
    if len(result) > _LOG_CURSOR_MAX_BYTES:
        raise ValueError("structured log cursor exceeds the size limit")
    return result


def encode_log_cursor(
    positions: tp.Mapping[int, str] | tp.Sequence[CursorPosition],
    *,
    context: CursorContext | None = None,
    bound: bool | None = None,
) -> str:
    """Encode every backend position in a deterministic opaque cursor.

    Empty when no rank reported one, which is distinct from a cursor pointing at
    the start: resuming from empty means "wherever you were", and a start cursor
    means "replay everything".

    Args:
        positions: Backend positions, as ``{rank: cursor}`` or as
            :class:`CursorPosition` records carrying cell/local diagnostics.
        context: The read these positions belong to. Required for the bound
            envelope, which records it.
        bound: Emit the bound DR-0002 envelope. ``None`` consults
            :func:`log_cursor_v2_emission_enabled`, which is off by default.

    Returns:
        The opaque cursor, or ``""`` when no rank reported a position.

    Raises:
        ValueError: If a rank, cursor, attempt, or the total size is unusable.
    """

    live = _validated_positions(positions)
    if not live:
        return ""
    context = context or CursorContext()
    if bound is None:
        bound = log_cursor_v2_emission_enabled()
    if not bound:
        return _encode_envelope(
            {
                "version": _LOG_CURSOR_VERSION,
                "attempt": context.attempt or 0,
                "cursors": [[item.rank, item.cursor] for item in live],
            }
        )
    if context.attempt is None:
        raise ValueError("a bound structured log cursor requires an attempt")
    if context.kind not in _KNOWN_CURSOR_KINDS:
        raise ValueError(f"unknown structured log cursor kind {context.kind!r}")
    if context.mode not in _KNOWN_CURSOR_MODES:
        raise ValueError(f"unknown structured log cursor mode {context.mode!r}")
    return _encode_envelope(
        {
            "v": _LOG_CURSOR_VERSION,
            "kind": context.kind,
            "attempt": context.attempt,
            "topology": context.topology,
            "mode": context.mode,
            "workload": context.workload,
            "positions": [
                {"rank": item.rank, "cursor": item.cursor}
                | ({"cell": item.cell} if item.cell else {})
                | ({"local": item.local} if item.local is not None else {})
                for item in live
            ],
        }
    )


def join_log_cursor(attempt: int, cursors: tp.Mapping[int, str]) -> str:
    """Encode every backend position for ``attempt`` in an opaque v2 cursor.

    Kept as the positional entry point the fan-out executors already call. Which
    envelope it writes is decided by :func:`log_cursor_v2_emission_enabled`, so
    the rollout flips in one place rather than at every call site.

    Args:
        attempt: Attempt the positions were read from.
        cursors: ``{rank: backend cursor}``; empty cursors are dropped.

    Returns:
        The opaque cursor, or ``""`` when no rank reported a position.

    Raises:
        ValueError: If the attempt, a rank, a cursor, or the total size is unusable.
    """

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("structured log cursor attempt must be a non-negative integer")
    return encode_log_cursor(cursors, context=CursorContext(attempt=attempt))


# --------------------------------------------------------------------------- archive positions
#
# A live position addresses a backend journal; an archive position addresses an
# immutable object. The handover between them is where a follower that outlives
# its cell either continues or starts printing the whole archive again on every
# poll, so the two are never conflated: a live position is *translated* into an
# archive offset by finding the line it names, and a position that cannot be
# translated is a named gap rather than a rewind to zero.

#: Marks a position as an offset into an archived rank object: how many of that
#: rank's archived lines the follower has already been served.
ARCHIVE_OFFSET_PREFIX = "eyvan-archive-offset:"
#: Marks a source position that could not be placed in the archive, retained
#: verbatim behind the tag. Dropping it would present the next poll as a first
#: read and replay the whole object; rewriting it as an offset would invent a
#: place for it. Kept, so the same gap is reported again instead of either.
ARCHIVE_SOURCE_PREFIX = "eyvan-archive-source:"

#: Stable reasons one rank's archived continuation could not be served. Routed
#: on by callers; the detail carries the human sentence.
ARCHIVE_POSITION_UNKNOWN = "archive_position_unknown"
ARCHIVE_POSITION_AMBIGUOUS = "archive_position_ambiguous"
ARCHIVE_POSITION_BEYOND_ARCHIVE = "archive_position_beyond_archive"
ARCHIVE_POSITION_MALFORMED = "archive_position_malformed"


@dataclass(frozen=True, slots=True)
class ArchiveOffset:
    """Where one rank's archived replay resumes.

    Attributes:
        rank: Global rank inside the attempt's flattened rank space.
        offset: Lines of that rank's object already served. ``0`` is a first
            read; ``len(object)`` is a follower that is fully caught up.
    """

    rank: int
    offset: int


@dataclass(frozen=True, slots=True)
class ArchiveGap:
    """Why one rank's archived continuation cannot be served.

    An explicit outcome rather than an offset of zero. The two are opposites: a
    zero offset replays the rank from its first line, which is precisely the
    duplicate output the translation exists to prevent, and it is
    indistinguishable from a first read to the follower receiving it.

    Attributes:
        rank: Global rank whose position could not be placed.
        reason: Stable machine-readable code.
        detail: One human sentence naming what failed.
        position: The source position, retained so the next poll reports the
            same gap instead of restarting the object.
    """

    rank: int
    reason: str
    detail: str
    position: str = ""


ArchivePlacement = ArchiveOffset | ArchiveGap


def encode_archive_offset(offset: int) -> str:
    """Return the tagged position meaning "``offset`` lines already served".

    Args:
        offset: Lines of one rank's archived object already served.

    Returns:
        The tagged position.

    Raises:
        ValueError: If ``offset`` is not a non-negative integer.
    """

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("an archive offset must be a non-negative integer")
    return f"{ARCHIVE_OFFSET_PREFIX}{offset}"


def encode_unplaced_archive_position(position: str) -> str:
    """Return ``position`` tagged as a source position the archive could not place."""

    if not isinstance(position, str):
        raise ValueError("an archive source position must be a string")
    if position.startswith(ARCHIVE_SOURCE_PREFIX):
        return position
    return f"{ARCHIVE_SOURCE_PREFIX}{position}"


def is_archive_position(value: object) -> bool:
    """Return whether ``value`` was minted by an archive read.

    The tag is on the value, not only on the envelope, because the envelope
    emitted by default carries no ``mode`` field. Without it a reader meeting a
    bare integer could not tell an archive offset from a backend cursor that
    happens to be numeric, and would hand one to the other's reader.
    """

    return isinstance(value, str) and value.startswith((ARCHIVE_OFFSET_PREFIX, ARCHIVE_SOURCE_PREFIX))


def resolve_archive_position(
    position: str,
    line_cursors: tp.Sequence[str],
    *,
    rank: int,
) -> ArchivePlacement:
    """Place one presented position inside one rank's archived object.

    Three inputs are accepted and told apart without guessing: empty (a first
    read), a tagged archive offset (a follower already reading the archive), and
    a backend cursor (a follower crossing the live-to-archive boundary). The last
    is translated by finding the archived line that carries exactly that cursor,
    which is what makes the handover continue rather than restart -- every
    archived record keeps the position its backend issued for it.

    A position that matches no line, or more than one, is a
    :class:`ArchiveGap`. Both are honestly unknown: the first cannot be placed at
    all, and the second names two different places in the same object. Resolving
    either to an offset would silently duplicate or silently drop lines, and the
    receiver could not tell which.

    Args:
        position: The presented position for this rank.
        line_cursors: The backend cursor of each archived line, in file order.
        rank: Global rank being placed, carried into the outcome.

    Returns:
        The offset to resume from, or the gap that explains why there is none.
    """

    if not position:
        return ArchiveOffset(rank=rank, offset=0)
    if position.startswith(ARCHIVE_OFFSET_PREFIX):
        text = position.removeprefix(ARCHIVE_OFFSET_PREFIX)
        if not _CANONICAL_DECIMAL.match(text):
            return ArchiveGap(
                rank=rank,
                reason=ARCHIVE_POSITION_MALFORMED,
                detail=f"archive offset {text!r} for rank {rank} is not a decimal integer",
                position=position,
            )
        try:
            offset = int(text)
        except ValueError:
            return ArchiveGap(
                rank=rank,
                reason=ARCHIVE_POSITION_MALFORMED,
                detail=f"archive offset for rank {rank} exceeds the decimal conversion limit",
                position=position,
            )
        if offset > len(line_cursors):
            return ArchiveGap(
                rank=rank,
                reason=ARCHIVE_POSITION_BEYOND_ARCHIVE,
                detail=(
                    f"rank {rank}'s archive holds {len(line_cursors)} line(s), "
                    f"fewer than the {offset} this position claims were already served"
                ),
                position=position,
            )
        return ArchiveOffset(rank=rank, offset=offset)
    source = position.removeprefix(ARCHIVE_SOURCE_PREFIX)
    matches = [index for index, cursor in enumerate(line_cursors) if cursor and cursor == source]
    if not matches:
        return ArchiveGap(
            rank=rank,
            reason=ARCHIVE_POSITION_UNKNOWN,
            detail=f"no archived line for rank {rank} carries the presented source position",
            position=source,
        )
    if len(matches) > 1:
        return ArchiveGap(
            rank=rank,
            reason=ARCHIVE_POSITION_AMBIGUOUS,
            detail=f"rank {rank}'s archive carries the presented source position {len(matches)} times",
            position=source,
        )
    return ArchiveOffset(rank=rank, offset=matches[0] + 1)


def _stream_from_wire(value: object) -> LogStream:
    """Accept a stream name the sender already decoded.

    Falls back to the priority mapping so a sender that forwards a raw journald
    PRIORITY is still understood; the two are distinguishable because the stream
    names are a closed set.
    """

    if value in ("stdout", "stderr", "unknown"):
        return tp.cast("LogStream", value)
    return stream_for_priority(value if isinstance(value, (str, int)) else None)


def log_page_from_json(payload: object) -> LogPage:
    """Rebuild a page from the wire shape ``LogPage`` writes.

    The inverse of ``as_json`` and the route's own document, kept beside them so
    the two cannot drift: a remote driver that decoded these fields itself would
    be a second spelling of the protocol.

    A malformed payload becomes an ``unavailable`` page rather than an
    exception, for the same reason ``log_page`` returns one: a follower polling
    every second should not have to catch an exception to learn a host answered
    badly, and "unavailable" is the honest description of what it got. A missing
    records list or unreadable record (including a missing/non-string message)
    refuses the whole page with no next cursor, even if some entries are valid:
    accepting its cursor could skip unreadable output on the next poll.
    Timezone-less timestamps retain their messages but become unknown time.

    An unusable ``rank`` is preserved as :data:`INVALID_LOG_RANK` rather than
    coerced to ``0``, and a malformed ``truncated_ranks`` entry is preserved
    rather than filtered out. Both normalisations produced the same failure: the
    controller validates each rank against the cell's durable process width, so
    a rank that was silently rewritten to zero -- or a truncation notice that was
    silently dropped -- made a partial page look complete. An absent ``rank``
    key is still the documented default of ``0``; only a *declared* rank that is
    not a non-negative integer is invalid.
    """

    if not isinstance(payload, dict):
        return LogPage(unavailable=True, detail="native worker returned a non-object log page")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return LogPage(unavailable=True, detail="native worker returned a log page without a records list")
    records: list[LogRecord] = []
    for item in raw_records:
        if not isinstance(item, dict) or not isinstance(item.get("message"), str):
            # The page cursor covers every entry, including an unreadable one.
            # Refuse the whole page so a corrected retry cannot skip that entry.
            return LogPage(unavailable=True, detail="native worker returned a malformed log record")
        stamp = item.get("timestamp")
        parsed: datetime | None = None
        if isinstance(stamp, str) and stamp:
            try:
                parsed = datetime.fromisoformat(stamp)
                # A timezone-less value has unknown time provenance. Assigning
                # local/UTC time would invent it, and comparing it with aware
                # records would make an otherwise readable page unmergeable.
                if parsed.utcoffset() is None:
                    parsed = None
            except ValueError:
                parsed = None
        rank = item.get("rank", 0)
        pid = item.get("pid")
        records.append(
            LogRecord(
                message=str(item.get("message", "")),
                # Already decoded by the sender. Feeding it back through
                # ``stream_for_priority`` -- which expects a journald PRIORITY --
                # silently turned every 'stdout' and 'stderr' into 'unknown', so
                # the one field structured logs add over plain text was empty on
                # every remote read.
                stream=_stream_from_wire(item.get("stream")),
                timestamp=parsed,
                cursor=str(item.get("cursor", "")),
                rank=rank if _is_rank(rank) else INVALID_LOG_RANK,
                pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                host=item.get("host") if isinstance(item.get("host"), str) else None,
            )
        )
    truncated = payload.get("truncated_ranks")
    if truncated is None:
        truncated_ranks: tuple[int, ...] = ()
    elif isinstance(truncated, list):
        truncated_ranks = tuple(rank if _is_rank(rank) else INVALID_LOG_RANK for rank in truncated)
    else:
        # A non-list truncation field is not "nothing was truncated". Reporting
        # it as empty is how a half-read cell is presented as a whole one.
        truncated_ranks = (INVALID_LOG_RANK,)
    return LogPage(
        records=tuple(records),
        next_cursor=str(payload.get("next_cursor", "")),
        unavailable=bool(payload.get("unavailable")),
        detail=str(payload.get("detail", "")),
        truncated_ranks=truncated_ranks,
    )


def stream_for_priority(priority: str | int | None) -> LogStream:
    """Map a journald PRIORITY to a stream name.

    Returns ``"unknown"`` for anything unrecognized. Guessing ``stdout`` would
    make an operator debugging a crash read stderr lines as ordinary output.
    """

    if priority is None:
        return "unknown"
    text = str(priority).strip()
    if text in _STDERR_PRIORITIES:
        return "stderr"
    if text in _STDOUT_PRIORITIES:
        return "stdout"
    return "unknown"


#: The unit supervisor writes start/stop bookkeeping into the same journal
#: the workload's output lands in. Its routine lines -- "Starting ...",
#: "Started ...", "Stopped ...", "Deactivated successfully", CPU accounting --
#: interleave with a job's real stdout and read as if the job had printed
#: them. They are transport noise, filtered at ingestion; anything systemd
#: says about a *failure* (a main process exiting non-zero, an OOM kill) does
#: not match these prefixes and is deliberately kept.
_SYSTEMD_IDENTIFIER = "systemd"
_ROUTINE_UNIT_LINES = (
    "Started ",
    "Starting ",
    "Stopped ",
    "Stopping ",
    "Finished ",
    "Deactivated successfully",
    "Consumed ",
)
#: journald usually prefixes the unit name: "eyvan-wl_<id>-a1-g1-<sha>.service:
#: Consumed 1min ...". The unit name varies, so match what follows it.
#: Per-process import noise that is identical on every rank. Kept exact-match:
#: a near-miss here hides a real failure mode someday.
_ROUTINE_PROCESS_LINES = frozenset(
    {
        "PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.",  # noqa: E501
        "please install `torch` (cpu) if you want to use `AutoEasyDeLModel*.from_torch_pretrained`",
    }
)

_UNIT_CHATTER = re.compile(r"\.service: (?:" + "|".join(re.escape(line.strip()) for line in _ROUTINE_UNIT_LINES) + ")")


#: Read-side companion: archived records carry no journald identifier, so the
#: transcript filter matches on message shape alone. Only unambiguous forms
#: qualify -- a unit-prefixed bookkeeping line, or the launcher echo for this
#: worker's own entrypoint -- never a bare "Starting ..." a job may print.
_WORKER_LAUNCH_CHATTER = re.compile(r"^(?:Starting |Started |Stopped ).*eyvan\.worker\.runtime\.callable_worker")


#: The multi-host join writes a machine-readable heartbeat to each rank's
#: stderr once the gang forms -- useful to scrapers, noise in a human
#: transcript, and it carries no facts a reader could act on ("rank=unknown").
#: The failure marker is deliberately absent here: a gang that never formed is
#: exactly what someone reading logs needs to see.
_GANG_JOINED_MARKER = "eyvan-gang-init-joined"
#: systemd answers post-teardown bookkeeping queries with this error once the
#: transient unit file is gone. It names no state a reader can act on.
_UNIT_FILE_GONE_CHATTER = ".service: Failed to open /run/systemd/transient/"


def routine_unit_chatter(message: str) -> bool:
    """True when one already-archived log line is transport noise, not output."""

    return (
        bool(_UNIT_CHATTER.search(message))
        or bool(_WORKER_LAUNCH_CHATTER.match(message))
        or message.startswith(_GANG_JOINED_MARKER)
        or _UNIT_FILE_GONE_CHATTER in message
    )


def _is_routine_unit_chatter(entry: dict[str, tp.Any]) -> bool:
    identifier = entry.get("SYSLOG_IDENTIFIER", entry.get("_COMM"))
    if identifier != _SYSTEMD_IDENTIFIER:
        return False
    message = entry.get("MESSAGE")
    message = message if isinstance(message, str) else ""
    return message.startswith(_ROUTINE_UNIT_LINES) or bool(_UNIT_CHATTER.search(message))


_LOCAL_ADDRESS_LOCK = __import__("threading").Lock()
_LOCAL_ADDRESS_CACHE: str | None | bool = False


def _local_worker_address() -> str | None:
    """This host's primary address, resolved once per process.

    A routing-table probe (connect a UDP socket toward a public address --
    no packet is sent) answers "which interface would carry traffic off this
    machine", which on a TPU VM is the internal address the rest of the
    fleet reaches it by. Ray prints exactly this in its worker prefix.
    """

    global _LOCAL_ADDRESS_CACHE
    with _LOCAL_ADDRESS_LOCK:
        if _LOCAL_ADDRESS_CACHE is not False:
            return tp.cast("str | None", _LOCAL_ADDRESS_CACHE)
        address: str | None = None
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("8.8.8.8", 80))
                address = probe.getsockname()[0]
            finally:
                probe.close()
        except OSError:
            address = None
        _LOCAL_ADDRESS_CACHE = address or None
        return tp.cast("str | None", _LOCAL_ADDRESS_CACHE)


def parse_journal_json(payload: str, *, rank: int = 0) -> LogPage:
    """Parse ``journalctl --output=json`` output into records.

    Tolerant by line: journald emits one JSON object per line, and a truncated
    final line during a live read is ordinary rather than a failure. Discarding
    the whole page for it would make ``--follow`` drop output at exactly the
    moments a job is producing the most.
    """

    records: list[LogRecord] = []
    cursor = ""
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        message = entry.get("MESSAGE")
        if isinstance(message, list):
            # journald returns a byte array for messages it considers non-UTF8.
            message = bytes(item for item in message if isinstance(item, int)).decode("utf-8", errors="replace")
        elif not isinstance(message, str):
            continue
        # Advance past it: a dropped line must never rewind the stream.
        cursor = str(entry.get("__CURSOR", cursor))
        if _is_routine_unit_chatter(entry):
            continue
        # Transformers prints the same two torch-absence lines on every rank
        # at import; four slices means eight identical announcements per run,
        # none of which ever changes. Exact-match, so a near-miss failure mode
        # still gets through.
        if message in _ROUTINE_PROCESS_LINES:
            continue
        records.append(
            LogRecord(
                message=message,
                stream=stream_for_priority(entry.get("PRIORITY")),
                timestamp=_microseconds_to_datetime(entry.get("__REALTIME_TIMESTAMP")),
                cursor=str(entry.get("__CURSOR", "")),
                rank=rank,
                pid=_optional_int(entry.get("_PID")),
                # The reachable address outranks the hostname in a prefix:
                # operators paste an ip into ssh/gcloud, not a node name.
                host=_local_worker_address()
                or (entry.get("_HOSTNAME") if isinstance(entry.get("_HOSTNAME"), str) else None),
            )
        )
    return LogPage(records=tuple(records), next_cursor=cursor)


def merge_pages(pages: tp.Mapping[int, LogPage]) -> LogPage:
    """Merge per-rank pages into one, keeping unreachable ranks visible.

    An unreachable rank is reported *in place* rather than dropped: a fan-out
    that silently returned the ranks it could reach would show a complete-looking
    log for a partially-failed slice, which is the exact failure the per-rank
    aggregation discipline exists to prevent.
    """

    records: list[LogRecord] = []
    truncated: set[int] = set()
    details: list[str] = []
    for rank in sorted(pages):
        page = pages[rank]
        page_truncated = set(page.truncated_ranks)
        truncated.update(page_truncated)
        if page.unavailable:
            # A nested fan-out (one cell containing several hosts) already names
            # every unavailable inner rank. A single-rank backend has no nested
            # list, so its mapping key remains the rank that failed.
            if not page_truncated:
                truncated.add(rank)
            if page.detail:
                details.append(page.detail if page_truncated else f"rank {rank}: {page.detail}")
            continue
        records.extend(page.records)
        if page.detail and page_truncated:
            details.append(page.detail)
    records.sort(key=lambda record: (record.timestamp or _EPOCH, record.rank))
    return LogPage(
        records=tuple(records),
        next_cursor="",
        unavailable=bool(truncated) and not records,
        detail="; ".join(details),
        truncated_ranks=tuple(sorted(truncated)),
    )


_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _microseconds_to_datetime(value: object) -> datetime | None:
    try:
        micros = int(str(value))
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(micros / 1_000_000, tz=UTC)


def _optional_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


__all__ = (
    "ARCHIVE_OFFSET_PREFIX",
    "ARCHIVE_POSITION_AMBIGUOUS",
    "ARCHIVE_POSITION_BEYOND_ARCHIVE",
    "ARCHIVE_POSITION_MALFORMED",
    "ARCHIVE_POSITION_UNKNOWN",
    "ARCHIVE_SOURCE_PREFIX",
    "CURSOR_ATTEMPT_CHANGED",
    "CURSOR_DUPLICATE_KEY",
    "CURSOR_DUPLICATE_RANK",
    "CURSOR_INVALID_ATTEMPT",
    "CURSOR_INVALID_POSITION",
    "CURSOR_INVALID_RANK",
    "CURSOR_KIND_MISMATCH",
    "CURSOR_KIND_WORKLOAD_LOG",
    "CURSOR_LEAF_TOO_LARGE",
    "CURSOR_MALFORMED",
    "CURSOR_MISSING_FIELD",
    "CURSOR_MODE_ARCHIVE",
    "CURSOR_MODE_LIVE",
    "CURSOR_MODE_MISMATCH",
    "CURSOR_RESET_REQUIRED",
    "CURSOR_TOO_LARGE",
    "CURSOR_TOO_MANY_POSITIONS",
    "CURSOR_TOPOLOGY_CHANGED",
    "CURSOR_UNKNOWN_FIELD",
    "CURSOR_UNKNOWN_KIND",
    "CURSOR_UNKNOWN_MODE",
    "CURSOR_UNKNOWN_RANK",
    "CURSOR_UNSUPPORTED_VERSION",
    "CURSOR_WORKLOAD_MISMATCH",
    "INVALID_LOG_RANK",
    "LOG_CURSOR_V2_EMISSION_ENV",
    "ArchiveGap",
    "ArchiveOffset",
    "ArchivePlacement",
    "CursorContext",
    "CursorFresh",
    "CursorIncompatible",
    "CursorInvalid",
    "CursorOutcome",
    "CursorPosition",
    "CursorResume",
    "LogPage",
    "LogRecord",
    "LogStream",
    "decode_log_cursor",
    "encode_archive_offset",
    "encode_log_cursor",
    "encode_unplaced_archive_position",
    "is_archive_position",
    "join_log_cursor",
    "log_cursor_v2_emission_enabled",
    "log_page_from_json",
    "merge_pages",
    "parse_journal_json",
    "resolve_archive_position",
    "split_log_cursor",
    "stream_for_priority",
)
