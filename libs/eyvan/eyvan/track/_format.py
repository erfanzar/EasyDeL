# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The storage format shared by the in-job tracking client and the tracker.

The object store is the source of truth for run tracking: jobs append
immutable, gzip-compressed NDJSON segments and the tracker only builds a
rebuildable index over them. This module is the single definition of that
format, imported by both sides, so the writer and the reader cannot disagree
about a path, a filename or how a non-finite value is spelled.

It is deliberately standard-library only: it runs inside training jobs, where
the package's lazy root exists to keep imports light.

Layout under a scope root (``EYVAN_TRACKING_URI``, which the control plane
already scopes to ``<store>/tracking/<org>/<project>``)::

    <scope-root>/<run_id>/<stream>/a<attempt>-r<rank>-<writer>-<seq>.ndjson.gz

Segments are never rewritten. A writer names a segment once, before its first
upload attempt, so a replay from the local spill directory reuses the name and
is idempotent instead of duplicating records.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import typing as tp
import zlib
from dataclasses import dataclass

#: Bumped only for an incompatible change; recorded in every ``start`` record.
FORMAT_VERSION = 1

#: Scope root a job writes under. Set by the control plane at dispatch.
TRACKING_URI_ENV = "EYVAN_TRACKING_URI"

STREAM_META = "meta"
STREAM_SCALARS = "scalars"
STREAM_HIST = "hist"
STREAM_EPISODES = "episodes"
STREAM_ROLLOUTS = "rollouts"
STREAM_COMPLETIONS = "completions"
STREAM_TABLES = "tables"
STREAM_MEDIA = "media"
#: Known streams. Readers must ignore streams they do not know, so later
#: phases stay additive.
STREAMS: tuple[str, ...] = (
    STREAM_META,
    STREAM_SCALARS,
    STREAM_HIST,
    STREAM_EPISODES,
    STREAM_ROLLOUTS,
    STREAM_COMPLETIONS,
    STREAM_TABLES,
    STREAM_MEDIA,
)

#: Directory beside the streams holding content-addressed media bytes. It is
#: deliberately not a stream: readers that only know streams skip it.
BLOB_DIR = "blobs"
#: Media types a writer may store and a reader may serve. Anything else is
#: refused on write and never served, so a blob can't become active content.
MEDIA_TYPES: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: Episode kinds a record may carry.
EPISODE_KINDS: tuple[str, ...] = ("train", "eval")
#: Upper bound on histogram bins, enforced by writer and reader alike.
MAX_HIST_BINS = 512

META_START = "start"
META_CONFIG = "config"
META_SUMMARY = "summary"
META_STATS = "stats"
META_FINISH = "finish"
META_OPS: tuple[str, ...] = (META_START, META_CONFIG, META_SUMMARY, META_STATS, META_FINISH)

#: Terminal statuses a ``finish`` record may carry.
FINISH_STATUSES: tuple[str, ...] = ("finished", "failed", "aborted")

SEGMENT_SUFFIX = ".ndjson.gz"
#: Decode ceilings: 256 MiB on each side of gzip, including all concatenated
#: members. A line may use the full byte budget; at most 1,000,000 records are
#: parsed. These bounds prevent stored segments from causing unbounded expansion
#: or JSON-object allocation, without changing normal segment bytes.
MAX_COMPRESSED_SEGMENT_BYTES = 256 * 1024 * 1024
MAX_SEGMENT_BYTES = 256 * 1024 * 1024
MAX_SEGMENT_LINE_BYTES = MAX_SEGMENT_BYTES
MAX_SEGMENT_RECORDS = 1_000_000
MAX_ATTEMPT = 9999
MAX_RANK = 9999
MAX_SEQ = 99_999_999

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_STREAM = re.compile(r"[a-z][a-z0-9_]{0,31}")
_WRITER = re.compile(r"[0-9a-f]{12}")
_SEGMENT = re.compile(r"a(\d{4})-r(\d{4})-([0-9a-f]{12})-(\d{8})" + re.escape(SEGMENT_SUFFIX))
_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

_NONFINITE_ENCODED = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged, or raise when it cannot be one path segment.

    Raises:
        ValueError: If the id is empty, too long, or contains characters that
            could escape or alias a store path (``/``, ``..`` prefixes, spaces).
    """

    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            f"invalid run id {run_id!r}: use 1-128 letters, digits, '.', '_' or '-', starting with a letter or digit"
        )
    return run_id


def validate_stream(stream: str) -> str:
    """Return ``stream`` unchanged, or raise when it is not a stream directory name."""

    if not isinstance(stream, str) or not _STREAM.fullmatch(stream):
        raise ValueError(f"invalid stream name {stream!r}")
    return stream


def run_id_for_workload(workload_id: str) -> str:
    """Derive the stable run id for an Eyvan workload.

    Every attempt of a workload derives the same id, which is what makes a
    retry or a Spot preemption resume the same run instead of starting another.
    """

    cleaned = _RUN_ID_UNSAFE.sub("-", workload_id.strip()).strip("-._")
    if not cleaned:
        raise ValueError(f"workload id {workload_id!r} cannot form a run id")
    return validate_run_id(cleaned[:128])


@dataclass(frozen=True, slots=True)
class SegmentName:
    """Identity of one immutable segment within a run's stream directory."""

    attempt: int
    rank: int
    writer: str
    seq: int

    def __post_init__(self) -> None:
        for field_name, value, ceiling in (
            ("attempt", self.attempt, MAX_ATTEMPT),
            ("rank", self.rank, MAX_RANK),
            ("seq", self.seq, MAX_SEQ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= ceiling:
                raise ValueError(f"segment {field_name} must be an integer from 0 through {ceiling}")
        if not isinstance(self.writer, str) or not _WRITER.fullmatch(self.writer):
            raise ValueError("segment writer must be 12 lowercase hex characters")

    @property
    def filename(self) -> str:
        """Return the canonical file name, which sorts by attempt, rank, writer and seq."""

        return f"a{self.attempt:04d}-r{self.rank:04d}-{self.writer}-{self.seq:08d}{SEGMENT_SUFFIX}"

    @classmethod
    def parse(cls, filename: str) -> SegmentName | None:
        """Parse a segment file name; ``None`` for anything that is not one."""

        match = _SEGMENT.fullmatch(filename)
        if match is None:
            return None
        attempt, rank, writer, seq = match.groups()
        return cls(attempt=int(attempt), rank=int(rank), writer=writer, seq=int(seq))


def segment_key(run_id: str, stream: str, name: SegmentName) -> str:
    """Return the object key of a segment relative to the scope root."""

    return f"{validate_run_id(run_id)}/{validate_stream(stream)}/{name.filename}"


def validate_sha256(digest: str) -> str:
    """Return ``digest`` unchanged, or raise when it is not 64 lowercase hex characters."""

    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("blob digest must be 64 lowercase hex characters")
    return digest


def blob_key(run_id: str, digest: str) -> str:
    """Return the object key of a content-addressed blob relative to the scope root."""

    return f"{validate_run_id(run_id)}/{BLOB_DIR}/{validate_sha256(digest)}"


def encode_value(value: float) -> float | str:
    """Spell one metric value for JSON: finite floats as numbers, others as strings.

    A diverging loss is information. Strict JSON has no NaN or infinity, so they
    travel as ``"nan"``, ``"inf"`` and ``"-inf"`` instead of being dropped.
    """

    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def decode_value(raw: object) -> float:
    """Inverse of :func:`encode_value`; raises ``ValueError`` for anything else."""

    if isinstance(raw, bool):
        raise ValueError("a boolean is not a metric value")
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str) and raw in _NONFINITE_ENCODED:
        return _NONFINITE_ENCODED[raw]
    raise ValueError(f"unsupported encoded metric value {raw!r}")


def encode_segment(records: tp.Iterable[tp.Mapping[str, object]]) -> bytes:
    """Serialize records as deterministic gzip NDJSON (``mtime=0``, compact JSON)."""

    lines = [json.dumps(record, separators=(",", ":"), allow_nan=False, ensure_ascii=False) for record in records]
    payload = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    return gzip.compress(payload, mtime=0)


def decode_segment(data: bytes) -> list[dict[str, object]]:
    """Parse bounded gzip NDJSON; corrupt or oversized segments raise ``ValueError``.

    Read at most the remaining byte/line budget plus one sentinel byte. Splitting
    on binary LF only preserves Unicode separators in old and new JSON strings;
    ``str.splitlines`` would incorrectly treat them as record delimiters.
    """

    if len(data) > MAX_COMPRESSED_SEGMENT_BYTES:
        raise ValueError("tracking segment exceeds the compressed byte limit")
    records: list[dict[str, object]] = []
    total = number = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as handle:
            while line := handle.readline(min(MAX_SEGMENT_LINE_BYTES, MAX_SEGMENT_BYTES - total) + 1):
                number += 1
                total += len(line)
                if total > MAX_SEGMENT_BYTES:
                    raise ValueError("tracking segment exceeds the decompressed byte limit")
                if len(line) > MAX_SEGMENT_LINE_BYTES:
                    raise ValueError("tracking segment exceeds the line byte limit")
                text = line.decode("utf-8")
                if not text.strip():
                    continue
                if len(records) >= MAX_SEGMENT_RECORDS:
                    raise ValueError("tracking segment exceeds the record count limit")
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    raise ValueError(f"corrupt tracking segment line {number}") from None
                if not isinstance(record, dict):
                    raise ValueError(f"tracking segment line {number} is not a JSON object")
                records.append(record)
    except (OSError, EOFError, UnicodeDecodeError, zlib.error):
        raise ValueError("corrupt tracking segment") from None
    return records


__all__ = (
    "BLOB_DIR",
    "EPISODE_KINDS",
    "FINISH_STATUSES",
    "FORMAT_VERSION",
    "MAX_ATTEMPT",
    "MAX_COMPRESSED_SEGMENT_BYTES",
    "MAX_HIST_BINS",
    "MAX_RANK",
    "MAX_SEGMENT_BYTES",
    "MAX_SEGMENT_LINE_BYTES",
    "MAX_SEGMENT_RECORDS",
    "MAX_SEQ",
    "MEDIA_TYPES",
    "META_CONFIG",
    "META_FINISH",
    "META_OPS",
    "META_START",
    "META_STATS",
    "META_SUMMARY",
    "SEGMENT_SUFFIX",
    "STREAMS",
    "STREAM_COMPLETIONS",
    "STREAM_EPISODES",
    "STREAM_HIST",
    "STREAM_MEDIA",
    "STREAM_META",
    "STREAM_ROLLOUTS",
    "STREAM_SCALARS",
    "STREAM_TABLES",
    "TRACKING_URI_ENV",
    "SegmentName",
    "blob_key",
    "decode_segment",
    "decode_value",
    "encode_segment",
    "encode_value",
    "run_id_for_workload",
    "segment_key",
    "validate_run_id",
    "validate_sha256",
    "validate_stream",
)
