# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Validation and record building for rollouts, completions, tables and media.

Pure functions used by :class:`eyvan.track.Run`. They raise for caller bugs
(wrong types, mismatched lengths, unsupported media types) and never touch the
store. Standard library only: this runs inside training jobs.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import typing as tp
from pathlib import Path

from eyvan.track import _format as fmt

MAX_ROLLOUT_STEPS = 10_000
MAX_TRACKS = 32
MAX_ACTION_NAMES = 256
MAX_TEXT = 32_768
MAX_CELL_TEXT = 4096
MAX_COLUMNS = 64
MAX_ROWS = 1000
MAX_GROUP_SIZE = 64
MAX_CAPTION = 1024
MAX_LABEL = 128
#: Largest single blob a writer stores and a tracker serves.
MAX_BLOB_BYTES = 64 * 1024 * 1024
_ZERO_STD = 1e-8

_KEY = re.compile(r"[A-Za-z0-9_./-]{1,128}")
_EXTENSIONS: dict[str, str] = {ext: ctype for ctype, ext in fmt.MEDIA_TYPES.items()} | {"jpeg": "image/jpeg"}
_BLOB_FIELDS = frozenset({"sha256", "content_type", "bytes"})


def check_name(value: object, what: str) -> str:
    """Return a metric-style name, or raise ``ValueError``."""

    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise ValueError(f"invalid {what} {value!r}: use 1-128 characters from letters, digits, '_', '.', '/', '-'")
    return value


def label(value: object, what: str, limit: int = MAX_LABEL) -> str | None:
    """Return an optional short free-text label (ints are accepted and stringified)."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"{what} must be a string")
    text = str(value)
    if not text or len(text) > limit:
        raise ValueError(f"{what} must be 1-{limit} characters")
    return text


def number(what: str, value: object) -> float:
    """Convert a Python, numpy, JAX or torch scalar to ``float``; strings are refused."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{what} must be a number, got {type(value).__name__}")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be a number, got {type(value).__name__}") from exc


def optional_int(value: object, what: str, *, minimum: int | None = None) -> int | None:
    """Return an optional integer, accepting integral floats and 0-d array scalars."""

    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, int | float | str | bytes):
        value = item()
    if isinstance(value, bool) or isinstance(value, str | bytes):
        raise TypeError(f"{what} must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise TypeError(f"{what} must be an integer")
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"{what} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be at least {minimum}")
    return value


def finite_or_none(value: object, what: str) -> float | None:
    """Return a finite float, or ``None`` for ``None`` and non-finite values."""

    if value is None:
        return None
    result = number(what, value)
    return result if math.isfinite(result) else None


def sequence(value: object, what: str) -> list[object]:
    """Return a list from a list, tuple, iterable or array (``tolist`` aware)."""

    if isinstance(value, str | bytes | bytearray | tp.Mapping):
        raise TypeError(f"{what} must be a sequence")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, list | tuple):
        return list(value)
    if isinstance(value, tp.Iterable):
        return list(value)
    raise TypeError(f"{what} must be a sequence")


def clip_text(value: object, what: str, limit: int) -> tuple[str, bool]:
    """Return ``value`` cut to ``limit`` characters and whether it was cut."""

    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    return (value[:limit], True) if len(value) > limit else (value, False)


# ------------------------------------------------------------------- rollouts


def rollout_body(steps: object, tracks: object, actions: object, action_names: object) -> dict[str, object]:
    """Return ``length``, ``tracks``, ``actions``, ``action_names`` and ``ret`` of a rollout."""

    track_values: dict[str, list[object]] = {}
    action_values: list[object] | None = None
    if steps is not None:
        if tracks is not None or actions is not None:
            raise ValueError("pass steps, or tracks and/or actions, not both")
        rows = sequence(steps, "steps")
        names: list[str] = []
        for row in rows:
            if not isinstance(row, tp.Mapping):
                raise TypeError("each rollout step must be a mapping such as {'reward': r, 'action': a}")
            for name in row:
                if name != "action" and name not in names:
                    names.append(check_name(name, "track name"))
        track_values = {name: [row.get(name) for row in rows] for name in names}  # type: ignore[union-attr]
        if any("action" in row for row in rows):  # type: ignore[operator]
            action_values = [row.get("action") for row in rows]  # type: ignore[union-attr]
    else:
        if tracks is None and actions is None:
            raise ValueError("a rollout needs steps, tracks or actions")
        if tracks is not None:
            if not isinstance(tracks, tp.Mapping):
                raise TypeError("tracks must be a mapping of track names to sequences")
            track_values = {check_name(n, "track name"): sequence(v, f"track {n!r}") for n, v in tracks.items()}
        if actions is not None:
            action_values = sequence(actions, "actions")
    if len(track_values) > MAX_TRACKS:
        raise ValueError(f"a rollout accepts at most {MAX_TRACKS} tracks")
    lengths = {len(values) for values in track_values.values()}
    if action_values is not None:
        lengths.add(len(action_values))
    if len(lengths) != 1:
        raise ValueError("every track and the actions must have the same length")
    length = lengths.pop()
    if not 1 <= length <= MAX_ROLLOUT_STEPS:
        raise ValueError(f"a rollout must have 1-{MAX_ROLLOUT_STEPS} steps")
    encoded = {
        name: [None if v is None else fmt.encode_value(number(f"track {name!r}", v)) for v in values]
        for name, values in track_values.items()
    }
    names_out: list[str] | None = None
    if action_names is not None:
        names_out = [label(n, "action name", 64) or "" for n in sequence(action_names, "action_names")]
        if len(names_out) > MAX_ACTION_NAMES:
            raise ValueError(f"at most {MAX_ACTION_NAMES} action names")
    ret: object = None
    if "reward" in encoded:
        # The return sums finite per-step rewards; a NaN step stays visible in
        # the track but does not erase the episode's return.
        rewards = [r for v in encoded["reward"] if v is not None and math.isfinite(r := fmt.decode_value(v))]
        ret = math.fsum(rewards) if rewards else None
    return {
        "length": length,
        "tracks": encoded,
        "actions": None if action_values is None else [optional_int(a, "action") for a in action_values],
        "action_names": names_out,
        "ret": ret,
    }


# ---------------------------------------------------------------- completions


def group_advantages(totals: list[float]) -> list[float]:
    """Group-normalized advantages: ``(r - mean) / std``; all zero when the group has no variance."""

    finite = [value for value in totals if math.isfinite(value)]
    if not finite:
        return [0.0] * len(totals)
    scale = max(abs(value) for value in finite)
    if not scale:
        return [0.0] * len(totals)
    scaled = [value / scale for value in finite]
    mean = math.fsum(scaled) / len(scaled)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in scaled) / len(scaled))
    if not std or std < _ZERO_STD / scale:
        return [0.0] * len(totals)
    return [(value / scale - mean) / std if math.isfinite(value) else math.nan for value in totals]


def completions_body(
    prompt: object,
    responses: object,
    *,
    rewards: object,
    advantages: object,
    answer: object,
    tokens: object,
    kl: object,
    truncated: object,
) -> tuple[dict[str, object], int]:
    """Return the completion-group fields and how many texts had to be clipped."""

    prompt_text, cut = clip_text(prompt, "prompt", MAX_TEXT)
    clipped = int(cut)
    texts: list[str] = []
    for index, response in enumerate(sequence(responses, "responses")):
        text, cut = clip_text(response, f"responses[{index}]", MAX_TEXT)
        texts.append(text)
        clipped += cut
    size = len(texts)
    if not 1 <= size <= MAX_GROUP_SIZE:
        raise ValueError(f"a completion group needs 1-{MAX_GROUP_SIZE} responses")

    def per_response(values: object, what: str) -> list[object]:
        items = sequence(values, what)
        if len(items) != size:
            raise ValueError(f"{what} must have one entry per response ({size})")
        return items

    components: dict[str, list[float]] = {}
    if rewards is None:
        totals = [math.nan] * size
    elif isinstance(rewards, tp.Mapping):
        if not rewards:
            raise ValueError("rewards mapping must name at least one component")
        for name, values in rewards.items():
            what = f"rewards[{name!r}]"
            components[check_name(name, "reward component")] = [number(what, v) for v in per_response(values, what)]
        totals = [sum(column[i] for column in components.values()) for i in range(size)]
    else:
        totals = [number("rewards", v) for v in per_response(rewards, "rewards")]
    if advantages is None:
        advantage_values = group_advantages(totals)
    else:
        advantage_values = [number("advantages", v) for v in per_response(advantages, "advantages")]
    token_values = (
        [optional_int(v, "tokens", minimum=0) for v in per_response(tokens, "tokens")]
        if tokens is not None
        else [None] * size
    )
    kl_values = [finite_or_none(v, "kl") for v in per_response(kl, "kl")] if kl is not None else [None] * size
    truncated_values: list[object] = per_response(truncated, "truncated") if truncated is not None else [None] * size
    if any(v is not None and not isinstance(v, bool) for v in truncated_values):
        raise TypeError("truncated entries must be booleans or None")
    answer_text = None
    if answer is not None:
        answer_text, cut = clip_text(answer, "answer", MAX_TEXT)
        clipped += cut
    return {
        "prompt": prompt_text,
        "answer": answer_text,
        "responses": [
            {
                "text": texts[i],
                "rewards": {name: fmt.encode_value(column[i]) for name, column in components.items()},
                "reward": fmt.encode_value(totals[i]),
                "advantage": fmt.encode_value(advantage_values[i]),
                "tokens": token_values[i],
                "kl": kl_values[i],
                "truncated": truncated_values[i],
            }
            for i in range(size)
        ],
    }, clipped


# --------------------------------------------------------------------- tables


def is_blob_ref(value: object) -> bool:
    """Return whether ``value`` is a well-formed BlobRef mapping."""

    if not isinstance(value, tp.Mapping) or set(value) != _BLOB_FIELDS:
        return False
    size = value["bytes"]
    try:
        fmt.validate_sha256(value["sha256"])
    except ValueError:
        return False
    return (
        value["content_type"] in fmt.MEDIA_TYPES
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 <= size <= MAX_BLOB_BYTES
    )


def _cell(value: object) -> tuple[object, bool]:
    if value is None or isinstance(value, bool):
        return value, False
    if isinstance(value, str):
        return clip_text(value, "table cell", MAX_CELL_TEXT)
    if isinstance(value, int):
        return value, False
    if isinstance(value, float):
        return fmt.encode_value(value), False
    if isinstance(value, tp.Mapping):
        if not is_blob_ref(value):
            raise TypeError("a mapping table cell must be a BlobRef returned by Run.blob()")
        return dict(value), False
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if isinstance(converted, bool | int | float | str):
            return _cell(converted)
    raise TypeError(f"unsupported table cell type {type(value).__name__}")


def table_body(rows: object, columns: object) -> tuple[list[str], list[list[object]], int]:
    """Return validated columns, rows and how many string cells were clipped."""

    row_list = sequence(rows, "rows")
    if len(row_list) > MAX_ROWS:
        raise ValueError(f"a table accepts at most {MAX_ROWS} rows")
    if columns is None:
        if not row_list or not all(isinstance(row, tp.Mapping) for row in row_list):
            raise ValueError("columns are required unless every row is a mapping")
        names: list[object] = []
        for row in row_list:
            for name in row:  # type: ignore[union-attr]
                if name not in names:
                    names.append(name)
    else:
        names = sequence(columns, "columns")
    for name in names:
        if not isinstance(name, str) or not 1 <= len(name) <= MAX_LABEL:
            raise ValueError("column names must be 1-128 character strings")
    if not 1 <= len(names) <= MAX_COLUMNS or len(set(names)) != len(names):
        raise ValueError(f"a table needs 1-{MAX_COLUMNS} unique columns")
    column_names = tp.cast(list[str], names)
    clipped = 0
    out: list[list[object]] = []
    for row in row_list:
        if isinstance(row, tp.Mapping):
            raw = [row.get(name) for name in column_names]
        else:
            raw = sequence(row, "table row")
            if len(raw) != len(column_names):
                raise ValueError("every table row must have one value per column")
        cells = []
        for value in raw:
            cell, cut = _cell(value)
            cells.append(cell)
            clipped += cut
        out.append(cells)
    return column_names, out, clipped


# ---------------------------------------------------------------------- media


def media_type(data_or_path: object, content_type: str | None, *, default: str | None = None) -> str:
    """Validate media arguments and resolve the allowlisted type without doing I/O."""

    if isinstance(data_or_path, bytes | bytearray | memoryview):
        content_type = content_type or default
        if content_type is None:
            raise ValueError("content_type is required for raw bytes")
    elif isinstance(data_or_path, str | os.PathLike):
        path = Path(data_or_path)
        content_type = content_type or _EXTENSIONS.get(path.suffix.lower().lstrip(".")) or default
        if content_type is None:
            supported = ", ".join(sorted(_EXTENSIONS))
            raise ValueError(f"cannot infer a media type from {path.name!r}; supported suffixes: {supported}")
    else:
        raise TypeError("media must be bytes or a file path")
    if content_type not in fmt.MEDIA_TYPES:
        raise ValueError(f"unsupported media type {content_type!r}; allowed: {', '.join(sorted(fmt.MEDIA_TYPES))}")
    return content_type


def read_media(data_or_path: object, content_type: str | None, *, default: str | None = None) -> tuple[bytes, str]:
    """Read at most the blob cap plus one byte; source OSErrors are handled by Run."""

    content_type = media_type(data_or_path, content_type, default=default)
    if isinstance(data_or_path, bytes | bytearray | memoryview):
        data = bytes(data_or_path)
    else:
        path = Path(data_or_path).expanduser()  # type: ignore[arg-type]
        size = path.stat().st_size
        if size > MAX_BLOB_BYTES:
            raise ValueError(f"media is {size} bytes; the limit is {MAX_BLOB_BYTES}")
        # The file can grow after stat; never read an unbounded amount even then.
        with path.open("rb") as source:
            data = source.read(MAX_BLOB_BYTES + 1)
    if len(data) > MAX_BLOB_BYTES:
        raise ValueError(f"media is {len(data)} bytes; the limit is {MAX_BLOB_BYTES}")
    return data, content_type


def blob_ref(data: bytes, content_type: str) -> dict[str, object]:
    """Return the BlobRef that names ``data`` by its SHA-256."""

    return {"sha256": hashlib.sha256(data).hexdigest(), "content_type": content_type, "bytes": len(data)}


__all__ = (
    "MAX_BLOB_BYTES",
    "blob_ref",
    "check_name",
    "completions_body",
    "group_advantages",
    "is_blob_ref",
    "label",
    "read_media",
    "rollout_body",
    "table_body",
)
