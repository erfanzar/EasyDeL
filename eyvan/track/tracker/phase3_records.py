# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strict, additive-field-tolerant validation of rich tracking records."""

from __future__ import annotations

import typing as tp

from eyvan.track._format import MEDIA_TYPES, decode_value, validate_sha256
from eyvan.track.tracker.records import axes, finite, integer, metric, text

Record = dict[str, tp.Any]
MAX_BLOB_BYTES = 64 * 1024**2
STREAMS = ("rollouts", "completions", "tables", "media")


def encoded(value: object) -> float | str:
    """Validate a finite JSON number or an explicitly encoded nonfinite value."""

    if isinstance(value, str):
        decode_value(value)
        return value
    return finite(value)


def blob_ref(value: object) -> Record:
    """Require a bounded, allowlisted content-addressed blob reference."""

    if not isinstance(value, dict) or not {"sha256", "content_type", "bytes"}.issubset(value):
        raise ValueError("incomplete blob reference")
    content_type = value["content_type"]
    if not isinstance(content_type, str) or content_type not in MEDIA_TYPES:
        raise ValueError("unsupported media content type")
    return {
        "sha256": validate_sha256(value["sha256"]),
        "content_type": content_type,
        "bytes": integer(value["bytes"], minimum=0, maximum=MAX_BLOB_BYTES),
    }


def _required(record: Record, fields: tp.Iterable[str]) -> None:
    if not set(fields).issubset(record):
        raise ValueError("incomplete tracking record")


def _string(value: object, maximum: int | None = None) -> str:
    if not isinstance(value, str) or (maximum is not None and len(value) > maximum):
        raise ValueError("invalid text field")
    value.encode("utf-8")
    return value


def _version(value: object) -> int | None:
    return integer(value, minimum=0) if value is not None else None


def _base(record: Record) -> Record:
    _required(record, ("t", "step"))
    return {
        "t": finite(record["t"]),
        "step": integer(record["step"], minimum=0, maximum=2**63 - 1) if record["step"] is not None else None,
    }


def _rollout(record: Record) -> Record:
    _required(
        record, ("episode_id", "actor", "env", "kind", "policy_version", "length", "tracks", "ret", "video", "fps", "x")
    )
    length = integer(record["length"], minimum=1, maximum=10000)
    tracks = record["tracks"]
    if not isinstance(tracks, dict) or len(tracks) > 32:
        raise ValueError("rollout tracks must be an object with at most 32 tracks")
    normalized = {}
    for name, values in tracks.items():
        if not isinstance(values, list) or len(values) != length:
            raise ValueError("track length does not match rollout length")
        normalized[metric(name)] = [encoded(value) if value is not None else None for value in values]
    actions = record.get("actions")
    if actions is not None:
        if not isinstance(actions, list) or len(actions) != length:
            raise ValueError("action length does not match rollout length")
        actions = [integer(value) if value is not None else None for value in actions]
    names = record.get("action_names")
    if names is not None:
        if not isinstance(names, list) or len(names) > 256:
            raise ValueError("invalid action names")
        names = [_string(value) for value in names]
    if record["kind"] not in ("train", "eval"):
        raise ValueError("invalid rollout kind")
    fps = finite(record["fps"]) if record["fps"] is not None else None
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    return {
        "episode_id": _string(record["episode_id"], 128),
        "actor": text(record["actor"], 64),
        "env": text(record["env"], 128),
        "kind": record["kind"],
        "policy_version": _version(record["policy_version"]),
        "length": length,
        "tracks": normalized,
        "actions": actions,
        "action_names": names,
        "ret": finite(record["ret"]) if record["ret"] is not None else None,
        "video": blob_ref(record["video"]) if record["video"] is not None else None,
        "fps": fps,
        "x": axes(record["x"]),
    }


def _response(response: object) -> Record:
    if not isinstance(response, dict):
        raise ValueError("completion response must be an object")
    _required(response, ("text", "rewards", "reward", "advantage", "tokens", "kl", "truncated"))
    rewards = response["rewards"]
    if not isinstance(rewards, dict):
        raise ValueError("reward components must be an object")
    truncated = response["truncated"]
    if truncated is not None and not isinstance(truncated, bool):
        raise ValueError("truncated must be boolean or null")
    return {
        "text": _string(response["text"], 32768),
        "rewards": {metric(key): encoded(value) for key, value in rewards.items()},
        "reward": encoded(response["reward"]),
        "advantage": encoded(response["advantage"]),
        "tokens": integer(response["tokens"], minimum=0) if response["tokens"] is not None else None,
        "kl": finite(response["kl"]) if response["kl"] is not None else None,
        "truncated": truncated,
    }


def _completion(record: Record) -> Record:
    _required(record, ("group", "prompt", "answer", "policy_version", "responses"))
    responses = record["responses"]
    if not isinstance(responses, list) or not 1 <= len(responses) <= 64:
        raise ValueError("completion groups require 1 through 64 responses")
    return {
        "group": _string(record["group"], 128),
        "prompt": _string(record["prompt"], 32768),
        "answer": _string(record["answer"]) if record["answer"] is not None else None,
        "policy_version": _version(record["policy_version"]),
        "responses": [_response(response) for response in responses],
    }


def _cell(value: object) -> tp.Any:
    if isinstance(value, dict):
        return blob_ref(value)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return _string(value, 4096)
    finite(value)
    return value  # Keep generic-table integers exact, rather than rounding through float.


def _table(record: Record) -> Record:
    _required(record, ("key", "columns", "rows"))
    columns, rows = record["columns"], record["rows"]
    if not isinstance(columns, list) or not 1 <= len(columns) <= 64:
        raise ValueError("tables require 1 through 64 columns")
    columns = [_string(column, 128) for column in columns]
    if any(not column for column in columns):
        raise ValueError("empty table column")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate table columns")
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("tables allow at most 1000 rows")
    normalized = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            raise ValueError("table row width does not match columns")
        normalized.append([_cell(value) for value in row])
    return {"key": metric(record["key"]), "columns": columns, "rows": normalized}


def _media(record: Record) -> Record:
    _required(record, ("key", "blob", "caption"))
    return {"key": metric(record["key"]), "blob": blob_ref(record["blob"]), "caption": text(record["caption"], 1024)}


def validate_record(stream: str, record: Record) -> Record:
    """Validate a complete rich record, discarding unknown additive fields."""

    base = _base(record)
    return {
        **base,
        **{"rollouts": _rollout, "completions": _completion, "tables": _table, "media": _media}[stream](record),
    }


__all__ = ("MAX_BLOB_BYTES", "STREAMS", "blob_ref", "validate_record")
