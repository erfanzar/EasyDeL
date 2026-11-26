# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Small strict JSON value definitions for untrusted local/provider documents."""

from __future__ import annotations

import math
import typing as tp
from typing import Any, TypeGuard

from pydantic import BaseModel

JsonScalar: tp.TypeAlias = str | int | float | bool | None
JsonValue: tp.TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: tp.TypeAlias = dict[str, JsonValue]


def is_json_value(value: object) -> tp.TypeGuard[JsonValue]:
    """Return whether a decoded value is composed only of JSON-native values."""

    if isinstance(value, float):
        return math.isfinite(value)
    if value is None or isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def is_json_object(value: object) -> bool | TypeGuard[str | int | float | bool | None | list[Any] | dict[str, Any]]:
    """Return whether a decoded value is a JSON object."""

    return isinstance(value, dict) and is_json_value(value)


def model_json(
    model: BaseModel,
    *,
    exclude_unset: bool = False,
    exclude_none: bool = False,
) -> JsonObject:
    """Serialize a validated Pydantic model into the strict JSON algebra."""

    document = model.model_dump(mode="json", exclude_none=exclude_none, exclude_unset=exclude_unset)
    if not is_json_object(document):
        raise TypeError(f"{type(model).__name__} did not serialize to a JSON object")
    return document


__all__ = ("JsonObject", "JsonScalar", "JsonValue", "is_json_object", "is_json_value", "model_json")
