# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared accelerator-type normalization and presentation helpers."""

from __future__ import annotations

import re
import typing as tp
import unicodedata
from dataclasses import dataclass

#: Every generation before the seventh is named ``v<n>`` with an optional
#: variant letter -- v4, v5p, v5litepod, v6e. The seventh broke the pattern:
#: Google ships it as ``tpu7x``, so a regex anchored on a leading ``v`` refused
#: the whole generation before any policy could look at it.
_TPU_ACCELERATOR_TYPE = re.compile(r"^(?P<generation>(?:v[0-9]|tpu[0-9])[a-z0-9]*?)-(?P<size>[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True)
class TpuAcceleratorType:
    """The generation and public slice size encoded in a TPU accelerator type."""

    accelerator_type: str
    generation: str
    size: int


def parse_tpu_accelerator_type(value: str) -> TpuAcceleratorType | None:
    """Parse one normalized TPU accelerator type without applying provider policy.

    This intentionally does not decide whether a type is supported by Eyvan's
    queued-resource autoscaler. Provider discovery must still report a valid
    type so the autoscale planner can explain why it was skipped.
    """

    matched = _TPU_ACCELERATOR_TYPE.fullmatch(value)
    if matched is None:
        return None
    return TpuAcceleratorType(
        accelerator_type=value,
        generation=matched.group("generation"),
        size=int(matched.group("size")),
    )


def accelerator_generation(value: str) -> str | None:
    """Return the normalized generation encoded by a valid TPU accelerator type."""

    parsed = parse_tpu_accelerator_type(value)
    return None if parsed is None else parsed.generation


def _accelerator_type_order(value: str) -> tuple[str, int, str, str]:
    """Sort a conventional ``family-size`` type by numeric size."""

    family, separator, raw_size = value.rpartition("-")
    if separator and family and raw_size.isdecimal():
        # Normalized digits compare numerically by length, then lexically,
        # without converting an unbounded provider name to an integer.
        size = "".join(str(unicodedata.decimal(digit)) for digit in raw_size).lstrip("0") or "0"
        return family, len(size), size, value
    return value, -1, "", value


def sort_accelerator_types(values: tp.Iterable[str]) -> tuple[str, ...]:
    """Return unique accelerator types in stable family and numeric-size order."""

    return tuple(sorted(set(values), key=_accelerator_type_order))


__all__ = (
    "TpuAcceleratorType",
    "accelerator_generation",
    "parse_tpu_accelerator_type",
    "sort_accelerator_types",
)
