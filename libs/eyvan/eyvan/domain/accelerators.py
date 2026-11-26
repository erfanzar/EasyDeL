# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared accelerator-type normalization and presentation helpers."""

from __future__ import annotations

import typing as tp
import unicodedata


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


__all__ = ("sort_accelerator_types",)
