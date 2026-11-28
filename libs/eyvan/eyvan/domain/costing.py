# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exact money and resource-hour arithmetic.

Every dollar Eyvan reports passes through here. The rules are small and the reason
they exist is specific: costs enter as JSON floats, are accumulated across many
settlements into a durable ``BudgetRecord.spent``, and are then compared against a
``limit`` by a *hard admission gate*. Binary floating point accumulates error under
exactly that pattern -- repeated addition of values like ``0.1`` -- so a fleet could
drift into admitting work past its ceiling, or refusing work below it, with no bug
anywhere obvious.

So: parse through ``str`` into ``Decimal``, do the arithmetic in ``Decimal``, and
return ``float`` only at the boundary, because the persisted models are JSON. Going
through ``str`` matters -- ``Decimal(0.1)`` captures the binary approximation while
``Decimal("0.1")`` is exact.

Kept in ``core`` because it is pure: no clock, no I/O, no records. That is also what
lets a cost be asserted for exact equality in tests rather than with a tolerance.
"""

from __future__ import annotations

import math
import typing as tp
from decimal import Decimal, Overflow, localcontext

#: Places retained when a Decimal result returns to float. Ten is well beyond any
#: real currency precision and well inside float's exact-integer range for the
#: magnitudes Eyvan handles, so it truncates accumulated noise without rounding away
#: a genuine fraction of a cent.
_QUANTUM = Decimal("0.0000000001")


def money(value: float | int | str | Decimal) -> Decimal:
    """Return ``value`` as an exact Decimal.

    Floats are routed through ``str`` on purpose: ``Decimal(0.1)`` is the binary
    approximation, ``Decimal(str(0.1))`` is ``0.1``. Nonfinite inputs are
    refused: a NaN that slipped into a settlement would silently poison every
    later budget comparison (NaN comparisons are all False, so a hard gate
    would pass), and an infinity would only crash ``quantize`` deep in
    reporting instead of failing here with a cause.
    """

    amount = value if isinstance(value, Decimal) else Decimal(str(value))
    if not amount.is_finite():
        raise ValueError(f"money must be finite, got {value!r}")
    return amount


def total(*values: float | int | str | Decimal) -> float:
    """Sum money exactly, returning a JSON-representable float."""

    try:
        with localcontext():
            amount = sum((money(value) for value in values), Decimal(0))
    except Overflow as exc:
        raise ValueError("money result must be a finite float") from exc
    return _to_float(amount)


def cost_for_hours(hourly_cost: float | int | str | Decimal, hours: float | int | str | Decimal) -> float:
    """Return the cost of holding capacity priced at ``hourly_cost`` for ``hours``."""

    try:
        with localcontext():
            amount = money(hourly_cost) * money(hours)
    except Overflow as exc:
        raise ValueError("money result must be a finite float") from exc
    return _to_float(amount)


def scale(values: tp.Mapping[str, float], factor: float | int | str | Decimal) -> dict[str, float]:
    """Scale a resource-hour vector, dropping entries that scale to nothing.

    Zero-valued entries are omitted rather than kept: ``accelerator_hours`` is read as
    "which accelerators did this consume", and a persisted ``{"gpu": 0.0}`` reads as a
    GPU job that used no GPU rather than as a job with no GPU at all.
    """

    multiplier = money(factor)
    scaled = {name: cost_for_hours(quantity, multiplier) for name, quantity in values.items()}
    return {name: quantity for name, quantity in scaled.items() if quantity}


def _to_float(value: Decimal) -> float:
    """Quantize away accumulated noise and return a JSON-safe float.

    The quantum bounds the *fraction*, not the magnitude: ``quantize`` raises
    InvalidOperation once the significant digits exceed the context precision,
    so the rounding runs in a context widened for the magnitude at hand.
    """

    # Check before widening precision: an arbitrary Decimal exponent must not
    # cause an allocation proportional to a magnitude JSON cannot represent.
    if not value.is_finite() or not math.isfinite(float(value)):
        raise ValueError("money result must be a finite float")
    with localcontext() as context:
        # Quantization retains ten fractional places, regardless of the input's
        # exponent. One additional digit accommodates a carry during rounding.
        context.prec = max(context.prec, (value.adjusted() if value else 0) + 12)
        rounded = value.quantize(_QUANTUM)
    # normalize() would round again using the caller's (possibly lower) precision.
    result = float(rounded)
    if not math.isfinite(result):
        raise ValueError("money result must be a finite float")
    return result


__all__ = ("cost_for_hours", "money", "scale", "total")
