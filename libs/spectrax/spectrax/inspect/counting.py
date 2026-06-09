# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parameter and byte counting helpers.

Re-exports :func:`count_parameters` and :func:`count_bytes` from
:mod:`spectrax.inspect.tabulate` and adds :func:`format_parameters` for
human-readable rendering of large counts.
"""

from __future__ import annotations

from .tabulate import count_bytes, count_parameters

__all__ = ["count_bytes", "count_parameters", "format_parameters"]


def format_parameters(n: int) -> str:
    """Render a parameter count compactly with a K / M / B suffix.

    Examples: ``999 -> "999"``, ``12_345 -> "12.3K"``,
    ``1_234_567 -> "1.2M"``, ``2_500_000_000 -> "2.5B"``. Numbers below
    1,000 are returned without a suffix.

    Args:
        n: A non-negative integer parameter count.

    Returns:
        The compact string form, with one decimal place when a
        suffix is used.
    """
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}B"
