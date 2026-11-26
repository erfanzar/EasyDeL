# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Summarising a set of accelerator shapes for a person.

A fleet here carries seventeen or twenty pools, and every surface that mentions
one has to answer "what can this run" in about half a line. The answer used to
be the first three shapes and a count -- ``v5p-16, v5p-32, v5p-64 +14 more`` --
which is not a summary but a truncation, and it truncates in the worst possible
place: shapes sort by generation, so the count swallowed whole families. A fleet
offering eleven v5p sizes and six v6e ones advertised itself as v5p-only, and
somebody checking whether it could run a v6e-8 was told no by a display
decision.

Collapse to one entry per generation with its size range instead. It is shorter
than the truncation it replaces for every fleet in this project, it cannot hide
a family, and it answers the question the line exists for.
"""

from __future__ import annotations

import typing as tp


def summarize_shapes(shapes: tp.Iterable[str]) -> str:
    """Return one entry per accelerator generation, with its size range.

    Args:
        shapes: Accelerator type names such as ``v5p-16``. Order is preserved by
            generation of first appearance, so a caller that sorted its input
            gets a stable line.

    Returns:
        A comma-separated summary such as ``v5p 16-1024, v6e 8-256``, or an
        empty string when there are no shapes.
    """

    grouped: dict[str, list[int]] = {}
    for shape in shapes:
        generation, _, size = shape.rpartition("-")
        # A name that is not <generation>-<size> ("cpu") is its own group: it
        # names no family to collapse into, and inventing one would be a guess.
        if generation and size.isdecimal():
            grouped.setdefault(generation, []).append(int(size))
        else:
            grouped.setdefault(shape, [])
    parts: list[str] = []
    for generation, sizes in grouped.items():
        if not sizes:
            parts.append(generation)
        elif len(sizes) == 1:
            parts.append(f"{generation}-{sizes[0]}")
        else:
            parts.append(f"{generation} {min(sizes)}-{max(sizes)}")
    return ", ".join(parts)


__all__ = ("summarize_shapes",)
