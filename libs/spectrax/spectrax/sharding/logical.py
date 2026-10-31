# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logical-to-mesh axis-name rules, managed as a thread-local stack.

Logical axis rules let a layer declare its sharding intent in
*semantic* terms (e.g. ``"embed"``, ``"heads"``, ``"vocab"``) and have
the runtime translate those names into concrete physical mesh axes
(``"tp"``, ``"fsdp"``, ``"sp"``, …) at constraint time. The mapping is
pushed onto a thread-local stack by :func:`logical_axis_rules` so the
same model code can run with different physical mappings depending on
the active mesh — and so multiple worker threads can carry independent
mappings without interference.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import TypeAlias

__all__ = ["current_axis_rules", "logical_axis_rules"]


_STACK: threading.local = threading.local()

AxisRuleValue: TypeAlias = str | tuple[str | None, ...] | None
"""Physical mesh-axis target for one logical axis-rule entry."""


def _get_stack() -> list[dict[str, AxisRuleValue]]:
    """Return the thread-local stack of axis-rule mappings.

    Each frame maps a logical name to a physical mesh-axis name, a fused
    tuple of physical mesh axes, or ``None`` to drop the axis. The list
    grows with each :func:`logical_axis_rules` ``with`` and shrinks on
    exit; per-thread storage means worker threads do not see each
    other's frames.

    Returns:
        Return the thread-local stack of axis-rule mappings.
    """
    s = getattr(_STACK, "stack", None)
    if s is None:
        s = []
        _STACK.stack = s
    return s


@contextlib.contextmanager
def logical_axis_rules(rules: Sequence[tuple[str, AxisRuleValue]]) -> Iterator[None]:
    """Push a logical-to-mesh axis mapping onto the stack for the ``with`` body.

    The mapping is consulted by :func:`current_axis_rules` (and through
    that by :func:`spectrax.sharding.with_sharding_constraint_by_name`,
    :func:`get_partition_spec`, and friends) when translating logical
    axis names declared on variable metadata into physical mesh axes.

    Inside the block the rules are *merged* with any outer frames —
    inner rules override outer entries with the same key, and
    inheriting the rest. On exit the pushed frame is popped.

    Args:
        rules: Sequence of ``(logical_name, mesh_axis_target)`` pairs.
            The target may be a physical mesh-axis name, a fused tuple
            such as ``("fsdp", "dp")``, or ``None`` to drop the logical
            axis and replicate along it.

    Yields:
        ``None``. Use :func:`current_axis_rules` inside the block to
        read the merged mapping.
    """
    mapping = dict(rules)
    stack = _get_stack()
    stack.append(mapping)
    try:
        yield
    finally:
        stack.pop()


def current_axis_rules() -> Mapping[str, AxisRuleValue]:
    """Return the merged logical-to-mesh axis mapping currently in effect.

    Frames pushed by :func:`logical_axis_rules` are merged from the
    bottom of the stack to the top, so inner ``with`` blocks override
    outer ones for the same logical name. Returns an empty mapping when
    no :func:`logical_axis_rules` context is active.

    Returns:
        A new ``dict`` (the caller may mutate it without affecting the
        stack).
    """
    merged: dict[str, AxisRuleValue] = {}
    for frame in _get_stack():
        merged.update(frame)
    return merged
