# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic, text-only summary of a :class:`~spectrax.Module`.

:func:`summary` walks the module's live :class:`~spectrax.Variable`
leaves via :func:`spectrax.live_variables` and produces a tabular
report suitable for logging or pasting into a bug report. When
example inputs are provided, the output spec is computed via
:func:`jax.eval_shape` with hooks suppressed so introspection cannot
trigger forward / variable observers.
"""

from __future__ import annotations

import jax

from ..core.graph import live_variables
from ..core.module import Module, _inside_transform, _set_inside_transform

__all__ = ["summary"]


def summary(module: Module, *example_inputs: object, **example_kwargs: object) -> str:
    """Return a deterministic multi-line text summary of ``module``.

    The report lists every live :class:`~spectrax.Variable` (after
    deduplication by ``ref_id`` — shared variables are reported once
    under their canonical path) with five columns: ``path``, ``class``,
    ``kind`` (collection), ``shape``, and ``dtype``. The footer prints
    aggregate scalar counts split as ``parameters`` (from the
    ``"parameters"`` collection) and ``other`` (everything else).

    When example inputs are supplied, :func:`jax.eval_shape` is run
    with hooks suppressed (via
    :func:`~spectrax.core.module._set_inside_transform`) so the
    summary does not trigger forward / variable hooks, and the output
    spec is appended after the header. Any exception from ``eval_shape``
    is caught and reported inline rather than propagated.

    Args:
        module: The module to summarize.
        *example_inputs: Optional positional example inputs forwarded
            to ``module.__call__`` under :func:`jax.eval_shape`.
        **example_kwargs: Optional keyword example inputs.

    Returns:
        The summary as a single string ready to ``print``.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    total_parameters = 0
    total_buffers = 0
    for path, var in live_variables(module):
        shape = tuple(getattr(var, "shape", ()))
        try:
            n = 1
            for s in shape:
                n *= int(s)
        except Exception:
            n = 0
        dtype = str(getattr(var, "dtype", ""))
        kind = var.kind
        rows.append((path, type(var).__name__, kind, str(shape), dtype))
        if kind == "parameters":
            total_parameters += n
        else:
            total_buffers += n

    lines = [f"{type(module).__name__} — {len(rows)} variables"]
    if example_inputs or example_kwargs:
        try:
            prev = _inside_transform()
            _set_inside_transform(True)
            try:
                out_spec = jax.eval_shape(lambda: module(*example_inputs, **example_kwargs))
            finally:
                _set_inside_transform(prev)
            lines.append(f"output: {out_spec}")
        except Exception as e:
            lines.append(f"output: (eval_shape failed: {e})")
    w0 = max((len(r[0]) for r in rows), default=4)
    w1 = max((len(r[1]) for r in rows), default=4)
    w2 = max((len(r[2]) for r in rows), default=4)
    w3 = max((len(r[3]) for r in rows), default=4)
    w4 = max((len(r[4]) for r in rows), default=4)
    header = f"{'path':<{w0}}  {'class':<{w1}}  {'kind':<{w2}}  {'shape':<{w3}}  {'dtype':<{w4}}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:<{w2}}  {r[3]:<{w3}}  {r[4]:<{w4}}")
    lines.append("-" * len(header))
    lines.append(f"parameters={total_parameters:,}  other={total_buffers:,}")
    return "\n".join(lines)
