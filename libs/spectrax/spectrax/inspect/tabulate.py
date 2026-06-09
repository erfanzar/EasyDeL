# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-submodule tabular reports and XLA cost-model probes.

Provides four entry points:

* :func:`count_parameters` and :func:`count_bytes` — totals over a
  named collection (default ``"parameters"``).
* :func:`tabulate` — PyTorch-style ``(path, class, parameters, bytes)``
  table walked via :func:`spectrax.iter_modules` (canonical path
  order). Optional example inputs append the output spec via
  :func:`jax.eval_shape` with hooks suppressed.
* :func:`hlo_cost` — XLA cost-model dict (``flops`` and
  ``bytes_accessed``) extracted from
  ``jax.jit(...).lower(...).compile().cost_analysis()``; returns an
  empty dict on failure.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core.graph import iter_modules, live_variables
from ..core.module import Module, _inside_transform, _set_inside_transform
from ..core.variable import Variable

__all__ = ["count_bytes", "count_parameters", "hlo_cost", "tabulate"]


def _var_size(v: Variable) -> tuple[int, int]:
    """Compute scalar count and byte size for a single variable.

    Treats unknown / unparseable shape entries as zero-element
    (element count 0) and unknown dtypes as 4 bytes per element so
    the function is total even on partially-initialised variables.

    Args:
        v: The :class:`~spectrax.Variable` to size.

    Returns:
        A ``(num_elements, num_bytes)`` pair.
    """
    shape = getattr(v, "shape", ())
    n = 1
    for s in shape:
        try:
            n *= int(s)
        except Exception:
            n = 0
            break
    dtype = getattr(v, "dtype", None)
    try:
        itemsize = jnp.dtype(dtype).itemsize if dtype is not None else 4
    except Exception:
        itemsize = 4
    return n, n * itemsize


def count_parameters(module: Module, *, collection: str = "parameters") -> int:
    """Sum scalar counts across every variable in ``collection``.

    Walks live (deduplicated) variables via
    :func:`spectrax.live_variables` and adds the element counts of
    those whose ``kind`` matches.

    Args:
        module: The module to count over.
        collection: Variable collection name. Defaults to
            ``"parameters"``; other common values are ``"buffers"``
            or any user-defined kind.

    Returns:
        Total number of scalar elements in the named collection.
    """
    total = 0
    for _, v in live_variables(module):
        if v.kind == collection:
            n, _ = _var_size(v)
            total += n
    return total


def count_bytes(module: Module, *, collection: str = "parameters") -> int:
    """Sum the in-memory byte size of every variable in ``collection``.

    Mirrors :func:`count_parameters` but accumulates ``bytes`` instead
    of element counts. Uses :func:`_var_size` per variable so dtype
    differences (bf16 vs f32 vs int8 etc.) are reflected.

    Args:
        module: The module to count over.
        collection: Variable collection name. Defaults to
            ``"parameters"``.

    Returns:
        Total byte size of the named collection.
    """
    total = 0
    for _, v in live_variables(module):
        if v.kind == collection:
            _, b = _var_size(v)
            total += b
    return total


def tabulate(
    module: Module,
    *example_args: object,
    depth: int | None = None,
    **example_kwargs: object,
) -> str:
    """Build a per-submodule ``(path, class, parameters, bytes)`` table.

    Iterates submodules in canonical-path order via
    :func:`spectrax.iter_modules`. Per-row counts include only
    variables owned directly by that submodule (paths without a
    ``"."``); nested submodule weights are reported on their own row,
    so column totals are not double-counted. The footer reports the
    full ``count_parameters`` and total bytes across the whole module.

    When example inputs are supplied, runs
    :func:`jax.eval_shape` with hooks suppressed (via
    :func:`~spectrax.core.module._set_inside_transform`) to append
    the output spec; eval_shape failures are caught and reported in
    the footer rather than raised.

    Args:
        module: The module to tabulate.
        *example_args: Optional positional inputs for an output-spec
            probe.
        depth: Maximum submodule depth to include (root is depth 0).
            ``None`` (default) prints every submodule.
        **example_kwargs: Optional keyword inputs for the probe.

    Returns:
        The rendered table as a single multi-line string.
    """
    rows: list[tuple[str, str, str, str]] = []
    for path, mod in iter_modules(module):
        if depth is not None and path.count(".") + (1 if path else 0) > depth:
            continue
        parameters_here = 0
        bytes_here = 0
        for p, v in live_variables(mod):
            if "." in p:
                continue
            n, b = _var_size(v)
            if v.kind == "parameters":
                parameters_here += n
            bytes_here += b
        cls = type(mod).__name__
        rows.append((path or "(root)", cls, f"{parameters_here:,}", f"{bytes_here:,}"))

    w0 = max((len(r[0]) for r in rows), default=4)
    w1 = max((len(r[1]) for r in rows), default=5)
    w2 = max((len(r[2]) for r in rows), default=6)
    w3 = max((len(r[3]) for r in rows), default=5)
    header = f"{'path':<{w0}}  {'class':<{w1}}  {'parameters':>{w2}}  {'bytes':>{w3}}"
    lines: list[str] = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:>{w2}}  {r[3]:>{w3}}")
    lines.append("-" * len(header))
    total_parameters = count_parameters(module)
    total_bytes = sum(_var_size(v)[1] for _, v in live_variables(module))
    lines.append(f"Total parameters: {total_parameters:,}  bytes: {total_bytes:,}")
    if example_args or example_kwargs:
        try:
            prev = _inside_transform()
            _set_inside_transform(True)
            try:
                out = jax.eval_shape(lambda: module(*example_args, **example_kwargs))
            finally:
                _set_inside_transform(prev)
            lines.append(f"Output: {out}")
        except Exception as e:
            lines.append(f"Output: (eval_shape failed: {e})")
    return "\n".join(lines)


def hlo_cost(module: Module, *example_args: object, **example_kwargs: object) -> dict[str, float]:
    """Probe XLA's cost model for ``module(*example_args, **example_kwargs)``.

    Wraps the call in a fresh closure, lowers it via :func:`jax.jit`
    with hooks suppressed (so introspection doesn't fire forward /
    variable observers), compiles, and reads the resulting cost
    analysis. The keys returned are the flat ones spectrax cares about.

    Any exception during lowering / compilation / analysis is swallowed
    and an empty dict is returned, so this is safe to call in
    diagnostics that should never fail user-visibly.

    Args:
        module: The module to probe.
        *example_args: Positional inputs sized for the module.
        **example_kwargs: Keyword inputs sized for the module.

    Returns:
        ``{"flops": float, "bytes_accessed": float}`` on success, or
        an empty dict on any failure or when the analysis is missing.
    """
    try:

        def run(*args: object, **kwargs: object) -> object:
            """Trivial closure that calls ``module`` with the given args.

            Exists purely so :func:`jax.jit` has a concrete callable
            to lower; calling ``module`` directly inside ``jit`` would
            also work but the closure makes the lowering boundary
            explicit.

            Args:
                *args: Additional positional arguments forwarded to the wrapped callable or backend.
                **kwargs: Additional keyword arguments forwarded to the wrapped callable or backend.

            Returns:
                Result described by this helper.
            """
            return module(*args, **kwargs)

        prev = _inside_transform()
        _set_inside_transform(True)
        try:
            lowered = jax.jit(run).lower(*example_args, **example_kwargs)
        finally:
            _set_inside_transform(prev)
        compiled = lowered.compile()
        analysis = compiled.cost_analysis()
        if analysis is None:
            return {}
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        return {
            "flops": float(analysis.get("flops", 0.0)),
            "bytes_accessed": float(analysis.get("bytes accessed", analysis.get("bytes_accessed", 0.0))),
        }
    except Exception:
        return {}
