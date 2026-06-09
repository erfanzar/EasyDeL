# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage-local forward/backward callable compilation for MPMD runtime."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax

from ...core._weakcache import weak_invalidate
from ...core.graph import bind, export
from ...core.module import Module
from ...core.state import State
from ...transforms.jit import jit as spx_jit
from .grad_core import _split_params_rest

_STAGE_CALLABLE_CACHE: dict[int, tuple[Callable[..., object], Callable[..., object], int]] = {}


def _build_stage_callables(
    stage: Module,
    donate_fwd: tuple[int, ...] = (),
    donate_bwd: tuple[int, ...] = (),
) -> tuple[
    Callable[..., object],
    Callable[..., object],
    State,
    State,
    object,
]:
    """Compile forward and unified-backward functions for a stage.

    Two jits per stage:

        * ``fwd_only(params, rest, x) -> y`` — the forward pass.
        * ``bwd_only(params, rest, x, g_y) -> (g_params, g_x)`` — the full
          VJP via :func:`jax.vjp` (``linearize`` is avoided because it
    fails on integer-valued inputs such as token-id embeddings).
    For :class:`~spectrax.runtime.schedules.ZeroBubbleH1`, both
    :attr:`Phase.BWD_I` and :attr:`Phase.BWD_W` call this same jit
    but discard one of the two outputs; XLA's dead-code elimination
    collapses each half-call to roughly half the work of the full
    backward.

    Compared to an earlier implementation with three separate VJP
    jits (``bwd``, ``bwd_i``, ``bwd_w``), this trims tracing cost by
    eliminating the redundant re-tracings of ``stage_fn``.

    ``rest`` and ``gdef`` are passed as explicit arguments to the jit
    (not closure-captured) so JAX's trace cache keys on ``(arg avals)``
    alone — every subsequent call with the same shape signature hits
    the cache, even across distinct ``sxcall`` invocations.

    Returns:
            ``(fwd_only, bwd_only, params, rest, gdef)`` — ``parameters`` /
            ``rest`` are the initial state split; ``gdef`` is the stage's
            :class:`GraphDef` (retained for potential reuse).

    Args:
        stage: Stage value consumed by this operation.
        donate_fwd: Donate fwd value consumed by this operation.
        donate_bwd: Donate bwd value consumed by this operation.
    """
    gdef, state = export(stage)
    params, rest = _split_params_rest(state)

    n_leaves = len(jax.tree.leaves(params))
    cache_key = (id(stage), donate_fwd, donate_bwd)
    cached = _STAGE_CALLABLE_CACHE.get(cache_key)
    if cached is not None:
        cached_fwd, cached_bwd, cached_n_leaves = cached
        if cached_n_leaves == n_leaves:
            return cached_fwd, cached_bwd, params, rest, gdef
        del _STAGE_CALLABLE_CACHE[cache_key]

    if donate_fwd:

        @functools.partial(spx_jit, donate_argnums=donate_fwd)
        def fwd_only(params, rest, x):
            """Run a single stage forward by re-binding ``(params, rest)`` into ``gdef``.

            Args:
                params: The differentiable parameter :class:`State`
                    placed on this rank.
                rest: Non-parameter state (overlaid on ``params``).
                x: Stage input activation.

            Returns:
                The stage's output activation.
            """
            with jax.named_scope("spectrax/mpmd/train/stage_forward"):
                module = bind(gdef, params.overlay(rest))
                return module(x)

    else:

        @spx_jit
        def fwd_only(params, rest, x):
            """Run a single stage forward by re-binding ``(params, rest)`` into ``gdef``.

            Args:
                params: The differentiable parameter :class:`State`
                    placed on this rank.
                rest: Non-parameter state (overlaid on ``params``).
                x: Stage input activation.

            Returns:
                The stage's output activation.
            """
            with jax.named_scope("spectrax/mpmd/train/stage_forward"):
                module = bind(gdef, params.overlay(rest))
                return module(x)

    if donate_bwd:

        @functools.partial(spx_jit, donate_argnums=donate_bwd)
        def bwd_only(params, rest, x, g_y):
            """(g_params, g_x) via :func:`jax.vjp`.

            Uses ``vjp`` instead of ``linearize + linear_transpose`` because
            ``linearize`` fails on stages whose inputs contain integers
            (e.g. an embedding layer taking token IDs). ``vjp`` handles the
            int-to-float boundary correctly.

            Args:
                params: Parameter mapping or primitive parameter dictionary.
                rest: Rest value consumed by this operation.
                x: Input value consumed by the operation.
                g_y: G y value consumed by this operation.
            """

            def stage_fn(p, r, xi):
                """Pure forward closure used as the :func:`jax.vjp` target.

                Re-binds the stage from ``gdef`` on every call so the
                VJP can differentiate through fresh leaves rather than
                the captured originals (necessary because :func:`vjp`
                tracks the identity of its inputs).

                Args:
                    p: P value consumed by this operation.
                    r: R value consumed by this operation.
                    xi: Xi value consumed by this operation.
                """
                return bind(gdef, p.overlay(r))(xi)

            with jax.named_scope("spectrax/mpmd/train/stage_backward"):
                _y, vjp_fn = jax.vjp(stage_fn, params, rest, x)
                g_params, _g_rest, g_x = vjp_fn(g_y)
                return g_params, g_x

    else:

        @spx_jit
        def bwd_only(params, rest, x, g_y):
            """(g_params, g_x) via :func:`jax.vjp`.

            Uses ``vjp`` instead of ``linearize + linear_transpose`` because
            ``linearize`` fails on stages whose inputs contain integers
            (e.g. an embedding layer taking token IDs). ``vjp`` handles the
            int-to-float boundary correctly.

            Args:
                params: Parameter mapping or primitive parameter dictionary.
                rest: Rest value consumed by this operation.
                x: Input value consumed by the operation.
                g_y: G y value consumed by this operation.
            """

            def stage_fn(p, r, xi):
                """Pure forward closure used as the :func:`jax.vjp` target.

                Re-binds the stage from ``gdef`` on every call so the
                VJP can differentiate through fresh leaves rather than
                the captured originals (necessary because :func:`vjp`
                tracks the identity of its inputs).

                Args:
                    p: P value consumed by this operation.
                    r: R value consumed by this operation.
                    xi: Xi value consumed by this operation.
                """
                return bind(gdef, p.overlay(r))(xi)

            with jax.named_scope("spectrax/mpmd/train/stage_backward"):
                _y, vjp_fn = jax.vjp(stage_fn, params, rest, x)
                g_params, _g_rest, g_x = vjp_fn(g_y)
                return g_params, g_x

    _STAGE_CALLABLE_CACHE[cache_key] = (fwd_only, bwd_only, n_leaves)
    weak_invalidate(stage, _STAGE_CALLABLE_CACHE, cache_key)
    return fwd_only, bwd_only, params, rest, gdef
