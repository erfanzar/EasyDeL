# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gradient accumulation and loss JIT helpers for the MPMD runtime."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ...core._weakcache import weak_invalidate
from ...core.state import State
from ...transforms.jit import jit as spx_jit
from .utils.tree import _add_grad, _is_leaf, _scale_grad

_FUSED_FWDBWD_CACHE: dict[tuple[int, int], Callable[..., object]] = {}
_VMAP_LOSS_CACHE: dict[tuple[int, tuple[int, ...]], Callable[..., object]] = {}
_LOSS_JIT_CACHE: dict[tuple[int, bool, tuple[int, ...]], Callable[..., object]] = {}


def _split_params_rest(state: State) -> tuple[State, State]:
    """Partition a :class:`State` into differentiable params and the remainder.

    The MPMD runtime treats the ``"parameters"`` collection as the
    grad-bearing portion and everything else (e.g. RNG state, batch
    norm running stats) as ``rest``. Splitting up front lets each
    stage's forward / backward jits accept ``params`` as the gradient
    target without touching the rest.

    Args:
        state: The full module state.

    Returns:
        ``(params_state, rest_state)`` — each a :class:`State` with
        the corresponding subset of collections.
    """
    raw = state.raw()
    params_raw: dict[str, dict[str, object]] = {}
    rest_raw: dict[str, dict[str, object]] = {}
    for c, d in raw.items():
        (params_raw if c == "parameters" else rest_raw)[c] = dict(d)
    return State(params_raw), State(rest_raw)


def _get_fused_fwd_bwd_jit(
    fwd_jit: Callable[..., object],
    bwd_jit: Callable[..., object],
) -> Callable[..., object]:
    """Return a cached jit that performs a ``(fwd_A, bwd_B)`` pair in one dispatch.

    For 1F1B-family schedules at steady state each rank alternates one
    forward (microbatch ``A``) and one backward (microbatch ``B``).
    Dispatching each as its own jit pays two trace/dispatch costs per
    microbatch; fusing them into one compiled kernel halves that cost
    and lets XLA interleave their HLO for better register reuse.

    The fused jit's signature is::

            (params, rest, x_fwd, x_bwd, g_y_bwd)
                -> (y_fwd, g_params_bwd, g_x_bwd)

        ``x_bwd`` is the saved activation for mb ``B`` (captured during
        its earlier forward); the runtime still manages saved_inputs /
        recv_cots / grad_accum the same way — only the dispatch count
        changes.

    Args:
        fwd_jit: Fwd jit value consumed by this operation.
        bwd_jit: Bwd jit value consumed by this operation.

    Returns:
        Return a cached jit that performs a ``(fwd_A, bwd_B)`` pair in one dispatch.
    """
    key = (id(fwd_jit), id(bwd_jit))
    cached = _FUSED_FWDBWD_CACHE.get(key)
    if cached is not None:
        return cached

    @spx_jit
    def fused(params, rest, x_fwd, x_bwd, g_y_bwd):
        """Run forward on ``x_fwd`` and backward on ``(x_bwd, g_y_bwd)`` in one HLO.

        Args:
            params: Parameter mapping or primitive parameter dictionary.
            rest: Rest value consumed by this operation.
            x_fwd: X fwd value consumed by this operation.
            x_bwd: X bwd value consumed by this operation.
            g_y_bwd: G y bwd value consumed by this operation.
        """
        with jax.named_scope("spectrax/mpmd/train/fused_fwdbwd"):
            with jax.named_scope("spectrax/mpmd/train/fused_fwdbwd/forward"):
                y_fwd = fwd_jit(params, rest, x_fwd)
            with jax.named_scope("spectrax/mpmd/train/fused_fwdbwd/backward"):
                g_params, g_x = bwd_jit(params, rest, x_bwd, g_y_bwd)
            return y_fwd, g_params, g_x

    _FUSED_FWDBWD_CACHE[key] = fused
    weak_invalidate(fwd_jit, _FUSED_FWDBWD_CACHE, key)
    weak_invalidate(bwd_jit, _FUSED_FWDBWD_CACHE, key)
    return fused


def _get_vmap_loss_and_g_y(
    loss_fn: Callable[..., jax.Array],
    donate_argnums: tuple[int, ...] = (),
) -> Callable[..., object]:
    """Return a cached jit that vmaps ``loss_fn`` + ``d_loss/d_y`` over microbatches.

    The wrapper takes ``y_stack`` of shape ``(M, ...)`` plus matching
    target stacks and returns ``(loss_stack, g_y_stack)``. Used by the
    GPipe vmap fast-path to compute every microbatch's loss/cotangent
    in a single device-side launch. Cached on
    ``(id(loss_fn), donate_argnums)`` in :data:`_VMAP_LOSS_CACHE`.

    Args:
        loss_fn: User loss callable ``(y, *targets) -> scalar``.
        donate_argnums: Argnums whose buffers may be donated.

    Returns:
        Jitted ``(y_stack, *t_stack) -> (loss_stack, g_y_stack)``.
    """
    key = (id(loss_fn), donate_argnums)
    cached = _VMAP_LOSS_CACHE.get(key)
    if cached is not None:
        return cached

    if donate_argnums:

        @functools.partial(jax.jit, donate_argnums=donate_argnums)
        def vmap_loss(y_stack, *t_stack):
            """Vmap ``per_mb`` over the leading microbatch axis under one jit.

            Returns ``(loss_stack, g_y_stack)`` with leading axis ``M``;
            both per-mb losses and per-mb cotangents are produced in
            one compiled program so the GPipe fast-path can fuse the
            terminal forward, loss, and backward.

            Args:
                y_stack: Y stack value consumed by this operation.
                *t_stack: Additional positional arguments forwarded to the wrapped callable or backend.
            """

            def per_mb(y_, *t_):
                """Compute ``(loss, d_loss/d_y)`` for a single microbatch slice.

                Wrapped in :func:`jax.vmap` upstream so this body sees
                one microbatch at a time even though the input tensors
                are the full ``(M, ...)`` stacks.

                Args:
                    y_: Y  value consumed by this operation.
                    *t_: Additional positional arguments forwarded to the wrapped callable or backend.
                """
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            with jax.named_scope("spectrax/mpmd/loss/vmap_loss_and_grad_y"):
                return jax.vmap(per_mb)(y_stack, *t_stack)

    else:

        @jax.jit
        def vmap_loss(y_stack, *t_stack):
            """Vmap ``per_mb`` over the leading microbatch axis under one jit.

            Returns ``(loss_stack, g_y_stack)`` with leading axis ``M``;
            both per-mb losses and per-mb cotangents are produced in
            one compiled program so the GPipe fast-path can fuse the
            terminal forward, loss, and backward.

            Args:
                y_stack: Y stack value consumed by this operation.
                *t_stack: Additional positional arguments forwarded to the wrapped callable or backend.
            """

            def per_mb(y_, *t_):
                """Compute ``(loss, d_loss/d_y)`` for a single microbatch slice.

                Wrapped in :func:`jax.vmap` upstream so this body sees
                one microbatch at a time even though the input tensors
                are the full ``(M, ...)`` stacks.

                Args:
                    y_: Y  value consumed by this operation.
                    *t_: Additional positional arguments forwarded to the wrapped callable or backend.
                """
                return jax.value_and_grad(lambda yy: loss_fn(yy, *t_))(y_)

            with jax.named_scope("spectrax/mpmd/loss/vmap_loss_and_grad_y"):
                return jax.vmap(per_mb)(y_stack, *t_stack)

    _VMAP_LOSS_CACHE[key] = vmap_loss
    weak_invalidate(loss_fn, _VMAP_LOSS_CACHE, key)
    return vmap_loss


@jax.jit
def _vmap_sum_grads(g_stack):
    """Sum a per-microbatch gradient stack along its leading axis.

    The GPipe vmap fast-path produces gradients shaped
    ``(M, *param_shape)`` because each microbatch contributes
    independently. Summing along axis 0 collapses the stack into the
    same parameter shape as a serial accumulation would yield.

    Args:
        g_stack: Pytree of arrays whose leading axis indexes
            microbatches.

    Returns:
        Pytree of arrays with the leading axis summed away.
    """
    with jax.named_scope("spectrax/mpmd/grad/vmap_sum"):
        return jax.tree.map(lambda x: x.sum(axis=0), g_stack, is_leaf=_is_leaf)


@jax.jit
def _accumulate_state(acc, add):
    """Module-level cached grad accumulator: ``acc + add`` leaf-wise.

    Defined at module scope so JAX's trace cache hits across every
    ``sxcall`` call and across every stage with matching pytree
    shape — eliminates the per-call re-trace cost that previously
    dominated step time at small batch sizes.

    Args:
        acc: Acc value consumed by this operation.
        add: Add value consumed by this operation.
    """
    with jax.named_scope("spectrax/mpmd/grad/accumulate_state"):
        return jax.tree.map(lambda a, b: a + b, acc, add, is_leaf=_is_leaf)


def _accumulate_grad_tree_impl(acc, add):
    """Add two grad pytrees leaf-wise under a cached jit, preserving ``float0``.

    Module-scope so JAX's trace cache reuses the compiled HLO across
    every :func:`sxcall` invocation that handles the same param tree
    shape.

    Args:
        acc: Running gradient accumulator pytree.
        add: New gradient contribution to fold in.

    Returns:
        ``acc + add`` leaf-wise, with ``float0`` leaves treated as
        additive zero.
    """
    with jax.named_scope("spectrax/mpmd/grad/accumulate_grad_tree"):
        return jax.tree.map(_add_grad, acc, add, is_leaf=_is_leaf)


_accumulate_grad_tree = jax.jit(_accumulate_grad_tree_impl)
_accumulate_grad_tree_donate = jax.jit(_accumulate_grad_tree_impl, donate_argnums=(0,))


@jax.jit
def _scale_grad_tree(state, scalar):
    """Scale every leaf of a grad pytree by ``scalar`` under a cached jit.

    Companion to :func:`_accumulate_grad_tree`; ``float0`` leaves
    pass through unchanged so integer-input branches stay valid.

    Args:
        state: Pytree of grad leaves.
        scalar: Multiplier (typically ``1/M``).

    Returns:
        Grad pytree with each leaf scaled.
    """
    with jax.named_scope("spectrax/mpmd/grad/scale_grad_tree"):
        return jax.tree.map(lambda x: _scale_grad(x, scalar), state, is_leaf=_is_leaf)


@jax.jit
def _zeros_like_state(state):
    """Module-level cached ``zeros_like`` over a State pytree.

    Replaces the per-call ``jax.tree.map(jnp.zeros_like, sp)`` which
    issued one eager dispatch per parameter leaf — ~0.3 ms each on
    TPU, easily 60+ ms per step on medium models.

    Args:
        state: SpectraX state tree or transform state passed into the operation.
    """
    with jax.named_scope("spectrax/mpmd/grad/zeros_like_state"):
        return jax.tree.map(jnp.zeros_like, state, is_leaf=_is_leaf)


@jax.jit
def _scale_state(state, scalar):
    """Multiply every array leaf of ``state`` by ``scalar`` under one jit.

    Used to apply the ``1/M`` mean-loss / mean-grad scaling. Defined at
    module scope so JAX's trace cache hits across every :func:`sxcall`
    invocation with the same state pytree shape.

    Args:
        state: Pytree of arrays (or :class:`State` /
            :class:`Variable`-leaved tree).
        scalar: Multiplicative factor (typically ``1.0 / M``).

    Returns:
        Same pytree structure with every array leaf scaled.
    """
    with jax.named_scope("spectrax/mpmd/grad/scale_state"):
        return jax.tree.map(lambda g: g * scalar, state, is_leaf=_is_leaf)


def _get_loss_and_g_y(
    loss_fn: Callable[..., jax.Array],
    has_aux: bool = False,
    donate_argnums: tuple[int, ...] = (),
) -> Callable[..., object]:
    """Return a jitted ``(y, *targets) -> (loss, grad_wrt_y, [aux])`` for ``loss_fn``.

    When ``has_aux=True``, ``loss_fn`` must return ``(scalar, aux_pytree)``.
    The returned wrapper yields ``(loss, g_y, aux)`` so the caller can
    accumulate aux across microbatches.

    Cached on ``(id(loss_fn), has_aux, donate_argnums)``.

    Args:
        loss_fn: Loss fn value consumed by this operation.
        has_aux: Has aux value consumed by this operation.
        donate_argnums: Donate argnums value consumed by this operation.

    Returns:
        Return a jitted ``(y, *targets) -> (loss, grad_wrt_y, [aux])`` for ``loss_fn``.
    """
    key = (id(loss_fn), has_aux, donate_argnums)
    cached = _LOSS_JIT_CACHE.get(key)
    if cached is not None:
        return cached

    if has_aux:
        if donate_argnums:

            @functools.partial(jax.jit, donate_argnums=donate_argnums)
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y, aux)`` for an aux-returning loss.

                The auxiliary pytree is passed through unchanged so the
                caller can accumulate it across microbatches without
                running a second pass through the loss.

                Args:
                    y: Secondary input value consumed by the operation.
                    *targets: Additional positional arguments forwarded to the wrapped callable or backend.
                """

                def local_loss(y_):
                    """Loss closure used by :func:`jax.value_and_grad`; returns ``(scalar, aux)``.

                    Args:
                        y_: Y  value consumed by this operation.
                    """
                    return loss_fn(y_, *targets)

                with jax.named_scope("spectrax/mpmd/loss/loss_and_grad_y_aux"):
                    (loss_val, aux), g_y = jax.value_and_grad(local_loss, has_aux=True)(y)
                    return loss_val, g_y, aux

        else:

            @jax.jit
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y, aux)`` for an aux-returning loss.

                The auxiliary pytree is passed through unchanged so the
                caller can accumulate it across microbatches without
                running a second pass through the loss.

                Args:
                    y: Secondary input value consumed by the operation.
                    *targets: Additional positional arguments forwarded to the wrapped callable or backend.
                """

                def local_loss(y_):
                    """Loss closure used by :func:`jax.value_and_grad`; returns ``(scalar, aux)``.

                    Args:
                        y_: Y  value consumed by this operation.
                    """
                    return loss_fn(y_, *targets)

                with jax.named_scope("spectrax/mpmd/loss/loss_and_grad_y_aux"):
                    (loss_val, aux), g_y = jax.value_and_grad(local_loss, has_aux=True)(y)
                    return loss_val, g_y, aux

    else:
        if donate_argnums:

            @functools.partial(jax.jit, donate_argnums=donate_argnums)
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y)`` for a plain scalar-loss callable.

                ``targets`` are bound at call time; the returned grad
                is taken with respect to ``y`` only.

                Args:
                    y: Secondary input value consumed by the operation.
                    *targets: Additional positional arguments forwarded to the wrapped callable or backend.
                """

                def local_loss(y_):
                    """Scalar loss closure passed to :func:`jax.value_and_grad`.

                    Args:
                        y_: Y  value consumed by this operation.
                    """
                    return loss_fn(y_, *targets)

                with jax.named_scope("spectrax/mpmd/loss/loss_and_grad_y"):
                    return jax.value_and_grad(local_loss)(y)

        else:

            @jax.jit
            def loss_and_g_y(y, *targets):
                """Return ``(loss, d_loss/d_y)`` for a plain scalar-loss callable.

                ``targets`` are bound at call time; the returned grad
                is taken with respect to ``y`` only.

                Args:
                    y: Secondary input value consumed by the operation.
                    *targets: Additional positional arguments forwarded to the wrapped callable or backend.
                """

                def local_loss(y_):
                    """Scalar loss closure passed to :func:`jax.value_and_grad`.

                    Args:
                        y_: Y  value consumed by this operation.
                    """
                    return loss_fn(y_, *targets)

                with jax.named_scope("spectrax/mpmd/loss/loss_and_grad_y"):
                    return jax.value_and_grad(local_loss)(y)

    _LOSS_JIT_CACHE[key] = loss_and_g_y
    weak_invalidate(loss_fn, _LOSS_JIT_CACHE, key)
    return loss_and_g_y
