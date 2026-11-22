# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training with parameters *stored* in a narrow float, for memory.

This is a different goal from the rest of this package. Quantization-aware
training keeps full-precision master weights and makes the forward pass
carry quantization error, so it costs memory rather than saving it. Storing
the parameters narrow is the opposite trade: the master weights themselves
are the narrow values, so parameter memory halves, and the difficulty moves
to keeping the optimizer able to move them at all.

Integers cannot be used this way — ``jax.grad`` refuses a non-inexact
dtype, so an int8-stored weight has no gradient. Narrow *floats* do have
one, which is what makes fp8 storage trainable where int8 storage is not.

Three things then break, and all three have to be handled together:

**Gradients underflow.** The cotangent is produced in the parameter dtype.
Measured on an 8-layer Llama with ``float8_e4m3b11fnuz`` weights, 93% of
gradient elements arrive as exactly zero. Scaling the loss before the
backward pass and unscaling in a wider type afterwards
(:func:`scale_loss_for_narrow_gradients`) moves them back into range.

**Optimizer moments overflow or collapse.** Adam's second moment is a
square, which underflows harder than the gradient did, and ``1 / sqrt(0)``
is what produces the NaN — observed at step 6 with fp8 moments. The
moments must live in a wider dtype; :func:`narrow_storage_update` does the
whole update wide and narrows only the resulting weight.

**Updates round away.** A narrow float has around three mantissa bits, so
an update smaller than the local spacing rounds straight back to where it
started and is lost — permanently, since it is lost identically on every
later step too. :func:`stochastic_round` keeps those updates in
expectation.

**Measured quality cost, and it is large.** On an 8-layer Llama trained on
a non-degenerate objective (random labels, plain Adam at 3e-4, 60 steps),
starting from loss ~8.7:

===================================  ==========  ============
storage                              parameters  loss after 60
===================================  ==========  ============
bfloat16                             1536 MB     0.002
float8_e4m3b11fnuz, round-to-nearest 1280 MB     6.979
float8_e4m3b11fnuz, stochastic       1280 MB     5.509
===================================  ==========  ============

Stochastic rounding is worth a lot relative to round-to-nearest, and
neither is close to bfloat16 — the bfloat16 run learns the batch while
both fp8 runs barely move. Three mantissa bits give roughly 12% relative
spacing, and an Adam update at a typical learning rate is far below that,
so stochastic rounding is converting a systematic loss into a large
unbiased noise term rather than recovering the signal.

Read that before reaching for this. Parameter memory halves and training
is stable, but convergence is materially worse, and the optimizer moments
are usually the larger pool anyway — narrowing *those* to bfloat16 costs
much less quality per byte saved. An earlier version of this note claimed
stochastic rounding closed the gap entirely; that was measured on a
degenerate task whose loss reaches exactly zero for reasons unrelated to
the dtype, and it is not true.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType

__all__ = [
    "narrow_storage_update",
    "scale_loss_for_narrow_gradients",
    "stochastic_round",
    "suggested_loss_scale",
]


def stochastic_round(x: Array, dtype: DType, key: Array) -> Array:
    """Cast to a narrow float, rounding up or down in proportion to position.

    Round-to-nearest discards any update smaller than half the spacing
    between neighbouring representable values. That is not a one-step
    rounding error: the same weight receives a similar update next step and
    loses it again, so the parameter is pinned and training plateaus.

    Stochastic rounding instead lands on the upper neighbour with
    probability equal to how far between the two the value sits, so the
    update survives in expectation. It is implemented by perturbing by up
    to half the local spacing before an ordinary cast; for a float the
    spacing is relative, ``|x| * eps``.

    Args:
        x: Values in a wider dtype.
        dtype: Narrow floating-point dtype to store them in.
        key: PRNG key for this cast. Use a fresh key each step — reusing
            one reintroduces a fixed bias.

    Returns:
        ``x`` cast to ``dtype``, unbiased in expectation.

    Raises:
        ValueError: If ``dtype`` is not a floating-point type. Integer
            storage cannot be trained: ``jax.grad`` rejects integer
            parameters outright.
    """
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(
            f"stochastic_round targets narrow floats, got {jnp.dtype(dtype).name}. Integer-stored parameters "
            f"have no gradient at all -- jax.grad rejects non-inexact dtypes -- so they cannot be trained "
            f"directly regardless of rounding."
        )
    spacing = jnp.abs(x) * float(jnp.finfo(dtype).eps)
    noise = jax.random.uniform(key, x.shape, x.dtype, minval=-0.5, maxval=0.5)
    return (x + noise * spacing).astype(dtype)


def suggested_loss_scale(dtype: DType, max_scale: float = 32768.0) -> float:
    """Suggest a starting loss scale for a narrow gradient dtype.

    Sizes the scale off the dtype's *dynamic range* rather than its
    underflow floor alone, placing gradients near the middle of the
    representable band in log space. Scaling only against the floor
    over-scales badly on the narrow-exponent types: for
    ``float8_e4m3b11fnuz``, whose largest finite value is 30, that approach
    suggested 8192 and produced intermittent overflow spikes mid-run.

    A static scale can always overflow when gradients grow. This is a
    starting point; for a real run prefer eformer's ``DynamicLossScale``,
    which adapts and backs off when it sees a non-finite gradient.

    Args:
        dtype: The parameter storage dtype the gradients inherit.
        max_scale: Upper clamp, so wide dtypes with an enormous dynamic
            range do not produce an absurd factor.

    Returns:
        A power-of-two loss scale in ``[1, max_scale]``.
    """
    info = jnp.finfo(dtype)
    dynamic_range = float(info.max) / float(info.tiny)
    centred = dynamic_range**0.5
    clamped = min(max(centred, 1.0), max_scale)
    return float(2.0 ** int(jnp.floor(jnp.log2(jnp.asarray(clamped)))))


def scale_loss_for_narrow_gradients(loss_fn: Callable[..., Array], loss_scale: float) -> Callable[..., Array]:
    """Wrap a loss so its backward pass runs at a larger magnitude.

    The cotangent is produced in the parameter dtype, so with narrow
    storage most of it underflows to zero before anything downstream can
    widen it. Scaling has to happen *before* the backward pass, which means
    scaling the loss itself; :func:`narrow_storage_update` removes the
    factor again once the gradient is in a wider dtype.

    Args:
        loss_fn: The scalar loss function.
        loss_scale: Factor to multiply the loss by.

    Returns:
        The wrapped loss function.
    """

    def scaled(*args: Any, **kwargs: Any) -> Array:
        """Return the loss multiplied by the scale.

        Args:
            *args: Forwarded to the wrapped loss.
            **kwargs: Forwarded to the wrapped loss.

        Returns:
            The scaled loss.
        """
        return loss_fn(*args, **kwargs) * loss_scale

    return scaled


def narrow_storage_update(
    params: Any,
    grads: Any,
    opt_state: Any,
    optimizer: Any,
    *,
    loss_scale: float = 1.0,
    key: Array | None = None,
    compute_dtype: DType = jnp.float32,
) -> tuple[Any, Any]:
    """Apply one optimizer step to narrow-stored parameters.

    Everything except the stored weight happens in ``compute_dtype``: the
    gradient is widened and unscaled, the optimizer moments are kept wide,
    the update is applied wide, and only the final weight is narrowed
    again. Doing any of it in the storage dtype is what produces the NaN.

    Args:
        params: Parameter pytree in the narrow storage dtype.
        grads: Gradient pytree, as produced against ``params`` and
            therefore also narrow and scaled by ``loss_scale``.
        opt_state: Optimizer state, whose moments must already be in
            ``compute_dtype`` — initialize it against a widened copy of the
            parameters.
        optimizer: An optax-style optimizer exposing ``update``.
        loss_scale: The factor the loss was multiplied by; divided out here.
        key: PRNG key enabling :func:`stochastic_round` for the cast back.
            ``None`` uses round-to-nearest, which loses sub-spacing updates
            permanently and is only appropriate for a wide storage dtype.
        compute_dtype: Dtype for the moments and the update arithmetic.

    Returns:
        ``(new_params, new_opt_state)`` with ``new_params`` back in the
        original storage dtype.
    """
    wide_grads = jax.tree.map(lambda g: g.astype(compute_dtype) / loss_scale, grads)
    wide_params = jax.tree.map(lambda p: p.astype(compute_dtype), params)

    updates, opt_state = optimizer.update(wide_grads, opt_state, wide_params)
    wide_new = jax.tree.map(lambda p, u: p + u, wide_params, updates)

    if key is None:
        new_params = jax.tree.map(lambda w, p: w.astype(p.dtype), wide_new, params)
    else:
        leaves, treedef = jax.tree.flatten(wide_new)
        keys = jax.random.split(key, len(leaves))
        narrowed = [
            stochastic_round(wide, original.dtype, subkey)
            if jnp.issubdtype(original.dtype, jnp.floating) and jnp.finfo(original.dtype).bits < 32
            else wide.astype(original.dtype)
            for wide, original, subkey in zip(leaves, jax.tree.leaves(params), keys, strict=True)
        ]
        new_params = jax.tree.unflatten(treedef, narrowed)
    return new_params, opt_state
