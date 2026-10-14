# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FP8 training primitives and layers.

Implements *delayed-scaling* per-tensor FP8 quantization with
amax-history tracking — the same algorithm used by
NVIDIA Transformer Engine — layered on
spectrax's native :class:`~spectrax.Module` / :class:`~spectrax.Variable`
primitives.

The high-level picture: every FP8-quantized matmul owns six metadata
cells — a scale factor and a rolling amax history for each of the
input activation, the kernel, and the output gradient. Forward calls
quantize the input and kernel through :data:`jax.numpy.float8_e4m3fn`
(E4M3 — balanced range for activations/weights); the backward pass
quantizes the incoming cotangent through :data:`jax.numpy.float8_e5m2`
(E5M2 — larger range, better for gradients). After each call the
amax/scale pair for each tensor is refreshed so the next step's
quantization stays inside fp8's representable range.

spectrax does *not* smuggle the meta updates through a
gradient slot (``OVERWRITE_WITH_GRADIENT``). The six metadata cells
live in a dedicated collection ``"fp8_meta"`` and are mutated directly
by the forward path (for input/kernel) and by the output-side
:func:`out_qdq` custom VJP (for the gradient scale). The user opts
in to those writes by declaring the collection mutable on their
transform::

    class Mlp(spx.Module):
        def __init__(self, d_in, d_hidden, rngs):
            super().__init__()
            self.fc1 = spx_nn.Fp8Linear(d_in, d_hidden, rngs=rngs)
            self.fc2 = spx_nn.Fp8Linear(d_hidden, d_in, rngs=rngs)

        def forward(self, x):
            return self.fc2(jax.nn.relu(self.fc1(x)))

    @spx.jit(mutable="fp8_meta")
    def train_step(model, x, y):
        def loss_fn(m):
            return ((m(x) - y) ** 2).mean()
        return spx.grad(loss_fn)(model)

Public surface exposed by this module:

* Primitives: :func:`quantize`, :func:`dequantize`, :func:`qdq`,
  :func:`compute_amax_history`, :func:`compute_scale`,
  :func:`update_fp8_meta`, :func:`quantize_dequantize`,
  :func:`in_qdq`, :func:`out_qdq`.
* :class:`Fp8Meta` — :class:`~spectrax.Variable` subclass owning the
  scale/history cells.
* :class:`Fp8DotGeneral` — the reusable :func:`~jax.lax.dot_general`
  wrapped in the full qdq machinery.
* :class:`Fp8Linear` — drop-in replacement for :class:`~spectrax.nn.Linear`.
* :class:`Fp8Einsum` — drop-in for :class:`~spectrax.nn.Einsum`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
from jax import lax

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import Parameter, Variable
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs

__all__ = [
    "Fp8DotGeneral",
    "Fp8Einsum",
    "Fp8Linear",
    "Fp8Meta",
    "compute_amax_history",
    "compute_scale",
    "dequantize",
    "in_qdq",
    "out_qdq",
    "qdq",
    "quantize",
    "quantize_dequantize",
    "update_fp8_meta",
]


class Fp8Meta(Variable):
    """Variable cell owning a scale factor or amax-history buffer.

    Defaults to the ``"fp8_meta"`` collection so ``spx.jit(mutable="fp8_meta")``
    lets :class:`Fp8DotGeneral` refresh the meta during forward and
    backward without the mutation being rejected as illegal. Each
    :class:`Fp8DotGeneral` instance owns six :class:`Fp8Meta` cells
    (three scales, three amax histories).
    """

    default_kind: ClassVar[str] = "fp8_meta"


def get_fp8_max(fp8_dtype: DType, out_dtype: DType) -> Array:
    """Return the largest finite value representable in ``fp8_dtype``.

    Thin wrapper around :func:`jax.numpy.finfo` that returns the value
    cast into ``out_dtype`` so it can participate directly in math with
    fp32 / bf16 / fp16 scale tensors without further promotion.

    Args:
        fp8_dtype: An fp8 dtype — typically :data:`jax.numpy.float8_e4m3fn`
            or :data:`jax.numpy.float8_e5m2`.
        out_dtype: Dtype to cast the max constant into.

    Returns:
        A 0-d :class:`jax.Array` holding ``finfo(fp8_dtype).max`` as
        ``out_dtype``.
    """
    return jnp.finfo(fp8_dtype).max.astype(out_dtype)


def quantize(x: ArrayLike, q_dtype: DType, scale: Array, compute_dtype: DType) -> Array:
    """Scale, clip, and cast ``x`` into an fp8 dtype.

    Implements ``round_to(fp8)(clip(x / scale, [-fp8_max, fp8_max]))``.
    Division by ``scale`` is done in ``compute_dtype`` (the surrounding
    math dtype, typically fp32 or bf16) so the quantize step does not
    round twice. The clamp is crucial: without it, values larger than
    the fp8 max would saturate to NaN on cast.

    Args:
        x: Tensor to quantize.
        q_dtype: Target fp8 dtype (E4M3 or E5M2).
        scale: Per-tensor scale, shape ``(1,)``. Divides ``x`` so that
            the scaled tensor's dynamic range fits fp8's.
        compute_dtype: Math dtype used for the divide/clip; should match
            the precision of the surrounding matmul.

    Returns:
        ``x`` as ``q_dtype``.
    """
    dtype_max = get_fp8_max(q_dtype, compute_dtype)
    scaled = jnp.asarray(x) / jnp.broadcast_to(scale.astype(compute_dtype), jnp.shape(x))
    clipped = jnp.clip(scaled, -dtype_max, dtype_max)
    return clipped.astype(q_dtype)


def dequantize(x: ArrayLike, dq_dtype: DType, scale: Array) -> Array:
    """Inverse of :func:`quantize`: cast back to ``dq_dtype`` and rescale.

    Applied symmetrically with :func:`quantize` so that ``dequantize(quantize(x))``
    is a rounded copy of ``x`` (never restores more precision than fp8
    can represent).

    Args:
        x: Fp8-dtyped tensor to restore.
        dq_dtype: Dtype to cast back to (usually the original math dtype).
        scale: The same scale used by :func:`quantize`; multiplied to
            reverse the divide.

    Returns:
        ``x`` as ``dq_dtype``, rescaled.
    """
    xa = jnp.asarray(x)
    return xa.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), xa.shape)


def qdq(x: ArrayLike, q_dtype: DType, scale: Array, compute_dtype: DType) -> Array:
    """Quantize then immediately dequantize — round-trip ``x`` through fp8.

    The output has the same dtype as ``x`` but only values representable
    in ``q_dtype`` at the given scale. Used inside the custom VJPs to
    create the "fake-quant" behaviour: math still runs in the higher
    precision, but the tensor behaves as if it had been sent through
    fp8 hardware.

    Args:
        x: Tensor to fake-quantize.
        q_dtype: Target fp8 dtype.
        scale: Per-tensor scale, shape ``(1,)``.
        compute_dtype: Math dtype for the intermediate divide/clip.

    Returns:
        A tensor of the same dtype as ``x`` but restricted to values
        representable in ``q_dtype`` at the given scale.
    """
    qx = quantize(x, q_dtype, scale, compute_dtype)
    return dequantize(qx, jnp.asarray(x).dtype, scale)


def compute_amax_history(x: ArrayLike, amax_history: Array) -> Array:
    """Roll the history one step and record ``max(|x|)`` at index 0.

    Implements a FIFO of recent maximum-absolute values sliding left
    (the oldest sample wraps to the tail via :func:`jax.numpy.roll`
    with ``shift=-1``) and the newly computed amax is dropped into
    index 0. :func:`compute_scale` uses ``max`` of this history as
    the representative amax for the next scale update.

    Args:
        x: Tensor whose amax should be pushed onto the history.
        amax_history: 1-d fp32 array of length :attr:`Fp8DotGeneral.amax_history_length`.

    Returns:
        The new history array, same shape and dtype as the input.
    """
    amax_update = jnp.max(jnp.abs(jnp.asarray(x))).astype(amax_history.dtype)
    return jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)


def compute_scale(amax: Array, scale: Array, fp8_max: Array, margin: int = 0) -> Array:
    """Pick the new scale so that future quantizations fill fp8's range.

    The target is ``fp8_max / amax / 2**margin`` — the scale that would
    push the previous amax exactly to fp8's maximum representable
    value (with an optional safety ``margin`` exponent). When ``amax``
    is non-positive or non-finite the previous scale is preserved;
    this matches the Transformer Engine reference implementation.

    Args:
        amax: Representative amax for the tensor.
        scale: Current per-tensor scale, shape ``(1,)``.
        fp8_max: Maximum finite value representable in the target
            fp8 dtype — typically computed via :func:`get_fp8_max`.
        margin: Extra safety bits (divides the ideal scale by ``2**margin``).
            Default ``0`` picks the tight fit.

    Returns:
        The new scale, shape ``(1,)``.
    """
    inv = 1.0 / scale
    sf = (fp8_max / amax) / (2**margin)
    sf = jnp.where(amax > 0.0, sf, inv)
    sf = jnp.where(jnp.isfinite(amax), sf, inv)
    return 1.0 / sf


def update_fp8_meta(x: ArrayLike, q_dtype: DType, scale: Array, amax_history: Array) -> tuple[Array, Array]:
    """Refresh the ``(scale, amax_history)`` pair for a tensor.

    Convenience that threads :func:`compute_scale` and
    :func:`compute_amax_history` in the canonical order (scale first
    off the current history, then roll ``x``'s amax onto the history).

    Args:
        x: Tensor whose amax should be folded into the history.
        q_dtype: Target fp8 dtype — used to look up ``fp8_max``.
        scale: Current scale, shape ``(1,)``.
        amax_history: Current amax history.

    Returns:
        A ``(new_scale, new_history)`` pair.
    """
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    new_amax = jnp.max(amax_history, axis=0)
    new_scale = compute_scale(new_amax, scale, dtype_max)
    new_history = compute_amax_history(x, amax_history)
    return new_scale, new_history


def quantize_dequantize(
    x: ArrayLike, q_dtype: DType, scale: Array, amax_history: Array, compute_dtype: DType
) -> tuple[Array, Array, Array]:
    """Update metas, then round-trip ``x`` through fp8.

    Returns both the fake-quantized tensor *and* the refreshed meta
    pair so callers can thread the metas back into their state (that
    is how the custom VJPs smuggle meta updates through the backward
    pass).

    Args:
        x: Tensor to fake-quantize.
        q_dtype: Target fp8 dtype.
        scale: Current per-tensor scale.
        amax_history: Current amax history.
        compute_dtype: Math dtype for the fake-quant arithmetic.

    Returns:
        A triple ``(qdq_x, new_scale, new_history)``.
    """
    new_scale, new_history = update_fp8_meta(x, q_dtype, scale, amax_history)
    qdq_x = qdq(x, q_dtype, new_scale, compute_dtype)
    return qdq_x, new_scale, new_history


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def in_qdq(compute_dtype: DType, q_dtype: DType, x: ArrayLike, scale: Array, amax_history: Array) -> Array:
    """Forward-direction fake-quant with a straight-through gradient.

    The public entry point; registered as a :func:`jax.custom_vjp` so
    that: the *forward* pass quantizes and dequantizes ``x`` (rounding
    through fp8), and the *backward* pass passes the upstream gradient
    through unchanged (STE — standard practice for fake-quant). The
    refreshed ``(scale, history)`` are returned as the gradient for
    those arguments so a surrounding :func:`spx.grad`/:func:`spx.jit`
    can propagate them back to the live :class:`Fp8Meta` cells.

    Args:
        compute_dtype: Math dtype used inside the quantize step.
        q_dtype: Target fp8 dtype (typically E4M3 for the forward path).
        x: Tensor to fake-quantize.
        scale: Current scale.
        amax_history: Current amax history.

    Returns:
        Fake-quantized ``x`` in its original dtype.
    """
    qx, _, _ = quantize_dequantize(x, q_dtype, scale, amax_history, compute_dtype)
    return qx


def _in_qdq_fwd(compute_dtype, q_dtype, x, scale, amax_history):
    """Custom-VJP forward rule for :func:`in_qdq`.

    Runs :func:`quantize_dequantize` and stashes the refreshed
    ``(scale, history)`` pair as residuals so the backward rule can
    return them as the gradient for the scale / history arguments.

    Args:
        compute_dtype: Math dtype passed through from :func:`in_qdq`.
        q_dtype: Target fp8 dtype.
        x: Tensor to fake-quantize.
        scale: Current scale.
        amax_history: Current amax history.

    Returns:
        ``(qdq_x, (new_scale, new_history))`` — the fake-quantized
        primal and the residual carried into the backward pass.
    """
    qx, new_scale, new_history = quantize_dequantize(x, q_dtype, scale, amax_history, compute_dtype)
    return qx, (new_scale, new_history)


def _in_qdq_bwd(compute_dtype, q_dtype, res, g):
    """Custom-VJP backward rule for :func:`in_qdq`.

    Implements a straight-through estimator on ``x`` (the cotangent
    is passed through unchanged) while emitting the refreshed metas
    as cotangents for ``scale`` / ``amax_history`` so a surrounding
    :func:`spx.grad` can write them back to the live :class:`Fp8Meta`
    cells.

    Args:
        compute_dtype: Forwarded from the primal call (unused).
        q_dtype: Forwarded from the primal call (unused).
        res: ``(new_scale, new_history)`` residual from
            :func:`_in_qdq_fwd`.
        g: Upstream cotangent on the fake-quantized tensor.

    Returns:
        ``(g, new_scale, new_history)`` — gradient for ``x`` (STE),
        and refreshed metas as the gradients for ``scale`` and
        ``amax_history``.
    """
    new_scale, new_history = res
    return g, new_scale, new_history


in_qdq.defvjp(_in_qdq_fwd, _in_qdq_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def out_qdq(compute_dtype: DType, q_dtype: DType, y: ArrayLike, scale: Array, amax_history: Array) -> Array:
    """Backward-direction fake-quant: identity forward, quantized cotangent backward.

    Inserted on the output path of an fp8 matmul. Forward is an
    identity — ``y`` flows through unchanged, so the computation graph
    is the same as without fp8. Backward runs
    :func:`quantize_dequantize` on the incoming cotangent (using
    ``q_dtype`` — typically E5M2, whose wider range suits gradients)
    and returns the refreshed meta pair as the gradient for scale /
    history, just like :func:`in_qdq`.

    Args:
        compute_dtype: Math dtype used during the backward qdq.
        q_dtype: Target fp8 dtype for the cotangent (typically E5M2).
        y: Forward output tensor (passed through untouched).
        scale: Current output-gradient scale.
        amax_history: Current output-gradient amax history.

    Returns:
        ``y`` unchanged.
    """
    return jnp.asarray(y)


def _out_qdq_fwd(compute_dtype, q_dtype, y, scale, amax_history):
    """Custom-VJP forward rule for :func:`out_qdq`.

    Identity on the primal — ``y`` flows through unchanged. The
    current ``(scale, amax_history)`` is stashed as the residual so
    the backward rule can fold the cotangent's amax into them.

    Args:
        compute_dtype: Forwarded; unused at forward time.
        q_dtype: Forwarded; unused at forward time.
        y: Forward output tensor.
        scale: Current output-gradient scale.
        amax_history: Current output-gradient amax history.

    Returns:
        ``(y, (scale, amax_history))``.
    """
    return y, (scale, amax_history)


def _out_qdq_bwd(compute_dtype, q_dtype, res, g):
    """Custom-VJP backward rule for :func:`out_qdq`.

    Runs :func:`quantize_dequantize` on the incoming cotangent
    (typically using the wide-range E5M2 dtype) and returns the
    refreshed meta pair as the gradient for ``scale`` / ``history``.

    Args:
        compute_dtype: Math dtype for the qdq arithmetic.
        q_dtype: Target fp8 dtype for the cotangent.
        res: ``(scale, amax_history)`` from :func:`_out_qdq_fwd`.
        g: Upstream cotangent.

    Returns:
        ``(qg, new_scale, new_history)`` — the qdq'd cotangent, plus
        refreshed metas to back-propagate into the
        :class:`Fp8Meta` cells.
    """
    scale, amax_history = res
    qg, new_scale, new_history = quantize_dequantize(g, q_dtype, scale, amax_history, compute_dtype)
    return qg, new_scale, new_history


out_qdq.defvjp(_out_qdq_fwd, _out_qdq_bwd)


class Fp8DotGeneral(Module):
    """FP8-quantized :func:`jax.lax.dot_general` with scale/amax tracking.

    Owns six :class:`Fp8Meta` cells — scale + amax history for each of
    the input activation, the kernel, and the output gradient — updated
    in place on every forward/backward pass. Calling the instance
    dispatches to :func:`jax.lax.dot_general` with both operands
    fake-quantized through :class:`jax.numpy.float8_e4m3fn` (E4M3),
    and wraps the output in :func:`out_qdq` so the backward pass
    quantizes the cotangent through :class:`jax.numpy.float8_e5m2`
    (E5M2).

    Meta updates require the ``"fp8_meta"`` collection to be declared
    mutable on the surrounding transform::

        @spx.jit(mutable="fp8_meta")
        def step(m, x):
            ...

    Args:
        amax_history_length: Number of past amax samples to keep per
            tensor. Larger values produce smoother scale updates but
            react slower to distribution shifts. Default ``1024``.
        e4m3_dtype: FP8 dtype for activations/weights (forward path).
            Default :data:`jax.numpy.float8_e4m3fn`.
        e5m2_dtype: FP8 dtype for gradients (backward path). Default
            :data:`jax.numpy.float8_e5m2`.

    Call signature::

        op(lhs, rhs, dimension_numbers, precision=None, compute_dtype=None)

    Mirrors :func:`jax.lax.dot_general`. ``compute_dtype`` defaults to
    the lhs's dtype and is the dtype used for the matmul itself (the
    fp8 operands are dequantized back to this dtype before the dot).
    """

    input_scale: Fp8Meta
    kernel_scale: Fp8Meta
    output_grad_scale: Fp8Meta
    input_amax_history: Fp8Meta
    kernel_amax_history: Fp8Meta
    output_grad_amax_history: Fp8Meta

    def __init__(
        self,
        *,
        amax_history_length: int = 1024,
        e4m3_dtype: DType = jnp.float8_e4m3fn,
        e5m2_dtype: DType = jnp.float8_e5m2,
    ) -> None:
        """Allocate the six fp8 meta cells and stash the dtype choices.

        Initializes:

        * ``input_scale`` / ``kernel_scale`` / ``output_grad_scale``
          — shape ``(1,)``, dtype ``float32``, value ``1.0`` (no
          rescaling on the first call).
        * ``input_amax_history`` / ``kernel_amax_history`` /
          ``output_grad_amax_history`` — shape
          ``(amax_history_length,)``, dtype ``float32``, value ``0``
          (the rolling FIFO grows in over the first calls).

        The fp8 dtypes are stashed as their :func:`jax.numpy.dtype`
        *names* (strings like ``"float8_e4m3fn"``) rather than the raw
        dtype objects — dtype classes are hashable but are not
        accepted by :class:`~spectrax.Module`'s static-scalar rule,
        whereas strings are first-class static fields that survive
        :func:`~spectrax.export` / :func:`~spectrax.bind` round-trips.

        Args:
            amax_history_length: Length of the rolling amax history
                kept per tensor. Larger values smooth the scale
                updates; smaller values react faster to distribution
                shifts. Default ``1024``.
            e4m3_dtype: FP8 dtype for the forward path (activations
                / weights). Default :data:`jax.numpy.float8_e4m3fn`.
            e5m2_dtype: FP8 dtype for the backward path (cotangents).
                Default :data:`jax.numpy.float8_e5m2`.
        """
        super().__init__()
        self.amax_history_length = int(amax_history_length)
        self.e4m3_name = jnp.dtype(e4m3_dtype).name
        self.e5m2_name = jnp.dtype(e5m2_dtype).name

        scale_shape = (1,)
        hist_shape = (self.amax_history_length,)
        self.input_scale = Fp8Meta(_ones_scale(scale_shape, jnp.float32))
        self.kernel_scale = Fp8Meta(_ones_scale(scale_shape, jnp.float32))
        self.output_grad_scale = Fp8Meta(_ones_scale(scale_shape, jnp.float32))
        self.input_amax_history = Fp8Meta(jnp.zeros(hist_shape, jnp.float32))
        self.kernel_amax_history = Fp8Meta(jnp.zeros(hist_shape, jnp.float32))
        self.output_grad_amax_history = Fp8Meta(jnp.zeros(hist_shape, jnp.float32))

    def forward(
        self,
        lhs: ArrayLike,
        rhs: ArrayLike,
        dimension_numbers: tuple,
        precision: lax.PrecisionLike = None,
        compute_dtype: DType | None = None,
    ) -> Array:
        """Fake-quantize both operands and run :func:`jax.lax.dot_general`.

        Flow, in order: (1) round-trip ``lhs`` and ``rhs`` through
        :class:`jax.numpy.float8_e4m3fn` via :func:`in_qdq`; (2)
        eagerly refresh the input/kernel meta cells so pure-forward
        inference (no :func:`spx.grad` outside) also gets correct
        scaling on the next call; (3) dispatch to
        :func:`jax.lax.dot_general` with the dequantized operands;
        (4) route the output through :func:`out_qdq` so the backward
        pass quantizes the incoming cotangent through
        :class:`jax.numpy.float8_e5m2`.

        The meta write in step 2 is what requires
        ``mutable="fp8_meta"`` on the surrounding transform.

        Args:
            lhs: Left-hand operand (typically the activation).
            rhs: Right-hand operand (typically the kernel).
            dimension_numbers: Contraction / batch dim spec — same
                shape as ``jax.lax.dot_general``'s argument.
            precision: Forwarded to :func:`jax.lax.dot_general`.
            compute_dtype: Dtype used for the dot itself. Defaults to
                ``lhs.dtype``.

        Returns:
            The dot result, potentially wrapped for cotangent-side qdq.
        """
        lhs = jnp.asarray(lhs)
        rhs = jnp.asarray(rhs)
        comp = compute_dtype if compute_dtype is not None else lhs.dtype
        e4m3 = jnp.dtype(self.e4m3_name)
        e5m2 = jnp.dtype(self.e5m2_name)

        q_lhs = in_qdq(comp, e4m3, lhs, self.input_scale.value, self.input_amax_history.value)
        q_rhs = in_qdq(comp, e4m3, rhs, self.kernel_scale.value, self.kernel_amax_history.value)
        self.input_scale.value, self.input_amax_history.value = update_fp8_meta(
            lhs, e4m3, self.input_scale.value, self.input_amax_history.value
        )
        self.kernel_scale.value, self.kernel_amax_history.value = update_fp8_meta(
            rhs, e4m3, self.kernel_scale.value, self.kernel_amax_history.value
        )

        y = lax.dot_general(q_lhs, q_rhs, dimension_numbers, precision=precision)
        y = out_qdq(comp, e5m2, y, self.output_grad_scale.value, self.output_grad_amax_history.value)
        return cast(Array, y)


def _ones_scale(shape: tuple[int, ...], dtype: DType) -> Array:
    """Return ``jnp.ones(shape, dtype)`` — the canonical initial scale factor.

    A scale of ``1`` means "no rescaling applied" and is safe even when
    the first call sees values far from fp8's representable range —
    :func:`compute_scale` picks a real value on the first update.

    Args:
        shape: Array shape requested by the initializer or helper.
        dtype: Array dtype requested for the produced value.

    Returns:
        Return ``jnp.ones(shape, dtype)`` — the canonical initial scale factor.
    """
    return jnp.ones(shape, dtype=dtype)


class Fp8Linear(Module):
    """Dense layer routed through :class:`Fp8DotGeneral`.

    API-compatible with :class:`~spectrax.nn.Linear`. State layout:

    * ``weight`` in the ``"parameters"`` collection (shape ``(in, out)``).
    * ``bias`` in the ``"parameters"`` collection when ``use_bias=True``.
    * Six :class:`Fp8Meta` cells under ``qdot.*`` in the ``"fp8_meta"``
      collection, managed by the embedded :class:`Fp8DotGeneral`.

    Forward computes ``qdot(x, W) + b`` where the ``qdot`` both
    fake-quantizes its operands and wraps the output cotangent for
    E5M2 qdq on the backward pass.

    Args:
        in_features: Trailing input feature count.
        out_features: Output feature count.
        use_bias: When ``True`` (default), allocate and add a bias.
        rngs: Source of PRNG keys for parameter init.
        w_init: Weight initializer (default:
            :func:`~spectrax.init.kaiming_uniform` with ``"linear"`` gain).
        b_init: Bias initializer (default: :func:`~spectrax.init.zeros`).
        dtype: Parameter dtype; defaults to ``float32``.
        amax_history_length: Forwarded to :class:`Fp8DotGeneral`.
    """

    weight: Parameter
    bias: Parameter
    qdot: Fp8DotGeneral

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        amax_history_length: int = 1024,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Allocate the weight, optional bias, and embedded :class:`Fp8DotGeneral`.

        Args:
            in_features: Trailing input feature count.
            out_features: Output feature count.
            use_bias: When ``True`` (default), allocate the bias.
            rngs: PRNG source for parameter initialization.
            w_init: Weight initializer; defaults to Kaiming-uniform
                with ``"linear"`` gain.
            b_init: Bias initializer; defaults to zeros.
            dtype: Storage dtype for the parameters; defaults to
                ``float32``.
            param_dtype: Alias for ``dtype``.
            amax_history_length: Forwarded to
                :class:`Fp8DotGeneral`; controls the rolling amax
                history length.
            sharding: Optional sharding for the weight (axis names
                ``("in", "out")``).
            bias_sharding: Optional sharding for the bias (axis
                names ``("out",)``).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        w_init = w_init or kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        self.weight = Parameter(
            w_init(resolved.parameters, (in_features, out_features), weight_dtype),
            sharding=sharding,
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, (out_features,), weight_dtype),
                sharding=bias_sharding,
                axis_names=("out",),
            )
        self.qdot = Fp8DotGeneral(amax_history_length=amax_history_length)

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Compute ``y = qdot(x, W) + b`` with fp8 fake-quant on ``x`` and ``W``.

        Downcasts the higher-precision operand to the lower-precision
        operand's dtype before dispatching (same rule applied by
        :func:`spectrax.functional.linear`). The bias, if present, is
        cast to the matmul output dtype before the add so a stored-in-fp32
        bias composes cleanly with a bf16 accumulator without surprise
        upcasts.

        Args:
            x: Input tensor; trailing axis is the contraction dimension.
            **_: Additional keyword arguments accepted for forward
                compatibility; unused.

        Returns:
            The fp8-quantized matmul (plus bias) with the original
            input's batch axes preserved.
        """
        xa = jnp.asarray(x)
        W = self.weight.value
        if xa.dtype != W.dtype and jnp.issubdtype(xa.dtype, jnp.floating) and jnp.issubdtype(W.dtype, jnp.floating):
            if jnp.finfo(xa.dtype).bits < jnp.finfo(W.dtype).bits:
                W = W.astype(xa.dtype)
            else:
                xa = xa.astype(W.dtype)
        dn = (((xa.ndim - 1,), (0,)), ((), ()))
        y = self.qdot(xa, W, dn)
        if self.use_bias:
            b = self.bias.value
            if b.dtype != y.dtype and jnp.issubdtype(b.dtype, jnp.floating) and jnp.issubdtype(y.dtype, jnp.floating):
                b = b.astype(y.dtype)
            y = y + b
        return cast(Array, y)


class Fp8Einsum(Module):
    """Learnable einsum with fp8 fake-quant on both operands.

    Mirror of :class:`~spectrax.nn.Einsum` routed through an embedded
    :class:`Fp8DotGeneral`. The equation describes how the input ``x``
    and the learnable ``weight`` combine; both are qdq'd through E4M3
    (forward) and the result is threaded through :func:`out_qdq` so
    the backward pass quantizes the cotangent through E5M2.

    Args:
        equation: An :func:`jax.numpy.einsum` equation where the first
            operand is the input and the second is :attr:`weight`.
        shape: Shape of :attr:`weight`.
        use_bias: When ``True``, add a broadcasting bias.
        bias_shape: Shape of the bias (defaults to the output shape
            inferred from the equation; pass explicitly if ambiguous).
        rngs: PRNG source for parameter init.
        w_init: Weight initializer.
        b_init: Bias initializer.
        dtype: Parameter dtype; defaults to float32.
        amax_history_length: Forwarded to :class:`Fp8DotGeneral`.
    """

    weight: Parameter
    bias: Parameter
    qdot: Fp8DotGeneral

    def __init__(
        self,
        equation: str,
        shape: Sequence[int],
        *,
        use_bias: bool = False,
        bias_shape: Sequence[int] | None = None,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        amax_history_length: int = 1024,
        sharding: Sharding | AxisNames | None = None,
        bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Allocate the weight, optional bias, and embedded :class:`Fp8DotGeneral`.

        Args:
            equation: An :func:`jax.numpy.einsum` equation in
                explicit form (must contain ``"->"``). The first
                operand is the input ``x``; the second is the
                learnable :attr:`weight`.
            shape: Shape of :attr:`weight`; allocated eagerly.
            use_bias: When ``True``, allocate a broadcast-add bias
                of shape ``bias_shape``.
            bias_shape: Shape of the bias; required when
                ``use_bias=True``.
            rngs: PRNG source for parameter initialization.
            w_init: Weight initializer; defaults to Kaiming-uniform
                with ``"linear"`` gain.
            b_init: Bias initializer; defaults to zeros.
            dtype: Storage dtype; defaults to ``float32``.
            param_dtype: Alias for ``dtype``.
            amax_history_length: Forwarded to
                :class:`Fp8DotGeneral`.
            sharding: Optional sharding for the weight.
            bias_sharding: Optional sharding for the bias.

        Raises:
            ValueError: If ``equation`` does not contain ``"->"``,
                or if ``use_bias=True`` is set without
                ``bias_shape``.
        """
        super().__init__()
        if "->" not in equation:
            raise ValueError("Fp8Einsum equation must contain '->'")
        self.equation = equation
        self.shape = tuple(shape)
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        w_init = w_init or kaiming_uniform("linear")
        weight_dtype = param_dtype or dtype or jnp.float32
        self.weight = Parameter(
            w_init(resolved.parameters, self.shape, weight_dtype),
            sharding=sharding,
        )
        if use_bias:
            if bias_shape is None:
                raise ValueError("Fp8Einsum(use_bias=True) requires bias_shape=(..)")
            b_init = b_init or zeros
            self.bias = Parameter(
                b_init(resolved.parameters, tuple(bias_shape), weight_dtype),
                sharding=bias_sharding,
            )
        self.qdot = Fp8DotGeneral(amax_history_length=amax_history_length)

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply ``einsum(x, W)`` with fake-quant on both operands and on the cotangent.

        Inlines the qdq machinery rather than going through
        :meth:`Fp8DotGeneral.forward` because :func:`jax.numpy.einsum`
        has no ``dimension_numbers``-shaped signature — the two calls
        differ only in the underlying primitive (``einsum`` vs
        ``dot_general``).

        Args:
            x: Input tensor.
            **_: Additional kwargs accepted for forward compatibility.

        Returns:
            The einsum result, plus bias when configured.
        """
        xa = jnp.asarray(x)
        W = self.weight.value
        comp = xa.dtype
        e4m3 = jnp.dtype(self.qdot.e4m3_name)
        e5m2 = jnp.dtype(self.qdot.e5m2_name)
        qx = in_qdq(comp, e4m3, xa, self.qdot.input_scale.value, self.qdot.input_amax_history.value)
        qw = in_qdq(comp, e4m3, W, self.qdot.kernel_scale.value, self.qdot.kernel_amax_history.value)
        self.qdot.input_scale.value, self.qdot.input_amax_history.value = update_fp8_meta(
            xa, e4m3, self.qdot.input_scale.value, self.qdot.input_amax_history.value
        )
        self.qdot.kernel_scale.value, self.qdot.kernel_amax_history.value = update_fp8_meta(
            W, e4m3, self.qdot.kernel_scale.value, self.qdot.kernel_amax_history.value
        )
        y = jnp.einsum(self.equation, qx, qw)
        y = out_qdq(
            comp,
            e5m2,
            y,
            self.qdot.output_grad_scale.value,
            self.qdot.output_grad_amax_history.value,
        )
        if self.use_bias:
            y = y + self.bias.value
        return cast(Array, y)
