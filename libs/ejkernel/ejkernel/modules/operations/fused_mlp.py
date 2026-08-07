# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public fused-MLP operation: one surface for inference and training.

``fused_mlp`` computes ``down(act(gate(x)) * up(x))`` with:

* **formats** — bf16 dense; channelwise int8; channelwise int4 (optionally
  bit-packed for the TPU decode kernel);
* **layouts** — gate/up separate, fused-concat, or TP-interleaved (normalized
  by :func:`split_gate_up` before any kernel runs);
* **activations** — the shared table (silu, gelu, gelu_tanh, relu, sigmoid);
* **forward** — Pallas W4A4 single-dispatch kernel on TPU decode shapes when
  packed weights are supplied; the XLA composition otherwise;
* **backward** — format-dependent, chosen by measurement:
  **dense** weights run the plain composition under JAX autodiff (XLA's
  saved-residual schedule measured strictly better than a recompute
  ``custom_vjp``: 0.78x); **integer-code** weights use a ``custom_vjp`` —
  naive autodiff through the quantized forward produces garbage gradients
  (cosine 0.03 vs ground truth, measured) — that recomputes pre-activations
  through the fast channelwise forward and, under ``quantize_activations``,
  dynamically quantizes the cotangents per token so the transposed products
  run on the int8 MXU. Integer codes are frozen (``float0`` cotangents) with
  correct ``dx`` (cosine 0.999 vs dequantized ground truth) — the
  LoRA / frozen-backbone training contract.

Gradient math (concat convention, ``a = x Wg``, ``b = x Wu``,
``h = act(a) * b``, ``y = h Wd``):

    dh  = gy Wd^T          dWd = h^T gy
    da  = dh * b * act'(a) db  = dh * act(a)
    dx  = da Wg^T + db Wu^T
    dWg = x^T da           dWu = x^T db

For channelwise-quantized weights every ``W`` above is ``codes * scale``; the
per-output-channel scale folds into the *cotangent* side of each transposed
product, so the transposed dots run on the raw integer codes with the same
fused-upcast behavior as the forward.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from jax import numpy as jnp

from ejkernel.kernels._xla.fused_mlp import ACTIVATIONS, fused_mlp_xla, split_gate_up

__all__ = ["fused_mlp", "split_gate_up"]


def _forward_reference(x, weights, scales, *, activation, quantize_activations, prefill_threshold):
    """Forward through the XLA composition (any format)."""
    w_gate, w_up, w_down = weights
    gate_scale, up_scale, down_scale = scales
    return fused_mlp_xla(
        x,
        w_gate,
        w_up,
        w_down,
        gate_scale,
        up_scale,
        down_scale,
        activation=activation,
        quantize_activations=quantize_activations,
        prefill_threshold=prefill_threshold,
    )


def _zero_cotangent(primal: jax.Array) -> jax.Array:
    """Cotangent for a frozen parameter, honoring JAX dtype rules.

    Integer primals take ``float0`` symbolic zeros; float primals (the scale
    arrays, treated as frozen calibration constants) take ordinary zeros.

    Args:
        primal: The primal value the cotangent corresponds to.

    Returns:
        The appropriate zero cotangent.
    """
    import numpy as np

    if jnp.issubdtype(primal.dtype, jnp.integer):
        return np.zeros(primal.shape, dtype=jax.dtypes.float0)
    return jnp.zeros_like(primal)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _fused_mlp_core(
    x: jax.Array,
    weights: tuple[jax.Array, jax.Array, jax.Array],
    scales: tuple[jax.Array | None, jax.Array | None, jax.Array | None],
    activation: str,
    quantize_activations: bool,
    prefill_threshold: int,
) -> jax.Array:
    """Differentiable core; see :func:`fused_mlp` for the public contract."""
    return _forward_reference(
        x,
        weights,
        scales,
        activation=activation,
        quantize_activations=quantize_activations,
        prefill_threshold=prefill_threshold,
    )


def _core_fwd(x, weights, scales, activation, quantize_activations, prefill_threshold):
    """Forward rule: save only primals — the backward recomputes the rest."""
    out = _forward_reference(
        x,
        weights,
        scales,
        activation=activation,
        quantize_activations=quantize_activations,
        prefill_threshold=prefill_threshold,
    )
    return out, (x, weights, scales)


def _core_bwd(activation, quantize_activations, prefill_threshold, residuals, gy):
    """Backward rule: recompute hidden state, differentiate through dequant.

    Args:
        activation: Static activation name.
        quantize_activations: Unused in backward (kept for signature parity —
            backward always runs the exact dequantized math).
        prefill_threshold: Unused in backward.
        residuals: ``(x, weights, scales)`` from the forward.
        gy: Output cotangent ``[m, k]``.

    Returns:
        Cotangents for ``(x, weights, scales)``.
    """
    del prefill_threshold
    x, weights, scales = residuals
    w_gate, w_up, w_down = weights
    gate_scale, up_scale, down_scale = scales
    act = ACTIVATIONS[activation]

    from ejkernel.kernels._xla.quantized_matmul import channelwise_quantized_matmul

    # Recompute the pre-activation products through the same fast forward
    # composition (fused-upcast / int8-dot), in bf16 — an f32 recompute
    # measured slower AND heavier than XLA's own residual saving.
    a = channelwise_quantized_matmul(x, w_gate, gate_scale).astype(jnp.float32)
    b = channelwise_quantized_matmul(x, w_up, up_scale).astype(jnp.float32)
    _, combine_vjp = jax.vjp(lambda a, b: act(a) * b, a, b)

    gy32 = gy.astype(jnp.float32)

    def transpose_dot(cotangent, w_codes, w_scale):
        """``cotangent @ dequant(w)^T`` — int8 MXU when activations quantize.

        The per-output-channel weight scale multiplies the cotangent BEFORE
        any quantization (it rides the contraction), so the transposed dot
        runs on raw codes. With ``quantize_activations`` the cotangent is
        dynamically quantized per token and the product runs at the int8 MXU
        rate — the backward twin of the forward's int-dot path.
        """
        scaled = cotangent * w_scale.reshape(1, w_codes.shape[-1])
        if quantize_activations:
            c_abs = jnp.max(jnp.abs(scaled), axis=1, keepdims=True)
            c_scale = c_abs / 127.0
            c_q = jnp.clip(
                jnp.round(scaled / jnp.where(c_scale == 0, 1, c_scale)), -127, 127
            ).astype(jnp.int8)
            w_dot = w_codes if w_codes.dtype == jnp.int8 else w_codes.astype(jnp.int8)
            out = jax.lax.dot_general(
                c_q, w_dot, dimension_numbers=(((1,), (1,)), ((), ())), preferred_element_type=jnp.int32
            )
            return out.astype(jnp.float32) * c_scale
        out = jax.lax.dot_general(
            scaled.astype(jnp.bfloat16),
            w_codes.astype(jnp.bfloat16),
            dimension_numbers=(((1,), (1,)), ((), ())),
        )
        return out.astype(jnp.float32)

    dh = transpose_dot(gy32, w_down, down_scale)
    da, db = combine_vjp(dh)
    dx = transpose_dot(da, w_gate, gate_scale) + transpose_dot(db, w_up, up_scale)
    dx = dx.astype(x.dtype)

    d_weights = tuple(_zero_cotangent(w) for w in weights)
    d_scales = tuple(None if s is None else _zero_cotangent(s) for s in scales)
    return dx, d_weights, d_scales


_fused_mlp_core.defvjp(_core_fwd, _core_bwd)


def fused_mlp(
    x: jax.Array,
    w_gate: jax.Array | None = None,
    w_up: jax.Array | None = None,
    w_down: jax.Array | None = None,
    *,
    gate_up: jax.Array | None = None,
    gate_up_layout: tp.Literal["concat", "interleaved"] = "concat",
    gate_up_segments: int = 1,
    gate_scale: jax.Array | None = None,
    up_scale: jax.Array | None = None,
    down_scale: jax.Array | None = None,
    activation: str = "silu",
    quantize_activations: bool = False,
    prefill_threshold: int = 256,
    packed_weights: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    packed_tile_i: int = 512,
    decode_threshold: int = 64,
) -> jax.Array:
    """Fused ``down(act(gate(x)) * up(x))`` for inference and training.

    Two execution regimes behind one call:

    * **Differentiable path** (default): the XLA composition with a
      recompute-based backward. Dense weights train normally; integer-code
      weights are frozen (symbolic-zero gradients) with exact ``dx`` — the
      LoRA / frozen-backbone contract.
    * **Packed decode path**: when ``packed_weights`` is given, the input is
      float, ``m < decode_threshold``, and the platform is TPU, the forward
      runs the single-dispatch Pallas W4A4 kernel (per-token input quant,
      per-(token, tile) hidden quant). Inference-only: it is wrapped in
      ``stop_gradient`` semantics by construction (no vjp), so training calls
      should not pass ``packed_weights``.

    Args:
        x: Input ``[m, k]``, floating dtype.
        w_gate: Gate weight ``[k, i]`` (dense or integer codes). Mutually
            exclusive with ``gate_up``.
        w_up: Up weight ``[k, i]``.
        w_down: Down weight ``[i, k]``.
        gate_up: Fused gate/up weight ``[k, 2i]`` instead of separate arrays.
        gate_up_layout: Fused layout — ``"concat"`` or ``"interleaved"``.
        gate_up_segments: TP segment count for the interleaved layout.
        gate_scale: Channel scale for integer gate weights.
        up_scale: Channel scale for integer up weights. When ``gate_up`` is
            given with one fused scale, pass it as ``gate_scale`` and leave
            this ``None`` — it is split with the weight.
        down_scale: Channel scale for integer down weights.
        activation: Gate-branch activation name.
        quantize_activations: Integer formats: use the int-MXU dot at prefill
            sizes (per-token dynamic activation quantization).
        prefill_threshold: Token count where the integer dot engages.
        packed_weights: ``(gate_packed, up_packed, down_packed)`` uint8 arrays
            in ``pack_int4_adjacent`` layout, enabling the fused decode
            kernel. Requires all three channel scales.
        packed_tile_i: I-tile width for the packed kernel.
        decode_threshold: Max ``m`` for which the packed kernel is used.

    Returns:
        ``[m, k]`` in ``x``'s dtype.

    Raises:
        ValueError: On inconsistent weight arguments.
    """
    if gate_up is not None:
        if w_gate is not None or w_up is not None:
            raise ValueError("Pass either gate_up or (w_gate, w_up), not both.")
        w_gate, w_up = split_gate_up(gate_up, layout=gate_up_layout, segments=gate_up_segments)
        if gate_scale is not None and up_scale is None:
            gate_scale, up_scale = split_gate_up(
                gate_scale.reshape(1, -1), layout=gate_up_layout, segments=gate_up_segments
            )
    if w_gate is None or w_up is None or w_down is None:
        raise ValueError("fused_mlp requires gate, up and down weights (separate or fused).")

    def _int4_mxu_available() -> bool:
        """Capability probe — no TPU generation is hardcoded anywhere."""
        from ejkernel.kernels._pallas.tpu.quantized_matmul._packed_gemv import supports_int4_mxu

        return supports_int4_mxu()

    use_packed = (
        packed_weights is not None
        and x.shape[0] < decode_threshold
        and jax.default_backend() == "tpu"
        and jnp.issubdtype(x.dtype, jnp.floating)
        and _int4_mxu_available()
    )
    if use_packed:
        from ejkernel.kernels._pallas.tpu.fused_mlp import fused_mlp_w4a4_pallas

        gate_packed, up_packed, down_packed = packed_weights
        x_abs = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=1, keepdims=True)
        x_scale = x_abs / 7.0
        x4 = jnp.clip(
            jnp.round(x.astype(jnp.float32) / jnp.where(x_scale == 0, 1, x_scale)), -7, 7
        ).astype(jnp.int4)
        out = fused_mlp_w4a4_pallas(
            x4,
            gate_packed,
            up_packed,
            down_packed,
            gate_scale,
            up_scale,
            down_scale,
            x_scale,
            activation=activation,
            tile_i=packed_tile_i,
        )
        return out.astype(x.dtype)

    if not jnp.issubdtype(w_gate.dtype, jnp.integer):
        # Dense weights: the plain composition in the input dtype, under JAX
        # autodiff. Two measured lessons: XLA's saved-residual backward beats
        # a recompute custom_vjp (0.78x), and an f32 elementwise upcast costs
        # 3-6% vs computing the activation in bf16 as XLA's own schedule does
        # — so this path is bit-identical to the naive composition.
        act = ACTIVATIONS[activation]
        return (act(x @ w_gate) * (x @ w_up)) @ w_down

    return _fused_mlp_core(
        x,
        (w_gate, w_up, w_down),
        (gate_scale, up_scale, down_scale),
        activation,
        quantize_activations,
        prefill_threshold,
    )
