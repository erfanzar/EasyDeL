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

"""Channelwise integer quantized matmul as a pure XLA composition.

The fast TPU path for W8A16/W8A8/W4A16 dense linears, found by measurement
rather than intuition (autoresearch on TPU v5, 2026-08-03):

* The historical v3 grouped-matmul configuration measured only 0.47-0.78x
  of XLA's plain BF16 matmul. This is not a hardware ceiling or a general
  limit on Pallas: newer kernels/tiles require fresh device-time comparisons.
* XLA fuses ``w_q.astype(bf16)`` into the matmul's weight stream, so the
  decode path stays bandwidth-bound on the *packed* bytes — but **any**
  arithmetic on the weight before the dot (scale folds, reshapes) breaks the
  fusion and materializes the dequantized weight.
* XLA emits the native int8 MXU path (459 TOPS/core on v5, 2x bf16) for
  ``int8 x int8 -> int32`` dots, and the native int4 path (920 TOPS/core,
  4x bf16) for ``int4 x int4``.
* Per-channel (full-K) scales are what make single-dot execution possible:
  K-blocked scales force either the fusion break above or ``[blocks, m, n]``
  int32 partials (1.9 GB on a 4096x28672 weight at m=2048).

Measured on TPU v5 vs jitted bf16 ``x @ w`` (per-chip 7B shapes / 27B-class):

============  ==================  ====================
path          decode (m<=32)      prefill (m=2048)
============  ==================  ====================
int8          1.08-1.52x          1.30-1.76x
int4          1.08-1.77x          (W4A8: same as int8)
W4A4 opt-in   —                   up to 2.88x, relerr ~0.11
============  ==================  ====================

Accuracy note: per-channel int8 weights measure ~0.004 relative error against
their own dequantization on gaussian data; dynamic per-token int8 activations
add ~0.007. W4A4 (``activation_bits=4``) is fast but coarse (~0.11) — it is
opt-in and belongs behind calibration/smoothing.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import numpy as jnp

from ..._registry import Backend, Platform, kernel_registry

__all__ = ["channelwise_quantized_matmul"]


@kernel_registry.register("channelwise_quantized_matmul", Platform.XLA, Backend.ANY)
def channelwise_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    channel_scale: jax.Array,
    *,
    quantize_activations: bool = False,
    activation_bits: int = 8,
    prefill_threshold: int = 256,
    platform: str = "xla",
) -> jax.Array:
    """Compute ``x @ (w_q * channel_scale)`` without dequantizing the weight.

    Two regimes, chosen by token count:

    * ``m < prefill_threshold`` (decode): ``x @ w_q.astype(x.dtype)`` — XLA
      fuses the upcast into the weight stream, keeping the op
      bandwidth-bound on the packed bytes — then the per-channel scale is
      applied to the small ``[m, n]`` output.
    * ``m >= prefill_threshold`` (prefill), when ``quantize_activations``:
      activations are quantized per token to the integer width, the dot runs
      on the int MXU path (int8: 2x bf16 rate; int4: 4x), and both scales are
      applied in the epilogue. Non-TPU backends widen int4 operands to int8
      for exact arithmetic without changing quantization or stored codes.

    The integer-activation branch uses an explicit straight-through activation
    derivative ``(dx @ w_q) * channel_scale``. Scale derivatives use the actual
    forward-quantized activations; integer weight codes are frozen. This is a
    training surrogate, not the natural derivative of hard quantization.
    Weight-only and below-threshold decode use ordinary autodiff.

    Args:
        x: Activations ``[m, k]``, floating dtype.
        w_q: Quantized weight ``[k, n]``; ``int8`` or ``int4``, symmetric,
            per-output-channel scaling.
        channel_scale: Per-output-channel scale, shape ``[n]``, ``[1, n]`` or
            any shape reshapeable to ``[1, n]``.
        quantize_activations: Enable the integer-dot prefill path. When
            ``False`` every token count uses the fused-upcast path (W-A16).
        activation_bits: 8 (default) or 4. Four-bit activations require
            ``int4`` weights and are markedly coarser (~0.11 relative error
            unsmoothed) — opt-in only.
        prefill_threshold: Token count at which the integer-dot path takes
            over from the fused-upcast path.
        platform: ``'xla'`` (default) or opt-in ``'pallas'``. Pallas requires
            active activation quantization, BF16 inputs, matching INT4/INT8
            weights, M divisible by 64, and K/N divisible by 128 and <=4096.
            It fuses row quantization with the dot, uses up to a 40 MiB VMEM
            budget, and retains XLA surrogate autodiff. It is not faster on
            every shape; notably 4096-square W8A8 remained slower in tests.
            No automatic platform selection or backward speedup is promised.

    Returns:
        ``[m, n]`` in ``x``'s dtype.

    Raises:
        ValueError: If ``w_q`` is not an integer array, or 4-bit activations
            are requested with a non-int4 weight.
    """
    if not jnp.issubdtype(w_q.dtype, jnp.integer):
        raise ValueError(f"w_q must be an integer array, got {w_q.dtype}.")
    if activation_bits not in (4, 8):
        raise ValueError(f"activation_bits must be 4 or 8, got {activation_bits}.")
    if activation_bits == 4 and w_q.dtype != jnp.int4:
        raise ValueError("activation_bits=4 requires int4 weights (the win is the int4 MXU path).")

    n_dim = w_q.shape[-1]
    scale = channel_scale.reshape(1, n_dim).astype(jnp.float32)
    tokens = x.shape[0]
    if platform not in ("xla", "pallas"):
        raise ValueError("platform must be 'xla' or 'pallas'")
    if platform == "pallas":
        if not quantize_activations or tokens < prefill_threshold:
            raise ValueError("Pallas dense path requires active integer activation quantization")
        from ..._pallas.tpu.quantized_matmul._channelwise import channelwise_quantized_matmul_pallas

        return channelwise_quantized_matmul_pallas(x, w_q, scale, activation_bits)

    if not quantize_activations or tokens < prefill_threshold:
        # Keep the weight expression a bare `astype`: XLA fuses exactly that
        # into the matmul stream; anything more materializes the bf16 weight.
        precision = jax.lax.Precision.HIGHEST if x.dtype == jnp.float32 else None
        out = jnp.matmul(x, w_q.astype(x.dtype), precision=precision)
        return (out.astype(jnp.float32) * scale).astype(x.dtype)

    return _quantized_activation_matmul(x, w_q, scale, activation_bits)


def _integer_dot(x, w_q, activation_bits):
    """Return the integer accumulator and per-token activation scale."""
    from ._integer_quantization import quantize_rows

    act_dtype = jnp.int4 if activation_bits == 4 else jnp.int8
    x_q, x_scale = quantize_rows(x, activation_bits)

    w_dot = w_q
    if act_dtype == jnp.int4 and jax.default_backend() != "tpu":
        # Preserve int4 quantization and stored codes, widening only arithmetic.
        # CPU/GPU XLA cannot lower the native sub-byte integer dot.
        x_q = x_q.astype(jnp.int8)
        w_dot = w_q.astype(jnp.int8)
    elif act_dtype == jnp.int8 and w_q.dtype != jnp.int8:
        # int8 x int4 is not MXU-native; the upcast is exact.
        w_dot = w_q.astype(jnp.int8)

    out = jax.lax.dot_general(
        x_q,
        w_dot,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    return out.astype(jnp.float32), x_scale


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _quantized_activation_matmul(x, w_q, scale, activation_bits):
    out, x_scale = _integer_dot(x, w_q, activation_bits)
    return (out * scale * x_scale).astype(x.dtype)


@_quantized_activation_matmul.defjvp
def _quantized_activation_matmul_jvp(activation_bits, primals, tangents):
    """Activation STE with frozen integer codes (not hard-rounding's derivative).

    The activation tangent is ``(dx @ w_q) * scale``: multiplication by
    represented weights, including at zero rows. Scale tangents use the exact
    forward-quantized activation accumulator, NOT unquantized ``x @ w_q``.
    Only the integer-activation regime uses this surrogate; A16 and legacy
    decode retain ordinary autodiff. Integer weight tangents are ignored.
    """
    x, w_q, scale = primals
    dx, _, dscale = tangents
    out, x_scale = _integer_dot(x, w_q, activation_bits)
    primal = (out * scale * x_scale).astype(x.dtype)
    precision = jax.lax.Precision.HIGHEST if x.dtype == jnp.float32 else None
    # Preserve the BF16 dot rounding boundary when XLA jointly compiles the
    # integer primal and floating tangent; otherwise fusion can elide it.
    activation_dot = jax.lax.optimization_barrier(jnp.matmul(dx, w_q.astype(x.dtype), precision=precision))
    activation_tangent = activation_dot.astype(jnp.float32) * scale
    scale_tangent = out * x_scale * dscale
    return primal, (activation_tangent + scale_tangent).astype(x.dtype)
