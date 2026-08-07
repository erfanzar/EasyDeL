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

* A Pallas kernel is the wrong tool here — the v3 grouped-matmul kernel's
  **pure bf16** ceiling measured 0.47-0.78x of XLA's plain matmul, so no
  amount of quantization cleverness inside it can win.
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

import jax
from jax import numpy as jnp

__all__ = ["channelwise_quantized_matmul"]


def channelwise_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    channel_scale: jax.Array,
    *,
    quantize_activations: bool = False,
    activation_bits: int = 8,
    prefill_threshold: int = 256,
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
      applied in the epilogue.

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

    if not quantize_activations or tokens < prefill_threshold:
        # Keep the weight expression a bare `astype`: XLA fuses exactly that
        # into the matmul stream; anything more materializes the bf16 weight.
        out = x @ w_q.astype(x.dtype)
        return (out.astype(jnp.float32) * scale).astype(x.dtype)

    if activation_bits == 4:
        act_dtype, act_max = jnp.int4, 7.0
    else:
        act_dtype, act_max = jnp.int8, 127.0

    x_abs = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(jnp.float32)
    x_scale = x_abs / act_max
    x_q = jnp.clip(
        jnp.round(x.astype(jnp.float32) / jnp.where(x_scale == 0, 1, x_scale)), -act_max, act_max
    ).astype(act_dtype)

    w_dot = w_q
    if act_dtype == jnp.int8 and w_q.dtype != jnp.int8:
        # int8 x int4 is not MXU-native; the upcast is exact.
        w_dot = w_q.astype(jnp.int8)

    out = jax.lax.dot_general(
        x_q,
        w_dot,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    return (out.astype(jnp.float32) * scale * x_scale).astype(x.dtype)
