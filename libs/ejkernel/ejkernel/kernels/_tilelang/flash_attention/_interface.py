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

"""tile-lang FlashAttention public interface.

This module exposes :func:`flash_attention`, the entry point for running
FlashAttention-2 (forward + backward) on NVIDIA GPUs via natively-authored
tile-lang kernels. The signature mirrors the XLA reference in
:mod:`ejkernel.kernels._xla.flash_attention`, and — unlike a feature gate —
every argument is honoured:

* ``attention_mask``, ``bias``, ``logits_soft_cap``, ``sliding_window``,
  ``q_segment_ids`` / ``kv_segment_ids``, ``softmax_aux`` (attention sinks),
  ``dropout_prob`` / ``dropout_seed``, ``normalize_output`` and GQA/MQA
  (``num_kv_heads`` dividing ``num_heads``) are all applied **natively
  inside the tile-lang kernel**, forward and backward.
* ``fwd_params`` / ``bwd_params`` are honoured as kernel tile-size hints.
* ``precision`` / ``logits_dtype`` are validated; the tile-lang kernel
  accumulates ``QK``, the softmax and ``PV`` in float32 (logits are always
  float32), which satisfies ``Precision.DEFAULT`` / ``HIGH`` / ``HIGHEST``.
* ``cum_seqlens_q`` / ``cum_seqlens_k`` and ``block_tables`` are gated with
  an explicit :class:`EjkernelRuntimeError` — exactly as the XLA reference
  does — because neither the XLA reference nor this backend implements
  ragged-packed or paged-KV flash attention.
"""

from __future__ import annotations

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._impl import flash_attention_tilelang

DenseKV = Float[Array, "batch seq_len_k num_kv_heads head_dim"]
PagedKV = Float[Array, "num_blocks block_size num_kv_heads head_dim"]
BlockTables = Int[Array, "batch max_blocks"]

_ALLOWED_PRECISION = {
    jax.lax.Precision.DEFAULT,
    jax.lax.Precision.HIGH,
    jax.lax.Precision.HIGHEST,
}


@kernel_registry.register("flash_attention", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: DenseKV | PagedKV,
    value: DenseKV | PagedKV,
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    block_tables: BlockTables | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """FlashAttention-2 implemented natively in tile-lang (forward + backward).

    H100-targeted GPU backend. The online-softmax accumulator, the dQ/dK/dV
    split, and every score-space modifier are written in the tile-lang DSL
    and lowered through TVM-FFI to a CUDA cubin invoked via
    ``jax.ffi.ffi_call``.

    See :func:`ejkernel.kernels._xla.flash_attention.flash_attention` for the
    canonical argument documentation. Every argument is honoured natively;
    the only gated parameters are ``cum_seqlens_*`` (ragged-packed) and
    ``block_tables`` (paged KV), which the XLA reference also rejects.

    Raises:
        EjkernelRuntimeError: if ``cum_seqlens_q`` / ``cum_seqlens_k`` is
            supplied without ``attention_mask``, or if ``block_tables`` is
            supplied (paged-KV flash attention is not implemented by this
            backend or the XLA reference).
        ValueError: on an unrecognised ``precision`` / ``logits_dtype`` or a
            ``num_kv_heads`` that does not divide ``num_heads``.
    """
    reasons: list[str] = []
    if cum_seqlens_q is not None and attention_mask is None:
        reasons.append("cum_seqlens_q requires attention_mask (fold the ragged layout into the mask)")
    if cum_seqlens_k is not None and attention_mask is None:
        reasons.append("cum_seqlens_k requires attention_mask (fold the ragged layout into the mask)")
    if block_tables is not None:
        reasons.append("block_tables (paged-KV) is not supported by flash_attention; use a paged-attention kernel")
    if reasons:
        raise EjkernelRuntimeError("flash_attention (platform=tilelang): " + "; ".join(reasons))

    if precision is not None and not isinstance(precision, int) and precision not in _ALLOWED_PRECISION:
        raise ValueError(
            "precision must be jax.lax.Precision.{DEFAULT|HIGH|HIGHEST}; "
            "the tile-lang kernel accumulates in float32 and satisfies all three."
        )
    jnp.dtype(logits_dtype)

    return flash_attention_tilelang(
        query,
        key,
        value,
        softmax_scale=softmax_scale,
        causal=causal,
        bias=bias,
        attention_mask=attention_mask,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=dropout_prob,
        dropout_seed=dropout_seed,
        normalize_output=normalize_output,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )


__all__ = ["flash_attention"]
