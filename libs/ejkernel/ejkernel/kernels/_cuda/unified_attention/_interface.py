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

"""CUDA unified attention interface (forward-only).

Provides the public, type-checked entry point
:func:`unified_attention` for the CUDA unified attention kernel.
The function is registered with the ejkernel kernel registry under the
name ``"unified_attention"`` for :attr:`Platform.CUDA` /
:attr:`Backend.GPU`, enabling automatic dispatch by the higher-level
operations layer.

Internally the function delegates to
:func:`~._cuda_impl.unified_attention_cuda` wrapped through
:func:`~ejkernel.callib._ejit.ejit` for persistent JIT-compilation
caching.

Limitations:
    - Only causal attention is supported (``causal`` must be ``True``).
    - ALiBi slopes (``alibi_slopes``) are not supported.
    - Query-query bias (``qq_bias``) is not supported.
    - ``head_dim`` must be <= 256.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.callib._ejit import ejit
from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl import unified_attention_cuda


@kernel_registry.register("unified_attention", Platform.CUDA, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    block_dim: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Compute unified paged-cache attention on a CUDA GPU.

    Performs multi-head (or grouped-query) attention where keys and values
    are stored in a block-paged cache. This is the registry-facing entry
    point that is automatically selected when the caller requests the
    ``"unified_attention"`` kernel on the CUDA / GPU platform.

    The function is type-checked at runtime via *jaxtyping* and
    *beartype*, and JIT-compiled with persistent caching via *ejit*.

    Args:
        queries: Query tensor of shape
            ``(total_tokens, num_q_heads, head_dim)`` containing the
            packed queries for all sequences in the batch.
        key_cache: Paged key cache of shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Paged value cache of shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        kv_lens: Number of valid KV tokens per sequence, shape
            ``(num_seqs,)``, dtype ``int32``.
        block_tables: Mapping from (sequence, block-index) to the
            physical cache block id, shape
            ``(num_seqs, max_blocks_per_seq)``, dtype ``int32``.
        query_start_loc: Cumulative token offsets indicating where each
            sequence's queries begin in the packed *queries* tensor,
            shape ``(num_seqs + 1,)``, dtype ``int32``.
        alibi_slopes: **Not supported by this CUDA backend.** Must be
            ``None``. Accepted for signature compatibility with other
            backends.
        qq_bias: **Not supported by this CUDA backend.** Must be
            ``None``. Accepted for signature compatibility.
        softmax_aux: Optional auxiliary buffer of shape
            ``(num_q_heads,)`` for sink-token attention. ``None``
            disables the feature.
        softmax_scale: Multiplicative scale applied to the QK^T dot
            products before softmax. Defaults to
            ``head_dim ** -0.5``.
        causal: Whether to apply causal (lower-triangular) masking.
            Must be ``True`` for this CUDA backend.
        sliding_window: Optional sliding-window length. When ``None``,
            full causal attention is used.
        logits_soft_cap: Optional soft-capping value applied to logits
            before the softmax. ``None`` disables capping.
        seq_threshold_3d: Ignored by this backend. Accepted for
            interface compatibility with Triton kernels.
        num_par_softmax_segments: Ignored by this backend.
        block_dim: CUDA thread-block dimension selected by the operation
            executor.
        num_warps: Ignored by this backend.
        num_stages: Ignored by this backend.

    Returns:
        Attention output tensor of the same shape and dtype as
        *queries* (``(total_tokens, num_q_heads, head_dim)``).

    Raises:
        ValueError: If *alibi_slopes* or *qq_bias* is not ``None``,
            or if *causal* is ``False`` (propagated from the CUDA
            implementation).

    Note:
        The parameters *seq_threshold_3d*, *num_par_softmax_segments*,
        *num_warps*, and *num_stages* are discarded. They exist
        in the signature solely to maintain a uniform interface across
        all registered attention kernel backends.
    """
    del seq_threshold_3d, num_par_softmax_segments, num_warps, num_stages

    reasons: list[str] = []
    if alibi_slopes is not None:
        reasons.append("alibi_slopes is not supported")
    if qq_bias is not None:
        reasons.append("qq_bias is not supported")
    if not causal:
        reasons.append("causal must be True")
    if queries.shape[-1] > 256:
        reasons.append("head_dim must be <= 256")
    if reasons:
        raise EjkernelRuntimeError("unified_attention (platform=cuda): " + "; ".join(reasons))

    return ejit(
        func=unified_attention_cuda,
        static_argnames=["softmax_scale", "causal", "sliding_window", "logits_soft_cap", "block_dim"],
    )(
        queries,
        key_cache,
        value_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        alibi_slopes,
        qq_bias,
        softmax_aux,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_dim=block_dim,
    )
