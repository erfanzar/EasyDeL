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

"""Paged attention interface for efficient KV cache management.

This module provides the public API for paged attention where the KV cache
is organized into fixed-size blocks. Enables efficient memory management
for variable-length sequences in decode/generation.
"""

import jax.numpy as jnp
import jaxtyping
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _page_attention_fwd

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@kernel_registry.register("page_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    sliding_window: int | None = None,
    num_warps: int = 4,
    num_stages: int = 3,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Compute paged attention for decode-phase inference using JAX/XLA.

    Each sequence contributes exactly one query token (decode step).  The KV
    cache is stored in fixed-size pages (blocks); the ``block_tables`` array
    maps each sequence's logical block positions to physical pages.  The
    function gathers all pages for each sequence, computes full attention
    with a validity mask (tokens beyond ``context_lens`` are masked), and
    returns the weighted sum of values.

    GQA / MQA is supported: ``num_heads`` must be a multiple of
    ``num_kv_heads``.

    The function accepts a ``key_cache`` / ``value_cache`` in either layout:

    - *Blocks-first*: ``[num_blocks, num_kv_heads, block_size, head_dim]``
    - *KV-heads-first*: ``[num_kv_heads, num_blocks, block_size, head_dim]``

    The layout is detected automatically via divisibility of ``num_heads``
    by each candidate first dimension; when the first dimension is
    ambiguous (e.g. ``num_blocks == num_kv_heads``), blocks-first is assumed.

    Args:
        query: Query tensor ``[num_seqs, num_heads, head_dim]``.
        key_cache: Paged key cache, blocks-first or KV-heads-first (see above).
        value_cache: Paged value cache, same layout as ``key_cache``.
        context_lens: Valid token count per sequence ``[num_seqs]``.  Tokens
            beyond this length are masked out with ``-1e9``.
        block_tables: Logical-to-physical block mapping ``[num_seqs, max_blocks]``.
        attn_scale: QK scaling factor.  Defaults to ``1 / sqrt(head_dim)``.
        max_context_len: Not supported; raises ``NotImplementedError`` if provided.
        num_splits: Partitioned-attention splits.  Only ``0`` is accepted; any
            other value raises ``NotImplementedError``.
        mask_value: Not used in this implementation; accepted for API compatibility.
        attn_logits_soft_cap: Logit soft-cap.  Raises ``NotImplementedError``
            if provided (not yet implemented on the XLA path).
        pages_per_compute_block: TPU tile size hint.  Raises ``NotImplementedError``
            if provided.
        megacore_mode: TPU megacore hint.  Raises ``NotImplementedError`` if provided.
        inline_seq_dim: Must be ``True``; ``False`` raises ``NotImplementedError``.
        sliding_window: Not used in this implementation; accepted for API
            compatibility but silently ignored.
        num_warps: Accepted for API compatibility with Triton; ignored by XLA.
        num_stages: Accepted for API compatibility with Triton; ignored by XLA.

    Returns:
        Attention output ``[num_seqs, num_heads, head_dim]``.

    Raises:
        NotImplementedError: If any unsupported parameter is non-default.

    Examples:
        >>> num_seqs, num_heads, head_dim = 2, 8, 64
        >>> num_kv_heads = 8
        >>> num_blocks, block_size = 10, 16
        >>>
        >>> query = jnp.ones((num_seqs, num_heads, head_dim))
        >>> key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>> value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>> context_lens = jnp.array([48, 32])
        >>> block_tables = jnp.array([[0, 1, 2, -1], [3, 4, -1, -1]])
        >>>
        >>> output = page_attention(query, key_cache, value_cache,
        ...                         context_lens, block_tables)
        >>> output.shape
        (2, 8, 64)
    """
    if max_context_len is not None:
        raise NotImplementedError("max_context_len is not supported in XLA implementation")
    if num_splits != 0:
        raise NotImplementedError("num_splits is not supported in XLA implementation")
    if pages_per_compute_block is not None:
        raise NotImplementedError("pages_per_compute_block is not supported in XLA implementation")
    if megacore_mode is not None:
        raise NotImplementedError("megacore_mode is not supported in XLA implementation")
    if not inline_seq_dim:
        raise NotImplementedError("inline_seq_dim=False is not supported in XLA implementation")
    if attn_logits_soft_cap is not None:
        raise NotImplementedError("attn_logits_soft_cap is not supported in XLA implementation")

    if attn_scale is None:
        attn_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)

    num_heads = query.shape[1]
    dim0, dim1 = key_cache.shape[0], key_cache.shape[1]
    dim0_div = (num_heads % dim0) == 0
    dim1_div = (num_heads % dim1) == 0

    if dim1_div and not dim0_div:
        key_cache_bf = key_cache
        value_cache_bf = value_cache
    elif dim0_div and not dim1_div:
        key_cache_bf = key_cache.transpose(1, 0, 2, 3)
        value_cache_bf = value_cache.transpose(1, 0, 2, 3)
    else:
        if dim0 >= dim1:
            key_cache_bf = key_cache
            value_cache_bf = value_cache
        else:
            key_cache_bf = key_cache.transpose(1, 0, 2, 3)
            value_cache_bf = value_cache.transpose(1, 0, 2, 3)

    block_size = key_cache_bf.shape[2]

    return _page_attention_fwd(
        query=query,
        key_cache=key_cache_bf,
        value_cache=value_cache_bf,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        block_size=block_size,
    )
