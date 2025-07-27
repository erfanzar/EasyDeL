# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import functools
import typing as tp

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from easydel.utils.compiling_utils import ejit

from ._forward_pallas import (
    DEFAULT_MASK_VALUE,
    paged_flash_attention_kernel,
    paged_flash_attention_kernel_inline_seq_dim,
    prefill_attention_impl,
)

MAX_SMEM_USAGE = 512 * 1024


@ejit
def _build_contiguous_kv_vectorized(pages, block_tables) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = block_tables.shape[0]
    num_heads, _, page_size, head_dim = pages.shape

    def gather_for_head(head_pages, indices_per_batch):
        return jax.vmap(lambda idx: head_pages[idx, :, :])(indices_per_batch)

    gathered_per_head = jax.vmap(gather_for_head, in_axes=(0, None))(pages, block_tables)
    gathered_swapped = gathered_per_head.transpose(1, 0, 2, 3, 4)
    max_seq_len = block_tables.shape[1] * page_size
    return gathered_swapped.reshape(batch_size, num_heads, max_seq_len, head_dim)


@ejit(static_argnames=["block_size", "num_total_blocks", "max_blocks_per_seq", "num_kv_heads", "head_dim"])
def _build_paged_kv(
    contiguous_k: jnp.ndarray,  # Shape: (batch, seq_len, num_kv_heads, head_dim)
    contiguous_v: jnp.ndarray,  # Shape: (batch, seq_len, num_kv_heads, head_dim)
    seq_context_lens: jnp.ndarray,  # Shape: (batch,). True context_lens of each sequence.
    block_size: int,
    num_total_blocks: int,  # Desired size of the physical cache.
    max_blocks_per_seq: int,  # Desired size of the block table per sequence.
    num_kv_heads: int,
    head_dim: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Builds paged KV caches (physical cache + block tables) from contiguous KV caches.
    """
    batch_size, _, _, _ = contiguous_k.shape  # Use different name
    assert contiguous_v.shape == contiguous_k.shape
    assert seq_context_lens.shape == (batch_size,)

    assert contiguous_k.shape[-2] == num_kv_heads
    assert contiguous_k.shape[-1] == head_dim

    # --- 1. Calculate block requirements and allocate physical indices ---
    # (create_table_body using static iota and masking should be correct now)
    num_blocks_per_seq = jnp.ceil(seq_context_lens / block_size).astype(jnp.int32)
    cum_blocks = jnp.cumsum(num_blocks_per_seq)
    start_indices = jnp.concatenate([jnp.zeros(1, dtype=cum_blocks.dtype), cum_blocks[:-1]])
    block_tables = jnp.zeros((batch_size, max_blocks_per_seq), dtype=jnp.int32)

    def create_table_body(i, tables):
        num_blocks = num_blocks_per_seq[i]
        start_idx = start_indices[i]
        max_iota = jax.lax.iota(dtype=jnp.int32, size=max_blocks_per_seq)
        potential_phys_indices = start_idx + max_iota
        mask = max_iota < num_blocks
        phys_indices_for_row = jnp.where(mask, potential_phys_indices, 0)
        tables_updated = tables.at[i].set(phys_indices_for_row)
        return tables_updated

    block_tables = jax.lax.fori_loop(0, batch_size, create_table_body, block_tables)

    physical_k_cache = jnp.zeros((num_total_blocks, block_size, num_kv_heads, head_dim), dtype=contiguous_k.dtype)
    physical_v_cache = jnp.zeros((num_total_blocks, block_size, num_kv_heads, head_dim), dtype=contiguous_v.dtype)

    def scatter_body_outer(i, caches):
        phys_k, phys_v = caches
        seq_len = seq_context_lens[i]  # Traced sequence context_lens for this item
        num_blocks = num_blocks_per_seq[i]  # Traced number of blocks for this item

        def scatter_body_inner(j, inner_caches):
            k_cache, v_cache = inner_caches
            # `j` is concrete loop index, `block_size` is static
            start_token = j * block_size

            # Get physical index (traced)
            physical_idx = block_tables[i, j]

            # --- FIX HERE: Use static slice_sizes for dynamic_slice ---
            # Define the STATIC slice size we want to extract
            static_slice_sizes = (block_size, num_kv_heads, head_dim)

            # Define the start indices (start_token is traced, which is allowed)
            slice_start_indices = (start_token, 0, 0)

            k_potential_block = jax.lax.dynamic_slice(
                contiguous_k[i],
                slice_start_indices,
                static_slice_sizes,
            )
            v_potential_block = jax.lax.dynamic_slice(contiguous_v[i], slice_start_indices, static_slice_sizes)

            block_indices = jnp.arange(block_size)
            original_indices = start_token + block_indices
            mask = original_indices < seq_len

            mask_expanded = mask[:, None, None]

            k_block_padded = jnp.where(mask_expanded, k_potential_block, 0.0)
            v_block_padded = jnp.where(mask_expanded, v_potential_block, 0.0)
            k_cache_updated = k_cache.at[physical_idx].set(k_block_padded)
            v_cache_updated = v_cache.at[physical_idx].set(v_block_padded)

            return k_cache_updated, v_cache_updated

        phys_k, phys_v = jax.lax.fori_loop(
            0,
            num_blocks,
            scatter_body_inner,
            (phys_k, phys_v),
        )
        return phys_k, phys_v

    physical_k_cache, physical_v_cache = jax.lax.fori_loop(
        0,
        batch_size,
        scatter_body_outer,
        (physical_k_cache, physical_v_cache),
    )

    return physical_k_cache, physical_v_cache, block_tables


class PagedAttention:
    def build_paged_kv(
        self,
        contiguous_k: jnp.ndarray,  # Shape: (batch, seq_len, num_kv_heads, head_dim)
        contiguous_v: jnp.ndarray,  # Shape: (batch, seq_len, num_kv_heads, head_dim)
        seq_context_lens: jnp.ndarray,  # Shape: (batch,). True context_lens of each sequence.
        block_size: int,
        num_total_blocks: int,  # Desired size of the physical cache.
        max_blocks_per_seq: int,  # Desired size of the block table per sequence.
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _build_paged_kv(
            contiguous_k,
            contiguous_v,
            seq_context_lens,
            block_size,
            num_total_blocks,
            max_blocks_per_seq,
            num_kv_heads,
            head_dim,
        )

    def build_contiguous_kv_vectorized(self, pages, block_tables) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Builds contiguous KV caches from paged KV caches using vectorized operations.

        The output sequence context_lens dimension will be max_blocks_per_seq * block_size.
        The caller needs external knowledge (e.g., original sequence context_lens) to
        correctly interpret or mask the padding positions in the returned tensors.

        Returns:
            A tuple containing (contiguous_k, contiguous_v).
        """
        return _build_contiguous_kv_vectorized(pages, block_tables)


def prefill_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    sm_scale: float | None = None,
):
    """Computes paged attention for the prefill phase.

    This function wraps the `prefill_attention_impl` Pallas kernel, handling
    data layout transformations and launching the kernel. It processes one
    chunk of the query sequence against the corresponding KV cache pages.

    Args:
      q: Query tensor for a chunk of the sequence.
      k_pages: Key cache stored in paged layout in HBM.
      v_pages: Value cache stored in paged layout in HBM.
      context_lens: The total sequence context_lens for the item being processed.
      block_tables: Array mapping sequence positions to page indices in k_pages/v_pages.
      sm_scale: normal softmax scale. By default it is None or auto.

    Returns:
      The attention output for the query chunk, shape [chunk_size, num_attn_heads * head_dim].
    """
    chunk_size, num_attn_heads, head_dim = q.shape
    num_kv_heads, _, page_size, _ = k_pages.shape

    assert num_attn_heads % num_kv_heads == 0
    assert chunk_size % page_size == 0
    attn_group_size = num_attn_heads // num_kv_heads
    page_per_chunk = chunk_size // page_size
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    q = q.transpose((1, 0, 2))
    q = q * sm_scale

    q_block_spec = pl.BlockSpec((attn_group_size, chunk_size, head_dim), lambda i, *_: (i, 0, 0))
    lm_block_spec = pl.BlockSpec((attn_group_size, chunk_size, 1), lambda *_: (0, 0, 0))
    lm_shape = jax.ShapeDtypeStruct(shape=(attn_group_size, chunk_size, 1), dtype=jnp.float32)

    out, _, _ = pl.pallas_call(
        prefill_attention_impl,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_block_spec,
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
                pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            ],
            out_specs=[
                q_block_spec,
                lm_block_spec,
                lm_block_spec,
            ],
            scratch_shapes=[
                pltpu.VMEM((2, page_per_chunk, page_size, head_dim), k_pages.dtype),
                pltpu.VMEM((2, page_per_chunk, page_size, head_dim), v_pages.dtype),
                pltpu.SemaphoreType.DMA,
            ],
            grid=(num_kv_heads,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            lm_shape,
            lm_shape,
        ],
    )(
        jnp.reshape(context_lens, (1,)),
        block_tables,
        jnp.asarray([0], jnp.int32),
        q,
        k_pages,
        v_pages,
    )
    out = out.transpose((1, 0, 2)).astype(q.dtype)

    return out


def chunked_prefill_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    sm_scale: float | None = None,
):
    return prefill_attention(q, k_pages, v_pages, context_lens, block_tables, sm_scale)


@ejit(
    static_argnames=[
        "pages_per_compute_block",
        "attn_logits_soft_cap",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
    ],
)
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    *,
    sm_scale: float = 1,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> jax.Array:
    """Paged grouped query attention.

    Args:
      q: A [batch_size, num_heads, head_dim] jax.Array.
      k_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
      v_pages: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
      context_lens: A i32[batch_size] jax.Array the context_lens of each example.
      block_tables: A i32[batch_size, max_num_pages_per_req] jax.Array. Each entry
        should be in the range of [0, total_num_pages), indicating where to locate
        the page in `k_pages` or `v_pages`.
      sm_scale: normal softmax scale. By default it is 1.0.
      mask_value: The value used for padding in attention. By default it is a very
        negative floating point number.
      attn_logits_soft_cap: The value used for soft capping the attention logits.
      pages_per_compute_block: how many pages to be processed in one flash
        attention block in the pallas kernel.
      megacore_mode: if set, enable megacore to parallelize the computation. Must
        be one of ['kv_head', 'batch', None]. Caveat: set this only if megacore is
        enabled, otherwise the kernel may hang. If you are not sure, leave it to
        None.
        * None: disable megacore parallelism.
        * kv_head: megacore parallelism on KV heads; requires number of KV heads
          divisible by 2.
        * batch: megacore parallelism on batch dimension; requires batch divisible
          by 2.
      inline_seq_dim: whether to fuse kernel instances along the sequence dim into
        one kernel.

    Returns:
      The output of attention([batch_size, num_heads, head_dim]).
    """

    batch_size, num_heads, head_dim = q.shape
    num_kv_heads, _, page_size, head_dim_k = k_pages.shape
    batch_size_paged_indices, max_num_pages_per_req = block_tables.shape

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if k_pages.shape != v_pages.shape:
        raise ValueError(
            f"k_pages and v_pages must have the same shape. Got {k_pages.shape} and"
            f" {v_pages.shape}"  # pytype: disable=attribute-error
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Number of Q heads must be divisible by number of KV heads. Got {num_heads} and {num_kv_heads}."
        )
    if head_dim_k != head_dim:
        raise ValueError(f"head_dim of Q must be the same as that of K/V. Got {head_dim} and {head_dim_k}.")
    if max_num_pages_per_req % pages_per_compute_block != 0:
        raise ValueError(
            "pages_per_compute_block must be divisible by pages per sequence. Got"
            f" {pages_per_compute_block} and {max_num_pages_per_req}."
        )
    if context_lens.shape != (batch_size,):
        raise ValueError("`context_lens` and `q` must have the same batch size")
    if batch_size_paged_indices != batch_size:
        raise ValueError("`block_tables` and `q` must have the same batch size")
    if context_lens.dtype != jnp.int32:
        raise ValueError("The dtype of `context_lens` must be int32. Got {context_lens.dtype}")

    if megacore_mode == "kv_head":
        if num_kv_heads % 2 != 0:
            raise ValueError("number of KV heads must be even when megacore_mode is 'kv_head'")
        num_cores = 2
    elif megacore_mode == "batch":
        if batch_size % 2 != 0:
            raise ValueError("batch size must be even when megacore_mode is 'batch'")
        num_cores = 2
    elif megacore_mode is None:
        num_cores = 1
    else:
        raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")
    if (num_heads // num_kv_heads) % 8 != 0:
        q = q.reshape(batch_size, num_heads, 1, head_dim)
        if megacore_mode == "kv_head":
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, None, head_dim),
                lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
            )
        elif megacore_mode == "batch":
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, None, head_dim),
                lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
            )
        else:
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, None, head_dim),
                lambda core_index, b, h, *_: (b, h, 0, 0),
            )
        q_dtype_for_kernel_launch = jnp.float32
    else:
        if megacore_mode == "kv_head":
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, head_dim),
                lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
            )
        elif megacore_mode == "batch":
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, head_dim),
                lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
            )
        else:
            q_block_spec = pl.BlockSpec(
                (None, num_heads // num_kv_heads, head_dim),
                lambda core_index, b, h, *_: (b, h, 0),
            )
        q_dtype_for_kernel_launch = q.dtype

    dimension_semantics: tp.Sequence[tp.Literal["parallel", "arbitrary"]]
    if inline_seq_dim:
        kernel = paged_flash_attention_kernel_inline_seq_dim
        grid = (
            num_cores,
            batch_size // num_cores if megacore_mode == "batch" else batch_size,
            num_kv_heads // num_cores if megacore_mode == "kv_head" else num_kv_heads,
        )
        dimension_semantics = ("parallel", "arbitrary", "arbitrary")
    else:
        kernel = paged_flash_attention_kernel
        grid = (
            num_cores,
            batch_size // num_cores if megacore_mode == "batch" else batch_size,
            num_kv_heads // num_cores if megacore_mode == "kv_head" else num_kv_heads,
            max_num_pages_per_req // pages_per_compute_block,
        )
        dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")

    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
    ]
    scratch_shapes = (
        pltpu.VMEM((2, pages_per_compute_block, page_size, head_dim), k_pages.dtype),
        pltpu.VMEM((2, pages_per_compute_block, page_size, head_dim), v_pages.dtype),
        pltpu.SemaphoreType.DMA,
    )

    out, _, _ = pl.pallas_call(
        functools.partial(
            kernel,
            max_num_pages_per_req=max_num_pages_per_req,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            megacore_mode=megacore_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=in_specs,
            out_specs=[
                q_block_spec,
                q_block_spec,
                q_block_spec,
            ],
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=dimension_semantics),
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
            jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
            jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
        ],
    )(
        context_lens,
        block_tables.reshape(-1),
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((1,), jnp.int32),
        q.astype(q_dtype_for_kernel_launch) * sm_scale,
        k_pages,
        v_pages,
    )

    return out.reshape(batch_size, num_heads, head_dim).astype(q.dtype)
