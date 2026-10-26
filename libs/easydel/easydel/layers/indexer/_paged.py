# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Paged (ragged-serving) selection for the k-pool :class:`SparseIndexer`.

The dense indexer re-reads a ``[batch, kv, 2*dim+1]`` packed state every step. On the
ragged-pages serving path that would mean materialising a per-row ``max_model_len``
window, which is what made the first attempt 145x slower. Here the state lives in pages
addressed through the *same page table as the KV cache*:

* ``token_pages [num_pages, page_size, 2*dim]``: each token's ``key | gate`` (key dtype,
  exactly as the dense indexer packs it).
* ``pool_pages [num_pages, page_size // kpool, dim]``: the gated summary of every
  *completed* pool, written once on the step its last member arrives.

Scoring reads only pool summaries (1/kpool of the tokens) and only up to each request's
own context, so the cost is proportional to the real context, never to ``max_model_len``.

Semantics equal :meth:`SparseIndexer._select_pool` for serving streams (every cached
token valid, pools aligned to position 0): a query at position ``pos`` may pick the
complete pools ``0 .. (pos + 1) // kpool - 1`` (the "pool end visible" rule) and always
gets the incomplete tail pool. Ties resolve lowest-pool-first like ``jax.lax.top_k`` over
the full pool axis, because the running top-k keeps the current best (lower pool ids)
ahead of every newly scored chunk.

Data-parallel serving passes rank-major metadata (``query_start_loc`` of shape
``[dp, rows + 1]`` with rank-local token offsets, tokens and pages sharded over the DP
axis). The selectors then run the page update and the top-k once per rank under a
``shard_map``, on that rank's rows, pages and block tables rebased to local page ids,
exactly like the DP-local MLA attention path.
"""

from __future__ import annotations

import functools
import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.layers.compressed_window import packed_row_of_token
from easydel.layers.rotary import apply_rope_interleaved, apply_rope_split_half

if tp.TYPE_CHECKING:
    from ._indexer import SparseIndexer
    from ._token import TokenIndexer


def _identity(x: jax.Array) -> jax.Array:
    return x


def _token_rope_interleaved(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Interleaved (even/odd pair) RoPE with half-width ``cos`` / ``sin`` of shape ``[batch, seq, rope_dim // 2]``.

    The tables gain a head axis so they broadcast over ``x`` of shape ``[batch, seq, ..., rope_dim]``.
    """
    return apply_rope_interleaved(x, cos[..., None, :], sin[..., None, :])


def _token_rope_split(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Split-half (NeoX) RoPE with half-width ``cos`` / ``sin`` of shape ``[batch, seq, rope_dim // 2]``."""
    cos = jnp.concatenate([cos, cos], axis=-1)
    sin = jnp.concatenate([sin, sin], axis=-1)
    return apply_rope_split_half(x, cos, sin, unsqueeze_dim=-2, compute_dtype=None)


def _paged_pool_topk_impl(platform: str):
    """Registered ``paged_pool_topk`` implementation for ``platform`` (``pallas`` or ``xla``)."""
    from ejkernel.kernels._registry import kernel_registry  # pyright: ignore[reportMissingTypeStubs]

    return kernel_registry.get("paged_pool_topk", platform=platform, backend="any")


def _run_pool_topk(kernel_mesh, *args, **kwargs) -> jax.Array:
    """Dispatch ``paged_pool_topk``: fused Pallas on TPU, XLA loop otherwise.

    A Pallas call cannot be SPMD-partitioned, so on a multi-device mesh it runs under a
    replicated ``shard_map`` over ``kernel_mesh`` (valid only while the indexer pages are
    replicated; the cache view returns ``None`` otherwise, which keeps the XLA path).
    """
    xla = _paged_pool_topk_impl("xla")
    if jax.default_backend() != "tpu":
        return xla(*args, **kwargs)
    pallas = _paged_pool_topk_impl("pallas")
    ambient = jax.sharding.get_abstract_mesh()
    if not ambient.empty and ambient.manual_axes:
        auto = [n for n in ambient.axis_names if n not in ambient.manual_axes and ambient.shape[n] > 1]
        return (xla if auto else pallas)(*args, **kwargs)
    mesh = getattr(kernel_mesh, "jax_mesh", kernel_mesh)
    if mesh is None:
        return (pallas if jax.device_count() == 1 else xla)(*args, **kwargs)
    if mesh.size == 1:
        return pallas(*args, **kwargs)
    rep = PartitionSpec()
    # args: (query [T,H,D], head_weights [T,H], pool_pages, block_tables, seq_ids [T], num_pools [T]).
    # Rows are independent, so split them over every mesh device: each device scores only its
    # slice and just the small [T, k] ids are gathered, instead of all-gathering the query and
    # repeating the whole top-k on every device. Pages and the block table stay replicated.
    if int(args[0].shape[0]) % int(mesh.size) == 0:
        row = PartitionSpec(tuple(mesh.axis_names))
        in_specs = (row, row, rep, rep, row, row)
        out_specs = row
    else:
        in_specs = (rep,) * len(args)
        out_specs = rep
    return jax.shard_map(
        lambda *a: pallas(*a, **kwargs),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def paged_token_layout(
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    total_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map packed query tokens to ``(sequence, absolute position, valid)``.

    Args:
        kv_lens: ``[S]`` context length per sequence *including* the current step.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        num_seqs: Scalar number of live sequences; later tokens are padding.
        total_tokens: Static packed token count.

    Returns:
        ``(seq [T], pos [T], valid [T])`` int32/int32/bool.
    """
    max_seqs = int(kv_lens.shape[0])
    tok = jnp.arange(total_tokens, dtype=jnp.int32)
    seq = packed_row_of_token(query_start_loc, tok, max_row=max_seqs - 1)
    q_start = query_start_loc[seq]
    q_len = query_start_loc[seq + 1] - q_start
    kv_len = kv_lens[seq]
    pos = kv_len - q_len + (tok - q_start)
    live_end = query_start_loc[jnp.clip(jnp.asarray(num_seqs, jnp.int32), 0, max_seqs)]
    # Tokens before query_start_loc[0] belong to other shards when a DP-local slice is passed.
    valid = (tok >= query_start_loc[0]) & (tok < live_end) & (pos >= 0) & (pos < kv_len)
    return seq, pos.astype(jnp.int32), valid


# ---------------------------------------------------------------------------------------
# Rank-major data-parallel helpers
# ---------------------------------------------------------------------------------------


def _dp_setup(
    dp_sharding: tuple[tp.Any, tp.Any] | None,
    *,
    total_tokens: int,
    num_rows: int,
    num_pages: int,
) -> tuple[jax.sharding.Mesh, tuple[str, ...], int]:
    """Resolve ``(mesh, page_axis_names, dp_size)`` for a rank-major selector call.

    Raises:
        ValueError: When no DP sharding is given or a dimension does not divide the DP size.
    """
    if dp_sharding is None:
        raise ValueError(
            "Rank-major (2-D query_start_loc) metadata needs dp_sharding=(mesh, page_axis_names) "
            "(see MLARaggedPagesCacheView.indexer_dp_sharding())."
        )
    mesh, names = dp_sharding
    mesh = getattr(mesh, "jax_mesh", mesh)
    names = tuple(names) if isinstance(names, (tuple, list)) else (names,)
    dp = 1
    for name in names:
        dp *= int(mesh.shape[name])
    for label, size in (("tokens", total_tokens), ("request rows", num_rows), ("pages", num_pages)):
        if int(size) % dp:
            raise ValueError(f"Rank-major DP indexer: {label}={size} is not divisible by the DP size {dp}.")
    return mesh, names, dp


def _dp_rank_local(
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    axis_names: tuple[str, ...],
    dp_size: int,
    local_num_pages: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Slice rank-major serving metadata to the calling DP rank (inside ``shard_map``).

    Mirrors the DP-local MLA attention path: rows are contiguous per rank, block tables hold
    global page ids and are rebased to the rank's local page shard.
    """
    from easydel.infra.sharding import axis_index

    shard_idx = axis_index(axis_names)
    kv_lens = kv_lens.reshape(-1)
    block_tables = block_tables.reshape(-1, block_tables.shape[-1])
    rows = int(kv_lens.shape[0]) // dp_size
    row_start = shard_idx * jnp.int32(rows)
    local_kv = lax.dynamic_slice_in_dim(kv_lens, row_start, rows, axis=0)
    local_tables = lax.dynamic_slice_in_dim(block_tables, row_start, rows, axis=0)
    local_tables = jnp.clip(local_tables - shard_idx * jnp.int32(local_num_pages), 0, local_num_pages - 1)
    local_qsl = query_start_loc[shard_idx]
    num_seqs = jnp.asarray(num_seqs, jnp.int32).reshape(-1)
    local_num = num_seqs[shard_idx] if num_seqs.shape[0] == dp_size else jnp.int32(rows)
    return local_kv, local_tables, local_qsl, local_num


# ---------------------------------------------------------------------------------------
# k-pool selector (glm5_next)
# ---------------------------------------------------------------------------------------


def _pool_select_core(
    query: jax.Array,
    key: jax.Array,
    gate: jax.Array,
    head_weights: jax.Array,
    ape: jax.Array,
    token_pages: jax.Array,
    pool_pages: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    kpool: int,
    select_k: int,
    sm_scale: float,
    relu: bool,
    precision: tp.Any,
    chunk: int,
    rows_c: tp.Callable[[jax.Array], jax.Array],
    pages_c: tp.Callable[[jax.Array], jax.Array],
    kernel_mesh: object | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Page update + pool selection for one set of 1-D ragged metadata (a whole batch or one DP rank)."""
    num_pages, page_size, packed_dim = token_pages.shape
    dim = packed_dim // 2
    pools_per_page = page_size // kpool
    pages_per_seq = int(block_tables.shape[1])
    total_tokens = int(query.shape[0])

    seq, pos, valid = paged_token_layout(kv_lens, query_start_loc, num_seqs, total_tokens)

    def physical_row(token_pos: jax.Array, seq_ids: jax.Array) -> jax.Array:
        logical = jnp.clip(token_pos // page_size, 0, pages_per_seq - 1)
        return block_tables[seq_ids, logical] * page_size + token_pos % page_size

    # 1) Write the current tokens' ``key | gate`` rows.
    current = rows_c(jnp.concatenate([key, gate.astype(key.dtype)], axis=-1).astype(token_pages.dtype))
    flat_tokens = pages_c(token_pages.reshape(num_pages * page_size, packed_dim))
    dst = jnp.where(valid, physical_row(jnp.maximum(pos, 0), seq), num_pages * page_size)
    flat_tokens = pages_c(flat_tokens.at[dst].set(current, mode="drop"))

    # 2) Summarise the pools whose last member arrived this step (members share a page).
    completes = valid & ((pos + 1) % kpool == 0)
    member_pos = jnp.maximum(pos - (kpool - 1), 0)[:, None] + jnp.arange(kpool, dtype=jnp.int32)[None]
    members = flat_tokens[physical_row(member_pos, seq[:, None])]  # [T, kpool, 2D]
    member_keys = members[..., :dim].astype(key.dtype)
    member_gates = members[..., dim:].astype(key.dtype).astype(jnp.float32)
    probabilities = jax.nn.softmax(member_gates + ape[None], axis=1)
    summary = jnp.sum(probabilities * member_keys.astype(jnp.float32), axis=1).astype(key.dtype)

    flat_pools = pool_pages.reshape(num_pages * pools_per_page, dim)
    pool_page = block_tables[seq, jnp.clip(jnp.maximum(pos, 0) // page_size, 0, pages_per_seq - 1)]
    pool_dst = pool_page * pools_per_page + (jnp.maximum(pos, 0) % page_size) // kpool
    pool_dst = jnp.where(completes, pool_dst, num_pages * pools_per_page)
    flat_pools = pages_c(flat_pools.at[pool_dst].set(rows_c(summary.astype(pool_pages.dtype)), mode="drop"))

    # 3) Exact top-k over each token's complete pools through the page table, one fused
    #    pass on TPU (ejkernel ``paged_pool_topk``), the running-top-k loop elsewhere.
    num_complete = jnp.where(valid, (pos + 1) // kpool, 0)
    best_ids = _run_pool_topk(
        kernel_mesh,
        query.astype(jnp.float32),
        head_weights,
        flat_pools.reshape(pool_pages.shape),
        block_tables,
        seq,
        num_complete,
        k=select_k,
        sm_scale=sm_scale,
        relu=relu,
        precision=precision,
        pool_chunk=chunk,
    )

    # 4) The incomplete tail pool is always attended (select_tail).
    has_tail = valid & ((pos + 1) % kpool != 0)
    tail = jnp.where(has_tail, (pos + 1) // kpool, -1)[:, None]
    selected = jnp.concatenate([best_ids, tail], axis=-1)
    selected = jnp.where(valid[:, None], selected, -1).astype(jnp.int32)
    return (
        jax.lax.stop_gradient(selected),
        pages_c(flat_tokens.reshape(token_pages.shape)),
        pages_c(flat_pools.reshape(pool_pages.shape)),
    )


def paged_pool_select(
    indexer: SparseIndexer,
    hidden_states: jax.Array,
    q_resid: jax.Array | None,
    token_pages: jax.Array,
    pool_pages: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    pool_chunk: int = 1024,
    constrain_rows: tp.Callable[[jax.Array], jax.Array] | None = None,
    constrain_pages: tp.Callable[[jax.Array], jax.Array] | None = None,
    kernel_mesh: object | None = None,
    dp_sharding: tuple[tp.Any, tp.Any] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Update the paged indexer state and select pools for every packed token.

    Args:
        indexer: A ``kind="pool"`` :class:`SparseIndexer` (projections, gate, ape).
        hidden_states: ``[T, hidden]`` packed layer inputs.
        q_resid: ``[T, q_input_dim]`` MLA low-rank residual (``query_source="q_lora"``).
        token_pages: ``[P, page_size, 2 * dim]`` per-token ``key | gate`` pages.
        pool_pages: ``[P, page_size // kpool, dim]`` completed-pool summaries.
        block_tables: ``[S, pages_per_seq]`` page table shared with the KV cache.
        kv_lens: ``[S]`` context lengths including the current step.
        query_start_loc: ``[S + 1]`` cumulative query counts, or rank-major ``[dp, S/dp + 1]``
            with rank-local token offsets (data-parallel serving).
        num_seqs: Scalar number of live sequences, or ``[dp]`` per rank (rank-major).
        pool_chunk: Pools scored per loop step (static). Each step is a serial ``top_k`` merge,
            so small chunks are latency-bound at long context: on TPU v5 with GLM-5.3 shapes
            (kpool 4, 512 kept pools) 128 cost 3.3 ms/layer at 28k vs 0.88 ms for 1024, and
            larger chunks slow short contexts (the whole static chunk is always scored).
        constrain_rows: Optional sharding constraint for scattered update rows (see
            ``MLARaggedPagesCacheView.constrain_indexer_rows``). Unused on the rank-major path.
        constrain_pages: Optional sharding constraint keeping page buffers in their
            allocation layout so they stay donated (``constrain_indexer_pages``).
        kernel_mesh: Mesh on which the fused TPU top-k runs replicated
            (``MLARaggedPagesCacheView.indexer_kernel_mesh()``); ``None`` on a
            multi-device mesh keeps the partitioned XLA path.
        dp_sharding: ``(mesh, page_axis_names)`` of the DP-sharded indexer pages
            (``MLARaggedPagesCacheView.indexer_dp_sharding()``); required for rank-major metadata.

    Returns:
        ``(selected [T, index_topk // kpool + 1], token_pages, pool_pages)``: chunk
        (pool) ids, ``-1``-padded, unique per row; the last column is the incomplete
        tail pool. Padding tokens select nothing.

    Raises:
        ValueError: On a non-pool indexer, a page size that pools do not tile, or rank-major
            metadata without a usable ``dp_sharding``.
    """
    config = indexer.config
    if config.kind != "pool":
        raise ValueError("paged_pool_select requires a kind='pool' indexer.")
    kpool = int(config.kpool_size)
    dim = int(indexer.head_dim)
    num_pages, page_size, packed_dim = token_pages.shape
    if packed_dim != 2 * dim:
        raise ValueError(f"token_pages last dim must be 2*index_head_dim={2 * dim}, got {packed_dim}.")
    if page_size % kpool:
        raise ValueError(f"page_size ({page_size}) must be a multiple of kpool ({kpool}).")
    pools_per_page = page_size // kpool
    if pool_pages.shape != (num_pages, pools_per_page, dim):
        raise ValueError(f"pool_pages must be {(num_pages, pools_per_page, dim)}, got {pool_pages.shape}.")
    qsl = jnp.asarray(query_start_loc, jnp.int32)
    if qsl.ndim == 1 and block_tables.ndim != 2:
        raise ValueError("block_tables must be [num_seqs, pages_per_seq].")
    total_tokens = int(hidden_states.shape[0])

    # Projections run outside any per-rank map so tensor-parallel weights stay auto-partitioned.
    query, key, gate = indexer._project(hidden_states[None], None if q_resid is None else q_resid[None], None, None)
    query, key, gate = query[0], key[0], gate[0]  # [T,H,D], [T,D], [T,D] (f32)
    if config.head_reduction == "uniform":
        head_weights = jnp.ones(query.shape[:2], jnp.float32)
    else:
        head_weights = indexer.weights_proj(hidden_states).astype(jnp.float32) * (indexer.n_heads**-0.5)
    ape = indexer.index_kpool_compress_ape.value.astype(jnp.float32)

    core = functools.partial(
        _pool_select_core,
        kpool=kpool,
        select_k=int(config.index_topk) // kpool,
        sm_scale=float(indexer.softmax_scale),
        relu=config.score_activation == "relu",
        # One resolved precision for both backends: the Pallas interface defaults ``None`` to
        # HIGHEST while the XLA loop would use the ambient default, so near-tied pool scores
        # could rank differently depending on the dispatch path.
        precision=indexer.precision if indexer.precision is not None else jax.lax.Precision.HIGHEST,
        chunk=max(1, int(pool_chunk)),
    )

    if qsl.ndim == 1:
        return core(
            query,
            key,
            gate,
            head_weights,
            ape,
            token_pages,
            pool_pages,
            block_tables,
            kv_lens,
            qsl,
            num_seqs,
            rows_c=constrain_rows or _identity,
            pages_c=constrain_pages or _identity,
            kernel_mesh=kernel_mesh,
        )

    mesh, names, dp = _dp_setup(
        dp_sharding, total_tokens=total_tokens, num_rows=int(jnp.size(kv_lens)), num_pages=int(num_pages)
    )

    def per_rank(q, k, g, w, ape_, tok, pool, tables, lens, qsl_, nseq):
        local_kv, local_tables, local_qsl, local_num = _dp_rank_local(
            lens, tables, qsl_, nseq, axis_names=names, dp_size=dp, local_num_pages=int(tok.shape[0])
        )
        return core(
            q,
            k,
            g,
            w,
            ape_,
            tok,
            pool,
            local_tables,
            local_kv,
            local_qsl,
            local_num,
            rows_c=_identity,
            pages_c=_identity,
            kernel_mesh=None,
        )

    row, rep = PartitionSpec(names), PartitionSpec()
    return jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=(row, row, row, row, rep, row, row, rep, rep, rep, rep),
        out_specs=(row, row, row),
        check_vma=False,
    )(
        query,
        key,
        gate,
        head_weights,
        ape,
        token_pages,
        pool_pages,
        jnp.asarray(block_tables, jnp.int32),
        jnp.asarray(kv_lens, jnp.int32),
        qsl,
        jnp.asarray(num_seqs, jnp.int32),
    )


# ---------------------------------------------------------------------------------------
# Context-bounded selection over per-slot dense key buffers (DeepSeek-V4.1 shared keys)
# ---------------------------------------------------------------------------------------


def _page_size_for(entries: int, preferred: int = 128) -> int:
    """Largest page size dividing both ``128`` (Pallas tiling) and ``entries``, else ``preferred`` (padded)."""
    size = preferred
    while size > 1:
        if entries % size == 0:
            return size
        size //= 2
    return preferred


def slot_buffer_pages(keys: jax.Array, page_size: int) -> tuple[jax.Array, jax.Array]:
    """View a per-slot key buffer ``[B, T, D]`` as pages plus a trivial page table.

    Slot ``b`` owns pages ``b * n .. b * n + n - 1`` with ``n = ceil(T / page_size)``, so a dense per-slot cache (or a
    training batch) is read through the same paged kernels as page-table caches. ``T`` is zero-padded to a page
    multiple only when it is not one already (decode buffers sized ``max_len // ratio`` normally are).

    Args:
        keys: Per-slot keys ``[B, T, D]``.
        page_size: Entries per page.

    Returns:
        ``(pages [B * n, page_size, D], block_tables [B, n])``.
    """
    batch, entries, dim = keys.shape
    pad = (-entries) % page_size
    if pad:
        keys = jnp.pad(keys, ((0, 0), (0, pad), (0, 0)))
    pages_per_slot = (entries + pad) // page_size
    pages = keys.reshape(batch * pages_per_slot, page_size, dim)
    tables = jnp.arange(batch * pages_per_slot, dtype=jnp.int32).reshape(batch, pages_per_slot)
    return pages, tables


def bounded_entry_topk(
    query: jax.Array,
    head_weights: jax.Array,
    keys: jax.Array,
    num_reachable: jax.Array,
    k: int,
    *,
    sm_scale: float = 1.0,
    relu: bool = True,
    precision: tp.Any = None,
    chunk: int = 1024,
    kernel_mesh: object | None = None,
) -> jax.Array:
    """Exact top-``k`` entries per query over its reachable prefix of a per-slot key buffer.

    ``score[b, s, t] = sum_h head_weights[b, s, h] * act(sm_scale * <query[b, s, h], keys[b, t]>)`` for
    ``t < num_reachable[b, s]``. Runs the registered ``paged_pool_topk`` kernel (fused Pallas on TPU, running-top-k XLA
    loop elsewhere) through :func:`slot_buffer_pages`, so the cost follows the live context, never the allocated
    buffer length.

    Args:
        query: Indexer queries ``[B, S, H, D]``.
        head_weights: Per-head weights ``[B, S, H]`` (all scaling folded in).
        keys: Per-slot keys ``[B, T, D]``; entries at or beyond ``num_reachable`` are never read.
        num_reachable: Reachable entry count per query ``[B, S]``.
        k: Entries kept per query (static).
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        chunk: Entries scored per running-top-k step (XLA path only).
        kernel_mesh: Mesh for the replicated-page Pallas dispatch (see :func:`_run_pool_topk`).

    Returns:
        ``[B, S, min(k, T)]`` int32 entry ids, ascending, ``-1``-padded: the set ``jax.lax.top_k`` keeps over the
        masked scores (ties to the lowest id) with unreachable picks dropped.
    """
    batch, seq, heads, dim = query.shape
    entries = int(keys.shape[1])
    k = min(int(k), entries)
    if k == 0 or seq == 0:
        return jnp.full((batch, seq, k), -1, jnp.int32)
    pages, tables = slot_buffer_pages(keys, _page_size_for(entries))
    rows = batch * seq
    ids = _run_pool_topk(
        kernel_mesh,
        query.reshape(rows, heads, dim).astype(jnp.float32),
        head_weights.reshape(rows, heads).astype(jnp.float32),
        pages,
        tables,
        jnp.repeat(jnp.arange(batch, dtype=jnp.int32), seq),
        jnp.clip(num_reachable.reshape(rows).astype(jnp.int32), 0, entries),
        k=k,
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=precision,
        pool_chunk=int(chunk),
    )
    return jax.lax.stop_gradient(ids.reshape(batch, seq, k))


def _slot_chunk_scores(
    query: jax.Array,
    head_weights: jax.Array,
    chunk_keys: jax.Array,
    *,
    sm_scale: float,
    relu: bool,
    precision: tp.Any,
) -> jax.Array:
    """``sum_h w[b, s, h] * act(sm_scale * <q[b, s, h], k[b, c]>)`` as fp32 ``[B, S, C]``.

    Heads accumulate in a ``fori_loop`` so no ``[B, S, H, C]`` intermediate exists (training rows can number in the
    tens of thousands; the head axis would multiply the chunk memory by ``H``).
    """
    batch, seq, heads, _ = query.shape
    keys = chunk_keys.astype(jnp.float32)

    def head(h, acc):
        qh = jax.lax.dynamic_index_in_dim(query, h, axis=2, keepdims=False)
        wh = jax.lax.dynamic_index_in_dim(head_weights, h, axis=2, keepdims=False)
        s = jnp.einsum("bsd,bcd->bsc", qh, keys, precision=precision) * sm_scale
        if relu:
            s = jax.nn.relu(s)
        return acc + s * wh[..., None]

    init = jnp.zeros((batch, seq, keys.shape[1]), jnp.float32)
    return jax.lax.fori_loop(0, heads, head, init)


def running_entry_topk(
    query: jax.Array,
    head_weights: jax.Array,
    keys: jax.Array,
    num_reachable: jax.Array,
    k: int,
    *,
    candidate_blocks: jax.Array | None = None,
    block_size: int = 1,
    sm_scale: float = 1.0,
    relu: bool = True,
    precision: tp.Any = None,
    chunk: int = 2048,
) -> jax.Array:
    """Running exact top-``k`` over each query's reachable prefix of its slot's keys (many queries per slot).

    Same contract as :func:`bounded_entry_topk` / :func:`bounded_candidate_entry_topk`, but the keys of a chunk are
    shared by every query of the slot (``[B, C, D]``), so training / prefill rows never gather per-row key copies.
    Only chunks up to the longest reachable prefix in the batch are visited; per chunk the live memory is
    ``[B, S, C]`` fp32.

    Args:
        query: Indexer queries ``[B, S, H, D]``.
        head_weights: Per-head weights ``[B, S, H]``.
        keys: Per-slot keys ``[B, T, D]``.
        num_reachable: Reachable entry count per query ``[B, S]``.
        k: Entries kept per query (static).
        candidate_blocks: Optional candidate block ids ``[B, S, KB]`` (ascending, ``-1``-padded); entries outside
            them are not eligible.
        block_size: Entries per candidate block (static).
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        chunk: Entries per loop step (static; rounded up to a block multiple with candidates).

    Returns:
        ``[B, S, min(k, T)]`` int32 entry ids, ascending, ``-1``-padded.
    """
    batch, seq, _, _ = query.shape
    entries = int(keys.shape[1])
    width = min(int(k), entries)
    if width == 0 or seq == 0:
        return jnp.full((batch, seq, width), -1, jnp.int32)
    c = max(1, min(int(chunk), entries))
    if candidate_blocks is not None:
        c = -(-c // block_size) * block_size
    padded = -(-entries // c) * c
    if padded != entries:
        keys = jnp.pad(keys, ((0, 0), (0, padded - entries), (0, 0)))
    q = query.astype(jnp.float32)
    w = head_weights.astype(jnp.float32)
    lens = jnp.clip(num_reachable.astype(jnp.int32), 0, entries)
    num_chunks = (jnp.max(lens, initial=0) + c - 1) // c
    off = jnp.arange(c, dtype=jnp.int32)
    chunk_k = min(width, c)
    big = jnp.iinfo(jnp.int32).max
    if candidate_blocks is not None:
        cand = candidate_blocks.astype(jnp.int32)
        cand = jnp.where(cand >= 0, cand, big)  # stays ascending
        search = jax.vmap(jax.vmap(jnp.searchsorted))
        blocks_per_chunk = c // block_size

    def body(carry):
        step, best_vals, best_ids = carry
        start = step * c
        chunk_keys = jax.lax.dynamic_slice_in_dim(keys, start, c, axis=1)
        scores = _slot_chunk_scores(q, w, chunk_keys, sm_scale=sm_scale, relu=relu, precision=precision)
        ids = start + off
        ok = ids < lens[..., None]
        if candidate_blocks is not None:
            blk = start // block_size + jnp.arange(blocks_per_chunk, dtype=jnp.int32)
            blk = jnp.broadcast_to(blk, (batch, seq, blocks_per_chunk))
            pos = jnp.clip(search(cand, blk), 0, cand.shape[-1] - 1)
            member = jnp.take_along_axis(cand, pos, axis=-1) == blk
            ok = ok & jnp.repeat(member, block_size, axis=-1)
        scores = jnp.where(ok, scores, -jnp.inf)
        # Two-stage merge: the chunk's own top-k (ids are ``start + pos``, no gather), then a merge with the carried
        # best. The id gather reads a ``[B, S, 2k]`` operand instead of ``[B, S, k + C]`` (measured on TPU at S=64k:
        # that gather was 3.4 s of a 20.6 s training step). Same id set: any chunk entry the full merge keeps is in
        # the chunk's top-k (ties to the lowest position), and carried entries still precede chunk entries.
        chunk_vals, chunk_pos = jax.lax.top_k(scores, chunk_k)
        merged_vals = jnp.concatenate([best_vals, chunk_vals], axis=-1)
        merged_ids = jnp.concatenate([best_ids, start + chunk_pos.astype(jnp.int32)], axis=-1)
        top_vals, top_pos = jax.lax.top_k(merged_vals, width)
        return step + 1, top_vals, jnp.take_along_axis(merged_ids, top_pos, axis=-1)

    init = (
        jnp.int32(0),
        jnp.full((batch, seq, width), -jnp.inf, jnp.float32),
        jnp.full((batch, seq, width), -1, jnp.int32),
    )
    _, best_vals, best_ids = jax.lax.while_loop(lambda cr: cr[0] < num_chunks, body, init)
    ids = jnp.sort(jnp.where(best_vals > -jnp.inf, best_ids, big), axis=-1)
    return jax.lax.stop_gradient(jnp.where(ids == big, -1, ids).astype(jnp.int32))


def slot_entry_topk(
    query: jax.Array,
    head_weights: jax.Array,
    keys: jax.Array,
    num_reachable: jax.Array,
    k: int,
    *,
    candidate_blocks: jax.Array | None = None,
    block_size: int = 1,
    sm_scale: float = 1.0,
    relu: bool = True,
    precision: tp.Any = None,
    kernel_mesh: object | None = None,
) -> jax.Array:
    """Context-bounded top-``k`` entry selection over per-slot key buffers (one entry point for every phase).

    Candidate users (any row count) and plain single-query rows go through the registered ``paged_pool_topk`` kernel
    (:func:`bounded_candidate_entry_topk` / :func:`bounded_entry_topk`); plain multi-query rows (training, prefill)
    use :func:`running_entry_topk`, which shares each chunk of slot keys across the slot's queries. None reads entries
    beyond each query's reach nor materializes a ``[B, S, T]`` score tensor.

    Args:
        query: Indexer queries ``[B, S, H, D]``.
        head_weights: Per-head weights ``[B, S, H]``.
        keys: Per-slot keys ``[B, T, D]``.
        num_reachable: Reachable entry count per query ``[B, S]``.
        k: Entries kept per query (static).
        candidate_blocks: Optional candidate block ids ``[B, S, KB]`` from :func:`bounded_block_max_topk`.
        block_size: Entries per candidate block.
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        kernel_mesh: Mesh for the replicated-page Pallas dispatch.

    Returns:
        ``[B, S, min(k, T)]`` int32 entry ids, ascending, ``-1``-padded.
    """
    kwargs = dict(sm_scale=sm_scale, relu=relu, precision=precision)
    if candidate_blocks is not None:
        # Candidate users read only their candidate pages through the paged kernel, for any number of rows: the cost is
        # ``O(S * KB * block_size)`` instead of scoring every reachable entry and masking (measured on one TPU v5 at
        # S=16k: 153 ms vs 1.83 s, identical ids).
        return bounded_candidate_entry_topk(
            query,
            head_weights,
            keys,
            candidate_blocks,
            num_reachable,
            k,
            block_size,
            kernel_mesh=kernel_mesh,
            **kwargs,
        )
    if query.shape[1] == 1:
        return bounded_entry_topk(query, head_weights, keys, num_reachable, k, kernel_mesh=kernel_mesh, **kwargs)
    # Plain many-query rows: the shared-key running top-k beats the per-row paged kernel here (the slot's key chunk is
    # scored once for all its queries).
    return running_entry_topk(query, head_weights, keys, num_reachable, k, **kwargs)


def bounded_block_max_topk(
    query: jax.Array,
    head_weights: jax.Array,
    keys: jax.Array,
    num_reachable: jax.Array,
    topk_blocks: int,
    block_size: int,
    *,
    sm_scale: float = 1.0,
    relu: bool = True,
    precision: tp.Any = None,
    chunk_blocks: int = 128,
) -> jax.Array:
    """Level one of a two-level top-k: each query's ``topk_blocks`` best blocks of ``block_size`` entries.

    A block scores the max of its reachable entries (same score as :func:`bounded_entry_topk`); the block holding
    the query's newest reachable entry is always kept and blocks without a reachable entry are never picked. Equal to
    ranking every block with ``jax.lax.top_k`` (ties to the lowest block id), but a running top-k over chunks of
    ``chunk_blocks`` blocks only visits blocks up to the longest reachable prefix in the batch.

    Args:
        query: Indexer queries ``[B, S, H, D]``.
        head_weights: Per-head weights ``[B, S, H]``.
        keys: Per-slot keys ``[B, T, D]``.
        num_reachable: Reachable entry count per query ``[B, S]``.
        topk_blocks: Blocks kept per query (static).
        block_size: Entries per block (static).
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        chunk_blocks: Blocks scored per loop step (static).

    Returns:
        ``[B, S, min(topk_blocks, ceil(T / block_size))]`` int32 block ids, ascending, ``-1``-padded.
    """
    batch, seq, _, _ = query.shape
    entries = int(keys.shape[1])
    num_blocks = -(-entries // block_size)
    kb = min(int(topk_blocks), num_blocks)
    if kb == 0 or seq == 0:
        return jnp.full((batch, seq, kb), -1, jnp.int32)
    cb = max(1, min(int(chunk_blocks), num_blocks))
    span = cb * block_size
    padded = -(-entries // span) * span
    if padded != entries:
        keys = jnp.pad(keys, ((0, 0), (0, padded - entries), (0, 0)))
    q = query.astype(jnp.float32)
    w = head_weights.astype(jnp.float32)
    lens = jnp.clip(num_reachable.astype(jnp.int32), 0, entries)
    last = jnp.where(lens > 0, (lens - 1) // block_size, -1)
    num_chunks = (jnp.max(lens, initial=0) + span - 1) // span
    entry_off = jnp.arange(span, dtype=jnp.int32)
    block_off = jnp.arange(cb, dtype=jnp.int32)

    def body(carry):
        step, best_vals, best_ids = carry
        start = step * span
        chunk_keys = jax.lax.dynamic_slice_in_dim(keys, start, span, axis=1)
        scores = _slot_chunk_scores(q, w, chunk_keys, sm_scale=sm_scale, relu=relu, precision=precision)
        scores = jnp.where(start + entry_off < lens[..., None], scores, -jnp.inf)
        ids = step * cb + block_off
        blocks = scores.reshape(batch, seq, cb, block_size).max(axis=-1)
        blocks = jnp.where(ids == last[..., None], jnp.inf, blocks)
        # Two-stage merge (see :func:`running_entry_topk`): chunk top-k first, so the id gather stays ``[B, S, 2 kb]``.
        chunk_vals, chunk_pos = jax.lax.top_k(blocks, min(kb, cb))
        merged_vals = jnp.concatenate([best_vals, chunk_vals], axis=-1)
        merged_ids = jnp.concatenate([best_ids, step * cb + chunk_pos.astype(jnp.int32)], axis=-1)
        top_vals, top_pos = jax.lax.top_k(merged_vals, kb)
        return step + 1, top_vals, jnp.take_along_axis(merged_ids, top_pos, axis=-1)

    init = (
        jnp.int32(0),
        jnp.full((batch, seq, kb), -jnp.inf, jnp.float32),
        jnp.full((batch, seq, kb), -1, jnp.int32),
    )
    _, best_vals, best_ids = jax.lax.while_loop(lambda c: c[0] < num_chunks, body, init)
    big = jnp.iinfo(jnp.int32).max
    ids = jnp.sort(jnp.where(best_vals > -jnp.inf, best_ids, big), axis=-1)
    return jax.lax.stop_gradient(jnp.where(ids == big, -1, ids).astype(jnp.int32))


def bounded_candidate_entry_topk(
    query: jax.Array,
    head_weights: jax.Array,
    keys: jax.Array,
    candidate_blocks: jax.Array,
    num_reachable: jax.Array,
    k: int,
    block_size: int,
    *,
    sm_scale: float = 1.0,
    relu: bool = True,
    precision: tp.Any = None,
    chunk: int = 1024,
    kernel_mesh: object | None = None,
) -> jax.Array:
    """Exact top-``k`` entries per query, scored only inside its published candidate blocks.

    Each query's candidate block ids (ascending, ``-1``-padded, newest reachable block included, as returned by
    :func:`bounded_block_max_topk` for the same reach) become its page table over ``block_size``-entry pages, so the
    registered ``paged_pool_topk`` kernel reads only candidate entries; the reachable ones form a prefix of that
    logical layout because only the last candidate block can be partial.

    Args:
        query: Indexer queries ``[B, S, H, D]``.
        head_weights: Per-head weights ``[B, S, H]``.
        keys: Per-slot keys ``[B, T, D]``.
        candidate_blocks: Candidate block ids ``[B, S, KB]``.
        num_reachable: Reachable entry count per query ``[B, S]``.
        k: Entries kept per query (static).
        block_size: Entries per candidate block (static).
        sm_scale: Score scale before the activation.
        relu: Per-head ``relu`` before weighting.
        precision: Matmul precision.
        chunk: Entries scored per running-top-k step (XLA path only).
        kernel_mesh: Mesh for the replicated-page Pallas dispatch.

    Returns:
        ``[B, S, min(k, T)]`` int32 entry ids, ascending, ``-1``-padded.
    """
    batch, seq, heads, dim = query.shape
    entries = int(keys.shape[1])
    width = min(int(k), entries)
    kb = int(candidate_blocks.shape[-1])
    k_eff = min(width, kb * block_size)
    if k_eff == 0 or seq == 0:
        return jnp.full((batch, seq, width), -1, jnp.int32)
    num_blocks = -(-entries // block_size)
    pad = num_blocks * block_size - entries
    if pad:
        keys = jnp.pad(keys, ((0, 0), (0, pad), (0, 0)))
    pages = keys.reshape(batch * num_blocks, block_size, dim)
    rows = batch * seq
    cand = candidate_blocks.reshape(rows, kb).astype(jnp.int32)
    valid = cand >= 0
    n_valid = jnp.sum(valid, axis=-1).astype(jnp.int32)
    lens = jnp.clip(num_reachable.reshape(rows).astype(jnp.int32), 0, entries)
    newest = jnp.take_along_axis(cand, jnp.maximum(n_valid - 1, 0)[:, None], axis=1)[:, 0]
    tail = jnp.clip(lens - newest * block_size, 0, block_size)
    reachable = jnp.where(n_valid > 0, (n_valid - 1) * block_size + tail, 0)
    slot = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), seq)
    tables = jnp.where(valid, slot[:, None] * num_blocks + cand, 0)
    logical = _run_pool_topk(
        kernel_mesh,
        query.reshape(rows, heads, dim).astype(jnp.float32),
        head_weights.reshape(rows, heads).astype(jnp.float32),
        pages,
        tables,
        jnp.arange(rows, dtype=jnp.int32),
        reachable,
        k=k_eff,
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=precision,
        pool_chunk=int(chunk),
    )
    block = jnp.take_along_axis(cand, jnp.clip(logical // block_size, 0, kb - 1), axis=1)
    entry = jnp.where(logical >= 0, block * block_size + logical % block_size, -1)
    if k_eff < width:
        entry = jnp.concatenate([entry, jnp.full((rows, width - k_eff), -1, jnp.int32)], axis=-1)
    return jax.lax.stop_gradient(entry.reshape(batch, seq, width).astype(jnp.int32))


# ---------------------------------------------------------------------------------------
# Per-token selector (glm_moe_dsa)
# ---------------------------------------------------------------------------------------


def _token_select_core(
    query: jax.Array,
    key: jax.Array,
    head_weights: jax.Array,
    freq: jax.Array,
    token_pages: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    rope_dim: int,
    rope_fn: tp.Callable | None,
    softmax_scale: float,
    select_k: int,
    chunk: int,
    precision: tp.Any,
    rows_c: tp.Callable[[jax.Array], jax.Array],
    pages_c: tp.Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Key-page update + running top-k for one set of 1-D ragged metadata."""
    num_pages, page_size, dim = token_pages.shape
    pages_per_seq = int(block_tables.shape[1])
    total_tokens = int(query.shape[0])

    seq, pos, valid = paged_token_layout(kv_lens, query_start_loc, num_seqs, total_tokens)

    if rope_dim > 0:
        # RoPE at the position the token is written to (derived from the page layout).
        rope_pos = jnp.clip(pos, 0, int(freq.shape[0]) - 1)
        cos, sin = jnp.split(freq[rope_pos], 2, axis=-1)  # [T, half]
        cos, sin = cos[None, :, : rope_dim // 2], sin[None, :, : rope_dim // 2]
        q_pe = rope_fn(query[None, ..., :rope_dim], cos, sin)[0]
        k_pe = rope_fn(key[None, :, None, :rope_dim], cos, sin)[0, :, 0]
        query = jnp.concatenate([q_pe, query[..., rope_dim:]], axis=-1)
        key = jnp.concatenate([k_pe, key[..., rope_dim:]], axis=-1)

    def physical_row(token_pos: jax.Array, seq_ids: jax.Array) -> jax.Array:
        logical = jnp.clip(token_pos // page_size, 0, pages_per_seq - 1)
        return block_tables[seq_ids, logical] * page_size + token_pos % page_size

    flat = pages_c(token_pages.reshape(num_pages * page_size, dim))
    dst = jnp.where(valid, physical_row(jnp.maximum(pos, 0), seq), num_pages * page_size)
    flat = pages_c(flat.at[dst].set(rows_c(key.astype(token_pages.dtype)), mode="drop"))

    visible = jnp.where(valid, pos + 1, 0)
    num_chunks = (jnp.max(visible) + chunk - 1) // chunk
    offsets = jnp.arange(chunk, dtype=jnp.int32)

    def body(carry):
        step, best_vals, best_ids = carry
        ids = step * chunk + offsets  # [C]
        ok = ids[None, :] < visible[:, None]  # [T, C]
        rows = physical_row(jnp.where(ok, ids[None, :], 0), seq[:, None])
        keys = flat[rows].astype(jnp.float32)  # [T, C, D]
        scores = jnp.einsum("thd,tcd->thc", query, keys, precision=precision) * softmax_scale
        scores = jax.nn.relu(scores)
        scores = jnp.einsum("thc,th->tc", scores, head_weights, precision=precision)
        scores = jnp.where(ok, scores, -jnp.inf)
        merged_vals = jnp.concatenate([best_vals, scores], axis=-1)
        merged_ids = jnp.concatenate([best_ids, jnp.broadcast_to(ids[None], scores.shape)], axis=-1)
        top_vals, top_pos = jax.lax.top_k(merged_vals, select_k)
        return step + 1, top_vals, jnp.take_along_axis(merged_ids, top_pos, axis=-1)

    init = (
        jnp.int32(0),
        jnp.full((total_tokens, select_k), -jnp.inf, jnp.float32),
        jnp.full((total_tokens, select_k), -1, jnp.int32),
    )
    _, best_vals, best_ids = jax.lax.while_loop(lambda c: c[0] < num_chunks, body, init)
    selected = jnp.where((best_vals > -jnp.inf) & valid[:, None], best_ids, -1).astype(jnp.int32)
    return jax.lax.stop_gradient(selected), pages_c(flat.reshape(token_pages.shape))


def paged_token_select(
    indexer: TokenIndexer,
    hidden_states: jax.Array,
    q_resid: jax.Array | None,
    frequencies: jax.Array | None,
    token_pages: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    key_chunk: int = 256,
    constrain_rows: tp.Callable[[jax.Array], jax.Array] | None = None,
    constrain_pages: tp.Callable[[jax.Array], jax.Array] | None = None,
    dp_sharding: tuple[tp.Any, tp.Any] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Update the paged key state of a per-token :class:`TokenIndexer` and select tokens.

    Mirrors ``GlmMoeDsaIndexerOp.forward_native``: keys are ``k_norm(wk(h))`` with RoPE on
    the leading ``rope_dim`` channels (stored post-RoPE), scores are
    ``sum_h w[h] * relu((q[h] . k) * scale)`` with ``w = kernels_proj(h) / sqrt(heads)``, and each
    query keeps the ``index_topk`` best causally visible keys. The running top-k keeps the
    current best (lower token ids) ahead of each new chunk, so ties resolve lowest-id-first
    like ``jax.lax.top_k`` over the full key axis. Only keys up to each request's own
    position are read, never a ``max_model_len`` window.

    Args:
        indexer: The layer's :class:`TokenIndexer`.
        hidden_states: ``[T, hidden]`` packed layer inputs.
        q_resid: ``[T, q_input_dim]`` MLA low-rank residual, or ``None``.
        frequencies: RoPE table ``[max_pos, 2 * half]`` or ``None`` (no RoPE).
        token_pages: ``[P, page_size, index_head_dim]`` post-RoPE key pages.
        block_tables: ``[S, pages_per_seq]`` page table shared with the KV cache.
        kv_lens: ``[S]`` context lengths including the current step.
        query_start_loc: ``[S + 1]`` cumulative query counts, or rank-major ``[dp, S/dp + 1]``.
        num_seqs: Scalar number of live sequences, or ``[dp]`` per rank (rank-major).
        key_chunk: Keys scored per loop step (static).
        constrain_rows: Optional sharding constraint for scattered update rows.
        constrain_pages: Optional sharding constraint keeping the page buffer in its
            allocation layout so it stays donated.
        dp_sharding: ``(mesh, page_axis_names)`` of the DP-sharded pages; required for
            rank-major metadata.

    Returns:
        ``(selected [T, index_topk] token ids (-1 padded, unique per row), token_pages)``.

    Raises:
        ValueError: On a page layout that does not match the indexer, or rank-major metadata
            without a usable ``dp_sharding``.
    """
    heads, dim = int(indexer.index_n_heads), int(indexer.index_head_dim)
    num_pages, _page_size, key_dim = token_pages.shape
    if key_dim != dim:
        raise ValueError(f"token_pages last dim must be index_head_dim={dim}, got {key_dim}.")
    qsl = jnp.asarray(query_start_loc, jnp.int32)
    if qsl.ndim == 1 and block_tables.ndim != 2:
        raise ValueError("block_tables must be [num_seqs, pages_per_seq].")
    total_tokens = int(hidden_states.shape[0])

    q_input = q_resid if q_resid is not None else hidden_states
    query = indexer.wq_b(q_input).reshape(total_tokens, heads, dim).astype(jnp.float32)
    key = indexer.k_norm(indexer.wk(hidden_states)).astype(jnp.float32)
    head_weights = indexer.kernels_proj(hidden_states).astype(jnp.float32) * (heads**-0.5)

    rope_dim, rope_fn = 0, None
    freq = jnp.zeros((1, 2), jnp.float32)
    if frequencies is not None:
        freq = frequencies.value if hasattr(frequencies, "value") else frequencies
        freq = jnp.asarray(freq)
        rope_dim = min(int(indexer.indexer_config.rope_dim), dim, (int(freq.shape[-1]) // 2) * 2)
        rope_dim -= rope_dim % 2
        rope_fn = _token_rope_interleaved if indexer.indexer_rope_interleave else _token_rope_split

    core = functools.partial(
        _token_select_core,
        rope_dim=int(rope_dim),
        rope_fn=rope_fn,
        softmax_scale=float(indexer.softmax_scale),
        select_k=int(indexer.index_topk),
        chunk=max(1, int(key_chunk)),
        precision=indexer.precision,
    )

    if qsl.ndim == 1:
        return core(
            query,
            key,
            head_weights,
            freq,
            token_pages,
            block_tables,
            kv_lens,
            qsl,
            num_seqs,
            rows_c=constrain_rows or _identity,
            pages_c=constrain_pages or _identity,
        )

    mesh, names, dp = _dp_setup(
        dp_sharding, total_tokens=total_tokens, num_rows=int(jnp.size(kv_lens)), num_pages=int(num_pages)
    )

    def per_rank(q, k, w, freq_, tok, tables, lens, qsl_, nseq):
        local_kv, local_tables, local_qsl, local_num = _dp_rank_local(
            lens, tables, qsl_, nseq, axis_names=names, dp_size=dp, local_num_pages=int(tok.shape[0])
        )
        return core(
            q,
            k,
            w,
            freq_,
            tok,
            local_tables,
            local_kv,
            local_qsl,
            local_num,
            rows_c=_identity,
            pages_c=_identity,
        )

    row, rep = PartitionSpec(names), PartitionSpec()
    return jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=(row, row, row, rep, row, rep, rep, rep, rep),
        out_specs=(row, row),
        check_vma=False,
    )(
        query,
        key,
        head_weights,
        freq,
        token_pages,
        jnp.asarray(block_tables, jnp.int32),
        jnp.asarray(kv_lens, jnp.int32),
        qsl,
        jnp.asarray(num_seqs, jnp.int32),
    )


__all__ = ("paged_pool_select", "paged_token_layout", "paged_token_select")
