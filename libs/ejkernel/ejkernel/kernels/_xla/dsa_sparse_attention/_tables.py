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

"""Block tables of the pool-bitmap block-sparse DSA attention (plain XLA, no ``O(S^2)`` tensor).

The selection arrives as the indexer's per-query POOL mask ``pool_mask [B, S, P]`` (pool ``p`` = keys
``p * kpool .. p * kpool + kpool - 1``, anchored at key 0, causal and same-document by the indexer's construction)
plus up to 8 always-selected TAIL keys per query ``tails [B, S, T]`` (``-1`` = none). :func:`build_tables` turns
them into what the Pallas kernels stream:

* ``bits_t [B, W_pad, S]`` int32 -- bit ``p % 32`` of word ``p // 32`` set iff pool ``p`` is selected; words on the
  sublane axis, queries on lanes, so a kernel reads one ``[1, block_q]`` row per word with a dynamic sublane index
  and never transposes a mask;
* ``tails_t [B, 8, S]`` int32 -- the tail keys, rows ``0..T-1`` used, the rest ``-1``;
* ``count [B, nQB]`` and ``kv_list [B, nQB, nKB]`` int32 -- per q-block how many kv-blocks are ACTIVE (some query of
  the block selected a key in it or holds a tail key in it, and the block is not entirely above the q-block's
  diagonal) and their ids ascending, padded with ``nKB``. Scalar-prefetched into SMEM.

The visit ``mode`` decides which causal pairs the tables list; the kernels only read the tables, so the mode changes
no kernel code:

* ``"sparse"`` -- the active pairs: the work follows the selection's block density;
* ``"dense"`` -- every causal pair: constant work per sequence and bit-identical results (an inactive block's keys
  are all closed, so the accumulators do not move);
* ``"budget:B"`` -- at most ``B`` kv-blocks per q-block, the ``B`` with the most selected keys (ties: the later
  block), the rest DROPPED from the selection: bounded work at a numerics change (dropped pools get a zero pool
  cotangent).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

TAIL_ROWS = 8
WORD = 32
_MODES = ("sparse", "dense", "budget")


def _round_up(x: int, m: int) -> int:
    """Round ``x`` up to a multiple of ``m``."""
    return -(-x // m) * m


def pack_pool_bits(pool_mask: Array) -> Array:
    """Pack a pool mask into 32-bit words.

    Args:
        pool_mask: ``[B, S, P]`` bool selected pools.

    Returns:
        ``[B, S, ceil(P / 32)]`` int32 where bit ``p % 32`` of word ``p // 32`` equals ``pool_mask[..., p]`` (exact).
    """
    batch, seq, num_pools = pool_mask.shape
    words = _round_up(num_pools, WORD) // WORD
    padded = jnp.pad(pool_mask.astype(jnp.uint32), ((0, 0), (0, 0), (0, words * WORD - num_pools)))
    weights = (jnp.uint32(1) << jnp.arange(WORD, dtype=jnp.uint32))[None, None, None, :]
    packed = jnp.sum(padded.reshape(batch, seq, words, WORD) * weights, axis=-1, dtype=jnp.uint32)
    return lax.bitcast_convert_type(packed, jnp.int32)


def parse_sparse_mode(mode: str | None) -> tuple[str, int]:
    """Parse a visit mode.

    Args:
        mode: ``"sparse"``, ``"dense"`` or ``"budget:B"`` with ``B`` a positive int; ``None`` / ``""`` mean sparse.

    Returns:
        ``(kind, budget)``; ``budget`` is 0 unless ``kind == "budget"``.

    Raises:
        ValueError: On an unknown mode or a non-positive budget.
    """
    text = "sparse" if mode is None else str(mode).strip().lower()
    if text == "":
        text = "sparse"
    kind, sep, rest = text.partition(":")
    if kind not in _MODES or (kind == "budget") != bool(sep):
        raise ValueError(f"dsa_sparse_attention mode={mode!r}: expected 'sparse', 'dense' or 'budget:<kv-blocks>'")
    if kind != "budget":
        return kind, 0
    if not rest.isdigit() or int(rest) < 1:
        raise ValueError(f"dsa_sparse_attention mode={mode!r}: the budget must be a positive integer of kv-blocks")
    return kind, int(rest)


def validate_blocks(block_q: int, block_kv: int, kpool: int) -> None:
    """Raise ``ValueError`` unless the tiles fit the kernels' layout rules.

    Args:
        block_q: Query tile (multiple of 128).
        block_kv: Key tile (multiple of 128 and of ``32 * kpool``: one bitmap word covers ``32 * kpool`` keys).
        kpool: Keys per pool (positive).
    """
    if kpool < 1:
        raise ValueError(f"dsa_sparse_attention: kpool must be positive, got {kpool}")
    if block_q < 128 or block_kv < 128 or block_q % 128 or block_kv % 128 or block_kv % (WORD * kpool):
        raise ValueError(
            f"dsa_sparse_attention: block_q={block_q} must be a multiple of 128 and block_kv={block_kv} a multiple of "
            f"128 and of {WORD * kpool} (32 * kpool)"
        )


BACKWARDS = ("fused", "two_pass")
POOL_SUMS = ("reshape", "matmul")


def validate_options(
    *,
    mode: str | None,
    backward: str,
    pool_sum: str,
    stale_tau: float | None,
    fused_groups: int | None,
) -> str:
    """Validate the call options shared by every implementation and return the normalised mode.

    Args:
        mode: Visit mode (see :func:`parse_sparse_mode`).
        backward: ``"fused"`` (single-pass backward) or ``"two_pass"`` (dQ pass + dKV pass).
        pool_sum: Pool-cotangent reduction, ``"reshape"`` or ``"matmul"``.
        stale_tau: ``None`` or a non-negative stale-max threshold in nats.
        fused_groups: ``None`` (automatic), ``1`` or ``2``.

    Returns:
        The lower-cased, stripped mode string.

    Raises:
        ValueError: On any unknown or out-of-range option.
    """
    parse_sparse_mode(mode)
    if backward not in BACKWARDS:
        raise ValueError(f"dsa_sparse_attention: backward={backward!r}, expected one of {BACKWARDS}")
    if pool_sum not in POOL_SUMS:
        raise ValueError(f"dsa_sparse_attention: pool_sum={pool_sum!r}, expected one of {POOL_SUMS}")
    if stale_tau is not None and not float(stale_tau) >= 0.0:
        raise ValueError(f"dsa_sparse_attention: stale_tau must be None or >= 0, got {stale_tau}")
    if fused_groups not in (None, 1, 2):
        raise ValueError(f"dsa_sparse_attention: fused_groups must be None, 1 or 2, got {fused_groups}")
    return "sparse" if mode is None or str(mode).strip() == "" else str(mode).strip().lower()


def build_tables(
    pool_mask: Array,
    tails: Array | None,
    *,
    kpool: int,
    block_q: int,
    block_kv: int,
    mode: str | None = "sparse",
) -> tuple[Array, Array, Array, Array]:
    """Build the kernels' tables from the pool selection and the tail keys (see the module docstring).

    Args:
        pool_mask: ``[B, S, P]`` bool selected pools (``P <= ceil(S / kpool)``).
        tails: ``[B, S, T]`` int32 tail keys, ``-1`` = none, ``T <= 8``; ``None`` = no tails.
        kpool: Keys per pool.
        block_q: Query block (multiple of 128).
        block_kv: Key block (multiple of 128 and of ``32 * kpool``).
        mode: Visit mode (``"sparse"``, ``"dense"`` or ``"budget:B"``).

    Returns:
        ``(bits_t [B, W_pad, S], tails_t [B, 8, S], count [B, nQB], kv_list [B, nQB, nKB])``.

    Raises:
        ValueError: On invalid tiles, mode, pool count or tail count.
    """
    kind, budget = parse_sparse_mode(mode)
    validate_blocks(block_q, block_kv, kpool)
    batch, seq, num_pools = pool_mask.shape
    n_qb = -(-seq // block_q)
    n_kb = -(-seq // block_kv)
    pools_per_kb = block_kv // kpool
    if num_pools > n_kb * pools_per_kb:
        raise ValueError(f"dsa_sparse_attention: {num_pools} pools exceed the {n_kb * pools_per_kb} the keys can hold")

    bits = pack_pool_bits(pool_mask)  # [B, S, W]
    w_pad = _round_up(bits.shape[-1], 8)
    bits = jnp.pad(bits, ((0, 0), (0, 0), (0, w_pad - bits.shape[-1])))
    bits_t = jnp.transpose(bits, (0, 2, 1))  # [B, W_pad, S]

    if tails is None:
        tails = jnp.full((batch, seq, 1), -1, jnp.int32)
    tails = tails.astype(jnp.int32)
    if tails.shape[-1] > TAIL_ROWS:
        raise ValueError(f"dsa_sparse_attention: at most {TAIL_ROWS} tail keys per query, got {tails.shape[-1]}")
    tails_t = jnp.transpose(
        jnp.pad(tails, ((0, 0), (0, 0), (0, TAIL_ROWS - tails.shape[-1])), constant_values=-1), (0, 2, 1)
    )

    # per-query kv-block hits: any selected pool inside the block, or a tail key inside it
    pm = jnp.pad(pool_mask, ((0, 0), (0, 0), (0, n_kb * pools_per_kb - num_pools)))
    hit = jnp.any(pm.reshape(batch, seq, n_kb, pools_per_kb), axis=-1)  # [B, S, nKB]
    tail_kb = jnp.where(tails >= 0, tails // block_kv, -1)  # [B, S, T]
    hit = hit | jnp.any(tail_kb[..., None] == jnp.arange(n_kb)[None, None, None, :], axis=2)
    pad_q = n_qb * block_q - seq
    if pad_q:
        hit = jnp.pad(hit, ((0, 0), (0, pad_q), (0, 0)))
    active = jnp.any(hit.reshape(batch, n_qb, block_q, n_kb), axis=2)  # [B, nQB, nKB]
    qb_last = (jnp.arange(n_qb, dtype=jnp.int32)[:, None] + 1) * block_q - 1
    kb_first = jnp.arange(n_kb, dtype=jnp.int32)[None, :] * block_kv
    causal = (kb_first <= qb_last)[None]  # [1, nQB, nKB]
    active = active & causal
    if kind == "dense":
        active = jnp.broadcast_to(causal, active.shape)
    elif kind == "budget":
        active = _budget_active(active, pm, tail_kb, n_qb, n_kb, block_q, pools_per_kb, kpool, pad_q, budget)
    count = jnp.sum(active, axis=-1, dtype=jnp.int32)
    kv_list = jnp.sort(jnp.where(active, jnp.arange(n_kb, dtype=jnp.int32)[None, None, :], n_kb), axis=-1)
    return bits_t, tails_t, count, kv_list.astype(jnp.int32)


def pair_weights(
    pm: Array, tail_kb: Array, n_qb: int, n_kb: int, block_q: int, pools_per_kb: int, kpool: int, pad_q: int
) -> Array:
    """Selected-key weight of every (q-block, kv-block) pair.

    Args:
        pm: ``[B, S, nKB * pools_per_kb]`` bool pool mask padded to whole kv-blocks.
        tail_kb: ``[B, S, T]`` kv-block of every tail key (``-1`` = none).
        n_qb: Number of q-blocks.
        n_kb: Number of kv-blocks.
        block_q: Query block.
        pools_per_kb: Pools per kv-block.
        kpool: Keys per pool.
        pad_q: Padding queries appended to reach ``n_qb * block_q``.

    Returns:
        ``[B, nQB, nKB]`` int32: ``kpool`` per selected pool plus one per tail key of the q-block's queries (a tail
        repeating a pool key counts twice; a ranking weight, not the open set).
    """
    batch, seq = pm.shape[:2]
    pools = jnp.sum(pm.reshape(batch, seq, n_kb, pools_per_kb), axis=-1, dtype=jnp.int32) * kpool
    tails_hit = jnp.sum(
        tail_kb[..., None] == jnp.arange(n_kb, dtype=jnp.int32)[None, None, None, :], axis=2, dtype=jnp.int32
    )
    per_query = pools + tails_hit
    if pad_q:
        per_query = jnp.pad(per_query, ((0, 0), (0, pad_q), (0, 0)))
    return jnp.sum(per_query.reshape(batch, n_qb, block_q, n_kb), axis=2, dtype=jnp.int32)


def _budget_active(active, pm, tail_kb, n_qb, n_kb, block_q, pools_per_kb, kpool, pad_q, budget: int):
    """Keep, per q-block, the ``budget`` active kv-blocks with the most selected keys (ties -> the later block)."""
    if budget >= n_kb:
        return active
    weight = pair_weights(pm, tail_kb, n_qb, n_kb, block_q, pools_per_kb, kpool, pad_q)
    kb_ids = jnp.arange(n_kb, dtype=jnp.int32)
    key = jnp.where(active, weight * n_kb + kb_ids[None, None, :], -1)  # rank by weight, then by block id
    vals, idx = lax.top_k(key, budget)  # [B, nQB, budget]
    keep = (kb_ids[None, None, None, :] == idx[..., None]) & (vals >= 0)[..., None]  # [B, nQB, budget, nKB]
    return jnp.any(keep, axis=2)


def build_q_tables(count: Array, kv_list: Array) -> tuple[Array, Array]:
    """Transpose the active lists for the kv-block-major dKV pass of the two-pass backward.

    Args:
        count: ``[B, nQB]`` active kv-block counts (:func:`build_tables`).
        kv_list: ``[B, nQB, nKB]`` ascending active kv-block ids, padded with ``nKB``.

    Returns:
        ``(q_count [B, nKB], q_list [B, nKB, nQB])``: per kv-block how many q-blocks visit it and their ids ascending,
        padded with ``nQB``.
    """
    n_qb, n_kb = kv_list.shape[1], kv_list.shape[2]
    ids = jnp.arange(n_kb, dtype=jnp.int32)
    valid = ids[None, None, :] < count[..., None]  # [B, nQB, i]: list entry i is a real block
    hit = (kv_list[:, :, None, :] == ids[None, None, :, None]) & valid[:, :, None, :]  # [B, nQB, kb, i]
    incidence = jnp.transpose(jnp.any(hit, axis=-1), (0, 2, 1))  # [B, nKB, nQB]
    q_count = jnp.sum(incidence, axis=-1, dtype=jnp.int32)
    q_list = jnp.sort(jnp.where(incidence, jnp.arange(n_qb, dtype=jnp.int32)[None, None, :], n_qb), axis=-1)
    return q_count, q_list.astype(jnp.int32)


def dma_block_indices(count: Array, lists: Array, n_blocks: int) -> Array:
    """The block every grid step of a kernel DMAs, vectorised over the whole grid.

    Step ``i`` past a row's ``count`` re-targets the last active block; a row with ``count == 0`` (a fully padded
    row, the padded tail of a packed row) holds only the pad sentinel, so the entry is clamped into range. Without
    the clamp the kernels DMA block ``n_blocks``, one past the buffer (a ``BoundsCheck`` fault on real TPUs; interpret
    mode clamps silently).

    Args:
        count: ``[B, nX]`` active counts.
        lists: ``[B, nX, L]`` ascending ids padded with ``n_blocks``.
        n_blocks: Number of blocks along the streamed axis.

    Returns:
        ``[B, nX, L]`` int32 block indices, always in ``[0, n_blocks)``.
    """
    length = lists.shape[-1]
    pos = jnp.minimum(jnp.arange(length, dtype=jnp.int32)[None, None, :], jnp.maximum(count[..., None] - 1, 0))
    entry = jnp.take_along_axis(lists, jnp.broadcast_to(pos, lists.shape), axis=-1)
    return jnp.clip(entry, 0, n_blocks - 1)


def expand_open_mask(pool_mask: Array, tails: Array | None, *, kpool: int, seq_len: int, causal: bool = True) -> Array:
    """Per-key open set: the keys of the selected pools plus the tail keys, optionally cut at the diagonal.

    Set semantics: a tail key repeating a pool key opens it once.

    Args:
        pool_mask: ``[B, S, P]`` bool selected pools.
        tails: ``[B, S, T]`` int32 tail keys (``-1`` = none) or ``None``.
        kpool: Keys per pool.
        seq_len: Key-axis length.
        causal: Also close keys past each query's position.

    Returns:
        ``[B, S, seq_len]`` bool.
    """
    batch, seq, num_pools = pool_mask.shape
    key_pool = jnp.arange(seq_len, dtype=jnp.int32) // kpool
    pad_pools = max(0, -(-seq_len // kpool) - num_pools)
    open_ = jnp.take(jnp.pad(pool_mask, ((0, 0), (0, 0), (0, pad_pools))), key_pool, axis=-1)
    if tails is not None:
        valid = tails >= 0
        safe = jnp.where(valid, tails, 0)
        b_ix = jnp.arange(batch)[:, None, None]
        s_ix = jnp.arange(seq)[None, :, None]
        tail_open = jnp.zeros((batch, seq, seq_len), jnp.bool_).at[b_ix, s_ix, safe].set(True)
        has0 = jnp.any(valid & (tails == 0), axis=-1)  # invalid slots parked on key 0 must not open it
        tail_open = tail_open & ((jnp.arange(seq_len) != 0)[None, None, :] | has0[..., None])
        open_ = open_ | tail_open
    if causal:
        open_ = open_ & (jnp.arange(seq_len)[None, :] <= jnp.arange(seq)[:, None])[None]
    return open_


def visited_open_mask(open_mask: Array, count: Array, kv_list: Array, block_q: int, block_kv: int) -> Array:
    """Cut ``open_mask [B, S, kv]`` to the (q-block, kv-block) pairs the tables visit (``budget:B`` semantics).

    Args:
        open_mask: ``[B, S, kv]`` bool open set.
        count: ``[B, nQB]`` active counts.
        kv_list: ``[B, nQB, nKB]`` active lists.
        block_q: Query block.
        block_kv: Key block.

    Returns:
        ``[B, S, kv]`` bool; equal to ``open_mask`` under ``sparse`` / ``dense``.
    """
    _, seq, kv_len = open_mask.shape
    n_qb, n_kb = kv_list.shape[1], kv_list.shape[2]
    ids = jnp.arange(n_kb, dtype=jnp.int32)
    valid = ids[None, None, :] < count[..., None]
    visited = jnp.any((kv_list[:, :, None, :] == ids[None, None, :, None]) & valid[:, :, None, :], axis=-1)
    q_of = jnp.minimum(jnp.arange(seq) // block_q, n_qb - 1)
    k_of = jnp.minimum(jnp.arange(kv_len) // block_kv, n_kb - 1)
    return open_mask & visited[:, q_of][:, :, k_of]


def expand_pool_values(values: Array, kpool: int, seq_len: int) -> Array:
    """Broadcast per-pool values ``[B, S, P]`` to keys ``[B, S, seq_len]`` (key ``k`` reads pool ``k // kpool``).

    The VJP of the gather is the pool-sum of the per-key cotangent.
    """
    num_pools = values.shape[-1]
    pad_pools = max(0, -(-seq_len // kpool) - num_pools)
    key_pool = jnp.arange(seq_len, dtype=jnp.int32) // kpool
    return jnp.take(jnp.pad(values, ((0, 0), (0, 0), (0, pad_pools))), key_pool, axis=-1)


__all__ = (
    "BACKWARDS",
    "POOL_SUMS",
    "TAIL_ROWS",
    "WORD",
    "build_q_tables",
    "build_tables",
    "dma_block_indices",
    "expand_open_mask",
    "expand_pool_values",
    "pack_pool_bits",
    "pair_weights",
    "parse_sparse_mode",
    "validate_blocks",
    "validate_options",
    "visited_open_mask",
)
