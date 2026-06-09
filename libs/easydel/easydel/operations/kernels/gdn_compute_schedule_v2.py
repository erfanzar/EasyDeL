# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grid schedule construction for ragged Gated Delta Rule kernels (v2).

Builds the per-grid-iteration metadata table consumed by the ragged GDN
Pallas prefill/decode kernels under continuous batching. Each grid
iteration is classified as either a decode batch (moving backwards from
the prefill/decode boundary in ``BT``-token strides) or a prefill chunk
(moving forwards in ``chunk_size`` strides), with transition rows used to
handle sequences whose start or end is not sublane-aligned.

See :func:`compute_schedule_table_v2` for the column layout of the
resulting table.
"""

import jax
import jax.numpy as jnp


def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Build the per-iteration work schedule table for the ragged GDN kernels.

    Each grid iteration represents either a *decode* batch (moving backwards
    from the decode/prefill boundary in ``BT``-token strides) or a *prefill*
    chunk (moving forwards from the boundary in ``chunk_size``-token
    strides). When a sequence boundary lands inside a sublane the schedule
    inserts an explicit transition block at each end so that the prefill
    kernel can fall back to token-by-token arithmetic for those rows; the
    rest of the prefill work is processed chunk-wise on sublane-aligned
    offsets.

    Args:
        query_start_loc: CSR-style cumulative token offsets per request,
            shape ``(num_requests + 1,)`` (``int32``).
        decode_tokens: Number of decode tokens at the start of the flat
            token stream (i.e. the size of the decode region preceding any
            prefill work). Scalar ``int`` or scalar JAX array.
        num_valid_seqs: Number of valid (non-padding) sequences in the
            batch. Scalar ``int`` or scalar JAX array.
        max_tokens: Static upper bound on the total number of tokens used
            to size the schedule table.
        chunk_size: Number of tokens processed per prefill block.
        BT: Number of tokens processed per decode block. Defaults to
            ``chunk_size`` when ``None``.
        alignment: Sublane alignment that block offsets must respect
            (typically 8 for TPU sublanes).

    Returns:
        tuple: ``(final_table, total_blocks)`` where ``final_table`` is an
        ``int32`` array of shape ``(safe_max_blocks, 11 + 3 * alignment)``
        carrying per-grid-iteration metadata (columns described below) and
        ``total_blocks`` is the ``int32`` count of valid grid iterations
        (``max(total_prefill_blocks, num_decode_batches)``).

    Note:
        Column layout of ``final_table``:

        * 0: ``prefill_valid_ints`` — 1 when this row carries prefill work.
        * 1: ``block_offset`` — token offset of the prefill block start.
        * 2: ``r_for_block`` — request index of the prefill block.
        * 3: ``block_count`` — valid tokens in the prefill block.
        * 4: ``decode_valid`` — 1 when this row carries decode work.
        * 5: ``decode_offsets`` — start token of the decode batch.
        * 6: ``decode_req_ids`` — starting request id in the decode batch.
        * 7: ``decode_counts`` — number of decode requests in the batch.
        * 8: ``block_is_last`` — 1 when this prefill block ends the request.
        * 9: ``block_is_first`` — 1 when this prefill block starts it.
        * 10: ``is_trans_block`` — 1 when the row is a sublane transition.
        * ``11 .. 10 + alignment``: per-sublane-token request ids.
        * next ``alignment``: per-sublane ``is_first_tok`` flags.
        * next ``alignment``: per-sublane ``is_last_tok`` flags.

    TODO:
        Compact the table — block offsets/counts can be reconstructed from
        block index plus sequence start/end, transition flags can be packed
        into fewer bits, and sublane metadata can be reduced to boundaries
        only.
    """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (max_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)

    r_idx = jnp.arange(num_seqs)
    is_last_seq = r_idx == num_seqs - 1
    seq_start = query_start_loc[:-1]
    seq_end = query_start_loc[1:]
    num_tokens = query_start_loc[num_valid_seqs]

    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )

    is_decode_boundary = prev_seq_end == decode_tokens
    is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

    next_aligned_start = (seq_end // alignment) * alignment
    needs_transition = (seq_end % alignment != 0) & (~is_last_seq) & (~is_swallowed)

    is_decode_boundary = prev_seq_end == decode_tokens

    needs_start_transition = (prev_seq_end % alignment != 0) & (~is_swallowed) & is_decode_boundary

    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)

    num_regular_blocks = (effective_end - effective_start + chunk_size - 1) // chunk_size
    total_blocks_per_seq = (
        num_regular_blocks + needs_transition.astype(jnp.int32) + needs_start_transition.astype(jnp.int32)
    )
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

    is_pure_decode = seq_end <= decode_tokens
    total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)

    base_idx = jnp.cumsum(total_blocks_per_seq) - total_blocks_per_seq
    total_prefill_blocks = jnp.sum(total_blocks_per_seq)

    b_idx = jnp.arange(safe_max_blocks)
    prefill_valid_mask = b_idx < total_prefill_blocks

    r_for_block = jnp.sum(b_idx[:, None] >= base_idx[None, :], axis=-1) - 1
    r_for_block = jnp.minimum(jnp.maximum(r_for_block, 0), num_seqs - 1)

    local_b = b_idx - base_idx[r_for_block]

    start_trans_offset = (seq_start[r_for_block] // alignment) * alignment

    is_start_trans = needs_start_transition[r_for_block] & (local_b == 0)

    adj_local_b = jnp.where(needs_start_transition[r_for_block], local_b - 1, local_b)

    is_end_trans = needs_transition[r_for_block] & (adj_local_b == num_regular_blocks[r_for_block])

    reg_offset = effective_start[r_for_block] + adj_local_b * chunk_size
    reg_count = jnp.minimum(chunk_size, effective_end[r_for_block] - reg_offset)

    trans_offset = next_aligned_start[r_for_block]

    block_offset = jnp.where(
        is_start_trans,
        start_trans_offset,
        jnp.where(is_end_trans, trans_offset, reg_offset),
    )

    block_count = jnp.where(
        is_start_trans,
        effective_start[r_for_block] - seq_start[r_for_block],
        jnp.where(is_end_trans, alignment, reg_count),
    )

    is_trans_block = is_start_trans | is_end_trans

    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    fixed_query_start_loc = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)
    glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

    valid_mask = glob_idxs < num_tokens
    t_reqs = jnp.sum(glob_idxs[:, :, None] >= fixed_query_start_loc[None, None, :], axis=-1) - 1
    last_valid_seq = jnp.max(jnp.where(total_blocks_per_seq > 0, jnp.arange(num_seqs), -1))
    t_reqs = jnp.where(valid_mask, t_reqs, last_valid_seq)
    t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

    is_first_tok = (glob_idxs == query_start_loc[t_reqs]).astype(jnp.int32)
    is_last_tok = (glob_idxs == query_start_loc[t_reqs + 1] - 1).astype(jnp.int32)

    decode_valid_mask = b_idx < num_decode_batches
    decode_batch_idx = jnp.where(decode_valid_mask, (num_decode_batches - 1) - b_idx, 0)
    decode_offsets = decode_batch_idx * BT
    decode_req_ids = decode_batch_idx * BT
    decode_counts = jnp.where(decode_valid_mask, jnp.minimum(BT, decode_tokens - decode_offsets), 0)

    prefill_valid_ints = prefill_valid_mask.astype(jnp.int32)
    block_offset = jnp.where(prefill_valid_mask, block_offset, 0)
    r_for_block = jnp.where(prefill_valid_mask, r_for_block, 0)
    block_count = jnp.where(prefill_valid_mask, block_count, 0)
    block_is_first = block_offset <= seq_start[r_for_block]
    block_is_last = (block_offset + block_count) >= seq_end[r_for_block]
    block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
    block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
    is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
    t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
    is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
    is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

    cols = [
        prefill_valid_ints,  # 0
        block_offset,  # 1
        r_for_block,  # 2
        block_count,  # 3
        decode_valid_mask.astype(jnp.int32),  # 4
        decode_offsets,  # 5
        decode_req_ids,  # 6
        decode_counts,  # 7
        block_is_last.astype(jnp.int32),  # 8
        block_is_first.astype(jnp.int32),  # 9
        is_trans_block.astype(jnp.int32),  # 10
    ]

    for i in range(alignment):
        cols.append(t_reqs[:, i])  # e.g., 11-18 if alignment=8
    for i in range(alignment):
        cols.append(is_first_tok[:, i])  # e.g., 19-26
    for i in range(alignment):
        cols.append(is_last_tok[:, i])  # e.g., 27-34

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks
