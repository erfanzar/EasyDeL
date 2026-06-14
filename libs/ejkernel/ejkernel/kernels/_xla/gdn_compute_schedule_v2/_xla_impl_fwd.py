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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grid schedule construction for ragged Gated Delta Rule kernels (v2).

This module holds the raw XLA (pure JAX) implementation that precomputes the
per-grid-iteration work schedule for the ragged Gated Delta Rule v2 Pallas
kernels. A packed token stream contains a block of leading decode tokens
followed by ragged per-request prefill tokens. The Pallas kernel runs a single
grid over a fixed number of steps and must decide, at each step, whether to do
decode or prefill work, which request and token range that work covers, and how
to stitch chunk-recurrent state across request boundaries that are not aligned
to the kernel's sublane width. Computing those decisions inside the kernel is
expensive and control-flow heavy, so this module materializes them ahead of
time into a fixed-shape integer table that the kernel reads from SMEM.

It contains a single public function, :func:`compute_schedule_table_v2`, which
returns the schedule table together with the number of active grid steps.
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
    """Build the per-iteration work schedule table for ragged GDN v2 kernels.

    Given the CSR request layout of a packed token stream (leading decode tokens
    followed by ragged prefill tokens), this returns a fixed-shape integer table
    whose rows describe the work each grid step of the Pallas kernel should run.
    Three kinds of work are encoded per row: decode batches, regular prefill
    chunks, and *transition* blocks. Transition blocks handle the partial,
    sublane-misaligned token spans at the start and/or end of a prefill request
    so that chunk-recurrent state can be correctly carried across request
    boundaries that are not multiples of ``alignment``.

    All shapes derive from ``num_seqs = query_start_loc.shape[0] - 1`` and a
    statically computed ``safe_max_blocks = (max_tokens + chunk_size - 1) //
    chunk_size + num_seqs * 2`` (extra rows reserve room for the up-to-two
    transition blocks per request). The table is laid out per grid step (row)
    with each column carrying one descriptor field; the decode and prefill
    descriptors share rows because at most one of them is active per step (the
    kernel guards on the validity flags).

    Args:
        query_start_loc: CSR-style request offsets into the packed token stream,
            integer array of shape ``[num_requests + 1]``. Entry ``i`` is the
            inclusive start token index of request ``i`` and entry ``i + 1`` its
            exclusive end. Trailing entries beyond ``num_valid_seqs`` may be
            padding and are masked internally.
        decode_tokens: Number of leading decode tokens in the packed stream,
            ``int`` or scalar integer array. Requests whose end is at or below
            this boundary are pure-decode and emit no prefill blocks.
        num_valid_seqs: Number of valid (non-padding) request rows, ``int`` or
            scalar integer array. ``query_start_loc[num_valid_seqs]`` is treated
            as the total valid token count and padding offsets are clamped to it.
        max_tokens: Static upper bound on the total packed token count; used to
            size ``safe_max_blocks``.
        chunk_size: Prefill chunk size in tokens. Regular prefill blocks cover up
            to this many tokens.
        BT: Decode token block size. When ``None`` it defaults to ``chunk_size``.
            ``num_decode_batches = ceil(decode_tokens / BT)`` rows are reserved
            for decode work.
        alignment: Sublane/token-row alignment, default ``8``. Request spans are
            rounded to this granularity to form transition blocks, and per-row
            metadata is emitted for ``alignment`` sublanes.

    Returns:
        A tuple ``(final_table, total_blocks)``:

        * ``final_table``: integer array of shape
          ``[safe_max_blocks, 11 + 3 * alignment]``, one row per grid step.
          Columns are:

          - 0: prefill-valid flag (1 if this step runs prefill work).
          - 1: prefill block token offset (global start index into the stream).
          - 2: prefill request index (the request this prefill block belongs to).
          - 3: prefill block token count (number of valid tokens in the block).
          - 4: decode-valid flag (1 if this step runs decode work).
          - 5: decode batch token offset (global start of the decode batch).
          - 6: decode starting request id of the batch.
          - 7: decode request/token count for the batch.
          - 8: prefill block-is-last flag (block ends at the request end).
          - 9: prefill block-is-first flag (block starts at the request start).
          - 10: transition-block flag (1 for start/end transition blocks).
          - ``11 .. 10 + alignment``: per-sublane request ids for the block's
            aligned token window.
          - next ``alignment`` columns: per-sublane first-token flags (1 where
            the sublane token is the first token of its request).
          - next ``alignment`` columns: per-sublane last-token flags (1 where
            the sublane token is the last token of its request).

          Rows beyond the active range are zero-filled so the table is safe to
          read fully.
        * ``total_blocks``: scalar integer array with the number of active grid
          steps, ``max(total_prefill_blocks, num_decode_batches)``.
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
        prefill_valid_ints,
        block_offset,
        r_for_block,
        block_count,
        decode_valid_mask.astype(jnp.int32),
        decode_offsets,
        decode_req_ids,
        decode_counts,
        block_is_last.astype(jnp.int32),
        block_is_first.astype(jnp.int32),
        is_trans_block.astype(jnp.int32),
    ]

    for i in range(alignment):
        cols.append(t_reqs[:, i])
    for i in range(alignment):
        cols.append(is_first_tok[:, i])
    for i in range(alignment):
        cols.append(is_last_tok[:, i])

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks
