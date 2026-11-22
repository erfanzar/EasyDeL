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

"""Packed-stream driver for per-slot compressed-window serving (eSurge).

The engine hands a layer a flat ``[1, T]`` token stream that concatenates every scheduled request's new tokens. These
helpers scatter it into a dense ``[num_slots, width]`` grid (one row per cache slot, columns are the request's new
tokens in order), advance every slot through the occupied columns in blocks of ``block`` columns, and gather the
result back into packed order. Pure decode buckets (at most one token per slot) use ``block == 1``; prefill and mixed
buckets advance whole blocks per loop step, so a prompt costs ``ceil(len / block)`` loop steps instead of ``len``.
"""

from __future__ import annotations

import typing as tp

import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Int

__all__ = (
    "PACKED_PREFILL_BLOCK",
    "PackedSlotMeta",
    "build_packed_slot_meta",
    "grid_to_tokens",
    "occupied_columns",
    "packed_block_size",
    "packed_row_of_token",
    "run_column_blocks",
    "tokens_to_grid",
)

# Token columns a packed prefill advances per loop step (callers also cap it by the sliding window so one block never
# writes the same ring slot twice).
PACKED_PREFILL_BLOCK = 64

_Carry = tp.TypeVar("_Carry")


def packed_block_size(
    total_tokens: int,
    num_slots: int,
    window: int,
    preferred: int = PACKED_PREFILL_BLOCK,
    *,
    cap_by_slot_share: bool = False,
) -> int:
    """Static columns per loop step: ``1`` for pure-decode buckets, else ``min(preferred, total, window)``.

    A bucket with ``total_tokens <= num_slots`` can hold at most one token per slot in the common decode case, so the
    single-column path (no wasted block cells) is kept there.

    With ``cap_by_slot_share`` the block is also capped at ``ceil(total_tokens / num_slots)``. Block work (per-cell
    selected-entry gathers, attention over every cell) scales with ``num_slots * block`` per loop step while the loop
    only visits occupied columns, so a block no wider than one slot's share of the bucket never adds work: a
    speculative verify window (``1 + k`` tokens on every slot) then runs one narrow step instead of a bucket-wide one.
    Large prefill buckets keep ``preferred``.
    """
    if total_tokens <= num_slots:
        return 1
    block = min(int(preferred), int(total_tokens), int(window))
    if cap_by_slot_share:
        block = min(block, -(-int(total_tokens) // max(1, int(num_slots))))
    return max(1, block)


def tokens_to_grid(
    values: Array,
    slot_of_token: Int[Array, " tokens"],
    col_of_token: Int[Array, " tokens"],
    num_slots: int,
    width: int,
) -> Array:
    """Scatter per-token values ``[T, ...]`` into a zero ``[num_slots, width, ...]`` grid (padding tokens dropped)."""
    grid = jnp.zeros((num_slots, width, *values.shape[1:]), dtype=values.dtype)
    return grid.at[slot_of_token, col_of_token].set(values, mode="drop")


def occupied_columns(valid_grid: Bool[Array, "slots width"]) -> Array:
    """Number of leading grid columns that carry at least one real token (int32 scalar)."""
    width = valid_grid.shape[1]
    col = jnp.arange(width, dtype=jnp.int32)
    return jnp.max(jnp.where(jnp.any(valid_grid, axis=0), col + 1, 0))


def run_column_blocks(
    step: tp.Callable[[Array, _Carry], _Carry],
    init: _Carry,
    *,
    num_used: Array,
    block: int,
) -> _Carry:
    """``while_loop`` calling ``step(start_column, carry)`` for ``start = 0, block, 2 * block, ...`` below ``num_used``.

    Only occupied columns are visited, so a chunked step that fills the leading columns does not pay for the padded
    bucket width. Shapes stay static (one compile per bucket).
    """

    def cond(state):
        return state[0] * block < num_used

    def body(state):
        index, carry = state
        return index + 1, step(index * block, carry)

    return jax.lax.while_loop(cond, body, (jnp.int32(0), init))[1]


def grid_to_tokens(
    columns: Array,
    slot_of_token: Int[Array, " tokens"],
    col_of_token: Int[Array, " tokens"],
    num_slots: int,
) -> Array:
    """Gather a column-major ``[width, num_slots, ...]`` result back into packed token order ``[T, ...]``.

    Padding tokens (``slot_of_token == num_slots``) read slot ``num_slots - 1``: garbage that downstream never reads.
    """
    return columns[col_of_token, jnp.minimum(slot_of_token, num_slots - 1)]


def packed_row_of_token(
    query_start_loc: Array,
    token_index: Int[Array, " tokens"],
    *,
    max_row: int | Array | None = None,
) -> Int[Array, " tokens"]:
    """Request row of each packed token: ``clip(searchsorted(qsl, t, "right") - 1, 0, max_row)``.

    Args:
        query_start_loc: ``[R + 1]`` cumulative scheduled-token offsets (padded with the total beyond the live rows).
        token_index: Packed token indices ``[T]``.
        max_row: Upper clip bound; defaults to ``R - 1`` (the last row). Padding tokens past the live rows clip to it,
            so callers mask them with their own validity test.

    Returns:
        ``[T]`` int32 row ids.
    """
    qsl = query_start_loc.astype(jnp.int32).reshape(-1)
    upper = qsl.shape[0] - 2 if max_row is None else max_row
    rows = jnp.searchsorted(qsl, token_index, side="right").astype(jnp.int32) - 1
    return jnp.clip(rows, 0, upper)


class PackedSlotMeta(tp.NamedTuple):
    """Token-to-slot mapping for eSurge packed multi-request serving.

    The engine's flat ``[1, T]`` token stream concatenates each scheduled request's new tokens. This mapping scatters
    those tokens into a dense ``[num_slots, T]`` grid (one row per cache slot at its own stream position) that
    compressed-window attention layers advance column by column (see :func:`run_column_blocks`).

    Attributes:
        slot_of_token: ``[T]`` int32 cache-slot row per packed token; padded tokens carry ``num_slots`` (dropped on
            scatter).
        col_of_token: ``[T]`` int32 grid column per packed token (its index within its request's scheduled chunk).
        valid_grid: ``[num_slots, T]`` bool; ``True`` where a real token occupies the cell.
        positions_grid: ``[num_slots, T]`` int32 absolute stream positions (garbage at invalid cells).
    """

    slot_of_token: Array
    col_of_token: Array
    valid_grid: Array
    positions_grid: Array


def build_packed_slot_meta(
    query_start_loc: Array,
    num_seqs: Array,
    recurrent_state_indices: Array | None,
    position_ids: Array,
    num_slots: int,
) -> PackedSlotMeta:
    """Derive the packed token-to-slot grid mapping from ragged metadata.

    Args:
        query_start_loc: ``[R + 1]`` cumulative scheduled-token offsets per request row (padded with the total beyond
            the live rows).
        num_seqs: ``[1]`` int32 live request count.
        recurrent_state_indices: Optional physical cache-slot id per request row (``[R]`` or ``[1, R]``); ``None``
            falls back to row identity.
        position_ids: Packed absolute positions ``[1, T]``.
        num_slots: Static number of cache slots (state rows).

    Returns:
        The scatter/gather mapping for this step.
    """
    qsl = query_start_loc.astype(jnp.int32).reshape(-1)
    num_rows = qsl.shape[0] - 1
    total_tokens = position_ids.shape[-1]
    token_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    row_of_token = packed_row_of_token(qsl, token_idx)
    live_rows = jnp.clip(num_seqs.reshape(-1)[0].astype(jnp.int32), 0, num_rows)
    total_real_tokens = jnp.take(qsl, live_rows)
    valid_token = token_idx < total_real_tokens
    if recurrent_state_indices is None:
        slots_by_row = jnp.arange(num_rows, dtype=jnp.int32)
    else:
        slots_by_row = recurrent_state_indices.astype(jnp.int32).reshape(-1)[:num_rows]
    slot_of_token = jnp.take(slots_by_row, row_of_token, mode="clip")
    slot_of_token = jnp.where(valid_token, slot_of_token, num_slots)
    col_of_token = token_idx - jnp.take(qsl, row_of_token)

    valid_grid = (
        jnp.zeros((num_slots, total_tokens), dtype=jnp.bool_)
        .at[slot_of_token, col_of_token]
        .set(valid_token, mode="drop")
    )
    positions_grid = (
        jnp.zeros((num_slots, total_tokens), dtype=jnp.int32)
        .at[slot_of_token, col_of_token]
        .set(position_ids.astype(jnp.int32).reshape(-1), mode="drop")
    )
    return PackedSlotMeta(
        slot_of_token=slot_of_token,
        col_of_token=col_of_token,
        valid_grid=valid_grid,
        positions_grid=positions_grid,
    )
