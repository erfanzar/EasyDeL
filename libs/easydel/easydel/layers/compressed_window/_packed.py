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
    "grid_to_tokens",
    "occupied_columns",
    "packed_block_size",
    "run_column_blocks",
    "tokens_to_grid",
)

# Token columns a packed prefill advances per loop step (callers also cap it by the sliding window so one block never
# writes the same ring slot twice).
PACKED_PREFILL_BLOCK = 64

_Carry = tp.TypeVar("_Carry")


def packed_block_size(total_tokens: int, num_slots: int, window: int, preferred: int = PACKED_PREFILL_BLOCK) -> int:
    """Static columns per loop step: ``1`` for pure-decode buckets, else ``min(preferred, total, window)``.

    A bucket with ``total_tokens <= num_slots`` can hold at most one token per slot in the common decode case, so the
    single-column path (no wasted block cells) is kept there.
    """
    if total_tokens <= num_slots:
        return 1
    return max(1, min(int(preferred), int(total_tokens), int(window)))


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
