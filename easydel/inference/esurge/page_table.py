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

"""Page table management for KV-cache allocation.

Provides efficient page-based memory management for attention KV-cache.
Uses NumPy arrays for CPU-based page tracking and slot mapping.

Classes:
    PageTable: Manages page allocation and slot mapping for sequences

Functions:
    cdiv: Ceiling division helper

Example:
    >>> table = PageTable(
    ...     page_size=16,
    ...     max_num_reqs=32,
    ...     max_num_pages_per_req=128,
    ...     max_num_batched_tokens=2048
    ... )
    >>> table.append_row([0, 1, 2], row_idx=0)
    >>> slots = table.get_slot_mapping([0, 1], [5, 10])
"""

import jax
import numpy as np
from jax import numpy as jnp

from easydel.utils.compiling_utils import ejit
from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


def cdiv(a: int, b: int) -> int:
    """Ceiling division.

    Computes ceil(a/b) using integer arithmetic.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Ceiling of a/b.

    Example:
        >>> cdiv(5, 2)  # 3
        >>> cdiv(4, 2)  # 2
    """
    return (a + b - 1) // b


SLOT_MAPPING_PADDING_VAL = 0
PAGE_TABLE_PADDING_VAL = 0


class PageTable:
    """Manages page allocation and slot mapping for KV-cache.

    Implements a page table structure for efficient memory management
    of attention caches. Each sequence gets allocated pages which are
    mapped to physical cache slots.

    Attributes:
        page_size: Number of tokens per page.
        max_num_reqs: Maximum number of concurrent requests.
        max_num_pages_per_req: Maximum pages per request.
        max_num_batched_tokens: Maximum tokens in a batch.
        page_table: 2D array mapping request to page IDs.
        num_pages_per_row: Number of pages allocated per request.
        slot_mapping: Maps token positions to cache slots.

    Example:
        >>> table = PageTable(page_size=16, max_num_reqs=32,
        ...                   max_num_pages_per_req=128,
        ...                   max_num_batched_tokens=2048)
        >>> table.set_row([10, 11, 12], row_idx=0)
        >>> slots = table.get_slot_mapping([0], [48])
    """

    def __init__(
        self,
        page_size: int,
        max_num_reqs: int,
        max_num_pages_per_req: int,
        max_num_batched_tokens: int,
    ):
        """Initialize PageTable.

        Args:
            page_size: Number of tokens per page.
            max_num_reqs: Maximum concurrent requests.
            max_num_pages_per_req: Maximum pages per request.
            max_num_batched_tokens: Maximum tokens in batch.
        """
        self.page_size = page_size
        self.max_num_reqs = max_num_reqs
        self.max_num_pages_per_req = max_num_pages_per_req
        self.max_num_batched_tokens = max_num_batched_tokens

        self.page_table = jnp.full(
            (self.max_num_reqs, self.max_num_pages_per_req),
            fill_value=PAGE_TABLE_PADDING_VAL,
            dtype=jnp.int32,
        )
        self.num_pages_per_row = jnp.zeros(self.max_num_reqs, dtype=jnp.int32)
        self.slot_mapping = jnp.full(
            (self.max_num_batched_tokens,),
            fill_value=SLOT_MAPPING_PADDING_VAL,
            dtype=jnp.int32,
        )

    def append_row(self, page_ids: list[int], row_idx: int) -> None:
        """Append page IDs to a single row."""
        if not page_ids:
            return
        start = int(self.num_pages_per_row[row_idx])
        remain = self.max_num_pages_per_req - start
        if remain <= 0:
            return
        to_write = page_ids[:remain]
        if not to_write:
            return
        arr = jnp.asarray(to_write, dtype=jnp.int32)
        end = start + arr.shape[0]
        self.page_table = self.page_table.at[row_idx, start:end].set(arr)
        self.num_pages_per_row = self.num_pages_per_row.at[row_idx].set(
            jnp.minimum(self.num_pages_per_row[row_idx] + arr.shape[0], self.max_num_pages_per_req)
        )

    def add_row(self, page_ids: list[int], row_idx: int) -> None:
        """Reset row to zero length, then append page IDs."""
        self.num_pages_per_row = self.num_pages_per_row.at[row_idx].set(0)
        self.append_row(page_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """Move a row (content and length) from src to tgt."""
        num_pages = int(self.num_pages_per_row[src])
        if num_pages > 0:
            self.page_table = self.page_table.at[tgt, :num_pages].set(self.page_table[src, :num_pages])
        self.num_pages_per_row = self.num_pages_per_row.at[tgt].set(num_pages)

    def swap_row(self, src: int, tgt: int) -> None:
        """Swap two rows (content and lengths)."""
        num_src = int(self.num_pages_per_row[src])
        num_tgt = int(self.num_pages_per_row[tgt])

        self.num_pages_per_row = self.num_pages_per_row.at[src].set(num_tgt)
        self.num_pages_per_row = self.num_pages_per_row.at[tgt].set(num_src)

        src_row = self.page_table[src]
        tgt_row = self.page_table[tgt]
        self.page_table = self.page_table.at[src].set(tgt_row)
        self.page_table = self.page_table.at[tgt].set(src_row)

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> None:
        """Compute flat slot indices for (req_index, position) pairs."""
        page_table_indices = req_indices * self.max_num_pages_per_req + positions // self.page_size
        page_table_flat = self.page_table.reshape((-1,))
        page_numbers = page_table_flat[page_table_indices]
        page_offsets = positions % self.page_size
        slot_values = page_numbers * self.page_size + page_offsets
        num_tokens = req_indices.shape[0]
        self.slot_mapping = self.slot_mapping.at[:num_tokens].set(slot_values)

    def clear(self) -> None:
        """Clear all internal buffers."""
        self.page_table = jnp.full_like(self.page_table, PAGE_TABLE_PADDING_VAL)
        self.num_pages_per_row = jnp.zeros_like(self.num_pages_per_row)
        self.slot_mapping = jnp.full_like(self.slot_mapping, SLOT_MAPPING_PADDING_VAL)

    def get_array(self) -> jax.Array:
        """Returns the dense page table array [max_num_reqs, max_num_pages_per_req]."""
        return self.page_table

    @staticmethod
    @ejit(donate_argnums=(0, 1))
    def _append_rows_batch_jit(
        page_table: jax.Array,
        num_pages_per_row: jax.Array,
        req_indices: jax.Array,  # [B]
        new_pages_padded: jax.Array,  # [B, M]
        lengths: jax.Array,  # [B]
        max_cols: jax.Array,  # scalar int32
    ) -> tuple[jax.Array, jax.Array]:
        """
        Fused append of variable-length page lists to multiple rows, using masked scatter-add.

        For each b in [0..B):
          - writes new_pages_padded[b, :lengths[b]] into row req_indices[b] starting at col starts[b]
          - increments num_pages_per_row[req_indices[b]] by lengths[b], clipped to capacity

        No boolean indexing (avoids NonConcreteBooleanIndex errors).
        """
        B, M = new_pages_padded.shape
        starts = num_pages_per_row[req_indices]  # [B]
        offs = jnp.arange(M, dtype=jnp.int32)[None, :].repeat(B, 0)  # [B, M]

        valid_write = offs < lengths[:, None]  # [B, M]
        cols = starts[:, None] + offs  # [B, M]
        in_bounds = cols < max_cols  # [B, M]
        valid = valid_write & in_bounds  # [B, M]

        # Broadcast rows and clip columns to safe range
        rows = jnp.broadcast_to(req_indices[:, None], (B, M))  # [B, M]
        safe_cols = jnp.clip(cols, 0, max_cols - 1)  # [B, M]

        # Flatten everything to 1D; we'll scatter to all positions, but compute zero deltas where invalid
        rows_f = rows.reshape(-1)  # [B*M]
        cols_f = safe_cols.reshape(-1)  # [B*M]
        vals_f = new_pages_padded.reshape(-1)  # [B*M]
        valid_f = valid.reshape(-1)  # [B*M]

        # Current values at targets (always valid indices)
        current_f = page_table[rows_f, cols_f]

        # Desired values: for invalid entries, keep current so delta=0
        desired_f = jnp.where(valid_f, vals_f, current_f)
        delta_f = desired_f - current_f

        # Scatter-add the deltas (set = add(val - current))
        page_table = page_table.at[(rows_f, cols_f)].add(delta_f)

        # Update lengths (clip to capacity)
        cap = max_cols - starts
        add_len = jnp.minimum(lengths, cap)  # [B]
        num_pages_per_row = num_pages_per_row.at[req_indices].add(add_len)

        return page_table, num_pages_per_row

    def append_rows_batch(self, page_ids_list: list[list[int]], req_indices_list: list[int]) -> None:
        """
        Batched append for one KV group.

        Args:
            page_ids_list: list of variable-length lists (pages) per request
            req_indices_list: list of target row indices (same length)
        """
        if not page_ids_list:
            return

        lengths_py = [len(x) for x in page_ids_list]
        max_len = max(lengths_py)
        if max_len == 0:
            return

        # Pad to next power-of-two to reduce recompiles; clamp to table width
        M = 1 << (max_len - 1).bit_length()
        M = min(M, int(self.max_num_pages_per_req))

        # Build a single padded host buffer then transfer once
        arr = np.zeros((len(page_ids_list), M), dtype=np.int32)
        for b, ids in enumerate(page_ids_list):
            if ids:
                upto = min(len(ids), M)
                arr[b, :upto] = np.asarray(ids[:upto], dtype=np.int32)

        req_indices = jnp.asarray(req_indices_list, dtype=jnp.int32)
        lengths = jnp.asarray(lengths_py, dtype=jnp.int32)
        new_pages_padded = jnp.asarray(arr)

        self.page_table, self.num_pages_per_row = self._append_rows_batch_jit(
            self.page_table,
            self.num_pages_per_row,
            req_indices,
            new_pages_padded,
            lengths,
            jnp.int32(self.max_num_pages_per_req),
        )

    def add_rows_batch(self, page_ids_list: list[list[int]], req_indices_list: list[int]) -> None:
        """
        Reset rows in req_indices_list to zero length, then append batched.

        Args:
            page_ids_list: list of new pages per request
            req_indices_list: row indices to reset and fill
        """
        if not page_ids_list:
            return
        req_indices = jnp.asarray(req_indices_list, dtype=jnp.int32)
        self.num_pages_per_row = self.num_pages_per_row.at[req_indices].set(0)
        self.append_rows_batch(page_ids_list, req_indices_list)


class MultiGroupPageTable:
    """The PageTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        page_sizes: list[int],
    ) -> None:
        self.page_tables = [
            PageTable(
                page_size=page_size,
                max_num_reqs=max_num_reqs,
                max_num_pages_per_req=cdiv(max_model_len, page_size),
                max_num_batched_tokens=max_num_batched_tokens,
            )
            for page_size in page_sizes
        ]

    def append_row(self, page_ids: tuple[list[int], ...], row_idx: int) -> None:
        """Append page IDs to a row for each KV group."""
        for gi, page_table in enumerate(self.page_tables):
            page_table.append_row(page_ids[gi], row_idx)

    def add_row(self, page_ids: tuple[list[int], ...], row_idx: int) -> None:
        """Reset row to zero length, then append per KV group."""
        for gi, page_table in enumerate(self.page_tables):
            page_table.add_row(page_ids[gi], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for page_table in self.page_tables:
            page_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for page_table in self.page_tables:
            page_table.swap_row(src, tgt)

    def append_rows_batch(self, page_ids_per_req: list[tuple[list[int], ...]], req_indices: list[int]) -> None:
        """
        Batched append across all KV groups.

        Args:
            page_ids_per_req: list of tuples (one per request), each tuple contains one list[int] per KV group
            req_indices: list[int], row indices for each request
        """
        if not page_ids_per_req:
            return
        num_groups = len(self.page_tables)
        for gi in range(num_groups):
            group_lists = [tpl[gi] for tpl in page_ids_per_req]  # list[list[int]]
            self.page_tables[gi].append_rows_batch(group_lists, req_indices)

    def add_rows_batch(self, page_ids_per_req: list[tuple[list[int], ...]], req_indices: list[int]) -> None:
        """
        Reset rows to zero length, then batched append across all KV groups.
        """
        if not page_ids_per_req:
            return
        num_groups = len(self.page_tables)
        for gi in range(num_groups):
            group_lists = [tpl[gi] for tpl in page_ids_per_req]
            self.page_tables[gi].add_rows_batch(group_lists, req_indices)

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> None:
        """Compute slot mapping in each KV group (if needed)."""
        for page_table in self.page_tables:
            page_table.compute_slot_mapping(req_indices, positions)

    def clear(self) -> None:
        for page_table in self.page_tables:
            page_table.clear()

    def __getitem__(self, idx: int) -> PageTable:
        """Returns the PageTable for the idx-th KV cache group."""
        return self.page_tables[idx]
