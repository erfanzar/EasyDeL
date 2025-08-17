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
from jax import numpy as jnp

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
            (max_num_reqs, max_num_pages_per_req), fill_value=PAGE_TABLE_PADDING_VAL, dtype=jnp.int32
        )
        self.num_pages_per_row = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.slot_mapping = jnp.full(self.max_num_batched_tokens, fill_value=SLOT_MAPPING_PADDING_VAL, dtype=jnp.int32)

    def append_row(self, page_ids: list[int], row_idx: int) -> None:
        """Append page IDs to a row.

        Args:
            page_ids: List of page IDs to append.
            row_idx: Row index to append to.
        """
        if not page_ids:
            return
        num_pages = len(page_ids)
        start = int(self.num_pages_per_row[row_idx])
        page_ids_array = jnp.array(page_ids, dtype=jnp.int32)
        self.page_table = self.page_table.at[row_idx, start : start + num_pages].set(page_ids_array)
        self.num_pages_per_row = self.num_pages_per_row.at[row_idx].set(self.num_pages_per_row[row_idx] + num_pages)

    def add_row(self, page_ids: list[int], row_idx: int) -> None:
        self.num_pages_per_row = self.num_pages_per_row.at[row_idx].set(0)
        self.append_row(page_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_pages = int(self.num_pages_per_row[src])
        self.page_table = self.page_table.at[tgt, :num_pages].set(self.page_table[src, :num_pages])
        self.num_pages_per_row = self.num_pages_per_row.at[tgt].set(num_pages)

    def swap_row(self, src: int, tgt: int) -> None:
        num_pages_src = int(self.num_pages_per_row[src])
        num_pages_tgt = int(self.num_pages_per_row[tgt])

        self.num_pages_per_row = self.num_pages_per_row.at[src].set(num_pages_tgt)
        self.num_pages_per_row = self.num_pages_per_row.at[tgt].set(num_pages_src)

        src_row = self.page_table[src]
        tgt_row = self.page_table[tgt]
        self.page_table = self.page_table.at[src].set(tgt_row)
        self.page_table = self.page_table.at[tgt].set(src_row)

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> None:
        page_table_indices = req_indices * self.max_num_pages_per_req + positions // self.page_size

        page_table_flat = self.page_table.flatten()
        page_numbers = page_table_flat[page_table_indices]

        page_offsets = positions % self.page_size

        slot_values = page_numbers * self.page_size + page_offsets
        num_tokens = req_indices.shape[0]
        self.slot_mapping = self.slot_mapping.at[:num_tokens].set(slot_values)

    def clear(self) -> None:
        self.page_table = jnp.zeros_like(self.page_table)
        self.num_pages_per_row = jnp.zeros_like(self.num_pages_per_row)
        self.slot_mapping = jnp.zeros_like(self.slot_mapping)

    def get_array(self) -> jax.Array:
        """Returns the array of the page table."""
        return self.page_table


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
                page_size,
                max_num_reqs,
                cdiv(max_model_len, page_size),
                max_num_batched_tokens,
            )
            for page_size in page_sizes
        ]

    def append_row(self, page_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, page_table in enumerate(self.page_tables):
            page_table.append_row(page_ids[i], row_idx)

    def add_row(self, page_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, page_table in enumerate(self.page_tables):
            page_table.add_row(page_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for page_table in self.page_tables:
            page_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for page_table in self.page_tables:
            page_table.swap_row(src, tgt)

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> None:
        for page_table in self.page_tables:
            page_table.compute_slot_mapping(req_indices, positions)

    def clear(self) -> None:
        for page_table in self.page_tables:
            page_table.clear()

    def __getitem__(self, idx: int) -> PageTable:
        """Returns the PageTable for the i-th KV cache group."""
        return self.page_tables[idx]
