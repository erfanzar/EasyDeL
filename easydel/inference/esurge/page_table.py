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

This module provides class-based page table management with dual CPU/GPU representation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from eformer.loggings import get_logger

logger = get_logger(__name__)


def cdiv(a: int, b: int) -> int:
    """Compute ceiling division.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The ceiling of a/b.

    Example:
        >>> cdiv(7, 3)  # Returns 3
        >>> cdiv(6, 3)  # Returns 2
        >>> cdiv(5, 3)  # Returns 2
    """
    return (a + b - 1) // b


SLOT_MAPPING_PADDING_VAL = 0
PAGE_TABLE_PADDING_VAL = 0


class PageTable:
    """Manages page allocation for paged KV-cache layouts.

    A class-based page table manager with dual CPU/GPU representation for
    efficient host-device synchronization.

    The page table maintains a 2D array where each row corresponds to a request
    and contains the page IDs allocated to that request. CPU-side modifications
    are explicitly committed to the GPU via the commit() method.

    Attributes:
        max_num_reqs: Maximum number of concurrent requests.
        max_num_pages_per_req: Maximum pages allocatable per request.
        max_num_batched_tokens: Maximum tokens processable in a batch.
        page_table: JAX device array [max_num_reqs, max_num_pages_per_req].
        page_table_cpu: NumPy CPU array [max_num_reqs, max_num_pages_per_req].
        num_pages_per_row: NumPy array tracking valid pages [max_num_reqs].

    Note:
        All modifications operate on CPU arrays and require explicit commit()
        to synchronize with the GPU.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_pages_per_req: int,
        max_num_batched_tokens: int,
        sharding: jax.sharding.Sharding | None = None,
    ):
        """Initialize a PageTable with specified capacity.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_num_pages_per_req: Maximum pages per request.
            max_num_batched_tokens: Maximum tokens in a batch.
            sharding: Optional sharding to use for the device page table.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_pages_per_req = max_num_pages_per_req
        self.max_num_batched_tokens = max_num_batched_tokens

        self.page_table = jnp.zeros(
            (max_num_reqs, max_num_pages_per_req),
            dtype=jnp.int32,
            device=sharding,
        )
        self.page_table_cpu = np.zeros(
            (max_num_reqs, max_num_pages_per_req),
            dtype=np.int32,
        )
        self.num_pages_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        # Incremented whenever the CPU-side page table changes. This enables
        # higher-level code to cache device views derived from `page_table_cpu`
        # (e.g., `pages_tables`) and refresh only when necessary.
        self.cpu_version: int = 0

    def append_row(
        self,
        page_ids: list[int],
        row_idx: int,
    ) -> None:
        """Append page IDs to a single row.

        Adds new pages to the end of an existing row's page list in the
        CPU array. Changes are not visible on GPU until commit() is called.

        Args:
            page_ids: Page IDs to append.
            row_idx: Index of the row to append to.

        Note:
            If page_ids is empty, this is a no-op.
            Call commit() to sync changes to GPU.
        """
        if not page_ids:
            return
        num_pages = len(page_ids)
        start = self.num_pages_per_row[row_idx]
        self.num_pages_per_row[row_idx] += num_pages
        self.page_table_cpu[row_idx, start : start + num_pages] = page_ids
        self.cpu_version += 1

    def add_row(self, page_ids: list[int], row_idx: int) -> None:
        """Replace a row with new page IDs.

        Resets the row to empty, then adds the provided page IDs to the
        CPU array. Changes are not visible on GPU until commit() is called.

        Args:
            page_ids: New page IDs for the row.
            row_idx: Index of the row to replace.

        Note:
            This is equivalent to clearing the row and then appending.
            Call commit() to sync changes to GPU.
        """
        self.num_pages_per_row[row_idx] = 0
        self.append_row(page_ids, row_idx)
        # `append_row` already bumps `cpu_version` when non-empty.
        if not page_ids:
            self.cpu_version += 1

    def move_row(self, src: int, tgt: int) -> None:
        """Move row content from source to target.

        Copies the page IDs and count from source row to target row in the
        CPU array. Only copies the valid pages (up to source length).

        Args:
            src: Source row index.
            tgt: Target row index.

        Note:
            The source row is not cleared; use this for copying.
            Call commit() to sync changes to GPU.
        """
        num_pages = self.num_pages_per_row[src]
        self.page_table_cpu[tgt, :num_pages] = self.page_table_cpu[src, :num_pages]
        self.num_pages_per_row[tgt] = num_pages
        self.cpu_version += 1

    def swap_row(self, src: int, tgt: int) -> None:
        """Swap two rows in the page table.

        Exchanges both page IDs and page counts between two rows in the
        CPU array.

        Args:
            src: First row index.
            tgt: Second row index.

        Note:
            This is a full swap including both content and metadata.
            Call commit() to sync changes to GPU.
        """
        num_pages_src = self.num_pages_per_row[src]
        num_pages_tgt = self.num_pages_per_row[tgt]
        self.num_pages_per_row[src] = num_pages_tgt
        self.num_pages_per_row[tgt] = num_pages_src

        self.page_table_cpu[[src, tgt]] = self.page_table_cpu[[tgt, src]]
        self.cpu_version += 1

    def commit(self, num_reqs: int) -> None:
        """Commit CPU modifications to GPU.

        Copies the first num_reqs rows from CPU array to GPU array.
        This synchronizes modifications made via append_row, add_row,
        move_row, or swap_row.

        Args:
            num_reqs: Number of request rows to commit.

        Note:
            Uses non-blocking transfer for better performance.
        """
        self.page_table = self.page_table.at[:num_reqs].set(self.page_table_cpu[:num_reqs])

    def clear(self) -> None:
        """Clear all data in the page table.

        Resets both CPU and GPU arrays to zero.

        Note:
            This clears both the page table and the page counts.
        """
        self.page_table = jnp.zeros_like(self.page_table)
        self.page_table_cpu.fill(0)
        self.num_pages_per_row.fill(0)
        self.cpu_version += 1

    def get_device_tensor(self) -> jax.Array:
        """Get the GPU device tensor of the page table.

        Returns:
            The 2D JAX array on device [max_num_reqs, max_num_pages_per_req].

        Note:
            This returns the GPU-side array. It may not reflect recent
            CPU-side modifications until commit() is called.
        """
        return self.page_table

    def get_cpu_tensor(self) -> np.ndarray:
        """Get the CPU tensor of the page table.

        Returns:
            The 2D NumPy array on CPU [max_num_reqs, max_num_pages_per_req].

        Note:
            This returns the CPU-side array where modifications are made.
        """
        return self.page_table_cpu


class MultiGroupPageTable:
    """Multi-group page table for grouped-query attention.

    Manages multiple PageTable instances, one per KV-cache group.
    This is used when the model uses grouped-query attention (GQA)
    where different attention heads may share KV-caches.

    Attributes:
        page_tables: List of PageTable instances, one per group.

    Note:
        All operations coordinate across all groups and modify state in-place.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        page_sizes: list[int],
        sharding: jax.sharding.Sharding | None = None,
    ) -> None:
        """Initialize a MultiGroupPageTable with page tables for each group.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_model_len: Maximum model sequence length.
            max_num_batched_tokens: Maximum batch token count.
            page_sizes: List of page sizes, one per KV-cache group.
            sharding: Optional sharding to use for the device page tables.
        """
        self.page_tables = [
            PageTable(
                max_num_reqs,
                cdiv(max_model_len, page_size),
                max_num_batched_tokens,
                sharding=sharding,
            )
            for page_size in page_sizes
        ]

    def append_row(self, page_ids: list[list[int]], row_idx: int) -> None:
        """Append pages to a row across all groups.

        Args:
            page_ids: List of page ID lists, one per group.
            row_idx: Row index to append to.

        Note:
            The length of page_ids must match the number of groups.
            Call commit() to sync changes to GPU.
        """
        for i, page_table in enumerate(self.page_tables):
            page_table.append_row(page_ids[i], row_idx)

    def add_row(self, page_ids: list[list[int]], row_idx: int) -> None:
        """Replace a row across all groups.

        Args:
            page_ids: List of new page ID lists, one per group.
            row_idx: Row index to replace.

        Note:
            Call commit() to sync changes to GPU.
        """
        for i, page_table in enumerate(self.page_tables):
            page_table.add_row(page_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """Move a row across all groups.

        Args:
            src: Source row index.
            tgt: Target row index.

        Note:
            Call commit() to sync changes to GPU.
        """
        for page_table in self.page_tables:
            page_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        """Swap two rows across all groups.

        Args:
            src: First row index.
            tgt: Second row index.

        Note:
            Call commit() to sync changes to GPU.
        """
        for page_table in self.page_tables:
            page_table.swap_row(src, tgt)

    def commit(self, num_reqs: int) -> None:
        """Commit CPU modifications to GPU for all groups.

        Args:
            num_reqs: Number of request rows to commit.

        Note:
            This commits changes across all page table groups.
        """
        for page_table in self.page_tables:
            page_table.commit(num_reqs)

    def clear(self) -> None:
        """Clear all page tables across all groups.

        Note:
            This clears both CPU and GPU arrays for all groups.
        """
        for page_table in self.page_tables:
            page_table.clear()

    def append_rows_batch(
        self,
        page_ids_per_req: list[list[list[int]]],
        req_indices: list[int],
    ) -> None:
        """Batch append pages across all groups.

        Args:
            page_ids_per_req: List of page ID lists per group per request.
                Shape: [num_requests][num_groups][variable_length_pages]
            req_indices: Row indices to append to.

        Note:
            Call commit() to sync changes to GPU after batch operations.

        Example:
            >>> # Two requests, two groups
            >>> multi_table.append_rows_batch(
            ...     [[[10, 11], [20, 21]],      # Request 0: group 0 and 1 pages
            ...      [[30], [40, 41, 42]]],     # Request 1: group 0 and 1 pages
            ...     [0, 1]                       # Append to rows 0 and 1
            ... )
            >>> multi_table.commit(2)
        """
        if not page_ids_per_req:
            return
        num_groups = len(self.page_tables)
        for req_idx, page_ids_for_req in zip(req_indices, page_ids_per_req, strict=True):
            for group_idx in range(num_groups):
                self.page_tables[group_idx].append_row(page_ids_for_req[group_idx], req_idx)

    def __getitem__(self, idx: int) -> PageTable:
        """Get a specific group's page table.

        Args:
            idx: Group index.

        Returns:
            The PageTable for the specified group.
        """
        return self.page_tables[idx]
