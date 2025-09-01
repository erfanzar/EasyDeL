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

"""Page table management for KV-cache allocation (JAX-jittable, dataclass PyTrees).

This version:
  - Converts PageTable and MultiGroupPageTable to dataclass PyTrees
  - Makes updates functional (return new instances) to be jit-friendly
  - Keeps a fused jittable kernel for batch appends with buffer donation

Example:
    >>> table = PageTable.create(
    ...     page_size=16,
    ...     max_num_reqs=32,
    ...     max_num_pages_per_req=128,
    ...     max_num_batched_tokens=2048,
    ... )
    >>> table = table.add_row([10, 11, 12], row_idx=0)
    >>> slots = table.get_slot_mapping(jnp.array([0]), jnp.array([48]))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import jax
import numpy as np
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp

from easydel.utils.compiling_utils import ejit

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


@auto_pytree(frozen=True)
class PageTable:
    """Manages page allocation and slot mapping for KV-cache.

    A functional, JAX-jittable dataclass that manages the mapping between
    request positions and physical page locations in the KV-cache. Supports
    efficient batched operations and is designed to work with paged attention.

    The page table maintains a 2D array where each row corresponds to a request
    and contains the page IDs allocated to that request. It also manages slot
    mapping for direct token-to-cache-position translation.

    Attributes:
        page_size: Number of tokens per page (static, non-pytree).
        max_num_reqs: Maximum number of concurrent requests (static).
        max_num_pages_per_req: Maximum pages allocatable per request (static).
        max_num_batched_tokens: Maximum tokens processable in a batch (static).
        page_table: Page allocation table [max_num_reqs, max_num_pages_per_req].
        num_pages_per_row: Number of valid pages per request [max_num_reqs].
        slot_mapping: Direct token-to-slot mapping [max_num_batched_tokens].

    Note:
        All operations are functional and return new instances rather than
        modifying in place, making them compatible with JAX transformations.
    """

    # Static (non-pytree leaves) configuration
    page_size: int = field(pytree_node=False)
    max_num_reqs: int = field(pytree_node=False)
    max_num_pages_per_req: int = field(pytree_node=False)
    max_num_batched_tokens: int = field(pytree_node=False)

    # PyTree leaves
    page_table: jax.Array
    num_pages_per_row: jax.Array
    slot_mapping: jax.Array

    @classmethod
    def create(
        cls,
        page_size: int,
        max_num_reqs: int,
        max_num_pages_per_req: int,
        max_num_batched_tokens: int,
    ) -> PageTable:
        """Create a new PageTable with initialized buffers.

        Factory method that creates a PageTable with properly initialized
        arrays filled with padding values.

        Args:
            page_size: Number of tokens per page.
            max_num_reqs: Maximum number of concurrent requests.
            max_num_pages_per_req: Maximum pages per request.
            max_num_batched_tokens: Maximum tokens in a batch.

        Returns:
            A new PageTable instance with initialized arrays.

        Example:
            >>> table = PageTable.create(
            ...     page_size=16,
            ...     max_num_reqs=32,
            ...     max_num_pages_per_req=128,
            ...     max_num_batched_tokens=2048
            ... )
        """
        page_table = jnp.full(
            (max_num_reqs, max_num_pages_per_req),
            fill_value=PAGE_TABLE_PADDING_VAL,
            dtype=jnp.int32,
        )
        num_pages_per_row = jnp.zeros((max_num_reqs,), dtype=jnp.int32)
        slot_mapping = jnp.full(
            (max_num_batched_tokens,),
            fill_value=SLOT_MAPPING_PADDING_VAL,
            dtype=jnp.int32,
        )
        return cls(
            page_size=page_size,
            max_num_reqs=max_num_reqs,
            max_num_pages_per_req=max_num_pages_per_req,
            max_num_batched_tokens=max_num_batched_tokens,
            page_table=page_table,
            num_pages_per_row=num_pages_per_row,
            slot_mapping=slot_mapping,
        )

    # ---------------------------
    # Jittable fused kernel
    # ---------------------------
    @staticmethod
    @ejit(donate_argnums=(0, 1))
    def _append_rows_batch_jit(
        page_table: jax.Array,
        num_pages_per_row: jax.Array,
        req_indices: jax.Array,  # [B] int32
        new_pages_padded: jax.Array,  # [B, M] int32
        lengths: jax.Array,  # [B] int32
        max_cols: jax.Array,  # scalar int32
    ) -> tuple[jax.Array, jax.Array]:
        """Fused kernel for batched page list appending.

        Efficiently appends variable-length page lists to multiple rows
        in a single operation with masked writes.

        Args:
            page_table: Current page table array to update.
            num_pages_per_row: Current page counts per row.
            req_indices: Indices of rows to append to [B].
            new_pages_padded: Padded array of new pages [B, M].
            lengths: Valid lengths in new_pages_padded [B].
            max_cols: Maximum columns (pages) per row.

        Returns:
            A tuple containing:
                - Updated page_table array
                - Updated num_pages_per_row array

        Note:
            This is a JIT-compiled kernel optimized for batched operations.
            Arrays are donated for in-place updates when possible.
        """
        B, M = new_pages_padded.shape
        starts = num_pages_per_row[req_indices]  # [B]
        offs = jnp.arange(M, dtype=jnp.int32)[None, :].repeat(B, 0)  # [B, M]

        valid_write = offs < lengths[:, None]  # [B, M]
        cols = starts[:, None] + offs  # [B, M]
        in_bounds = cols < max_cols  # [B, M]
        valid = valid_write & in_bounds  # [B, M]

        rows = jnp.broadcast_to(req_indices[:, None], (B, M))  # [B, M]
        safe_cols = jnp.clip(cols, 0, max_cols - 1)  # [B, M]

        rows_f = rows.reshape(-1)  # [B*M]
        cols_f = safe_cols.reshape(-1)  # [B*M]
        vals_f = new_pages_padded.reshape(-1)  # [B*M]
        valid_f = valid.reshape(-1)  # [B*M]

        current_f = page_table[rows_f, cols_f]
        desired_f = jnp.where(valid_f, vals_f, current_f)
        delta_f = desired_f - current_f

        page_table = page_table.at[(rows_f, cols_f)].add(delta_f)

        cap = max_cols - starts
        add_len = jnp.minimum(lengths, cap)  # [B]
        num_pages_per_row = num_pages_per_row.at[req_indices].add(add_len)
        return page_table, num_pages_per_row

    def append_row(self, page_ids: Sequence[int] | jax.Array, row_idx: int) -> PageTable:
        """Append page IDs to a single row.

        Adds new pages to the end of an existing row's page list.

        Args:
            page_ids: Page IDs to append (sequence or array).
            row_idx: Index of the row to append to.

        Returns:
            A new PageTable with pages appended to the specified row.

        Note:
            Works with both Python sequences and JAX arrays. When JIT-compiled,
            Python sequences are treated as static constants.
        """
        if isinstance(page_ids, jax.Array):
            arr = page_ids.astype(jnp.int32)
        else:
            if not page_ids:
                return self
            arr = jnp.asarray(page_ids, dtype=jnp.int32)

        req_indices = jnp.asarray([row_idx], dtype=jnp.int32)
        new_pages_padded = arr[None, :]  # [1, M]
        lengths = jnp.asarray([arr.shape[0]], dtype=jnp.int32)

        new_pt, new_npr = self._append_rows_batch_jit(
            self.page_table,
            self.num_pages_per_row,
            req_indices,
            new_pages_padded,
            lengths,
            jnp.int32(self.max_num_pages_per_req),
        )
        return replace(self, page_table=new_pt, num_pages_per_row=new_npr)

    def add_row(self, page_ids: Sequence[int] | jax.Array, row_idx: int) -> PageTable:
        """Replace a row with new page IDs.

        Resets the row to empty, then adds the provided page IDs.

        Args:
            page_ids: New page IDs for the row.
            row_idx: Index of the row to replace.

        Returns:
            A new PageTable with the row replaced.

        Note:
            This is equivalent to clearing the row and then appending.
        """
        new_npr = self.num_pages_per_row.at[row_idx].set(jnp.int32(0))
        tmp = replace(self, num_pages_per_row=new_npr)
        return tmp.append_row(page_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> PageTable:
        """Move row content from source to target.

        Copies the page IDs and count from source row to target row.
        Only copies the valid pages (up to source length).

        Args:
            src: Source row index.
            tgt: Target row index.

        Returns:
            A new PageTable with the row moved.

        Note:
            The source row is not cleared; use this for copying.
            Invalid pages in the target row beyond source length are preserved.
        """
        num_pages = self.num_pages_per_row[src]  # dynamic scalar
        cols = jnp.arange(self.max_num_pages_per_req, dtype=jnp.int32)
        mask = cols < num_pages  # [C]
        new_tgt_row = jnp.where(mask, self.page_table[src], self.page_table[tgt])
        new_pt = self.page_table.at[tgt].set(new_tgt_row)
        new_npr = self.num_pages_per_row.at[tgt].set(num_pages)
        return replace(self, page_table=new_pt, num_pages_per_row=new_npr)

    def swap_row(self, src: int, tgt: int) -> PageTable:
        """Swap two rows in the page table.

        Exchanges both page IDs and page counts between two rows.

        Args:
            src: First row index.
            tgt: Second row index.

        Returns:
            A new PageTable with the rows swapped.

        Note:
            This is a full swap including both content and metadata.
        """
        pt = self.page_table
        npr = self.num_pages_per_row
        src_row = pt[src]
        tgt_row = pt[tgt]
        pt = pt.at[src].set(tgt_row)
        pt = pt.at[tgt].set(src_row)
        npr_src = npr[src]
        npr_tgt = npr[tgt]
        npr = npr.at[src].set(npr_tgt)
        npr = npr.at[tgt].set(npr_src)
        return replace(self, page_table=pt, num_pages_per_row=npr)

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> PageTable:
        """Compute and store slot mapping for token positions.

        Calculates the flat cache slot indices for given request-position pairs
        and stores them in the internal slot_mapping array.

        Args:
            req_indices: Request indices for each token.
            positions: Position indices within each request.

        Returns:
            A new PageTable with updated slot_mapping.

        Note:
            The slot mapping translates (request, position) pairs to
            flat indices in the KV-cache storage.
        """
        page_table_indices = req_indices * self.max_num_pages_per_req + positions // self.page_size
        page_table_flat = self.page_table.reshape((-1,))
        page_numbers = page_table_flat[page_table_indices]
        page_offsets = positions % self.page_size
        slot_values = page_numbers * self.page_size + page_offsets
        num_tokens = req_indices.shape[0]
        new_slot_mapping = self.slot_mapping.at[:num_tokens].set(slot_values)
        return replace(self, slot_mapping=new_slot_mapping)

    def get_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> jax.Array:
        """Get slot indices for token positions.

        Pure function that computes slot mappings without modifying state.

        Args:
            req_indices: Request indices for each token.
            positions: Position indices within each request.

        Returns:
            Array of flat slot indices in the KV-cache.

        Note:
            This is a pure function that doesn't modify the PageTable.
        """
        page_table_indices = req_indices * self.max_num_pages_per_req + positions // self.page_size
        page_table_flat = self.page_table.reshape((-1,))
        page_numbers = page_table_flat[page_table_indices]
        page_offsets = positions % self.page_size
        return page_numbers * self.page_size + page_offsets

    def clear(self) -> PageTable:
        """Clear all data in the page table.

        Returns:
            A new PageTable with all arrays reset to initial values.

        Note:
            Page table is filled with PAGE_TABLE_PADDING_VAL,
            slot mapping with SLOT_MAPPING_PADDING_VAL,
            and page counts are zeroed.
        """
        new_pt = jnp.full_like(self.page_table, PAGE_TABLE_PADDING_VAL)
        new_npr = jnp.zeros_like(self.num_pages_per_row)
        new_sm = jnp.full_like(self.slot_mapping, SLOT_MAPPING_PADDING_VAL)
        return replace(self, page_table=new_pt, num_pages_per_row=new_npr, slot_mapping=new_sm)

    def get_array(self) -> jax.Array:
        """Get the underlying page table array.

        Returns:
            The 2D page table array of shape [max_num_reqs, max_num_pages_per_req].

        Note:
            This returns a reference to the internal array, not a copy.
        """
        return self.page_table

    # ---------------------------
    # Batched convenience (Python lists) -> jittable kernel
    # ---------------------------
    def append_rows_batch(self, page_ids_list: Sequence[Sequence[int]], req_indices_list: Sequence[int]) -> PageTable:
        """Batch append pages to multiple rows.

        Efficiently appends variable-length page lists to multiple rows
        in a single operation. Automatically pads to power-of-two widths
        to reduce JAX recompilation.

        Args:
            page_ids_list: List of page ID sequences, one per request.
            req_indices_list: List of row indices to append to.

        Returns:
            A new PageTable with pages appended to specified rows.

        Example:
            >>> table = table.append_rows_batch(
            ...     [[10, 11], [20, 21, 22]],
            ...     [0, 1]
            ... )

        Note:
            Padding to power-of-two widths reduces the number of
            unique compiled kernels needed.
        """
        if not page_ids_list:
            return self

        lengths_py = [len(x) for x in page_ids_list]
        max_len = max(lengths_py) if lengths_py else 0
        if max_len == 0:
            return self

        M = 1 << (max_len - 1).bit_length()
        M = min(M, int(self.max_num_pages_per_req))

        arr = np.zeros((len(page_ids_list), M), dtype=np.int32)
        for b, ids in enumerate(page_ids_list):
            if ids:
                upto = min(len(ids), M)
                arr[b, :upto] = np.asarray(ids[:upto], dtype=np.int32)

        req_indices = jnp.asarray(req_indices_list, dtype=jnp.int32)
        lengths = jnp.asarray(lengths_py, dtype=jnp.int32)
        new_pages_padded = jnp.asarray(arr)

        new_pt, new_npr = self._append_rows_batch_jit(
            self.page_table,
            self.num_pages_per_row,
            req_indices,
            new_pages_padded,
            lengths,
            jnp.int32(self.max_num_pages_per_req),
        )
        return replace(self, page_table=new_pt, num_pages_per_row=new_npr)

    def add_rows_batch(self, page_ids_list: Sequence[Sequence[int]], req_indices_list: Sequence[int]) -> PageTable:
        """Batch replace multiple rows with new pages.

        Resets specified rows to empty, then adds new page IDs.

        Args:
            page_ids_list: List of new page ID sequences.
            req_indices_list: List of row indices to replace.

        Returns:
            A new PageTable with rows replaced.

        Note:
            This is the batched version of add_row().
        """
        if not page_ids_list:
            return self
        req_indices = jnp.asarray(req_indices_list, dtype=jnp.int32)
        new_npr = self.num_pages_per_row.at[req_indices].set(0)
        return replace(self, num_pages_per_row=new_npr).append_rows_batch(page_ids_list, req_indices_list)

    # ---------------------------
    # Jittable batch append that accepts pre-padded arrays
    # ---------------------------
    def append_rows_batch_from_padded(
        self,
        req_indices: jax.Array,  # [B] int32
        new_pages_padded: jax.Array,  # [B, M] int32
        lengths: jax.Array,  # [B] int32
    ) -> PageTable:
        """Batch append using pre-padded arrays.

        Fully JIT-compatible version that accepts already-padded arrays,
        avoiding dynamic shapes.

        Args:
            req_indices: Row indices to append to [B].
            new_pages_padded: Pre-padded page IDs [B, M].
            lengths: Valid lengths in each row [B].

        Returns:
            A new PageTable with pages appended.

        Note:
            This version is preferred when arrays are already padded,
            as it avoids dynamic shape handling.
        """
        new_pt, new_npr = self._append_rows_batch_jit(
            self.page_table,
            self.num_pages_per_row,
            req_indices,
            new_pages_padded,
            lengths,
            jnp.int32(self.max_num_pages_per_req),
        )
        return replace(self, page_table=new_pt, num_pages_per_row=new_npr)


@auto_pytree(frozen=True)
class MultiGroupPageTable:
    """Multi-group page table for grouped-query attention.

    Manages multiple PageTable instances, one per KV-cache group.
    This is used when the model uses grouped-query attention (GQA)
    where different attention heads may share KV-caches.

    Attributes:
        max_num_reqs: Maximum concurrent requests (static).
        max_model_len: Maximum model sequence length (static).
        max_num_batched_tokens: Maximum batch token count (static).
        page_sizes: Tuple of page sizes per group (static).
        page_tables: Tuple of PageTable instances, one per group.

    Note:
        All operations are functional and coordinate across all groups.
    """

    # Static configuration for convenience (non-leaves)
    max_num_reqs: int = field(pytree_node=False)
    max_model_len: int = field(pytree_node=False)
    max_num_batched_tokens: int = field(pytree_node=False)
    page_sizes: tuple[int, ...] = field(pytree_node=False)

    # Leaves
    page_tables: tuple[PageTable, ...]

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        page_sizes: Sequence[int],
    ) -> MultiGroupPageTable:
        """Create a MultiGroupPageTable with initialized page tables.

        Factory method that creates a page table for each KV-cache group.

        Args:
            max_num_reqs: Maximum concurrent requests.
            max_model_len: Maximum sequence length.
            max_num_batched_tokens: Maximum tokens per batch.
            page_sizes: Page size for each KV group.

        Returns:
            A new MultiGroupPageTable with initialized page tables.

        Example:
            >>> multi_table = MultiGroupPageTable.create(
            ...     max_num_reqs=32,
            ...     max_model_len=2048,
            ...     max_num_batched_tokens=4096,
            ...     page_sizes=[16, 32]  # Two groups with different page sizes
            ... )
        """
        pts = tuple(
            PageTable.create(
                page_size=ps,
                max_num_reqs=max_num_reqs,
                max_num_pages_per_req=cdiv(max_model_len, ps),
                max_num_batched_tokens=max_num_batched_tokens,
            )
            for ps in page_sizes
        )
        return cls(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            page_sizes=tuple(page_sizes),
            page_tables=pts,
        )

    def append_row(self, page_ids: tuple[Sequence[int], ...], row_idx: int) -> MultiGroupPageTable:
        """Append pages to a row across all groups.

        Args:
            page_ids: Tuple of page ID sequences, one per group.
            row_idx: Row index to append to.

        Returns:
            A new MultiGroupPageTable with pages appended.

        Note:
            The length of page_ids must match the number of groups.
        """
        new_pts = tuple(pt.append_row(page_ids[gi], row_idx) for gi, pt in enumerate(self.page_tables))
        return replace(self, page_tables=new_pts)

    def add_row(self, page_ids: tuple[Sequence[int], ...], row_idx: int) -> MultiGroupPageTable:
        """Replace a row across all groups.

        Args:
            page_ids: Tuple of new page ID sequences, one per group.
            row_idx: Row index to replace.

        Returns:
            A new MultiGroupPageTable with the row replaced.
        """
        new_pts = tuple(pt.add_row(page_ids[gi], row_idx) for gi, pt in enumerate(self.page_tables))
        return replace(self, page_tables=new_pts)

    def move_row(self, src: int, tgt: int) -> MultiGroupPageTable:
        """Move a row across all groups.

        Args:
            src: Source row index.
            tgt: Target row index.

        Returns:
            A new MultiGroupPageTable with rows moved.
        """
        new_pts = tuple(pt.move_row(src, tgt) for pt in self.page_tables)
        return replace(self, page_tables=new_pts)

    def swap_row(self, src: int, tgt: int) -> MultiGroupPageTable:
        """Swap two rows across all groups.

        Args:
            src: First row index.
            tgt: Second row index.

        Returns:
            A new MultiGroupPageTable with rows swapped.
        """
        new_pts = tuple(pt.swap_row(src, tgt) for pt in self.page_tables)
        return replace(self, page_tables=new_pts)

    def append_rows_batch(
        self,
        page_ids_per_req: Sequence[tuple[Sequence[int], ...]],
        req_indices: Sequence[int],
    ) -> MultiGroupPageTable:
        """Batch append pages across all groups.

        Args:
            page_ids_per_req: List of tuples, each tuple contains page IDs
                for all groups for one request.
            req_indices: Row indices to append to.

        Returns:
            A new MultiGroupPageTable with pages appended.

        Example:
            >>> # Two requests, two groups
            >>> multi_table = multi_table.append_rows_batch(
            ...     [([10, 11], [20, 21]),  # Request 0: group 0 and 1 pages
            ...      ([30], [40, 41, 42])],  # Request 1: group 0 and 1 pages
            ...     [0, 1]  # Append to rows 0 and 1
            ... )
        """
        if not page_ids_per_req:
            return self
        num_groups = len(self.page_tables)
        new_pts = list(self.page_tables)
        for gi in range(num_groups):
            group_lists = [tpl[gi] for tpl in page_ids_per_req]  # list[list[int]]
            new_pts[gi] = new_pts[gi].append_rows_batch(group_lists, req_indices)
        return replace(self, page_tables=tuple(new_pts))

    def add_rows_batch(
        self,
        page_ids_per_req: Sequence[tuple[Sequence[int], ...]],
        req_indices: Sequence[int],
    ) -> MultiGroupPageTable:
        """Batch replace rows across all groups.

        Args:
            page_ids_per_req: List of tuples with new page IDs per group.
            req_indices: Row indices to replace.

        Returns:
            A new MultiGroupPageTable with rows replaced.
        """
        if not page_ids_per_req:
            return self
        num_groups = len(self.page_tables)
        new_pts = list(self.page_tables)
        for gi in range(num_groups):
            group_lists = [tpl[gi] for tpl in page_ids_per_req]
            new_pts[gi] = new_pts[gi].add_rows_batch(group_lists, req_indices)
        return replace(self, page_tables=tuple(new_pts))

    def compute_slot_mapping(self, req_indices: jax.Array, positions: jax.Array) -> MultiGroupPageTable:
        """Compute slot mappings for all groups.

        Args:
            req_indices: Request indices for tokens.
            positions: Position indices within requests.

        Returns:
            A new MultiGroupPageTable with updated slot mappings.
        """
        new_pts = tuple(pt.compute_slot_mapping(req_indices, positions) for pt in self.page_tables)
        return replace(self, page_tables=new_pts)

    def clear(self) -> MultiGroupPageTable:
        """Clear all page tables.

        Returns:
            A new MultiGroupPageTable with all groups cleared.
        """
        new_pts = tuple(pt.clear() for pt in self.page_tables)
        return replace(self, page_tables=new_pts)

    def __getitem__(self, idx: int) -> PageTable:
        """Get a specific group's page table.

        Args:
            idx: Group index.

        Returns:
            The PageTable for the specified group.
        """
        return self.page_tables[idx]
