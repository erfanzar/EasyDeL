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


import numpy as np

from easydel.utils.helpers import get_logger

logger = get_logger(__name__)


def cdiv(a: int, b: int) -> int:
    """Ceiling division"""
    return (a + b - 1) // b


class PageTable:
    def __init__(
        self,
        page_size: int,
        max_num_reqs: int,
        max_num_pages_per_req: int,
        max_num_batched_tokens: int,
    ):
        self.page_size = page_size
        self.max_num_reqs = max_num_reqs
        self.max_num_pages_per_req = max_num_pages_per_req
        self.max_num_batched_tokens = max_num_batched_tokens

        self.page_table = np.full((max_num_reqs, max_num_pages_per_req), fill_value=-1, dtype=np.int32)
        self.num_pages_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = np.full(self.max_num_batched_tokens, fill_value=-1, dtype=np.int32)

    def append_row(self, page_ids: list[int], row_idx: int) -> None:
        if not page_ids:
            return
        num_pages = len(page_ids)
        start = int(self.num_pages_per_row[row_idx])
        page_ids_array = np.array(page_ids, dtype=np.int32)
        self.page_table[row_idx, start : start + num_pages] = page_ids_array
        self.num_pages_per_row[row_idx] += num_pages

    def add_row(self, page_ids: list[int], row_idx: int) -> None:
        self.num_pages_per_row[row_idx] = 0
        self.append_row(page_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_pages = int(self.num_pages_per_row[src])
        self.page_table[tgt, :num_pages] = self.page_table[src, :num_pages]
        self.num_pages_per_row[tgt] = num_pages

    def swap_row(self, src: int, tgt: int) -> None:
        num_pages_src = int(self.num_pages_per_row[src])
        num_pages_tgt = int(self.num_pages_per_row[tgt])

        self.num_pages_per_row[src] = num_pages_tgt
        self.num_pages_per_row[tgt] = num_pages_src

        src_row = self.page_table[src].copy()
        tgt_row = self.page_table[tgt].copy()
        self.page_table[src] = tgt_row
        self.page_table[tgt] = src_row

    def compute_slot_mapping(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        page_table_indices = req_indices * self.max_num_pages_per_req + positions // self.page_size

        page_table_flat = self.page_table.flatten()
        page_numbers = page_table_flat[page_table_indices]

        page_offsets = positions % self.page_size

        slot_values = page_numbers * self.page_size + page_offsets
        num_tokens = req_indices.shape[0]
        self.slot_mapping[:num_tokens] = slot_values

    def clear(self) -> None:
        self.page_table = np.zeros_like(self.page_table)
        self.num_pages_per_row = np.zeros_like(self.num_pages_per_row)
        self.slot_mapping = np.zeros_like(self.slot_mapping)

    def get_array(self) -> np.ndarray:
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

    def compute_slot_mapping(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        for page_table in self.page_tables:
            page_table.compute_slot_mapping(req_indices, positions)

    def clear(self) -> None:
        for page_table in self.page_tables:
            page_table.clear()

    def __getitem__(self, idx: int) -> PageTable:
        """Returns the PageTable for the i-th KV cache group."""
        return self.page_tables[idx]
