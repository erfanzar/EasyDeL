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
import jax
import jax.numpy as jnp
import numpy as np


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class BlockTable:
    """
    Manages the mapping of logical blocks to physical blocks for a single
    KV cache group, implemented in JAX.

    This class uses JAX's immutable arrays. Instead of modifying arrays
    in-place, it creates new arrays with updated values using the `.at[...].set(...)`
    pattern. It maintains a single source-of-truth array for the block table,
    simplifying the state management compared to the original PyTorch version.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_table = jnp.zeros((max_num_reqs, max_num_blocks_per_req), dtype=jnp.int32)
        self.num_blocks_per_row = jnp.zeros(max_num_reqs, dtype=jnp.int32)
        self.slot_mapping = jnp.zeros(self.max_num_batched_tokens, dtype=jnp.int64)

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return

        num_new_blocks = len(block_ids)
        start = int(self.num_blocks_per_row[row_idx])
        end = start + num_new_blocks
        self.num_blocks_per_row = self.num_blocks_per_row.at[row_idx].set(end)
        self.block_table = self.block_table.at[row_idx, start:end].set(jnp.array(block_ids, dtype=jnp.int32))

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row = self.num_blocks_per_row.at[row_idx].set(0)
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        src_row_data = self.block_table[src]

        self.block_table = self.block_table.at[tgt].set(src_row_data)
        self.num_blocks_per_row = self.num_blocks_per_row.at[tgt].set(num_blocks)

    def swap_row(self, src: int, tgt: int) -> None:
        src_row_data, tgt_row_data = self.block_table[src], self.block_table[tgt]
        self.block_table = self.block_table.at[src].set(tgt_row_data)
        self.block_table = self.block_table.at[tgt].set(src_row_data)

        src_num, tgt_num = self.num_blocks_per_row[src], self.num_blocks_per_row[tgt]
        self.num_blocks_per_row = self.num_blocks_per_row.at[src].set(tgt_num)
        self.num_blocks_per_row = self.num_blocks_per_row.at[tgt].set(src_num)

    def clear(self) -> None:
        """Resets the block table and metadata to zeros."""
        self.block_table = jnp.zeros_like(self.block_table)
        self.num_blocks_per_row = jnp.zeros_like(self.num_blocks_per_row)

    def get_device_tensor(self) -> jnp.ndarray:
        """Returns the JAX array for the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> np.ndarray:
        """Returns the block table as a NumPy array on the CPU."""
        return jax.device_get(self.block_table)

    def get_numpy_array(self) -> np.ndarray:
        """Returns the block table as a NumPy array on the CPU."""
        return self.get_cpu_tensor()


class MultiGroupBlockTable:
    """A container for BlockTables, one for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        block_sizes: list[int],
    ) -> None:
        self.block_tables = [
            BlockTable(max_num_reqs, cdiv(max_model_len, block_size), max_num_batched_tokens)
            for block_size in block_sizes
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> BlockTable:
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
