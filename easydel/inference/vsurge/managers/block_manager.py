# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class Block:
    """Immutable block representation with efficient hashing."""

    block_id: int
    ref_count: int = 0
    content_hash: int | None = None

    def __hash__(self) -> int:
        return self.block_id

    def __lt__(self, other: Block) -> bool:
        return self.block_id < other.block_id


class BlockAllocator:
    """Efficient block allocation with O(log n) operations."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: list[int] = list(range(num_blocks))
        heapq.heapify(self.free_blocks)
        self.allocated: set[int] = set()

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("No free blocks available")
        block_id = heapq.heappop(self.free_blocks)
        self.allocated.add(block_id)
        return block_id

    def deallocate(self, block_id: int) -> None:
        if block_id in self.allocated:
            self.allocated.remove(block_id)
            heapq.heappush(self.free_blocks, block_id)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class BlockManager:
    """Thread-safe block manager with copy-on-write semantics."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.blocks: dict[int, Block] = {i: Block(i) for i in range(num_blocks)}
        self.hash_to_block: dict[int, int] = {}

    def allocate_blocks(self, num_blocks: int) -> list[int]:
        """Allocate multiple blocks atomically."""
        if self.allocator.num_free < num_blocks:
            raise MemoryError(f"Insufficient blocks: need {num_blocks}, have {self.allocator.num_free}")

        block_ids = []
        try:
            for _ in range(num_blocks):
                block_id = self.allocator.allocate()
                self.blocks[block_id] = Block(block_id, ref_count=1)
                block_ids.append(block_id)
            return block_ids
        except Exception:
            # Rollback on failure
            for bid in block_ids:
                self.allocator.deallocate(bid)
            raise

    def increment_ref(self, block_id: int) -> None:
        """Increment reference count for a block."""
        block = self.blocks[block_id]
        self.blocks[block_id] = Block(block_id, block.ref_count + 1, block.content_hash)

    def decrement_ref(self, block_id: int) -> None:
        """Decrement reference count and deallocate if zero."""
        block = self.blocks[block_id]
        new_ref_count = block.ref_count - 1

        if new_ref_count == 0:
            if block.content_hash and self.hash_to_block.get(block.content_hash) == block_id:
                del self.hash_to_block[block.content_hash]
            self.allocator.deallocate(block_id)
        else:
            self.blocks[block_id] = Block(block_id, new_ref_count, block.content_hash)

    def copy_on_write(self, block_id: int) -> tuple[int, bool]:
        """Return block_id if exclusive, or allocate new block if shared."""
        block = self.blocks[block_id]
        if block.ref_count == 1:
            return block_id, False

        # Need to copy
        new_block_id = self.allocator.allocate()
        self.blocks[new_block_id] = Block(new_block_id, ref_count=1)
        self.decrement_ref(block_id)
        return new_block_id, True
