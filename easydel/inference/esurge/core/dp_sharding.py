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

"""Helpers for data-parallel page-shard math.

The page tensor's first axis is DP-sharded, so ``num_pages`` itself must be
evenly divisible by ``data_parallel_size``.  Each shard owns a contiguous
slice of ``num_pages // dp_size`` pages.  Global page-id 0 (null/padding)
lives in shard 0.
"""

from __future__ import annotations


def usable_pages_count(num_pages: int) -> int:
    """Return the number of allocatable pages (excluding null page 0)."""
    return max(0, int(num_pages) - 1)


def pages_per_dp_shard(num_pages: int, data_parallel_size: int | None) -> int | None:
    """Return pages per DP shard, or ``None`` when not evenly partitionable.

    The check is on ``num_pages % dp_size == 0`` (matching the JAX sharding
    requirement on the full page tensor).
    """
    dp_size = max(1, int(data_parallel_size or 1))
    if dp_size <= 1:
        return None

    num_pages = int(num_pages)
    if num_pages <= 0 or num_pages % dp_size != 0:
        return None
    return num_pages // dp_size


def dp_shard_for_page_id(page_id: int, pages_per_shard: int, dp_size: int) -> int | None:
    """Map a non-null page ID to a DP shard index."""
    pid = int(page_id)
    if pid <= 0 or pages_per_shard <= 0 or dp_size <= 0:
        return None
    return min(pid // pages_per_shard, dp_size - 1)


def dp_shard_page_bounds(shard_index: int, pages_per_shard: int) -> tuple[int, int]:
    """Return inclusive-exclusive page-ID bounds for a DP shard.

    Shard 0 contains the null page (id 0), so its *usable* range starts at 1.
    """
    shard = max(0, int(shard_index))
    lo = shard * int(pages_per_shard)
    hi = lo + int(pages_per_shard)
    # Shard 0 owns page 0 (null), usable starts at 1.
    if shard == 0:
        lo = 1
    return lo, hi
