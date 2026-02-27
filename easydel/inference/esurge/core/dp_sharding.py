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

The page pool reserves page-id ``0`` as a null/padding page. DP-local page
sharding must therefore partition only usable page IDs ``[1, num_pages)``.
"""

from __future__ import annotations


def usable_pages_count(num_pages: int) -> int:
    """Return the number of allocatable pages (excluding null page 0)."""
    return max(0, int(num_pages) - 1)


def pages_per_dp_shard(num_pages: int, data_parallel_size: int | None) -> int | None:
    """Return usable pages per DP shard, or ``None`` when not evenly partitionable."""
    dp_size = max(1, int(data_parallel_size or 1))
    if dp_size <= 1:
        return None

    usable_pages = usable_pages_count(num_pages)
    if usable_pages <= 0 or usable_pages % dp_size != 0:
        return None
    return usable_pages // dp_size


def dp_shard_for_page_id(page_id: int, pages_per_shard: int, dp_size: int) -> int | None:
    """Map a non-null page ID to a DP shard index."""
    pid = int(page_id)
    if pid <= 0 or pages_per_shard <= 0 or dp_size <= 0:
        return None
    return min((pid - 1) // pages_per_shard, dp_size - 1)


def dp_shard_page_bounds(shard_index: int, pages_per_shard: int) -> tuple[int, int]:
    """Return inclusive-exclusive usable page-ID bounds for a DP shard."""
    shard = max(0, int(shard_index))
    lo = 1 + shard * int(pages_per_shard)
    hi = lo + int(pages_per_shard)
    return lo, hi
