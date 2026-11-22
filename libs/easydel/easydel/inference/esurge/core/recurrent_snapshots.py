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

"""Host-side index of recurrent-state snapshots taken at prefix-cache page boundaries.

Linear-attention / SSM layers keep one running state per request that the page table does not cover. The scheduler
ends a prefill chunk on a page boundary, the runner copies the request's recurrent row into a snapshot slot after that
step, and a later request that hits the same prefix gets that slot copied back. Snapshots are keyed by the page key of
the boundary page (``CacheManager.request_page_hashes``). The index is surgecore's ``SnapshotIndex`` (Rust): a slot is
pending until committed, ready slots are recycled least recently used first, and excluded or pending slots never are.
"""

from __future__ import annotations


class RecurrentSnapshotIndex:
    """``RecurrentSnapshotIndex(num_slots, slot_offset=0)`` builds a ``surgecore.SnapshotIndex``.

    Methods: ``reset()``, ``contains(key)``, ``is_ready(key)``, ``lookup(key)``, ``reserve(key, exclude=())``,
    ``cancel(slot)``, ``commit(slot)``; attributes ``num_slots`` and ``slot_offset``.
    """

    def __new__(cls, num_slots: int, slot_offset: int = 0):
        from surgecore import SnapshotIndex

        return SnapshotIndex(int(num_slots), int(slot_offset))
