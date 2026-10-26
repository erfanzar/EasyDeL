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

Paged attention KV can be reused from the prefix cache, but linear-attention / SSM layers keep one running
state per request that the page table does not cover. The scheduler ends a prefill chunk exactly on a page
boundary, the runner copies the request's recurrent row into a snapshot slot after that step, and a later
request that hits the same prefix gets that slot copied back into its row before its first step. Snapshots
are keyed by the chained hash of the boundary page, which identifies the whole prefix.
"""

from __future__ import annotations

import typing
from collections import OrderedDict


class RecurrentSnapshotIndex:
    """Slot bookkeeping for recurrent-state snapshots.

    A slot is *pending* from the schedule step that asks the runner to write it until the scheduler sees that
    step's output, and *ready* afterwards. Only ready slots are returned by :meth:`lookup`. When every slot is
    taken, the least recently used ready slot is recycled.
    """

    def __init__(self, num_slots: int) -> None:
        if int(num_slots) <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.num_slots = int(num_slots)
        self.reset()

    def reset(self) -> None:
        """Forget every snapshot."""
        self._ready: OrderedDict[typing.Hashable, int] = OrderedDict()
        self._pending: dict[int, typing.Hashable] = {}
        self._pending_keys: set[typing.Hashable] = set()
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))

    def contains(self, key: typing.Hashable) -> bool:
        """Whether ``key`` is ready or about to be written."""
        return key in self._ready or key in self._pending_keys

    def is_ready(self, key: typing.Hashable) -> bool:
        """Whether a written snapshot of ``key`` exists."""
        return key in self._ready

    def lookup(self, key: typing.Hashable) -> int | None:
        """Return the ready slot holding ``key`` and mark it recently used."""
        slot = self._ready.get(key)
        if slot is not None:
            self._ready.move_to_end(key)
        return slot

    def reserve(self, key: typing.Hashable, exclude: typing.Collection[int] = ()) -> int | None:
        """Claim a slot for a new snapshot of ``key``.

        Args:
            key: Prefix key the snapshot will hold.
            exclude: Ready slots that must not be recycled (restores issued in the same step).

        Returns:
            The claimed slot, or ``None`` if ``key`` already exists or no slot can be freed.
        """
        if self.contains(key):
            return None
        if self._free:
            slot = self._free.pop()
        else:
            victim = next((k for k, s in self._ready.items() if s not in exclude), None)
            if victim is None:
                return None
            slot = self._ready.pop(victim)
        self._pending[slot] = key
        self._pending_keys.add(key)
        return slot

    def cancel(self, slot: int) -> None:
        """Drop a reservation whose write will not happen."""
        key = self._pending.pop(int(slot), None)
        if key is None:
            return
        self._pending_keys.discard(key)
        self._free.append(int(slot))

    def commit(self, slot: int) -> None:
        """Mark a reserved slot as written by the runner."""
        key = self._pending.pop(int(slot), None)
        if key is None:
            return
        self._pending_keys.discard(key)
        self._ready[key] = int(slot)
