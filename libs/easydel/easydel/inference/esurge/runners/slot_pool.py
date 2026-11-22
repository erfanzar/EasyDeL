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

"""Physical recurrent-state slot bookkeeping for the eSurge runner.

Pure host-side bookkeeping: requests whose decode state lives in pooled physical slots (rank-major SPMD DP recurrent
caches, compressed-window caches, recurrent layers that declare the eSurge slot contract) keep a stable group of slots
for their lifetime even when their :class:`SequenceBuffer` row moves. :class:`RecurrentSlotPool` owns the free-lists
and the request-to-slot/rank maps.
"""

from __future__ import annotations

import typing


class RecurrentSlotPool:
    """Stable physical recurrent-state slot groups partitioned by DP rank.

    Every request owns a contiguous group of ``group_size`` (W) slots ``g .. g + W - 1`` for its lifetime, where ``g``
    is the group's base slot: ``W = 1 + K_max`` when speculative verify steps capture per-token states (the eSurge
    recurrent step contract, ``runners/slot_contract.py``), 1 otherwise. Under rank-major SPMD DP, rank ``r`` owns the
    contiguous slot block ``[r * rows_per_rank * W, (r + 1) * rows_per_rank * W)`` and every group of its requests lies
    inside it.

    Attributes:
        metadata: Paged-attention cache metadata; only ``data_parallel_size`` is read (live, at every call).
        max_num_reqs: Request rows managed; each owns one group.
        slot_indexed_state: Whether requests are pinned to pooled slots without SPMD DP (compressed-window caches and
            recurrent layers that declare the slot contract).
        group_size: Slots per request, ``W``.
        slot_by_req: Mapping from request id to the base slot of its group.
        dp_rank_by_req: Mapping from request id to its recorded DP rank.
        free_slots_by_rank: Per-rank free-lists of group base slots, lowest first.
    """

    def __init__(
        self,
        *,
        metadata: typing.Any,
        max_num_reqs: int,
        slot_indexed_state: bool,
        group_size: int = 1,
    ) -> None:
        """Initialize the pool and build the per-rank free-lists.

        Args:
            metadata: Cache metadata carrying ``data_parallel_size``.
            max_num_reqs: Request rows to manage (one slot group each).
            slot_indexed_state: Whether slot pooling is required independent of SPMD DP.
            group_size: Slots per request, ``W`` (at least 1).
        """
        self.metadata = metadata
        self.max_num_reqs = int(max_num_reqs)
        self.slot_indexed_state = bool(slot_indexed_state)
        self.group_size = max(1, int(group_size))
        self.reset()

    def uses_spmd_dp(self) -> bool:
        """Return whether rank-major data-parallel execution is active."""
        return int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1

    def is_enabled(self) -> bool:
        """Whether requests are pinned to pooled physical state slots.

        True under rank-major SPMD DP (per-rank slot blocks) and for pooled state (compressed-window caches, recurrent
        layers that declare the slot contract), where the request's slots must survive SequenceBuffer row moves.
        """
        return self.uses_spmd_dp() or self.slot_indexed_state

    def _dp_size(self) -> int:
        """Data-parallel size of the cache metadata (at least 1)."""
        return max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))

    def rows_per_dp_rank(self) -> int:
        """Return the number of request rows owned by each DP rank.

        Returns:
            ``max_num_reqs // data_parallel_size`` as an int.

        Raises:
            ValueError: If ``max_num_reqs`` is not divisible by the DP size.
        """
        dp_size = self._dp_size()
        if int(self.max_num_reqs) % dp_size != 0:
            raise ValueError(
                "Rank-major SPMD DP requires recurrent slots divisible by DP size: "
                f"max_num_reqs={self.max_num_reqs}, dp_size={dp_size}."
            )
        return int(self.max_num_reqs) // dp_size

    def slots_per_dp_rank(self) -> int:
        """Return the number of physical slots in each DP rank's block, ``rows_per_dp_rank() * group_size``."""
        return self.rows_per_dp_rank() * self.group_size

    def num_slots(self) -> int:
        """Return the number of physical slots the recurrent pools hold, ``max_num_reqs * group_size``."""
        return int(self.max_num_reqs) * self.group_size

    def group_slots(self, base_slot: int) -> list[int]:
        """Return every physical slot of the group whose base slot is ``base_slot``."""
        return [int(base_slot) + j for j in range(self.group_size)]

    def reset(self) -> None:
        """Forget every assignment and rebuild the per-rank free-lists of group base slots."""
        self.slot_by_req: dict[str, int] = {}
        self.dp_rank_by_req: dict[str, int] = {}
        width = self.group_size
        if not self.uses_spmd_dp():
            self.free_slots_by_rank = [list(range(0, self.num_slots(), width))]
            return
        rows_per_rank = self.rows_per_dp_rank()
        slots_per_rank = rows_per_rank * width
        self.free_slots_by_rank = [
            list(range(rank * slots_per_rank, (rank + 1) * slots_per_rank, width)) for rank in range(self._dp_size())
        ]

    def _rank_of_slot(self, slot: int) -> int:
        """DP rank whose slot block contains ``slot``."""
        return min(max(int(slot), 0) // self.slots_per_dp_rank(), self._dp_size() - 1)

    def assign_slot(self, req_id: str, dp_rank: int | None) -> int | None:
        """Assign or return the base slot of a request's stable slot group.

        A request that already owns a group gets the same base slot back; otherwise the lowest free group of the
        target rank is allocated and recorded.

        Args:
            req_id: Request identifier to assign a group for.
            dp_rank: Target DP rank for the request, or ``None`` for rank ``0``.

        Returns:
            The group's base slot, or ``None`` when the pool is disabled (neither SPMD DP nor pooled state).

        Raises:
            RuntimeError: If an existing group's rank disagrees with ``dp_rank``, if ``dp_rank`` is out of range, or
                if no free group remains in the target rank.
        """
        if not self.is_enabled():
            return None
        existing = self.slot_by_req.get(req_id)
        if existing is not None:
            existing_rank = self._rank_of_slot(int(existing))
            if dp_rank is not None and existing_rank != int(dp_rank):
                raise RuntimeError(
                    "Existing recurrent slot rank does not match scheduler DP rank: "
                    f"req_id={req_id} slot={existing} slot_rank={existing_rank} dp_rank={dp_rank}."
                )
            return int(existing)

        rank = 0 if dp_rank is None else int(dp_rank)
        if rank < 0 or rank >= len(self.free_slots_by_rank):
            raise RuntimeError(f"Invalid DP rank for recurrent slot assignment: req_id={req_id} dp_rank={rank}.")
        free_slots = self.free_slots_by_rank[rank]
        if not free_slots:
            raise RuntimeError(f"No free recurrent state slots in DP rank {rank} for request {req_id}.")
        slot = int(free_slots.pop(0))
        self.slot_by_req[req_id] = slot
        self.dp_rank_by_req[req_id] = rank
        return slot

    def release_slot(self, req_id: str, *, forget_rank: bool) -> int | None:
        """Release a request's slot group and return its base slot for clearing.

        Pops the request's mapping and, when the pool is enabled, returns the group to its owning rank's free-list
        (kept sorted). The caller clears every slot of the group (:meth:`group_slots`).

        Args:
            req_id: Request identifier whose group should be released.
            forget_rank: When True, also drop the request's recorded DP rank (finished requests; preemptions keep it).

        Returns:
            The released base slot, or ``None`` when the request held no group.
        """
        slot = self.slot_by_req.pop(req_id, None)
        if forget_rank:
            self.dp_rank_by_req.pop(req_id, None)
        if slot is None or not self.is_enabled():
            return slot
        free_slots = self.free_slots_by_rank[self._rank_of_slot(int(slot))]
        if int(slot) not in free_slots:
            free_slots.append(int(slot))
            free_slots.sort()
        return int(slot)
