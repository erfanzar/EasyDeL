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

Pure host-side bookkeeping: requests whose decode state lives in per-request
physical rows (rank-major SPMD DP recurrent caches, compressed-window caches)
must keep a stable slot for their lifetime even when their
:class:`SequenceBuffer` row moves. :class:`RecurrentSlotPool` owns the
free-lists and request-to-slot/rank maps the runner used to keep inline.
"""

from __future__ import annotations

import typing


class RecurrentSlotPool:
    """Stable physical recurrent-state slots partitioned by DP rank.

    Attributes:
        metadata: Paged-attention cache metadata; only
            ``data_parallel_size`` is read (live, at every call).
        max_num_reqs: Total number of recurrent-state rows managed.
        slot_indexed_state: Whether the model family keeps ALL decode state
            in per-request slots (e.g. compressed-window caches), which
            forces slot pooling even without SPMD DP.
        slot_by_req: Mapping from request id to its assigned physical slot.
        dp_rank_by_req: Mapping from request id to its recorded DP rank.
        free_slots_by_rank: Per-rank free-lists of physical slot indices.
    """

    def __init__(
        self,
        *,
        metadata: typing.Any,
        max_num_reqs: int,
        slot_indexed_state: bool,
    ) -> None:
        """Initialize the pool and build the per-rank free-lists.

        Args:
            metadata: Cache metadata carrying ``data_parallel_size``.
            max_num_reqs: Total number of recurrent-state rows to manage.
            slot_indexed_state: Whether slot pooling is required independent
                of SPMD DP (slot-indexed cache families).
        """
        self.metadata = metadata
        self.max_num_reqs = int(max_num_reqs)
        self.slot_indexed_state = bool(slot_indexed_state)
        self.reset()

    def uses_spmd_dp(self) -> bool:
        """Return whether rank-major data-parallel execution is active."""
        return int(getattr(self.metadata, "data_parallel_size", 1) or 1) > 1

    def is_enabled(self) -> bool:
        """Whether requests are pinned to pooled physical state slots.

        True under rank-major SPMD DP (per-rank slot blocks) and for
        slot-indexed state families (compressed-window caches), where the
        request's state row must survive SequenceBuffer row moves.
        """
        return self.uses_spmd_dp() or self.slot_indexed_state

    def rows_per_dp_rank(self) -> int:
        """Return the number of recurrent-state rows owned by each DP rank.

        Splits the pool's ``max_num_reqs`` recurrent slots evenly across the
        data-parallel cache axis so each rank gets a contiguous block of rows.

        Returns:
            ``max_num_reqs // data_parallel_size`` as an int.

        Raises:
            ValueError: If ``max_num_reqs`` is not divisible by the DP size.
        """
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
        if int(self.max_num_reqs) % dp_size != 0:
            raise ValueError(
                "Rank-major SPMD DP requires recurrent slots divisible by DP size: "
                f"max_num_reqs={self.max_num_reqs}, dp_size={dp_size}."
            )
        return int(self.max_num_reqs) // dp_size

    def reset(self) -> None:
        """Initialize physical recurrent-state slots partitioned by DP rank."""
        self.slot_by_req: dict[str, int] = {}
        self.dp_rank_by_req: dict[str, int] = {}
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
        if not self.uses_spmd_dp():
            self.free_slots_by_rank = [list(range(int(self.max_num_reqs)))]
            return
        rows_per_rank = self.rows_per_dp_rank()
        self.free_slots_by_rank = [
            list(range(rank * rows_per_rank, (rank + 1) * rows_per_rank)) for rank in range(dp_size)
        ]

    def assign_slot(self, req_id: str, dp_rank: int | None) -> int | None:
        """Assign or return the stable physical recurrent-state slot for a request.

        Under rank-major SPMD DP, each request must keep a fixed recurrent-state
        row within its DP rank's contiguous slot block. If the request already
        owns a slot, that slot is returned unchanged; otherwise a free slot in
        the target rank is allocated and recorded.

        Args:
            req_id: Request identifier to assign a slot for.
            dp_rank: Target DP rank for the request, or ``None`` to default to
                rank ``0``.

        Returns:
            The physical recurrent-state slot index, or ``None`` when the pool
            is disabled (neither SPMD DP nor slot-indexed state).

        Raises:
            RuntimeError: If an existing slot's inferred rank disagrees with the
                provided ``dp_rank``, if ``dp_rank`` is out of range, or if no
                free slot remains in the target rank.
        """
        if not self.is_enabled():
            return None
        existing = self.slot_by_req.get(req_id)
        rows_per_rank = self.rows_per_dp_rank()
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
        if existing is not None:
            existing_rank = min(max(int(existing), 0) // rows_per_rank, dp_size - 1)
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
        """Release a request's physical recurrent-state slot and return it for clearing.

        Pops the request's slot mapping and, under SPMD DP, returns the slot to
        its owning rank's free list (kept sorted). The returned slot index lets
        the caller schedule a state clear for the freed row.

        Args:
            req_id: Request identifier whose slot should be released.
            forget_rank: When True, also drop the request's recorded DP rank
                (used for finished requests; preemptions keep the rank).

        Returns:
            The released slot index, or ``None`` when the request held no slot.
        """
        slot = self.slot_by_req.pop(req_id, None)
        if forget_rank:
            self.dp_rank_by_req.pop(req_id, None)
        if slot is None or not self.is_enabled():
            return slot
        rows_per_rank = self.rows_per_dp_rank()
        dp_size = max(1, int(getattr(self.metadata, "data_parallel_size", 1) or 1))
        rank = min(max(int(slot), 0) // rows_per_rank, dp_size - 1)
        free_slots = self.free_slots_by_rank[rank]
        if int(slot) not in free_slots:
            free_slots.append(int(slot))
            free_slots.sort()
        return int(slot)
