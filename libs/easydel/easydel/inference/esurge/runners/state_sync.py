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

"""Scheduler-to-runner sequence-state synchronization for eSurge.

Owns the per-step reconciliation between the scheduler's decisions and the
runner's host-side request state: request lifecycle in the
:class:`SequenceBuffer` (add/remove/preempt/condense with DP-local row
placement), decode-first row reordering, and the async-scheduling placeholder
repair paths that materialize previously sampled tokens back into the buffer.

The runner constructs one :class:`SequenceStateSync` per buffer generation
(inside ``_setup_variables``) so the injected sequence buffer, requests dict,
slot pool, and spec-commit scratch stay in lockstep with the runner's own
handles.
"""

from __future__ import annotations

import typing

import numpy as np
from eformer.loggings import get_logger

from ..core.dp_sharding import dp_shard_for_page_id, pages_per_dp_shard
from .states import CachedRequestState

if typing.TYPE_CHECKING:
    from ..scheduler import SchedulerOutput
    from .async_types import AsyncPreResults
    from .sequence_buffer import SequenceBuffer
    from .slot_pool import RecurrentSlotPool

logger = get_logger("eSurge")


class SequenceStateSync:
    """Synchronizes runner request/sequence state with scheduler output.

    Attributes:
        sequence_buffer: Live row-backed request state the model step reads.
        requests: Live mapping of request id to :class:`CachedRequestState`.
        slot_pool: Recurrent-state slot pool (assign/release + DP-rank map).
        metadata: Paged-attention cache metadata (DP size, page counts).
        executor_manager: Execution manager; only ``clear_recurrent_slots``
            is called from here.
        max_model_len: Maximum sequence length (token-position guard).
        spec_decode_recurrent_candidates: Whether speculative recurrent
            candidate commits are active (spec scratch must be cleared for
            freed rows).
        spec_recurrent_commit_cpu: Host scratch of pending spec recurrent
            commits, cleared for removed rows.
        pending_spec_recurrent_commit_by_req: Pending spec-commit token counts
            by request id, dropped on finish/preempt.
    """

    def __init__(
        self,
        *,
        sequence_buffer: SequenceBuffer,
        requests: dict[str, CachedRequestState],
        slot_pool: RecurrentSlotPool,
        metadata: typing.Any,
        executor_manager: typing.Any,
        max_model_len: int,
        spec_decode_recurrent_candidates: bool,
        spec_recurrent_commit_cpu: np.ndarray,
        pending_spec_recurrent_commit_by_req: dict[str, int],
    ) -> None:
        """Capture the live state handles this synchronizer operates on.

        Args:
            sequence_buffer: Row-backed request state buffer.
            requests: Request-id to cached-request-state mapping.
            slot_pool: Recurrent slot pool for slot/rank bookkeeping.
            metadata: Cache metadata carrying DP size and page counts.
            executor_manager: Manager exposing ``clear_recurrent_slots``.
            max_model_len: Maximum sequence length.
            spec_decode_recurrent_candidates: Whether spec recurrent-candidate
                commits are enabled.
            spec_recurrent_commit_cpu: Pending spec-commit host scratch.
            pending_spec_recurrent_commit_by_req: Pending spec-commit counts.
        """
        self.sequence_buffer = sequence_buffer
        self.requests = requests
        self.slot_pool = slot_pool
        self.metadata = metadata
        self.executor_manager = executor_manager
        self.max_model_len = int(max_model_len)
        self.spec_decode_recurrent_candidates = bool(spec_decode_recurrent_candidates)
        self.spec_recurrent_commit_cpu = spec_recurrent_commit_cpu
        self.pending_spec_recurrent_commit_by_req = pending_spec_recurrent_commit_by_req

    def update_states(self, scheduler_output: SchedulerOutput) -> bool:
        """Update internal states based on scheduler output.

        Synchronizes the runner's internal state with the scheduler's decisions.
        Handles request lifecycle: adding new requests, removing finished ones,
        updating cached requests, and managing the sequence buffer.

        State Updates:
            1. Remove finished requests from tracking
            2. Remove unscheduled requests from buffer
            3. Add new requests with their metadata
            4. Update cached request states
            5. Reorganize sequence buffer for efficiency

        Args:
            scheduler_output: Contains request scheduling decisions including:
                - finished_req_ids: Requests that completed
                - scheduled_new_reqs: New requests to add
                - scheduled_cached_reqs: Existing requests to update
                - num_scheduled_tokens: Tokens to generate per request

        Returns:
            True if state changed (requests added/removed), indicating
            potential buffer reorganization. False if no changes occurred.

        Side Effects:
            - Updates the ``requests`` dictionary
            - Modifies sequence buffer contents
            - May trigger buffer condensation

        Note:
            This method is called at the beginning of each execution cycle
            to ensure the runner's state matches the scheduler's decisions.
        """
        dp_size = int(getattr(self.metadata, "data_parallel_size", 1) or 1)
        pages_per_shard_opt = pages_per_dp_shard(int(getattr(self.metadata, "num_pages", 0) or 0), dp_size)
        use_dp_local_rows = (
            dp_size > 1
            and int(self.sequence_buffer.max_num_reqs) > 0
            and int(self.sequence_buffer.max_num_reqs) % dp_size == 0
            and pages_per_shard_opt is not None
        )
        rows_per_shard = int(self.sequence_buffer.max_num_reqs) // dp_size if use_dp_local_rows else 0
        pages_per_shard = int(pages_per_shard_opt or 0) if use_dp_local_rows else 0

        def infer_req_shard(page_ids: tuple[list[int], ...]) -> int | None:
            """Infer the DP shard index that owns a request based on its page IDs.

            Examines page IDs across all cache groups and returns the shard
            index if all non-null pages belong to the same shard. Returns
            None if DP-local rows are disabled or pages span multiple shards.
            """
            if not use_dp_local_rows or pages_per_shard <= 0:
                return None
            inferred: int | None = None
            for group_ids in page_ids:
                for pid in group_ids:
                    # 0 is reserved for null/padding page in page pool.
                    if int(pid) <= 0:
                        continue
                    shard = dp_shard_for_page_id(int(pid), pages_per_shard, dp_size)
                    if shard is None:
                        continue
                    if inferred is None:
                        inferred = shard
                    elif inferred != shard:
                        return None
            return inferred

        for req_id, dp_rank in getattr(scheduler_output, "req_id_to_dp_rank", {}).items():
            self.slot_pool.dp_rank_by_req[str(req_id)] = int(dp_rank)

        removed_recurrent_slots: list[int] = []
        for req_id in sorted(scheduler_output.finished_req_ids):
            slot = self.slot_pool.release_slot(str(req_id), forget_rank=True)
            if slot is not None:
                removed_recurrent_slots.append(int(slot))
            self.requests.pop(req_id, None)
            self.pending_spec_recurrent_commit_by_req.pop(str(req_id), None)

        # 2) Remove finished from sequence buffer (functional)
        removed_req_indices: list[int] = []
        removed_req_index_by_id: dict[str, int] = {}
        for req_id in sorted(scheduler_output.finished_req_ids):
            req_index = self.sequence_buffer.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # 3) Remove preempted requests from buffer.
        # Only remove requests the scheduler explicitly preempted (evicted from
        # running to waiting). Running requests that were merely skipped due to
        # token budget exhaustion still hold valid rows and pages — removing them
        # would force re-insertion next cycle and trigger "No free sequence row
        # in target DP shard" errors when shard rows are full.
        for req_id in sorted(scheduler_output.preempted_req_ids):
            req_index = self.sequence_buffer.remove_request(req_id)
            self.pending_spec_recurrent_commit_by_req.pop(str(req_id), None)
            slot = self.slot_pool.release_slot(str(req_id), forget_rank=False)
            if slot is not None:
                removed_recurrent_slots.append(int(slot))
            if req_index is not None:
                removed_req_indices.append(req_index)
                removed_req_index_by_id[req_id] = req_index

        # 3b) Clear recurrent/SSM state for freed slots so the next request
        # assigned to the same slot starts from a clean state. Slot-pooled
        # runners clear physical slots; row-identity runners clear buffer rows.
        slots_to_clear = removed_recurrent_slots if self.slot_pool.is_enabled() else removed_req_indices
        if slots_to_clear:
            self.executor_manager.clear_recurrent_slots(slots_to_clear)
            if self.spec_decode_recurrent_candidates:
                for req_index in removed_req_indices:
                    if 0 <= int(req_index) < int(self.spec_recurrent_commit_cpu.shape[1]):
                        self.spec_recurrent_commit_cpu[:, int(req_index)] = 0

        # 4) Add new requests to tracking
        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            if new_req_data.sampling_params is None:
                raise ValueError("Pooling not supported in TPU")
            if self.slot_pool.uses_spmd_dp() and new_req_data.has_vision:
                raise ValueError("Rank-major SPMD DP currently supports text-only requests.")
            req_id = new_req_data.req_id
            if new_req_data.dp_rank is not None:
                self.slot_pool.dp_rank_by_req[str(req_id)] = int(new_req_data.dp_rank)
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                page_ids=new_req_data.page_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                # Vision-language model data
                pixel_values=new_req_data.pixel_values,
                image_grid_thw=new_req_data.image_grid_thw,
                pixel_values_videos=new_req_data.pixel_values_videos,
                video_grid_thw=new_req_data.video_grid_thw,
                mm_features=new_req_data.mm_features,
            )
            req_ids_to_add.append(req_id)

        # 5) Update cached requests and page tables
        req_data = scheduler_output.scheduled_cached_reqs
        upd_req_indices: list[int] = []
        upd_num_computed_vals: list[int] = []
        batched_page_rows: list[tuple[int, tuple[list[int], ...]]] = []

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            if i < len(req_data.dp_ranks) and req_data.dp_ranks[i] is not None:
                self.slot_pool.dp_rank_by_req[str(req_id)] = int(req_data.dp_ranks[i])

            nct = req_data.num_computed_tokens[i]
            new_page_ids = req_data.new_page_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            req_state.num_computed_tokens = nct
            if not resumed_from_preemption:
                for page_ids, new_ids in zip(req_state.page_ids, new_page_ids, strict=False):
                    page_ids.extend(new_ids)
            else:
                req_state.page_ids = new_page_ids

            req_index = self.sequence_buffer.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue
            if self.slot_pool.uses_spmd_dp():
                self.slot_pool.assign_slot(req_id, self.slot_pool.dp_rank_by_req.get(str(req_id)))

            upd_req_indices.append(req_index)
            upd_num_computed_vals.append(int(nct))
            if resumed_from_preemption:
                # Resumed requests may provide a full replacement page table.
                self.sequence_buffer.page_table.add_row(new_page_ids, req_index)
            else:
                if any(len(ids) for ids in new_page_ids):
                    batched_page_rows.append((req_index, new_page_ids))

        if upd_req_indices:
            # num_computed_tokens is now a NumPy array, use standard indexing
            idx_arr = np.array(upd_req_indices, dtype=np.int32)
            val_arr = np.array(upd_num_computed_vals, dtype=np.int32)
            self.sequence_buffer.num_computed_tokens[idx_arr] = val_arr

        if batched_page_rows:
            indices = [ix for ix, _ in batched_page_rows]
            pages_per_req = [ids for _, ids in batched_page_rows]
            self.sequence_buffer.page_table.append_rows_batch(pages_per_req, indices)

        # 6) Add new / reinserted requests
        # Prefer stable index reuse (same request index when possible), and under
        # DP-local page sharding, try to keep request rows in the shard range that
        # matches their current page IDs.
        removed_pool = set(removed_req_indices)

        def _find_reuse_index_in_shard(shard_idx: int) -> int | None:
            """Locate a row index inside ``shard_idx`` safe to reuse for a new request.

            Under DP-local page sharding, each request must occupy a row whose
            shard matches the shard that owns its allocated KV pages. This
            helper first prefers rows being vacated this step (``removed_pool``
            membership) at the highest available index, then falls back to any
            currently empty slot inside the shard. Returns ``None`` when DP-local
            row enforcement is disabled or no candidate exists in the shard.
            """
            if not use_dp_local_rows:
                return None
            lo = int(shard_idx) * rows_per_shard
            hi = lo + rows_per_shard

            shard_removed = [ix for ix in removed_pool if lo <= int(ix) < hi]
            if shard_removed:
                return max(shard_removed)

            req_slots = self.sequence_buffer.req_ids
            for ix in range(lo, hi):
                if ix >= len(req_slots) or req_slots[ix] is None:
                    return ix
            return None

        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            reuse_index = removed_req_index_by_id.pop(req_id, None)
            if reuse_index is not None and reuse_index not in removed_pool:
                reuse_index = None

            target_shard = self.slot_pool.dp_rank_by_req.get(str(req_id))
            if target_shard is None:
                target_shard = infer_req_shard(req_state.page_ids)
            if target_shard is not None:
                self.slot_pool.dp_rank_by_req[str(req_id)] = int(target_shard)
            if target_shard is not None and use_dp_local_rows:
                lo = target_shard * rows_per_shard
                hi = lo + rows_per_shard

                if reuse_index is not None and not (lo <= int(reuse_index) < hi):
                    logger.warning(
                        "Dropping out-of-shard row reuse for req %s: reuse_index=%s target_shard=%s range=[%s,%s).",
                        req_id,
                        reuse_index,
                        target_shard,
                        lo,
                        hi,
                    )
                    reuse_index = None

                if reuse_index is None:
                    reuse_index = _find_reuse_index_in_shard(target_shard)

                # Preserve DP-local block-table invariants: a request that already
                # owns shard-local pages must never be inserted into a different
                # row shard. If no row is available in this shard, surface a hard
                # error so scheduler/accounting can be fixed.
                if reuse_index is None:
                    raise RuntimeError(
                        "No free sequence row available in target DP shard for request insertion. "
                        f"req_id={req_id} shard={target_shard} rows_per_shard={rows_per_shard} "
                        f"removed_pool_size={len(removed_pool)}."
                    )

            if reuse_index is None and removed_pool:
                reuse_index = max(removed_pool)

            if reuse_index is not None:
                removed_pool.discard(reuse_index)
            if self.slot_pool.uses_spmd_dp():
                self.slot_pool.assign_slot(req_id, target_shard)
            self.sequence_buffer.add_request(req_state, reuse_index)

        if removed_pool and not use_dp_local_rows:
            self.sequence_buffer.condense(sorted(removed_pool))

        # Drop cached VLM prompt helpers once prefill is complete to free host RAM.
        for req_state in self.requests.values():
            if (
                req_state.prefill_inputs_embeds is not None
                and req_state.num_computed_tokens >= req_state.num_prompt_tokens
            ):
                req_state.prefill_inputs_embeds = None
                req_state.prefill_position_ids = None
                req_state.prefill_visual_pos_masks = None
                req_state.prefill_deepstack_visual_embeds = None

        has_changes = len(scheduler_output.preempted_req_ids) > 0 or len(req_ids_to_add) > 0
        return has_changes

    def modify_prev_results(self, pre_results: AsyncPreResults | None) -> None:
        """Apply previous iteration's tokens to sequence buffer.

        This method is called at the beginning of each iteration when async
        scheduling is enabled. It retrieves the tokens that were sampled
        asynchronously in the previous iteration and applies them to the
        sequence buffer.

        The method blocks until the async token transfer is complete, then
        updates the token_ids array and request output_token_ids lists.

        Args:
            pre_results: Deferred sampled-token payload from the previous
                overlap window, or ``None`` for a no-op.
        """
        if pre_results is None:
            return

        pre_windows = pre_results.windows
        pre_request_seq_lens = pre_results.request_seq_lens

        valid_sampled_token_ids: list[np.ndarray] = []
        for window in pre_windows:
            next_tokens_cpu = np.asarray(window.sampled_token_ids)
            for row_pos, is_valid in zip(window.row_positions, window.valid_mask, strict=False):
                if not is_valid:
                    valid_sampled_token_ids.append(np.array([], dtype=np.int32))
                    continue
                valid_sampled_token_ids.append(np.array([int(next_tokens_cpu[row_pos])], dtype=np.int32))

        for pre_req_idx, _, req_state, placeholder_idx in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if len(sampled_ids) == 0:
                continue

            req_id = req_state.req_id
            if req_id not in self.sequence_buffer.req_id_to_index or req_id not in self.requests:
                continue

            req_idx = self.sequence_buffer.req_id_to_index[req_id]
            if req_state is not self.requests[req_id]:
                raise RuntimeError("Request state mismatch")

            start_idx = int(placeholder_idx)
            end_idx = start_idx + 1
            if start_idx < 0 or end_idx > self.max_model_len:
                raise ValueError(f"Token position {start_idx} exceeds max_model_len {self.max_model_len}")

            self.sequence_buffer.token_ids[req_idx, start_idx:end_idx] = sampled_ids
            req_state.output_token_ids[-1] = int(sampled_ids[-1])

    @staticmethod
    def finalize_async_runner_state(
        sampled_token_ids: list[list[int]],
        *,
        request_seq_lens: list[tuple[int, int, CachedRequestState, int]],
        sequence_buffer: typing.Any,
        requests: dict[str, CachedRequestState],
        max_model_len: int,
    ) -> None:
        """Repair runner-side placeholders after async scheduler output drains.

        AsyncScheduler inserts optimistic placeholders before the runner's
        sampled token is host-visible. For PP decode we may launch the next
        step from the previous device token directly, so the CPU repair should
        happen when the previous async handle is drained, not inside the next
        launch path. This finalizer consumes the tokens already materialized by
        :class:`_AsyncExecutionHandle` and replaces the placeholder in both the
        sequence buffer and the request's public output-token list.

        Static so the runner delegator can pass its live buffer/request handles
        explicitly (the payload-identity bookkeeping for
        ``expected_pre_results`` stays on the runner).

        Args:
            sampled_token_ids: Per-output-index sampled-token lists.
            request_seq_lens: ``(out_idx, row_idx, req_state, placeholder_idx)``
                tuples recorded when the placeholders were installed.
            sequence_buffer: Live sequence buffer to repair.
            requests: Live request-id to cached-request-state mapping.
            max_model_len: Maximum sequence length (token-position guard).

        Raises:
            RuntimeError: If a tracked request state no longer matches the
                registered request.
            ValueError: If a placeholder position exceeds ``max_model_len``.
        """
        for pre_req_idx, _, req_state, placeholder_idx in request_seq_lens:
            if pre_req_idx >= len(sampled_token_ids):
                continue
            sampled_ids = sampled_token_ids[pre_req_idx]
            if len(sampled_ids) == 0:
                continue

            req_id = req_state.req_id
            if req_id not in sequence_buffer.req_id_to_index or req_id not in requests:
                continue

            req_idx = sequence_buffer.req_id_to_index[req_id]
            if req_state is not requests[req_id]:
                raise RuntimeError("Request state mismatch")

            start_idx = int(placeholder_idx)
            end_idx = start_idx + 1
            if start_idx < 0 or end_idx > max_model_len:
                raise ValueError(f"Token position {start_idx} exceeds max_model_len {max_model_len}")

            tid = int(sampled_ids[-1])
            sequence_buffer.token_ids[req_idx, start_idx:end_idx] = np.asarray([tid], dtype=np.int32)
            output_pos = start_idx - int(req_state.num_prompt_tokens)
            if 0 <= output_pos < len(req_state.output_token_ids):
                req_state.output_token_ids[output_pos] = tid
            else:
                req_state.output_token_ids[-1] = tid

    def reorder_decode_first(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder active requests so decode requests are placed first.

        Partitions the request buffer so all decode requests (single token,
        with computed tokens > 0) appear before prefill requests. This ordering
        matches the TPU runner behavior and enables optimized v3 attention
        with request distribution.

        Args:
            scheduler_output: Used to determine scheduled tokens per request.

        Side Effects:
            - Modifies sequence_buffer ordering via swap_states().
        """
        i, j = 0, self.sequence_buffer.num_reqs - 1
        while i < j:
            i_req_id = self.sequence_buffer.req_ids[i]
            j_req_id = self.sequence_buffer.req_ids[j]
            if i_req_id is None or j_req_id is None:
                break

            i_is_decode = (
                scheduler_output.num_scheduled_tokens.get(i_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[i] > 0
            )
            j_is_decode = (
                scheduler_output.num_scheduled_tokens.get(j_req_id, 0) == 1
                and self.sequence_buffer.num_computed_tokens[j] > 0
            )

            if i_is_decode:
                i += 1
            elif not j_is_decode:
                j -= 1
            else:
                # Swap to move a decode request forward.
                self.sequence_buffer.swap_states(i, j)
                i += 1
                j -= 1

    def reorder_decode_first_per_shard(
        self,
        scheduler_output: SchedulerOutput,
        dp_size: int,
    ) -> None:
        """Reorder decode requests first within each DP shard's row range.

        Unlike reorder_decode_first which reorders across the entire buffer
        (and would move requests across shard boundaries), this method
        reorders decode-first independently within each shard's contiguous
        row range: [shard * rows_per_shard, (shard+1) * rows_per_shard).

        This preserves DP-local row placement while giving each shard's
        rows the decode-first ordering that the v3 attention kernel expects.

        Args:
            scheduler_output: Used to determine scheduled tokens per request.
            dp_size: Number of data-parallel shards.
        """
        # Use max_num_reqs (not num_slots) for shard boundaries to match
        # update_states and the validation in batch_preparer, which both
        # partition rows based on the fixed max_num_reqs capacity.
        max_reqs = self.sequence_buffer.max_num_reqs
        if max_reqs <= 1 or dp_size <= 1:
            return
        rows_per_shard = max_reqs // dp_size
        if rows_per_shard <= 1 or max_reqs % dp_size != 0:
            return

        num_slots = self.sequence_buffer.num_slots
        for shard in range(dp_size):
            lo = shard * rows_per_shard
            hi = min(lo + rows_per_shard, num_slots)

            # 1) Compact holes (None slots) to the end of the shard range.
            #    This ensures the attention kernel never encounters a 0-token
            #    row in the middle of its processing range.
            self.sequence_buffer.compact_holes_in_range(lo, hi)

            # 2) Decode-first partitioning on the compacted (hole-free) prefix.
            #    Find the boundary between non-None rows and holes.
            shard_end = hi
            while shard_end > lo and (
                shard_end - 1 >= len(self.sequence_buffer.req_ids) or self.sequence_buffer.req_ids[shard_end - 1] is None
            ):
                shard_end -= 1

            i, j = lo, shard_end - 1
            while i < j:
                i_req_id = self.sequence_buffer.req_ids[i]
                j_req_id = self.sequence_buffer.req_ids[j]

                # Guard against empty slots that survived compaction
                # (e.g. when prompt count < max_num_seqs).
                if i_req_id is None:
                    break  # no more populated slots from the left
                if j_req_id is None:
                    j -= 1
                    continue

                i_is_decode = (
                    scheduler_output.num_scheduled_tokens.get(i_req_id, 0) == 1
                    and self.sequence_buffer.num_computed_tokens[i] > 0
                )
                j_is_decode = (
                    scheduler_output.num_scheduled_tokens.get(j_req_id, 0) == 1
                    and self.sequence_buffer.num_computed_tokens[j] > 0
                )

                if i_is_decode:
                    i += 1
                elif not j_is_decode:
                    j -= 1
                else:
                    self.sequence_buffer.swap_states(i, j)
                    i += 1
                    j -= 1
