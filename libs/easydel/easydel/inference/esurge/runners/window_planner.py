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

"""Compile-bucket math and scheduler-window row planning for the eSurge runner.

Owns the token/request padding ladders that decide which compiled executable a
step maps to, and the window row collection that compacts interior zero-token
gaps before bucket selection. All logic is host-side and allocation-light; the
runner keeps its public ``num_tokens_paddings`` / ``max_num_seq_buckets``
attributes (read by the scheduler and tests) and delegates the math here.
"""

from __future__ import annotations

import typing

import numpy as np
from eformer.loggings import get_logger

logger = get_logger("eSurge")


class WindowPlanner:
    """Bucket ladders and schedulable-window row planning.

    The constructor captures the static bucket parameters computed at runner
    initialization; per-step lookups take the live runtime state (active
    bucket list, sequence-buffer row ids) as explicit arguments.

    Attributes:
        num_tokens_paddings: Token compile-bucket ladder (ascending).
        max_num_seq_buckets: Configured request-count compile buckets.
    """

    def __init__(
        self,
        *,
        num_tokens_paddings: list[int],
        max_num_seq_buckets: list[int],
    ) -> None:
        """Capture the static bucket ladders.

        Args:
            num_tokens_paddings: Token compile-bucket sizes (ascending).
            max_num_seq_buckets: Request-count compile bucket sizes.
        """
        self.num_tokens_paddings = list(num_tokens_paddings)
        self.max_num_seq_buckets = list(max_num_seq_buckets)

    @staticmethod
    def get_token_paddings(min_token_size: int, max_token_size: int, padding_gap: int) -> list[int]:
        """Generate padding sizes for efficient compilation.

        Args:
            min_token_size: Minimum token size (must be power of 2)
            max_token_size: Maximum token size to cover
            padding_gap: Gap between padding sizes (0 for exponential growth)

        Returns:
            List of padding sizes
        """
        if not ((min_token_size & (min_token_size - 1) == 0) and min_token_size > 0):
            logger.error(f"Invalid min_token_size={min_token_size}, must be power of 2")
            raise ValueError(f"min_token_size must be a power of 2, got {min_token_size}")
        paddings = []
        num = min_token_size

        if padding_gap == 0:
            while num <= max_token_size:
                paddings.append(num)
                num *= 2
        else:
            while num <= padding_gap:
                paddings.append(num)
                num *= 2
            num //= 2
            while num < max_token_size:
                num += padding_gap
                paddings.append(num)
        if paddings[-1] != max_token_size:
            paddings.append(max_token_size)
        return paddings

    @staticmethod
    def get_request_paddings(min_bucket: int, max_bucket: int) -> list[int]:
        """Generate request count buckets using exponential growth.

        Args:
            min_bucket: Minimum bucket size.
            max_bucket: Maximum bucket size (must be included).

        Returns:
            List of bucket sizes from min_bucket to max_bucket,
            doubling at each step.
        """
        min_bucket = max(1, min(min_bucket, max_bucket))
        buckets: list[int] = []
        current = min_bucket
        while current < max_bucket:
            buckets.append(current)
            current *= 2
        if not buckets or buckets[-1] != max_bucket:
            buckets.append(max_bucket)
        return buckets

    @staticmethod
    def init_seq_buckets(
        user_buckets: list[int] | None,
        max_num_seqs: int,
        min_input_pad: int,
    ) -> list[int]:
        """Initialize sequence count buckets for compilation.

        Args:
            user_buckets: Optional user-provided compile bucket sizes. Values
                may exceed ``max_num_seqs`` so the compiled static request
                width can match another runtime's padding policy while the
                scheduler still caps active concurrency at ``max_num_seqs``.
            max_num_seqs: Maximum number of concurrently running sequences.
            min_input_pad: Minimum input padding.

        Returns:
            Sorted list of request compile buckets. Without explicit buckets,
            derives the usual padding ladder from ``min_input_pad`` to
            ``max_num_seqs``. With explicit buckets, preserves valid positive
            buckets and ensures at least one bucket can hold ``max_num_seqs``.
        """
        if user_buckets:
            buckets = sorted({int(b) for b in user_buckets if int(b) > 0})
        else:
            buckets = WindowPlanner.get_request_paddings(min_input_pad, max_num_seqs)
        if not buckets or buckets[-1] < max_num_seqs:
            buckets.append(max_num_seqs)
        return buckets

    def get_current_bucket(self, num_reqs: int, active_buckets: list[int] | None = None) -> int:
        """Select the smallest bucket that can accommodate num_reqs.

        Args:
            num_reqs: Number of active requests
            active_buckets: Runtime-clamped bucket list, or ``None`` to fall
                back to the configured ``max_num_seq_buckets``.

        Returns:
            Smallest sufficient bucket size from the active runtime buckets.
        """
        buckets = active_buckets if active_buckets is not None else self.max_num_seq_buckets
        if num_reqs <= 0:
            return buckets[0]
        for bucket in buckets:
            if num_reqs <= bucket:
                return bucket
        return buckets[-1]

    @staticmethod
    def clamp_request_buckets_to_runtime_cap(buckets: list[int], runtime_cap: int) -> list[int]:
        """Clamp request-count buckets to the runtime execution cap.

        The runner may admit more requests globally than it can execute in a
        single scheduler window.  Compilation and bucket lookup should
        therefore only consider request-count buckets that are reachable
        under the current runtime window cap.

        Args:
            buckets: Original list of request-count bucket sizes.
            runtime_cap: Maximum number of requests executable in one
                scheduler window.

        Returns:
            Sorted list of bucket sizes where every entry is at most
            ``runtime_cap``, with ``runtime_cap`` itself always included
            as the final element.
        """
        runtime_cap = max(1, int(runtime_cap))
        clamped = sorted({int(bucket) for bucket in buckets if 0 < int(bucket) <= runtime_cap})
        if not clamped or clamped[-1] != runtime_cap:
            clamped.append(runtime_cap)
        return clamped

    @staticmethod
    def collect_schedulable_window_rows(
        *,
        req_ids: list[str | None],
        start_index: int,
        stop_index: int,
        scheduled_tokens_by_req: dict[str, int],
        allow_sparse_packing: bool,
    ) -> tuple[np.ndarray, list[str | None], list[int], int, bool]:
        """Collect runnable rows for a window, compacting interior zero-token gaps.

        The scheduler keeps some RUNNING requests resident even when they
        receive zero tokens in the current step. When such rows appear in the
        middle of a window, the execution key can become `(few tokens, many
        requests)`, which is not a real batch shape. This helper preserves the
        common contiguous-prefix fast path and only packs rows when interior
        zero-token gaps are present.

        Args:
            req_ids: Sequence-buffer row-ordered request ids (``None`` = hole).
            start_index: First buffer row of the window (inclusive).
            stop_index: Last buffer row of the window (exclusive).
            scheduled_tokens_by_req: Scheduled token count per request id.
            allow_sparse_packing: Whether interior zero-token rows may be
                compacted out of the window.

        Returns:
            Tuple of ``(row_indices, req_ids_window, scheduled_list,
            next_start_index, packed)``.
        """
        start_index = max(0, int(start_index))
        stop_index = max(start_index, int(stop_index))

        window_req_ids: list[str | None] = []
        window_scheduled: list[int] = []
        last_positive_offset = -1

        for global_row_index in range(start_index, stop_index):
            rid = req_ids[global_row_index]
            scheduled = int(scheduled_tokens_by_req.get(rid, 0)) if rid is not None else 0
            window_req_ids.append(rid)
            window_scheduled.append(scheduled)
            if rid is not None and scheduled > 0:
                last_positive_offset = global_row_index - start_index

        if last_positive_offset < 0:
            return np.empty((0,), dtype=np.int32), [], [], stop_index, False

        prefix_stop = last_positive_offset + 1
        has_interior_zero_rows = any(
            rid is None or scheduled <= 0
            for rid, scheduled in zip(window_req_ids[:prefix_stop], window_scheduled[:prefix_stop], strict=False)
        )

        if not has_interior_zero_rows:
            row_indices = np.arange(start_index, start_index + prefix_stop, dtype=np.int32)
            req_ids_window: list[str | None] = [typing.cast(str, rid) for rid in window_req_ids[:prefix_stop]]
            scheduled_list = [int(scheduled) for scheduled in window_scheduled[:prefix_stop]]
            return row_indices, req_ids_window, scheduled_list, start_index + prefix_stop, False

        if not allow_sparse_packing:
            row_indices = np.arange(start_index, start_index + prefix_stop, dtype=np.int32)
            req_ids_window = [typing.cast(str | None, rid) for rid in window_req_ids[:prefix_stop]]
            scheduled_list = [int(scheduled) for scheduled in window_scheduled[:prefix_stop]]
            return row_indices, req_ids_window, scheduled_list, start_index + prefix_stop, False

        row_indices_list: list[int] = []
        req_ids_window = []
        scheduled_list = []
        for offset in range(prefix_stop):
            rid = window_req_ids[offset]
            scheduled = int(window_scheduled[offset])
            if rid is None or scheduled <= 0:
                continue
            row_indices_list.append(start_index + offset)
            req_ids_window.append(rid)
            scheduled_list.append(scheduled)
        return (
            np.asarray(row_indices_list, dtype=np.int32),
            req_ids_window,
            scheduled_list,
            start_index + prefix_stop,
            True,
        )
