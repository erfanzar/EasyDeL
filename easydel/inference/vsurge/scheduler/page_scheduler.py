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

import queue
import threading
import time

import jax
from jax import numpy as jnp

from easydel.layers.caching import PagesCacheMetaData
from easydel.utils.helpers import get_logger

from ..utils import ActiveRequest
from ._utils import DecodeScheduleInfo, PagedScheduleResult, PrefillScheduleInfo, RequestPriority, SchedulePolicy
from .page_manager import PageManager

logger = get_logger("PageScheduler")


@jax.jit
def get_free_size(page_manager: PageManager, num_tokens):
    pages_needed = (num_tokens + page_manager.page_size - 1) // page_manager.page_size
    release_sequence_slot = jnp.sum(page_manager.page_ownership < 0)
    return release_sequence_slot >= pages_needed


class PagedScheduler:
    """
    Scheduler that integrates with PageManager for memory management
    """

    def __init__(
        self,
        metadata: PagesCacheMetaData,
        max_batch_size: int,
        max_prefill_batch_size: int,
        max_prefill_chunk_size: int = 512,
        schedule_policy: SchedulePolicy = SchedulePolicy.ADAPTIVE,
        high_load_threshold: float = 0.8,
    ):
        self.page_manager = PageManager.create(
            num_pages=metadata.num_pages,
            max_sequences=max_batch_size,
            page_size=metadata.page_size,
            pages_per_sequence=metadata.pages_per_sequence,
        )

        self.prefill_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.decode_queue: queue.Queue = queue.Queue()

        self.max_batch_size = max_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.schedule_policy = schedule_policy
        self.high_load_threshold = high_load_threshold
        self.max_prefill_chunk_size = max_prefill_chunk_size
        # Active request tracking
        self.active_requests: dict[int, ActiveRequest] = {}  # seq_id -> request
        self.request_to_seq: dict[str, int] = {}  # request.id -> seq_id
        self.seq_to_slot: dict[int, int] = {}  # seq_id -> slot
        self.in_progress_prefills: dict[str, ActiveRequest] = {}

        # Metrics
        self.total_requests = 0
        self.completed_requests = 0

        self._lock = threading.Lock()

    def add_request(self, request: ActiveRequest, priority: RequestPriority = RequestPriority.NORMAL):
        """Add a new request to the scheduler"""
        with self._lock:
            self.total_requests += 1
            self.prefill_queue.put((priority.value, time.time(), request))

    def _estimate_tokens_for_request(self, request: ActiveRequest) -> int:
        """Estimate number of tokens for a request"""
        if isinstance(request.prefill_content, str):
            return len(request.prefill_content.split()) * 2
        elif isinstance(request.prefill_content, list):
            return len(request.prefill_content)
        return 100

    def _can_allocate_pages(self, num_tokens: int) -> bool:
        """Check if we have enough free pages for the tokens"""
        return get_free_size(self.page_manager, num_tokens)

    def schedule(self) -> PagedScheduleResult:
        """Make scheduling decisions with chunked prefill support"""
        with self._lock:
            prefill_infos = []
            decode_info = None
            active_count = len(self.active_requests)
            load = active_count / self.max_batch_size

            if self.schedule_policy == SchedulePolicy.ADAPTIVE:
                effective_policy = SchedulePolicy.OFFLINE if load > self.high_load_threshold else SchedulePolicy.ONLINE
            else:
                effective_policy = self.schedule_policy

            max_tokens_this_round = min(
                self.max_prefill_chunk_size,
                self.max_prefill_batch_size * self.max_prefill_chunk_size,
            )

            all_tokens = []
            all_sequence_ids = []
            request_info = []

            for request_id, request in list(self.in_progress_prefills.items()):
                if not request.prefill_tokens_remaining:
                    del self.in_progress_prefills[request_id]
                    continue

                seq_id = request.current_seq_id
                if seq_id < 0:
                    self.log(f"Warning: Invalid seq_id for request {request_id}")
                    del self.in_progress_prefills[request_id]
                    continue
                chunk_size = min(len(request.prefill_tokens_remaining), max_tokens_this_round - len(all_tokens))

                if chunk_size > 0:
                    chunk_tokens = request.prefill_tokens_remaining[:chunk_size]
                    request.prefill_tokens_remaining = request.prefill_tokens_remaining[chunk_size:]

                    token_start = len(all_tokens)
                    all_tokens.extend(chunk_tokens)
                    all_sequence_ids.extend([seq_id] * chunk_size)
                    token_end = len(all_tokens)

                    request_info.append((seq_id, request, token_start, token_end))

                    request.prefill_tokens_processed += chunk_size
                    if not request.prefill_tokens_remaining:
                        request.is_prefill_complete = True
                        del self.in_progress_prefills[request_id]

            temp_queue = []
            remaining_token_budget = max_tokens_this_round - len(all_tokens)

            while not self.prefill_queue.empty() and remaining_token_budget > 0:
                try:
                    priority, timestamp, request = self.prefill_queue.get_nowait()

                    if request.prefill_tokens_processed is None:
                        request.prefill_tokens_processed = 0
                        request.is_prefill_complete = False
                        request.current_seq_id = -1
                        print(request.prefill_content)
                        if isinstance(request.prefill_content, str):
                            estimated_tokens = self._estimate_tokens_for_request(request)
                            request._token_ids = list(range(estimated_tokens))
                        else:
                            request._token_ids = list(request.prefill_content)

                    if self._can_allocate_pages(min(len(request._token_ids), remaining_token_budget)):
                        self.page_manager, seq_id = self.page_manager.allocate_sequence_slot()

                        if seq_id >= 0:
                            seq_id = int(seq_id)
                            request.current_seq_id = seq_id

                            chunk_size = min(len(request._token_ids), remaining_token_budget)
                            chunk_tokens = request._token_ids[:chunk_size]
                            request.prefill_tokens_remaining = request._token_ids[chunk_size:]

                            token_start = len(all_tokens)
                            all_tokens.extend(chunk_tokens)
                            all_sequence_ids.extend([seq_id] * chunk_size)
                            token_end = len(all_tokens)

                            request_info.append((seq_id, request, token_start, token_end))

                            request.prefill_tokens_processed = chunk_size
                            self.active_requests[seq_id] = request
                            self.request_to_seq[request.id] = seq_id

                            if request.prefill_tokens_remaining:
                                self.in_progress_prefills[request.id] = request
                            else:
                                request.is_prefill_complete = True

                            remaining_token_budget -= chunk_size
                        else:
                            temp_queue.append((priority, timestamp, request))
                            break
                    else:
                        temp_queue.append((priority, timestamp, request))
                        if effective_policy == SchedulePolicy.ONLINE:
                            break

                except queue.Empty:
                    break

            for item in temp_queue:
                self.prefill_queue.put(item)

            if all_tokens:
                token_sequence_ids = jnp.array(all_sequence_ids, dtype=jnp.int32)
                self.page_manager, page_batch_info = self.page_manager.allocate_pages_for_tokens(token_sequence_ids)
                for seq_id, request, token_start, token_end in request_info:
                    if seq_id in self.seq_to_slot:
                        slot = self.seq_to_slot[seq_id]
                    else:
                        slot = self._find_available_slot()
                        if slot >= 0:
                            self.seq_to_slot[seq_id] = slot
                        else:
                            self.log(f"Warning: No slot available for seq_id {seq_id}")
                            continue

                    chunk_tokens = jnp.array(all_tokens[token_start:token_end], dtype=jnp.int32)
                    prefill_info = PrefillScheduleInfo(
                        slot=slot,
                        request=request,
                        seq_id=seq_id,
                        token_ids=chunk_tokens,
                        page_batch_info=page_batch_info,
                    )
                    prefill_infos.append(prefill_info)
            active_decode_seqs = []
            active_decode_requests = {}

            for seq_id, request in self.active_requests.items():
                if (
                    request.is_prefill_complete
                    and hasattr(request, "prefill_result")
                    and request.prefill_result is not None
                    and seq_id not in [info.seq_id for info in prefill_infos]
                ):  # Don't decode if we just prefilled
                    if seq_id in self.seq_to_slot:
                        slot = self.seq_to_slot[seq_id]
                        active_decode_seqs.append(seq_id)
                        active_decode_requests[slot] = request

            # Determine if we should decode
            should_decode = len(active_decode_seqs) > 0

            if effective_policy == SchedulePolicy.OFFLINE and should_decode:
                if len(active_decode_seqs) < self.max_batch_size * 0.5:
                    should_decode = False

            # Prepare decode info
            if should_decode and active_decode_seqs:
                # For decode, each sequence generates one token
                decode_sequence_ids = jnp.array(active_decode_seqs, dtype=jnp.int32)

                # Allocate pages for decode tokens
                self.page_manager, decode_page_info = self.page_manager.allocate_pages_for_tokens(decode_sequence_ids)

                decode_info = DecodeScheduleInfo(
                    active_slots=list(active_decode_requests.keys()),
                    active_requests=active_decode_requests,
                    sequence_ids=decode_sequence_ids,
                    page_batch_info=decode_page_info,
                )

            return PagedScheduleResult(
                prefill_infos=prefill_infos,
                decode_info=decode_info,
                updated_page_manager=self.page_manager,
                should_prefill=len(prefill_infos) > 0,
                should_decode=should_decode,
            )

    def _find_available_slot(self) -> int:
        """Find an available slot index"""
        used_slots = set(self.seq_to_slot.values())
        for slot in range(self.max_batch_size):
            if slot not in used_slots:
                return slot
        return -1

    def complete_request(self, request_id: str):
        """Mark a request as completed and free resources"""
        with self._lock:
            if request_id in self.request_to_seq:
                seq_id = self.request_to_seq[request_id]
                self.page_manager = self.page_manager.release_sequence_slot(seq_id)
                if seq_id in self.active_requests:
                    del self.active_requests[seq_id]
                if seq_id in self.seq_to_slot:
                    del self.seq_to_slot[seq_id]
                del self.request_to_seq[request_id]
                if request_id in self.in_progress_prefills:
                    del self.in_progress_prefills[request_id]
                self.completed_requests += 1
