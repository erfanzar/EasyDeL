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

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from easydel.layers.caching.page.cache import PagesCacheMetaData
from easydel.utils.helpers import get_logger

from ..cache_manager import KVCacheManager
from ..request_type import EngineRequest, EngineRequestStatus
from ..utils import EngineCoreOutput, EngineCoreOutputs, ModelRunnerOutput
from ._utils import check_stop
from .output import ScheduledCacheRequestData, ScheduledNewRequestData, SchedulerOutput
from .queue_types import SchedulingPolicy, create_request_queue
from .scheduler_config import SchedulerConfig
from .scheduler_interface import SchedulerInterface

logger = get_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        metadata: PagesCacheMetaData,
        scheduler_config: SchedulerConfig,
        enable_caching: bool = True,
        num_kv_cache_groups: int = 1,
        include_finished_set: bool = False,
    ) -> None:
        self.scheduler_config = scheduler_config

        self.finished_req_ids_dict: dict[int, set[str]] | None = defaultdict(set) if include_finished_set else None

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        self.requests: dict[str, EngineRequest] = {}

        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(f"Unknown scheduling policy: {self.scheduler_config.policy}")

        self.waiting = create_request_queue(self.policy)
        self.running: list[EngineRequest] = []
        self.finished_req_ids: set[str] = set()
        self.finished_recving_kv_req_ids: set[str] = set()
        self.num_kv_cache_groups = num_kv_cache_groups
        self.kv_cache_manager = KVCacheManager(
            metadata=metadata,
            enable_caching=enable_caching,
            use_eagle=False,
            num_kv_cache_groups=num_kv_cache_groups,
        )

    def schedule(self) -> SchedulerOutput:
        scheduled_new_reqs: list[EngineRequest] = []
        scheduled_resumed_reqs: list[EngineRequest] = []
        scheduled_running_reqs: list[EngineRequest] = []
        preempted_reqs: list[EngineRequest] = []
        req_to_new_page_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens

            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)

            if num_new_tokens == 0:
                req_index += 1
                continue

            while True:
                new_pages = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
                if new_pages is None:
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = EngineRequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_pages is not None

            scheduled_running_reqs.append(request)
            req_to_new_page_ids[request.request_id] = new_pages.get_page_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        skipped_waiting_requests = create_request_queue(self.policy)

        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                if request.status == EngineRequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = EngineRequestStatus.WAITING
                    else:
                        logger.debug("%s is still in WAITING_FOR_REMOTE_KVS state.", request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if request.status == EngineRequestStatus.WAITING_FOR_FSM:
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    new_computed_pages, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_pages(request)

                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens

                else:
                    new_computed_pages = self.kv_cache_manager.create_empty_page_list()
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0

                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = self.scheduler_config.long_prefill_token_threshold

                    if not self.scheduler_config.chunked_prefill_enabled and num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                new_pages = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_pages,
                    num_lookahead_tokens=0,
                    delay_cache_pages=load_kv_async,
                )
                if new_pages is None:
                    break

                request = self.waiting.pop_request()
                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = EngineRequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                req_index += 1
                self.running.append(request)
                if request.status == EngineRequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == EngineRequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                req_to_new_page_ids[request.request_id] = self.kv_cache_manager.get_page_ids(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = EngineRequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        num_common_prefix_pages = [0] * self.num_kv_cache_groups
        if self.running:
            any_request = self.running[0]
            num_common_prefix_pages = self.kv_cache_manager.get_num_common_prefix_pages(any_request, len(self.running))

        new_reqs_data = [
            ScheduledNewRequestData.from_request(req, req_to_new_page_ids[req.request_id]) for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            req_to_new_page_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            num_common_prefix_pages=num_common_prefix_pages,
            finished_req_ids=self.finished_req_ids,
        )

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[EngineRequest],
        resumed_reqs: list[EngineRequest],
        num_scheduled_tokens: dict[str, int],
        req_to_new_page_ids: dict[str, tuple[list[int], ...]],
    ) -> ScheduledCacheRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_page_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            new_page_ids.append(req_to_new_page_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)

        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return ScheduledCacheRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_page_ids=new_page_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        stopped_running_reqs: set[EngineRequest] = set()
        stopped_preempted_reqs: set[EngineRequest] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == EngineRequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if request.sampling_params is not None and request.sampling_params.logprobs is not None and logprobs:
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or kv_transfer_params:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        finish_reason=request.get_finished_reason(),
                        stop_reason=request.stop_reason,
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
            else:
                assert not prompt_logprobs_tensors

        if stopped_running_reqs:
            self.running = [req for req in self.running if req not in stopped_running_reqs]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

        self._update_from_kv_xfer_finished(model_runner_output)

        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            for client_index, finished_set in finished_req_ids.items():
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: EngineRequest,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]
                break
        return new_token_ids, stopped

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: EngineRequest) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: EngineRequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert EngineRequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue

            valid_requests.append(request)
            if request.status == EngineRequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: EngineRequest) -> dict[str, Any] | None:
        assert request.is_finished()
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)
        self._free_pages(request)
        return None

    def _free_pages(self, request: EngineRequest):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_page_hashes(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def shutdown(self) -> None: ...

    def _update_from_kv_xfer_finished(self, model_runner_output: ModelRunnerOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the pages

            scheduler the request during the next step.
        """
        for req_id in model_runner_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in model_runner_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_pages(self.requests[req_id])
