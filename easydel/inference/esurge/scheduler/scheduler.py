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
import typing
from collections import defaultdict
from collections.abc import Iterable

from ..config import Config
from ..core.interface import CacheGroupsConfig
from ..core.manager import CacheManager
from ..engine_types import EngineCoreOutput, EngineCoreOutputs
from ..metrics import get_metrics_collector
from ..outputs import ModelRunnerOutput
from ..request import EngineRequest, EngineRequestStatus
from .interface import SchedulerInterface
from .output import CachedRequestData, NewRequestData, SchedulerOutput
from .request_queue import SchedulingPolicy, create_request_queue
from .utils import check_stop

if typing.TYPE_CHECKING:
    from ..runners.model_runner import eSurgeRunner


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        config: Config,
        kv_cache_config: CacheGroupsConfig,
        include_finished_set: bool = False,
    ) -> None:
        self.config = config
        self.scheduler_config = config.scheduler_config
        self.cache_config = config.cache_config
        self.kv_cache_config = kv_cache_config

        self.finished_req_ids_dict: dict[int, set[str]] | None = defaultdict(set) if include_finished_set else None

        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        num_pages = self.cache_config.num_pages
        assert num_pages is not None and num_pages > 0

        self.page_size = self.cache_config.page_size

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

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0

        self.kv_cache_manager = CacheManager(
            num_pages=num_pages,
            kv_cache_groups=kv_cache_config.kv_cache_groups,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
        )

    @classmethod
    def from_runner(
        cls,
        runner: eSurgeRunner,
        max_num_batched_tokens: int | None = None,
        enable_prefix_caching: bool = True,
    ) -> Scheduler:
        """Create a Scheduler instance from an eSurgeRunner."""
        from ..config import CacheConfig, SchedulerConfig
        from ..core.interface import CacheGroupSpec, FullAttentionSpec

        metadata = runner.metadata

        if max_num_batched_tokens is None:
            max_num_batched_tokens = runner.max_model_len
        return Scheduler(
            config=Config(
                scheduler_config=SchedulerConfig(
                    max_num_seqs=runner.max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_model_len=runner.max_model_len,
                ),
                cache_config=CacheConfig(
                    num_pages=metadata.num_pages,
                    page_size=metadata.page_size,
                    enable_prefix_caching=enable_prefix_caching,
                ),
            ),
            kv_cache_config=CacheGroupsConfig(
                num_pages=metadata.num_pages,
                kv_cache_groups=[
                    CacheGroupSpec(
                        FullAttentionSpec(
                            page_size=metadata.page_size,
                            num_kv_heads=metadata.num_kv_heads,
                            head_size=metadata.k_headdim,
                            dtype=runner.executor_manager.kv_pages.views[-1].kv_pages.dtype,
                            use_mla=False,
                        )
                    )
                ],
            ),
        )

    def schedule(self) -> SchedulerOutput:
        import time

        schedule_start_time = time.time()
        scheduled_new_reqs: list[EngineRequest] = []
        scheduled_resumed_reqs: list[EngineRequest] = []
        scheduled_running_reqs: list[EngineRequest] = []
        preempted_reqs: list[EngineRequest] = []

        req_to_new_page_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

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
                new_pages = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens
                )
                if new_pages is None:
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
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

            if request.spec_token_ids:
                num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = request.spec_token_ids

        skipped_waiting_requests = create_request_queue(self.policy)

        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                if request.status == EngineRequestStatus.WAITING_FOR_REMOTE_KVS:
                    request.status = EngineRequestStatus.WAITING

                if request.status == EngineRequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = EngineRequestStatus.WAITING
                    else:
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
                    num_lookahead_tokens=self.num_lookahead_tokens,
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

        num_common_prefix_pages = [0] * len(self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_pages = self.kv_cache_manager.get_num_common_prefix_pages(any_request, len(self.running))
        new_reqs_data = [
            NewRequestData.from_request(req, req_to_new_page_ids[req.request_id]) for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_page_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_common_prefix_pages=num_common_prefix_pages,
            finished_req_ids=self.finished_req_ids,
        )

        self._update_after_schedule(scheduler_output)

        # Log scheduler metrics
        schedule_time = time.time() - schedule_start_time
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_scheduler_metrics(
                num_waiting=len(self.waiting),
                num_running=len(self.running),
                num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
                num_preempted=len(preempted_reqs),
                batch_size=len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs),
                schedule_time=schedule_time,
            )

            # Log cache metrics
            cache_manager = self.kv_cache_manager
            total_pages = cache_manager.num_pages
            used_pages = total_pages - cache_manager.page_pool.get_num_free_pages()

            num_cached_pages = len(cache_manager.page_pool.cached_page_hash_to_page)
            cache_hit_rate = num_cached_pages / max(total_pages, 1)

            metrics_collector.record_cache_metrics(
                total_pages=total_pages,
                used_pages=used_pages,
                cache_hit_rate=cache_hit_rate,
            )

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
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_page_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_page_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(spec_decode_tokens.get(req_id, ()))
            token_ids = req.all_token_ids[req.num_computed_tokens : req.num_computed_tokens + num_tokens]
            new_token_ids.append(token_ids)

            new_page_ids.append(req_to_new_page_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
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
            new_token_ids = generated_token_ids
            status_before_stop = request.status

            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
            if stopped:
                self._free_request(request)
                if status_before_stop == EngineRequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )
            assert not prompt_logprobs_tensors

        if stopped_running_reqs:
            self.running = [req for req in self.running if req not in stopped_running_reqs]
        if stopped_preempted_reqs:
            self.waiting.remove_requests(stopped_preempted_reqs)

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

    def _free_request(self, request: EngineRequest):
        assert request.is_finished()

        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        self._free_pages(request)

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
