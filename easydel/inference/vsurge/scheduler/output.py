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

from eformer.pytree import auto_pytree

from ...sampling_params import SamplingParams
from ..request_type import EngineRequest


@auto_pytree
class ScheduledNewRequestData:
    req_id: str | int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams | None
    page_ids: tuple[list[int], ...]
    num_computed_tokens: int

    @classmethod
    def from_request(
        cls,
        request: EngineRequest,
        page_ids: tuple[list[int], ...],
    ) -> ScheduledNewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            page_ids=page_ids,
            num_computed_tokens=request.num_computed_tokens,
        )

    def __repr__(self):
        return (
            f"ScheduledNewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"sampling_params={self.sampling_params},"
            f"page_ids={self.page_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            ")"
        )


@auto_pytree
class ScheduledCacheRequestData:
    req_ids: list[str]
    resumed_from_preemption: list[bool]
    new_token_ids: list[list[int]]
    new_page_ids: list[tuple[list[int], ...]]
    num_computed_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> ScheduledCacheRequestData:
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_page_ids=[],
            num_computed_tokens=[],
        )


@auto_pytree
class SchedulerOutput:
    scheduled_new_reqs: list[ScheduledNewRequestData]
    scheduled_cached_reqs: ScheduledCacheRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    num_common_prefix_pages: list[int]
    finished_req_ids: set[str]
