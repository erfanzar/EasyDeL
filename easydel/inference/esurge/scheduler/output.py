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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...sampling_params import SamplingParams
    from ..multimodal import MultiModalFeature
    from ..request import EngineRequest


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams | None
    page_ids: tuple[list[int], ...]
    num_computed_tokens: int
    # Vision-language model support
    pixel_values: Any | None = None
    image_grid_thw: Any | None = None
    pixel_values_videos: Any | None = None
    video_grid_thw: Any | None = None
    mm_features: list["MultiModalFeature"] = field(default_factory=list)

    @classmethod
    def from_request(
        cls,
        request: EngineRequest,
        page_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            page_ids=page_ids,
            num_computed_tokens=request.num_computed_tokens,
            # Vision-language model data
            pixel_values=request.pixel_values,
            image_grid_thw=request.image_grid_thw,
            pixel_values_videos=request.pixel_values_videos,
            video_grid_thw=request.video_grid_thw,
            mm_features=request.mm_features,
        )

    @property
    def has_vision(self) -> bool:
        """Check if request has vision data."""
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    def __repr__(self):
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"sampling_params={self.sampling_params},"
            f"page_ids={self.page_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"has_vision={self.has_vision}"
            ")"
        )

    def anon_repr(self):
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={len(self.prompt_token_ids)},"
            f"sampling_params={self.sampling_params},"
            f"page_ids={self.page_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"has_vision={self.has_vision}"
            ")"
        )


@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_from_preemption: list[bool]
    new_token_ids: list[list[int]]
    new_page_ids: list[tuple[list[int], ...]]
    num_computed_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_page_ids=[],
            num_computed_tokens=[],
        )


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    num_common_prefix_pages: list[int]
    finished_req_ids: set[str]
    suggested_bucket: int | None = None  # Optimal bucket size hint for runner
    async_scheduling: bool = False  # Enable async token sampling
