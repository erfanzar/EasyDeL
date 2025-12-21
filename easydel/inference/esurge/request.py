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

"""Request management for the eSurge engine.

Defines the core request structures and status tracking for managing
inference requests throughout their lifecycle.

Classes:
    EngineRequest: Main request object for tracking generation
    EngineRequestStatus: Enum of request statuses

Example:
    >>> request = EngineRequest(
    ...     request_id="req_123",
    ...     prompt_token_ids=[1, 2, 3],
    ...     sampling_params=params,
    ...     eos_token_id=2
    ... )
    >>> request.status = EngineRequestStatus.RUNNING
"""

import enum
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..sampling_params import SamplingParams
from .engine_types import EngineCoreEvent, EngineCoreEventType, EngineCoreRequest, FinishReason
from .utils import ConstantList

if TYPE_CHECKING:
    from .multimodal import MultiModalFeature


class EngineRequest:
    """Request object for tracking generation through the engine.

    Manages the state and metadata of a single inference request,
    including tokens, sampling parameters, and execution status.

    Attributes:
        request_id: Unique identifier for the request.
        prompt_token_ids: Input token IDs.
        sampling_params: Parameters controlling generation.
        eos_token_id: End-of-sequence token ID.
        client_index: Index of the client making request.
        arrival_time: Timestamp when request arrived.
        priority: Request priority for scheduling.
        parent_request_id: ID of parent request for n>1 sampling (None for n=1).
        sample_index: Index of this sample (0 to n-1) for n>1 sampling.
        status: Current request status.
        events: List of events during processing.
        stop_reason: Reason for stopping generation.
        pixel_values: Image pixel values for vision-language models.
        image_grid_thw: Grid shape (T, H, W) for each image.
        pixel_values_videos: Video pixel values for vision-language models.
        video_grid_thw: Grid shape (T, H, W) for each video.
        mm_features: List of multimodal features with metadata for caching.

    Example:
        >>> request = EngineRequest(
        ...     request_id="req_123",
        ...     prompt_token_ids=[1, 2, 3],
        ...     sampling_params=sampling_params,
        ...     eos_token_id=2
        ... )
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        priority: int = 0,
        parent_request_id: str | None = None,
        sample_index: int = 0,
        # Vision-language model support
        pixel_values: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        pixel_values_videos: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
        mm_features: list["MultiModalFeature"] | None = None,
    ) -> None:
        """Initialize EngineRequest.

        Args:
            request_id: Unique request identifier.
            prompt_token_ids: Input token IDs.
            sampling_params: Generation parameters.
            eos_token_id: End-of-sequence token.
            client_index: Client index.
            arrival_time: Request arrival time.
            priority: Request priority.
            parent_request_id: Parent request ID for n>1 sampling.
            sample_index: Sample index (0 to n-1) for n>1 sampling.
            pixel_values: Image pixel values for VLMs.
            image_grid_thw: Grid shape (T, H, W) for each image.
            pixel_values_videos: Video pixel values for VLMs.
            video_grid_thw: Grid shape (T, H, W) for each video.
            mm_features: List of multimodal features with metadata.
        """
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.parent_request_id = parent_request_id
        self.sample_index = sample_index
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = EngineRequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        self.kv_transfer_params: dict[str, Any] | None = None
        self.structured_output_request = getattr(sampling_params, "guided_decoding", None) if sampling_params else None
        self.use_structured_output = self.structured_output_request is not None

        if sampling_params is not None:
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.num_output_placeholders = 0
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        self.num_cached_tokens = -1

        self.num_nans_in_logits = 0

        # Vision-language model data
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.pixel_values_videos = pixel_values_videos
        self.video_grid_thw = video_grid_thw
        self.mm_features: list["MultiModalFeature"] = mm_features or []
        self._vision_processed = False

    @property
    def has_vision(self) -> bool:
        """Check if request has vision data (images or videos)."""
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    def clear_vision_data(self) -> None:
        """Clear raw vision data after prefill to free memory.

        Note: mm_features with cached_embeddings are preserved for multi-turn.
        """
        self.pixel_values = None
        self.image_grid_thw = None
        self.pixel_values_videos = None
        self.video_grid_thw = None
        # Clear pixel values from features but preserve cached embeddings
        for feat in self.mm_features:
            feat.clear_pixel_values()
        self._vision_processed = True

    @property
    def vision_processed(self) -> bool:
        """Check if vision data has been processed (prefill complete)."""
        return self._vision_processed

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "EngineRequest":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            priority=request.priority,
            # Vision-language model data
            pixel_values=request.pixel_values,
            image_grid_thw=request.image_grid_thw,
            pixel_values_videos=request.pixel_values_videos,
            video_grid_thw=request.video_grid_thw,
            mm_features=getattr(request, "mm_features", None),
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

    @property
    def is_output_corrupted(self) -> bool:
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return EngineRequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return EngineRequestStatus.get_finished_reason(self.status)

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class EngineRequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()

    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "EngineRequestStatus") -> bool:
        return status > EngineRequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "EngineRequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    EngineRequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    EngineRequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    EngineRequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    EngineRequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
