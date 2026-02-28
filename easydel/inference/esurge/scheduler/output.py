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

"""Scheduler output data structures.

This module defines the data classes used to represent the output of scheduling
decisions. These structures contain all the information needed by the model
runner to execute the scheduled batch.

Classes:
    NewRequestData: Data for newly scheduled requests (first time in batch).
    CachedRequestData: Data for cached/running requests continuing execution.
    SchedulerOutput: Complete output from a scheduling decision.

Example:
    >>> # Creating scheduler output structures
    >>> new_req = NewRequestData.from_request(request, page_ids)
    >>> cached_reqs = CachedRequestData.make_empty()
    >>> output = SchedulerOutput(
    ...     scheduled_new_reqs=[new_req],
    ...     scheduled_cached_reqs=cached_reqs,
    ...     num_scheduled_tokens={"req_1": 128},
    ...     total_num_scheduled_tokens=128,
    ...     scheduled_spec_decode_tokens={},
    ...     num_common_prefix_pages=[0],
    ...     finished_req_ids=set()
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...sampling_params import SamplingParams
    from ..multimodal import MultiModalFeature
    from ..request import EngineRequest


@dataclass
class NewRequestData:
    """Data structure for new requests being scheduled for the first time.

    This class contains all the information needed to process a new request
    that is being scheduled for its first forward pass. It includes prompt
    tokens, sampling parameters, KV cache page allocations, and optional
    vision-language model data.

    Attributes:
        req_id: Unique identifier for the request.
        prompt_token_ids: List of token IDs in the prompt.
        sampling_params: Sampling parameters for token generation, or None
            if using default sampling.
        page_ids: Tuple of page ID lists, one per KV cache group. Each list
            contains the page indices allocated for this request.
        num_computed_tokens: Number of tokens already computed (from prefix
            cache hits).
        pixel_values: Pixel values for image inputs (vision models).
        image_grid_thw: Grid dimensions (temporal, height, width) for images.
        pixel_values_videos: Pixel values for video inputs.
        video_grid_thw: Grid dimensions for video frames.
        mm_features: List of multimodal features for the request.

    Example:
        >>> new_data = NewRequestData.from_request(request, page_ids)
        >>> print(f"Request {new_data.req_id} has vision: {new_data.has_vision}")
    """

    req_id: str
    """Unique identifier for the request."""

    prompt_token_ids: list[int]
    """List of token IDs in the prompt."""

    sampling_params: SamplingParams | None
    """Sampling parameters for token generation."""

    page_ids: tuple[list[int], ...]
    """Tuple of page ID lists, one per KV cache group."""

    num_computed_tokens: int
    """Number of tokens already computed from prefix cache."""

    # Vision-language model support
    pixel_values: Any | None = None
    """Pixel values for image inputs (vision models)."""

    image_grid_thw: Any | None = None
    """Grid dimensions (temporal, height, width) for images."""

    pixel_values_videos: Any | None = None
    """Pixel values for video inputs."""

    video_grid_thw: Any | None = None
    """Grid dimensions for video frames."""

    mm_features: list["MultiModalFeature"] = field(default_factory=list)
    """List of multimodal features for the request."""

    @classmethod
    def from_request(
        cls,
        request: EngineRequest,
        page_ids: tuple[list[int], ...],
    ) -> NewRequestData:
        """Create a NewRequestData instance from an EngineRequest.

        This factory method extracts all necessary data from an EngineRequest
        to create a NewRequestData instance for scheduling.

        Args:
            request: The engine request to extract data from.
            page_ids: The allocated KV cache page IDs for this request.

        Returns:
            NewRequestData: A new instance containing the request data.

        Example:
            >>> page_ids = ([0, 1, 2],)  # Single cache group
            >>> new_data = NewRequestData.from_request(request, page_ids)
        """
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
        """Check if request has vision data.

        Returns:
            bool: True if the request contains image, video, or multimodal
                features; False otherwise.
        """
        return self.pixel_values is not None or self.pixel_values_videos is not None or len(self.mm_features) > 0

    def __repr__(self) -> str:
        """Return a detailed string representation.

        Returns:
            str: String representation including all key attributes.
        """
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

    def anon_repr(self) -> str:
        """Return an anonymized string representation.

        This method returns a representation that hides the actual token IDs,
        showing only their count. Useful for logging without exposing content.

        Returns:
            str: Anonymized string representation with token count instead
                of actual token IDs.

        Example:
            >>> print(new_data.anon_repr())
            NewRequestData(req_id=...,prompt_token_ids_len=128,...)
        """
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
    """Data structure for cached/running requests in the batch.

    This class contains batched data for requests that are continuing
    execution (not new). It uses parallel lists where the i-th element
    of each list corresponds to the same request.

    Attributes:
        req_ids: List of request identifiers.
        resumed_from_preemption: List of flags indicating if each request
            was resumed from preemption.
        new_token_ids: List of token ID lists to process for each request.
        new_page_ids: List of newly allocated page ID tuples per request.
        num_computed_tokens: List of computed token counts per request.

    Example:
        >>> cached = CachedRequestData(
        ...     req_ids=["req_1", "req_2"],
        ...     resumed_from_preemption=[False, True],
        ...     new_token_ids=[[42], [100, 101]],
        ...     new_page_ids=[([5],), ([6, 7],)],
        ...     num_computed_tokens=[127, 64]
        ... )
        >>> print(f"Processing {cached.num_reqs} cached requests")
    """

    req_ids: list[str]
    """List of request identifiers."""

    resumed_from_preemption: list[bool]
    """List of flags indicating if each request was resumed from preemption."""

    new_token_ids: list[list[int]]
    """List of token ID lists to process for each request."""

    new_page_ids: list[tuple[list[int], ...]]
    """List of newly allocated page ID tuples per request."""

    num_computed_tokens: list[int]
    """List of computed token counts per request."""

    @property
    def num_reqs(self) -> int:
        """Get the number of cached requests.

        Returns:
            int: Number of requests in this cached data structure.
        """
        return len(self.req_ids)

    @classmethod
    def make_empty(cls) -> CachedRequestData:
        """Create an empty CachedRequestData instance.

        This factory method creates a CachedRequestData with empty lists,
        useful when there are no cached requests to process.

        Returns:
            CachedRequestData: An empty instance with all lists initialized
                to empty.

        Example:
            >>> empty_cached = CachedRequestData.make_empty()
            >>> assert empty_cached.num_reqs == 0
        """
        return cls(
            req_ids=[],
            resumed_from_preemption=[],
            new_token_ids=[],
            new_page_ids=[],
            num_computed_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """Complete output from a scheduling decision.

    This class contains all the information produced by a scheduler's
    ``schedule()`` method. It includes data for new and cached requests,
    token counts, speculative decoding information, and various hints
    for the model runner.

    Attributes:
        scheduled_new_reqs: List of new requests being scheduled.
        scheduled_cached_reqs: Data for cached/running requests.
        num_scheduled_tokens: Dict mapping request ID to number of tokens
            scheduled for that request.
        total_num_scheduled_tokens: Total tokens across all scheduled requests.
        scheduled_spec_decode_tokens: Dict mapping request ID to speculative
            decode token IDs for that request.
        num_common_prefix_pages: List of common prefix page counts per cache group.
        finished_req_ids: Set of request IDs that finished in the previous step.
        suggested_bucket: Suggested batch size bucket for the runner's buffer
            selection, or None if not applicable.
        async_scheduling: Whether async token sampling is enabled.

    Example:
        >>> output = scheduler.schedule()
        >>> print(f"Scheduled {output.total_num_scheduled_tokens} tokens")
        >>> print(f"New requests: {len(output.scheduled_new_reqs)}")
        >>> print(f"Cached requests: {output.scheduled_cached_reqs.num_reqs}")
    """

    scheduled_new_reqs: list[NewRequestData]
    """List of new requests being scheduled for their first forward pass."""

    scheduled_cached_reqs: CachedRequestData
    """Data for cached/running requests continuing execution."""

    num_scheduled_tokens: dict[str, int]
    """Dict mapping request ID to number of tokens scheduled."""

    total_num_scheduled_tokens: int
    """Total number of tokens scheduled across all requests."""

    scheduled_spec_decode_tokens: dict[str, list[int]]
    """Dict mapping request ID to speculative decode token IDs."""

    num_common_prefix_pages: list[int]
    """List of common prefix page counts per KV cache group."""

    finished_req_ids: set[str]
    """Set of request IDs that finished in the previous scheduling step."""

    preempted_req_ids: set[str] = field(default_factory=set)
    """Set of request IDs that were preempted (evicted from running to waiting)."""

    suggested_bucket: int | None = None
    """Optimal bucket size hint for runner's buffer selection."""

    async_scheduling: bool = False
    """Enable async token sampling for overlapped execution."""
