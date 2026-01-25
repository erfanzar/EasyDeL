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

"""Type definitions and data structures for the eSurge engine.

This module defines the core data types used throughout the eSurge engine,
including request types, event types, and output structures. Uses msgspec
for efficient serialization and deserialization.

Classes:
    FinishReason: Enumeration of reasons why generation finished.
    EngineCoreRequest: Core request structure for engine processing.
    EngineCoreEventType: Types of engine events.
    EngineCoreEvent: Timestamped engine event.
    EngineCoreOutput: Output from engine core processing.
    EngineCoreOutputs: Batch of engine outputs.
    UtilityResult: Wrapper for special serialization handling.
    UtilityOutput: Output from utility operations.
    EngineCoreRequestType: Request type identifiers for socket communication.
    ReconfigureDistributedRequest: Request to reconfigure distributed setup.
    ReconfigureRankType: Rank type for reconfiguration.

Example:
    >>> from easydel.inference.esurge.engine_types import (
    ...     EngineCoreRequest,
    ...     FinishReason,
    ...     EngineCoreOutput
    ... )
    >>>
    >>> # Create a request
    >>> request = EngineCoreRequest(
    ...     request_id="req_123",
    ...     prompt_token_ids=[1, 2, 3],
    ...     sampling_params=None,
    ...     eos_token_id=2,
    ...     arrival_time=time.time(),
    ...     data_parallel_rank=0
    ... )
    >>>
    >>> # Check finish reason
    >>> print(FinishReason.STOP)  # Output: "stop"
"""

import enum
import time
from typing import Any

import msgspec
import numpy as np

from ..sampling_params import SamplingParams
from .outputs import LogprobsLists, LogprobsTensors


class FinishReason(enum.IntEnum):
    """Reason why text generation finished.

    Uses integer values for compact serialization and efficient comparison.

    Attributes:
        STOP: A stop string or stop token was generated.
        LENGTH: Maximum tokens or model length was reached.
        ABORT: Generation was aborted for another reason.

    Example:
        >>> reason = FinishReason.STOP
        >>> print(reason)  # Output: "stop"
        >>> reason.value  # Output: 0
        >>> str(reason)  # Output: "stop"
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        """Return human-readable string representation.

        Returns:
            String name of the finish reason ('stop', 'length', or 'abort').
        """
        return ("stop", "length", "abort")[self.value]


class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Core request structure for engine processing.

    Efficient msgspec-based structure for request data, designed for
    fast serialization when passing between processes or threads.

    Attributes:
        request_id: Unique identifier for the request.
        prompt_token_ids: List of token IDs in the prompt.
        sampling_params: Parameters controlling generation behavior.
        eos_token_id: End-of-sequence token ID.
        arrival_time: Timestamp when request arrived.
        data_parallel_rank: Rank for data parallel processing.
        client_index: Index of the client making the request.
        current_wave: Current processing wave number.
        priority: Request priority for scheduling (higher = more urgent).
        pixel_values: Image pixel values for vision-language models.
        image_grid_thw: Grid shape (T, H, W) for each image.
        pixel_values_videos: Video pixel values for vision-language models.
        video_grid_thw: Grid shape (T, H, W) for each video.

    Example:
        >>> request = EngineCoreRequest(
        ...     request_id="req_123",
        ...     prompt_token_ids=[1, 2, 3, 4, 5],
        ...     sampling_params=params,
        ...     eos_token_id=2,
        ...     arrival_time=time.time(),
        ...     data_parallel_rank=0,
        ...     priority=1
        ... )
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams | None
    eos_token_id: int | None
    arrival_time: float
    data_parallel_rank: int | None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    # Vision-language model support
    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None
    pixel_values_videos: np.ndarray | None = None
    video_grid_thw: np.ndarray | None = None


class EngineCoreEventType(enum.IntEnum):
    """Types of engine core events.

    Used to track request lifecycle and performance metrics.

    Attributes:
        QUEUED: Request was added to the queue.
        SCHEDULED: Request was scheduled for processing.
        PREEMPTED: Request was preempted by higher priority work.

    Example:
        >>> event_type = EngineCoreEventType.SCHEDULED
        >>> print(event_type.value)  # Output: 2
    """

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """Timestamped engine core event.

    Records events that occur during request processing with monotonic
    timestamps for accurate interval calculation.

    Attributes:
        type: Type of the event (QUEUED, SCHEDULED, or PREEMPTED).
        timestamp: Monotonic timestamp of when event occurred.

    Note:
        Timestamps are monotonic and should only be compared within
        the same process. They are used to calculate intervals between
        events for performance monitoring.

    Example:
        >>> event = EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED)
        >>> print(f"Event type: {event.type}, time: {event.timestamp}")
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(cls, event_type: EngineCoreEventType, timestamp: float | None = None) -> "EngineCoreEvent":
        """Create a new engine event.

        Factory method to create an event with the current time if
        no timestamp is provided.

        Args:
            event_type: Type of the event.
            timestamp: Optional timestamp (uses current monotonic time if None).

        Returns:
            New EngineCoreEvent instance with the specified type and timestamp.

        Example:
            >>> event = EngineCoreEvent.new_event(EngineCoreEventType.QUEUED)
            >>> event_with_time = EngineCoreEvent.new_event(
            ...     EngineCoreEventType.SCHEDULED,
            ...     timestamp=time.monotonic()
            ... )
        """
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Output from engine core processing.

    Contains generated tokens and associated metadata for a single request.

    Attributes:
        request_id: ID of the request this output belongs to.
        new_token_ids: List of newly generated token IDs.
        new_logprobs: Log probabilities for generated tokens.
        new_prompt_logprobs_tensors: Log probabilities for prompt tokens.
        finish_reason: Reason generation finished (if finished).
        stop_reason: Specific stop string/token that triggered finish.
        events: List of events that occurred during processing.
        num_cached_tokens: Number of tokens retrieved from cache.

    Example:
        >>> output = EngineCoreOutput(
        ...     request_id="req_123",
        ...     new_token_ids=[42, 43, 44],
        ...     finish_reason=FinishReason.STOP
        ... )
        >>> print(output.finished)  # True
    """

    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    num_cached_tokens: int = 0

    @property
    def finished(self) -> bool:
        """Check if generation has finished.

        Returns:
            True if finish_reason is set, False otherwise.
        """
        return self.finish_reason is not None


class UtilityResult:
    """Wrapper for special serialization/deserialization handling.

    Provides a container for results that require custom serialization
    behavior or special handling during data transfer.

    Attributes:
        result: The wrapped result object.

    Example:
        >>> result = UtilityResult({"status": "ok", "data": [1, 2, 3]})
        >>> print(result.result)  # {"status": "ok", "data": [1, 2, 3]}
    """

    def __init__(self, r: Any = None):
        """Initialize with an optional result object.

        Args:
            r: The result object to wrap. Can be any type.
        """
        self.result = r


class UtilityOutput(msgspec.Struct, array_like=True, gc=False):
    """Output from utility operations.

    Contains the result of utility operations like cache management
    or configuration changes.

    Attributes:
        call_id: Unique identifier for the utility call.
        failure_message: Error message if the operation failed, None otherwise.
        result: The result wrapped in UtilityResult if successful.

    Example:
        >>> output = UtilityOutput(call_id=1, result=UtilityResult("success"))
        >>> if output.failure_message is None:
        ...     print("Operation succeeded")
    """

    call_id: int

    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Batch of engine outputs.

    Contains outputs from multiple requests processed in a single batch,
    along with batch-level metadata.

    Attributes:
        engine_index: Index of the engine that produced these outputs.
        outputs: List of individual request outputs.
        timestamp: Monotonic timestamp when outputs were generated.
        utility_output: Optional utility operation result.
        finished_requests: Set of request IDs that finished in this batch.
        wave_complete: Wave number that completed (for data parallel).
        start_wave: Wave number that started (for data parallel).

    Example:
        >>> outputs = EngineCoreOutputs(
        ...     engine_index=0,
        ...     outputs=[output1, output2],
        ...     finished_requests={"req_1", "req_2"}
        ... )
    """

    engine_index: int = 0
    outputs: list[EngineCoreOutput] = msgspec.field(default_factory=list)
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None

    def __post_init__(self):
        """Set timestamp to current monotonic time if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """Engine request types defined as hex byte strings.

    Used for socket communication where requests need to be identified
    by type without separate encoding. Each type is a single byte.

    Attributes:
        ADD: Add a new request to the engine.
        ABORT: Abort an existing request.
        START_DP_WAVE: Start a data parallel wave.
        UTILITY: Execute a utility operation.
        EXECUTOR_FAILED: Signal that the executor has failed.

    Example:
        >>> req_type = EngineCoreRequestType.ADD
        >>> print(req_type.value)  # b'\\x00'
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"

    EXECUTOR_FAILED = b"\x04"


class ReconfigureDistributedRequest(msgspec.Struct):
    """Request to reconfigure distributed processing setup.

    Used to dynamically change the data parallel configuration during
    runtime, such as scaling up or down the number of workers.

    Attributes:
        new_data_parallel_size: New total number of data parallel workers.
        new_data_parallel_rank: New rank for this worker.
        new_data_parallel_rank_local: New local rank within the node.
        new_data_parallel_master_ip: IP address of the master node.
        new_data_parallel_master_port: Port of the master node.

    Example:
        >>> config = ReconfigureDistributedRequest(
        ...     new_data_parallel_size=4,
        ...     new_data_parallel_rank=1,
        ...     new_data_parallel_rank_local=1,
        ...     new_data_parallel_master_ip="192.168.1.1",
        ...     new_data_parallel_master_port=29500
        ... )
    """

    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int


class ReconfigureRankType(enum.IntEnum):
    """Rank type for reconfiguring distributed request.

    Special values used during distributed reconfiguration to indicate
    whether a worker should keep its current role or shut down.

    Attributes:
        KEEP_CURRENT_RANK: Worker should keep its current rank assignment.
        SHUTDOWN_CURRENT_RANK: Worker should shut down gracefully.

    Example:
        >>> if rank_type == ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
        ...     worker.shutdown()
    """

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
