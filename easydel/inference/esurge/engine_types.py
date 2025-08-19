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
including request types, event types, and output structures.

Classes:
    FinishReason: Enumeration of reasons why generation finished
    EngineCoreRequest: Core request structure for engine processing
    EngineCoreEventType: Types of engine events
    EngineCoreEvent: Timestamped engine event
    EngineCoreOutput: Output from engine core processing
    UtilityResult: Wrapper for special serialization handling
"""

import enum
import time
from typing import Any

import msgspec

from ..sampling_params import SamplingParams
from .outputs import LogprobsLists, LogprobsTensors


class FinishReason(enum.IntEnum):
    """Reason why text generation finished.

    Uses integer values for compact serialization.

    Attributes:
        STOP: A stop string/token was generated.
        LENGTH: Maximum tokens or model length was reached.
        ABORT: Generation was aborted for another reason.

    Example:
        >>> reason = FinishReason.STOP
        >>> print(reason)  # Output: "stop"
        >>> reason.value  # Output: 0
    """

    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return ("stop", "length", "abort")[self.value]


class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Core request structure for engine processing.

    Efficient msgspec-based structure for request data.

    Attributes:
        request_id: Unique identifier for the request.
        prompt_token_ids: List of token IDs in the prompt.
        sampling_params: Parameters controlling generation behavior.
        eos_token_id: End-of-sequence token ID.
        arrival_time: Timestamp when request arrived.
        data_parallel_rank: Rank for data parallel processing.
        client_index: Index of the client making the request.
        current_wave: Current processing wave number.
        priority: Request priority for scheduling.
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


class EngineCoreEventType(enum.IntEnum):
    """Types of engine core events.

    Attributes:
        QUEUED: Request was added to the queue.
        SCHEDULED: Request was scheduled for processing.
        PREEMPTED: Request was preempted by higher priority work.
    """

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """Timestamped engine core event.

    Records events that occur during request processing with monotonic
    timestamps for accurate interval calculation.

    Attributes:
        type: Type of the event.
        timestamp: Monotonic timestamp of when event occurred.

    Note:
        Timestamps are monotonic and should only be compared within
        the same process. They are used to calculate intervals between
        events for performance monitoring.
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(cls, event_type: EngineCoreEventType, timestamp: float | None = None) -> "EngineCoreEvent":
        """Create a new engine event.

        Args:
            event_type: Type of the event.
            timestamp: Optional timestamp (uses current time if None).

        Returns:
            New EngineCoreEvent instance.
        """
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    """Output from engine core processing.

    Contains generated tokens and associated metadata.

    Attributes:
        request_id: ID of the request this output belongs to.
        new_token_ids: List of newly generated token IDs.
        new_logprobs: Log probabilities for generated tokens.
        new_prompt_logprobs_tensors: Log probabilities for prompt tokens.
        finish_reason: Reason generation finished (if finished).
        stop_reason: Specific stop string/token that triggered finish.
        events: List of events that occurred during processing.
        num_cached_tokens: Number of tokens retrieved from cache.
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
        r: The wrapped result object.
    """

    def __init__(self, r: Any = None):
        self.result = r


class UtilityOutput(msgspec.Struct, array_like=True, gc=False):
    call_id: int

    failure_message: str | None = None
    result: UtilityResult | None = None


class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class EngineCoreRequestType(enum.Enum):
    """
    EngineRequest types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"

    EXECUTOR_FAILED = b"\x04"


class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int


class ReconfigureRankType(enum.IntEnum):
    """
    Rank type for reconfiguring distributed request.
    """

    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
