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

import enum
import time
from typing import Any

import msgspec

from ..sampling_params import SamplingParams
from .outputs import LogprobsLists, LogprobsTensors


class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, or abort.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted for another reason

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2

    def __str__(self):
        return ("stop", "length", "abort")[self.value]


class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
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
    """The type of engine core request event."""

    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3


class EngineCoreEvent(msgspec.Struct):
    """A timestamped engine core event associated with a request.

    The timestamp is a monotonic timestamps and is used for by the engine
    frontend to calculate intervals between engine core events. These
    timestamps should not be compared with timestamps from other processes.
    """

    type: EngineCoreEventType
    timestamp: float

    @classmethod
    def new_event(cls, event_type: EngineCoreEventType, timestamp: float | None = None) -> "EngineCoreEvent":
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
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
        return self.finish_reason is not None


class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

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
