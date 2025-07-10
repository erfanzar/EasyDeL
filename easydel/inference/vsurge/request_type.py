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

import enum
import time
import typing as tp

import msgspec

from ..sampling_params import SamplingParams
from .utils.utils import ConstantList, LogprobsLists, LogprobsTensors, SchedulerStats, UtilityOutput

T = tp.TypeVar("T")


class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: list[EngineCoreEvent] | None = None
    kv_transfer_params: dict[str, tp.Any] | None = None
    num_cached_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    engine_index: int = 0
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: SchedulerStats | None = None
    timestamp: float = 0.0
    utility_output: UtilityOutput | None = None
    finished_requests: set[str] | None = None
    wave_complete: int | None = None
    start_wave: int | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


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
    def new_event(cls, event_type: EngineCoreEventType, timestamp: float | None = None) -> EngineCoreEvent:
        timestamp = time.monotonic() if timestamp is None else timestamp
        return cls(event_type, timestamp)


class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    eos_token_id: int | None
    arrival_time: float
    cache_salt: str | None
    data_parallel_rank: int | None
    client_index: int = 0
    current_wave: int = 0


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


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_id: int | None,
        client_index: int = 0,
        cache_salt: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.sampling_params = sampling_params
        self.eos_token_id = eos_token_id

        self.status = (
            RequestStatus.WAITING_FOR_FSM if sampling_params.guided_decoding is not None else RequestStatus.WAITING
        )
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        self.has_encoder_inputs = self.num_encoder_inputs > 0
        kv_params = None if sampling_params.extra_args is None else sampling_params.extra_args.get("kv_transfer_params")
        self.kv_transfer_params: dict[str, tp.Any] | None = kv_params
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        self.num_cached_tokens = -1

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> Request:
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            cache_salt=request.cache_salt,
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
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> str | None:
        return RequestStatus.get_finished_reason(self.status)

    @property
    def use_structured_output(self) -> bool:
        return self.sampling_params.guided_decoding is not None

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


class RequestStatus(enum.IntEnum):
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

    @staticmethod
    def is_finished(status: RequestStatus) -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: RequestStatus) -> str | None:
        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
