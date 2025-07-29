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
from dataclasses import dataclass
from typing import Any

from ..sampling_params import SamplingParams
from .utils import ConstantList

T = tp.TypeVar("T")


class vSurgeMetadata:
    """Tracks timing information for requests processed by vSurge.

    Attributes:
        start_time (float): The Unix timestamp (seconds) when request processing began.
    """

    def __init__(self):
        """Initializes the metadata, capturing the current time as the start time."""
        self.start_time = time.time()


@dataclass
class vSurgeRequest:
    """Represents a request for text completion within the vSurge system.

    Attributes:
        prompt (str): The input prompt for text completion.
        max_tokens (int): The maximum number of tokens to generate.
        top_p (float): Nucleus sampling probability. Defaults to 0.95.
        top_k (int): Number of highest probability tokens for top-k filtering. Defaults to 0.
        min_p (float): Minimum probability for a token to be considered. Defaults to 0.0.
        n (int): Number of independent samples to generate. Defaults to 1.
        stop (tp.Optional[tp.Union[str, tp.list[str]]]): String or list of strings to
            stop generation if encountered. Defaults to None.
        temperature (float): Sampling temperature. Defaults to 0.7.
        presence_penalty (float): Penalty for token presence. Defaults to 0.0.
        frequency_penalty (float): Penalty for token frequency. Defaults to 0.0.
        repetition_penalty (float): Penalty for repeated tokens. Defaults to 1.0.
        metadata (tp.Optional[vSurgeMetadata]): Metadata associated with the request.
            Auto-initialized if None.
        is_client_side_tokenization (bool): If True, prompt is tokenized and client expects
            token IDs. Defaults to False.
    """

    prompt: str
    sampling_params: SamplingParams
    metadata: vSurgeMetadata | None = None
    is_client_side_tokenization: bool = False

    @classmethod
    def from_sampling_params(cls, prompt: str, sampling_params: SamplingParams) -> vSurgeRequest:
        """Creates a vSurgeRequest from a prompt and SamplingParams.

        Args:
            prompt (str): The input prompt string.
            sampling_params (SamplingParams): An object containing sampling parameters.

        Returns:
            vSurgeRequest: A new vSurgeRequest instance.
        """
        return vSurgeRequest(prompt=prompt, sampling_params=sampling_params)

    def __post_init__(self):
        """Ensures metadata is initialized and validates prompt type."""
        if self.metadata is None:
            self.metadata = vSurgeMetadata()
        assert isinstance(self.prompt, str), "prompt should be a single string"


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
    def is_finished(status: EngineRequestStatus) -> bool:
        return status > EngineRequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: EngineRequestStatus) -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


_FINISHED_REASON_MAP = {
    EngineRequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    EngineRequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    EngineRequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    EngineRequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}


class EngineRequest:
    """
    Represents a request to the engine, including prompt, sampling parameters,
    and tracking of output and status.
    """

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        cache_salt: str | None = None,
        priority: int = 0,
    ) -> None:
        self.request_id: str = request_id
        self.client_index: int = client_index
        self.priority: int = priority
        self.sampling_params: Any = sampling_params
        self.eos_token_id: int | None = eos_token_id
        self.arrival_time: float = arrival_time if arrival_time is not None else time.time()
        self.status: EngineRequestStatus = (
            EngineRequestStatus.WAITING_FOR_FSM
            if sampling_params and getattr(sampling_params, "guided_decoding", None) is not None
            else EngineRequestStatus.WAITING
        )
        self.stop_reason: int | str | None = None
        self.kv_transfer_params: dict[str, Any] | None = None

        assert getattr(sampling_params, "max_tokens", None) is not None, "max_tokens must be set in sampling_params"
        self.max_tokens: int = sampling_params.max_tokens

        if getattr(sampling_params, "extra_args", None) is not None:
            self.kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")

        self.prompt_token_ids: list[int] = prompt_token_ids
        self.num_prompt_tokens: int = len(prompt_token_ids)
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = prompt_token_ids.copy()
        self.num_output_placeholders: int = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens: int = 0
        self.cache_salt: str | None = cache_salt

        self.output_token_ids: ConstantList = ConstantList(self._output_token_ids)
        self.all_token_ids: ConstantList = ConstantList(self._all_token_ids)
        self.num_cached_tokens: int = -1
        self.num_nans_in_logits: int = 0

    def append_output_token_ids(self, token_ids: int | list[int]) -> None:
        """Append one or more output token ids to the request."""
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

    @property
    def is_output_corrupted(self) -> bool:
        """Returns True if any NaNs were found in logits."""
        return self.num_nans_in_logits > 0

    @property
    def num_tokens(self) -> int:
        """Total number of tokens (prompt + output)."""
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        """Total number of tokens including spec tokens."""
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """Number of output tokens generated."""
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        """Returns True if the request is finished."""
        return EngineRequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        """Returns the reason the request finished, if any."""
        return EngineRequestStatus.get_finished_reason(self.status)
