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

import copy
import enum
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from collections.abc import Callable, Mapping
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Literal, NotRequired, TypedDict

import jax
import msgspec
from jax import numpy as jnp

from ..sampling_params import RequestOutputKind, SamplingParams

INVALID_TOKEN_ID = -1

ARRAY_TYPE = "l"


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    prompt_token_ids: list[int]
    token_type_ids: NotRequired[list[int]]
    prompt: NotRequired[str]


def array_full(token_id: int, count: int):
    """[`array`][] equivalent of [numpy.full][]."""
    return array("l", [token_id]) * count


@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


PromptLogprobs = list[dict[int, Logprob] | None]
SampleLogprobs = list[dict[int, Logprob]]


class SequenceStatus(enum.IntEnum):
    """Status of a sequence."""

    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "SequenceStatus") -> bool:
        return status > SequenceStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "SequenceStatus") -> str | None:
        if status == SequenceStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == SequenceStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == SequenceStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SequenceStage(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()


@dataclass
class RequestMetrics:
    arrival_time: float
    last_token_time: float
    first_scheduled_time: float | None
    first_token_time: float | None
    time_in_queue: float | None
    finished_time: float | None = None
    scheduler_time: float | None = None
    model_forward_time: float | None = None
    model_execute_time: float | None = None
    spec_token_acceptance_counts: list[int] | None = None


class SequenceDataDelta(msgspec.Struct, array_like=True, omit_defaults=True):
    """Delta SequenceData to send to workers per step."""

    new_output_token_ids: list[int]
    new_cumulative_logprob: float
    new_num_computed_tokens: int
    new_stage: SequenceStage


class SequenceData(msgspec.Struct, omit_defaults=True):
    """Data associated with a sequence.

    Args:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output. Set to an empty list if
            None.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """

    _prompt_token_ids: array
    _output_token_ids: array = msgspec.field(default_factory=lambda: array(ARRAY_TYPE, []))

    _prompt_embeds: jax.Array | None = None
    _output_embeds: jax.Array | None = None

    _cumulative_logprob: float = 0.0
    _prompt_token_ids_tuple: tuple[int, ...] = msgspec.field(default_factory=tuple)
    _num_computed_tokens: int = 0
    _num_cached_tokens: int = 0
    _stage: SequenceStage = SequenceStage.PREFILL
    _cached_all_token_ids: list[int] = msgspec.field(default_factory=list)
    _cached_all_token_embeds: jax.Array | None = None

    _new_appended_tokens: list[int] = msgspec.field(default_factory=list)
    _mrope_position_delta: int | None = None

    @staticmethod
    def from_prompt_token_counts(*token_counts: tuple[int, int]) -> "SequenceData":
        """
        Construct a `SequenceData`instance
        by concatenating prompt token sequences.

        Each tuple represents one token sequence, expressed in the form
        `(token_id, count)`.
        """
        if len(token_counts) == 0:
            return SequenceData.from_seqs([])

        prompt_token_ids_arr = reduce(
            array.__iadd__,
            (array_full(token_id, count) for token_id, count in token_counts),
        )

        return SequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(
        prompt_token_ids: GenericSequence[int],
        output_token_ids: GenericSequence[int] | None = None,
        *,
        prompt_embeds: jax.Array | None = None,
    ) -> "SequenceData":
        """
        Construct a `SequenceData` instance from prompt and output token sequences.
        """
        prompt_token_ids_arr = array(ARRAY_TYPE, prompt_token_ids)

        if output_token_ids is None:
            return SequenceData(prompt_token_ids_arr, _prompt_embeds=prompt_embeds)

        output_token_ids_arr = array(ARRAY_TYPE, output_token_ids)

        return SequenceData(prompt_token_ids_arr, _output_token_ids=output_token_ids_arr, _prompt_embeds=prompt_embeds)

    def __post_init__(self) -> None:
        assert self._prompt_token_ids.typecode == "l"
        assert self._output_token_ids.typecode == "l"
        self._prompt_token_ids_tuple: tuple[int, ...] = tuple(self._prompt_token_ids)
        self._update_cached_all_tokens()
        if self._prompt_embeds is not None:
            self._update_cached_all_token_embeds()

    def _update_cached_all_tokens(self):
        assert isinstance(self._prompt_token_ids, array)
        assert isinstance(self._output_token_ids, array)
        self._cached_all_token_ids: list[int] = list(self._prompt_token_ids + self._output_token_ids)

    def _update_cached_all_token_embeds(self):
        assert isinstance(self._prompt_embeds, jax.Array)
        self._cached_all_token_embeds: jax.Array = self._prompt_embeds
        if self._output_embeds is not None:
            self._cached_all_token_embeds = jnp.concatenate(
                (self._cached_all_token_embeds, self._output_embeds),
                axis=0,
            )

    @property
    def cumulative_logprob(self) -> float:
        return self._cumulative_logprob

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        raise NotImplementedError

    @property
    def prompt_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self, new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = array(ARRAY_TYPE, new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_embeds(self) -> jax.Array | None:
        return self._output_embeds

    @output_embeds.setter
    def output_embeds(self, new_output_token_embeds: jax.Array) -> None:
        self._output_token_embeds = new_output_token_embeds
        self._update_cached_all_token_embeds()

    @property
    def output_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        assert isinstance(self._output_token_ids, array)
        return self._output_token_ids

    @property
    def prompt_embeds(self) -> jax.Array | None:
        return self._prompt_embeds

    @prompt_embeds.setter
    def prompt_embeds(self, prompt_embeds: jax.Array) -> None:
        self._prompt_embeds = prompt_embeds
        self._update_cached_all_token_embeds()

    @property
    def mrope_position_delta(self) -> int | None:
        return self._mrope_position_delta

    @mrope_position_delta.setter
    def mrope_position_delta(self, new_mrope_position_delta):
        self._mrope_position_delta = new_mrope_position_delta

    def append_token_id(self, token_id: int, logprob: float, token_embed: jax.Array | None = None) -> None:
        self._output_token_ids.append(token_id)
        self._new_appended_tokens.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self._cumulative_logprob += logprob
        if token_embed is not None:
            assert token_embed.ndim == 1
            token_embed = jnp.expand_dims(token_embed, 0)
            if self._output_embeds is None:
                self._output_embeds = token_embed
            else:
                self._output_embeds = jnp.concatenate(
                    [self._output_embeds, token_embed],
                    axis=0,
                )
            assert self._cached_all_token_embeds is not None
            self._cached_all_token_embeds = jnp.concatenate(
                [self._cached_all_token_embeds, token_embed],
                axis=0,
            )

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> list[int]:
        return self._cached_all_token_ids

    def get_token_embeddings(self) -> jax.Array | None:
        return self._cached_all_token_embeds

    def get_prefix_token_ids(self, num_tokens: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Get prefix tokens, and make the return value hashable"""
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple, tuple(self._output_token_ids[: num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (self._num_computed_tokens, self.get_len())
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def get_num_cached_tokens(self) -> int:
        """Return the number of tokens with prefix cache hit."""
        return self._num_cached_tokens

    def update_num_cached_tokens(self, num_cached_tokens: int):
        """Update the number of tokens with prefix cache hit."""
        self._num_cached_tokens = num_cached_tokens

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL
        self._new_appended_tokens = []

    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> tuple[int, ...]:
        return self.output_token_ids

    def get_delta_and_reset(self) -> SequenceDataDelta:
        delta = SequenceDataDelta(
            self._new_appended_tokens, self._cumulative_logprob, self.get_num_computed_tokens(), self.stage
        )
        # Reset delta state.
        self._new_appended_tokens = []
        return delta

    def apply_delta(self, delta: SequenceDataDelta):
        self._num_computed_tokens = delta.new_num_computed_tokens
        self._cumulative_logprob = delta.new_cumulative_logprob
        self._stage = delta.new_stage
        self._output_token_ids.extend(delta.new_output_token_ids)
        self._cached_all_token_ids.extend(delta.new_output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (
            f"SequenceData("
            f"prompt_token_ids={self._prompt_token_ids}, "
            f"prompt_embeds.shape="
            f"{getattr(self._prompt_embeds, 'shape', None)}, "
            f"output_token_ids={self.output_token_ids}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"get_num_computed_tokens={self.get_num_computed_tokens()})"
        )


class Sequence:
    """Stores the data, status, and block information of a sequence."""

    def __init__(
        self,
        seq_id: int,
        inputs: TokenInputs,
        block_size: int,
        eos_token_id: int | None = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = inputs
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.data = SequenceData.from_seqs(self.prompt_token_ids, prompt_embeds=None)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""
        self.status = SequenceStatus.WAITING
        self.stop_reason: int | str | None = None
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0
        self.prefix_offset = 0
        self.read_offset = 0
        self.tokens: list[str] | None = None

    @property
    def n_blocks(self) -> int:
        return (self.get_len() + self.block_size - 1) // self.block_size

    @property
    def prompt(self) -> str | None:
        if self.inputs["type"] == "embeds":
            return None
        return self.inputs.get("prompt")

    @property
    def prompt_token_ids(self) -> list[int]:
        if self.inputs["type"] == "embeds":
            return [0] * len(self.inputs["prompt_embeds"])
        return self.inputs["prompt_token_ids"]

    @property
    def token_type_ids(self) -> list[int]:
        if self.inputs["type"] == "embeds":
            return []
        return self.inputs.get("token_type_ids", [])

    @property
    def lora_int_id(self) -> int:
        return 0

    @property
    def prompt_adapter_id(self) -> int:
        return 0

    def get_output_text_to_return(self, buffer_length: int, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        truncate = buffer_length and not self.is_finished()
        if not delta:
            return self.output_text[:-buffer_length] if truncate else (self.output_text)
        length = len(self.output_text)
        if truncate:
            length -= buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""

    def get_output_token_ids_to_return(self, delta: bool) -> GenericSequence[int] | int:
        """If delta is True, only new tokens since the last call to
        this method are returned"""
        if not delta:
            return self.get_output_token_ids()

        output_len = self.get_output_len()
        num_new_tokens = output_len - self._last_output_token_ids_offset
        self._last_output_token_ids_offset = output_len

        if num_new_tokens == 1:
            return self.data._cached_all_token_ids[-1]

        if num_new_tokens == 0:
            return []

        return self.data._cached_all_token_ids[-num_new_tokens:]

    def hash_of_block(self, logical_idx: int) -> int:
        num_tokens = self.num_hashed_tokens_of_block(logical_idx)
        hashed_tokens = self.data.get_prefix_token_ids(num_tokens)
        return hash((hashed_tokens, self.lora_int_id))

    def extra_hash(self) -> int | None:
        """
        This function computes an extra hash for a sequence, specifically
        designed for prefix caching mode. The final sequence hash is determined
        by applying token_ids from the sequence's blocks.
        """
        if self.prompt_adapter_id == 0 and self.lora_int_id == 0:
            return None

        return hash((self.prompt_adapter_id, self.lora_int_id))

    def num_hashed_tokens_of_block(self, logical_idx: int):
        return logical_idx * self.block_size + self.block_size

    def reset_state_for_recompute(self):
        """Reset the sequence states for recomputation."""
        self.data.reset_state_for_recompute()

    def append_token_id(self, token_id: int, logprobs: dict[int, Logprob], token_embed: jax.Array | None = None) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)
        self.data.append_token_id(token_id, logprobs[token_id].logprob, token_embed)

    def get_len(self) -> int:
        return self.data.get_len()

    def get_prompt_len(self) -> int:
        return self.data.get_prompt_len()

    def get_output_len(self) -> int:
        return self.data.get_output_len()

    def get_token_ids(self) -> list[int]:
        return self.data.get_token_ids()

    def get_prompt_token_ids(self) -> tuple[int, ...]:
        return self.data.get_prompt_token_ids()

    def get_last_token_id(self) -> int:
        return self.data.get_last_token_id()

    def get_output_token_ids(self) -> tuple[int, ...]:
        return self.data.get_output_token_ids()

    def get_cumulative_logprob(self) -> float:
        return self.data.cumulative_logprob

    def is_finished(self) -> bool:
        return SequenceStatus.is_finished(self.status)

    def fork(self, new_seq_id: int) -> "Sequence":
        new_seq = copy.deepcopy(self)
        new_seq.seq_id = new_seq_id
        return new_seq

    def get_num_new_tokens(self) -> int:
        """Get the number of new tokens to be computed.

        Returns:
            The new number of tokens to be computed. I.e., 1 for decode, or
            the remaining prompt size for prefill.
        """
        if self.data.stage == SequenceStage.DECODE:
            return 1
        return self.data.get_num_uncomputed_tokens()

    def get_num_computed_tokens(self) -> int:
        return self.data.get_num_computed_tokens()

    def is_prefill(self) -> bool:
        return self.data.stage == SequenceStage.PREFILL

    def __repr__(self) -> str:
        return f"Sequence(seq_id={self.seq_id}, status={self.status.name}, num_blocks={self.n_blocks})"


class SequenceGroupState(msgspec.Struct, omit_defaults=True):
    """Mutable state tied to a specific sequence group"""

    # for multi-step decoding
    num_steps: int = 1
    current_step: int = 0

    @property
    def remaining_steps(self) -> int:
        return self.num_steps - self.current_step


class SequenceGroup:
    """A group of sequences that are generated from the same prompt.

    Args:
        request_id: The ID of the request.
        seqs: The list of sequences.
        sampling_params: The sampling parameters used to generate the outputs.
        arrival_time: The arrival time of the request.
            for a pooling model.
        pooled_data: The extracted hidden states from a pooling model.
        encoder_seq: Optional, the single encoder sequence. Should be None
                     unless you are working with an encoder/decoder model.
        trace_headers: OpenTelemetry trace headers.
        priority: User-defined priority of the request.
        draft_size: The number of speculative tokens plus one from the target
                    model; equal to max number of tokens a step can generate
                    for single-draft speculative decoding but larger than
                    that for multi-draft SD (currently not supported).
    """

    def __init__(
        self,
        request_id: str,
        seqs: list[Sequence],
        arrival_time: float,
        sampling_params: SamplingParams | None = None,
        pooled_data: jax.Array | None = None,
        encoder_seq: Sequence | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        draft_size: int = 1,
    ) -> None:
        self.request_id = request_id
        self.seqs = seqs
        self.first_seq = seqs[0]
        self.arrival_time = arrival_time
        self.is_single_seq = len(seqs) == 1
        self.seqs_dict = {seq.seq_id: seq for seq in seqs}

        self.sampling_params = sampling_params
        self.metrics = RequestMetrics(
            arrival_time=arrival_time,
            last_token_time=arrival_time,
            first_scheduled_time=None,
            first_token_time=None,
            time_in_queue=None,
            spec_token_acceptance_counts=[0] * draft_size,
        )
        self.last_token_latency = 0.0
        self.prompt_logprobs: PromptLogprobs | None = None
        self.state = SequenceGroupState()
        self.pooled_data = pooled_data
        self.encoder_seq = encoder_seq
        self.trace_headers = trace_headers
        self.priority = priority

        self.cached_request_output = None

    @property
    def prompt(self) -> str | None:
        return self.first_seq.prompt

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.first_seq.prompt_token_ids

    @property
    def encoder_prompt(self) -> str | None:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt is distinct
        # from the decoder's.
        return self.encoder_seq.prompt if self.encoder_seq is not None else None

    @property
    def encoder_prompt_token_ids(self) -> list[int] | None:
        # There are either 0 or 1 encoder sequences
        # If one is present, its prompt token ids are
        # distinct from the decoder's.
        return self.encoder_seq.prompt_token_ids if self.encoder_seq is not None else None

    @property
    def token_type_ids(self) -> list[int] | None:
        return self.first_seq.token_type_ids

    @property
    def lora_int_id(self) -> int:
        return 0

    @property
    def prompt_adapter_id(self) -> int:
        return 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return 0

    def init_multi_step(self, num_steps: int) -> None:
        self.state.num_steps = num_steps
        self.state.current_step = 0

    def init_multi_step_from_lookahead_slots(
        self, num_lookahead_slots: int, num_scheduler_steps: int, is_multi_step: bool, enable_chunking: bool
    ) -> None:
        if not is_multi_step:
            self.init_multi_step(num_steps=num_scheduler_steps)
            return

        # Multi-Step case
        is_prefill = self.is_prefill()

        # The asserts below reflect the expectations of the current system.
        if is_prefill and enable_chunking:
            assert num_lookahead_slots == num_scheduler_steps
            self.init_multi_step(num_steps=num_lookahead_slots)
        else:
            is_decode: bool = not is_prefill
            # If it is a prefill, num_lookahead_slots must be 0
            assert num_lookahead_slots == 0 or is_decode
            # If it is a decode, num_lookahead_slots + 1 must match
            # the scheduler steps.
            assert num_lookahead_slots + 1 == num_scheduler_steps or is_prefill
            self.init_multi_step(num_steps=num_lookahead_slots + 1)

    def set_last_token_time(self, now: float) -> None:
        """Sets the last token time for Request level timings."""
        # If still in prefill phase, assertion fails.
        assert not self.is_prefill(), (
            "seq_group.set_last_token_time() should not be called if the seq_group is in prefill phase."
        )
        self.last_token_latency = now - self.metrics.last_token_time
        self.metrics.last_token_time = now

    def get_last_token_latency(self) -> float:
        """Returns the latency of the last token."""
        assert not self.is_prefill(), (
            "seq_group.get_last_token_latency() should not be called if the seq_group is in prefill phase."
        )
        return self.last_token_latency

    def maybe_set_first_token_time(self, time: float) -> None:
        """Sets the first token time for Request level timings."""
        # Note: in a case where a sequence_group is swapped and
        #   recomputed, the time between iterations is counted
        #   in TPOT, rather than recalculating TTFT (since from the )
        #   POV of the user, there is simply a long generation delay.
        if self.metrics.first_token_time is None and self.first_seq.get_output_len() == 1:
            self.metrics.first_token_time = time

    def maybe_set_first_scheduled_time(self, time: float) -> None:
        """Sets the first scheduled time and time in queue for Request
        level timings."""
        if self.metrics.first_scheduled_time is None:
            self.metrics.first_scheduled_time = time
            self.metrics.time_in_queue = time - self.metrics.arrival_time

    def set_finished_time(self, time: float | None) -> None:
        """Sets the finished time for Request level timings."""
        self.metrics.finished_time = time

    def get_max_num_running_seqs(self) -> int:
        """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
        if self.is_single_seq:
            return 0 if self.first_seq.is_finished() else 1
        return self.num_seqs() - self.num_finished_seqs()

    def get_seqs(
        self,
        status: SequenceStatus | None = None,
    ) -> list[Sequence]:
        if status is None:
            return self.seqs

        if self.is_single_seq:
            return self.seqs if self.first_seq.status == status else []

        return [seq for seq in self.seqs if seq.status == status]

    def is_encoder_decoder(self) -> bool:
        return self.encoder_seq is not None

    def get_encoder_seq(self) -> Sequence | None:
        return self.encoder_seq

    def get_finished_seqs(self) -> list[Sequence]:
        if self.is_single_seq:
            return self.seqs if self.first_seq.is_finished() else []

        return [seq for seq in self.seqs if seq.is_finished()]

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        for seq in self.seqs:
            if not seq.is_finished():
                seq.data.update_num_computed_tokens(num_new_computed_tokens)

    def get_num_uncomputed_tokens(self) -> int:
        num_uncomputed_tokens = 0
        for seq in self.seqs:
            if not seq.is_finished():
                num_uncomputed_tokens += seq.data.get_num_uncomputed_tokens()
        return num_uncomputed_tokens

    def num_seqs(self, status: SequenceStatus | None = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        if status is None:
            return len(self.seqs)

        if self.is_single_seq:
            return 1 if self.seqs[0].status == status else 0

        return len(self.get_seqs(status))

    def num_finished_seqs(self) -> int:
        if self.is_single_seq:
            return 1 if self.seqs[0].is_finished() else 0
        return len(self.get_finished_seqs())

    def is_finished(self) -> bool:
        if self.is_single_seq:
            return self.first_seq.is_finished()
        return all(seq.is_finished() for seq in self.seqs)

    def is_prefill(self) -> bool:
        return self.first_seq.is_prefill()

    def __repr__(self) -> str:
        return (
            f"SequenceGroup(request_id={self.request_id}, "
            f"sampling_params={self.sampling_params}, "
            f"num_seqs={len(self.seqs)})"
        )

    def uses_prompt_embeds(self) -> bool:
        """Returns True if the sequence group uses input embeds."""
        return any(seq.data.prompt_embeds is not None for seq in self.seqs)


class SequenceGroupMetadataDelta(msgspec.Struct, tag=True, array_like=True, omit_defaults=True):
    """Delta of SequenceGroupMetadata.

    After sending the first SequenceGroupMetadata, scheduler
    only sends delta to reduce the data payload size.
    """

    seq_data_delta: dict[int, SequenceDataDelta]
    request_id: str
    block_tables: dict[int, list[int]]
    is_prompt: bool
    do_sample: bool = True
    token_chunk_size: int | None = None
    computed_block_nums: list[int] | None = None
    state: SequenceGroupState | None = msgspec.field(default_factory=lambda: SequenceGroupState())


class SequenceGroupMetadata(msgspec.Struct, tag=True, array_like=True, omit_defaults=True):
    """Metadata for a sequence group. Used to create `AttentionMetadata`.

    Args:
        request_id: The ID of the request.
        is_prompt: Whether the request is at prompt stage.
        seq_data: The sequence data. (Seq id -> sequence data)
        sampling_params: The sampling parameters used to generate the outputs.
        block_tables: The block tables. (Seq id -> list of physical block
            numbers)
        do_sample: True if sampling is required. Sampling is not required when
            e.g., prefill is chunked, and the current iteration only computes
            query tokens for prefill, we don't need sampling.
        token_chunk_size: The number of tokens to be processed (per sequence).
            None if chunking is not required.
        computed_block_nums: The block numbers that are already computed,
            used in prefix caching.
        state: Internal state tied to this sequence group.
        encoder_seq_data: Optional sequence data for encoder prompt
                          (SequenceGroup.encoder_seq). Should be None
                          unless you are working with an encoder/decoder
                          model.
        cross_block_table: Optional cross-attention block table associated
                           with the encoder prompt
                           (SequenceGroup.encoder_seq). Should be None
                           unless you are working with an encoder/decoder
                           model.
    """

    request_id: str
    is_prompt: bool
    seq_data: dict[int, SequenceData]
    sampling_params: SamplingParams | None
    block_tables: dict[int, list[int]]
    do_sample: bool = True
    computed_block_nums: list[int] | None = None
    state: SequenceGroupState | None = msgspec.field(default_factory=lambda: SequenceGroupState())
    token_type_ids: list[int] | None = None
    encoder_seq_data: SequenceData | None = None
    cross_block_table: list[int] | None = None
    token_chunk_size: int | None = None
    num_speculative_tokens: int | None = None

    def __post_init__(self):
        if self.seq_data is not None and self.token_chunk_size is None:
            if self.is_prompt:
                self.token_chunk_size = next(iter(self.seq_data.values())).get_len()
            else:
                self.token_chunk_size = 1

    @property
    def lora_int_id(self) -> int:
        return 0

    @property
    def prompt_adapter_id(self) -> int:
        return 0

    @property
    def prompt_adapter_num_virtual_tokens(self) -> int:
        return 0

    @property
    def is_single_step_prompt(self) -> bool:
        return self.is_prompt and self.do_sample

    def get_first_seq_id(self) -> int:
        return next(iter(self.seq_data))

    def apply_delta(self, sequence_group_metadata_delta: SequenceGroupMetadataDelta):
        for id_, delta in sequence_group_metadata_delta.seq_data_delta.items():
            self.seq_data[id_].apply_delta(delta)
        assert self.request_id == sequence_group_metadata_delta.request_id
        self.block_tables = sequence_group_metadata_delta.block_tables
        self.token_chunk_size = sequence_group_metadata_delta.token_chunk_size
        self.do_sample = sequence_group_metadata_delta.do_sample
        self.is_prompt = sequence_group_metadata_delta.is_prompt

    def finish_step(self) -> None:
        assert self.state is not None
        assert self.state.current_step < self.state.num_steps, (
            f"current step {self.state.current_step}, num_steps {self.state.num_steps}"
        )
        self.state.current_step += 1


class SequenceOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    parent_seq_id: int
    output_token: int
    logprobs: dict[int, Logprob]
    output_embed: jax.Array | None = None

    def __repr__(self) -> str:
        output_embed_shape = self.output_embed.shape if self.output_embed is not None else None
        return (
            f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
            f"output_token={self.output_token}, "
            f"output_embed.shape={output_embed_shape}, "
            f"logprobs={self.logprobs})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = self.parent_seq_id == other.parent_seq_id and self.output_token == other.output_token
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


class SequenceGroupOutput(ABC):
    """The base class for model outputs associated with a sequence group."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class CompletionSequenceGroupOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    """The model output associated with a completion sequence group."""

    __metaclass__ = SequenceGroupOutput
    samples: list[SequenceOutput]
    prompt_logprobs: PromptLogprobs | None
    step_index: int | None = 0

    def __repr__(self) -> str:
        return f"CompletionSequenceGroupOutput(samples={self.samples}, prompt_logprobs={self.prompt_logprobs})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionSequenceGroupOutput):
            raise NotImplementedError()
        return self.samples == other.samples and self.prompt_logprobs == other.prompt_logprobs


class PoolingSequenceGroupOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    """The model output associated with a pooling sequence group."""

    __metaclass__ = SequenceGroupOutput
    # Annotated as Any to be compatible with msgspec
    # The actual type is in SequenceGroup.pooled_data
    data: Any

    def __repr__(self) -> str:
        return f"PoolingSequenceGroupOutput(data={self.data}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoolingSequenceGroupOutput):
            raise NotImplementedError()
        return self.data == other.data


# cannot use msgspec.Struct here because Dynamo does not support it
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict[str, jax.Array]

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    def __getitem__(self, key: str | slice):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value: jax.Array):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


class PoolerOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    """The output from a pooling operation in the pooling model."""

    outputs: list[PoolingSequenceGroupOutput]

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.outputs == other.outputs


def get_all_seq_ids(seq_group_metadata_list: list[SequenceGroupMetadata]) -> list[int]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    return [seq_id for sg in seq_group_metadata_list for seq_id in sg.seq_data]


def get_all_seq_ids_and_request_ids(
    seq_group_metadata_list: list[SequenceGroupMetadata],
) -> tuple[list[int], dict[str, set[int]]]:
    """Given a list of SequenceGroupMetadata, create a list of all
    sequence ids.
    """
    seq_ids: list[int] = []
    request_id_seq_ids_mapping: defaultdict[str, set[int]] = defaultdict(set)
    for sg in seq_group_metadata_list:
        for seq_id in sg.seq_data:
            seq_ids.append(seq_id)
            request_id_seq_ids_mapping[sg.request_id].add(seq_id)
    return seq_ids, request_id_seq_ids_mapping


class HiddenStates(msgspec.Struct, array_like=True, omit_defaults=True):
    """Hidden states corresponding to in-progress sequences.
    Used in speculative decoding to pass hidden states from
    the target model to the proposer model.

    seq_ids are the sequence ids of each entry of the batch
    dimension of the hidden_states tensor"""

    hidden_states: jax.Array
    seq_group_metadata_list: list[SequenceGroupMetadata] | None = None
    second_last_token_hidden_states: jax.Array | None = None

    _seq_ids: list[int] = msgspec.field(default_factory=list)

    def __post_init__(self):
        if self.seq_group_metadata_list is not None:
            assert len(self.seq_group_metadata_list) == len(self.hidden_states)
            self._seq_ids = get_all_seq_ids(self.seq_group_metadata_list)

    @property
    def seq_ids(self) -> list[int]:
        return self._seq_ids

    def update(
        self,
        hidden_states: jax.Array,
        seq_group_metadata_list: list[SequenceGroupMetadata],
        second_last_token_hidden_states: jax.Array | None = None,
    ):
        """Update hidden states from target model invocation. Only used for
        decode steps"""
        assert len(seq_group_metadata_list) == len(hidden_states)
        self._seq_ids.extend(get_all_seq_ids(seq_group_metadata_list))
        self.hidden_states = jnp.concatenate([self.hidden_states, hidden_states])

        if self.second_last_token_hidden_states is not None:
            # Adding dummy hidden_states to this to maintain same shape
            self.second_last_token_hidden_states = jnp.concatenate(
                [
                    self.second_last_token_hidden_states,
                    jnp.zeros_like(hidden_states)
                    if second_last_token_hidden_states is None
                    else second_last_token_hidden_states,
                ]
            )

    def prune(self, seq_group_metadata_list: list[SequenceGroupMetadata]) -> None:
        """Prune to provided list of sequence ids. Only used for decode steps."""
        seq_ids = get_all_seq_ids(seq_group_metadata_list)
        seq_ids = [seq_id for seq_id in seq_ids if seq_id in self._seq_ids]
        if seq_ids != self._seq_ids:
            index = [self._seq_ids.index(seq_id) for seq_id in seq_ids]
            self.hidden_states = self.hidden_states[index]
            if self.second_last_token_hidden_states is not None:
                self.second_last_token_hidden_states = self.second_last_token_hidden_states[index]
            self._seq_ids = seq_ids

    def expand_with_bonus_tokens(self, seq_with_bonus_token_in_last_step: set) -> None:
        """Expand hidden states for sequences with bonus tokens. This is in
        alignment with `MultiStepWorker._expand_execute_model_request`."""
        if self.second_last_token_hidden_states is None or not seq_with_bonus_token_in_last_step:
            return

        index = []
        for seq_id in self._seq_ids:
            i = self._seq_ids.index(seq_id)
            if seq_id in seq_with_bonus_token_in_last_step:
                index.append(i + len(self._seq_ids))
            index.append(i)

        self.hidden_states = jnp.concatenate([self.hidden_states, self.second_last_token_hidden_states])[index]


class ExecuteModelRequest(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
):
    """The model execution request, containing CPU metadata only. The
    engine should create an instance of this class for each request batch."""

    seq_group_metadata_list: list[SequenceGroupMetadata | SequenceGroupMetadataDelta]
    blocks_to_swap_in: list[tuple[int, int]] = msgspec.field(default_factory=list)
    blocks_to_swap_out: list[tuple[int, int]] = msgspec.field(default_factory=list)
    blocks_to_copy: list[tuple[int, int]] = msgspec.field(default_factory=list)
    virtual_engine: int = 0
    num_lookahead_slots: int = 0
    running_queue_size: int = 0
    previous_hidden_states: HiddenStates | None = None
    num_steps: int = 1
    spec_step_idx: int | None = None
    finished_requests_ids: list[str] = msgspec.field(default_factory=list)
    last_sampled_token_ids: jax.Array | None = None
    async_callback: Callable | None = None

    @property
    def is_first_multi_step(self) -> bool:
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.current_step == 0

    @property
    def is_last_step(self) -> bool:
        assert len(self.seq_group_metadata_list) > 0
        first_seq_group = self.seq_group_metadata_list[0]
        assert first_seq_group.state is not None
        return first_seq_group.state.remaining_steps == 1

    @property
    def current_step(self) -> int:
        assert len(self.seq_group_metadata_list) > 0
        state = self.seq_group_metadata_list[0].state
        assert state is not None
        return state.current_step

    def clone(
        self, seq_group_metadata_list: list[SequenceGroupMetadata | SequenceGroupMetadataDelta]
    ) -> "ExecuteModelRequest":
        """Clone the request with a new sequence group metadata list."""
        return ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=self.blocks_to_swap_in.copy(),
            blocks_to_swap_out=self.blocks_to_swap_out.copy(),
            blocks_to_copy=self.blocks_to_copy.copy(),
            virtual_engine=self.virtual_engine,
            num_lookahead_slots=self.num_lookahead_slots,
            running_queue_size=self.running_queue_size,
            previous_hidden_states=self.previous_hidden_states,
            num_steps=self.num_steps,
            finished_requests_ids=self.finished_requests_ids,
            last_sampled_token_ids=self.last_sampled_token_ids.clone()
            if self.last_sampled_token_ids is not None
            else None,
            async_callback=self.async_callback,
        )


@dataclass
class SequenceGroupBase:
    group_id: str
    assembled_seq_group: SequenceGroup | None = None
    seq_id_to_index: dict[str, int] = field(default_factory=dict)
    to_be_finished: dict[str, SequenceGroup] = field(default_factory=dict)
    finished_reqs: dict[str, SequenceGroup] = field(default_factory=dict)
    streaming: bool = False
    output_produced: bool = False

    @staticmethod
    def add_request(request_id: str, engine, params, *args, **kwargs):
        """When we are ready to add a request with request_id and params
        into the engine, we can split the request into multiple requests.
        """
        raise NotImplementedError

    def finish_seq(self, seq: SequenceGroup):
        """The sequence `seq` finishes, we should record the information."""
        del self.to_be_finished[seq.request_id]
        self.finished_reqs[seq.request_id] = seq

    def maybe_assemble_group(self, seq_group: SequenceGroup) -> SequenceGroup | None:
        """Assemble the sequence group, for producing the final
        output, or adding request in the engine again.
        """
        raise NotImplementedError


class ParallelSampleSequenceGroup(SequenceGroupBase):
    @staticmethod
    def add_request(request_id: str, engine, params, **kwargs):
        original_params = params
        group = ParallelSampleSequenceGroup(request_id)
        seqs = []
        for i in range(original_params.n):
            request_id_i = f"{request_id}_parallel_sample_{i}"
            group.seq_id_to_index[request_id_i] = i
            params = original_params.clone()
            params.n = 1
            if params.seed is not None:
                params.seed += i
            seq_group = engine._add_processed_request(
                request_id_i,
                params=params,
                **kwargs,
            )
            assert seq_group is not None
            engine.seq_id_tountil_seq_group[request_id_i] = group
            group.to_be_finished[request_id_i] = seq_group
            seqs.append(seq_group.seqs[0])

        group.assembled_seq_group = SequenceGroup(
            request_id=request_id,
            seqs=seqs,
            arrival_time=seq_group.arrival_time,
            sampling_params=original_params,
            pooled_data=seq_group.pooled_data,
            encoder_seq=seq_group.encoder_seq,
            trace_headers=seq_group.trace_headers,
            priority=seq_group.priority,
        )

        group.streaming = params.output_kind == RequestOutputKind.DELTA
        group.output_produced = False

    def maybe_assemble_group(self, seq_group: SequenceGroup) -> SequenceGroup | None:
        # in the streaming mode, we will return the assembled sequence
        # for the first remaining sequence, and then return None for the
        # rest of sequences
        if self.streaming:
            first_remaining_id = next(iter(self.to_be_finished))
            if seq_group.request_id == first_remaining_id:
                return self.assembled_seq_group
            return None

        # in the non-streaming mode, we will return the assembled sequence
        # when the last sequences finishes, and then return None for the
        # rest of the time
        if len(self.to_be_finished) == 1 and seq_group.request_id in self.to_be_finished and seq_group.is_finished():
            assert self.assembled_seq_group is not None
            params = self.assembled_seq_group.sampling_params
            assert isinstance(params, SamplingParams)
            if not self.output_produced:
                self.output_produced = True
                if params._real_n is not None:
                    # Get the top-n sequences.
                    n = params._real_n or params.n
                    seqs = self.assembled_seq_group.seqs
                    sorted_seqs = sorted(seqs, key=lambda seq: seq.get_cumulative_logprob(), reverse=True)
                    top_n_seqs = sorted_seqs[:n]
                    self.assembled_seq_group.seqs = top_n_seqs
                return self.assembled_seq_group
            if self.output_produced:
                return None
        return None
