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

"""Sampling parameters and configurations for text generation.

This module defines the parameters that control text generation behavior,
including temperature, top-k, top-p, penalties, and other sampling strategies.
It provides both host-side Python configurations and JAX-compatible versions
for efficient on-device processing.

Classes:
    SamplingType: Enum for sampling strategies (greedy vs random)
    RequestOutputKind: Enum for output formats (cumulative, delta, final)
    GuidedDecodingParams: Parameters for constrained generation
    JitableSamplingParams: JAX-compatible sampling parameters
    SamplingParams: High-level sampling configuration
    BeamSearchParams: Parameters for beam search decoding

Example:
    >>> from easydel.inference import SamplingParams
    >>> params = SamplingParams(
    ...     temperature=0.8,
    ...     top_p=0.95,
    ...     max_tokens=100,
    ...     stop=["\n", "END"]
    ... )
    >>> # Convert to JAX-compatible format
    >>> jit_params = params.make_jitable()
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import jax
from eformer.escale import with_sharding_constraint
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers import AutoTokenizer

from .logits_process import (
    FrequencyPenaltyLogitsProcessor,
    LogitsProcessorList,
    MinPLogitsWarper,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

logger = get_logger(__name__)


class SamplingType(IntEnum):
    """Defines the sampling strategy.

    Attributes:
        GREEDY: Deterministic selection of highest probability token
        RANDOM: Probabilistic sampling from the distribution
    """

    GREEDY = 0
    RANDOM = 1


class RequestOutputKind(Enum):
    """Defines the kind of output for a request.

    Attributes:
        CUMULATIVE: Return the full generated text so far
        DELTA: Return only newly generated tokens
        FINAL_ONLY: Return only the final complete output
    """

    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


@auto_pytree
class GuidedDecodingParams:
    """Parameters for guided decoding.

    Enables constrained generation to match specific formats or patterns.
    Only one guided decoding mode can be active at a time.

    Attributes:
        json: JSON schema or object for structured output
        regex: Regular expression pattern to match
        choice: List of allowed string choices
        grammar: Context-free grammar specification
        json_object: Force output to be valid JSON object
        backend: Decoding backend to use
        backend_was_auto: Whether backend was auto-selected
        disable_fallback: Disable fallback to unconstrained generation
        disable_any_whitespace: Disable whitespace in structured output
        disable_additional_properties: Restrict JSON to defined properties only
        whitespace_pattern: Custom whitespace pattern
        structural_tag: Tag for structural elements

    Raises:
        ValueError: If multiple guided decoding modes are specified
    """

    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    backend: str | None = None
    backend_was_auto: bool = False
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    def __post_init__(self):
        """Validates that only one guided decoding mode is specified.

        Raises:
            ValueError: If more than one guided decoding mode is active
        """
        guide_count = sum(
            (
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
            )
        )
        if guide_count > 1:
            raise ValueError(
                f"Only one guided decoding mode can be used, but multiple were specified: {dataclasses.asdict(self)}"
            )


@auto_pytree(frozen=True)
class JitableSamplingParams:
    """
    A JAX-native, device-ready version of sampling parameters.

    This class contains only JAX arrays and static information, making it
    suitable for passing into jit-compiled functions. All Python-specific
    types like strings and lists have been converted or removed.
    """

    random_sampling: jax.Array  # [1] bool

    temperature: jax.Array
    top_k: jax.Array
    top_p: jax.Array
    min_p: jax.Array
    repetition_penalty: jax.Array
    frequency_penalty: jax.Array
    presence_penalty: jax.Array

    max_tokens: jax.Array
    min_tokens: jax.Array

    all_stop_token_ids: jax.Array
    bad_words_token_ids: jax.Array  # Padded to be rectangular
    bad_words_lengths: jax.Array  # Stores the true length of each bad word sequence
    allowed_token_ids: jax.Array | None = None

    def insert(self, second_sample: JitableSamplingParams, slot: int) -> JitableSamplingParams:
        self = self.view_1d()

        def update_idx1d(x, y):
            sharding = getattr(x, "sharding", PartitionSpec())
            return with_sharding_constraint(jax.lax.dynamic_update_slice(x, y, (slot,)), sharding)

        return JitableSamplingParams(
            random_sampling=update_idx1d(self.random_sampling, second_sample.random_sampling),
            temperature=update_idx1d(self.temperature, second_sample.temperature),
            top_k=update_idx1d(self.top_k, second_sample.top_k),
            top_p=update_idx1d(self.top_p, second_sample.top_p),
            min_p=update_idx1d(self.min_p, second_sample.min_p),
            repetition_penalty=update_idx1d(self.repetition_penalty, second_sample.repetition_penalty),
            frequency_penalty=update_idx1d(self.frequency_penalty, second_sample.frequency_penalty),
            presence_penalty=update_idx1d(self.presence_penalty, second_sample.presence_penalty),
            max_tokens=update_idx1d(self.max_tokens, second_sample.max_tokens),
            min_tokens=update_idx1d(self.min_tokens, second_sample.min_tokens),
            all_stop_token_ids=self.all_stop_token_ids,
            bad_words_token_ids=self.bad_words_token_ids,
            bad_words_lengths=self.bad_words_lengths,
            allowed_token_ids=self.allowed_token_ids,
        )

    def view_1d(self) -> JitableSamplingParams:
        return JitableSamplingParams(
            random_sampling=self.random_sampling.reshape(-1),
            temperature=self.temperature.reshape(-1),
            top_k=self.top_k.reshape(-1),
            top_p=self.top_p.reshape(-1),
            min_p=self.min_p.reshape(-1),
            repetition_penalty=self.repetition_penalty.reshape(-1),
            frequency_penalty=self.frequency_penalty.reshape(-1),
            presence_penalty=self.presence_penalty.reshape(-1),
            max_tokens=self.max_tokens.reshape(-1),
            min_tokens=self.min_tokens.reshape(-1),
            all_stop_token_ids=self.all_stop_token_ids.reshape(-1),
            bad_words_token_ids=self.bad_words_token_ids.reshape(-1),
            bad_words_lengths=self.bad_words_lengths.reshape(-1),
            allowed_token_ids=self.allowed_token_ids.reshape(-1) if self.allowed_token_ids is not None else None,
        )

    def view_2d(self) -> JitableSamplingParams:
        return JitableSamplingParams(
            random_sampling=self.random_sampling.reshape(-1, 1),
            temperature=self.temperature.reshape(-1, 1),
            top_k=self.top_k.reshape(-1, 1),
            top_p=self.top_p.reshape(-1, 1),
            min_p=self.min_p.reshape(-1, 1),
            repetition_penalty=self.repetition_penalty.reshape(-1, 1),
            frequency_penalty=self.frequency_penalty.reshape(-1, 1),
            presence_penalty=self.presence_penalty.reshape(-1, 1),
            max_tokens=self.max_tokens.reshape(-1, 1),
            min_tokens=self.min_tokens.reshape(-1, 1),
            all_stop_token_ids=self.all_stop_token_ids.reshape(-1, 1),
            bad_words_token_ids=self.bad_words_token_ids.reshape(-1, 1),
            bad_words_lengths=self.bad_words_lengths.reshape(-1, 1),
            allowed_token_ids=self.allowed_token_ids.reshape(-1, 1) if self.allowed_token_ids is not None else None,
        )

    @classmethod
    def init_empty(cls, batch_size: int):
        return cls(
            random_sampling=jnp.zeros([batch_size], dtype="b1"),
            temperature=jnp.zeros([batch_size], dtype="f4"),
            top_k=jnp.zeros([batch_size], dtype="i4"),
            top_p=jnp.zeros([batch_size], dtype="f4"),
            min_p=jnp.zeros([batch_size], dtype="f4"),
            repetition_penalty=jnp.zeros([batch_size], dtype="f4"),
            frequency_penalty=jnp.zeros([batch_size], dtype="f4"),
            presence_penalty=jnp.zeros([batch_size], dtype="f4"),
            max_tokens=jnp.zeros([batch_size], dtype="i4"),
            min_tokens=jnp.zeros([batch_size], dtype="i4"),
            all_stop_token_ids=jnp.array([[]], dtype=jnp.int32),
            bad_words_token_ids=jnp.array([[]], dtype=jnp.int32),
            bad_words_lengths=jnp.array([[]], dtype=jnp.int32),
        )

    @classmethod
    def from_host_params(cls, params: SamplingParams) -> JitableSamplingParams:
        """Converts the host-side SamplingParams to a JIT-compatible version."""
        if params._bad_words_token_ids:
            max_len = max(len(ids) for ids in params._bad_words_token_ids)
            lengths = jnp.array([len(ids) for ids in params._bad_words_token_ids], dtype=jnp.int32)
            padded_ids = jnp.array(
                [ids + [-100] * (max_len - len(ids)) for ids in params._bad_words_token_ids],
                dtype=jnp.int32,
            )
        else:
            lengths = jnp.array([], dtype=jnp.int32)
            padded_ids = jnp.array([[]], dtype=jnp.int32)

        return cls(
            random_sampling=jnp.asarray(params.sampling_type.value == SamplingType.RANDOM.value, dtype=jnp.bool),
            temperature=jnp.asarray(params.temperature, dtype=jnp.float32),
            top_k=jnp.asarray(params.top_k, dtype=jnp.int32),
            top_p=jnp.asarray(params.top_p, dtype=jnp.float32),
            min_p=jnp.asarray(params.min_p, dtype=jnp.float32),
            repetition_penalty=jnp.asarray(params.repetition_penalty, dtype=jnp.float32),
            frequency_penalty=jnp.asarray(params.frequency_penalty, dtype=jnp.float32),
            presence_penalty=jnp.asarray(params.presence_penalty, dtype=jnp.float32),
            max_tokens=jnp.asarray(params.max_tokens if params.max_tokens is not None else -1, dtype=jnp.int32),
            min_tokens=jnp.asarray(params.min_tokens, dtype=jnp.int32),
            all_stop_token_ids=jnp.asarray(list(params.all_stop_token_ids), dtype=jnp.int32),
            bad_words_token_ids=padded_ids,
            bad_words_lengths=lengths,
            allowed_token_ids=jnp.asarray(params.allowed_token_ids, dtype=jnp.int32)
            if params.allowed_token_ids
            else None,
        )

    def get_logits_warper(self):
        """
        Constructs a `LogitsProcessorList` containing the configured logits warpers.

        Logits warpers modify the probability distribution derived from logits, typically
        used for techniques like temperature scaling, top-k, top-p, and min-p sampling.

        Returns:
            A `LogitsProcessorList` containing the enabled logits warpers based on the
            sampling parameters.
        """
        warpers = LogitsProcessorList()
        warpers.append(TemperatureLogitsWarper(temperature=self.temperature))
        warpers.append(TopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=1))
        warpers.append(TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1))
        warpers.append(MinPLogitsWarper(min_p=self.min_p, min_tokens_to_keep=1))
        return warpers

    def get_logits_processor(self):
        """
        Constructs a `LogitsProcessorList` containing the configured logits processors.

        Logits processors modify the logits directly, often used for applying penalties
        (presence, frequency, repetition) or suppressing specific tokens.

        Returns:
            A `LogitsProcessorList` containing the enabled logits processors based on the
            sampling parameters.
        """
        processors = LogitsProcessorList()
        processors.append(PresencePenaltyLogitsProcessor(self.presence_penalty))
        processors.append(FrequencyPenaltyLogitsProcessor(self.frequency_penalty))
        processors.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))
        return processors

    @cached_property
    def logits_processor(self):
        return self.get_logits_processor()

    @cached_property
    def logits_warper(self):
        return self.get_logits_warper()

    def make_jitable(self):
        return self


@auto_pytree
class SamplingParams:
    """Sampling parameters for text generation.

    Comprehensive configuration for controlling text generation behavior.
    Supports both OpenAI-compatible parameters and EasyDeL-specific extensions.

    Attributes:
        n: Number of sequences to generate
        best_of: Sample n sequences and return the best one
        presence_penalty: Penalty for tokens based on presence (-2.0 to 2.0)
        frequency_penalty: Penalty for tokens based on frequency (-2.0 to 2.0)
        repetition_penalty: Multiplicative penalty for repeated tokens
        temperature: Controls randomness (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling threshold (0.0 to 1.0)
        min_p: Minimum probability threshold relative to top token
        top_k: Number of highest probability tokens to consider
        seed: Random seed for reproducibility
        stop: List of stop strings
        stop_token_ids: List of stop token IDs
        bad_words: List of strings to avoid generating
        ignore_eos: Whether to ignore end-of-sequence token
        max_tokens: Maximum number of tokens to generate
        min_tokens: Minimum number of tokens to generate
        logprobs: Number of log probabilities to return
        prompt_logprobs: Number of prompt log probabilities to return
        detokenize: Whether to convert tokens to text
        skip_special_tokens: Whether to skip special tokens in output
        spaces_between_special_tokens: Add spaces between special tokens
        include_stop_str_in_output: Include stop string in output
        output_kind: Type of output to return
        truncate_prompt_tokens: Truncate prompt to this many tokens
        guided_decoding: Parameters for constrained generation
        logit_bias: Bias to apply to specific token logits
        allowed_token_ids: Whitelist of allowed token IDs
        extra_args: Additional custom arguments

    Example:
        >>> params = SamplingParams(
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     max_tokens=100,
        ...     stop=["\n\n"]
        ... )
    """

    n: int = 1
    best_of: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    seed: int | None = None

    # Stopping Conditions
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    bad_words: list[str] = field(default_factory=list)
    ignore_eos: bool = False
    max_tokens: int | None = 16
    min_tokens: int = 0

    # Output Control
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    detokenize: bool = True
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool = False
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    # Advanced & Guided Decoding
    truncate_prompt_tokens: Annotated[int, "ge=1"] | None = None
    guided_decoding: GuidedDecodingParams | None = None
    logit_bias: dict[int, float] | None = None
    allowed_token_ids: list[int] | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    # Internal fields computed during initialization or via update methods.
    # They are not part of the constructor (`init=False`).
    _real_n: int | None = field(default=None, init=False)
    _output_text_buffer_length: int = field(default=0, init=False)
    _all_stop_token_ids: set[int] = field(default_factory=set, init=False)
    _bad_words_token_ids: list[list[int]] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Initializes and validates parameters.
        """
        if self.best_of is not None:
            if self.best_of < self.n:
                raise ValueError(f"best_of ({self.best_of}) must be >= n ({self.n}).")
            if not self._real_n:
                self._real_n = self.n
                self.n = self.best_of
        if self.temperature is None:
            self.temperature = 1
        if 0 < self.temperature < 1e-2:
            logger.warning(
                f"temperature {self.temperature} is below {1e-2}, which may cause numerical instability. "
                f"Clamping to {1e-2}."
            )
            self.temperature = 1e-2
        if self.seed == -1:
            self.seed = None

        if isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.logprobs is True:
            self.logprobs = 1
        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        if self.stop and not self.include_stop_str_in_output:
            buffer_len = max(len(s) for s in self.stop) - 1
            self._output_text_buffer_length = buffer_len

        self._verify_args()

        if self.temperature < 1e-5:
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        self._all_stop_token_ids = set(self.stop_token_ids)

    def _verify_args(self) -> None:
        """Performs detailed validation of parameter values."""
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(f"presence_penalty must be in [-2, 2], got {self.presence_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise ValueError(f"min_tokens ({self.min_tokens}) must be <= max_tokens ({self.max_tokens}).")
        if self.stop and not self.detokenize:
            raise ValueError("stop strings require detokenize=True.")

    def _verify_greedy_sampling(self) -> None:
        """Validates parameters for greedy sampling."""
        if self.n > 1:
            raise ValueError(f"n must be 1 for greedy sampling, got {self.n}.")

    def update_with_generation_config(
        self,
        generation_config: dict[str, Any],
        model_eos_token_id: int | None = None,
    ) -> SamplingParams:
        """
        Creates a new `SamplingParams` instance updated with a model's generation_config.
        Returns a new instance to maintain immutability.
        """
        all_stop_ids = self._all_stop_token_ids.copy()
        if model_eos_token_id is not None:
            all_stop_ids.add(model_eos_token_id)

        new_stop_token_ids = self.stop_token_ids
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            eos_ids_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if model_eos_token_id is not None:
                eos_ids_set.discard(model_eos_token_id)
            if eos_ids_set and not self.ignore_eos:
                new_stop_token_ids = list(set(self.stop_token_ids) | eos_ids_set)
                all_stop_ids.update(eos_ids_set)

        self.stop_token_ids = new_stop_token_ids
        self._all_stop_token_ids = all_stop_ids
        return self

    def update_with_tokenizer(self, tokenizer: AutoTokenizer) -> SamplingParams:
        """
        Creates a new `SamplingParams` instance with bad_words encoded into token IDs.
        Returns a new instance to maintain immutability.
        """
        if not self.bad_words:
            return self

        bad_words_token_ids = []
        for word in self.bad_words:
            for add_prefix_space in [False, True]:
                text = (" " if add_prefix_space else "") + word.lstrip()
                token_ids = tokenizer.encode(text=text, add_special_tokens=False)
                if not add_prefix_space or (
                    add_prefix_space and len(bad_words_token_ids) > 0 and token_ids != bad_words_token_ids[-1]
                ):
                    bad_words_token_ids.append(token_ids)

        vocab_size = getattr(tokenizer, "vocab_size", tokenizer.model_max_length)
        invalid_ids = [tid for ids in bad_words_token_ids for tid in ids if not (0 <= tid < vocab_size)]
        if invalid_ids:
            raise ValueError(
                f"Bad words resulted in invalid token IDs: {invalid_ids}. "
                f"All token IDs must be within the vocab size of {vocab_size}."
            )

        self._bad_words_token_ids = bad_words_token_ids
        return self

    @cached_property
    def sampling_type(self) -> SamplingType:
        """Determines the sampling type based on parameters."""
        if self.temperature < 1e-5:
            return SamplingType.GREEDY
        return SamplingType.RANDOM

    @property
    def all_stop_token_ids(self) -> set[int]:
        """Returns all stop token IDs, including EOS."""
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        """Returns the tokenized versions of bad_words."""
        return self._bad_words_token_ids

    def make_jitable(self) -> JitableSamplingParams:
        """
        Converts this host-side configuration into a JAX-jittable object.

        This method should be called after all pre-processing (like tokenization)
        is complete.
        """
        if self.bad_words and self._bad_words_token_ids is None:
            raise RuntimeError("Must call `with_tokenizer()` before `make_jitable()` when `bad_words` is set.")
        return JitableSamplingParams.from_host_params(self)

    def clone(self) -> SamplingParams:
        """Creates a deep copy of the instance."""
        return copy.deepcopy(self)


@auto_pytree
class BeamSearchParams:
    """Beam search parameters for text generation.

    Configuration for beam search decoding, which maintains multiple
    candidate sequences and selects the best ones based on cumulative probability.
    This class is immutable (`frozen=True`) for JAX compatibility.

    Attributes:
        beam_width: Number of beams to maintain
        max_tokens: Maximum tokens to generate
        ignore_eos: Whether to ignore end-of-sequence token
        temperature: Temperature for beam search (0.0 = greedy)
        length_penalty: Penalty factor for sequence length
        include_stop_str_in_output: Include stop string in output

    Note:
        Beam search is typically used when deterministic, high-quality
        output is desired, at the cost of increased computation.
    """

    beam_width: int
    max_tokens: int
    ignore_eos: bool = False
    temperature: float = 0.0
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
