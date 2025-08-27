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

"""Logits processing and warping utilities for text generation.

This module provides a comprehensive set of logits processors and warpers
for controlling text generation behavior in language models. These utilities
allow fine-grained control over the probability distributions during sampling.

Key Components:
    - LogitsProcessor: Base class for modifying logits before sampling
    - LogitsWarper: Base class for rescaling probability distributions
    - Various specialized processors for temperature, top-k, top-p, penalties, etc.

Example:
    >>> from easydel.inference.logits_process import (
    ...     LogitsProcessorList,
    ...     TemperatureLogitsWarper,
    ...     TopKLogitsWarper
    ... )
    >>> processors = LogitsProcessorList()
    >>> processors.append(TemperatureLogitsWarper(temperature=0.7))
    >>> processors.append(TopKLogitsWarper(top_k=50))
    >>> # Apply processors to logits during generation
    >>> processed_scores = processors(input_ids, scores, cur_len)
"""

import inspect

import jax
import jax.lax as lax
import jax.numpy as jnp
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
from jax.experimental import sparse

from easydel.utils.compiling_utils import hash_fn


def add_start_docstrings(*docstr):
    """
    A decorator that prepends a given docstring section to the decorated function's docstring.

    This is useful for adding standard documentation sections (like parameter descriptions)
    to multiple functions without repetition.

    Args:
        *docstr: One or more strings that will be joined and prepended to the
            decorated function's existing docstring.

    Returns:
        A decorator function.
    """

    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search
        kwargs (`Dict[str, Any]`, *optional*):
            Additional logits processor specific kwargs.

    Return:
        `jnp.ndarray` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """
    Abstract base class for all logit processors.

    Logits processors are callable classes that modify the logits predicted by a
    language model *before* sampling. They are used to implement various decoding
    strategies and constraints, such as forcing specific tokens, applying penalties,
    or preventing repetitions.

    Inheriting classes should implement the `__call__` method taking `input_ids`,
    `scores`, and `cur_len` as arguments.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        """
        Applies the processor to the logits.

        Args:
            input_ids: The sequence of token ids generated so far
                (shape: `(batch_size, sequence_length)`).
            scores: The logits predicted by the model for the next token
                (shape: `(batch_size, vocab_size)`).
            cur_len: The current length of the sequence (integer).

        Returns:
            The modified logits.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    __hash__ = hash_fn


class LogitsWarper:
    """
    Abstract base class for all logit warpers.

    Logit warpers are callable classes that modify the logits predicted by a
    language model *after* potential processing but *before* sampling, typically
    by re-scaling or filtering the probability distribution.
    They are used for techniques like temperature scaling, top-k, and top-p (nucleus) sampling.

    Inheriting classes should implement the `__call__` method taking `input_ids`,
    `scores`, and `cur_len` as arguments.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        """
        Applies the warper to the logits.

        Args:
            input_ids: The sequence of token ids generated so far
                (shape: `(batch_size, sequence_length)`).
            scores: The logits predicted by the model for the next token
                (shape: `(batch_size, vocab_size)`).
            cur_len: The current length of the sequence (integer).

        Returns:
            The modified logits.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    __hash__ = hash_fn


@auto_pytree
class EmptyProcessor(LogitsProcessor):
    r"""
    A placeholder `LogitsProcessor` that performs no operation.

    This processor simply returns the input scores unchanged. It can be useful
    in configurations where a processor slot needs to be filled but no actual
    processing is desired at that stage.
    """

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        return scores


class LogitsProcessorList(list):
    """
    A container class, inheriting from `list`, designed to hold a sequence of
    `LogitsProcessor` and `LogitsWarper` objects.

    The primary purpose of this class is to provide a convenient way to apply
    a chain of processors/warpers sequentially to a set of logits. It overrides
    the `__call__` method to iterate through the contained objects and apply each
    one to the logits.

    It intelligently handles processors that might require additional keyword arguments
    by inspecting their `__call__` method signatures using `inspect.signature`.
    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs) -> jnp.ndarray:
        """
        Applies all contained processors and warpers sequentially to the logits.

        Args:
            input_ids: Tensor of input IDs generated so far (shape: `(batch_size, sequence_length)`).
            scores: Logits for the next token prediction (shape: `(batch_size, vocab_size)`).
            cur_len: The current length of the sequences being generated (integer).
            **kwargs: Additional keyword arguments passed down to processors/warpers
                that accept them in their `__call__` method (beyond `input_ids`, `scores`, `cur_len`).

        Returns:
            The final modified logits tensor after applying all processors/warpers in the list.

        Raises:
            ValueError: If a processor in the list requires specific keyword arguments
                (beyond the standard three) that are not provided in `**kwargs`.
        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 3:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, cur_len, **kwargs)
            else:
                scores = processor(input_ids, scores, cur_len)
        return scores

    __hash__ = hash_fn


@auto_pytree
class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that applies temperature scaling to the logits distribution.

    Divides the logits by the `temperature` value. A temperature of 0.0 or 1.0 results
    in no change. Temperatures below 1.0 make the distribution sharper (less random),
    while temperatures above 1.0 make it flatter (more random).

    Args:
        temperature: The temperature value for scaling. Must be non-negative.
            Setting to 0.0 disables the warper effectively.
    """

    temperature: jnp.ndarray

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        return jax.lax.cond(
            self.temperature != -1,
            lambda x, temp: x / temp.astype(x.dtype),
            lambda *x: x[0],
            scores,
            self.temperature,
        )


@auto_pytree
class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that implements top-p (nucleus) sampling.

    Filters the vocabulary distribution by keeping only the smallest set of tokens
    whose cumulative probability mass exceeds the threshold `top_p`. The logits
    of the filtered tokens are set to `filter_value`.

    Reference: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
    by Holtzman et al. (2019).

    Args:
        top_p: The cumulative probability threshold. Must be in (0, 1].
            Setting `top_p=1.0` disables the filter.
        filter_value: The value assigned to the logits of filtered tokens.
            Defaults to -infinity.
        min_tokens_to_keep: Minimum number of tokens to retain, even if their
            cumulative probability exceeds `top_p`. Defaults to 1.
    """

    top_p: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        top_p = jnp.asarray(self.top_p)
        min_keep = self.min_tokens_to_keep

        def _apply(x):
            topk_scores, topk_indices = lax.top_k(x, x.shape[-1])
            mask_scores = jnp.full_like(x, self.filter_value)
            cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
            score_mask = cumulative_probs < top_p
            score_mask = jnp.roll(score_mask, 1)
            score_mask |= score_mask.at[:, 0].set(True)
            score_mask = score_mask.at[:, :min_keep].set(True)
            topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
            x = jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]
            return x

        return jax.lax.cond((top_p > 0) & (top_p < 1), _apply, lambda x: x, scores)


@auto_pytree
class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that implements top-k sampling.

    Filters the vocabulary distribution by keeping only the `top_k` tokens with the
    highest probabilities (logits). The logits of the filtered tokens are set to
    `filter_value`.

    Args:
        top_k: The number of highest probability tokens to keep. Setting `top_k=0`
            disables the filter.
        filter_value: The value assigned to the logits of filtered tokens.
            Defaults to -infinity.
        min_tokens_to_keep: Minimum number of tokens to retain, overriding `top_k`
            if `top_k` is smaller. Ensures at least this many tokens are considered.
            Defaults to 1.
    """

    top_k: int
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        k_val = jnp.asarray(self.top_k)
        vocab_size = scores.shape[-1]
        effective_k = jnp.maximum(k_val, self.min_tokens_to_keep)
        effective_k = jnp.minimum(effective_k, vocab_size).astype(jnp.int32)

        def _filter_scores(s: jnp.ndarray) -> jnp.ndarray:
            """Applies the dynamic filtering logic."""
            sorted_scores = jnp.sort(s, axis=-1)[:, ::-1]
            k_index = effective_k - 1
            k_index = jnp.maximum(0, k_index)
            threshold = sorted_scores[:, k_index]
            threshold = threshold[:, None]
            mask = s >= threshold
            return jnp.where(mask, s, self.filter_value)

        def _identity(s: jnp.ndarray) -> jnp.ndarray:
            """Returns scores unchanged."""
            return s

        return lax.cond(
            (k_val > 0) & (effective_k < vocab_size),
            _filter_scores,  # Apply filtering
            _identity,  # Pass through unchanged
            scores,  # Operand
        )


@auto_pytree
class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that ensures the beginning-of-sequence (BOS) token is
    generated as the very first token.

    This processor modifies the logits only at the first generation step (`cur_len` = 1).
    It sets the logit of the `bos_token_id` to 0 (probability 1) and all other logits
    to `filter_value` (-infinity).

    Args:
        bos_token_id: The integer ID of the Beginning-Of-Sequence token.
    """

    bos_token_id: int

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))
        apply_penalty = 1 - jnp.bool_(cur_len - 1)
        scores = jnp.where(apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores)
        return scores


@auto_pytree
class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that forces the end-of-sequence (EOS) token to be generated
    when the generation process reaches the predefined `max_length`.

    This processor modifies the logits only at the step where `cur_len` equals
    `max_length - 1`. It sets the logit of the `eos_token_id` to 0 (probability 1)
    and all other logits to `filter_value` (-infinity).

    Args:
        max_length: The maximum allowed sequence length (including prompt).
        eos_token_id: The integer ID of the End-Of-Sequence token.
    """

    max_length: int
    eos_token_id: int

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        new_scores = jnp.full(scores.shape, -float("inf"))
        apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)
        scores = jnp.where(apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores)
        return scores


@auto_pytree
class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the generation of the end-of-sequence (EOS)
    token until a minimum sequence length (`min_length`) has been reached.

    This processor sets the logit of the `eos_token_id` to `filter_value` (-infinity)
    if the current sequence length `cur_len` is less than `min_length`.

    Args:
        min_length: The minimum number of tokens that must be generated before the
            EOS token is allowed.
        eos_token_id: The integer ID of the End-Of-Sequence token.
    """

    min_length: int
    eos_token_id: int

    def __post_init__(self):
        if not isinstance(self.min_length, int) or self.min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {self.min_length}")

        if not isinstance(self.eos_token_id, int) or self.eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {self.eos_token_id},")

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)
        scores = jnp.where(apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores)
        return scores


@auto_pytree
class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that suppresses a specified list of tokens only at a specific
    early step in the generation process.

    This is useful for preventing certain tokens (like BOS) from being generated
    immediately after the prompt.

    The suppression occurs only when `cur_len` equals `begin_index`. The logits of the
    `begin_suppress_tokens` are set to `filter_value` (-infinity) at that step.

    Args:
        begin_suppress_tokens: A list or tuple of token IDs to suppress at the start.
        begin_index: The generation step index (0-based relative to the start of generation)
            at which to apply the suppression.
    """

    begin_suppress_tokens: list
    begin_index: int

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)
        scores = jnp.where(apply_penalty, scores.at[:, self.begin_suppress_tokens].set(-float("inf")), scores)
        return scores


@auto_pytree
class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that suppresses a specified list of tokens throughout the
    entire generation process.

    This processor sets the logits of the `suppress_tokens` to `filter_value` (-infinity)
    at every generation step where the list is not empty.

    Args:
        suppress_tokens: A list or tuple of token IDs to suppress consistently.
    """

    suppress_tokens: list

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        if len(self.suppress_tokens) != 0:
            scores = scores.at[..., self.suppress_tokens].set(-float("inf"))
        return scores


class ForceTokensLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that forces specific tokens to be generated at predefined
    positions during the generation.

    This processor uses a mapping (`force_token_map`) where keys are the generation
    indices (0-based, relative to the start of generation) and values are the token IDs
    to be forced at those indices.

    When the current generation step `cur_len` matches an index in the map, the logit
    of the corresponding forced token ID is set to 0 (probability 1), and all other
    logits are set to `filter_value` (-infinity).

    Args:
        force_token_map: A mapping from generation index to the token ID to force.
            Can be provided as a `dict` or a `list` of `(index, token_id)` pairs.
    """

    def __init__(self, force_token_map):
        assert isinstance(force_token_map, dict)
        force_token_array = jnp.ones((max(force_token_map.keys()) + 1), dtype=jnp.int32) * -1
        for index, token in force_token_map.items():
            if token is not None:
                force_token_array = force_token_array.at[index].set(token)
        self.force_token_array = jnp.int32(force_token_array)

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _force_token(generation_idx):
            batch_size = scores.shape[0]
            current_token = self.force_token_array[generation_idx]

            new_scores = jnp.ones_like(scores, dtype=scores.dtype) * -float("inf")
            updates = jnp.zeros((batch_size, 1), dtype=scores.dtype)
            new_scores = lax.dynamic_update_slice(new_scores, updates, (0, current_token))
            return new_scores

        scores = lax.cond(
            cur_len >= self.force_token_array.shape[0],
            lambda: scores,
            lambda: lax.cond(
                self.force_token_array[cur_len] >= 0,
                lambda: _force_token(cur_len),
                lambda: scores,
            ),
        )
        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""
    A specialized [`LogitsProcessor`] tailored for handling timestamp tokens during
    generation with Whisper-style models used for Automatic Speech Recognition (ASR).

    It enforces several constraints specific to timestamp prediction:
    1. **Suppresses `<|notimestamps|>`:** Prevents the model from predicting the token
       that indicates the absence of timestamps.
    2. **Alternating Tokens:** Enforces that text tokens and timestamp tokens generally
       alternate. If the last generated token was a timestamp, it biases against
       predicting another timestamp immediately after (unless it's the very beginning
       or certain edge cases).
    3. **Initial Timestamp Limit:** Restricts the maximum value of the *first* timestamp
       token predicted using `max_initial_timestamp_index`.
    4. **Timestamp Probability Check:** If the total probability mass assigned to all
       valid timestamp tokens is higher than the probability of the single most likely
       *non-timestamp* token, it forces the model to sample a timestamp token by
       suppressing all non-timestamp tokens.

    Note:
        This processor assumes the existence of specific token IDs related to timestamps
        (e.g., `eos_token_id`, `no_timestamps_token_id`, `timestamp_begin`) which are
        typically defined in the model's generation configuration.

    Args:
        generate_config: Configuration object containing Whisper-specific generation parameters
            like `eos_token_id`, `no_timestamps_token_id`, `is_multilingual`,
            `max_initial_timestamp_index`.
        model_config: The model's configuration (used for `vocab_size` as a fallback).
        decoder_input_length: The length of the initial input sequence provided to the decoder
            (e.g., the prompt length).
    """

    def __init__(self, generate_config, model_config, decoder_input_length):
        self.eos_token_id = generate_config.eos_token_id
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1

        self.begin_index = decoder_input_length + 1

        if generate_config.is_multilingual:
            self.begin_index += 2
        if hasattr(generate_config, "max_initial_timestamp_index"):
            self.max_initial_timestamp_index = generate_config.max_initial_timestamp_index
        else:
            self.max_initial_timestamp_index = model_config.vocab_size
        if self.max_initial_timestamp_index is None:
            self.max_initial_timestamp_index = model_config.vocab_size

    def __call__(self, input_ids, scores, cur_len):
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores = scores.at[:, self.no_timestamps_token_id].set(-float("inf"))

        def handle_pairs(input_ids_k, scores_k):
            last_was_timestamp = jnp.where((cur_len - self.begin_index) >= 1, True, False)
            last_was_timestamp = jnp.where(
                input_ids_k[cur_len - 1] >= self.timestamp_begin,
                True and last_was_timestamp,
                False,
            )

            penultimate_was_timestamp = jnp.where((cur_len - self.begin_index) < 2, True, False)
            penultimate_was_timestamp = jnp.where(
                input_ids_k[cur_len - 2] >= self.timestamp_begin,
                True,
                penultimate_was_timestamp,
            )

            return jnp.where(
                last_was_timestamp,
                jnp.where(
                    penultimate_was_timestamp > 0,
                    scores_k.at[self.timestamp_begin :].set(-float("inf")),
                    scores_k.at[: self.eos_token_id].set(-float("inf")),
                ),
                scores_k,
            )

        scores = jax.vmap(handle_pairs)(input_ids, scores)

        apply_max_initial_timestamp = jnp.where(cur_len == self.begin_index, True, False)
        apply_max_initial_timestamp = jnp.where(
            self.max_initial_timestamp_index is not None,
            True and apply_max_initial_timestamp,
            False,
        )

        last_allowed = self.timestamp_begin + self.max_initial_timestamp_index

        scores = jnp.where(
            apply_max_initial_timestamp,
            scores.at[:, last_allowed + 1 :].set(-float("inf")),
            scores,
        )

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = jax.nn.log_softmax(scores, axis=-1)

        def handle_cumulative_probs(logprobs_k, scores_k):
            timestamp_logprob = jax.nn.logsumexp(logprobs_k[self.timestamp_begin :], axis=-1)
            max_text_token_logprob = jnp.max(logprobs_k[: self.timestamp_begin])
            return jnp.where(
                timestamp_logprob > max_text_token_logprob,
                scores_k.at[: self.timestamp_begin].set(-float("inf")),
                scores_k,
            )

        scores = jax.vmap(handle_cumulative_probs)(logprobs, scores)

        return scores


@auto_pytree
class PresencePenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that penalizes tokens based on their presence in the sequence
    generated so far (`input_ids`).

    This processor subtracts a fixed `presence_penalty` value from the logits of all
    tokens that have appeared at least once in the `input_ids`.
    Positive penalties discourage the model from reusing tokens, promoting topic diversity.

    Args:
        presence_penalty: The penalty value subtracted from the logits of present tokens.
            Must be non-negative. Defaults to 0.0 (no penalty).
    """

    presence_penalty: float

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _apply(x, ids, presence_penalty):
            org_dtype = x.dtype
            batch_size, vocab_size = x.shape
            one_hot_presence = jax.nn.one_hot(ids, num_classes=vocab_size, dtype=x.dtype)
            presence_mask = jnp.sum(one_hot_presence, axis=1) > 0
            penalty_values = jnp.where(presence_mask, presence_penalty, 0.0)
            x = x - penalty_values
            return x.astype(org_dtype)

        return jax.lax.cond(
            self.presence_penalty == 0.0,
            lambda *x: x[0],
            _apply,
            scores,
            input_ids,
            self.presence_penalty,
        )


@auto_pytree
class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that penalizes tokens based on their frequency (number of
    occurrences) in the sequence generated so far (`input_ids`).

    This processor subtracts a penalty proportional to the token's count from its logit.
    The penalty is calculated as `count * frequency_penalty`.
    Positive penalties discourage the model from repeating specific tokens frequently.

    Args:
        frequency_penalty: The penalty factor. Must be non-negative.
            Defaults to 0.0 (no penalty).
    """

    frequency_penalty: float

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _apply(x, ids, frequency_penalty):
            org_dtype = x.dtype
            batch_size, vocab_size = x.shape
            one_hot_counts = jax.nn.one_hot(ids, num_classes=vocab_size, dtype=x.dtype)
            token_counts = jnp.sum(one_hot_counts, axis=1)
            penalty_values = token_counts * frequency_penalty
            x = x - penalty_values
            return x.astype(org_dtype)

        return jax.lax.cond(
            self.frequency_penalty == 0.0,
            lambda *x: x[0],
            _apply,
            scores,
            input_ids,
            self.frequency_penalty,
        )


@auto_pytree
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that applies a multiplicative penalty to the logits of tokens
    that have already appeared in the generated sequence (`input_ids`).

    For previously seen tokens:
    - If the original logit is positive, it's divided by `repetition_penalty`.
    - If the original logit is negative, it's multiplied by `repetition_penalty`.

    This aims to discourage repetition.

    Reference: [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858)
    by Keskar et al. (2019).

    Args:
        repetition_penalty: The penalty factor. Must be positive.
            - 1.0 means no penalty.
            - Values > 1.0 discourage repetition.
            - Values < 1.0 encourage repetition.
            Defaults to 1.0.
    """

    repetition_penalty: float

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def _apply(x, ids, repetition_penalty):
            org_dtype = x.dtype
            batch_size, vocab_size = x.shape
            one_hot_presence = jax.nn.one_hot(ids, num_classes=vocab_size, dtype=x.dtype)
            presence_mask = jnp.sum(one_hot_presence, axis=1) > 0
            positive_penalized_scores = x / repetition_penalty
            negative_penalized_scores = x * repetition_penalty
            scores_intermediate = jnp.where(x > 0, positive_penalized_scores, scores)
            penalized_scores = jnp.where(x < 0, negative_penalized_scores, scores_intermediate)
            x = jnp.where(presence_mask, penalized_scores, x)
            return x.astype(org_dtype)

        return jax.lax.cond(
            self.repetition_penalty == 1.0,
            lambda *x: x[0],
            _apply,
            scores,
            input_ids,
            self.repetition_penalty,
        )


@auto_pytree
class MinPLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] implementing min-p sampling.

    Filters the vocabulary distribution by removing tokens whose probability `P(token)`
    is less than `min_p` times the probability of the most likely token `P(max)`. That is,
    it keeps tokens where `P(token) >= min_p * P(max)`.

    This is an alternative filtering strategy to top-p or top-k.

    Args:
        min_p: The minimum probability threshold relative to the peak probability.
            Must be in [0, 1]. Setting `min_p=0.0` disables the filter.
        filter_value: The value assigned to the logits of filtered tokens.
            Defaults to -infinity.
        min_tokens_to_keep: Minimum number of tokens to retain, even if their
            probability falls below the `min_p * P(max)` threshold. Defaults to 1.
    """

    min_p: float
    filter_value: float = -float("Inf")
    min_tokens_to_keep: int = 1

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        min_p = jnp.asarray(self.min_p)

        def _apply(x):
            batch_size, vocab_size = x.shape
            sorted_logits, sorted_indices = lax.top_k(x, k=vocab_size)
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            shifted_cum_probs = jnp.pad(cumulative_probs[:, :-1], ((0, 0), (1, 0)), constant_values=0.0)
            sorted_indices_to_remove_mask = shifted_cum_probs >= min_p
            min_keep_mask = jnp.arange(vocab_size) < self.min_tokens_to_keep
            sorted_indices_to_remove_mask = sorted_indices_to_remove_mask & (~min_keep_mask)
            indices_to_remove_mask = jnp.zeros_like(x, dtype=jnp.bool_)
            batch_idx_mesh, _ = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(vocab_size), indexing="ij")
            update_indices = (batch_idx_mesh, sorted_indices)
            indices_to_remove_mask = indices_to_remove_mask.at[update_indices].set(sorted_indices_to_remove_mask)
            final_scores = jnp.where(indices_to_remove_mask, self.filter_value, x)
            return final_scores

        return jax.lax.cond((min_p > 0) & (min_p < 1), _apply, lambda x: x, scores)


@auto_pytree
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the generation of n-grams that have already
    occurred in the sequence generated so far (`input_ids`).

    At each step, it considers the last `ngram_size - 1` tokens generated. It then identifies
    all tokens in the vocabulary that would complete an n-gram already present in the full
    `input_ids` sequence. The logits for these banned tokens are set to `filter_value` (-infinity).

    Reference: [Fairseq Sequence Generator](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345)

    Args:
        ngram_size: The size of the n-gram to prevent from repeating. Setting `ngram_size=0`
            disables the processor.
    """

    ngram_size: int

    def forward(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        def true_fn():
            def get_previous_ngrams(input_ids: jnp.ndarray, vocab_size: int, cur_len: int):
                batch_size, seq_len = input_ids.shape
                seq_ngrams = seq_len - (self.ngram_size - 1)
                cur_ngrams = cur_len - (self.ngram_size - 1)

                def body_fun(i, val):
                    b = i % batch_size
                    pos = i // batch_size
                    return val.at[i].set(
                        jnp.array([b] + [jnp.array(input_ids)[b, pos + j] for j in range(self.ngram_size)])
                    )

                shape = (batch_size * seq_ngrams, self.ngram_size + 1)
                all_update_indices = jax.lax.fori_loop(
                    0,
                    batch_size * cur_ngrams,
                    body_fun,
                    jnp.zeros(shape, dtype=input_ids.dtype),
                )

                data = (jnp.arange(batch_size * seq_ngrams) < batch_size * cur_ngrams).astype("float32")

                return sparse.BCOO(
                    (data, all_update_indices),
                    shape=(batch_size,) + (vocab_size,) * self.ngram_size,
                )

            def get_banned_tokens_mask(
                latest_tokens: jnp.ndarray,
                previous_ngrams,
            ) -> jnp.ndarray:
                """
                Determines which tokens must be banned given latest tokens and the previously seen
                ngrams.
                """

                @sparse.sparsify
                @jax.vmap
                def inner_fn(latest_tokens, previous_ngrams):
                    return previous_ngrams[tuple(latest_tokens)]

                return sparse.bcoo_todense(inner_fn(latest_tokens, previous_ngrams))

            _, vocab_size = scores.shape
            previous_ngrams = get_previous_ngrams(input_ids, vocab_size, cur_len)
            latest_tokens = jnp.zeros(
                (input_ids.shape[0], self.ngram_size - 1),
                dtype=input_ids.dtype,
            )
            latest_tokens = jax.lax.dynamic_update_slice(
                latest_tokens,
                jax.lax.dynamic_slice(
                    input_ids,
                    (0, cur_len - (self.ngram_size - 1)),
                    (input_ids.shape[0], (self.ngram_size - 1)),
                ),
                (0, 0),
            )
            banned_tokens_indices_mask = get_banned_tokens_mask(latest_tokens, previous_ngrams).astype("b1")
            return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)

        output = jax.lax.cond((cur_len >= self.ngram_size - 1), true_fn, lambda: scores)
        return output

    def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int) -> jnp.ndarray:
        return jax.lax.cond(
            self.ngram_size != 0,
            lambda a, b, c: self.forward(a, b, c),
            lambda a, b, c: b,
            input_ids,
            scores,
            cur_len,
        )
