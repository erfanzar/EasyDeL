# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


import inspect

import jax
import jax.lax as lax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from jax.experimental import sparse

from easydel.utils.compiling_utils import hash_fn
from easydel.utils.helpers import get_logger


def add_start_docstrings(*docstr):
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


@auto_pytree
class LogitsProcessor:
	"""Abstract base class for all logit processors that can be applied during generation."""

	@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
	def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
		"""Flax method for processing logits."""
		raise NotImplementedError(
			f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
		)

	__hash__ = hash_fn


@auto_pytree
class LogitsWarper:
	"""Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

	@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
	def __call__(self, input_ids: jnp.ndarray, scores: jnp.ndarray) -> jnp.ndarray:
		"""Flax method for warping logits."""
		raise NotImplementedError(
			f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
		)

	__hash__ = hash_fn


@auto_pytree
class EmptyProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
	to be `-inf` so they are not sampled.

	Args:
	    suppress_tokens (`list`):
	        Tokens to not sample.
	"""

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		return scores


class LogitsProcessorList(list):
	"""
	This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process
	a `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
	[`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
	"""

	@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
	def __call__(
		self, input_ids: jnp.ndarray, scores: jnp.ndarray, cur_len: int, **kwargs
	) -> jnp.ndarray:
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
	[`LogitsWarper`] for temperature (exponential scaling output probability distribution).

	Args:
	    temperature (`float`):
	        The value used to module the logits distribution.
	"""

	temperature: float

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		return jax.lax.cond(
			self.temperature != 0.0,
			lambda x, temp: x / temp,
			lambda *x: x[0],
			scores,
			self.temperature,
		)


@auto_pytree
class TopPLogitsWarper(LogitsWarper):
	"""
	[`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.

	Args:
	    top_p (`float`):
	        If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
	        higher are kept for generation.
	    filter_value (`float`, *optional*, defaults to -inf):
	        All filtered values will be set to this float value.
	    min_tokens_to_keep (`int`, *optional*, defaults to 1):
	        Minimum number of tokens that cannot be filtered.
	"""

	top_p: float
	filter_value: float = -float("Inf")
	min_tokens_to_keep: int = 1

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
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

		return jax.lax.cond(
			(top_p > 0) & (top_p < 1),
			_apply,
			lambda x: x,
			scores,
		)


@auto_pytree
class TopKLogitsWarper(LogitsWarper):
	r"""
	[`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

	Args:
	    top_k (`int`):
	        The number of highest probability vocabulary tokens to keep for top-k-filtering.
	    filter_value (`float`, *optional*, defaults to -inf):
	        All filtered values will be set to this float value.
	    min_tokens_to_keep (`int`, *optional*, defaults to 1):
	        Minimum number of tokens that cannot be filtered.
	"""

	top_k: int
	filter_value: float = -float("Inf")
	min_tokens_to_keep: int = 1

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
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
	[`LogitsProcessor`] that enforces the specified token as the first generated token.

	Args:
	    bos_token_id (`int`):
	        The id of the token to force as the first generated token.
	"""

	bos_token_id: int

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		new_scores = jnp.full(scores.shape, -float("inf"))

		apply_penalty = 1 - jnp.bool_(cur_len - 1)

		scores = jnp.where(
			apply_penalty, new_scores.at[:, self.bos_token_id].set(0), scores
		)

		return scores


@auto_pytree
class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] that enforces the specified token as the last generated token when `max_length` is reached.

	Args:
	    max_length (`int`):
	        The maximum length of the sequence to be generated.
	    eos_token_id (`int`):
	        The id of the token to force as the last generated token when `max_length` is reached.
	"""

	max_length: int
	eos_token_id: int

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		new_scores = jnp.full(scores.shape, -float("inf"))

		apply_penalty = 1 - jnp.bool_(cur_len - self.max_length + 1)

		scores = jnp.where(
			apply_penalty, new_scores.at[:, self.eos_token_id].set(0), scores
		)

		return scores


@auto_pytree
class MinLengthLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.

	Args:
	    min_length (`int`):
	        The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
	    eos_token_id (`int`):
	        The id of the *end-of-sequence* token.
	"""

	min_length: int
	eos_token_id: int

	def __post_init__(self):
		if not isinstance(self.min_length, int) or self.min_length < 0:
			raise ValueError(
				f"`min_length` has to be a positive integer, but is {self.min_length}"
			)

		if not isinstance(self.eos_token_id, int) or self.eos_token_id < 0:
			raise ValueError(
				f"`eos_token_id` has to be a positive integer, but is {self.eos_token_id},"
			)

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		# create boolean flag to decide if min length penalty should be applied
		apply_penalty = 1 - jnp.clip(cur_len - self.min_length, 0, 1)

		scores = jnp.where(
			apply_penalty, scores.at[:, self.eos_token_id].set(-float("inf")), scores
		)

		return scores


@auto_pytree
class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] supressing a list of tokens as soon as the `generate` function starts generating using
	`begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the
	begining of the generation.

	Args:
	    begin_suppress_tokens (`List[int]`):
	        Tokens to not sample.
	    begin_index (`int`):
	        Index where the tokens are suppressed.
	"""

	begin_suppress_tokens: list
	begin_index: int

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		apply_penalty = 1 - jnp.bool_(cur_len - self.begin_index)

		scores = jnp.where(
			apply_penalty,
			scores.at[:, self.begin_suppress_tokens].set(-float("inf")),
			scores,
		)

		return scores


@auto_pytree
class SuppressTokensLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] suppressing a list of tokens at each decoding step. The processor will set their log probs
	to be `-inf` so they are not sampled.

	Args:
	    suppress_tokens (`list`):
	        Tokens to not sample.
	"""

	suppress_tokens: list

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		if len(self.suppress_tokens) != 0:
			scores = scores.at[..., self.suppress_tokens].set(-float("inf"))
		return scores


@auto_pytree
class ForceTokensLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] that takes a list of pairs of integers which indicates a mapping from generation indices to
	token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens
	to `-inf` so that they are sampled at their corresponding index.

	Args:
	    force_token_map (`list`):
	        Map giving token ids and indices where they will be forced to be sampled.
	"""

	force_token_map: list | dict
	force_token_array: jax.Array

	def __post_init__(self):
		force_token_map = dict(self.force_token_map)
		force_token_array = (
			jnp.ones((max(force_token_map.keys()) + 1), dtype=jnp.int32) * -1
		)
		for index, token in force_token_map.items():
			if token is not None:
				force_token_array = force_token_array.at[index].set(token)
		self.force_token_array = jnp.int32(force_token_array)

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
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
	Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log
	probs to `inf` so that they are sampled at their corresponding index.

	Args:
	    generate_config (`GenerateConfig`):
	        The generate config used to generate the output. The following parameters are required:
	            eos_token_id (`int`, *optional*, defaults to 50257):
	                The id of the *end-of-sequence* token.
	            no_timestamps_token_id (`int`, *optional*, defaults to 50363):
	                The id of the `"<|notimestamps|>"` token.
	            max_initial_timestamp_index (`int`, *optional*, defaults to 1):
	                Used to set the maximum value of the initial timestamp. This is used to prevent the model from
	                predicting timestamps that are too far in the future.
	"""

	def __post_init__(self, generate_config, model_config, decoder_input_length):
		self.eos_token_id = generate_config.eos_token_id
		self.no_timestamps_token_id = generate_config.no_timestamps_token_id
		self.timestamp_begin = generate_config.no_timestamps_token_id + 1

		self.begin_index = decoder_input_length + 1

		if generate_config.is_multilingual:
			# room for language token and task token
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

			penultimate_was_timestamp = jnp.where(
				(cur_len - self.begin_index) < 2, True, False
			)
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
	[`LogitsProcessor`] that applies a penalty to the logit of tokens that have already appeared in the
	`input_ids`.

	Args:
	    presence_penalty (`float`):
	        The penalty value. It is subtracted from the logits of tokens that are present in the `input_ids`.
	        A positive value penalizes new tokens according to whether they appear in the text so far,
	        increasing the model's likelihood to talk about new topics. Must be >= 0.
	"""

	presence_penalty: float

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		def _apply(x, ids, presence_penalty):
			batch_size, vocab_size = x.shape
			one_hot_presence = jax.nn.one_hot(
				ids,
				num_classes=vocab_size,
				dtype=x.dtype,
			)
			presence_mask = jnp.sum(one_hot_presence, axis=1) > 0
			penalty_values = jnp.where(presence_mask, presence_penalty, 0.0)
			x = x - penalty_values

			return x

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
	[`LogitsProcessor`] that applies a penalty based on the frequency of tokens that have already
	appeared in the `input_ids`.

	Args:
	    frequency_penalty (`float`):
	        The penalty value. It is subtracted from the logits proportionally to the frequency of the
	        token in the `input_ids`. A positive value penalizes new tokens based on their existing
	        frequency in the text so far, decreasing the model's likelihood to repeat the same line
	        verbatim. Must be >= 0.
	"""

	frequency_penalty: float

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		def _apply(x, ids, frequency_penalty):
			batch_size, vocab_size = x.shape

			one_hot_counts = jax.nn.one_hot(
				ids,
				num_classes=vocab_size,
				dtype=x.dtype,
			)
			token_counts = jnp.sum(one_hot_counts, axis=1)
			penalty_values = token_counts * frequency_penalty
			x = x - penalty_values

			return x

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
	[`LogitsProcessor`] enforcing an exponential penalty on repeated sequences.

	Args:
	    repetition_penalty (`float`):
	        The parameter for repetition penalty. 1.0 means no penalty. Values > 1.0 discourage repetition,
	        values < 1.0 encourage it. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
	"""

	repetition_penalty: float

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		def _apply(x, ids, repetition_penalty):
			batch_size, vocab_size = x.shape
			one_hot_presence = jax.nn.one_hot(
				ids,
				num_classes=vocab_size,
				dtype=x.dtype,
			)
			presence_mask = jnp.sum(one_hot_presence, axis=1) > 0
			positive_penalized_scores = x / repetition_penalty
			negative_penalized_scores = x * repetition_penalty

			scores_intermediate = jnp.where(x > 0, positive_penalized_scores, scores)
			penalized_scores = jnp.where(
				x < 0,
				negative_penalized_scores,
				scores_intermediate,
			)

			x = jnp.where(presence_mask, penalized_scores, x)

			return x

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
	[`LogitsWarper`] that performs nucleus filtering, also known as top-p sampling.
	Filters vocabulary based on cumulative probability threshold `min_p`.

	Args:
	    min_p (`float`):
	        If set to < 1, only the most probable tokens with probabilities that add up to `min_p` or higher are
	        kept for generation.
	    filter_value (`float`, *optional*, defaults to -inf):
	        All filtered values will be set to this float value.
	    min_tokens_to_keep (`int`, *optional*, defaults to 1):
	        Minimum number of tokens that cannot be filtered, even if their cumulative probability is above `min_p`.
	"""

	min_p: float
	filter_value: float = -float("Inf")
	min_tokens_to_keep: int = 1

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		min_p = jnp.asarray(self.min_p)

		def _apply(x):
			batch_size, vocab_size = x.shape
			sorted_logits, sorted_indices = lax.top_k(x, k=vocab_size)
			sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
			cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

			shifted_cum_probs = jnp.pad(
				cumulative_probs[:, :-1],
				((0, 0), (1, 0)),
				constant_values=0.0,
			)
			sorted_indices_to_remove_mask = shifted_cum_probs >= min_p
			min_keep_mask = jnp.arange(vocab_size) < self.min_tokens_to_keep
			sorted_indices_to_remove_mask = sorted_indices_to_remove_mask & (~min_keep_mask)
			indices_to_remove_mask = jnp.zeros_like(x, dtype=jnp.bool_)
			batch_idx_mesh, _ = jnp.meshgrid(
				jnp.arange(batch_size),
				jnp.arange(vocab_size),
				indexing="ij",
			)
			update_indices = (batch_idx_mesh, sorted_indices)
			indices_to_remove_mask = indices_to_remove_mask.at[update_indices].set(
				sorted_indices_to_remove_mask
			)

			final_scores = jnp.where(indices_to_remove_mask, self.filter_value, x)

			return final_scores

		return jax.lax.cond(
			(min_p > 0) & (min_p < 1),
			_apply,
			lambda x: x,
			scores,
		)


@auto_pytree
class NoRepeatNGramLogitsProcessor(LogitsProcessor):
	r"""
	[`LogitsProcessor`] that enforces no repetition of n-grams. See
	[Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

	Args:
	    ngram_size (`int`):
	        All ngrams of size `ngram_size` can only occur once.
	"""

	ngram_size: int

	def forward(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		def true_fn():
			def get_previous_ngrams(
				input_ids: jnp.ndarray,
				vocab_size: int,
				cur_len: int,
			):
				batch_size, seq_len = input_ids.shape
				seq_ngrams = seq_len - (self.ngram_size - 1)
				cur_ngrams = cur_len - (self.ngram_size - 1)

				def body_fun(i, val):
					b = i % batch_size
					pos = i // batch_size
					return val.at[i].set(
						jnp.array(
							[
								b,
							]
							+ [jnp.array(input_ids)[b, pos + j] for j in range(self.ngram_size)]
						)
					)

				shape = (batch_size * seq_ngrams, self.ngram_size + 1)
				all_update_indices = jax.lax.fori_loop(
					0,
					batch_size * cur_ngrams,
					body_fun,
					jnp.zeros(shape, dtype=input_ids.dtype),
				)

				data = (jnp.arange(batch_size * seq_ngrams) < batch_size * cur_ngrams).astype(
					"float32"
				)

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
			banned_tokens_indices_mask = get_banned_tokens_mask(
				latest_tokens, previous_ngrams
			).astype("b1")
			return jnp.where(banned_tokens_indices_mask, -float("inf"), scores)

		output = jax.lax.cond((cur_len >= self.ngram_size - 1), true_fn, lambda: scores)
		return output

	def __call__(
		self,
		input_ids: jnp.ndarray,
		scores: jnp.ndarray,
		cur_len: int,
	) -> jnp.ndarray:
		return jax.lax.cond(
			self.ngram_size != 0,
			lambda a, b, c: self.forward(a, b, c),
			lambda a, b, c: b,
			input_ids,
			scores,
			cur_len,
		)
