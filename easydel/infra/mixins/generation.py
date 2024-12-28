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
import copy
import inspect
import typing as tp
import warnings
from functools import partial

import chex
import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers.generation.configuration_utils import GenerationConfig

from easydel.etils.etils import get_logger
from easydel.inference.logits_process import (
	FlaxForcedBOSTokenLogitsProcessor,
	FlaxForcedEOSTokenLogitsProcessor,
	FlaxForceTokensLogitsProcessor,
	FlaxLogitsProcessorList,
	FlaxMinLengthLogitsProcessor,
	FlaxNoRepeatNGramLogitsProcessor,
	FlaxSuppressTokensAtBeginLogitsProcessor,
	FlaxSuppressTokensLogitsProcessor,
	FlaxTemperatureLogitsWarper,
	FlaxTopKLogitsWarper,
	FlaxTopPLogitsWarper,
)
from easydel.layers.caching.transformer_cache import (
	TransformerCache,
	TransformerCacheMetaData,
)
from easydel.utils.quantizers import EasyQuantizer

from ..base_config import EasyDeLBaseConfig
from ..modeling_outputs import (
	FlaxBeamSearchOutput,
	FlaxGreedySearchOutput,
	FlaxSampleOutput,
)

logger = get_logger(__name__)


@chex.dataclass
class GreedyState:
	"""
	State for greedy search generation.

	Attributes:
	    cur_len (chex.Array): Current length of the generated sequence.
	    sequences (chex.Array): Generated sequences so far.
	    running_token (chex.Array): Current token being processed.
	    is_sent_finished (chex.Array): Boolean array indicating if a sequence is finished.
	    model_kwargs (tp.Dict[str, chex.Array]): Model specific keyword arguments.
	"""

	cur_len: chex.Array
	sequences: chex.Array
	running_token: chex.Array
	is_sent_finished: chex.Array
	model_kwargs: tp.Dict[str, chex.Array]


@chex.dataclass
class SampleState:
	"""
	State for sampling generation.

	Attributes:
	    cur_len (chex.Array): Current length of the generated sequence.
	    sequences (chex.Array): Generated sequences so far.
	    running_token (chex.Array): Current token being processed.
	    is_sent_finished (chex.Array): Boolean array indicating if a sequence is finished.
	    prng_key (chex.Array): PRNG key for sampling.
	    model_kwargs (tp.Dict[str, chex.Array]): Model specific keyword arguments.
	"""

	cur_len: chex.Array
	sequences: chex.Array
	running_token: chex.Array
	is_sent_finished: chex.Array
	prng_key: chex.Array
	model_kwargs: tp.Dict[str, chex.Array]


@chex.dataclass
class BeamSearchState:
	"""
	State for beam search generation.

	Attributes:
	    cur_len (chex.Array): Current length of the generated sequence.
	    running_sequences (chex.Array): Generated sequences being tracked in the beam.
	    running_scores (chex.Array): Scores of the sequences being tracked in the beam.
	    sequences (chex.Array): Best generated sequences.
	    scores (chex.Array): Scores of the best generated sequences.
	    is_sent_finished (chex.Array): Boolean array indicating if a sequence is finished.
	    model_kwargs (tp.Dict[str, chex.Array]): Model specific keyword arguments.
	"""

	cur_len: chex.Array
	running_sequences: chex.Array
	running_scores: chex.Array
	sequences: chex.Array
	scores: chex.Array
	is_sent_finished: chex.Array
	model_kwargs: tp.Dict[str, chex.Array]


class EasyGenerationMixin:
	config_class: tp.Type[EasyDeLBaseConfig]
	config: EasyDeLBaseConfig
	base_model_prefix: str
	_model_task: tp.Optional[str] = None
	_model_type: tp.Optional[str] = None

	def init_cache(self, batch_size: int, max_length: int):
		head_dim = getattr(self.config, "head_dim", None)
		if head_dim is None:
			head_dim = self.config.hidden_size // self.config.num_attention_heads
		num_key_value_heads = getattr(self.config, "num_key_value_heads", None)
		if num_key_value_heads is None:
			num_key_value_heads = self.config.num_attention_heads
		return TransformerCache.init_layers_cache(
			num_hidden_layers=self.config.num_hidden_layers,
			dtype=self.dtype,
			key_values_partition_specs=PartitionSpec(
				self.config.partition_axis.batch_axis,
				self.config.partition_axis.key_sequence_axis,
				self.config.partition_axis.head_axis,
				self.config.partition_axis.attention_dim_axis,
			),
			metadata=TransformerCacheMetaData.create(
				batch_size=batch_size,
				sequence_length=max_length,
				num_heads=num_key_value_heads,
				head_dim=head_dim,
			),
			quantizer=EasyQuantizer(
				quantization_method=self.config.kv_cache_quantization_method,
				block_size=self.config.kv_cache_quantization_blocksize,
				quantization_platform=self.config.platform,
			),
		)

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		attention_mask: tp.Optional[chex.Array] = None,
	):
		"""The prepare_inputs_for_generation function is used to prepare the inputs for a generation task.

		Args:
		    self: Access variables that belong to the class
		    input_ids: Pass in the input tokens
		    max_length: Set the length of the sequence to be generated
		    attention_mask: tp.Optional[chex.Array]: Mask the attention
		        weights

		Returns:
		    A dictionary of the past_key_values, attention_mask and
		    position ids
		"""
		batch_size, seq_length = input_ids.shape
		past_key_values = self.init_cache(batch_size, max_length)

		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			position_ids = attention_mask.cumsum(axis=-1) - 1
			extended_attention_mask = lax.dynamic_update_slice(
				extended_attention_mask,
				attention_mask,
				(0, 0),
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :],
				(batch_size, seq_length),
			)

		return self.prepare_inputs_for_call(
			**{
				"past_key_values": past_key_values,
				"attention_mask": extended_attention_mask,
				"position_ids": position_ids,
			}
		)

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		model_kwargs["past_key_values"] = model_outputs.past_key_values
		model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
		return model_kwargs

	def _validate_signature(
		self,
		method,
		args: tuple,
		kwargs: tp.Dict[str, tp.Any],
	) -> tp.Dict[str, tp.Any]:
		"""
		Validates and filters arguments based on the method's signature.

		Args:
				method: The method to check signature against
				args: Positional arguments
				kwargs: Keyword arguments

		Returns:
				tp.Dict[str, tp.Any]: Filtered kwargs containing only valid parameters
		"""
		sig = inspect.signature(method)
		valid_params = sig.parameters

		args_as_kwargs = {}
		positional_params = [
			param
			for param in valid_params.values()
			if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
		]

		for i, arg in enumerate(args):
			if i < len(positional_params):
				args_as_kwargs[positional_params[i].name] = arg

		filtered_kwargs = {}
		for name, value in {**args_as_kwargs, **kwargs}.items():
			if name in valid_params:
				param = valid_params[name]
				if param.annotation != inspect.Parameter.empty:
					try:
						if (
							getattr(param.annotation, "__origin__", None) is tp.Optional
							and value is not None
						):
							expected_type = param.annotation.__args__[0]
							if not isinstance(value, expected_type):
								print(
									f"Warning: Parameter '{name}' expected type {expected_type}, "
									f"got {type(value)}. Skipping parameter."
								)
								continue
					except Exception:
						pass
				filtered_kwargs[name] = value
			else:
				warnings.warn(
					f"  Parameter '{name}' not found in child class signature. Skipping.",
					stacklevel=1,
				)

		return filtered_kwargs

	@staticmethod
	def _run_loop_in_debug(cond_fn, body_fn, init_state):
		"""
		Run generation in untraced mode. This should only be used for debugging purposes.
		"""
		state = init_state
		while cond_fn(state):
			state = body_fn(state)
		return state

	def _prepare_encoder_decoder_kwargs_for_generation(
		self,
		input_ids,
		model_kwargs,
	):
		encoder_kwargs = {
			argument: value
			for argument, value in model_kwargs.items()
			if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
		}
		model_kwargs["encoder_outputs"] = self.encode(
			input_ids,
			return_dict=True,
			**encoder_kwargs,
		)
		return model_kwargs

	def _prepare_decoder_input_ids_for_generation(
		self,
		batch_size: int,
		decoder_start_token_id: int = None,
		bos_token_id: int = None,
		model_kwargs: tp.Optional[tp.Dict[str, chex.Array]] = None,
	) -> chex.Array:
		if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
			decoder_input_ids = model_kwargs.pop("decoder_input_ids")
			if decoder_input_ids is not None:
				return decoder_input_ids
		decoder_start_token_id = self._get_decoder_start_token_id(
			decoder_start_token_id, bos_token_id
		)
		return (
			jnp.array(decoder_start_token_id, dtype="i4")
			.reshape(1, -1)
			.repeat(batch_size, axis=0)
		)

	def _get_decoder_start_token_id(
		self,
		decoder_start_token_id: int = None,
		bos_token_id: int = None,
	) -> int:
		decoder_start_token_id = (
			decoder_start_token_id
			if decoder_start_token_id is not None
			else self.generation_config.decoder_start_token_id
		)
		bos_token_id = (
			bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id
		)
		if decoder_start_token_id is not None:
			return decoder_start_token_id
		elif (
			hasattr(self.config, "decoder")
			and hasattr(self.config.decoder, "decoder_start_token_id")
			and self.config.decoder.decoder_start_token_id is not None
		):
			return self.config.decoder.decoder_start_token_id
		elif bos_token_id is not None:
			return bos_token_id
		elif (
			hasattr(self.config, "decoder")
			and hasattr(self.config.decoder, "bos_token_id")
			and self.config.decoder.bos_token_id is not None
		):
			return self.config.decoder.bos_token_id
		raise ValueError(
			"`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
		)

	@staticmethod
	def _expand_to_num_beams(tensor, num_beams):
		return jnp.broadcast_to(
			tensor[:, None], (tensor.shape[0], num_beams) + tensor.shape[1:]
		)

	def _adapt_logits_for_beam_search(self, logits):
		"""
		This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
		search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
		"""
		return logits

	def _validate_model_kwargs(self, model_kwargs: tp.Dict[str, tp.Any]):
		"""Validates model kwargs for generation. Generate argument typos will also be caught here."""
		unused_model_args = []
		model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)

		if "kwargs" in model_args or "model_kwargs" in model_args:
			model_args |= set(inspect.signature(self.__call__).parameters)
		for key, value in model_kwargs.items():
			if value is not None and key not in model_args:
				unused_model_args.append(key)

		if unused_model_args:
			raise ValueError(
				f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
				" generate arguments will also show up in this list)"
			)

	def generate(
		self,
		input_ids: chex.Array,
		generation_config: tp.Optional[GenerationConfig] = None,
		prng_key: tp.Optional[chex.Array] = None,
		trace: bool = True,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		**kwargs,
	):
		r"""
		Generates sequences of token ids for models with a language modeling head.

		Parameters:
				input_ids (`chex.Array` of shape `(batch_size, sequence_length)`):
						The sequence used as a prompt for the generation.
				generation_config (`~generation.GenerationConfig`, *optional*):
						The generation configuration to be used as base parametrization for the generation call. `**kwargs`
						passed to generate matching the attributes of `generation_config` will override them. If
						`generation_config` is not provided, the default will be used, which had the following loading
						priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
						configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
						default values, whose documentation should be checked to parameterize generation.
				trace (`bool`, *optional*, defaults to `True`):
						Whether to trace generation. Setting `trace=False` should only be used for debugging and will lead to a
						considerably slower runtime.
				logits_processor (`FlaxLogitsProcessorList `, *optional*):
						Custom logits processors that complement the default logits processors built from arguments and
						generation config. If a logit processor is passed that is already created with the arguments or a
						generation config an error is thrown. This feature is intended for advanced users.
				kwargs (`tp.Dict[str, Any]`, *optional*):
						Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
						forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
						specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

		Return:
				[`~utils.ModelOutput`].

		"""

		if generation_config is None:
			if (
				self.generation_config._from_model_config
				and self.generation_config._original_object_hash == hash(self.generation_config)
			):
				new_generation_config = GenerationConfig.from_model_config(self.config)
				if new_generation_config != self.generation_config:
					warnings.warn(
						"You have modified the pretrained model configuration to control generation. This is a"
						" deprecated strategy to control generation and will be removed soon, in a future version."
						" Please use and modify the model generation configuration (see"
						" https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
						stacklevel=1,
					)
					self.generation_config = new_generation_config
			generation_config = self.generation_config

		generation_config = copy.deepcopy(generation_config)
		model_kwargs = generation_config.update(**kwargs)
		self._validate_model_kwargs(model_kwargs.copy())

		logits_processor = (
			logits_processor if logits_processor is not None else FlaxLogitsProcessorList()
		)

		# set init values
		prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

		if (
			generation_config.pad_token_id is None
			and generation_config.eos_token_id is not None
		):
			if model_kwargs.get("attention_mask") is None:
				logger.warning(
					"The attention mask and the pad token id were not set. As a consequence, you may observe "
					"unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
				)
			eos_token_id = generation_config.eos_token_id
			if isinstance(eos_token_id, list):
				eos_token_id = eos_token_id[0]
			logger.warning(
				f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
			)
			generation_config.pad_token_id = eos_token_id

		if (
			generation_config.decoder_start_token_id is None
			and self.config.is_encoder_decoder
		):
			raise ValueError(
				"`decoder_start_token_id` has to be defined for encoder-decoder generation."
			)

		# decoder-only models should use left-padding for generation (can't be checked with `trace=True`)
		if not self.config.is_encoder_decoder and not trace:
			if (
				generation_config.pad_token_id is not None
				and jnp.sum(input_ids[:, -1] == generation_config.pad_token_id) > 0
			):
				logger.warning(
					"A decoder-only architecture is being used, but right-padding was detected! For correct "
					"generation results, please set `padding_side='left'` when initializing the tokenizer."
				)

		batch_size = input_ids.shape[0]

		if self.config.is_encoder_decoder:
			# add encoder_outputs to model_kwargs
			if model_kwargs.get("encoder_outputs") is None:
				model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
					input_ids,
					model_kwargs,
				)
			# prepare decoder_input_ids for generation
			input_ids = self._prepare_decoder_input_ids_for_generation(
				batch_size,
				decoder_start_token_id=generation_config.decoder_start_token_id,
				bos_token_id=generation_config.bos_token_id,
				model_kwargs=model_kwargs,
			)

		# Prepare `max_length` depending on other stopping criteria.
		input_ids_seq_length = input_ids.shape[-1]
		has_default_max_length = (
			kwargs.get("max_length") is None and generation_config.max_length is not None
		)
		if (
			has_default_max_length
			and generation_config.max_new_tokens is None
			and generation_config.max_length == 20
		):
			# 20 is the default max_length of the generation config
			warnings.warn(
				f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
				"to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation.",
				UserWarning,
				stacklevel=1,
			)
		elif generation_config.max_new_tokens is not None:
			if not has_default_max_length and generation_config.max_length is not None:
				logger.warning(
					f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
					f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
					"Please refer to the documentation for more information. "
					"(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
				)
			generation_config.max_length = (
				generation_config.max_new_tokens + input_ids_seq_length
			)
		else:  # by default let's always generate 10 new tokens
			if generation_config.max_length == GenerationConfig().max_length:
				generation_config.max_length = (
					generation_config.max_length + input_ids_seq_length
				)
				max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
				if max_position_embeddings is not None:
					generation_config.max_length = min(
						generation_config.max_length, max_position_embeddings
					)

		if (
			generation_config.min_length is not None
			and generation_config.min_length > generation_config.max_length
		):
			raise ValueError(
				f"Unfeasable length constraints: the minimum length ({generation_config.min_length}) is larger than"
				f" the maximum length ({generation_config.max_length})"
			)
		if input_ids_seq_length >= generation_config.max_length:
			input_ids_string = (
				"decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
			)
			logger.warning(
				f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
				f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
				" increasing`max_new_tokens`."
			)

		logits_processor = self._get_logits_processor(
			generation_config=generation_config,
			input_ids_seq_length=input_ids_seq_length,
			logits_processor=logits_processor,
		)

		if not generation_config.do_sample and generation_config.num_beams == 1:
			return self._greedy_search(
				input_ids,
				generation_config.max_length,
				generation_config.pad_token_id,
				generation_config.eos_token_id,
				logits_processor=logits_processor,
				trace=trace,
				model_kwargs=model_kwargs,
			)
		elif generation_config.do_sample and generation_config.num_beams == 1:
			logits_warper = self._get_logits_warper(generation_config=generation_config)
			return self._sample(
				input_ids,
				generation_config.max_length,
				generation_config.pad_token_id,
				generation_config.eos_token_id,
				prng_key,
				logits_warper=logits_warper,
				logits_processor=logits_processor,
				trace=trace,
				model_kwargs=model_kwargs,
			)
		elif not generation_config.do_sample and generation_config.num_beams > 1:
			# broadcast input_ids & encoder_outputs
			input_ids = self._expand_to_num_beams(
				input_ids, num_beams=generation_config.num_beams
			)

			if "encoder_outputs" in model_kwargs:
				model_kwargs["encoder_outputs"]["last_hidden_state"] = (
					self._expand_to_num_beams(
						model_kwargs["encoder_outputs"]["last_hidden_state"],
						num_beams=generation_config.num_beams,
					)
				)

			for kwarg in ["attention_mask", "decoder_attention_mask"]:
				if kwarg in model_kwargs:
					model_kwargs[kwarg] = self._expand_to_num_beams(
						model_kwargs[kwarg], num_beams=generation_config.num_beams
					)

			return self._beam_search(
				input_ids,
				generation_config.max_length,
				generation_config.pad_token_id,
				generation_config.eos_token_id,
				length_penalty=generation_config.length_penalty,
				early_stopping=generation_config.early_stopping,
				logits_processor=logits_processor,
				trace=trace,
				num_return_sequences=generation_config.num_return_sequences,
				model_kwargs=model_kwargs,
			)
		else:
			raise NotImplementedError("`Beam sampling is currently not implemented.")

	def _get_logits_warper(
		self, generation_config: GenerationConfig
	) -> FlaxLogitsProcessorList:
		"""
		This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsWarper`]
		instances used for multinomial sampling.
		"""
		warpers = FlaxLogitsProcessorList()

		if (
			generation_config.temperature is not None and generation_config.temperature != 1.0
		):
			warpers.append(FlaxTemperatureLogitsWarper(generation_config.temperature))
		if generation_config.top_k is not None and generation_config.top_k != 0:
			warpers.append(
				FlaxTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1)
			)
		if generation_config.top_p is not None and generation_config.top_p < 1.0:
			warpers.append(
				FlaxTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1)
			)

		return warpers

	def _get_logits_processor(
		self,
		generation_config: GenerationConfig,
		input_ids_seq_length: int,
		logits_processor: tp.Optional[FlaxLogitsProcessorList],
	) -> FlaxLogitsProcessorList:
		"""
		This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsProcessor`]
		instances used to modify the scores of the language model head.
		"""
		processors = FlaxLogitsProcessorList()

		if (
			generation_config.min_length is not None
			and generation_config.min_length > 0
			and generation_config.eos_token_id is not None
			and generation_config.min_length > -1
		):
			processors.append(
				FlaxMinLengthLogitsProcessor(
					generation_config.min_length,
					generation_config.eos_token_id,
				)
			)
		if generation_config.forced_bos_token_id is not None:
			processors.append(
				FlaxForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id)
			)
		if generation_config.forced_eos_token_id is not None:
			processors.append(
				FlaxForcedEOSTokenLogitsProcessor(
					generation_config.max_length, generation_config.forced_eos_token_id
				)
			)
		if generation_config.suppress_tokens is not None:
			processors.append(
				FlaxSuppressTokensLogitsProcessor(generation_config.suppress_tokens)
			)
		if generation_config.begin_suppress_tokens is not None:
			begin_index = input_ids_seq_length
			begin_index = (
				begin_index
				if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
				else begin_index + 1
			)
			if (
				generation_config.forced_decoder_ids is not None
				and len(generation_config.forced_decoder_ids) > 0
			):
				# generation starts after the last token that is forced
				begin_index += generation_config.forced_decoder_ids[-1][0]
			processors.append(
				FlaxSuppressTokensAtBeginLogitsProcessor(
					generation_config.begin_suppress_tokens, begin_index
				)
			)
		if generation_config.forced_decoder_ids is not None:
			forced_decoder_ids = [
				[input_ids_seq_length + i[0] - 1, i[1]]
				for i in generation_config.forced_decoder_ids
			]
			processors.append(FlaxForceTokensLogitsProcessor(forced_decoder_ids))
		if (
			generation_config.no_repeat_ngram_size is not None
			and generation_config.no_repeat_ngram_size > 0
		):
			processors.append(
				FlaxNoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size)
			)
		processors = self._merge_criteria_processor_list(processors, logits_processor)

		return processors

	def _merge_criteria_processor_list(
		self,
		default_list: FlaxLogitsProcessorList,
		custom_list: FlaxLogitsProcessorList,
	) -> FlaxLogitsProcessorList:
		if len(custom_list) == 0:
			return default_list
		for default in default_list:
			for custom in custom_list:
				if type(custom) is type(default):
					object_type = "logits processor"
					raise ValueError(
						f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
						f" `generate`, but it has already been created with the values {default}. {default} has been"
						" created by passing the corresponding arguments to generate or by the model's config default"
						f" values. If you just want to change the default values of {object_type} consider passing"
						f" them as arguments to `generate` instead of using a custom {object_type}."
					)
		default_list.extend(custom_list)
		return default_list

	def _greedy_search(
		self,
		input_ids: None,
		max_length: tp.Optional[int] = None,
		pad_token_id: tp.Optional[int] = None,
		eos_token_id: tp.Optional[int] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		trace: bool = True,
		model_kwargs: tp.Optional[tp.Dict[str, chex.Array]] = None,
	):
		max_length = (
			max_length if max_length is not None else self.generation_config.max_length
		)
		pad_token_id = (
			pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
		)
		eos_token_id = (
			eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
		)

		batch_size, cur_len = input_ids.shape

		eos_token_id = jnp.array(
			eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None
		)
		pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
		cur_len = jnp.array(cur_len)

		sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
		sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

		is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

		model = self.decode if self.config.is_encoder_decoder else self
		model_kwargs = self.prepare_inputs_for_generation(
			input_ids, max_length, **model_kwargs
		)

		state = GreedyState(
			cur_len=cur_len,
			sequences=sequences,
			running_token=input_ids,
			is_sent_finished=is_sent_finished,
			model_kwargs=model_kwargs,
		)

		def greedy_search_cond_fn(state):
			"""state termination condition fn."""
			has_reached_max_length = state.cur_len == max_length
			all_sequence_finished = jnp.all(state.is_sent_finished)
			finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
			return ~finish_generation

		def greedy_search_body_fn(state):
			"""state update fn."""
			model_outputs = model(state.running_token, **state.model_kwargs)
			logits = model_outputs.logits[:, -1]

			logits = logits_processor(state.sequences, logits, state.cur_len)

			next_token = jnp.argmax(logits, axis=-1)

			next_token = (
				next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
			)
			next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
			next_token = next_token[:, None]

			next_sequences = lax.dynamic_update_slice(
				state.sequences, next_token, (0, state.cur_len)
			)
			next_model_kwargs = self.update_inputs_for_generation(
				model_outputs, state.model_kwargs
			)
			return GreedyState(
				cur_len=state.cur_len + 1,
				sequences=next_sequences,
				running_token=next_token,
				is_sent_finished=next_is_sent_finished,
				model_kwargs=next_model_kwargs,
			)

		if input_ids.shape[1] > 1:
			state = greedy_search_body_fn(state)

		if not trace:
			state = self._run_loop_in_debug(
				greedy_search_cond_fn, greedy_search_body_fn, state
			)
		else:
			state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

		return FlaxGreedySearchOutput(sequences=state.sequences)

	def _sample(
		self,
		input_ids: None,
		max_length: tp.Optional[int] = None,
		pad_token_id: tp.Optional[int] = None,
		eos_token_id: tp.Optional[int] = None,
		prng_key: tp.Optional[chex.Array] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		logits_warper: tp.Optional[FlaxLogitsProcessorList] = None,
		trace: bool = True,
		model_kwargs: tp.Optional[tp.Dict[str, chex.Array]] = None,
	):
		# init values
		max_length = (
			max_length if max_length is not None else self.generation_config.max_length
		)
		pad_token_id = (
			pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
		)
		eos_token_id = (
			eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
		)
		prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

		batch_size, cur_len = input_ids.shape

		eos_token_id = jnp.array(
			eos_token_id,
			dtype=jnp.int32 if eos_token_id is not None else None,
		)
		pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
		cur_len = jnp.array(cur_len)

		# per batch-item holding current token in loop.
		sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
		sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
		is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
		model = self.decode if self.config.is_encoder_decoder else self

		# initialize state
		state = SampleState(
			cur_len=cur_len,
			sequences=sequences,
			running_token=input_ids,
			is_sent_finished=is_sent_finished,
			prng_key=prng_key,
			model_kwargs=self.prepare_inputs_for_generation(
				input_ids,
				max_length,
				**model_kwargs,
			),
		)

		def sample_search_cond_fn(state):
			"""state termination condition fn."""
			has_reached_max_length = state.cur_len == max_length
			all_sequence_finished = jnp.all(state.is_sent_finished)
			finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
			return ~finish_generation

		def sample_search_body_fn(state):
			"""state update fn."""
			prng_key, prng_key_next = jax.random.split(state.prng_key)
			model_outputs = model(state.running_token, **state.model_kwargs)
			logits = model_outputs.logits[:, -1]
			logits = logits_processor(state.sequences, logits, state.cur_len)
			logits = logits_warper(logits, logits, state.cur_len)
			next_token = (
				jax.random.categorical(prng_key, logits, axis=-1) * ~state.is_sent_finished
				+ pad_token_id * state.is_sent_finished
			)
			next_is_sent_finished = state.is_sent_finished | jnp.isin(
				next_token,
				eos_token_id,
			)
			next_token = next_token[:, None]
			next_sequences = lax.dynamic_update_slice(
				state.sequences,
				next_token,
				(0, state.cur_len),
			)
			next_model_kwargs = self.update_inputs_for_generation(
				model_outputs,
				state.model_kwargs,
			)

			return SampleState(
				cur_len=state.cur_len + 1,
				sequences=next_sequences,
				running_token=next_token,
				is_sent_finished=next_is_sent_finished,
				model_kwargs=next_model_kwargs,
				prng_key=prng_key_next,
			)

		# The very first prompt often has sequence length > 1, so run outside of `lax.while_loop` to comply with TPU
		if input_ids.shape[1] > 1:
			state = sample_search_body_fn(state)

		if not trace:
			state = self._run_loop_in_debug(
				sample_search_cond_fn,
				sample_search_body_fn,
				state,
			)
		else:
			state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

		return FlaxSampleOutput(sequences=state.sequences)

	def _beam_search(
		self,
		input_ids: None,
		max_length: tp.Optional[int] = None,
		pad_token_id: tp.Optional[int] = None,
		eos_token_id: tp.Optional[int] = None,
		length_penalty: tp.Optional[float] = None,
		early_stopping: tp.Optional[tp.Union[bool, str]] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		trace: bool = True,
		num_return_sequences: tp.Optional[int] = None,
		model_kwargs: tp.Optional[tp.Dict[str, chex.Array]] = None,
	):
		"""
		This beam search function is heavily inspired by Flax's official example:
		https://github.com/google/flax/blob/main/examples/wmt/decode.py
		"""

		def flatten_beam_dim(tensor):
			"""Flattens the first two dimensions of a non-scalar array."""
			# ignore scalars (e.g. cache index)
			if tensor.ndim == 0:
				return tensor
			return tensor.reshape((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

		def unflatten_beam_dim(tensor, batch_size, num_beams):
			"""Unflattens the first, flat batch*beam dimension of a non-scalar array."""
			# ignore scalars (e.g. cache index)
			if tensor.ndim == 0:
				return tensor
			return tensor.reshape((batch_size, num_beams) + tensor.shape[1:])

		def gather_beams(nested, beam_indices, batch_size, new_num_beams):
			"""
			Gathers the beam slices indexed by beam_indices into new beam array.
			"""
			batch_indices = jnp.reshape(
				jnp.arange(batch_size * new_num_beams) // new_num_beams,
				(batch_size, new_num_beams),
			)

			def gather_fn(tensor):
				# ignore scalars (e.g. cache index)
				if tensor.ndim == 0:
					return tensor
				else:
					return tensor[batch_indices, beam_indices]

			return jax.tree_util.tree_map(gather_fn, nested)

		# init values
		max_length = (
			max_length if max_length is not None else self.generation_config.max_length
		)
		pad_token_id = (
			pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
		)
		eos_token_id = (
			eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
		)
		length_penalty = (
			length_penalty
			if length_penalty is not None
			else self.generation_config.length_penalty
		)
		early_stopping = (
			early_stopping
			if early_stopping is not None
			else self.generation_config.early_stopping
		)
		num_return_sequences = (
			num_return_sequences
			if num_return_sequences is not None
			else self.generation_config.num_return_sequences
		)

		batch_size, num_beams, cur_len = input_ids.shape

		eos_token_id = jnp.array(
			eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None
		)
		pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
		cur_len = jnp.array(cur_len)

		# record the prompt length of decoder
		decoder_prompt_len = input_ids.shape[-1]

		# per batch,beam-item holding current token in loop.
		sequences = jnp.full(
			(batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32
		)
		running_sequences = jnp.full(
			(batch_size, num_beams, max_length), pad_token_id, dtype=jnp.int32
		)
		running_sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0, 0))

		# per batch,beam-item state bit indicating if sentence has finished.
		is_sent_finished = jnp.zeros((batch_size, num_beams), dtype=jnp.bool_)

		# per batch,beam-item score, logprobs
		running_scores = jnp.tile(
			jnp.array([0.0] + [np.array(-1.0e7)] * (num_beams - 1)), [batch_size, 1]
		)
		scores = jnp.ones((batch_size, num_beams)) * np.array(-1.0e7)

		# For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
		# and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
		model = self.decode if self.config.is_encoder_decoder else self

		# flatten beam dim
		if "encoder_outputs" in model_kwargs:
			model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
				model_kwargs["encoder_outputs"]["last_hidden_state"]
			)
		for kwarg in ["attention_mask", "decoder_attention_mask"]:
			if kwarg in model_kwargs:
				model_kwargs[kwarg] = flatten_beam_dim(model_kwargs[kwarg])

		# initialize model specific kwargs
		model_kwargs = self.prepare_inputs_for_generation(
			flatten_beam_dim(input_ids), max_length, **model_kwargs
		)

		# initialize state
		state = BeamSearchState(
			cur_len=cur_len,
			running_sequences=running_sequences,
			running_scores=running_scores,
			sequences=sequences,
			scores=scores,
			is_sent_finished=is_sent_finished,
			model_kwargs=model_kwargs,
		)

		def beam_search_cond_fn(state):
			"""beam search state termination condition fn."""
			not_max_length_yet = state.cur_len < max_length

			if early_stopping == "never" and length_penalty > 0.0:
				best_running_score = state.running_scores[:, :1] / (
					(max_length - decoder_prompt_len) ** length_penalty
				)
			else:
				best_running_score = state.running_scores[:, :1] / (
					(state.cur_len - decoder_prompt_len) ** length_penalty
				)
			worst_finished_score = jnp.where(
				state.is_sent_finished,
				jnp.min(state.scores, axis=1, keepdims=True),
				np.array(-1.0e7),
			)
			improvement_still_possible = jnp.any(best_running_score > worst_finished_score)

			# 3. is there still a beam that has not finished?
			still_open_beam = ~(jnp.all(state.is_sent_finished) & (early_stopping is True))

			return not_max_length_yet & still_open_beam & improvement_still_possible

		def beam_search_body_fn(state, input_ids_length=1):
			"""beam search state update fn."""

			input_token = flatten_beam_dim(
				lax.dynamic_slice(
					state.running_sequences,
					(0, 0, state.cur_len - input_ids_length),
					(batch_size, num_beams, input_ids_length),
				)
			)
			model_outputs = model(input_token, **state.model_kwargs)

			logits = unflatten_beam_dim(model_outputs.logits[:, -1], batch_size, num_beams)
			cache = jax.tree_util.tree_map(
				lambda tensor: unflatten_beam_dim(tensor, batch_size, num_beams),
				model_outputs.past_key_values,
			)

			logits = self._adapt_logits_for_beam_search(logits)

			log_probs = jax.nn.log_softmax(logits)
			log_probs = logits_processor(
				flatten_beam_dim(state.running_sequences),
				flatten_beam_dim(log_probs),
				state.cur_len,
			)
			log_probs = unflatten_beam_dim(log_probs, batch_size, num_beams)
			log_probs = log_probs + jnp.expand_dims(state.running_scores, axis=2)
			vocab_size = log_probs.shape[2]
			log_probs = log_probs.reshape((batch_size, num_beams * vocab_size))

			beams_to_keep = 2 * num_beams
			topk_log_probs, topk_indices = lax.top_k(log_probs, k=beams_to_keep)
			topk_beam_indices = topk_indices // vocab_size
			topk_running_sequences = gather_beams(
				state.running_sequences, topk_beam_indices, batch_size, beams_to_keep
			)
			topk_ids = jnp.expand_dims(topk_indices % vocab_size, axis=2)
			topk_sequences = lax.dynamic_update_slice(
				topk_running_sequences, topk_ids, (0, 0, state.cur_len)
			)

			did_topk_just_finished = topk_sequences[:, :, state.cur_len] == eos_token_id
			running_topk_log_probs = topk_log_probs + did_topk_just_finished * np.array(
				-1.0e7
			)

			next_topk_indices = lax.top_k(running_topk_log_probs, k=num_beams)[1]
			next_running_sequences, next_running_scores = gather_beams(
				[topk_sequences, running_topk_log_probs],
				next_topk_indices,
				batch_size,
				num_beams,
			)

			topk_log_probs = topk_log_probs / (
				(state.cur_len + 1 - decoder_prompt_len) ** length_penalty
			)
			beams_in_batch_are_full = jnp.broadcast_to(
				state.is_sent_finished.all(axis=-1, keepdims=True), did_topk_just_finished.shape
			) & (early_stopping is True)
			add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
			topk_log_probs += add_penalty * np.array(-1.0e7)

			merged_sequences = jnp.concatenate([state.sequences, topk_sequences], axis=1)
			merged_scores = jnp.concatenate([state.scores, topk_log_probs], axis=1)
			merged_is_sent_finished = jnp.concatenate(
				[state.is_sent_finished, did_topk_just_finished], axis=1
			)
			topk_merged_indices = lax.top_k(merged_scores, k=num_beams)[1]
			next_sequences, next_scores, next_is_sent_finished = gather_beams(
				[merged_sequences, merged_scores, merged_is_sent_finished],
				topk_merged_indices,
				batch_size,
				num_beams,
			)

			next_running_indices = gather_beams(
				topk_beam_indices, next_topk_indices, batch_size, num_beams
			)
			next_cache = gather_beams(cache, next_running_indices, batch_size, num_beams)
			model_outputs["past_key_values"] = jax.tree_util.tree_map(
				lambda x: flatten_beam_dim(x), next_cache
			)
			next_model_kwargs = self.update_inputs_for_generation(
				model_outputs, state.model_kwargs
			)

			return BeamSearchState(
				cur_len=state.cur_len + 1,
				running_scores=next_running_scores,
				running_sequences=next_running_sequences,
				scores=next_scores,
				sequences=next_sequences,
				is_sent_finished=next_is_sent_finished,
				model_kwargs=next_model_kwargs,
			)

		state = partial(beam_search_body_fn, input_ids_length=input_ids.shape[-1])(state)

		if not trace:
			state = self._run_loop_in_debug(beam_search_cond_fn, beam_search_body_fn, state)
		else:
			state = lax.while_loop(beam_search_cond_fn, beam_search_body_fn, state)

		none_finished = jnp.any(state.is_sent_finished, axis=1)
		sequences = jnp.where(
			none_finished[:, None, None], state.sequences, state.running_sequences
		)
		scores = jnp.where(none_finished[:, None], state.scores, state.running_scores)

		sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
		scores = flatten_beam_dim(scores[:, :num_return_sequences])

		return FlaxBeamSearchOutput(sequences=sequences, scores=scores)
