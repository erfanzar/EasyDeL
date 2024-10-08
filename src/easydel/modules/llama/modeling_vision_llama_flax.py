
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

# Adapted from https://github.com/LargeWorldModel/LWM/blob/main/lwm/vision_llama.py
import copy
import warnings
from typing import Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import Dense
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers import GenerationConfig
from transformers.generation.flax_utils import (
	FlaxLogitsProcessorList,
	FlaxSampleOutput,
	SampleState,
)

from easydel.etils.etils import get_logger
from easydel.modules.llama.modeling_llama_flax import (
	FlaxLlamaBlockCollection,
	RMSNorm,
	precompute_frequencies,
)
from easydel.modules.llama.vision_llama_configuration import VisionLlamaConfig
from easydel.modules.modeling_flax_outputs import (
	FlaxBaseModelOutput,
	FlaxCausalLMOutput,
)
from easydel.modules.modeling_utils import EDPretrainedModel

logger = get_logger(__name__)


class FlaxVisionLlamaPreTrainedModel(EDPretrainedModel):
	config_class = VisionLlamaConfig
	base_model_prefix = "model"
	module_class: nn.Module = None

	def __init__(
		self,
		config: VisionLlamaConfig,
		input_shape: Tuple = (4, 1),
		seed: int = 0,
		dtype: jnp.dtype = jnp.float32,
		_do_init: bool = True,
		**kwargs,
	):
		module = self.module_class(config=config, dtype=dtype, **kwargs)
		super().__init__(
			config,
			module,
			input_shape=input_shape,
			seed=seed,
			dtype=dtype,
			_do_init=_do_init,
		)

	def init_cache(self, batch_size, max_length):
		input_ids = jnp.ones((batch_size, max_length))
		attention_mask = jnp.ones_like(input_ids)
		position_ids = jnp.broadcast_to(
			jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape
		)
		vision_mask = jnp.ones((batch_size, max_length), dtype=bool)

		init_variables = self.module.init(
			jax.random.PRNGKey(0),
			input_ids,
			vision_mask,
			attention_mask,
			position_ids,
			return_dict=False,
			init_cache=True,
		)
		return init_variables["cache"]

	def init_weights(
		self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
	) -> FrozenDict:
		"""
		The init_weights function is used to initialize the weights of a model.

		:param self: Access variables that belong to the class
		:param rng: jax.random.PRNGKey: Initialize the weights of the model
		:param input_shape: Tuple: Specify the shape of the input tensor
		:param params: FrozenDict: Pass in the parameters of a pre-trained model
		:return: A frozendict of parameters

		"""
		input_ids = jnp.zeros(input_shape, dtype="i4")
		attention_mask = jnp.ones_like(input_ids)
		vision_mask = jnp.ones(input_ids.shape, dtype=bool)
		position_ids = jnp.broadcast_to(
			jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape
		)
		params_rng, dropout_rng = jax.random.split(rng)

		random_params = self.module.init(
			{"params": params_rng, "dropout": dropout_rng},
			input_ids,
			vision_mask,
			attention_mask,
			position_ids,
			return_dict=False,
		)["params"]

		if params is not None:
			random_params = flatten_dict(unfreeze(random_params))
			params = flatten_dict(unfreeze(params))
			for missing_key in self._missing_keys:
				params[missing_key] = random_params[missing_key]
			self._missing_keys = set()
			return freeze(unflatten_dict(params))
		else:
			return random_params

	def __call__(
		self,
		input_ids: chex.Array,
		vision_mask: Optional[chex.Array] = None,
		attention_mask: Optional[chex.Array] = None,
		position_ids: Optional[chex.Array] = None,
		params: dict = None,
		past_key_values: Optional[dict] = None,
		dropout_rng: jax.random.PRNGKey = None,
		train: bool = False,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		extra_embedding: Optional[Union[jnp.ndarray, None]] = None,
		add_params_field: bool = False,
		**kwargs,
	):
		output_attentions = (
			output_attentions
			if output_attentions is not None
			else self.config.output_attentions
		)
		output_hidden_states = (
			output_hidden_states
			if output_hidden_states is not None
			else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.return_dict

		batch_size, sequence_length = input_ids.shape

		if position_ids is None:
			if past_key_values is not None:
				raise ValueError(
					"Make sure to provide `position_ids` when passing `past_key_values`."
				)

			position_ids = jnp.broadcast_to(
				jnp.arange(sequence_length)[None, :], (batch_size, sequence_length)
			)

		if attention_mask is None:
			attention_mask = jnp.ones((batch_size, sequence_length))

		# Handle any PRNG if needed
		rngs = {}
		if dropout_rng is not None:
			rngs["dropout"] = dropout_rng

		inputs = {"params": params or self.params}

		if past_key_values is not None:
			inputs["cache"] = past_key_values
			mutable = ["cache"]
		else:
			mutable = False

		outputs = self.module.apply(
			inputs,
			jnp.array(input_ids, dtype="i4"),
			jnp.array(vision_mask, dtype="f4"),
			jnp.array(attention_mask, dtype="i4"),
			jnp.array(position_ids, dtype="i4"),
			not train,
			False,
			output_attentions,
			output_hidden_states,
			return_dict,
			rngs=rngs,
			mutable=mutable,
		)

		# add updated cache to model output
		if past_key_values is not None and return_dict:
			outputs, past_key_values = outputs
			outputs["past_key_values"] = unfreeze(past_key_values["cache"])
			return outputs
		elif past_key_values is not None and not return_dict:
			outputs, past_key_values = outputs
			outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

		return outputs


class FlaxVisionLlamaModule(nn.Module):
	config: VisionLlamaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		config = self.config
		self.embed_dim = config.hidden_size

		self.embed_vision = nn.Embed(
			config.vision_vocab_size,
			config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)

		self.embed_tokens = nn.Embed(
			config.vocab_size,
			config.hidden_size,
			embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.dropout = nn.Dropout(rate=config.embd_pdrop)
		self.layers = FlaxLlamaBlockCollection(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.norm = RMSNorm(
			self.config.hidden_size,
			eps=self.config.rms_norm_eps,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
		)
		self.causal_mask = nn.make_causal_mask(
			jnp.ones(
				(
					1,
					getattr(
						self.config,
						"mask_max_position_embeddings",
						self.config.max_position_embeddings,
					),
				),
				dtype="bool",
			),
			dtype="bool",
		)

		initial_rope_kwargs = dict(rope_type="none")
		if config.rope_scaling is not None:
			scaling_type = config.rope_scaling["type"]
			scaling_factor = config.rope_scaling["factor"]
			initial_rope_kwargs = dict(scaling_factor=scaling_factor, rope_type=scaling_type)
		self.frequencies = precompute_frequencies(
			max_position_embeddings=(
				getattr(
					self.config,
					"freq_max_position_embeddings",
					self.config.max_position_embeddings,
				)
			),
			dim=config.hidden_size // config.num_attention_heads,
			base=config.rope_theta,
			**initial_rope_kwargs,
		)

	def __call__(
		self,
		input_ids,
		vision_mask,
		attention_mask,
		position_ids,
		deterministic=True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		input_ids = input_ids.astype("i4")

		if input_ids.shape[1] == 1:
			if self.config.sample_mode == "text":
				input_embeds = self.embed_tokens(input_ids)
			elif self.config.sample_mode == "vision":
				input_embeds = self.embed_vision(input_ids)
			elif self.config.sample_mode == "all":
				raise NotImplementedError
			else:
				raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")
		else:
			input_text_embeds = self.embed_tokens(jnp.where(vision_mask, 0, input_ids))
			input_vision_embeds = self.embed_vision(jnp.where(vision_mask, input_ids, 0))
			vision_mask = vision_mask[..., None].astype("f4")
			input_embeds = (
				input_text_embeds * (1 - vision_mask) + input_vision_embeds * vision_mask
			)

		hidden_states = self.dropout(input_embeds, deterministic=deterministic)

		outputs = self.layers(
			hidden_states=hidden_states,
			attention_mask=attention_mask,
			position_ids=position_ids,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			causal_mask=self.causal_mask,
			frequencies=self.frequencies,
		)

		hidden_states = outputs[0]
		hidden_states = self.norm(hidden_states)

		if output_hidden_states:
			all_hidden_states = outputs[1] + (hidden_states,)
			outputs = (hidden_states, all_hidden_states) + outputs[2:]
		else:
			outputs = (hidden_states,) + outputs[1:]

		if not return_dict:
			return tuple(v for v in outputs if v is not None)

		return FlaxBaseModelOutput(
			last_hidden_state=hidden_states,
			hidden_states=outputs[1],
			attentions=outputs[-1],
		)


class FlaxVisionLlamaForCausalLMModule(nn.Module):
	config: VisionLlamaConfig
	dtype: jnp.dtype = jnp.float32
	param_dtype: jnp.dtype = jnp.float32
	precision: Optional[Union[jax.lax.Precision, str]] = None

	def setup(self):
		self.model = FlaxVisionLlamaForCausalLMModule(
			self.config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
		)
		self.vision_head = Dense(
			self.config.vision_vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			precision=self.precision,
		)
		self.lm_head = Dense(
			self.config.vocab_size,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			use_bias=False,
			kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
			precision=self.precision,
		)

	def __call__(
		self,
		input_ids,
		vision_mask,
		attention_mask=None,
		position_ids=None,
		deterministic: bool = True,
		init_cache: bool = False,
		output_attentions: bool = False,
		output_hidden_states: bool = False,
		return_dict: bool = True,
	):
		batch_size, seq_length = input_ids.shape
		if attention_mask is None:
			attention_mask = jnp.ones_like(input_ids)
		if position_ids is None:
			position_ids = jnp.broadcast_to(
				jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
				(batch_size, seq_length),
			)

		outputs = self.transformer(
			input_ids,
			vision_mask,
			attention_mask,
			position_ids,
			deterministic=deterministic,
			init_cache=init_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0]

		if self.config.tie_vision_embeddings:
			shared_kernel = self.transformer.variables["params"]["embed_vision"][
				"embedding"
			].T
			vision_logits = self.vision_head.apply(
				{"params": {"kernel": shared_kernel}}, hidden_states
			)
		else:
			vision_logits = self.vision_head(hidden_states)

		if self.config.tie_word_embeddings:
			shared_kernel = self.transformer.variables["params"]["embed_tokens"][
				"embedding"
			].T.astype(self.param_dtype)
			lm_logits = self.lm_head.apply(
				{"params": {"kernel": shared_kernel}},
				hidden_states,
			)
		else:
			lm_logits = self.lm_head(hidden_states)

		if self.config.sample_mode == "all":
			if not return_dict:
				return (
					vision_logits,
					lm_logits,
				) + outputs[1:]

			return FlaxCausalLMOutput(
				logits=(vision_logits, lm_logits),
				hidden_states=outputs.hidden_states,
				attentions=outputs.attentions,
			)
		elif self.config.sample_mode == "vision":
			if not return_dict:
				return (vision_logits,) + outputs[1:]

			return FlaxCausalLMOutput(
				logits=vision_logits,
				hidden_states=outputs.hidden_states,
				attentions=outputs.attentions,
			)
		elif self.config.sample_mode == "text":
			if not return_dict:
				return (lm_logits,) + outputs[1:]

			return FlaxCausalLMOutput(
				logits=lm_logits,
				hidden_states=outputs.hidden_states,
				attentions=outputs.attentions,
			)
		else:
			raise ValueError(f"Invalid sample_mode: {self.config.sample_mode}")


class FlaxVisionLlamaForCausalLM(FlaxVisionLlamaPreTrainedModel):
	module_class = FlaxVisionLlamaForCausalLMModule

	def prepare_inputs_for_generation(
		self,
		input_ids,
		max_length,
		attention_mask: Optional[jax.Array] = None,
		vision_mask=None,
	):
		# initializing the cache
		batch_size, seq_length = input_ids.shape

		past_key_values = self.init_cache(batch_size, max_length)
		extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
		if attention_mask is not None:
			position_ids = attention_mask.cumsum(axis=-1) - 1
			extended_attention_mask = lax.dynamic_update_slice(
				extended_attention_mask, attention_mask, (0, 0)
			)
		else:
			position_ids = jnp.broadcast_to(
				jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length)
			)

		return {
			"past_key_values": past_key_values,
			"attention_mask": extended_attention_mask,
			"position_ids": position_ids,
			"vision_mask": vision_mask,
		}

	def update_inputs_for_generation(self, model_outputs, model_kwargs):
		return {
			"past_key_values": model_outputs.past_key_values,
			"position_ids": model_kwargs["position_ids"][:, -1:] + 1,
			"attention_mask": model_kwargs["attention_mask"],
			"vision_mask": model_kwargs["vision_mask"],
		}

	def _sample_vision(
		self,
		input_ids: None,
		max_length: Optional[int] = None,
		pad_token_id: Optional[int] = None,
		eos_token_id: Optional[int] = None,
		prng_key: Optional[jnp.ndarray] = None,
		logits_processor: Optional[FlaxLogitsProcessorList] = None,
		logits_warper: Optional[FlaxLogitsProcessorList] = None,
		cfg_scales: jnp.ndarray = 1.0,
		trace: bool = True,
		params: Optional[Dict[str, jnp.ndarray]] = None,
		model_kwargs: Optional[Dict[str, jnp.ndarray]] = None,
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
		initial_len = cur_len

		eos_token_id = jnp.array(
			eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None
		)
		pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
		cur_len = jnp.array(cur_len)

		# per batch-item holding current token in loop.
		sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
		sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

		# per batch-item state bit indicating if sentence has finished.
		is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

		# For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
		# and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
		model = self.decode if self.config.is_encoder_decoder else self

		# initialize model specific kwargs
		model_kwargs = self.prepare_inputs_for_generation(
			input_ids, max_length, **model_kwargs
		)

		# initialize state
		state = SampleState(
			cur_len=cur_len,
			sequences=sequences,
			running_token=input_ids,
			is_sent_finished=is_sent_finished,
			prng_key=prng_key,
			model_kwargs=model_kwargs,
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
			model_outputs = model(state.running_token, params=params, **state.model_kwargs)

			logits = model_outputs.logits[:, -1]
			cond_logits, uncond_logits = jnp.split(logits, 2, axis=0)
			logits = uncond_logits + cfg_scales[:, None] * (cond_logits - uncond_logits)

			# apply min_length, ...
			logits = logits_processor(state.sequences, logits, state.cur_len)
			# apply top_p, top_k, temperature
			logits = logits_warper(logits, logits, state.cur_len)

			next_token = jax.random.categorical(prng_key, logits, axis=-1)
			next_token = jax.lax.cond(
				(state.cur_len - initial_len + 1) % 257 == 0,
				lambda: jnp.full_like(next_token, 8192),
				lambda: next_token,
			)
			next_token = jnp.concatenate([next_token, next_token], axis=0)

			# next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
			next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
			next_token = next_token[:, None]

			next_sequences = lax.dynamic_update_slice(
				state.sequences, next_token, (0, state.cur_len)
			)
			next_model_kwargs = self.update_inputs_for_generation(
				model_outputs, state.model_kwargs
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
				sample_search_cond_fn, sample_search_body_fn, state
			)
		else:
			state = lax.while_loop(sample_search_cond_fn, sample_search_body_fn, state)

		return FlaxSampleOutput(sequences=state.sequences)

	def generate_vision(
		self,
		input_ids: jnp.ndarray,
		cfg_scales: jnp.ndarray,
		generation_config: Optional[GenerationConfig] = None,
		prng_key: Optional[jnp.ndarray] = None,
		trace: bool = True,
		params: Optional[Dict[str, jnp.ndarray]] = None,
		logits_processor: Optional[FlaxLogitsProcessorList] = None,
		**kwargs,
	):
		# Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
		self._validate_model_class()

		# priority: `generation_config` argument > `model.generation_config` (the default generation config)
		if generation_config is None:
			# legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
			# two conditions must be met
			# 1) the generation config must have been created from the model config (`_from_model_config` field);
			# 2) the generation config must have seen no modification since its creation (the hash is the same).
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
						" https://huggingface.co/docs/transformers/generation_strategies#d"
						"efault-text-generation-configuration )"
					)
					self.generation_config = new_generation_config
			generation_config = self.generation_config

		generation_config = copy.deepcopy(generation_config)
		model_kwargs = generation_config.update(
			**kwargs
		)  # All unused kwargs must be model kwargs
		generation_config.validate()
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
				warnings.warn(
					"The attention mask and the pad token id were not set. As a consequence, you may observe "
					"unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
				)
			eos_token_id = generation_config.eos_token_id
			if isinstance(eos_token_id, list):
				eos_token_id = eos_token_id[0]
			warnings.warn(
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
				warnings.warn(
					"A decoder-only architecture is being used, but right-padding was detected! For correct "
					"generation results, please set `padding_side='left'` when initializing the tokenizer."
				)

		batch_size = input_ids.shape[0]

		if self.config.is_encoder_decoder:
			# add encoder_outputs to model_kwargs
			if model_kwargs.get("encoder_outputs") is None:
				model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
					input_ids, params, model_kwargs
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
				"to control the generation length.  recommend setting `max_new_tokens` to control "
				"the maximum length of the generation.",
				UserWarning,
			)
		elif generation_config.max_new_tokens is not None:
			if not has_default_max_length and generation_config.max_length is not None:
				warnings.warn(
					f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
					f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
					"Please refer to the documentation for more information. "
					"(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
				)
			generation_config.max_length = (
				generation_config.max_new_tokens + input_ids_seq_length
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
			warnings.warn(
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
			raise NotImplementedError
		elif generation_config.do_sample and generation_config.num_beams == 1:
			logits_warper = self._get_logits_warper(generation_config=generation_config)
			return self._sample_vision(
				input_ids,
				generation_config.max_length,
				generation_config.pad_token_id,
				generation_config.eos_token_id,
				prng_key,
				logits_warper=logits_warper,
				logits_processor=logits_processor,
				cfg_scales=cfg_scales,
				trace=trace,
				params=params,
				model_kwargs=model_kwargs,
			)
		elif not generation_config.do_sample and generation_config.num_beams > 1:
			raise NotImplementedError
		else:
			raise NotImplementedError("`Beam sampling is currently not implemented.")
