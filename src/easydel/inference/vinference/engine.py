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

"""Module for text generation pipeline using JAX/Flax."""

from functools import partial
import warnings
from typing import Optional, Union, Generator, Any

import flax.core
from fjformer import GenerateRNG, with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PreTrainedTokenizer
from easydel.etils.etils import get_logger
from easydel.inference.utils import (
	vInferenceConfig,
	SampleState,
	inference_step_compiled,
)
from jax import numpy as jnp
from jax import random as jrand
from jax import lax
import jax
from easydel.modules.modeling_utils import EDPretrainedModel
from fjformer.core import implicit_compact

logger = get_logger(__name__)


@partial(jax.jit, static_argnames=["model", "generation_config"])
def _compiled_generate(
	model: EDPretrainedModel,
	params: dict,
	input_ids: jax.Array,
	attention_mask: jax.Array,
	position_ids: jax.Array,
	generation_config: vInferenceConfig,
	rng: jrand.PRNGKey,
) -> SampleState:
	partition_axes = model.config.partition_axis
	mesh = model.config.mesh

	eos_token_id = jnp.array(generation_config.eos_token_id, dtype=jnp.int32)
	pad_token_id = jnp.array(generation_config.pad_token_id, dtype=jnp.int32)

	generation_spec = PartitionSpec(
		partition_axes.batch_axis,
		partition_axes.key_sequence_axis,
	)

	batch_size, current_length = input_ids.shape
	max_length = current_length + generation_config.max_new_tokens
	current_length = jnp.array(current_length)
	sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
	sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))
	is_sequence_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

	if attention_mask is None:
		warnings.warn(
			"`attention_mask` is not provided, it's recommended to "
			"pass an attention mask for better results.",
			stacklevel=1,
		)
		attention_mask = jnp.ones_like(input_ids)

	if position_ids is None:
		position_ids = attention_mask.cumsum(axis=-1, dtype="i4") - 1
	with mesh:
		input_ids = with_sharding_constraint(input_ids, generation_spec)
		attention_mask = with_sharding_constraint(attention_mask, generation_spec)
		position_ids = with_sharding_constraint(position_ids, generation_spec)
	assert (
		position_ids.shape == attention_mask.shape
	), "`position_ids` and `attention_mask` must have the same shape."

	state = SampleState(
		current_length=current_length,
		sequences=sequences,
		running_token=input_ids,
		is_sequence_finished=is_sequence_finished,
		prng_key=rng,
		model_kwargs=model.prepare_inputs_for_generation(
			input_ids=input_ids,
			max_length=max_length,
			attention_mask=attention_mask,
		),
	)

	def cond_fn(state):
		"""state termination condition fn."""
		all_sequence_finished = jnp.all(state.is_sequence_finished)
		return ~jnp.logical_or(
			all_sequence_finished,
			state.current_length >= (current_length + generation_config.streaming_chunks),
		)

	@implicit_compact
	def sampling_step(params, state: SampleState):
		"""
		Performs a single sampling step for text generation.

		Args:
				params: Model parameters.
				state (inference_utils.SampleState): The current generation state.

		Returns:
				inference_utils.SampleState: The updated generation state.
		"""
		model_outputs = model(
			input_ids=state.running_token,
			params=params,
			add_params_field=True,
			return_dict=True,
			**state.model_kwargs,
		)
		next_token = inference_step_compiled(
			model_outputs.logits[:, -1],
			state.sequences,
			state.prng_key,
			generation_config,
			current_length,
			generation_config.max_new_tokens,
		)

		next_token = (
			next_token * ~state.is_sequence_finished
			+ pad_token_id * state.is_sequence_finished
		)

		next_sequence_finished = state.is_sequence_finished | jnp.isin(
			next_token,
			eos_token_id,
		)
		next_token = next_token[:, None]
		next_sequences = lax.dynamic_update_slice(
			state.sequences,
			next_token,
			(0, state.current_length),
		)
		next_model_kwargs = model.update_inputs_for_generation(
			model_outputs,
			state.model_kwargs,
		)

		return SampleState(
			current_length=state.current_length + 1,
			sequences=next_sequences,
			running_token=next_token,
			is_sequence_finished=next_sequence_finished,
			prng_key=jrand.split(state.prng_key, 2)[0],
			model_kwargs=next_model_kwargs,
		)

	with mesh:
		if input_ids.shape[-1] > 1:
			state = sampling_step(params=params, state=state)

	# def interval_sample(state):
	# 	return sampling_step(params=params, state=state)

	# state = jax.lax.while_loop(cond_fn, body_fun=interval_sample, init_val=state)
	return state


@partial(jax.jit, static_argnames=["model", "generation_config"])
def _compiled_interval_generate(
	model: EDPretrainedModel,
	params: dict,
	state: SampleState,
	generation_config: vInferenceConfig,
	loop_max_tokens: int,
	start_length: int,
) -> SampleState:
	mesh = model.config.mesh

	eos_token_id = jnp.array(generation_config.eos_token_id, dtype=jnp.int32)
	pad_token_id = jnp.array(generation_config.pad_token_id, dtype=jnp.int32)
	tlen = state.current_length + loop_max_tokens

	def cond_fn(state):
		"""state termination condition fn."""
		all_sequence_finished = jnp.all(state.is_sequence_finished)
		return ~jnp.logical_or(all_sequence_finished, state.current_length >= tlen)

	@implicit_compact
	def sampling_step(params, state: SampleState):
		"""
		Performs a single sampling step for text generation.

		Args:
				params: Model parameters.
				state (inference_utils.SampleState): The current generation state.

		Returns:
				inference_utils.SampleState: The updated generation state.
		"""
		model_outputs = model(
			input_ids=state.running_token,
			params=params,
			add_params_field=True,
			return_dict=True,
			**state.model_kwargs,
		)
		next_token = inference_step_compiled(
			model_outputs.logits[:, -1],
			state.sequences,
			state.prng_key,
			generation_config,
			start_length,
			generation_config.max_new_tokens,
		)

		next_token = (
			next_token * ~state.is_sequence_finished
			+ pad_token_id * state.is_sequence_finished
		)

		next_sequence_finished = state.is_sequence_finished | jnp.isin(
			next_token,
			eos_token_id,
		)
		next_token = next_token[:, None]
		next_sequences = lax.dynamic_update_slice(
			state.sequences,
			next_token,
			(0, state.current_length),
		)
		next_model_kwargs = model.update_inputs_for_generation(
			model_outputs,
			state.model_kwargs,
		)

		return SampleState(
			current_length=state.current_length + 1,
			sequences=next_sequences,
			running_token=next_token,
			is_sequence_finished=next_sequence_finished,
			prng_key=jrand.split(state.prng_key, 2)[0],
			model_kwargs=next_model_kwargs,
		)

	with mesh:

		def interval_sample(state):
			return sampling_step(params=params, state=state)

		state = jax.lax.while_loop(cond_fn, body_fun=interval_sample, init_val=state)
	return state


class vInference:
	def __init__(
		self,
		model: EDPretrainedModel,
		params: Union[flax.core.FrozenDict, dict],
		tokenizer: PreTrainedTokenizer,
		generation_config: Optional[vInferenceConfig] = None,
		seed: Optional[int] = 42,
		input_partition_spec: Optional[PartitionSpec] = None,
		max_new_tokens: int = 512,
	):
		self.model = model
		self.params = self._validate_params(params)
		self.tokenizer = tokenizer
		self.generation_config = self._init_generation_config(
			generation_config,
			max_new_tokens,
		)
		self._rng_generator = GenerateRNG(seed)
		self.input_partition_spec = input_partition_spec or PartitionSpec(("dp", "fsdp"))
		self.mesh = self.model.config.mesh

		self._init_shardings()
		self._validate_token_ids()

	def _validate_params(
		self, params: Union[flax.core.FrozenDict, dict]
	) -> Union[flax.core.FrozenDict, dict]:
		if "params" in params:
			warnings.warn(
				"`params` field should be like {k:v} not {'params':{k:v}}",
				DeprecationWarning,
				stacklevel=2,
			)
			return params["params"]
		return params

	def _init_generation_config(
		self, generation_config: Optional[vInferenceConfig], max_new_tokens: int
	) -> vInferenceConfig:
		if generation_config is None:
			if self.model.generation_config is not None:
				return vInferenceConfig(
					bos_token_id=self.model.generation_config.bos_token_id,
					eos_token_id=self.model.generation_config.eos_token_id,
					pad_token_id=self.model.generation_config.pad_token_id,
					top_k=self.model.generation_config.top_k,
					top_p=self.model.generation_config.top_p,
					temperature=self.model.generation_config.temperature,
					max_new_tokens=self.model.generation_config.max_new_tokens or max_new_tokens,
				)
			return vInferenceConfig(max_new_tokens=max_new_tokens)
		return generation_config

	def _init_shardings(self):
		self.input_sharding = NamedSharding(
			spec=self.input_partition_spec,
			mesh=self.model.mesh,
		)
		self.empty_sharding = NamedSharding(
			spec=PartitionSpec(),
			mesh=self.model.mesh,
		)
		self.gen_input_sharding = NamedSharding(
			spec=PartitionSpec(self.input_partition_spec[0], None),
			mesh=self.model.mesh,
		)

	def _validate_token_ids(self):
		if self.generation_config.pad_token_id is None:
			self.generation_config.pad_token_id = self.tokenizer.pad_token_id
		if self.generation_config.eos_token_id is None:
			self.generation_config.eos_token_id = self.tokenizer.eos_token_id
		if self.generation_config.bos_token_id is None:
			self.generation_config.bos_token_id = self.tokenizer.bos_token_id

		assert self.generation_config.pad_token_id is not None, (
			"`pad_token_id` cannot be None. "
			"(Set `tokenizer.pad_token_id = tokenizer.eos_token_id` if undefined)"
		)
		assert (
			self.generation_config.eos_token_id is not None
		), "`eos_token_id` cannot be None."

	def generate(
		self,
		input_ids: jax.Array,
		attention_mask: Optional[jax.Array] = None,
		position_ids: Optional[jax.Array] = None,
	) -> Generator[SampleState, SampleState, Any]:
		input_ids = jnp.array(input_ids)
		if attention_mask is None:
			warnings.warn(
				"`attention_mask` is not provided, it's recommended to "
				"pass an attention mask for better results.",
				stacklevel=1,
			)
			attention_mask = jnp.ones_like(input_ids)

		if position_ids is None:
			position_ids = attention_mask.cumsum(axis=-1, dtype="i4") - 1

		input_ids = jnp.array(input_ids)
		attention_mask = jnp.array(attention_mask)
		start_length = input_ids.shape[-1]

		state: SampleState = _compiled_generate(
			model=self.model,
			params=self.params,
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			generation_config=self.generation_config,
			rng=self._rng_generator.rng,
		)
		if not state.is_sequence_finished and self.generation_config._loop_rows - 1 != 0:
			for _ in range(self.generation_config._loop_rows - 1):
				state = _compiled_interval_generate(
					model=self.model,
					params=self.params,
					state=state,
					generation_config=self.generation_config,
					loop_max_tokens=self.generation_config.streaming_chunks,
					start_length=start_length,
				)
				yield state
				if state.is_sequence_finished:
					break
		else:
			yield state

	def precompile(self, batch_size: int, input_tokens_length: int):
		input_ids = jnp.ones((batch_size, input_tokens_length), dtype="i4")
		attention_mask = jnp.ones_like(input_ids)
		position_ids = attention_mask.cumsum(axis=-1, dtype="i4") - 1
		state = _compiled_generate(
			model=self.model,
			params=self.params,
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			generation_config=self.generation_config,
			rng=self._rng_generator.rng,
		)
		_compiled_interval_generate(
			model=self.model,
			params=self.params,
			state=state,
			generation_config=self.generation_config,
			loop_max_tokens=self.generation_config.streaming_chunks,
			start_length=input_tokens_length,
		)

		return True
