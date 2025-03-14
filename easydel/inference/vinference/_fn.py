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

import time
import typing as tp
import warnings  # noqa: F401

import jax
from jax import numpy as jnp

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = object
from eformer.jaximus import implicit

from ..utils import (
	SampleState,
	create_sampling_step,
	vInferenceConfig,
	vInferencePreCompileConfig,
)


def measure_flops(func, *args, **kwargs):
	try:
		flops = func.cost_analysis()[0]["flops"]
	except Exception:
		flops = 1
	start_time = time.perf_counter()
	result = jax.block_until_ready(func(*args, **kwargs))
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	return result, flops, flops / elapsed_time, elapsed_time


def basic_generation_first_iter_fn(
	graphdef: EasyDeLBaseModule,
	graphstate: dict,
	graphother,
	state: SampleState,
	generation_config: vInferenceConfig,
) -> SampleState:
	"""
	Compiled function for performing the initial generation step.

	This function takes the graphdef, parameters, input IDs, attention mask, position IDs,
	generation configuration, and a random number generator key as input. It initializes
	the generation state and performs the first sampling step.

	Returns:
		SampleState: The initial generation state after the first sampling step.
	"""

	if state.running_token.shape[-1] > 1:
		runner = create_sampling_step(
			eos_token_id=jnp.array(generation_config.eos_token_id, dtype=jnp.int32),
			pad_token_id=jnp.array(generation_config.pad_token_id, dtype=jnp.int32),
			logits_processor=generation_config.get_logits_processor(),
			logits_warper=generation_config.get_logits_processor(),
			do_sample=generation_config.do_sample,
		)
		runner = implicit(runner)
		state = runner(
			graphdef=graphdef,
			graphstate=graphstate,
			graphother=graphother,
			state=state,
		)
	return state


def basic_generation_iter_fn(
	graphdef: EasyDeLBaseModule,
	graphstate: dict,
	graphother,
	state: SampleState,
	generation_config: vInferenceConfig,
	loop_max_tokens: int,
) -> SampleState:
	"""
	Compiled function for performing interval generation steps.

	This function takes the graphdef, parameters, current generation state, generation
	configuration, maximum number of tokens for the loop, and the starting length as input.
	It continues the generation process until the termination condition is met.

	Returns:
		SampleState: The updated generation state after the interval generation steps.
	"""

	tlen = state.current_length + loop_max_tokens

	def cond_fn(state):
		"""state termination condition fn."""
		all_sequence_finished = jnp.all(state.is_sequence_finished)
		return ~jnp.logical_or(all_sequence_finished, state.current_length >= tlen)

	sampling_step = create_sampling_step(
		eos_token_id=jnp.array(generation_config.eos_token_id, dtype=jnp.int32),
		pad_token_id=jnp.array(generation_config.pad_token_id, dtype=jnp.int32),
		logits_processor=generation_config.get_logits_processor(),
		logits_warper=generation_config.get_logits_processor(),
		do_sample=generation_config.do_sample,
	)

	sampling_step = implicit(sampling_step)

	def interval_sample(state):
		return sampling_step(
			graphdef=graphdef,
			graphstate=graphstate,
			graphother=graphother,
			state=state,
		)

	state = jax.lax.while_loop(cond_fn, body_fun=interval_sample, init_val=state)
	return state


COMPILED_FUNCS = {}


def get_compiled_funcs(
	standalone_config: vInferencePreCompileConfig,
	id: str,
	safe: bool = True,
	false_instance: tp.Any = None,
):
	"""
	Retrieves compiled generation functions from a cache.

	Args:
		batch_size: The batch size.
		input_tokens_length: The length of the input tokens.
		vision_active: whenever the model using it vision features.
		id: A unique identifier for the compilation.

	"""
	search_key = f"{standalone_config.get_default_hash()}-UUID{id}"
	outs = COMPILED_FUNCS.get(search_key, false_instance)
	if outs is None and safe:
		raise RuntimeError(
			"wasn't able to find requested functions please `precompile`"
			" inference before using `generate` function."
		)
	return outs


def put_compiled_funcs(
	funcs: tp.Any,
	standalone_config: vInferencePreCompileConfig,
	id: str,
):
	"""
	Stores compiled generation functions in a cache.

	Args:
		funcs: functions to put.
		standalone_config: vinference precompile config
		id: A unique identifier for the compilation.
	"""
	search_key = f"{standalone_config.get_default_hash()}-UUID{id}"
	COMPILED_FUNCS[search_key] = funcs
