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
import dataclasses
import typing as tp

import chex
import fjformer
import jax
import jax.experimental
import jax.experimental.pallas
import jax.random
from jax import core, random, sharding
from jax import numpy as jnp

from .logits_process import (
	FlaxForcedBOSTokenLogitsProcessor,
	FlaxForcedEOSTokenLogitsProcessor,
	FlaxLogitsProcessorList,
	FlaxMinLengthLogitsProcessor,
	FlaxNoRepeatNGramLogitsProcessor,
	FlaxSuppressTokensLogitsProcessor,
	FlaxTemperatureLogitsWarper,
	FlaxTopKLogitsWarper,
	FlaxTopPLogitsWarper,
	hash_fn,
)

if hasattr(core, "new_main"):
	ic = fjformer.core.implicit_compact
else:

	def ic(fn):
		# warnings.warn(
		# 	"fjformer quantizers wont support current jax version (soon will be fixed).",
		# 	stacklevel=3,
		# )
		return fn


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class vInferenceConfig:
	max_new_tokens: int = 64
	min_length: tp.Optional[int] = None
	streaming_chunks: int = 16
	temperature: float = 0.0
	top_p: float = 0.95
	top_k: int = 50
	do_sample: bool = True
	no_repeat_ngram_size: tp.Optional[int] = None
	suppress_tokens: tp.Optional[list] = None
	forced_bos_token_id: tp.Optional[int] = None
	forced_eos_token_id: tp.Optional[int] = None
	pad_token_id: tp.Optional[int] = None
	bos_token_id: tp.Optional[int] = None
	eos_token_id: tp.Optional[tp.Union[int, tp.List[int]]] = None
	_loop_rows: tp.Optional[int] = None

	def tree_flatten(self):
		return (
			self.max_new_tokens,
			self.min_length,
			self.streaming_chunks,
			self.temperature,
			self.top_p,
			self.top_k,
			self.do_sample,
			self.no_repeat_ngram_size,
			self.suppress_tokens,
			self.forced_bos_token_id,
			self.forced_eos_token_id,
			self.pad_token_id,
			self.bos_token_id,
			self.eos_token_id,
			self._loop_rows,
		), {}

	def tree_unflatten(cls, aux, children):
		return cls(*children)

	def __post_init__(self):
		if isinstance(self.max_new_tokens, int):
			self._loop_rows = (
				self.max_new_tokens + self.streaming_chunks - 1
			) // self.streaming_chunks

	def __repr__(self):
		# fmt:off
		string = f"{self.__class__.__name__}(\n"
		for k, v in self.__dict__.items():
			if not k.startswith("_"):
				try:
					repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
					string += repr_src if len(repr_src) < 500 else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
				except TypeError: pass #noqa
		return string.strip() + "\n)"
		# fmt:on

	__str__ = __repr__
	__hash__ = hash_fn

	def get_logits_warper(self):
		warpers = FlaxLogitsProcessorList()
		if self.temperature is not None and self.temperature != 1.0:
			warpers.append(FlaxTemperatureLogitsWarper(self.temperature))
		if self.top_k is not None and self.top_k != 0:
			warpers.append(FlaxTopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=1))
		if self.top_p is not None and self.top_p < 1.0:
			warpers.append(FlaxTopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1))
		print(hash(warpers))
		if len(warpers) == 0:
			return None

		return warpers

	def get_logits_processor(self):
		processors = FlaxLogitsProcessorList()
		eos_id = (
			self.eos_token_id[0] if isinstance(self.eos_token_id, list) else self.eos_token_id
		)
		if (
			self.min_length is not None
			and self.eos_token_id is not None
			and self.min_length > -1
		):
			processors.append(FlaxMinLengthLogitsProcessor(self.min_length, eos_id))
		if self.forced_bos_token_id is not None:
			processors.append(FlaxForcedBOSTokenLogitsProcessor(self.forced_bos_token_id))
		if self.forced_eos_token_id is not None:
			fet = FlaxForcedEOSTokenLogitsProcessor(self.max_length, self.forced_eos_token_id)
			processors.append(fet)
		if self.suppress_tokens is not None:
			processors.append(FlaxSuppressTokensLogitsProcessor(self.suppress_tokens))
		if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size > 0:
			processors.append(FlaxNoRepeatNGramLogitsProcessor(self.no_repeat_ngram_size))
		if len(processors) == 0:
			return None
		return processors


def lower_function(
	func,
	func_input_args,
	func_input_kwargs,
	mesh=None,
	in_shardings=None,
	out_shardings=None,
	static_argnums=None,
	donate_argnums=None,
):
	"""
	lower a JAX function with optional sharding and mesh configuration.

	Args:
	    func: The JAX function to compile.
	    func_input_args: Input arguments for the function.
	    func_input_kwargs: Input keyword arguments for the function.
	    mesh: tp.Optional JAX mesh for distributed execution.
	    in_shardings: tp.Optional input sharding specifications.
	    out_shardings: tp.Optional output sharding specifications.
	    static_argnums: Indices of static arguments.
	    donate_argnums: Indices of arguments to donate.

	Returns:
	    lowered JAX function.
	"""
	if mesh is None:
		return jax.jit(
			func,
			in_shardings=in_shardings,
			out_shardings=out_shardings,
			static_argnums=static_argnums,
			donate_argnums=donate_argnums,
		).lower(*func_input_args, **func_input_kwargs)
	with mesh:
		return jax.jit(
			func,
			in_shardings=in_shardings,
			out_shardings=out_shardings,
			static_argnums=static_argnums,
			donate_argnums=donate_argnums,
		).lower(*func_input_args, **func_input_kwargs)


def compile_function(
	func,
	func_input_args,
	func_input_kwargs,
	mesh=None,
	in_shardings=None,
	out_shardings=None,
	static_argnums=None,
	donate_argnums=None,
):
	"""
	Compiles a JAX function with optional sharding and mesh configuration.

	Args:
	    func: The JAX function to compile.
	    func_input_args: Input arguments for the function.
	    func_input_kwargs: Input keyword arguments for the function.
	    mesh: tp.Optional JAX mesh for distributed execution.
	    in_shardings: tp.Optional input sharding specifications.
	    out_shardings: tp.Optional output sharding specifications.
	    static_argnums: Indices of static arguments.
	    donate_argnums: Indices of arguments to donate.

	Returns:
	    Compiled JAX function.
	"""
	return lower_function(
		func,
		func_input_args,
		func_input_kwargs,
		mesh=mesh,
		in_shardings=in_shardings,
		out_shardings=out_shardings,
		static_argnums=static_argnums,
		donate_argnums=donate_argnums,
	).compile()


@chex.dataclass
class SampleState:
	"""
	Data class representing the state of the sampling process.
	"""

	current_length: tp.Union[jax.Array, sharding.NamedSharding]
	sequences: tp.Union[jax.Array, sharding.NamedSharding]
	running_token: tp.Union[jax.Array, sharding.NamedSharding]
	is_sequence_finished: tp.Union[jax.Array, sharding.NamedSharding]
	prng_key: tp.Union[random.PRNGKey, sharding.NamedSharding]
	model_kwargs: tp.Union[tp.Dict[str, jax.Array], sharding.NamedSharding]

	# vInference Ops
	generate_func_flops: tp.Optional[float] = float("-inf")
	interval_func_flops: tp.Optional[float] = float("-inf")
	tokens_pre_second: tp.Optional[float] = float("-inf")
	generated_tokens: tp.Optional[int] = 0

	def __repr__(self):
		"""
		Args:
		    self: Refer to the instance of the class

		Returns:
		    A string representation of the object
		"""
		string = f"{self.__class__.__name__}(\n"
		for k, v in self.__dict__.items():
			if not k.startswith("_"):
				try:
					repr_src = f"  {k} : " + v.__str__().replace("\n", "\n  ") + "\n"
					string += (
						repr_src
						if len(repr_src) < 500
						else f"  {k} : " + f"{v.__class__.__name__}(...)" + "\n"
					)
				except TypeError:
					pass
		return string.strip() + "\n)"

	__str__ = __repr__


def create_sampling_step(
	logits_processor: FlaxLogitsProcessorList,
	logits_warper: FlaxLogitsProcessorList,
	eos_token_id: jax.Array,
	pad_token_id: jax.Array,
	do_sample: bool = True,
):
	@ic
	def sampling_step(model, state: SampleState):
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
			return_dict=True,
			**state.model_kwargs,
		)

		logits = model_outputs.logits[:, -1]
		if logits_processor is not None:
			logits = logits_processor(state.sequences, logits, state.current_length)

		if do_sample:
			if logits_warper is not None:
				logits = logits_warper(logits, logits, state.current_length)
			next_token = jax.random.categorical(state.prng_key, logits, axis=-1)
		else:
			next_token = jnp.argmax(logits, axis=-1)

		next_token = (
			next_token * ~state.is_sequence_finished
			+ pad_token_id * state.is_sequence_finished
		)
		next_sequence_finished = state.is_sequence_finished | jnp.isin(
			next_token,
			eos_token_id,
		)
		next_token = next_token[:, None]
		next_sequences = jax.lax.dynamic_update_slice(
			state.sequences,
			next_token,
			(0, state.current_length),
		)
		next_model_kwargs = model.update_inputs_for_generation(
			model_outputs, state.model_kwargs
		)

		return SampleState(
			current_length=state.current_length + 1,
			sequences=next_sequences,
			running_token=next_token,
			is_sequence_finished=next_sequence_finished,
			prng_key=jax.random.split(state.prng_key, 2)[0],
			model_kwargs=next_model_kwargs,
			generated_tokens=state.generated_tokens + 1,
		)

	return sampling_step
