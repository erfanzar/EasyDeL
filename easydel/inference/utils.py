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
from functools import partial
from typing import Dict, Optional, Union, List

import fjformer
import jax
import jax.experimental
import jax.experimental.pallas
import jax.random
from jax import numpy as jnp
from jax import random, sharding


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class vInferenceConfig:
	max_new_tokens: int = 64
	streaming_chunks: int = 16
	temperature: float = 0.0
	top_p: float = 0.95
	top_k: int = 50
	repetition_penalty: float = 1.0
	length_penalty: float = 1.0
	pad_token_id: Optional[int] = None
	bos_token_id: Optional[int] = None
	eos_token_id: Optional[Union[int, List[int]]] = None
	_loop_rows: Optional[int] = None

	def __post_init__(self):
		if isinstance(self.max_new_tokens, int):
			self._loop_rows = (
				self.max_new_tokens + self.streaming_chunks - 1
			) // self.streaming_chunks

	def tree_flatten(self):
		return (
			self.max_new_tokens,
			self.streaming_chunks,
			self.temperature,
			self.top_p,
			self.top_k,
			self.repetition_penalty,
			self.length_penalty,
			self.pad_token_id,
			self.bos_token_id,
			self.eos_token_id,
			self._loop_rows,
		), {}

	@classmethod
	def tree_unflatten(cls, aux, children):
		return cls(*children)

	def __hash__(self) -> int:
		int_hash = int(
			(
				"---".join(
					str(cu) for cu in self.__dict__.values() if isinstance(cu, (float, int))
				)
			)
			.replace("---", "")
			.replace(".", "")
		)

		return int_hash

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
	    mesh: Optional JAX mesh for distributed execution.
	    in_shardings: Optional input sharding specifications.
	    out_shardings: Optional output sharding specifications.
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
	    mesh: Optional JAX mesh for distributed execution.
	    in_shardings: Optional input sharding specifications.
	    out_shardings: Optional output sharding specifications.
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


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SampleState:
	"""
	Data class representing the state of the sampling process.

	Attributes:
	    current_length: Current length of the generated sequence.
	    sequences: Generated token sequences.
	    running_token: The last generated token for each sequence.
	    is_sequence_finished: Boolean array indicating finished sequences.
	    prng_key: JAX PRNG key for random sampling.
	    model_kwargs: Keyword arguments passed to the model.
	"""

	current_length: Union[jax.Array, sharding.NamedSharding]
	sequences: Union[jax.Array, sharding.NamedSharding]
	running_token: Union[jax.Array, sharding.NamedSharding]
	is_sequence_finished: Union[jax.Array, sharding.NamedSharding]
	prng_key: Union[random.PRNGKey, sharding.NamedSharding]
	model_kwargs: Union[Dict[str, jax.Array], sharding.NamedSharding]

	# vInference Ops
	generate_func_flops: Optional[float] = float("-inf")
	interval_func_flops: Optional[float] = float("-inf")
	tokens_pre_second: Optional[float] = float("-inf")
	generated_tokens: Optional[int] = 0

	def tree_flatten(self):
		return (
			self.current_length,
			self.sequences,
			self.running_token,
			self.is_sequence_finished,
			self.prng_key,
			self.model_kwargs,
			self.generate_func_flops,
			self.interval_func_flops,
			self.tokens_pre_second,
			self.generated_tokens,
		), {}

	@classmethod
	def tree_unflatten(cls, aux, children):
		return cls(*children)

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


def apply_repetition_penalty(logits, tokens, penalty):
	"""
	Applies repetition penalty to the logits.

	Args:
	    logits: Logits tensor.
	    tokens: Previously generated tokens.
	    penalty: Repetition penalty factor.

	Returns:
	    Logits tensor with repetition penalty applied.
	"""

	# Create a mask for the tokens that appear in the input
	vocab_size = logits.shape[-1]
	token_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
	token_mask = token_mask.at[tokens].set(True)

	# Apply the penalty
	logits = jnp.where(token_mask, logits / penalty, logits * penalty)

	return logits


def apply_length_penalty(logits, current_length, max_new_tokens, length_penalty):
	"""
	Applies length penalty to the logits.

	Args:
	    logits: Logits tensor.
	    current_length: Current length of the generated sequence.
	    max_len: Maximum length of the sequence.
	    length_penalty: Length penalty factor.

	Returns:
	    Logits tensor with length penalty applied.
	"""

	# Calculate the penalty factor
	penalty_factor = ((5 + current_length) / 6) ** length_penalty

	# Apply the penalty
	return logits / penalty_factor


@partial(jax.jit, static_argnames=["k"])
def cal_top_k(x, k):
	def scan(x, unused):
		indice = jnp.argmax(x, axis=1)
		return (
			jax.vmap(lambda x, y: x.at[y].set(-jnp.inf))(x, indice),
			(
				jax.vmap(lambda x, y: x[y])(x, indice),
				indice,
			),
		)

	x, (values, indices) = jax.lax.scan(scan, x, (), k)
	return values.T, indices.T


@partial(jax.jit, static_argnames=["top_p"])
def calculate_top_p(logits, top_p):
	
	topk_scores, topk_indices = jax.lax.top_k(logits, k=logits.shape[-1])
	mask_scores = jnp.full_like(logits, -float("inf"))
	cumulative_probs = jax.nn.softmax(topk_scores, axis=-1).cumsum(axis=-1)
	score_mask = cumulative_probs < top_p
	score_mask = jnp.roll(score_mask, 1)
	score_mask |= score_mask.at[:, 0].set(True)
	score_mask = score_mask.at[:, :1].set(True)
	topk_next_scores = jnp.where(score_mask, topk_scores, mask_scores)
	return jax.lax.sort_key_val(topk_indices, topk_next_scores)[-1]


@partial(jax.jit, static_argnames=["top_k"])
def calculate_top_k(logits, top_k):
	batch_size, vocab_size = logits.shape
	next_scores_flat = jnp.full(batch_size * vocab_size, -float("inf"))
	topk_scores, topk_indices = jax.lax.top_k(logits, k=top_k)
	shift = jnp.broadcast_to(
		(jnp.arange(batch_size) * vocab_size)[:, None],
		(batch_size, top_k),
	).flatten()
	topk_scores_flat = topk_scores.flatten()
	topk_indices_flat = topk_indices.flatten() + shift
	next_scores_flat = next_scores_flat.at[topk_indices_flat].set(topk_scores_flat)
	return next_scores_flat.reshape(batch_size, vocab_size)


def temperature_branch(logits, prng_key, top_k, temperature, top_p):
	"""
	Applies temperature scaling, top-k and top-p sampling to the logits.

	Args:
	    logits: Logits tensor.
	    prng_key: JAX PRNG key.
	    top_k: Number of top logits to consider.
	    temperature: Temperature scaling factor.
	    top_p: Top-p sampling threshold.

	Returns:
	    Sampled token IDs.
	"""
	logits = logits / temperature
	if top_k > 1:
		logits = calculate_top_k(logits, top_k)
	if 0 < top_p < 1.0:
		logits = calculate_top_p(logits, top_p)
	return jax.random.categorical(key=prng_key, logits=logits)


def vgready_branch(logits, prng_key, top_k, temperature, top_p):
	"""
	Performs greedy decoding on the logits.

	Args:
	    logits: Logits tensor.

	Returns:
	    Token IDs with the highest logits.
	"""
	return jnp.argmax(logits, axis=-1).reshape(-1)


def vinference_step(
	logits: jax.Array,
	tokens: jax.Array,
	prng_key: jax.random.PRNGKey,
	current_length: int,
	max_new_tokens: int,
	repetition_penalty: float,
	length_penalty: float,
	top_k: int,
	top_p: float,
	temperature: float,
):
	"""
	Performs a single inference step in the text generation process.

	This function applies repetition and length penalties to the logits,
	and then performs either temperature-based sampling or greedy decoding.
	Returns:
	    jax.Array: An array of generated token IDs.
	"""
	# Apply repetition penalty
	logits = jax.lax.cond(
		repetition_penalty != 1.0,
		apply_repetition_penalty,
		lambda x, *u: x,
		logits,
		tokens,
		repetition_penalty,
	)

	# Apply length penalty
	logits = jax.lax.cond(
		length_penalty != 1.0,
		apply_length_penalty,
		lambda x, *u: x,
		logits,
		current_length,
		max_new_tokens,
		length_penalty,
	)
	if temperature > 0.0:
		return temperature_branch(
			logits=logits,
			prng_key=prng_key,
			top_k=top_k,
			top_p=top_p,
			temperature=temperature,
		)
	return vgready_branch(
		logits=logits,
		prng_key=prng_key,
		top_k=top_k,
		top_p=top_p,
		temperature=temperature,
	)


vinference_step_compiled = jax.jit(
	vinference_step,
	static_argnames=[
		"max_new_tokens",
		"repetition_penalty",
		"length_penalty",
		"top_k",
		"top_p",
		"temperature",
	],
)


def create_sampling_step(
	model,
	max_new_tokens: int,
	repetition_penalty: float,
	length_penalty: float,
	top_k: int,
	top_p: float,
	temperature: float,
	eos_token_id: jax.Array,
	pad_token_id: jax.Array,
	current_length: int,
):
	@fjformer.core.implicit_compact
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
		next_token = vinference_step_compiled(
			model_outputs.logits[:, -1],
			state.sequences,
			state.prng_key,
			current_length,
			max_new_tokens,
			repetition_penalty,
			length_penalty,
			top_k,
			top_p,
			temperature,
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
		next_sequences = jax.lax.dynamic_update_slice(
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
			prng_key=jax.random.split(state.prng_key, 2)[0],
			model_kwargs=next_model_kwargs,
			generated_tokens=state.generated_tokens + 1,
		)

	return sampling_step
