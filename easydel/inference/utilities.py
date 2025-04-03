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
import queue
import threading
import typing as tp
from dataclasses import field

import jax
import jax.random
import numpy as np
from eformer.escale import PartitionAxis
from eformer.pytree import auto_pytree
from flax import nnx as nn
from jax import numpy as jnp
from jax import random, sharding
from jax.sharding import PartitionSpec

from easydel.utils.compiling_utils import get_safe_hash_int

from .logits_process import (
	FrequencyPenaltyLogitsProcessor,
	LogitsProcessorList,
	MinPLogitsWarper,
	# NoRepeatNGramLogitsProcessor,
	PresencePenaltyLogitsProcessor,
	RepetitionPenaltyLogitsProcessor,
	SuppressTokensLogitsProcessor,
	TemperatureLogitsWarper,
	TopKLogitsWarper,
	TopPLogitsWarper,
	hash_fn,
)


@auto_pytree
class vInferencePreCompileConfig:
	batch_size: tp.Union[int, tp.List[int]] = 1
	prefill_length: tp.Optional[tp.Union[int, tp.List[int]]] = None
	vision_included: tp.Union[bool, tp.List[bool]] = False
	vision_batch_size: tp.Optional[tp.Union[int, tp.List[int]]] = None
	vision_channels: tp.Optional[tp.Union[int, tp.List[int]]] = None
	vision_height: tp.Optional[tp.Union[int, tp.List[int]]] = None
	vision_width: tp.Optional[tp.Union[int, tp.List[int]]] = None
	required_props: tp.Optional[
		tp.Union[
			tp.Mapping[str, tp.Dict[str, tp.Any]],
			tp.List[tp.Mapping[str, tp.Dict[str, tp.Any]]],
		]
	] = None

	def _im_standalone(self):
		standalone = True
		for rfield in dataclasses.fields(self):
			attr = getattr(self, rfield.name)
			if isinstance(attr, list):
				standalone = False
		return standalone

	_is_standalone = _im_standalone

	def extract(self) -> dict:
		return dataclasses.asdict(self)

	def get_default_hash(self):
		hash_str = ""
		hash_str += str(self.batch_size) + "-"
		hash_str += str(self.prefill_length) + "-"
		hash_str += str(self.vision_included) + "-"
		hash_str += str(self.vision_batch_size) + "-"
		hash_str += str(self.vision_channels) + "-"
		hash_str += str(self.vision_height) + "-"
		hash_str += str(self.vision_width) + "-"
		hash_str += str(self.required_props)
		hash_out = get_safe_hash_int(hash_str)
		return hash_out

	__hash__ = get_default_hash

	def get_standalones(self):
		"""
		Creates standalone configurations when any field contains a list.
		Returns a list of standalone vInferencePreCompileConfig instances.

		For example, if batch_size=[1, 2, 3, 4], it will create 4 standalone configs
		with batch_size values 1, 2, 3, and 4 respectively.
		"""
		if self._is_standalone():
			return [self]

		list_fields = {}
		max_length = 0

		for rfield in dataclasses.fields(self):
			attr = getattr(self, rfield.name)
			if isinstance(attr, list):
				list_fields[rfield.name] = attr
				max_length = max(max_length, len(attr))

		# Create standalone configs
		standalone_configs = []

		for i in range(max_length):
			config_kwargs = {}

			for rfield in dataclasses.fields(self):
				attr = getattr(self, rfield.name)
				field_name = rfield.name

				if field_name in list_fields:
					list_attr = list_fields[field_name]
					# Use value at index i if available, otherwise use the last value
					if i < len(list_attr):
						config_kwargs[field_name] = list_attr[i]
					else:
						config_kwargs[field_name] = list_attr[-1]
				else:
					# For non-list fields, use the original value
					config_kwargs[field_name] = attr

			standalone_configs.append(vInferencePreCompileConfig(**config_kwargs))

		return standalone_configs


vInferencePreCompileConfig.__hash__ = vInferencePreCompileConfig.get_default_hash


@auto_pytree
class SamplingParams:
	max_tokens: int = 16
	presence_penalty: float = 0.0
	frequency_penalty: float = 0.0
	repetition_penalty: float = 1.0
	temperature: float = 0.0
	top_p: float = 1.0
	top_k: int = 0
	min_p: float = 0.0
	suppress_tokens: list[int] = field(default_factory=lambda: list())

	def get_logits_warper(self):
		warpers = LogitsProcessorList()
		warpers.append(TemperatureLogitsWarper(temperature=self.temperature))
		warpers.append(TopKLogitsWarper(top_k=self.top_k, min_tokens_to_keep=1))
		warpers.append(TopPLogitsWarper(top_p=self.top_p, min_tokens_to_keep=1))
		warpers.append(MinPLogitsWarper(min_p=self.min_p, min_tokens_to_keep=1))
		return warpers

	def get_logits_processor(self):
		processors = LogitsProcessorList()
		processors.append(SuppressTokensLogitsProcessor(self.suppress_tokens))
		processors.append(PresencePenaltyLogitsProcessor(self.presence_penalty))
		processors.append(FrequencyPenaltyLogitsProcessor(self.frequency_penalty))
		processors.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))

		return processors

	__hash__ = hash_fn


@auto_pytree
class vInferenceConfig:
	max_new_tokens: int = 64
	streaming_chunks: int = 16

	num_return_sequences: tp.Optional[tp.Union[int, tp.Dict[int, int]]] = 1
	pad_token_id: tp.Optional[int] = None
	bos_token_id: tp.Optional[int] = None
	eos_token_id: tp.Optional[tp.Union[int, tp.List[int]]] = None
	partition_rules: tp.Optional[tp.Tuple[tp.Tuple[str, tp.Any]]] = None
	partition_axis: tp.Optional[PartitionAxis] = None
	_loop_rows: tp.Optional[int] = None

	sampling_params: tp.Optional[SamplingParams] = None

	def get_partition_rules(
		self,
		# in case that someone needs to customize this
		runtime_config: tp.Optional[vInferencePreCompileConfig] = None,
	):
		if self.partition_rules is not None:
			return self.partition_rules
		assert self.partition_axis is not None, (
			"partition axis is required for state sharding"
		)
		paxis = self.partition_axis
		kvps = PartitionSpec(
			paxis.batch_axis,
			paxis.key_sequence_axis,
			paxis.head_axis,
			paxis.attention_dim_axis,
		)
		idps = PartitionSpec(paxis.batch_axis, paxis.sequence_axis)
		return (
			("(sequences|running_token)", idps),
			("model_kwargs/(attention_mask|position_ids)", idps),
			# A8BIT
			("model_kwargs/past_key_values/views/[0-9]+/(key|value)/(scale|weight)", kvps),
			# NF4
			("model_kwargs/past_key_values/views/[0-9]+/(key|value)/(packed|absmax)", kvps),
			("model_kwargs/past_key_values/views/[0-9]+/(key|value)", kvps),
			(".*", PartitionSpec()),
		)

	def __post_init__(self):
		if isinstance(self.max_new_tokens, int):
			self._loop_rows = (
				self.max_new_tokens + self.streaming_chunks - 1
			) // self.streaming_chunks
		if self.sampling_params is None:
			self.sampling_params = SamplingParams(max_tokens=self.max_new_tokens)

	__hash__ = hash_fn


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


@auto_pytree
class PagedAttentionGenerateState:
	"""Generate phase state"""

	input_ids: jax.Array  # batch_size, 1
	position_ids: jax.Array  # batch_size, 1
	page_table: jax.Array  # batch_size, num_pages_per_seq
	available_slots: queue.SimpleQueue
	active_slot_req_map: dict
	map_mutex: threading.Lock = threading.Lock()

	@classmethod
	def init(cls, metadata: "PagedAttentionCacheMetaData"):  # noqa #type:ignore
		batch_size = metadata.batch_size
		page_size = metadata.page_size
		max_seq_len = metadata.max_sequences
		tables = max_seq_len // page_size
		batch_tensor = jnp.ones((batch_size), dtype=jnp.int32)
		page_table_tensor = jnp.ones((batch_size, tables), dtype=jnp.int32)
		return cls(
			input_ids=batch_tensor,
			positions=batch_tensor,
			page_table=page_table_tensor,
			available_slots=0,
			active_slot_req_map={},
		)


@auto_pytree
class SchedulerPostProcessRequest:
	prefill_request_id: tp.Optional[str]
	prefill_input_id: tp.Union[jax.Array, np.ndarray]
	prefill_done: tp.Union[jax.Array, np.ndarray]
	generate_active_slots: tp.List[int]
	generate_active_request_ids: tp.List[str]
	generate_input_ids: tp.Union[jax.Array, np.ndarray]
	generate_done: tp.Union[jax.Array, np.ndarray]


@auto_pytree
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
	padded_length: tp.Optional[int] = 0

	_time_spent_computing: tp.Optional[float] = 0.0
	_compile_config: tp.Optional[vInferencePreCompileConfig] = None


def create_sampling_step(
	logits_processor: LogitsProcessorList,
	logits_warper: LogitsProcessorList,
	eos_token_id: jax.Array,
	pad_token_id: jax.Array,
):
	def sampling_step(graphdef, graphstate, graphother, state: SampleState):
		"""
		Performs a single sampling step for text generation.

		Args:
		    params: Model parameters.
		    state (inference_utils.SampleState): The current generation state.

		Returns:
		    inference_utils.SampleState: The updated generation state.
		"""
		model = nn.merge(graphdef, graphstate, graphother)
		model_outputs = model(
			input_ids=state.running_token,
			return_dict=True,
			**state.model_kwargs,
		)

		logits = model_outputs.logits[:, -1]
		if logits_processor is not None:
			logits = logits_processor(state.sequences, logits, state.current_length)

		if logits_warper is not None:
			logits = logits_warper(logits, logits, state.current_length)
		next_token = jax.random.categorical(state.prng_key, logits, axis=-1)

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

		return state.replace(
			current_length=state.current_length + 1,
			sequences=next_sequences,
			running_token=next_token,
			is_sequence_finished=next_sequence_finished,
			prng_key=jax.random.split(state.prng_key, 2)[0],
			model_kwargs=next_model_kwargs,
			generated_tokens=state.generated_tokens + state.sequences.shape[0],
		)

	return sampling_step
