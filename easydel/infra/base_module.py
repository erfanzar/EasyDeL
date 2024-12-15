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
from __future__ import annotations

import typing as tp
from functools import cached_property

import jax
import jax.extend
import jax.tree_util
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh
from transformers.generation.flax_utils import FlaxSampleOutput

from easydel.etils.etils import (
	EasyDeLQuantizationMethods,
	get_logger,
)
from easydel.inference.logits_process import FlaxLogitsProcessorList
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.mixins import (
	BaseModuleProtocol,
	EasyBridgeMixin,
	EasyGenerationMixin,
)
from easydel.infra.utils import quantize_linear_layers
from easydel.utils.traversals import (
	flatten_dict,
	unflatten_dict,
)

PartitionLike = tp.Optional[
	tp.Union[tp.Mapping[str, tp.Callable], tp.Mapping[tuple, tp.Callable]]
]


logger = get_logger(__name__)


_CP = tp.TypeVar("CP")


class EasyDeLBaseModule(
	nn.Module,
	BaseModuleProtocol,
	EasyBridgeMixin,
	EasyGenerationMixin,
):
	"""
	Base class for EasyDeL modules, providing common functionalities for model initialization,
	parameter handling, and integration with the EasyDeL ecosystem.
	"""

	config_class: tp.Type[EasyDeLBaseConfig]
	base_model_prefix: str
	_model_task: tp.Optional[str] = None
	_model_type: tp.Optional[str] = None

	def __init__(
		self,
		config: tp.Union[EasyDeLBaseConfig, _CP],
		dtype: jnp.dtype,
		param_dtype: jnp.dtype,
		precision: lax.PrecisionLike,
		rngs: nn.Rngs,
	):
		"""Initializes the EasyDeLBaseModule.

		Args:
		    config (EasyDeLBaseConfig): The model configuration.
		    dtype (jnp.dtype): The data type for computation.
		    param_dtype (jnp.dtype): The data type for parameters.
		    precision (jax.lax.PrecisionLike): The numerical precision.
		    rngs (nn.Rngs): The random number generators.
		"""
		self.config: tp.Union[EasyDeLBaseConfig, _CP] = config
		self.dtype: jnp.dtype = dtype
		self.param_dtype: jnp.dtype = param_dtype
		self.precision: lax.PrecisionLike = precision
		self.rngs: nn.Rngs = rngs

	@cached_property
	def graphtree_params_shape(self) -> tp.Dict:
		"""Evaluates the shape of the model's parameters and returns a dictionary."""
		graphtree = nn.eval_shape(lambda: nn.split(self, nn.Param, ...)[1])
		flattened_tree = flatten_dict(graphtree)

		param_shapes = {key: val.value for key, val in flattened_tree.items()}
		return unflatten_dict(param_shapes)

	@cached_property
	def mesh(self) -> jax.sharding.Mesh:
		"""Returns the mesh from the config."""
		return self.config.mesh

	@property
	def model_task(self) -> tp.Optional[str]:
		"""Returns the model task."""
		return self._model_task

	@property
	def model_type(self) -> tp.Optional[str]:
		"""Returns the model type."""
		return self._model_type

	@cached_property
	def causal_mask(self) -> jnp.ndarray:
		"""Returns a causal mask from the config."""
		return self.config.get_basic_causal_mask()

	@cached_property
	def frequencies(self) -> jnp.ndarray:
		"""Returns frequency values from the config."""
		return self.config.get_basic_frequencies()

	def _get_mesh(self, mesh: tp.Optional[Mesh] = None) -> Mesh:
		"""Retrieves the mesh, either from the provided argument or the config."""
		if mesh is None:
			if (
				not hasattr(self, "config")
				or not hasattr(self.config, "mesh")
				or self.config.mesh is None
			):
				raise ValueError(
					"A mesh must be provided, either as an argument or through the model config."
				)
			return self.config.mesh
		return mesh

	def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
		"""Retrieves the partition rules from input or the config"""
		if partition_rules is None:
			if not hasattr(self, "config"):
				raise ValueError(
					"Partition rules must be provided either as an argument or through the model config."
				)

			return self.config.get_partition_rules(fully_sharded_data_parallel=True)
		return partition_rules

	def _apply_sharding_fns(
		self, sharding_fns: tp.Mapping[str, tp.Callable]
	) -> nn.Module:
		"""Applies sharding functions to the model's state."""
		gdef, state, other = nn.split(self, nn.Param, ...)
		sharding_fns = flatten_dict(sharding_fns)
		_shard_keys = list(sharding_fns.keys())

		def _map(path, val: nn.VariableState):
			if val.value is not None and path in _shard_keys:
				val.value = sharding_fns[path](val.value)
			return val

		state = state.map(_map)
		return nn.merge(gdef, state, other)

	def shard_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	) -> EasyDeLBaseModule:
		"""Shards the model's parameters using the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for sharding.
		    mesh (jax.sharding.Mesh, optional): The mesh to shard across.

		Returns:
		    EasyDeLBaseModule: The sharded model.
		"""
		mesh = self._get_mesh(mesh)
		partition_rules = self._get_partition_rules(partition_rules)

		shard_fns = make_shard_and_gather_fns(
			partition_specs=match_partition_rules(
				rules=partition_rules,
				params=self.graphtree_params_shape,
			),
			mesh=mesh,
		)[0]

		return self._apply_sharding_fns(shard_fns)

	def gather_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	) -> EasyDeLBaseModule:
		"""Gathers the model's parameters based on the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for gathering.
		    mesh (jax.sharding.Mesh, optional): The mesh to gather from.

		Returns:
		    EasyDeLBaseModule: The gathered model.
		"""
		mesh = self._get_mesh(mesh)
		partition_rules = self._get_partition_rules(partition_rules)

		gather_fns = make_shard_and_gather_fns(
			partition_specs=match_partition_rules(
				rules=partition_rules,
				params=self.graphtree_params_shape,
			),
			mesh=mesh,
		)[1]
		return self._apply_sharding_fns(gather_fns)

	def quantize(
		self,
		method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
		block_size: int = 128,
		quantization_pattern: tp.Optional[str] = None,
	) -> EasyDeLBaseModule:
		"""Quantizes the model's linear layers.

		Args:
		    method (EasyDeLQuantizationMethods, optional): The quantization method to use.
		    block_size (int, optional): The block size for quantization.
		    quantization_pattern (str, optional): The quantization pattern to use.

		Returns:
		    EasyDeLBaseModule: The quantized model.
		"""
		return quantize_linear_layers(
			self,
			method=method,
			block_size=block_size,
			quantization_pattern=quantization_pattern,
		)

	# def __repr__(self):
	#   """Provides a human-readable string representation of the module."""
	#   return (
	#     f"{self.__class__.__name__}(\n"
	#     f"  model_type={self.model_type},\n"
	#     f"  model_task={self.model_task},\n"
	#     f"  config={self.config},\n"
	#     f"  dtype={self.dtype},\n"
	#     f"  param_dtype={self.param_dtype}\n"
	#     f")"
	#   )

	# __str__ = __repr__


class EasyDeLBaseVisionModule(EasyDeLBaseModule):
	def prepare_inputs_for_generation(
		self,
		input_ids: jax.Array,
		max_length: int,
		attention_mask: tp.Optional[jax.Array] = None,
		vision_mask: tp.Optional[jax.Array] = None,
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
		max_length: tp.Optional[int] = None,
		pad_token_id: tp.Optional[int] = None,
		eos_token_id: tp.Optional[int] = None,
		prng_key: tp.Optional[jnp.ndarray] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		logits_warper: tp.Optional[FlaxLogitsProcessorList] = None,
		cfg_scales: jnp.ndarray = 1.0,
		trace: bool = True,
		params: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
		model_kwargs: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
	):
		from easydel.inference.utils import SampleState

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
		generation_config: tp.Optional["transformers.GenerationConfig"] = None,  # noqa #type:ignore
		prng_key: tp.Optional[jnp.ndarray] = None,
		trace: bool = True,
		params: tp.Optional[tp.Dict[str, jnp.ndarray]] = None,
		logits_processor: tp.Optional[FlaxLogitsProcessorList] = None,
		**kwargs,
	):
		self._validate_model_class()

		if generation_config is None:
			if (
				self.generation_config._from_model_config
				and self.generation_config._original_object_hash == hash(self.generation_config)
			):
				from transformers import GenerationConfig

				new_generation_config = GenerationConfig.from_model_config(self.config)
				if new_generation_config != self.generation_config:
					logger.warn(
						"You have modified the pretrained model configuration to control generation. This is a"
						" deprecated strategy to control generation and will be removed soon, in a future version."
						" Please use and modify the model generation configuration (see"
						" https://huggingface.co/docs/transformers/generation_strategies#"
						"default-text-generation-configuration )"
					)
					self.generation_config = new_generation_config
			generation_config = self.generation_config
		import copy

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
				logger.warn(
					"The attention mask and the pad token id were not set. As a consequence, you may observe "
					"unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
				)
			eos_token_id = generation_config.eos_token_id
			if isinstance(eos_token_id, list):
				eos_token_id = eos_token_id[0]
			logger.warn(
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
				logger.warn(
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
			logger.warn(
				f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
				"to control the generation length.  recommend setting `max_new_tokens` to control"
				" the maximum length of the generation.",
				UserWarning,
			)
		elif generation_config.max_new_tokens is not None:
			if not has_default_max_length and generation_config.max_length is not None:
				logger.warn(
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
			logger.warn(
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


M = tp.TypeVar("M", bound=nn.Module)


def wrap_easydel_module(
	config_class: tp.Type[EasyDeLBaseConfig],
	base_model_prefix: str = "model",
):
	def wrapper(mdl: tp.Type[M]) -> tp.Type[EasyDeLBaseModule]:
		class_dict = {
			"config_class": config_class,
			"base_model_prefix": base_model_prefix,
			"__annotations__": {
				"config_class": tp.Type[EasyDeLBaseConfig],
				"base_model_prefix": str,
				"flax_module": tp.Type[M],
				"module_class": tp.Union[nn.Module, tp.Type[M]],
			},
		}

		for name, attr in mdl.__dict__.items():
			if not name.startswith("__"):
				class_dict[name] = attr

		WrappedModule = type(mdl.__name__, (EasyDeLBaseModule,), class_dict)
		WrappedModule.__module__ = mdl.__module__
		WrappedModule.__qualname__ = mdl.__qualname__
		WrappedModule.__doc__ = mdl.__doc__

		return WrappedModule

	return wrapper


def wrap_custom_easydel_module(
	base,
	config_class: tp.Type[EasyDeLBaseConfig],
	base_model_prefix: str = "model",
):
	def wrapper(mdl: tp.Type[M]) -> tp.Type[EasyDeLBaseModule]:
		class_dict = {
			"config_class": config_class,
			"base_model_prefix": base_model_prefix,
			"module_class": mdl,
			"flax_module": mdl,
			"__annotations__": {
				"config_class": tp.Type[EasyDeLBaseConfig],
				"base_model_prefix": str,
				"flax_module": tp.Type[M],
				"module_class": tp.Union[nn.Module, tp.Type[M]],
			},
		}

		for name, attr in mdl.__dict__.items():
			if not name.startswith("__"):
				class_dict[name] = attr

		WrappedModule = type(mdl.__name__, (base,), class_dict)
		WrappedModule.__module__ = mdl.__module__
		WrappedModule.__qualname__ = mdl.__qualname__
		WrappedModule.__doc__ = mdl.__doc__

		return WrappedModule

	return wrapper
