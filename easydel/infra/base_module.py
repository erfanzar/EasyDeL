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

import re
import typing as tp
import warnings
from functools import cached_property, partial

import chex
import jax
import jax.extend
import jax.tree_util
from fjformer.sharding import make_shard_and_gather_fns, match_partition_rules
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh

from easydel.etils.etils import EasyDeLQuantizationMethods, get_logger
from easydel.utils.traversals import flatten_dict, unflatten_dict

from .base_config import EasyDeLBaseConfig
from .loss_utils import (
	LOSS_MAPPING,
	ForCausalLMLoss,
	LossConfig,
	LossMetrics,
)
from .mixins import (
	BaseModuleProtocol,
	EasyBridgeMixin,
	EasyGenerationMixin,
)
from .utils import quantize_linear_layers

if tp.TYPE_CHECKING:
	from easydel.etils.easystate import EasyDeLState
else:
	EasyDeLState = tp.Any

PartitionLike = tp.Optional[
	tp.Union[
		tp.Mapping[str, tp.Callable],
		tp.Mapping[tuple, tp.Callable],
	]
]


logger = get_logger(__name__)

MO = tp.TypeVar("MO")
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

		# these useless call's are just here to init values in graphdef
		_ = self.graphtree_params_shape
		_ = self.mesh
		_ = self.model_task
		_ = self.model_type

	@property
	def parameters(self):
		from easydel.utils.graph_utils import iter_module_search

		parameters = {}
		for key, value in iter_module_search(self, nn.Param):
			parameters[key] = value.value
		return parameters

	@property
	def graphtree_params_shape(self) -> tp.Dict:
		"""Evaluates the shape of the model's parameters and returns a dictionary."""
		graphtree = nn.eval_shape(lambda: nn.split(self, nn.Param, ...)[1])

		flattened_tree = flatten_dict(graphtree)

		param_shapes = {key: val.value for key, val in flattened_tree.items()}
		return unflatten_dict(param_shapes)

	@property
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

	@cached_property
	def static_arguments(self) -> tp.Tuple:
		return self.get_static_arguments()

	@cached_property
	def loss_function(self):
		if getattr(self.config, "loss_type", None) is not None:
			loss_type = self.config.loss_type
		else:
			loss_type = self.__class__.__name__
			if loss_type not in LOSS_MAPPING:
				loss_groups = f"({'|'.join(LOSS_MAPPING)})"
				loss_type = re.findall(loss_groups, self.__class__.__name__)
				if len(loss_type) > 0:
					loss_type = loss_type[0]
				else:
					loss_type = None
		if (
			loss_type is None
			or loss_type not in LOSS_MAPPING
			and getattr(self.config, "loss_type", None) is not None
		):
			warnings.warn(
				f"`loss_type={loss_type}` was set in the config but it is unrecognised."
				f"Using the default loss: `ForCausalLMLoss`.",
				stacklevel=1,
			)
			loss_type = "ForCausalLM"
		return LOSS_MAPPING[loss_type]

	@property
	def module_dtype(self) -> jnp.dtype:
		params_state = nn.split(self, nn.Param, ...)[1].flat_state()
		return jax.tree_util.tree_leaves(params_state)[0].dtype

	def half(self) -> EasyDeLBaseModule:
		return self._reformat_dtype(jnp.float16)

	def float(self) -> EasyDeLBaseModule:
		return self._reformat_dtype(jnp.float32)

	def _reformat_dtype(self, dtype) -> EasyDeLBaseModule:
		gdef, gtree, others = nn.split(self, nn.Param, ...)

		def _map(array):
			if array.dtype in [
				jnp.bfloat16,
				jnp.float16,
				jnp.float32,
				jnp.float64,
				jnp.float_,
			]:
				array = array.astype(dtype)
			return array

		gtree = jax.tree_util.tree_map(_map, gtree)
		self = nn.merge(gdef, gtree, others)
		self.dtype = dtype
		self.param_dtype = dtype
		return self

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

	@property
	def _shard_fns(self):
		mesh = self._get_mesh(None)
		partition_specs = match_partition_rules(
			rules=self._get_partition_rules(None),
			params=self.graphtree_params_shape,
		)
		return make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)[0]

	@property
	def _gather_fns(self):
		mesh = self._get_mesh(None)
		partition_specs = match_partition_rules(
			rules=self._get_partition_rules(None),
			params=self.graphtree_params_shape,
		)
		return make_shard_and_gather_fns(partition_specs=partition_specs, mesh=mesh)[1]

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

	def to_state(self) -> EasyDeLState:
		"""converts current model to a EasyDeLState"""
		from easydel.etils.easystate import EasyDeLState

		return EasyDeLState.create(step=0, model=self)

	def to_torch(self, **kwargs):
		from easydel.utils.parameters_transformation import module_to_huggingface_model

		hf_autoloader = self.get_torch_loader()
		model_class = hf_autoloader._model_mapping[type(self.config)]
		hf_model = module_to_huggingface_model(
			module=self,
			base_huggingface_module=model_class,
			config=self.config,
			**kwargs,
		)
		return hf_model

	def prepare_inputs_for_call(self, **kwargs):
		return kwargs

	def get_static_arguments(self) -> tp.Tuple:
		return ()

	@classmethod
	def lazy_init(cls: tp.Type[MO], *args, **kwargs) -> MO:
		return nn.eval_shape(lambda: cls(*args, **kwargs))

	def apply_lora_to_layers(
		self,
		lora_rank: int,
		lora_pattern: tp.Optional[str] = None,
		verbose: bool = False,
		rngs: tp.Optional[nn.Rngs] = None,
	):
		from easydel.infra.utils import apply_lora_to_layers

		self = apply_lora_to_layers(
			self,
			lora_pattern=lora_pattern,
			lora_rank=lora_rank,
			rngs=rngs,
			verbose=verbose,
		)
		return self

	def unwrap_lora_to_layers(self, verbose: bool = False):
		from easydel.infra.utils import unwrap_lora_to_layers

		self = unwrap_lora_to_layers(self, verbose=verbose)
		return unwrap_lora_to_layers

	@property
	def transform_fn(self):
		from easydel.utils import graph_utils
		from easydel.utils.parameters_transformation import torch_dict_to_easydel_params

		embedding_path = [
			pa[-1]
			for pa, _ in graph_utils.iter_module_search(self, nn.Embed)
			if not isinstance(pa[-1], int)
		]
		layernorm_path = [
			pa[-1]
			for pa, _ in graph_utils.iter_module_search(self, nn.LayerNorm)
			if not isinstance(pa[-1], int)
		]

		return partial(
			torch_dict_to_easydel_params,
			embedding_layer_names=embedding_path,
			layernorm_names=layernorm_path,
			dtype=self.param_dtype,
			shard_fns=self._shard_fns,
		)

	@property
	def pure_transform_fn(self):
		from easydel.utils import graph_utils
		from easydel.utils.parameters_transformation import torch_dict_to_easydel_params

		embedding_path = [
			pa[-1]
			for pa, _ in graph_utils.iter_module_search(self, nn.Embed)
			if not isinstance(pa[-1], int)
		]
		layernorm_path = [
			pa[-1]
			for pa, _ in graph_utils.iter_module_search(self, nn.LayerNorm)
			if not isinstance(pa[-1], int)
		]

		return partial(
			torch_dict_to_easydel_params,
			embedding_layer_names=embedding_path,
			layernorm_names=layernorm_path,
			dtype=self.param_dtype,
		)

	def compute_loss(
		self,
		*,
		labels: tp.Optional[chex.Array] = None,
		loss_config: tp.Optional[LossConfig] = None,
		loss_kwargs: tp.Optional[tp.Dict] = None,
		**batch,
	) -> tp.Tuple[tp.Any, LossMetrics]:
		"""basic `compute_loss` call"""
		if labels is None and self.loss_function.__name__ == ForCausalLMLoss.__name__:
			labels = batch.get("input_ids", None)
		assert labels is not None, "`labels` can not be `None` for computing loss."
		loss_kwargs = loss_kwargs or {}
		batch.pop("return_dict", None)
		outputs = self(**batch, return_dict=True)
		loss_output: LossMetrics = self.loss_function(
			labels=labels,
			config=loss_config,
			**loss_kwargs,
			**outputs,
		)
		if hasattr(outputs, "aux_loss"):
			if outputs.aux_loss is not None:
				loss_output.loss = loss_output.loss + outputs.aux_loss
		outputs = outputs.replace(loss=loss_output.loss)
		return outputs, loss_output
