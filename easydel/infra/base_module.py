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
import flax
import flax.struct
import jax
import jax.extend
import jax.tree_util
from eformer.escale import make_shard_and_gather_fns, match_partition_rules
from flax import nnx as nn
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.utils import traversals
from easydel.utils.helpers import get_logger
from easydel.utils.traversals import flatten_dict, is_flatten, unflatten_dict

from .base_config import EasyDeLBaseConfig
from .etils import EasyDeLGradientCheckPointers, EasyDeLQuantizationMethods
from .loss_utils import (
	LOSS_MAPPING,
	ForCausalLMLoss,
	ForSequenceClassificationLoss,
	LossConfig,
	LossMetrics,
)
from .mixins import (
	BaseModuleProtocol,
	EasyBridgeMixin,
	EasyGenerationMixin,
)

if tp.TYPE_CHECKING:
	from easydel.infra.base_state import EasyDeLState
else:
	EasyDeLState = tp.Any

PartitionLike = tp.Optional[
	tp.Union[
		tp.Mapping[str, tp.Callable],
		tp.Mapping[tuple, tp.Callable],
	]
]


logger = get_logger(__name__)

_CP = tp.TypeVar("CP")
SELF = tp.TypeVar("SELF")


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
		_ = self.graphtree_shape
		_ = self.graphtree_params_shape
		_ = self.mesh
		_ = self.model_task
		_ = self.model_type

	@property
	def parameters(self) -> tp.Dict:
		from easydel.utils.graph_utils import iter_module_search

		parameters = {}
		for key, value in iter_module_search(self, nn.Param):
			parameters[key] = value.value
		return parameters

	@property
	def graphdef(self) -> nn.GraphDef:
		return nn.split(self, nn.Param, ...)[0]

	@property
	def graphstate(self) -> nn.GraphState:
		return nn.split(self, nn.Param, ...)[1]

	@property
	def graphother(self) -> nn.GraphState:
		return nn.split(self, nn.Param, ...)[-1]

	@property
	def graphtree_params_shape(self) -> tp.Dict:
		"""Evaluates the shape of the model's parameters and returns a dictionary."""
		graphtree = nn.eval_shape(lambda: nn.split(self, nn.Param, ...)[1])

		flattened_tree = flatten_dict(graphtree)

		param_shapes = {key: val.value for key, val in flattened_tree.items()}
		return unflatten_dict(param_shapes)

	@property
	def graphtree_shape(self) -> tp.Dict:
		"""Evaluates the shape of the modeland returns a dictionary."""
		graphtree = nn.eval_shape(lambda: nn.split(self)[1])

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

	@property
	def params(self) -> tp.Dict:
		return nn.split(self)[-1]

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
		elif getattr(self, "loss_type", None) is not None:
			loss_type = self.loss_type
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

	def to_dtype(self: SELF, dtype: jnp.dtype) -> SELF:
		"""Applies sharding functions to the model's state."""
		from easydel.utils.graph_utils import iter_module_search

		gdef, state, others = nn.split(self, nn.Param, ...)

		def _map(path, val: nn.VariableState):
			if val.value is not None:
				if not path[-1].startswith("quant_"):
					val.value = val.value.astype(dtype)
			return val

		state.update(state.map(_map))
		self = nn.merge(gdef, state, others)

		for path, module in iter_module_search(self):
			if hasattr(module, "param_dtype"):
				module.param_dtype = dtype
		return self

	def half(self: SELF, change_runtime_dtype: bool = True) -> SELF:
		if change_runtime_dtype:
			self = self._reformat_runtime_dtype(jnp.float16)
		return self._reformat_dtype(jnp.float16)

	def float(self: SELF, change_runtime_dtype: bool = True) -> SELF:
		if change_runtime_dtype:
			self = self._reformat_runtime_dtype(jnp.float32)
		return self._reformat_dtype(jnp.float32)

	def _reformat_runtime_dtype(self: SELF, dtype) -> SELF:
		from easydel.utils.graph_utils import iter_module_search

		for path, module in iter_module_search(self):
			if hasattr(module, "dtype"):
				if str(type(module.dtype)).endswith(
					"lax_numpy._ScalarMeta'>"
				):  # dont change numpy based dtypes
					module.dtype = dtype
		self.dtype = dtype
		return self

	def _reformat_dtype(self: SELF, dtype) -> SELF:
		from easydel.utils.graph_utils import iter_module_search

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

		for path, module in iter_module_search(self):
			if hasattr(module, "param_dtype"):
				if isinstance(module.param_dtype, jnp.dtype):
					module.param_dtype = dtype

		self.param_dtype = dtype
		return self

	def _match_partition_rules(self, partition_rules: tp.Any = None):
		return match_partition_rules(
			rules=self._get_partition_rules(partition_rules),
			tree=self.graphtree_params_shape,
		)

	@property
	def _specs_sharding(self):
		def _map(array):
			if hasattr(array, "sharding"):
				sharding = array.sharding
				if isinstance(sharding, NamedSharding):
					return sharding.spec
			return PartitionSpec()

		return nn.from_tree(
			jax.tree_util.tree_map(
				_map,
				nn.to_tree(self),
			)
		)

	@property
	def _shardings(self):
		return nn.from_tree(
			jax.tree_util.tree_map(
				lambda x: x.sharding if hasattr(x, "sharding") else PartitionSpec(),
				nn.to_tree(self),
			)
		)

	@property
	def _named_shardings(self):
		return nn.from_tree(
			jax.tree_util.tree_map(
				lambda x: x.sharding if hasattr(x, "sharding") else None,
				nn.to_tree(self),
			)
		)

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
		self: SELF,
		sharding_fns: tp.Mapping[str, tp.Callable],
	) -> SELF:
		"""Applies sharding functions to the model's state."""
		gdef, state, others = nn.split(self, nn.Param, ...)
		sharding_fns = flatten_dict(sharding_fns)
		_shard_keys = list(sharding_fns.keys())

		def _map(path, val: nn.VariableState):
			if val.value is not None and path in _shard_keys:
				try:
					val.value = sharding_fns[path](val.value)
				except TypeError:
					path = map(str, path)
					warnings.warn(f"couldn't shard/gather {'.'.join(path)}", stacklevel=1)
			return val

		state.update(state.map(_map))
		self = nn.merge(gdef, state, others)
		return self

	def shard_model(
		self: SELF,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
		overlay_fns: tp.Optional[tp.Mapping[str, tp.Callable]] = None,
	) -> SELF:
		"""Shards the model's parameters using the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for sharding.
		    mesh (jax.sharding.Mesh, optional): The mesh to shard across.
		                overlay_fns (tp.Optional[tp.Mapping[str, tp.Callable]]): Overlay functions to apply to the model.

		Returns:
		    EasyDeLBaseModule: The sharded model.
		"""
		mesh = self._get_mesh(mesh)
		partition_rules = self._get_partition_rules(partition_rules)
		partition_specs = match_partition_rules(
			rules=partition_rules,
			tree=self.graphtree_params_shape,
		)
		shard_fns, _ = make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)
		if overlay_fns is not None:
			shard_fns.update(overlay_fns)
		self = self._apply_sharding_fns(shard_fns)
		return self

	def gather_model(
		self: SELF,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
		overlay_fns: tp.Optional[tp.Mapping[str, tp.Callable]] = None,
	) -> SELF:
		"""Gathers the model's parameters based on the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for gathering.
		    mesh (jax.sharding.Mesh, optional): The mesh to gather from.
		                overlay_fns (tp.Optional[tp.Mapping[str, tp.Callable]]): Overlay functions to apply to the model.
		Returns:
		    EasyDeLBaseModule: The gathered model.
		"""
		mesh = self._get_mesh(mesh)
		partition_rules = self._get_partition_rules(partition_rules)
		partition_specs = match_partition_rules(
			rules=partition_rules,
			tree=self.graphtree_params_shape,
		)
		_, gather_fns = make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)

		if overlay_fns is not None:
			gather_fns.update(overlay_fns)
		return self._apply_sharding_fns(gather_fns)

	@property
	def _shard_fns(self):
		mesh = self._get_mesh(None)
		partition_specs = match_partition_rules(
			rules=self._get_partition_rules(None),
			tree=self.graphtree_params_shape,
		)
		return make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)[0]

	@property
	def _gather_fns(self):
		mesh = self._get_mesh(None)
		partition_specs = match_partition_rules(
			rules=self._get_partition_rules(None),
			tree=self.graphtree_params_shape,
		)
		return make_shard_and_gather_fns(
			partition_specs=partition_specs,
			mesh=mesh,
		)[1]

	def fully_shard(self: SELF, partition_rules: PartitionLike = None) -> SELF:
		class ShardState(flax.struct.PyTreeNode):
			graphdef: nn.GraphDef
			graphstate: nn.GraphState

		gdef, gstate = nn.split(self)
		mock = ShardState(graphdef=gdef, graphstate=gstate)
		shardings = jax.tree_util.tree_map(
			lambda x: NamedSharding(mesh=self.mesh, spec=x),
			match_partition_rules(
				self._get_partition_rules(partition_rules), nn.eval_shape(lambda: mock)
			),
		)

		@partial(jax.jit, out_shardings=shardings)
		def _call(cl):
			return cl

		mock = _call(mock)
		self = nn.merge(mock.graphdef, mock.graphstate)
		return self

	def fully_gather(self: SELF) -> SELF:
		class ShardState(flax.struct.PyTreeNode):
			graphdef: nn.GraphDef
			graphstate: nn.GraphState

		gdef, gstate = nn.split(self)
		mock = ShardState(graphdef=gdef, graphstate=gstate)
		shardings = jax.tree_util.tree_map(
			lambda x: NamedSharding(mesh=self.mesh, spec=PartitionSpec()),
			match_partition_rules(
				self._get_partition_rules(None), nn.eval_shape(lambda: mock)
			),
		)

		@partial(jax.jit, out_shardings=shardings)
		def _call(cl):
			return cl

		mock = _call(mock)
		self = nn.merge(mock.graphdef, mock.graphstate)
		return self

	def quantize(
		self: SELF,
		method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
		block_size: int = 128,
		quantization_pattern: tp.Optional[str] = None,
		quantize_tensors: bool = True,
		verbose: tp.Optional[bool] = None,
	) -> SELF:
		"""Quantizes the model's linear layers.

		Args:
		    method (EasyDeLQuantizationMethods, optional): The quantization method to use.
		    block_size (int, optional): The block size for quantization.
		    quantization_pattern (str, optional): The quantization pattern to use.
				quantize_tensors (bool): whenever to quantize tensors or quantize Linear Layers.`
				verbose (bool, optional): Verbose quantizing process
		Returns:
		    EasyDeLBaseModule: The quantized model.
		"""
		from easydel.layers.quantization.quantizers import EasyQuantizer

		quantizer = EasyQuantizer(
			quantization_method=method,
			block_size=block_size,
			quantization_pattern=quantization_pattern,
		)
		if verbose is None:
			verbose = jax.process_index() == 0
		if quantize_tensors:
			...
		else:
			self = quantizer.quantize_linears(
				self,
				quantization_pattern=quantization_pattern,
				verbose=verbose,
			)
		return self

	def to_state(self) -> EasyDeLState:
		"""converts current model to a EasyDeLState"""
		from easydel.infra.base_state import EasyDeLState

		return EasyDeLState.create(step=0, model=self)

	def to_torch(self, **kwargs):
		from easydel.utils.parameters_transformation import module_to_huggingface_model

		hf_autoloader = self.get_torch_loader()
		model_class = hf_autoloader._model_mapping[type(self.config)]
		hf_model = module_to_huggingface_model(
			module=self,
			base_huggingface_module=model_class,
			config=self.config,
			dtype=self.param_dtype,
			**kwargs,
		)
		return hf_model

	def prepare_inputs_for_call(self, **kwargs):
		return kwargs

	def get_static_arguments(self) -> tp.Tuple:
		return ()

	@classmethod
	def lazy_init(cls: tp.Type[SELF], *args, **kwargs) -> SELF:
		return nn.eval_shape(lambda: cls(*args, **kwargs))

	def merge_lora_params(self: SELF, pytree: tp.Dict) -> SELF:
		from easydel.infra.utils import merge_lora_params

		self = merge_lora_params(self, pytree)
		return self

	def split_lora_params(self: SELF) -> tp.Dict:
		from easydel.infra.utils import split_lora_params

		pytree = split_lora_params(self)
		return pytree

	def apply_lora_to_layers(
		self: SELF,
		lora_rank: int,
		lora_pattern: tp.Optional[str] = None,
		verbose: bool = False,
		rngs: tp.Optional[nn.Rngs] = None,
	) -> SELF:
		from easydel.infra.utils import apply_lora_to_layers

		self = apply_lora_to_layers(
			self,
			lora_pattern=lora_pattern,
			lora_rank=lora_rank,
			rngs=rngs,
			verbose=verbose,
		)
		return self

	def unwrap_lora_to_layers(self: SELF, verbose: bool = False) -> SELF:
		from easydel.infra.utils import unwrap_lora_to_layers

		self = unwrap_lora_to_layers(self, verbose=verbose)
		return self

	@property
	def transform_fn(self):
		from easydel.utils import graph_utils
		from easydel.utils.parameters_transformation import torch_dict_to_easydel_params

		embedding_path = [
			".".join(tuple(map(str, pa)))
			for pa, _ in graph_utils.iter_module_search(self, nn.Embed)
			if not isinstance(pa[-1], int)
		]
		layernorm_path = [
			".".join(tuple(map(str, pa)))
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
	def _generate_compatible_graphdef(self):
		from copy import deepcopy

		adjusted_config = deepcopy(self.config)
		adjusted_config.gradient_checkpointing = EasyDeLGradientCheckPointers.NONE
		dummy = type(self).lazy_init(
			config=adjusted_config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			rngs=self.rngs,
		)
		gdef, _, _ = nn.split(dummy, nn.Param, ...)
		return gdef

	@property
	def _generate_compatible_graphother(self):
		from copy import deepcopy

		adjusted_config = deepcopy(self.config)
		adjusted_config.gradient_checkpointing = EasyDeLGradientCheckPointers.NONE
		dummy = type(self).lazy_init(
			config=adjusted_config,
			dtype=self.dtype,
			param_dtype=self.param_dtype,
			precision=self.precision,
			rngs=self.rngs,
		)
		_, _, gother = nn.split(dummy, nn.Param, ...)
		gother = traversals.recreate_meta_values(gother)
		return gother

	@property
	def params_sharding(self) -> tp.Dict:
		return jax.tree_util.tree_map(
			lambda x: x.sharding if hasattr(x, "sharding") else None,
			self.split_params_dict(),
		)

	def merge_params(self, tree):
		"""merge state to the current model"""
		gdef, _, gother = nn.split(self, nn.Param, ...)
		self = nn.merge(gdef, tree, gother)
		return self

	def split_params(self):
		"""split the model parameters"""
		return nn.split(self, nn.Param, ...)[1]

	def split_params_dict(
		self,
		extract_fn: tp.Optional[tp.Callable] = None,
		remove_none: bool = True,
	) -> tp.Dict:
		"""Splits the model parameters and returns them as a dictionary, removing `VariableState` from the tree.

		Args:
		        extract_fn (tp.Optional[tp.Callable], optional): Function to extract values from the parameters.
		        remove_none (bool, optional): Whether to remove `None` values from the dictionary.

		Returns:
		        tp.Dict: The dictionary of split parameters.
		"""
		flat_params = flatten_dict(self.split_params().to_pure_dict(extract_fn=extract_fn))
		if remove_none:
			flat_params = {
				k: v.value if hasattr(v, "value") else v
				for k, v in flat_params.items()
				if (v.value if hasattr(v, "value") else v) is not None
			}
		else:
			flat_params = {
				k: v.value if hasattr(v, "value") else v for k, v in flat_params.items()
			}
		return unflatten_dict(flat_params)

	def merge_params_dict(self: SELF, params_dict: tp.Dict) -> SELF:
		"""Merges the model parameters from a dictionary into the current model.

		Args:
		        params_dict (tp.Dict): A dictionary containing the parameters to merge.

		Returns:
		        EasyDeLBaseModule: The model with merged parameters.
		"""
		current_state = self.split_params().flat_state()
		if not is_flatten(params_dict):
			params_dict = flatten_dict(params_dict)
		for key, value in params_dict.items():
			if key in current_state:
				current_state[key].value = value
			else:
				raise KeyError(f"Parameter key {key} not found in the current model state.")
		self = self.merge_params(unflatten_dict(current_state))
		return self

	def _flop(self, *args, **kwargs) -> tp.Optional[float]:
		"""Calculates the FLOP (Floating Point Operations) from JaxPr"""
		from .utils import count_flop_jaxpr

		return count_flop_jaxpr(jax.make_jaxpr(self.__call__)(*args, **kwargs))

	@property
	def pure_transform_fn(self):
		from easydel.utils import graph_utils
		from easydel.utils.parameters_transformation import torch_dict_to_easydel_params

		embedding_path = [
			".".join(tuple(map(str, pa)))
			for pa, _ in graph_utils.iter_module_search(self, nn.Embed)
			if not isinstance(pa[-1], int)
		]
		layernorm_path = [
			".".join(tuple(map(str, pa)))
			for pa, _ in graph_utils.iter_module_search(self, nn.LayerNorm)
			if not isinstance(pa[-1], int)
		]

		return partial(
			torch_dict_to_easydel_params,
			embedding_layer_names=embedding_path,
			layernorm_names=layernorm_path,
			dtype=self.param_dtype,
		)

	@property
	def _default_loss_config(self) -> tp.Optional[LossConfig]:
		return None

	@_default_loss_config.setter
	def _default_loss_config(self, val):
		return val

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

		if self.loss_function.__name__ == ForSequenceClassificationLoss.__name__:
			if loss_config is None:
				assert hasattr(self.config, "num_labels"), (
					"in order to use `SequenceClassification` Models in `EasyDeL` you first need to attach `num_labels` to model `config`"
				)
				loss_config = LossConfig(num_labels=self.config.num_labels)

		assert labels is not None, "`labels` can not be `None` for computing loss."
		loss_kwargs = loss_kwargs or {}
		batch.pop("return_dict", None)
		outputs = self(**batch, return_dict=True)

		loss_output: LossMetrics = self.loss_function(
			labels=labels,
			config=loss_config,
			paxis=self.config.partition_axis,
			**loss_kwargs,
			**outputs,
			**batch,
		)
		if hasattr(outputs, "aux_loss"):
			if outputs.aux_loss is not None:
				loss_output.loss = loss_output.loss + outputs.aux_loss
		outputs = outputs.replace(loss=loss_output.loss)
		return outputs, loss_output
