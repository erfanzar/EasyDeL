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

import typing as tp
import chex

from easydel.etils.etils import EasyDeLQuantizationMethods
from easydel.layers.caching.transformer_cache import TransformerCache
from easydel.modules._base.base_config import EasyDeLBaseConfig
from easydel.modules.modeling_flax_outputs import (
	FlaxCausalLMOutput,
	FlaxSequenceClassifierOutput,
	MoeModelOutput,
	MoeCausalLMOutput,
)
from jax.sharding import Mesh
from flax import nnx as nn


PartitionLike = tp.Optional[
	tp.Union[tp.Mapping[str, tp.Callable], tp.Mapping[tuple, tp.Callable]]
]
_CP = tp.Type[EasyDeLBaseConfig]
_T = tp.TypeVar("_T")


def return_type_adjuster(
	original_return_type: tp.Type[_T],
) -> tp.Callable[[tp.Callable[..., nn.Module]], tp.Callable[..., _T]]:
	def decorator(func: tp.Callable[..., nn.Module]) -> tp.Callable[..., _T]:
		def wrapper(*args: tp.Any, **kwargs: tp.Any) -> _T:
			return tp.cast(_T, func(*args, **kwargs))

		return wrapper

	return decorator


class BaseModuleProtocol:
	"""
	Protocol defining the common interface for EasyDeL modules.
	"""

	config_class: tp.Type[EasyDeLBaseConfig]
	config: EasyDeLBaseConfig
	base_model_prefix: str
	_model_task: tp.Optional[str] = None
	_model_type: tp.Optional[str] = None

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxCausalLMOutput, tp.Tuple]: ...

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		return_dict: bool = True,
	) -> tp.Union[FlaxSequenceClassifierOutput, tp.Tuple]: ...

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeModelOutput, tp.Tuple]: ...

	@tp.overload
	def __call__(
		self,
		input_ids: tp.Optional[chex.Array] = None,
		input_embeds: tp.Optional[chex.Array] = None,
		attention_mask: tp.Optional[chex.Array] = None,
		position_ids: tp.Optional[chex.Array] = None,
		segment_ids: tp.Optional[chex.Array] = None,
		output_attentions: tp.Optional[bool] = None,
		output_hidden_states: tp.Optional[bool] = None,
		output_router_logits: tp.Optional[bool] = None,
		past_key_values: tp.Optional[TransformerCache] = None,
		return_dict: bool = True,
	) -> tp.Union[MoeCausalLMOutput, tp.Tuple]: ...

	def _get_mesh(self, mesh: tp.Optional[Mesh] = None) -> Mesh:
		"""Retrieves the mesh, either from the provided argument or the config."""

	def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
		"""Retrieves the partition rules from input or the config"""

	def _apply_sharding_fns(self, sharding_fns: tp.Mapping[str, tp.Callable]):
		"""Applies sharding functions to the model's state."""

	def shard_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	):
		"""Shards the model's parameters using the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for sharding.
		    mesh (jax.sharding.Mesh, optional): The mesh to shard across.

		Returns:
		    nn.Module: The sharded model.
		"""

	def gather_model(
		self,
		partition_rules: PartitionLike = None,
		mesh: tp.Optional[Mesh] = None,
	):
		"""Gathers the model's parameters based on the specified partitioning rules and mesh.

		Args:
		    partition_rules (PartitionLike, optional): Partitioning rules for gathering.
		    mesh (jax.sharding.Mesh, optional): The mesh to gather from.

		Returns:
		    nn.Module: The gathered model.
		"""

	def quantize(
		self,
		method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.A8BIT,
		block_size: int = 128,
		quantization_pattern: tp.Optional[str] = None,
	):
		"""Quantizes the model's linear layers.

		Args:
		    method (EasyDeLQuantizationMethods, optional): The quantization method to use.
		    block_size (int, optional): The block size for quantization.
		    quantization_pattern (str, optional): The quantization pattern to use.

		Returns:
		    nn.Module: The quantized model.
		"""
