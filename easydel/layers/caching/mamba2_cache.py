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

import chex as cx
from fjformer import with_sharding_constraint
from fjformer.core import ImplicitArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.partition_module import PartitionAxis


@cx.dataclass
class Mamba2CacheMetaData:
	"""Metadata for Mamba2 cache configuration."""

	batch_size: int
	intermediate_size: int
	num_heads: int
	head_dim: int
	state_size: int
	conv_kernel_size: int
	n_groups: int

	@classmethod
	def create(
		cls,
		batch_size: int,
		intermediate_size: int,
		num_heads: int,
		head_dim: int,
		state_size: int,
		conv_kernel_size: int,
		n_groups: int,
	) -> "Mamba2CacheMetaData":
		"""Create a Mamba2CacheMetaData instance with validation."""
		if batch_size <= 0:
			raise ValueError("batch_size must be positive")
		if intermediate_size <= 0:
			raise ValueError("intermediate_size must be positive")
		if num_heads <= 0:
			raise ValueError("num_heads must be positive")
		if head_dim <= 0:
			raise ValueError("head_dim must be positive")
		if state_size <= 0:
			raise ValueError("state_size must be positive")
		if conv_kernel_size <= 0:
			raise ValueError("conv_kernel_size must be positive")
		if n_groups <= 0:
			raise ValueError("n_groups must be positive")

		return cls(
			batch_size=batch_size,
			intermediate_size=intermediate_size,
			num_heads=num_heads,
			head_dim=head_dim,
			state_size=state_size,
			conv_kernel_size=conv_kernel_size,
			n_groups=n_groups,
		)


@cx.dataclass
class Mamba2CacheView:
	conv_states: tp.Union[cx.Array, ImplicitArray]
	ssm_states: tp.Union[cx.Array, ImplicitArray]
	positions: cx.Array
	seqlen_offset: int
	metadata: Mamba2CacheMetaData
	layer_index: tp.Optional[int] = None

	@classmethod
	def init(
		cls,
		metadata: Mamba2CacheMetaData,
		partition_specs: PartitionSpec,
		dtype: jnp.dtype,
		layer_index: tp.Optional[int] = None,
	):
		return cls(
			conv_states=with_sharding_constraint(
				x=jnp.zeros(
					shape=(
						metadata.batch_size,
						metadata.intermediate_size + 2 * metadata.n_groups * metadata.state_size,
						metadata.conv_kernel_size,
					),
					dtype=dtype,
				),
				partition_specs=partition_specs,
			),
			ssm_states=with_sharding_constraint(
				x=jnp.zeros(
					shape=(
						metadata.batch_size,
						metadata.num_heads,
						metadata.head_dim,
						metadata.state_size,
					),
					dtype=dtype,
				),
				partition_specs=partition_specs,
			),
			positions=jnp.zeros((metadata.batch_size,), "i4"),
			metadata=metadata,
			layer_index=layer_index,
			seqlen_offset=0,
		)

	def update_conv_state(
		self,
		new_conv_state: cx.Array,
		cache_position: cx.Array,
	) -> "Mamba2CacheView":
		"""Update the convolutional state of the cache."""
		cache_position = jnp.clip(cache_position, 0, self.metadata.conv_kernel_size - 1)
		conv_state = jnp.roll(self.conv_states, shift=-1, axis=-1)
		updated_conv_states = conv_state.at[:, :, cache_position].set(new_conv_state)
		self.conv_states = updated_conv_states
		return self

	def update_ssm_state(
		self,
		new_ssm_state: cx.Array,
	) -> "Mamba2CacheView":
		"""Update the SSM state of the cache."""
		self.ssm_states = new_ssm_state
		return self

	def reset(self) -> "Mamba2CacheView":
		"""Reset both conv and ssm states to zeros."""
		self.conv_states = jnp.zeros_like(self.conv_states)
		self.ssm_states = jnp.zeros_like(self.ssm_states)
		return self


@cx.dataclass
class Mamba2Cache:
	views: tp.List[tp.Optional[Mamba2CacheView]]

	@classmethod
	def init_layers_cache(
		cls,
		num_hidden_layers: int,
		metadata: Mamba2CacheMetaData,
		dtype: tp.Optional[jnp.dtype] = None,
		partition_specs: tp.Optional[PartitionSpec] = None,
	):
		paxis = PartitionAxis()
		partition_specs = partition_specs or PartitionSpec(
			paxis.batch_axis,
			paxis.head_axis,
			paxis.sequence_axis,
		)
		if dtype is None:
			dtype = jnp.bfloat16

		return cls(
			views=[
				Mamba2CacheView.init(
					metadata=metadata,
					partition_specs=partition_specs,
					dtype=dtype,
					layer_index=layer_index,
				)
				for layer_index in range(num_hidden_layers)
			]
		)

	def update_conv_state(
		self,
		layer_idx: int,
		new_conv_state: cx.Array,
		cache_position: cx.Array,
	) -> "Mamba2Cache":
		"""
		Update the convolutional state for a specific layer.

		Arguments:
		    layer_idx: Index of the layer to update
		    new_conv_state: New state to be inserted
		    cache_position: Position in the cache to update

		Returns:
		    Updated MambaCache
		"""
		if self.views[layer_idx] is None:
			raise ValueError(f"Cache view for layer {layer_idx} is None")

		updated_view = self.views[layer_idx].update_conv_state(
			new_conv_state=new_conv_state,
			cache_position=cache_position,
		)

		new_views = list(self.views)
		new_views[layer_idx] = updated_view
		return self.replace(views=new_views)

	def update_ssm_state(
		self,
		layer_idx: int,
		new_ssm_state: cx.Array,
	) -> "Mamba2Cache":
		"""
		Update the SSM state for a specific layer.

		Arguments:
		    layer_idx: Index of the layer to update
		    new_ssm_state: New SSM state to replace the current one

		Returns:
		    Updated MambaCache
		"""
		if self.views[layer_idx] is None:
			raise ValueError(f"Cache view for layer {layer_idx} is None")

		updated_view = self.views[layer_idx].update_ssm_state(
			new_ssm_state=new_ssm_state,
		)

		new_views = list(self.views)
		new_views[layer_idx] = updated_view
		return self.replace(views=new_views)

	def reset(self) -> "Mamba2Cache":
		"""
		Reset all cache views to their initial state.

		Returns:
		    Reset MambaCache
		"""
		new_views = [view.reset() if view is not None else None for view in self.views]
		return self.replace(views=new_views)

	@classmethod
	def init_empty(cls, num_hidden_layers):
		return cls(views=[None for _ in range(num_hidden_layers)])

	def update_seq(self, num):
		for view in self.views:
			if view is not None:
				view.positions += num
				view.seqlen_offset += num

	def __repr__(self):
		return (
			f"{self.__class__.__name__}(\n  "
			+ "\n  ".join(str(view) for view in self.views)
			+ "\n)"
		)

	__str__ = __repr__
