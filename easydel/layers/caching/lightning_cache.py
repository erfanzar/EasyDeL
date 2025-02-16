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

import chex as cx
from eformer.jaximus import ImplicitArray

if tp.TYPE_CHECKING:
	from easydel.layers.quantization.quantizers import EasyQuantizer
else:
	EasyQuantizer = object


@cx.dataclass
class LightningCacheMetaData:
	"""Metadata for transformer cache configuration."""

	batch_size: tp.Optional[int]
	num_heads: tp.Optional[int]
	head_dim: tp.Optional[int]
	key_heads: tp.Optional[int]
	value_heads: tp.Optional[int]
	key_dim: tp.Optional[int]
	value_dim: tp.Optional[int]

	@classmethod
	def create(
		cls,
		batch_size: tp.Optional[int] = None,
		num_heads: tp.Optional[int] = None,
		head_dim: tp.Optional[int] = None,
		key_heads: tp.Optional[int] = None,
		value_heads: tp.Optional[int] = None,
		key_dim: tp.Optional[int] = None,
		value_dim: tp.Optional[int] = None,
	) -> LightningCacheMetaData:
		"""
		Create a LightningCacheMetaData instance with validation.
		Returns:
		    LightningCacheMetaData instance

		Raises:
		    ValueError: If required parameters are missing or invalid.
		"""
		return cls(
			batch_size=batch_size,
			num_heads=num_heads,
			head_dim=head_dim,
			key_heads=key_heads,
			value_heads=value_heads,
			key_dim=key_dim,
			value_dim=value_dim,
		)


@cx.dataclass
class LightningCacheView:
	key_value: tp.Union[cx.Array, ImplicitArray]
	metadata: LightningCacheMetaData
	layer_index: tp.Optional[int] = None

	@classmethod
	def init(cls, metadata: LightningCacheMetaData, layer_index: tp.Optional[int] = None):
		return cls(
			key_value=None,
			metadata=metadata,
			layer_index=layer_index,
		)

	def __repr__(self):
		return (
			self.__class__.__name__
			+ f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
		)

	__str__ = __repr__


@cx.dataclass
class LightningCache:
	views: tp.List[tp.Optional[LightningCacheView]]

	@classmethod
	def init_layers_cache(cls, num_hidden_layers: int, metadata: LightningCacheMetaData):
		return cls(
			views=[
				LightningCacheView.init(metadata=metadata, layer_index=layer_index)
				for layer_index in range(num_hidden_layers)
			]
		)

	@classmethod
	def init_empty(cls, num_hidden_layers):
		return cls(views=[None for _ in range(num_hidden_layers)])

	def __repr__(self):
		return (
			f"{self.__class__.__name__}(\n  "
			+ "\n  ".join(str(view) for view in self.views)
			+ "\n)"
		)

	__str__ = __repr__
