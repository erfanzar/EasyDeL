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

import math
import typing as tp

import chex as cx
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from eformer import common_types
from eformer import escale as es
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps

from .._abstracts import (
	BaseCache,
	BaseCacheMetadata,
	BaseCacheView,
)

if tp.TYPE_CHECKING:
	from easydel.layers.quantization.quantizers import EasyQuantizer
else:
	EasyQuantizer = object


@auto_pytree
class PagedAttentionCacheMetaData(BaseCacheMetadata):
	"""
	Metadata holding configuration parameters for the Paged Attention KV cache.

	This class stores static configuration details required to initialize and manage
	a paged KV cache, such as dimensions, page sizes, and resource utilization hints.
	It inherits from `BaseCacheMetadata`.

	Attributes:
	    batch_size (int): The maximum number of sequences processed concurrently during decoding.
	    num_hidden_layers (int): The total number of transformer layers in the model.
	    num_pages_per_layer (int): The total number of physical memory pages allocated
	        for the KV cache per layer across all sequences. This is calculated based on
	        available memory and `hbm_utilization`.
	    num_pages_per_sequence (int): The maximum number of pages a single sequence
	        can occupy, determined by `max_sequences` and `page_size`.
	    max_sequences (int): The maximum sequence length supported by the cache allocation.
	    page_size (int): The number of tokens stored per page in the KV cache.
	    num_kv_heads (int): The number of key/value heads in the attention mechanism.
	    kv_head_dim_size (int): The dimension size of each key/value head.
	    hbm_utilization (float): The target fraction of available High Bandwidth Memory (HBM)
	        to be utilized for the KV cache pages. Should be between 0.0 and 1.0.
	"""

	batch_size: int
	num_hidden_layers: int
	num_pages_per_layer: int
	num_pages_per_sequence: int
	max_sequences: int
	page_size: int
	num_kv_heads: int
	kv_head_dim_size: int
	hbm_utilization: float

	@staticmethod
	def _usable_hbm(hbm_utilization: float, mesh: Mesh) -> int:
		"""
		Calculates the usable HBM in bytes based on utilization factor and mesh.
		(Internal helper method).
		"""
		per_device_memory_stats = jax.devices()[0].memory_stats()
		limit = per_device_memory_stats["bytes_reservable_limit"]
		used = per_device_memory_stats["bytes_in_use"]
		return (int(limit * hbm_utilization) - used) * mesh.devices.size

	@classmethod
	def create(
		cls,
		mesh: Mesh,
		batch_size: int,
		num_hidden_layers: int,
		max_sequences: int,
		page_size: int,
		num_kv_heads: int,
		kv_head_dim_size: int,
		hbm_utilization: float,
		dtype: jnp.dtype = jnp.bfloat16,
	) -> "PagedAttentionCacheMetaData":
		"""
		Factory method to create and initialize a PagedAttentionCacheMetaData instance.

		Calculates derived values like `num_pages_per_layer` and `num_pages_per_sequence`
		based on the provided parameters and estimated available memory.

		Args:
		    mesh (Mesh): The JAX device mesh.
		    batch_size (int): Maximum concurrent sequences for decode.
		    num_hidden_layers (int): Number of transformer layers.
		    max_sequences (int): Maximum supported sequence length.
		    page_size (int): Number of tokens per cache page.
		    num_kv_heads (int): Number of KV heads.
		    kv_head_dim_size (int): Dimension of each KV head.
		    hbm_utilization (float): Target HBM utilization fraction (0.0 to 1.0).
		    dtype (jnp.dtype): Data type used for cache size calculation.

		Returns:
		    PagedAttentionCacheMetaData: An initialized metadata object.

		Raises:
		    ValueError: If input parameters are invalid (e.g., non-positive dimensions,
		        invalid utilization factor).
		"""
		if batch_size <= 0:
			raise ValueError("`batch_size` must be positive")
		if num_hidden_layers <= 0:
			raise ValueError("`num_hidden_layers` must be positive")
		if max_sequences <= 0:
			raise ValueError("`max_sequences` must be positive")
		if page_size <= 0:
			raise ValueError("`page_size` must be positive")
		if num_kv_heads <= 0:
			raise ValueError("`num_kv_heads` must be positive")
		if kv_head_dim_size <= 0:
			raise ValueError("`kv_head_dim_size` must be positive")
		if not (0.0 < hbm_utilization < 1.0):
			raise ValueError("`hbm_utilization` must be positive float value in range 0~1")

		hbm_bytes = cls._usable_hbm(hbm_utilization, mesh)
		item_size = np.dtype(dtype).itemsize
		per_kv_bytes = num_kv_heads * kv_head_dim_size * item_size * 2
		num_pages_per_layer = (hbm_bytes // num_hidden_layers) // (page_size * per_kv_bytes)
		num_pages_per_sequence = math.ceil(max_sequences / page_size)
		return cls(
			batch_size=batch_size,
			num_hidden_layers=num_hidden_layers,
			num_pages_per_layer=num_pages_per_layer,
			num_pages_per_sequence=num_pages_per_sequence,
			max_sequences=max_sequences,
			page_size=page_size,
			num_kv_heads=num_kv_heads,
			kv_head_dim_size=kv_head_dim_size,
			hbm_utilization=hbm_utilization,
		)


@auto_pytree
class PagedAttentionCacheView(BaseCacheView):
	"""
	Represents the view of the Paged Attention KV cache for a single transformer layer.

	It holds references to the physical key and value pages allocated for this layer
	and the associated metadata. It provides methods to write new key/value pairs
	into the correct pages based on runtime metadata. It inherits from `BaseCacheView`.

	Attributes:
	    metadata (PagedAttentionCacheMetaData): The static configuration metadata for the
	        entire paged cache.
	    layer_index (int): The index of the transformer layer this view corresponds to.
	    key_pages (tp.Union[cx.Array, ImplicitArray]): The tensor holding all key pages for this layer.
	        Shape: (num_kv_heads, num_pages_per_layer, page_size, kv_head_dim_size).
	        Can be a JAX array or an ImplicitArray if quantization is used.
	    value_pages (tp.Union[cx.Array, ImplicitArray]): The tensor holding all value pages for this layer.
	        Shape: (num_kv_heads, num_pages_per_layer, page_size, kv_head_dim_size).
	        Can be a JAX array or an ImplicitArray if quantization is used.
	"""

	metadata: PagedAttentionCacheMetaData
	layer_index: int

	key_pages: tp.Union[cx.Array, ImplicitArray]
	value_pages: tp.Union[cx.Array, ImplicitArray]

	@classmethod
	def init(
		cls,
		mesh: Mesh,
		dtype: jnp.dtype,
		metadata: PagedAttentionCacheMetaData,
		layer_index: int,
		partition_manager: es.PartitionManager,
		quantizer: tp.Optional["EasyQuantizer"] = None,
	):
		"""
		Initializes the PagedAttentionCacheView for a specific layer.

		Allocates the `key_pages` and `value_pages` tensors with the appropriate
		shape, dtype, and sharding based on the provided metadata and partition manager.
		Optionally applies quantization if a quantizer is provided.

		Args:
		    mesh (Mesh): The JAX device mesh.
		    dtype (jnp.dtype): The data type for the cache pages (e.g., jnp.bfloat16).
		    metadata (PagedAttentionCacheMetaData): Static configuration for the cache.
		    layer_index (int): The index of the layer this view is for.
		    partition_manager (es.PartitionManager): Manages tensor sharding across the mesh.
		    quantizer (tp.Optional["EasyQuantizer"]): Optional quantizer to apply to the pages.

		Returns:
		    PagedAttentionCacheView: An initialized cache view for the specified layer.
		"""
		from easydel.infra.etils import EasyDeLQuantizationMethods
		from easydel.layers.quantization.quantizers import EasyQuantizer

		quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)

		kv_pages_shape = (
			metadata.num_kv_heads,
			metadata.num_pages_per_layer,
			metadata.page_size,
			metadata.kv_head_dim_size,
		)

		kv_pages_sharding = partition_manager.resolve(
			[
				common_types.HEAD,
				common_types.EMPTY,
				common_types.EMPTY,
				common_types.EMPTY,
			],
			mode=common_types.MODE_PREFILL,
			shape=kv_pages_shape,
		)

		kv_pages_sharding = Ns(mesh=mesh, spec=kv_pages_sharding)

		with jax.named_scope("easydel-paged-attention-cache-init"):
			key_pages = jnp.zeros(
				shape=kv_pages_shape,
				dtype=dtype,
				device=kv_pages_sharding,
			)
			value_pages = jnp.zeros(
				shape=kv_pages_shape,
				dtype=dtype,
				device=kv_pages_sharding,
			)

			key_pages = quantizer(key_pages)
			value_pages = quantizer(value_pages)

			return cls(
				metadata=metadata,
				layer_index=layer_index,
				key_pages=key_pages,
				value_pages=value_pages,
			)

	def concatenate_to_cache(self, *args, **kwargs):
		"""
		Concatenation is not applicable for Paged Attention.
		Raises NotImplementedError.
		"""
		raise NotImplementedError()

	def write_prefill_to_cache(
		self,
		key: cx.Array,
		value: cx.Array,
		metadata: PagedAttentionMetadata,
	):
		"""
		Writes the key/value pairs from a prefill step into the appropriate cache pages.

		Uses the `prefill_page_table` from the runtime `metadata` to determine which
		physical pages (`key_pages`, `value_pages`) correspond to the logical pages
		of the prefill sequence. It transposes and reshapes the input key/value tensors
		and uses `jax.lax.dynamic_update_slice_in_dim` within a `while_loop` to update
		the relevant pages.

		Args:
		    key (cx.Array): Key tensor for the prefill sequence. Shape
		        (padded_prefill_len, num_kv_heads, kv_head_dim_size).
		    value (cx.Array): Value tensor for the prefill sequence. Shape
		        (padded_prefill_len, num_kv_heads, kv_head_dim_size).
		    metadata (PagedAttentionMetadata): Runtime metadata containing the
		        `prefill_length` and `prefill_page_table`.

		Returns:
		    PagedAttentionCacheView: Returns `self` after updating the pages.
		"""
		padded_prefill_len = key.shape[0]
		kv_heads = self.key_pages.shape[0]
		page_size = self.key_pages.shape[2]
		num_pages = padded_prefill_len // page_size
		num_pages = jnp.where(num_pages < 1, 1, num_pages)
		num_active_pages, reminder = jnp.divmod(metadata.prefill_length, page_size)
		num_active_pages += jnp.where(reminder > 0, 1, 0)
		head_dim = self.key_pages.shape[-1]
		key = (
			key.transpose((1, 0, 2))
			.reshape((kv_heads, -1, page_size, head_dim))
			.astype(self.key_pages.dtype)
		)
		value = (
			value.transpose((1, 0, 2))
			.reshape((kv_heads, -1, page_size, head_dim))
			.astype(self.value_pages.dtype)
		)

		def update_cond(carry):
			_, idx = carry
			return idx < num_active_pages

		def per_page_update(carry):
			(kp, vp), idx = carry
			page_k = key[:, idx, :, :][:, None, :, :]
			page_v = value[:, idx, :, :][:, None, :, :]
			mapped_idx = metadata.prefill_page_table[idx]
			kp = jax.lax.dynamic_update_slice_in_dim(
				kp,
				page_k,
				mapped_idx,
				axis=1,
			)
			vp = jax.lax.dynamic_update_slice_in_dim(
				vp,
				page_v,
				mapped_idx,
				axis=1,
			)
			idx += 1
			return (kp, vp), idx

		idx = 0
		(self.key_pages, self.value_pages), idx = jax.lax.while_loop(
			update_cond,
			per_page_update,
			((self.key_pages, self.value_pages), idx),
		)
		return self

	def write_decodes_to_cache(
		self,
		key: cx.Array,
		value: cx.Array,
		metadata: PagedAttentionMetadata,
	):
		"""
		Writes the key/value pairs from a decode step into the appropriate cache pages.

		Uses the `decodes_position` and `decodes_page_table` from the runtime `metadata`
		to calculate the exact page index and offset within that page where the new
		key/value pair for each sequence in the batch should be written. It reshapes
		the cache pages and input keys/values for efficient scattered updates using
		`.at[...].set(...)`.

		Args:
		    key (cx.Array): Key tensor for the decode tokens. Shape
		        (batch_size, num_kv_heads, kv_head_dim_size).
		    value (cx.Array): Value tensor for the decode tokens. Shape
		        (batch_size, num_kv_heads, kv_head_dim_size).
		    metadata (PagedAttentionMetadata): Runtime metadata containing
		        `decodes_position` and `decodes_page_table`.

		Returns:
		    PagedAttentionCacheView: Returns `self` after updating the pages.
		"""
		key = key.transpose((1, 0, 2))
		value = value.transpose((1, 0, 2))

		key = key.astype(self.key_pages.dtype)
		value = value.astype(self.value_pages.dtype)
		num_tokens = key.shape[1]
		kv_heads, num_pages, page_size, head_dim = self.key_pages.shape
		page_idx, offset = jnp.divmod(metadata.decodes_position, page_size)
		page_to_update = metadata.decodes_page_table[jnp.arange(0, num_tokens), page_idx]
		mapped_page_to_update = page_to_update * page_size + offset
		mapped_page_to_update = jnp.tile(mapped_page_to_update, kv_heads)
		kv_heads_axis_stride = (
			jnp.repeat(jnp.arange(0, kv_heads), num_tokens) * num_pages * page_size
		)
		mapped_page_to_update = kv_heads_axis_stride + mapped_page_to_update

		key = key.reshape(-1, head_dim)
		value = value.reshape(-1, head_dim)

		self.key_pages = self.key_pages.reshape(-1, head_dim)
		self.value_pages = self.value_pages.reshape(-1, head_dim)

		self.key_pages = self.key_pages.at[mapped_page_to_update, :].set(key)
		self.value_pages = self.value_pages.at[mapped_page_to_update, :].set(value)

		self.key_pages = self.key_pages.reshape(
			kv_heads,
			num_pages,
			page_size,
			head_dim,
		)
		self.value_pages = self.value_pages.reshape(
			kv_heads,
			num_pages,
			page_size,
			head_dim,
		)

		return self

	def __repr__(self):
		return f"{self.__class__.__name__}(layer_index={self.layer_index}, kv_shape={self.key_pages.shape})"

	__str__ = __repr__


@auto_pytree
class PagedAttentionCache(BaseCache):
	"""
	Represents the complete Paged Attention KV cache for all layers of a model.

	It holds a list of `PagedAttentionCacheView` objects, one for each layer.
	It inherits from `BaseCache`.

	Attributes:
	    views (tp.List[PagedAttentionCacheView]): A list containing the cache view
	        for each layer in the model.
	"""

	views: tp.List[PagedAttentionCacheView]

	@classmethod
	def init_cache(
		cls,
		mesh: Mesh,
		dtype: jnp.dtype,
		metadata: PagedAttentionCacheMetaData,
		partition_manager: es.PartitionManager,
		quantizer: tp.Optional["EasyQuantizer"] = None,
	):
		"""
		Initializes the entire PagedAttentionCache for all layers.

		Creates a list of `PagedAttentionCacheView` instances, one for each layer
		specified in the `metadata`, by calling `PagedAttentionCacheView.init` for each layer.

		Args:
		    mesh (Mesh): The JAX device mesh.
		    dtype (jnp.dtype): The data type for the cache pages.
		    metadata (PagedAttentionCacheMetaData): Static configuration for the cache.
		    partition_manager (es.PartitionManager): Manages tensor sharding.
		    quantizer (tp.Optional["EasyQuantizer"]): Optional quantizer to apply.

		Returns:
		    PagedAttentionCache: An initialized cache object containing views for all layers.
		"""
		views = [
			PagedAttentionCacheView.init(
				mesh=mesh,
				dtype=dtype,
				metadata=metadata,
				quantizer=quantizer,
				layer_index=i,
				partition_manager=partition_manager,
			)
			for i in range(metadata.num_hidden_layers)
		]
		return cls(views=views)

	def init_empty(self, *args, **kwargs):
		"""Not typically used for PagedAttentionCache; returns None."""
		return None

	def __repr__(self):
		"""Provides a string representation of the entire paged cache."""
		idx = self.views[-1]
		try:
			k_shape = idx.key_pages.shape
			v_shape = idx.value_pages.shape
		except AttributeError:
			k_shape = "Uninitialized"
			v_shape = "Uninitialized"
		return (
			f"{self.__class__.__name__}(\n"
			f"  key_pages={k_shape},\n"
			f"  value_pages={v_shape},\n"
			f"  num_layers={len(self.views)},\n"
			")"
		)

	__str__ = __repr__


@auto_pytree
class PagedAttentionMetadata:
	"""
	Runtime metadata required for performing a Paged Attention computation step.

	This object holds the necessary information for a single forward pass of the
	paged attention mechanism, distinguishing between prefill and decode steps
	and providing the mappings (page tables) from logical sequence positions to
	physical cache pages.

	Attributes:
	    prefill_length (jax.Array): Scalar JAX array containing the actual length of the
	        prompt being processed in a prefill step. Shape (). Set to 0 if not in prefill.
	    prefill_position (jax.Array): JAX array of positions for the prefill tokens.
	        Shape (padded_prompt_length,). Empty shape () if not in prefill.
	    prefill_page_table (jax.Array): JAX array mapping logical page indices of the
	        prefill sequence to physical page indices in the KV cache. Shape (num_pages_for_prefill,).
	        Empty shape () if not in prefill.
	    decodes_position (jax.Array): JAX array containing the current sequence position
	        (or length - 1) for each sequence in the decode batch. Shape (batch_size,).
	        Empty shape () if not in decode.
	    decodes_page_table (jax.Array): JAX array mapping logical page indices to physical
	        page indices for each sequence in the decode batch.
	        Shape (batch_size, num_pages_per_sequence). Empty shape () if not in decode.
	"""

	prefill_length: jax.Array
	prefill_position: jax.Array
	prefill_page_table: jax.Array
	decodes_position: jax.Array
	decodes_page_table: jax.Array

	def is_prefill_mode(self) -> bool:
		"""
		Checks if the current metadata represents a prefill-only step.

		Returns:
		    bool: True if only prefill information is present (decode arrays have empty shape),
		        False otherwise.
		"""
		return (
			hasattr(self.decodes_position, "shape") and len(self.decodes_position.shape) == 0
		)

	def is_decode_mode(self) -> bool:
		"""
		Creates an initial or placeholder PagedAttentionMetadata object.
		(Internal helper method).

		Returns:
		    PagedAttentionMetadata: An instance with scalar placeholder values.
		"""
		return (
			hasattr(self.prefill_position, "shape") and len(self.prefill_position.shape) == 0
		)

	@classmethod
	def init_empty(cls):
		scalar = jax.device_put(
			jnp.asarray(1e6, dtype=jnp.int32),
			Ns(es.get_incontext_mesh(), Ps()),
		)
		return PagedAttentionMetadata(
			prefill_length=scalar,
			prefill_position=scalar,
			prefill_page_table=scalar,
			decodes_position=scalar,
			decodes_page_table=scalar,
		)
