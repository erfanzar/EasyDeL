from __future__ import annotations

import math
import queue
import typing as tp

import chex as cx
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from eformer.jaximus import ImplicitArray
from eformer import escale as es
from eformer.pytree import auto_pytree
from jax.sharding import Mesh, NamedSharding, PartitionSpec

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
	"""Metadata for Paged Attention KV cache configuration."""

	partition_axis: es.PartitionAxis
	dtype: jnp.dtype
	batch_size: int
	num_hidden_layers: int
	num_pages_per_layer: int
	num_pages_per_sequence: int
	max_sequences: int
	page_size: int
	num_kv_heads: int
	kv_head_dim_size: int
	hbm_utilization: float
	hbm_bytes: float

	@staticmethod
	def _usable_hbm(hbm_utilization: float, mesh: Mesh) -> int:
		per_device_memory_stats = jax.devices()[0].memory_stats()
		limit = per_device_memory_stats["bytes_reservable_limit"]
		used = per_device_memory_stats["bytes_in_use"]
		return (int(limit * hbm_utilization) - used) * mesh.devices.size

	@classmethod
	def create(
		cls,
		mesh: Mesh,
		partition_axis: es.PartitionAxis,
		batch_size: int,
		num_hidden_layers: int,
		max_sequences: int,
		page_size: int,
		num_kv_heads: int,
		kv_head_dim_size: int,
		hbm_utilization: float,
		dtype: jnp.dtype = jnp.bfloat16,
	) -> "PagedAttentionCacheMetaData":
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
			partition_axis=partition_axis,
			dtype=dtype,
			batch_size=batch_size,
			num_hidden_layers=num_hidden_layers,
			num_pages_per_layer=num_pages_per_layer,
			num_pages_per_sequence=num_pages_per_sequence,
			max_sequences=max_sequences,
			page_size=page_size,
			num_kv_heads=num_kv_heads,
			kv_head_dim_size=kv_head_dim_size,
			hbm_utilization=hbm_utilization,
			hbm_bytes=hbm_bytes,
		)


@auto_pytree
class PagedAttentionCacheView(BaseCacheView):
	"""Minimal view for a layer within the PagedAttentionCache."""

	metadata: PagedAttentionCacheMetaData
	layer_index: int

	key_pages: tp.Union[cx.Array, ImplicitArray]
	value_pages: tp.Union[cx.Array, ImplicitArray]
	kv_pages_sharding: NamedSharding

	@classmethod
	def init(
		cls,
		mesh: Mesh,
		metadata: PagedAttentionCacheMetaData,
		layer_index: int,
		quantizer: tp.Optional["EasyQuantizer"] = None,
		kv_pages_sharding: tp.Optional[PartitionSpec] = None,
	):
		from easydel.infra.etils import EasyDeLQuantizationMethods
		from easydel.layers.quantization.quantizers import EasyQuantizer

		quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)
		default_ps = PartitionSpec(metadata.partition_axis.head_axis, None, None, None)
		kv_pages_sharding = kv_pages_sharding or default_ps
		dtype = metadata.dtype
		kv_pages_sharding = NamedSharding(mesh=mesh, spec=kv_pages_sharding)

		kv_pages_shape = (
			metadata.num_kv_heads,
			metadata.num_pages_per_layer,
			metadata.page_size,
			metadata.kv_head_dim_size,
		)

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
				kv_pages_sharding=kv_pages_sharding,
			)

	def concatenate_to_cache(self, *args, **kwargs):
		raise NotImplementedError()

	def write_prefill_to_cache(
		self,
		key: cx.Array,
		value: cx.Array,
		metadata: PagedAttentionMetadata,
	):
		padded_prefill_len = key.shape[0]
		num_kv_heads_per_device = self.key_pages.shape[0]
		page_size = self.key_pages.shape[2]
		num_pages = padded_prefill_len // page_size
		num_pages = jnp.where(num_pages < 1, 1, num_pages)
		num_active_pages, reminder = jnp.divmod(metadata.prefill_length, page_size)
		num_active_pages += jnp.where(reminder > 0, 1, 0)
		head_dim = self.key_pages.shape[-1]
		key = (
			key.transpose((1, 0, 2))
			.reshape((num_kv_heads_per_device, -1, page_size, head_dim))
			.astype(self.key_pages.dtype)
		)
		value = (
			value.transpose((1, 0, 2))
			.reshape((num_kv_heads_per_device, -1, page_size, head_dim))
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

	def write_generate_to_cache(
		self,
		key: cx.Array,
		value: cx.Array,
		metadata: PagedAttentionMetadata,
	):
		key = key.transpose((1, 0, 2))
		value = value.transpose((1, 0, 2))

		key = key.astype(self.key_pages.dtype)
		value = value.astype(self.value_pages.dtype)

		num_tokens = key.shape[1]
		num_kv_heads_per_device, num_pages, page_size, head_dim = self.key_pages.shape
		page_idx, offset = jnp.divmod(metadata.generate_pos, page_size)
		page_to_update = metadata.generate_page_table[jnp.arange(0, num_tokens), page_idx]

		mapped_page_to_update = page_to_update * page_size + offset
		mapped_page_to_update = jnp.tile(mapped_page_to_update, num_kv_heads_per_device)

		kv_heads_axis_stride = (
			jnp.repeat(jnp.arange(0, num_kv_heads_per_device), num_tokens)
			* num_pages
			* page_size
		)
		mapped_page_to_update = kv_heads_axis_stride + mapped_page_to_update

		key = key.reshape(-1, head_dim)
		value = value.reshape(-1, head_dim)

		self.key_pages = self.key_pages.reshape(-1, head_dim)
		self.value_pages = self.value_pages.reshape(-1, head_dim)

		self.key_pages = self.key_pages.at[mapped_page_to_update, :].set(key)
		self.value_pages = self.value_pages.at[mapped_page_to_update, :].set(value)

		self.key_pages = self.key_pages.reshape(
			num_kv_heads_per_device,
			num_pages,
			page_size,
			head_dim,
		)
		self.value_pages = self.value_pages.reshape(
			num_kv_heads_per_device,
			num_pages,
			page_size,
			head_dim,
		)

	def __repr__(self):
		return f"{self.__class__.__name__}(layer_index={self.layer_index}, kv_shape={self.key_pages.shape})"

	__str__ = __repr__


@auto_pytree
class PagedAttentionCache(BaseCache):
	views: tp.List[PagedAttentionCacheView]

	@classmethod
	def init_cache(
		cls,
		mesh: Mesh,
		metadata: PagedAttentionCacheMetaData,
		quantizer: tp.Optional["EasyQuantizer"] = None,
		kv_pages_sharding: tp.Optional[PartitionSpec] = None,
	):
		views = [
			PagedAttentionCacheView.init(
				mesh=mesh,
				metadata=metadata,
				layer_index=i,
				quantizer=quantizer,
				kv_pages_sharding=kv_pages_sharding,
			)
			for i in range(metadata.num_hidden_layers)
		]
		return cls(views=views)

	def init_empty(self, *args, **kwargs):
		return None

	def __repr__(self):
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
			f"  kv_pages_sharding={idx.kv_pages_sharding}\n"
			")"
		)

	__str__ = __repr__


@auto_pytree
class PagedAttentionMetadata:
	prefill_length: jax.Array
	prefill_pos: jax.Array
	prefill_page_table: jax.Array
	generate_pos: jax.Array
	generate_page_table: jax.Array

	@classmethod
	def _init_decode_state(cls):
		return PagedAttentionMetadata(
			prefill_length=jnp.asarray(1e7, dtype=jnp.int32),
			prefill_pos=jnp.asarray(1e7, dtype=jnp.int32),
			prefill_page_table=jnp.asarray(1e7, dtype=jnp.int32),
			generate_pos=jnp.asarray(1e7, dtype=jnp.int32),
			generate_page_table=jnp.asarray(1e7, dtype=jnp.int32),
		)


class PagedAttentionCacheManager:
	"""Logical KV Cache Manager"""

	def __init__(self, metadata: PagedAttentionCacheMetaData):
		"""Initializes the PagedAttentionCacheViewManager."""
		self._metadata = metadata
		self._current_page_index = 0
		self._available_hbm_pages = queue.SimpleQueue()
		for p in range(1, metadata.num_pages_per_layer):
			self._available_hbm_pages.put_nowait(p)

	@property
	def page_size(self):
		"""Returns the page size in the number of per-token kv cache items."""
		return self._metadata.page_size

	@property
	def current_page_index(self):
		"""Returns the dummy page index (0)."""
		return self._current_page_index

	def alloc_prefill_hbm_pages(self, prompt_len) -> list[int]:
		"""Allocates HBM pages for prompt prefill."""
		n = math.ceil(prompt_len / self._metadata.page_size)
		return self.alloc_hbm_pages(n)

	def alloc_hbm_pages(self, n: int) -> list[int]:
		"""Allocates `n` HBM pages."""
		if 0 < n <= self._available_hbm_pages.qsize():
			return [self._available_hbm_pages.get(block=True) for _ in range(n)]
		else:
			return []

	def free_hbm_pages(self, pages: list[int]):
		"""Frees the given HBM pages."""
		for p in pages:
			if p != self._current_page_index:
				self._available_hbm_pages.put_nowait(p)
