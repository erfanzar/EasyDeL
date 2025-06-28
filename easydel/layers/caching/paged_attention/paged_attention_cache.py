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
import jax
import jax.experimental
import jax.numpy as jnp
from eformer import common_types
from eformer import escale as es
from eformer.escale import PartitionManager
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object


def _store_kvcache(
    key: jax.Array,
    value: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    slot_mapping: jax.Array,
):
    """
    Stores key-value pairs in cache with block structure, supporting padding in slot mapping.

    Args:
        key: [N, num_kv_heads, kv_head_dim_size] tensor of keys
        value: [N, num_kv_heads, kv_head_dim_size] tensor of values
        k_cache: [num_kvcache_blocks, kvcache_block_size, num_kv_heads, kv_head_dim_size] key cache
        v_cache: [num_kvcache_blocks, kvcache_block_size, num_kv_heads, kv_head_dim_size] value cache
        slot_mapping: [N] tensor of flat slot indices (-1 indicates padding)

    Returns:
        Updated (k_cache, v_cache)
    """
    N, num_kv_heads, kv_head_dim_size = key.shape
    num_kvcache_blocks, kvcache_block_size, cache_num_heads, cache_head_dim = k_cache.shape
    assert value.shape == (N, num_kv_heads, kv_head_dim_size)
    assert v_cache.shape == (num_kvcache_blocks, kvcache_block_size, num_kv_heads, kv_head_dim_size)
    assert cache_num_heads == num_kv_heads
    assert cache_head_dim == kv_head_dim_size
    assert slot_mapping.shape == (N,)
    total_slots = num_kvcache_blocks * kvcache_block_size
    block_indices = slot_mapping // kvcache_block_size
    block_offsets = slot_mapping % kvcache_block_size

    valid_mask = (slot_mapping >= 0) & (slot_mapping < total_slots)

    safe_block_indices = jnp.where(valid_mask, block_indices, 0)
    safe_block_offsets = jnp.where(valid_mask, block_offsets, 0)
    update_mask = valid_mask.astype(jnp.int32)[:, None, None]

    k_cache = k_cache.at[safe_block_indices, safe_block_offsets].add(
        (key * update_mask) - k_cache[safe_block_indices, safe_block_offsets] * update_mask
    )

    v_cache = v_cache.at[safe_block_indices, safe_block_offsets].add(
        (value * update_mask) - v_cache[safe_block_indices, safe_block_offsets] * update_mask
    )

    return k_cache, v_cache


@auto_pytree
class PagedAttentionCacheMetaData(BaseCacheMetadata):
    """
    Metadata holding configuration parameters for the Paged Attention KV cache.

    This class stores static configuration details required to initialize and manage
    a paged KV cache, such as dimensions, page sizes, and resource utilization hints.
    It inherits from `BaseCacheMetadata`.
    """

    num_hidden_layers: int
    num_kv_heads: int
    kv_head_dim_size: int
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    hbm_utilization: float = 0.9
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    @staticmethod
    def _compute_free_hbm(
        mesh: Mesh,
        partition_manager: PartitionManager,
        hbm_utilization: float,
    ):
        partition_axis = partition_manager.paxis
        size = mesh.shape[partition_axis.kv_head_axis]
        try:
            per_device_memory_stats = jax.local_devices()[0].memory_stats()
            limit = per_device_memory_stats.get("bytes_limit", per_device_memory_stats.get("bytes_reservable_limit"))
            used = per_device_memory_stats["bytes_in_use"]
            usable = int(limit * hbm_utilization) - used
            return usable * size
        except Exception as e:
            print(e)
            return 22 * (1024**3)

    @classmethod
    def create(
        cls,
        mesh: Mesh,
        partition_manager: PartitionManager,
        kvdtype: jnp.dtype,
        num_hidden_layers: int,
        num_kv_heads: int,
        kv_head_dim_size: int,
        max_num_batched_tokens: int = 32768,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        hbm_utilization: float = 0.9,
        kvcache_block_size: int = 256,
    ) -> PagedAttentionCacheMetaData:
        if num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive")
        if num_kv_heads <= 0:
            raise ValueError("`num_kv_heads` must be positive")
        if kv_head_dim_size <= 0:
            raise ValueError("`kv_head_dim_size` must be positive")
        free = cls._compute_free_hbm(mesh=mesh, partition_manager=partition_manager, hbm_utilization=hbm_utilization)
        block_bytes = (
            2 * num_hidden_layers * kvcache_block_size * num_kv_heads * kv_head_dim_size * (jnp.finfo(kvdtype).bits // 8)
        )
        num_kvcache_blocks = int(free) // block_bytes
        return cls(
            num_hidden_layers=num_hidden_layers,
            num_kv_heads=num_kv_heads,
            kv_head_dim_size=kv_head_dim_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            hbm_utilization=hbm_utilization,
            kvcache_block_size=kvcache_block_size,
            num_kvcache_blocks=num_kvcache_blocks,
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

    key_pages: cx.Array | ImplicitArray
    value_pages: cx.Array | ImplicitArray

    @classmethod
    def init(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: PagedAttentionCacheMetaData,
        layer_index: int,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
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
            metadata.num_kvcache_blocks,
            metadata.kvcache_block_size,
            metadata.num_kv_heads,
            metadata.kv_head_dim_size,
        )

        kv_pages_sharding = partition_manager.resolve(
            [
                common_types.EMPTY,
                common_types.EMPTY,
                common_types.HEAD,
                common_types.EMPTY,
            ],
            mode=common_types.MODE_PREFILL,
            shape=kv_pages_shape,
        )

        kv_pages_sharding = Ns(mesh=mesh, spec=kv_pages_sharding)

        with jax.named_scope("easydel-paged-attention-cache-init"):
            key_pages = jnp.zeros(shape=kv_pages_shape, dtype=dtype, device=kv_pages_sharding)
            value_pages = jnp.zeros(shape=kv_pages_shape, dtype=dtype, device=kv_pages_sharding)
            key_pages = quantizer(key_pages)
            value_pages = quantizer(value_pages)
            return cls(metadata=metadata, layer_index=layer_index, key_pages=key_pages, value_pages=value_pages)

    def concatenate_to_cache(
        self,
        key: cx.Array,
        value: cx.Array,
        cache_metadata: PagedAttentionMetadata,
    ):
        """
        Concatenation is not applicable for Paged Attention.
        """
        self.key_pages, self.value_pages = _store_kvcache(
            key,
            value,
            self.key_pages,
            self.value_pages,
            cache_metadata.slot_mapping,
        )
        return self.key_pages, self.value_pages

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

    views: list[PagedAttentionCacheView]

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: PagedAttentionCacheMetaData,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
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
    is_prefill: bool
    slot_mapping: jax.Array
    block_tables: jax.Array | None = None
    context_lens: jax.Array | None = None
    cu_seqlens_q: jax.Array | None = None
    cu_seqlens_k: jax.Array | None = None
    max_seqlen_q: jax.Array | int | None = None
    max_seqlen_k: jax.Array | int | None = None
