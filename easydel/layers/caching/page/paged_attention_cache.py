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


@auto_pytree
class PagesCacheMetaData(BaseCacheMetadata):
    """
    Metadata holding configuration parameters for the Paged Attention KV cache.

    This class stores static configuration details required to initialize and manage
    a paged KV cache, such as dimensions, page sizes, and resource utilization hints.
    It inherits from `BaseCacheMetadata`.
    """

    num_hidden_layers: int
    num_kv_heads: int
    k_headdim: int
    v_headdim: int
    max_sequence_length: int
    hbm_utilization: float = 0.9
    page_size: int = 128
    num_pages: int = -1

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
            return 4 * (1024**3)

    @classmethod
    def create(
        cls,
        mesh: Mesh,
        partition_manager: PartitionManager,
        kvdtype: jnp.dtype,
        num_hidden_layers: int,
        num_kv_heads: int,
        kv_head_dim_size: int | None = None,
        k_headdim: int | None = None,
        v_headdim: int | None = None,
        max_sequence_length: int = 4096,
        hbm_utilization: float = 0.9,
        page_size: int = 128,
    ) -> PagesCacheMetaData:
        if k_headdim is None:
            assert kv_head_dim_size is not None, "Either `k_headdim` or `kv_head_dim_size` must be provided"
            k_headdim = kv_head_dim_size
        if v_headdim is None:
            assert kv_head_dim_size is not None, "Either `v_headdim` or `kv_head_dim_size` must be provided"
            v_headdim = kv_head_dim_size
        if num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive")
        if num_kv_heads <= 0:
            raise ValueError("`num_kv_heads` must be positive")
        if kv_head_dim_size <= 0:
            raise ValueError("`kv_head_dim_size` must be positive")
        free = cls._compute_free_hbm(mesh=mesh, partition_manager=partition_manager, hbm_utilization=hbm_utilization)
        bytes_av = jnp.finfo(kvdtype).bits // 8
        block_bytes = 2 * num_hidden_layers * page_size * num_kv_heads * kv_head_dim_size * bytes_av
        num_pages = int(free) // block_bytes
        return cls(
            num_hidden_layers=num_hidden_layers,
            num_kv_heads=num_kv_heads,
            k_headdim=k_headdim,
            v_headdim=v_headdim,
            max_sequence_length=max_sequence_length,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            num_pages=num_pages,
        )


@auto_pytree
class PagesCacheView(BaseCacheView):
    """
    Represents the view of the Paged Attention KV cache for a single transformer layer.

    It holds references to the physical key and value pages allocated for this layer
    and the associated metadata. It provides methods to write new key/value pairs
    into the correct pages based on runtime metadata. It inherits from `BaseCacheView`.

    Attributes:
        metadata (PagesCacheMetaData): The static configuration metadata for the
            entire paged cache.
        layer_index (int): The index of the transformer layer this view corresponds to.
        key_pages (tp.Union[cx.Array, ImplicitArray]): The tensor holding all key pages for this layer.
            Shape: (num_kv_heads, num_pages_per_layer, page_size, kv_head_dim_size).
            Can be a JAX array or an ImplicitArray if quantization is used.
        value_pages (tp.Union[cx.Array, ImplicitArray]): The tensor holding all value pages for this layer.
            Shape: (num_kv_heads, num_pages_per_layer, page_size, kv_head_dim_size).
            Can be a JAX array or an ImplicitArray if quantization is used.
    """

    metadata: PagesCacheMetaData
    layer_index: int

    key_pages: cx.Array | ImplicitArray
    value_pages: cx.Array | ImplicitArray

    @classmethod
    def init(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: PagesCacheMetaData,
        layer_index: int,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
    ):
        """
        Initializes the PagesCacheView for a specific layer.

        Allocates the `key_pages` and `value_pages` tensors with the appropriate
        shape, dtype, and sharding based on the provided metadata and partition manager.
        Optionally applies quantization if a quantizer is provided.

        Args:
            mesh (Mesh): The JAX device mesh.
            dtype (jnp.dtype): The data type for the cache pages (e.g., jnp.bfloat16).
            metadata (PagesCacheMetaData): Static configuration for the cache.
            layer_index (int): The index of the layer this view is for.
            partition_manager (es.PartitionManager): Manages tensor sharding across the mesh.
            quantizer (tp.Optional["EasyQuantizer"]): Optional quantizer to apply to the pages.

        Returns:
            PagesCacheView: An initialized cache view for the specified layer.
        """
        from easydel.infra.etils import EasyDeLQuantizationMethods
        from easydel.layers.quantization.quantizers import EasyQuantizer

        quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)

        k_pages_shape = (metadata.num_kv_heads, metadata.num_pages, metadata.page_size, metadata.k_headdim)
        k_pages_sharding = partition_manager.resolve(
            [common_types.HEAD, common_types.EMPTY, common_types.EMPTY, common_types.EMPTY],
            mode=common_types.MODE_PREFILL,
            shape=k_pages_shape,
        )
        v_pages_shape = (metadata.num_kv_heads, metadata.num_pages, metadata.page_size, metadata.v_headdim)
        v_pages_sharding = partition_manager.resolve(
            [common_types.HEAD, common_types.EMPTY, common_types.EMPTY, common_types.EMPTY],
            mode=common_types.MODE_PREFILL,
            shape=v_pages_shape,
        )

        k_pages_sharding = Ns(mesh=mesh, spec=k_pages_sharding)
        v_pages_sharding = Ns(mesh=mesh, spec=v_pages_sharding)

        with jax.named_scope("easydel-paged-attention-cache-init"):
            key_pages = quantizer(jnp.zeros(shape=k_pages_shape, dtype=dtype, device=k_pages_sharding))
            value_pages = quantizer(jnp.zeros(shape=v_pages_shape, dtype=dtype, device=v_pages_sharding))
            return cls(metadata=metadata, layer_index=layer_index, key_pages=key_pages, value_pages=value_pages)

    def concatenate_to_cache(self, key: cx.Array, value: cx.Array, cache_metadata: PagesMetadata):
        """
        Concatenation is not applicable for Paged Attention.
        """
        dist = cache_metadata.destination_pages
        is_valid_token = dist >= 0
        dest_pages = jnp.where(is_valid_token, dist // self.metadata.page_size, self.metadata.num_pages)
        dest_slots = jnp.where(is_valid_token, dist % self.metadata.page_size, 0)
        self.key_pages = self.key_pages.at[:, dest_pages, dest_slots, :].set(jnp.swapaxes(key, 0, 1))
        self.value_pages = self.value_pages.at[:, dest_pages, dest_slots, :].set(jnp.swapaxes(value, 0, 1))
        return self.key_pages, self.value_pages

    def interleave_by_reshaping(self):
        """
        Recombines cache by stacking and reshaping.
        """
        stacked = jnp.stack([self.key_pages, self.value_pages], axis=3)
        b, h, s_half, _, d = stacked.shape
        final_shape = (b, h, s_half * 2, d)
        return stacked.reshape(final_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(layer_index={self.layer_index}, kv_shape={self.key_pages.shape})"

    __str__ = __repr__


@auto_pytree
class PagesCache(BaseCache):
    """
    Represents the complete Paged Attention KV cache for all layers of a model.

    It holds a list of `PagesCacheView` objects, one for each layer.
    It inherits from `BaseCache`.

    Attributes:
        views (tp.List[PagesCacheView]): A list containing the cache view
            for each layer in the model.
    """

    views: list[PagesCacheView]

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: PagesCacheMetaData,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
    ):
        """
        Initializes the entire PagesCache for all layers.

        Creates a list of `PagesCacheView` instances, one for each layer
        specified in the `metadata`, by calling `PagesCacheView.init` for each layer.

        Args:
            mesh (Mesh): The JAX device mesh.
            dtype (jnp.dtype): The data type for the cache pages.
            metadata (PagesCacheMetaData): Static configuration for the cache.
            partition_manager (es.PartitionManager): Manages tensor sharding.
            quantizer (tp.Optional["EasyQuantizer"]): Optional quantizer to apply.

        Returns:
            PagesCache: An initialized cache object containing views for all layers.
        """
        views = [
            PagesCacheView.init(
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
        """Not typically used for PagesCache; returns None."""
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
class PagesMetadata:
    is_prefill: bool | None
    page_indices: jax.Array  # [sequence, page] i32
    destination_pages: jax.Array  # [max_new_tokens] i32
    sequence_lengths: jax.Array  # [sequence] i32
    cumulative_sequence_lengths: jax.Array  # [sequence + 1] i32
    num_sequence: jax.Array  # scalar i32
    page_size: int = 128
