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
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns

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
    """

    batch_size: int
    tokens_per_page: int

    max_prefill_length: int
    max_decodes_length: int
    max_pages_per_group: int

    num_hidden_layers: int
    num_pages: int
    num_kv_heads: int

    kv_head_dim_size: int

    @classmethod
    def create(
        cls,
        batch_size: int,
        tokens_per_page: int,
        max_prefill_length: int,
        max_decodes_length: int,
        num_hidden_layers: int,
        num_pages: int,
        num_kv_heads: int,
        kv_head_dim_size: int,
    ) -> PagedAttentionCacheMetaData:
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive")
        if tokens_per_page <= 0:
            raise ValueError("`tokens_per_page` must be positive")
        if max_prefill_length <= 0:
            raise ValueError("`max_prefill_length` must be positive")
        if max_decodes_length <= 0:
            raise ValueError("`max_decodes_length` must be positive")
        if num_pages <= 0:
            raise ValueError("`num_pages` must be positive")
        if num_kv_heads <= 0:
            raise ValueError("`num_kv_heads` must be positive")
        if kv_head_dim_size <= 0:
            raise ValueError("`kv_head_dim_size` must be positive")

        return cls(
            batch_size=batch_size,
            tokens_per_page=tokens_per_page,
            max_prefill_length=max_prefill_length,
            max_decodes_length=max_decodes_length,
            max_pages_per_group=(max_decodes_length + tokens_per_page - 1) // tokens_per_page,
            num_hidden_layers=num_hidden_layers,
            num_pages=num_pages,
            num_kv_heads=num_kv_heads,
            kv_head_dim_size=kv_head_dim_size,
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
            metadata.num_kv_heads,
            metadata.num_pages,
            metadata.tokens_per_page,
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
        batch_size, seq_len, n_kv_head, head_dim = key.shape
        tokens_per_page = self.key_pages.shape[2]

        if batch_size == 1:
            key = jnp.squeeze(key)
            value = jnp.squeeze(value)
        else:
            key = key[0]
            value = value[0]

        key = jnp.transpose(key, axes=(1, 0, 2))
        value = jnp.transpose(value, axes=(1, 0, 2))

        shape = (n_kv_head, max(1, seq_len // tokens_per_page), tokens_per_page, head_dim)

        self.key_pages = jnp.reshape(key, shape=shape)
        self.value_pages = jnp.reshape(value, shape=shape)
        return self

    def write_decodes_to_cache(
        self,
        key: cx.Array,
        value: cx.Array,
        metadata: PagedAttentionMetadata,
    ):
        batch_size, _, kv_heads, head_dim = key.shape
        kv_heads, _, _, head_dim = self.key_pages.shape

        new_key = key.reshape(batch_size, kv_heads, head_dim)[:, :, :]
        new_key = jnp.transpose(new_key, (1, 0, 2))
        new_value = value.reshape(batch_size, kv_heads, head_dim)[:, :, :]
        new_value = jnp.transpose(new_value, (1, 0, 2))
        broadcast_pages = jnp.tile(metadata.active_page, (kv_heads, 1))
        broadcast_pos = jnp.tile(metadata.active_page_position, (kv_heads, 1))
        kv_indices = jnp.arange(kv_heads)[:, None]
        kv_indices = jnp.tile(kv_indices, (1, batch_size))
        key_pages_updated = self.key_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_key)
        value_pages_updated = self.value_pages.at[kv_indices, broadcast_pages, broadcast_pos].set(new_value)

        self.key_pages_var = key_pages_updated
        self.value_pages_var = value_pages_updated

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
    page_status: jax.Array
    page_map: jax.Array
    num_pages_used: jax.Array
    sequence_lengths: jax.Array
    active_page: jax.Array
    has_active_page: jax.Array
    active_page_position: jax.Array

    @classmethod
    def create(
        cls,
        num_pages: int,
        max_page_groups: int,
        max_pages_per_group: int,
    ) -> PagedAttentionMetadata:
        return cls(
            page_map=jnp.zeros((max_page_groups, max_pages_per_group), dtype="i4"),
            active_page=jnp.zeros((max_page_groups,), dtype="i4"),
            page_status=jnp.zeros((num_pages,), dtype="i4").at[0].set(1),
            num_pages_used=jnp.zeros((max_page_groups,), dtype="i4"),
            has_active_page=jnp.zeros((max_page_groups,), dtype="b1"),
            sequence_lengths=jnp.zeros((max_page_groups,), dtype="i4"),
            active_page_position=jnp.zeros((max_page_groups,), dtype="i4"),
        )
