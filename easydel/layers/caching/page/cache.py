# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from eformer.escale import PartitionAxis, PartitionManager
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView
from .utils import kv_cache_update, kv_cache_update_jax

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object

EMPTY = common_types.EMPTY
KV_HEAD = common_types.KV_HEAD
MODE_PREFILL = common_types.MODE_PREFILL


def cdiv(a, b):
    return (a + b - 1) // b


def previous_power_of_2(n: int) -> int:
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def get_num_slices_per_kv_cache_update_page(page_size_bytes: int) -> int:
    num_slices_per_page = (16 * 1024 * 1024) // page_size_bytes
    assert num_slices_per_page > 0, "Number of slices should be positive"
    num_slices_per_page = previous_power_of_2(num_slices_per_page)
    if num_slices_per_page > 64:
        num_slices_per_page = 64
    return num_slices_per_page


def get_dtype_packing(dtype):
    bits = jnp.finfo(dtype).bits
    if 32 % bits != 0:
        raise ValueError(f"The bit width must be divisible by 32, but got bits={bits}, dtype={{dtype}}")
    return 32 // bits


def get_page_size_bytes(
    page_size: int,
    num_kv_heads: int,
    head_size: int,
    kv_cache_dtype: jnp.dtype,
) -> int:
    """Returns the size in bytes of one page of the KV cache."""
    padded_head_size = cdiv(head_size, 128) * 128
    num_combined_kv_heads = num_kv_heads * 2
    packing = get_dtype_packing(kv_cache_dtype)
    num_combined_kv_heads = cdiv(num_combined_kv_heads, packing) * packing
    kv_cache_dtype_bits = jnp.finfo(kv_cache_dtype).bits
    return page_size * num_combined_kv_heads * padded_head_size * kv_cache_dtype_bits // 8


@auto_pytree
class PagesCacheMetaData(BaseCacheMetadata):
    """
    Metadata holding configuration parameters for the Paged Attention KV cache.

    This class stores static configuration details required to initialize and manage
    a paged KV cache, such as dimensions, page sizes, and resource utilization hints.
    It inherits from `BaseCacheMetadata`.
    """

    num_hidden_layers: int
    max_model_length: int
    num_kv_heads: int
    k_headdim: int
    v_headdim: int
    hbm_utilization: float = 0.9
    page_size: int = 128
    num_pages: int = -1
    max_num_pages_per_req: int = -1
    num_slices_per_kv_cache_update_page: int = -1
    _kvdtype_str: str = field(pytree_node=False, default="bf16")

    @staticmethod
    def _compute_free_hbm(mesh: Mesh, partition_manager: PartitionManager, hbm_utilization: float):
        partition_axis = partition_manager.paxis
        size = mesh.shape[partition_axis.kv_head_axis]
        try:
            per_device_memory_stats = jax.local_devices()[0].memory_stats()
            limit = per_device_memory_stats.get("bytes_limit", per_device_memory_stats.get("bytes_reservable_limit"))
            used = per_device_memory_stats["bytes_in_use"]
            usable = int(limit * hbm_utilization) - used
            return abs(usable * size)
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
        max_model_length: int,
        kv_head_dim_size: int | None = None,
        k_headdim: int | None = None,
        v_headdim: int | None = None,
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
        page_bytes = 2 * num_hidden_layers * page_size * num_kv_heads * kv_head_dim_size * bytes_av
        num_pages = int(free) // page_bytes
        from eformer.mpric import DTYPE_TO_STRING_MAP

        return cls(
            num_hidden_layers=num_hidden_layers,
            max_model_length=max_model_length,
            num_kv_heads=num_kv_heads,
            k_headdim=k_headdim,
            v_headdim=v_headdim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            num_pages=num_pages,
            max_num_pages_per_req=cdiv(max_model_length, page_size),
            num_slices_per_kv_cache_update_page=get_num_slices_per_kv_cache_update_page(
                get_page_size_bytes(
                    page_size=page_size,
                    num_kv_heads=num_kv_heads,
                    head_size=k_headdim,
                    kv_cache_dtype=kvdtype,
                )
            ),
            _kvdtype_str=DTYPE_TO_STRING_MAP[kvdtype],
        )

    @property
    def kvdtype(self) -> jnp.dtype:
        from eformer.mpric import STRING_TO_DTYPE_MAP

        return STRING_TO_DTYPE_MAP[self._kvdtype_str]

    def get_padded_num_slices(self, num_tokens: int, max_num_reqs: int):
        padded_num_slices = 2 * max_num_reqs + num_tokens // self.page_size
        padded_num_slices = min(padded_num_slices, num_tokens)
        padded_num_slices = (
            (padded_num_slices + self.num_slices_per_kv_cache_update_page - 1)
            // self.num_slices_per_kv_cache_update_page
            * self.num_slices_per_kv_cache_update_page
        )
        return padded_num_slices

    def get_max_num_seqs(self) -> int:
        num_page_per_req = cdiv(self.max_model_length, self.page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4


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
        kv_pages (tp.Union[cx.Array, ImplicitArray]): The tensor holding all key value pages for this layer.
            Shape: (num_kv_heads, num_pages_per_layer, page_size, kv_head_dim_size).
            Can be a JAX array or an ImplicitArray if quantization is used.
    """

    metadata: PagesCacheMetaData
    layer_index: int

    kv_pages: cx.Array | ImplicitArray
    partition_manager: PartitionManager = field(
        pytree_node=False,
        default_factory=lambda: PartitionManager(PartitionAxis()),
    )

    @classmethod
    def init(
        cls,
        mesh: Mesh,
        metadata: PagesCacheMetaData,
        layer_index: int,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
    ):
        """
        Initializes the PagesCacheView for a specific layer.

        Allocates the `kv_pages` tensors with the appropriate
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
        kv_pages_shape = (metadata.num_pages, metadata.page_size, metadata.num_kv_heads * 2, metadata.k_headdim)
        axes = [common_types.EMPTY, common_types.EMPTY, common_types.HEAD, common_types.EMPTY]
        kv_pages_sharding = partition_manager.resolve(axes=axes, mode=common_types.MODE_PREFILL, shape=kv_pages_shape)
        kv_pages_sharding = Ns(mesh=mesh, spec=kv_pages_sharding)
        with jax.named_scope("easydel-paged-attention-cache-init"):
            kv_pages = quantizer(jnp.zeros(shape=kv_pages_shape, dtype=metadata.kvdtype, device=kv_pages_sharding))

        return cls(metadata=metadata, layer_index=layer_index, kv_pages=kv_pages, partition_manager=partition_manager)

    def concatenate_to_cache(self, key: cx.Array, value: cx.Array, cache_metadata: PagesMetadata):
        num_kv_heads = key.shape[2]
        head_size = key.shape[3]
        key = key.reshape(-1, num_kv_heads, head_size).astype(self.kv_pages)
        value = value.reshape(-1, num_kv_heads, head_size).astype(self.kv_pages)
        use_kernel = jax.default_backend() == "tpu"
        use_shardmap = use_kernel

        def _update_fn(kv, slots, pages, num_update_slices):
            orgshape = pages.shape
            pages = pages.reshape(-1, *orgshape[2:])
            if use_kernel:
                pages = kv_cache_update(
                    kv,
                    slots,
                    pages,
                    num_update_slices,
                    page_size=cache_metadata.page_size,
                    slices_per_processing_page=cache_metadata.num_slices_per_kv_cache_update_page,
                )
            else:
                pages = kv_cache_update_jax(
                    kv,
                    slots,
                    pages,
                    num_update_slices,
                    page_size=cache_metadata.page_size,
                )
            return pages.reshape(*orgshape)

        if use_shardmap:
            resolve = self.partition_manager.resolve
            _update_fn = jax.shard_map(
                _update_fn,
                in_specs=(
                    resolve([EMPTY, KV_HEAD, EMPTY], mode=MODE_PREFILL),
                    resolve([EMPTY, EMPTY], mode=MODE_PREFILL),
                    resolve([EMPTY, EMPTY, KV_HEAD, EMPTY], mode=MODE_PREFILL),
                    resolve([EMPTY], mode=MODE_PREFILL),
                ),
                out_specs=resolve([EMPTY, EMPTY, KV_HEAD, EMPTY], mode=MODE_PREFILL),
                mesh=es.get_incontext_mesh(),
                check_vma=False,
            )

        kvs = jnp.stack([key, value], axis=2).reshape(-1, num_kv_heads * 2, head_size)
        kv_pages = _update_fn(kvs, cache_metadata.slot_mapping, self.kv_pages, cache_metadata.num_kv_update_slices)
        return self.replace(kv_pages=kv_pages)

    @property
    def key_pages(self) -> jax.Array:
        return self.kv_pages[:, :, 0::2, :]

    @property
    def value_pages(self) -> jax.Array:
        return self.kv_pages[:, :, 1::2, :]

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

    @property
    def metadata(self) -> PagesCacheMetaData | None:
        if self.views[-1] is None:
            return None
        return self.views[-1].metadata

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
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
            kv_shape = idx.kv_pages.shape
        except AttributeError:
            kv_shape = "Uninitialized"
        return f"{self.__class__.__name__}(\n  kv_pages={kv_shape},\n  num_layers={len(self.views)},\n)"

    __str__ = __repr__


@auto_pytree(max_print_length=3000)
class PagesMetadata:
    pages_tables: jax.Array
    context_lens: jax.Array
    query_start_loc: jax.Array
    num_seqs: jax.Array
    slot_mapping: jax.Array
    position_ids: jax.Array | None = None
    num_kv_update_slices: jax.Array | None = None
    num_slices_per_kv_cache_update_page: int | None = field(pytree_node=False, default_factory=lambda: None)
    page_size: int = field(pytree_node=False, default=128)
    prefill_chunk_size: int = field(pytree_node=False, default=512)

    @classmethod
    def create_empty(cls, num_tokens: int, max_num_reqs: int, max_pages: int, page_size: int = 128) -> PagesMetadata:
        """Create empty metadata with proper shapes."""
        return cls(
            slot_mapping=jnp.zeros([num_tokens], dtype=jnp.int32),
            pages_tables=jnp.zeros((max_num_reqs, max_pages), dtype=jnp.int32),
            context_lens=jnp.zeros([max_num_reqs], dtype=jnp.int32),
            query_start_loc=jnp.zeros([max_num_reqs + 1], dtype=jnp.int32),
            position_ids=jnp.zeros([max_num_reqs], dtype=jnp.int32),
            num_seqs=jnp.zeros([max_num_reqs], dtype=jnp.int32),
            page_size=page_size,
        )
