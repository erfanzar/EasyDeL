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

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer import escale as es
from eformer.escale import PartitionAxis, PartitionManager
from eformer.loggings import get_logger
from eformer.mpric import DTYPE_TO_STRING_MAP
from eformer.pytree import auto_pytree, field
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array, Float

from easydel.layers.caching.ragged_page.utils import kv_cache_update_jax

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, unwrap_metadata

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object

logger = get_logger(__name__)

EMPTY = common_types.EMPTY
KV_HEAD = common_types.KV_HEAD
MODE_PREFILL = common_types.MODE_PREFILL


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def per_device_hbm_budget_bytes(util: float = 0.9, mode: str = "free", safety_margin: int = 256 << 20) -> int:
    budgets: list[int] = []
    for device in jax.local_devices():
        try:
            stats = device.memory_stats()
        except Exception:
            continue
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit") or stats.get("bytes_total")
        used_in_use = stats.get("bytes_in_use", 0)
        used_reserved = stats.get("bytes_reserved", 0)
        used = max(used_in_use, used_reserved)
        if limit is None:
            continue

        free = max(0, int(limit) - int(used))
        if mode == "free":
            usable = max(0, int(free * float(util)) - safety_margin)
        else:
            usable = max(0, int(int(limit) * float(util)) - int(used) - safety_margin)
        budgets.append(usable)

    return min(budgets) if budgets else 4 * (1024**3)


def _previous_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n.bit_length() - 1)


@auto_pytree
class UnifiedAttentionCacheConfig(BaseCacheConfig):
    """Paged KV-cache config for vLLM-style unified attention.

    Storage layout:
        - key_cache/value_cache per layer: [num_blocks, block_size, num_kv_heads, head_dim]

    This matches ejkernel's Triton UnifiedAttention kernel input contract.
    """

    num_hidden_layers: int = field(pytree_node=False)
    max_model_length: int = field(pytree_node=False)
    num_kv_heads: int = field(pytree_node=False)
    head_dim: int = field(pytree_node=False)

    hbm_utilization: float = field(pytree_node=False, default=0.9)
    page_size: int = field(pytree_node=False, default=128)
    num_pages: int = field(pytree_node=False, default=-1)
    max_num_pages_per_req: int = field(pytree_node=False, default=-1)

    # Used by eSurge's slot-mapping padding logic (v2-style cache updates).
    num_slices_per_kv_cache_update_page: int = field(pytree_node=False, default=-1)

    # Exposed to keep eSurge metadata builder compatible.
    version: tp.Literal["v2"] = field(pytree_node=False, default="v2")

    _kvdtype_str: str = field(pytree_node=False, default="bf16")

    @staticmethod
    def _compute_free_hbm(mesh: Mesh, partition_manager: PartitionManager, hbm_utilization: float) -> int:
        kv_head_axis = partition_manager.paxis.kv_head_axis
        size = int(mesh.shape[kv_head_axis])
        budget = per_device_hbm_budget_bytes(hbm_utilization, mode="free")
        available_alloc = budget * size
        logger.info(f"{kv_head_axis=} {size=} {budget=} {available_alloc=} {hbm_utilization=}")
        return available_alloc

    @classmethod
    def create(
        cls,
        mesh: Mesh,
        partition_manager: PartitionManager,
        kvdtype: jnp.dtype,
        num_hidden_layers: int,
        num_kv_heads: int,
        max_model_length: int,
        head_dim: int,
        *,
        hbm_utilization: float = 0.9,
        page_size: int = 128,
    ) -> "UnifiedAttentionCacheConfig":
        if num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive")
        if num_kv_heads <= 0:
            raise ValueError("`num_kv_heads` must be positive")
        if head_dim <= 0:
            raise ValueError("`head_dim` must be positive")
        if page_size <= 0:
            raise ValueError("`page_size` must be positive")
        if max_model_length <= 0:
            raise ValueError("`max_model_length` must be positive")

        free = cls._compute_free_hbm(mesh=mesh, partition_manager=partition_manager, hbm_utilization=hbm_utilization)
        bytes_av = jnp.finfo(kvdtype).bits // 8
        # Two tensors (K+V) per layer.
        page_bytes = 2 * num_hidden_layers * page_size * num_kv_heads * head_dim * bytes_av
        num_pages = int(free) // int(page_bytes)
        logger.info(
            f"Creating UnifiedAttentionCacheConfig with {num_pages=} {page_bytes=} "
            f"sequence_capacity={int((num_pages * page_size) / 1000)}K"
        )

        # A lightweight heuristic for slot-mapping padding; matches eSurge expectations.
        page_size_bytes = 2 * page_size * num_kv_heads * head_dim * bytes_av
        # Keep this conservative; it only affects padding of the update schedule.
        slices_raw = (16 * 1024 * 1024) // max(1, int(page_size_bytes))
        num_slices_per_page = min(64, _previous_power_of_2(int(slices_raw)))

        return cls(
            num_hidden_layers=num_hidden_layers,
            max_model_length=max_model_length,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            num_pages=num_pages,
            max_num_pages_per_req=cdiv(max_model_length, page_size),
            num_slices_per_kv_cache_update_page=int(num_slices_per_page),
            _kvdtype_str=DTYPE_TO_STRING_MAP[kvdtype],
        )

    @property
    def kvdtype(self) -> jnp.dtype:
        from eformer.mpric import STRING_TO_DTYPE_MAP

        return STRING_TO_DTYPE_MAP[self._kvdtype_str]

    def get_padded_num_slices(
        self,
        num_tokens: int | None = None,
        max_num_reqs: int | None = None,
    ) -> int:
        if num_tokens is None or num_tokens <= 0:
            num_tokens = self.max_model_length
        if max_num_reqs is None or max_num_reqs <= 0:
            max_num_reqs = self.get_max_num_seqs()

        padded_num_slices = 2 * int(max_num_reqs) + int(num_tokens) // int(self.page_size)
        padded_num_slices = min(int(padded_num_slices), int(num_tokens))

        slices_per_page = max(1, int(self.num_slices_per_kv_cache_update_page))
        padded_num_slices = ((padded_num_slices + slices_per_page - 1) // slices_per_page) * slices_per_page
        return int(padded_num_slices)

    def get_max_num_seqs(self) -> int:
        # Same heuristic as RaggedPagesCacheConfig.
        num_page_per_req = cdiv(self.max_model_length, self.page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4


@auto_pytree
class UnifiedAttentionCacheView(BaseCacheView):
    """Per-layer KV-cache view for unified attention."""

    metadata: UnifiedAttentionCacheConfig
    layer_index: int = field(pytree_node=False)

    key_cache: Float[Array, "num_pages page_size num_kv_heads head_dim"]
    value_cache: Float[Array, "num_pages page_size num_kv_heads head_dim"]

    partition_manager: PartitionManager = field(
        pytree_node=False,
        default_factory=lambda: PartitionManager(PartitionAxis()),
    )

    @classmethod
    def init(
        cls,
        config: UnifiedAttentionCacheConfig,
        layer_index: int | None = None,
        *,
        mesh: Mesh | None = None,
        partition_manager: es.PartitionManager | None = None,
        quantizer: EasyQuantizer | None = None,
    ) -> "UnifiedAttentionCacheView":
        if partition_manager is None:
            partition_manager = PartitionManager(PartitionAxis())

        key_shape = (config.num_pages, config.page_size, config.num_kv_heads, config.head_dim)
        axes = [EMPTY, EMPTY, KV_HEAD, EMPTY]
        sharding_spec = partition_manager.resolve(axes=axes, mode=MODE_PREFILL, shape=key_shape)
        sharding = Ns(mesh=mesh, spec=sharding_spec)

        with jax.named_scope("easydel-unified-attention-cache-init"):
            key_cache = jnp.zeros(shape=key_shape, dtype=config.kvdtype, device=sharding)
            value_cache = jnp.zeros(shape=key_shape, dtype=config.kvdtype, device=sharding)

        if quantizer is not None and callable(quantizer):
            key_cache = quantizer(key_cache)
            value_cache = quantizer(value_cache)

        return cls(
            metadata=config,
            layer_index=layer_index or 0,
            key_cache=key_cache,
            value_cache=value_cache,
            partition_manager=partition_manager,
        )

    def concatenate_to_cache(
        self,
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        cache_metadata: tp.Any,
    ) -> "UnifiedAttentionCacheView":
        cache_metadata = unwrap_metadata(cache_metadata, "ragged")

        if cache_metadata.slot_mapping is None or cache_metadata.num_kv_update_slices is None:
            raise ValueError("UnifiedAttentionCacheView requires v2-style `slot_mapping` and `num_kv_update_slices`.")

        # Flatten to `[total_tokens, num_kv_heads, head_dim]`.
        key_tokens = key.reshape(-1, *key.shape[-2:]).astype(self.key_cache.dtype)
        value_tokens = value.reshape(-1, *value.shape[-2:]).astype(self.value_cache.dtype)

        pages_k = self.key_cache.reshape(-1, *self.key_cache.shape[-2:])
        pages_v = self.value_cache.reshape(-1, *self.value_cache.shape[-2:])

        pages_k = kv_cache_update_jax(
            key_tokens,
            cache_metadata.slot_mapping,
            pages_k,
            cache_metadata.num_kv_update_slices,
            page_size=int(cache_metadata.page_size),
        )
        pages_v = kv_cache_update_jax(
            value_tokens,
            cache_metadata.slot_mapping,
            pages_v,
            cache_metadata.num_kv_update_slices,
            page_size=int(cache_metadata.page_size),
        )

        new_key_cache = pages_k.reshape(self.key_cache.shape)
        new_value_cache = pages_v.reshape(self.value_cache.shape)
        return self.replace(key_cache=new_key_cache, value_cache=new_value_cache)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(layer_index={self.layer_index}, "
            f"key_cache_shape={getattr(self.key_cache, 'shape', None)}, "
            f"value_cache_shape={getattr(self.value_cache, 'shape', None)})"
        )

    __str__ = __repr__


@auto_pytree
class UnifiedAttentionCache(BaseCache):
    """Cache container holding per-layer unified attention cache views."""

    views: list[UnifiedAttentionCacheView]

    @property
    def metadata(self) -> UnifiedAttentionCacheConfig | None:
        if not self.views or self.views[-1] is None:
            return None
        return self.views[-1].metadata

    @classmethod
    def init_cache(
        cls,
        *,
        mesh: Mesh,
        config: UnifiedAttentionCacheConfig,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
    ) -> "UnifiedAttentionCache":
        views = [
            UnifiedAttentionCacheView.init(
                config=config,
                layer_index=i,
                mesh=mesh,
                partition_manager=partition_manager,
                quantizer=quantizer,
            )
            for i in range(config.num_hidden_layers)
        ]
        return cls(views=views)

    @classmethod
    def init_empty(cls, num_hidden_layers: int, *args, **kwargs) -> "UnifiedAttentionCache":
        return cls(views=[None] * int(num_hidden_layers))
