# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""MLA-specific ragged page cache.

The Multi-Latent Ragged Page Attention (MLA) kernel stores a compressed KV state
per token in the layout expected by ``ejkernel``:
``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer import escale as es
from eformer.escale import PartitionAxis, PartitionManager
from eformer.jaximus import ImplicitArray
from eformer.loggings import get_logger
from eformer.mpric import DTYPE_TO_STRING_MAP
from eformer.pytree import auto_pytree
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array, Float

from easydel.axis import ATTN_DP

from ..ragged_page.cache import (
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
    _mesh_axis_size,
    align_to_multiple,
    cdiv,
    get_dtype_packing,
    get_num_slices_per_kv_cache_update_page,
    per_device_hbm_budget_bytes,
)

if tp.TYPE_CHECKING:
    from easydel.layers.quantization._quants import EasyQuantizer
else:
    EasyQuantizer = object

logger = get_logger(__name__)


@auto_pytree
class MLARaggedPagesCacheConfig(RaggedPagesCacheConfig):
    """Configuration for Multi-Latent Attention (MLA) compressed ragged pages cache.

    Extends :class:`RaggedPagesCacheConfig` for architectures (e.g. DeepSeek-V2/v1)
    that compress key-value projections into a single low-rank latent per token.
    Instead of separate K and V page buffers, each page stores a unified compressed
    KV state with shape
    ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.

    The compressed dimension (``kv_dim_padded``) is composed of two independently
    128-aligned components: ``kv_lora_rank`` (the low-rank projection size) and
    ``qk_rope_head_dim`` (the RoPE-specific head dimension).

    Use :meth:`create` to build a config from model hyperparameters; the factory
    automatically computes page counts from available HBM.
    """

    @staticmethod
    def _compute_free_hbm(
        mesh: Mesh,
        partition_manager: PartitionManager,
        hbm_utilization: float,
    ):
        """Compute free HBM budget available for MLA cache pages.

        Unlike standard MHA caches, MLA pages are not sharded across the
        KV-head axis.  The budget is therefore scaled only by the
        data-parallel axis size.

        Args:
            mesh: JAX device mesh used to look up axis sizes.
            partition_manager: Provides the data-parallel axis name.
            hbm_utilization: Fraction of total HBM to consider available
                (0.0-1.0).

        Returns:
            Total allocatable bytes across all devices on the page axis.
        """
        budget = per_device_hbm_budget_bytes(hbm_utilization, mode="free")
        page_axis_size = _mesh_axis_size(mesh, partition_manager.paxis.data_parallel_axis)
        available_alloc = budget * page_axis_size
        logger.info(f"{page_axis_size=} {budget=} {available_alloc=} {hbm_utilization=}")
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
        *,
        kv_lora_rank: int | None = None,
        qk_rope_head_dim: int | None = None,
        kv_head_dim_size: int | None = None,
        k_headdim: int | None = None,
        v_headdim: int | None = None,
        hbm_utilization: float = 0.9,
        page_size: int = 128,
        version: tp.Literal["v1"] = "v1",
    ) -> "MLARaggedPagesCacheConfig":
        """Create an MLA ragged-page cache config from model hyperparameters.

        Computes the padded KV dimension, page byte footprint, and maximum
        number of pages that fit in the available HBM budget, then returns a
        fully populated :class:`MLARaggedPagesCacheConfig`.

        ``kv_lora_rank`` and ``qk_rope_head_dim`` are the preferred way to
        specify the compressed KV dimensions.  For backward compatibility the
        aliases ``k_headdim`` / ``v_headdim`` and ``kv_head_dim_size`` are
        also accepted (resolved in that priority order).

        Args:
            mesh: JAX device mesh for sharding and HBM budget estimation.
            partition_manager: Resolves sharding specs and provides the
                data-parallel axis name.
            kvdtype: Data type for the KV cache (e.g. ``jnp.bfloat16``).
            num_hidden_layers: Number of transformer layers that require
                a cache view.
            num_kv_heads: Number of KV attention heads in the model.
            max_model_length: Maximum sequence length the cache must support.
            kv_lora_rank: Low-rank KV projection dimension.  Falls back to
                ``k_headdim`` then ``kv_head_dim_size`` if ``None``.
            qk_rope_head_dim: RoPE head dimension appended to the latent.
                Falls back to ``v_headdim``; defaults to 0 if still ``None``.
            kv_head_dim_size: Legacy alias for ``kv_lora_rank``.
            k_headdim: Legacy alias for ``kv_lora_rank``.
            v_headdim: Legacy alias for ``qk_rope_head_dim``.
            hbm_utilization: Fraction of free HBM to allocate (0.0-1.0).
            page_size: Number of tokens per page.
            version: Cache layout version (only ``"v1"`` is supported).

        Returns:
            A fully initialized :class:`MLARaggedPagesCacheConfig`.

        Raises:
            ValueError: If any dimension is non-positive, ``kv_lora_rank``
                cannot be resolved, or the HBM budget is too small.
        """
        if kv_lora_rank is None:
            kv_lora_rank = k_headdim
        if qk_rope_head_dim is None:
            qk_rope_head_dim = v_headdim

        if kv_lora_rank is None and kv_head_dim_size is not None:
            kv_lora_rank = kv_head_dim_size
        if qk_rope_head_dim is None:
            qk_rope_head_dim = 0

        if num_hidden_layers <= 0:
            raise ValueError("`num_hidden_layers` must be positive")
        if num_kv_heads <= 0:
            raise ValueError("`num_kv_heads` must be positive")
        if max_model_length <= 0:
            raise ValueError("`max_model_length` must be positive")
        if kv_lora_rank is None or kv_lora_rank <= 0:
            raise ValueError("`kv_lora_rank` must be positive")
        if qk_rope_head_dim < 0:
            raise ValueError("`qk_rope_head_dim` must be non-negative")
        if page_size <= 0:
            raise ValueError("`page_size` must be positive")
        if version != "v1":
            raise ValueError(f"MLA ragged cache only supports version='v1', got {version!r}")

        data_parallel_size = _mesh_axis_size(mesh, partition_manager.paxis.data_parallel_axis)
        if data_parallel_size > 1:
            logger.info(f"Scaling MLA KV page budget by data-parallel page axis: {data_parallel_size=}.")

        free = cls._compute_free_hbm(
            mesh=mesh,
            partition_manager=partition_manager,
            hbm_utilization=hbm_utilization,
        )
        bytes_av = jnp.finfo(kvdtype).bits // 8
        kv_dim_padded = align_to_multiple(int(kv_lora_rank), 128) + align_to_multiple(int(qk_rope_head_dim), 128)
        kv_packing = get_dtype_packing(kvdtype)
        page_size_per_kv_packing = cdiv(page_size, kv_packing)
        packed_page_tokens = page_size_per_kv_packing * kv_packing

        page_bytes_per_layer = packed_page_tokens * kv_dim_padded * bytes_av
        page_bytes = num_hidden_layers * page_bytes_per_layer
        num_pages = int(free) // int(page_bytes)
        if data_parallel_size > 1:
            num_pages = (num_pages // data_parallel_size) * data_parallel_size
        if num_pages <= 0:
            raise ValueError(
                "Computed `num_pages` is non-positive; increase `hbm_utilization` or reduce page footprint."
            )
        logger.info(
            f"Creating MLARaggedPagesCacheConfig with {num_pages=} {page_bytes=} "
            f"sequence_capacity={int((num_pages * page_size) / 1000)}K"
        )

        return cls(
            num_hidden_layers=num_hidden_layers,
            max_model_length=max_model_length,
            num_kv_heads=num_kv_heads,
            k_headdim=int(kv_lora_rank),
            v_headdim=int(qk_rope_head_dim),
            hbm_utilization=hbm_utilization,
            data_parallel_size=data_parallel_size,
            page_size=page_size,
            num_pages=num_pages,
            max_num_pages_per_req=cdiv(max_model_length, page_size),
            num_slices_per_kv_cache_update_page=get_num_slices_per_kv_cache_update_page(
                packed_page_tokens * kv_dim_padded * bytes_av
            ),
            version="v1",
            _kvdtype_str=DTYPE_TO_STRING_MAP[kvdtype.type if hasattr(kvdtype, "type") else kvdtype],
        )

    @property
    def kv_lora_rank(self) -> int:
        """Low-rank KV projection dimension (stored as ``k_headdim``)."""
        return int(self.k_headdim)

    @property
    def qk_rope_head_dim(self) -> int:
        """RoPE head dimension for queries/keys (stored as ``v_headdim``)."""
        return int(self.v_headdim)

    @property
    def kv_dim(self) -> int:
        """Total unpadded KV dimension (``kv_lora_rank + qk_rope_head_dim``)."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_dim_padded(self) -> int:
        """Total KV dimension after padding each component to a multiple of 128.

        The ``ejkernel`` backend pads the low-rank KV and RoPE components
        separately to 128-element boundaries for efficient memory access.
        """
        # ejkernel pads lkv and rope components separately.
        return align_to_multiple(self.kv_lora_rank, 128) + align_to_multiple(self.qk_rope_head_dim, 128)

    @property
    def kv_packing(self) -> int:
        """Number of elements packed per dtype unit (e.g. 2 for ``bfloat16``)."""
        return get_dtype_packing(self.kvdtype)

    @property
    def page_size_per_kv_packing(self) -> int:
        """Page size divided by the dtype packing factor, rounded up."""
        return cdiv(int(self.page_size), int(self.kv_packing))

    def get_shape_and_axes(self):
        """Return the page tensor shape and corresponding sharding axes.

        Returns:
            A ``(shape, axes)`` tuple where *shape* is
            ``(num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded)``
            and *axes* assigns the data-parallel axis to the page dimension
            (or ``EMPTY`` when running on a single device).
        """
        page_axis = ATTN_DP if self.data_parallel_size > 1 else common_types.EMPTY
        kv_pages_shape = (
            self.num_pages,
            self.page_size_per_kv_packing,
            self.kv_packing,
            self.kv_dim_padded,
        )
        axes = [page_axis, common_types.EMPTY, common_types.EMPTY, common_types.EMPTY]
        return kv_pages_shape, axes


@auto_pytree
class MLARaggedPagesCacheView(RaggedPagesCacheView):
    """Per-layer view into the MLA ragged page cache.

    Each view owns the page buffer for a single transformer layer.  Because MLA
    compresses keys and values into a shared latent, there is only one page
    tensor (``kv_pages``) rather than separate key and value buffers.  The
    ``key_pages`` and ``value_pages`` properties both return this unified buffer
    for API compatibility with non-MLA cache views.

    Attributes:
        metadata: The :class:`MLARaggedPagesCacheConfig` describing page layout.
        kv_pages: Compressed KV page buffer of shape
            ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``,
            or an ``ImplicitArray`` when quantization is applied.
    """

    metadata: MLARaggedPagesCacheConfig
    kv_pages: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"] | ImplicitArray

    @classmethod
    def init(
        cls,
        config: MLARaggedPagesCacheConfig,
        layer_index: int | None = None,
        *,
        mesh: Mesh | None = None,
        partition_manager: es.PartitionManager | None = None,
        quantizer: EasyQuantizer | None = None,
    ) -> "MLARaggedPagesCacheView":
        """Allocate a single-layer MLA ragged page cache view.

        Args:
            config: MLA cache configuration describing page layout and dimensions.
            layer_index: Transformer layer index this view belongs to (defaults to 0).
            mesh: JAX device mesh used for sharding the page buffer.
            partition_manager: Partition manager that resolves sharding specs.
            quantizer: Optional quantizer applied to the allocated page buffer.

        Returns:
            An initialized ``MLARaggedPagesCacheView`` with zeroed page storage.
        """
        from easydel.layers.quantization._quants import EasyQuantizer as EQ

        if quantizer is None:
            quantizer = EQ(quantization_config=None)
        if partition_manager is None:
            partition_manager = PartitionManager(PartitionAxis())

        kv_pages_shape, axes = config.get_shape_and_axes()
        kv_pages_sharding = partition_manager.resolve(axes=axes, mode=common_types.MODE_PREFILL, shape=kv_pages_shape)
        kv_pages_sharding = Ns(mesh=mesh, spec=kv_pages_sharding)

        with jax.named_scope("easydel-mla-ragged-cache-init"):
            kv_pages = quantizer(jnp.zeros(shape=kv_pages_shape, dtype=config.kvdtype, device=kv_pages_sharding))

        return cls(
            metadata=config,
            layer_index=layer_index or 0,
            kv_pages=kv_pages,
            partition_manager=partition_manager,
        )

    def flattened_kv_pages(self) -> jax.Array:
        """Materialize and return the KV page buffer as a concrete JAX array.

        If the underlying storage is an :class:`ImplicitArray` (e.g. after
        quantization), it is materialized into a dense JAX array first.

        Returns:
            jax.Array: Dense page buffer of shape
                ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.
        """
        pages = self.kv_pages
        if isinstance(pages, ImplicitArray):
            pages = pages.materialize()
        return pages

    @property
    def key_pages(self):
        """Return the key page buffer (alias for :meth:`flattened_kv_pages`).

        In MLA the key and value projections share a single compressed
        latent, so this returns the same unified buffer as ``value_pages``.
        """
        return self.flattened_kv_pages()

    @property
    def value_pages(self):
        """Return the value page buffer (alias for :meth:`flattened_kv_pages`).

        In MLA the key and value projections share a single compressed
        latent, so this returns the same unified buffer as ``key_pages``.
        """
        return self.flattened_kv_pages()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(layer_index={self.layer_index}, "
            f"kv_shape={getattr(self.kv_pages, 'shape', None)})"
        )

    __str__ = __repr__


@auto_pytree
class MLARaggedPagesCache(RaggedPagesCache):
    """Top-level container holding one :class:`MLARaggedPagesCacheView` per transformer layer.

    This is the object passed through the model during inference.  Use
    :meth:`init_cache` to allocate the full cache from a
    :class:`MLARaggedPagesCacheConfig`.

    Attributes:
        views: List of per-layer cache views, indexed by layer number.
            Entries may be ``None`` for layers that do not use KV caching
            (e.g. the first layer in hybrid architectures).
    """

    views: list[MLARaggedPagesCacheView | None]

    @property
    def metadata(self) -> MLARaggedPagesCacheConfig | None:
        """Return the shared config from the last layer view, or ``None`` if empty."""
        if not self.views or self.views[-1] is None:
            return None
        return self.views[-1].metadata

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        config: MLARaggedPagesCacheConfig,
        partition_manager: es.PartitionManager,
        quantizer: EasyQuantizer | None = None,
    ) -> "MLARaggedPagesCache":
        """Allocate an MLA ragged page cache for all transformer layers.

        Creates one :class:`MLARaggedPagesCacheView` per layer, each backed by
        a zeroed page buffer sharded according to *partition_manager*.

        Args:
            mesh: JAX device mesh for sharding.
            config: MLA cache configuration (page count, dimensions, etc.).
            partition_manager: Resolves sharding specs for page tensors.
            quantizer: Optional quantizer applied to each layer's page buffer.

        Returns:
            An ``MLARaggedPagesCache`` containing ``config.num_hidden_layers`` views.
        """
        views = [
            MLARaggedPagesCacheView.init(
                config=config,
                layer_index=i,
                mesh=mesh,
                partition_manager=partition_manager,
                quantizer=quantizer,
            )
            for i in range(config.num_hidden_layers)
        ]
        return cls(views=views)
