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
import spectrax as spx
from eformer.jaximus import ImplicitArray
from eformer.loggings import get_logger
from eformer.mpric import DTYPE_TO_STRING_MAP
from eformer.pytree import auto_pytree, field
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array, Float
from spectrax import common_types

from easydel.axis import ATTN_DP, resolve_attention_data_parallel_axis
from easydel.infra.sharding import (
    MeshLike,
    RuntimeShardingResolver,
    coerce_runtime_sharding_resolver,
    is_mpmd_mesh,
    mesh_axis_size,
    normalize_axis_names,
    resolve_stage_cache_mesh,
)

from ..budget import cap_paged_cache_pages
from ..ragged_page.cache import (
    HEAD_DIM_ALIGNMENT,
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
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

    Sparse-attention (DSA) models may also keep their indexer state in pages that
    share the KV page table (``indexer_key_dim > 0``): per-token ``key | gate`` rows
    and, for pooled indexers (``indexer_kpool > 1``), one summary per completed pool.
    Those bytes are included in the per-page HBM budget.
    """

    indexer_key_dim: int = field(pytree_node=False, default=0)
    indexer_gate_dim: int = field(pytree_node=False, default=0)
    indexer_kpool: int = field(pytree_node=False, default=1)
    _indexer_dtype_str: str = field(pytree_node=False, default="bf16")
    # Layers charged for indexer pages by ``create`` (``None`` = every layer) and, optionally,
    # which cache-view indices own them. ``init_cache`` allocates exactly that set (r11 S09-1).
    indexer_num_layers: int | None = field(pytree_node=False, default=None)
    indexer_layer_indices: tuple[int, ...] | None = field(pytree_node=False, default=None)

    @staticmethod
    def _compute_free_hbm(
        mesh: MeshLike,
        runtime_sharding_resolver: RuntimeShardingResolver,
        hbm_utilization: float,
    ):
        """Compute free HBM budget available for MLA cache pages.

        Unlike standard MHA caches, MLA pages are not sharded across the
        KV-head axis.  The budget is therefore scaled only by the
        data-parallel axis size.

        Args:
            mesh: JAX device mesh used to look up axis sizes.
            runtime_sharding_resolver: Provides the data-parallel axis name.
            hbm_utilization: Fraction of total HBM to consider available
                (0.0-1.0).

        Returns:
            Total allocatable bytes across all devices on the page axis.
        """
        budget = per_device_hbm_budget_bytes(hbm_utilization, mode="free")
        page_axis_size = mesh_axis_size(mesh, resolve_attention_data_parallel_axis(runtime_sharding_resolver))
        available_alloc = budget * page_axis_size
        logger.info(f"{page_axis_size=} {budget=} {available_alloc=} {hbm_utilization=}")
        return available_alloc

    @classmethod
    def create(
        cls,
        mesh: MeshLike,
        runtime_sharding_resolver: RuntimeShardingResolver,
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
        max_cache_tokens: int | None = None,
        indexer_key_dim: int = 0,
        indexer_gate_dim: int = 0,
        indexer_kpool: int = 1,
        indexer_dtype: jnp.dtype | None = None,
        indexer_num_layers: int | None = None,
        indexer_layer_indices: tp.Sequence[int] | None = None,
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
            runtime_sharding_resolver: Resolves sharding specs and provides the
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
            max_cache_tokens: Optional hard ceiling on the total tokens the
                page pool may hold (``num_pages * page_size``); the final page
                count is ``min(hbm_utilization-derived, this)``. ``None`` keeps
                pure HBM sizing.
            indexer_key_dim: Paged DSA indexer key width (0 disables indexer pages).
            indexer_gate_dim: Per-token gate width stored beside each key (pooled
                indexers; 0 for per-token indexers).
            indexer_kpool: Tokens per indexer pool; ``> 1`` also allocates pool
                summary pages and requires ``page_size % indexer_kpool == 0``.
            indexer_dtype: Storage dtype of the indexer pages (default: ``kvdtype`` when it
                is at least 16 bits wide, otherwise bf16).
            indexer_num_layers: Number of layers that own indexer pages (layers with a
                ``"full"`` indexer). ``None`` charges every cached layer. Only these layers'
                indexer bytes are counted in the page budget.
            indexer_layer_indices: Explicit cache-view indices that own indexer pages. When
                given, it defines ``indexer_num_layers`` and lets :meth:`MLARaggedPagesCache.init_cache`
                allocate exactly the budgeted owners; a reduced ``indexer_num_layers`` without it
                can only be allocated by callers that pass ``with_indexer_pages`` per view.

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

        data_parallel_size = mesh_axis_size(mesh, resolve_attention_data_parallel_axis(runtime_sharding_resolver))
        if data_parallel_size > 1:
            logger.info(f"Scaling MLA KV page budget by data-parallel page axis: {data_parallel_size=}.")

        free = cls._compute_free_hbm(
            mesh=mesh,
            runtime_sharding_resolver=runtime_sharding_resolver,
            hbm_utilization=hbm_utilization,
        )
        bytes_av = jnp.finfo(kvdtype).bits // 8
        kv_dim_padded = align_to_multiple(int(kv_lora_rank), HEAD_DIM_ALIGNMENT) + align_to_multiple(
            int(qk_rope_head_dim), HEAD_DIM_ALIGNMENT
        )
        kv_packing = get_dtype_packing(kvdtype)
        page_size_per_kv_packing = cdiv(page_size, kv_packing)
        packed_page_tokens = page_size_per_kv_packing * kv_packing

        page_bytes_per_layer = packed_page_tokens * kv_dim_padded * bytes_av
        if indexer_key_dim < 0 or indexer_gate_dim < 0 or indexer_kpool < 1:
            raise ValueError("indexer dims must be non-negative and indexer_kpool >= 1.")
        if indexer_dtype is None:
            # The dense indexer keeps its packed state in the model (KV) dtype. A narrower
            # (quantized, e.g. fp8) KV cache must not drag the indexer keys down with it:
            # top-k selection needs the raw scores, so such caches store indexer state in bf16.
            indexer_dtype = kvdtype if jnp.dtype(kvdtype).itemsize >= 2 else jnp.bfloat16
        if indexer_key_dim > 0:
            if page_size % indexer_kpool:
                raise ValueError(
                    f"page_size ({page_size}) must be a multiple of indexer_kpool ({indexer_kpool}) "
                    "so indexer pools never straddle a page."
                )
            indexer_bytes = jnp.dtype(indexer_dtype).itemsize
            indexer_page_bytes = page_size * (indexer_key_dim + indexer_gate_dim) * indexer_bytes
            if indexer_kpool > 1:
                indexer_page_bytes += (page_size // indexer_kpool) * indexer_key_dim * indexer_bytes
        else:
            indexer_page_bytes = 0
        if indexer_layer_indices is not None:
            indexer_layer_indices = tuple(sorted({int(i) for i in indexer_layer_indices}))
            if any(not 0 <= i < int(num_hidden_layers) for i in indexer_layer_indices):
                raise ValueError(f"indexer_layer_indices {indexer_layer_indices} must lie in [0, {num_hidden_layers}).")
            if indexer_num_layers is not None and int(indexer_num_layers) != len(indexer_layer_indices):
                raise ValueError(
                    f"indexer_num_layers={indexer_num_layers} disagrees with "
                    f"{len(indexer_layer_indices)} indexer_layer_indices."
                )
            indexer_num_layers = len(indexer_layer_indices)
        if indexer_num_layers is None:
            indexer_num_layers = num_hidden_layers
        indexer_num_layers = max(0, min(int(indexer_num_layers), int(num_hidden_layers)))
        page_bytes = num_hidden_layers * page_bytes_per_layer + indexer_num_layers * indexer_page_bytes
        num_pages = int(free) // int(page_bytes)
        num_pages = cap_paged_cache_pages(
            num_pages,
            page_size=page_size,
            data_parallel_size=data_parallel_size,
            max_cache_tokens=max_cache_tokens,
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
            indexer_key_dim=int(indexer_key_dim),
            indexer_gate_dim=int(indexer_gate_dim) if indexer_key_dim > 0 else 0,
            indexer_kpool=int(indexer_kpool) if indexer_key_dim > 0 else 1,
            _indexer_dtype_str=DTYPE_TO_STRING_MAP[jnp.dtype(indexer_dtype).type],
            indexer_num_layers=int(indexer_num_layers) if indexer_key_dim > 0 else None,
            indexer_layer_indices=indexer_layer_indices if indexer_key_dim > 0 else None,
        )

    @property
    def has_indexer_pages(self) -> bool:
        """Whether paged DSA indexer state is allocated beside the KV pages."""
        return int(self.indexer_key_dim) > 0

    @property
    def indexer_dtype(self) -> jnp.dtype:
        """Storage dtype of the indexer pages."""
        from eformer.mpric import STRING_TO_DTYPE_MAP

        return STRING_TO_DTYPE_MAP[self._indexer_dtype_str]

    def get_indexer_shapes(self) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        """Return ``(token_pages_shape, pool_pages_shape)``; ``None`` when not allocated.

        Token pages are ``[num_pages, page_size, key + gate]``; pool pages
        ``[num_pages, page_size // kpool, key]`` exist only for pooled indexers.
        """
        if not self.has_indexer_pages:
            return None, None
        token = (self.num_pages, self.page_size, self.indexer_key_dim + self.indexer_gate_dim)
        pool = None
        if self.indexer_kpool > 1:
            pool = (self.num_pages, self.page_size // self.indexer_kpool, self.indexer_key_dim)
        return token, pool

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
        """Total KV dimension after padding each component to ``HEAD_DIM_ALIGNMENT``.

        The ``ejkernel`` backend pads the low-rank KV and RoPE components
        separately for efficient memory access.
        """
        return align_to_multiple(self.kv_lora_rank, HEAD_DIM_ALIGNMENT) + align_to_multiple(
            self.qk_rope_head_dim, HEAD_DIM_ALIGNMENT
        )

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
    indexer_token_pages: Float[Array, "num_pages page_size indexer_dim"] | None = None
    indexer_pool_pages: Float[Array, "num_pages pools_per_page indexer_key_dim"] | None = None

    @classmethod
    def init(
        cls,
        config: MLARaggedPagesCacheConfig,
        layer_index: int | None = None,
        *,
        mesh: MeshLike | None = None,
        runtime_sharding_resolver: RuntimeShardingResolver | None = None,
        quantizer: EasyQuantizer | None = None,
        with_indexer_pages: bool = True,
    ) -> "MLARaggedPagesCacheView":
        """Allocate a single-layer MLA ragged page cache view.

        Args:
            config: MLA cache configuration describing page layout and dimensions.
            layer_index: Transformer layer index this view belongs to (defaults to 0).
            mesh: JAX device mesh used for sharding the page buffer.
            runtime_sharding_resolver: Runtime sharding resolver for page tensors.
            quantizer: Optional quantizer applied to the allocated page buffer.
            with_indexer_pages: Allocate this layer's paged DSA indexer buffers when the
                config enables them. Layers that reuse another layer's selection
                (``"shared"`` indexers) pass ``False``: they never read indexer state.
                Callers decide the serving path from ``config.has_indexer_pages``, not
                from these buffers.

        Returns:
            An initialized ``MLARaggedPagesCacheView`` with zeroed page storage.
        """
        from easydel.layers.quantization._quants import EasyQuantizer as EQ

        if quantizer is None:
            quantizer = EQ(quantization_config=None)
        mesh = resolve_stage_cache_mesh(mesh)
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(runtime_sharding_resolver, mesh=mesh)

        kv_pages_shape, axes = config.get_shape_and_axes()
        kv_pages_sharding = runtime_sharding_resolver.resolve(
            axes=axes,
            mode=common_types.MODE_PREFILL,
            shape=kv_pages_shape,
        )
        kv_pages_sharding = Ns(mesh=mesh, spec=kv_pages_sharding)

        with jax.named_scope("easydel-mla-ragged-cache-init"):
            kv_pages = quantizer(jnp.zeros(shape=kv_pages_shape, dtype=config.kvdtype, device=kv_pages_sharding))

        # Paged DSA indexer state shares the KV page table, so it is sharded on the
        # same page axis and is never quantized (scores need the raw keys).
        def _indexer_buffer(shape_: tuple[int, ...] | None) -> jax.Array | None:
            if shape_ is None:
                return None
            spec = runtime_sharding_resolver.resolve(
                axes=[axes[0], common_types.EMPTY, common_types.EMPTY],
                mode=common_types.MODE_PREFILL,
                shape=shape_,
            )
            return jnp.zeros(shape_, dtype=config.indexer_dtype, device=Ns(mesh=mesh, spec=spec))

        token_shape, pool_shape = config.get_indexer_shapes() if with_indexer_pages else (None, None)
        with jax.named_scope("easydel-mla-ragged-indexer-init"):
            indexer_token_pages = _indexer_buffer(token_shape)
            indexer_pool_pages = _indexer_buffer(pool_shape)

        return cls(
            metadata=config,
            layer_index=layer_index or 0,
            kv_pages=kv_pages,
            runtime_sharding_resolver=runtime_sharding_resolver,
            indexer_token_pages=indexer_token_pages,
            indexer_pool_pages=indexer_pool_pages,
        )

    def reset(self) -> "MLARaggedPagesCacheView":
        """Zero latent KV and optional indexer state, preserving layout metadata."""
        return self.replace(
            kv_pages=jnp.zeros_like(self.kv_pages),
            indexer_token_pages=None if self.indexer_token_pages is None else jnp.zeros_like(self.indexer_token_pages),
            indexer_pool_pages=None if self.indexer_pool_pages is None else jnp.zeros_like(self.indexer_pool_pages),
        )

    def indexer_kernel_mesh(self):
        """Mesh for running un-partitionable indexer kernels on replicated pages, or ``None``.

        Returns the resolver mesh only while the page axis is unsharded (DP size 1): the
        indexer pages are then replicated, so a Pallas call can run per device under a
        replicated ``shard_map``. Page-sharded caches keep the auto-partitioned XLA path.
        """
        resolver = getattr(self, "runtime_sharding_resolver", None)
        if resolver is None:
            return None
        _, axes = self.metadata.get_shape_and_axes()
        if axes[0] != common_types.EMPTY:
            return None
        mesh = getattr(resolver, "mesh", None)
        # A pipeline (MPMD) mesh spans devices outside the executing stage; a replicated
        # shard_map over it would be wrong inside a per-stage jit, so keep the XLA path.
        if mesh is None or is_mpmd_mesh(mesh):
            return None
        # Re-resolve against the active stage context: the resolver mesh was captured at cache
        # init and must match the mesh the attention op itself resolves at call time.
        mesh = resolve_stage_cache_mesh(mesh)
        if mesh is None or is_mpmd_mesh(mesh):
            return None
        return mesh

    def indexer_dp_sharding(self) -> tuple[tp.Any, tuple[str, ...]] | None:
        """``(mesh, page_axis_names)`` of DP-sharded indexer pages, or ``None`` when unsharded.

        Rank-major data-parallel serving runs the paged indexer once per DP rank under a
        ``shard_map`` over these axes (see ``layers.indexer._paged``), on the rank's own page
        shard, exactly like the DP-local MLA attention path.
        """
        resolver = getattr(self, "runtime_sharding_resolver", None)
        if resolver is None:
            return None
        _, axes = self.metadata.get_shape_and_axes()
        if axes[0] == common_types.EMPTY:
            return None
        mesh = getattr(resolver, "mesh", None)
        if mesh is None or is_mpmd_mesh(mesh):
            return None
        mesh = resolve_stage_cache_mesh(mesh)
        if mesh is None or is_mpmd_mesh(mesh):
            return None
        spec = resolver.resolve(axes=[axes[0]], mode=common_types.MODE_PREFILL, shape=(int(self.metadata.num_pages),))
        names = normalize_axis_names(spec[0] if len(spec) > 0 else None)
        if not names:
            return None
        return mesh, tuple(names)

    def constrain_indexer_pages(self, x: jax.Array) -> jax.Array:
        """Pin an indexer page buffer (or its flat token view) to its allocation sharding.

        Without this, the tensor-parallel sharding of the indexer projections propagates
        through the per-step scatter into the (page-axis-only) page buffers, so the layer
        returns them re-sharded: donation is lost and every step relayouts whole buffers.

        Args:
            x: Page buffer ``[P, ...]`` or flat view ``[P * rows, D]``.

        Returns:
            ``x`` constrained to ``P(page_axis, None, ...)``.
        """
        resolver = getattr(self, "runtime_sharding_resolver", None)
        if resolver is None:
            return x
        _, axes = self.metadata.get_shape_and_axes()
        return resolver.shard(x, axes=[axes[0]] + [common_types.EMPTY] * (x.ndim - 1), mode=common_types.MODE_PREFILL)

    def constrain_indexer_rows(self, x: jax.Array) -> jax.Array:
        """Fully replicate per-token rows before they are scattered into indexer pages.

        Args:
            x: Update rows ``[T, ...]``.

        Returns:
            ``x`` constrained to a fully replicated spec (only the small rows move).
        """
        resolver = getattr(self, "runtime_sharding_resolver", None)
        if resolver is None:
            return x
        return resolver.shard(x, axes=[common_types.EMPTY] * x.ndim, mode=common_types.MODE_PREFILL)

    def flattened_kv_pages(self) -> jax.Array:
        """Materialize and return the KV page buffer as a concrete JAX array.

        If the underlying storage is an :class:`ImplicitArray` (e.g. after
        quantization), it is materialized into a dense JAX array first.

        Returns:
            jax.Array: Dense page buffer of shape
                ``[num_pages, page_size_per_kv_packing, kv_packing, kv_dim_padded]``.
        """
        pages = self.kv_pages
        if pages is None:
            raise ValueError("MLA ragged page cache has no KV pages.")
        if isinstance(pages, ImplicitArray):
            pages = pages.materialize()
        return tp.cast(Array, pages)

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
        """Return a short ``repr`` showing layer index and KV page shape.

        Returns:
            A human-readable representation of the cache view.
        """
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
        mesh: MeshLike,
        config: MLARaggedPagesCacheConfig,
        runtime_sharding_resolver: RuntimeShardingResolver,
        quantizer: EasyQuantizer | None = None,
    ) -> "MLARaggedPagesCache":
        """Allocate an MLA ragged page cache for all transformer layers.

        Creates one :class:`MLARaggedPagesCacheView` per layer, each backed by
        a zeroed page buffer sharded according to *runtime_sharding_resolver*.

        Args:
            mesh: JAX device mesh for sharding.
            config: MLA cache configuration (page count, dimensions, etc.).
            runtime_sharding_resolver: Resolves sharding specs for page tensors.
            quantizer: Optional quantizer applied to each layer's page buffer.

        Returns:
            An ``MLARaggedPagesCache`` containing ``config.num_hidden_layers`` views.
        """
        num_layers = int(config.num_hidden_layers)
        owners = None  # None: every layer owns indexer pages (what ``create`` budgets by default)
        if config.has_indexer_pages:
            charged = config.indexer_num_layers
            if config.indexer_layer_indices is not None:
                owners = frozenset(int(i) for i in config.indexer_layer_indices)
            elif charged is not None and int(charged) < num_layers:
                # r11 S09-1: allocating every layer would exceed the page budget create() computed.
                raise ValueError(
                    f"MLA cache config budgets indexer pages for {charged} of {num_layers} layers but "
                    "records no owner schedule; pass `indexer_layer_indices` to create(), or build the "
                    "views with MLARaggedPagesCacheView.init(..., with_indexer_pages=...)."
                )
        views = []
        for i in range(num_layers):
            with spx.assign_stage(total=num_layers, current=i):
                views.append(
                    MLARaggedPagesCacheView.init(
                        config=config,
                        layer_index=i,
                        mesh=mesh,
                        runtime_sharding_resolver=runtime_sharding_resolver,
                        quantizer=quantizer,
                        with_indexer_pages=owners is None or i in owners,
                    )
                )
        return cls(views=views)
