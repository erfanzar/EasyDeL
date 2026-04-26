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

"""TurboQuant-compressed ragged page KV cache implementation.

Stores KV caches using TurboQuant two-stage vector quantization:
- Key indices pages: 4-bit packed Lloyd-Max codebook indices
- Key signs pages: bit-packed QJL residual signs
- Key norms pages: original + residual norms (bf16)
- Value indices pages: 4-bit packed Lloyd-Max codebook indices
- Value norms pages: original norms (bf16)

Plus precomputed constants: rotation matrix, QJL projection, codebooks.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree, field
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array, Float
from spectrax import common_types

from easydel.axis import ATTN_DP, resolve_attention_data_parallel_axis
from easydel.infra.sharding import RuntimeShardingResolver, coerce_runtime_sharding_resolver
from easydel.layers.quantization import TurboQuantConfig, TurboQuantConstants

from .._abstracts import OperationsMetadata
from ..ragged_page.cache import (
    RaggedPagesCache,
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    _mesh_axis_size,
    cdiv,
    per_device_hbm_budget_bytes,
)

if tp.TYPE_CHECKING:
    from easydel.layers.quantization._quants import EasyQuantizer
else:
    EasyQuantizer = object

logger = get_logger(__name__)


@auto_pytree
class TurboQuantRaggedPagesCacheConfig(RaggedPagesCacheConfig):
    """Configuration for TurboQuant-compressed ragged pages cache.

    Extends RaggedPagesCacheConfig with TurboQuant-specific parameters.
    The page layout stores 5 separate tensors per layer instead of one
    interleaved KV tensor.

    The ``turboquant_config`` and ``turboquant_constants`` fields are
    marked as non-pytree nodes since they contain configuration and
    precomputed constants that don't participate in JAX transformations.
    """

    turboquant_config: TurboQuantConfig = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        mesh: Mesh,
        runtime_sharding_resolver: RuntimeShardingResolver,
        turboquant_config: TurboQuantConfig,
        num_hidden_layers: int,
        num_kv_heads: int,
        max_model_length: int,
        kv_head_dim_size: int,
        hbm_utilization: float = 0.9,
        page_size: int = 128,
    ) -> "TurboQuantRaggedPagesCacheConfig":
        """Create a TurboQuant cache config from model hyperparameters.

        Computes the compressed page sizes and maximum page count that
        fits in the available HBM budget.

        Args:
            mesh: JAX device mesh.
            runtime_sharding_resolver: Resolves runtime shardings for cache tensors.
            turboquant_config: TurboQuant algorithm configuration.
            num_hidden_layers: Number of transformer layers.
            num_kv_heads: Number of KV attention heads.
            max_model_length: Maximum sequence length.
            kv_head_dim_size: Attention head dimension (head_dim).
            hbm_utilization: Fraction of free HBM to use.
            page_size: Tokens per page.

        Returns:
            Configured TurboQuantRaggedPagesCacheConfig.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(runtime_sharding_resolver, mesh=mesh)
        data_parallel_size = _mesh_axis_size(mesh, resolve_attention_data_parallel_axis(runtime_sharding_resolver))
        kv_head_size = _mesh_axis_size(mesh, runtime_sharding_resolver.paxis.kv_head_axis)

        budget = per_device_hbm_budget_bytes(hbm_utilization, mode="free")
        page_axis_size = data_parallel_size if data_parallel_size > 1 else 1
        available_alloc = budget * kv_head_size * page_axis_size

        head_dim = kv_head_dim_size
        qjl_dim = turboquant_config.qjl_dim if turboquant_config.qjl_dim is not None else head_dim

        # Compute bytes per page per layer for all 5 tensors
        # Key indices: [page_size, num_kv_heads, head_dim // 2] uint8
        ki_bytes = page_size * num_kv_heads * (head_dim // 2)
        # Key signs: [page_size, num_kv_heads, qjl_dim // 8] uint8
        ks_bytes = page_size * num_kv_heads * (qjl_dim // 8)
        # Key norms: [page_size, num_kv_heads, 2] bf16
        kn_bytes = page_size * num_kv_heads * 2 * 2  # 2 bytes per bf16
        # Value indices: [page_size, num_kv_heads, head_dim // 2] uint8
        vi_bytes = page_size * num_kv_heads * (head_dim // 2)
        # Value norms: [page_size, num_kv_heads] bf16
        vn_bytes = page_size * num_kv_heads * 2

        page_bytes_per_layer = ki_bytes + ks_bytes + kn_bytes + vi_bytes + vn_bytes
        page_bytes_total = num_hidden_layers * page_bytes_per_layer

        num_pages = int(available_alloc) // int(page_bytes_total)
        if data_parallel_size > 1:
            num_pages = (num_pages // data_parallel_size) * data_parallel_size
        if num_pages <= 0:
            raise ValueError("Computed `num_pages` is non-positive; increase `hbm_utilization` or reduce model size.")

        # Compare with uncompressed bf16 equivalent
        uncompressed_bytes = page_size * num_kv_heads * head_dim * 2 * 2  # K+V, bf16
        compression_ratio = uncompressed_bytes / page_bytes_per_layer

        logger.info(
            f"TurboQuant cache: {num_pages=} pages, {page_bytes_per_layer=}B/layer/page, "
            f"compression={compression_ratio:.1f}x vs bf16, "
            f"capacity={int(num_pages * page_size / 1000)}K tokens"
        )

        return cls(
            num_hidden_layers=num_hidden_layers,
            max_model_length=max_model_length,
            num_kv_heads=num_kv_heads,
            k_headdim=head_dim,
            v_headdim=head_dim,
            hbm_utilization=hbm_utilization,
            data_parallel_size=data_parallel_size,
            page_size=page_size,
            num_pages=num_pages,
            max_num_pages_per_req=cdiv(max_model_length, page_size),
            num_slices_per_kv_cache_update_page=1,
            version="v3",
            _kvdtype_str="bfloat16",
            turboquant_config=turboquant_config,
        )


@auto_pytree
class TurboQuantRaggedPagesCacheView(RaggedPagesCacheView):
    """Per-layer view into the TurboQuant-compressed page cache.

    Stores 5 separate page arrays for the compressed KV representation,
    plus references to precomputed TurboQuant constants.

    The ``kv_pages`` field from the parent class is set to a dummy scalar
    since TurboQuant uses separate arrays. Use the specific page arrays
    (key_indices_pages, etc.) for actual data access.
    """

    key_indices_pages: Array = None  # int32 (pages, page_size, nkv, packed_idx_dim_aligned)
    key_signs_pages: Array = None  # int32 (pages, page_size, nkv, packed_sign_dim_aligned)
    key_norms_pages: Float[Array, "num_pages page_size num_kv_heads two"] = None
    value_indices_pages: Array = None  # int32 (pages, page_size, nkv, packed_idx_dim_aligned)
    value_norms_pages: Float[Array, "num_pages page_size num_kv_heads"] = None
    constants: TurboQuantConstants = field(pytree_node=False, default=None)

    @classmethod
    def init(
        cls,
        config: TurboQuantRaggedPagesCacheConfig,
        layer_index: int | None = None,
        *,
        mesh: "Mesh | None" = None,
        runtime_sharding_resolver: RuntimeShardingResolver | None = None,
        quantizer: "EasyQuantizer | None" = None,
        constants: TurboQuantConstants | None = None,
    ) -> "TurboQuantRaggedPagesCacheView":
        """Initialize a TurboQuant cache view with zeroed page buffers.

        Args:
            config: TurboQuant cache configuration.
            layer_index: Transformer layer index.
            mesh: JAX device mesh for sharding.
            runtime_sharding_resolver: Runtime sharding resolver.
            quantizer: Ignored (TurboQuant handles its own compression).
            constants: Precomputed TurboQuant constants. If None, generated
                from config.

        Returns:
            Initialized TurboQuantRaggedPagesCacheView.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(runtime_sharding_resolver, mesh=mesh)

        layer_idx = layer_index or 0
        tq_config = config.turboquant_config
        head_dim = config.k_headdim
        num_kv_heads = config.num_kv_heads
        num_pages = config.num_pages
        page_size = config.page_size
        qjl_dim = tq_config.qjl_dim if tq_config.qjl_dim is not None else head_dim

        if constants is None:
            constants = TurboQuantConstants.generate(tq_config, head_dim, layer_idx, mesh=mesh)

        page_axis = ATTN_DP if config.data_parallel_size > 1 else common_types.EMPTY

        def _make_pages(shape, np_dtype, axes):
            spec = runtime_sharding_resolver.resolve(axes=axes, mode=common_types.MODE_PREFILL, shape=shape)
            sharding = Ns(mesh=mesh, spec=spec)
            return jax.device_put(np.zeros(shape, dtype=np_dtype), sharding)

        # Common axes: [page_axis, EMPTY, KV_HEAD, EMPTY]
        axes_4d = [page_axis, common_types.EMPTY, common_types.KV_HEAD, common_types.EMPTY]
        axes_3d = [page_axis, common_types.EMPTY, common_types.KV_HEAD]

        packed_idx_dim = head_dim // 2
        packed_sign_dim = qjl_dim // 8

        with jax.named_scope("easydel-turboquant-cache-init"):
            ki = _make_pages((num_pages, page_size, num_kv_heads, packed_idx_dim), np.uint8, axes_4d)
            ks = _make_pages((num_pages, page_size, num_kv_heads, packed_sign_dim), np.uint8, axes_4d)
            kn = _make_pages((num_pages, page_size, num_kv_heads, 2), jnp.bfloat16, axes_4d)
            vi = _make_pages((num_pages, page_size, num_kv_heads, packed_idx_dim), np.uint8, axes_4d)
            vn = _make_pages((num_pages, page_size, num_kv_heads), jnp.bfloat16, axes_3d)

        # Dummy kv_pages for parent class compatibility
        dummy_kv = jnp.zeros((), dtype=jnp.bfloat16)

        return cls(
            metadata=config,
            layer_index=layer_idx,
            kv_pages=dummy_kv,
            runtime_sharding_resolver=runtime_sharding_resolver,
            key_indices_pages=ki,
            key_signs_pages=ks,
            key_norms_pages=kn,
            value_indices_pages=vi,
            value_norms_pages=vn,
            constants=constants,
        )

    @classmethod
    def init_all_layers(
        cls,
        config: TurboQuantRaggedPagesCacheConfig,
        num_layers: int,
        *,
        mesh: "Mesh | None" = None,
        runtime_sharding_resolver: RuntimeShardingResolver | None = None,
        layer_indices: list[int] | None = None,
    ) -> list["TurboQuantRaggedPagesCacheView"]:
        """Batch-allocate cache views for all layers at once.

        Uses ``numpy.zeros`` + ``jax.device_put`` to bypass XLA compilation
        overhead. Precomputes shardings once per tensor type, then reuses
        them across all layers.  This reduces the allocation from
        ``5 * num_layers`` XLA-compiled ``jnp.zeros`` calls to
        ``5 * num_layers`` fast ``device_put`` transfers (no compilation).

        Args:
            config: TurboQuant cache configuration.
            num_layers: Number of transformer layers.
            mesh: JAX device mesh for sharding.
            runtime_sharding_resolver: Runtime sharding resolver.
            layer_indices: Optional transformer layer indices. When provided,
                generated constants and view metadata keep the original layer
                numbering instead of using ``range(num_layers)``.

        Returns:
            List of ``num_layers`` initialized TurboQuantRaggedPagesCacheView.
        """
        runtime_sharding_resolver = coerce_runtime_sharding_resolver(runtime_sharding_resolver, mesh=mesh)
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        elif len(layer_indices) != int(num_layers):
            raise ValueError("`layer_indices` length must match `num_layers` for TurboQuant batch cache init.")

        tq_config = config.turboquant_config
        head_dim = config.k_headdim
        num_kv_heads = config.num_kv_heads
        num_pages = config.num_pages
        page_size = config.page_size
        qjl_dim = tq_config.qjl_dim if tq_config.qjl_dim is not None else head_dim

        page_axis = ATTN_DP if config.data_parallel_size > 1 else common_types.EMPTY

        # Per-layer axes (same as the single-layer init path)
        axes_4d = [page_axis, common_types.EMPTY, common_types.KV_HEAD, common_types.EMPTY]
        axes_3d = [page_axis, common_types.EMPTY, common_types.KV_HEAD]

        packed_idx_dim = head_dim // 2
        packed_sign_dim = qjl_dim // 8
        shapes_and_shardings = {
            "ki": ((num_pages, page_size, num_kv_heads, packed_idx_dim), np.uint8, axes_4d),
            "ks": ((num_pages, page_size, num_kv_heads, packed_sign_dim), np.uint8, axes_4d),
            "kn": ((num_pages, page_size, num_kv_heads, 2), jnp.bfloat16, axes_4d),
            "vi": ((num_pages, page_size, num_kv_heads, packed_idx_dim), np.uint8, axes_4d),
            "vn": ((num_pages, page_size, num_kv_heads), jnp.bfloat16, axes_3d),
        }
        resolved_shardings = {}
        for name, (shape, _, axes) in shapes_and_shardings.items():
            spec = runtime_sharding_resolver.resolve(axes=axes, mode=common_types.MODE_PREFILL, shape=shape)
            resolved_shardings[name] = Ns(mesh=mesh, spec=spec)

        logger.info(f"Batch-allocating TurboQuant cache for {num_layers} layers via device_put")

        views = []
        for _idx, layer_index in enumerate(layer_indices):
            dummy_kv = jax.device_put(np.zeros((), dtype=np.float32))
            constants = TurboQuantConstants.generate(tq_config, head_dim, int(layer_index), mesh=mesh)

            ki_shape = shapes_and_shardings["ki"][0]
            ks_shape = shapes_and_shardings["ks"][0]
            kn_shape = shapes_and_shardings["kn"][0]
            vi_shape = shapes_and_shardings["vi"][0]
            vn_shape = shapes_and_shardings["vn"][0]

            ki = jax.device_put(np.zeros(ki_shape, dtype=np.uint8), resolved_shardings["ki"])
            ks = jax.device_put(np.zeros(ks_shape, dtype=np.uint8), resolved_shardings["ks"])
            kn = jax.device_put(np.zeros(kn_shape, dtype=jnp.bfloat16), resolved_shardings["kn"])
            vi = jax.device_put(np.zeros(vi_shape, dtype=np.uint8), resolved_shardings["vi"])
            vn = jax.device_put(np.zeros(vn_shape, dtype=jnp.bfloat16), resolved_shardings["vn"])

            views.append(
                cls(
                    metadata=config,
                    layer_index=int(layer_index),
                    kv_pages=dummy_kv,
                    runtime_sharding_resolver=runtime_sharding_resolver,
                    key_indices_pages=ki,
                    key_signs_pages=ks,
                    key_norms_pages=kn,
                    value_indices_pages=vi,
                    value_norms_pages=vn,
                    constants=constants,
                )
            )

        logger.info(f"Batch allocation complete for {num_layers} layers")
        return views

    def concatenate_to_cache(
        self,
        key: Float[Array, "batch seq_len num_key_heads head_dim"],
        value: Float[Array, "batch seq_len num_value_heads head_dim"],
        cache_metadata: "RaggedPagesMetadata | OperationsMetadata",
    ) -> "TurboQuantRaggedPagesCacheView":
        """Update cache with new KV pairs.

        For v3 format, the kernel handles cache writes internally.
        This method returns self unchanged — the actual compression
        and cache update happens inside the ragged_page_attention_v3_turboquant
        kernel.

        Args:
            key: New key states [batch, seq_len, num_kv_heads, head_dim].
            value: New value states [batch, seq_len, num_kv_heads, head_dim].
            cache_metadata: Runtime metadata.

        Returns:
            Self (cache update is deferred to the kernel).
        """
        return self

    def __repr__(self) -> str:
        ki_shape = getattr(self.key_indices_pages, "shape", None)
        return f"{self.__class__.__name__}(layer_index={self.layer_index}, key_indices_shape={ki_shape})"

    __str__ = __repr__


@auto_pytree
class TurboQuantRaggedPagesCache(RaggedPagesCache):
    """Multi-layer container for TurboQuant-compressed ragged page caches.

    Attributes:
        views: List of per-layer TurboQuantRaggedPagesCacheView instances.
    """

    views: list[TurboQuantRaggedPagesCacheView | None]

    @property
    def metadata(self) -> TurboQuantRaggedPagesCacheConfig | None:
        if not self.views or self.views[-1] is None:
            return None
        return self.views[-1].metadata

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        config: TurboQuantRaggedPagesCacheConfig,
        runtime_sharding_resolver: RuntimeShardingResolver,
        quantizer: EasyQuantizer | None = None,
    ) -> "TurboQuantRaggedPagesCache":
        """Allocate TurboQuant cache for all transformer layers.

        Args:
            mesh: JAX device mesh.
            config: TurboQuant cache configuration.
            runtime_sharding_resolver: Resolves runtime shardings for cache tensors.
            quantizer: Ignored (TurboQuant handles its own compression).

        Returns:
            TurboQuantRaggedPagesCache with one view per layer.
        """
        views = TurboQuantRaggedPagesCacheView.init_all_layers(
            config=config,
            num_layers=config.num_hidden_layers,
            mesh=mesh,
            runtime_sharding_resolver=runtime_sharding_resolver,
        )
        return cls(views=views)
