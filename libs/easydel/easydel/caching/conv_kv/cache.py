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

"""Reusable KV + multi-channel short causal convolution cache.

Serves transformer architectures (such as Inkling 2 and Inkling 2 Loop) whose
decoder layers pair a standard full or sliding-window KV cache with one or more
float32 rolling short-convolution buffers per logical layer slot (for example
key/value short convolutions plus post-attention and post-MLP convolutions).

Each :class:`ConvKVCacheView` wraps a :class:`TransformerCacheView` alongside a
tuple of ``[batch, kernel_size - 1, dim]`` float32 convolution history buffers
and exposes ``indexes``, ``starts``, and ``transformer`` accessors so
:class:`~easydel.infra.mixins.generation.EasyGenerationMixin` and
:class:`~easydel.layers.attention.FlexibleAttentionModule` interoperate with it
directly.
"""

from __future__ import annotations

import contextlib
import typing as tp

from eformer.pytree import auto_pytree, field
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array
from spectrax import PartitionManager, common_types

from easydel.infra.sharding import (
    MeshLike,
    RuntimeShardingResolver,
    coerce_runtime_sharding_resolver,
    resolve_stage_cache_mesh,
    sanitize_sharding_axes_for_shape,
)
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

from .._abstracts import BaseCache, BaseCacheConfig, BaseCacheView, OperationsMetadata
from ..transformer import TransformerCacheConfig, TransformerCacheView, TransformerMetadata

__all__ = (
    "ConvKVCache",
    "ConvKVCacheConfig",
    "ConvKVCacheView",
)


@auto_pytree
class ConvKVCacheConfig(BaseCacheConfig):
    """Static configuration for :class:`ConvKVCache`.

    Attributes:
        batch_size: Number of sequences in the batch.
        sequence_length: Maximum cached sequence length.
        num_hidden_layers: Total number of logical cache slots (may exceed
            physical layer count for looped architectures).
        conv_kernel_size: Causal short-convolution kernel width; each rolling
            buffer stores ``max(0, conv_kernel_size - 1)`` history steps.
        layer_kv_shapes: Per-slot ``(num_key_value_heads, head_dim)`` pairs.
        layer_conv_dims: Per-slot tuple of feature widths for each convolution
            buffer owned by that slot.
        layer_sliding_windows: Per-slot sliding-window size, or ``None`` for
            full-attention slots.
        pad_token_id: Padding token id forwarded to :class:`TransformerCacheConfig`.
        masking_details: Optional per-slot :class:`AttnMaskDetail` metadata.
    """

    batch_size: int = field(pytree_node=False)
    sequence_length: int = field(pytree_node=False)
    num_hidden_layers: int = field(pytree_node=False)
    conv_kernel_size: int = field(pytree_node=False)
    layer_kv_shapes: tuple[tuple[int, int], ...] = field(pytree_node=False)
    layer_conv_dims: tuple[tuple[int, ...], ...] = field(pytree_node=False)
    layer_sliding_windows: tuple[int | None, ...] = field(pytree_node=False)
    pad_token_id: int = field(pytree_node=False, default=-100)
    masking_details: dict[int, AttnMaskDetail] | None = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        sequence_length: int,
        conv_kernel_size: int,
        layer_kv_shapes: tp.Sequence[tuple[int, int]],
        layer_conv_dims: tp.Sequence[tp.Sequence[int]],
        layer_sliding_windows: tp.Sequence[int | None] | None = None,
        pad_token_id: int | None = None,
        masking_details: dict[int, AttnMaskDetail] | None = None,
    ) -> ConvKVCacheConfig:
        """Create and validate a :class:`ConvKVCacheConfig`.

        Args:
            batch_size: Positive batch size.
            sequence_length: Positive maximum cached sequence length.
            conv_kernel_size: Positive causal convolution kernel width.
            layer_kv_shapes: Sequence of ``(num_kv_heads, head_dim)`` pairs,
                one per logical cache slot.
            layer_conv_dims: Sequence of per-slot convolution feature widths.
            layer_sliding_windows: Optional per-slot sliding window sizes
                (``None`` entries indicate full attention).
            pad_token_id: Optional pad token id (defaults to ``-100``).
            masking_details: Optional per-slot :class:`AttnMaskDetail` map.

        Returns:
            Validated :class:`ConvKVCacheConfig`.

        Raises:
            ValueError: If dimensions are non-positive or per-slot sequences
                disagree in length.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}.")
        kv_shapes = tuple((int(h), int(d)) for h, d in layer_kv_shapes)
        if not kv_shapes:
            raise ValueError("layer_kv_shapes must be non-empty.")
        conv_dims = tuple(tuple(int(c) for c in dims) for dims in layer_conv_dims)
        if len(conv_dims) != len(kv_shapes):
            raise ValueError(
                f"layer_conv_dims length ({len(conv_dims)}) must match layer_kv_shapes length ({len(kv_shapes)})."
            )
        if layer_sliding_windows is None:
            windows: tuple[int | None, ...] = (None,) * len(kv_shapes)
        else:
            windows = tuple(None if w is None else int(w) for w in layer_sliding_windows)
            if len(windows) != len(kv_shapes):
                raise ValueError(
                    f"layer_sliding_windows length ({len(windows)}) must match layer_kv_shapes ({len(kv_shapes)})."
                )
        resolved_details = masking_details
        if resolved_details is None:
            resolved_details = {
                idx: (
                    AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=win)
                    if win is not None
                    else AttnMaskDetail(mask_type=AttnMaskType.FULL, size=int(sequence_length))
                )
                for idx, win in enumerate(windows)
            }
        return cls(
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
            num_hidden_layers=len(kv_shapes),
            conv_kernel_size=int(conv_kernel_size),
            layer_kv_shapes=kv_shapes,
            layer_conv_dims=conv_dims,
            layer_sliding_windows=windows,
            pad_token_id=-100 if pad_token_id is None else int(pad_token_id),
            masking_details=resolved_details,
        )


@auto_pytree
class ConvKVCacheView(BaseCacheView):
    """Per-layer-slot cache holding KV storage and float32 short-conv history buffers."""

    kv: TransformerCacheView
    conv_states: tuple[Array, ...]
    metadata: ConvKVCacheConfig = field(pytree_node=False)
    layer_index: int = field(pytree_node=False)

    @property
    def transformer(self) -> TransformerCacheView:
        """Expose the underlying KV view for generic cache utilities."""
        return self.kv

    @property
    def indexes(self) -> Array:
        """Current write cursor per batch element (delegates to ``self.kv.indexes``)."""
        return self.kv.indexes

    @property
    def starts(self) -> Array:
        """Per-batch prompt start offset (delegates to ``self.kv.starts``)."""
        return self.kv.starts

    @property
    def masking_details(self) -> AttnMaskDetail | None:
        """Per-slot attention masking details (delegates to ``self.kv.masking_details``)."""
        return self.kv.masking_details

    @classmethod
    def init(
        cls,
        metadata: ConvKVCacheConfig,
        layer_index: int,
        *,
        mesh: MeshLike | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        starts: Array | None = None,
        quantizer: tp.Any = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
    ) -> ConvKVCacheView:
        """Allocate one KV + short-convolution cache view for ``layer_index``."""
        mesh = resolve_stage_cache_mesh(mesh)
        resolver = coerce_runtime_sharding_resolver(
            runtime_sharding_resolver if runtime_sharding_resolver is not None else partition_manager,
            mesh=mesh,
        )
        num_kv_heads, head_dim = metadata.layer_kv_shapes[layer_index]
        sliding_window = metadata.layer_sliding_windows[layer_index]
        mask_detail = metadata.masking_details.get(layer_index) if metadata.masking_details else None

        kv_config = TransformerCacheConfig.create(
            batch_size=metadata.batch_size,
            sequence_length=metadata.sequence_length,
            num_hidden_layers=metadata.num_hidden_layers,
            pad_token_id=metadata.pad_token_id,
            key_heads=num_kv_heads,
            value_heads=num_kv_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            sliding_window=sliding_window,
        )
        kv_view = TransformerCacheView.init(
            config=kv_config,
            layer_index=layer_index,
            mesh=mesh,
            dtype=dtype,
            starts=starts,
            quantizer=quantizer,
            runtime_sharding_resolver=resolver,
            masking_details=mask_detail,
        )

        state_len = max(0, metadata.conv_kernel_size - 1)
        conv_states: list[Array] = []
        for dim in metadata.layer_conv_dims[layer_index]:
            shape = (metadata.batch_size, state_len, dim)
            axes = sanitize_sharding_axes_for_shape(
                mesh=mesh,
                runtime_sharding_resolver=resolver,
                axes=(common_types.BATCH, None, None),
                mode=common_types.MODE_PREFILL,
                shape=shape,
            )
            sharding = Ns(mesh, resolver.resolve(axes=axes, mode=common_types.MODE_PREFILL, shape=shape))
            conv_states.append(jnp.zeros(shape, dtype=jnp.float32, device=sharding))

        return cls(
            kv=kv_view,
            conv_states=tuple(conv_states),
            metadata=metadata,
            layer_index=layer_index,
        )

    def update_conv_state(self, conv_idx: int, masked_x_fp32: Array) -> tuple[Array, ConvKVCacheView]:
        """Prepend cached conv history to ``masked_x_fp32`` and return ``(context, updated_view)``."""
        state = self.conv_states[conv_idx]
        state_len = state.shape[1]
        context = jnp.concatenate([state, masked_x_fp32], axis=1)
        new_state = context[:, -state_len:, :] if state_len > 0 else state
        states = list(self.conv_states)
        states[conv_idx] = new_state
        return context, self.replace(conv_states=tuple(states))

    def concatenate_to_cache(
        self,
        query: Array,
        key: Array,
        value: Array,
        mode: common_types.RUNTIME_MODE_TYPES,
        quantizer: tp.Any,
        cache_metadata: TransformerMetadata | OperationsMetadata | None,
        mask_info: MaskInfo,
        runtime_sharding_resolver: RuntimeShardingResolver | PartitionManager | None = None,
        partition_manager: RuntimeShardingResolver | PartitionManager | None = None,
        causal: bool = True,
    ) -> tuple[Array, Array, MaskInfo, ConvKVCacheView, AttnMaskDetail | None]:
        """Delegate KV insertion and mask alignment to the underlying :class:`TransformerCacheView`."""
        k_out, v_out, aligned_mask, new_kv, mask_detail = self.kv.concatenate_to_cache(
            query=query,
            key=key,
            value=value,
            mode=mode,
            quantizer=quantizer,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            runtime_sharding_resolver=runtime_sharding_resolver,
            partition_manager=partition_manager,
            causal=causal,
        )
        return k_out, v_out, aligned_mask, self.replace(kv=new_kv), mask_detail

    def reset(self) -> ConvKVCacheView:
        """Return a zeroed copy of this cache view."""
        return self.replace(
            kv=self.kv.reset(),
            conv_states=tuple(jnp.zeros_like(s) for s in self.conv_states),
        )


@auto_pytree
class ConvKVCache(BaseCache):
    """Multi-layer cache container pairing KV views with rolling short-convolution buffers."""

    views: list[ConvKVCacheView | None]

    @classmethod
    def init_cache(
        cls,
        mesh: MeshLike,
        metadata: ConvKVCacheConfig,
        partition_manager: RuntimeShardingResolver | PartitionManager,
        dtype: jnp.dtype = jnp.bfloat16,
        starts: Array | None = None,
        quantizer: tp.Any = None,
        layer_context_fn: tp.Callable[[int], tp.ContextManager[tp.Any]] | None = None,
    ) -> ConvKVCache:
        """Allocate a :class:`ConvKVCache` with all logical layer views initialized.

        Args:
            mesh: Device mesh (SPMD ``Mesh`` or MPMD ``SpxMesh``).
            metadata: Static cache configuration.
            partition_manager: Runtime sharding resolver or partition manager.
            dtype: Data type for KV buffers (short-conv buffers stay float32).
            starts: Optional per-batch start offsets.
            quantizer: Optional KV cache quantizer.
            layer_context_fn: Optional callable returning a context manager for
                each logical slot index (e.g. ``self._layer_stage_context`` for
                stage-local MPMD cache allocation).
        """
        ctx_fn = layer_context_fn if layer_context_fn is not None else (lambda _idx: contextlib.nullcontext())
        views: list[ConvKVCacheView | None] = []
        with mesh:
            for idx in range(metadata.num_hidden_layers):
                with ctx_fn(idx):
                    views.append(
                        ConvKVCacheView.init(
                            metadata=metadata,
                            layer_index=idx,
                            mesh=mesh,
                            dtype=dtype,
                            starts=starts,
                            quantizer=quantizer,
                            runtime_sharding_resolver=partition_manager,
                        )
                    )
        return cls(views=views)

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> ConvKVCache:
        """Create a stateless placeholder cache with ``None`` views."""
        return cls(views=[None for _ in range(num_hidden_layers)])

    def reset(self) -> ConvKVCache:
        """Zero every allocated cache view."""
        return self.replace(views=[None if v is None else v.reset() for v in self.views])

    def get_seq_length(self, layer_idx: int = 0) -> Array | int:
        """Return the current cached sequence length (or 0 if uninitialized)."""
        if not self.views or self.views[layer_idx] is None:
            return 0
        return self.views[layer_idx].kv.indexes  # type: ignore[union-attr]
