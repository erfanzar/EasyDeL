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
from enum import Enum
from functools import partial

import chex as cx
import jax
from eformer import common_types
from eformer.escale import PartitionManager, apply_logical_sharding
from eformer.jaximus import ImplicitArray, register
from eformer.pytree import auto_pytree, field
from flax import nnx as nn
from jax import Array, lax
from jax import numpy as jnp
from jax.extend.core import Primitive
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object


@auto_pytree
class AttnMaskDetail:
    mask_type: Enum
    size: int
    offset: int | None = None
    chunks: int | None = None
    bricks: int | None = None


NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ
MODE_PREFILL = common_types.MODE_PREFILL


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: ImplicitArray,
    update: tp.Any,
    *args,
    **kwargs,
):
    operand = operand.materialize()
    return primitive.bind(operand, update, *args)


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: tp.Any,
    update: ImplicitArray,
    *args,
    **kwargs,
):
    update = update.materialize()
    return primitive.bind(operand, update, *args)


@register("dynamic_update_slice")
def _(
    primitive: Primitive,
    operand: ImplicitArray,
    update: ImplicitArray,
    *args,
    **kwargs,
):
    operand = operand.materialize()
    update = update.materialize()
    return primitive.bind(operand, update, *args)


@auto_pytree
class TransformerCacheMetaData(BaseCacheMetadata):
    """Metadata for transformer cache configuration."""

    batch_size: int
    sequence_length: int
    num_hidden_layers: int
    pad_token_id: int
    # Optional attention-related fields
    num_heads: int | None
    head_dim: int | None
    key_heads: int | None
    value_heads: int | None
    key_dim: int | None
    value_dim: int | None
    sliding_window: int | None

    # Configuration flags
    update_causal_mask: bool
    create_attention_bias: bool

    @classmethod
    def create(
        cls,
        batch_size: int,
        sequence_length: int,
        num_hidden_layers: int,
        pad_token_id: int,
        num_heads: int | None = None,
        head_dim: int | None = None,
        key_heads: int | None = None,
        value_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
        update_causal_mask: bool = True,
        create_attention_bias: bool = True,
        sliding_window: int | None = None,
    ) -> TransformerCacheMetaData:
        """
        Create a TransformerCacheMetaData instance with validation.

        Arguments:
            batch_size: Size of the batch.
            sequence_length: Length of the sequence.
            num_hidden_layers: number of hidden layers.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            key_heads: Number of key heads.
            value_heads: Number of value heads.
            key_dim: Dimension of keys.
            value_dim: Dimension of values.
            update_causal_mask: Whether to update causal mask.
            create_attention_bias: Whether to create attention bias.

        Returns:
            TransformerCacheMetaData instance

        Raises:
            ValueError: If required parameters are missing or invalid.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if head_dim is not None:
            key_dim = key_dim or head_dim
            value_dim = value_dim or head_dim
        else:
            if key_dim is None or value_dim is None:
                raise ValueError("Either head_dim or both key_dim and value_dim must be specified")

        # Derive heads from num_heads if not specified
        if num_heads is not None:
            key_heads = key_heads or num_heads
            value_heads = value_heads or num_heads
        else:
            if key_heads is None or value_heads is None:
                raise ValueError("Either num_heads or both key_heads and value_heads must be specified")

        return cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            num_heads=num_heads,
            head_dim=head_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            update_causal_mask=update_causal_mask,
            create_attention_bias=create_attention_bias,
            sliding_window=sliding_window,
        )


@auto_pytree(frozen=False)
class TransformerCacheView(BaseCacheView):
    key: cx.Array | ImplicitArray
    value: cx.Array | ImplicitArray
    indexs: cx.Array | ImplicitArray
    starts: cx.Array | ImplicitArray

    metadata: TransformerCacheMetaData

    maximum_sequence_length: int = field(pytree_node=False)

    layer_index: int | None = None
    masking_details: AttnMaskDetail | None = None

    @classmethod
    def init(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: TransformerCacheMetaData,
        quantizer: EasyQuantizer,
        partition_manager: PartitionManager,
        starts: jax.Array | None = None,
        layer_index: int | None = None,
        masking_details: AttnMaskDetail | None = None,
    ):
        from easydel.infra.utils import AttnMaskType

        with jax.named_scope("easydel-transformer-cacheview-init"):
            mt = metadata
            kshape = (mt.batch_size, mt.sequence_length, mt.key_heads, mt.key_dim)
            vshape = (mt.batch_size, mt.sequence_length, mt.value_heads, mt.value_dim)
            kv_sharding_axes = [BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM]

            if masking_details is not None:
                if masking_details.mask_type == AttnMaskType.SLIDING:
                    kshape = (mt.batch_size, masking_details.size, mt.key_heads, mt.key_dim)
                    vshape = (mt.batch_size, masking_details.size, mt.value_heads, mt.value_dim)

            kshardings = Ns(mesh, partition_manager.resolve(axes=kv_sharding_axes, mode=MODE_PREFILL, shape=kshape))
            vshardings = Ns(mesh, partition_manager.resolve(axes=kv_sharding_axes, mode=MODE_PREFILL, shape=vshape))
            ishardings = Ns(mesh, partition_manager.resolve(axes=[BATCH], mode=MODE_PREFILL, shape=(mt.batch_size,)))

            if starts is None:
                starts = jnp.zeros((mt.batch_size,), dtype=jnp.int32)

            starts = apply_logical_sharding(starts, axes=[BATCH], mode=MODE_PREFILL, partition_manager=partition_manager)

            out = cls(
                key=quantizer(jnp.zeros(shape=kshape, dtype=dtype, device=kshardings)),
                value=quantizer(jnp.zeros(shape=vshape, dtype=dtype, device=vshardings)),
                indexs=jnp.zeros((metadata.batch_size,), dtype=jnp.int32, device=ishardings),
                starts=starts,
                metadata=metadata,
                layer_index=layer_index,
                masking_details=masking_details,
                maximum_sequence_length=mt.sequence_length,
            )
        return out

    @jax.named_scope("easydel-transformer-cacheview-concatenate-to-cache")
    def concatenate_to_cache(
        self,
        query: cx.Array,
        key: cx.Array,
        value: cx.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        quantizer: EasyQuantizer,
        cache_metadata: TransformerMetadata | None,
        attention_mask: cx.Array,
        partition_manager: PartitionManager,
        causal_mask: cx.Array | None = None,
        token_type_ids: cx.Array | None = None,
    ) -> tuple[cx.Array, cx.Array, cx.Array, TransformerCacheView, AttnMaskDetail | None]:
        """
        Updates the KV cache functionally and returns the updated tensors along with the appropriate attention mask.

        Args:
            query: Current query states.
            key: Current key states to add to the cache.
            value: Current value states to add to the cache.
            cache_metadata: Optional metadata. If provided and contains slot/length info, enables pooled caching.
            attention_mask: Base attention mask.
            quantizer: Quantizer for the cache.
            causal_mask: Optional causal mask.
            token_type_ids: Optional token type IDs for segment masking.

        Returns:
            Tuple[Array, Array, Array]:
                - Updated key cache tensor (functional update).
                - Updated value cache tensor (functional update).
                - Final attention mask to be used (either original or calculated).
        """
        from easydel.infra.utils import AttnMaskType

        runtime_dtype = query.dtype
        num_updated_cache_vectors = query.shape[1]
        masking_details = self.masking_details
        indexs = self.indexs
        batch_dims = self.value.shape[0]
        sharding_statics = dict(mode=MODE_PREFILL, partition_manager=partition_manager)

        def _kv_struct_shard(x: cx.Array) -> cx.Array:
            return apply_logical_sharding(x, axes=[BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM], **sharding_statics)

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if causal_mask is not None:
            if hasattr(causal_mask, "value"):
                causal_mask = causal_mask.value
            if causal_mask.shape[0] != query.shape[0]:
                causal_mask = jnp.broadcast_to(causal_mask, (query.shape[0], *causal_mask.shape[1:]))

            @partial(jax.vmap, in_axes=(0, 0), out_axes=0)
            def _mask_slice(mask, slot):
                return lax.dynamic_slice(
                    mask,
                    (0, slot, 0),
                    (1, num_updated_cache_vectors, self.maximum_sequence_length),
                )

            causal_mask = _mask_slice(causal_mask, self.indexs)
            if token_type_ids is not None and num_updated_cache_vectors != 1:
                token_type_mask = jnp.equal(jnp.expand_dims(token_type_ids, 2), jnp.expand_dims(token_type_ids, 1))
                token_type_mask = jnp.where(token_type_ids == 0, False, token_type_mask)
                token_type_mask = jnp.expand_dims(token_type_mask, 1)
                sequence_length = token_type_ids.shape[1]
                masked_portion = jnp.logical_or(
                    token_type_mask[:, :, :num_updated_cache_vectors, :],
                    causal_mask[:, :, :, :sequence_length],
                )
                causal_mask = causal_mask.at[:, :, :, :sequence_length].set(masked_portion)

            attention_mask = nn.combine_masks(attention_mask, causal_mask)
        else:
            attention_mask = attention_mask

        def _maybe_materialize(x):
            if hasattr(x, "materialize"):
                x = x.materialize()
            return x

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0))
        def _update_kv(old, new, slot):
            return lax.dynamic_update_slice(old, new.astype(old.dtype), (slot, 0, 0))

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0))
        def _update_kv_sliding(old_cache, new_values, current_index):
            """Update sliding window KV cache."""
            new_len = new_values.shape[0]
            window_size = old_cache.shape[0]
            if new_len >= window_size:
                return new_values[-window_size:, :, :].astype(old_cache.dtype)

            total_tokens = current_index + new_len

            def _fits_in_window():
                return lax.dynamic_update_slice(old_cache, new_values.astype(old_cache.dtype), (current_index, 0, 0))

            def _overflow_window():
                return jnp.concatenate([old_cache[new_len:, :, :], new_values.astype(old_cache.dtype)], axis=0)

            return lax.cond(total_tokens <= window_size, _fits_in_window, _overflow_window)

        if masking_details is not None and masking_details.mask_type == AttnMaskType.SLIDING:
            value_cache_updated = _update_kv_sliding(_maybe_materialize(self.value), value, indexs)
            key_cache_updated = _update_kv_sliding(_maybe_materialize(self.key), key, indexs)
        else:
            value_cache_updated = _update_kv(_maybe_materialize(self.value), value, indexs)
            key_cache_updated = _update_kv(_maybe_materialize(self.key), key, indexs)

        indexs = indexs + num_updated_cache_vectors
        pad_mask = jnp.broadcast_to(
            (jnp.arange(self.maximum_sequence_length)[None, :] < indexs[:, None])[:, None, None, :],
            (batch_dims, 1, num_updated_cache_vectors, self.maximum_sequence_length),
        )

        value_cache_updated = _kv_struct_shard(value_cache_updated).astype(runtime_dtype)
        key_cache_updated = _kv_struct_shard(key_cache_updated).astype(runtime_dtype)
        indexs_updated = apply_logical_sharding(indexs, axes=[BATCH], **sharding_statics)

        return (
            key_cache_updated,
            value_cache_updated,
            _kv_struct_shard(jnp.logical_and(pad_mask, attention_mask)),
            self.replace(key=quantizer(key_cache_updated), value=quantizer(value_cache_updated), indexs=indexs_updated),
            masking_details,
        )

    def __repr__(self):
        try:
            return (
                self.__class__.__name__
                + f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
            )
        except AttributeError:
            return self.__class__.__name__ + f"(key={self.key}, value={self.value}, layer_index={self.layer_index})"

    @property
    def is_empty(self):
        return self.key is None

    __str__ = __repr__


@auto_pytree
class TransformerCache(BaseCache):
    views: list[TransformerCacheView | None]

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        metadata: TransformerCacheMetaData,
        partition_manager: PartitionManager,
        dtype: jnp.dtype | None = None,
        starts: jax.Array | None = None,
        quantizer: EasyQuantizer | None = None,
        mask_type_details: dict[int, AttnMaskDetail] | None = None,
    ):
        from easydel.infra.etils import EasyDeLQuantizationMethods
        from easydel.layers.quantization.quantizers import EasyQuantizer

        quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)
        if dtype is None:
            dtype = jnp.bfloat16
        with mesh:
            return cls(
                views=[
                    # i have to somehow fix my OCD
                    TransformerCacheView.init(
                        mesh=mesh,
                        dtype=dtype,
                        starts=starts,
                        metadata=metadata,
                        quantizer=quantizer,
                        layer_index=layer_index,
                        masking_details=mask_type_details.get(layer_index) if mask_type_details is not None else None,
                        partition_manager=partition_manager,
                    )
                    for layer_index in range(metadata.num_hidden_layers)
                ]
            )

    def to_pure(self):
        return (
            [[layer.key, layer.value, layer.indexs, layer.starts] for i, layer in enumerate(self.views)],
            self.views[-1].metadata,
        )

    @classmethod
    def from_pure(cls, pure, metadata):
        return cls(
            views=[
                TransformerCacheView(
                    key=layer[0],
                    value=layer[1],
                    indexs=layer[2],
                    starts=layer[3],
                    metadata=metadata,
                )
                for layer in pure
            ]
        )

    def insert_starts(self, starts, slot: int, partition_manager: PartitionManager):
        for idx in range(len(self.views)):
            view = self.views[idx]
            starts = jnp.array(starts).reshape(-1)

            self.views[idx] = self.views[idx].replace(
                starts=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.starts, starts, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                )
            )
        return self

    def insert_index(self, index, slot: int, partition_manager: PartitionManager):
        for idx in range(len(self.views)):
            view = self.views[idx]
            index = jnp.array(index).reshape(-1)
            self.views[idx] = self.views[idx].replace(
                indexs=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.indexs, index, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                )
            )
        return self

    def insert(
        self,
        other: TransformerCache,
        slot: int,
        quantizer: EasyQuantizer,
        partition_manager: PartitionManager,
    ):
        def _maybe_materialize(x: ImplicitArray | Array):
            if hasattr(x, "materialize"):
                x = x.materialize()
            return x

        for idx in range(len(self.views)):
            view = self.views[idx]
            oview = other.views[idx]

            new_val = lax.dynamic_update_slice(
                _maybe_materialize(view.value),
                _maybe_materialize(oview.value.astype(view.value.dtype)),
                (slot, 0, 0, 0),
            )
            new_key = lax.dynamic_update_slice(
                _maybe_materialize(view.key),
                _maybe_materialize(oview.key.astype(view.key.dtype)),
                (slot, 0, 0, 0),
            )

            self.views[idx] = self.views[idx].replace(
                key=quantizer(
                    apply_logical_sharding(
                        new_key,
                        axes=[BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM],
                        mode=MODE_PREFILL,
                        partition_manager=partition_manager,
                    )
                ),
                value=quantizer(
                    apply_logical_sharding(
                        new_val,
                        axes=[BATCH, KV_LENGTH, KV_HEAD, KV_HEAD_DIM],
                        mode=MODE_PREFILL,
                        partition_manager=partition_manager,
                    )
                ),
                indexs=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.indexs, oview.indexs, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                ),
                starts=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.starts, oview.starts, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                ),
                metadata=view.metadata,
            )
        return self

    @classmethod
    def init_empty(cls, num_hidden_layers):
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


@auto_pytree
class TransformerMetadata(BaseRunTimeMetadata):
    """
    holds optional metadata for attention runtime
    """

    postpadded: bool = False
    starts: jax.Array | None = None
    indexs: jax.Array | None = None
