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
from eformer import escale as es
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView, BaseRunTimeMetadata

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object


@auto_pytree
class LightningCacheMetaData(BaseCacheMetadata):
    """Metadata for transformer cache configuration."""

    partition_axis: es.PartitionAxis
    batch_size: int | None
    num_heads: int | None
    head_dim: int | None
    key_heads: int | None
    value_heads: int | None
    key_dim: int | None
    value_dim: int | None

    @classmethod
    def create(
        cls,
        partition_axis: es.PartitionAxis,
        batch_size: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        key_heads: int | None = None,
        value_heads: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
    ) -> LightningCacheMetaData:
        """
        Create a LightningCacheMetaData instance with validation.
        Returns:
            LightningCacheMetaData instance

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        return cls(
            partition_axis=partition_axis,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            key_heads=key_heads,
            value_heads=value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
        )


@auto_pytree
class LightningCacheView(BaseCacheView):
    key_value: cx.Array | ImplicitArray
    metadata: LightningCacheMetaData
    layer_index: int | None = None

    @classmethod
    def init(cls, metadata: LightningCacheMetaData, layer_index: int | None = None):
        return cls(
            key_value=None,
            metadata=metadata,
            layer_index=layer_index,
        )

    @jax.named_scope("easydel-lightning-cacheview-concatenate-to-cache")
    def concatenate_to_cache(
        self,
        query: cx.Array,
        key: cx.Array,
        value: cx.Array,
        attention_mask: cx.Array,
        kv_sharding: PartitionSpec,
        quantizer: EasyQuantizer,
        causal_mask: cx.Array | bool | None = None,
        token_type_ids: cx.Array | None = None,
    ) -> tuple[cx.Array, cx.Array, cx.Array]:
        """
        Updates the KV cache with new key/value states and adjusts the attention mask.

        Internal helper function used when KV caching is enabled.

        Args:
            query (Array): Current query states.
            key (Array): Current key states.
            value (Array): Current value states.
            attention_mask (Array): Base attention mask.
            causal_mask (tp.Optional[Array], optional): Causal mask. Defaults to None.
            token_type_ids (tp.Optional[Array], optional): Token type IDs for segment-based masking.
                                                            Defaults to None.

        Returns:
            tp.Tuple[Array, Array, Array]:
                - Updated key cache tensor.
                - Updated value cache tensor.
                - Updated attention mask reflecting the cached sequence length.
        """
        num_updated_cache_vectors = query.shape[1]
        end_index = self.index[0]

        *batch_dims, max_length, num_heads, depth_per_head = self.value.shape

        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if causal_mask is not None:
            if hasattr(causal_mask, "value"):
                causal_mask = causal_mask.value
            causal_mask = jax.lax.dynamic_slice(
                causal_mask,
                (0, 0, end_index, 0),
                (1, 1, num_updated_cache_vectors, max_length),
            )
            if token_type_ids is not None and num_updated_cache_vectors != 1:
                token_type_mask = jnp.equal(
                    jnp.expand_dims(token_type_ids, 2),
                    jnp.expand_dims(token_type_ids, 1),
                )

                token_type_mask = jnp.where(token_type_ids == 0, False, token_type_mask)
                token_type_mask = jnp.expand_dims(token_type_mask, 1)
                sequence_length = token_type_ids.shape[1]
                masked_portion = jnp.logical_or(
                    token_type_mask[:, :, :num_updated_cache_vectors, :],
                    causal_mask[:, :, :, :sequence_length],
                )
                causal_mask = causal_mask.at[:, :, :, :sequence_length].set(masked_portion)

            causal_mask = jnp.broadcast_to(causal_mask, (query.shape[0], *causal_mask.shape[1:]))

            attention_mask = jnp.broadcast_to(attention_mask, causal_mask.shape)
            attention_mask = jnp.logical_and(attention_mask, causal_mask)

        slice_indices = (0, end_index % self.value.shape[1], 0, 0)

        value_cache = es.with_sharding_constraint(
            jax.lax.dynamic_update_slice(
                self.value,
                value.astype(self.value.dtype),
                slice_indices,
            ),
            sharding=kv_sharding,
        )
        key_cache = es.with_sharding_constraint(
            jax.lax.dynamic_update_slice(
                self.key,
                key.astype(self.key.dtype),
                slice_indices,
            ),
            sharding=kv_sharding,
        )
        pad_mask = jnp.broadcast_to(
            jnp.arange(max_length) < end_index + num_updated_cache_vectors,
            (*tuple(batch_dims), 1, num_updated_cache_vectors, max_length),
        )
        attention_mask = jnp.logical_and(pad_mask, attention_mask)

        self.key = quantizer(key_cache)
        self.value = quantizer(value_cache)

        self.index = self.index + num_updated_cache_vectors
        return key_cache, value_cache, attention_mask

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(key={self.key.shape}, value={self.value.shape}, layer_index={self.layer_index})"
        )

    __str__ = __repr__


@auto_pytree
class LightningCache(BaseCache):
    views: list[LightningCacheView | None]

    @classmethod
    def init_cache(
        cls,
        num_hidden_layers: int,
        metadata: LightningCacheMetaData,
    ):
        return cls(
            views=[
                LightningCacheView.init(metadata=metadata, layer_index=layer_index)
                for layer_index in range(num_hidden_layers)
            ]
        )

    @classmethod
    def init_empty(cls, num_hidden_layers):
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__


class LightningMetadata(BaseRunTimeMetadata): ...
