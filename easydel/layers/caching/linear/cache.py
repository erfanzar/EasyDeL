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
from eformer import common_types
from eformer.escale import PartitionManager, apply_logical_sharding
from eformer.jaximus import ImplicitArray
from eformer.pytree import auto_pytree, field
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding as Ns
from jaxtyping import Array as JAXArray
from jaxtyping import Float, Int

from .._abstracts import BaseCache, BaseCacheMetadata, BaseCacheView

if tp.TYPE_CHECKING:
    from easydel.layers.quantization.quantizers import EasyQuantizer
else:
    EasyQuantizer = object


# Reuse your existing constants/axes from common_types
NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
KV_LENGTH = common_types.KV_LENGTH
KV_HEAD = common_types.KV_HEAD
KV_HEAD_DIM = common_types.KV_HEAD_DIM
MODE_PREFILL = common_types.MODE_PREFILL


@auto_pytree
class LinearAttnCacheMetaData(BaseCacheMetadata):
    """Metadata for linear attention (Gated Delta) caching.

    Stores static config for conv_state and recurrent_state buffers.

    Attributes:
        batch_size: Batch size.
        sequence_length: Max seq length (used for indices/tracking only).
        num_hidden_layers: Number of decoder layers.
        pad_token_id: Padding id.
        num_k_heads: Number of key heads in linear block.
        num_v_heads: Number of value heads in linear block.
        k_head_dim: Key head dim for linear block.
        v_head_dim: Value head dim for linear block.
        conv_state_len: Length of the conv state buffer (e.g., kernel_size or library-specific state length).
        conv_channels: Channels for conv_state buffer. Commonly equals 2 * (num_k_heads * k_head_dim) + (num_v_heads * v_head_dim).
    """

    batch_size: int
    sequence_length: int
    num_hidden_layers: int
    pad_token_id: int

    num_k_heads: int
    num_v_heads: int
    k_head_dim: int
    v_head_dim: int

    conv_state_len: int
    conv_channels: int

    # flags for API parity (usually not used for linear attention masking)
    update_causal_mask: bool = False
    create_attention_bias: bool = False

    @classmethod
    def create(
        cls,
        batch_size: int,
        sequence_length: int,
        num_hidden_layers: int,
        pad_token_id: int,
        num_k_heads: int,
        num_v_heads: int,
        k_head_dim: int,
        v_head_dim: int,
        conv_state_len: int,
        conv_channels: int | None = None,
        update_causal_mask: bool = False,
        create_attention_bias: bool = False,
    ) -> "LinearAttnCacheMetaData":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if num_k_heads <= 0 or num_v_heads <= 0:
            raise ValueError("num_k_heads and num_v_heads must be positive")
        if k_head_dim <= 0 or v_head_dim <= 0:
            raise ValueError("k_head_dim and v_head_dim must be positive")
        if conv_state_len <= 0:
            raise ValueError("conv_state_len must be positive")

        if conv_channels is None:
            key_dim = num_k_heads * k_head_dim
            value_dim = num_v_heads * v_head_dim
            conv_channels = 2 * key_dim + value_dim

        return cls(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            conv_state_len=conv_state_len,
            conv_channels=conv_channels,
            update_causal_mask=update_causal_mask,
            create_attention_bias=create_attention_bias,
        )


@auto_pytree(frozen=False)
class LinearAttnCacheView(BaseCacheView):
    """Single-layer view for linear attention (Gated Delta) states.

    Stores:
      - conv_state: [B, C, Ls]  (channels-first causal conv state buffer)
      - recurrent_state: [B, k_dim, v_dim]  (per-layer recurrent matrix state)
      - indexs/starts: [B] tracking like standard cache (optional but kept for consistency)

    Note: We do not compute the linear attn updates here; the layer computes them and writes them back into the cache.
    """

    conv_state: Float[JAXArray, "batch conv_channels conv_state_len"] | ImplicitArray
    recurrent_state: Float[JAXArray, "batch k_dim v_dim"] | ImplicitArray
    indexs: Int[JAXArray, "batch"] | ImplicitArray  # noqa: F821
    starts: Int[JAXArray, "batch"] | ImplicitArray  # noqa: F821

    metadata: LinearAttnCacheMetaData

    maximum_sequence_length: int = field(pytree_node=False)

    layer_index: int | None = None

    @classmethod
    def init(
        cls,
        mesh: Mesh,
        dtype: jnp.dtype,
        metadata: LinearAttnCacheMetaData,
        quantizer: "EasyQuantizer",
        partition_manager: PartitionManager,
        starts: Int[JAXArray, "batch"] | None = None,  # noqa: F821
        layer_index: int | None = None,
    ) -> "LinearAttnCacheView":
        with jax.named_scope("easydel-linearattn-cacheview-init"):
            bsz = metadata.batch_size
            c = metadata.conv_channels
            ls = metadata.conv_state_len
            kdim = metadata.k_head_dim
            vdim = metadata.v_head_dim

            # Sharding axes (reusing KV_HEAD/KV_LENGTH for channels/length)
            conv_axes = [BATCH, KV_HEAD, KV_LENGTH]  # [B, C, Ls]
            rec_axes = [BATCH, KV_HEAD, KV_HEAD_DIM]  # [B, k_dim, v_dim]
            idx_axes = [BATCH]

            conv_shape = (bsz, c, ls)
            rec_shape = (bsz, kdim, vdim)

            conv_sharding = Ns(mesh, partition_manager.resolve(axes=conv_axes, mode=MODE_PREFILL, shape=conv_shape))
            rec_sharding = Ns(mesh, partition_manager.resolve(axes=rec_axes, mode=MODE_PREFILL, shape=rec_shape))
            idx_sharding = Ns(mesh, partition_manager.resolve(axes=idx_axes, mode=MODE_PREFILL, shape=(bsz,)))

            if starts is None:
                starts = jnp.zeros((bsz,), dtype=jnp.int32)
            starts = apply_logical_sharding(
                starts, axes=idx_axes, mode=MODE_PREFILL, partition_manager=partition_manager
            )

            return cls(
                conv_state=quantizer(jnp.zeros(conv_shape, dtype=dtype, device=conv_sharding)),
                recurrent_state=quantizer(jnp.zeros(rec_shape, dtype=dtype, device=rec_sharding)),
                indexs=jnp.zeros((bsz,), dtype=jnp.int32, device=idx_sharding),
                starts=starts,
                metadata=metadata,
                layer_index=layer_index,
                maximum_sequence_length=metadata.sequence_length,
            )

    @jax.named_scope("easydel-linear-cacheview-concatenate-to-cache")
    def concatenate_to_cache(
        self,
        new_conv_state: Float[JAXArray, "batch conv_channels conv_state_len"] | None,
        new_recurrent_state: Float[JAXArray, "batch k_dim v_dim"] | None,
        partition_manager: PartitionManager,
        quantizer: "EasyQuantizer",
        tokens_appended: Int[JAXArray, "batch"] | int | None = None,  # <- allow per-batch  # noqa: F821
    ) -> "LinearAttnCacheView":
        sharding_statics = dict(mode=MODE_PREFILL, partition_manager=partition_manager)

        def _maybe_mat(x):
            return x.materialize() if hasattr(x, "materialize") else x

        conv = _maybe_mat(self.conv_state)
        rec = _maybe_mat(self.recurrent_state)

        if new_conv_state is not None:
            conv = apply_logical_sharding(new_conv_state, axes=[BATCH, KV_HEAD, KV_LENGTH], **sharding_statics).astype(
                conv.dtype
            )

        if new_recurrent_state is not None:
            rec = apply_logical_sharding(
                new_recurrent_state, axes=[BATCH, KV_HEAD, KV_HEAD_DIM], **sharding_statics
            ).astype(rec.dtype)

        indexs = self.indexs
        if tokens_appended is not None:
            if jnp.ndim(tokens_appended) == 0:  # scalar
                indexs = indexs + jnp.asarray(tokens_appended, dtype=indexs.dtype)
            else:  # [batch]
                tokens_appended = apply_logical_sharding(
                    tokens_appended.astype(indexs.dtype), axes=[BATCH], **sharding_statics
                )
                indexs = indexs + tokens_appended

        return self.replace(
            conv_state=quantizer(conv),
            recurrent_state=quantizer(rec),
            indexs=apply_logical_sharding(indexs, axes=[BATCH], **sharding_statics),
        )

    def __repr__(self):
        try:
            return (
                self.__class__.__name__
                + f"(conv_state={self.conv_state.shape}, recurrent_state={self.recurrent_state.shape}, "
                + f"layer_index={self.layer_index})"
            )
        except AttributeError:
            return (
                self.__class__.__name__
                + f"(conv_state={self.conv_state}, recurrent_state={self.recurrent_state}, layer_index={self.layer_index})"
            )

    @property
    def is_empty(self) -> bool:
        return self.conv_state is None and self.recurrent_state is None

    __str__ = __repr__


@auto_pytree
class LinearAttnCache(BaseCache):
    """Multi-layer container for linear attention states."""

    views: list[LinearAttnCacheView | None]

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        metadata: LinearAttnCacheMetaData,
        partition_manager: PartitionManager,
        dtype: jnp.dtype | None = None,
        starts: Int[JAXArray, "batch"] | None = None,  # noqa: F821
        quantizer: "EasyQuantizer" | None = None,
    ) -> "LinearAttnCache":
        from easydel.infra.etils import EasyDeLQuantizationMethods
        from easydel.layers.quantization.quantizers import EasyQuantizer

        quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)
        dtype = dtype or jnp.bfloat16
        with mesh:
            return cls(
                views=[
                    LinearAttnCacheView.init(
                        mesh=mesh,
                        dtype=dtype,
                        metadata=metadata,
                        quantizer=quantizer,
                        partition_manager=partition_manager,
                        starts=starts,
                        layer_index=layer_idx,
                    )
                    for layer_idx in range(metadata.num_hidden_layers)
                ]
            )

    def insert_starts(
        self, starts: Int[JAXArray, "..."], slot: int, partition_manager: PartitionManager
    ) -> "LinearAttnCache":
        for i in range(len(self.views)):
            view = self.views[i]
            starts_arr = jnp.array(starts).reshape(-1)
            self.views[i] = view.replace(
                starts=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.starts, starts_arr, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                )
            )
        return self

    def insert_index(
        self, index: Int[JAXArray, "..."], slot: int, partition_manager: PartitionManager
    ) -> "LinearAttnCache":
        for i in range(len(self.views)):
            view = self.views[i]
            index_arr = jnp.array(index).reshape(-1)
            self.views[i] = view.replace(
                indexs=apply_logical_sharding(
                    lax.dynamic_update_slice_in_dim(view.indexs, index_arr, slot, 0),
                    axes=[BATCH],
                    mode=MODE_PREFILL,
                    partition_manager=partition_manager,
                )
            )
        return self

    def insert(
        self,
        other: "LinearAttnCache",
        slot: int,
        quantizer: "EasyQuantizer",
        partition_manager: PartitionManager,
    ) -> "LinearAttnCache":
        def _mat(x):
            return x.materialize() if hasattr(x, "materialize") else x

        for i in range(len(self.views)):
            view = self.views[i]
            oview = other.views[i]

            new_conv = lax.dynamic_update_slice(
                _mat(view.conv_state),
                _mat(oview.conv_state.astype(view.conv_state.dtype)),
                (slot, 0, 0),
            )
            new_rec = lax.dynamic_update_slice(
                _mat(view.recurrent_state),
                _mat(oview.recurrent_state.astype(view.recurrent_state.dtype)),
                (slot, 0, 0),
            )

            self.views[i] = view.replace(
                conv_state=quantizer(
                    apply_logical_sharding(
                        new_conv,
                        axes=[BATCH, KV_HEAD, KV_LENGTH],
                        mode=MODE_PREFILL,
                        partition_manager=partition_manager,
                    )
                ),
                recurrent_state=quantizer(
                    apply_logical_sharding(
                        new_rec,
                        axes=[BATCH, KV_HEAD, KV_HEAD_DIM],
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
    def init_empty(cls, num_hidden_layers: int) -> "LinearAttnCache":
        return cls(views=[None for _ in range(num_hidden_layers)])

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__
