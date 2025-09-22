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

from eformer import common_types
from eformer.escale import PartitionManager, apply_logical_sharding
from eformer.pytree import auto_pytree
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array as JAXArray
from jaxtyping import Int

from easydel.layers.caching.linear.cache import LinearAttnCacheMetaData, LinearAttnCacheView

from .._abstracts import BaseCache, BaseCacheView
from .._utils import AttnMaskDetail
from ..transformer.cache import TransformerCacheMetaData, TransformerCacheView

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
class FlexAttentionCache(BaseCache):
    """Flexible cache routing by attention type (FULL, SLIDING -> Transformer, LINEAR -> LinearAttn).

    Pass a mapping attn_type_per_layer (layer_idx -> AttnMaskType value).
    """

    views: list[BaseCacheView | None]
    attn_type_per_layer: dict[int, "Enum"]

    @classmethod
    def init_cache(
        cls,
        mesh: Mesh,
        partition_manager: PartitionManager,
        attn_type_per_layer: dict[int, "Enum"],  # layer_idx -> AttnMaskType.{FULL,SLIDING,LINEAR}
        transformer_metadata: TransformerCacheMetaData | None = None,
        linear_metadata: LinearAttnCacheMetaData | None = None,
        mask_type_details: dict[int, AttnMaskDetail] | None = None,  # for SLIDING/FULL layers
        dtype: jnp.dtype | None = None,
        starts: Int[JAXArray, "batch"] | None = None,  # noqa: F821
        quantizer: "EasyQuantizer" | None = None,
    ) -> FlexAttentionCache:
        from easydel.infra.etils import EasyDeLQuantizationMethods
        from easydel.infra.utils import AttnMaskType
        from easydel.layers.quantization.quantizers import EasyQuantizer

        if not attn_type_per_layer:
            raise ValueError("attn_type_per_layer cannot be empty.")

        # Resolve layer count
        total_layers = max(attn_type_per_layer.keys()) + 1
        if transformer_metadata is not None and transformer_metadata.num_hidden_layers != total_layers:
            raise ValueError("transformer_metadata.num_hidden_layers must match attn_type_per_layer.")
        if linear_metadata is not None and linear_metadata.num_hidden_layers != total_layers:
            raise ValueError("linear_metadata.num_hidden_layers must match attn_type_per_layer.")

        dtype = dtype or jnp.bfloat16
        quantizer = quantizer or EasyQuantizer(EasyDeLQuantizationMethods.NONE)

        # Optional sanity checks if both given
        if transformer_metadata and linear_metadata:
            if transformer_metadata.batch_size != linear_metadata.batch_size:
                raise ValueError("Batch size mismatch between transformer and linear metadata.")

        views: list[BaseCacheView | None] = []
        with mesh:
            for layer_idx in range(total_layers):
                atype = attn_type_per_layer[layer_idx]
                if atype == AttnMaskType.LINEAR:
                    if linear_metadata is None:
                        raise ValueError("linear_metadata must be provided for LINEAR layers.")
                    views.append(
                        LinearAttnCacheView.init(
                            mesh=mesh,
                            dtype=dtype,
                            metadata=linear_metadata,
                            quantizer=quantizer,
                            partition_manager=partition_manager,
                            starts=starts,
                            layer_index=layer_idx,
                        )
                    )
                else:
                    if transformer_metadata is None:
                        raise ValueError("transformer_metadata must be provided for FULL/SLIDING layers.")
                    views.append(
                        TransformerCacheView.init(
                            mesh=mesh,
                            dtype=dtype,
                            metadata=transformer_metadata,
                            quantizer=quantizer,
                            partition_manager=partition_manager,
                            starts=starts,
                            layer_index=layer_idx,
                            masking_details=mask_type_details.get(layer_idx) if mask_type_details else None,
                        )
                    )

        return cls(views=views, attn_type_per_layer=dict(attn_type_per_layer))

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        """Return seq length of KV cache from any transformer layer (0 if none)."""
        tl_idx = None
        if layer_idx is not None:
            view = self.views[layer_idx]

            if isinstance(view, TransformerCacheView):
                tl_idx = layer_idx
        if tl_idx is None:
            for i, v in enumerate(self.views):
                if isinstance(v, TransformerCacheView):
                    tl_idx = i
                    break
        if tl_idx is None:
            return 0
        v = self.views[tl_idx]
        try:
            return int(jnp.max(v.indexs))
        except Exception:
            return 0

    def get_mask_sizes(self, cache_position: JAXArray, layer_idx: int) -> tuple[int, int]:
        """Return (kv_length, kv_offset) like HF's API for transformer layers."""
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = int(query_length + past_seen_tokens)
        kv_offset = 0
        return kv_length, kv_offset

    @property
    def has_previous_state(self) -> bool:
        """True if at least one LINEAR layer has a non-empty conv_state."""
        for v in self.views[::-1]:
            if isinstance(v, LinearAttnCacheView):
                if hasattr(v.conv_state, "materialize"):
                    arr = v.conv_state.materialize()
                else:
                    arr = v.conv_state
                return bool(jnp.any(arr != 0.0))
        return False

    def reorder_cache(self, beam_idx: Int[JAXArray, "new_batch"]) -> None:  # noqa: F821
        """Reorder batch dimension according to beam indices for all views."""
        beam_idx = jnp.asarray(beam_idx)

        def reorder_first_dim(x):
            if hasattr(x, "materialize"):
                x = x.materialize()
            return x[beam_idx]

        for i, v in enumerate(self.views):
            if isinstance(v, TransformerCacheView):
                self.views[i] = v.replace(
                    key=reorder_first_dim(v.key),
                    value=reorder_first_dim(v.value),
                    indexs=reorder_first_dim(v.indexs),
                    starts=reorder_first_dim(v.starts),
                )
            elif isinstance(v, LinearAttnCacheView):
                self.views[i] = v.replace(
                    conv_state=reorder_first_dim(v.conv_state),
                    recurrent_state=reorder_first_dim(v.recurrent_state),
                    indexs=reorder_first_dim(v.indexs),
                    starts=reorder_first_dim(v.starts),
                )

    def insert_starts(
        self,
        starts: Int[JAXArray, "..."],
        slot: int,
        partition_manager: PartitionManager,
    ) -> "FlexAttentionCache":
        for i, view in enumerate(self.views):
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
    ) -> "FlexAttentionCache":
        for i, view in enumerate(self.views):
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
        other: "FlexAttentionCache",
        slot: int,
        quantizer: "EasyQuantizer",
        partition_manager: PartitionManager,
    ) -> "FlexAttentionCache":
        for i, (view, oview) in enumerate(zip(self.views, other.views, strict=False)):
            if isinstance(view, TransformerCacheView) and isinstance(oview, TransformerCacheView):

                def _mat(x):
                    return x.materialize() if hasattr(x, "materialize") else x

                new_val = lax.dynamic_update_slice(
                    _mat(view.value), _mat(oview.value.astype(view.value.dtype)), (slot, 0, 0, 0)
                )
                new_key = lax.dynamic_update_slice(
                    _mat(view.key), _mat(oview.key.astype(view.key.dtype)), (slot, 0, 0, 0)
                )
                self.views[i] = view.replace(
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
                )
            elif isinstance(view, LinearAttnCacheView) and isinstance(oview, LinearAttnCacheView):

                def _mat(x):
                    return x.materialize() if hasattr(x, "materialize") else x

                new_conv = lax.dynamic_update_slice(
                    _mat(view.conv_state), _mat(oview.conv_state.astype(view.conv_state.dtype)), (slot, 0, 0)
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
                )
            else:
                raise TypeError(f"Mismatched view types at layer {i}: {type(view).__name__} vs {type(oview).__name__}")
        return self

    @classmethod
    def init_empty(cls, num_hidden_layers: int) -> "FlexAttentionCache":
        return cls(views=[None for _ in range(num_hidden_layers)], attn_type_per_layer={})

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  " + "\n  ".join(str(view) for view in self.views) + "\n)"

    __str__ = __repr__
