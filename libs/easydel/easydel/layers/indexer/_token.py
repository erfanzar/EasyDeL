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

"""Operation-backed weighted token indexer with explicit key-cache ownership."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import spectrax as spx
from jax import numpy as jnp

from easydel.layers.linears import ColumnParallelLinear

from ._selection import BaseIndexer, IndexerSelection, SelectionSpec

if tp.TYPE_CHECKING:
    from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOutput


@dataclass(frozen=True)
class TokenIndexerConfig:
    """Dimensions and RoPE policy for weighted per-token operation dispatch."""

    hidden_size: int
    q_input_dim: int
    num_heads: int
    head_dim: int
    topk: int
    rope_dim: int
    rope_interleave: bool = False
    initializer_range: float = 0.02


class TokenIndexer(BaseIndexer):
    """Weighted token scores dispatched through the GLM DSA operation.

    This strategy deliberately retains that operation's mask, rotary and
    cache semantics rather than substituting the pooled indexer's scorer.
    The ``kernels_proj`` checkpoint name and HF ``weights_proj`` reform rule
    are part of this implementation's public parameter layout.
    """

    selection_spec = SelectionSpec(candidate_unit="token", output_unit="token")

    def __init__(
        self,
        config: TokenIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        base_config,
        norm_factory: tp.Callable[[], spx.Module],
    ):
        """Build checkpoint-compatible projections and operation metadata.

        ``base_config`` supplies the standard EasyDeL runtime/sharding policy;
        ``norm_factory`` supplies the key norm with its exact family layout.
        """
        # Operations import shared layers too; defer until construction so
        # exporting this class from the layers root cannot form an import cycle.
        from easydel.operations import OperationMetadata
        from easydel.operations.kernels.glm_moe_dsa_indexer import GlmMoeDsaIndexerOp

        self.indexer_config = config
        self.dtype, self.param_dtype, self.precision = dtype, param_dtype, precision
        self.index_n_heads, self.index_head_dim, self.index_topk = config.num_heads, config.head_dim, config.topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.indexer_rope_interleave = config.rope_interleave
        linear = partial(
            ColumnParallelLinear,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        self.wq_b = linear(config.q_input_dim, config.num_heads * config.head_dim)
        self.wk = linear(config.hidden_size, config.head_dim)
        self.k_norm = norm_factory()
        self.kernels_proj = linear(config.hidden_size, config.num_heads)
        self.indexer_op = GlmMoeDsaIndexerOp(
            OperationMetadata(runtime_dtype=dtype, runtime_softmax_dtype=jnp.float32, base_config=base_config)
        )

    @property
    def reform_param(self):
        """Map the HF head-weight projection to the historical runtime name."""
        return {
            "weights_proj.weight$": {
                "splits": [{"name": "kernels_proj.weight", "spliter": lambda x: x.swapaxes(-1, -2)}],
                "inverse_spliter": lambda x: x.swapaxes(-1, -2),
            },
        }

    @staticmethod
    def selection_from_output(output: GlmMoeDsaIndexerOutput) -> IndexerSelection:
        """View an already-computed operation output without updating its cache again."""
        return IndexerSelection(output.topk_indices)

    def forward(
        self,
        hidden_states,
        q_resid,
        position_ids,
        frequencies=None,
        attention_mask=None,
        cached_keys=None,
        use_cache=False,
    ) -> GlmMoeDsaIndexerOutput:
        """Return legacy operation output (token IDs and optional updated key cache).

        Multi-token prefill resets the passed cache; single-token decode
        appends to it. Rotary transformation and mask interpretation remain
        owned by the registered operation.
        """
        q_input = q_resid if q_resid is not None else hidden_states
        query_states = self.wq_b(q_input).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.index_n_heads, self.index_head_dim
        )
        key_states = self.k_norm(self.wk(hidden_states))
        head_weights = self.kernels_proj(hidden_states).astype(jnp.float32) * (self.index_n_heads**-0.5)
        return self.indexer_op(
            query_states=query_states,
            key_states=key_states,
            head_weights=head_weights,
            position_ids=position_ids,
            qk_rope_head_dim=self.indexer_config.rope_dim,
            index_topk=self.index_topk,
            softmax_scale=self.softmax_scale,
            frequencies=frequencies,
            attention_mask=attention_mask,
            cached_keys=cached_keys,
            use_cache=use_cache,
            reset_cache=hidden_states.shape[1] > 1,
            indexer_rope_interleave=self.indexer_rope_interleave,
            precision=self.precision,
        )
