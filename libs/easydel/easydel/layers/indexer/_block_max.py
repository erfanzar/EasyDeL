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

"""Per-group token-score max pooling with a fixed block-selection budget."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from functools import partial

import jax
import spectrax as spx
from jax import numpy as jnp

from easydel.layers.linears import ColumnParallelLinear

from ._selection import BaseIndexer, IndexerSelection, SelectionSpec


@dataclass(frozen=True)
class BlockMaxIndexerConfig:
    """Projection dimensions and stateless block-selection policy.

    Local blocks receive infinite scores *inside* ``topk_blocks``; they are
    not appended to the selected budget. Each indexer head selects separately.
    """

    hidden_size: int
    num_heads: int
    head_dim: int
    block_size: int
    topk_blocks: int
    local_blocks: int = 0
    initializer_range: float = 0.02

    def __post_init__(self):
        if min(self.hidden_size, self.num_heads, self.head_dim, self.block_size, self.topk_blocks) < 1:
            raise ValueError("Projection dimensions, block_size and topk_blocks must be positive.")
        if self.local_blocks < 0:
            raise ValueError("local_blocks must be nonnegative.")


class BlockMaxIndexer(BaseIndexer):
    """Project, normalize, rotate and max-pool token scores independently per head.

    Norm factories and rotary callbacks allow families to retain their exact
    norm parameterization and frequency-table layout without importing models.
    Parameters live directly under ``q_proj``, ``k_proj``, ``q_norm``, ``k_norm``.
    """

    selection_spec = SelectionSpec(candidate_unit="block", output_unit="token", grouped=True)

    def __init__(
        self,
        config: BlockMaxIndexerConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        norm_factory: tp.Callable[[], spx.Module],
        gather_cos_sin: tp.Callable,
        apply_rotary: tp.Callable,
    ):
        """Build projections and family-supplied normalizers and rotary transforms."""
        self.indexer_config = config
        self.dtype, self.param_dtype, self.precision = dtype, param_dtype, precision
        self.head_dim, self.num_heads = config.head_dim, config.num_heads
        self.block_size, self.topk_blocks, self.local_blocks = config.block_size, config.topk_blocks, config.local_blocks
        self.gather_cos_sin, self.apply_rotary = gather_cos_sin, apply_rotary
        linear = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.q_proj = linear(config.hidden_size, config.num_heads * config.head_dim)
        self.k_proj = linear(config.hidden_size, config.head_dim)
        self.q_norm = norm_factory()
        self.k_norm = norm_factory()

    def select(self, hidden_states, position_ids, frequencies, *, cache_view=None) -> IndexerSelection:
        """Return causal token IDs ``[B, index_heads, Q, topk_blocks * block_size]``.

        Padding, future members and empty blocks have ID ``-1``. The indexer is
        stateless: supplying a cache is an error, not a silent cache bypass.
        """
        if cache_view is not None:
            raise NotImplementedError(
                "BlockMaxIndexer supports stateless selection only; KV-cache decoding is unsupported."
            )
        batch, seq_len, _ = hidden_states.shape
        idx_q = self.q_norm(self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim))
        idx_k = self.k_norm(self.k_proj(hidden_states).reshape(batch, seq_len, 1, self.head_dim))
        cos, sin = self.gather_cos_sin(frequencies, position_ids, width=self.head_dim)
        cos, sin = cos[:, :, None, :].astype(idx_q.dtype), sin[:, :, None, :].astype(idx_q.dtype)
        idx_q, idx_k = self.apply_rotary(idx_q, cos, sin), self.apply_rotary(idx_k, cos, sin)
        scores = jnp.einsum(
            "bqhd,bkd->bhqk", idx_q.astype(jnp.float32), idx_k[:, :, 0, :].astype(jnp.float32), precision=self.precision
        )
        future = jnp.arange(seq_len)[None, None, None, :] > position_ids[:, None, :, None]
        scores = jnp.where(future, -jnp.inf, scores)
        pad = (-seq_len) % self.block_size
        if pad:
            scores = jnp.pad(scores, ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=-jnp.inf)
        num_blocks = (seq_len + pad) // self.block_size
        block_scores = scores.reshape(batch, self.num_heads, seq_len, num_blocks, self.block_size).max(-1)
        if self.local_blocks > 0:
            local_idx = jnp.clip(
                position_ids[:, :, None] // self.block_size - jnp.arange(self.local_blocks)[None, None, :], min=0
            )
            local_idx = jnp.broadcast_to(local_idx[:, None, :, :], (batch, self.num_heads, seq_len, self.local_blocks))
            block_scores = jnp.put_along_axis(block_scores, local_idx, jnp.inf, axis=-1, inplace=False)
        selected = self.select_candidates(block_scores, min(self.topk_blocks, num_blocks), mesh_source=self.mesh_source)
        indices = selected.indices[..., None] * self.block_size + jnp.arange(self.block_size)
        valid = (selected.indices[..., None] >= 0) & (indices < seq_len)
        valid &= indices <= position_ids[:, None, :, None, None]
        indices = jnp.where(valid, indices, -1).reshape(batch, self.num_heads, seq_len, -1)
        return IndexerSelection(indices)

    def forward(self, hidden_states, position_ids, frequencies, *, cache_view=None) -> IndexerSelection:
        """Standard selection entry point; see :meth:`select`."""
        return self.select(hidden_states, position_ids, frequencies, cache_view=cache_view)

    def compute_block_bias(self, hidden_states, position_ids, frequencies, num_attention_heads, bias_dtype):
        """Expand grouped selection to GQA heads with the legacy dtype-min bias."""
        if num_attention_heads % self.num_heads:
            raise ValueError("num_attention_heads must be divisible by the indexer head count.")
        selection = self.select(hidden_states, position_ids, frequencies)
        bias = selection.to_bias(hidden_states.shape[1], bias_dtype, mask_value=jnp.finfo(bias_dtype).min)
        return jnp.repeat(bias, num_attention_heads // self.num_heads, axis=1)
