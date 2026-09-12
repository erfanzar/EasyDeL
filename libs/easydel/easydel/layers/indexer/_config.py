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

"""Declarative configuration for the unified sparse-attention indexer."""

from __future__ import annotations

import dataclasses

__all__ = ("IndexerConfig", "IndexerKind")


class IndexerKind:
    """Selection strategies supported by :class:`~easydel.layers.indexer.SparseIndexer`.

    Attributes:
        TOKEN: Score every key token directly (GLM-MoE-DSA, DeepSeek
            Lightning-style indexers operating on raw or externally
            compressed keys).
        POOL: Group keys into pools of ``kpool_size`` consecutive valid
            tokens, score pool summaries, top-k pools expanded back to token
            indices (GLM-5-Next k-pool).
    """

    TOKEN = "token"
    POOL = "pool"


@dataclasses.dataclass(frozen=True)
class IndexerConfig:
    """Everything :class:`~easydel.layers.indexer.SparseIndexer` needs.

    The defaults reproduce GLM-MoE-DSA's per-token indexer; GLM-5-Next's
    k-pool indexer is ``kind="pool"`` + ``score_activation="relu"`` +
    ``packed_state="key_gate_valid"`` + ``stop_gradient=True`` plus the
    ``kpool_*`` fields.

    Attributes:
        kind: Selection strategy, ``"token"`` or ``"pool"``.
        index_n_heads: Indexer query head count.
        index_head_dim: Per-head indexer width.
        index_topk: Maximum selections per query (tokens for ``token``,
            *pool-expanded member slots* for ``pool`` — the module derives
            ``index_topk // kpool_size`` pool picks).
        hidden_size: Model hidden width (K-side projection input and the
            head-weight projection input).
        q_input_dim: Width of the indexer query projection input (the
            ``q_lora_rank`` residual when the family indexes from the MLA
            low-rank residual, else ``hidden_size``).
        score_activation: ``"none"`` or ``"relu"`` applied to the per-head
            scaled dot products before head reduction.
        head_reduction: How per-head scores collapse to one score per
            (query, key): ``"weighted"`` = learned per-token head weights
            (``weights_proj(hidden) * n_heads**-0.5``), ``"uniform"`` = plain
            sum over heads.
        query_source: ``"q_lora"`` — ``wq_b`` consumes the MLA low-rank
            residual passed as ``q_resid``; ``"hidden"`` — ``wq_b`` consumes
            ``hidden_states`` and ``q_resid`` is ignored.
        rope_style: ``"split_half"`` (NeoX rotate-half), ``"interleaved"``
            (adjacent-channel), or ``"none"`` (NoPE indexers). Cos/sin are
            truncated to ``rope_dim`` before application.
        rope_dim: Rotary width to truncate cos/sin to; ``None`` uses the full
            table width.
        norm_eps: LayerNorm epsilon for the index-key norm.
        packed_state: Per-token cache layout appended to (and expected from)
            the cache view: ``"keys"`` — normalized keys only
            ``[B, T, index_head_dim]``; ``"key_gate_valid"`` — raw key, gate
            projection scores and validity flag ``[B, T, 2*dim + 1]`` (needed
            when pooling re-weights cached tokens); ``"none"`` — stateless.
        stop_gradient: Wrap the whole selection in ``jax.lax.stop_gradient``
            (HF reference wraps the indexer in ``@torch.no_grad``).
        kpool_size: Tokens per pool (``pool`` kind only).
        select_tail: Always append the current incomplete tail pool's raw
            token indices to the selection (``pool`` kind only).
        initializer_range: Std of the normal initializer for projections.
    """

    kind: str = IndexerKind.TOKEN
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    hidden_size: int = 4096
    q_input_dim: int = 4096
    score_activation: str = "none"
    head_reduction: str = "weighted"
    query_source: str = "q_lora"
    rope_style: str = "split_half"
    rope_dim: int | None = None
    norm_eps: float = 1e-6
    packed_state: str = "keys"
    stop_gradient: bool = False
    kpool_size: int = 1
    select_tail: bool = False
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.kind not in (IndexerKind.TOKEN, IndexerKind.POOL):
            raise ValueError(f"IndexerConfig.kind must be 'token' or 'pool', got {self.kind!r}.")
        if self.score_activation not in ("none", "relu"):
            raise ValueError(f"score_activation must be 'none' or 'relu', got {self.score_activation!r}.")
        if self.head_reduction not in ("weighted", "uniform"):
            raise ValueError(f"head_reduction must be 'weighted' or 'uniform', got {self.head_reduction!r}.")
        if self.query_source not in ("q_lora", "hidden"):
            raise ValueError(f"query_source must be 'q_lora' or 'hidden', got {self.query_source!r}.")
        if self.rope_style not in ("split_half", "interleaved", "none"):
            raise ValueError(f"rope_style must be 'split_half', 'interleaved' or 'none', got {self.rope_style!r}.")
        if self.packed_state not in ("keys", "key_gate_valid", "none"):
            raise ValueError(f"packed_state must be 'keys', 'key_gate_valid' or 'none', got {self.packed_state!r}.")
        if self.kind == IndexerKind.POOL:
            if self.kpool_size < 1:
                raise ValueError(f"kpool_size must be >= 1 for pool indexers, got {self.kpool_size}.")
            if self.index_topk % self.kpool_size:
                raise ValueError(
                    f"index_topk ({self.index_topk}) must be divisible by kpool_size ({self.kpool_size})."
                )
            # Without the tail pool, queries seeing fewer than kpool visible
            # tokens (the first kpool-1 rows of every prefill, and all rows
            # when kv < kpool) select nothing -> all-masked attention.
            if not self.select_tail:
                raise ValueError(
                    "kind='pool' requires select_tail=True: without the tail pool, queries with fewer"
                    " than kpool_size visible tokens select nothing (all-masked attention rows)."
                )
            # Pooling reads the gate/valid channels of the packed state; a
            # keys-only or stateless layout structurally cannot pool.
            if self.packed_state != "key_gate_valid":
                raise ValueError(
                    f"kind='pool' requires packed_state='key_gate_valid', got {self.packed_state!r}."
                )
        if self.rope_dim is not None:
            if self.rope_dim > self.index_head_dim:
                raise ValueError(
                    f"rope_dim ({self.rope_dim}) must be <= index_head_dim ({self.index_head_dim})."
                )
            if self.rope_dim % 2:
                raise ValueError(f"rope_dim must be even for split-half/interleaved RoPE, got {self.rope_dim}.")
        if self.index_n_heads < 1 or self.index_head_dim < 1 or self.index_topk < 1:
            raise ValueError("index_n_heads, index_head_dim and index_topk must be positive.")

    def with_changes(self, **kwargs) -> "IndexerConfig":
        """Return a copy with the given fields replaced.

        Args:
            **kwargs: Field names and replacement values.

        Returns:
            A new :class:`IndexerConfig` (re-validated).
        """
        return dataclasses.replace(self, **kwargs)
