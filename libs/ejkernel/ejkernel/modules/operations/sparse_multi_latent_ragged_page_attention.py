# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Sparse (top-k selected chunks) Multi-Latent Ragged Page Attention operation module.

Same paged-cache contract as ``multi_latent_ragged_page_attention_v2`` (the current
step's latent tokens are written into ``kv_cache`` and the updated cache is returned),
plus a per-token selection ``selected [total_tokens, width]`` of chunk ids (``-1`` =
padding). Query ``t`` attends only to cached tokens ``j`` with ``j // chunk_size`` in
``selected[t]``, ``j <= pos(t)`` and ``j < kv_lens[seq(t)]``. The work per query is
proportional to ``width * chunk_size``, never to the context length.

Example:
    >>> outputs, updated_cache = sparse_multi_latent_ragged_page_attention(
    ...     queries_nope, queries_pe, keys_values, keys_pe,
    ...     kv_cache, kv_lens, block_tables, query_start_loc, distribution, selected,
    ...     chunk_size=16,
    ... )
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float, Int32

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import SparseMultiLatentRaggedPageAttentionConfig

_OP_ID = "sparse_multi_latent_ragged_page_attention"


class SparseMultiLatentRaggedPageAttention(Kernel[SparseMultiLatentRaggedPageAttentionConfig, tuple[Array, Array]]):
    """Sparse MLA ragged paged attention over per-token selected chunks."""

    version = "1"

    def __init__(self):
        """Initialize with the registry operation identifier."""
        super().__init__(op_id=_OP_ID)

    def get_impl(self, cfg: SparseMultiLatentRaggedPageAttentionConfig):
        """Resolve the implementation; Pallas on TPU when registered, else XLA.

        Args:
            cfg: Configuration carrying platform/backend preferences.

        Returns:
            The registered kernel callable.
        """
        platform = detect_platform(_OP_ID, cfg.platform, prefer_pallas=True)
        return kernel_registry.get(_OP_ID, platform=platform, backend=cfg.backend)

    def run(
        self,
        queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
        queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
        keys_values: Float[Array, "total_tokens kv_latent_dim"],
        keys_pe: Float[Array, "total_tokens qk_pe_dim"],
        kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        selected: Int32[Array, "total_tokens num_selected"],
        *,
        chunk_size: int = 1,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: SparseMultiLatentRaggedPageAttentionConfig,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads kv_latent_dim"],
        Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    ]:
        """Write new latent tokens into the paged cache and attend over selected chunks.

        Args:
            queries_nope: ``[T, H, L]`` weight-absorbed latent queries.
            queries_pe: ``[T, H, R]`` rope queries.
            keys_values: ``[T, L]`` new latent tokens.
            keys_pe: ``[T, R]`` new rope keys.
            kv_cache: Paged cache ``[P, page_size/pack, pack, L_pad + R_pad]``.
            kv_lens: ``[S]`` context lengths including the current step.
            block_tables: ``[S * pages_per_seq]`` flat page table.
            query_start_loc: ``[S + 1]`` cumulative query counts.
            distribution: ``[n_decode, n_prefill, n_total]``.
            selected: ``[T, W]`` chunk ids, ``-1``-padded, unique per row.
            chunk_size: Tokens per chunk (static).
            softmax_scale: Logit scale (default ``(L + R) ** -0.5``).
            sliding_window: Optional local window.
            logits_soft_cap: Optional tanh soft cap.
            mask_value: Accepted for API parity with v2.
            q_scale: Optional query dequantisation scale.
            k_scale: Optional key dequantisation scale.
            v_scale: Optional value dequantisation scale.
            platform: Override for platform selection.
            cfg: Kernel configuration.

        Returns:
            ``(outputs [T, H, L], updated_kv_cache)``.
        """
        if platform is not None:
            cfg = SparseMultiLatentRaggedPageAttentionConfig(
                num_queries_per_block=cfg.num_queries_per_block,
                num_chunks_per_block=cfg.num_chunks_per_block,
                vmem_limit_bytes=cfg.vmem_limit_bytes,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries_nope=queries_nope,
            queries_pe=queries_pe,
            keys_values=keys_values,
            keys_pe=keys_pe,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            selected=selected,
            chunk_size=chunk_size,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            num_queries_per_block=cfg.num_queries_per_block,
            num_chunks_per_block=cfg.num_chunks_per_block,
            vmem_limit_bytes=cfg.vmem_limit_bytes,
        )

    def heuristic_cfg(
        self,
        inv: Invocation[SparseMultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ) -> SparseMultiLatentRaggedPageAttentionConfig:
        """Default configuration: 8 query rows and 128 selected tokens per KV step."""
        chunk_size = max(1, int(inv.kwargs.get("chunk_size", 1) or 1))
        return SparseMultiLatentRaggedPageAttentionConfig(
            num_queries_per_block=8,
            num_chunks_per_block=max(1, 128 // chunk_size),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(
        self,
        inv: Invocation[SparseMultiLatentRaggedPageAttentionConfig, tuple[Array, Array]],
    ):
        """Autotuning candidates over the query-block size."""
        chunk_size = max(1, int(inv.kwargs.get("chunk_size", 1) or 1))
        return [
            SparseMultiLatentRaggedPageAttentionConfig(
                num_queries_per_block=q_block,
                num_chunks_per_block=max(1, 128 // chunk_size),
                platform="auto",
                backend="any",
            )
            for q_block in (8, 16)
        ]


_sparse_mlrpa_executor: Executor[SparseMultiLatentRaggedPageAttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=40),
        persistent=PersistentCache(_OP_ID),
    )
)


def sparse_multi_latent_ragged_page_attention(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    selected: Int32[Array, "total_tokens num_selected"],
    /,
    *,
    chunk_size: int = 1,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: SparseMultiLatentRaggedPageAttentionConfig | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Sparse MLA ragged paged attention with automatic backend/config selection.

    See :class:`SparseMultiLatentRaggedPageAttention.run` for arguments.

    Returns:
        ``(outputs [T, H, L], updated_kv_cache)``.
    """
    return _sparse_mlrpa_executor(
        SparseMultiLatentRaggedPageAttention(),
        queries_nope=queries_nope,
        queries_pe=queries_pe,
        keys_values=keys_values,
        keys_pe=keys_pe,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        selected=selected,
        chunk_size=chunk_size,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        platform=platform,
        _cfg=cfg,
    )


__all__ = (
    "SparseMultiLatentRaggedPageAttention",
    "sparse_multi_latent_ragged_page_attention",
)
