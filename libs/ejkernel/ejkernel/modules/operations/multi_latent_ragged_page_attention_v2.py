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

"""Multi-Latent Ragged Page Attention v2 (MLRPA-v2) operation module.

This module extends the v1 MLRPA kernel with per-case block-size tuning:
``num_kv_pages_per_block`` and ``num_queries_per_block`` may each be an
integer (broadcast to all three cases) or a ``(decode, prefill, mixed)``
tuple so the Pallas grid can be specialised for each request type that
appears in a single continuous batch.

Key differences from v1:
    - Block-size parameters accept ``(decode, prefill, mixed)`` triples.
    - ``MultiLatentRaggedPageAttentionV2Config`` normalises list payloads
      (from JSON de-serialisation) back into tuples.
    - Uses operation key ``"multi_latent_ragged_page_attention_v2"`` so
      the registry resolves to the v2-specific Pallas/XLA implementations.

The paged-KV cache layout, ragged batching contract, and MLA decomposition
(nope + pe streams) are identical to v1 — see
``multi_latent_ragged_page_attention.py`` for the conceptual overview.

Example:
    >>> from ejkernel.modules.operations import multi_latent_ragged_page_attention_v2
    >>>
    >>> outputs, updated_cache = multi_latent_ragged_page_attention_v2(
    ...     queries_nope, queries_pe,
    ...     keys_values, keys_pe,
    ...     kv_cache, kv_lens, block_tables,
    ...     query_start_loc, distribution,
    ...     softmax_scale=1.0 / head_dim**0.5,
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
from .configs import MultiLatentRaggedPageAttentionV2Config


def _repeat_case(value: int) -> tuple[int, int, int]:
    """Broadcast a scalar block size into a ``(decode, prefill, mixed)`` triple.

    The v2 Pallas kernel accepts per-case block sizes.  When auto-tuning
    produces a single value, this helper replicates it for all three cases.

    Args:
        value: Block size to replicate.

    Returns:
        Three-element tuple ``(value, value, value)``.
    """
    return (value, value, value)


class MultiLatentRaggedPageAttentionV2(Kernel[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]]):
    """Multi-Latent Ragged Page Attention v2 with per-case block-size tuning.

    Extends the v1 MLA ragged paged attention kernel with ``(decode,
    prefill, mixed)`` block-size triples so the Pallas grid can be
    specialised for each request type within a single batch.

    The kernel writes new tokens into the paged KV cache *and* computes
    attention output in a single fused pass.
    """

    version = "2"

    def __init__(self):
        """Initialize MLRPA-v2 with its registry operation identifier."""
        super().__init__(op_id="multi_latent_ragged_page_attention_v2")

    def get_impl(self, cfg: MultiLatentRaggedPageAttentionV2Config):
        """Get kernel implementation from registry based on configuration.

        Pallas/TPU is preferred when available (``prefer_pallas=True``);
        otherwise the XLA fallback is used.

        Args:
            cfg: Configuration specifying platform and backend preferences.

        Returns:
            Callable kernel implementation for MLA ragged paged attention v2.
        """
        platform = detect_platform(
            "multi_latent_ragged_page_attention_v2",
            cfg.platform,
            prefer_pallas=True,
        )
        return kernel_registry.get(
            "multi_latent_ragged_page_attention_v2",
            platform=platform,
            backend=cfg.backend,
        )

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
        *,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        mask_value: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        cfg: MultiLatentRaggedPageAttentionV2Config,
    ) -> tuple[
        Float[Array, "total_tokens num_q_heads kv_latent_dim"],
        Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    ]:
        """Execute MLA ragged paged attention v2 and update the paged KV cache.

        Dispatches to the platform-specific kernel and forwards block-size
        parameters from the config.  When *platform* is given explicitly it
        overrides the value stored in *cfg*.

        Args:
            queries_nope: Non-positional query projections [total_tokens, num_q_heads, kv_latent_dim].
            queries_pe: Positional-encoding query projections [total_tokens, num_q_heads, qk_pe_dim].
            keys_values: New KV tokens to insert into the cache [total_tokens, kv_latent_dim].
            keys_pe: Positional-encoding keys for new tokens [total_tokens, qk_pe_dim].
            kv_cache: Paged KV cache [num_pages, page_size/kv_packing, kv_packing, kv_dim_padded].
            kv_lens: Context length per sequence [max_num_seqs].
            block_tables: Flat page-index table [max_num_seqs * pages_per_seq].
            query_start_loc: Cumulative query counts [max_num_seqs + 1].
            distribution: Three-element array ``[n_decode, n_prefill, n_mixed]``.
            softmax_scale: Attention scaling factor (default: ``(nope_dim + pe_dim)^-0.5``).
            sliding_window: Optional local-attention window size.
            logits_soft_cap: Optional Gemma-2-style soft cap for logits.
            mask_value: Fill value for masked positions (default from kernel).
            q_scale: Optional FP8 query de-quantisation scale.
            k_scale: Optional FP8 key de-quantisation scale.
            v_scale: Optional FP8 value de-quantisation scale.
            platform: Override for platform selection.
            cfg: Kernel configuration with block-size tuning parameters.

        Returns:
            Tuple of (attention_output, updated_kv_cache).
        """
        if platform is not None:
            cfg = MultiLatentRaggedPageAttentionV2Config(
                chunk_prefill_size=cfg.chunk_prefill_size,
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
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
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=cfg.chunk_prefill_size,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
            vmem_limit_bytes=cfg.vmem_limit_bytes,
        )

    def _estimate_kv_pages(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]],
    ) -> int:
        """Estimate a reasonable KV-pages-per-block from invocation shapes.

        Uses ``block_tables.shape[0] / kv_lens.shape[0]`` to approximate
        the average number of pages per sequence, then selects a tile size
        that balances parallelism against per-tile memory usage.

        Args:
            inv: Invocation containing ``block_tables`` and ``kv_lens`` kwargs.

        Returns:
            Heuristic KV-pages-per-block (2, 4, 8, or 16).
        """
        try:
            kv_lens = inv.kwargs["kv_lens"]
            block_tables = inv.kwargs["block_tables"]
            pages_per_seq = max(1, int(block_tables.shape[0]) // int(kv_lens.shape[0]))
        except Exception:
            pages_per_seq = 16

        if pages_per_seq >= 256:
            return 16
        if pages_per_seq >= 128:
            return 8
        if pages_per_seq >= 64:
            return 4
        return 2

    def heuristic_cfg(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]],
    ) -> MultiLatentRaggedPageAttentionV2Config:
        """Generate default configuration based on input characteristics.

        Block sizes are estimated from the invocation shapes so that first
        calls (before autotuning completes) get a reasonable starting point.

        Args:
            inv: Invocation containing input tensors and metadata.

        Returns:
            Configuration with ``num_kv_pages_per_block`` derived from the
            average pages-per-sequence and ``num_queries_per_block`` set to 32.
        """
        kv_pages = self._estimate_kv_pages(inv)
        return MultiLatentRaggedPageAttentionV2Config(
            chunk_prefill_size=None,
            num_kv_pages_per_block=_repeat_case(kv_pages),
            num_queries_per_block=_repeat_case(32),
            vmem_limit_bytes=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]],
    ):
        """Generate candidate configurations for autotuning.

        Explores two query-block sizes (16 and 32) while keeping the
        KV-pages-per-block fixed at the heuristic estimate.  The tuner
        benchmarks each candidate and caches the winner.

        Args:
            inv: Invocation containing input tensors and metadata.

        Returns:
            List of ``MultiLatentRaggedPageAttentionV2Config`` candidates.
        """
        kv_pages = self._estimate_kv_pages(inv)
        return [
            MultiLatentRaggedPageAttentionV2Config(
                chunk_prefill_size=None,
                num_kv_pages_per_block=_repeat_case(kv_pages),
                num_queries_per_block=_repeat_case(q_block),
                vmem_limit_bytes=None,
                num_warps=4,
                num_stages=1,
                platform="auto",
                backend="any",
            )
            for q_block in (16, 32)
        ]

    def candidate_cfgs_gpu(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]],
    ):
        """Generate GPU candidates for TileLang and XLA MLRPA-v2.

        MLRPA-v2 has per-mode (decode / prefill / mixed) tile tuples but
        the helper ``_repeat_case`` projects a single value across all three.
        Sweeps span the kv-pages estimate bracket and four query-block sizes.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        kv_pages = self._estimate_kv_pages(inv)
        kv_choices = sorted({max(1, kv_pages // 2), kv_pages, kv_pages * 2})
        candidates: list[MultiLatentRaggedPageAttentionV2Config] = []
        for platform, backend in (("tilelang", "gpu"), ("xla", "any")):
            if platform not in platforms:
                continue
            candidates.extend(
                MultiLatentRaggedPageAttentionV2Config(
                    chunk_prefill_size=None,
                    num_kv_pages_per_block=_repeat_case(kv_block),
                    num_queries_per_block=_repeat_case(q_block),
                    vmem_limit_bytes=None,
                    num_warps=4,
                    num_stages=1,
                    platform=platform,
                    backend=backend,
                )
                for kv_block in kv_choices
                for q_block in (8, 16, 32, 64)
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(
        self,
        inv: Invocation[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]],
    ):
        """Generate TPU candidates for Pallas and XLA MLRPA-v2."""
        kv_pages = self._estimate_kv_pages(inv)
        return [
            MultiLatentRaggedPageAttentionV2Config(
                chunk_prefill_size=None,
                num_kv_pages_per_block=_repeat_case(kv_pages),
                num_queries_per_block=_repeat_case(q_block),
                vmem_limit_bytes=None,
                num_warps=4,
                num_stages=1,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for q_block in (16, 32)
        ]


_mlrpa_v2_executor: Executor[MultiLatentRaggedPageAttentionV2Config, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=3, iters=40),
        persistent=PersistentCache("multi_latent_ragged_page_attention_v2"),
    )
)


def multi_latent_ragged_page_attention_v2(
    queries_nope: Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    queries_pe: Float[Array, "total_tokens num_q_heads qk_pe_dim"],
    keys_values: Float[Array, "total_tokens kv_latent_dim"],
    keys_pe: Float[Array, "total_tokens qk_pe_dim"],
    kv_cache: Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: MultiLatentRaggedPageAttentionV2Config | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads kv_latent_dim"],
    Float[Array, "num_pages page_size_per_kv_packing kv_packing kv_dim_padded"],
]:
    """Execute Multi-Latent Ragged Page Attention v2 with automatic optimization.

    Functional API that wraps :class:`MultiLatentRaggedPageAttentionV2` with
    the autotuning executor.  Block-size tuning parameters (``num_kv_pages_per_block``,
    ``num_queries_per_block``, etc.) are managed by the config; the functional
    signature only exposes model-level knobs.

    Args:
        queries_nope: Query non-positional component
            ``[total_tokens, num_q_heads, kv_latent_dim]``.
        queries_pe: Query positional component
            ``[total_tokens, num_q_heads, qk_pe_dim]``.
        keys_values: New cache KV-compressed tokens ``[total_tokens, kv_latent_dim]``.
        keys_pe: New cache key positional component ``[total_tokens, qk_pe_dim]``.
        kv_cache: Paged KV cache
            ``[num_pages, page_size/kv_packing, kv_packing, kv_dim_padded]``.
        kv_lens: Context length per sequence ``[max_num_seqs]``.
        block_tables: Flat page-index table ``[max_num_seqs * pages_per_seq]``.
        query_start_loc: Cumulative query token counts ``[max_num_seqs + 1]``.
        distribution: Three-element array ``[n_decode, n_prefill, n_mixed]``.
        softmax_scale: Attention scaling factor (default: ``(nope_dim + pe_dim)^-0.5``).
        sliding_window: Optional local-attention window size.
        logits_soft_cap: Optional Gemma-2-style soft cap for logits.
        mask_value: Fill value for masked positions.
        q_scale: Optional FP8 query de-quantisation scale.
        k_scale: Optional FP8 key de-quantisation scale.
        v_scale: Optional FP8 value de-quantisation scale.
        platform: Override for platform selection.
        cfg: Optional configuration override (bypasses autotuning).

    Returns:
        Tuple ``(outputs, updated_kv_cache)`` where:
        - ``outputs``: ``[total_tokens, num_q_heads, kv_latent_dim]``
        - ``updated_kv_cache``: same shape as ``kv_cache``
    """
    return _mlrpa_v2_executor(
        MultiLatentRaggedPageAttentionV2(),
        queries_nope=queries_nope,
        queries_pe=queries_pe,
        keys_values=keys_values,
        keys_pe=keys_pe,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
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
    "MultiLatentRaggedPageAttentionV2",
    "multi_latent_ragged_page_attention_v2",
)
