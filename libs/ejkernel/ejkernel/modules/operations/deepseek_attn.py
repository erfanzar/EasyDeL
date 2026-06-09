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


"""DeepSeek Sparse Attention (DSA) module with automatic optimization.

Implements DeepSeek-V3.2's sparse attention on top of MLA. Follows the same
tensor conventions as flash_mla: compressed KV latent with on-the-fly
projection via w_kc/w_vc, plus optional RoPE via b_q/b_k.

The DSA-specific additions are the Lightning Indexer inputs:
  - query_index: Lightweight indexer query projections (reuses MLA's q_lora)
  - key_index: Indexer key projections (from hidden states, shared across heads)
  - index_weights: Learned per-head aggregation weights

Architecture:
    Phase 1 — Lightning Indexer: Cheap FP8-friendly scorer that selects
    top-k KV tokens per query using ReLU activation + learned head weights.
    Phase 2 — Sparse MLA: Standard MLA attention with non-selected tokens
    masked to -inf before softmax.

References:
    - DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
    - DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
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
from .configs import DeepSeekAttentionConfig


class DeepSeekAttention(Kernel[DeepSeekAttentionConfig, Array]):
    """DeepSeek Sparse Attention with MLA + Lightning Indexer.

    Combines MLA attention (compressed KV latent, on-the-fly projection)
    with a lightweight learned indexer that dynamically selects top-k
    KV tokens per query position.
    """

    def __init__(self):
        super().__init__(op_id="deepseek_attn")

    def get_impl(self, cfg: DeepSeekAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend preferences.

        Returns:
            Callable kernel implementation for DSA (Pallas/TPU or XLA fallback).
        """
        platform = detect_platform("deepseek_attn", cfg.platform)
        return kernel_registry.get("deepseek_attn", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len q_heads q_head_dim"],
        key_value: Float[Array, "batch seq_len kv_lora_rank"],
        w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
        w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
        query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
        key_index: Float[Array, "batch seq_len index_head_dim"],
        index_weights: Float[Array, "batch seq_len index_heads"],
        index_topk: int = 2048,
        softmax_scale: float | None = None,
        index_softmax_scale: float | None = None,
        b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        causal: bool = True,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DeepSeekAttentionConfig,
    ) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
        """Execute DeepSeek Sparse Attention.

        Args:
            query: Query tensor [batch, seq_len, q_heads, q_head_dim].
            key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
            w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
            w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
            query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
            key_index: Indexer keys [batch, seq_len, index_head_dim].
            index_weights: Per-head weights [batch, seq_len, index_heads].
            index_topk: Tokens to select (default: 2048).
            softmax_scale: Attention scale.
            index_softmax_scale: Indexer scale.
            b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
            b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
            causal: Causal masking (default: True).
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Attention output [batch, seq_len, q_heads, v_head_dim].
        """
        cfg_index_topk = int(getattr(cfg, "index_topk", 2048))
        cfg_block_q = int(getattr(cfg, "block_q", 128))
        cfg_block_k = int(getattr(cfg, "block_k", 128))
        cfg_gemm_block = int(getattr(cfg, "gemm_block", self._heuristic_gemm_block(int(query.shape[1]))))
        cfg_num_warps = int(getattr(cfg, "num_warps", 4))
        cfg_num_stages = int(getattr(cfg, "num_stages", 2))
        cfg_backend = getattr(cfg, "backend", "any")

        if platform is not None:
            cfg = DeepSeekAttentionConfig(
                index_topk=cfg_index_topk,
                block_q=cfg_block_q,
                block_k=cfg_block_k,
                gemm_block=cfg_gemm_block,
                num_warps=cfg_num_warps,
                num_stages=cfg_num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg_backend,
            )
            cfg_index_topk = cfg.index_topk
            cfg_gemm_block = cfg.gemm_block
        impl = self.get_impl(cfg)
        resolved = detect_platform("deepseek_attn", cfg.platform)
        kwargs = dict(
            query=query,
            key_value=key_value,
            w_kc=w_kc,
            w_vc=w_vc,
            query_index=query_index,
            key_index=key_index,
            index_weights=index_weights,
            index_topk=index_topk if index_topk != 2048 else cfg_index_topk,
            softmax_scale=softmax_scale,
            index_softmax_scale=index_softmax_scale,
            b_q=b_q,
            b_k=b_k,
            causal=causal,
        )
        if resolved == Platform.TILELANG:
            kwargs["gemm_block"] = int(cfg_gemm_block)
        return impl(**kwargs)

    @staticmethod
    def _seq_len_from_inv(inv: Invocation[DeepSeekAttentionConfig, Array]) -> int:
        """Pull ``seq_len`` from the invocation's ``query`` tensor."""
        q = inv.kwargs.get("query")
        if q is None and inv.args:
            q = inv.args[0]
        shape = getattr(q, "shape", None)
        if shape is None or len(shape) < 2:
            return 0
        return int(shape[1])

    @staticmethod
    def _heuristic_gemm_block(seq_len: int) -> int:
        """Operation-side tile heuristic — single source of truth.

        Mirrors the historical kernel-side ``_pick_block`` ladder.
        """
        if seq_len == 0 or seq_len <= 64:
            return 64
        return 128

    def heuristic_cfg(self, inv: Invocation[DeepSeekAttentionConfig, Array]) -> DeepSeekAttentionConfig:
        """Cold-start configuration with shape-aware ``gemm_block``."""
        index_topk = int(inv.kwargs.get("index_topk", 2048))
        return DeepSeekAttentionConfig(
            index_topk=index_topk,
            block_q=128,
            block_k=128,
            gemm_block=self._heuristic_gemm_block(self._seq_len_from_inv(inv)),
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[DeepSeekAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Keeps ``index_topk`` fixed to the caller's requested value because it
        changes sparsity semantics and output numerics.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[DeepSeekAttentionConfig, Array]):
        """Generate GPU candidates for TileLang and XLA DeepSeek attention.

        DSA enumerates two orthogonal axes:

        * ``gemm_block`` — tile shared by the DSA-internal sub-kernels
          (indexer, KV-recon GEMMs, elementwise add/cast, top-k bias).
          On H100, 64 helps short prompts (``S<=512``) where CTA-count
          dominates; 128 helps longer prompts; 256 only at S>=4K and
          large heads (it's gated on seq_len below).
        * ``(block_q, block_k)`` — FlashAttention inner core. The
          MLA score dim can be wide (e.g. ``128+64`` with RoPE), so
          ``block_k=64`` is the safer default; ``block_k=128`` only
          when head_dim is small.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        index_topk = int(inv.kwargs.get("index_topk", 2048))
        seq_len = self._seq_len_from_inv(inv)
        gb_choices: list[int] = [64, 128]
        if seq_len == 0 or seq_len >= 4096:
            gb_choices.append(256)
        fa_pairs = [(128, 64), (128, 128), (64, 128), (64, 64)]
        candidates: list[DeepSeekAttentionConfig] = []
        if "tilelang" in platforms:
            for gb in gb_choices:
                for bq, bk in fa_pairs:
                    candidates.append(
                        DeepSeekAttentionConfig(
                            index_topk=index_topk,
                            block_q=bq,
                            block_k=bk,
                            gemm_block=gb,
                            num_warps=8 if max(bq, bk) >= 128 else 4,
                            num_stages=2,
                            platform="tilelang",
                            backend="gpu",
                        )
                    )
        if "xla" in platforms:
            candidates.append(
                DeepSeekAttentionConfig(
                    index_topk=index_topk,
                    block_q=128,
                    block_k=128,
                    gemm_block=128,
                    num_warps=4,
                    num_stages=2,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or [self.heuristic_cfg(inv)]

    def candidate_cfgs_tpu(self, inv: Invocation[DeepSeekAttentionConfig, Array]):
        """Generate TPU candidates for Pallas and XLA DeepSeek attention."""
        index_topk = int(inv.kwargs.get("index_topk", 2048))
        return [
            DeepSeekAttentionConfig(
                index_topk=index_topk,
                block_q=128,
                block_k=128,
                gemm_block=128,
                num_warps=4,
                num_stages=2,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
        ]


_dsa_executor: Executor[DeepSeekAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("dsa"),
    )
)


def deepseek_attn(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    /,
    *,
    index_topk: int = 2048,
    softmax_scale: float | None = None,
    index_softmax_scale: float | None = None,
    causal: bool = True,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: DeepSeekAttentionConfig | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Execute DeepSeek Sparse Attention with automatic optimization.

    DSA = MLA + Lightning Indexer. Same tensor conventions as flash_mla:
      - key_value: Compressed KV latent [batch, seq_len, kv_lora_rank]
      - w_kc/w_vc: On-the-fly K/V projection weights
      - b_q/b_k: Optional RoPE bias

    Plus DSA-specific Lightning Indexer inputs:
      - query_index: Indexer queries (derived from MLA's q_lora)
      - key_index: Indexer keys (from hidden states, shared across heads)
      - index_weights: Learned per-head aggregation weights

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
        query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
        key_index: Indexer keys [batch, seq_len, index_head_dim].
        index_weights: Per-head weights [batch, seq_len, index_heads].
        b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
        b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
        index_topk: Tokens to select per query (default: 2048).
        softmax_scale: Attention scale (default: effective_head_dim^-0.5).
        index_softmax_scale: Indexer scale (default: index_head_dim^-0.5).
        causal: Causal masking (default: True).
        platform: Platform override ("triton", "pallas", "cuda", "xla").
        cfg: Optional configuration override.

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim].

    Example:
        >>> output = deepseek_attn(
        ...     query, key_value, w_kc, w_vc,
        ...     query_index, key_index, index_weights,
        ...     index_topk=2048, causal=True,
        ... )
    """
    return _dsa_executor(
        DeepSeekAttention(),
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        query_index=query_index,
        key_index=key_index,
        index_weights=index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
        platform=platform,
        _cfg=cfg,
    )
