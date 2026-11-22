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


"""Multi-head Latent Attention (MLA) module with automatic optimization.

This module implements Multi-head Latent Attention, a memory-efficient attention
variant that uses low-rank compression for key-value pairs. MLA reduces the KV cache
size by projecting keys and values through a low-rank bottleneck while maintaining
attention quality.

Key Features:
    - Low-rank KV compression reducing memory by up to 93.3%
    - On-the-fly key-value reconstruction using learned weights
    - Compatible with RoPE positional encoding via bias terms
    - Flash attention-style tiling for memory efficiency
    - Support for variable-length sequences and causal masking

The Compression Scheme:
    The key innovation is compressing the KV representations:
        1. Keys and values are projected to a low-rank space (kv_lora_rank)
        2. Compressed representations are stored efficiently
        3. Full-rank keys/values are reconstructed on-the-fly using learned weights

    Mathematical formulation:
        - Compressed KV: c_kv = [h_1, h_2, ..., h_n] where h_i in R^{kv_lora_rank}
        - Key reconstruction: K = c_kv @ W_kc  (W_kc in R^{kv_lora_rank x d_k})
        - Value reconstruction: V = c_kv @ W_vc  (W_vc in R^{kv_lora_rank x d_v})
        - With RoPE: K = concat(c_kv @ W_kc, RoPE(b_k))

Use Cases:
    - Long context inference where KV cache dominates memory
    - Multi-query or grouped-query attention patterns
    - Deployment scenarios with memory constraints
    - Models requiring efficient inference with large context windows

Example:
    >>> from ejkernel.modules.operations import flash_mla
    >>>
    >>> # Basic MLA with low-rank compression
    >>> output = flash_mla(
    ...     query, key_value, w_kc, w_vc,
    ...     causal=True,
    ... )
    >>>
    >>> # With RoPE positional encoding
    >>> output = flash_mla(
    ...     query, key_value, w_kc, w_vc,
    ...     b_q=rope_query_bias,
    ...     b_k=rope_key_bias,
    ...     causal=True,
    ... )

References:
    - DeepSeek-V2: https://arxiv.org/abs/2405.04434
    - Flash Attention: https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

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
from .configs import FlashMLAConfig


class FlashMLA(Kernel[FlashMLAConfig, Array]):
    """Flash Multi-head Latent Attention with custom optimization logic.

    Combines flash attention's memory efficiency with MLA's low-rank KV compression.
    This implementation uses tiling and on-the-fly decompression to achieve both
    reduced memory footprint and computational efficiency.

    Features:
        - Low-rank KV compression via w_kc and w_vc weight matrices
        - Optional RoPE bias for positional encoding (b_q, b_k)
        - Flash attention-style tiling for memory efficiency
        - Support for causal masking and variable-length sequences
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The compression scheme:
        - key_value: Compressed KV tensor [batch, seq_len, kv_lora_rank]
        - w_kc, w_vc: Decompression weights [kv_lora_rank, kv_heads, head_dim]
        - Keys/values are reconstructed as: key = key_value @ w_kc
    """

    def __init__(self):
        """Initialize Flash MLA module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="flash_mla")

    def get_impl(self, cfg: FlashMLAConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for flash MLA

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("flash_mla", cfg.platform)
        return kernel_registry.get("flash_mla", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len q_heads q_head_dim"],
        key_value: Float[Array, "batch seq_len kv_lora_rank"],
        w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
        w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
        b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
        softmax_aux: Float[Array, "..."] | None = None,
        logits_soft_cap: float | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        dropout_prob: float = 0.0,
        sliding_window: int | tuple[int, int] | None = None,
        softmax_dtype: DTypeLike | None = None,
        platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
        *,
        cfg: FlashMLAConfig,
    ) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
        """Execute flash multi-head latent attention.

        Args:
            query: Query tensor [batch, seq_len, q_heads, head_dim]
            key_value: Compressed key-value tensor [batch, seq_len, kv_lora_rank]
            w_kc: Key decompression weights [kv_lora_rank, kv_heads, head_dim]
            w_vc: Value decompression weights [kv_lora_rank, kv_heads, head_dim]
            b_q: Optional query RoPE bias [batch, seq_len, qk_rope_head_dim]
            b_k: Optional key RoPE bias [batch, seq_len, qk_rope_head_dim]
            softmax_scale: Optional scaling factor for attention scores
            causal: Whether to apply causal masking (default: False)
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            attention_mask: Optional boolean mask [batch, 1/heads, seq_len, kv_len]
            bias: Optional additive attention bias [batch, 1/heads, seq_len, kv_len]
            softmax_aux: Optional attention sink logits
            logits_soft_cap: Optional logits capping via tanh
            deterministic: If True, disables dropout (default: True)
            dropout_rng: JAX PRNG key for dropout
            dropout_prob: Dropout probability (default: 0.0)
            sliding_window: Optional sliding window constraint
            softmax_dtype: Dtype for softmax accumulation
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len, q_heads, head_dim]
        """

        if platform is not None:
            cfg = FlashMLAConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key_value=key_value,
            w_kc=w_kc,
            w_vc=w_vc,
            b_q=b_q,
            b_k=b_k,
            softmax_scale=softmax_scale,
            causal=causal,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            bias=bias,
            softmax_aux=softmax_aux,
            logits_soft_cap=logits_soft_cap,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            dropout_prob=dropout_prob,
            sliding_window=sliding_window,
            softmax_dtype=softmax_dtype,
        )

    def heuristic_cfg(self, inv: Invocation[FlashMLAConfig, Array]) -> FlashMLAConfig:
        """Provide default configuration with block sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration optimized for MLA's low-rank decompression
            and on-the-fly reconstruction requirements
        """
        return FlashMLAConfig(
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FlashMLAConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            MLA performance depends on the compression rank and decompression
            overhead. Candidates balance memory efficiency with compute cost.
        """
        return [self.heuristic_cfg(inv)]

    def candidate_cfgs_gpu(self, inv: Invocation[FlashMLAConfig, Array]):
        """Generate GPU candidates for TileLang and XLA FlashMLA.

        MLA's effective head dim is ``qk_nope_dim + qk_rope_dim`` (often
        128 + 64 = 192), so ``block_k`` is capped to keep SMEM in budget.
        On H100:

        * ``(64, 64)`` — short prompts / decode-style.
        * ``(128, 64)`` — common training tile, wide head dim.
        * ``(64, 128)`` — long KV with smaller Q chunks.
        * ``(128, 128)`` — only when head dim fits SMEM.
        """
        requested = inv.kwargs.get("platform", None)
        platforms = ("tilelang", "xla") if requested in (None, "auto") else (str(requested),)
        candidates: list[FlashMLAConfig] = []
        if "tilelang" in platforms:
            for bq, bk in ((64, 64), (128, 64), (64, 128), (128, 128)):
                for stages in (2, 3):
                    candidates.append(
                        FlashMLAConfig(
                            block_q=bq,
                            block_k=bk,
                            num_warps=8 if max(bq, bk) >= 128 else 4,
                            num_stages=stages,
                            platform="tilelang",
                            backend="gpu",
                        )
                    )
        if "xla" in platforms:
            candidates.append(
                FlashMLAConfig(
                    block_q=128,
                    block_k=128,
                    num_warps=4,
                    num_stages=2,
                    platform="xla",
                    backend="any",
                )
            )
        return candidates or self.candidate_cfgs(inv)

    def candidate_cfgs_tpu(self, inv: Invocation[FlashMLAConfig, Array]):
        """Generate TPU candidates for Pallas and XLA FlashMLA."""
        return [
            FlashMLAConfig(
                block_q=block_q,
                block_k=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                platform=platform,
                backend=backend,
            )
            for platform, backend in (("pallas", "tpu"), ("xla", "any"))
            for block_q, block_k, num_warps, num_stages in ((64, 64, 4, 1), (128, 128, 4, 2))
        ]


_mla_executor: Executor[FlashMLAConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("mla"),
    )
)


def flash_mla(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    attention_mask: Bool[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch heads_or_1 seq_len kv_len"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
    logits_soft_cap: float | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    dropout_prob: float = 0.0,
    sliding_window: int | tuple[int, int] | None = None,
    softmax_dtype: DTypeLike | None = None,
    platform: Literal["triton", "pallas", "cuda", "tilelang", "xla", "auto", "cute"] | None = None,
    cfg: FlashMLAConfig | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Execute flash multi-head latent attention with automatic optimization.

    MLA uses low-rank compression for key-value pairs to reduce memory
    and computation while maintaining attention quality. The key innovation
    is compressing KV representations through a low-rank bottleneck and
    reconstructing them on-the-fly during attention computation.

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim]
        key_value: Compressed key-value tensor [batch, seq_len, kv_lora_rank]
        w_kc: Key decompression weights [kv_lora_rank, kv_heads, qk_nope_head_dim]
        w_vc: Value decompression weights [kv_lora_rank, kv_heads, v_head_dim]
        b_q: Optional query RoPE bias [batch, seq_len, qk_rope_head_dim]
        b_k: Optional key RoPE bias [batch, seq_len, qk_rope_head_dim]
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking (default: False)
        attention_mask: Optional boolean mask [batch, 1/heads, seq_len, kv_len]
        bias: Optional additive attention bias [batch, 1/heads, seq_len, kv_len]
        softmax_aux: Optional attention sink logits
        logits_soft_cap: Optional logits capping via tanh
        deterministic: If True, disables dropout (default: True)
        dropout_rng: JAX PRNG key for dropout
        dropout_prob: Dropout probability (default: 0.0)
        sliding_window: Optional sliding window constraint
        softmax_dtype: Dtype for softmax accumulation
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional configuration override

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim]

    Example:
        >>> # Basic MLA with low-rank compression
        >>> output = flash_mla(
        ...     query, key_value, w_kc, w_vc,
        ...     causal=True,
        ... )

        >>> # With RoPE positional encoding
        >>> output = flash_mla(
        ...     query, key_value, w_kc, w_vc,
        ...     b_q=rope_query_bias,
        ...     b_k=rope_key_bias,
        ...     causal=True,
        ... )

        >>> # With attention mask and sliding window
        >>> output = flash_mla(
        ...     query, key_value, w_kc, w_vc,
        ...     causal=True,
        ...     logits_soft_cap=50.0,
        ...     sliding_window=256,
        ... )
    """
    return _mla_executor(
        FlashMLA(),
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seqlens=cu_seqlens,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        logits_soft_cap=logits_soft_cap,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        dropout_prob=dropout_prob,
        sliding_window=sliding_window,
        softmax_dtype=softmax_dtype,
        platform=platform,
        _cfg=cfg,
    )
