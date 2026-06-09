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


"""Standard multi-head attention module with automatic optimization.

This module implements standard multi-head attention (MHA) with XLA-optimized kernels.
It provides a flexible interface supporting various attention patterns including causal
masking, dropout, sliding windows, and variable-length sequences.

Unlike FlashAttention which uses tiling for memory efficiency, this implementation
leverages XLA's compiler optimizations for straightforward attention computation.

Key Features:
    - XLA-optimized attention computation
    - Returns both attention output and attention weights
    - Causal and bidirectional attention patterns
    - Dropout support for training with configurable probability
    - Sliding window attention for local context patterns
    - Attention biasing and masking support
    - Logit soft capping for numerical stability
    - Lazy bias initialization for memory efficiency
    - Grouped Query Attention (GQA) support

Use Cases:
    - Training scenarios requiring attention weight access
    - Debugging and visualization of attention patterns
    - Situations where XLA optimization is sufficient
    - Research and experimentation with attention patterns

Mathematical Foundation:
    Multi-head attention:
        head_i = softmax(Q_i @ K_i.T / sqrt(d_k)) @ V_i
        output = Concat(head_1, ..., head_h) @ W_o

    With optional features:
        - Causal mask: mask[i,j] = 1 if j <= i else 0
        - Sliding window: mask[i,j] = 1 if |i-j| <= window_size
        - Bias: scores = Q @ K.T / sqrt(d_k) + bias
        - Soft cap: scores = soft_cap * tanh(scores / soft_cap)

Output Format:
    Returns tuple of:
        - output: [batch, seq_len, num_heads, head_dim] - attention output
        - weights: [batch, num_heads, seq_len, kv_len] - attention probabilities

    The attention weights are useful for:
        - Visualization and interpretability
        - Attention-based pooling
        - Research and debugging

Performance Characteristics:
    - Memory: O(N^2) for storing attention weights
    - Compute: O(N^2 * d) standard attention complexity
    - Best suited for: Moderate sequence lengths, training with weight access

Note:
    For memory-efficient attention without weight access, prefer FlashAttention.
    For inference-only workloads, prefer PageAttention or UnifiedAttention.
"""

from __future__ import annotations

from collections.abc import Callable

from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from ejkernel.kernels._registry import Platform, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    BwdParams,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    FwdParams,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache
from ejkernel.types.mask import MaskInfo

from ..base import detect_platform
from .configs import AttentionConfig


class Attention(Kernel[AttentionConfig, tuple[Array, Array]]):
    """Attention with custom optimization logic.

    Supports causal masking, dropout, sliding windows, and variable-length sequences.

    Features:
        - Automatic platform/backend selection (XLA-only; no Triton/Pallas dispatch
          is registered for this op — see FlashAttention for GPU/TPU alternatives).
        - Configuration caching for consistent performance.
        - Optional autotuning to find optimal implementation.
        - Custom gradient support for efficient backpropagation.
        - Support for variable-length sequences via cumulative sequence lengths.
        - Sliding window attention for local attention patterns.
        - Logits soft capping for numerical stability.

    Example:
        >>> from ejkernel.modules import Attention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = Attention()
        >>>
        >>>
        >>> output = executor(attn, query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>>
        >>> output = executor(
        ...     attn, query, key, value,...
        ... )
        >>>
        >>>
        >>> output = executor(attn, query, key, value, sliding_window=(256, 256))
    """

    def __init__(self):
        """Initialize  Attention module."""
        super().__init__(op_id="attention")

    def get_impl(self, cfg: AttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="attention",
            platform=detect_platform("attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        softmax_aux: Float[Array, "num_sinks"] | None = None,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dtype: DTypeLike | None = jnp.bfloat16,
        softmax_dtype: DTypeLike | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        *,
        cfg: AttentionConfig,
    ) -> tuple[
        Float[Array, "batch seq_len num_q_heads vhead_dim"],
        Float[Array, "batch num_heads seq_len kv_len"],
    ]:
        """Execute standard multi-head attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len, num_q_heads, head_dim]
            key: Key tensor [batch, kv_len, num_kv_heads, head_dim]
            value: Value tensor [batch, kv_len, num_kv_heads, vhead_dim]
            attention_mask: Optional boolean/integer mask [batch, num_heads_or_1,
                seq_len, kv_len]. Positions where mask is False/0 are excluded
                from attention. Prefer ``bias`` for additive masking.
            bias: Optional additive bias added to raw attention logits
                [batch, num_heads, seq_len, kv_len].
            init_bias: Lazy factory for ``bias``. Called only when ``bias``
                is None; avoids materialising large bias tensors when unused.
            deterministic: Disable dropout when True (default: True).
            dropout_rng: PRNG key for dropout; required when
                ``dropout_prob > 0`` and ``deterministic=False``.
            softmax_aux: Optional attention-sink logits added before softmax.
            softmax_scale: Scale factor applied to logits before softmax.
                Defaults to ``1 / sqrt(head_dim)``.
            logits_soft_cap: If set, clamps logits via
                ``soft_cap * tanh(logits / soft_cap)`` (Gemma-2 style).
            dtype: Dtype for intermediate computations (default: bfloat16).
            softmax_dtype: Dtype for softmax accumulation; falls back to
                ``dtype`` when None.
            dropout_prob: Dropout probability (default: 0.0).
            causal: Apply causal masking (default: False).
            sliding_window: Local attention window.  An ``int`` applies a
                symmetric window; a ``(left, right)`` tuple is asymmetric.
            cfg: Kernel configuration (platform/backend selection).

        Returns:
            Tuple of:
                - output: Attention output [batch, seq_len, num_q_heads, vhead_dim]
                - weights: Attention probabilities [batch, num_heads, seq_len, kv_len]
        """
        cfg_platform = getattr(cfg, "platform", "auto")
        cfg_backend = getattr(cfg, "backend", "any")
        block_q = int(getattr(cfg, "block_q", 128))
        block_k = int(getattr(cfg, "block_k", 128))
        num_warps = int(getattr(cfg, "num_warps", 4))
        num_stages = int(getattr(cfg, "num_stages", 2))
        weights_block_q = int(getattr(cfg, "weights_block_q", self._heuristic_weights_block(int(query.shape[1]))))
        weights_block_k = int(getattr(cfg, "weights_block_k", self._heuristic_weights_block(int(key.shape[1]))))

        resolved_platform = detect_platform("attention", cfg_platform)
        impl = kernel_registry.get(
            algorithm="attention",
            platform=resolved_platform,
            backend=cfg_backend,
        )
        impl_kwargs = dict(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            dropout_prob=dropout_prob,
            init_bias=init_bias,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            dtype=dtype,
            softmax_dtype=softmax_dtype,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            causal=causal,
        )
        if resolved_platform == Platform.TILELANG:
            impl_kwargs["fwd_params"] = FwdParams(
                q_blocksize=block_q,
                kv_blocksize=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            impl_kwargs["bwd_params"] = BwdParams(
                q_blocksize=max(32, block_q // 2),
                kv_blocksize=max(32, block_k // 2),
                num_warps=num_warps,
                num_stages=num_stages,
            )
        impl_kwargs["weights_block_q"] = weights_block_q
        impl_kwargs["weights_block_k"] = weights_block_k
        return impl(**impl_kwargs)

    @staticmethod
    def _seqlens_from_inv(inv: Invocation[AttentionConfig, Array]) -> tuple[int, int]:
        """Pull ``(seq_len_q, seq_len_k)`` from the invocation's q/k tensors."""
        query = inv.kwargs.get("query")
        key = inv.kwargs.get("key")
        q_len = int(query.shape[1]) if getattr(query, "shape", None) else 0
        k_len = int(key.shape[1]) if getattr(key, "shape", None) else 0
        return q_len, k_len

    @staticmethod
    def _heuristic_weights_block(seq_len: int) -> int:
        """Operation-side dense-weights tile heuristic — single source of truth.

        Mirrors the historical kernel-side ladder verbatim.
        """
        return 32 if 0 < seq_len <= 32 else 64

    def heuristic_cfg(self, inv: Invocation[AttentionConfig, Array]) -> AttentionConfig:
        """Cold-start configuration with shape-aware ``weights_block_q/k``."""
        q_len, k_len = self._seqlens_from_inv(inv)
        return AttentionConfig(
            block_q=128,
            block_k=128,
            weights_block_q=self._heuristic_weights_block(q_len),
            weights_block_k=self._heuristic_weights_block(k_len),
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[AttentionConfig, Array]):
        """Generate candidate configurations for autotuning."""
        return [
            self.heuristic_cfg(inv),
            AttentionConfig(
                block_q=128,
                block_k=128,
                weights_block_q=64,
                weights_block_k=64,
                num_warps=4,
                num_stages=2,
                platform="xla",
                backend="any",
            ),
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[AttentionConfig, Array]):
        """Generate GPU candidates for dense attention with weights.

        Two interleaved kernels share this op: the FlashAttention forward
        (tuned via ``block_q``/``block_k``) and the dense ``(B, H, Sq, Sk)``
        weights kernel (tuned via ``weights_block_q``/``weights_block_k``).
        The sweep crosses both spaces, with shape-aware pruning:

        * FA tiles ∈ {64, 128, 256} pruned by seq_len and head_dim.
        * Weights tiles ∈ {32, 64} — smaller than FA because the dense
          ``(Sq, Sk)`` matrix is the bottleneck.
        * ``num_warps`` ∈ {4, 8}, ``num_stages`` ∈ {2, 3}.
        """
        q_len, k_len = self._seqlens_from_inv(inv)
        query = inv.kwargs.get("query")
        head_dim = int(query.shape[-1]) if getattr(query, "shape", None) else 64
        q_opts = [32, 64, 128, 256] if q_len <= 256 else [64, 128, 256]
        k_opts = [32, 64, 128, 256] if k_len <= 256 else [64, 128, 256]
        if head_dim >= 128:
            k_opts = [k for k in k_opts if k <= 128] or [128]
            q_opts = [q for q in q_opts if q <= 128] or [128]
        if q_len < 512:
            q_opts = [q for q in q_opts if q < 256] or [128]
        if k_len < 512:
            k_opts = [k for k in k_opts if k < 256] or [128]
        w_q_opts = [32, 64] if q_len <= 128 else [64]
        w_k_opts = [32, 64] if k_len <= 128 else [64]
        candidates: list[AttentionConfig] = []
        for block_q in q_opts:
            for block_k in k_opts:
                big = max(block_q, block_k) >= 128
                warps = 8 if (head_dim >= 128 and big) else 4
                for wbq in w_q_opts:
                    for wbk in w_k_opts:
                        for stages in (2, 3) if k_len >= 1024 else (2,):
                            candidates.append(
                                AttentionConfig(
                                    block_q=block_q,
                                    block_k=block_k,
                                    weights_block_q=wbq,
                                    weights_block_k=wbk,
                                    num_warps=warps,
                                    num_stages=stages,
                                    platform="tilelang",
                                    backend="gpu",
                                )
                            )
        candidates.append(
            AttentionConfig(
                block_q=128,
                block_k=128,
                weights_block_q=64,
                weights_block_k=64,
                num_warps=4,
                num_stages=2,
                platform="xla",
                backend="any",
            )
        )
        return candidates

    def candidate_cfgs_tpu(self, inv: Invocation[AttentionConfig, Array]):
        """Return TPU candidates for the portable XLA dense-attention path."""
        return [
            AttentionConfig(
                block_q=128,
                block_k=128,
                weights_block_q=64,
                weights_block_k=64,
                num_warps=4,
                num_stages=2,
                platform="xla",
                backend="any",
            )
        ]


_executor: Executor[AttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=True),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("attention"),
    )
)


def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    dropout_rng: PRNGKeyArray | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    dtype: DTypeLike | None = jnp.bfloat16,
    softmax_dtype: DTypeLike | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> tuple[Float[Array, "batch seq_len num_q_heads vhead_dim"], Float[Array, "batch num_heads seq_len kv_len"]]:
    """Execute standard multi-head attention with automatic optimization.

    Convenience wrapper around ``Attention.run`` that drives the operation
    through a pre-configured ``Executor`` with config caching.

    Unlike ``flash_attention``, this function returns **both** the attention
    output and the full attention weight matrix, which requires O(N²) memory.
    Prefer ``flash_attention`` when the attention weights are not needed.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim].
        key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor [batch, kv_len, num_kv_heads, vhead_dim].
        bias: Optional additive bias added to raw attention logits
            [batch, num_heads, seq_len, kv_len].
        dropout_rng: PRNG key required when ``dropout_prob > 0`` and
            ``deterministic=False``.
        softmax_aux: Optional attention-sink logits.
        mask_info: Optional ``MaskInfo`` container; the attention mask is
            extracted via ``get_or_compute_attention_mask()``.
        init_bias: Lazy factory for ``bias``; called only when ``bias``
            is None.
        deterministic: Disable dropout (default: True).
        softmax_scale: Scale factor for logits (default: 1/sqrt(head_dim)).
        logits_soft_cap: Soft cap value for attention logits (Gemma-2 style).
        dtype: Dtype for intermediate computations (default: bfloat16).
        softmax_dtype: Dtype for softmax accumulation.
        dropout_prob: Dropout probability (default: 0.0).
        causal: Apply causal masking (default: False).
        sliding_window: Local attention window (int or (left, right) tuple).

    Returns:
        Tuple of:
            - output: Attention output [batch, seq_len, num_q_heads, vhead_dim]
            - weights: Attention probabilities [batch, num_heads, seq_len, kv_len]

    Note:
        This operation is XLA-only; it does not dispatch to Triton or Pallas
        backends. For hardware-efficient fused attention use ``flash_attention``.

    Example:
        >>> out, weights = attention(query, key, value)
        >>>
        >>> out, weights = attention(query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>> out, weights = attention(query, key, value, causal=True, sliding_window=256)
    """

    attention_mask = None

    if mask_info is not None:
        attention_mask = mask_info.get_or_compute_attention_mask()

    out, w = _executor(
        Attention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=dropout_prob,
        init_bias=init_bias,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        dtype=dtype,
        softmax_dtype=softmax_dtype,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        causal=causal,
    )
    return out, w
