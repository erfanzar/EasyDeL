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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Bool, Callable, Float, Int
from ._xla_impl_fwd import scaled_dot_product_attention as _scaled_dot_product_attention_impl


@kernel_registry.register("scaled_dot_product_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Compute scaled dot-product attention using XLA backend.
    This function wraps JAX's native dot_product_attention with the XLA
    implementation, which is optimized for CPU and general-purpose hardware.
    Args:
        query: Query tensor of shape [batch, seq_len, num_q_heads, head_dim].
        key: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim].
        attention_mask: Optional boolean mask of shape [batch, 1, seq_len, kv_len].
            Positions with True are attended, False positions are masked out.
        bias: Optional attention bias of shape [batch, num_heads, seq_len, kv_len].
            Added to attention logits before softmax.
        init_bias: Optional callable that generates bias tensor lazily.
            Called only if bias is None. Useful for avoiding computation when bias
            is not needed.
        softmax_scale: Scaling factor applied to attention logits before softmax.
            Defaults to 1/sqrt(head_dim) if None.
        causal: If True, applies causal (autoregressive) masking where each position
            can only attend to earlier positions.
        sliding_window: Optional local attention window size. Can be:
            - int: symmetric window size (past and future)
            - tuple[int, int]: (past_window, future_window) for asymmetric windows
            When set, limits attention to local context.
        cum_seqlens_q: Optional per-example query sequence lengths, shape [batch].
            Passed through to ``jax.nn.dot_product_attention`` as
            ``query_seq_lengths`` to mask padding in variable-length batches.
        cum_seqlens_k: Optional per-example key/value sequence lengths,
            shape [batch].  Passed as ``key_value_seq_lengths``.
        fwd_params: Operation-level tuning hint accepted for API parity; ignored by XLA.
        bwd_params: Operation-level tuning hint accepted for API parity; ignored by XLA.
    Returns:
        Attention output tensor of shape [batch, seq_len, num_q_heads, head_dim].
    Note:
        This implementation uses the XLA backend which provides good portability
        but may not be as optimized as specialized implementations (cuDNN, Flash Attention)
        for GPU hardware.
    Example:
        >>> import jax.numpy as jnp
        >>> query = jnp.ones((2, 128, 8, 64))
        >>> key = jnp.ones((2, 128, 8, 64))
        >>> value = jnp.ones((2, 128, 8, 64))
        >>> output = scaled_dot_product_attention(query, key, value, causal=True)
        >>> output.shape
        (2, 128, 8, 64)
    """
    _ = fwd_params, bwd_params
    return _scaled_dot_product_attention_impl(
        query,
        key,
        value,
        attention_mask,
        bias,
        init_bias,
        softmax_scale,
        causal,
        sliding_window,
        cum_seqlens_q,
        cum_seqlens_k,
    )


__all__ = ("scaled_dot_product_attention",)
