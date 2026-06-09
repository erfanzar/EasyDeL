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

"""Flash Attention backward pass implementation using XLA/JAX.

This module provides the backward pass (gradient computation) for Flash
Attention using JAX's automatic differentiation. The gradients are computed
with respect to queries, keys, and values.

Key Components:
    - _flash_attention_bwd: Main backward function using JAX autodiff

Algorithm:
    The backward pass uses JAX's grad transformation on the forward pass:
    1. Reconstruct forward computation with saved residuals
    2. Apply chain rule through attention computation
    3. Accumulate gradients for Q, K, V

Features:
    - Supports all forward pass features (causal, sliding window, dropout)
    - Uses same precision settings as forward pass
    - Handles optional bias gradients

Note:
    This implementation uses JAX autodiff rather than a custom backward
    kernel. While this may be less memory-efficient than a fused backward
    kernel, it ensures correctness and supports all forward pass features.
"""

import chex
import jax


def _flash_attention_bwd(
    bias: chex.Array | None,
    mask: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
    window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: chex.Array | None,
    res: tuple,
    g: chex.Array,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Backward pass for flash attention using JAX autodiff.

    Reconstructs the forward computation from saved residuals and uses
    JAX's VJP (vector-Jacobian product) to compute gradients for query,
    key, and value tensors. All static parameters (precision, chunk sizes,
    causal mode, etc.) are provided as encoded integers or plain values
    and decoded internally.

    Args:
        bias: Optional attention bias tensor used in the forward pass.
        mask: Optional boolean attention mask used in the forward pass.
        q_segment_ids: Optional query segment IDs for packed sequences.
        kv_segment_ids: Optional key/value segment IDs for packed sequences.
        softmax_aux: Optional attention sink logits used in the forward pass.
        window: Optional (left, right) sliding window bounds.
        softmax_scale: Scaling factor applied to attention scores.
        logits_soft_cap: Optional soft cap value for attention logits.
        chunk_size_q: Query chunk size used in the forward pass.
        chunk_size_k: Key/value chunk size used in the forward pass.
        normalize_output: Whether output was normalized by attention weight sum.
        precision_code: Integer code encoding the JAX precision setting.
        logits_dtype_code: Integer code encoding the logits computation dtype.
        causal: Whether causal masking was applied in the forward pass.
        dropout_prob: Dropout probability used in the forward pass.
        dropout_key: Optional PRNG key for dropout reproducibility.
        res: Tuple of (query, key, value) residuals saved from the forward pass.
        g: Gradient of the loss with respect to the attention output.

    Returns:
        Tuple of (dq, dk, dv) where each gradient has the same shape
        as its corresponding input tensor.
    """
    from ._xla_impl_fwd import _flash_attention_fwd

    _CODE_TO_PREC = {
        0: jax.lax.Precision.DEFAULT,
        1: jax.lax.Precision.HIGHEST,
        2: jax.lax.Precision.HIGH,
    }
    _CODE_TO_DTYPE = {
        0: jax.numpy.float16,
        1: jax.numpy.bfloat16,
        2: jax.numpy.float32,
        3: jax.numpy.float64,
    }

    q, k, v = res
    precision = _CODE_TO_PREC[precision_code]
    logits_dtype = _CODE_TO_DTYPE[logits_dtype_code]

    def f(q_, k_, v_):
        """Reconstruct the forward pass for VJP computation.

        Args:
            q_: Query tensor.
            k_: Key tensor.
            v_: Value tensor.

        Returns:
            Forward pass attention output.
        """
        return _flash_attention_fwd(
            q_,
            k_,
            v_,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            window=window,
            chunk_size_q=chunk_size_q,
            chunk_size_k=chunk_size_k,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=logits_dtype,
            softmax_aux=softmax_aux,
            causal=causal,
            dropout_prob=dropout_prob,
            dropout_key=dropout_key,
        )

    _, pullback = jax.vjp(f, q, k, v)
    dq, dk, dv = pullback(g)
    return dq, dk, dv
