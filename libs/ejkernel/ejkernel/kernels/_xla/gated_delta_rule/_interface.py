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

"""Gated Delta Rule (GDR) public interface for XLA backend.

This module provides the public API for GDR, a linear attention mechanism used
in hybrid transformer architectures (e.g., Qwen3Next). It registers the kernel
under the ``"gated_delta_rule"`` key in the ejkernel registry for the XLA
platform.

GDR maintains a [head_dim, d_state] memory matrix per head that is updated via
a *gated delta rule*:
    h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

The public tensor convention is [batch, seq_len, num_heads, dim]; internally
tensors are transposed to [batch, num_heads, seq_len, dim] before dispatching
to the implementation functions.

Supported computation modes (dispatched by ``gated_delta_rule``):
    - Single-step (seq_len=1 + initial_state): fast path for autoregressive decoding.
    - Chunked (use_chunked=True, default): exact triangular-solve per chunk,
      sequential state propagation across chunks.
    - Recurrent (use_chunked=False): pure sequential scan, lowest memory.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _chunk_gdr_fwd, _recurrent_gdr_fwd, _single_step_gdr_fwd


@kernel_registry.register("gated_delta_rule", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def gated_delta_rule(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    *,
    chunk_size: int = 256,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
    seg_ids: Int[Array, "batch seq_len"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "batch num_heads qk_head_dim v_head_dim"],
]:
    """Gated Delta Rule (GDR) linear attention using XLA backend.

    GDR is a linear attention variant that combines gated delta rule updates
    with learnable decay for efficient sequence processing. It is used in
    hybrid transformer architectures like Qwen3Next.

    The core recurrence is:
        h_t = exp(decay_t) * h_{t-1} + k_t (x) (beta_t * (v_t - h_{t-1} @ k_t))
        o_t = h_t @ q_t

    Where:
        - h_t is the [head_dim, v_head_dim] memory matrix per head
        - exp(decay_t) controls memory retention (decay <= 0 for stability)
        - beta_t controls the gating for delta updates
        - The delta term (v_t - h_{t-1} @ k_t) computes what's new in v_t

    Algorithm Modes:
        - Chunked (default): Parallel within chunks using Neumann series,
          sequential across chunks. Best throughput for training.
        - Recurrent: Pure sequential scan. Lower memory, good for inference.
        - Single-step: Optimized path when seq_len=1 with initial_state.

    Args:
        query: Query tensor for attention
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        key: Key tensor for memory addressing
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        value: Value tensor to store in memory
            Shape: [batch, seq_len, num_heads, v_head_dim]
        beta: Per-token gating for delta updates (typically in [0, 1])
            Shape: [batch, seq_len, num_heads]
        decay: Per-token decay for memory retention (should be <= 0)
            Shape: [batch, seq_len, num_heads]
            If None, defaults to zeros (no decay, full retention)
        chunk_size: Block size for chunked algorithm (default: 256)
        initial_state: Optional initial memory state for incremental inference
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
        use_qk_l2norm: If True, apply L2 normalization to queries and keys
            before attention. Improves numerical stability (default: True)
        use_chunked: If True, use chunked algorithm; else use recurrent scan.
            Chunked is faster for training, recurrent for long inference

    Returns:
        Tuple of:
            - output: Attention output
                Shape: [batch, seq_len, num_heads, v_head_dim]
            - final_state: Final memory state for incremental inference
                Shape: [batch, num_heads, qk_head_dim, v_head_dim]

    Example:
        >>> import jax.numpy as jnp
        >>> from jax import random
        >>>
        >>> batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
        >>> key = random.PRNGKey(0)
        >>> q = random.normal(random.fold_in(key, 0), (batch, seq_len, num_heads, head_dim))
        >>> k = random.normal(random.fold_in(key, 1), (batch, seq_len, num_heads, head_dim))
        >>> v = random.normal(random.fold_in(key, 2), (batch, seq_len, num_heads, head_dim))
        >>> beta = jax.nn.sigmoid(random.normal(random.fold_in(key, 3), (batch, seq_len, num_heads)))
        >>> decay = random.normal(random.fold_in(key, 4), (batch, seq_len, num_heads)) * -0.01
        >>>
        >>> output, state = gated_delta_rule(q, k, v, beta, decay, chunk_size=16)
        >>> output.shape
        (2, 64, 8, 32)

    References:
        - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
    """
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)

    b = beta.transpose(0, 2, 1)
    d = decay.transpose(0, 2, 1) if decay is not None else None

    if query.shape[1] == 1 and initial_state is not None:
        out, final_state = _single_step_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            recurrent_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )
    elif use_chunked:
        out, final_state = _chunk_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
            seg_ids=seg_ids,
        )
    else:
        out, final_state = _recurrent_gdr_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            seg_ids=seg_ids,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )

    out = out.transpose(0, 2, 1, 3)
    return out, final_state
