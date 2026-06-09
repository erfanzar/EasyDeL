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

"""Kernel Delta Attention (KDA) public interface for XLA backend.

This module provides the public API for KDA, a linear attention mechanism that
maintains a [head_dim, v_head_dim] memory matrix per head and uses a delta-rule
update to selectively store and overwrite information.

Registered kernel keys (both map to the same implementation):
    - ``"kda"``
    - ``"kernel_delta_attention"``

The module also exports ``kda_decay``, a utility that computes Mamba-style
per-token decay values from learnable ``A_log`` and ``dt_bias`` parameters.

Public tensor convention: [batch, seq_len, num_heads, dim].  Internally,
tensors are transposed to [batch, num_heads, seq_len, dim] before dispatching
to the implementation functions.

Supported computation modes (dispatched by ``kernel_delta_attention``):
    - Single-step (seq_len=1 + initial_state): fast path for autoregressive decoding.
    - Chunked (use_chunked=True, default): parallel intra-chunk with sequential
      inter-chunk state propagation.
    - Recurrent (use_chunked=False): pure sequential scan, lowest memory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _chunk_kda_fwd, _recurrent_kda_fwd, _single_step_kda_fwd


def kda_decay(
    gate: Float[Array, "batch seq_len num_heads"],
    A_log: Float[Array, "num_heads"],
    dt_bias: Float[Array, "num_heads"],
) -> Float[Array, "batch seq_len num_heads"]:
    """Compute KDA per-token decay from gate, A_log, and dt_bias.

    This function computes the decay term used in Kernel Delta Attention,
    following the Mamba-style discretization where decay controls how much
    of the previous state is retained.

    The computation is:
        A = -exp(A_log)  # Ensure A is negative for stability
        decay = A * softplus(gate + dt_bias)

    Args:
        gate: Gating signal from input projection
            Shape: [batch, seq_len, num_heads]
        A_log: Learnable log-scale decay parameter (typically initialized near 0)
            Shape: [num_heads]
        dt_bias: Learnable bias added to gate before softplus
            Shape: [num_heads]

    Returns:
        Per-token decay values (always <= 0 for stable state decay)
            Shape: [batch, seq_len, num_heads]

    Example:
        >>> gate = jnp.zeros((2, 10, 4))  # batch=2, seq_len=10, num_heads=4
        >>> A_log = jnp.zeros((4,))
        >>> dt_bias = jnp.zeros((4,))
        >>> decay = kda_decay(gate, A_log, dt_bias)
        >>> assert jnp.all(decay <= 0)  # Decay is always non-positive
    """
    A = -jnp.exp(A_log.astype(jnp.float32))
    return A[None, None, :] * jax.nn.softplus(gate.astype(jnp.float32) + dt_bias.astype(jnp.float32))


@kernel_registry.register("kda", Platform.XLA, Backend.ANY)
@kernel_registry.register("kernel_delta_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def kernel_delta_attention(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_heads v_head_dim"],
    beta: Float[Array, "batch seq_len num_heads"],
    decay: Float[Array, "batch seq_len num_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads qk_head_dim v_head_dim"] | None = None,
    use_qk_l2norm: bool = True,
    use_chunked: bool = True,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "batch num_heads qk_head_dim v_head_dim"],
]:
    """Kernel Delta Attention (KDA) linear attention using XLA backend.

    KDA is a linear attention variant that maintains a key-value memory matrix
    and uses delta updates to efficiently store and retrieve information. It
    combines ideas from linear attention and delta networks for O(N) complexity.

    The core recurrence is:
        h_t = exp(decay_t) * h_{t-1} + k_t ⊗ (beta_t * (v_t - h_{t-1} @ k_t))
        o_t = h_t @ q_t

    Where:
        - h_t is the [head_dim, value_dim] memory matrix per head
        - exp(decay_t) controls memory retention (decay <= 0 for stability)
        - beta_t controls the learning rate for delta updates
        - The delta term (v_t - h_{t-1} @ k_t) computes what's new in v_t

    Algorithm Modes:
        - Chunked (default): Parallel within chunks, sequential across chunks.
          Better throughput for training with moderate sequence lengths.
        - Recurrent: Pure sequential scan. Lower memory, good for inference.
        - Single-step: Optimized path when seq_len=1 with initial_state.

    Args:
        query: Query tensor for attention
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        key: Key tensor for memory addressing
            Shape: [batch, seq_len, num_heads, qk_head_dim]
        value: Value tensor to store in memory
            Shape: [batch, seq_len, num_heads, v_head_dim]
        beta: Per-token learning rate for delta updates (typically in [0, 1])
            Shape: [batch, seq_len, num_heads]
        decay: Per-token decay for memory retention (should be <= 0)
            Shape: [batch, seq_len, num_heads]
            If None, defaults to zeros (no decay, full retention)
        softmax_scale: Scaling factor for queries. If None, uses head_dim^-0.5
        chunk_size: Block size for chunked algorithm (default: 64)
        initial_state: Optional initial memory state for incremental inference
            Shape: [batch, num_heads, qk_head_dim, v_head_dim]
        use_qk_l2norm: If True, apply L2 normalization to queries and keys
            before attention. Improves stability (default: True)
        use_chunked: If True, use chunked algorithm; else use recurrent scan
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
        >>> # Basic usage
        >>> batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
        >>> key = random.PRNGKey(0)
        >>> q = random.normal(random.fold_in(key, 0), (batch, seq_len, num_heads, head_dim))
        >>> k = random.normal(random.fold_in(key, 1), (batch, seq_len, num_heads, head_dim))
        >>> v = random.normal(random.fold_in(key, 2), (batch, seq_len, num_heads, head_dim))
        >>> beta = jax.nn.sigmoid(random.normal(random.fold_in(key, 3), (batch, seq_len, num_heads)))
        >>> decay = random.normal(random.fold_in(key, 4), (batch, seq_len, num_heads)) * -0.01
        >>>
        >>> output, state = kernel_delta_attention(q, k, v, beta, decay, chunk_size=16)
        >>> output.shape
        (2, 64, 8, 32)
        >>>
        >>> # Incremental inference
        >>> q_new = random.normal(random.fold_in(key, 5), (batch, 1, num_heads, head_dim))
        >>> k_new = random.normal(random.fold_in(key, 6), (batch, 1, num_heads, head_dim))
        >>> v_new = random.normal(random.fold_in(key, 7), (batch, 1, num_heads, head_dim))
        >>> beta_new = jax.nn.sigmoid(random.normal(random.fold_in(key, 8), (batch, 1, num_heads)))
        >>> decay_new = random.normal(random.fold_in(key, 9), (batch, 1, num_heads)) * -0.01
        >>>
        >>> output_new, state_new = kernel_delta_attention(
        ...     q_new, k_new, v_new, beta_new, decay_new, initial_state=state
        ... )

    References:
        - Delta Networks: https://arxiv.org/abs/1612.04859
        - Linear Transformers: https://arxiv.org/abs/2006.16236
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)

    b = beta.transpose(0, 2, 1)
    d = decay.transpose(0, 2, 1) if decay is not None else None

    if query.shape[1] == 1 and initial_state is not None:
        out, final_state = _single_step_kda_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            softmax_scale=softmax_scale,
            recurrent_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )
    elif use_chunked:
        out, final_state = _chunk_kda_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            softmax_scale=softmax_scale,
            chunk_size=chunk_size,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )
    else:
        out, final_state = _recurrent_kda_fwd(
            query=q,
            key=k,
            value=v,
            beta=b,
            decay=d,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            use_qk_l2norm=use_qk_l2norm,
        )

    out = out.transpose(0, 2, 1, 3)
    return out, final_state


kda = kernel_delta_attention
