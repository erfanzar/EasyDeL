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
"""Public interface for the RWKV-4 XLA kernel.

Registers ``rwkv4`` in the kernel registry under ``Platform.XLA``/``Backend.ANY``
and re-exports it with jaxtyping/beartype runtime shape checks.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float
from ._xla_impl_fwd import rwkv4 as _rwkv4_impl


@kernel_registry.register("rwkv4", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
    *,
    block_c: int = 64,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 recurrent time-mix attention using XLA backend.
    Implements the RWKV-4 time-mixing mechanism which computes a weighted
    key-value (WKV) attention using an O(N) recurrent formulation. This
    replaces the O(N^2) attention matrix with a sequential scan that
    maintains running statistics.
    The mechanism uses exponential decay controlled by `w` and adds a
    bonus term `u` for the current token's contribution. The computation
    is numerically stabilized using log-sum-exp tracking.
    Args:
        w: Time-decay parameter in log space. The actual decay used is
            -exp(w), so larger w values mean faster decay (less memory
            of past tokens). Shape: [channels]
        u: Time-mix bonus/bias that controls how much the current token
            contributes relative to historical context. Shape: [channels]
        k: Key tensor representing input features to match against.
            Shape: [batch, seq_len, channels]
        v: Value tensor representing content to aggregate.
            Shape: [batch, seq_len, channels]
        state: Optional initial state tuple packed as [B, 3, C] where:
            - state[:, 0, :] = alpha (numerator accumulator)
            - state[:, 1, :] = beta (denominator accumulator)
            - state[:, 2, :] = eps (log-scale normalization)
            If None, initializes with alpha=0, beta=0, eps=-1e30.
    Returns:
        Tuple of:
            - wkv: Weighted key-value output matching input dtype.
                Shape: [batch, seq_len, channels]
            - final_state: Final state for continuation, always float32.
                Shape: [batch, 3, channels]
    Raises:
        ValueError: If k and v shapes don't match, or dimension mismatches.
    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> # Basic usage
        >>> batch, seq_len, channels = 2, 100, 512
        >>> w = -jnp.ones(channels) * 0.5  # moderate decay
        >>> u = jnp.zeros(channels)
        >>> k = jnp.ones((batch, seq_len, channels))
        >>> v = jnp.ones((batch, seq_len, channels))
        >>>
        >>> wkv, state = rwkv4(w, u, k, v)
        >>> wkv.shape
        (2, 100, 512)
        >>>
        >>> # Incremental inference with state
        >>> k_next = jnp.ones((batch, 1, channels))
        >>> v_next = jnp.ones((batch, 1, channels))
        >>> wkv_next, state_next = rwkv4(w, u, k_next, v_next, state=state)
    """
    return _rwkv4_impl(w, u, k, v, state)


__all__ = ("rwkv4",)
