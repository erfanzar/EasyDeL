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
"""Registry entry point for the XLA Lightning Attention kernel.

This module registers ``lightning_attn`` under the ``(Platform.XLA,
Backend.ANY)`` key so that the kernel registry can dispatch calls on any
XLA-compatible device (CPU, GPU, TPU) to the pure-JAX recurrent
implementation in ``_xla_impl_fwd``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import lightning_attn as _lightning_attn_impl


@kernel_registry.register("lightning_attn", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def lightning_attn(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    block_k: int = 64,
    block_v: int = 64,
    num_warps: int = 4,
    num_stages: int = 1,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Compute Lightning Attention via the XLA recurrent kernel.

    Thin registry wrapper.  Builds the layer-specific per-head decay vector
    ``g_gamma`` and forwards all arguments to the underlying ``recurrent``
    implementation (see ``_xla_impl_fwd.lightning_attn`` for the full
    algorithm description).

    Args:
        query: Query tensor of shape ``[batch, seq_len, num_heads, qk_head_dim]``.
            When ``cu_seqlens`` is provided the batch dimension must be 1 and
            ``seq_len`` is the total number of packed tokens.
        key: Key tensor of shape ``[batch, seq_len, num_kv_heads, qk_head_dim]``.
        value: Value tensor of shape ``[batch, seq_len, num_kv_heads, v_head_dim]``.
        layer_idx: 0-based index of the current transformer layer.  Together
            with ``num_layers`` it determines the per-head decay magnitudes:
            larger ``layer_idx`` values produce smaller (closer to zero) decay.
        num_layers: Total number of transformer layers in the model.
        softmax_scale: Scaling factor applied to query vectors before the
            recurrent update.  Defaults to ``1 / sqrt(qk_head_dim)``.
        initial_state: Optional initial recurrent hidden state of shape
            ``[..., num_heads, qk_head_dim, v_head_dim]``.  When
            ``cu_seqlens`` is provided the leading dimension must equal the
            number of packed sequences (``len(cu_seqlens) - 1``).
        reverse: If ``True``, the sequence is scanned right-to-left.
        cu_seqlens: Cumulative sequence lengths ``[0, l1, l1+l2, ...]`` of
            shape ``[num_seqs + 1]``.  Enables packed variable-length inputs
            with batch size 1.
        block_k: Accepted for operation-level config parity with Triton; XLA
            chooses its own tiling.
        block_v: Accepted for operation-level config parity with Triton; XLA
            chooses its own tiling.
        num_warps: Accepted for operation-level config parity with Triton.
        num_stages: Accepted for operation-level config parity with Triton.

    Returns:
        Tuple ``(output, final_state)`` where:
            - ``output``: ``[batch, seq_len, num_heads, v_head_dim]``
            - ``final_state``: ``[..., num_heads, qk_head_dim, v_head_dim]``

    Raises:
        ValueError: If ``cu_seqlens`` is provided and ``query.shape[0] != 1``.
        ValueError: If ``cu_seqlens`` is provided and
            ``initial_state.shape[0] != len(cu_seqlens) - 1``.

    Example:
        >>> import jax.numpy as jnp
        >>> q = jnp.ones((2, 100, 8, 64))
        >>> k = jnp.ones((2, 100, 8, 64))
        >>> v = jnp.ones((2, 100, 8, 64))
        >>> output, state = lightning_attn(q, k, v, layer_idx=5, num_layers=24)
        >>> output.shape
        (2, 100, 8, 64)
    """
    return _lightning_attn_impl(
        query, key, value, layer_idx, num_layers, softmax_scale, initial_state, reverse, cu_seqlens
    )


__all__ = ("lightning_attn",)
