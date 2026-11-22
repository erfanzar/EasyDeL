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


"""Pallas TPU registry entry for the fused KDA decode step (conv window update, conv, L2 norm, recurrence)."""

from __future__ import annotations

import typing as tp

from jaxtyping import Array, Bool, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.ragged_kda_conv_decode._xla_impl import ragged_kda_conv_decode_xla, validate_shapes
from ._pallas_impl import ragged_kda_conv_decode_pallas, unsupported_reason


@kernel_registry.register("ragged_kda_conv_decode", Platform.PALLAS, Backend.TPU, priority=10)
def ragged_kda_conv_decode(
    query: Float[Array, "num_tokens key_dim"],
    key: Float[Array, "num_tokens key_dim"],
    value: Float[Array, "num_tokens value_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads head_dim"],
    conv_weights: tuple[Array, Array, Array],
    conv_pools: tuple[Array, Array, Array],
    recurrent_state: Float[Array, "num_slots num_heads head_dim v_dim"],
    token_slots: Int[Array, "num_tokens"],
    token_init: Bool[Array, "num_tokens"],
    *,
    conv_dtype: tp.Any = None,
    use_qk_l2norm: bool = True,
    block_tokens: int = 0,
) -> tuple[Float[Array, "num_tokens num_heads v_dim"], tuple[Array, Array, Array, Array]]:
    """Fused KDA decode step (Pallas TPU, pools updated in place); same contract as the XLA reference.

    Calls the kernel does not take (head sizes that are not multiples of the 128 lanes, dtypes other than bfloat16 /
    float32, a token count no VMEM block fits) run the XLA reference instead.

    Args:
        query: Raw (pre-conv) queries ``[T, H*K]``.
        key: Raw keys ``[T, H*K]``.
        value: Raw values ``[T, H*V]``.
        beta: Write strengths ``[T, H]`` (sigmoid applied).
        decay: Per-channel log-decay ``[T, H, K]``.
        conv_weights: Depthwise conv kernels ``(q, k, v)``, each ``[dim, d_conv]``.
        conv_pools: Conv windows ``(q, k, v)``, each ``[S, dim, d_conv]``.
        recurrent_state: State pool ``[S, H, K, V]``.
        token_slots: ``[T]`` pool slot advanced by token ``t`` (out of range: the token outputs zeros, no write).
        token_init: ``[T]`` whether that slot's windows and state carry over (``False``: zeros).
        conv_dtype: Dtype of the conv outputs (``None``: the query dtype).
        use_qk_l2norm: L2-normalise queries and keys.
        block_tokens: Tokens per grid step (``0``: from the VMEM budget).

    Returns:
        ``(out [T, H, V] fp32, (q_pool, k_pool, v_pool, recurrent_state))``.
    """
    validate_shapes(query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state, token_slots, token_init)
    conv_dtype = query.dtype if conv_dtype is None else conv_dtype
    reason = unsupported_reason(
        query, key, value, beta, decay, conv_weights, conv_pools, recurrent_state, conv_dtype, block_tokens=block_tokens
    )
    if reason is not None:
        return ragged_kda_conv_decode_xla(
            query,
            key,
            value,
            beta,
            decay,
            conv_weights,
            conv_pools,
            recurrent_state,
            token_slots,
            token_init,
            conv_dtype=conv_dtype,
            use_qk_l2norm=use_qk_l2norm,
        )
    return ragged_kda_conv_decode_pallas(
        query,
        key,
        value,
        beta,
        decay,
        tuple(conv_weights),
        tuple(conv_pools),
        recurrent_state,
        token_slots,
        token_init,
        conv_dtype=conv_dtype,
        use_qk_l2norm=use_qk_l2norm,
        block_tokens=int(block_tokens),
    )


__all__ = ("ragged_kda_conv_decode",)
