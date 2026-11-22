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


"""XLA registry entry (priority 0 fallback and reference) for the W8A8 shared expert (``shared_expert_w8a8``)."""

from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ..fused_mlp._interface import resolve_mlp_combine
from ..grouped_matmul_quant._channelwise import grouped_matmul_channelwise


@kernel_registry.register("shared_expert_w8a8", Platform.XLA, Backend.ANY, priority=0)
def shared_expert_w8a8(
    rows: Float[Array, "rows hidden"],
    gate_up_codes: Int[Array, "hidden two_inter"],
    gate_up_scales: Float[Array, "1 two_inter"],
    down_codes: Int[Array, "inter hidden"],
    down_scales: Float[Array, "1 hidden"],
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
) -> Float[Array, "rows hidden"]:
    """W8A8 gated MLP on a few rows: two one-group A8 ``grouped_matmul_channelwise`` calls around the combine.

    The gate/up projection returns the rows' dtype, ``combine(gate, up)`` runs in float32 and is cast back to the rows'
    dtype, and the down projection returns float32 (EasyDeL's folded shared expert, ``_shared_expert_rows``).

    Args:
        rows: Floating rows ``[r, H]``.
        gate_up_codes: int8 ``[H, 2I]`` (``[gate | up]``).
        gate_up_scales: Per-channel scales ``[1, 2I]``.
        down_codes: int8 ``[I, H]``.
        down_scales: Per-channel scales ``[1, H]``.
        activation: ``resolve_mlp_combine`` name.
        act_params: Its static parameters.

    Returns:
        float32 ``[r, H]``.
    """
    combine = resolve_mlp_combine(activation, tuple(act_params))
    sizes = jnp.full((1,), rows.shape[0], dtype=jnp.int32)
    gate_up = grouped_matmul_channelwise(
        rows,
        gate_up_codes[None],
        gate_up_scales.reshape(1, 1, -1).astype(jnp.float32),
        sizes,
        activation_bits=8,
        preferred_element_type=rows.dtype,
    )
    gate, up = jnp.split(gate_up, 2, axis=-1)
    hidden = combine(gate.astype(jnp.float32), up.astype(jnp.float32)).astype(rows.dtype)
    return grouped_matmul_channelwise(
        hidden,
        down_codes[None],
        down_scales.reshape(1, 1, -1).astype(jnp.float32),
        sizes,
        activation_bits=8,
        preferred_element_type=jnp.float32,
    )


__all__ = ("shared_expert_w8a8",)
