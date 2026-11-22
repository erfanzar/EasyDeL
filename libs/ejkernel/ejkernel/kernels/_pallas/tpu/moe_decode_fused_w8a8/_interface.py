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


"""Pallas TPU registry entry of the fused decode MoE segment (``moe_decode_fused_w8a8``)."""

from __future__ import annotations

from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.fused_mlp._interface import resolve_mlp_combine
from ._pallas_impl import moe_decode_fused_w8a8_pallas


@kernel_registry.register("moe_decode_fused_w8a8", Platform.PALLAS, Backend.TPU, priority=10)
def moe_decode_fused_w8a8(
    x: Float[Array, "tokens hidden"],
    expert_ids: Int[Array, "tokens k"],
    combine_weights: Float[Array, "tokens k"],
    gate_codes: Int[Array, "experts hidden inter"],
    gate_scales: Float[Array, "experts 1 inter"],
    up_codes: Int[Array, "experts hidden inter"],
    up_scales: Float[Array, "experts 1 inter"],
    down_codes: Int[Array, "experts inter hidden"],
    down_scales: Float[Array, "experts 1 hidden"],
    control: Int[Array, "3"],
    shared_gate_up_codes: Int[Array, "hidden two_sinter"] | None = None,
    shared_gate_up_scales: Float[Array, "1 two_sinter"] | None = None,
    shared_down_codes: Int[Array, "sinter hidden"] | None = None,
    shared_down_scales: Float[Array, "1 hidden"] | None = None,
    *,
    activation: str = "silu",
    act_params: tuple[float, ...] = (),
    shared_activation: str = "silu",
    shared_act_params: tuple[float, ...] = (),
) -> Float[Array, "tokens hidden"]:
    """Decode MoE segment in one Pallas TPU kernel (approximate against the XLA reference; see ``_pallas_impl``).

    Args:
        x: Token rows ``[T, H]`` (floating; its dtype sets the intermediate roundings).
        expert_ids: Global expert id per (token, slot) ``[T, k]``.
        combine_weights: Combine weight per (token, slot) ``[T, k]``.
        gate_codes: This shard's int8 gate codes ``[E, H, M]``.
        gate_scales: Per-channel gate scales ``[E, 1, M]``.
        up_codes: int8 up codes ``[E, H, M]``.
        up_scales: Per-channel up scales ``[E, 1, M]``.
        down_codes: int8 down codes ``[E, M, H]``.
        down_scales: Per-channel down scales ``[E, 1, H]``.
        control: int32 ``[3]``: global id of local expert 0, shared-expert row start, shared-expert row end.
        shared_gate_up_codes: Optional shared expert int8 ``[H, 2I]`` (``[gate | up]``).
        shared_gate_up_scales: Its per-channel scales ``[1, 2I]``.
        shared_down_codes: Optional shared expert int8 ``[I, H]``.
        shared_down_scales: Its per-channel scales ``[1, H]``.
        activation: Routed ``resolve_mlp_combine`` name.
        act_params: Its static parameters.
        shared_activation: Shared-expert ``resolve_mlp_combine`` name.
        shared_act_params: Its static parameters.

    Returns:
        float32 ``[T, H]``: the shard's weighted routed rows summed per token plus its shared-expert rows.
    """
    has_shared = shared_gate_up_codes is not None
    return moe_decode_fused_w8a8_pallas(
        x,
        expert_ids,
        combine_weights,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        down_codes,
        down_scales,
        control,
        shared_gate_up_codes,
        shared_gate_up_scales,
        shared_down_codes,
        shared_down_scales,
        resolve_mlp_combine(activation, tuple(act_params)),
        resolve_mlp_combine(shared_activation, tuple(shared_act_params)) if has_shared else None,
    )


__all__ = ("moe_decode_fused_w8a8",)
