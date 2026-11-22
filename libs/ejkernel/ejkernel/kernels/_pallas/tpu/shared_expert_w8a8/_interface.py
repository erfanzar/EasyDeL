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


"""Pallas TPU registry entry for the W8A8 shared expert (``shared_expert_w8a8``)."""

from __future__ import annotations

from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.fused_mlp._interface import resolve_mlp_combine
from ._pallas_impl import shared_expert_w8a8_pallas


@kernel_registry.register("shared_expert_w8a8", Platform.PALLAS, Backend.TPU, priority=10)
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
    """W8A8 gated MLP on a few rows in two Pallas TPU stages, bitwise equal to the XLA reference.

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
    return shared_expert_w8a8_pallas(rows, gate_up_codes, gate_up_scales, down_codes, down_scales, combine)


__all__ = ("shared_expert_w8a8",)
