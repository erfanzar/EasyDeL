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


"""XLA registry entry (priority 0 fallback and reference) of the fused decode MoE segment (``moe_decode_fused_w8a8``)."""

from __future__ import annotations

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ..fused_mlp._interface import resolve_mlp_combine
from ..quantized_matmul._integer_quantization import quantize_rows
from ..shared_expert_w8a8._interface import shared_expert_w8a8

F32 = jnp.float32


@kernel_registry.register("moe_decode_fused_w8a8", Platform.XLA, Backend.ANY, priority=0)
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
    """Decode MoE segment as dense XLA ops over every local expert (the reference of the Pallas kernel).

    Per expert: per-row int8 gate/up projections of the token rows (rounded to the rows' dtype), the float32 combine
    rounded to the rows' dtype, per-row int8 down projection (rounded to the rows' dtype); the per-token combine
    weight of each expert (zero where the token is not routed there) then weights the rows in one float32 contraction.
    The shared expert (shared_expert_w8a8's XLA entry) runs on every row and keeps the rows in
    [control[1], control[2]).

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
    t = x.shape[0]
    e = gate_codes.shape[0]
    rows_dtype = x.dtype
    combine = resolve_mlp_combine(activation, tuple(act_params))
    codes, scale = quantize_rows(x, 8)

    def project(q, s, w, ws):
        spec = "tk,ekn->etn" if q.ndim == 2 else "etk,ekn->etn"
        acc = jnp.einsum(spec, q, w, preferred_element_type=jnp.int32)
        return ((acc.astype(F32) * s) * ws.reshape(e, 1, -1).astype(F32)).astype(rows_dtype).astype(F32)

    gate = project(codes, scale, gate_codes, gate_scales)
    up = project(codes, scale, up_codes, up_scales)
    hidden = combine(gate, up).astype(rows_dtype)
    hq, hs = jax.vmap(lambda r: quantize_rows(r, 8))(hidden)
    down = project(hq, hs, down_codes, down_scales)
    local = expert_ids.astype(jnp.int32) - control[0]
    wmat = jnp.sum(
        jnp.where(local[..., None] == jnp.arange(e, dtype=jnp.int32), combine_weights.astype(F32)[..., None], 0.0),
        axis=1,
    )
    out = jnp.einsum("te,eth->th", wmat, down, precision=jax.lax.Precision.HIGHEST)
    if shared_gate_up_codes is not None:
        shared = shared_expert_w8a8(
            x,
            shared_gate_up_codes,
            shared_gate_up_scales,
            shared_down_codes,
            shared_down_scales,
            activation=shared_activation,
            act_params=tuple(shared_act_params),
        )
        row = jnp.arange(t, dtype=jnp.int32)[:, None]
        out = out + jnp.where((row >= control[1]) & (row < control[2]), shared, 0.0)
    return out


__all__ = ("moe_decode_fused_w8a8",)
