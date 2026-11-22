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

"""Ragged causal conv1d XLA kernel interface and registry entry."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_causal_conv1d as _ragged_causal_conv1d_xla


@kernel_registry.register("ragged_causal_conv1d", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_causal_conv1d(
    x: Float[Array, "num_tokens conv_dim"],
    conv_state: Float[Array, "num_slots conv_dim d_conv_state"],
    kernel: Float[Array, "conv_dim d_conv_kernel"],
    query_start_loc: Int[Array, "num_requests_plus_1"],
    state_indices: Int[Array, "num_requests"],
    distribution: Int[Array, "three"],
    *,
    d_conv: int,
    apply_silu: bool = True,
) -> tuple[
    Float[Array, "num_tokens conv_dim"],
    Float[Array, "num_slots conv_dim d_conv_state"],
]:
    """Run ragged causal depthwise conv1d on the XLA backend.

    Registered into ``kernel_registry`` under name ``"ragged_causal_conv1d"`` for
    ``Platform.XLA`` / ``Backend.ANY`` and decorated with ``jaxtyping.jaxtyped`` +
    ``beartype`` so that the array shape/dtype contracts in the signature are
    runtime-checked. This is a thin dispatch shim that forwards every argument to
    the XLA implementation :func:`._xla_impl_fwd.ragged_causal_conv1d`, which
    applies a causal depthwise convolution independently per packed (ragged)
    request and refreshes each slot's rolling convolution state.

    Args:
        x: Packed input token stream, shape ``(num_tokens, conv_dim)`` float.
            Tokens for all requests are concatenated along the first axis.
        conv_state: Per-slot rolling convolution state, shape
            ``(num_slots, conv_dim, d_conv_state)`` float, where ``d_conv_state``
            equals ``d_conv``. Position ``d_conv - 1`` holds the most recent
            historical token and position ``0`` the oldest.
        kernel: Depthwise convolution kernel, shape ``(conv_dim, d_conv_kernel)``
            float, where ``d_conv_kernel`` equals ``d_conv``.
        query_start_loc: Cumulative per-request token offsets, shape
            ``(num_requests + 1,)`` int; ``query_start_loc[i+1] -
            query_start_loc[i]`` is the length of request ``i``.
        state_indices: Request-to-state-slot mapping, shape ``(num_requests,)``
            int; selects which ``conv_state`` slot each request reads from and
            writes back to.
        distribution: ``(decode_end, prefill_end, mixed_end)`` int triple of
            shape ``(3,)``; only ``distribution[2]`` (count of valid sequences)
            is consumed by the implementation.
        d_conv: Convolution window / state size (keyword-only). Must match
            ``kernel.shape[-1]`` and ``conv_state.shape[-1]``.
        apply_silu: If True (default), fuse ``jax.nn.silu`` into the output
            (Qwen3-Next / GatedDeltaNet convention); if False, return the raw
            linear convolution result (keyword-only).

    Returns:
        tuple: ``(output, updated_conv_state)`` where ``output`` has shape
        ``(num_tokens, conv_dim)`` (dtype of ``x``) and ``updated_conv_state``
        has the same shape/dtype as ``conv_state`` with the slots indexed by
        ``state_indices`` refreshed to the trailing ``d_conv`` tokens of each
        request.
    """
    return _ragged_causal_conv1d_xla(
        x,
        conv_state,
        kernel,
        query_start_loc,
        state_indices,
        distribution,
        d_conv=d_conv,
        apply_silu=apply_silu,
    )
