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

"""XLA registry entry (priority 0 fallback) for the experts-over-tp local expert FFN."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import moe_local_ffn_xla


@kernel_registry.register("moe_local_ffn", Platform.XLA, Backend.ANY)
def moe_local_ffn(
    x: Float[Array, "tokens hidden"],
    weights: Float[Array, "tokens k"],
    experts: Int[Array, "tokens k"],
    w_gate: Float[Array, "experts hidden inter"],
    w_up: Float[Array, "experts hidden inter"],
    w_down: Float[Array, "experts inter hidden"],
    activation: Callable[[Array, Array], Array] | None = None,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
) -> Float[Array, "tokens hidden"]:
    """This device's partial MoE output over its local experts (dropless dynamic row windows, custom VJP).

    Args:
        x: ``[tokens, hidden]`` token rows (replicated over the expert-sharding axis).
        weights: ``[tokens, k]`` routing (combine) weights; differentiable.
        experts: ``[tokens, k]`` local expert ids; ids outside ``[0, experts)`` are other devices' routes.
        w_gate: ``[experts, hidden, inter]`` gate projection.
        w_up: ``[experts, hidden, inter]`` up projection.
        w_down: ``[experts, inter, hidden]`` down projection.
        activation: ``(gate, up) -> h``; default ``silu(gate) * up``. Must close over no traced values.
        tokens_per_chunk: Tokens per outer chunk (default: all tokens).
        rows_per_window: Rows per inner window (default: every route of a chunk). The windows cover every route
            (none is dropped); those past the actual local row count are skipped.

    Returns:
        ``[tokens, hidden]`` in ``x.dtype``: ``sum_{j local} weights[t, j] * FFN_{experts[t, j]}(x[t])``.
    """
    return moe_local_ffn_xla(x, weights, experts, w_gate, w_up, w_down, activation, tokens_per_chunk, rows_per_window)


__all__ = ("moe_local_ffn",)
