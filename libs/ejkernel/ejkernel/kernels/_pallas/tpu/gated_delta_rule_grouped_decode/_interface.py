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

"""TPU Pallas interface for grouped GDR single-step decode."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import gated_delta_rule_grouped_decode as _gated_delta_rule_grouped_decode_pallas


@kernel_registry.register("gated_delta_rule_grouped_decode", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def gated_delta_rule_grouped_decode(
    query: Float[Array, "batch num_k_heads head_dim"],
    key: Float[Array, "batch num_k_heads head_dim"],
    value: Float[Array, "batch num_k_heads expand_ratio value_dim"],
    beta: Float[Array, "batch num_k_heads expand_ratio"],
    decay: Float[Array, "batch num_k_heads expand_ratio"] | None,
    recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
) -> tuple[
    Float[Array, "batch num_v_heads value_dim"],
    Float[Array, "batch num_v_heads head_dim value_dim"],
]:
    """Grouped GDR (gated delta-rule) decode step using the TPU Pallas kernel.

    Registered with ``kernel_registry`` under name
    ``"gated_delta_rule_grouped_decode"`` for the Pallas platform and TPU
    backend. When ``decay`` is ``None`` the wrapper substitutes a zero decay
    (no decay) before delegating to the Pallas implementation.

    Args:
        query: Query tensor ``[batch, num_k_heads, head_dim]``.
        key: Key tensor ``[batch, num_k_heads, head_dim]``.
        value: Value tensor ``[batch, num_k_heads, expand_ratio, value_dim]``.
        beta: Gating coefficients ``[batch, num_k_heads, expand_ratio]``.
        decay: Optional log-space decay factors
            ``[batch, num_k_heads, expand_ratio]``. If ``None``, treated as
            zeros (no decay).
        recurrent_state: Current recurrent state
            ``[batch, num_v_heads, head_dim, value_dim]``, where
            ``num_v_heads == num_k_heads * expand_ratio``.

    Returns:
        A tuple ``(output, new_state)`` where ``output`` is the decode output
        ``[batch, num_v_heads, value_dim]`` and ``new_state`` is the updated
        recurrent state ``[batch, num_v_heads, head_dim, value_dim]``, both with
        the dtype of ``recurrent_state``.
    """
    if decay is None:
        decay = jnp.zeros_like(beta)
    return _gated_delta_rule_grouped_decode_pallas(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )
