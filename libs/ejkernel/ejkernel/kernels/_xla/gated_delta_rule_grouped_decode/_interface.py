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

"""Registry interface for the XLA grouped gated-delta-rule (GDR) decode kernel.

This module exposes the public, registry-registered entry point for one grouped GDR decode
step. The function is decorated so that:

    - :func:`kernel_registry.register` records it under the ``"gated_delta_rule_grouped_decode"``
      algorithm name for ``Platform.XLA`` and ``Backend.ANY`` (matching every hardware target),
      making it the XLA fallback the registry dispatches to.
    - :func:`jaxtyping.jaxtyped` with the :func:`beartype` checker enforces the declared array
      shapes and dtypes at call time.

The actual numerics are delegated to the pure-JAX implementation in :mod:`._xla_impl_fwd`;
this module only adds registration and runtime type checking.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import gated_delta_rule_grouped_decode as _gated_delta_rule_grouped_decode_xla


@kernel_registry.register("gated_delta_rule_grouped_decode", Platform.XLA, Backend.ANY)
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
    """Run one grouped gated-delta-rule decode step via the XLA/JAX reference implementation.

    This is the registry-facing, runtime-typechecked wrapper around
    :func:`ejkernel.kernels._xla.gated_delta_rule_grouped_decode._xla_impl_fwd.gated_delta_rule_grouped_decode`.
    It performs no computation of its own beyond forwarding the arguments; all numerics and
    shape conventions are defined by that implementation. ``num_v_heads`` is implied to equal
    ``num_k_heads * expand_ratio`` (each key head is shared by ``expand_ratio`` value heads).

    Args:
        query: Query for the new token, ``[batch, num_k_heads, head_dim]``. Any float dtype;
            it is cast to ``recurrent_state.dtype`` and scaled by ``head_dim ** -0.5`` inside
            the implementation.
        key: Key for the new token, ``[batch, num_k_heads, head_dim]``. Any float dtype; cast
            to ``recurrent_state.dtype``.
        value: Value for the new token, ``[batch, num_k_heads, expand_ratio, value_dim]``. The
            ``expand_ratio`` axis carries the per-group expansion shared across the key head.
        beta: Per-group delta-rule gating coefficient, ``[batch, num_k_heads, expand_ratio]``,
            scaling the write of ``(value - state-read)`` into the state.
        decay: Optional per-group log-space decay, ``[batch, num_k_heads, expand_ratio]``. When
            given, the state is multiplied by ``exp(decay)`` before the update; when ``None`` no
            decay is applied.
        recurrent_state: Current linear recurrent state,
            ``[batch, num_v_heads, head_dim, value_dim]``. Its dtype defines the compute dtype,
            and the returned new state has the same dtype/shape.

    Returns:
        tuple: ``(output, new_state)`` where ``output`` is the decoded attention output of shape
        ``[batch, num_v_heads, value_dim]`` and ``new_state`` is the updated recurrent state of
        shape ``[batch, num_v_heads, head_dim, value_dim]``, both cast to
        ``recurrent_state.dtype``.
    """
    return _gated_delta_rule_grouped_decode_xla(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
    )
