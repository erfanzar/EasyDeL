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

"""XLA interface for fused conv-state decode."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import fused_conv_decode as _fused_conv_decode_xla


@kernel_registry.register("fused_conv_decode", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_conv_decode(
    conv_state: Float[Array, "num_slots conv_dim d_conv"],
    new_tokens: Float[Array, "num_slots conv_dim"],
    kernel: Float[Array, "conv_dim d_conv"],
    *,
    output_dtype: jnp.dtype,
    activation: Callable[[Array], Array] | None = jax.nn.silu,
) -> tuple[
    Float[Array, "num_slots conv_dim d_conv"],
    Float[Array, "num_slots conv_dim"],
]:
    """Fused conv-state shift, depthwise convolution, and activation (XLA reference).

    Registered entry point for the ``"fused_conv_decode"`` kernel on the XLA platform with the
    ``ANY`` backend (see :func:`kernel_registry.register`). The body delegates directly to the
    pure-JAX reference implementation :func:`_xla_impl_fwd.fused_conv_decode`; this wrapper exists
    to attach registry metadata and runtime shape/dtype checking via ``jaxtyping.jaxtyped`` +
    ``beartype``. It performs one decode step: the per-slot convolution cache is shifted left by
    one position, ``new_tokens`` is appended as the newest column, then a depthwise (per-channel)
    convolution over the ``d_conv`` window is computed and optionally passed through ``activation``.

    Args:
        conv_state: Current convolution cache, shape ``[num_slots, conv_dim, d_conv]``. The last
            axis is the temporal window, ordered oldest-to-newest.
        new_tokens: New per-channel token values for this step, shape ``[num_slots, conv_dim]``.
        kernel: Depthwise convolution weights, shape ``[conv_dim, d_conv]`` (one ``d_conv``-length
            filter per channel; broadcast across slots).
        output_dtype: Dtype to cast the convolution output (second tuple element) to.
        activation: Elementwise activation applied to the convolution output, or ``None`` to skip
            activation. Defaults to ``jax.nn.silu``.

    Returns:
        A tuple ``(updated_state, conv_output)`` where ``updated_state`` is the shifted cache of
        shape ``[num_slots, conv_dim, d_conv]`` (dtype unchanged from ``conv_state``) and
        ``conv_output`` is the activated convolution result of shape ``[num_slots, conv_dim]`` cast
        to ``output_dtype``.
    """
    return _fused_conv_decode_xla(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=output_dtype,
        activation=activation,
    )
