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

"""TPU Pallas interface for fused conv-state decode."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import fused_conv_decode as _fused_conv_decode_pallas


@kernel_registry.register("fused_conv_decode", Platform.PALLAS, Backend.TPU)
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
    """Fused conv-state shift, depthwise convolution, and activation (TPU Pallas).

    Registered with ``kernel_registry`` under name ``"fused_conv_decode"`` for
    the Pallas platform and TPU backend. This is a thin wrapper that drops the
    ``activation`` argument and delegates to the Pallas implementation, which
    hard-codes SiLU.

    Args:
        conv_state: Current conv state ``[num_slots, conv_dim, d_conv]``, last
            axis holding the most recent ``d_conv`` tokens per channel.
        new_tokens: New token embeddings ``[num_slots, conv_dim]`` to append.
        kernel: Depthwise conv kernel ``[conv_dim, d_conv]`` (per-channel
            weights).
        output_dtype: Desired dtype for the convolution output array.
        activation: Activation function applied after convolution. Defaults to
            ``jax.nn.silu``. The Pallas kernel currently always applies SiLU;
            other activations are accepted for signature compatibility but
            ignored.

    Returns:
        A tuple ``(updated_state, conv_output)`` where ``updated_state`` is the
        shifted conv state ``[num_slots, conv_dim, d_conv]`` (same dtype as
        ``conv_state``) and ``conv_output`` is the post-SiLU convolution result
        ``[num_slots, conv_dim]`` with dtype ``output_dtype``.
    """
    del activation  # The Pallas kernel hard-codes SiLU for now.
    return _fused_conv_decode_pallas(
        conv_state,
        new_tokens,
        kernel,
        output_dtype=output_dtype,
    )
