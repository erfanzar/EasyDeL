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

"""Registry entry point for the XLA fused KL divergence."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import fused_kl_divergence as _fused_kl_xla


@kernel_registry.register("fused_kl_divergence", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_kl_divergence(
    student_logits: Float[Array, "... vocab_size"],
    teacher_logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    *,
    reduction: str = "mean",
    direction: str = "forward",
    temperature: float = 1.0,
    beta: float = 0.5,
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
) -> Float[Array, "..."]:
    """XLA reference matching the TileLang KL kernel contract.

    See :func:`ejkernel.kernels._tilelang.fused_kl_divergence.fused_kl_divergence`
    for parameter semantics. This implementation uses ``jax.nn.log_softmax``
    + autodiff. ``block_v`` / ``block_m`` are accepted for signature
    compatibility but ignored — XLA does its own tiling.
    """
    del block_v, block_m
    return _fused_kl_xla(
        student_logits,
        teacher_logits,
        weights,
        reduction=reduction,
        direction=direction,
        temperature=temperature,
        beta=beta,
        vocab_parallel_axis=vocab_parallel_axis,
    )


__all__ = ("fused_kl_divergence",)
