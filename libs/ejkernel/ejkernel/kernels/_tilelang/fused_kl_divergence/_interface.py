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

"""tile-lang fused forward-KL public interface.

Registers ``fused_kl_divergence`` for ``Platform.TILELANG / Backend.GPU``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import fused_kl_divergence_tilelang


@kernel_registry.register("fused_kl_divergence", Platform.TILELANG, Backend.GPU)
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
    """Tile-lang fused KL divergence with forward / reverse / JSD modes.

    Args:
        student_logits: ``(..., V)`` student logits. Differentiable.
        teacher_logits: ``(..., V)`` teacher logits (detached).
        weights: Optional ``logits.shape[:-1]`` per-token weights. Pass
            ``completion_mask`` for assistant-only loss.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        direction: ``"forward"`` (``KL(p_t‖p_s)``, default), ``"reverse"``
            (``KL(p_s‖p_t)``), or ``"jsd"`` (Jensen-Shannon mixture).
        temperature: Softmax temperature ``T`` (Hinton distillation
            convention — loss is scaled by ``T²``).
        beta: JSD mixture coefficient; ignored unless ``direction='jsd'``.
        vocab_parallel_axis: TP mesh axis (forward + ``T=1`` only).
    """
    if student_logits.shape != teacher_logits.shape:
        raise EjkernelRuntimeError(
            f"fused_kl_divergence: shape mismatch student={student_logits.shape} vs teacher={teacher_logits.shape}"
        )
    if weights is not None and weights.shape != student_logits.shape[:-1]:
        raise EjkernelRuntimeError(
            f"fused_kl_divergence: weights.shape={weights.shape} "
            f"must equal logits.shape[:-1]={student_logits.shape[:-1]}"
        )
    return fused_kl_divergence_tilelang(
        student_logits,
        teacher_logits,
        weights,
        reduction=reduction,
        direction=direction,
        temperature=temperature,
        beta=beta,
        vocab_parallel_axis=vocab_parallel_axis,
        block_v=block_v,
        block_m=block_m,
    )


__all__ = ["fused_kl_divergence"]
