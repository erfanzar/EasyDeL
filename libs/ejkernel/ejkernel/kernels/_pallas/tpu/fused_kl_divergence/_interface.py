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

"""Registry interface for TPU Pallas fused KL divergence.

The public operation layer resolves this registry entry when
``platform="pallas"`` on TPU. This wrapper performs type checking and delegates
to the implementation module, which contains both replicated-vocab and
vocab-parallel Pallas paths.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import fused_kl_divergence_pallas


@kernel_registry.register("fused_kl_divergence", Platform.PALLAS, Backend.TPU)
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
    """Dispatch fused KL divergence to the TPU Pallas implementation.

    Args:
        student_logits: Differentiable student logits with shape
            ``(..., vocab_size)``.
        teacher_logits: Teacher logits with the same shape. The Pallas custom
            VJP returns zero teacher gradients, matching distillation use.
        weights: Optional per-row mask/weights. Zero rows are skipped by the
            streaming Pallas kernels.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``.
        direction: ``"forward"`` or ``"reverse"`` for Pallas. ``"jsd"`` falls
            back to the XLA implementation.
        temperature: Softmax temperature; the public loss applies the standard
            ``T**2`` scaling after the fused core.
        beta: JSD interpolation parameter, used only by the XLA fallback.
        vocab_parallel_axis: Mesh axis name for vocab sharding in ``shard_map``.
        block_v: Optional vocab tile override.
        block_m: Optional row tile override.

    Returns:
        Per-row or reduced KL loss according to ``reduction``.
    """
    return fused_kl_divergence_pallas(
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
