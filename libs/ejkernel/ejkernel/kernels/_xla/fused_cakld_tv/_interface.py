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

"""Registry entry point for the XLA fused dynamic-CAKLD + TV blend."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import fused_cakld_tv as _fused_cakld_tv_xla


@kernel_registry.register("fused_cakld_tv", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_cakld_tv(
    student_logits: Float[Array, "... vocab_size"],
    teacher_logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    *,
    reduction: str = "mean",
    temperature: float = 1.0,
    tv_weight: float = 0.0,
    entropy_weight: float = 0.0,
    entropy_alpha: float = 1.0,
    entropy_gate: float | None = None,
    vocab_parallel_axis: str | None = None,
) -> Float[Array, "..."]:
    """XLA implementation of the fused dynamic-CAKLD + TV distillation loss.

    See :func:`ejkernel.kernels._xla.fused_cakld_tv._xla_impl.fused_cakld_tv`
    for the math and parameter semantics. XLA does its own tiling, so there are
    no block-size parameters.
    """
    return _fused_cakld_tv_xla(
        student_logits,
        teacher_logits,
        weights,
        reduction=reduction,
        temperature=temperature,
        tv_weight=tv_weight,
        entropy_weight=entropy_weight,
        entropy_alpha=entropy_alpha,
        entropy_gate=entropy_gate,
        vocab_parallel_axis=vocab_parallel_axis,
    )


__all__ = ("fused_cakld_tv",)
