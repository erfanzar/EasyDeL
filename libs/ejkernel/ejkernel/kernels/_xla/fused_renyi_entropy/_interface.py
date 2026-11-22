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

"""Registry entry point for the XLA fused Renyi entropy."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl import fused_renyi_entropy as _fused_renyi_xla


@kernel_registry.register("fused_renyi_entropy", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_renyi_entropy(
    logits: Float[Array, "... vocab_size"],
    weights: Float[Array, "..."] | None = None,
    *,
    alpha: float = 1.0,
    reduction: str = "mean",
    temperature: float = 1.0,
    vocab_parallel_axis: str | None = None,
) -> Float[Array, "..."]:
    """XLA implementation of the fused per-row Renyi entropy.

    See :func:`ejkernel.kernels._xla.fused_renyi_entropy._xla_impl.fused_renyi_entropy`
    for the math and parameter semantics. XLA does its own tiling, so there are
    no block-size parameters.
    """
    return _fused_renyi_xla(
        logits,
        weights,
        alpha=alpha,
        reduction=reduction,
        temperature=temperature,
        vocab_parallel_axis=vocab_parallel_axis,
    )


__all__ = ("fused_renyi_entropy",)
