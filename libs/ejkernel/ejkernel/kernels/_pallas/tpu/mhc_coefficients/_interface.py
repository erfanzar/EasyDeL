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

"""TPU entry point for packed mHC coefficients."""

import math

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.mhc_coefficients._interface import mhc_coefficients as reference
from ...._xla.mhc_coefficients._interface import validate_inputs
from ._pallas_impl import coefficients


@kernel_registry.register("mhc_coefficients", Platform.PALLAS, Backend.TPU)
def mhc_coefficients(logits, base, scale, hc_mult: int = 4, n_iters: int = 20, eps: float = 1e-6):
    """Compute coefficients, using packed TPU kernels for four streams.

    Inputs and output shapes match the XLA backend. The optimized path supports
    first-order reverse AD only. Other stream counts, empty inputs, and iteration
    counts above 20 use the XLA reference. Explicit ``platform='xla'`` on the
    public operation supports forward-mode and higher-order differentiation.
    """
    validate_inputs(logits, base, scale, hc_mult, n_iters, eps)
    shape = logits.shape[:-1]
    tokens = math.prod(shape)
    if hc_mult != 4 or n_iters > 20 or tokens == 0:
        return reference(logits, base, scale, hc_mult, n_iters, eps)
    pre, post, matrix = coefficients(logits.reshape(tokens, 24), base, scale, n_iters, eps)
    return pre.reshape(*shape, 4), post.reshape(*shape, 4), matrix.reshape(*shape, 4, 4)
