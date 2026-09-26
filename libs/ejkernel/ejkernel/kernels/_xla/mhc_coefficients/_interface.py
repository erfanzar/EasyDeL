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

"""XLA reference for manifold hyper-connection coefficients."""

import math
import operator

import jax
from jax import numpy as jnp

from ..._registry import Backend, Platform, kernel_registry


def validate_inputs(logits, base, scale, hc_mult, n_iters, eps):
    """Validate static shapes/settings without inspecting traced array values."""
    hc_mult, n_iters = operator.index(hc_mult), operator.index(n_iters)
    if hc_mult < 1 or n_iters < 1:
        raise ValueError("hc_mult and n_iters must be positive integers")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    width = hc_mult * (hc_mult + 2)
    if logits.ndim < 1 or logits.shape[-1] != width or base.shape != (width,) or scale.shape != (3,):
        raise ValueError(f"Expected logits[..., {width}], base[{width}], scale[3]")
    if any(x.dtype != jnp.float32 for x in (logits, base, scale)):
        raise TypeError("mHC coefficient inputs must be float32")


@kernel_registry.register("mhc_coefficients", Platform.XLA, Backend.ANY)
def mhc_coefficients(logits, base, scale, hc_mult: int = 4, n_iters: int = 20, eps: float = 1e-6):
    """Return fp32 read gates, write gates and Sinkhorn residual mixers.

    Args:
        logits: Unscaled projection output ``[..., hc_mult * (hc_mult + 2)]``.
        base: Additive logits offset, one value per projection channel.
        scale: Three multipliers for read, write and mixer channels.
        hc_mult: Number of residual streams (static, positive).
        n_iters: Number of column normalizations (static, positive).
        eps: Offset on read gates, softmax probabilities, and every denominator.

    Returns:
        Tuple of arrays with shapes ``[..., hc_mult]``, ``[..., hc_mult]`` and
        ``[..., hc_mult, hc_mult]``. The first normalization is over rows of
        each column, followed by ``n_iters - 1`` row/column pairs.
    """
    validate_inputs(logits, base, scale, hc_mult, n_iters, eps)
    indices = jnp.asarray([0] * hc_mult + [1] * hc_mult + [2] * (hc_mult * hc_mult))
    z = logits * scale[indices] + base
    pre = jax.nn.sigmoid(z[..., :hc_mult]) + eps
    post = 2 * jax.nn.sigmoid(z[..., hc_mult : 2 * hc_mult])
    matrix = jax.nn.softmax(z[..., 2 * hc_mult :].reshape(*z.shape[:-1], hc_mult, hc_mult), axis=-1) + eps
    matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=-1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    return pre, post, matrix
