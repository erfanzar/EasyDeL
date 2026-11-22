# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Shared kernel initializers for linear layers."""

from __future__ import annotations

import math

import jax
from jax import numpy as jnp

__all__ = ("rescaled_residual_init",)


def rescaled_residual_init(num_layers: int):
    """Kernel initializer for residual-branch output projections (``rescale_prenorm_residual``).

    Mirrors the reference Mamba ``_init_weights``: ``kaiming_uniform(a=sqrt(5))`` (uniform in
    ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``) divided by ``sqrt(num_layers)``, so the variance contributed by the
    pre-norm residual stream does not grow with depth.

    Args:
        num_layers: Number of residual blocks (``config.num_hidden_layers``).

    Returns:
        An initializer ``(key, shape, dtype) -> Array`` for a ``(fan_in, fan_out)`` kernel.
    """
    scale = 1.0 / math.sqrt(num_layers)

    def init(key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(shape[0])
        weight = jax.random.uniform(key, shape, jnp.float32, -bound, bound)
        return (weight * scale).astype(dtype)

    return init
