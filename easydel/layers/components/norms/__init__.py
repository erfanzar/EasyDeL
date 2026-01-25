# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""EasyDeL normalization layers package.

This package provides efficient normalization layer implementations optimized
for JAX/Flax, with support for mixed precision training and float8 data types.

Modules:
    _norms: Core normalization layer implementations.

Exports:
    RMSNorm: Root Mean Square normalization layer for efficient normalization
        in transformer models.
    lowfloats: List of supported float8 data types for low-precision computation.

Example:
    >>> from easydel.layers.components.norms import RMSNorm, lowfloats
    >>> import jax.numpy as jnp
    >>>
    >>> # Create RMSNorm layer
    >>> norm = RMSNorm(dim=768, eps=1e-6, dtype=jnp.bfloat16)
    >>>
    >>> # Apply normalization
    >>> inputs = jnp.ones((2, 512, 768))
    >>> normalized = norm(inputs)

Note:
    RMSNorm is the preferred normalization for large language models as it
    provides computational efficiency while maintaining model performance.
    It requires fewer parameters than LayerNorm since it does not learn
    separate shift parameters.

See Also:
    - :class:`RMSNorm`: Main normalization layer class.
    - :data:`lowfloats`: Supported low-precision float types.
"""

from ._norms import RMSNorm, lowfloats

__all__ = "RMSNorm", "lowfloats"
