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
    LayerNorm: Layer Normalization that normalizes across feature dimensions
        independently per sample.
    BatchNorm: Batch Normalization that normalizes across the batch dimension
        with running statistics.
    lowfloats: List of supported float8 data types for low-precision computation.

Example:
    >>> from easydel.layers.components.norms import RMSNorm, LayerNorm, BatchNorm
    >>> import jax.numpy as jnp
    >>>
    >>> # RMSNorm for transformer hidden states
    >>> rms = RMSNorm(dim=768, eps=1e-6, dtype=jnp.bfloat16)
    >>> normalized = rms(jnp.ones((2, 512, 768)))
    >>>
    >>> # LayerNorm for standard normalization
    >>> ln = LayerNorm(num_features=768, rngs=nn.Rngs(0))
    >>> normalized = ln(jnp.ones((2, 512, 768)))
    >>>
    >>> # BatchNorm for convolutional or dense layers
    >>> bn = BatchNorm(num_features=64, rngs=nn.Rngs(0))
    >>> normalized = bn(jnp.ones((8, 64)), use_running_average=False)

See Also:
    - :class:`RMSNorm`: Efficient normalization for LLMs.
    - :class:`LayerNorm`: Standard layer normalization.
    - :class:`BatchNorm`: Batch normalization with running statistics.
    - :data:`lowfloats`: Supported low-precision float types.
"""

from ._norms import BatchNorm, LayerNorm, RMSNorm, lowfloats

__all__ = "RMSNorm", "lowfloats", LayerNorm, BatchNorm
