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

"""Embedding layer components for EasyDeL neural networks.

This package provides embedding layer implementations for converting discrete
token indices into continuous dense vector representations. These components
are fundamental building blocks for natural language processing models.

Available Components:
    Embed: Standard embedding layer that maps integer indices to dense vectors.
        Supports configurable dimensions, custom initialization, and dtype
        promotion for mixed-precision training.

Example:
    >>> from easydel.layers.embeddings import Embed
    >>> from flax import nnx as nn
    >>> import jax.numpy as jnp
    >>>
    >>> # Create embedding layer with 32K vocabulary and 512-dim embeddings
    >>> embed = Embed(
    ...     num_embeddings=32000,
    ...     features=512,
    ...     rngs=nn.Rngs(0)
    ... )
    >>>
    >>> # Embed token sequences
    >>> tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> embeddings = embed(tokens)  # Shape: (2, 3, 512)

See Also:
    - easydel.layers.components.embeddings._embeddings: Implementation module
    - flax.nnx.nn.Embed: Flax's native embedding implementation
"""

from ._embeddings import Embed

__all__ = ("Embed",)
