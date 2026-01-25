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

"""Embedding layer implementations for EasyDeL.

This module provides embedding layer components used in neural network models,
particularly for natural language processing tasks. The primary component is
the `Embed` class which converts integer token indices into dense vector
representations.

The embedding implementation supports:
    - Configurable embedding dimensions and vocabulary size
    - Custom initialization strategies
    - Dtype promotion for mixed-precision training
    - Efficient attention mechanism via the `attend` method for weight-tying

Example:
    >>> from easydel.layers.components.embeddings import Embed
    >>> from flax import nnx as nn
    >>> import jax.numpy as jnp
    >>>
    >>> # Create embedding layer
    >>> embed = Embed(
    ...     num_embeddings=32000,  # vocabulary size
    ...     features=768,          # embedding dimension
    ...     rngs=nn.Rngs(0)
    ... )
    >>>
    >>> # Embed token indices
    >>> tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> embeddings = embed(tokens)  # Shape: (2, 3, 768)
"""

from flax import nnx as nn
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
    Dtype,
    Initializer,
    PromoteDtypeFn,
)
from jax import Array
from jax import numpy as jnp

default_embed_init = initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
"""Default embedding initializer using variance scaling.

Uses fan-in variance scaling with normal distribution and output axis 0,
which is standard practice for embedding layers to maintain proper gradient flow.
"""


class Embed(nn.Module):
    """Embedding layer that converts integer indices to dense vector representations.

    This module implements a standard embedding lookup table that maps discrete
    token indices to continuous vector representations. It is commonly used as
    the first layer in NLP models to convert tokenized text into dense embeddings.

    The embedding supports dtype promotion for mixed-precision training and
    includes an `attend` method for efficient weight-tying between input embeddings
    and output projections (common in language models).

    Attributes:
        embedding: The embedding weight matrix of shape (num_embeddings, features).
        num_embeddings: Size of the vocabulary (number of unique tokens).
        features: Dimensionality of each embedding vector.
        dtype: Data type for embedding computations (default: same as embedding weights).
        param_dtype: Data type used for parameter initialization.
        embedding_init: Initializer function for embedding weights.
        promote_dtype: Function to promote dtypes during computation.

    Example:
        >>> from easydel.layers.components.embeddings import Embed
        >>> from flax import nnx as nn
        >>> import jax.numpy as jnp
        >>>
        >>> # Create an embedding layer for a vocabulary of 1000 tokens
        >>> embed = Embed(
        ...     num_embeddings=1000,
        ...     features=256,
        ...     rngs=nn.Rngs(42)
        ... )
        >>>
        >>> # Embed a batch of token sequences
        >>> tokens = jnp.array([[5, 10, 15], [20, 25, 30]])
        >>> embeddings = embed(tokens)  # Shape: (2, 3, 256)
        >>>
        >>> # Use attend for weight-tying (e.g., in language model head)
        >>> hidden_states = jnp.ones((2, 3, 256))
        >>> logits = embed.attend(hidden_states)  # Shape: (2, 3, 1000)

    Note:
        For single embedding vocabularies (num_embeddings=1), the embedding is
        broadcast to match the input shape, which is useful for special tokens
        or bias terms.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        embedding_init: Initializer = default_embed_init,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: nn.Rngs,
    ):
        """Initialize the embedding layer.

        Args:
            num_embeddings: Number of embeddings in the vocabulary (vocab size).
                Must be a positive integer.
            features: Dimensionality of each embedding vector. Must be a positive
                integer.
            dtype: Data type for embedding vectors during computation. If None,
                uses the same dtype as the embedding weights. Defaults to None.
            param_dtype: Data type for parameter initialization. For special dtypes
                like float4_e2m1fn, initialization is done in float32 and then
                converted. Defaults to jnp.float32.
            embedding_init: Initializer function for the embedding weight matrix.
                Should be a callable that takes (rng_key, shape, dtype) and returns
                an array. Defaults to variance scaling initialization.
            promote_dtype: Function to promote array dtypes during computation.
                Used to ensure consistent dtypes between embeddings and queries.
                Defaults to flax's promote_dtype function.
            rngs: Random number generator state for parameter initialization.

        Note:
            When param_dtype is jnp.float4_e2m1fn, the embedding is initialized
            in float32 to avoid numerical issues during initialization.
        """
        param_to_init = param_dtype
        if param_dtype in [jnp.float4_e2m1fn]:
            param_to_init = jnp.float32

        self.embedding = nn.Param(embedding_init(rngs.params(), (num_embeddings, features), param_to_init))

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: Array) -> Array:
        """Embed the input indices along the last dimension.

        Performs an embedding lookup by indexing into the embedding weight matrix
        using the provided integer indices. Each integer in the input is replaced
        with the corresponding embedding vector.

        Args:
            inputs: Integer array of token indices with arbitrary shape. All
                dimensions are treated as batch dimensions. Values must be
                valid indices in the range [0, num_embeddings).

        Returns:
            Array of shape (*inputs.shape, features) containing the embedded
            vectors. The output has the same leading dimensions as the input,
            with an additional `features` dimension appended.

        Raises:
            ValueError: If the input dtype is not an integer type.

        Example:
            >>> tokens = jnp.array([[1, 2], [3, 4]])  # Shape: (2, 2)
            >>> embeddings = embed(tokens)            # Shape: (2, 2, features)
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")

        (embedding,) = self.promote_dtype((self.embedding.value,), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        """Compute attention scores between query vectors and embeddings.

        Performs a batched inner product between query vectors and all embedding
        vectors in the vocabulary. This operation is commonly used for weight-tying
        in language models, where the same embedding matrix is used both for input
        embeddings and for the output projection to vocabulary logits.

        Args:
            query: Array of query vectors with shape (..., features), where the
                last dimension must match the embedding feature dimension.

        Returns:
            Array of shape (..., num_embeddings) containing the dot product
            similarity scores between each query vector and all embeddings.
            These scores are typically interpreted as logits over the vocabulary.

        Example:
            >>> # In a language model with weight-tying
            >>> hidden_states = model.decoder(inputs)  # Shape: (batch, seq, features)
            >>> logits = embed.attend(hidden_states)   # Shape: (batch, seq, vocab_size)
            >>> probs = jax.nn.softmax(logits, axis=-1)
        """
        query, embedding = self.promote_dtype((query, self.embedding.value), dtype=self.dtype)
        return jnp.dot(query, embedding.T)
