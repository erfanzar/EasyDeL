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
    >>> from easydel.layers.embeddings import Embed
    >>> import spectrax as spx
    >>> from spectrax import nn
    >>> import jax.numpy as jnp
    >>>
    >>> # Create embedding layer
    >>> embed = Embed(
    ...     num_embeddings=32000,  # vocabulary size
    ...     features=768,          # embedding dimension
    ...     rngs=spx.Rngs(0)
    ... )
    >>>
    >>> # Embed token indices
    >>> tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> embeddings = embed(tokens)  # Shape: (2, 3, 768)
"""

import typing as tp

import jax
import spectrax as spx
from jax import Array
from jax import numpy as jnp
from spectrax.common_types import ColumnWise

from easydel.infra.sharding import sharding_for_layout

Dtype = jnp.dtype | type
Initializer = jax.nn.initializers.Initializer
PromoteDtypeFn = tp.Callable[..., tuple]


def _promote_dtype(values, *, dtype=None, inexact=None):
    """Cast a tuple of arrays to a common dtype.

    Used by :class:`Embed` to align embedding weights with the queried
    inputs before performing the lookup or matrix multiply.

    Args:
        values: Iterable of arrays (or array-like values) to promote.
        dtype: Target dtype. If ``None``, the values are returned unchanged.
        inexact: Unused; kept for compatibility with the upstream ``flax``/
            ``spectrax`` ``promote_dtype`` signature.

    Returns:
        Tuple of arrays cast to ``dtype`` if it was provided, otherwise the
        original ``values`` argument unchanged.
    """
    if dtype is None:
        return values
    return tuple(jnp.asarray(v, dtype=dtype) for v in values)


default_embed_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)
"""Default embedding initializer using variance scaling.

Uses fan-in variance scaling with normal distribution and output axis 0,
which is standard practice for embedding layers to maintain proper gradient flow.
"""


class Embed(spx.Module):
    """Embedding layer that converts integer indices to dense vector representations.

    Maps integer token indices in the range ``[0, num_embeddings)`` into a
    learnable dense matrix of shape ``(num_embeddings, features)``. The forward
    pass is a straightforward gather (``jnp.take``) and the :meth:`attend`
    method exposes the same matrix as a logit projection so the layer can be
    used in weight-tied language model heads.

    The embedding weight is registered as a column-wise sharded
    :class:`spectrax.Parameter` so that vocabulary-parallel embedding tables
    work transparently across model-parallel meshes.

    Attributes:
        weight: Parameter of shape ``(num_embeddings, features)`` holding the
            embedding lookup table. Sharded column-wise.
        num_embeddings: Vocabulary size (number of distinct tokens).
        features: Embedding dimensionality.
        dtype: Computation dtype; weights are cast to this dtype before lookup.
        param_dtype: Storage dtype for the parameter (may differ from ``dtype``
            for mixed-precision training; e.g. fp4/fp8 storage).
        embedding_init: Initializer used to populate the weight matrix.
        promote_dtype: Callable used to promote tensors to a common dtype
            before lookup or attend.

    Note:
        Overrides ``__setattr__`` to allow JAX dtype values (e.g.
        ``jnp.float32``) which are not recognised as static scalars by the
        base :class:`spectrax.Module`.
    """

    def __setattr__(self, name: str, value: tp.Any) -> None:
        """Set an attribute, falling back to ``object.__setattr__`` for JAX dtypes.

        Args:
            name: Attribute name to set.
            value: Value to assign. JAX dtype values are accepted even though
                the parent ``spectrax.Module`` rejects them as non-static.

        Returns:
            None.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        try:
            super().__setattr__(name, value)
        except TypeError:
            object.__setattr__(self, name, value)

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        embedding_init: Initializer = default_embed_init,
        promote_dtype: PromoteDtypeFn = _promote_dtype,
        rngs: spx.Rngs,
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
                Defaults to spx's promote_dtype function.
            rngs: Random number generator state for parameter initialization.

        Note:
            When param_dtype is jnp.float4_e2m1fn, the embedding is initialized
            in float32 to avoid numerical issues during initialization.
        """
        super().__init__()
        param_to_init = param_dtype
        if param_dtype in [jnp.float4_e2m1fn]:
            param_to_init = jnp.float32

        self.weight = spx.Parameter(
            embedding_init(rngs.parameters, (num_embeddings, features), param_to_init),
            sharding=sharding_for_layout(ColumnWise),
        )

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.weight.value.dtype
        self.param_dtype = param_dtype
        self.embedding_init = embedding_init
        self.promote_dtype = promote_dtype

    def forward(self, inputs: Array) -> Array:
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

        (embedding,) = self.promote_dtype((self.weight.value,), dtype=self.dtype, inexact=False)
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
        query, embedding = self.promote_dtype((query, self.weight.value), dtype=self.dtype)
        return jnp.dot(query, embedding.T)
