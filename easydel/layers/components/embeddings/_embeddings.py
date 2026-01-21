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


class Embed(nn.Module):
    """
    Embedding Module.

      num_embeddings: number of embeddings / vocab size.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: same as embedding).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      embedding_init: embedding initializer.
      promote_dtype: function to promote the dtype of the arrays to the desired
        dtype. The function should accept a tuple of ``(embedding,)`` during ``__call__``
        or ``(query, embedding)`` during ``attend``, and a ``dtype`` keyword argument,
        and return a tuple of arrays with the promoted dtype.
      rngs: rng key.
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
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.
            Values in the input array must be integers.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional ``features`` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")

        (embedding,) = self.promote_dtype((self.embedding.value,), dtype=self.dtype, inexact=False)
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, (*inputs.shape, self.features))
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: Array) -> Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth ``features`` of the
            embedding.

        Returns:
          An array with final dim ``num_embeddings`` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = self.promote_dtype((query, self.embedding.value), dtype=self.dtype)
        return jnp.dot(query, embedding.T)
