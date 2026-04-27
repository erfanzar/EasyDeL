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

"""Generic base class for Embedding tasks.

This module provides BaseEmbeddingModule, a generic, type-safe base class for
creating embedding model wrappers that produce dense vector representations
from input sequences. These embeddings are suitable for semantic search,
retrieval-augmented generation, clustering, and similarity computation.

Unlike sequence classification models, embedding models have no task-specific
head — they pool the hidden states and optionally L2-normalize the result.

Key Features:
    - Generic typing with ModelT and ConfigT type parameters
    - Configurable pooling strategies (last, first, mean, weighted_mean, max)
    - Optional L2 normalization (standard for embedding models)
    - Matryoshka truncation support for variable-dimension embeddings
    - Convenience ``encode()`` method for tokenize-and-embed workflows
    - Static similarity computation utilities

Example:
    Creating an embedding model::

        from easydel.modules._base import BaseEmbeddingModule

        class MyModelForEmbedding(
            BaseEmbeddingModule[MyModel, MyConfig]
        ):
            _task_type = TaskType.EMBEDDING
            _model_type = "my_model"
            _config_class = MyConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                super().__init__(
                    config=config,
                    base_model_class=MyModel,
                    dtype=dtype,
                    rngs=rngs,
                    pooling_strategy="last",
                )

See Also:
    - BaseTaskModule: Parent class with common task functionality
    - BaseSequenceClassificationModule: Similar pattern with classification head
"""

import jax
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from spectrax import common_types

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerMetadata,
)
from easydel.infra.modeling_outputs import EmbeddingOutput

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseEmbeddingModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Embedding models.

    This class provides a fully-featured, type-safe base for creating embedding
    model wrappers with support for:

    - Generic typing (ModelT, ConfigT) for type safety
    - Automatic model registration via class attributes
    - Configurable pooling strategies (last, first, mean, weighted_mean, max)
    - L2 normalization for cosine-similarity-ready embeddings
    - Matryoshka (MRL) truncation for variable-dimension embeddings
    - Convenience ``encode()`` method for raw text input

    The embedding task produces a fixed-size vector for each input sequence by
    pooling the transformer hidden states and optionally normalizing. No
    task-specific learned head is required — the pooled hidden state IS the
    embedding.

    Example:
        Basic usage::

            model = Qwen2ForEmbedding.from_pretrained(...)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.embeddings  # (batch_size, hidden_size)

        Computing similarity::

            sim = BaseEmbeddingModule.cosine_similarity(emb_a, emb_b)

    Type Parameters:
        ModelT: The base model type (must implement BaseModelProtocol).
        ConfigT: The configuration type containing model hyperparameters.

    Attributes:
        _normalize_embeddings: Whether to L2-normalize output embeddings.
        _embedding_dim: Optional truncation dimension for Matryoshka support.
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "model",
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        pooling_strategy: str = "last",
        router_aux_loss_coef: float | None = None,
        normalize_embeddings: bool = True,
        embedding_dim: int | None = None,
    ):
        """Initialize the Embedding module.

        Creates an embedding model by wrapping a base transformer model with
        pooling and optional normalization. No classification head is created.

        Args:
            config: Model configuration object. Must have ``hidden_size``
                and optionally ``pad_token_id`` attributes.
            base_model: Pre-instantiated base model instance.
            base_model_class: Base model class to instantiate.
            base_model_name: Attribute name for storing the base model.
            dtype: Data type for computations.
            param_dtype: Data type for parameters.
            precision: JAX precision setting.
            rngs: SpecTrax random number generators.
            pooling_strategy: Strategy for pooling sequence representations.
                One of ``"last"``, ``"first"``, ``"mean"``, ``"weighted_mean"``,
                ``"max"``. Defaults to ``"last"`` (standard for decoder-only
                models like GTE-Qwen2, E5-Mistral).
            router_aux_loss_coef: Coefficient for MoE router auxiliary loss.
            normalize_embeddings: Whether to L2-normalize output embeddings.
                Defaults to ``True`` (standard for all modern embedding models).
            embedding_dim: Optional truncation dimension for Matryoshka
                (MRL) support. ``None`` means full ``hidden_size``.
        """
        super().__init__(
            config=config,
            base_model=base_model,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy=pooling_strategy,
            router_aux_loss_coef=router_aux_loss_coef,
        )
        self._normalize_embeddings = normalize_embeddings
        self._embedding_dim = embedding_dim

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> EmbeddingOutput:
        """Forward pass through the Embedding model.

        Processes input through the base transformer, pools hidden states
        to a single vector per sequence, optionally truncates to
        ``embedding_dim``, and optionally L2-normalizes.

        Args:
            input_ids: Input token IDs ``(batch_size, sequence_length)``.
            inputs_embeds: Pre-computed embeddings ``(batch_size, seq_len, hidden)``.
            attention_mask: Binary mask ``(batch_size, sequence_length)``.
            mask_info: Structured mask information for advanced attention.
            position_ids: Position indices ``(batch_size, sequence_length)``.
            mode: Runtime mode (train, eval, decode).
            past_key_values: Cached KV states.
            cache_metadata: Cache management metadata.
            output_attentions: Include attention weights in output.
            output_hidden_states: Include all layer hidden states in output.

        Returns:
            EmbeddingOutput with ``embeddings`` of shape
            ``(batch_size, embedding_dim or hidden_size)``.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state
        embeddings = self.pool_sequence(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self._embedding_dim is not None:
            embeddings = embeddings[:, : self._embedding_dim]

        if self._normalize_embeddings:
            embeddings = embeddings / jnp.clip(
                jnp.linalg.norm(embeddings, axis=-1, keepdims=True),
                min=1e-12,
            )

        return EmbeddingOutput(
            embeddings=embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_task_head(self):
        """Embedding models have no task-specific head."""
        return None

    def get_lm_head(self):
        """Embedding models do not have a language modeling head."""
        raise NotImplementedError("Embedding models do not have a language modeling head.")

    def encode(
        self,
        texts: list[str] | str,
        tokenizer=None,
        max_length: int | None = None,
        batch_size: int = 32,
        normalize: bool | None = None,
        embedding_dim: int | None = None,
    ) -> Array:
        """Encode texts into dense embedding vectors.

        High-level convenience method that handles tokenization, batching,
        forward pass, and concatenation. Accepts raw strings and returns
        a single array of embeddings.

        Args:
            texts: Single text string or list of strings to encode.
            tokenizer: Tokenizer instance. If ``None``, uses the tokenizer
                stored in ``self.processing_class`` (set during loading).
            max_length: Maximum sequence length for tokenization. If ``None``,
                uses the model's configured ``max_position_embeddings``.
            batch_size: Number of texts to process in each forward pass.
            normalize: Override L2 normalization. If ``None``, uses the
                model's configured ``_normalize_embeddings`` setting.
            embedding_dim: Override Matryoshka truncation dimension.
                If ``None``, uses the model's configured ``_embedding_dim``.

        Returns:
            Array of shape ``(num_texts, embedding_dim or hidden_size)``
            containing the embedding vectors.

        Raises:
            ValueError: If no tokenizer is available.

        Example:
            >>> embeddings = model.encode(["Hello world", "How are you?"])
            >>> embeddings.shape
            (2, 1536)
        """
        if isinstance(texts, str):
            texts = [texts]

        if tokenizer is None:
            tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            raise ValueError(
                "No tokenizer available. Pass one explicitly or load the model "
                "with AutoEasyDeLModelForEmbedding.from_pretrained() which sets "
                "processing_class automatically."
            )

        if max_length is None:
            max_length = getattr(self.config, "max_position_embeddings", 8192)

        orig_normalize = self._normalize_embeddings
        orig_dim = self._embedding_dim
        try:
            if normalize is not None:
                self._normalize_embeddings = normalize
            if embedding_dim is not None:
                self._embedding_dim = embedding_dim

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="np",
                )
                input_ids = jnp.array(encoded["input_ids"])
                attention_mask = jnp.array(encoded["attention_mask"])

                outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                all_embeddings.append(outputs.embeddings)

            return jnp.concatenate(all_embeddings, axis=0)
        finally:
            self._normalize_embeddings = orig_normalize
            self._embedding_dim = orig_dim

    @staticmethod
    def cosine_similarity(
        a: Float[Array, "n dim"],
        b: Float[Array, "m dim"],
    ) -> Float[Array, "n m"]:
        """Compute pairwise cosine similarity between two sets of embeddings.

        Args:
            a: First set of embeddings ``(n, dim)``.
            b: Second set of embeddings ``(m, dim)``.

        Returns:
            Similarity matrix ``(n, m)`` where each entry ``[i, j]`` is the
            cosine similarity between ``a[i]`` and ``b[j]``.
        """
        a_norm = a / jnp.clip(jnp.linalg.norm(a, axis=-1, keepdims=True), min=1e-12)
        b_norm = b / jnp.clip(jnp.linalg.norm(b, axis=-1, keepdims=True), min=1e-12)
        return a_norm @ b_norm.T

    @staticmethod
    def dot_product_similarity(
        a: Float[Array, "n dim"],
        b: Float[Array, "m dim"],
    ) -> Float[Array, "n m"]:
        """Compute pairwise dot-product similarity between embeddings.

        For L2-normalized embeddings, this is equivalent to cosine similarity.

        Args:
            a: First set of embeddings ``(n, dim)``.
            b: Second set of embeddings ``(m, dim)``.

        Returns:
            Similarity matrix ``(n, m)``.
        """
        return a @ b.T
