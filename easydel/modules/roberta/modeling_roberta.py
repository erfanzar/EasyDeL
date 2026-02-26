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


import jax
from eformer import common_types
from eformer.common_types import Replicated
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from flax.nnx.nn.attention import dot_product_attention_weights
from jax import lax
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    DecoderLayerOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.norms import LayerNorm
from easydel.modules._base import (
    BaseCausalLMModule,
    BaseQuestionAnsweringModule,
    BaseSequenceClassificationModule,
    BaseTaskModule,
    BaseTokenClassificationModule,
)

from .roberta_configuration import RobertaConfig as RobertaConfig


class RobertaEmbeddings(nn.Module):
    """Embedding layer for RoBERTa model.

    This layer constructs the combined embeddings from word, position, and
    token type embeddings used in the RoBERTa encoder. It includes layer
    normalization and dropout for regularization.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations. Defaults to jnp.float32.
        param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
        precision (lax.Precision): Precision setting for JAX operations.
        word_embeddings (Embed): Token embedding layer.
        position_embeddings (Embed): Position embedding layer.
        token_type_embeddings (Embed): Token type (segment) embedding layer.
        LayerNorm (LayerNorm): Layer normalization applied after embedding sum.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.word_embeddings = Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.position_embeddings = Embed(
            num_embeddings=self.config.max_position_embeddings,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.token_type_embeddings = Embed(
            num_embeddings=self.config.type_vocab_size,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.LayerNorm = LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(
            rate=self.config.hidden_dropout_prob,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
    ):
        """Forward pass of the RobertaEmbeddings layer.

        Computes the combined embeddings by summing word, position, and token type
        embeddings, then applies layer normalization and dropout.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            token_type_ids (Array): Token type IDs of shape (batch_size, sequence_length)
                for distinguishing different segments (e.g., sentence A vs B).
            position_ids (Array): Position IDs of shape (batch_size, sequence_length)
                for positional encoding.
            attention_mask (Array): Attention mask of shape (batch_size, sequence_length)
                indicating which tokens to attend to.

        Returns:
            Array: Combined embeddings of shape (batch_size, sequence_length, hidden_size)
                after layer normalization and dropout.
        """
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.astype("i4"))

        hidden_states = checkpoint_name(inputs_embeds + token_type_embeddings + position_embeds, "embeddings")

        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class RobertaSelfAttention(AttentionModule):
    """Multi-head self-attention module for RoBERTa.

    This module implements the standard multi-head self-attention mechanism used
    throughout RoBERTa encoder layers. It supports both self-attention and
    cross-attention configurations.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        causal (bool): Whether to apply causal (unidirectional) attention masking.
            Defaults to False for bidirectional attention.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        head_dim (int): Dimension of each attention head.
        attention_performer (FlexibleAttentionModule): Module for computing attention.
        query (ColumnParallelLinear): Query projection layer.
        key (ColumnParallelLinear): Key projection layer.
        value (ColumnParallelLinear): Value projection layer.
    """

    def __init__(
        self,
        config: RobertaConfig,
        causal: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(config)
        self.causal = causal
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads` "
                "                   : {self.config.num_attention_heads}"
            )
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
            requires_cache=False,  # RoBERTa is encoder-only, doesn't need KV cache
        )
        self.query = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )
        self.key = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )
        self.value = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )

    def _split_heads(self, hidden_states):
        """Splits the hidden states into multiple attention heads.

        Args:
            hidden_states (Array): Hidden states of shape (batch, seq_len, hidden_size).

        Returns:
            Array: Reshaped states of shape (batch, seq_len, num_heads, head_dim).
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.config.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """
        Merges the attention heads into a single hidden state tensor.

        Args:
            hidden_states (Array): The hidden states with separate head dimensions.

        Returns:
            Array: The hidden states with merged head dimensions.
        """
        return hidden_states.reshape((*hidden_states.shape[:2], self.config.hidden_size))

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        layer_head_mask: Bool[Array, "num_heads"] | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        key_value_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass of the RobertaSelfAttention module.

        Computes multi-head self-attention (or cross-attention if key_value_states
        is provided) on the input hidden states.

        Args:
            hidden_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo, optional): Pre-computed mask information for attention.
            layer_head_mask (Array, optional): Mask for individual attention heads of shape
                (num_heads,). Used to selectively disable specific heads.
            mode (RUNTIME_MODE_TYPES): Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional): Cache view
                for key/value states in autoregressive generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata
                for cache handling.
            key_value_states (Array, optional): Key/value states for cross-attention.
                If provided, performs cross-attention instead of self-attention.
            output_attentions (bool): Whether to return attention weights.

        Returns:
            AttentionLayerOutput: Named tuple containing:
                - attention_output: Attention output of shape (batch, seq_len, hidden_dim)
                - attention_weight: Attention weights if output_attentions=True, else None
                - cache_view: Updated cache view
        """
        is_cross_attention = key_value_states is not None

        query_states = checkpoint_name(self.query(hidden_states), "attn_query")
        if is_cross_attention:
            key_states = checkpoint_name(self.key(key_value_states), "attn_key")
            value_states = checkpoint_name(self.value(key_value_states), "attn_value")
        else:
            key_states = checkpoint_name(self.key(hidden_states), "attn_key")
            value_states = checkpoint_name(self.value(hidden_states), "attn_value")

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
        )

        if layer_head_mask is None:
            out = self.attention_performer.forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                mode=mode,
                causal=self.causal,
                init_bias=init_attention_bias,
                mask_info=mask_info,
                cache_view=cache_view,
                cache_metadata=cache_metadata,
            )
            attn_weights = out.attention_weights
            attn_output = out.attention_outputs
        else:
            attn_weights = dot_product_attention_weights(
                query_states,
                key_states,
                init_bias=init_attention_bias,
                dropout_rate=self.config.attention_probs_dropout_prob,
                broadcast_dropout=True,
                dtype=self.dtype,
                precision=None,
            )

            attn_weights = jnp.einsum("...hqk,h->...hqk", attn_weights, layer_head_mask)
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        attn_output = checkpoint_name(
            self.shard_attention_prod(attn_output.reshape((*attn_output.shape[:2], -1))), "attn_output"
        )

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attn_weights if output_attentions else None,
            cache_view=cache_view,
        )


class RobertaSelfOutput(nn.Module):
    """Output projection layer following RoBERTa self-attention.

    This module applies a dense projection, dropout, and residual connection
    with layer normalization after the self-attention operation.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (RowParallelLinear): Dense projection layer.
        LayerNorm (LayerNorm): Layer normalization for residual connection.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.LayerNorm = LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, hidden_states, input_tensor):
        """Forward pass of the RobertaSelfOutput layer.

        Args:
            hidden_states (Array): Output from the self-attention layer of shape
                (batch, seq_len, hidden_dim).
            input_tensor (Array): Original input to the attention block for residual
                connection of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Output hidden states of shape (batch, seq_len, hidden_dim) after
                projection, dropout, and layer normalization with residual connection.
        """
        hidden_states = checkpoint_name(self.dense(hidden_states), "attn_dense")
        hidden_states = self.dropout(hidden_states)
        hidden_states = checkpoint_name(self.LayerNorm(hidden_states + input_tensor), "residual")
        return hidden_states


class RobertaAttention(nn.Module):
    """Full attention module combining self-attention and output projection.

    This module wraps the self-attention mechanism and its output projection,
    providing the complete attention sub-layer used in each RoBERTa encoder layer.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        causal (bool): Whether to apply causal attention masking.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        self (RobertaSelfAttention): Self-attention module.
        output (RobertaSelfOutput): Output projection module.
    """

    def __init__(
        self,
        config: RobertaConfig,
        causal: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.causal = causal
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.self = RobertaSelfAttention(
            config=config,
            causal=causal,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.output = RobertaSelfOutput(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states,
        mask_info: MaskInfo | None,
        layer_head_mask,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        key_value_states=None,
        output_attentions: bool = False,
    ):
        """Forward pass of the RobertaAttention module.

        Args:
            hidden_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo, optional): Pre-computed mask information for attention.
            layer_head_mask (Array): Mask for individual attention heads.
            mode (RUNTIME_MODE_TYPES): Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional): Cache view
                for key/value states.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata
                for cache handling.
            key_value_states (Array, optional): Key/value states for cross-attention.
            output_attentions (bool): Whether to return attention weights.

        Returns:
            AttentionLayerOutput: Output containing attention results with residual connection
                and layer normalization applied.
        """
        attn_outputs = self.self(
            hidden_states=hidden_states,
            mask_info=mask_info,
            mode=mode,
            layer_head_mask=layer_head_mask,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            key_value_states=key_value_states,
            output_attentions=output_attentions,
        )

        hidden_states = self.output(attn_outputs.attention_output, hidden_states)

        return attn_outputs.replace(attention_output=hidden_states)


class RobertaIntermediate(nn.Module):
    """Intermediate (up-projection) layer of the RoBERTa feed-forward network.

    This module implements the first dense layer of the two-layer MLP used
    in each RoBERTa encoder layer, expanding the hidden dimension to the
    intermediate size and applying an activation function.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (ColumnParallelLinear): Dense layer projecting to intermediate size.
        activation (callable): Activation function (typically GELU).
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the RobertaIntermediate layer.

        Args:
            hidden_states (Array): Input hidden states of shape
                (batch, seq_len, hidden_dim).

        Returns:
            Array: Intermediate hidden states of shape (batch, seq_len, intermediate_size)
                after dense projection and activation.
        """
        hidden_states = checkpoint_name(self.dense(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.activation(hidden_states), "mlp_gate")
        return hidden_states


class RobertaOutput(nn.Module):
    """Output (down-projection) layer of the RoBERTa feed-forward network.

    This module implements the second dense layer of the two-layer MLP,
    projecting back from intermediate size to hidden size. Includes dropout
    for regularization and layer normalization with residual connection.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (RowParallelLinear): Dense layer projecting back to hidden size.
        dropout (nn.Dropout): Dropout layer for regularization.
        LayerNorm (LayerNorm): Layer normalization for residual connection.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = RowParallelLinear(
            self.config.intermediate_size,
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=dtype,
            precision=precision,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(
            rate=self.config.hidden_dropout_prob,
            rngs=rngs,
        )
        self.LayerNorm = LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states, attention_output):
        """Forward pass of the RobertaOutput layer.

        Args:
            hidden_states (Array): Intermediate hidden states from RobertaIntermediate
                of shape (batch, seq_len, intermediate_size).
            attention_output (Array): Output from the attention sub-layer for residual
                connection of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Output hidden states of shape (batch, seq_len, hidden_dim) after
                down-projection, dropout, and layer normalization with residual.
        """
        hidden_states = checkpoint_name(self.dense(hidden_states), "mlp_down")
        hidden_states = self.dropout(hidden_states)
        hidden_states = checkpoint_name(self.LayerNorm(hidden_states + attention_output), "layer_output")
        return hidden_states


class RobertaLayer(nn.Module):
    """Single RoBERTa transformer encoder layer.

    This module represents a complete encoder layer in the RoBERTa model,
    containing self-attention, optional cross-attention, and feed-forward
    sub-layers with residual connections and layer normalization.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        attention (RobertaAttention): Self-attention sub-layer.
        intermediate (RobertaIntermediate): MLP up-projection sub-layer.
        output (RobertaOutput): MLP down-projection sub-layer.
        crossattention (RobertaAttention, optional): Cross-attention sub-layer
            for encoder-decoder configurations.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.attention = RobertaAttention(
            config=config,
            causal=config.is_decoder,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.intermediate = RobertaIntermediate(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.output = RobertaOutput(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        if self.config.add_cross_attention:
            self.crossattention = RobertaAttention(
                config=config,
                causal=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

    def __call__(
        self,
        hidden_states,
        mask_info: MaskInfo | None,
        layer_head_mask,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
    ):
        """Forward pass of a single RobertaLayer.

        Processes input through self-attention, optional cross-attention (if
        encoder_hidden_states provided), and feed-forward sub-layers.

        Args:
            hidden_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo, optional): Pre-computed mask information for self-attention.
            layer_head_mask (Array): Mask for individual attention heads.
            mode (RUNTIME_MODE_TYPES): Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            cache_view (TransformerCacheView | RaggedPagesCacheView, optional): Cache view
                for key/value states.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata
                for cache handling.
            encoder_hidden_states (Array, optional): Hidden states from encoder for
                cross-attention in encoder-decoder configurations.
            encoder_mask_info (MaskInfo, optional): Mask information for encoder states.
            output_attentions (bool): Whether to return attention weights.

        Returns:
            DecoderLayerOutput: Named tuple containing:
                - hidden_states: Output hidden states of shape (batch, seq_len, hidden_dim)
                - attention_weight: Self-attention weights if output_attentions=True
                - cross_attention: Cross-attention output if encoder_hidden_states provided
                - cache_view: Updated cache view
        """
        # Self Attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            mask_info=mask_info,
            layer_head_mask=layer_head_mask,
            cache_view=cache_view,
            mode=mode,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs.attention_output

        # Cross-Attention Block
        cross_attention = None
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                hidden_states=attention_output,
                mask_info=encoder_mask_info,
                layer_head_mask=layer_head_mask,
                cache_view=None,  # Cross-attention typically doesn't use cache
                mode=mode,
                cache_metadata=None,
                key_value_states=encoder_hidden_states,
                output_attentions=output_attentions,
            )
            cross_attention = cross_attention_outputs.attention_output

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output)

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attention_outputs.attention_weight if output_attentions else None,
            cross_attention=cross_attention,
            cache_view=attention_outputs.cache_view,
        )


class RobertaEncoder(nn.Module):
    """Stack of RoBERTa encoder layers.

    This module contains the stack of transformer encoder layers that form
    the core of the RoBERTa model. Supports gradient checkpointing for
    memory-efficient training of deep models.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        layer (list[RobertaLayer]): List of encoder layers.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        block = RobertaLayer
        block = auto_remat(
            block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layer = nn.List(
            [
                block(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def __call__(
        self,
        hidden_states,
        mask_info: MaskInfo | None,
        head_mask,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_mask_info: MaskInfo | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through all RoBERTa encoder layers.

        Processes input through the full stack of encoder layers, optionally
        collecting intermediate hidden states and attention weights.

        Args:
            hidden_states (Array): Input hidden states from embeddings of shape
                (batch, seq_len, hidden_dim).
            mask_info (MaskInfo, optional): Pre-computed mask information for attention.
            head_mask (Array): Mask for attention heads with shape (num_layers, num_heads).
            mode (RUNTIME_MODE_TYPES): Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER).
            encoder_hidden_states (Array, optional): Hidden states from encoder for
                cross-attention in encoder-decoder configurations.
            encoder_mask_info (MaskInfo, optional): Mask information for encoder states.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key/value states for efficient autoregressive generation.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata, optional): Metadata
                for cache handling.
            output_attentions (bool): Whether to return attention weights from all layers.
            output_hidden_states (bool): Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Named tuple containing:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_dim)
                - past_key_values: Updated cache for next generation step
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - cross_attentions: Tuple of cross-attention outputs if applicable

        Raises:
            ValueError: If head_mask has incorrect number of layers specified.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_cross_attentions = () if encoder_hidden_states is not None else None

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layer)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layer)} layer, but it is for                  "
                    f"       {head_mask.shape[0]}."
                )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layer))
        for i, layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                mask_info=mask_info,
                layer_head_mask=head_mask[i] if head_mask is not None else None,
                mode=mode,
                cache_view=past_key_values.views[i],
                cache_metadata=cache_metadata,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask_info=encoder_mask_info,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[i] = layer_outputs.cache_view

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs.cross_attention,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            past_key_values=past_key_values,
        )


class RobertaPooler(nn.Module):
    """Pooling layer for sequence-level representations.

    This module extracts the representation of the first token ([CLS] token)
    and projects it through a dense layer with tanh activation, producing
    a fixed-size vector for sequence classification tasks.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (RowParallelLinear): Dense projection layer for pooled output.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the RobertaPooler layer.

        Extracts the first token representation and applies dense projection
        with tanh activation.

        Args:
            hidden_states (Array): Hidden states from encoder of shape
                (batch, seq_len, hidden_dim).

        Returns:
            Array: Pooled output of shape (batch, hidden_dim) representing
                the sequence-level encoding.
        """
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class RobertaLMHead(nn.Module):
    """Language modeling head for masked language modeling.

    This module implements the prediction head for masked language modeling (MLM)
    tasks on top of RoBERTa. It consists of a dense layer, GELU activation,
    layer normalization, and a decoder layer projecting to vocabulary size.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (RowParallelLinear): Dense transformation layer.
        layer_norm (LayerNorm): Layer normalization.
        decoder (RowParallelLinear): Projection layer to vocabulary size.
        bias (ArrayParam): Output bias for vocabulary predictions.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )
        self.layer_norm = LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.decoder = RowParallelLinear(
            self.config.hidden_size,
            self.config.vocab_size,
            dtype=dtype,
            use_bias=False,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )
        self.bias = ArrayParam.bound(
            shape=(self.config.vocab_size,),
            dtype=self.param_dtype,
            init_method="zeros",
            key=rngs.params(),
        )

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for LM head bias."""
        return {"bias": Replicated}

    def __call__(self, hidden_states, shared_embedding=None):
        """Forward pass of the RobertaLMHead.

        Transforms hidden states through dense layer, GELU activation,
        layer normalization, and decoder projection to produce vocabulary logits.

        Args:
            hidden_states (Array): Hidden states from encoder of shape
                (batch, seq_len, hidden_dim).
            shared_embedding (Array, optional): Shared embedding weights for
                weight tying. If provided, uses transposed embeddings as decoder weights.

        Returns:
            Array: Vocabulary logits of shape (batch, seq_len, vocab_size).
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = ACT2FN["gelu"](hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        if shared_embedding is not None:
            self.decoder.kernel.value = shared_embedding.T
        hidden_states = self.decoder(hidden_states)

        bias = self.bias.astype(self.dtype)
        hidden_states += bias
        return hidden_states


class RobertaClassificationHead(nn.Module):
    """Classification head for sequence-level classification tasks.

    This module implements the classification head used for sequence
    classification tasks like sentiment analysis. It takes the [CLS] token
    representation and applies dropout, dense layer, tanh activation,
    another dropout, and final projection to class logits.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        dense (RowParallelLinear): Hidden layer projection.
        dropout (nn.Dropout): Dropout for regularization.
        out_proj (RowParallelLinear): Output projection to number of labels.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.dense = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(
            rate=classifier_dropout,
            rngs=rngs,
        )
        self.out_proj = RowParallelLinear(
            self.config.hidden_size,
            self.config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the RobertaClassificationHead.

        Extracts [CLS] token, applies transformations, and projects to class logits.

        Args:
            hidden_states (Array): Hidden states from encoder of shape
                (batch, seq_len, hidden_dim).

        Returns:
            Array: Classification logits of shape (batch, num_labels).
        """
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


@register_module(TaskType.BASE_MODULE, config=RobertaConfig, model_type="roberta")
class RobertaModel(EasyDeLBaseModule):
    """RoBERTa base model implementation.

    This class implements the main RoBERTa encoder architecture, consisting of
    embedding layers (word, position, and token type), multiple encoder layers,
    and an optional pooling layer. RoBERTa uses dynamic masking and removes the
    next sentence prediction objective compared to BERT.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        embeddings (RobertaEmbeddings): Embedding layer for input processing.
        encoder (RobertaEncoder): Stack of transformer encoder layers.
        pooler (RobertaPooler, optional): Pooling layer for sequence-level representations.
        add_pooling_layer (bool): Whether to include the pooling layer.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        add_pooling_layer: bool = True,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embeddings = RobertaEmbeddings(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.encoder = RobertaEncoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.pooler = (
            RobertaPooler(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            if add_pooling_layer
            else None
        )
        self.add_pooling_layer = add_pooling_layer

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        token_type_ids: Int[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        head_mask: Bool[Array, "num_heads"] | None = None,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_attention_mask: Bool[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the RoBERTa model.

        Processes input tokens through embeddings, encoder layers, and optional
        pooling layer. Supports bidirectional encoding and optional cross-attention
        for encoder-decoder configurations.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Array, optional): Boolean mask of shape (batch_size, sequence_length)
                indicating which tokens to attend to (True) and which to ignore (False).
                Auto-generated as all ones if not provided.
            token_type_ids (Array, optional): Token type IDs of shape (batch_size, sequence_length)
                for distinguishing segments. Defaults to zeros if not provided.
            position_ids (Array, optional): Position indices of shape (batch_size, sequence_length).
                Auto-generated following RoBERTa's scheme if not provided.
            head_mask (Array, optional): Mask for attention heads with shape (num_layers, num_heads).
            encoder_hidden_states (Array, optional): Hidden states from encoder for cross-attention.
            encoder_attention_mask (Array, optional): Mask for encoder hidden states.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key/value states for efficient generation.
            output_attentions (bool): Whether to return attention weights from all layers.
            output_hidden_states (bool): Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions: Named tuple containing:
                - last_hidden_state: Final layer output of shape (batch, seq_len, hidden_size)
                - pooler_output: Pooled [CLS] representation if add_pooling_layer=True
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - cross_attentions: Tuple of cross-attention weights if applicable
        """
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)

        # make sure `position_ids` is correctly initialized when not passed
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
        else:
            attention_mask = attention_mask.astype(jnp.bool_)
        if position_ids is None:
            # Match HuggingFace RoBERTa's position id scheme:
            # position_ids start at `padding_idx + 1` for non-padding tokens.
            padding_idx = getattr(self.config, "pad_token_id", 0) or 0
            position_ids = jnp.cumsum(attention_mask.astype("i4"), axis=1) + jnp.asarray(padding_idx, dtype="i4")
            position_ids = jnp.where(attention_mask, position_ids, jnp.asarray(padding_idx, dtype="i4"))

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Initialize MaskInfo
        mask_info = MaskInfo.dynamic_init(
            mask_info=None,
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        # Initialize encoder MaskInfo for cross-attention if encoder_hidden_states provided
        encoder_mask_info = None
        if encoder_hidden_states is not None:
            batch_size = hidden_states.shape[0]
            decoder_seq_len = hidden_states.shape[1]
            encoder_seq_len = encoder_hidden_states.shape[1]

            # Create cross-attention mask: [batch, decoder_seq, encoder_seq]
            if encoder_attention_mask is not None:
                # Broadcast encoder mask to match decoder queries
                # encoder_attention_mask: [batch, encoder_seq] -> [batch, decoder_seq, encoder_seq]
                cross_attn_mask = jnp.broadcast_to(
                    encoder_attention_mask[:, None, :], (batch_size, decoder_seq_len, encoder_seq_len)
                )
            else:
                # No padding - all ones
                cross_attn_mask = jnp.ones((batch_size, decoder_seq_len, encoder_seq_len), dtype=jnp.bool_)

            encoder_mask_info = MaskInfo.from_attention_mask(
                attention_mask=cross_attn_mask[:, None, :, :],
            )

        outputs = self.encoder(
            hidden_states=hidden_states,
            mask_info=mask_info,
            head_mask=head_mask,
            mode=common_types.MODE_TRAIN,  # Default mode, can be parameterized if needed
            encoder_hidden_states=encoder_hidden_states,
            encoder_mask_info=encoder_mask_info,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = checkpoint_name(outputs.last_hidden_state, "model_output")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        """
        return self

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        RoBERTa is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.embeddings


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=RobertaConfig, model_type="roberta")
class RobertaForSequenceClassification(BaseSequenceClassificationModule[RobertaModel, RobertaConfig]):  # type: ignore
    """RoBERTa model with a classification head for sequence classification.

    This model extends the base RoBERTa model by adding a classification head
    on top for sequence-level classification tasks such as sentiment analysis,
    natural language inference, or text categorization.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        roberta (RobertaModel): Base RoBERTa encoder model.
        classifier (RobertaClassificationHead): Classification head for sequence labels.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        roberta = RobertaModel(
            config=config,
            dtype=dtype,
            add_pooling_layer=False,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        BaseTaskModule.__init__(
            self,
            config=config,
            base_model=roberta,
            base_model_name="roberta",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="first",
        )
        self.classifier = RobertaClassificationHead(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        token_type_ids: Int[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        head_mask: Bool[Array, "num_heads"] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for sequence classification.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Array, optional): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (Array, optional): Token type IDs for segment distinction.
            position_ids (Array, optional): Position indices for positional encoding.
            head_mask (Array, optional): Mask for attention heads.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.

        Returns:
            SequenceClassifierOutput: Named tuple containing:
                - logits: Classification logits of shape (batch, num_labels)
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
        """
        # Model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """Returns the encoder part of the model."""
        return self.roberta

    def get_decoder(self):
        """Returns the decoder part of the model.

        Raises:
            NotImplementedError: RoBERTa is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Returns the language model head.

        Raises:
            NotImplementedError: This model has a classification head, not an LM head.
        """
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the model."""
        return self.roberta.get_embedding()

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.classifier


class RobertaForMultipleChoice(EasyDeLBaseModule):
    """RoBERTa model for multiple-choice classification tasks.

    This model extends RoBERTa for multiple-choice tasks where each example
    has several possible answers. The input is reshaped to process all choices
    in parallel, and the model outputs a score for each choice.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        roberta (RobertaModel): Base RoBERTa encoder model with pooling.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (ColumnParallelLinear): Linear layer projecting to single score per choice.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.roberta = RobertaModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(
            rate=self.config.hidden_dropout_prob,
            rngs=rngs,
        )
        self.classifier = ColumnParallelLinear(
            self.config.hidden_size,
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for multiple-choice classification.

        Reshapes inputs from (batch, num_choices, seq_len) to (batch*num_choices, seq_len),
        processes through RoBERTa, and produces scores for each choice.

        Args:
            input_ids (Array): Input token IDs of shape (batch, num_choices, seq_len).
            attention_mask (Array): Attention mask of shape (batch, num_choices, seq_len).
            token_type_ids (Array): Token type IDs of shape (batch, num_choices, seq_len).
            position_ids (Array): Position IDs of shape (batch, num_choices, seq_len).
            head_mask (Array): Mask for attention heads.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.

        Returns:
            MultipleChoiceModelOutput: Named tuple containing:
                - logits: Choice scores of shape (batch, num_choices)
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
        """
        num_choices = input_ids.shape[1]
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.reshape(-1, position_ids.shape[-1]) if position_ids is not None else None

        # Model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.reshape(-1, num_choices)

        return MultipleChoiceModelOutput(
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """Returns the encoder part of the model."""
        return self.roberta

    def get_decoder(self):
        """Returns the decoder part of the model.

        Raises:
            NotImplementedError: RoBERTa is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Returns the language model head.

        Raises:
            NotImplementedError: This model has a multiple choice head, not an LM head.
        """
        raise NotImplementedError("This model has a multiple choice classification head, not a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the model."""
        return self.roberta.get_embedding()


class RobertaForTokenClassification(BaseTokenClassificationModule[RobertaModel, RobertaConfig]):  # type: ignore
    """RoBERTa model with a token classification head.

    This model extends RoBERTa for token-level classification tasks such as
    Named Entity Recognition (NER) or Part-of-Speech (POS) tagging, where
    each token in the sequence receives a label.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        roberta (RobertaModel): Base RoBERTa encoder model without pooling.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer projecting to number of labels per token.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        roberta = RobertaModel(
            config=config,
            dtype=dtype,
            add_pooling_layer=False,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        super().__init__(
            config=config,
            base_model=roberta,
            base_model_name="roberta",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_dropout=classifier_dropout,
            classifier_bias=True,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for token classification.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Array): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (Array): Token type IDs for segment distinction.
            position_ids (Array): Position indices for positional encoding.
            head_mask (Array): Mask for attention heads.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.

        Returns:
            TokenClassifierOutput: Named tuple containing:
                - logits: Token classification logits of shape (batch, seq_len, num_labels)
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
        """
        # Model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """Returns the encoder part of the model."""
        return self.roberta

    def get_decoder(self):
        """Returns the decoder part of the model.

        Raises:
            NotImplementedError: RoBERTa is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Returns the language model head.

        Raises:
            NotImplementedError: This model has a token classification head, not an LM head.
        """
        raise NotImplementedError("This model has a token classification head, not a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the model."""
        return self.roberta.get_embedding()

    def get_task_head(self):
        """Returns the token classification head."""
        return self.classifier


class RobertaForQuestionAnswering(BaseQuestionAnsweringModule[RobertaModel, RobertaConfig]):  # type: ignore
    """RoBERTa model for extractive question answering.

    This model extends RoBERTa for extractive QA tasks where the answer is
    a span within the input context. It predicts start and end positions
    of the answer span.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        roberta (RobertaModel): Base RoBERTa encoder model without pooling.
        qa_outputs (nn.Linear): Linear layer for start/end position predictions.
    """

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        roberta = RobertaModel(
            config=config,
            dtype=dtype,
            add_pooling_layer=False,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        super().__init__(
            config=config,
            base_model=roberta,
            base_model_name="roberta",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for question answering.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Array): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (Array): Token type IDs for segment distinction.
            position_ids (Array): Position indices for positional encoding.
            head_mask (Array): Mask for attention heads.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.

        Returns:
            QuestionAnsweringModelOutput: Named tuple containing:
                - start_logits: Start position logits of shape (batch, seq_len)
                - end_logits: End position logits of shape (batch, seq_len)
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
        """
        # Model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = jnp.split(logits, 2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_encoder(self):
        """Returns the encoder part of the model."""
        return self.roberta

    def get_decoder(self):
        """Returns the decoder part of the model.

        Raises:
            NotImplementedError: RoBERTa is an encoder-only model.
        """
        raise NotImplementedError("This is an encoder-only model and does not have a decoder.")

    def get_lm_head(self):
        """Returns the language model head.

        Raises:
            NotImplementedError: This model has a QA head, not an LM head.
        """
        raise NotImplementedError("This model has a question answering head, not a language model head.")

    def get_embedding(self):
        """Returns the embedding layer of the model."""
        return self.roberta.get_embedding()

    def get_task_head(self):
        """Returns the question answering head."""
        return self.qa_outputs


@register_module(TaskType.CAUSAL_LM, config=RobertaConfig, model_type="roberta")
class RobertaForCausalLM(BaseCausalLMModule[RobertaModel, RobertaConfig]):  # type: ignore
    """RoBERTa model with a causal language modeling head.

    This model adapts RoBERTa for causal (autoregressive) language modeling,
    predicting the next token in a sequence. It uses a language modeling head
    with dense projection, GELU activation, and output vocabulary projection.

    Attributes:
        config (RobertaConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (lax.Precision): Precision setting for JAX operations.
        roberta (RobertaModel): Base RoBERTa encoder model without pooling.
        lm_head (RobertaLMHead): Language modeling head for next token prediction.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "roberta"
    _config_class = RobertaConfig

    def __init__(
        self,
        config: RobertaConfig,
        dtype: jnp.dtype = jnp.float32,  # the dtype of the computation
        param_dtype: jnp.dtype = jnp.float32,
        precision: lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        roberta = RobertaModel(
            config=config,
            add_pooling_layer=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        BaseTaskModule.__init__(
            self,
            config=config,
            base_model=roberta,
            base_model_name="roberta",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self._lm_head_name = "lm_head"
        lm_head_block = RobertaLMHead
        lm_head_block = auto_remat(
            lm_head_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.lm_head = lm_head_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply the language modeling head to hidden states.

        Args:
            hidden_states (Array): Hidden states from encoder of shape
                (batch, seq_len, hidden_dim).

        Returns:
            Array: Vocabulary logits of shape (batch, seq_len, vocab_size).
        """
        shared_embedding = (
            self.roberta.embeddings.word_embeddings.embedding.value if self.config.tie_word_embeddings else None
        )
        return self.lm_head(hidden_states, shared_embedding=shared_embedding)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Int[Array, "batch seq_len"] | None = None,
        head_mask: Bool[Array, "num_heads"] | None = None,
        encoder_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None,
        encoder_attention_mask: Bool[Array, "batch seq_len"] | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass for causal language modeling.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Array, optional): Attention mask of shape (batch_size, sequence_length).
            mask_info (MaskInfo, optional): Pre-computed mask information.
            position_ids (Array, optional): Position indices for positional encoding.
            token_type_ids (Array, optional): Token type IDs for segment distinction.
            head_mask (Array, optional): Mask for attention heads.
            encoder_hidden_states (Array, optional): Hidden states from encoder for cross-attention.
            encoder_attention_mask (Array, optional): Mask for encoder hidden states.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache, optional):
                Cached key/value states for efficient autoregressive generation.
            output_attentions (bool): Whether to return attention weights.
            output_hidden_states (bool): Whether to return all hidden states.

        Returns:
            CausalLMOutputWithCrossAttentions: Named tuple containing:
                - logits: Vocabulary logits of shape (batch, seq_len, vocab_size)
                - hidden_states: Tuple of hidden states if output_hidden_states=True
                - attentions: Tuple of attention weights if output_attentions=True
                - cross_attentions: Tuple of cross-attention weights if applicable
        """
        # Model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        logits = self.apply_lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def get_encoder(self):
        """Returns the encoder part of the model.

        Raises:
            NotImplementedError: This causal LM model does not have a separate encoder.
        """
        raise NotImplementedError("This CausalLM model does not have a separate encoder.")

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.roberta.get_decoder()

    def get_lm_head(self):
        """Returns the language model head."""
        return self.lm_head

    def get_embedding(self):
        """Returns the embedding layer of the model."""
        return self.roberta.get_embedding()
