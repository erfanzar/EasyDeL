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


# coding=utf-8
# Copyright 2022 The Fairseq Authors and The Google Flax Team Authors And The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# THIS SCRIPT IS EDITED FROM ORIGINAL IMPLEMENTATION OF TRANSFORMERS OPT
"""Flax OPT model."""

from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear

from .opt_configuration import OPTConfig


class OPTAttention(AttentionModule):
    """OPT Attention mechanism module.

    This module implements the multi-head self-attention mechanism used in the OPT model.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        embed_dim (int): The dimensionality of the embedding layer.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability for the attention scores.
        causal (bool): Whether to use causal masking.
        bias (bool): Whether to include bias in the linear projections.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        head_dim (int): Dimensionality of each attention head.
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        out_proj (ParallelLinear): Linear layer for the output projection.
        dropout_layer (nn.Dropout): Dropout layer applied after attention.
        attention_module (AttentionModule): The core attention computation module.
    """

    def __init__(
        self,
        config: OPTConfig,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initializes the OPTAttention module.

        Args:
            config (OPTConfig): The configuration object for the OPT model.
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate for the attention scores. Defaults to 0.0.
            causal (bool, optional): Whether to apply causal masking. Defaults to False.
            bias (bool, optional): Whether to use bias in linear projections. Defaults to True.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike, optional): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__(config=config)

        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.bias = bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`: {num_heads})."
            )

        linear = partial(
            ColumnParallelLinear,
            use_bias=bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
        )

        self.q_proj, self.k_proj, self.v_proj = (
            linear(embed_dim, embed_dim, rngs=rngs),
            linear(embed_dim, embed_dim, rngs=rngs),
            linear(embed_dim, embed_dim, rngs=rngs),
        )
        self.out_proj = linear(embed_dim, embed_dim, rngs=rngs)

        self.dropout_layer = nn.Dropout(rate=self.dropout, rngs=rngs)
        self.attention_module = FlexibleAttentionModule(base_config=config, softmax_scale=self.head_dim**-0.5)

    def _split_heads(self, hidden_states):
        """Splits the hidden states into multiple heads."""
        return hidden_states.reshape((*hidden_states.shape[:2], self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        """Merges the attention heads back into a single hidden state tensor."""
        return hidden_states.reshape((*hidden_states.shape[:2], self.embed_dim))

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        key_value_states: Array | None = None,
        output_attentions: bool = False,
    ) -> AttentionLayerOutput:
        """Forward pass of the OPTAttention module.

        Args:
            hidden_states (Array): Input hidden states. Shape: (batch_size, sequence_length, embed_dim).
            mask_info (MaskInfo | None): Mask information for attention.
            mode (common_types.RUNTIME_MODE_TYPES): Runtime mode (train/eval/infer).
            cache_view (TransformerCacheView | RaggedPagesCacheView | None): Cache view for attention KVs.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | None): Metadata for paged attention.
            key_value_states (Array | None): Optional hidden states for cross-attention. If provided,
                these are used as keys and values. Shape: (batch_size, key_sequence_length, embed_dim).
            output_attentions (bool): Whether to return attention weights.

        Returns:
            AttentionLayerOutput: Attention output containing attention_output, attention_weight, and cache_view.
        """
        is_cross_attention = key_value_states is not None
        query_states = checkpoint_name(self.q_proj(hidden_states), "attn_query")

        if is_cross_attention:
            key_states = checkpoint_name(self.k_proj(key_value_states), "attn_key")
            value_states = checkpoint_name(self.v_proj(key_value_states), "attn_value")
        else:
            key_states = checkpoint_name(self.k_proj(hidden_states), "attn_key")
            value_states = checkpoint_name(self.v_proj(hidden_states), "attn_value")

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

        attentions = self.attention_module.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=self.causal,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.out_proj(attn_output), "attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class OPTDecoderLayer(nn.Module):
    """OPT Decoder Layer.

    This module represents a single layer in the OPT decoder stack.
    It consists of a self-attention mechanism, optional layer normalization,
    a feed-forward network (FFN), and residual connections.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        embed_dim (int): Dimensionality of the embedding layer.
        self_attn (OPTAttention): The self-attention module.
        do_layer_norm_before (bool): Whether to apply layer normalization before the attention/FFN blocks.
        dropout_layer (nn.Dropout): Dropout layer applied to the hidden states.
        activation_fn (callable): The activation function used in the FFN.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization applied before the self-attention module.
        fc1 (ParallelLinear): The first linear layer of the FFN.
        fc2 (ParallelLinear): The second linear layer (output) of the FFN.
        final_layer_norm (nn.LayerNorm): Layer normalization applied before the FFN module.
    """

    def __init__(
        self,
        config: OPTConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize OPT decoder layer.

        Args:
            config: OPTConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        super().__init__()

        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.embed_dim = self.config.hidden_size
        self.self_attn = OPTAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            causal=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.do_layer_norm_before = self.config.do_layer_norm_before
        self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)
        self.activation_fn = ACT2FN[self.config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            epsilon=1e-05,
        )
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
            rngs=rngs,
        )
        self.fc2 = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(config.init_std),
            rngs=rngs,
        )
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            dtype=self.dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            epsilon=1e-05,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
    ) -> DecoderLayerOutput:
        """Forward pass of the OPTDecoderLayer.

        Applies pre-norm or post-norm architecture based on config.do_layer_norm_before:
        - Pre-norm (default): LayerNorm -> Attention -> Residual, LayerNorm -> FFN -> Residual
        - Post-norm: Attention -> Residual -> LayerNorm, FFN -> Residual -> LayerNorm

        Args:
            hidden_states: Input hidden states with shape (batch_size, sequence_length, hidden_size).
            mask_info: Masking information containing attention masks and positions.
            mode: Runtime mode (train/decode/prefill) for cache handling.
            cache_view: Optional cache view for key/value states in decoder inference.
            cache_metadata: Optional metadata for cache handling.
            output_attentions: Whether to return attention weights (default: False).

        Returns:
            DecoderLayerOutput containing:
                - hidden_states: Output hidden states with shape (batch_size, sequence_length, hidden_size).
                - attention_weight: Optional attention weights if output_attentions=True.
                - cache_view: Updated cache view if cache is used.
        """
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = checkpoint_name(self.fc1(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.activation_fn(hidden_states), "mlp_gate")

        hidden_states = checkpoint_name(self.fc2(hidden_states), "mlp_down")
        hidden_states = self.dropout_layer(hidden_states)

        hidden_states = checkpoint_name((residual + hidden_states).reshape(hidden_states_shape), "layer_output")

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight if output_attentions else None,
            cache_view=attn_outputs.cache_view,
        )


class OPTLearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding for OPT.

    This module learns positional embeddings up to a maximum specified length.
    It includes an offset, typically used to account for padding tokens.

    Attributes:
        offset (int): The offset added to position IDs before embedding lookup.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        offset: int = 2,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        embedding_init=None,
        rngs: nn.Rngs,
    ):
        """Initializes the OPTLearnedPositionalEmbedding module.

        Args:
            num_embeddings (int): The maximum number of positions to embed (vocabulary size).
            features (int): The dimensionality of the embedding vectors.
            offset (int, optional): The offset added to position IDs. Defaults to 2.
            dtype (tp.Optional[jnp.dtype], optional): Data type for the embeddings. Defaults to None.
            param_dtype (jnp.dtype, optional): Data type for the parameters. Defaults to jnp.float32.
            embedding_init (callable, optional): Initializer function for the embeddings.
                Defaults to `jax.nn.initializers.normal(stddev=1.0)`.
            rngs (nn.Rngs): Random number generators.
        """
        self.offset = offset
        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or param_dtype
        self.param_dtype = param_dtype

        # Create embedding parameter directly to match HuggingFace structure
        # HuggingFace uses 'weight' (PyTorch) which becomes 'kernel' in Flax
        if embedding_init is not None:
            value = embedding_init(rngs.params(), (num_embeddings + offset, features), param_dtype)
            self.kernel = ArrayParam.bound(
                shape=(num_embeddings + offset, features),
                dtype=param_dtype,
                init_method="variance_scaling",
                init_kwargs={"scale": 1.0, "mode": "fan_in", "distribution": "normal"},
                key=rngs.params(),
                value=value,
            )
        else:
            self.kernel = ArrayParam.bound(
                shape=(num_embeddings + offset, features),
                dtype=param_dtype,
                init_method="variance_scaling",
                init_kwargs={"scale": 1.0, "mode": "fan_in", "distribution": "normal"},
                key=rngs.params(),
            )

    def __call__(self, inputs: Array) -> Array:
        """Apply learned positional embeddings with offset.

        Args:
            inputs: Position indices [batch, seq_len]

        Returns:
            Positional embeddings [batch, seq_len, features]
        """
        # Add offset to inputs and lookup embeddings
        indices = inputs + self.offset
        # Use take for embedding lookup, matching nn.Embed behavior
        embedded = jnp.take(self.kernel.value, indices, axis=0)
        # Cast to output dtype if needed
        if self.dtype != self.param_dtype:
            embedded = embedded.astype(self.dtype)
        return embedded


class OPTDecoder(EasyDeLBaseModule):
    """OPT Decoder stack.

    This module comprises the main transformer decoder layers for the OPT model,
    including token embeddings, positional embeddings, the decoder layers themselves,
    and optional final layer normalization.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        padding_idx (int): Index of the padding token.
        max_target_positions (int): Maximum sequence length the model can handle.
        embed_scale (float): Scaling factor for embeddings (usually 1.0).
        embed_tokens (nn.Embed): Token embedding layer.
        embed_positions (OPTLearnedPositionalEmbedding): Positional embedding layer.
        project_out (ColumnParallelLinear, optional): Optional linear projection layer after embeddings.
        project_in (ColumnParallelLinear, optional): Optional linear projection layer before embeddings.
        layers (tp.List[OPTDecoderLayer]): List of OPT decoder layers.
        dropout_layer (nn.Dropout): Dropout layer applied after embeddings.
        final_layer_norm (nn.LayerNorm, optional): Optional final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.dropout_layer = nn.Dropout(rate=self.config.dropout, rngs=rngs)

        embed_dim = self.config.hidden_size
        self.padding_idx = self.config.pad_token_id
        self.max_target_positions = self.config.max_position_embeddings

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            config.vocab_size,
            config.word_embed_proj_dim,
            embedding_init=nn.initializers.normal(config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.position_offset = offset
        # Use `nn.Embed` directly so HF -> EasyDeL conversion treats this as an embedding
        # (no weight transpose) and maps `*.weight` -> `*.embedding`.
        self.embed_positions = nn.Embed(
            self.config.max_position_embeddings + offset,
            embed_dim,
            embedding_init=nn.initializers.normal(config.init_std),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.project_in = ColumnParallelLinear(
                self.config.word_embed_proj_dim,
                self.config.hidden_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            self.project_out = RowParallelLinear(
                self.config.hidden_size,
                self.config.word_embed_proj_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        else:
            self.project_in = None
            self.project_out = None

        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=param_dtype,
                epsilon=1e-05,
                rngs=rngs,
            )
        else:
            self.final_layer_norm = None

        self.layers = [
            OPTDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the OPTDecoder.

        Args:
            input_ids (Array): Input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (tp.Optional[Array]): Mask to prevent attention to padding tokens.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states for all layers. Defaults to False.

        Returns:
            BaseModelOutput: The decoder's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        input_shape = input_ids.shape
        input_ids = input_ids.reshape(-1, input_shape[-1])
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        inputs_embeds = checkpoint_name(self.embed_tokens(input_ids), "embeddings")
        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            # OPT uses absolute (learned) positional embeddings. `MaskInfo` already
            # provides per-token positions via `q_position_ids`.
            position_ids = jnp.clip(mask_info.q_position_ids[:, :sequence_length], min=0).astype(jnp.int32)

        positions = self.embed_positions((position_ids + self.position_offset).astype("i4"))
        hidden_states = inputs_embeds + positions

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                mask_info=mask_info,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_self_attns += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        if self.final_layer_norm is not None:
            hidden_state = checkpoint_name(self.final_layer_norm(hidden_states), "model_output")
        else:
            hidden_state = checkpoint_name(hidden_states, "model_output")

        if self.project_out is not None:
            hidden_state = self.project_out(hidden_state)

        if output_hidden_states:
            all_hidden_states += (hidden_state,)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            past_key_values=past_key_values,
        )


class OPTModel(EasyDeLBaseModule):
    """Base OPT Model class.

    This class represents the core OPT model architecture, consisting primarily of the OPTDecoder.

    Attributes:
        config (OPTConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        decoder (OPTDecoder): The OPT decoder stack.
    """

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.decoder = OPTDecoder(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            offset=offset,
        )

    def _get_decoder_module(self):
        return self.decoder

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass of the OPTModel.

        Args:
            input_ids (Array): Input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (tp.Optional[Array]): Mask to prevent attention to padding tokens.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]): Metadata for paged attention.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_hidden_states (bool): Whether to return hidden states for all layers. Defaults to False.

        Returns:
            BaseModelOutput: The model's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        return BaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            past_key_values=decoder_outputs.past_key_values,
        )

    def set_embeddings(self, value):
        """Sets the input embeddings for the model."""
        self.decoder.embed_tokens = value

    def get_embedding(self):
        """Gets the input embeddings from the model."""
        return self.decoder.embed_tokens

    def get_decoder(self):
        """Returns the decoder part of the model."""
        return self.decoder


@register_module(TaskType.CAUSAL_LM, config=OPTConfig, model_type="opt")
class OPTForCausalLM(BaseCausalLMModule[OPTModel, OPTConfig]):
    """OPT Model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "opt"
    _config_class = OPTConfig

    def __init__(
        self,
        config: OPTConfig,
        offset: int = 2,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ) -> None:
        """Initialize OPT for Causal LM.

        Args:
            config: OPT configuration
            offset: Offset for position embeddings (OPT-specific, default 2)
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: JAX precision setting
            rngs: Random number generators
        """
        # Pre-instantiate base model with OPT-specific offset parameter
        base_model = OPTModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            offset=offset,
        )

        super().__init__(
            config=config,
            base_model=base_model,  # Pass pre-instantiated model
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            tie_word_embeddings=config.tie_word_embeddings if hasattr(config, "tie_word_embeddings") else False,
        )

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """Apply LM head, supporting tied embeddings without mutating params."""
        if self.config.tie_word_embeddings:
            shared_kernel = self.base_model.decoder.embed_tokens.embedding.value.T
            return self.lm_head(hidden_states, w=shared_kernel)
        return self.lm_head(hidden_states)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings=None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
    ):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        if starts is None:
            starts = self.compute_prefill_length(input_ids, pad_token_id)
        past_key_values = self.init_cache(
            batch_size,
            max_length,
            starts,
            shardings,
            pad_token_id,
        )
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="b1")

        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return self.prepare_inputs_for_call(
            **{
                "past_key_values": past_key_values,
                "attention_mask": extended_attention_mask,
                "position_ids": position_ids,
            }
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
