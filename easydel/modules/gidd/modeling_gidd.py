# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi) and @dvruette.
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

"""
This module provides the core components of the GIDD model, including:
- GiddMLP: A feed-forward network with squared ReLU activation
- GiddAttention: An attention mechanism with optional query-key normalization
- GiddRMSNorm: Root Mean Square normalization layer
- GiddLayer: A transformer layer combining attention and MLP components
- GiddModel: The base transformer model
- GiddForDiffusionLM: A version of the model adapted for diffusion language modeling

The implementation leverages JAX for efficient computation and supports various
optimizations including gradient checkpointing, mixed precision, and model parallelism.
"""

import typing as tp
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import auto_remat, block_wise_ffn, get_dot_general_by_bits
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagesCache,
    PagesCacheView,
    PagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear

from .gidd_configuration import GiddConfig


class GiddMLP(nn.Module):
    """
    GIDD-specific MLP (Multi-Layer Perceptron) implementation.

    This MLP uses a two-layer structure with a squared ReLU activation function
    between the layers. It's designed to be used within the GIDD transformer layers.

    Attributes:
        config (GiddConfig): Configuration object containing model parameters.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        up_proj (ParallelLinear): First linear layer projecting from hidden_size to intermediate_size.
        down_proj (ParallelLinear): Second linear layer projecting from intermediate_size back to hidden_size.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """
        Initialize the GiddMLP.

        Args:
            config: Configuration object containing model parameters.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators for parameter initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        linear_class = partial(
            ParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=self.config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.up_proj = linear_class(config.hidden_size, config.intermediate_size)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size)

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the MLP.

        Args:
            h: Input tensor of shape [..., hidden_size].

        Returns:
            Output tensor of shape [..., hidden_size].
        """
        # Apply logical sharding for distributed computation
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        # First projection and activation
        h = self.up_proj(h)
        h = nn.relu(h) ** 2  # Squared ReLU activation

        # Second projection
        h = self.down_proj(h)

        # Apply logical sharding for distributed computation
        h = apply_logical_sharding(
            h,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return h


class GiddAttention(AttentionModule):
    """
    GIDD-specific attention mechanism with optional query-key normalization.

    This attention module implements a multi-head attention mechanism with support for
    query-key normalization, rotary position embeddings, and flexible attention patterns.

    Attributes:
        config (GiddConfig): Configuration object containing model parameters.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        hidden_size (int): Dimensionality of the hidden states.
        head_dim (int): Dimensionality of each attention head.
        use_qk_norm (bool): Whether to apply normalization to query and key vectors.
        qk_norm_eps (float): Epsilon value for numerical stability in QK normalization.
        qk_scale (float or nn.Param): Scaling factor for attention scores.
        q_proj (ParallelLinear): Linear projection for queries.
        k_proj (ParallelLinear): Linear projection for keys.
        v_proj (ParallelLinear): Linear projection for values.
        o_proj (ParallelLinear): Linear projection for outputs.
        rotary: Rotary position embedding module.
        attention_performer (FlexibleAttentionModule): Module that performs the actual attention computation.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """
        Initialize the GiddAttention module.

        Args:
            config: Configuration object containing model parameters.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators for parameter initialization.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size

        # Calculate head dimension, allowing for explicit override in config
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", head_dim)

        # QK normalization settings
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_eps = config.qk_norm_eps

        if self.use_qk_norm:
            # Initialize learnable scale parameter for QK normalization
            self.qk_scale = nn.Param(
                jnp.full(
                    (1, 1, self.config.num_attention_heads, 1),
                    2 * jnp.log(config.max_position_embeddings),
                    dtype=self.param_dtype,
                ),
            )
        else:
            # Fixed scale based on head dimension
            self.qk_scale = 1.0

        # Create linear projections for Q, K, V, and O
        linear_class = partial(
            ParallelLinear,
            scale="fan_in",
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.attention_bias,
            kernel_init=jax.nn.initializers.normal(config.init_scale),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.q_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
        )
        self.o_proj = linear_class(config.num_attention_heads * self.head_dim, config.hidden_size, rngs=rngs)

        # Initialize rotary position embeddings
        self.rotary = self.config.get_basic_rope(
            self.dtype,
            self.head_dim,
            self.head_dim,
            True,
        )

        # Initialize attention performer
        self.attention_performer = FlexibleAttentionModule(
            base_config=self.config,
            softmax_scale=1.0 if self.use_qk_norm else 1.0 / self.head_dim**0.5,
            dropout_prob=0.0,
        )

    @jax.named_scope("gidd-flax-attention-concatenate")
    def concatenate(
        self,
        *,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        attention_mask: chex.Array,
        noise_mask: chex.Array,
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
    ) -> tuple[chex.Array, chex.Array, chex.Array, tp.Callable[[], chex.Array]]:
        """
        Prepare and concatenate key, value, and attention mask for attention computation.

        This method handles the preprocessing of attention inputs, including:
        - Validating and reshaping attention masks
        - Combining attention masks with noise masks
        - Creating a function to initialize attention bias

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim].
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim].
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim].
            attention_mask: Binary mask of shape [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len].
            noise_mask: Binary mask for noise tokens of shape [batch_size, seq_len].
            cache_view: View into the key/value cache for incremental decoding.
            cache_metadata: Metadata for cache operations.

        Returns:
            A tuple containing:
            - key: Processed key tensor.
            - value: Processed value tensor.
            - attention_mask: Processed attention mask.
            - init_attention_bias: Function to initialize attention bias.
            - cache_view: Updated cache view.
        """
        # Validate that query and key have matching sequence lengths
        assert query.shape[1] == key.shape[1], "Query and Key lengths must match for GIDD attention."

        # Process attention mask
        if attention_mask is not None:
            if attention_mask.dtype != jnp.bool:
                warnings.warn("attention_mask should be a boolean array", stacklevel=1)
                attention_mask = (attention_mask == 1).astype("b1")

        # Expand attention mask to match attention computation dimensions
        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_mask = jnp.repeat(attention_mask, query.shape[1], -2)  # [Batch, 1, q_len, kv_len]

        # Process noise mask if provided
        if noise_mask is not None:
            if noise_mask.dtype != jnp.bool:
                warnings.warn("noise_mask should be a boolean array", stacklevel=1)
                noise_mask = (noise_mask == 1).astype("b1")

            # Create noise attention mask
            noise_mask_q = jnp.expand_dims(noise_mask, axis=-1)
            noise_mask_kv = jnp.expand_dims(noise_mask, axis=-2)
            noise_attn_mask = jnp.expand_dims(noise_mask_q >= noise_mask_kv, axis=-3)

            # Combine with attention mask
            attention_mask = jnp.logical_and(attention_mask, noise_attn_mask)

        # Function to initialize attention bias
        def init_attention_bias():
            return jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )

        return key, value, attention_mask, init_attention_bias, cache_view

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply normalization to query or key vectors.

        Args:
            x: Input tensor of shape [..., num_heads, head_dim].

        Returns:
            Normalized tensor of the same shape.
        """
        return x * jax.lax.rsqrt(jnp.square(x).sum(-1, keepdims=True) + self.qk_norm_eps)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        noise_mask: chex.Array,
        position_ids: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Forward pass through the attention module.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask: Binary mask for attention.
            noise_mask: Binary mask for noise tokens.
            position_ids: Position indices for rotary embeddings.
            mode: Runtime mode (train, decode, etc.).
            cache_view: View into the key/value cache.
            cache_metadata: Metadata for cache operations.
            segment_ids: Segment indices for segment embeddings.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed frequencies for rotary embeddings.

        Returns:
            AttentionLayerOutput containing:
            - attention_output: Output tensor of shape [batch_size, seq_len, hidden_size].
            - attention_weight: Attention weights if output_attentions is True.
            - cache_view: Updated cache view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        # Project inputs to Q, K, V
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        # Apply QK normalization if enabled
        if self.use_qk_norm:
            query_states = self._norm(query_states)
            key_states = self._norm(key_states)

        # Reshape for multi-head attention
        qshape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        kv_shape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        query_states = query_states.reshape(qshape)
        key_states = key_states.reshape(kv_shape)
        value_states = value_states.reshape(kv_shape)

        # Apply sharding for distributed computation
        (
            query_states,
            key_states,
            value_states,
        ) = self.apply_qkv_shardings(query_states, key_states, value_states)

        # Apply rotary position embeddings
        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        # Prepare inputs for attention computation
        (
            key_states,
            value_states,
            attention_mask,
            init_attention_bias,
            cache_view,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            value=value_states,
            attention_mask=attention_mask,
            noise_mask=noise_mask,
        )

        # Compute attention
        attentions = self.attention_performer.forward(
            query_states=query_states * self.qk_scale,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=False,
        )

        # Project attention outputs back to hidden dimension
        attn_output = self.o_proj(self.shard_attention_prod(attn_output=self._merge_heads(attentions.attention_outputs)))

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class GiddRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) for the GIDD model.

    RMSNorm is a simplified variant of LayerNorm that normalizes the input by
    its root mean square value and applies a learnable scale parameter.

    Attributes:
        config (GiddConfig): Configuration object containing model parameters.
        epsilon (float): Small constant for numerical stability.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        kernel (nn.Param): Learnable scale parameter.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        """
        Initialize the GiddRMSNorm.

        Args:
            config: Configuration object containing model parameters.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
        """
        self.config = config
        self.epsilon = self.config.rms_norm_eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = nn.Param(jnp.zeros(self.config.hidden_size, dtype=param_dtype))

    def __call__(self, hidden_states):
        """
        Apply RMSNorm to the input tensor.

        Args:
            hidden_states: Input tensor of shape [..., hidden_size].

        Returns:
            Normalized tensor of the same shape.
        """
        variance = hidden_states.astype(jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)

        # Normalize and apply scale
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        return (1 + self.kernel.value.astype(self.dtype)) * jnp.asarray(hidden_states, dtype=self.dtype)


class GiddLayer(nn.Module):
    """
    A single transformer layer for the GIDD model.

    This layer combines a self-attention mechanism with a feed-forward network (MLP),
    using residual connections and layer normalization. It's designed to be stacked
    to form the complete transformer model.

    Attributes:
        config (GiddConfig): Configuration object containing model parameters.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        resid_scale (float): Scaling factor for residual connections.
        self_attn (GiddAttention): Self-attention module.
        mlp (GiddMLP): Feed-forward network module.
        input_layernorm (GiddRMSNorm): Layer normalization before attention.
        post_attention_layernorm (GiddRMSNorm): Layer normalization before MLP.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        resid_scale: float = 1.0,
    ):
        """
        Initialize the GiddLayer.

        Args:
            config: Configuration object containing model parameters.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators for parameter initialization.
            resid_scale: Scaling factor for residual connections. Defaults to 1.0.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.resid_scale = resid_scale

        # Apply gradient checkpointing if enabled
        attn_block = GiddAttention
        mlp_block = GiddMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )

        # Initialize sub-modules
        self.self_attn: GiddAttention = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp: GiddMLP = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)
        self.post_attention_layernorm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        noise_mask: chex.Array,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass through the transformer layer.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask: Binary mask for attention.
            position_ids: Position indices for rotary embeddings.
            noise_mask: Binary mask for noise tokens.
            mode: Runtime mode (train, decode, etc.).
            cache_view: View into the key/value cache.
            cache_metadata: Metadata for cache operations.
            segment_ids: Segment indices for segment embeddings.
            output_attentions: Whether to return attention weights.
            frequencies: Precomputed frequencies for rotary embeddings.

        Returns:
            DecoderLayerOutput containing:
            - hidden_states: Output tensor of shape [batch_size, seq_len, hidden_size].
            - attention_weight: Attention weights if output_attentions is True.
            - cache_view: Updated cache view.
        """
        # Self-attention block with residual connection
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            noise_mask,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            frequencies,
        )
        hidden_states = hidden_states + self.resid_scale * attn_outputs.attention_output

        # Feed-forward block with residual connection
        feed_forward_input = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            # Use block-wise computation for memory efficiency
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)
        hidden_states = hidden_states + self.resid_scale * feed_forward_hidden_states

        # Apply logical sharding for distributed computation
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GiddConfig, model_type="Gidd")
class GiddModel(EasyDeLBaseModule):
    """
    Base GIDD model implementation.

    This model implements the core transformer architecture of the GIDD model,
    consisting of an embedding layer, multiple transformer layers, and a final
    normalization layer. It serves as the foundation for more specialized models
    like GiddForDiffusionLM.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        resid_scale (float): Scaling factor for residual connections.
        embed_tokens (nn.Embed): Token embedding layer.
        layers (list[GiddLayer]): List of transformer layers.
        norm (GiddRMSNorm): Final normalization layer.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """
        Initialize the GiddModel.

        Args:
            config: Configuration for the model.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators for parameter initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Calculate residual scale factor
        self.resid_scale = config.resid_scale / config.num_hidden_layers

        # Initialize token embeddings
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.emb_init_scale),
            rngs=rngs,
        )

        # Initialize transformer layers
        self.layers = [
            GiddLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                resid_scale=self.resid_scale,
            )
            for _ in range(self.config.num_hidden_layers)
        ]

        self.norm = GiddRMSNorm(config=config, dtype=dtype, param_dtype=param_dtype)

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        log_snr: chex.Array | None = None,
        noise_mask: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """
        Forward pass through the GiddModel.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            inputs_embeds: Input embeddings of shape [batch_size, seq_len, hidden_size].
            attention_mask: Binary mask to avoid attention on padding tokens.
            position_ids: Position indices of each input sequence token.
            log_snr: Log signal-to-noise ratio for diffusion models.
            noise_mask: Binary mask for noise tokens.
            segment_ids: Segment token indices for segment embeddings.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cache containing precomputed key/value states.
            cache_metadata: Metadata for cache handling.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states of all layers.

        Returns:
            BaseModelOutput containing:
            - last_hidden_state: Final hidden state of shape [batch_size, seq_len, hidden_size].
            - hidden_states: Hidden states of all layers if output_hidden_states is True.
            - attentions: Attention weights of all layers if output_attentions is True.
            - past_key_values: Updated cache with key/value states.
        """
        # Validate input
        if (input_ids is None) ^ (inputs_embeds is None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        batch_size, sequence_length, _ = inputs_embeds.shape

        # Initialize outputs
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Validate sequence length
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        # Process attention mask
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        # Start with input embeddings
        hidden_states = inputs_embeds

        # Determine runtime mode
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        # Initialize cache if not provided
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        # Apply logical sharding for distributed computation
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        # Process through transformer layers
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                noise_mask=noise_mask,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                segment_ids=segment_ids,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.

        Note:
            This is a decoder-only model and does not have an encoder.

        Raises:
            NotImplementedError: Always raised as this is a decoder-only model.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.

        Returns:
            The model itself, as it is a decoder-only model.
        """
        return self

    def get_lm_head(self):
        """
        Returns the language model head of the module.

        Note:
            The base model does not have a language model head.

        Raises:
            NotImplementedError: Always raised as the base model does not have an LM head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.

        Returns:
            The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.DIFFUSION_LM, config=GiddConfig, model_type="Gidd")
class GiddForDiffusionLM(EasyDeLBaseModule):
    """
    GIDD model with a language modeling head for diffusion language modeling tasks.

    This model extends the base GiddModel with a language modeling head, making it
    suitable for autoregressive language generation tasks, particularly in the
    context of diffusion models.

    Attributes:
        config (GiddConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        model (GiddModel): The base transformer model.
        lm_head (ParallelLinear): Language modeling head.
    """

    def __init__(
        self,
        config: GiddConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """
        Initialize the GiddForDiffusionLM.

        Args:
            config: Configuration for the model.
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: Precision setting for JAX operations. Defaults to None.
            rngs: Random number generators for parameter initialization.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Initialize base model
        self.model = GiddModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Initialize language modeling head
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.head_init_scale),
            precision=self.precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        log_snr: chex.Array | None = None,
        noise_mask: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """
        Forward pass through the GiddForDiffusionLM.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            inputs_embeds: Input embeddings of shape [batch_size, seq_len, hidden_size].
            attention_mask: Binary mask to avoid attention on padding tokens.
            position_ids: Position indices of each input sequence token.
            segment_ids: Segment token indices for segment embeddings.
            log_snr: Log signal-to-noise ratio for diffusion models.
            noise_mask: Binary mask for noise tokens.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cache containing precomputed key/value states.
            cache_metadata: Metadata for cache handling.
            apply_lm_head: Whether to apply the language modeling head.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states of all layers.

        Returns:
            CausalLMOutput containing:
            - logits: Output logits of shape [batch_size, seq_len, vocab_size] if apply_lm_head is True.
            - hidden_states: Hidden states of all layers if output_hidden_states is True.
            - last_hidden_state: Final hidden state of shape [batch_size, seq_len, hidden_size].
            - attentions: Attention weights of all layers if output_attentions is True.
            - past_key_values: Updated cache with key/value states.
        """
        # Get outputs from base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            log_snr=log_snr,
            noise_mask=noise_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
        )

        hidden_states = outputs.last_hidden_state

        # Apply logical sharding for distributed computation
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        # Apply language modeling head if requested
        lm_logits = None
        if apply_lm_head:
            lm_logits = self.apply_lm_head(hidden_states)

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.

        Note:
            This is a decoder-only model and does not have an encoder.

        Raises:
            NotImplementedError: Always raised as this is a decoder-only model.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.

        Returns:
            The base model, which serves as the decoder.
        """
        return self.model

    def get_lm_head(self):
        """
        Returns the language model head of the module.

        Returns:
            The language modeling head.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.

        Returns:
            The token embedding layer from the base model.
        """
        return self.model.embed_tokens
