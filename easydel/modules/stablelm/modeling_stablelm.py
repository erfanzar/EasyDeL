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


from functools import cached_property, partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn, get_dot_general_by_bits
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

from .stablelm_configuration import StableLmConfig


class StableLmMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) block for the StableLM model.

    Attributes:
        config (StableLmConfig): Configuration object for the model.
        gate_proj (ParallelLinear): Linear layer for the gating mechanism.
        down_proj (ParallelLinear): Linear layer for down-projection.
        up_proj (ParallelLinear): Linear layer for up-projection.
        act_fn (callable): Activation function (specified in config).
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
    """

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmMLP module.

        Args:
            config (StableLmConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.gate_proj = linear_class(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = linear_class(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the MLP block.

        Args:
            hidden_states (jnp.ndarray): Input hidden states.

        Returns:
            jnp.ndarray: Output hidden states after MLP transformation.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class StableLmLayerNormPerHead(nn.Module):
    """Applies Layer Normalization independently to each attention head's dimension.

    Attributes:
        norms (list[nn.LayerNorm]): List of LayerNorm modules, one per head.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        eps: float = 1e-5,
        bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmLayerNormPerHead module.

        Args:
            head_dim (int): The dimension of each attention head.
            num_heads (int): The number of attention heads.
            eps (float): Epsilon value for LayerNorm (default: 1e-5).
            bias (bool): Whether to include bias in LayerNorm (default: False).
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            rngs (nn.Rngs): Random number generators.
        """
        self.norms = [
            nn.LayerNorm(
                head_dim,
                epsilon=eps,
                use_bias=bias,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for idx in range(num_heads)
        ]

    def __call__(self, hidden_states):
        """Applies LayerNorm per head.

        Args:
            hidden_states (chex.Array): Input hidden states, expected shape (..., num_heads * head_dim).

        Returns:
            chex.Array: Hidden states after applying LayerNorm per head, same shape as input.
        """
        # hidden_states: [batch, seq_len, num_heads * head_dim]
        # Reshape to [batch, seq_len, num_heads, head_dim]
        states_per_heads = jnp.split(hidden_states, 1, axis=1)
        # Normalize and merge the heads back together
        return jnp.concatenate(
            [norm(hidden_states) for norm, hidden_states in zip(self.norms, states_per_heads, strict=False)],
            axis=1,
        )


class StableLmAttention(AttentionModule):
    """StableLM Attention module with Rotary Position Embeddings and optional LayerNorm on QK.

    Attributes:
        config (StableLmConfig): Configuration object for the model.
        hidden_size (int): Dimensionality of the hidden states.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        num_key_value_heads (int): Number of key/value heads (for GQA).
        num_key_value_groups (int): Number of query heads per key/value head.
        max_position_embeddings (int): Maximum sequence length.
        rope_theta (float): Base value for RoPE.
        partial_rotary_factor (float): Factor determining the portion of head dimension subject to RoPE.
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        o_proj (ParallelLinear): Linear layer for output projection.
        rotary_emb_dim (int): Dimensionality of the rotary embeddings.
        attention_performer (FlexibleAttentionModule): Module for performing attention computation.
        qk_layernorm (bool): Whether to apply LayerNorm to query and key states.
        q_layernorm (StableLmLayerNormPerHead): LayerNorm for query states (if qk_layernorm is True).
        k_layernorm (StableLmLayerNormPerHead): LayerNorm for key states (if qk_layernorm is True).
        rotary (RotaryEmbedding): Rotary positional embedding module.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmAttention module.

        Args:
            config (StableLmConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(config=config)
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if self.num_key_value_groups == 1:
            assert self.config.num_attention_heads == self.config.num_key_value_heads

        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.q_proj = linear_class(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            use_bias=self.config.use_qkv_bias,
            rngs=rngs,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=self.config.use_qkv_bias,
            rngs=rngs,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            use_bias=self.config.use_qkv_bias,
            rngs=rngs,
        )
        self.o_proj = linear_class(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            rngs=rngs,
        )

        self.rotary_emb_dim = int(self.config.partial_rotary_factor * self.head_dim)
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = StableLmLayerNormPerHead(
                head_dim=self.head_dim,
                num_heads=config.num_attention_heads,
                eps=config.layer_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.k_layernorm = StableLmLayerNormPerHead(
                head_dim=self.head_dim,
                num_heads=config.num_key_value_heads,
                eps=config.layer_norm_eps,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )

        self.rotary = self.config.get_basic_rope(
            self.dtype,
            head_size=int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads)),
            rotary_dim=self.rotary_emb_dim,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the attention module.

        Args:
            hidden_states (chex.Array): Input hidden states (batch, seq_len, hidden_size).
            attention_mask (chex.Array): Mask to apply on the attention scores (batch, 1, seq_len, kv_seq_len).
            position_ids (chex.Array): Position indices for the tokens (batch, seq_len).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]):
                Cache view for key/value states (optional).
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]):
                Metadata for paged attention (optional).
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): If True, outputs attention weights alongside the hidden states (default: False).
            fcm_mask (tp.Optional[chex.Array]): Forward causal mask (FCM) mask (optional).
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequencies (optional).

        Returns:
            tp.Tuple[chex.Array, chex.Array | None]: A tuple containing the attention output
                (batch, seq_len, hidden_size)
                and optionally the attention weights (batch, num_heads, seq_len, kv_seq_len).
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            key_states = self.k_layernorm(key_states.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self.rotary(
            positions=position_ids,
            query=query_states,
            key=key_states,
            frequencies=frequencies,
        )

        (
            key_states,
            value_states,
            attention_mask,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.o_proj(attn_output)
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class StableLmDecoderLayer(nn.Module):
    """A single decoder layer for the StableLM model.

    This layer combines self-attention, MLP, and residual connections with layer normalization.
    It supports parallel residual connections.

    Attributes:
        config (StableLmConfig): Configuration object for the model.
        self_attn (StableLmAttention): Self-attention module.
        mlp (StableLmMLP): MLP module.
        input_layernorm (nn.LayerNorm): Layer normalization applied before self-attention.
        post_attention_layernorm (nn.LayerNorm): Layer normalization applied after self-attention and before the MLP.
        dropout_rng_key (str): Name of the RNG key for dropout.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmDecoderLayer module.

        Args:
            config (StableLmConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = StableLmAttention
        mlp_block = StableLmMLP
        self.use_parallel_residual = self.config.use_parallel_residual
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
        )
        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if not self.use_parallel_residual:
            self.post_attention_layernorm = nn.LayerNorm(
                config.hidden_size,
                epsilon=config.layer_norm_eps,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        self.dropout = nn.Dropout(self.config.hidden_dropout, rngs=rngs)

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the decoder layer.

        Args:
            hidden_states (chex.Array): Input hidden states (batch, seq_len, hidden_size).
            attention_mask (chex.Array): Attention mask (batch, 1, seq_len, kv_seq_len).
            position_ids (chex.Array): Position IDs (batch, seq_len).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]):
                Cache view for key/value states (optional).
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]):
                Metadata for paged attention (optional).
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to output attention weights (default: False).
            fcm_mask (tp.Optional[chex.Array]): Forward causal mask (FCM) mask (optional).
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequencies (optional).

        Returns:
            tp.Tuple[chex.Array, chex.Array | None]: A tuple containing:
                - hidden_states (chex.Array): Output hidden states after the decoder layer.
                - attention_outputs (chex.Array | None): Attention weights (if `output_attentions` is True).
        """
        assert hidden_states.ndim == 3, f"Input hidden_states should be 3 dimensions, got {hidden_states.ndim}"

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        attn_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            causal_mask,
            mode,
            cache_view,
            cache_metadata,
            segment_ids,
            output_attentions,
            fcm_mask,
            frequencies,
        )

        if self.use_parallel_residual:
            if self.config.use_scan_mlp:
                hidden_states = block_wise_ffn(self.mlp, hidden_states, self.config.scan_mlp_chunk_size)
            else:
                hidden_states = self.mlp(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + residual + attn_outputs.attention_output
        else:
            residual = residual + attn_outputs.attention_output
            if self.config.use_scan_mlp:
                hidden_states = block_wise_ffn(
                    self.mlp,
                    self.post_attention_layernorm(residual),
                    self.config.scan_mlp_chunk_size,
                )
            else:
                hidden_states = self.mlp(self.post_attention_layernorm(residual))
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + residual

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


@register_module(TaskType.BASE_MODULE, config=StableLmConfig, model_type="stablelm")
class StableLmModel(EasyDeLBaseModule):
    """The base StableLM transformer model.

    This class implements the core transformer architecture, including embedding layers,
    decoder layers, and final normalization.

    Attributes:
        config (StableLmConfig): Configuration object for the model.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (nn.List[StableLmDecoderLayer]): List of decoder layers.
        norm (nn.LayerNorm): Final layer normalization.
        gradient_checkpointing (str): Gradient checkpointing strategy.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmModel module.

        Args:
            config (StableLmConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            StableLmDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.norm = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        """Cached property for precomputed rotary frequencies."""
        rotary_emb_dim = int(
            self.config.partial_rotary_factor * (self.config.hidden_size // self.config.num_attention_heads)
        )
        self._frequencies = self.config.get_basic_frequencies(
            head_size=rotary_emb_dim,
            rotary_dim=rotary_emb_dim,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> BaseModelOutput:
        """Forward pass of the StableLM model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs (batch, seq_len).
                Mutually exclusive with `inputs_embeds`.
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings (batch, seq_len, hidden_size).
                Mutually exclusive with `input_ids`.
            attention_mask (tp.Optional[chex.Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[chex.Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all layers
                (default defined by config).
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for caching.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]):
                Metadata for paged attention (optional).

        Returns:
            BaseModelOutput: The model output, either as a `BaseModelOutput` object or a tuple.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided or neither is provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)
        if attention_mask.ndim == 2:
            attention_mask = jnp.expand_dims(attention_mask, (1, 2))
        hidden_states = inputs_embeds

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
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                causal_mask=self.causal_mask,
                output_attentions=output_attentions,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

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
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

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
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=StableLmConfig, model_type="stablelm")
class StableLmForCausalLM(EasyDeLBaseModule):
    """StableLM model with a Causal Language Modeling (CLM) head.

    This class wraps the base `StableLmModel` and adds a linear layer (language model head)
    to predict the next token logits.

    Attributes:
        config (StableLmConfig): Configuration object for the model.
        model (StableLmModel): The base StableLM model.
        lm_head (ParallelLinear): The language model head (linear layer).
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the StableLmForCausalLM module.

        Args:
            config (StableLmConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = StableLmModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = self.config.vocab_size
        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Forward pass of the StableLM model for Causal Language Modeling.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs (batch, seq_len).
                Mutually exclusive with `inputs_embeds`.
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings (batch, seq_len, hidden_size).
                Mutually exclusive with `input_ids`.
            attention_mask (tp.Optional[chex.Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[chex.Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all
                layers (default defined by config).
            past_key_values (tp.Optional[TransformerCache | PagesCache]):
                Precomputed key/value states for caching.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]):
                Metadata for paged attention (optional).

        Returns:
            CausalLMOutput: The model output, including logits, hidden states, and attentions.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
        )
        hidden_states = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

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
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()
