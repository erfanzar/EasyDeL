# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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


from functools import partial

import chex
import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import (
    ACT2FN,
    auto_remat,
    block_wise_ffn,
    get_dot_general_by_bits,
)
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule
from easydel.layers.caching import (
    PagedAttentionCache,
    PagedAttentionCacheView,
    PagedAttentionMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ParallelLinear
from easydel.layers.norms import RMSNorm as RMSNorm

from .qwen3_moe_configuration import Qwen3MoeConfig


class Qwen3MoeMLP(nn.Module):
    """Qwen3Moe MLP module.

    This module implements the feed-forward network (MLP) used in the Qwen3Moe model.
    It uses a Gated Linear Unit (GLU) structure with SiLU activation.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        gate_proj (ParallelLinear): Linear layer for the GLU gate.
        down_proj (ParallelLinear): Linear layer for the down projection.
        up_proj (ParallelLinear): Linear layer for the GLU value.
        act_fn (callable): Activation function (SiLU).
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        intermediate_size=None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeMLP module.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        linear_class = partial(
            ParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(
            config.hidden_size,
            intermediate_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = linear_class(
            config.hidden_size,
            intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the Qwen3MoeMLP module.

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


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts (MoE) block for Qwen3 MoE.

    This block routes input hidden states to a selected subset of experts
    and combines their outputs.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        gate (ParallelLinear): Linear layer for the gating network.
        experts (nn.List[Qwen3MoeMLP]): List of expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeSparseMoeBlock module.

        Args:
            config (Qwen3MoeConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = [
            Qwen3MoeMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                intermediate_size=config.moe_intermediate_size,
                rngs=rngs,
            )
            for i in range(self.config.num_experts)
        ]

    def __call__(self, hidden_states: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Forward pass of the Sparse MoE block.

        Args:
            hidden_states (chex.Array): Input hidden states (batch_size * sequence_length, hidden_dim).

        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing:
                - final_hidden_states (chex.Array): The output hidden states after MoE processing.
                - router_logits (chex.Array): The logits output by the gating network.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        router_logits = self.gate(hidden_states)
        routing_weights = jax.nn.softmax(router_logits.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1)

        routing_weights, selected_experts = jax.lax.top_k(
            routing_weights,
            k=self.config.num_experts_per_tok,
        )

        if self.config.norm_topk_prob:
            routing_weights /= routing_weights.sum(axis=-1, keepdims=True)
        final_hidden_state = jnp.zeros_like(hidden_states)

        for index in range(self.config.num_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_layer_output_exp = (
                jnp.sum(jnp.multiply(selected_experts == index, routing_weights), axis=-1)[:, :, None]
                * expert_layer_output
            )
            final_hidden_state += expert_layer_output_exp
        return (final_hidden_state, router_logits)


class Qwen3MoeAttention(AttentionModule):
    """Qwen3Moe Attention module.

    This module implements the multi-head attention mechanism used in the Qwen3Moe model.
    It supports Grouped Query Attention (GQA) and Rotary Position Embeddings (RoPE).

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        hidden_size (int): Dimensionality of the hidden states.
        head_dim (int): Dimensionality of each attention head.
        num_key_value_groups (int): Number of query head groups for each key/value head.
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        o_proj (ParallelLinear): Linear layer for the output projection.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
        rotary (RoPE): Rotary position embedding module.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeAttention module.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
            layer_idx (int): The index of the layer in the model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_attention_heads`.
        """
        super().__init__(config=config)
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads

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
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.o_proj = linear_class(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=config.attention_bias,
        )

        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )
        self.q_norm = RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.k_norm = RMSNorm(
            dim=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.rotary = self.config.get_basic_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            base=config.rope_theta,
            dtype=self.dtype,
        )
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """
        Forward pass of the Qwen3MoeAttention module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores. Shape:
                (batch_size, 1, query_length, key_length).
            position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagedAttentionCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

        Returns:
            tp.Union[tp.Tuple[chex.Array, chex.Array], tp.Tuple[chex.Array]]:
                A tuple containing the attention output hidden states. If `output_attentions` is True,
                it also includes the attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = self.q_norm(
            query_states.reshape(
                batch_size,
                sequence_length,
                self.config.num_attention_heads,
                self.head_dim,
            )
        )
        key_states = self.k_norm(
            key_states.reshape(
                batch_size,
                sequence_length,
                self.config.num_key_value_heads,
                self.head_dim,
            )
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

        (
            query_states,
            key_states,
            value_states,
        ) = self.apply_qkv_shardings(query_states, key_states, value_states)

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
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            fcm_mask=fcm_mask,
            sliding_window=self.sliding_window,
        )

        attentions = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            sliding_window=self.sliding_window,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            causal=True,
        )
        attn_output = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen3MoeDecoderLayer(nn.Module):
    """Qwen3Moe Transformer Decoder Layer.

    This module represents a single decoder layer in the Qwen3Moe model,
    combining self-attention and MLP sub-layers with residual connections
    and RMS normalization.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        layer_idx (int): The index of the layer in the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        input_layernorm (RMSNorm): RMS normalization applied before the attention layer.
        self_attn (Qwen3MoeAttention): The self-attention module.
        mlp (Qwen3MoeMLP): The feed-forward (MLP) module.
        post_attention_layernorm (RMSNorm): RMS normalization applied after the attention layer and before the MLP layer.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeDecoderLayer.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
            layer_idx (int): The index of the layer in the model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = Qwen3MoeAttention
        mlp_block = Qwen3MoeMLP
        moe_block = Qwen3MoeSparseMoeBlock
        attn_block, mlp_block, moe_block = auto_remat(
            attn_block,
            mlp_block,
            moe_block,
            policy=config.gradient_checkpointing,
        )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.is_moe = (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if self.is_moe:
            self.mlp = moe_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = mlp_block(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: chex.Array,
        attention_mask: chex.Array,
        position_ids: chex.Array,
        causal_mask: chex.Array | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagedAttentionCacheView | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ):
        """Forward pass of the Qwen3MoeDecoderLayer module.

        Args:
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores. Shape:
                (batch_size, 1, query_length, key_length).
            position_ids (chex.Array): Position indices for the tokens. Shape: (batch_size, sequence_length).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for ensuring autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | PagedAttentionCacheView]): Cache view for attention KVs.
            cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (tp.Optional[chex.Array]): Flash Chunking Mask (FCM) for attention.
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequency embeddings.

        Returns:
            tp.Tuple[chex.Array, tp.Optional[chex.Array]]:
                A tuple containing the output hidden states and optionally the attention weights.
        """

        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
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
        hidden_states = hidden_states + attn_outputs.attention_output

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        # if self.config.use_scan_mlp:
        # 	feed_forward_hidden_states = block_wise_ffn(
        # 		self.mlp,
        # 		feed_forward_input,
        # 		self.config.scan_mlp_chunk_size,
        # 	)
        # else:
        feed_forward_hidden_states = self.mlp(feed_forward_input)
        if self.is_moe:
            feed_forward_hidden_states, router_logits = feed_forward_hidden_states
        else:
            router_logits = None
        hidden_states = hidden_states + feed_forward_hidden_states
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3MoeConfig, model_type="qwen3_moe")
class Qwen3MoeModel(EasyDeLBaseModule):
    """The base Qwen3Moe model transformer.

    This class represents the core transformer architecture of the Qwen3Moe model,
    consisting of an embedding layer, multiple Qwen3MoeDecoderLayer layers,
    and a final RMS normalization layer.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[Qwen3MoeDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeModel.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Qwen3MoeDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
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
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass of the Qwen3MoeModel.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.


        Returns:
            MoeModelOutput: The model's output.
                returns a `MoeModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        batch_size, sequence_length, _ = inputs_embeds.shape
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_router_logits = () if output_router_logits else None
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
                output_router_logits=output_router_logits,
                segment_ids=segment_ids,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )


@register_module(TaskType.CAUSAL_LM, config=Qwen3MoeConfig, model_type="qwen3_moe")
class Qwen3MoeForCausalLM(EasyDeLBaseModule):
    """Qwen3Moe model with a Causal Language Modeling head.

    This model consists of the base Qwen3Moe transformer (`Qwen3MoeModel`) followed by a
    linear layer (`lm_head`) that projects the transformer's output hidden states
    to the vocabulary size, producing logits for next token prediction.
    Optionally, the input token embeddings can be tied to the output projection layer.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        model (Qwen3MoeModel): The core Qwen3Moe transformer model.
        lm_head (ParallelLinear): The linear layer for projecting hidden states to vocabulary logits.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeForCausalLM model.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = Qwen3MoeModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.lm_head = ParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
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
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass of the Qwen3MoeForCausalLM model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.

        Returns:
            MoeCausalLMOutput: The model's output.
                returns a `MoeCausalLMOutput` object containing `logits`, `hidden_states` (optional),
                and `attentions` (optional).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
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

        if self.config.tie_word_embeddings:
            lm_logits = jax.lax.dot_general(
                hidden_states,
                self.model.embed_tokens.embedding.value.T,
                (((hidden_states.ndim - 1), (0,)), ((), ())),
            )
        else:
            lm_logits = self.lm_head(hidden_states)

        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=outputs.router_logits,
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
            aux_loss += aux_loss * self.config.router_aux_loss_coef

        return MoeCausalLMOutput(
            logits=lm_logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Qwen3MoeConfig, model_type="qwen3_moe")
class Qwen3MoeForSequenceClassification(EasyDeLBaseModule):
    """Qwen3Moe model with a Sequence Classification head.

    This model consists of the base Qwen3Moe transformer (`Qwen3MoeModel`) followed by a
        linear layer (`score`) that projects the transformer's output hidden states
        (typically the hidden state of the last token or a pooled representation) to the number of classes
        for classification.

    Attributes:
        config (Qwen3MoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        model (Qwen3MoeModel): The core Qwen3Moe transformer model.
        score (ParallelLinear): The linear layer for classification.
    """

    def __init__(
        self,
        config: Qwen3MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MoeForSequenceClassification model.

        Args:
            config (Qwen3MoeConfig): The configuration object for the Qwen3Moe model.
                Must include `num_labels`.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.

        Raises:
            AssertionError: If `config.num_labels` is not defined.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.model = Qwen3MoeModel(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        assert hasattr(config, "num_labels"), (
            "in order to use `SequenceClassification` Models in `EasyDeL` "
            "you first need to attach `num_labels` to model `config`"
        )
        self.score = ParallelLinear(
            self.config.hidden_size,
            config.num_labels,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: chex.Array | None = None,
        inputs_embeds: chex.Array | None = None,
        attention_mask: chex.Array | None = None,
        position_ids: chex.Array | None = None,
        segment_ids: chex.Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagedAttentionCache | None = None,
        cache_metadata: TransformerMetadata | PagedAttentionMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the Qwen3MoeForSequenceClassification model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs. Shape: (batch_size, sequence_length).
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (tp.Optional[chex.Array]): Mask to avoid performing attention on padding token indices.
                Shape: (batch_size, sequence_length).
            position_ids (tp.Optional[chex.Array]): Position indices for the tokens.
                Shape: (batch_size, sequence_length).
            segment_ids (tp.Optional[chex.Array]): Segment IDs (unused).
            past_key_values (tp.Optional[TransformerCache | PagedAttentionCache]):
                Precomputed key/value states for attention.
            cache_metadata (tp.Optional[TransformerMetadata | PagedAttentionMetadata]): Metadata for paged attention.
            output_attentions (tp.Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (tp.Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.


        Returns:
            SequenceClassifierOutput: The model's output,
                returns a `SequenceClassifierOutput` object containing `logits`, `hidden_states` (optional),
                and `attentions` (optional).

        Raises:
            ValueError: If `config.pad_token_id` is None and `batch_size > 1`.
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
            segment_ids=segment_ids,
        )

        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[jnp.arange(batch_size), sequence_lengths]

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
