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


from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
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
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.norms import RMSNorm as RMSNorm

from .qwen3_configuration import Qwen3Config


class Qwen3MLP(nn.Module):
    """Qwen3 MLP module.

    This module implements the feed-forward network (MLP) used in the Qwen3 model.
    It uses a Gated Linear Unit (GLU) structure with SiLU activation.

    Attributes:
        config (Qwen3Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        gate_proj (ParallelLinear): Linear layer for the GLU gate.
        down_proj (ParallelLinear): Linear layer for the down projection.
        up_proj (ParallelLinear): Linear layer for the GLU value.
        act_fn (callable): Activation function (SiLU).
    """

    config: Qwen3Config
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike | None
    gate_proj: ColumnParallelLinear
    down_proj: RowParallelLinear
    up_proj: ColumnParallelLinear
    act_fn: callable

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3MLP module.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.down_proj = row_parallel_linear(
            config.intermediate_size,
            config.hidden_size,
            rngs=rngs,
        )
        self.up_proj = column_parallel_linear(
            config.hidden_size,
            config.intermediate_size,
            rngs=rngs,
        )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the Qwen3MLP module.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states.

        Returns:
            Float[Array, "batch seq_len hidden_dim"]: Output hidden states after MLP transformation.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate: Float[Array, "batch seq_len intermediate_size"] = checkpoint_name(
            self.act_fn(self.gate_proj(hidden_states)), "mlp_gate"
        )
        up: Float[Array, "batch seq_len intermediate_size"] = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states: Float[Array, "batch seq_len hidden_dim"] = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3Attention(AttentionModule):
    """Qwen3 Attention module.

    This module implements the multi-head attention mechanism used in the Qwen3 model.
    It supports Grouped Query Attention (GQA) and Rotary Position Embeddings (RoPE).

    Attributes:
        config (Qwen3Config): Configuration object for the model.
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

    layer_idx: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike | None
    rngs: nn.Rngs
    hidden_size: int
    head_dim: int
    num_key_value_groups: int
    q_proj: ColumnParallelLinear
    k_proj: ColumnParallelLinear
    v_proj: ColumnParallelLinear
    o_proj: RowParallelLinear
    attention_performer: FlexibleAttentionModule
    q_norm: RMSNorm
    k_norm: RMSNorm
    sliding_window: int | None

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3Attention module.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
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

        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )

        self.q_proj = column_parallel_linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.k_proj = column_parallel_linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.v_proj = column_parallel_linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.attention_bias,
        )
        self.o_proj = row_parallel_linear(
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
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"],
        position_ids: Int[Array, "batch seq_len"],
        causal_mask: Bool[Array, "batch 1 seq_len seq_len"] | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
        fcm_mask: Bool[Array, "batch seq_len seq_len"] | None = None,
        frequencies: Float[Array, "seq_len head_dim//2 2"] | None = None,
    ) -> AttentionLayerOutput:
        """
        Forward pass of the Qwen3Attention module.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states.
            attention_mask (Bool[Array, "batch seq_len"]): Mask to apply on the attention scores.
            position_ids (Int[Array, "batch seq_len"]): Position indices for the tokens.
            causal_mask (Union[Bool[Array, "batch 1 seq_len seq_len"], bool, None]): Causal mask for ensuring autoregressive behavior.
            cache_view (Optional[Union[TransformerCacheView, PagesCacheView]]): Cache view for attention KVs.
            cache_metadata (Optional[Union[TransformerMetadata, PagesMetadata]]): Metadata for paged attention.
            segment_ids (Optional[Int[Array, "batch seq_len"]]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (Optional[Bool[Array, "batch seq_len seq_len"]]): Flash Chunking Mask (FCM) for attention.
            frequencies (Optional[Float[Array, "seq_len head_dim//2 2"]]): Precomputed rotary frequency embeddings.

        Returns:
            AttentionLayerOutput: A tuple containing the attention output hidden states, optionally attention weights, and cache view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        query_states: Float[Array, "batch seq_len num_heads*head_dim"]
        key_states: Float[Array, "batch seq_len num_kv_heads*head_dim"]
        value_states: Float[Array, "batch seq_len num_kv_heads*head_dim"]

        query_states, key_states, value_states = (
            checkpoint_name(self.q_proj(hidden_states), "attn_query"),
            checkpoint_name(self.k_proj(hidden_states), "attn_key"),
            checkpoint_name(self.v_proj(hidden_states), "attn_value"),
        )

        query_states: Float[Array, "batch seq_len num_heads head_dim"] = self.q_norm(
            query_states.reshape(
                batch_size,
                sequence_length,
                self.config.num_attention_heads,
                self.head_dim,
            )
        )
        key_states: Float[Array, "batch seq_len num_kv_heads head_dim"] = self.k_norm(
            key_states.reshape(
                batch_size,
                sequence_length,
                self.config.num_key_value_heads,
                self.head_dim,
            )
        )
        value_states: Float[Array, "batch seq_len num_kv_heads head_dim"] = value_states.reshape(
            batch_size,
            sequence_length,
            self.config.num_key_value_heads,
            self.head_dim,
        )

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
            sliding_window=self.sliding_window,
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
            sliding_window=self.sliding_window,
        )
        attn_output: Float[Array, "batch seq_len hidden_dim"] = self._merge_heads(attentions.attention_outputs)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output: Float[Array, "batch seq_len hidden_dim"] = checkpoint_name(self.o_proj(attn_output), "attn_output")
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Transformer Decoder Layer.

    This module represents a single decoder layer in the Qwen3 model,
    combining self-attention and MLP sub-layers with residual connections
    and RMS normalization.

    Attributes:
        config (Qwen3Config): Configuration object for the model.
                    layer_idx (int): The index of the layer in the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        input_layernorm (RMSNorm): RMS normalization applied before the attention layer.
        self_attn (Qwen3Attention): The self-attention module.
        mlp (Qwen3MLP): The feed-forward (MLP) module.
        post_attention_layernorm (RMSNorm): RMS normalization applied after the attention layer and before the MLP layer.
    """

    config: Qwen3Config
    dtype: jnp.dtype
    param_dtype: jnp.dtype
    precision: jax.lax.PrecisionLike | None
    self_attn: Qwen3Attention
    mlp: Qwen3MLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3DecoderLayer.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
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
        attn_block = Qwen3Attention
        mlp_block = Qwen3MLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"],
        position_ids: Int[Array, "batch seq_len"],
        causal_mask: Bool[Array, "batch 1 seq_len seq_len"] | bool | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool = False,
        fcm_mask: Bool[Array, "batch seq_len seq_len"] | None = None,
        frequencies: Float[Array, "seq_len head_dim//2 2"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass of the Qwen3DecoderLayer module.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states.
            attention_mask (Bool[Array, "batch seq_len"]): Mask to apply on the attention scores.
            position_ids (Int[Array, "batch seq_len"]): Position indices for the tokens.
            causal_mask (Union[Bool[Array, "batch 1 seq_len seq_len"], bool, None]): Causal mask for ensuring autoregressive behavior.
            cache_view (Optional[Union[TransformerCacheView, PagesCacheView]]): Cache view for attention KVs.
            cache_metadata (Optional[Union[TransformerMetadata, PagesMetadata]]): Metadata for paged attention.
            segment_ids (Optional[Int[Array, "batch seq_len"]]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to return attention weights. Default is False.
            fcm_mask (Optional[Bool[Array, "batch seq_len seq_len"]]): Flash Chunking Mask (FCM) for attention.
            frequencies (Optional[Float[Array, "seq_len head_dim//2 2"]]): Precomputed rotary frequency embeddings.

        Returns:
            DecoderLayerOutput: A tuple containing the output hidden states, optionally the attention weights, and cache view.
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"] = checkpoint_name(
            hidden_states + attn_outputs.attention_output, "residual"
        )

        feed_forward_input: Float[Array, "batch seq_len hidden_dim"] = self.post_attention_layernorm(hidden_states)

        if self.config.use_scan_mlp:
            feed_forward_hidden_states: Float[Array, "batch seq_len hidden_dim"] = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states: Float[Array, "batch seq_len hidden_dim"] = self.mlp(feed_forward_input)
        hidden_states: Float[Array, "batch seq_len hidden_dim"] = checkpoint_name(
            hidden_states + feed_forward_hidden_states, "residual"
        )
        hidden_states = checkpoint_name(hidden_states, "layer_output")
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


@register_module(TaskType.BASE_MODULE, config=Qwen3Config, model_type="qwen3")
class Qwen3Model(EasyDeLBaseModule):
    """The base Qwen3 model transformer.

    This class represents the core transformer architecture of the Qwen3 model,
    consisting of an embedding layer, multiple Qwen3DecoderLayer layers,
    and a final RMS normalization layer.

    Attributes:
        config (Qwen3Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (tp.List[Qwen3DecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (EasyDeLGradientCheckPointers): Gradient checkpointing configuration.
    """

    embed_tokens: nn.Embed
    layers: list[Qwen3DecoderLayer]
    norm: RMSNorm

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3Model.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
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

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Qwen3DecoderLayer(
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> BaseModelOutput:
        """Forward pass of the Qwen3Model.

        Args:
            input_ids (Optional[Int[Array, "batch seq_len"]]): Input token IDs.
            inputs_embeds (Optional[Float[Array, "batch seq_len hidden_dim"]]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (Optional[Bool[Array, "batch seq_len"]]): Mask to avoid performing attention on padding token indices.
            position_ids (Optional[Int[Array, "batch seq_len"]]): Position indices for the tokens.
            segment_ids (Optional[Int[Array, "batch seq_len"]]): Segment IDs (unused).
            output_attentions (Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (Optional[Union[TransformerCache, PagesCache]]):
                Precomputed key/value states for attention.
            cache_metadata (Optional[Union[TransformerMetadata, PagesMetadata]]): Metadata for paged attention.

        Returns:
            BaseModelOutput: The model's output.
                returns a `BaseModelOutput` object containing `last_hidden_state`, `hidden_states` (optional),
                and `attentions` (optional).

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds: Float[Array, "batch seq_len hidden_dim"] = checkpoint_name(
                self.embed_tokens(input_ids.astype("i4")), "embeddings"
            )
        batch_size, sequence_length, _ = inputs_embeds.shape

        all_attentions: tuple[Float[Array, ...], ...] | None = () if output_attentions else None
        all_hidden_states: tuple[Float[Array, "batch seq_len hidden_dim"], ...] | None = (
            () if output_hidden_states else None
        )
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        if attention_mask is None:
            attention_mask: Bool[Array, "batch seq_len"] = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask: Bool[Array, "batch seq_len"] = jnp.astype(attention_mask == 1, "b1")
        if position_ids is None:
            position_ids: Int[Array, "batch seq_len"] = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        hidden_states: Float[Array, "batch seq_len hidden_dim"] = inputs_embeds
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

        hidden_states: Float[Array, "batch seq_len hidden_dim"] = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self) -> None:
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> "Qwen3Model":
        """
        Returns the decoder part of the model's graph definition.
        """
        return self

    def get_lm_head(self) -> None:
        """
        Returns the language model head of the module.
        Base Models don't have a Language Model Head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self) -> nn.Embed:
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Qwen3Config, model_type="qwen3")
class Qwen3ForCausalLM(EasyDeLBaseModule):
    """Qwen3 model with a Causal Language Modeling head.

    This model consists of the base Qwen3 transformer (`Qwen3Model`) followed by a
    linear layer (`lm_head`) that projects the transformer's output hidden states
    to the vocabulary size, producing logits for next token prediction.
    Optionally, the input token embeddings can be tied to the output projection layer.

    Attributes:
        config (Qwen3Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        model (Qwen3Model): The core Qwen3 transformer model.
        lm_head (ParallelLinear): The linear layer for projecting hidden states to vocabulary logits.
    """

    model: Qwen3Model
    lm_head: ColumnParallelLinear

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3ForCausalLM model.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
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
        self.model = Qwen3Model(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        lm_head_block = ColumnParallelLinear
        lm_head_block = auto_remat(
            lm_head_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.lm_head = lm_head_block(
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:
        """Forward pass of the Qwen3ForCausalLM model.

        Args:
            input_ids (Optional[Int[Array, "batch seq_len"]]): Input token IDs.
            inputs_embeds (Optional[Float[Array, "batch seq_len hidden_dim"]]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (Optional[Bool[Array, "batch seq_len"]]): Mask to avoid performing attention on padding token indices.
            position_ids (Optional[Int[Array, "batch seq_len"]]): Position indices for the tokens.
            segment_ids (Optional[Int[Array, "batch seq_len"]]): Segment IDs (unused).
            output_attentions (Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (Optional[bool]): Whether to return hidden states for all layers.
                Defaults to `config.output_hidden_states`.
            past_key_values (Optional[Union[TransformerCache, PagesCache]]):
                Precomputed key/value states for attention.
            cache_metadata (Optional[Union[TransformerMetadata, PagesMetadata]]): Metadata for paged attention.


        Returns:
            CausalLMOutput: The model's output.
                returns a `CausalLMOutput` object containing `logits`, `hidden_states` (optional),
                and `attentions` (optional).
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

        hidden_states: Float[Array, "batch seq_len hidden_dim"] = outputs.last_hidden_state

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits: Float[Array, "batch seq_len vocab_size"] | None = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

        return CausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self) -> None:
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> Qwen3Model:
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self) -> ColumnParallelLinear:
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self) -> nn.Embed:
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Qwen3Config, model_type="qwen3")
class Qwen3ForSequenceClassification(EasyDeLBaseModule):
    """Qwen3 model with a Sequence Classification head.

    This model consists of the base Qwen3 transformer (`Qwen3Model`) followed by a
    linear layer (`score`) that projects the transformer's output hidden states
    (typically the hidden state of the last token or a pooled representation) to the number of classes
        for classification.

    Attributes:
        config (Qwen3Config): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        model (Qwen3Model): The core Qwen3 transformer model.
        score (ParallelLinear): The linear layer for classification.
    """

    model: Qwen3Model
    score: ColumnParallelLinear

    def __init__(
        self,
        config: Qwen3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen3ForSequenceClassification model.

        Args:
            config (Qwen3Config): The configuration object for the Qwen3 model.
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
        self.model = Qwen3Model(
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
        self.score = ColumnParallelLinear(
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
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        segment_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the Qwen3ForSequenceClassification model.

        Args:
            input_ids (Optional[Int[Array, "batch seq_len"]]): Input token IDs.
            inputs_embeds (Optional[Float[Array, "batch seq_len hidden_dim"]]): Input embeddings.
                Either `input_ids` or `inputs_embeds` must be provided.
            attention_mask (Optional[Bool[Array, "batch seq_len"]]): Mask to avoid performing attention on padding token indices.
            position_ids (Optional[Int[Array, "batch seq_len"]]): Position indices for the tokens.
            segment_ids (Optional[Int[Array, "batch seq_len"]]): Segment IDs (unused).
            past_key_values (Optional[Union[TransformerCache, PagesCache]]):
                Precomputed key/value states for attention.
            cache_metadata (Optional[Union[TransformerMetadata, PagesMetadata]]): Metadata for paged attention.
            output_attentions (Optional[bool]): Whether to return attention weights.
                Defaults to `config.output_attentions`.
            output_hidden_states (Optional[bool]): Whether to return hidden states for all layers.
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

        hidden_states: Float[Array, "batch seq_len hidden_dim"] = transformer_outputs.last_hidden_state
        logits: Float[Array, "batch seq_len num_labels"] = self.score(hidden_states)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        sequence_lengths: int | Int[Array, ...]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.config.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits: Float[Array, "batch num_labels"] = logits[jnp.arange(batch_size), sequence_lengths]

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_encoder(self) -> None:
        """
        Returns the encoder part of the model's graph definition.
        Decoder-Only models don't have an encoder.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> Qwen3Model:
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.model.get_decoder()

    def get_lm_head(self) -> None:
        """
        Returns the language model head of the module.
        This model has a sequence classification head, not an LM Head.
        """
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self) -> nn.Embed:
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()
