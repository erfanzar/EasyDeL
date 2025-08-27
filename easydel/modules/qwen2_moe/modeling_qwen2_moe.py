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
from easydel.infra.utils import auto_remat, get_dot_general_by_bits
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
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm as RMSNorm
from easydel.modules.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig as Qwen2MoeConfig


class Qwen2MoeMLPStack(nn.Module):
    """Qwen2Moe MoE MLP using the new ParallelMoELinear layers."""

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            use_pallas_group_matmul=config.use_pallas_group_matmul,
            partition_manager=config.partition_manager,
        )
        self.act_fn = nn.silu

    def __call__(self, x: chex.Array, group_sizes: chex.Array) -> chex.Array:
        """Forward pass through MoE MLP."""
        return self.down_proj(self.act_fn(self.gate_proj(x, group_sizes)) * self.up_proj(x, group_sizes), group_sizes)


class Qwen2MoeMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) block for the Qwen2 MoE model.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        gate_proj (ParallelLinear): Linear layer for the gating mechanism.
        down_proj (ParallelLinear): Linear layer for down-projection.
        up_proj (ParallelLinear): Linear layer for up-projection.
        act_fn (callable): Activation function (SiLU).
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeMLP module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
            intermediate_size (int): The size of the intermediate layer.
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
            rngs=rngs,
            **get_dot_general_by_bits(config.bits, config.easy_method),
        )
        self.gate_proj = linear_class(config.hidden_size, intermediate_size, rngs=rngs)
        self.down_proj = linear_class(intermediate_size, config.hidden_size, rngs=rngs)
        self.up_proj = linear_class(config.hidden_size, intermediate_size, rngs=rngs)
        self.act_fn = nn.silu

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


class Qwen2MoeAttention(AttentionModule):
    """Qwen2 MoE Attention module.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        hidden_size (int): Dimensionality of the hidden states.
        head_dim (int): Dimensionality of each attention head.
        num_key_value_groups (int): Number of groups for key/value heads (for GQA).
        q_proj (ParallelLinear): Linear layer for query projection.
        k_proj (ParallelLinear): Linear layer for key projection.
        v_proj (ParallelLinear): Linear layer for value projection.
        o_proj (ParallelLinear): Linear layer for output projection.
        attention_performer (FlexibleAttentionModule): Module for performing attention computation.
        resid_dropout (nn.Dropout): Dropout layer for residual connections.
        rotary (RotaryEmbedding): Rotary positional embedding module.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeAttention module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
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
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
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
            use_bias=config.qkv_bias,
        )
        self.k_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.qkv_bias,
        )
        self.v_proj = linear_class(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.qkv_bias,
        )
        self.o_proj = linear_class(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=False,
        )
        self.sliding_window = config.sliding_window if config.use_sliding_window else None
        self.attention_performer = FlexibleAttentionModule(
            rngs=rngs,
            base_config=self.config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
        )
        self.resid_dropout = nn.Dropout(rate=config.attention_dropout, rngs=rngs)
        self.rotary = self.config.get_basic_rope(
            head_size=config.hidden_size // config.num_attention_heads,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            dtype=self.dtype,
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
            hidden_states (chex.Array): Input hidden states.
            attention_mask (chex.Array): Mask to apply on the attention scores.
            position_ids (chex.Array): Position indices for the tokens.
            causal_mask (chex.Array): Causal mask for ensuring autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            deterministic (bool): If True, disables dropout for deterministic behavior.
            init_cache (bool): If True, initializes cache for caching keys and values.
            output_attentions (bool): If True, outputs attention weights alongside the hidden states.
            fcm_mask (tp.Optional[chex.Array]): fcm mask to be combined with attn mask and causal mask.
        Returns:
            tp.Tuple[chex.Array, chex.Array]: A tuple containing the attention output and the attention weights.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        query_states = query_states.reshape(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
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

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = self.o_proj(attn_output)

        attn_output = self.resid_dropout(attn_output)
        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen2MoeSparseBlock(BaseMoeModule):
    """Sparse Mixture of Experts (MoE) block for Qwen2 MoE.

    This block routes input hidden states to a selected subset of experts
    and combines their outputs.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        gate (ParallelLinear): Linear layer for the gating network.
        experts (nn.List[Qwen2MoeMLP]): List of expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeSparseBlock module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.NONE,
        )
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

        self.experts = Qwen2MoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert = Qwen2MoeMLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.shared_expert_gate = ParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _moe_call(
        self,
        hidden_state_flat: jax.Array,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass of the MoE block."""

        router_logits = self.gate(hidden_state_flat).astype(jnp.float32)
        routing_weights = jax.nn.softmax(router_logits, axis=-1)
        selected_weights, selected_experts = self._route(routing_weights)
        x_repeat_sort, group_sizes, sort_idx = self._permute(hidden_state_flat, selected_experts.reshape(-1))
        out_repeat_sort = self.experts(x_repeat_sort, group_sizes)
        out_repeat_unflat = self._unpermute(out_repeat_sort, sort_idx, (batch_size, seq_len, hidden_size))
        output = jnp.sum(out_repeat_unflat * selected_weights[:, :, None], axis=1)

        return output, router_logits

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
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        out, router_logits = self._moe_call(
            hidden_state_flat=hidden_states,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
        )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = jax.nn.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        out = out + shared_expert_output

        return out.reshape(batch_size, seq_len, hidden_size), router_logits


class Qwen2MoeDecoderLayer(nn.Module):
    """A single decoder layer for the Qwen2 MoE model.

    This layer combines self-attention, a sparse MoE block (or a standard MLP),
    and residual connections with layer normalization.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        layer_idx (int): Index of the current layer.
        self_attn (Qwen2MoeAttention): Self-attention module.
        mlp (Qwen2MoeSparseBlock | Qwen2MoeMLP): MoE block or standard MLP.
        input_layernorm (RMSNorm): Layer normalization applied before self-attention.
        post_attention_layernorm (RMSNorm): Layer normalization applied after self-attention and
            before the MLP/MoE block.
        dropout_rng_key (str): Name of the RNG key for dropout.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeDecoderLayer module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
            layer_idx (int): The index of the current layer.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        attn_block = Qwen2MoeAttention

        mlp_block = (
            Qwen2MoeSparseBlock
            if (self.layer_idx not in self.config.mlp_only_layers)
            and (self.config.num_experts > 0 and (self.layer_idx + 1) % self.config.decoder_sparse_step == 0)
            else Qwen2MoeMLP
        )
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
        self.input_layernorm = RMSNorm(
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            dim=self.config.hidden_size,
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
        cache_view: TransformerCacheView | PagesCacheView | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        segment_ids: chex.Array | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        fcm_mask: chex.Array | None = None,
        frequencies: chex.Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass of the decoder layer.

        Args:
            hidden_states (chex.Array): Input hidden states (batch, seq_len, hidden_size).
            attention_mask (chex.Array): Attention mask (batch, 1, seq_len, kv_seq_len).
            position_ids (chex.Array): Position IDs (batch, seq_len).
            causal_mask (tp.Optional[chex.Array | bool]): Causal mask for autoregressive behavior.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            cache_view (tp.Optional[TransformerCacheView | PagesCacheView]): Cache view for
                key/value states (optional).
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for
                paged attention (optional).
            output_attentions (bool): Whether to output attention weights (default: False).
            output_router_logits (bool): Whether to output router logits (default: False).
            fcm_mask (tp.Optional[chex.Array]): Forward causal mask (FCM) mask (optional).
            frequencies (tp.Optional[chex.Array]): Precomputed rotary frequencies (optional).

        Returns:
            DecoderLayerOutput: A tuple containing:
                - hidden_states (chex.Array): Output hidden states after the decoder layer.
                - attention_outputs (chex.Array): Attention weights (if `output_attentions` is True).
                - router_logits (tp.Optional[chex.Array]): Router logits (if `output_router_logits` is
                    True and it's an MoE layer).
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

        mlp_out = self.mlp(feed_forward_input)

        if self.config.num_experts > 0:
            feed_forward_hidden_states, router_logits = mlp_out
        else:
            feed_forward_hidden_states = mlp_out
            router_logits = None

        hidden_states = hidden_states + feed_forward_hidden_states

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen2MoeConfig, model_type="qwen2_moe")
class Qwen2MoeModel(EasyDeLBaseModule):
    """The base Qwen2 MoE transformer model.

    This class implements the core transformer architecture, including embedding layers,
    decoder layers, and final normalization.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        embed_tokens (nn.Embed): Embedding layer for input tokens.
        layers (nn.List[Qwen2MoeDecoderLayer]): List of decoder layers.
        norm (RMSNorm): Final layer normalization.
        gradient_checkpointing (str): Gradient checkpointing strategy.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeModel module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
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
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            Qwen2MoeDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
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
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeModelOutput:
        """Forward pass of the Qwen2 MoE model.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs (batch, seq_len). Mutually exclusive with
                `inputs_embeds`.
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings (batch, seq_len, hidden_size). Mutually
                exclusive with `input_ids`.
            attention_mask (tp.Optional[chex.Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[chex.Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all layers
                (default defined by config).
            output_router_logits (tp.Optional[bool]): Whether to output router logits (default defined by config).
            past_key_values (tp.Optional[TransformerCache | PagesCache]): Precomputed key/value states
                for caching.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for
                paged attention (optional).

        Returns:
            MoeModelOutput: The model output.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided or neither is provided.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        all_hidden_states = ()
        all_router_logits = ()
        all_self_attns = ()
        batch_size, sequence_length, _ = inputs_embeds.shape
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

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            inputs_embeds,
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
                causal_mask=self.causal_mask,
                segment_ids=segment_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_self_attns += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
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


@register_module(TaskType.CAUSAL_LM, config=Qwen2MoeConfig, model_type="qwen2_moe")
class Qwen2MoeForCausalLM(EasyDeLBaseModule):
    """Qwen2 MoE model with a Causal Language Modeling (CLM) head.

    This class wraps the base `Qwen2MoeModel` and adds a linear layer (language model head)
    to predict the next token logits.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        model (Qwen2MoeModel): The base Qwen2 MoE model.
        lm_head (ParallelLinear): The language model head (linear layer).
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeForCausalLM module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
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
        self.model = Qwen2MoeModel(
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
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            precision=precision,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass of the Qwen2 MoE model for Causal Language Modeling.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs (batch, seq_len). Mutually exclusive
                with `inputs_embeds`.
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings (batch, seq_len, hidden_size). Mutually
                exclusive with `input_ids`.
            attention_mask (tp.Optional[chex.Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[chex.Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all layers
                (default defined by config).
            output_router_logits (tp.Optional[bool]): Whether to output router logits (default defined by config).
            past_key_values (tp.Optional[TransformerCache | PagesCache]): Precomputed key/value states
                for caching.
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged
                attention (optional).

        Returns:
            MoeCausalLMOutput: The model output, including logits, hidden states, attentions, and router logits.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if output_attentions is None:
            output_attentions = self.config.output_attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            segment_ids=segment_ids,
        )
        hidden_states = outputs.last_hidden_state

        logits = None
        if apply_lm_head:
            logits = self.apply_lm_head(hidden_states)

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
            aux_loss=aux_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_hidden_state=outputs.last_hidden_state,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
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


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Qwen2MoeConfig, model_type="qwen2_moe")
class Qwen2MoeForSequenceClassification(EasyDeLBaseModule):
    """Qwen2 MoE model with a sequence classification head.

    This class wraps the base `Qwen2MoeModel` and adds a linear layer on top
    to perform sequence classification tasks.

    Attributes:
        config (Qwen2MoeConfig): Configuration object for the model.
        model (Qwen2MoeModel): The base Qwen2 MoE model.
        score (ParallelLinear): The sequence classification head (linear layer).
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
        rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the Qwen2MoeForSequenceClassification module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
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
        self.model = Qwen2MoeModel(
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
        past_key_values: TransformerCache | PagesCache | None = None,
        cache_metadata: TransformerMetadata | PagesMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass of the Qwen2 MoE model for sequence classification.

        Args:
            input_ids (tp.Optional[chex.Array]): Input token IDs (batch, seq_len). Mutually
                exclusive with `inputs_embeds`.
            inputs_embeds (tp.Optional[chex.Array]): Input embeddings (batch, seq_len, hidden_size).
                Mutually exclusive with `input_ids`.
            attention_mask (tp.Optional[chex.Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[chex.Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[chex.Array]): Segment IDs for segment-based attention (optional).
            past_key_values (tp.Optional[TransformerCache | PagesCache]): Precomputed key/value states for
                caching (ignored in classification).
            cache_metadata (tp.Optional[TransformerMetadata | PagesMetadata]): Metadata for paged attention
                (ignored in classification).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all layers
                (default defined by config).

        Returns:
           SequenceClassifierOutput: The model output, including classification logits, hidden states, and attentions.
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
        This model has a sequence classification head, not an LM Head.
        """
        raise NotImplementedError("This model has a sequence classification head, not a language model head.")

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()
