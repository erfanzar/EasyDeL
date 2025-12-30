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

from __future__ import annotations

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import AttentionLayerOutput, DecoderLayerOutput, MoeCausalLMOutput, MoeModelOutput
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesMetadata,
    RecurrentCacheView,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm

from .minimax_configuration import MiniMaxConfig


class MiniMaxLightningAttention(nn.Module):
    """Lightning Attention module for MiniMax models.

    This module implements a linear attention mechanism with exponential decay,
    designed for efficient long-sequence processing. It uses a block-wise computation
    strategy with intra-block and inter-block attention components.

    Attributes:
        config: Model configuration object.
        layer_idx: Index of this layer in the transformer stack.
        hidden_size: Dimension of hidden states.
        num_attention_heads: Number of attention heads.
        num_hidden_layers: Total number of layers in the model.
        head_dim: Dimension of each attention head.
        block_size: Size of blocks for block-wise attention computation.
        act_fn: Activation function used in QKV projections.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize the MiniMaxLightningAttention module.

        Args:
            config: Model configuration containing attention parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.block_size = config.block_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim * 3,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.out_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.output_gate = ColumnParallelLinear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )
        self.norm = RMSNorm(
            self.num_attention_heads * self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _get_slope_rate(self) -> Array:
        """Compute the slope rate for exponential decay in attention.

        The slope rate determines the decay factor for each attention head,
        with earlier layers having faster decay rates.

        Returns:
            Array of shape (num_heads, 1, 1) containing decay rates per head.
        """
        base = 1.0 / (2.0 ** (8.0 / self.num_attention_heads))
        exponent = jnp.arange(self.num_attention_heads, dtype=jnp.float32) + 1.0
        factor = 1.0 - float(self.layer_idx) / (float(self.num_hidden_layers) - 1.0 + 1e-5) + 1e-5
        rate = (base**exponent) * factor
        return rate[:, None, None].astype(jnp.float32)

    def _decay_factors(self, slope_rate: Array) -> tuple[Array, Array, Array]:
        """Compute decay factors for block-wise attention computation.

        Args:
            slope_rate: Decay rate per attention head, shape (num_heads, 1, 1).

        Returns:
            A tuple of three arrays:
                - query_decay: Decay factors for queries, shape (num_heads, block_size, 1).
                - key_decay: Decay factors for keys, shape (num_heads, block_size, 1).
                - diagonal_decay: Causal mask with decay, shape (1, 1, block_size, block_size).
        """
        block_size_range = jnp.arange(self.block_size, dtype=jnp.float32) + 1.0
        block_size_range_2d = block_size_range[:, None]

        query_decay = jnp.exp(-slope_rate * block_size_range_2d)
        key_decay = jnp.exp(-slope_rate * (float(self.block_size) - block_size_range_2d))

        diagonal_decay = block_size_range_2d - block_size_range[None, :]
        diagonal_decay = diagonal_decay[None, None, :, :]
        diagonal_decay = slope_rate * diagonal_decay
        diagonal_decay = jnp.where(diagonal_decay >= 0, -diagonal_decay, -jnp.inf)
        diagonal_decay = jnp.exp(diagonal_decay)

        return query_decay, key_decay, diagonal_decay

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: RecurrentCacheView | None = None,
    ) -> tuple[Float[Array, "batch seq_len hidden_dim"], RecurrentCacheView | None]:
        """Perform lightning attention forward pass.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            attention_mask: Optional boolean mask of shape (batch, seq_len).
            mode: Runtime mode (train, decode, etc.).
            cache_view: Optional recurrent cache for incremental decoding.

        Returns:
            A tuple containing:
                - Output hidden states of shape (batch, seq_len, hidden_dim).
                - Updated cache view (or None if caching is disabled).
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        slope_rate = self._get_slope_rate()
        query_decay, key_decay, diagonal_decay = self._decay_factors(slope_rate)

        qkv_states = self.act_fn(checkpoint_name(self.qkv_proj(hidden_states), "attn_qkv"))
        qkv_states = qkv_states.reshape(batch_size, seq_len, self.num_attention_heads, 3 * self.head_dim)
        query_states, key_states, value_states = jnp.split(qkv_states, 3, axis=-1)

        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))

        attn_weights_inter: Array
        if cache_view is not None and cache_view.recurrent_state is not None:
            attn_weights_inter = cache_view.recurrent_state
        else:
            attn_weights_inter = jnp.zeros(
                (batch_size, self.num_attention_heads, self.head_dim, self.head_dim), dtype=value_states.dtype
            )

        if mode != common_types.MODE_DECODE:
            if attention_mask is not None:
                attention_mask = attention_mask.astype(jnp.bool_)
                value_states = jnp.where(attention_mask[:, None, :, None], value_states, jnp.zeros_like(value_states))

            attn_outputs = []
            for i in range(num_blocks):
                start_idx = i * self.block_size
                end_idx = min(start_idx + self.block_size, seq_len)
                current_block_size = end_idx - start_idx

                current_query_states = query_states[:, :, start_idx:end_idx]
                current_key_states = key_states[:, :, start_idx:end_idx]
                current_value_states = value_states[:, :, start_idx:end_idx]

                current_query_decay = query_decay[:, :current_block_size]
                current_key_decay = key_decay[:, -current_block_size:]
                current_diagonal_decay = diagonal_decay[:, :, :current_block_size, :current_block_size]
                block_decay = jnp.exp(-slope_rate * float(current_block_size))

                attn_weights_intra = jnp.matmul(current_query_states, jnp.swapaxes(current_key_states, -1, -2))
                attn_output_intra = jnp.matmul(attn_weights_intra * current_diagonal_decay, current_value_states)

                attn_output_inter = jnp.matmul(current_query_states * current_query_decay, attn_weights_inter)
                current_attn_output = attn_output_inter + attn_output_intra
                attn_outputs.append(current_attn_output)

                next_attn_weights_inter = jnp.matmul(
                    jnp.swapaxes(current_key_states * current_key_decay, -1, -2),
                    current_value_states,
                )
                attn_weights_inter = attn_weights_inter * block_decay + next_attn_weights_inter

            attn_output = jnp.concatenate(attn_outputs, axis=-2)
        else:
            ratio = jnp.exp(-slope_rate)
            attn_outputs = []
            for i in range(seq_len):
                current_query_states = query_states[:, :, i : i + 1]
                current_key_states = key_states[:, :, i : i + 1]
                current_value_states = value_states[:, :, i : i + 1]

                current_attn_weights_inter = jnp.matmul(
                    jnp.swapaxes(current_key_states, -1, -2),
                    current_value_states,
                )
                attn_weights_inter = ratio * attn_weights_inter + current_attn_weights_inter
                current_attn_output = jnp.matmul(current_query_states, attn_weights_inter)
                attn_outputs.append(current_attn_output)
            attn_output = jnp.concatenate(attn_outputs, axis=-2)

        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_attention_heads * self.head_dim)
        attn_output = self.norm(attn_output)
        attn_output = jax.nn.sigmoid(checkpoint_name(self.output_gate(hidden_states), "attn_gate")) * attn_output
        attn_output = checkpoint_name(self.out_proj(attn_output), "attn_output")

        if cache_view is not None:
            cache_view = cache_view.update_recurrent_state(attn_weights_inter)

        return attn_output, cache_view


class MiniMaxAttention(UnifiedAttention[MiniMaxConfig]):
    """Standard multi-head attention module for MiniMax models.

    This module extends UnifiedAttention to provide standard causal self-attention
    with optional sliding window support. Used in layers that require full attention
    as opposed to lightning (linear) attention.

    Attributes:
        Inherits all attributes from UnifiedAttention.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize the MiniMaxAttention module.

        Args:
            config: Model configuration containing attention parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
            layer_idx: Index of this layer in the transformer stack.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=getattr(config, "sliding_window", None),
        )


class MiniMaxExperts(nn.Module):
    """Expert feed-forward networks for MiniMax Mixture-of-Experts layers.

    This module implements the expert networks used in the sparse MoE architecture.
    Each expert consists of a gated feed-forward network with up-projection (w1),
    gate projection (w3), and down-projection (w2) layers.

    Attributes:
        config: Model configuration object.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting for matrix operations.
        w1: Up-projection linear layer.
        w2: Down-projection linear layer.
        w3: Gate projection linear layer.
        act_fn: Activation function for the gating mechanism.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the MiniMaxExperts module.

        Args:
            config: Model configuration containing MoE parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        init = nn.initializers.normal(config.initializer_range)
        self.w1 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=init,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.w2 = RowParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=init,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.w3 = ColumnParallelMoELinear(
            num_experts=config.num_local_experts,
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=init,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "tokens hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Forward pass through the expert networks.

        Computes gated feed-forward transformation: w2(act(w1(x)) * w3(x)).

        Args:
            hidden_states: Input tensor of shape (tokens, hidden_dim).
            group_sizes: Number of tokens assigned to each expert.
            sorted_experts: Optional sorted expert indices for routing.

        Returns:
            Output tensor of shape (tokens, hidden_dim).
        """
        return self.w2(
            self.act_fn(self.w1(hidden_states, group_sizes, sorted_experts))
            * self.w3(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class MiniMaxSparseMoeBlock(BaseMoeModule):
    """Sparse Mixture-of-Experts block for MiniMax models.

    This module implements top-k expert routing with load balancing. Each token
    is routed to a subset of experts based on router logits, and the outputs
    are combined using weighted averaging.

    Attributes:
        config: Model configuration object.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting for matrix operations.
        gate: Router network that produces expert selection logits.
        experts: The MiniMaxExperts module containing all expert networks.
        jitter_noise: Noise level for router jitter during training.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the MiniMaxSparseMoeBlock module.

        Args:
            config: Model configuration containing MoE parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )

        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )
        self.experts = MiniMaxExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        def _normalize_weights(weights: Array) -> Array:
            return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)

        self.moe_hooks = self.moe_hooks.replace(refine_weights_hook=_normalize_weights)

        self.jitter_noise = config.router_jitter_noise

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        training: bool = False,
        layer_idx: int | None = None,
    ) -> tuple[Array, Array]:
        """Forward pass through the sparse MoE block.

        Routes each token to top-k experts and combines their outputs.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            training: Whether the model is in training mode (enables jitter noise).
            layer_idx: Optional layer index for auxiliary loss computation.

        Returns:
            A tuple containing:
                - Output hidden states of shape (batch, seq_len, hidden_dim).
                - Router logits for auxiliary loss computation.
        """
        if training and self.jitter_noise > 0:
            hidden_states = hidden_states * jax.random.uniform(
                self.rngs.param(),
                shape=hidden_states.shape,
                minval=1.0 - self.jitter_noise,
                maxval=1.0 + self.jitter_noise,
            )

        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.w1.kernel.value,
            wu_kernel=self.experts.w3.kernel.value,
            wd_kernel=self.experts.w2.kernel.value,
            act_fn=self.experts.act_fn,
            layer_idx=layer_idx,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class MiniMaxDecoderLayer(nn.Module):
    """Single decoder layer for MiniMax transformer models.

    Each layer consists of an attention block (either lightning or standard attention)
    followed by a sparse MoE feed-forward block, with residual connections and
    layer normalization.

    Attributes:
        config: Model configuration object.
        layer_idx: Index of this layer in the transformer stack.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting for matrix operations.
        layer_type: Type of attention ("linear_attention" or "full_attention").
        self_attn: Attention module (MiniMaxLightningAttention or MiniMaxAttention).
        block_sparse_moe: Sparse MoE feed-forward block.
        input_layernorm: Pre-attention layer normalization.
        post_attention_layernorm: Pre-FFN layer normalization.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize the MiniMaxDecoderLayer module.

        Args:
            config: Model configuration containing layer parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
            layer_idx: Index of this layer in the transformer stack.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            attn_block: type[nn.Module] = MiniMaxLightningAttention
            self.attn_alpha_factor = config.linear_attn_alpha_factor
            self.attn_beta_factor = config.linear_attn_beta_factor
        else:
            attn_block = MiniMaxAttention
            self.attn_alpha_factor = config.full_attn_alpha_factor
            self.attn_beta_factor = config.full_attn_beta_factor

        mlp_block: type[nn.Module] = MiniMaxSparseMoeBlock

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.block_sparse_moe = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.mlp_alpha_factor = config.mlp_alpha_factor
        self.mlp_beta_factor = config.mlp_beta_factor

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None,
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RecurrentCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Array | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info: Mask information for attention computation.
            position_ids: Position indices of shape (batch, seq_len).
            attention_mask: Optional boolean mask of shape (batch, seq_len).
            mode: Runtime mode (train, decode, etc.).
            cache_view: Optional cache view for key-value or recurrent state caching.
            cache_metadata: Optional metadata for cache operations.
            output_attentions: Whether to return attention weights.
            output_router_logits: Whether to return router logits.
            frequencies: Optional rotary embedding frequencies.

        Returns:
            DecoderLayerOutput containing hidden states and optional attention/router outputs.
        """
        residual = self.input_layernorm(hidden_states)
        residual = apply_logical_sharding(
            residual,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        attn_weights = None
        if self.layer_type == "linear_attention":
            linear_cache = cache_view if isinstance(cache_view, RecurrentCacheView) else None
            attn_out, linear_cache = self.self_attn(
                residual,
                attention_mask=attention_mask,
                mode=mode,
                cache_view=linear_cache,
            )
            cache_view = linear_cache
        else:
            kv_cache = cache_view if isinstance(cache_view, TransformerCacheView) else None
            attn_outputs: AttentionLayerOutput = self.self_attn(
                residual,
                mask_info,
                position_ids,
                mode,
                kv_cache,
                cache_metadata,
                output_attentions,
                frequencies,
            )
            attn_out = attn_outputs.attention_output
            attn_weights = attn_outputs.attention_weight
            cache_view = attn_outputs.cache_view

        hidden_states = residual * self.attn_alpha_factor + attn_out * self.attn_beta_factor

        ff_input = self.post_attention_layernorm(hidden_states)
        training = mode == common_types.MODE_TRAIN
        ff_out, router_logits = self.block_sparse_moe(ff_input, training=training, layer_idx=self.layer_idx)
        hidden_states = ff_input * self.mlp_alpha_factor + ff_out * self.mlp_beta_factor

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_weights if output_attentions else None,
            cache_view=cache_view,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=MiniMaxConfig, model_type="minimax")
@register_module(TaskType.BASE_MODULE, config=MiniMaxConfig, model_type="minimax_text_01")
@register_module(TaskType.BASE_MODULE, config=MiniMaxConfig, model_type="MiniMaxText01")
class MiniMaxModel(EasyDeLBaseModule):
    """Base transformer model for MiniMax architecture.

    This model implements the core MiniMax transformer with hybrid attention
    (combining lightning and standard attention) and sparse Mixture-of-Experts
    feed-forward layers. It serves as the backbone for downstream tasks.

    Attributes:
        config: Model configuration object.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting.
        embed_tokens: Token embedding layer.
        layers: List of MiniMaxDecoderLayer modules.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the MiniMaxModel.

        Args:
            config: Model configuration containing architecture parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
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
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            rngs=rngs,
        )

        self.layers = [
            MiniMaxDecoderLayer(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
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
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the MiniMax model.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed input embeddings. Mutually exclusive with input_ids.
            attention_mask: Optional boolean mask of shape (batch, seq_len).
            mask_info: Optional pre-computed mask information.
            position_ids: Optional position indices of shape (batch, seq_len).
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router logits from MoE layers.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Optional cache for incremental decoding.
            cache_metadata: Optional metadata for cache operations.

        Returns:
            MoeModelOutput containing last hidden state and optional intermediate outputs.

        Raises:
            ValueError: If neither or both of input_ids and inputs_embeds are provided.
            ValueError: If sequence length exceeds max_position_embeddings.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Maximum Position Embedding Reached (expected <= {self.config.max_position_embeddings}, got {sequence_length})."
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )

        if past_key_values is None:
            past_key_values = HybridCache.init_empty(len(self.layers))

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
                mask_info=mask_info,
                position_ids=position_ids,
                attention_mask=attention_mask,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=bool(output_attentions),
                output_router_logits=bool(output_router_logits),
                frequencies=self.frequencies,
            )

            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        router_losses = None
        if output_router_logits and all_router_logits is not None:
            router_losses = (
                auxiliary_load_balancing_loss_func(
                    gate_logits=all_router_logits,
                    num_experts=self.config.num_local_experts,
                    top_k=self.config.num_experts_per_tok,
                    attention_mask=attention_mask,
                ),
            )

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            all_router_losses=router_losses,
            past_key_values=past_key_values,
        )

    def get_encoder(self):
        """Get the encoder module.

        Raises:
            NotImplementedError: MiniMax is a decoder-only architecture.
        """
        raise NotImplementedError("MiniMax is a decoder-only model and does not have an encoder.")

    def get_decoder(self):
        """Get the decoder module.

        Returns:
            The model itself, as MiniMax is a decoder-only architecture.
        """
        return self

    def get_lm_head(self):
        """Get the language model head.

        Raises:
            NotImplementedError: The base model does not include an LM head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            The embed_tokens module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=MiniMaxConfig, model_type="minimax")
@register_module(TaskType.CAUSAL_LM, config=MiniMaxConfig, model_type="minimax_text_01")
@register_module(TaskType.CAUSAL_LM, config=MiniMaxConfig, model_type="MiniMaxText01")
class MiniMaxForCausalLM(BaseCausalLMModule[MiniMaxModel, MiniMaxConfig]):
    """MiniMax model with a causal language modeling head.

    This model extends MiniMaxModel with a language modeling head for next-token
    prediction tasks. It supports both training and generation with hybrid caching
    for efficient inference.

    Attributes:
        _task_type: The task type (CAUSAL_LM).
        _model_type: The model type identifier.
        _config_class: The configuration class for this model.
        model: The underlying MiniMaxModel backbone.
        lm_head: Linear layer projecting hidden states to vocabulary logits.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "minimax_text_01"
    _config_class = MiniMaxConfig

    def __init__(
        self,
        config: MiniMaxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the MiniMaxForCausalLM model.

        Args:
            config: Model configuration containing architecture parameters.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix operations.
            rngs: Random number generators for initialization.
        """
        super().__init__(
            config=config,
            base_model_class=MiniMaxModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=config.router_aux_loss_coef,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed input embeddings. Mutually exclusive with input_ids.
            attention_mask: Optional boolean mask of shape (batch, seq_len).
            mask_info: Optional pre-computed mask information.
            position_ids: Optional position indices of shape (batch, seq_len).
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router logits from MoE layers.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Optional cache for incremental decoding.
            cache_metadata: Optional metadata for cache operations.
            apply_lm_head: Whether to apply the language model head to compute logits.

        Returns:
            MoeCausalLMOutput containing logits and optional intermediate outputs.
        """
        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=None,
        )

    def get_inference_cache_type(self) -> str:
        """Get the cache type used for inference.

        Returns:
            The string "hybrid" indicating this model uses hybrid caching
            (combining transformer and recurrent caches).
        """
        return "hybrid"

    def get_operations_cache_view(self) -> dict[int, type]:
        """Get the cache view type for each layer.

        Returns:
            A dictionary mapping layer indices to their respective cache view types
            (RecurrentCacheView for lightning attention, TransformerCacheView for
            standard attention).
        """
        from easydel.layers.caching import RecurrentCacheView, TransformerCacheView

        layer_types = self.config.get_text_config().layer_types
        return {
            idx: (RecurrentCacheView if layer_type == "linear_attention" else TransformerCacheView)
            for idx, layer_type in enumerate(layer_types)
        }

    def create_recurrent_cache_config(self, batch_size: int):
        """Create configuration for recurrent cache used in lightning attention layers.

        Args:
            batch_size: The batch size for cache allocation.

        Returns:
            RecurrentCacheConfig with appropriate dimensions for the model's
            lightning attention layers.
        """
        from eformer.escale import PartitionAxis

        from easydel.layers.caching import RecurrentCacheConfig

        text_config = self.config.get_text_config()
        partition_axis = getattr(text_config, "partition_axis", None) or PartitionAxis()
        num_hidden_layers = int(getattr(text_config, "num_hidden_layers", 1))
        num_heads = int(getattr(text_config, "num_attention_heads", 1))
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = int(text_config.hidden_size // text_config.num_attention_heads)

        return RecurrentCacheConfig.create(
            num_hidden_layers=num_hidden_layers,
            partition_axis=partition_axis,
            batch_size=int(batch_size),
            conv_dim=1,
            conv_kernel_size=1,
            recurrent_state_shape=(num_heads, int(head_dim), int(head_dim)),
        )
