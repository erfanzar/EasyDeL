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


from functools import cached_property
from typing import ClassVar

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
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule
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
from easydel.layers.components import ColumnParallelLinear, Embed, RowParallelLinear

from .dbrx_configuration import DbrxConfig


class DbrxAttention(UnifiedAttention):
    """DBRX Attention module with fused QKV projection.

    This module implements the multi-head attention mechanism used in the DBRX model.
    It supports Grouped Query Attention (GQA) and Rotary Position Embeddings (RoPE).
    The query, key, and value projections are combined into a single fused linear layer
    for efficiency, and supports optional QKV clipping.

    Overrides forward_standard to efficiently handle fused QKV projection.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        Wqkv (ColumnParallelLinear): Fused linear layer for query, key, and value projections.
        out_proj (RowParallelLinear): Linear layer for the output projection.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
        rotary (RoPE): Rotary position embedding module.
        resid_dropout (nn.Dropout): Residual dropout layer.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "output_projection": "out_proj",
        "query_key_value_projection": "Wqkv",
    }

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initializes the DbrxAttention module.

        Args:
            config (DbrxConfig): The configuration object for the DBRX model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
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
        )

    def define_network(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ):
        """Override to create fused QKV projection instead of separate Q/K/V.

        Args:
            config: Model configuration
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: JAX precision setting
            rngs: Random number generators
        """
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        qkv_size = num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim

        self.Wqkv = ColumnParallelLinear(
            config.hidden_size,
            qkv_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.attention_performer = self._create_attention_performer(config, rngs)
        self.rotary = self._create_rotary(config, dtype)
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def _create_rotary(self, config: DbrxConfig, dtype: jnp.dtype):
        """Create rotary position embedding layer with DBRX specific configuration.

        Args:
            config: Model configuration
            dtype: Data type for computations
        """
        return config.get_basic_rope(
            dtype=dtype,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            is_neox_style=True,
            base=config.attn_config.rope_theta,
        )

    def _create_attention_performer(self, config: DbrxConfig, rngs: nn.Rngs):
        """Create attention performer module with DBRX specific settings.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=0.0,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ):
        """Forward pass for DBRX attention with fused QKV projection.

        Handles the fused QKV projection efficiently with optional clipping,
        applies rotary position embeddings, and computes attention outputs.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information for the attention computation.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for key-value caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.
            alibi (Array | None, optional): ALiBi position bias (not used in DBRX). Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output, optional attention weights, and cache view.
        """
        batch_size, sequence_length = hidden_states.shape[:2]
        qkv_states = checkpoint_name(self.Wqkv(hidden_states), "attn_qkv")
        if self.config.attn_config.clip_qkv is not None:
            qkv_states = qkv_states.clip(
                min=-self.config.attn_config.clip_qkv,
                max=self.config.attn_config.clip_qkv,
            )
        query_size = self.hidden_size
        key_size = self.num_key_value_heads * self.head_dim
        query_states = qkv_states[..., :query_size]
        key_states = qkv_states[..., query_size : query_size + key_size]
        value_states = qkv_states[..., query_size + key_size :]

        query_states = query_states.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)
        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)
        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)
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
            sliding_window=getattr(self, "sliding_window", None),
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
            mask_info=mask_info,
            causal=self.causal,
            sliding_window=getattr(self, "sliding_window", None),
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )

    def _get_output_proj(self):
        """Override to access output projection with DBRX's naming convention.

        Returns:
            Output projection layer
        """
        return self.out_proj


class DbrxNormAttentionNorm(nn.Module):
    """Normalization-Attention-Normalization module for DBRX models.

    Implements a unique architecture pattern with normalization layers
    surrounding the attention mechanism for improved gradient flow.
    This module applies layer normalization before and after attention,
    with residual connections for stable training.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        norm_1 (nn.LayerNorm): Pre-attention layer normalization.
        attn (DbrxAttention): DBRX attention module.
        norm_2 (nn.LayerNorm): Post-attention layer normalization.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize the DbrxNormAttentionNorm module.

        Args:
            config (DbrxConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.norm_1 = nn.LayerNorm(
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )
        self.attn = DbrxAttention(  # statics 3,5,6,7
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.norm_2 = nn.LayerNorm(
            self.config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )

        self.dropout = nn.Dropout(
            self.config.resid_pdrop,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the normalization-attention-normalization module.

        Applies pre-attention normalization, attention mechanism with residual connection,
        and post-attention normalization.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information for attention computation.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for key-value caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains normalized hidden states, residual states,
                optional attention weights, and cache view.
        """
        residual_states = hidden_states
        hidden_states = self.norm_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states=hidden_states,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            frequencies=frequencies,
            mode=mode,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
        )
        hidden_states = attn_outputs.attention_output
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + residual_states

        residual_states = hidden_states
        hidden_states = self.norm_2(hidden_states)

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            residual_states=residual_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            gate_loss=None,
            cache_view=attn_outputs.cache_view,
        )


class DbrxExpertGLU(nn.Module):
    """Gated Linear Unit expert module for DBRX mixture of experts.

    Implements a single expert network with gated activation for
    specialized processing in the MoE architecture. Uses a SwiGLU-style
    gating mechanism with two parallel projections (w1, v1) and an output
    projection (w2).

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        w1 (ArrayParam): Gate projection weights for all experts.
        v1 (ArrayParam): Up projection weights for all experts.
        w2 (ArrayParam): Down projection weights for all experts.
        activation_fn: Activation function for the gating mechanism.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
    ):
        """Initialize the DbrxExpertGLU module.

        Args:
            config (DbrxConfig): Model configuration with FFN parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int | None, optional): Index of this layer. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        shape = (
            self.config.ffn_config.moe_num_experts * self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        self.w1 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.params())
        self.v1 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.params())
        self.w2 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.params())
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

    def __call__(self, x: Array, expert_idx: int) -> Array:
        """Apply the gated linear unit transformation for a specific expert.

        Computes SwiGLU-style transformation: (act(x @ w1) * (x @ v1)) @ w2
        for the specified expert.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            expert_idx (int): Index of the expert to use for computation.

        Returns:
            Array: Transformed tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        expert_shape = (
            self.config.ffn_config.moe_num_experts,
            self.config.ffn_config.ffn_hidden_size,
            self.config.d_model,
        )
        expert_w1 = checkpoint_name(self.w1.value.reshape(expert_shape)[expert_idx], name="moe_expert_w1")
        expert_v1 = checkpoint_name(self.v1.value.reshape(expert_shape)[expert_idx], name="moe_expert_v1")
        expert_w2 = checkpoint_name(self.w2.value.reshape(expert_shape)[expert_idx], name="moe_expert_w2")

        x1 = jnp.matmul(
            x,
            jnp.expand_dims(expert_w1.T, 0),
            precision=self.precision,
        )
        x2 = jnp.matmul(
            x,
            jnp.expand_dims(expert_v1.T, 0),
            precision=self.precision,
        )
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = jnp.matmul(
            x1,
            jnp.expand_dims(expert_w2, 0),
            precision=self.precision,
        )
        return x1


class DbrxExperts(nn.Module):
    """Collection of expert networks for DBRX mixture of experts.

    Manages multiple expert networks that can be selected and combined
    based on routing decisions for conditional computation. Iterates through
    all experts and aggregates their weighted outputs.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        mlp (DbrxExpertGLU): Shared expert MLP containing all expert weights.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DbrxExperts module.

        Args:
            config (DbrxConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.mlp = DbrxExpertGLU(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Array,
        weights: Array,
        top_weights: Array,
        top_experts: Array,
    ):
        """Forward pass through all experts with weighted aggregation.

        Iterates through all experts, applies each to the input, and aggregates
        the outputs based on the routing weights assigned by the router.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            weights (Array): Full routing weights for all experts.
            top_weights (Array): Top-k routing weights for selected experts.
            top_experts (Array): Indices of top-k selected experts.

        Returns:
            Array: Aggregated expert outputs of shape (batch_size, sequence_length, hidden_dim).
        """
        final_hidden_state = jnp.zeros_like(x)
        for index in range(self.config.ffn_config.moe_num_experts):
            output_moe_layer = self.mlp(x, index)
            final_hidden_state += (
                jnp.sum(jnp.multiply(index == top_experts, top_weights), axis=-1)[:, :, None] * output_moe_layer
            )
        return final_hidden_state


class DbrxRouter(nn.Module):
    """Router module for DBRX mixture of experts.

    Determines which experts to activate for each input token,
    implementing sparse routing for efficient computation. Uses top-k
    routing with optional jittering and weight normalization.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        hidden_size (int): Dimension of input hidden states.
        moe_num_experts (int): Total number of experts.
        moe_top_k (int): Number of experts to select for each token.
        moe_jitter_eps (float | None): Jitter epsilon for training regularization.
        moe_normalize_expert_weights (int | None): Normalization order for weights.
        uniform_expert_assignment (bool): Whether to use uniform expert assignment.
        layer (ColumnParallelLinear): Router projection layer.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DbrxRouter module.

        Args:
            config (DbrxConfig): Model configuration with router parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = self.config.d_model
        self.moe_num_experts = self.config.ffn_config.moe_num_experts
        self.moe_top_k = self.config.ffn_config.moe_top_k
        self.moe_jitter_eps = self.config.ffn_config.moe_jitter_eps
        self.moe_normalize_expert_weights = self.config.ffn_config.moe_normalize_expert_weights
        self.uniform_expert_assignment = self.config.ffn_config.uniform_expert_assignment

        self.layer = ColumnParallelLinear(
            config.hidden_size,
            self.moe_num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def jitter(self, x: Array) -> Array:
        """Apply multiplicative jitter noise for training regularization.

        Adds random noise to the input to improve router training stability
        and encourage exploration of different expert assignments.

        Args:
            x (Array): Input tensor to apply jitter to.

        Returns:
            Array: Jittered input tensor with same shape as input.

        Raises:
            RuntimeError: If moe_jitter_eps is not set in config.
        """
        if self.moe_jitter_eps is None:
            raise RuntimeError("The router does not have moe_jitter_eps set.")
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = jax.random.normal(self.make_rng("params"), x.shape, dtype=x.dtype)
        return low + noise * (high - low)

    def __call__(self, x: Array, deterministic: bool = True) -> tuple[Array, Array, Array]:
        """Route tokens to experts using top-k selection.

        Computes routing scores for all experts, selects top-k experts for each token,
        and optionally normalizes the routing weights.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            deterministic (bool, optional): If False, applies jitter noise. Defaults to True.

        Returns:
            tuple[Array, Array, Array]: A tuple containing:
                - weights: Full routing weights for all experts (batch, seq_len, num_experts).
                - top_weights: Weights for selected experts (batch, seq_len, top_k).
                - top_experts: Indices of selected experts (batch, seq_len, top_k).
        """
        if not deterministic and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        weights = self.layer(x.astype(jnp.promote_types(self.dtype, jnp.float32)))
        weights = jax.nn.softmax(weights.astype(jnp.promote_types(self.dtype, jnp.float32)))
        top_weights, top_experts = jax.lax.top_k(weights, self.moe_top_k)

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / jnp.linalg.norm(
                top_weights,
                ord=int(self.moe_normalize_expert_weights),
                axis=-1,
                keepdims=True,
            )

        if self.uniform_expert_assignment:
            top_experts = jax.lax.stop_gradient(
                (
                    jnp.arange(
                        0,
                        jnp.prod(
                            jnp.asarray(top_experts.shape, dtype=jnp.int32),
                            dtype=jnp.int32,
                        ),
                        dtype=top_experts.dtype,
                    )
                    % self.moe_num_experts
                ).reshape(top_experts.shape)
            )

        weights = weights.astype(x.dtype)
        top_weights = top_weights.astype(x.dtype)
        return weights, top_weights, top_experts


class DbrxFFN(nn.Module):
    """Feedforward network with mixture of experts for DBRX models.

    Combines router and expert networks to implement sparse MoE
    feedforward layers with conditional computation. The router determines
    which experts process each token, and the experts' outputs are combined
    based on routing weights.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        router (DbrxRouter): Router module for expert selection.
        experts (DbrxExperts): Collection of expert MLP networks.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DbrxFFN module.

        Args:
            config (DbrxConfig): Model configuration with FFN and MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.router = DbrxRouter(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.experts = DbrxExperts(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """Forward pass through the MoE feedforward network.

        Routes input tokens to selected experts and combines their outputs.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - Output hidden states after MoE processing (batch, seq_len, hidden_dim).
                - Router logits for auxiliary loss computation (batch, seq_len, num_experts).
        """
        x = apply_logical_sharding(
            x,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        weights, top_weights, top_experts = self.router(x)
        weights = checkpoint_name(weights, name="moe_router_logits")
        out = checkpoint_name(self.experts(x, weights, top_weights, top_experts), name="moe_expert_output")
        out = apply_logical_sharding(
            out,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return out, weights


class DbrxBlock(nn.Module):
    """Single transformer block for DBRX models.

    Integrates attention mechanisms with mixture of experts feedforward
    networks, using residual connections and normalization. Each block
    contains a norm-attention-norm module followed by a MoE FFN.

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        hidden_size (int): Dimension of hidden states.
        resid_pdrop (float): Residual dropout probability.
        norm_attn_norm (DbrxNormAttentionNorm): Attention module with normalization.
        ffn (DbrxFFN): Mixture of experts feedforward network.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize a DBRX decoder block.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.hidden_size = self.config.d_model
        self.resid_pdrop = self.config.resid_pdrop
        attn_block = DbrxNormAttentionNorm
        ffn_block = DbrxFFN
        attn_block, ffn_block = auto_remat(
            attn_block,
            ffn_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.norm_attn_norm = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )
        self.ffn = ffn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the DBRX decoder block.

        Applies attention with normalization followed by MoE feedforward network
        with residual connections.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information for attention computation.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES | None): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for key-value caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """

        decoder_output = self.norm_attn_norm(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = decoder_output.hidden_states
        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = decoder_output.residual_states + hidden_states

        return decoder_output.replace(
            hidden_states=hidden_states,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=DbrxConfig, model_type="dbrx")
class DbrxModel(EasyDeLBaseModule):
    """The base DBRX model transformer.

    This class represents the core transformer architecture of the DBRX model,
    consisting of an embedding layer, multiple DbrxBlock layers (with sparse MoE),
    and a final layer normalization.

    The DBRX architecture features:
    - Fused QKV attention with optional clipping
    - Mixture of Experts (MoE) feedforward networks
    - Rotary Position Embeddings (RoPE)
    - Pre-normalization architecture

    Attributes:
        config (DbrxConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        padding_idx (int | None): Token ID used for padding.
        vocab_size (int): Size of the vocabulary.
        emb_pdrop (float): Embedding dropout probability.
        wte (Embed): Token embedding layer.
        blocks (list[DbrxBlock]): List of decoder blocks.
        norm_f (nn.LayerNorm): Final layer normalization.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the DBRX base model.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size
        self.emb_pdrop = self.config.emb_pdrop

        self.wte = Embed(
            self.config.vocab_size,
            self.config.d_model,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.blocks = nn.List([
            DbrxBlock(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ])
        self.norm_f = nn.LayerNorm(
            self.config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        """Compute and cache the rotary position embedding frequencies.

        Returns:
            Array: Precomputed RoPE frequencies for position embeddings.
        """
        return self.config.get_basic_frequencies(
            rotary_dim=self.config.hidden_size // self.config.num_attention_heads,
            head_size=self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.attn_config.rope_theta,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the DBRX base model.

        Processes input tokens through embedding, all decoder blocks with MoE,
        and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None (uses config.output_attentions).
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None (uses config.output_hidden_states).
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None (uses config.output_router_logits).
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                optional router_logits, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        hidden_states = inputs_embeds
        all_hidden_states = ()
        all_router_logits = ()
        all_attentions = ()
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.blocks))
        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = outputs.hidden_states

            hidden_states = apply_logical_sharding(
                hidden_states,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.partition_manager,
            )

            if output_attentions:
                all_attentions += (outputs.attention_weight,)
            if output_router_logits:
                all_router_logits += (outputs.router_logits,)
            past_key_values[idx] = outputs.cache_view
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For DbrxModel (decoder-only), this is not applicable.
        """
        raise NotImplementedError("DbrxModel is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For DbrxModel, this is the model itself.
        """
        return self

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        DbrxModel does not include the lm_head.
        """
        raise NotImplementedError("DbrxModel does not include the language model head. See DbrxForCausalLM.")

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=DbrxConfig, model_type="dbrx")
class DbrxForCausalLM(BaseCausalLMModule[DbrxModel, DbrxConfig]):
    """DBRX model with a language modeling head for causal language modeling tasks.

    This model is a sparse MoE transformer-based language model with causal attention masks
    applied to perform autoregressive language generation.

    Attributes:
        config (DbrxConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "dbrx"
    _config_class = DbrxConfig

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize DBRX model for causal language modeling.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=DbrxModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass through the DBRX causal language model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply language model head projection.
                Defaults to True.

        Returns:
            MoeCausalLMOutput: Contains logits, optional hidden_states, optional attentions,
                optional router_logits, auxiliary loss, and updated past_key_values.
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
            aux_loss_fn=self._compute_aux_loss,
        )

    def _compute_aux_loss(self, outputs, attention_mask):
        """Compute auxiliary load balancing loss from router logits.

        Args:
            outputs: Model outputs containing router_logits from MoE layers.
            attention_mask: Attention mask to exclude padding tokens from loss computation.

        Returns:
            Optional auxiliary loss value for load balancing, or None if router_logits unavailable.
        """
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.ffn_config.moe_num_experts,
            top_k=self.config.ffn_config.moe_top_k,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=DbrxConfig, model_type="dbrx")
class DbrxForSequenceClassification(BaseSequenceClassificationModule[DbrxModel, DbrxConfig]):
    """DBRX model for sequence classification tasks.

    This class extends the base DBRX model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (DbrxConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "dbrx"
    _config_class = DbrxConfig

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize DBRX model for sequence classification.

        Args:
            config (DbrxConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=DbrxModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the DBRX sequence classification model.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return router logits from MoE layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            SequenceClassifierOutput: Contains classification logits, optional hidden_states,
                optional attentions, auxiliary loss, and updated past_key_values.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        transformer_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
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

        aux_loss = None
        if output_router_logits and transformer_outputs.router_logits is not None:
            aux_loss = auxiliary_load_balancing_loss_func(
                gate_logits=transformer_outputs.router_logits,
                num_experts=self.config.ffn_config.moe_num_experts,
                top_k=self.config.ffn_config.moe_top_k,
                attention_mask=attention_mask,
            )
            aux_loss = aux_loss * self.config.router_aux_loss_coef

        return SequenceClassifierOutput(
            logits=pooled_logits,
            past_key_values=past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            aux_loss=aux_loss,
        )
