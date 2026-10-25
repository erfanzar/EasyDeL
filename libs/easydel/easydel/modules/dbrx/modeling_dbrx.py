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

"""Spectrax implementation of the DBRX (Databricks) Mixture-of-Experts decoder.

DBRX is a fine-grained MoE transformer with 16 experts per layer and top-4
routing in its largest configuration. The reference architecture also uses
multi-query attention (single KV head), QKV clipping, and standard RoPE.

Building blocks:

- :class:`DbrxAttention` — fused-QKV multi-query attention with optional QKV
  clipping.
- :class:`DbrxNormAttentionNorm` — pre/post norm wrapper around the attention
  module.
- :class:`DbrxExpertGLU` / :class:`DbrxExperts` — fused per-expert SwiGLU FFN
  bank.
- :class:`DbrxRouter` — softmax router with optional jitter and uniform-
  assignment debug knob.
- :class:`DbrxFFN` — router + experts wired together.
- :class:`DbrxBlock` — single decoder layer.

Public model classes (registered with the factory):

- :class:`DbrxModel` — base decoder.
- :class:`DbrxForCausalLM` — causal LM head + auxiliary load-balancing loss.
- :class:`DbrxForSequenceClassification` — pooled classifier head.
"""

from functools import cached_property
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import spectrax as spx
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int
from spectrax import apply_logical_sharding, common_types, nn

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
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.sharding import is_mpmd_mesh
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat
from easydel.layers import (
    ColumnParallelLinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    ParallelLinear,
    RowParallelLinear,
)
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.layers.moe import BaseMoeModule
from easydel.layers.norms import LayerNorm
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .dbrx_configuration import DbrxConfig


def _config_uses_mpmd(config) -> bool:
    """Return ``True`` when the resolved mesh is MPMD (pipeline-parallel).

    Checks the live mesh first via :func:`is_mpmd_mesh` or a ``pp`` axis
    size greater than 1. Falls back to inspecting the declared
    ``sharding_axis_names`` / ``sharding_axis_dims`` on ``config`` for the
    pre-mesh planning phase, so MoE-routing code can pick the correct
    expert-parallel implementation before the mesh is built.
    """
    mesh = getattr(config, "_hidden_mesh", None) or getattr(config, "mesh", None)
    if is_mpmd_mesh(mesh) or int(getattr(mesh, "shape", {}).get("pp", 1)) > 1:
        return True
    axis_names = tuple(getattr(config, "sharding_axis_names", ()))
    axis_dims = tuple(getattr(config, "sharding_axis_dims", ()))
    if "pp" not in axis_names:
        return False
    pp_index = axis_names.index("pp")
    return pp_index < len(axis_dims) and int(axis_dims[pp_index]) > 1


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
        rngs (spx.Rngs): Random number generators.
        Wqkv (ColumnParallelLinear): Fused linear layer for query, key, and value projections.
        out_proj (RowParallelLinear): Linear layer for the output projection.
        attention_performer (FlexibleAttentionModule): Module to perform the core attention computation.
        rotary (RoPE): Rotary position embedding module.
        resid_dropout (nn.Dropout): Residual dropout, shared with ``DbrxNormAttentionNorm.dropout``.

    Note:
        The residual dropout (``config.resid_pdrop``) is applied once per layer in
        :class:`DbrxNormAttentionNorm`, matching the reference DBRX implementation, not inside the attention.
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
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initializes the DbrxAttention module.

        Args:
            config (DbrxConfig): The configuration object for the DBRX model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (spx.Rngs): Random number generators.
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
        self.rngs = rngs

    def define_network(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: spx.Rngs,
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

    def _create_attention_performer(self, config: DbrxConfig, rngs: spx.Rngs):
        """Create attention performer module with DBRX specific settings.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attn_config.attn_pdrop,
        )

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
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
            deterministic=not self.training,
            dropout_rng=(self.rngs.dropout if self.training and self.config.attn_config.attn_pdrop > 0 else None),
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")
        attn_output = self.shard_attention_prod(attn_output)

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


class DbrxNormAttentionNorm(spx.Module):
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
        norm_1 (LayerNorm): Pre-attention layer normalization.
        attn (DbrxAttention): DBRX attention module.
        norm_2 (LayerNorm): Post-attention layer normalization.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    kernel_init = staticmethod(jax.nn.initializers.ones)

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize the DbrxNormAttentionNorm module.

        Args:
            config (DbrxConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        self.norm_1 = LayerNorm(
            self.config.hidden_size,
            epsilon=1e-5,
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
        self.norm_2 = LayerNorm(
            self.config.hidden_size,
            epsilon=1e-5,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            rngs=rngs,
        )

        # Preserve both public dropout attributes while applying one shared layer once.
        self.dropout = self.attn.resid_dropout

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type: ignore
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


class DbrxExpertGLU(spx.Module):
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
        rngs: spx.Rngs,
        layer_idx: int | None = None,
    ):
        """Initialize the DbrxExpertGLU module.

        Args:
            config (DbrxConfig): Model configuration with FFN parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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
        self.w1 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.parameters)
        self.v1 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.parameters)
        self.w2 = ArrayParam.bound(shape=shape, dtype=self.param_dtype, init_method="normal", key=rngs.parameters)
        self.activation_fn = ACT2FN[self.config.ffn_config.ffn_act_fn["name"]]

    def forward(self, x: Array, expert_idx: int) -> Array:
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

        # DBRX stores every expert matrix as ``[ffn_hidden_size, d_model]``:
        # gate/up project with the transpose, down with the matrix itself
        # (original DBRX and transformers>=5.17; 5.13 had these flipped, which
        # only type-checks when ffn_hidden_size == d_model).
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


class DbrxExperts(spx.Module):
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
        rngs: spx.Rngs,
    ):
        """Initialize the DbrxExperts module.

        Args:
            config (DbrxConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(self, rows: Array, group_sizes: Array, sorted_experts: Array | None = None) -> Array:
        """Expert SwiGLU over expert-sorted token rows (the ``BaseMoeModule`` standard-path contract).

        ``DbrxFFN`` normally dispatches through the fused path with stacked kernels built from the same weights; this
        method serves ``MoEMethods.STANDARD_MOE``, which calls the expert layer directly.

        Args:
            rows (Array): Token rows sorted by expert, shape ``(num_rows, d_model)``.
            group_sizes (Array): Rows per expert, shape ``(moe_num_experts,)``.
            sorted_experts (Array | None): Expert id per row (unused; ``group_sizes`` defines the grouping).

        Returns:
            Array: Expert outputs per row, shape ``(num_rows, d_model)``.
        """
        del sorted_experts
        experts = self.config.ffn_config.moe_num_experts
        expert_shape = (experts, self.config.ffn_config.ffn_hidden_size, self.config.d_model)
        # DBRX stores each expert matrix as ``[ffn_hidden_size, d_model]``: gate/up use the transpose, down the matrix.
        wi = self.mlp.w1.value.reshape(expert_shape).swapaxes(-1, -2)
        wu = self.mlp.v1.value.reshape(expert_shape).swapaxes(-1, -2)
        wd = self.mlp.w2.value.reshape(expert_shape)
        group_sizes = group_sizes.astype(jnp.int32)
        gate = jax.lax.ragged_dot(rows, wi.astype(rows.dtype), group_sizes, precision=self.precision)
        up = jax.lax.ragged_dot(rows, wu.astype(rows.dtype), group_sizes, precision=self.precision)
        hidden = (self.mlp.activation_fn(gate) * up).astype(rows.dtype)
        return jax.lax.ragged_dot(hidden, wd.astype(rows.dtype), group_sizes, precision=self.precision)


class DbrxRouter(spx.Module):
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
        rngs: spx.Rngs,
    ):
        """Initialize the DbrxRouter module.

        Args:
            config (DbrxConfig): Model configuration with router parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

        self.layer = ParallelLinear(
            config.hidden_size,
            self.moe_num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
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
        # Reference DBRX draws the multiplicative jitter from U(1 - eps, 1 + eps).
        return jax.random.uniform(self.rngs.parameters, x.shape, dtype=x.dtype, minval=low, maxval=high)

    def forward(self, x: Array, deterministic: bool = True) -> tuple[Array, Array, Array]:
        """Route tokens to experts using top-k selection.

        Computes routing scores for all experts, selects top-k experts for each token,
        and optionally normalizes the routing weights.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            deterministic (bool, optional): If False, applies jitter noise. Defaults to True.

        Returns:
            tuple[Array, Array, Array]: A tuple containing:
                - router_logits: Raw router logits for all experts (batch, seq_len, num_experts).
                - top_weights: Weights for selected experts (batch, seq_len, top_k).
                - top_experts: Indices of selected experts (batch, seq_len, top_k).
        """
        if not deterministic and self.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        batch_size, sequence_length, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim).astype(jnp.promote_types(self.dtype, jnp.float32))

        router_logits_flat = self.layer(x_flat)
        routing_weights_flat = jax.nn.softmax(
            router_logits_flat.astype(jnp.promote_types(self.dtype, jnp.float32)), axis=-1
        )
        top_weights_flat, top_experts_flat = jax.lax.top_k(routing_weights_flat, self.moe_top_k)

        if self.moe_normalize_expert_weights is not None:
            norm_order = float(self.moe_normalize_expert_weights)
            if norm_order == 1.0:
                denom = jnp.sum(top_weights_flat, axis=-1, keepdims=True)
            else:
                denom = jnp.linalg.norm(top_weights_flat, ord=norm_order, axis=-1, keepdims=True)
            top_weights_flat = top_weights_flat / denom

        if self.uniform_expert_assignment:
            top_experts_flat = jax.lax.stop_gradient(
                (
                    jnp.arange(
                        0,
                        jnp.prod(
                            jnp.asarray(top_experts_flat.shape, dtype=jnp.int32),
                            dtype=jnp.int32,
                        ),
                        dtype=top_experts_flat.dtype,
                    )
                    % self.moe_num_experts
                ).reshape(top_experts_flat.shape)
            )

        router_logits = router_logits_flat.reshape(batch_size, sequence_length, self.moe_num_experts)
        top_weights = top_weights_flat.reshape(batch_size, sequence_length, self.moe_top_k)
        top_experts = top_experts_flat.reshape(batch_size, sequence_length, self.moe_top_k)
        return router_logits, top_weights, top_experts


class DbrxFFN(BaseMoeModule):
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
        rngs: spx.Rngs,
    ):
        """Initialize the DbrxFFN module.

        Args:
            config (DbrxConfig): Model configuration with FFN and MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.ffn_config.moe_num_experts,
            num_experts_per_tok=config.ffn_config.moe_top_k,
            hidden_size=config.d_model,
            routing_strategy=MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
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

    def forward(self, x: Array) -> tuple[Array, Array]:
        """Forward pass through the MoE feedforward network.

        Routes input tokens to selected experts and combines their outputs.

        Args:
            x (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - Output hidden states after MoE processing (batch, seq_len, hidden_dim).
                - Raw router logits for auxiliary loss computation (batch, seq_len, num_experts).
        """
        if not _config_uses_mpmd(self.config):
            x = apply_logical_sharding(
                x,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )

        def select_hook(scores: Array, _pre_bias: Array | None, k: int) -> tuple[Array, Array]:
            weights, indices = jax.lax.top_k(scores, k=k)
            norm_order = self.config.ffn_config.moe_normalize_expert_weights
            if norm_order is not None:
                order = float(norm_order)
                denom = (
                    jnp.sum(weights, axis=-1, keepdims=True)
                    if order == 1.0
                    else jnp.linalg.norm(weights, ord=order, axis=-1, keepdims=True)
                )
                weights = weights / denom
            if self.config.ffn_config.uniform_expert_assignment:
                indices = jax.lax.stop_gradient(
                    (jnp.arange(indices.size, dtype=indices.dtype) % self.n_routed_experts).reshape(indices.shape)
                )
            return weights, indices

        hooks = self.moe_hooks.replace(
            normalize_gate_logits=lambda logits: jax.nn.softmax(logits.astype(jnp.float32), axis=-1),
            select_hook=select_hook,
            refine_weights_hook=None,
        )

        def gate_layer(gate_input: Array) -> Array:
            """Perturb only the router input, including on the standard MoE path."""
            if self.training and self.config.ffn_config.moe_jitter_eps is not None:
                gate_input = gate_input * self.router.jitter(gate_input)
            return self.router.layer(gate_input)

        experts = self.experts.mlp
        expert_count = self.config.ffn_config.moe_num_experts
        intermediate = self.config.ffn_config.ffn_hidden_size
        wi = experts.w1.value.reshape(expert_count, intermediate, self.config.d_model).swapaxes(-1, -2)
        wu = experts.v1.value.reshape(expert_count, intermediate, self.config.d_model).swapaxes(-1, -2)
        wd = experts.w2.value.reshape(expert_count, intermediate, self.config.d_model)
        out, router_logits = self.moe_call(
            hidden_state=x,
            gate_layer=gate_layer,
            expert_layer=self.experts,
            wi_kernel=wi,
            wu_kernel=wu,
            wd_kernel=wd,
            act_fn=experts.activation_fn,
            hooks=hooks,
        )
        router_logits = checkpoint_name(router_logits.reshape(x.shape[0], x.shape[1], -1), name="moe_router_logits")
        out = checkpoint_name(out, name="moe_expert_output")
        if not _config_uses_mpmd(self.config):
            out = apply_logical_sharding(
                out,
                dynamic_axes=common_types.HiddenStateSharding,
                partition_manager=self.config.runtime_sharding_resolver,
            )
        return cast(Array, out), router_logits


class DbrxBlock(spx.Module):
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
        rngs: spx.Rngs,
        layer_idx: int,
    ):
        """Initialize a DBRX decoder block.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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
        self.resid_dropout = nn.Dropout(rate=self.resid_pdrop, rngs=rngs)
        attn_block = DbrxNormAttentionNorm
        ffn_block = DbrxFFN
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

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES | None,  # type: ignore
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
        hidden_states = self.resid_dropout(hidden_states, deterministic=not self.training)
        hidden_states = decoder_output.residual_states + hidden_states

        return decoder_output.replace(
            hidden_states=hidden_states,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=DbrxConfig, model_type="dbrx")
class DbrxModel(EasyDeLBaseModule):
    """Databricks DBRX backbone (registered as ``BASE_MODULE``).

    Stack: token embedding -> ``n_layers`` :class:`DbrxBlock` blocks (each
    a norm-attention-norm + MoE FFN) -> final ``LayerNorm``. Attention
    uses MQA/GQA with ``kv_n_heads`` (default 1, multi-query), fused QKV,
    optional symmetric QKV clipping (``attn_config.clip_qkv``), and RoPE
    (``rope_theta``). Every layer's FFN is the fine-grained DBRX MoE:
    ``ffn_config.moe_num_experts`` (16 by default) routed via
    :class:`DbrxRouter` with top-``moe_top_k`` selection (4 by default),
    optional jitter, and Lp-normalised expert weights. Auxiliary
    load-balancing loss is gated by ``router_aux_loss_coef``. Returns
    :class:`MoeModelOutput`.

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
        norm_f (LayerNorm): Final layer normalization.
    """

    def __init__(
        self,
        config: DbrxConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: spx.Rngs,
    ):
        """Initialize the DBRX base model.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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
        self.emb_dropout = nn.Dropout(rate=self.emb_pdrop, rngs=rngs)

        with self.assign_layer_stage(0, total_layers=self.config.num_hidden_layers):
            self.wte = Embed(
                self.config.vocab_size,
                self.config.d_model,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        remat_layer_block = auto_remat(
            DbrxBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.blocks = nn.ModuleList([])
        for i in range(self.config.n_layers):
            with self.assign_layer_stage(i, total_layers=self.config.n_layers):
                self.blocks.append(
                    remat_layer_block(
                        config=config,
                        layer_idx=i,
                        dtype=dtype,
                        param_dtype=param_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
        with self.assign_layer_stage(self.config.num_hidden_layers - 1, total_layers=self.config.num_hidden_layers):
            self.norm_f = LayerNorm(
                self.config.hidden_size,
                epsilon=1e-5,
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

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
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
        inputs_embeds = self.emb_dropout(inputs_embeds)

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

        views = past_key_values.views if past_key_values is not None else None
        has_cache_views = views is not None and any(v is not None for v in views)
        needs_trace_cache = mode == common_types.MODE_DECODE or has_cache_views
        # Router-logit aggregation grows a Python tuple per layer, which
        # lax.scan cannot carry — collecting it forces the trace path too.
        trace_layers = self._layer_scan_trace(
            False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_views=views,
            extra=needs_trace_cache or bool(output_router_logits),
        )
        cache_views = views if trace_layers else None
        # Hoisted out of the loop body: the caching property must not be
        # first materialized inside the lax.scan trace (tracer leak).
        frequencies = self.frequencies

        def _layer_loop(block, carry):
            """Per-block body for the DBRX decoder ``scan``.

            Carry layout: ``(hidden_states, cv, all_hidden_states,
            all_attentions, all_router_logits, idx)``. Records the input
            hidden state when requested, runs one DBRX decoder block at the
            corresponding pipeline stage, optionally accumulates attention
            weights and the block's MoE router logits, updates the per-block
            KV cache slot, and returns the next carry.
            """
            hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx = carry
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            with self._layer_stage_context(idx, layers=self.blocks):
                outputs = block(
                    hidden_states=hidden_states,
                    mask_info=mask_info,
                    position_ids=position_ids,
                    mode=mode,
                    cache_view=self._layer_cache_view_at(cv, idx, enabled=trace_layers, cache=past_key_values),
                    cache_metadata=cache_metadata,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    frequencies=frequencies,
                )
            hidden_states = self._mark_layer_stage_boundary(outputs.hidden_states, idx, layers=self.blocks)

            if not _config_uses_mpmd(self.config):
                hidden_states = apply_logical_sharding(
                    hidden_states,
                    dynamic_axes=common_types.HiddenStateSharding,
                    partition_manager=self.config.runtime_sharding_resolver,
                )

            if output_attentions:
                all_attentions += (outputs.attention_weight,)
            if output_router_logits:
                all_router_logits += (outputs.router_logits,)
            cv = self._layer_cache_view_update(
                cv,
                idx,
                outputs.cache_view,
                enabled=trace_layers,
                cache=past_key_values,
            )
            return hidden_states, cv, all_hidden_states, all_attentions, all_router_logits, idx + 1

        hidden_states, _, all_hidden_states, all_attentions, all_router_logits, _ = self.blocks.scan(
            _layer_loop,
            (hidden_states, cache_views, all_hidden_states, all_attentions, all_router_logits, 0),
            trace=trace_layers,
            **self._layer_scan_kwargs("text"),
        )
        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            router_logits=all_router_logits,
            past_key_values=past_key_values,
        )

    def get_encoder(self) -> spx.Module:
        """
        Returns the encoder part of the model's graph definition.
        For DbrxModel (decoder-only), this is not applicable.
        """
        raise NotImplementedError("DbrxModel is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> spx.Module:
        """
        Returns the decoder part of the model's graph definition.
        For DbrxModel, this is the model itself.
        """
        return self

    def get_lm_head(self) -> spx.Module:
        """
        Returns the language model head of the module.
        DbrxModel does not include the lm_head.
        """
        raise NotImplementedError("DbrxModel does not include the language model head. See DbrxForCausalLM.")

    def get_embedding(self) -> spx.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=DbrxConfig, model_type="dbrx")
class DbrxForCausalLM(BaseCausalLMModule[DbrxModel, DbrxConfig]):  # type: ignore
    """DBRX backbone + ``vocab_size`` LM head for autoregressive generation.

    Wraps :class:`DbrxModel` with an unbiased linear ``lm_head``. Goes
    through the MoE-aware ``forward_moe`` path so per-layer router logits
    flow into the output and the auxiliary load-balancing loss
    (weighted by ``config.router_aux_loss_coef``) is folded into the main
    objective. Tied embeddings are *not* supported by DBRX
    (the config raises if ``tie_word_embeddings=True``).

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
        rngs: spx.Rngs,
    ):
        """Initialize DBRX model for causal language modeling.

        Args:
            config (DbrxConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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
            # The causal-LM router loss is weighted by 1.0 when the coefficient is explicitly ``None``.
            router_aux_loss_coef=(
                1.0 if getattr(config, "router_aux_loss_coef", None) is None else config.router_aux_loss_coef
            ),
        )

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
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
        )

    def _router_top_k(self) -> int:
        """DBRX keeps the routing width under ``ffn_config.moe_top_k``."""
        return self.config.ffn_config.moe_top_k


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=DbrxConfig, model_type="dbrx")
class DbrxForSequenceClassification(BaseSequenceClassificationModule[DbrxModel, DbrxConfig]):  # type: ignore
    """DBRX backbone + linear ``score`` head for sequence-level classification.

    Pools the last non-pad token's hidden state from :class:`DbrxModel` and
    feeds it through an unbiased ``score`` head with ``num_labels``
    outputs. ``config.pad_token_id`` is required for batched inputs;
    auxiliary router load-balancing loss is still threaded through during
    fine-tuning to keep the experts balanced.

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
        rngs: spx.Rngs,
    ):
        """Initialize DBRX model for sequence classification.

        Args:
            config (DbrxConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (spx.Rngs): Random number generator state.
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

    def forward(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
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

        if self.config.pad_token_id is None and attention_mask is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if input_ids is None and attention_mask is None:
            pooled_logits = logits[:, -1]
        else:
            pooled_logits = self.pool_sequence(logits, input_ids=input_ids, attention_mask=attention_mask)

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
