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


import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
from easydel.layers.caching import (
    HybridCache,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheConfig,
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
from easydel.layers.rotary_embedding import yarn_get_mscale

from .xerxes2_configuration import Xerxes2Config as Xerxes2Config

logger = get_logger(__name__)


class Xerxes2Attention(UnifiedAttention):
    """Xerxes2 Multi-head Latent Attention (MLA) layer.

    Implements Multi-head Latent Attention from DeepSeek-V2 architecture with compressed
    key-value representations using LoRA-style low-rank projections and separate nope/rope
    dimensions for efficient long-context processing.

    Features:
        - Compressed KV representation with low-rank LoRA projections
        - Separate nope (non-positional) and rope (rotary) head dimensions
        - Optional query LoRA compression for reduced memory footprint
        - YaRN-based attention scaling for extended context lengths
        - Grouped-query attention support via shared KV projections
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
        "output_projection": "o_proj",
    }

    def __init__(
        self,
        config: Xerxes2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 Multi-head Latent Attention layer.

        Args:
            config (Xerxes2Config): Model configuration with MLA parameters including
                qk_nope_head_dim, qk_rope_head_dim, vhead_dim, and kv_lora_dim.
            layer_idx (int): Index of this layer in the model.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        # Set MLA-specific dimensions before calling super().__init__()
        # so they're available in define_network
        self.config = config
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.vhead_dim
        self.kv_lora_rank = config.kv_lora_dim

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="mla",
            causal=True,
            use_mla_lora=config.q_lora_dim is not None,
        )

        # Override head_dim for MLA - use value head dimension for output merging
        self.head_dim = self.v_head_dim

    def define_network(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision,
        rngs: nn.Rngs,
    ):
        """Define MLA-specific network structure for Xerxes2.

        Creates the projection layers for Multi-head Latent Attention, including
        optional LoRA compression for queries and mandatory LoRA compression for
        key-value pairs.

        Args:
            config (Xerxes2Config): Model configuration.
            dtype (jnp.dtype): Data type for computation.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.Precision): Numerical precision for operations.
            rngs (nn.Rngs): Random number generator state.
        """

        if not self.use_mla_lora:
            setattr(
                self,
                self.projection_mapping["mla_q_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
        else:
            setattr(
                self,
                self.projection_mapping["mla_q_a_proj"],
                ColumnParallelLinear(
                    config.hidden_size,
                    config.q_lora_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_a_layernorm"],
                nn.LayerNorm(
                    config.q_lora_dim,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                ),
            )
            setattr(
                self,
                self.projection_mapping["mla_q_b_proj"],
                ColumnParallelLinear(
                    config.q_lora_dim,
                    config.num_attention_heads * self.q_head_dim,
                    rngs=rngs,
                    use_bias=False,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    kernel_init=jax.nn.initializers.normal(config.initializer_range),
                    precision=precision,
                ),
            )

        setattr(
            self,
            self.projection_mapping["mla_kv_a_proj_with_mqa"],
            ColumnParallelLinear(
                config.hidden_size,
                config.kv_lora_dim + config.qk_rope_head_dim,
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_a_layernorm"],
            nn.LayerNorm(
                config.kv_lora_dim,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            ),
        )
        setattr(
            self,
            self.projection_mapping["mla_kv_b_proj"],
            ColumnParallelLinear(
                config.kv_lora_dim,
                config.num_attention_heads * (config.qk_nope_head_dim + config.vhead_dim),
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )
        setattr(
            self,
            self.projection_mapping["output_projection"],
            RowParallelLinear(
                config.num_attention_heads * self.v_head_dim,
                config.hidden_size,
                rngs=rngs,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
                precision=precision,
            ),
        )
        self.rotary = self._create_rotary(config, dtype)
        self.attention_performer = self._create_attention_performer(config, rngs)

    def _create_attention_performer(self, config, rngs):
        """Create attention performer module with YaRN scaling.

        Args:
            config: Model configuration with rope_scaling parameters.
            rngs: Random number generator state.

        Returns:
            FlexibleAttentionModule: Configured attention performer with optional
                YaRN-based mscale for extended context attention.
        """
        softmax_scale = self.q_head_dim**-0.5
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=softmax_scale,
            dropout_prob=0.0,
        )


class Xerxes2MLP(nn.Module):
    """Multi-Layer Perceptron module for dense Xerxes2 decoder layers.

    Implements a gated feedforward network with SiLU activation using fused gate/up
    projections for efficient computation. Used in non-MoE decoder layers.

    Features:
        - Fused gate_up projection for memory efficiency
        - SiLU (Swish) gating activation
        - Column/Row parallel linear layers for tensor parallelism
    """

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 MLP block.

        Args:
            config (Xerxes2Config): Model configuration with MLP parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.act = nn.silu
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            rngs=rngs,
        )
        self.gate_up_proj = column_parallel_linear(config.hidden_size, 2 * config.intermediate_size, rngs=rngs)
        self.down_proj = row_parallel_linear(config.intermediate_size, config.hidden_size, rngs=rngs)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply gated feedforward transformation with fused projections.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim]
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = jnp.split(up_states, 2, axis=-1)
        hidden_states = checkpoint_name(self.down_proj(up_states * nn.silu(gate)), "mlp_output")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class Xerxes2MoeMLPStack(nn.Module):
    """Mixture-of-Experts MLP stack using parallel MoE linear layers.

    Implements the expert feedforward networks for the MoE block with efficient
    parallel computation across all experts simultaneously.

    Features:
        - Parallel gate/up/down projections across all experts
        - Expert tensor mode for efficient batched computation
        - Configurable hidden activation function
    """

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 MoE MLP stack.

        Args:
            config (Xerxes2Config): Model configuration with MoE parameters including
                num_experts and moe_intermediate_size.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
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
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply gated MoE feedforward transformation.

        Processes routed tokens through the parallel expert MLPs with gated activation.

        Args:
            hidden_states: Input tensor for routed tokens.
            group_sizes: Number of tokens assigned to each expert.
            sorted_experts: Optional sorted expert indices for efficient routing.

        Returns:
            Transformed hidden states after expert processing.
        """
        return checkpoint_name(
            self.down_proj(
                self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
                * self.up_proj(hidden_states, group_sizes, sorted_experts),
                group_sizes,
                sorted_experts,
            ),
            "moe_output",
        )


class Xerxes2MoeSparseBlock(BaseMoeModule):
    """Sparse Mixture-of-Experts block for Xerxes2 models.

    Implements a top-k routing mechanism where each token is processed by
    a subset of expert MLPs, enabling parameter-efficient scaling. Uses
    configurable routing and load balancing strategies.

    Features:
        - Top-k expert selection with configurable routing strategy
        - Optional probability normalization via norm_topk_prob
        - Standard load balancing for training stability
        - Efficient parallel expert computation via MLP stack

    Attributes:
        config (Xerxes2Config): Configuration object for the model.
        gate (ColumnParallelLinear): Linear layer for the gating/routing network.
        experts (Xerxes2MoeMLPStack): Parallel expert MLP modules.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for matrix multiplications.
    """

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 Sparse MoE block.

        Args:
            config (Xerxes2Config): Model configuration with MoE parameters including
                num_experts and num_experts_per_tok.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Xerxes2MoeMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Apply sparse MoE transformation with top-k expert routing.

        Routes each token to a subset of experts based on gating scores,
        processes through the selected experts, and combines outputs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            tuple: A tuple containing:
                - final_hidden_states: Transformed hidden states [batch, seq_len, hidden_dim]
                - router_logits: Router logits for load balancing loss [batch, seq_len, num_experts]
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Xerxes2DecoderLayer(nn.Module):
    """Single decoder layer for Xerxes2 models.

    Combines Multi-head Latent Attention with either dense MLP or sparse MoE
    feedforward networks. Uses pre-norm architecture with RMS normalization
    and residual connections.

    Features:
        - Multi-head Latent Attention (MLA) with compressed KV
        - Layer-specific MoE vs dense MLP selection based on decoder_sparse_step
        - Pre/post normalization for both attention and feedforward blocks
        - Gradient checkpointing support for memory efficiency
    """

    def __init__(
        self,
        config: Xerxes2Config,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 decoder layer.

        Args:
            config (Xerxes2Config): Model configuration.
            layer_idx (int): Index of this layer in the model, used to determine MoE vs dense.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        attn_block, mlp_block, moe_block = auto_remat(
            Xerxes2Attention,
            Xerxes2MLP,
            Xerxes2MoeSparseBlock,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.self_attn = attn_block(
            config=self.config,
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
        rms = functools.partial(
            RMSNorm,
            dim=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.input_layernorm = rms()
        self.post_attention_layernorm = rms()
        self.pre_feedforward_layernorm = rms()
        self.post_feedforward_layernorm = rms()

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        frequencies: tuple[Array, Array],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture with MLA attention followed by
        either dense MLP or sparse MoE feedforward, with residual connections.

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            frequencies (tuple[Array, Array]): Precomputed RoPE frequencies (cos, sin).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view
                for key-value caching during generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata for cache management. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return MoE router logits. Defaults to False.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, router logits, and cache view.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        attn_outputs = self.self_attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            None,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        router_logits = None
        if self.is_moe:
            hidden_states, router_logits = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
            router_logits=router_logits if output_router_logits else None,
        )


@register_module(TaskType.BASE_MODULE, config=Xerxes2Config, model_type="xerxes2")
class Xerxes2Model(EasyDeLBaseModule):
    """Xerxes2 base model implementation.

    Implements the Xerxes2 decoder-only transformer architecture with Multi-head
    Latent Attention (MLA), hybrid dense/MoE feedforward layers, and RMSNorm.
    Supports efficient long-context processing via compressed KV representations.

    Features:
        - Multi-head Latent Attention with LoRA-compressed KV states
        - Hybrid architecture with configurable dense/MoE layer patterns
        - YaRN-based rotary position embeddings for extended contexts
        - Gradient checkpointing for memory-efficient training

    Attributes:
        config (Xerxes2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Xerxes2 base model.

        Args:
            config (Xerxes2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
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
            Xerxes2DecoderLayer(
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
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
        )

    @functools.cached_property
    def frequencies(self) -> jnp.ndarray:
        """Compute and cache rotary position embedding frequencies.

        Returns:
            jnp.ndarray: Precomputed RoPE frequencies for the rope head dimension.
        """
        return self.config.get_basic_frequencies(self.config.qk_rope_head_dim)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the Xerxes2 base model.

        Processes input tokens through embedding, all decoder layers with MLA attention
        and hybrid dense/MoE feedforward, and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return MoE router logits from all layers.
                Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                past_key_values, and optional router_logits.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs=input_ids.astype("i4"))

        sequence_length = inputs_embeds.shape[1]

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )
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
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
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


class Xerxes2ForCausalLM(BaseCausalLMModule[Xerxes2Model, Xerxes2Config]):
    """Xerxes2 model with a language modeling head for causal language modeling tasks.

    Extends the base Xerxes2Model by adding a linear language modeling head for
    autoregressive text generation. Supports hybrid dense/MoE architecture with
    auxiliary load balancing loss for stable MoE training.

    Features:
        - Multi-head Latent Attention for efficient KV caching
        - Hybrid dense/MoE feedforward with configurable layer patterns
        - Auxiliary load balancing loss for MoE expert utilization
        - YaRN-based position embeddings for extended context support

    Attributes:
        config (Xerxes2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "xerxes2"
    _config_class = Xerxes2Config

    def __init__(
        self,
        config: Xerxes2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the Xerxes2ForCausalLM model.

        Args:
            config (Xerxes2Config): The model configuration.
            dtype (jnp.dtype, optional): The data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): The data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): The precision to use for matrix multiplication.
                Defaults to None.
            rngs (nn.Rngs): The random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=Xerxes2Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> MoeCausalLMOutput:
        """Forward pass through the Xerxes2 causal language model.

        Processes input through the base model and applies the language modeling head
        with optional MoE auxiliary loss computation.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens,
                shape (batch_size, sequence_length). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head. Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            output_router_logits (bool | None, optional): Whether to return MoE router logits from all layers.
                Defaults to None.

        Returns:
            MoeCausalLMOutput: Contains logits, hidden_states, last_hidden_state, attentions,
                past_key_values, router_logits, and auxiliary loss.
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
        """Compute auxiliary load balancing loss for MoE layers.

        Calculates the auxiliary loss that encourages balanced expert utilization
        across all MoE layers, scaled by the router_aux_loss_coef.

        Args:
            outputs: Model outputs containing router_logits from MoE layers.
            attention_mask: Attention mask to exclude padding tokens from loss computation.

        Returns:
            Auxiliary loss value if router_logits are available, None otherwise.
        """
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None

        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )

        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)

    def create_transformer_cache_config(self, batch_size: int, max_length: int):
        """Create cache configuration for MLA-based key-value caching.

        Creates a TransformerCacheConfig with dimensions appropriate for Multi-head
        Latent Attention, using separate key and value dimensions based on the
        nope/rope head dimensions.

        Args:
            batch_size: Number of sequences in the batch.
            max_length: Maximum sequence length for the cache.

        Returns:
            TransformerCacheConfig: Configuration for initializing the KV cache
                with MLA-specific key_dim (qk_rope + qk_nope) and value_dim (vhead_dim).
        """
        head_dim = getattr(self.config, "head_dim", None)
        if head_dim is None:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_key_value_heads = getattr(self.config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = self.config.num_attention_heads
        return TransformerCacheConfig.create(
            num_hidden_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            sequence_length=max_length,
            num_heads=self.config.num_attention_heads,
            key_dim=self.config.qk_rope_head_dim + self.config.qk_nope_head_dim,
            value_dim=self.config.vhead_dim,
        )
