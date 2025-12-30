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


import typing
from functools import partial

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
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import auto_remat
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
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear
from easydel.layers.moe import (
    BaseMoeModule,
    ColumnParallelMoELinear,
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RowParallelMoELinear,
)
from easydel.layers.norms import RMSNorm as RMSNorm

from .qwen2_moe_configuration import Qwen2MoeConfig


class Qwen2MoeMLPStack(nn.Module):
    """Qwen2Moe MoE MLP using the new ParallelMoELinear layers."""

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {"name": "gate_proj.kernel", "spliter": lambda x: x[..., : x.shape[-1] // 2]},
                {"name": "up_proj.kernel", "spliter": lambda x: x[..., x.shape[-1] // 2 :]},
            ],
            "inverse_spliter": lambda torch, gate, up: torch.stack((gate, up), dim=-1).flatten(-2),
        },
        "down_proj$": {
            "splits": [
                {"name": "down_proj.kernel", "spliter": lambda x: x},
            ],
            "inverse_spliter": lambda x: x,
        },
    }

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int | None = None,
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
        self.act_fn = nn.silu

    def __call__(self, x: Array, group_sizes: Array, sorted_experts: Array | None = None) -> Array:
        """Forward pass through MoE MLP stack.

        Args:
            x (Array): Input tensor of shape (total_tokens, hidden_size) where total_tokens
                is the sum of all tokens assigned to each expert.
            group_sizes (Array): Array of shape (num_experts,) containing the number of tokens
                assigned to each expert.
            sorted_experts (Array | None): Optional array containing expert indices in sorted order.
                Used for efficient expert batching.

        Returns:
            Array: Output tensor of shape (total_tokens, hidden_size) after MoE MLP transformation.
        """
        gate = self.gate_proj(x, group_sizes, sorted_experts)
        up = self.up_proj(x, group_sizes, sorted_experts)
        return self.down_proj(self.act_fn(gate) * up, group_sizes, sorted_experts)


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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.down_proj = row_parallel_linear(intermediate_size, config.hidden_size, rngs=rngs)
        self.up_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.act_fn = nn.silu

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
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
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen2MoeAttention(UnifiedAttention):
    """Qwen2 MoE Attention module with sliding window support.

    Inherits from UnifiedAttention with Qwen2Moe-specific customizations:
    - Sliding window attention
    - Custom bias configuration (Q/K/V use qkv_bias, O doesn't)
    - Attention dropout
    """

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initializes the Qwen2MoeAttention module.

        Args:
            config (Qwen2MoeConfig): The configuration object for the model.
            dtype (jnp.dtype): Data type for computations (default: jnp.float32).
            param_dtype (jnp.dtype): Data type for parameters (default: jnp.float32).
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations (default: None).
            rngs (nn.Rngs): Random number generators.
        """

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if config.use_sliding_window else None,
        )

    def _create_q_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use qkv_bias for query projection (Qwen2Moe-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )

    def _create_k_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use qkv_bias for key projection (Qwen2Moe-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )

    def _create_v_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use qkv_bias for value projection (Qwen2Moe-specific)."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            rngs=rngs,
            use_bias=config.qkv_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )

    def _create_o_proj(self, config, dtype, param_dtype, precision, rngs):
        """Override to use bias=False for output projection (Qwen2Moe-specific)."""
        from easydel.layers.linear import RowParallelLinear

        return RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )

    def _create_rotary(self, config: Qwen2MoeConfig, dtype: jnp.dtype):
        """Create Qwen2Moe-specific rotary embedding layer."""
        return config.get_basic_rope(
            head_size=config.hidden_size // config.num_attention_heads,
            rotary_dim=config.hidden_size // config.num_attention_heads,
            base=config.rope_theta,
            dtype=dtype,
        )

    def _create_attention_performer(self, config: Qwen2MoeConfig, rngs: nn.Rngs):
        """Create attention performer with Qwen2Moe's attention dropout."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
            dropout_prob=config.attention_dropout,
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
        self.shared_expert_gate = ColumnParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.moe_hooks = MoeFusedHooks()

    def __call__(self, hidden_states: Array) -> tuple[Array, Array]:
        """Forward pass of the Sparse MoE block.

        Args:
            hidden_states (Array): Input hidden states of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            tuple[Array, Array]: A tuple containing:
                - final_hidden_states (Array): Output hidden states of shape (batch_size, sequence_length, hidden_dim)
                  combining routed expert outputs and shared expert output.
                - router_logits (Array): Router logits of shape (batch_size * sequence_length, num_experts)
                  representing the gating scores for each expert.
        """
        B, S, H = hidden_states.shape

        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
            output_metrics=False,
        )

        hs_flat = hidden_states.reshape(-1, H)
        shared_out = self.shared_expert(hs_flat)
        shared_gate = jax.nn.sigmoid(self.shared_expert_gate(hs_flat))
        shared_out = shared_gate * shared_out
        shared_out = shared_out.reshape(B, S, H)
        out = out + shared_out

        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


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
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass of the decoder layer.

        Args:
            hidden_states (Float[Array, "batch seq_len hidden_dim"]): Input hidden states of shape
                (batch_size, sequence_length, hidden_size).
            mask_info (MaskInfo): Attention mask information containing causal mask and padding mask details.
            position_ids (Int[Array, "batch seq_len"]): Position indices for each token of shape
                (batch_size, sequence_length).
            mode (common_types.RUNTIME_MODE_TYPES): Runtime mode (train, decode, or prefill) controlling
                attention implementation.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None): View into the KV cache for
                efficient inference. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None):
                Metadata for paged attention operations. Defaults to None.
            output_attentions (bool): Whether to return attention weights. Defaults to False.
            output_router_logits (bool): Whether to return router logits from MoE layers. Defaults to False.
            frequencies (Float[Array, "seq_len head_dim"] | None): Precomputed RoPE frequencies. If None,
                computed on-the-fly. Defaults to None.

        Returns:
            DecoderLayerOutput: Named tuple containing:
                - hidden_states (Array): Output hidden states of shape (batch_size, sequence_length, hidden_size).
                - attention_weight (Array | None): Attention weights of shape
                  (batch_size, num_heads, sequence_length, kv_sequence_length) if output_attentions=True.
                - router_logits (Array | None): Router logits of shape (batch_size * sequence_length, num_experts)
                  if output_router_logits=True and this is an MoE layer.
                - cache_view (TransformerCacheView | RaggedPagesCacheView | None): Updated cache view.
        """
        attn_outputs = self.self_attn(
            self.input_layernorm(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        mlp_out = self.mlp(feed_forward_input)

        if self.config.num_experts > 0:
            feed_forward_hidden_states, router_logits = mlp_out
        else:
            feed_forward_hidden_states = mlp_out
            router_logits = None

        hidden_states = checkpoint_name(hidden_states + feed_forward_hidden_states, "residual")

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
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

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
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
    ) -> MoeModelOutput:
        """Forward pass of the Qwen2 MoE model.

        Args:
            input_ids (Int[Array, "batch seq_len"] | None): Input token IDs of shape (batch_size, sequence_length).
                Mutually exclusive with `inputs_embeds`. Defaults to None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Mutually exclusive with `input_ids`. Defaults to None.
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask of shape (batch_size, sequence_length)
                indicating which tokens should be attended to (1) and which should be ignored (0). Defaults to None.
            mask_info (MaskInfo | None): Precomputed mask information. If provided, overrides `attention_mask`.
                Defaults to None.
            position_ids (Int[Array, "batch seq_len"] | None): Position indices of shape (batch_size, sequence_length).
                If None, automatically generated as consecutive integers. Defaults to None.
            output_attentions (bool | None): Whether to return attention weights for all layers. If None, uses
                config.output_attentions. Defaults to None.
            output_hidden_states (bool | None): Whether to return hidden states for all layers. If None, uses
                config.output_hidden_states. Defaults to None.
            output_router_logits (bool | None): Whether to return router logits from MoE layers. If None, uses
                config.output_router_logits. Defaults to None.
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode controlling attention implementation.
                Automatically inferred if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): Cached key/value states
                from previous forward passes for efficient autoregressive generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None): Metadata for
                paged attention operations. Defaults to None.
            apply_lm_head (bool): Whether to apply the language model head (unused in base model). Defaults to True.

        Returns:
            MoeModelOutput: Named tuple containing:
                - last_hidden_state (Array): Final hidden states of shape (batch_size, sequence_length, hidden_size).
                - hidden_states (tuple[Array, ...] | None): Hidden states from all layers if output_hidden_states=True.
                - attentions (tuple[Array, ...] | None): Attention weights from all layers if output_attentions=True.
                - router_logits (tuple[Array, ...] | None): Router logits from all MoE layers if output_router_logits=True.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated KV cache.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are provided or if neither is provided.
        """
        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")
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
        sequence_length = inputs_embeds.shape[1]
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
                all_self_attns += (layer_outputs.attention_weight,)

            if output_router_logits:
                all_router_logits += (layer_outputs.router_logits,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

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
class Qwen2MoeForCausalLM(BaseCausalLMModule[Qwen2MoeModel, Qwen2MoeConfig]):
    """Qwen2 MoE model with a Causal Language Modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "qwen2_moe"
    _config_class = Qwen2MoeConfig

    def __init__(
        self,
        config: Qwen2MoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=Qwen2MoeModel,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass of the Qwen2MoeForCausalLM model.

        Args:
            input_ids (Int[Array, "batch seq_len"] | None): Input token IDs of shape (batch_size, sequence_length).
                Mutually exclusive with `inputs_embeds`. Defaults to None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Mutually exclusive with `input_ids`. Defaults to None.
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask of shape (batch_size, sequence_length).
                Defaults to None.
            mask_info (MaskInfo | None): Precomputed mask information. Defaults to None.
            position_ids (Int[Array, "batch seq_len"] | None): Position indices of shape (batch_size, sequence_length).
                Defaults to None.
            output_attentions (bool | None): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None): Whether to return hidden states. Defaults to None.
            output_router_logits (bool | None): Whether to return router logits from MoE layers. Defaults to None.
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): Cached KV states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None): Cache metadata.
                Defaults to None.
            apply_lm_head (bool): Whether to apply language modeling head to compute logits. Defaults to True.

        Returns:
            MoeCausalLMOutput: Named tuple containing:
                - logits (Array): Language modeling logits of shape (batch_size, sequence_length, vocab_size).
                - aux_loss (Array | None): Auxiliary load balancing loss for MoE routing.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated KV cache.
                - hidden_states (tuple[Array, ...] | None): All hidden states if output_hidden_states=True.
                - attentions (tuple[Array, ...] | None): All attention weights if output_attentions=True.
                - router_logits (tuple[Array, ...] | None): All router logits if output_router_logits=True.
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
        """Compute auxiliary loss from router logits."""
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Qwen2MoeConfig, model_type="qwen2_moe")
class Qwen2MoeForSequenceClassification(BaseSequenceClassificationModule[Qwen2MoeModel, Qwen2MoeConfig]):
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

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "qwen2_moe"
    _config_class = Qwen2MoeConfig

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
            base_model_class=Qwen2MoeModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy="last",
            score_head_bias=False,
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
    ) -> SequenceClassifierOutput:
        """Forward pass of the Qwen2 MoE model for sequence classification.

        Args:
            input_ids (Int[Array, "batch seq_len"] | None): Input token IDs of shape (batch_size, sequence_length).
                Mutually exclusive with `inputs_embeds`. Defaults to None.
            inputs_embeds (Float[Array, "batch seq_len hidden_dim"] | None): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Mutually exclusive with `input_ids`. Defaults to None.
            attention_mask (Bool[Array, "batch seq_len"] | None): Attention mask of shape (batch_size, sequence_length).
                Defaults to None.
            mask_info (MaskInfo | None): Precomputed mask information. Defaults to None.
            position_ids (Int[Array, "batch seq_len"] | None): Position indices of shape (batch_size, sequence_length).
                Defaults to None.
            mode (common_types.RUNTIME_MODE_TYPES | None): Runtime mode. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None): Cached KV states (ignored
                in classification). Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None): Cache metadata
                (ignored in classification). Defaults to None.
            apply_lm_head (bool): Unused in classification. Defaults to True.
            output_attentions (bool | None): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None): Whether to return hidden states. Defaults to None.

        Returns:
            SequenceClassifierOutput: Named tuple containing:
                - logits (Array): Classification logits of shape (batch_size, num_labels) pooled from the last token.
                - past_key_values (TransformerCache | RaggedPagesCache | HybridCache): Updated KV cache.
                - hidden_states (tuple[Array, ...] | None): All hidden states if output_hidden_states=True.
                - attentions (tuple[Array, ...] | None): All attention weights if output_attentions=True.
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            inputs_embeds=inputs_embeds,
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

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.score
