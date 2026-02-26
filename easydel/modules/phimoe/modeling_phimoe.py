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


import functools

import jax.lax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

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
from easydel.infra.modeling_outputs import BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers import RMSNorm as RMSNorm
from easydel.layers.attention import UnifiedAttention
from easydel.modules._base import BaseCausalLMModule

from .phimoe_configuration import PhiMoeConfig


class PhiMoEBlockSparseTop2MLP(nn.Module):
    """Expert MLP module for PhiMoE Sparse Mixture of Experts.

    Implements the feedforward network used by each expert in the PhiMoE model's
    Sparse MoE layer. Uses SwiGLU activation with gated projections (w1, w3)
    and a down projection (w2) for efficient expert computation.
    """

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize PhiMoE expert MLP block.

        Args:
            config (PhiMoeConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        column_parallel_linear = functools.partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = functools.partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        ffn_dim = config.intermediate_size
        hidden_dim = config.hidden_size

        self.w1 = column_parallel_linear(hidden_dim, ffn_dim, rngs=rngs)
        self.w2 = row_parallel_linear(ffn_dim, hidden_dim, rngs=rngs)
        self.w3 = column_parallel_linear(hidden_dim, ffn_dim, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        """Apply SwiGLU feedforward transformation for expert processing.

        Args:
            hidden_states (Array): Input tensor for the expert MLP of shape
                (num_tokens_routed_to_expert, hidden_size).

        Returns:
            Array: Transformed hidden states after expert MLP processing with shape
                (num_tokens_routed_to_expert, hidden_size).
        """
        gate = checkpoint_name(self.act_fn(self.w1(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.w3(hidden_states), "mlp_up")
        down = checkpoint_name(self.w2(gate * up), "mlp_down")
        return checkpoint_name(down, "mlp_output")


class PhiMoEAttention(UnifiedAttention):
    """PhiMoE Attention module with sliding window support.

    Inherits from UnifiedAttention with PhiMoE-specific customizations:
    - Sliding window attention support for efficient long-context modeling
    - Grouped Query Attention (GQA) for memory efficiency
    - Rotary Position Embeddings (RoPE) for position encoding
    """

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize PhiMoE attention with sliding window configuration.

        Args:
            config (PhiMoeConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        # PhiMoE router decisions are highly sensitive to attention noise.
        # Keep attention computations in model dtype (fp32 in tests) to align
        # routing with HF and avoid cascading top-k expert mismatches.
        if getattr(config, "attn_dtype", None) is not None:
            config.attn_dtype = dtype
        if getattr(config, "attn_softmax_dtype", None) is not None:
            config.attn_softmax_dtype = dtype

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window,
        )


class PhiMoeSparseMoeBlock(nn.Module):
    """Sparse Mixture of Experts block for PhiMoE models.

    Implements the sparse MoE layer that routes tokens to a subset of expert MLPs.
    Uses SparseMixer-style top-k routing with optional jitter noise for training
    stability and load balancing across experts.
    """

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize PhiMoE Sparse MoE block.

        Args:
            config (PhiMoeConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise
        self.gate = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.num_local_experts,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nn.initializers.normal(),
        )

        self.experts = nn.List(
            [
                PhiMoEBlockSparseTop2MLP(
                    config=config,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for _ in range(self.config.num_local_experts)
            ]
        )

    def __call__(
        self,
        hidden_states: Array,
        deterministic: bool = True,
    ) -> tuple[Array, Array]:
        """Forward pass through the Sparse MoE block.

        Routes input tokens to selected experts and combines their outputs using
        SparseMixer-style routing for deterministic mode or standard top-k softmax
        for non-deterministic (training) mode.

        Args:
            hidden_states (Array): Input hidden states of shape (batch, seq_len, hidden_dim).
            deterministic (bool, optional): If True, uses SparseMixer routing with jitter
                thresholds. If False, uses standard top-k with softmax. Defaults to True.

        Returns:
            tuple[Array, Array]: A tuple containing:
                - Output hidden states after MoE processing (batch, seq_len, hidden_dim).
                - Router logits for auxiliary loss computation (batch, seq_len, num_experts).
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self.gate(hidden_states).astype(  # no reshaping is needed
            jnp.promote_types(self.dtype, jnp.float32)
        )

        def _sparsemixer_eval(scores: Array, jitter_eps: float) -> tuple[Array, Array]:
            tokens = scores.shape[0]
            max_vals = jnp.max(scores, axis=-1, keepdims=True)
            max_ind = jnp.argmax(scores, axis=-1, keepdims=True)

            factor = jnp.maximum(jnp.abs(scores), max_vals)
            threshold_mask = ((max_vals - scores) / factor) > (2 * jitter_eps)
            masked_gates = jnp.where(threshold_mask, -jnp.inf, scores)
            masked_probs = jax.nn.softmax(masked_gates, axis=-1)
            multiplier_1 = jnp.take_along_axis(masked_probs, max_ind, axis=-1)

            masked_scores = scores.at[jnp.arange(tokens), max_ind.squeeze(-1)].set(-jnp.inf)
            max_vals_2 = jnp.max(masked_scores, axis=-1, keepdims=True)
            max_ind_2 = jnp.argmax(masked_scores, axis=-1, keepdims=True)

            factor_2 = jnp.maximum(jnp.abs(scores), max_vals_2)
            threshold_mask_2 = ((max_vals_2 - scores) / factor_2) > (2 * jitter_eps)
            masked_gates_2 = jnp.where(threshold_mask_2, -jnp.inf, masked_scores)
            masked_probs_2 = jax.nn.softmax(masked_gates_2, axis=-1)
            multiplier_2 = jnp.take_along_axis(masked_probs_2, max_ind_2, axis=-1)

            multiplier = jnp.concatenate([multiplier_1, multiplier_2], axis=-1)
            selected_experts = jnp.concatenate([max_ind, max_ind_2], axis=-1).astype(jnp.int32)
            return multiplier, selected_experts

        if deterministic:
            routing_weights, selected_experts = _sparsemixer_eval(
                router_logits,
                jitter_eps=float(self.router_jitter_noise),
            )
        else:
            routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
            routing_weights = jax.nn.softmax(
                routing_weights.astype(jnp.promote_types(self.dtype, jnp.float32)),
                axis=-1,
            )
        # HF compatibility: Phimoe router returns a dense expert-weight matrix
        # by scattering top-k multipliers into expert columns. The current HF
        # expert block then indexes this dense matrix by top-k *position* (0/1)
        # rather than expert id. Reproduce this behavior for strict parity.
        routing_weights_dense = jnp.zeros(
            (routing_weights.shape[0], self.config.num_local_experts),
            dtype=routing_weights.dtype,
        )
        token_indices = jnp.arange(routing_weights.shape[0])
        for topk_pos in range(selected_experts.shape[-1]):
            routing_weights_dense = routing_weights_dense.at[
                token_indices,
                selected_experts[:, topk_pos],
            ].set(routing_weights[:, topk_pos])

        final_hidden_state = jnp.zeros_like(hidden_states)
        for index in range(self.config.num_local_experts):
            expert_layer_output = (
                block_wise_ffn(
                    self.experts[index],
                    hidden_states,
                    self.config.scan_mlp_chunk_size,
                )
                if self.config.use_scan_mlp
                else self.experts[index](hidden_states)
            )
            expert_weight = jnp.zeros((hidden_states.shape[0],), dtype=routing_weights.dtype)
            for topk_pos in range(selected_experts.shape[-1]):
                expert_weight = expert_weight + jnp.where(
                    selected_experts[:, topk_pos] == index,
                    routing_weights_dense[:, topk_pos],
                    0.0,
                )
            expert_layer_output_exp = expert_layer_output * expert_weight[:, None]
            final_hidden_state += checkpoint_name(expert_layer_output_exp, "moe_expert_output")
        final_hidden_state = final_hidden_state.reshape(batch_size, sequence_length, hidden_dim)
        router_logits = router_logits.reshape(batch_size, sequence_length, -1)
        return final_hidden_state, checkpoint_name(router_logits, "moe_router_logits")


class PhiMoeDecoderLayer(nn.Module):
    """Single decoder layer for PhiMoE models.

    Combines multi-head attention with sliding window and Sparse MoE feedforward
    networks, using LayerNorm normalization and residual connections.
    """

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize PhiMoE decoder layer.

        Args:
            config (PhiMoeConfig): Model configuration with layer parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        gate_up_splits = []
        down_splits = []
        for expert_idx in range(config.num_local_experts):
            gate_up_splits.append(
                {
                    "name": f"block_sparse_moe.experts.{expert_idx}.w1.kernel",
                    "spliter": (lambda x, idx=expert_idx: x[idx, : x.shape[1] // 2, :].swapaxes(-1, -2)),
                }
            )
            gate_up_splits.append(
                {
                    "name": f"block_sparse_moe.experts.{expert_idx}.w3.kernel",
                    "spliter": (lambda x, idx=expert_idx: x[idx, x.shape[1] // 2 :, :].swapaxes(-1, -2)),
                }
            )
            down_splits.append(
                {
                    "name": f"block_sparse_moe.experts.{expert_idx}.w2.kernel",
                    "spliter": (lambda x, idx=expert_idx: x[idx].swapaxes(-1, -2)),
                }
            )

        # Accept HF PhiMoE MoE naming (`mlp.*`) directly during state-dict conversion.
        # HF stores expert projections as consolidated tensors over experts:
        # - mlp.experts.gate_up_proj: [num_experts, 2 * intermediate, hidden]
        # - mlp.experts.down_proj:    [num_experts, hidden, intermediate]
        # while EasyDeL stores per-expert w1/w2/w3 modules.
        self.reform_param = {
            "mlp.router.weight$": {
                "splits": [{"name": "block_sparse_moe.gate.kernel", "spliter": lambda x: x.swapaxes(-1, -2)}]
            },
            "mlp.experts.gate_up_proj$": {"splits": gate_up_splits},
            "mlp.experts.down_proj$": {"splits": down_splits},
        }

        attn_block = PhiMoEAttention
        mlp_block = PhiMoeSparseMoeBlock
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
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
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
    ):
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture: x + attn(norm(x)) followed by x + moe(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal and sliding window masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

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
        )
        hidden_states, self_attn_weights = (
            attn_outputs.attention_output,
            attn_outputs.attention_weight,
        )

        hidden_states = checkpoint_name(residual + hidden_states, "residual")

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = checkpoint_name(hidden_states, "layer_output")

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=self_attn_weights if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=PhiMoeConfig, model_type="phimoe")
class PhiMoeModel(EasyDeLBaseModule):
    """The base PhiMoE model transformer.

    This class represents the core transformer architecture of the PhiMoE model,
    consisting of an embedding layer, multiple PhiMoeDecoderLayer layers (with sparse MoE),
    and a final layer normalization. Combines Phi architecture with sparse
    Mixture of Experts for efficient scaling.

    Attributes:
        config (PhiMoeConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computation.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize PhiMoE base model.

        Args:
            config (PhiMoeConfig): Model configuration.
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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.List(
            [
                PhiMoeDecoderLayer(
                    config=config,
                    layer_idx=idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for idx in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            dim=config.hidden_size,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the PhiMoE base model.

        Processes input tokens through embedding, all decoder layers with sliding window
        attention, Sparse MoE, and final normalization.

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
                Defaults to None (uses config.output_attentions).
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None (uses config.output_hidden_states).

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states, optional attentions,
                and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
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
                output_router_logits=bool(output_router_logits),
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states

            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view

        hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint_name(hidden_states, "model_output")

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


@register_module(TaskType.CAUSAL_LM, config=PhiMoeConfig, model_type="phimoe")
class PhiMoeForCausalLM(BaseCausalLMModule[PhiMoeModel, PhiMoeConfig]):
    """PhiMoE model with a language modeling head for causal language modeling tasks.

    This model is a sparse MoE transformer-based language model with causal attention masks
    and sliding window attention applied to perform autoregressive language generation.

    Attributes:
        config (PhiMoeConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "phimoe"
    _config_class = PhiMoeConfig

    def __init__(
        self,
        config: PhiMoeConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize PhiMoE model for causal language modeling.

        Args:
            config (PhiMoeConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=PhiMoeModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=config.lm_head_bias,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_router_logits: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass through the PhiMoE causal language model.

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
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
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
            CausalLMOutput: Contains logits, optional hidden_states, optional attentions,
                and updated past_key_values.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs.last_hidden_state
        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

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
