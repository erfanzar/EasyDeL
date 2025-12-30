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
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, CausalLMOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule
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
        layer_idx: int,
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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
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
        self.act_fn = ACT2FN[config.hidden_act]

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
            hidden_states (Array): Input hidden states, expected shape (..., num_heads * head_dim).

        Returns:
            Array: Hidden states after applying LayerNorm per head, same shape as input.
        """
        # hidden_states: [batch, seq_len, num_heads * head_dim]
        states_per_heads = jnp.split(hidden_states, 1, axis=1)
        # Normalize and merge the heads back together
        return jnp.concatenate(
            [norm(hidden_states) for norm, hidden_states in zip(self.norms, states_per_heads, strict=False)],
            axis=1,
        )


class StableLmAttention(UnifiedAttention):
    """StableLM Attention with Q/K normalization.

    Inherits Q/K normalization from QKNormAttention.
    Features:
    - Uses LayerNorm instead of RMSNorm
    - Per-head normalization (StableLmLayerNormPerHead)
    - Partial RoPE (partial_rotary_factor)
    """

    norms_mapping: ClassVar = {
        "query_normalization": "q_layernorm",
        "key_normalization": "k_layernorm",
    }

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        self.qk_layernorm = config.qk_layernorm
        self.partial_rotary_factor = config.partial_rotary_factor

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb_dim = int(config.partial_rotary_factor * self.head_dim)

        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            use_qk_norm=config.qk_layernorm,
        )

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Override to use per-head LayerNorm if qk_layernorm is enabled."""
        if not self.qk_layernorm:
            return None
        return StableLmLayerNormPerHead(
            head_dim=self.head_dim,
            num_heads=config.num_attention_heads,
            eps=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Override to use per-head LayerNorm if qk_layernorm is enabled."""
        if not self.qk_layernorm:
            return None
        return StableLmLayerNormPerHead(
            head_dim=self.head_dim,
            num_heads=config.num_key_value_heads,
            eps=config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_rotary(self, config, dtype):
        """Override for partial RoPE."""
        return config.get_basic_rope(
            dtype,
            head_size=int(config.partial_rotary_factor * (config.hidden_size // config.num_attention_heads)),
            rotary_dim=self.rotary_emb_dim,
            base=config.rope_theta,
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
    ):
        """Forward pass with per-head LayerNorm requiring transpose operations."""
        batch_size, sequence_length = hidden_states.shape[:2]

        # Project to Q/K/V
        query_states, key_states, value_states = (
            checkpoint_name(self.query_projection(hidden_states), "attn_query"),
            checkpoint_name(self.key_projection(hidden_states), "attn_key"),
            checkpoint_name(self.value_projection(hidden_states), "attn_value"),
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
            query_states = self.query_normalization(query_states.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            key_states = self.key_normalization(key_states.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
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
            causal=True,
        )

        attn_output = self.shard_attention_prod(self._merge_heads(attentions.attention_outputs))
        attn_output = checkpoint_name(self.output_projection(attn_output), "attn_output")

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
        layer_idx: int,
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
            layer_idx=layer_idx,
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
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ):
        """Forward pass of the decoder layer.

        Args:
            hidden_states (Array): Input hidden states (batch, seq_len, hidden_size).
            attention_mask (Array): Attention mask (batch, 1, seq_len, kv_seq_len).
            position_ids (Array): Position IDs (batch, seq_len).
            causal_mask (tp.Optional[Array | bool]): Causal mask for autoregressive behavior.
            cache_view (tp.Optional[TransformerCacheView | RaggedPagesCacheView]):
                Cache view for key/value states (optional).
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]):
                Metadata for paged attention (optional).
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            output_attentions (bool): Whether to output attention weights (default: False).
            fcm_mask (tp.Optional[Array]): Forward causal mask (FCM) mask (optional).
            frequencies (tp.Optional[Array]): Precomputed rotary frequencies (optional).

        Returns:
            tp.Tuple[Array, Array | None]: A tuple containing:
                - hidden_states (Array): Output hidden states after the decoder layer.
                - attention_outputs (Array | None): Attention weights (if `output_attentions` is True).
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
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
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

        embed_block = auto_remat(
            nn.Embed,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.embed_tokens = embed_block(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = [
            StableLmDecoderLayer(
                config=config,
                layer_idx=idx,
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
    ) -> BaseModelOutput:
        """Forward pass of the StableLM model.

        Args:
            input_ids (tp.Optional[Array]): Input token IDs (batch, seq_len).
                Mutually exclusive with `inputs_embeds`.
            inputs_embeds (tp.Optional[Array]): Input embeddings (batch, seq_len, hidden_size).
                Mutually exclusive with `input_ids`.
            attention_mask (tp.Optional[Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all layers
                (default defined by config).
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for caching.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]):
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
class StableLmForCausalLM(BaseCausalLMModule[StableLmModel, StableLmConfig]):
    """StableLM model with a Causal Language Modeling (CLM) head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "stablelm"
    _config_class = StableLmConfig

    def __init__(
        self,
        config: StableLmConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=StableLmModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
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
    ) -> CausalLMOutput:
        """Forward pass of the StableLM model for Causal Language Modeling.

        Args:
            input_ids (tp.Optional[Array]): Input token IDs (batch, seq_len).
                Mutually exclusive with `inputs_embeds`.
            inputs_embeds (tp.Optional[Array]): Input embeddings (batch, seq_len, hidden_size).
                Mutually exclusive with `input_ids`.
            attention_mask (tp.Optional[Array]): Attention mask (batch, seq_len). Usually used for padding tokens.
            position_ids (tp.Optional[Array]): Position IDs (batch, seq_len). If None, automatically generated.
            segment_ids (tp.Optional[Array]): Segment IDs for segment-based attention (optional).
            output_attentions (tp.Optional[bool]): Whether to output attention weights (default defined by config).
            output_hidden_states (tp.Optional[bool]): Whether to output hidden states for all
                layers (default defined by config).
            past_key_values (tp.Optional[TransformerCache | RaggedPagesCache]):
                Precomputed key/value states for caching.
            cache_metadata (tp.Optional[TransformerMetadata | RaggedPagesMetadata]):
                Metadata for paged attention (optional).

        Returns:
            CausalLMOutput: The model output, including logits, hidden states, and attentions.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
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
