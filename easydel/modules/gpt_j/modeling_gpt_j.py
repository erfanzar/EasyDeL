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
from eformer.loggings import get_logger
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
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
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.layers.linear import ColumnParallelLinear, RowParallelLinear

from .gpt_j_configuration import GPTJConfig as GPTJConfig

logger = get_logger(__name__)


class GPTJAttention(UnifiedAttention):
    """GPT-J Attention with partial RoPE.

    Inherits from UnifiedAttention.
    Uses separate Q/K/V projections with partial rotary embeddings.
    """

    projection_mapping: ClassVar[dict[str, str]] = {
        "query_projection": "q_proj",
        "key_projection": "k_proj",
        "value_projection": "v_proj",
        "output_projection": "out_proj",
        "qkv_projection": "qkv_proj",
        "mla_q_proj": "q_proj",
        "mla_q_a_proj": "q_a_proj",
        "mla_q_a_layernorm": "q_a_layernorm",
        "mla_q_b_proj": "q_b_proj",
        "mla_kv_a_proj_with_mqa": "kv_a_proj_with_mqa",
        "mla_kv_a_layernorm": "kv_a_layernorm",
        "mla_kv_b_proj": "kv_b_proj",
    }

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize GPT-J attention module.

        Args:
            config: GPTJConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
            layer_idx: Index of this layer in the model (0-indexed).
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
        )

    def _create_rotary(self, config: GPTJConfig, dtype: jnp.dtype):
        """Create GPT-J-specific rotary embedding with partial RoPE."""
        return config.get_basic_rope(
            dtype,
            head_size=self.head_dim,
            rotary_dim=config.rotary_dim,  # Partial RoPE
            base=10000,
            is_neox_style=False,
        )

    def _create_attention_performer(self, config: GPTJConfig, rngs: nn.Rngs):
        """Create attention performer with config dropout."""
        return FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attn_pdrop,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def _create_q_proj(self, config, dtype, param_dtype, precision, rngs):
        """Create query projection with checkpointing."""
        return ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_k_proj(self, config, dtype, param_dtype, precision, rngs):
        """Create key projection."""
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_v_proj(self, config, dtype, param_dtype, precision, rngs):
        """Create value projection."""
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

    def _create_o_proj(self, config, dtype, param_dtype, precision, rngs):
        """Create output projection (named out_proj for GPT-J)."""
        self.out_proj = ColumnParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        return self.out_proj

    def define_network(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ):
        """Define GPT-J-specific network with residual dropout."""
        # Call parent to create standard Q/K/V/O projections
        super().define_network(config, dtype, param_dtype, precision, rngs)

        # GPT-J has residual dropout
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)

    def _split_heads(self, hidden_states):
        """Split hidden states into attention heads."""
        return hidden_states.reshape((*hidden_states.shape[:2], self.config.num_attention_heads, self.head_dim))

    def _get_query_proj(self, hidden_states: Array) -> Array:
        """Apply query projection with checkpoint naming and head splitting."""
        query_states = checkpoint_name(self.q_proj(hidden_states), "attn_query")
        return self._split_heads(query_states)

    def _get_key_proj(self, hidden_states: Array) -> Array:
        """Apply key projection with checkpoint naming and head splitting."""
        key_states = checkpoint_name(self.k_proj(hidden_states), "attn_key")
        return self._split_heads(key_states)

    def _get_value_proj(self, hidden_states: Array) -> Array:
        """Apply value projection with checkpoint naming and head splitting."""
        value_states = checkpoint_name(self.v_proj(hidden_states), "attn_value")
        return self._split_heads(value_states)

    def _get_output_proj(self, attn_output: Array) -> Array:
        """Apply output projection with checkpoint naming and residual dropout."""
        attn_output = checkpoint_name(self.out_proj(attn_output), "attn_output")
        return self.resid_dropout(attn_output)


class GPTJMLP(nn.Module):
    """GPT-J MLP module.

    This module implements the feed-forward network used in the GPT-J model.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            intermediate_size (int): Dimensionality of the intermediate layer.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        intermediate_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-J MLP module.

        Args:
            config: GPTJConfig containing model hyperparameters.
            intermediate_size: Dimensionality of the intermediate feed-forward layer.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.intermediate_size = intermediate_size
        embed_dim = config.hidden_size
        kernel_init = nn.initializers.normal(config.initializer_range)

        self.fc_in = ColumnParallelLinear(
            embed_dim,
            intermediate_size,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.fc_out = RowParallelLinear(
            intermediate_size,
            embed_dim,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(rate=config.resid_pdrop)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass of the GPTJMLP module.

        Args:
            hidden_states: Input hidden states with shape (batch_size, sequence_length, hidden_size).

        Returns:
            Output hidden states with shape (batch_size, sequence_length, hidden_size) after
            processing through fc_in -> activation -> fc_out -> dropout.
        """
        gate = checkpoint_name(self.act(self.fc_in(hidden_states)), "mlp_gate")
        hidden_states = checkpoint_name(self.dropout(self.fc_out(gate)), "mlp_output")
        return hidden_states


class GPTJBlock(nn.Module):
    """GPT-J Transformer block.

    This module represents a single transformer block in the GPT-J model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-J transformer block.

        Args:
            config: GPTJConfig containing model hyperparameters.
            layer_idx: Index of this layer in the model (0-indexed).
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        self.config: GPTJConfig = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        hidden_size = self.config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        attn_block = GPTJAttention
        mlp_block = GPTJMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.ln_1 = nn.LayerNorm(
            self.config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.attn = attn_block(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=rngs,
        )

        self.mlp = mlp_block(
            config,
            inner_dim,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
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
        """Forward pass of the GPTJBlock module with parallel attention and FFN.

        Note: GPT-J uses a unique architecture where attention and FFN are computed in parallel
        on the same normalized input, then summed together before adding to the residual.

        Args:
            hidden_states: Input hidden states with shape (batch_size, sequence_length, hidden_size).
            mask_info: Masking information containing attention masks and positions.
            position_ids: Position indices for tokens with shape (batch_size, sequence_length).
            mode: Runtime mode (train/decode/prefill) for cache handling.
            cache_view: Optional cache view for key/value states in decoder inference.
            cache_metadata: Optional metadata for cache handling.
            output_attentions: Whether to return attention weights (default: False).
            frequencies: Optional precomputed rotary embedding frequencies with shape
                (sequence_length, head_dim).

        Returns:
            DecoderLayerOutput containing:
                - hidden_states: Output hidden states with shape (batch_size, sequence_length, hidden_size).
                - attention_weight: Optional attention weights if output_attentions=True.
                - cache_view: Updated cache view if cache is used.
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
        )
        attn_output = attn_outputs.attention_output
        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = checkpoint_name(attn_output + feed_forward_hidden_states + residual, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=GPTJConfig, model_type="gptj")
class GPTJModel(EasyDeLBaseModule):
    """GPT-J model implementation.

    This class implements the main GPT-J transformer model architecture, consisting of
    an embedding layer, multiple GPTJBlock layers, and a final layer normalization.

    Attributes:
            config (GPTJConfig): Configuration object for the model.
            dtype (jnp.dtype): Data type for computations.
            param_dtype (jnp.dtype): Data type for parameters.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
            rngs (nn.Rngs): Random number generators.
    """

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-J base model.

        Args:
            config: GPTJConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.embed_dim = config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(
            rate=self.config.embd_pdrop,
            rngs=rngs,
        )
        self.h = [
            GPTJBlock(
                config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.ln_f = nn.LayerNorm(
            self.config.hidden_size,
            epsilon=self.config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    @cached_property
    def frequencies(self):
        embed_dim = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = embed_dim // num_heads

        rotary_dim = self.config.rotary_dim
        return self.config.get_basic_frequencies(
            rotary_dim=rotary_dim,
            head_size=head_dim,
            base=10000,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        extra_embedding: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """Forward pass through the GPTJModel.

        Args:
            input_ids: Input token IDs with shape (batch_size, sequence_length). Mutually exclusive
                with inputs_embeds.
            attention_mask: Boolean mask with shape (batch_size, sequence_length) to avoid attention
                on padding tokens. Values are True for tokens to attend to, False for padding.
            mask_info: Pre-computed masking information. If None, computed from attention_mask.
            position_ids: Position indices with shape (batch_size, sequence_length). If None,
                automatically generated as sequential positions.
            mode: Runtime mode for cache handling. One of MODE_TRAIN, MODE_DECODE, or MODE_PREFILL.
                Auto-detected if None.
            past_key_values: Cache containing precomputed key/value states from previous decode steps.
            cache_metadata: Metadata for cache operations (positions, sequence lengths, etc.).
            inputs_embeds: Pre-computed input embeddings with shape (batch_size, sequence_length, hidden_size).
                Mutually exclusive with input_ids.
            extra_embedding: Additional embeddings to add to inputs_embeds, with shape
                (batch_size, sequence_length, hidden_size).
            output_attentions: Whether to return attention weights from all layers (default: False).
            output_hidden_states: Whether to return hidden states from all layers (default: False).

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final hidden states with shape (batch_size, sequence_length, hidden_size).
                - hidden_states: Tuple of hidden states from all layers if output_hidden_states=True.
                - attentions: Tuple of attention weights from all layers if output_attentions=True.
                - past_key_values: Updated cache for next decode step.
        """
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.wte(input_ids.astype("i4")), "embeddings")
        sequence_length = inputs_embeds.shape[1]
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

        hidden_states = inputs_embeds + extra_embedding if extra_embedding is not None else inputs_embeds

        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.h))

        hidden_states = self.dropout(inputs_embeds)
        for idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                position_ids=position_ids,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            past_key_values[idx] = layer_outputs.cache_view
        hidden_states = checkpoint_name(self.ln_f(hidden_states), "model_output")

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
        return self.wte


@register_module(TaskType.CAUSAL_LM, config=GPTJConfig, model_type="gptj")
class GPTJForCausalLM(BaseCausalLMModule[GPTJModel, GPTJConfig]):
    """GPT-J model with a language modeling head for autoregressive text generation.

    This model extends GPTJModel with a linear language modeling head that projects
    hidden states to vocabulary logits for next-token prediction.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gptj"
    _config_class = GPTJConfig

    def __init__(
        self,
        config: GPTJConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize GPT-J for causal language modeling.

        Args:
            config: GPTJConfig containing model hyperparameters.
            dtype: Data type for computations (default: jnp.bfloat16).
            param_dtype: Data type for parameters (default: jnp.bfloat16).
            precision: JAX precision setting for matrix operations (default: None).
            rngs: Flax NNX random number generators.
        """
        super().__init__(
            config=config,
            base_model_class=GPTJModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )
