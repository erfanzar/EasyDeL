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


import math
from functools import cached_property, partial

import jax
from eformer import common_types
from eformer.escale import apply_logical_sharding
from einops import rearrange
from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.modeling_outputs import AttentionLayerOutput, BaseModelOutput, DecoderLayerOutput
from easydel.infra.utils import auto_remat
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

from .mosaic_configuration import MptConfig as MptConfig


class MptMLP(nn.Module):
    """MPT MLP module.

    This module implements the feed-forward network (MLP) used in the MPT model.
    It consists of an up-projection, GELU activation, and a down-projection, followed by dropout.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        up_proj (ParallelLinear): Linear layer for up-projection.
        down_proj (ParallelLinear): Linear layer for down-projection.
        hidden_dropout (nn.Dropout): Dropout layer applied to the output.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initializes the MptMLP module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        linear_class = partial(
            ColumnParallelLinear,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.up_proj = linear_class(
            self.config.hidden_size,
            self.config.expansion_ratio * self.config.hidden_size,
            rngs=rngs,
        )
        self.down_proj = linear_class(
            self.config.expansion_ratio * self.config.hidden_size,
            self.config.hidden_size,
            rngs=rngs,
        )
        self.hidden_dropout = nn.Dropout(
            self.config.attn_config.attn_pdrop,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        residual: Float[Array, "batch seq_len hidden_dim"],
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        up = jax.nn.gelu(checkpoint_name(self.up_proj(hidden_states), name="mlp_up"), approximate=False)
        hidden_states = checkpoint_name(self.down_proj(up), name="mlp_down")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return self.hidden_dropout(hidden_states) + residual


class MptAttention(UnifiedAttention):
    """MPT Attention module with ALiBi positional bias.

    Inherits from UnifiedAttention.
    Uses fused QKV projection and ALiBi (Attention with Linear Biases) for positional information.
    Overrides forward_alibi to handle custom ALiBi bias computation with masking.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        Wqkv (ColumnParallelLinear): Fused linear layer for query, key, and value projections.
        out_proj (RowParallelLinear): Linear layer for the output projection.
        resid_dropout (nn.Dropout): Dropout layer applied after the output projection.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize MPT attention with ALiBi support."""
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="alibi",
            causal=True,
        )

    def define_network(
        self,
        config: MptConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ):
        """Define MPT-specific network with fused QKV projection."""
        self.Wqkv = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size * 3,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            use_bias=config.use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )

        self.resid_dropout = nn.Dropout(
            config.attn_config.attn_pdrop,
            rngs=rngs,
        )

        self.attention_performer = self._create_attention_performer(config, rngs)
        self._create_alibi_slopes(config)

    def _create_attention_performer(self, config: MptConfig, rngs: nn.Rngs):
        """Create attention performer with MPT-specific settings."""
        softmax_scale = config.attn_config.softmax_scale
        if softmax_scale is None:
            softmax_scale = 1 / math.sqrt(self.head_dim)

        return FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=float(config.attn_config.attn_pdrop) if config.attn_config.attn_pdrop is not None else 0.0,
            base_config=config,
            softmax_scale=softmax_scale,
        )

    def _compute_alibi_bias(self, sequence_length):
        config: MptConfig = self.config
        return build_mpt_alibi_tensor(config.n_heads, sequence_length, config.attn_config.alibi_bias_max)

    def forward_alibi(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Override ALiBi forward with fused QKV projection.

        Important: ALiBi does not enforce causality by itself, so we must still
        apply causal masking (and padding masking) via `mask_info` / `causal`.
        """
        batch_size, sequence_length = hidden_states.shape[:2]

        mixed_qkv = checkpoint_name(self.Wqkv(hidden_states), "attn_qkv")
        query_states, key_states, value_states = jnp.split(mixed_qkv, 3, -1)

        query_states = rearrange(query_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        key_states = rearrange(key_states, "b s (h d) -> b s h d", h=self.config.n_heads)
        value_states = rearrange(value_states, "b s (h d) -> b s h d", h=self.config.n_heads)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        causal_for_kernel = self.causal
        if mask_info is not None and getattr(mask_info, "_causal_baked", False):
            causal_for_kernel = False

        (
            key_states,
            value_states,
            mask_info,
            _,
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

        if alibi is None:
            alibi_bias = self._compute_alibi_bias(self.config.max_seq_len)
        else:
            alibi_bias = alibi

        alibi_bias = jnp.asarray(alibi_bias, dtype=self.dtype)
        if alibi_bias.ndim == 3:
            alibi_bias = alibi_bias[None, ...]
        elif alibi_bias.ndim == 2:
            alibi_bias = alibi_bias[None, :, None, :]

        q_len = query_states.shape[1]
        kv_len = key_states.shape[1]

        if alibi_bias.shape[-1] != kv_len:
            start_k = max(0, alibi_bias.shape[-1] - kv_len)
            alibi_bias = alibi_bias[..., start_k:]

        if alibi_bias.shape[0] == 1 and batch_size != 1:
            alibi_bias = jnp.broadcast_to(alibi_bias, (batch_size, *alibi_bias.shape[1:]))

        if alibi_bias.shape[-2] == 1 and q_len != 1:
            alibi_bias = jnp.broadcast_to(alibi_bias, (*alibi_bias.shape[:-2], q_len, kv_len))
        elif alibi_bias.shape[-2] != q_len:
            start_q = max(0, alibi_bias.shape[-2] - q_len)
            alibi_bias = alibi_bias[..., start_q:, :]

        attention = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=alibi_bias,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=None,
            mask_info=mask_info,
            causal=causal_for_kernel,
        )

        attn_output = self.shard_attention_prod(
            attention.attention_outputs.reshape(batch_size, sequence_length, self.config.hidden_size)
        )
        attn_output = checkpoint_name(self.out_proj(attn_output), name="attn_output")
        attn_output = self.resid_dropout(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attention.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class MptBlock(nn.Module):
    """MPT Transformer block.

    This module represents a single transformer block in the MPT model,
    containing self-attention and MLP sub-layers with residual connections
    and layer normalization. It utilizes ALiBi for positional information.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        norm_1 (nn.LayerNorm): Layer normalization before the attention layer.
        attn (MptAttention): The self-attention module.
        norm_2 (nn.LayerNorm): Layer normalization before the MLP layer.
        ffn (MptMLP): The feed-forward (MLP) module.
        resid_attn_dropout (nn.Dropout): Dropout applied after the attention layer's residual connection.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initializes the MptBlock module.

        Args:
            config (MptConfig): The configuration object for the MPT model.
            dtype (jnp.dtype): Data type for computation. Defaults to jnp.float32.
            param_dtype (jnp.dtype): Data type for parameters. Defaults to jnp.float32.
            precision (jax.lax.PrecisionLike): Precision setting for JAX operations. Defaults to None.
            rngs (nn.Rngs): Random number generators.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        attn_block = MptAttention
        mlp_block = MptMLP
        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.norm_1 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.attn = attn_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.norm_2 = nn.LayerNorm(
            config.hidden_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )
        self.ffn = mlp_block(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
        )

        self.dropout_rate = self.config.attn_config.attn_pdrop
        self.resid_attn_dropout = nn.Dropout(self.dropout_rate, rngs=rngs)

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
        position_bias: Float[Array, "batch heads seq_len seq_len"] | None = None,
    ) -> DecoderLayerOutput:
        attn_outputs = self.attn(
            self.norm_1(hidden_states),
            mask_info,
            position_ids,
            mode,
            cache_view,
            cache_metadata,
            output_attentions,
            frequencies,
            alibi=position_bias,
        )

        hidden_states = self.resid_attn_dropout(attn_outputs.attention_output) + hidden_states
        output = self.ffn(self.norm_2(hidden_states), hidden_states)

        return DecoderLayerOutput(
            hidden_states=output,
            attention_weight=attn_outputs.attention_weight,
            cache_view=attn_outputs.cache_view,
        )


def build_mpt_alibi_tensor(num_heads, sequence_length, alibi_bias_max=8):
    """Builds the ALiBi tensor for MPT models.

    ALiBi (Attention with Linear Biases) is a method to incorporate positional information
    into transformer models without explicit position embeddings. It adds a bias to the
    attention scores based on the distance between query and key positions.

    Args:
        num_heads (int): The number of attention heads.
        sequence_length (int): The length of the sequence.
        alibi_bias_max (int, optional): The maximum bias value allowed by ALiBi. Defaults to 8.

    Returns:
        Array: The ALiBi tensor of shape (1, num_heads, sequence_length, sequence_length).
    """
    alibi = jnp.arange(
        1 - sequence_length,
        1,
        dtype="i4",
    ).reshape(
        1,
        1,
        1,
        sequence_length,
    )
    num_heads_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
    base = jnp.arange(1, num_heads_power_of_2 + 1, dtype=jnp.int32).astype("float32")
    base = base * (alibi_bias_max / num_heads_power_of_2)

    slopes = 1.0 / jnp.pow(2, base)
    slopes = slopes.reshape(
        1,
        num_heads_power_of_2,
        1,
        1,
    )

    if num_heads_power_of_2 != num_heads:
        slopes = jnp.concat(
            [slopes[:, 1::2, ...], slopes[:, ::2, ...]],
            axis=1,
        )[:, :num_heads, ...]

    alibi = alibi * slopes
    return alibi


@register_module(TaskType.BASE_MODULE, config=MptConfig, model_type="mpt")
class MptModel(EasyDeLBaseModule):
    """MPT model implementation.

    This class implements the main MPT transformer model architecture, consisting of
    an embedding layer (token and optional positional), multiple MptBlock layers,
    and a final layer normalization.

    Attributes:
        config (MptConfig): Configuration object for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision (jax.lax.PrecisionLike): Precision setting for JAX operations.
        rngs (nn.Rngs): Random number generators.
        wte (nn.Embed): Token embedding layer.
        emb_drop (nn.Dropout): Dropout layer applied after embeddings.
        blocks (tp.List[MptBlock]): List of transformer blocks.
        norm_f (nn.LayerNorm): Final layer normalization.
        alibi (Array, optional): Precomputed ALiBi tensor if using ALiBi.
    """

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initializes the MptModel.

        Args:
            config (MptConfig): The configuration object for the MPT model.
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
        self.wte = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.d_model,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
        )

        self.blocks = [
            MptBlock(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.n_layers)
        ]

        self.norm_f = nn.LayerNorm(
            config.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            epsilon=config.layer_norm_epsilon,
            use_bias=config.use_norm_bias,
            rngs=rngs,
        )

    @cached_property
    def alibi(self):
        return build_mpt_alibi_tensor(
            sequence_length=self.config.max_seq_len,
            num_heads=self.config.n_heads,
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
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.astype("i4"))
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

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = TransformerCache.init_empty(len(self.blocks))

        for idx, block in enumerate(self.blocks):
            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=None,
                position_bias=self.alibi,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)
            past_key_values[idx] = layer_outputs.cache_view
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        hidden_states = self.norm_f(hidden_states)
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


@register_module(TaskType.CAUSAL_LM, config=MptConfig, model_type="mpt")
class MptForCausalLM(BaseCausalLMModule[MptModel, MptConfig]):
    """MPT model with a language modeling head."""

    _task_type = TaskType.CAUSAL_LM
    _model_type = "mpt"
    _config_class = MptConfig

    def __init__(
        self,
        config: MptConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        super().__init__(
            config=config,
            base_model_class=MptModel,
            base_model_name="transformer",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=config.use_bias if hasattr(config, "use_bias") else False,
        )
