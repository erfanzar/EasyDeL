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
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat, block_wise_ffn
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
from easydel.layers.norms import RMSNorm

from .seed_oss_configuration import SeedOssConfig


class SeedOssMLP(nn.Module):
    """Seed OSS gated MLP with SiLU activation.

    This implements a gated feed-forward network using the SwiGLU activation pattern,
    where the hidden representation is computed as: down_proj(act_fn(gate_proj(x)) * up_proj(x)).
    Uses column-parallel projections for gate and up, and row-parallel for down projection.

    Attributes:
        config: Model configuration object.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting for matrix multiplications.
        gate_proj: Column-parallel linear projection for gating.
        up_proj: Column-parallel linear projection for values.
        down_proj: Row-parallel linear projection to hidden size.
        dropout: Dropout layer for residual dropout.
        act_fn: Activation function (default: SiLU).
    """

    def __init__(
        self,
        config: SeedOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssMLP layer.

        Args:
            config: Model configuration containing hidden_size, intermediate_size,
                initializer_range, and other settings.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        column_parallel = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=config.mlp_bias,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )

        self.gate_proj = column_parallel(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.up_proj = column_parallel(config.hidden_size, config.intermediate_size, rngs=rngs)
        self.down_proj = row_parallel(config.intermediate_size, config.hidden_size, rngs=rngs)
        self.dropout = nn.Dropout(rate=config.resid_pdrop, rngs=rngs)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply the gated MLP transformation.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim) after gated MLP transformation.
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.gate_proj(hidden_states), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(self.act_fn(gate) * up), "mlp_down")
        hidden_states = self.dropout(hidden_states)
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class SeedOssAttention(UnifiedAttention[SeedOssConfig]):
    """Seed OSS attention with biased QKV projections and bias-free output projection.

    This attention module extends UnifiedAttention with Seed OSS-specific features including:
    - Support for sliding window attention based on layer type configuration
    - Bias-free output projection
    - Standard causal attention with optional sliding window

    Attributes:
        layer_idx: Index of this layer in the transformer stack.
        sliding_window: Sliding window size if this layer uses sliding attention, None otherwise.
    """

    def __init__(
        self,
        config: SeedOssConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssAttention layer.

        Args:
            config: Model configuration containing attention settings.
            layer_idx: Index of this layer in the transformer stack, used to determine
                whether sliding window attention should be applied.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.
        """
        self.layer_idx = layer_idx
        if config.layer_types is not None and layer_idx < len(config.layer_types):
            self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        else:
            self.sliding_window = None

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

    def _create_o_proj(
        self,
        config: SeedOssConfig,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.PrecisionLike,
        rngs: nn.Rngs,
    ) -> RowParallelLinear:
        """Create the output projection layer without bias.

        Args:
            config: Model configuration.
            dtype: Data type for computation.
            param_dtype: Data type for parameters.
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.

        Returns:
            A RowParallelLinear layer for output projection.
        """
        return RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            rngs=rngs,
            use_bias=self.config.attention_out_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
        )


class SeedOssDecoderLayer(nn.Module):
    """Single transformer decoder layer for Seed OSS.

    Implements a standard transformer decoder block with pre-normalization (Pre-LN),
    consisting of:
    1. RMSNorm -> Self-Attention -> Residual connection
    2. RMSNorm -> MLP -> Residual connection

    Supports gradient checkpointing through auto_remat for memory efficiency.

    Attributes:
        config: Model configuration object.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
        precision: JAX precision setting for matrix multiplications.
        self_attn: Self-attention module.
        mlp: Feed-forward network module.
        input_layernorm: RMSNorm applied before attention.
        post_attention_layernorm: RMSNorm applied before MLP.
    """

    def __init__(
        self,
        config: SeedOssConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssDecoderLayer.

        Args:
            config: Model configuration containing layer settings.
            layer_idx: Index of this layer in the transformer stack.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        attn_block = SeedOssAttention
        mlp_block = SeedOssMLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            config=config,
            layer_idx=layer_idx,
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

    def __call__(
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
        """Process input through attention and MLP sublayers.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info: Attention mask information for causal masking.
            position_ids: Position indices of shape (batch, seq_len).
            mode: Runtime mode (train, decode, etc.).
            cache_view: Optional KV cache view for incremental decoding.
            cache_metadata: Optional metadata for cache management.
            output_attentions: Whether to return attention weights.
            frequencies: Optional rotary embedding frequencies.

        Returns:
            DecoderLayerOutput containing hidden states, optional attention weights,
            and updated cache view.
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
        hidden_states = checkpoint_name(hidden_states + attn_outputs.attention_output, "residual_attn")

        ff_inputs = self.post_attention_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            mlp_outputs = block_wise_ffn(
                self.mlp,
                ff_inputs,
                self.config.scan_mlp_chunk_size,
            )
        else:
            mlp_outputs = self.mlp(ff_inputs)

        hidden_states = checkpoint_name(hidden_states + mlp_outputs, "residual_mlp")
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


class SeedOssModel(EasyDeLBaseModule):
    """Base Seed OSS transformer model without task-specific heads.

    This is the core transformer model that processes input tokens through
    embedding, multiple decoder layers, and final normalization. It serves
    as the backbone for task-specific models like SeedOssForCausalLM.

    The architecture follows a decoder-only transformer with:
    - Token embeddings with optional gradient checkpointing
    - Stack of SeedOssDecoderLayer blocks
    - Final RMSNorm layer

    Attributes:
        config: Model configuration object.
        embed_tokens: Token embedding layer.
        dropout: Embedding dropout layer.
        layers: List of decoder layers.
        norm: Final RMSNorm layer.
    """

    def __init__(
        self,
        config: SeedOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssModel.

        Args:
            config: Model configuration containing vocab_size, hidden_size,
                num_hidden_layers, and other architecture settings.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
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
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.dropout = nn.Dropout(rate=config.embd_pdrop)
        self.layers = [
            SeedOssDecoderLayer(
                config=config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(config.num_hidden_layers)
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Process input through the transformer encoder stack.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len). Mutually exclusive
                with inputs_embeds.
            inputs_embeds: Pre-computed input embeddings of shape (batch, seq_len, hidden_dim).
                Mutually exclusive with input_ids.
            attention_mask: Boolean mask of shape (batch, seq_len) indicating valid tokens.
            mask_info: Precomputed mask information for attention.
            position_ids: Position indices of shape (batch, seq_len). Auto-computed if None.
            mode: Runtime mode (train, decode, etc.). Auto-inferred if None.
            past_key_values: Cached key-value states for incremental decoding.
            cache_metadata: Metadata for cache management.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing last hidden state, optional all hidden states,
            optional attention weights, and updated cache.

        Raises:
            ValueError: If both or neither of input_ids and inputs_embeds are provided.
            AssertionError: If sequence length exceeds max_position_embeddings.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must provide exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        batch_size, sequence_length, _ = inputs_embeds.shape
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum sequence length exceeded: got {sequence_length}, "
            f"but config.max_position_embeddings={self.config.max_position_embeddings}"
        )

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(mask_info.q_segment_ids, axis=-1) - 1, a_min=0),
                (batch_size, sequence_length),
            ).astype(jnp.int32)

        hidden_states = self.dropout(inputs_embeds)
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

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
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
        """Get the encoder module.

        Raises:
            NotImplementedError: SeedOssModel is decoder-only and has no encoder.
        """
        raise NotImplementedError("SeedOssModel is decoder-only.")

    def get_decoder(self):
        """Get the decoder module.

        Returns:
            The model itself, as it is a decoder-only architecture.
        """
        return self

    def get_lm_head(self):
        """Get the language modeling head.

        Raises:
            NotImplementedError: Base model does not define an LM head.
        """
        raise NotImplementedError("Base model does not define an LM head.")

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            The embed_tokens layer used for input token embeddings.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=SeedOssConfig, model_type="seed_oss")
class SeedOssForCausalLM(BaseCausalLMModule[SeedOssModel, SeedOssConfig]):
    """Seed OSS model with a causal language modeling head.

    This model extends the base SeedOssModel with a linear head for next-token
    prediction. It is suitable for text generation, completion, and other
    autoregressive language modeling tasks.

    Attributes:
        model: The underlying SeedOssModel transformer.
        lm_head: Linear projection from hidden states to vocabulary logits.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "seed_oss"
    _config_class = SeedOssConfig

    def __init__(
        self,
        config: SeedOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssForCausalLM model.

        Args:
            config: Model configuration containing vocab_size, hidden_size,
                and other architecture settings.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.
        """
        super().__init__(
            config=config,
            base_model_class=SeedOssModel,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:  # type:ignore
        """Perform forward pass for causal language modeling.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed input embeddings of shape (batch, seq_len, hidden_dim).
            attention_mask: Boolean mask of shape (batch, seq_len) indicating valid tokens.
            mask_info: Precomputed mask information for attention.
            position_ids: Position indices of shape (batch, seq_len).
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for incremental decoding.
            cache_metadata: Metadata for cache management.
            apply_lm_head: Whether to apply the LM head to compute logits.

        Returns:
            CausalLMOutput containing logits, last hidden state, optional all hidden states,
            optional attention weights, and updated cache.
        """
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

        return CausalLMOutput(
            logits=lm_logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

    def get_encoder(self):
        """Get the encoder module.

        Raises:
            NotImplementedError: SeedOssForCausalLM is decoder-only and has no encoder.
        """
        raise NotImplementedError("SeedOssForCausalLM is decoder-only.")

    def get_decoder(self):
        """Get the decoder module.

        Returns:
            The decoder from the underlying model.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """Get the language modeling head.

        Returns:
            The lm_head layer used for vocabulary projection.
        """
        return self.lm_head

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            The embed_tokens layer from the underlying model.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=SeedOssConfig, model_type="seed_oss")
class SeedOssForSequenceClassification(BaseSequenceClassificationModule[SeedOssModel, SeedOssConfig]):
    """Seed OSS model with a sequence classification head.

    This model extends the base SeedOssModel with a linear classification head
    for sequence-level predictions. It pools the last hidden state using the
    last valid token position (determined by padding) and projects to the
    number of classes.

    Suitable for tasks like sentiment analysis, text classification, and
    natural language inference.

    Attributes:
        model: The underlying SeedOssModel transformer.
        score: Linear projection from hidden states to class logits.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "seed_oss"
    _config_class = SeedOssConfig

    def __init__(
        self,
        config: SeedOssConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the SeedOssForSequenceClassification model.

        Args:
            config: Model configuration containing hidden_size, num_labels,
                and other architecture settings.
            dtype: Data type for computation (default: bfloat16).
            param_dtype: Data type for parameters (default: bfloat16).
            precision: JAX precision setting for matrix multiplications.
            rngs: Random number generators for initialization.
        """
        super().__init__(
            config=config,
            base_model_class=SeedOssModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            classifier_name="score",
            classifier_bias=False,
        )

    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type: ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Perform forward pass for sequence classification.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed input embeddings of shape (batch, seq_len, hidden_dim).
            attention_mask: Boolean mask of shape (batch, seq_len) indicating valid tokens.
            mask_info: Precomputed mask information for attention.
            position_ids: Position indices of shape (batch, seq_len).
            segment_ids: Segment IDs for multi-segment inputs (currently unused).
            mode: Runtime mode (train, decode, etc.).
            past_key_values: Cached key-value states for incremental decoding.
            cache_metadata: Metadata for cache management.
            apply_lm_head: Whether to apply the classification head to compute logits.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput containing logits, optional hidden states,
            optional attention weights, and updated cache.

        Raises:
            ValueError: If neither input_ids nor inputs_embeds are provided for classification.
            ValueError: If batch_size > 1 and no pad_token_id is defined in config.
        """
        transformer_outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = None
        if apply_lm_head:
            hidden_states = transformer_outputs.last_hidden_state
            logits = self.score(hidden_states)
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("Either input_ids or inputs_embeds must be provided for classification.")

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

            logits = logits[jnp.arange(batch_size), sequence_lengths]

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            past_key_values=transformer_outputs.past_key_values,
        )

    def get_encoder(self):
        """Get the encoder module.

        Raises:
            NotImplementedError: SeedOssForSequenceClassification is decoder-only and has no encoder.
        """
        raise NotImplementedError("SeedOssForSequenceClassification is decoder-only.")

    def get_decoder(self):
        """Get the decoder module.

        Returns:
            The decoder from the underlying model.
        """
        return self.model.get_decoder()

    def get_lm_head(self):
        """Get the classification head.

        Returns:
            The score layer used for classification projection.
        """
        return self.score

    def get_embedding(self):
        """Get the token embedding layer.

        Returns:
            The embed_tokens layer from the underlying model.
        """
        return self.model.get_embedding()


__all__ = [
    "SeedOssForCausalLM",
    "SeedOssForSequenceClassification",
    "SeedOssModel",
]
