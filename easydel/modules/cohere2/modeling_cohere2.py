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


from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.common_types import Replicated
from eformer.escale import apply_logical_sharding
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
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
from easydel.infra.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    DecoderLayerOutput,
    SequenceClassifierOutput,
)
from easydel.infra.utils import ArrayParam, auto_remat, block_wise_ffn
from easydel.layers import ColumnParallelLinear, Embed, RowParallelLinear
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from easydel.modules._base import BaseCausalLMModule, BaseSequenceClassificationModule

from .cohere2_configuration import Cohere2Config


class Cohere2LayerNorm(nn.Module):
    """Layer Normalization for Cohere2 models.

    Implements standard layer normalization with learnable scale parameters,
    providing training stability through mean centering and variance normalization.
    Unlike RMSNorm, this includes mean subtraction for full layer normalization.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        dim: int | tuple,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nn.Rngs | None = None,
    ):
        """Initialize Cohere2LayerNorm layer.

        Args:
            dim (int | tuple): Dimension(s) of the normalization layer.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs, optional): Random number generator state. Defaults to None.
        """
        super().__init__()

        if rngs is None:
            rngs = nn.Rngs(0)
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.kernel = ArrayParam.bound(
            shape=(self.dim,) if isinstance(self.dim, int) else self.dim,
            dtype=self.param_dtype,
            init_method="ones",
            key=rngs.params(),
        )

    def _norm(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute layer normalization with mean centering.

        Args:
            x (jnp.ndarray): Input tensor to normalize.

        Returns:
            jnp.ndarray: Layer-normalized tensor with zero mean and unit variance.
        """
        mean = jnp.mean(x, -1, keepdims=True)
        variance = jnp.mean(jnp.pow((x - mean), 2), -1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + self.eps)

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        return {"kernel": Replicated}

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply layer normalization with learnable scale.

        Args:
            x (jnp.ndarray): Input tensor to normalize.

        Returns:
            jnp.ndarray: Normalized and scaled tensor.
        """
        if self.dtype in [
            jnp.float8_e4m3b11fnuz,
            jnp.float8_e4m3fn,
            jnp.float8_e4m3fnuz,
            jnp.float8_e5m2,
            jnp.float8_e5m2fnuz,
        ]:
            x = x.astype(jnp.float32)
        else:
            x = x.astype(jnp.promote_types(self.dtype, jnp.float32))
        output = self._norm(x).astype(self.dtype)
        weight = self.kernel.value.astype(self.dtype)
        return output * weight


class Cohere2Attention(UnifiedAttention):
    """Multi-head attention layer with layer-specific sliding window for Cohere2 models.

    Inherits from UnifiedAttention with Cohere2-specific customizations:
    - Layer-specific sliding window (only applies to sliding_attention layers)
    - Conditional RoPE application (only when sliding window is enabled)
    - Custom attention dropout configuration
    """

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize Cohere2 attention with layer-specific sliding window configuration.

        Args:
            config (Cohere2Config): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model, used to determine
                whether sliding window attention should be applied based on layer_types.
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
            sliding_window=config.sliding_window
            if config.layer_types is not None and config.layer_types[layer_idx] == "sliding_attention"
            else None,
        )

    def _create_rotary(self, config: Cohere2Config, dtype: jnp.dtype):
        """Create Cohere2-specific rotary embedding layer.

        Args:
            config (Cohere2Config): Model configuration.
            dtype (jnp.dtype): Data type for the rotary embeddings.

        Returns:
            Rotary embedding module configured for Cohere2 architecture.
        """
        return config.get_basic_rope(dtype, self.head_dim, self.head_dim, False)

    def _create_attention_performer(self, config: Cohere2Config, rngs: nn.Rngs):
        """Create attention performer with Cohere2's attention dropout.

        Args:
            config (Cohere2Config): Model configuration with attention dropout settings.
            rngs (nn.Rngs): Random number generator state.

        Returns:
            FlexibleAttentionModule: Attention module configured for Cohere2.
        """
        return FlexibleAttentionModule(
            rngs=rngs,
            dropout_prob=config.attention_dropout,
            base_config=config,
            softmax_scale=self.head_dim**-0.5,
        )

    def _apply_rotary(self, query_states, key_states, position_ids, frequencies):
        """Apply rotary position embeddings only when sliding window is enabled.

        In Cohere2, RoPE is conditionally applied based on whether the layer
        uses sliding window attention. This allows for hybrid attention patterns.

        Args:
            query_states: Query tensor from attention projection.
            key_states: Key tensor from attention projection.
            position_ids: Position indices for the tokens.
            frequencies: Precomputed rotary frequencies.

        Returns:
            Tuple of (query_states, key_states), with RoPE applied if sliding window
            is enabled, or unchanged if sliding window is disabled.
        """
        if self.sliding_window is not None:
            return self.rotary(
                query=query_states,
                key=key_states,
                positions=position_ids,
                frequencies=frequencies,
            )
        return query_states, key_states


class Cohere2MLP(nn.Module):
    """Multi-Layer Perceptron module for Cohere2 models.

    Implements feedforward network with SwiGLU activation and gated linear units
    for enhanced representation learning. Uses parallel column and row linear
    layers for efficient tensor parallelism.
    """

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Cohere2 MLP block.

        Args:
            config (Cohere2Config): Model configuration with MLP parameters.
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
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=self.precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)
        self.down_proj = row_parallel_linear(config.intermediate_size, config.hidden_size)
        self.up_proj = column_parallel_linear(config.hidden_size, config.intermediate_size)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Array: Transformed hidden states of shape (batch, seq_len, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = jax.nn.silu(checkpoint_name(self.gate_proj(hidden_states), name="mlp_gate"))
        up = checkpoint_name(self.up_proj(hidden_states), name="mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), name="mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return hidden_states


class Cohere2Block(nn.Module):
    """Single decoder layer for Cohere2 models.

    Combines multi-head attention and feedforward networks with
    layer normalization and residual connections. Uses a parallel
    attention-feedforward architecture where both are applied
    to the same normalized input, with layer-specific sliding window
    attention patterns.
    """

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ) -> None:
        """Initialize Cohere2 decoder block.

        Args:
            config (Cohere2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs
        attn_block = Cohere2Attention
        mlp_block = Cohere2MLP

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.self_attn = attn_block(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.mlp = mlp_block(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_layernorm = Cohere2LayerNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.is_sliding = (layer_idx + 1) % self.config.sliding_window_pattern != 0
        self.sliding_window = config.sliding_window

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
        """Forward pass through the decoder block.

        Applies parallel attention-feedforward architecture: x + attn(norm(x)) + mlp(norm(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): Cache view.
                Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, attention weights, and cache view.
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

        feed_forward_input = hidden_states

        if self.config.use_scan_mlp:
            feed_forward_hidden_states = block_wise_ffn(
                self.mlp,
                feed_forward_input,
                self.config.scan_mlp_chunk_size,
            )
        else:
            feed_forward_hidden_states = self.mlp(feed_forward_input)

        hidden_states = attn_outputs.attention_output + feed_forward_hidden_states + residual
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return DecoderLayerOutput(
            hidden_states=hidden_states,
            attention_weight=attn_outputs.attention_weight,
            router_logits=None,
            gate_loss=None,
            cache_view=attn_outputs.cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Cohere2Config, model_type="cohere2")
class Cohere2Model(EasyDeLBaseModule):
    """Cohere2 model implementation.

    This implements the Cohere2 language model architecture, utilizing transformer blocks
    with LayerNorm, rotary position embeddings, layer-specific sliding window attention,
    and a parallel attention-feedforward architecture.

    Attributes:
        config (Cohere2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Cohere2 base model.

        Args:
            config (Cohere2Config): Model configuration.
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

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.List(
            [
                Cohere2Block(
                    config=config,
                    layer_idx=idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Cohere2LayerNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_eps,
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
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Performs forward pass through the Cohere2 transformer model.

        Processes input tokens through embeddings, stacked Cohere2 decoder blocks with LayerNorm,
        layer-specific sliding window attention, SwiGLU gated FFNs, and conditional RoPE
        position encoding.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length). Either this
                or `inputs_embeds` must be provided but not both.
            inputs_embeds: Pre-computed input embeddings of shape (batch_size, sequence_length,
                hidden_size). Use instead of `input_ids` for custom embeddings.
            attention_mask: Boolean mask of shape (batch_size, sequence_length) indicating
                which tokens to attend to (True) and which to ignore (False).
            mask_info: Pre-computed mask information. If provided, overrides `attention_mask`.
            position_ids: Explicit position indices of shape (batch_size, sequence_length).
                Auto-generated from mask_info if not provided.
            mode: Runtime mode (MODE_TRAIN, MODE_DECODE, MODE_INFER). Auto-detected if None.
            past_key_values: Cached key/value states for efficient autoregressive generation.
            cache_metadata: Metadata for paged attention mechanisms.
            output_attentions: Whether to return attention weights from all layers.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput containing:
                - last_hidden_state: Final LayerNorm output of shape (batch, seq_len, hidden_size)
                - hidden_states: Tuple of all layer outputs if output_hidden_states=True
                - attentions: Tuple of all attention weights if output_attentions=True
                - past_key_values: Updated cache for next generation step

        Raises:
            ValueError: If neither `input_ids` nor `inputs_embeds` is provided, or if both
                are provided simultaneously.
            AssertionError: If sequence length exceeds max_position_embeddings.
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
            assert all_hidden_states is not None
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
        )

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2Model (decoder-only), this is not applicable.
        """
        raise NotImplementedError("Cohere2Model is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2Model, this is the model itself.
        """
        return self

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        Cohere2Model does not include the lm_head.
        """
        raise NotImplementedError("Cohere2Model does not include the language model head. See Cohere2ForCausalLM.")

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Cohere2Config, model_type="cohere2")
class Cohere2ForCausalLM(BaseCausalLMModule[Cohere2Model, Cohere2Config]):
    """Cohere2 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation. Includes logit scaling
    as configured in Cohere2Config and layer-specific sliding window attention.

    Attributes:
        config (Cohere2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "cohere2"
    _config_class = Cohere2Config

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Cohere2 model for causal language modeling.

        Args:
            config (Cohere2Config): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Cohere2Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
        )
        self.logit_scale = self.config.logit_scale

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
    ) -> CausalLMOutput:
        """Forward pass through the Cohere2 model for causal language modeling.

        Processes input through the Cohere2 transformer and applies the language
        modeling head to produce next-token predictions with logit scaling.

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
            apply_lm_head (bool, optional): Whether to apply the language modeling head.
                Defaults to True.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.

        Returns:
            CausalLMOutput: Contains logits (scaled by logit_scale), last_hidden_state,
                optional all hidden_states, optional attentions, and updated past_key_values.
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

    def apply_lm_head(self, hidden_states: Array) -> Array:
        """
        Applies the language model head to the hidden states.

        Args:
            hidden_states (Array): The last hidden states from the model.

        Returns:
            Array: The logits after applying the language model head.
        """
        lm_logits = self.lm_head(hidden_states)
        if self.logit_scale is not None:
            lm_logits *= self.logit_scale
        return lm_logits

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2ForCausalLM (decoder-only), this is not applicable.
        """
        raise NotImplementedError("Cohere2ForCausalLM is a decoder-only model and does not have a separate encoder.")

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2ForCausalLM, this is the underlying Cohere2Model.
        """
        return self.model.get_decoder()  # self.model is the Cohere2Model instance

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        """
        return self.lm_head

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (Cohere2Model)
        return self.model.get_embedding()  # Leverages Cohere2Model's get_embedding


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Cohere2Config, model_type="cohere2")
class Cohere2ForSequenceClassification(BaseSequenceClassificationModule[Cohere2Model, Cohere2Config]):
    """Cohere2 model for sequence classification tasks.

    This class extends the base Cohere2 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.

    Attributes:
        config (Cohere2Config): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "cohere2"
    _config_class = Cohere2Config

    def __init__(
        self,
        config: Cohere2Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Cohere2 model for sequence classification.

        Args:
            config (Cohere2Config): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Cohere2Model,
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
    ) -> SequenceClassifierOutput:
        """Forward pass for sequence classification.

        Processes input through the Cohere2 transformer and applies the classification
        head to produce class predictions using pooled representations.

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
                Cache with precomputed key-value states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            SequenceClassifierOutput: Contains pooled logits of shape (batch_size, num_labels),
                optional past_key_values, optional all hidden_states, and optional attentions.

        Raises:
            ValueError: If batch_size > 1 and no padding token is defined in config.
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

    def get_encoder(self) -> nn.Module:
        """
        Returns the encoder part of the model's graph definition.
        For Cohere2ForSequenceClassification (decoder-only), this is not applicable.
        """
        raise NotImplementedError(
            "Cohere2ForSequenceClassification is a decoder-only model and does not have a separate encoder."
        )

    def get_decoder(self) -> nn.Module:
        """
        Returns the decoder part of the model's graph definition.
        For Cohere2ForSequenceClassification, this is the underlying Cohere2Model.
        """
        return self.model  # self.model is the Cohere2Model instance

    def get_lm_head(self) -> nn.Module:
        """
        Returns the language model head of the module.
        Cohere2ForSequenceClassification uses a classification head instead.
        """
        raise NotImplementedError(
            "Cohere2ForSequenceClassification uses a classification head (self.score), not an lm_head."
        )

    def get_embedding(self) -> nn.Module:
        """
        Returns the embedding layer of the module.
        """
        # Access the embedding layer through the decoder (Cohere2Model)
        return self.model.get_embedding()  # Leverages Cohere2Model's get_embedding
