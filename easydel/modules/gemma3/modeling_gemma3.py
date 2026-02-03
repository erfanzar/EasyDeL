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

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.escale import apply_logical_sharding
from eformer.loggings import get_logger
from eformer.pytree import auto_pytree
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
    ModelOutput,
    SequenceClassifierOutput,
    VLMCausalLMOutput,
)
from easydel.infra.utils import ACT2FN, ArrayParam, auto_remat, block_wise_ffn
from easydel.layers.attention import FlexibleAttentionModule
from easydel.layers.attention_unified import UnifiedAttention
from easydel.layers.base_modules import BaseCausalLMModule, BaseSequenceClassificationModule, BaseVisionLanguageModule
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
from easydel.layers.components.norms._norms import lowfloats
from easydel.modules.auto.auto_modeling import AutoEasyDeLVisionModel

from .gemma3_configuration import Gemma3Config, Gemma3TextConfig

logger = get_logger(__name__)


@auto_pytree
class Gemma3ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`tuple(tuple(Array))`):
        Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`Array`, *optional*):
        A `Array` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    last_hidden_state: Array | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None


@auto_pytree
class Gemma3CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Gemma3 causal language model (or autoregressive) outputs.

    Args:
        loss (`Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`Array` of shape `(batch_size, sequence_length, config.get_text_config().vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(Array))`):
            Tuple of `tuple(Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(Array)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(Array)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`Array`, *optional*):
            A `Array` of size `(batch_size, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Array | None = None
    logits: Array | None = None
    last_hidden_state: Array | None = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Float[Array, "batch seq_len hidden_dim"] | None = None


class Gemma3RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for Gemma3 models.

    Implements RMS normalization with Float8 support for efficient computation
    and memory usage in Gemma3 architecture. This normalization technique
    normalizes inputs by the root mean square, providing training stability.
    """

    kernel_init = staticmethod(nn.initializers.ones)

    def __init__(
        self,
        config: Gemma3TextConfig,
        param_dtype: jnp.dtype = jnp.float32,
        dim: int | None = None,
        epsilon: float | None = None,
    ):
        """Initialize Gemma3 RMS normalization layer.

        Args:
            config (Gemma3TextConfig): Model configuration containing rms_norm_eps and hidden_size.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.float32.
            dim (int | None, optional): Dimension for the normalization kernel. If None,
                uses config.hidden_size. Defaults to None.
            epsilon (float | None, optional): Small constant for numerical stability. If None,
                uses config.rms_norm_eps. Defaults to None.
        """
        self.config = config
        self.epsilon = self.config.rms_norm_eps if epsilon is None else epsilon
        self.param_dtype = param_dtype
        dim = self.config.hidden_size if dim is None else dim
        self.kernel = ArrayParam.bound(
            shape=(dim,),
            dtype=param_dtype,
            init_method="ones",
            key=None,
        )

    def _norm(self, x: jax.Array) -> jax.Array:
        """Apply RMS normalization without learnable scale.

        Args:
            x (jax.Array): Input tensor to normalize.

        Returns:
            jax.Array: Normalized tensor.
        """
        return x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.epsilon))

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jax.Array:
        """Apply RMS normalization with learnable scale.

        Args:
            hidden_states (Array): Input tensor to normalize of shape
                (batch_size, sequence_length, hidden_dim).

        Returns:
            jax.Array: Normalized and scaled hidden states. If output dtype is Float8,
                automatically casts to bfloat16 for compatibility.
        """
        variance = self._norm(hidden_states.astype(jnp.float32)).astype(self.param_dtype)
        out = (1 + self.kernel.value.astype(self.param_dtype)) * variance

        if out.dtype in lowfloats:
            out = out.astype(jnp.bfloat16)
        return out


class Gemma3Attention(UnifiedAttention):
    """Multi-head attention layer with Q/K normalization for Gemma3 models.

    Inherits from UnifiedAttention and extends it with Gemma3-specific features:
    - Custom Gemma3RMSNorm for Q/K normalization to stabilize attention
    - Layer-specific sliding window attention for efficient long-context processing
    - Custom softmax scaling using query_pre_attn_scalar for improved training dynamics

    Attributes:
        is_sliding (bool): Whether this layer uses sliding window attention.
        layer_idx (int): Index of this layer in the model stack.
        is_cross_attention (bool): Whether this is a cross-attention layer.
        causal (bool): Whether to apply causal masking.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        causal: bool = True,
        is_cross_attention: bool = False,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 attention layer.

        Args:
            config (Gemma3TextConfig): Model configuration with attention parameters.
            layer_idx (int): Index of this layer in the model, used to determine
                whether to apply sliding window attention.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for
                matrix operations. Defaults to None.
            causal (bool, optional): Whether to apply causal masking. Defaults to True.
            is_cross_attention (bool, optional): Whether this is a cross-attention layer.
                Defaults to False.
            rngs (nn.Rngs): Random number generator state.
        """
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=config.sliding_window if self.is_sliding else None,
            use_qk_norm=True,
        )

        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.causal = causal

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Override to use Gemma3RMSNorm instead of standard RMSNorm."""
        return Gemma3RMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Override to use Gemma3RMSNorm instead of standard RMSNorm."""
        return Gemma3RMSNorm(config, param_dtype=param_dtype, dim=self.head_dim)

    def _create_attention_performer(self, config, rngs):
        """Override to use custom softmax scale with query_pre_attn_scalar."""
        return FlexibleAttentionModule(
            rngs=rngs,
            base_config=config,
            softmax_scale=config.query_pre_attn_scalar**-0.5,
            dropout_prob=config.attention_dropout,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Post-process Q/K/V after projection and reshape, before RoPE/sharding.

        **KEY METHOD**: Override this to apply Q/K normalization or other transformations.
        By default, applies Q/K norm if configured, otherwise returns unchanged.

        Pattern for Q/K normalization:
            >>> def _postprocess_qkv(self, q, k, v):
            ...     if hasattr(self, 'q_norm'):
            ...         q = self.q_norm(q)
            ...         k = self.k_norm(k)
            ...     return q, k, v

        Args:
            query_states: Query tensor [batch, seq_len, num_heads, head_dim]
            key_states: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value_states: Value tensor [batch, seq_len, num_kv_heads, head_dim]

        Returns:
            Tuple of (processed_query, processed_key, processed_value)
        """

        return self.q_norm(query_states), self.k_norm(key_states), value_states


class Gemma3MLP(nn.Module):
    """Multi-Layer Perceptron module for Gemma3 models.

    Implements the feedforward network component of the transformer architecture
    with gated linear units and activation functions. Uses column and row parallel
    linear layers for efficient distributed computation.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 MLP block.

        Args:
            config (Gemma3TextConfig): Model configuration with MLP parameters including
                hidden_size, intermediate_size, and hidden_activation.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for
                matrix operations. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        embed_dim = self.config.hidden_size
        inner_dim = self.config.intermediate_size if self.config.intermediate_size is not None else 4 * embed_dim
        kernel_init = jax.nn.initializers.normal(config.initializer_range)

        self.act = ACT2FN[self.config.hidden_activation]

        column_parallel_linear = partial(
            ColumnParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(embed_dim, inner_dim)
        self.down_proj = row_parallel_linear(inner_dim, embed_dim)
        self.up_proj = column_parallel_linear(embed_dim, inner_dim)

    def __call__(
        self, hidden_states: Float[Array, "batch seq_len hidden_dim"]
    ) -> Float[Array, "batch seq_len hidden_dim"]:
        """Forward pass through the MLP block.

        Applies gated linear units with activation function: down_proj(act(gate_proj(x)) * up_proj(x))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).

        Returns:
            Array: Output tensor of shape (batch_size, sequence_length, hidden_dim).
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Gemma3DecoderLayer(nn.Module):
    """Single decoder layer for Gemma3 models.

    Combines self-attention and feedforward networks with residual connections
    and RMS layer normalization to form a complete transformer decoder layer.
    Features pre-normalization and post-normalization for both attention and MLP blocks.

    Attributes:
        config (Gemma3TextConfig): Model configuration.
        layer_idx (int): Index of this layer in the model stack.
        is_sliding (bool): Whether this layer uses sliding window attention.
        sliding_window (int): Size of the sliding window for local attention.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: str | jax.lax.Precision | None = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 decoder layer.

        Args:
            config (Gemma3TextConfig): Model configuration.
            layer_idx (int): Index of this layer in the model, determines attention type.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (str | jax.lax.Precision | None, optional): Numerical precision for
                matrix operations. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision

        mlp_block = Gemma3MLP
        attn_block = Gemma3Attention

        attn_block, mlp_block = auto_remat(
            attn_block,
            mlp_block,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )

        self.self_attn = attn_block(
            self.config,
            layer_idx=self.layer_idx,
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

        self.input_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.post_attention_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.config, param_dtype=self.param_dtype)

        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        default_frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture with both pre and post normalization for
        attention and MLP blocks: x + post_norm(attn(pre_norm(x))) and x + post_norm(mlp(pre_norm(x)))

        Args:
            hidden_states (Array): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for each token, shape (batch_size, sequence_length).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode, etc.) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | None, optional): View into the
                key-value cache for this layer. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache operations. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies for global attention.
                Defaults to None.
            default_frequencies (Array | None, optional): Precomputed RoPE frequencies for sliding
                window attention. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, optional attention weights, and updated cache view.
        """
        residual = hidden_states
        frequencies = default_frequencies if self.is_sliding else frequencies
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
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        hidden_states = self.post_attention_layernorm(attn_outputs.attention_output)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if self.config.use_scan_mlp:
            hidden_states = block_wise_ffn(
                self.mlp,
                hidden_states,
                self.config.scan_mlp_chunk_size,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = checkpoint_name(residual + hidden_states, "residual")
        hidden_states = checkpoint_name(hidden_states, "layer_output")
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


@register_module(TaskType.BASE_MODULE, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3TextModel(EasyDeLBaseModule):
    """Gemma3 text model implementation.

    This implements the Gemma3 decoder-only language model architecture, utilizing
    transformer blocks with RMSNorm, rotary position embeddings, Q/K normalization,
    and alternating global/sliding window attention for efficient long-context processing.
    Embeddings are scaled by sqrt(hidden_size) for training stability.

    Attributes:
        config (Gemma3TextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
        hidden_size (int): Dimension of the hidden states.
    """

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 text base model.

        Args:
            config (Gemma3TextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.hidden_size = self.config.hidden_size

        self.embed_tokens = Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nn.List([
            Gemma3DecoderLayer(
                self.config,
                layer_idx=i,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
            for i in range(self.config.num_hidden_layers)
        ])
        self.norm = Gemma3RMSNorm(self.config, param_dtype=self.dtype)

    @cached_property
    def default_frequencies(self):
        """Compute default RoPE frequencies for sliding window attention layers.

        Returns:
            ModuleCaches: Cached RoPE frequencies with local base frequency.
        """
        from easydel.infra.utils import ModuleCaches
        from easydel.layers.components import get_frequencies

        frequencies = get_frequencies(
            head_size=self.config.head_dim,
            rotary_dim=self.config.head_dim,
            max_position=self.config.granted_freq_max_position_embedding,
            base=self.config.rope_local_base_freq,
            rope_scaling=None,
        ).astype(jnp.bfloat16)

        return ModuleCaches(frequencies)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> BaseModelOutput:
        """Forward pass through the Gemma3 text base model.

        Processes input tokens through embedding, all decoder layers, and final normalization.
        Embeddings are scaled by sqrt(hidden_size) for training stability. Supports both
        global and sliding window attention based on layer configuration.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask of shape (batch_size, sequence_length)
                to avoid attention on padding tokens. Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention operations.
                Defaults to None.
            position_ids (Array | None, optional): Position indices for each token in the sequence,
                shape (batch_size, sequence_length). Defaults to None.
            token_type_ids (Array | None, optional): Token type IDs for distinguishing image/text
                tokens in multimodal inputs. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from all layers.
                Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states from all layers.
                Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for optimizations.
                Defaults to None (auto-detected).
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache containing precomputed key-value states for fast generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for managing the cache. Defaults to None.

        Returns:
            BaseModelOutput: Contains last_hidden_state, optional all hidden_states,
                optional attention weights, and updated past_key_values.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings") * (
                self.config.hidden_size**0.5
            )
        sequence_length = inputs_embeds.shape[1]

        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        # HF Gemma3 uses per-layer attention masks:
        # - full attention: causal
        # - sliding attention: causal + sliding window
        # For multimodal (token_type_ids), HF *OR*s a bidirectional image-block mask
        # with the base mask (so image tokens can see each other even outside the window).
        #
        # To match that, we construct two MaskInfo variants and select per layer:
        #   full:    causal | image_block_mask
        #   sliding: (causal & window) | image_block_mask
        mask_info_full = mask_info
        mask_info_sliding = mask_info
        if token_type_ids is not None:
            token_type_ids = jnp.asarray(token_type_ids, dtype=jnp.int32)
            is_image = token_type_ids == 1
            prev_is_image = jnp.pad(is_image, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
            new_image_start = is_image & (~prev_is_image)
            image_group_ids = jnp.cumsum(new_image_start.astype(jnp.int32), axis=1) - 1
            # Use per-image group IDs so equality masking matches HF's "same_image_block" logic.
            # Text tokens are 0 (disabled by MaskInfo.apply_token_type_ids' default zero_policy="q").
            grouped_token_types = jnp.where(is_image, image_group_ids + 1, 0).astype(jnp.int32)

            causal_mask_info = mask_info.apply_causal()
            mask_info_full = causal_mask_info.apply_token_type_ids(grouped_token_types)
            mask_info_sliding = causal_mask_info.apply_sliding_window(self.config.sliding_window).apply_token_type_ids(
                grouped_token_types
            )

            # We've baked causal (and for sliding layers, window) into the attention mask, so the
            # attention kernel shouldn't apply causal/window again.
            object.__setattr__(mask_info_full, "_causal_baked", True)
            object.__setattr__(mask_info_sliding, "_causal_baked", True)
        if position_ids is None:
            position_ids = mask_info.q_position_ids
        inputs_embeds = inputs_embeds
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Excepted <= {self.config.max_position_embeddings} got {sequence_length})"
        )

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
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask_info = (
                mask_info_sliding if self.config.layer_types[idx] == "sliding_attention" else mask_info_full
            )
            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=layer_mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                frequencies=self.frequencies,
                default_frequencies=self.default_frequencies,
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


@register_module(TaskType.CAUSAL_LM, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3ForCausalLM(BaseCausalLMModule[Gemma3TextModel, Gemma3TextConfig]):
    """Gemma3 model with a language modeling head for causal language modeling tasks.

    This model is a transformer-based language model with causal attention masks
    applied to perform autoregressive language generation. Supports both global
    and sliding window attention with optional logit softcapping.

    Attributes:
        config (Gemma3TextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations (default is jnp.bfloat16).
        param_dtype (jnp.dtype): Data type for parameters (default is jnp.bfloat16).
        precision: Precision setting for JAX operations.

    Note:
        Gemma3 requires bfloat16 precision. Using float16 may result in incorrect
        predictions or degraded output quality.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "gemma3_text"
    _config_class = Gemma3TextConfig

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 model for causal language modeling.

        Args:
            config (Gemma3TextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
                Note: float16 is not recommended and will trigger a warning.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        if param_dtype == jnp.float16 or param_dtype == "f2":
            logger.error(
                "Gemma-3's recommended dtype is bfloat16, but you are using float16. "
                "This may result in junk responses or incorrect predictions."
            )
        super().__init__(
            config=config,
            base_model_class=Gemma3TextModel,
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
        token_type_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> CausalLMOutput:  # type:ignore
        """Forward pass through the Gemma3 causal language model.

        Runs the base model and optionally applies the language modeling head to produce
        token logits for next-token prediction. Applies final_logit_softcapping if configured.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch_size, sequence_length). Defaults to None.
            token_type_ids (Array | None, optional): Token type IDs for distinguishing image/text
                tokens in multimodal inputs. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimizations. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache management metadata. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language modeling head. Defaults to True.

        Returns:
            CausalLMOutput: Contains logits (if apply_lm_head=True), hidden states, attentions,
                and updated cache.
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
            token_type_ids=token_type_ids,
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
        if self.config.final_logit_softcapping is not None:
            cap = jnp.array(self.config.final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)

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
        Base Models don't have a Language Model Head.
        """
        return self.lm_head

    def get_embedding(self):
        """
        Returns the embedding layer of the module.
        """
        return self.model.get_embedding()


@register_module(TaskType.SEQUENCE_CLASSIFICATION, config=Gemma3TextConfig, model_type="gemma3_text")
class Gemma3ForSequenceClassification(BaseSequenceClassificationModule[Gemma3TextModel, Gemma3TextConfig]):
    """Gemma3 model for sequence classification tasks.

    This class extends the base Gemma3 model by adding a linear classification head
    to perform sequence classification tasks such as sentiment analysis or text classification.
    Uses the last token's hidden state for classification (last-token pooling).

    Attributes:
        config (Gemma3TextConfig): Configuration for the model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
    """

    _task_type = TaskType.SEQUENCE_CLASSIFICATION
    _model_type = "gemma3_text"
    _config_class = Gemma3TextConfig

    def __init__(
        self,
        config: Gemma3TextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 model for sequence classification.

        Args:
            config (Gemma3TextConfig): Model configuration with num_labels for classification.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma3TextModel,
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
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass through the Gemma3 sequence classification model.

        Runs the base model and applies a classification head to the last token's hidden state
        to produce class logits.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings. Defaults to None.
            attention_mask (Array | None, optional): Mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information. Defaults to None.
            position_ids (Array | None, optional): Position indices for tokens. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached states. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return all hidden states. Defaults to None.

        Returns:
            SequenceClassifierOutput: Contains classification logits, hidden states, and attentions.

        Raises:
            ValueError: If batch size > 1 and no padding token is defined.
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
        return self.model

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

    def get_task_head(self):
        """Returns the sequence classification head."""
        return self.score


class Gemma3MultiModalProjector(nn.Module):
    """Multi-modal projector for Gemma3 vision-language models.

    Projects vision features into the text embedding space, enabling
    cross-modal understanding and generation in Gemma3. Uses average pooling
    to reduce spatial dimensions followed by linear projection.

    Attributes:
        config (Gemma3Config): Model configuration.
        patches_per_image (int): Number of patches per image side from vision encoder.
        tokens_per_side (int): Target tokens per image side after pooling.
        kernel_size (int): Pooling kernel size for spatial reduction.
    """

    def __init__(
        self,
        config: Gemma3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the multi-modal projector.

        Args:
            config (Gemma3Config): Model configuration containing vision and text configs.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

        self.mm_input_projection_weight = ArrayParam.bound(
            shape=(
                config.get_text_config().hidden_size,
                config.vision_config.hidden_size,
            ),
            dtype=param_dtype,
            init_method="zeros",
            key=None,
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config,
            param_dtype=param_dtype,
            dim=config.vision_config.hidden_size,
            epsilon=config.vision_config.layer_norm_eps,
        )
        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        kernel_size = self.patches_per_image // self.tokens_per_side
        self.kernel_size = kernel_size
        self.avg_pool = lambda x: jax.lax.reduce_window(
            x,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, 1, kernel_size, kernel_size),
            window_strides=(1, 1, kernel_size, kernel_size),
            padding="VALID",
        ) / (kernel_size * kernel_size)

    def __call__(self, vision_outputs: Float[Array, "batch hidden_dim num_patches"]) -> Array:
        """Project vision features to text embedding space.

        Applies average pooling to reduce spatial dimensions, normalizes with RMSNorm,
        then projects to the text model's hidden dimension.

        Args:
            vision_outputs (Array): Vision encoder outputs of shape
                (batch_size, hidden_dim, num_patches).

        Returns:
            Array: Projected vision features of shape (batch_size, num_tokens, text_hidden_size)
                ready for merging with text embeddings.
        """
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = jnp.transpose(vision_outputs, (0, 2, 1))

        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size,
            seq_length,
            self.patches_per_image,
            self.patches_per_image,
        )
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.reshape(batch_size, seq_length, -1)
        pooled_vision_outputs = jnp.transpose(pooled_vision_outputs, (0, 2, 1))
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = jax.lax.dot_general(
            normed_vision_outputs,
            self.mm_input_projection_weight.T,
            (((normed_vision_outputs.ndim - 1), (0,)), ((), ())),
        )
        return projected_vision_outputs.astype(vision_outputs.dtype)


@register_module(TaskType.BASE_MODULE, config=Gemma3Config, model_type="gemma3")
class Gemma3Model(EasyDeLBaseModule):
    """Multimodal Gemma3 model combining a vision tower, projector, and language model.

    This is the base multimodal model that processes both images and text. It consists of:
    - A vision tower for encoding images into feature representations
    - A multi-modal projector for mapping vision features to the text embedding space
    - A language model for processing the combined multimodal embeddings

    Attributes:
        config (Gemma3Config): Configuration for the multimodal model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.
        vocab_size (int): Size of the vocabulary from text config.
        pad_token_id (int): ID of the padding token.
    """

    def __init__(
        self,
        config: Gemma3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize the Gemma3 multimodal base model.

        Args:
            config (Gemma3Config): Multimodal model configuration containing vision_config
                and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vision_tower = AutoEasyDeLVisionModel.from_config(
            config=config.vision_config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.multi_modal_projector = Gemma3MultiModalProjector(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.vocab_size = config.get_text_config().vocab_size
        self.language_model = Gemma3TextModel(
            config=config.get_text_config(),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: Array) -> Array:
        """Extract image features from pixel values.

        Processes images through the vision tower and projects them to the
        text embedding space.

        Args:
            pixel_values (Array): Input images of shape (batch_size, channels, height, width).

        Returns:
            Array: Projected image features ready for merging with text embeddings.
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    def compute_embedding(
        self,
        input_ids: Int[Array, "batch seq_len"],
        *,
        image_features: Array | None = None,
        pixel_values: Array | None = None,
        **kwargs,
    ) -> Array:
        """Compute embeddings for multimodal inputs.

        Computes text embeddings from input_ids and optionally merges image features
        at positions marked by image_token_id. Embeddings are scaled by sqrt(hidden_size).

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            image_features (Array | None, optional): Pre-computed image features. If None
                and pixel_values is provided, features will be computed. Defaults to None.
            pixel_values (Array | None, optional): Raw image pixel values. Used to compute
                image_features if not provided directly. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Array: Combined multimodal embeddings of shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If input_ids is None.
        """
        if input_ids is None:
            raise ValueError("`input_ids` must be provided when calling `compute_embedding`.")

        image_token_id = self.config.image_token_id
        if image_token_id >= self.vocab_size:
            llm_input_ids = jnp.where(input_ids == image_token_id, 0, input_ids)
        else:
            llm_input_ids = input_ids

        hidden_size = self.config.get_text_config().hidden_size
        inputs_embeds = super().compute_embedding(llm_input_ids) * (hidden_size**0.5)

        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if image_features is not None:
            multimodal_embeddings = image_features.reshape(-1, image_features.shape[-1]).astype(inputs_embeds.dtype)
            inputs_embeds = BaseVisionLanguageModule.merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=image_token_id,
            )

        return inputs_embeds

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        token_type_ids: Array | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> Gemma3ModelOutputWithPast:
        """Forward pass through the Gemma3 multimodal base model.

        Processes images through the vision tower (if provided), computes embeddings,
        and runs the language model on the combined multimodal representation.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch_size, sequence_length).
                Must be provided if inputs_embeds is None.
            pixel_values (Array | None, optional): Input images of shape
                (batch_size, channels, height, width). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on padding tokens.
                Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode for optimizations. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cached key-value states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Cache management metadata. Defaults to None.
            token_type_ids (Array | None, optional): Token type IDs for distinguishing image/text tokens.
                Defaults to None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states. Defaults to None.
            **lm_kwargs: Additional arguments passed to the language model.

        Returns:
            Gemma3ModelOutputWithPast: Contains last_hidden_state, image_hidden_states,
                past_key_values, optional hidden_states, and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

        if inputs_embeds is None:
            inputs_embeds = self.compute_embedding(
                input_ids,
                image_features=image_features,
            )
        elif image_features is not None:
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_embedding()(
                    jnp.array(self.config.image_token_id, dtype="i4")
                )
            else:
                special_image_mask = jnp.expand_dims(input_ids == self.config.image_token_id, axis=-1)
                special_image_mask = jnp.broadcast_to(special_image_mask, inputs_embeds.shape)
            inputs_embeds = jnp.place(
                inputs_embeds,
                special_image_mask,
                image_features.astype(inputs_embeds.dtype),
                inplace=False,
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            **lm_kwargs,
        )
        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ) -> TransformerCache:
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length for the cache.
            starts (int | None, optional): Starting position for generation. Defaults to None.
            shardings (dict | None, optional): Sharding specifications for distributed caching.
                Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
        return self.language_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def prepare_inputs_for_generation(
        self,
        input_ids: Int[Array, "batch seq_len"],
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        pixel_values: Array | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        token_type_ids: Array | None = None,
    ) -> dict:
        """Prepare inputs for autoregressive generation.

        Prepares the language model inputs and adds pixel_values for multimodal generation.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            max_length (int): Maximum generation length.
            pad_token_id (int): Padding token ID.
            starts (int | None, optional): Starting position for generation. Defaults to None.
            pixel_values (Array | None, optional): Input images for the first generation step.
                Defaults to None.
            attention_mask (Array | None, optional): Attention mask. Defaults to None.
            token_type_ids (Array | None, optional): Token type IDs for multimodal inputs.
                Defaults to None.

        Returns:
            dict: Dictionary of model inputs including pixel_values.
        """
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            starts=starts,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def update_inputs_for_generation(self, model_outputs, model_kwargs: dict) -> dict:
        """Update inputs for the next generation step.

        Removes pixel_values and token_type_ids after the first generation step
        since images are only processed once.

        Args:
            model_outputs: Outputs from the previous generation step.
            model_kwargs (dict): Current model keyword arguments.

        Returns:
            dict: Updated model kwargs for the next generation step.
        """
        model_kwargs = super().update_inputs_for_generation(model_outputs, model_kwargs)
        model_kwargs.pop("pixel_values", None)  # only effect first iter
        model_kwargs.pop("token_type_ids", None)  # only effect first iter
        return model_kwargs

    def get_encoder(self):
        """
        Returns the encoder part of the model's graph definition.
        Gemma3 is a multi-modal model with a vision tower, but for typical LLM usage,
        it's considered a decoder-only architecture.
        """
        return self.vision_tower

    def get_decoder(self):
        """
        Returns the decoder part of the model's graph definition.
        """
        return self.language_model.get_decoder()

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
        return self.language_model.get_embedding()


@register_module(TaskType.IMAGE_TEXT_TO_TEXT, config=Gemma3Config, model_type="gemma3")
class Gemma3ForConditionalGeneration(BaseVisionLanguageModule[Gemma3Model, Gemma3Config]):
    """Gemma3 multimodal language model for conditional generation.

    Combines a vision tower, language model, and multi-modal projector for
    vision-language tasks such as image captioning, visual question answering,
    and multimodal dialogue. Inherits from BaseVisionLanguageModule.

    Features:
    - Vision tower for encoding images into feature representations
    - Multi-modal projector for aligning vision and text embeddings
    - Language model with alternating global/sliding window attention
    - Final logit softcapping for improved training stability

    Attributes:
        config (Gemma3Config): Configuration for the multimodal model.
        dtype (jnp.dtype): Data type for computations.
        param_dtype (jnp.dtype): Data type for parameters.
        precision: Precision setting for JAX operations.

    Class Attributes:
        _task_type: IMAGE_TEXT_TO_TEXT task type
        _model_type: "gemma3" model identifier
        _supports_video: False (Gemma3 is image-only)
        _uses_mrope: False (uses standard RoPE)
    """

    # Class attributes for registration and capabilities
    _task_type = TaskType.IMAGE_TEXT_TO_TEXT
    _model_type = "gemma3"
    _config_class = Gemma3Config
    _auto_register = False  # Already registered via decorator
    _supports_video = False
    _uses_mrope = False

    # Component name mapping
    _vision_tower_name = "vision_tower"
    _projector_name = "multi_modal_projector"
    _language_model_name = "language_model"

    loss_type = "ForCausalLM"

    def __init__(
        self,
        config: Gemma3Config,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Gemma3 for conditional generation.

        Args:
            config (Gemma3Config): Multimodal model configuration containing vision_config
                and text_config.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for matrix operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Gemma3Model,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            # VLM-specific configuration
            vision_feature_layer=getattr(config, "vision_feature_layer", -1),
            vision_feature_select_strategy=getattr(config, "vision_feature_select_strategy", "default"),
            image_token_index=getattr(config, "image_token_id", None),
            # LM head configuration
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            lm_head_bias=False,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        **kwargs,
    ) -> Float[Array, "batch num_patches hidden"]:
        """Extract and project image features from pixel values.

        Delegates to the base model's get_image_features implementation.

        Args:
            pixel_values: Input image pixel values
            **kwargs: Additional arguments (unused for Gemma3)

        Returns:
            Projected image features ready for merging with text embeddings
        """
        return self.base_model.get_image_features(pixel_values)

    def compute_embedding(self, input_ids: Int[Array, "batch seq_len"], *args, **kwargs) -> Array:
        """Compute embeddings for multimodal inputs.

        Delegates to the base model's compute_embedding method to handle both
        text and image embedding computation.

        Args:
            input_ids (Array): Input token IDs of shape (batch_size, sequence_length).
            *args: Additional positional arguments passed to base model.
            **kwargs: Additional keyword arguments (e.g., image_features, pixel_values).

        Returns:
            Array: Combined multimodal embeddings.
        """
        return self.base_model.compute_embedding(input_ids, *args, **kwargs)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] = None,
        pixel_values: Array = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        token_type_ids: Array | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **lm_kwargs,
    ) -> VLMCausalLMOutput:
        """Forward pass for the Gemma3 model.

        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            pixel_values: Input pixel values for images
            attention_mask: Attention mask
            mask_info: Mask information
            position_ids: Position IDs for text
            mode: Runtime mode
            past_key_values: Cached keys/values for language model
            cache_metadata: Metadata for paged attention
            apply_lm_head: Whether to apply the LM head
            token_type_ids: Token type IDs for distinguishing image/text tokens
            inputs_embeds: Input embeddings (alternative to input_ids)
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            **lm_kwargs: Additional arguments passed to the language model

        Returns:
            VLMCausalLMOutput: Model outputs including logits and optional states
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            **lm_kwargs,
        )
        hidden_states = outputs.last_hidden_state

        lm_logits = None
        if apply_lm_head:
            lm_logits = checkpoint_name(self.apply_lm_head(hidden_states), "lm_head_output")

        return VLMCausalLMOutput(
            logits=lm_logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states if pixel_values is not None else None,
        )

    def apply_lm_head(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> Array:
        """Apply the language modeling head with optional logit softcapping.

        Gemma3 uses final_logit_softcapping to prevent extreme logit values,
        which improves training stability.

        Args:
            hidden_states: Hidden states from the model

        Returns:
            LM logits with optional softcapping applied
        """
        lm_logits = self.lm_head(hidden_states)
        if self.config.get_text_config().final_logit_softcapping is not None:
            cap = jnp.array(self.config.get_text_config().final_logit_softcapping, dtype=lm_logits.dtype)
            lm_logits = cap * jax.nn.tanh(lm_logits / cap)
        return lm_logits

    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ) -> TransformerCache:
        """Initialize the KV cache for autoregressive generation.

        Args:
            batch_size (int): Batch size for the cache.
            max_length (int): Maximum sequence length for the cache.
            starts (int | None, optional): Starting position for generation. Defaults to None.
            shardings (dict | None, optional): Sharding specifications for distributed caching.
                Defaults to None.
            pad_token_id (int | None, optional): Padding token ID. Defaults to None.

        Returns:
            TransformerCache: Initialized cache for the language model.
        """
        return self.base_model.init_cache(batch_size, max_length, starts, shardings, pad_token_id)

    def get_vision_tower(self) -> nn.Module:
        """Returns the vision tower component."""
        return self.base_model.vision_tower

    def get_projector(self) -> nn.Module:
        """Returns the multimodal projector component."""
        return self.base_model.multi_modal_projector

    def get_language_model(self) -> nn.Module:
        """Returns the language model component."""
        return self.base_model.language_model
