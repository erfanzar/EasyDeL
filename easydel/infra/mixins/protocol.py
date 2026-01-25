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

"""Protocol definitions for EasyDeL base modules.

This module defines the protocol (interface) that all EasyDeL models must implement.
It provides the BaseModuleProtocol abstract base class which specifies the required
methods and properties for model implementations, along with utility functions for
module representation and formatting.

The protocol ensures consistency across different model implementations and provides
type hints for better IDE support and type checking. It combines interfaces from the
base module, bridge functionality (EasyBridgeMixin), and generation capabilities
(EasyGenerationMixin).

Classes:
    BaseModuleProtocol: Abstract base class defining the interface for EasyDeL modules

Functions:
    return_type_adjuster: Decorator to adjust return types for type checking
    get_module_repr: Get string representation of module parameters
    prettify_nnx: Format module structure for display
    printify_nnx: Create printable representation of NNX modules

Type Aliases:
    PartitionLike: Type for partition rule specifications
    Self: Type variable for self-referencing types

The protocol includes methods for:
- Model forward passes and loss computation (overloaded for different model types)
- Parameter management (sharding, gathering, quantization, LoRA)
- Model I/O (saving, loading, HuggingFace Hub integration)
- Text generation (greedy search, sampling, beam search)
- Cache management (standard and paged attention)
- Framework conversion (PyTorch ↔ JAX/Flax)

Supported model types:
- Causal Language Models
- Sequence Classification
- Mixture of Experts (MoE)
- Vision models (CLIP)
- Multi-modal models

Example:
    >>> from easydel.infra.mixins.protocol import BaseModuleProtocol
    >>>
    >>> class MyModel(BaseModuleProtocol):
    ...     # Implement required methods
    ...     def __call__(self, input_ids, ...):
    ...         # Forward pass implementation
    ...         pass
    ...
    ...     def compute_loss(self, input_ids, labels, ...):
    ...         # Loss computation
    ...         pass
    ...
    ...     def generate(self, input_ids, ...):
    ...         # Generation implementation
    ...         pass
"""

from __future__ import annotations

import os
import typing as tp
from abc import ABCMeta, abstractmethod
from mimetypes import common_types

from ejkernel.types import MaskInfo
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import Mesh
from jaxtyping import Array, Bool, Float, Int, Shaped

from easydel.layers.components import ParallelLinear, QuantizationConfig

from ..base_config import EasyDeLBaseConfig
from ..loss_utils import LossConfig, LossMetrics
from ..modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutput,
    CLIPOutput,
    CLIPTextModelOutput,
    ImageClassifierOutput,
    ModelOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    VLMCausalLMOutput,
)

PartitionLike = tp.Optional[tp.Mapping[str, tp.Callable] | tp.Mapping[tuple, tp.Callable]]  # noqa
_CP = type[EasyDeLBaseConfig]
_T = tp.TypeVar("_T")
Self = tp.TypeVar("Self")

AnyArray = Shaped[Array, "..."]
Tokens = Int[Array, "batch seq_len"]
TokenEmbeds = Float[Array, "batch seq_len hidden_dim"]
AttentionMask = Bool[Array, "batch ..."]
PositionIds = Int[Array, "batch ..."]
SegmentIds = Int[Array, "batch ..."]
TokenTypeIds = Int[Array, "batch ..."]
HeadMask = Bool[Array, "..."]
EncoderHiddenStates = Float[Array, "batch ..."]
EncoderAttentionMask = Bool[Array, "batch ..."]
DecoderInputIds = Int[Array, "batch dec_seq_len"]
DecoderAttentionMask = Bool[Array, "batch dec_seq_len"]
DecoderPositionIds = Int[Array, "batch dec_seq_len"]
PixelValues = Float[Array, "batch ..."]
VideoPixelValues = Float[Array, "batch ..."]
ImageSizes = Int[Array, "batch ..."]
VisualPosMasks = Bool[Array, "batch ..."]
DeepstackVisualEmbeds = list[Float[Array, "..."]]
RopeDeltas = Int[Array, "..."]
CachePosition = Int[Array, "..."]
LogSnr = Float[Array, "..."]
NoiseMask = Bool[Array, "..."]
AudioFeatures = Float[Array, "batch n_mels frames"]
Labels = AnyArray
RecurrentState = list[Float[Array, "..."]]

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_state import EasyDeLState
    from easydel.layers.caching import (
        HybridCache,
        OperationsMetadata,
        RaggedPagesCache,
        RaggedPagesMetadata,
        RecurrentCache,
        TransformerCache,
        TransformerMetadata,
    )


def return_type_adjuster(
    original_return_type: type[_T],
) -> tp.Callable[[tp.Callable[..., nn.Module]], tp.Callable[..., _T]]:
    """Decorator factory to adjust return type annotations for type checking.

    This decorator is used to cast the return type of a function that returns
    an nn.Module to a more specific type for improved type checking support
    in IDEs and static analysis tools.

    Args:
        original_return_type: The desired return type to cast to.

    Returns:
        A decorator that wraps a function and casts its return value to the
        specified type.

    Example:
        >>> @return_type_adjuster(MyModelClass)
        ... def load_model(config) -> nn.Module:
        ...     return MyModelClass(config)
        ...
        >>> model = load_model(config)  # Type checker sees MyModelClass
    """

    def decorator(func: tp.Callable[..., nn.Module]) -> tp.Callable[..., _T]:
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> _T:
            return tp.cast(_T, func(*args, **kwargs))

        return wrapper

    return decorator


def get_module_repr(module: nn.Module) -> str:
    """Get a string representation of module parameters.

    Creates a human-readable string representation of a Flax NNX module,
    showing key parameters like input/output features for linear layers,
    dropout rates, embedding dimensions, and normalization epsilon values.

    Args:
        module: The Flax NNX module to represent.

    Returns:
        A formatted string representation of the module. The format depends
        on the module type:
        - ParallelLinear: "Linear(in_features=X, out_features=Y, bias=Z)"
        - Dropout: "Dropout(p=X)"
        - Embed: "Embedding(num_embeddings, embedding_dim)"
        - Modules with eps: "ModuleName(shape, eps=X)"
        - Other modules: Just the class name

    Example:
        >>> linear = ParallelLinear(in_features=512, out_features=1024)
        >>> get_module_repr(linear)
        'Linear(in_features=512, out_features=1024, bias=False)'
    """
    module_name = type(module).__name__

    if isinstance(module, ParallelLinear):
        in_features = (
            (module.kernel.shape[0] if hasattr(module.kernel, "shape") else "Null")
            if hasattr(module, "kernel")
            else module.kernel_init.__wrapped__.__code__.co_argcount - 1
        )
        out_features = (
            module.features
            if hasattr(module, "features")
            else (module.kernel.shape[-1] if hasattr(module.kernel, "shape") else "Null")
        )
        use_bias = module.use_bias if hasattr(module, "use_bias") else False
        return f"Linear(in_features={in_features}, out_features={out_features}, bias={use_bias})"

    elif isinstance(module, nn.Dropout):
        rate = module.rate if hasattr(module, "rate") else 0.0
        return f"Dropout(p={rate})"

    elif isinstance(module, nn.Embed):
        if hasattr(module, "embedding"):
            num_embeddings, embedding_dim = module.embedding.shape
            return f"Embedding({num_embeddings}, {embedding_dim})"
        return "Embedding(...)"

    elif hasattr(module, "eps"):
        shape_str = ""
        if hasattr(module, "kernel"):
            shape_str = str(tuple(module.kernel.shape))
        elif hasattr(module, "scale"):
            shape_str = str(tuple(module.scale.shape))
        return f"{module_name}({shape_str}, eps={module.eps})"

    return module_name


def prettify_nnx(
    module: nn.Module,
    indent: str = "",
    depth: int = 0,
    max_depth: int | None = None,
    module_param=None,
) -> str:
    """Format the structure of a Flax NNX module for display.

    Recursively creates a human-readable representation of a module's
    structure, similar to PyTorch's module printing.

    Args:
        module: The module to format.
        indent: Current indentation string.
        depth: Current recursion depth.
        max_depth: Maximum depth to recurse.
        module_param: Optional parameter dictionary.

    Returns:
        Formatted string representation of the module hierarchy.

    Example:
        >>> print(prettify_nnx(my_model, max_depth=2))
        MyModel(
          (encoder): Encoder(
            (layers): ModuleList(...)
          )
          (decoder): Decoder(...)
        )
    """
    if max_depth is not None and depth > max_depth:
        return ""

    output = []
    module_repr = get_module_repr(module)

    current_line = f"{indent}{module_repr}"

    children = list(module.iter_children())

    if module_param is not None:
        params_children = {key: param for key, param in module_param.items()}
    else:
        params_children = {}

    if children or any(
        isinstance(value, list) and all(isinstance(item, nn.Module) for item in value)
        for value in module.__dict__.values()
    ):
        output.append(current_line + "(")
        new_indent = indent + "  "

        for key, child in children:
            child_param = params_children.get(key, None)
            child_str = prettify_nnx(
                child,
                new_indent,
                depth + 1,
                max_depth,
                child_param,
            ).lstrip()
            output.append(f"{new_indent}({key}): {child_str}")

        for key, value in module.__dict__.items():
            if isinstance(value, list) and all(isinstance(item, nn.Module) for item in value):
                output.append(f"{new_indent}({key}): ModuleList(")

                if value:
                    first_item = value[0]
                    item_param = params_children.get(key, [None])[0] if params_children else None

                    if len(value) > 1:
                        child_str = prettify_nnx(
                            first_item,
                            new_indent + "  ",
                            depth + 1,
                            max_depth,
                            item_param,
                        ).lstrip()
                        output.append(f"{new_indent}  (0-{len(value) - 1}): {len(value)} x {child_str}")
                    else:
                        child_str = prettify_nnx(
                            first_item,
                            new_indent + "  ",
                            depth + 1,
                            max_depth,
                            item_param,
                        ).lstrip()
                        output.append(f"{new_indent}  {child_str}")

                output.append(f"{new_indent})")

        output.append(f"{indent})")
    else:
        output.append(current_line)

    return "\n".join(output)


class BaseModuleProtocol(metaclass=ABCMeta):
    """Protocol defining the common interface for EasyDeL modules.

    This abstract base class defines the complete interface that all EasyDeL
    model implementations must adhere to. It combines functionality from the
    base module, bridge functionality (EasyBridgeMixin), and generation
    capabilities (EasyGenerationMixin).

    The protocol ensures consistency across different model architectures and
    provides comprehensive type hints for IDE support and static type checking.
    It supports a wide variety of model types including:

    - Causal Language Models (decoder-only)
    - Encoder-only models (BERT-style)
    - Encoder-Decoder models (T5, BART-style)
    - Vision models (ViT, CLIP)
    - Vision-Language models (LLaVA, Qwen-VL)
    - Mixture of Experts (MoE) models
    - Audio models (Whisper)
    - State Space Models (Mamba, RWKV)

    Attributes:
        config_class: The configuration class type for this model.
        config: The model's configuration instance.
        base_model_prefix: String prefix used for identifying the base model
            in state dictionaries.
        _model_task: Optional string identifying the model's task type.
        _model_type: Optional string identifying the model architecture type.

    The protocol defines several categories of methods:

    Forward Pass Methods:
        - __call__: Multiple overloaded signatures for different model types
        - mesh_call: Execute forward pass within the model's mesh context
        - compute_loss: Compute loss for training

    Parameter Management:
        - shard_model: Distribute parameters across devices
        - gather_model: Collect distributed parameters
        - quantize: Apply quantization to model weights
        - apply_lora_to_layers: Add LoRA adapters
        - merge_lora_params/split_lora_params: Manage LoRA parameters

    Serialization:
        - save_pretrained: Save model to disk
        - push_to_hub: Upload to Hugging Face Hub
        - from_pretrained: Load from disk or Hub
        - to_state: Convert to EasyDeLState
        - to_torch: Convert to PyTorch model

    Generation:
        - generate: Autoregressive text generation
        - init_cache: Initialize KV cache
        - init_ragged_pages: Initialize paged attention cache
        - prepare_inputs_for_generation: Setup generation inputs
        - update_inputs_for_generation: Update inputs between steps

    Example:
        >>> from easydel import AutoEasyDeLModelForCausalLM
        >>>
        >>> # Load a model (implements BaseModuleProtocol)
        >>> model = AutoEasyDeLModelForCausalLM.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf"
        ... )
        >>>
        >>> # Use the forward pass
        >>> outputs = model(input_ids=input_ids, attention_mask=mask)
        >>>
        >>> # Generate text
        >>> generated = model.generate(input_ids, max_length=100)
        >>>
        >>> # Shard across devices
        >>> model.shard_model()
    """

    config_class: type[EasyDeLBaseConfig]
    config: EasyDeLBaseConfig
    base_model_prefix: str
    _model_task: str | None = None
    _model_type: str | None = None

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass for decoder-only backbones without a task head.

        Accepts token IDs or input embeddings, mask_info/attention_mask, and
        optional KV cache arguments. Returns BaseModelOutput with last_hidden_state,
        optional hidden_states/attentions, and updated cache state.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_attention_mask: EncoderAttentionMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass for decoder-only backbones with optional cross-attention.

        Provide encoder_hidden_states/encoder_attention_mask to enable cross-attention
        (e.g., GPT-2 conditional setups). Returns BaseModelOutput with cache updates.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_attention_mask: Encoder attention mask for cross-attention.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        extra_embedding: TokenEmbeds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass for models that accept extra embeddings.

        `extra_embedding` is merged with token embeddings (model-specific) before
        attention layers. Returns BaseModelOutput with hidden states/attentions
        and updated cache.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            extra_embedding: Additional embedding injected into the model.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        log_snr: LogSnr | None = None,
        noise_mask: NoiseMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Forward pass for diffusion language models.

        `log_snr` and `noise_mask` condition the diffusion timestep/noise schedule.
        Returns BaseModelOutput with hidden states and optional attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            log_snr: Log signal-to-noise ratio for diffusion conditioning.
            noise_mask: Mask for diffusion/noise conditioning.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Causal LM forward pass with optional LM head.

        Use apply_lm_head=False to return backbone states only. Returns CausalLMOutput
        with logits, cache state, and optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        log_snr: LogSnr | None = None,
        noise_mask: NoiseMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Diffusion LM forward pass with optional LM head.

        Uses log_snr/noise_mask for diffusion conditioning and returns CausalLMOutput
        with logits plus optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            log_snr: Log signal-to-noise ratio for diffusion conditioning.
            noise_mask: Mask for diffusion/noise conditioning.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """MoE backbone forward pass.

        Set output_router_logits=True to return router/gating diagnostics.
        Returns MoeModelOutput with hidden states, attentions, and cache updates.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.

        Returns:
            MoeModelOutput: MoE backbone output with optional router logits and cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """MoE causal LM forward pass.

        Returns MoeCausalLMOutput with logits plus optional router logits, hidden
        states, attentions, and cache updates.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.

        Returns:
            MoeCausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Sequence classification forward pass for decoder-style backbones.

        Runs the backbone and task head, returning SequenceClassifierOutput with
        logits plus optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        segment_ids: SegmentIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Sequence classification with optional LM head and segment IDs.

        `segment_ids` is model-specific (often unused). apply_lm_head toggles the
        classification head. Returns SequenceClassifierOutput with logits and states.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            segment_ids: Segment ids for packed/segmented inputs (model-specific).
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TokenClassifierOutput:
        """Token-level classification forward pass.

        Applies a per-token classifier and returns TokenClassifierOutput with logits
        plus optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            TokenClassifierOutput: Per-token logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> QuestionAnsweringModelOutput:
        """Extractive QA forward pass.

        Applies a span head to produce start/end logits and returns
        QuestionAnsweringModelOutput with optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            QuestionAnsweringModelOutput: Start/end span logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_attention_mask: EncoderAttentionMask | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """Encoder-only forward pass with pooling and optional cross-attention.

        Supports token_type_ids/head_mask and optional encoder_hidden_states for
        cross-attention. Returns BaseModelOutputWithPoolingAndCrossAttentions.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_attention_mask: Encoder attention mask for cross-attention.
            past_key_values: KV cache for autoregressive decoding.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Encoder-only sequence classification.

        Returns SequenceClassifierOutput with logits plus optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TokenClassifierOutput:
        """Encoder-only token classification.

        Returns TokenClassifierOutput with per-token logits and optional states.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            TokenClassifierOutput: Per-token logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> QuestionAnsweringModelOutput:
        """Encoder-only extractive QA.

        Returns QuestionAnsweringModelOutput with span logits and optional states.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            QuestionAnsweringModelOutput: Start/end span logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> MultipleChoiceModelOutput:
        """Multiple-choice classification over encoder outputs.

        Inputs are shaped (batch, num_choices, seq_len) and flattened internally by
        model implementations. Returns MultipleChoiceModelOutput with choice logits.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            MultipleChoiceModelOutput: Choice logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPooling:
        """Text encoder forward pass for CLIP/SigLIP-style models.

        Returns BaseModelOutputWithPooling with pooled text representation and
        optional hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            attention_mask: Attention mask (True/1 for tokens to attend).
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPooling: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Text encoder forward pass with projection into multimodal space.

        Returns CLIPTextModelOutput with projected text embeddings plus optional
        hidden states/attentions.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            attention_mask: Attention mask (True/1 for tokens to attend).
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CLIPTextModelOutput: Projected text embeddings plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        pixel_values: PixelValues | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
    ) -> BaseModelOutputWithPooling:
        """Vision encoder forward pass for CLIP/SigLIP/Vision Transformers.

        Supports interpolate_pos_encoding for resized inputs. Returns
        BaseModelOutputWithPooling with pooled image features and optional states.

        Args:
            pixel_values: Image pixel values (preprocessed).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            interpolate_pos_encoding: Model-specific argument.

        Returns:
            BaseModelOutputWithPooling: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        pixel_values: PixelValues | None = None,
        labels: Labels | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
    ) -> ImageClassifierOutput:
        """Image classification forward pass.

        If labels are provided, models may compute loss. Returns ImageClassifierOutput
        with logits plus optional hidden states/attentions.

        Args:
            pixel_values: Image pixel values (preprocessed).
            labels: Target labels for loss computation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            interpolate_pos_encoding: Model-specific argument.

        Returns:
            ImageClassifierOutput: Image logits plus optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        pixel_values: PixelValues | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CLIPOutput:
        """Full CLIP dual-encoder forward pass.

        Encodes text and images, projects them into a shared embedding space,
        and returns similarity logits along with text/image embeddings.

        Args:
            input_ids: Token ids for text inputs.
            pixel_values: Image pixel values (preprocessed).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CLIPOutput: Text/image embeddings and similarity logits."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        decoder_input_ids: DecoderInputIds | None = None,
        decoder_attention_mask: DecoderAttentionMask | None = None,
        pixel_values: PixelValues | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """Conditional generation forward pass for encoder-decoder models.

        Supports encoder text inputs, optional decoder_input_ids, and optional
        pixel_values for multimodal variants. Returns Seq2SeqLMOutput with logits,
        cache state, and encoder/decoder hidden states.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            decoder_input_ids: Decoder token ids.
            decoder_attention_mask: Decoder attention mask.
            pixel_values: Image pixel values (preprocessed).
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Seq2SeqLMOutput: Seq2Seq LM output with logits, cache, and encoder/decoder states."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        token_type_ids: TokenTypeIds | None = None,
        pixel_values: PixelValues | None = None,
        pixel_values_videos: VideoPixelValues | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_grid_hws: tuple | None = None,
        image_sizes: ImageSizes | None = None,
        image_max_grid_size: tuple | None = None,
        video_max_grid_size: tuple | None = None,
        visual_pos_masks: VisualPosMasks | None = None,
        deepstack_visual_embeds: DeepstackVisualEmbeds | None = None,
        rope_deltas: RopeDeltas | None = None,
        cache_position: CachePosition | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Vision-language causal LM forward pass.

        Combines text tokens with visual/video features (grid metadata, masks,
        mRoPE deltas, cache_position, etc.) and returns VLMCausalLMOutput with
        logits plus optional router diagnostics, hidden states, and attentions.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            pixel_values: Image pixel values (preprocessed).
            pixel_values_videos: Video pixel values (preprocessed).
            image_grid_thw: Image grid layout metadata (T, H, W).
            video_grid_thw: Video grid layout metadata (T, H, W).
            image_grid_hws: Image grid metadata (H, W) for some models.
            image_sizes: Original image sizes for dynamic resizing/positioning.
            image_max_grid_size: Max image grid size for visual encoders.
            video_max_grid_size: Max video grid size for visual encoders.
            visual_pos_masks: Visual position masks for multimodal alignment.
            deepstack_visual_embeds: Precomputed DeepStack visual embeddings.
            rope_deltas: RoPE delta offsets for multimodal positions.
            cache_position: Cache position indices for streaming decoding.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            VLMCausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        token_type_ids: TokenTypeIds | None = None,
        pixel_values: PixelValues | None = None,
        pixel_values_videos: VideoPixelValues | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_grid_hws: tuple | None = None,
        image_sizes: ImageSizes | None = None,
        image_max_grid_size: tuple | None = None,
        video_max_grid_size: tuple | None = None,
        visual_pos_masks: VisualPosMasks | None = None,
        deepstack_visual_embeds: DeepstackVisualEmbeds | None = None,
        rope_deltas: RopeDeltas | None = None,
        cache_position: CachePosition | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Vision-language backbone forward pass without an LM head.

        Accepts the same multimodal inputs as the LM variant but returns
        backbone ModelOutput (hidden states, attentions, cache) without logits.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            pixel_values: Image pixel values (preprocessed).
            pixel_values_videos: Video pixel values (preprocessed).
            image_grid_thw: Image grid layout metadata (T, H, W).
            video_grid_thw: Video grid layout metadata (T, H, W).
            image_grid_hws: Image grid metadata (H, W) for some models.
            image_sizes: Original image sizes for dynamic resizing/positioning.
            image_max_grid_size: Max image grid size for visual encoders.
            video_max_grid_size: Max video grid size for visual encoders.
            visual_pos_masks: Visual position masks for multimodal alignment.
            deepstack_visual_embeds: Precomputed DeepStack visual embeddings.
            rope_deltas: RoPE delta offsets for multimodal positions.
            cache_position: Cache position indices for streaming decoding.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: Backbone output with last_hidden_state, optional hidden_states/attentions, and multimodal metadata."""

    @tp.overload
    def __call__(
        self,
        input_features: AudioFeatures,
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Audio encoder forward pass for log-Mel features.

        Args:
            input_features: Log-Mel spectrogram features (batch, n_mels, frames).
            mask_info: Optional attention mask information for padding/timestamps.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return per-layer hidden states.

        Returns:
            BaseModelOutput with last_hidden_state and optional hidden_states/attentions.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Decoder forward pass with optional cross-attention.

        Uses decoder token ids with optional encoder_hidden_states/encoder_mask_info
        for cross-attention and returns BaseModelOutputWithPastAndCrossAttentions.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_mask_info: MaskInfo for encoder cross-attention.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def __call__(
        self,
        input_features: AudioFeatures,
        decoder_input_ids: DecoderInputIds,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: DecoderPositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Seq2SeqModelOutput:
        """Encoder-decoder audio forward pass (Whisper base model).

        Runs the audio encoder then the text decoder and returns Seq2SeqModelOutput
        with decoder states plus optional encoder/decoder attentions.

        Args:
            input_features: Audio log-Mel features.
            decoder_input_ids: Decoder token ids.
            decoder_mask_info: MaskInfo for decoder self-attention.
            decoder_position_ids: Decoder position indices.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Seq2SeqModelOutput: Seq2Seq base output with decoder states and optional encoder states."""

    @tp.overload
    def __call__(
        self,
        input_features: AudioFeatures,
        decoder_input_ids: DecoderInputIds,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: DecoderPositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> Seq2SeqLMOutput:
        """Whisper conditional generation forward pass with LM head.

        Returns Seq2SeqLMOutput with decoder logits, cache state, and optional
        encoder/decoder hidden states and attentions.

        Args:
            input_features: Audio log-Mel features.
            decoder_input_ids: Decoder token ids.
            decoder_mask_info: MaskInfo for decoder self-attention.
            decoder_position_ids: Decoder position indices.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Seq2SeqLMOutput: Seq2Seq LM output with logits, cache, and encoder/decoder states."""

    @tp.overload
    def __call__(
        self,
        input_features: AudioFeatures,
        encoder_outputs: BaseModelOutput | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Audio classification forward pass over encoder outputs.

        Args:
            input_features: Log-Mel spectrogram features (batch, n_mels, frames).
            encoder_outputs: Optional precomputed encoder outputs to reuse. If provided,
                it should include last_hidden_state and (when needed) hidden_states.
            output_attentions: Whether to return encoder attention weights.
            output_hidden_states: Whether to return per-layer hidden states.

        Returns:
            SequenceClassifierOutput with logits (batch, num_labels) and optional
            hidden_states/attentions from the encoder.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache: RecurrentCache | None = None,
        position_ids: PositionIds | None = None,
        attention_mask: AttentionMask | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """SSM/recurrent backbone forward pass (Mamba-style).

        Uses `cache` for recurrent state and returns ModelOutput with
        last_hidden_state, optional hidden_states, and updated cache.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache: Recurrent cache state (SSM/Mamba).
            position_ids: Position indices for tokens.
            attention_mask: Attention mask (True/1 for tokens to attend).
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache: RecurrentCache | None = None,
        position_ids: PositionIds | None = None,
        attention_mask: AttentionMask | None = None,
        apply_lm_head: bool = True,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """SSM/recurrent causal LM forward pass (Mamba-style).

        Applies the LM head when apply_lm_head=True and returns ModelOutput
        containing logits plus optional states and cache.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache: Recurrent cache state (SSM/Mamba).
            position_ids: Position indices for tokens.
            attention_mask: Attention mask (True/1 for tokens to attend).
            apply_lm_head: Whether to apply the LM head and return logits.
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated cache."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        cache_position: CachePosition | None = None,
        attention_mask: AttentionMask | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Mamba2/Falcon-Mamba backbone forward pass.

        Uses cache_params and cache_position for streaming/incremental updates and
        returns ModelOutput with last_hidden_state plus updated cache_params.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache_params: Recurrent cache parameters (Mamba2/Falcon-Mamba).
            output_hidden_states: Whether to return hidden states from all layers.
            cache_position: Cache position indices for streaming decoding.
            attention_mask: Attention mask (True/1 for tokens to attend).
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated cache_params."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        cache_position: CachePosition | None = None,
        attention_mask: AttentionMask | None = None,
        apply_lm_head: bool = True,
        **kwargs,
    ) -> ModelOutput:
        """Mamba2/Falcon-Mamba causal LM forward pass.

        Applies the LM head when apply_lm_head=True and returns ModelOutput with
        logits plus optional hidden states and cache_params.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache_params: Recurrent cache parameters (Mamba2/Falcon-Mamba).
            output_hidden_states: Whether to return hidden states from all layers.
            cache_position: Cache position indices for streaming decoding.
            attention_mask: Attention mask (True/1 for tokens to attend).
            apply_lm_head: Whether to apply the LM head and return logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated cache_params."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        state: RecurrentState | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> ModelOutput:
        """RWKV backbone forward pass with recurrent state.

        Uses `state` for recurrence and returns ModelOutput with last_hidden_state,
        optional hidden_states/attentions, and updated state when caching is enabled.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            state: Recurrent state for RWKV.
            use_cache: Whether to use/return recurrent state.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated state when use_cache=True."""

    @tp.overload
    def __call__(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        state: RecurrentState | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> ModelOutput:
        """RWKV causal LM forward pass.

        Returns ModelOutput with logits and recurrent state; output_router_logits is
        accepted for API consistency but unused by RWKV.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            state: Recurrent state for RWKV.
            use_cache: Whether to use/return recurrent state.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated state when use_cache=True."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_attention_mask: EncoderAttentionMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_attention_mask: Encoder attention mask for cross-attention.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        extra_embedding: TokenEmbeds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            extra_embedding: Additional embedding injected into the model.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        log_snr: LogSnr | None = None,
        noise_mask: NoiseMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            log_snr: Log signal-to-noise ratio for diffusion conditioning.
            noise_mask: Mask for diffusion/noise conditioning.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        log_snr: LogSnr | None = None,
        noise_mask: NoiseMask | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            log_snr: Log signal-to-noise ratio for diffusion conditioning.
            noise_mask: Mask for diffusion/noise conditioning.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.

        Returns:
            MoeModelOutput: MoE backbone output with optional router logits and cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.

        Returns:
            MoeCausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        segment_ids: SegmentIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            segment_ids: Segment ids for packed/segmented inputs (model-specific).
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TokenClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            TokenClassifierOutput: Per-token logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> QuestionAnsweringModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            QuestionAnsweringModelOutput: Start/end span logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_attention_mask: EncoderAttentionMask | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_attention_mask: Encoder attention mask for cross-attention.
            past_key_values: KV cache for autoregressive decoding.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> TokenClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            TokenClassifierOutput: Per-token logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> QuestionAnsweringModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            QuestionAnsweringModelOutput: Start/end span logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        position_ids: PositionIds | None = None,
        head_mask: HeadMask | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> MultipleChoiceModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            position_ids: Position indices for tokens.
            head_mask: Head mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            MultipleChoiceModelOutput: Choice logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPooling:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            attention_mask: Attention mask (True/1 for tokens to attend).
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPooling: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        attention_mask: AttentionMask | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            attention_mask: Attention mask (True/1 for tokens to attend).
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CLIPTextModelOutput: Projected text embeddings plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        pixel_values: PixelValues | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
    ) -> BaseModelOutputWithPooling:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            pixel_values: Image pixel values (preprocessed).
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            interpolate_pos_encoding: Model-specific argument.

        Returns:
            BaseModelOutputWithPooling: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        pixel_values: PixelValues | None = None,
        labels: Labels | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool | None = None,
    ) -> ImageClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            pixel_values: Image pixel values (preprocessed).
            labels: Target labels for loss computation.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            interpolate_pos_encoding: Model-specific argument.

        Returns:
            ImageClassifierOutput: Image logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        pixel_values: PixelValues | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CLIPOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            pixel_values: Image pixel values (preprocessed).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            CLIPOutput: Text/image embeddings and similarity logits."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        decoder_input_ids: DecoderInputIds | None = None,
        decoder_attention_mask: DecoderAttentionMask | None = None,
        pixel_values: PixelValues | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            decoder_input_ids: Decoder token ids.
            decoder_attention_mask: Decoder attention mask.
            pixel_values: Image pixel values (preprocessed).
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Seq2SeqLMOutput: Seq2Seq LM output with logits, cache, and encoder/decoder states."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        token_type_ids: TokenTypeIds | None = None,
        pixel_values: PixelValues | None = None,
        pixel_values_videos: VideoPixelValues | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_grid_hws: tuple | None = None,
        image_sizes: ImageSizes | None = None,
        image_max_grid_size: tuple | None = None,
        video_max_grid_size: tuple | None = None,
        visual_pos_masks: VisualPosMasks | None = None,
        deepstack_visual_embeds: DeepstackVisualEmbeds | None = None,
        rope_deltas: RopeDeltas | None = None,
        cache_position: CachePosition | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> VLMCausalLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            pixel_values: Image pixel values (preprocessed).
            pixel_values_videos: Video pixel values (preprocessed).
            image_grid_thw: Image grid layout metadata (T, H, W).
            video_grid_thw: Video grid layout metadata (T, H, W).
            image_grid_hws: Image grid metadata (H, W) for some models.
            image_sizes: Original image sizes for dynamic resizing/positioning.
            image_max_grid_size: Max image grid size for visual encoders.
            video_max_grid_size: Max video grid size for visual encoders.
            visual_pos_masks: Visual position masks for multimodal alignment.
            deepstack_visual_embeds: Precomputed DeepStack visual embeddings.
            rope_deltas: RoPE delta offsets for multimodal positions.
            cache_position: Cache position indices for streaming decoding.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            VLMCausalLMOutput: Causal LM output with logits plus optional hidden_states/attentions and cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        attention_mask: AttentionMask | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        token_type_ids: TokenTypeIds | None = None,
        pixel_values: PixelValues | None = None,
        pixel_values_videos: VideoPixelValues | None = None,
        image_grid_thw: tuple | None = None,
        video_grid_thw: tuple | None = None,
        image_grid_hws: tuple | None = None,
        image_sizes: ImageSizes | None = None,
        image_max_grid_size: tuple | None = None,
        video_max_grid_size: tuple | None = None,
        visual_pos_masks: VisualPosMasks | None = None,
        deepstack_visual_embeds: DeepstackVisualEmbeds | None = None,
        rope_deltas: RopeDeltas | None = None,
        cache_position: CachePosition | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            attention_mask: Attention mask (True/1 for tokens to attend).
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            token_type_ids: Token type ids for segment embeddings (encoder-style models).
            pixel_values: Image pixel values (preprocessed).
            pixel_values_videos: Video pixel values (preprocessed).
            image_grid_thw: Image grid layout metadata (T, H, W).
            video_grid_thw: Video grid layout metadata (T, H, W).
            image_grid_hws: Image grid metadata (H, W) for some models.
            image_sizes: Original image sizes for dynamic resizing/positioning.
            image_max_grid_size: Max image grid size for visual encoders.
            video_max_grid_size: Max video grid size for visual encoders.
            visual_pos_masks: Visual position masks for multimodal alignment.
            deepstack_visual_embeds: Precomputed DeepStack visual embeddings.
            rope_deltas: RoPE delta offsets for multimodal positions.
            cache_position: Cache position indices for streaming decoding.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: Backbone output with last_hidden_state, optional hidden_states/attentions, and multimodal metadata."""

    @tp.overload
    def mesh_call(
        self,
        input_features: AudioFeatures,
        mask_info: MaskInfo | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_features: Audio log-Mel features.
            mask_info: Optional MaskInfo for attention masking.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutput: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens,
        mask_info: MaskInfo | None = None,
        position_ids: PositionIds | None = None,
        encoder_hidden_states: EncoderHiddenStates | None = None,
        encoder_mask_info: MaskInfo | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            mask_info: Optional MaskInfo for attention masking.
            position_ids: Position indices for tokens.
            encoder_hidden_states: Encoder hidden states for cross-attention.
            encoder_mask_info: MaskInfo for encoder cross-attention.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Backbone output with last_hidden_state and optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_features: AudioFeatures,
        decoder_input_ids: DecoderInputIds,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: DecoderPositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Seq2SeqModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_features: Audio log-Mel features.
            decoder_input_ids: Decoder token ids.
            decoder_mask_info: MaskInfo for decoder self-attention.
            decoder_position_ids: Decoder position indices.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Seq2SeqModelOutput: Seq2Seq base output with decoder states and optional encoder states."""

    @tp.overload
    def mesh_call(
        self,
        input_features: AudioFeatures,
        decoder_input_ids: DecoderInputIds,
        decoder_mask_info: MaskInfo | None = None,
        decoder_position_ids: DecoderPositionIds | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> Seq2SeqLMOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_features: Audio log-Mel features.
            decoder_input_ids: Decoder token ids.
            decoder_mask_info: MaskInfo for decoder self-attention.
            decoder_position_ids: Decoder position indices.
            mode: Runtime mode (prefill/decode/auto).
            past_key_values: KV cache for autoregressive decoding.
            cache_metadata: Metadata for cache operations/paged attention.
            apply_lm_head: Whether to apply the LM head and return logits.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            Seq2SeqLMOutput: Seq2Seq LM output with logits, cache, and encoder/decoder states."""

    @tp.overload
    def mesh_call(
        self,
        input_features: AudioFeatures,
        encoder_outputs: BaseModelOutput | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_features: Audio log-Mel features.
            encoder_outputs: Precomputed encoder outputs to reuse.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            SequenceClassifierOutput: Classification logits plus optional hidden_states/attentions."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache: RecurrentCache | None = None,
        position_ids: PositionIds | None = None,
        attention_mask: AttentionMask | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache: Recurrent cache state (SSM/Mamba).
            position_ids: Position indices for tokens.
            attention_mask: Attention mask (True/1 for tokens to attend).
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache: RecurrentCache | None = None,
        position_ids: PositionIds | None = None,
        attention_mask: AttentionMask | None = None,
        apply_lm_head: bool = True,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache: Recurrent cache state (SSM/Mamba).
            position_ids: Position indices for tokens.
            attention_mask: Attention mask (True/1 for tokens to attend).
            apply_lm_head: Whether to apply the LM head and return logits.
            output_hidden_states: Whether to return hidden states from all layers.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated cache."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        cache_position: CachePosition | None = None,
        attention_mask: AttentionMask | None = None,
        **kwargs,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache_params: Recurrent cache parameters (Mamba2/Falcon-Mamba).
            output_hidden_states: Whether to return hidden states from all layers.
            cache_position: Cache position indices for streaming decoding.
            attention_mask: Attention mask (True/1 for tokens to attend).
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated cache_params."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        cache_params: RecurrentCache | None = None,
        output_hidden_states: bool | None = None,
        cache_position: CachePosition | None = None,
        attention_mask: AttentionMask | None = None,
        apply_lm_head: bool = True,
        **kwargs,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            cache_params: Recurrent cache parameters (Mamba2/Falcon-Mamba).
            output_hidden_states: Whether to return hidden states from all layers.
            cache_position: Cache position indices for streaming decoding.
            attention_mask: Attention mask (True/1 for tokens to attend).
            apply_lm_head: Whether to apply the LM head and return logits.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated cache_params."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        state: RecurrentState | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            state: Recurrent state for RWKV.
            use_cache: Whether to use/return recurrent state.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            ModelOutput: last_hidden_state plus optional hidden_states and updated state when use_cache=True."""

    @tp.overload
    def mesh_call(
        self,
        input_ids: Tokens | None = None,
        attention_mask: AttentionMask | None = None,
        inputs_embeds: TokenEmbeds | None = None,
        state: RecurrentState | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
    ) -> ModelOutput:
        """Run the forward pass inside `self.mesh` (auto/normal mesh).

        This is equivalent to `with self.mesh: self(*args, **kwargs)` and uses the
        same arguments and return type as the matching `__call__` overload. It does
        not switch to `explicit_mesh` or `manual_mesh`; enter those meshes explicitly
        if needed.

        Args:
            input_ids: Token ids for text inputs.
            attention_mask: Attention mask (True/1 for tokens to attend).
            inputs_embeds: Precomputed token embeddings (use instead of input_ids).
            state: Recurrent state for RWKV.
            use_cache: Whether to use/return recurrent state.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states from all layers.
            output_router_logits: Whether to return router/gating logits.

        Returns:
            ModelOutput: logits plus optional hidden_states and updated state when use_cache=True."""

    @tp.overload
    def compute_loss(
        self,
        *,
        labels: Labels | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        """Compute loss for a forward pass.

        Runs `self(**batch)` and applies the configured loss function. For causal
        LM losses, labels can be inferred from input_ids when omitted. loss_kwargs
        are forwarded to the loss function.

        Args:
            labels: Target labels (optional for causal LM).
            loss_config: Optional loss configuration override.
            loss_kwargs: Extra kwargs forwarded to the loss function.
            **batch: Forward-pass inputs (input_ids, attention_mask, etc.).

        Returns:
            Tuple of (model_output, loss_metrics). model_output includes `loss`
            populated from the computed metrics.
        """

    @property
    @abstractmethod
    def graphdef(self) -> nn.GraphDef:
        """Returns the static graph definition of the model.

        The graphdef contains the model's structure without any variable data,
        used for JAX transformations and serialization.

        Returns:
            nn.GraphDef: The Flax NNX graph definition.
        """
        ...

    @property
    @abstractmethod
    def graphstate(self) -> nn.GraphState:
        """Returns the trainable state (parameters) of the model.

        The graphstate contains all trainable parameters like weights and biases
        that are updated during training.

        Returns:
            nn.GraphState: The trainable state containing model parameters.
        """
        ...

    @property
    @abstractmethod
    def graphother(self) -> nn.GraphState:
        """Returns the non-trainable state of the model.

        The graphother contains non-trainable state like batch normalization
        statistics, RNG keys, and other mutable but non-optimized state.

        Returns:
            nn.GraphState: The non-trainable state of the model.
        """
        ...

    @abstractmethod
    def to_dtype(self: Self, dtype) -> Self:
        """Convert model parameters to the specified data type.

        Args:
            dtype: The target JAX dtype to convert parameters to
                (e.g., jnp.float16, jnp.bfloat16, jnp.float32).

        Returns:
            Self: The model with parameters converted to the specified dtype.
        """

    @abstractmethod
    def half(self, change_runtime_dtype: bool = True):
        """Convert model parameters to float16 (half precision).

        Args:
            change_runtime_dtype: If True, also updates the runtime dtype
                configuration. Defaults to True.

        Returns:
            The model with parameters converted to float16.
        """

    @abstractmethod
    def float(self, change_runtime_dtype: bool = True):
        """Convert model parameters to float32 (full precision).

        Args:
            change_runtime_dtype: If True, also updates the runtime dtype
                configuration. Defaults to True.

        Returns:
            The model with parameters converted to float32.
        """

    @abstractmethod
    def _reformat_dtype(self, dtype):
        """Internal method to convert model parameters to a given data type.

        This is an internal implementation detail used by to_dtype, half,
        and float methods.

        Args:
            dtype: The target JAX dtype to convert parameters to.

        Returns:
            The model with reformatted parameters.
        """

    @abstractmethod
    def _get_mesh(self, mesh: Mesh | None = None) -> Mesh:
        """Retrieve the JAX mesh for distributed computation.

        Gets the mesh either from the provided argument or from the model's
        configuration if no mesh is provided.

        Args:
            mesh: Optional mesh to use. If None, uses the mesh from config.

        Returns:
            Mesh: The JAX sharding mesh to use for distributed operations.
        """

    @abstractmethod
    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Retrieve partition rules for model sharding.

        Gets the partition rules either from the provided argument or from
        the model's configuration if no rules are provided.

        Args:
            partition_rules: Optional partition rules to use. If None,
                uses the rules from config.

        Returns:
            PartitionLike: The partition rules for sharding model parameters.
        """

    @abstractmethod
    def _apply_sharding_fns(self, sharding_fns: tp.Mapping[str, tp.Callable]):
        """Apply sharding functions to the model's state.

        Internal method that applies a mapping of sharding functions to
        transform the model's parameters for distributed execution.

        Args:
            sharding_fns: A mapping from parameter names to sharding functions
                that define how each parameter should be distributed.
        """

    @abstractmethod
    def shard_model(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ):
        """Shards the model's parameters using the specified partitioning rules and mesh.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules for sharding.
            mesh (jax.sharding.Mesh, optional): The mesh to shard across.

        Returns:
            nn.Module: The sharded model.
        """

    @abstractmethod
    def gather_model(
        self,
        partition_rules: PartitionLike = None,
        mesh: Mesh | None = None,
    ):
        """Gathers the model's parameters based on the specified partitioning rules and mesh.

        Args:
            partition_rules (PartitionLike, optional): Partitioning rules for gathering.
            mesh (jax.sharding.Mesh, optional): The mesh to gather from.

        Returns:
            nn.Module: The gathered model.
        """

    @property
    @abstractmethod
    def _shard_fns(self):
        """property shard functions for model state and parameters."""

    @property
    @abstractmethod
    def _gather_fns(self):
        """property gather functions for model state and parameters."""

    @abstractmethod
    def quantize(
        self: Self,
        quantization_config: QuantizationConfig | None = None,
        quantize_tensors: bool = True,
        verbose: bool | None = None,
    ):
        """Quantizes the model's linear layers.

        Args:
            quantization_config: Quantization configuration. Pass None to use default INT8.
            quantize_tensors: Whether to quantize tensors directly.
            verbose: Whether to print verbose output.

        Returns:
            The quantized model.
        """

    @abstractmethod
    def to_state(self) -> EasyDeLState:
        """Convert the current model to an EasyDeLState object.

        Creates a state object that encapsulates the model's parameters,
        configuration, and other metadata in a format suitable for
        checkpointing and distributed training.

        Returns:
            EasyDeLState: A state object containing the model's parameters
            and configuration.
        """

    @abstractmethod
    def to_torch(self) -> PreTrainedModel:
        """Convert the current EasyDeL model to a HuggingFace PyTorch model.

        Transforms the JAX/Flax model parameters into their PyTorch equivalents
        and returns a compatible HuggingFace PreTrainedModel instance.

        Returns:
            PreTrainedModel: The equivalent HuggingFace PyTorch model with
            converted weights.
        """
        ...

    @abstractmethod
    def prepare_inputs_for_call(self, **kwargs):
        """Prepare and validate inputs before calling the model.

        This method processes and validates keyword arguments before they
        are passed to the model's forward method. It may add default values,
        reshape tensors, or perform other preprocessing.

        Args:
            **kwargs: Input keyword arguments for the model forward pass.

        Returns:
            A dictionary of processed inputs ready for the model call.
        """
        ...

    @abstractmethod
    def get_static_arguments(self) -> tuple:
        """Get static arguments for JIT compilation.

        Returns a tuple of argument names that should be treated as static
        during JAX JIT compilation (jax.jit or ejit). Static arguments
        are traced separately for each unique value.

        Returns:
            tuple: Names of arguments that should be static during JIT.
        """
        ...

    @classmethod
    @abstractmethod
    def lazy_init(cls: type[Self], *args, **kwargs) -> Self:
        """Initialize the model lazily using nnx.eval_shape.

        Creates a model instance without actually allocating memory for
        parameters. This is useful for determining model structure and
        memory requirements before full initialization.

        Args:
            *args: Positional arguments for model initialization.
            **kwargs: Keyword arguments for model initialization.

        Returns:
            Self: A lazily initialized model instance where parameters
            are represented by shape/dtype metadata rather than actual arrays.
        """
        ...

    @abstractmethod
    def apply_lora_to_layers(
        self: Self,
        lora_rank: int,
        lora_pattern: str | None = None,
        verbose: bool = False,
        rngs: nn.Rngs | None = None,
    ) -> Self:
        """Apply LoRA (Low-Rank Adaptation) to specified linear layers.

        LoRA enables efficient fine-tuning by adding low-rank decomposition
        matrices to linear layers while keeping the original weights frozen.

        Args:
            lora_rank: The rank of the low-rank decomposition matrices.
                Higher ranks allow more expressive adaptations but use more
                memory.
            lora_pattern: Optional regex pattern to match layer names for
                LoRA application. If None, applies to default layers
                (typically attention projections).
            verbose: If True, prints information about which layers are
                being modified.
            rngs: Optional Flax random number generators for initializing
                LoRA parameters.

        Returns:
            Self: The model with LoRA layers applied.
        """
        ...

    @abstractmethod
    def merge_lora_params(self: Self, pytree: dict) -> Self:
        """Merge LoRA parameters into the base model parameters.

        Combines the low-rank adaptation matrices with the original weights
        to produce a single set of weights that incorporate the LoRA
        adaptations.

        Args:
            pytree: A dictionary containing the LoRA parameters to merge.

        Returns:
            Self: The model with LoRA parameters merged into base weights.
        """
        ...

    @abstractmethod
    def split_lora_params(self: Self) -> dict:
        """Split LoRA parameters from the base model parameters.

        Extracts the LoRA adaptation matrices from the model, returning
        them as a separate dictionary. This is useful for saving only
        the LoRA weights.

        Returns:
            dict: A dictionary containing only the LoRA parameters.
        """
        ...

    @abstractmethod
    def unwrap_lora_to_layers(self, verbose: bool = False):
        """Remove LoRA adapters from linear layers within the model.

        Unwraps LoRA layers by either merging the adaptations into base
        weights or discarding them, restoring the original linear layer
        structure.

        Args:
            verbose: If True, prints information about which layers are
                being unwrapped.
        """
        ...

    @property
    @abstractmethod
    def transform_fn(self) -> tp.Callable:
        """Get the transform function for converting PyTorch to EasyDeL module.

        Returns a function that transforms PyTorch state dictionaries
        to the format expected by this EasyDeL model.

        Returns:
            Callable: A function that takes PyTorch weights and returns
            EasyDeL-compatible parameters.
        """
        ...

    @property
    @abstractmethod
    def pure_transform_fn(self) -> tp.Callable:
        """Get a pure transform function for PyTorch to EasyDeL conversion.

        Similar to transform_fn but returns a pure function without side
        effects, suitable for use with JAX transformations.

        Returns:
            Callable: A pure function for weight transformation.
        """
        ...

    @property
    @abstractmethod
    def params_sharding(self) -> dict:
        """Get the sharding specification for model parameters.

        Returns a dictionary mapping parameter paths to their sharding
        specifications, describing how parameters are distributed across
        devices.

        Returns:
            dict: A dictionary mapping parameter names to sharding specs.
        """
        ...

    @abstractmethod
    def merge_params(self, tree):
        """Merge a parameter tree into the current model.

        Updates the model's parameters with values from the provided
        parameter tree.

        Args:
            tree: A pytree of parameters to merge into the model.
        """
        ...

    @abstractmethod
    def split_params(self):
        """Split the model into its parameter tree.

        Extracts the model's parameters as a pytree structure that can
        be manipulated independently of the model object.

        Returns:
            The model's parameters as a pytree.
        """
        ...

    @abstractmethod
    def split_params_dict(
        self,
        params_dict: dict,
    ) -> dict:
        """Split model parameters from a dictionary into state components.

        Takes a flat parameter dictionary and organizes it into separate
        state components (trainable params, non-trainable state, etc.).

        Args:
            params_dict: A dictionary of model parameters.

        Returns:
            dict: A dictionary with parameters organized by state type.
        """

    @abstractmethod
    def merge_params_dict(self, params_dict: dict):
        """Merge model parameters from a dictionary into the current model.

        Takes a dictionary of parameters and updates the model's internal
        state with these values.

        Args:
            params_dict: A dictionary of parameters to merge.
        """

    @abstractmethod
    def _flop(self, *args, **kwargs) -> float | None:
        """Calculate the FLOP count from JaxPr.

        Estimates the number of floating point operations required for
        a forward pass through the model.

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            float or None: The estimated FLOP count, or None if calculation
            is not possible.
        """
        ...

    @abstractmethod
    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        push_to_hub: bool = False,
        token: str | bool | None = None,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: jnp.dtype | None = None,
        step: int | None = None,
        **kwargs,
    ):
        """Saves the model, its configuration, and optionally pushes it to the Hugging Face Hub.

        Args:
            save_directory: Directory where to save the model.
            push_to_hub: If True, pushes the model to the Hugging Face Hub.
            token: The Hugging Face Hub token.
            gather_fns: Custom gather functions for checkpoint saving.
            float_dtype: Data type for saving weights.
            step: Optional step number for checkpoint naming.
            **kwargs: Additional keyword arguments for Hugging Face Hub.
        """
        ...

    @abstractmethod
    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool | None = None,
        commit_message: str | None = None,
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: jnp.dtype | None = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        revision: str | None = None,
        commit_description: str | None = None,
    ) -> str:
        """Pushes the model to the Hugging Face Hub.

        Args:
            repo_id: The repository ID on Hugging Face Hub.
            use_temp_dir: If True, uses a temporary directory.
            commit_message: The commit message for the push.
            private: If True, creates a private repository.
            token: The Hugging Face Hub token.
            create_pr: If True, creates a pull request.
            gather_fns: Custom gather functions for checkpoint saving.
            float_dtype: Data type for saving weights.
            verbose: Whether to print verbose messages.
            mismatch_allowed: If True, allows mismatch in parameters while loading.
            revision: The revision to push to.
            commit_description: The commit description for the push.

        Returns:
            The URL of the created repository.
        """
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        **kwargs,
    ):
        """Loads an EasyDeL model from a pretrained model or path.

        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model.
            **kwargs: Additional keyword arguments for loading configuration.

        Returns:
            The loaded EasyDeL model.
        """
        ...

    @classmethod
    @abstractmethod
    def can_generate(cls) -> bool:
        """Checks if the model can generate sequences with `.generate()`.

        Returns:
            True if the model can generate, False otherwise.
        """
        ...

    @classmethod
    @abstractmethod
    def get_torch_loader(cls):
        """Gets the appropriate PyTorch AutoModel loader for this model type.

        Returns:
            The PyTorch AutoModel class for loading this model type.
        """
        ...

    @abstractmethod
    def generate(
        self,
        input_ids: Tokens,
        generation_config: tp.Any | None = None,
        prng_key: AnyArray | None = None,
        trace: bool = True,
        logits_processor: tp.Any | None = None,
        **kwargs,
    ):
        """Generates sequences of token ids for models with a language modeling head.

        Args:
            input_ids: The sequence used as a prompt for the generation.
            generation_config: The generation configuration to be used as base parametrization.
            prng_key: Random key for sampling-based generation.
            trace: Whether to trace generation for better performance.
            logits_processor: Custom logits processors.
            **kwargs: Additional generation parameters.

        Returns:
            Generated sequences and optionally scores.
        """
        ...

    @abstractmethod
    def init_cache(
        self,
        batch_size: int,
        max_length: int,
        starts: int | None = None,
        shardings: dict | None = None,
        pad_token_id: int | None = None,
    ):
        """Initializes and returns a standard (non-paged) Key-Value cache.

        Args:
            batch_size: The batch size for the cache.
            max_length: The maximum sequence length the cache needs to support.
            starts: Optional starting positions for the cache sequences.
            shardings: Optional dictionary specifying sharding configurations.
            pad_token_id: The ID of the padding token.

        Returns:
            An initialized standard TransformerCache object.
        """
        ...

    @abstractmethod
    def init_ragged_pages(
        self,
        metadata: tp.Any | None = None,
        page_size: int | None = None,
        hbm_utilization: float | None = None,
        max_model_length: int | None = None,
    ):
        """Initializes and returns the actual Paged Attention KV Cache tensors.

        Args:
            metadata: An optional pre-configured metadata object.
            page_size: Number of tokens per page. Required if metadata is None.
            hbm_utilization: Target HBM usage. Required if metadata is None.
            max_model_length: Maximum model sequence length. Required if metadata is None.

        Returns:
            An initialized RaggedPagesCache object containing the allocated cache tensors.
        """
        ...

    @abstractmethod
    def get_inference_cache_type(self) -> str:
        """Determine the appropriate cache type for inference.

        Returns:
            str: Either "hybrid" (for models with layer_types) or "ragged" (for pure attention).
        """
        ...

    @abstractmethod
    def prepare_inputs_for_generation(
        self,
        input_ids: Tokens,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings: int | None = None,
        attention_mask: AttentionMask | None = None,
        token_type_ids: TokenTypeIds | None = None,
        mask_info: tp.Any | None = None,
    ) -> dict[str, tp.Any]:
        """Sets up the initial inputs required for starting autoregressive generation.

        Args:
            input_ids: The initial sequence of token IDs.
            max_length: The maximum sequence length that the KV cache should support.
            pad_token_id: The ID used for padding tokens.
            starts: Optional pre-calculated starting positions.
            shardings: Optional sharding configuration passed to init_cache.
            attention_mask: An optional mask indicating which tokens should be attended to.
            token_type_ids: Optional segment IDs for models that use them.
            mask_info: Optional pre-constructed MaskInfo object.

        Returns:
            A dictionary containing the prepared inputs for generation.
        """
        ...

    @abstractmethod
    def update_inputs_for_generation(
        self,
        model_outputs: tp.Any,
        model_kwargs: dict[str, tp.Any],
    ) -> dict[str, tp.Any]:
        """Updates the keyword arguments for the next generation step.

        Args:
            model_outputs: The output object from the model's forward pass in the previous step.
            model_kwargs: The dictionary of keyword arguments used for the model call.

        Returns:
            The updated model_kwargs dictionary ready for the next generation step.
        """
        ...

    def __str__(self) -> str:
        """Return a human-readable string representation of the model.

        Returns:
            str: A formatted string showing the model's structure and layers,
            prefixed with "EasyDeL-".
        """
        return printify_nnx(self)

    def __repr__(self) -> str:
        """Return a detailed string representation of the model.

        Returns:
            str: A formatted string showing the model's structure and layers,
            prefixed with "EasyDeL-".
        """
        return printify_nnx(self)


def printify_nnx(model: nn.Module) -> str:
    """Create a printable string representation of an EasyDeL NNX module.

    This function wraps prettify_nnx to create a standardized string
    representation prefixed with "EasyDeL-". It handles edge cases
    gracefully by returning a fallback string if the module cannot
    be properly formatted.

    Args:
        model: The Flax NNX module to create a string representation for.

    Returns:
        A string representation of the model prefixed with "EasyDeL-".
        If the model cannot be formatted (e.g., it's a partition object
        without proper attributes), returns "EasyDeL-Partitions".

    Example:
        >>> model = LlamaForCausalLM(config)
        >>> print(printify_nnx(model))
        EasyDeL-LlamaForCausalLM(
          (model): LlamaModel(
            (embed_tokens): Embedding(32000, 4096)
            ...
          )
        )
    """
    try:
        return "EasyDeL-" + prettify_nnx(model)
    except AttributeError:
        return "EasyDeL-Partitions"
