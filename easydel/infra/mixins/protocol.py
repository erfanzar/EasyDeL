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
from jaxtyping import Array

from easydel.layers.linear import ParallelLinear
from easydel.layers.quantization import EasyDeLQuantizationConfig

from ..base_config import EasyDeLBaseConfig
from ..loss_utils import LossConfig, LossMetrics
from ..modeling_outputs import (
    CausalLMOutput,
    CLIPOutput,
    CLIPTextModelOutput,
    ImageClassifierOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
    SequenceClassifierOutput,
)

PartitionLike = tp.Optional[tp.Mapping[str, tp.Callable] | tp.Mapping[tuple, tp.Callable]]  # noqa
_CP = type[EasyDeLBaseConfig]
_T = tp.TypeVar("_T")
Self = tp.TypeVar("Self")

if tp.TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easydel.infra.base_state import EasyDeLState
    from easydel.layers.caching import (
        HybridCache,
        OperationsMetadata,
        RaggedPagesCache,
        RaggedPagesMetadata,
        TransformerCache,
        TransformerMetadata,
    )


def return_type_adjuster(
    original_return_type: type[_T],
) -> tp.Callable[[tp.Callable[..., nn.Module]], tp.Callable[..., _T]]:
    def decorator(func: tp.Callable[..., nn.Module]) -> tp.Callable[..., _T]:
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> _T:
            return tp.cast(_T, func(*args, **kwargs))

        return wrapper

    return decorator


def get_module_repr(module: nn.Module) -> str:
    """Get a string representation of module parameters."""
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
    """
    Protocol defining the common interface for EasyDeL modules.
    """

    config_class: type[EasyDeLBaseConfig]
    config: EasyDeLBaseConfig
    base_model_prefix: str
    _model_task: str | None = None
    _model_type: str | None = None

    @tp.overload
    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> CausalLMOutput:
        """Forward pass for Causal Language Models (e.g., GPT, Llama).

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            inputs_embeds (Array | None): Optional array of input embeddings. Use this if you've
                pre-computed embeddings and want to bypass the embedding layer.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
                Preferred over attention_mask for document-level boundaries.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            mode (RUNTIME_MODE_TYPES | None): Runtime mode ("prefill", "decode", or None for auto).
            past_key_values (Cache | None): Optional cache containing key and value tensors
                from previous model passes. Useful for faster inference.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            apply_lm_head (bool): Whether to apply the language model head. Defaults to True.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.

        Returns:
            CausalLMOutput: Model output containing logits and optionally hidden states/attentions.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass for Sequence Classification Models (e.g., BERT for sentiment).

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.

        Returns:
            SequenceClassifierOutput: Model output containing logits for classification.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeModelOutput:
        """Forward pass for Mixture-of-Experts (MoE) Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            output_router_logits (bool | None): Optional flag to return router logits
                that determine expert selection.
            mode (RUNTIME_MODE_TYPES | None): Runtime mode ("prefill", "decode", or None).
            past_key_values (Cache | None): Optional cache from previous model passes.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            apply_lm_head (bool): Whether to apply the language model head. Defaults to True.

        Returns:
            MoeModelOutput: Model output with logits, router logits, and optional states.
        """

    @tp.overload
    def __call__(
        self,
        input_ids: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass for Mixture-of-Experts (MoE) Causal Language Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            output_router_logits (bool | None): Optional flag to return router logits
                that determine expert selection.
            mode (RUNTIME_MODE_TYPES | None): Runtime mode ("prefill", "decode", or None).
            past_key_values (Cache | None): Optional cache from previous model passes.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            apply_lm_head (bool): Whether to apply the language model head. Defaults to True.

        Returns:
            MoeCausalLMOutput: Model output with logits, router logits, and optional states.
        """

    @tp.overload
    def __call__(
        self,
        pixel_values: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> ImageClassifierOutput:
        """Process image inputs through the CLIP vision encoder.

        Args:
            pixel_values: Optional array of shape (batch_size, num_channels, height, width)
                containing the pixel values of the images to encode.
            output_attentions: Optional bool indicating whether to return attention weights.
            output_hidden_states: Optional bool indicating whether to return all hidden states.

        Returns:
            ImageClassifierOutput containing the model outputs.
        """
        ...

    @tp.overload
    def __call__(
        self,
        input_ids: Array,
        attention_mask: Array,
        position_ids: Array,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> CLIPTextModelOutput:
        """Process text inputs through the CLIP text encoder.

        Args:
            input_ids: Array of shape (batch_size, sequence_length) containing the input
                token ids.
            attention_mask: Array of shape (batch_size, sequence_length) containing the
                attention mask for padding tokens.
            position_ids: Array of shape (batch_size, sequence_length) containing position
                indices for tokens.
            output_attentions: Whether to return attention weights. Defaults to False.
            output_hidden_states: Whether to return all hidden states. Defaults to False.

        Returns:
            CLIPTextModelOutput containing the model outputs.
        """
        ...

    @tp.overload
    def __call__(
        self,
        input_ids: Array | None = None,
        pixel_values: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        output_attentions=None,
        output_hidden_states=None,
    ) -> CLIPOutput:
        """Process both text and image inputs through the full CLIP model.

        This method handles the full CLIP model forward pass, encoding both text and image
        inputs and computing their similarity.

        Args:
            input_ids: Optional array of shape (batch_size, sequence_length) containing the
                input token ids for text.
            pixel_values: Optional array of shape (batch_size, num_channels, height, width)
                containing the pixel values of the images.
            attention_mask: Optional array of shape (batch_size, sequence_length) containing
                the attention mask for text padding tokens.
            position_ids: Optional array of shape (batch_size, sequence_length) containing
                position indices for text tokens.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.


        Returns:
            CLIPOutput containing the model outputs (including text embeddings,
            image embeddings, and their similarity).
        """
        ...

    @tp.overload
    def compute_loss(
        self,
        input_ids: Array | None = None,
        labels: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[CausalLMOutput, LossMetrics]:
        """Computes the loss for Causal Language Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            labels (Array | None): Optional array of target token IDs for computing loss.
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            past_key_values (Cache | None): Optional cache from previous model passes.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            loss_config (LossConfig | None): Optional configuration for loss computation.
            loss_kwargs (dict | None): Optional additional keyword arguments for loss computation.

        Returns:
            tuple[CausalLMOutput, LossMetrics]: Model output and loss metrics.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: Array | None = None,
        labels: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[SequenceClassifierOutput, LossMetrics]:
        """Computes the loss for Sequence Classification Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            labels (Array | None): Optional array of target classification labels.
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            loss_config (LossConfig | None): Optional configuration for loss computation.
            loss_kwargs (dict | None): Optional additional keyword arguments for loss computation.

        Returns:
            tuple[SequenceClassifierOutput, LossMetrics]: Model output and loss metrics.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: Array | None = None,
        labels: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[MoeModelOutput, LossMetrics]:
        """Computes the loss for Mixture-of-Experts (MoE) Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            labels (Array | None): Optional array of target token IDs for computing loss.
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            output_router_logits (bool | None): Optional flag to return router logits.
            past_key_values (Cache | None): Optional cache from previous model passes.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            loss_config (LossConfig | None): Optional configuration for loss computation.
            loss_kwargs (dict | None): Optional additional keyword arguments for loss.

        Returns:
            tuple[MoeModelOutput, LossMetrics]: Model output and loss metrics.
        """

    @tp.overload
    def compute_loss(
        self,
        input_ids: Array | None = None,
        labels: Array | None = None,
        inputs_embeds: Array | None = None,
        attention_mask: Array | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Array | None = None,
        segment_ids: Array | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
    ) -> tuple[MoeCausalLMOutput, LossMetrics]:
        """Computes the loss for Mixture-of-Experts (MoE) Causal Language Models.

        Args:
            input_ids (Array | None): Optional array of token IDs. Shape (batch_size, seq_length).
            labels (Array | None): Optional array of target token IDs for computing loss.
            inputs_embeds (Array | None): Optional array of input embeddings.
            attention_mask (Array | None): Optional array indicating which tokens should be attended to.
            mask_info (MaskInfo | None): Optional MaskInfo object for segment-aware masking.
            position_ids (Array | None): Optional array specifying token positions.
            segment_ids (Array | None): Optional array indicating segment IDs.
            output_attentions (bool | None): Optional flag to return attention weights.
            output_hidden_states (bool | None): Optional flag to return hidden states.
            output_router_logits (bool | None): Optional flag to return router logits.
            past_key_values (Cache | None): Optional cache from previous model passes.
            cache_metadata (Metadata | None): Optional metadata for cache operations.
            loss_config (LossConfig | None): Optional configuration for loss computation.
            loss_kwargs (dict | None): Optional additional keyword arguments for loss.

        Returns:
            tuple[MoeCausalLMOutput, LossMetrics]: Model output and loss metrics.
        """

    @tp.overload
    def compute_loss(
        self,
        *,
        labels: Array | None = None,
        loss_config: LossConfig | None = None,
        loss_kwargs: dict | None = None,
        **batch,
    ) -> tuple[tp.Any, LossMetrics]:
        """basic `compute_loss` call"""

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
        """Converts Model paramters to given dtype"""

    @abstractmethod
    def half(self, change_runtime_dtype: bool = True):
        """Converts Model paramters to float16."""

    @abstractmethod
    def float(self, change_runtime_dtype: bool = True):
        """Converts Model paramters to float32."""

    @abstractmethod
    def _reformat_dtype(self, dtype):
        """Converts Model paramters to given data type."""

    @abstractmethod
    def _get_mesh(self, mesh: Mesh | None = None) -> Mesh:
        """Retrieves the mesh, either from the provided argument or the config."""

    @abstractmethod
    def _get_partition_rules(self, partition_rules: PartitionLike) -> PartitionLike:
        """Retrieves the partition rules from input or the config"""

    @abstractmethod
    def _apply_sharding_fns(self, sharding_fns: tp.Mapping[str, tp.Callable]):
        """Applies sharding functions to the model's state."""

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
        quantization_config: EasyDeLQuantizationConfig | None = None,
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
        """converts current model to a EasyDeLState"""

    @abstractmethod
    def to_torch(self) -> PreTrainedModel:
        """converts current model to a huggingface torch model"""
        ...

    @abstractmethod
    def prepare_inputs_for_call(self, **kwargs):
        """update inputs for calling model"""
        ...

    @abstractmethod
    def get_static_arguments(self) -> tuple:
        """return static arguments kwargs for `jax.jit` / `ejit`"""
        ...

    @classmethod
    @abstractmethod
    def lazy_init(cls: type[Self], *args, **kwargs) -> Self:
        """initialize the base class with nnx.eval_shape carefully"""
        ...

    @abstractmethod
    def apply_lora_to_layers(
        self: Self,
        lora_rank: int,
        lora_pattern: str | None = None,
        verbose: bool = False,
        rngs: nn.Rngs | None = None,
    ) -> Self:
        """Apply LoRA (Low-Rank Adaptation) to specified linear layers within a model."""
        ...

    @abstractmethod
    def merge_lora_params(self: Self, pytree: dict) -> Self:
        """
        Merge LoRA (Low-Rank Adaptation) parameters into the base model parameters.
        """
        ...

    @abstractmethod
    def split_lora_params(self: Self) -> dict:
        """
        Split LoRA (Low-Rank Adaptation) parameters from the base model parameters.
        """
        ...

    @abstractmethod
    def unwrap_lora_to_layers(self, verbose: bool = False):
        """UnWrap LoRA (Low-Rank Adaptation) from specified linear layers within a model."""
        ...

    @property
    @abstractmethod
    def transform_fn(self) -> tp.Callable:
        """generate transform function for converting torch to easydel module."""
        ...

    @property
    @abstractmethod
    def pure_transform_fn(self) -> tp.Callable:
        """generates a pure transform function for converting torch to easydel module."""
        ...

    @property
    @abstractmethod
    def params_sharding(self) -> dict:
        """return the sharding of the model parameters"""
        ...

    @abstractmethod
    def merge_params(self, tree):
        """merge state to the current model"""
        ...

    @abstractmethod
    def split_params(self):
        """split the model parameters"""
        ...

    @abstractmethod
    def split_params_dict(
        self,
        params_dict: dict,
    ) -> dict:
        """Splits the model parameters from a dictionary into separate state components."""

    @abstractmethod
    def merge_params_dict(self, params_dict: dict):
        """
        Merges the model parameters from a dictionary into the current model.
        """

    @abstractmethod
    def _flop(self, *args, **kwargs) -> float | None:
        """Calculates the FLOP (Floating Point Operations) from JaxPr."""
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
        input_ids: Array,
        generation_config: tp.Any | None = None,
        prng_key: Array | None = None,
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
        input_ids: Array,
        max_length: int,
        pad_token_id: int,
        starts: int | None = None,
        shardings: int | None = None,
        attention_mask: Array | None = None,
        token_type_ids: Array | None = None,
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

    def __str__(self):
        return printify_nnx(self)

    def __repr__(self):
        return printify_nnx(self)


def printify_nnx(model):
    try:
        return "EasyDeL-" + prettify_nnx(model)
    except AttributeError:
        return "EasyDeL-Partitions"
