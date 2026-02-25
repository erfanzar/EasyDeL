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

"""Model output classes for EasyDeL.

This module defines standardized output structures for various model types and tasks.
These dataclasses provide consistent interfaces for model outputs while maintaining
compatibility with JAX pytrees for automatic differentiation and JIT compilation.

The output classes follow a hierarchical design where specialized outputs inherit
from base classes, allowing for consistent handling across different model
architectures while preserving model-specific information.

Classes:
    ModelOutput: Base class for all model outputs providing dictionary-like access.
    EmbeddingInfo: Auxiliary outputs from embedding computation for multimodal models.
    AttentionLayerOutput: Output from a single attention layer.
    EncoderLayerOutput: Output from a single encoder layer.
    DecoderLayerOutput: Output from a single decoder layer.
    BaseModelOutput: Basic model output with hidden states and attentions.
    BaseModelOutputWithNoAttention: Model output without attention weights.
    BaseModelOutputWithPoolingAndNoAttention: Output with pooling but no attention.
    ImageClassifierOutputWithNoAttention: Image classification output without attention.
    BaseModelOutputWithPast: Model output with cached key-values for generation.
    BaseModelOutputWithPooling: Output with pooled representations.
    BaseModelOutputWithPoolingAndCrossAttentions: Output with pooling and cross-attention.
    BaseModelOutputWithPastAndCrossAttentions: Output with cache and cross-attention.
    Seq2SeqModelOutput: Output for encoder-decoder models.
    CausalLMOutputWithCrossAttentions: Causal LM output with cross-attention.
    MaskedLMOutput: Output for masked language models.
    CausalLMOutput: Output for causal language models.
    Seq2SeqLMOutput: Output for sequence-to-sequence language models.
    NextSentencePredictorOutput: Output for next sentence prediction.
    SequenceClassifierOutput: Output for sequence classification.
    Seq2SeqSequenceClassifierOutput: Seq2seq sequence classification output.
    MultipleChoiceModelOutput: Output for multiple choice tasks.
    TokenClassifierOutput: Output for token classification.
    QuestionAnsweringModelOutput: Output for question answering.
    Seq2SeqQuestionAnsweringModelOutput: Seq2seq question answering output.
    MoeModelOutput: Output for Mixture-of-Experts models.
    MoeCausalLMOutput: Output for MoE causal language models.
    VLMCausalLMOutput: Output for Vision-Language Models.
    MambaOutput: Output for Mamba state-space models.
    MambaCausalLMOutput: Output for Mamba causal language models.
    CLIPTextModelOutput: Output for CLIP text encoders.
    ImageClassifierOutput: Output for image classification.
    CLIPOutput: Output for CLIP models.
    GreedySearchOutput: Output for greedy generation.
    SampleOutput: Output for sampling generation.
    BeamSearchOutput: Output for beam search generation.

Key Features:
    - Consistent interface across model types
    - JAX pytree compatibility via @auto_pytree decorator
    - Optional fields with None defaults
    - Dictionary-like access patterns (both attribute and key-based)
    - Automatic validation of dataclass fields
    - Immutable item deletion for safety

Example:
    Basic usage with CausalLMOutput::

        >>> from easydel.infra.modeling_outputs import CausalLMOutput
        >>> import jax.numpy as jnp
        >>> logits = jnp.zeros((2, 10, 50000))  # batch=2, seq=10, vocab=50000
        >>> output = CausalLMOutput(
        ...     logits=logits,
        ...     hidden_states=None,
        ...     attentions=None
        ... )
        >>> # Access as attribute
        >>> logits = output.logits
        >>> # Access as dictionary
        >>> logits = output["logits"]
        >>> # Convert to tuple
        >>> output_tuple = output.to_tuple()

Note:
    All output classes must use the @auto_pytree decorator to ensure JAX
    compatibility. The decorator automatically registers the class as a
    pytree node, enabling it to be used with JAX transformations like
    jit, grad, and vmap.
"""

from __future__ import annotations

import typing as tp
from dataclasses import fields, is_dataclass

from eformer.pytree import auto_pytree
from jax.core import Tracer
from jaxtyping import Array

if tp.TYPE_CHECKING:
    from easydel.caching import TransformerCache, TransformerCacheView
else:
    TransformerCacheView = tp.Any
    TransformerCache = tp.Any


def _is_array(array):
    """Check if an object is a JAX array or tracer.

    This function is used internally to detect JAX arrays and tracers
    during output validation. Tracers are returned during JAX
    transformations like jit and grad.

    Args:
        array: The object to check.

    Returns:
        bool: True if the object is a JAX Tracer, False otherwise.

    Note:
        This function currently only checks for Tracer instances.
        Regular JAX arrays (DeviceArray) return False, which may be
        intentional for the validation logic in ModelOutput.__post_init__.
    """
    if isinstance(array, Tracer):
        return True
    return False


class ModelOutput(tp.OrderedDict):
    """Base class for all model outputs.

    Provides a consistent interface for model outputs that behaves like
    both a tuple (for positional access) and a dictionary (for named access).
    Automatically filters out None values and provides validation.

    This class inherits from OrderedDict to maintain insertion order of
    fields while providing both attribute-style (output.logits) and
    dictionary-style (output["logits"]) access patterns.

    Subclasses must use the @auto_pytree decorator to ensure JAX compatibility
    for JIT compilation and automatic differentiation.

    Attributes:
        All subclass fields are accessible as both attributes and dictionary keys.

    Example:
        Creating a custom output class::

            >>> @auto_pytree
            ... class CustomOutput(ModelOutput):
            ...     predictions: Array
            ...     features: Array | None = None
            ...
            >>> output = CustomOutput(predictions=jnp.array([1, 2, 3]))
            >>> output.predictions  # Attribute access
            Array([1, 2, 3], dtype=int32)
            >>> output["predictions"]  # Dictionary access
            Array([1, 2, 3], dtype=int32)

    Note:
        - All fields except the first should have None as default.
        - Item deletion, setdefault, pop, and update operations are disabled
          to maintain output immutability.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the ModelOutput instance.

        Validates that subclasses are properly decorated with @auto_pytree.

        Args:
            *args: Positional arguments passed to OrderedDict.
            **kwargs: Keyword arguments passed to OrderedDict.

        Raises:
            TypeError: If a subclass is not decorated with @auto_pytree,
                which makes it not a dataclass.
        """
        super().__init__(*args, **kwargs)
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @auto_pytree decorator."
            )

    def to_tuple(self) -> tuple[tp.Any]:
        """Convert the output to a tuple containing all non-None values.

        This method is useful for unpacking model outputs or passing them
        to functions that expect tuple inputs.

        Returns:
            tuple[Any]: A tuple containing all values in the output dictionary.
                The order matches the insertion order of fields.

        Example:
            >>> output = CausalLMOutput(logits=logits, hidden_states=hidden_states)
            >>> logits, hidden_states = output.to_tuple()
        """
        return tuple(self[k] for k in self.keys())

    def __post_init__(self):
        """Validate and initialize the ModelOutput dataclass.

        This method is called automatically after dataclass initialization
        when the @auto_pytree decorator is used. It performs validation
        and populates the underlying OrderedDict.

        The method handles several initialization patterns:
        1. Standard field initialization with explicit values
        2. Dictionary-style initialization from an iterable of (key, value) pairs
        3. Single-field initialization when only the first field is provided

        Raises:
            ValueError: If the dataclass has no fields.
            ValueError: If more than one required field is defined.
            ValueError: If an iterable element cannot be converted to (key, value).
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not _is_array(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:
                            raise ValueError(f"Cannot set key/value for {element}. It needs to be a tuple (key, value).")
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        """Prevent item deletion to maintain output immutability.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Exception: Always raised to prevent deletion.
        """
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        """Prevent setdefault to maintain output immutability.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Exception: Always raised to prevent setdefault.
        """
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        """Prevent pop to maintain output immutability.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Exception: Always raised to prevent pop.
        """
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        """Prevent update to maintain output immutability.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            Exception: Always raised to prevent update.
        """
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        """Get an item by string key or integer index.

        Supports both dictionary-style access with string keys and
        tuple-style access with integer indices.

        Args:
            k: Either a string key or an integer index.

        Returns:
            The value associated with the key or at the index.

        Raises:
            KeyError: If a string key is not found.
            IndexError: If an integer index is out of range.

        Example:
            >>> output["logits"]  # Dictionary access
            >>> output[0]  # Tuple-style access (first element)
        """
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        """Set an attribute and sync with the underlying dictionary.

        Ensures that attribute assignments are reflected in the
        OrderedDict for consistency between access patterns.

        Args:
            name: The attribute name.
            value: The value to set.
        """
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        """Set an item and sync with attributes.

        Ensures that dictionary assignments are reflected as
        attributes for consistency between access patterns.

        Args:
            key: The dictionary key.
            value: The value to set.

        Raises:
            KeyError: If the key is not a valid field name.
        """
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        """Support pickling of ModelOutput instances.

        Handles serialization for both dataclass-decorated and
        plain ModelOutput instances.

        Returns:
            tuple: A tuple suitable for pickle reconstruction containing
                the constructor function and arguments.
        """
        if not is_dataclass(self):
            return super().__reduce__()
        _fn, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return _fn, args, *remaining


@auto_pytree
class EmbeddingInfo:
    """Auxiliary outputs produced while computing input embeddings.

    This class is primarily used by multimodal models to return extra tensors
    (e.g., DeepStack visual features, mRoPE indices) alongside `inputs_embeds`
    without forcing callers to recompute them.

    This is particularly useful for Vision-Language Models (VLMs) that need
    to pass additional positional or visual information through the model
    alongside the main embeddings.

    Attributes:
        position_ids: Position IDs for the tokens. Shape varies by model.
            May be modified for multimodal inputs.
        rope_deltas: Delta values for multi-dimensional RoPE computation.
            Used in models like Qwen2-VL for spatial-temporal positional encoding.
        visual_pos_masks: Masks indicating positions of visual tokens in the
            sequence. Shape: (batch_size, sequence_length).
        deepstack_visual_embeds: List of visual embeddings at different layers
            for DeepStack architectures.
        deepstack_image_embeds: List of image-specific embeddings for DeepStack.
        deepstack_video_embeds: List of video-specific embeddings for DeepStack.

    Example:
        >>> embedding_info = EmbeddingInfo(
        ...     position_ids=position_ids,
        ...     rope_deltas=rope_deltas
        ... )
        >>> # Pass to model forward
        >>> outputs = model(inputs_embeds=embeds, embedding_info=embedding_info)
    """

    position_ids: Array | None = None
    rope_deltas: Array | None = None
    visual_pos_masks: Array | None = None
    deepstack_visual_embeds: list[Array] | None = None
    deepstack_image_embeds: list[Array] | None = None
    deepstack_video_embeds: list[Array] | None = None


@auto_pytree
class AttentionLayerOutput(ModelOutput):
    """Output from a single attention layer.

    Contains the attention computation results from a transformer attention layer,
    including optional attention weights and cache views for efficient generation.

    This output is typically returned by individual attention modules within
    transformer layers and can be used to inspect attention patterns or
    for efficient autoregressive decoding.

    Attributes:
        attention_output: Output tensor from the attention layer with shape
            (batch_size, sequence_length, hidden_size).
        attention_weight: Optional attention weights after softmax with shape
            (batch_size, num_heads, sequence_length, sequence_length).
            Only returned when output_attentions=True.
        cache_view: Optional cache view for efficient autoregressive generation.
            Contains cached key-value pairs from previous steps.

    Example:
        >>> attn_output = AttentionLayerOutput(
        ...     attention_output=jnp.zeros((2, 10, 768)),
        ...     attention_weight=jnp.zeros((2, 12, 10, 10))
        ... )
        >>> print(attn_output.attention_output.shape)
        (2, 10, 768)
    """

    attention_output: Array
    attention_weight: Array | None = None
    cache_view: TransformerCacheView | None = None


@auto_pytree
class EncoderLayerOutput(ModelOutput):
    """Output from a single encoder layer.

    Contains the outputs from a transformer encoder layer, including
    the processed hidden states and optional attention weights.

    Encoder layers typically apply self-attention followed by feed-forward
    processing to transform input representations.

    Attributes:
        hidden_states: Output hidden states from the encoder layer with shape
            (batch_size, sequence_length, hidden_size).
        residual_states: Optional residual connection states before layer norm
            with shape (batch_size, sequence_length, hidden_size). Useful for
            certain normalization strategies.
        attention_weight: Optional attention weights after softmax with shape
            (batch_size, num_heads, sequence_length, sequence_length).
            Only returned when output_attentions=True.

    Example:
        >>> encoder_output = EncoderLayerOutput(
        ...     hidden_states=jnp.zeros((2, 10, 768)),
        ...     attention_weight=jnp.zeros((2, 12, 10, 10))
        ... )
    """

    hidden_states: Array
    residual_states: Array | None = None
    attention_weight: Array | None = None


@auto_pytree
class DecoderLayerOutput(ModelOutput):
    """Output from a single decoder layer.

    Contains the outputs from a transformer decoder layer, including
    hidden states, attention weights, and optional MoE routing information.

    Decoder layers may include self-attention, cross-attention (for
    encoder-decoder models), and feed-forward processing. For MoE models,
    router logits and auxiliary losses are also tracked.

    Attributes:
        hidden_states: Output hidden states from the decoder layer with shape
            (batch_size, sequence_length, hidden_size).
        residual_states: Optional residual connection states before layer norm
            with shape (batch_size, sequence_length, hidden_size).
        cross_attention: Optional cross-attention outputs when using encoder-decoder
            architecture with shape (batch_size, sequence_length, hidden_size).
        attention_weight: Optional self-attention weights after softmax with shape
            (batch_size, num_heads, sequence_length, sequence_length).
        router_logits: Optional MoE router logits for expert selection with shape
            (batch_size, sequence_length, num_experts). Used for MoE models.
        gate_loss: Optional auxiliary loss for MoE load balancing. Scalar value
            used to encourage balanced expert utilization.
        cache_view: Optional cache view for efficient autoregressive generation.
            Contains cached key-value pairs from previous steps.

    Example:
        >>> decoder_output = DecoderLayerOutput(
        ...     hidden_states=jnp.zeros((2, 10, 768)),
        ...     attention_weight=jnp.zeros((2, 12, 10, 10)),
        ...     router_logits=jnp.zeros((2, 10, 8))  # 8 experts
        ... )
    """

    hidden_states: Array
    residual_states: Array | None = None
    cross_attention: Array | None = None
    attention_weight: Array | None = None
    router_logits: Array | None = None
    gate_loss: Array | None = None
    cache_view: TransformerCacheView | None = None


@auto_pytree
class BaseModelOutput(ModelOutput):
    """Base class for model outputs with potential hidden states and attentions.

    This is a fundamental output class used by many transformer models. It provides
    the basic structure for returning the model's final hidden states along with
    optional intermediate representations.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size). Hidden-states of the
            model at the output of each layer plus the initial embedding outputs.
            Only returned when output_hidden_states=True.
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights after the attention softmax, used to compute
            the weighted average in the self-attention heads.
            Only returned when output_attentions=True.
        past_key_values: Dictionary of pre-computed hidden-states (key and values
            in the attention blocks) that can be used for fast auto-regressive
            decoding.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutput(
        ...     last_hidden_state=jnp.zeros((2, 10, 768)),
        ...     hidden_states=tuple(jnp.zeros((2, 10, 768)) for _ in range(13)),
        ...     attentions=tuple(jnp.zeros((2, 12, 10, 10)) for _ in range(12))
        ... )
    """

    last_hidden_state: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    past_key_values: dict[str, Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithNoAttention(ModelOutput):
    """Base class for model outputs without attention weights.

    Used for models where attention weights are not tracked or for efficiency
    when attention patterns are not needed. Common in vision models where
    the output shape differs from sequence models.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. For vision models: (batch_size, num_channels,
            height, width). For sequence models: (batch_size, sequence_length,
            hidden_size).
        hidden_states: Tuple of Arrays (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer). Shape matches last_hidden_state. Hidden-states of the model
            at the output of each layer plus the optional initial embedding outputs.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithNoAttention(
        ...     last_hidden_state=jnp.zeros((2, 768, 14, 14)),  # Vision format
        ...     hidden_states=None
        ... )
    """

    last_hidden_state: Array = None
    hidden_states: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """Base class for model outputs with pooling but without attention weights.

    Adds a pooler output for tasks requiring a fixed-size representation
    of variable-length inputs, commonly used in vision models for classification.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, num_channels, height, width)
            for vision models.
        pooler_output: Last layer hidden-state after a pooling operation on
            the spatial dimensions. Shape: (batch_size, hidden_size). This
            provides a fixed-size representation regardless of input dimensions.
        hidden_states: Tuple of Arrays (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer). Hidden-states of the model at the output of each layer plus
            the optional initial embedding outputs.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithPoolingAndNoAttention(
        ...     last_hidden_state=jnp.zeros((2, 768, 14, 14)),
        ...     pooler_output=jnp.zeros((2, 768))
        ... )
    """

    last_hidden_state: Array = None
    pooler_output: Array = None
    hidden_states: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """Output class for image classification models without attention weights.

    Provides classification logits along with optional hidden states
    for image classification tasks.

    Attributes:
        logits: Classification (or regression if config.num_labels==1) scores
            before SoftMax. Shape: (batch_size, config.num_labels).
        hidden_states: Tuple of Arrays (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            stage). Shape: (batch_size, num_channels, height, width). Hidden-states
            (also called feature maps) of the model at the output of each stage.
            Only returned when output_hidden_states=True.
        loss: Optional classification loss when labels are provided.

    Example:
        >>> output = ImageClassifierOutputWithNoAttention(
        ...     logits=jnp.zeros((2, 1000)),  # 1000 ImageNet classes
        ...     hidden_states=None,
        ...     loss=jnp.array(0.5)
        ... )
    """

    logits: Array = None
    hidden_states: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithPast(ModelOutput):
    """Base class for model outputs with cached key-values for efficient generation.

    Extends BaseModelOutput with past_key_values for autoregressive decoding,
    allowing the model to reuse previous computations during generation.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        past_key_values: Dictionary of pre-computed hidden-states (key and values
            in the attention blocks) that can be used for fast auto-regressive
            decoding. Pre-computed key and value hidden-states are of shape
            [batch_size, max_length].
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size). Hidden-states of the
            model at the output of each layer plus the initial embedding outputs.
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights after the attention softmax.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithPast(
        ...     last_hidden_state=jnp.zeros((2, 1, 768)),  # Single new token
        ...     past_key_values=cache_dict
        ... )
    """

    last_hidden_state: Array = None
    past_key_values: dict[str, Array] | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithPooling(ModelOutput):
    """Base class for model outputs with pooled representations.

    Adds a pooler output for tasks requiring a single vector representation
    of the input sequence, such as classification tasks using [CLS] tokens.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        pooler_output: Last layer hidden-state of the first token of the sequence
            (classification token) further processed by a Linear layer and a
            Tanh activation function. Shape: (batch_size, hidden_size). The
            Linear layer weights are trained from the next sentence prediction
            (classification) objective during pretraining.
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithPooling(
        ...     last_hidden_state=jnp.zeros((2, 10, 768)),
        ...     pooler_output=jnp.zeros((2, 768))
        ... )
    """

    last_hidden_state: Array = None
    pooler_output: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """Base class for model outputs with pooling and cross-attention.

    Combines pooled representations with cross-attention capabilities,
    typically used in encoder-decoder architectures where the decoder
    attends to encoder outputs.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        pooler_output: Last layer hidden-state of the first token of the sequence
            (classification token) after further processing through the layers
            used for the auxiliary pretraining task. Shape: (batch_size, hidden_size).
            For BERT-family models, this returns the classification token after
            processing through a linear layer and a tanh activation function.
        hidden_states: Tuple of Arrays (one for the output of the embeddings,
            if the model has an embedding layer, + one for the output of each
            layer). Shape of each: (batch_size, sequence_length, hidden_size).
        past_key_values: Cached key-values for efficient generation.
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        cross_attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights of the decoder's cross-attention layer.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithPoolingAndCrossAttentions(
        ...     last_hidden_state=jnp.zeros((2, 10, 768)),
        ...     pooler_output=jnp.zeros((2, 768)),
        ...     cross_attentions=tuple(jnp.zeros((2, 12, 10, 20)) for _ in range(12))
        ... )
    """

    last_hidden_state: Array = None
    pooler_output: Array = None
    hidden_states: tuple[Array] | None = None
    past_key_values: TransformerCache | None = None
    attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """Base class for outputs with cached key-values and cross-attention.

    Used in encoder-decoder models during autoregressive generation,
    where both self-attention caching and cross-attention are needed.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. If past_key_values is used, only the last
            hidden-state of the sequences of shape (batch_size, 1, hidden_size)
            is output.
        past_key_values: Tuple of tuples of Arrays of length config.n_layers,
            with each tuple having 2 tensors of shape (batch_size, num_heads,
            sequence_length, embed_size_per_head) and optionally if
            config.is_encoder_decoder=True, 2 additional tensors of shape
            (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
            Contains pre-computed hidden-states for fast sequential decoding.
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        cross_attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights of the decoder's cross-attention layer.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = BaseModelOutputWithPastAndCrossAttentions(
        ...     last_hidden_state=jnp.zeros((2, 1, 768)),
        ...     past_key_values=cache
        ... )
    """

    last_hidden_state: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class Seq2SeqModelOutput(ModelOutput):
    """Output class for sequence-to-sequence encoder-decoder models.

    Contains outputs from both encoder and decoder components, including
    cached key-values for efficient generation.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the decoder of the model. If past_key_values is used, only
            the last hidden-state of shape (batch_size, 1, hidden_size) is output.
        past_key_values: Tuple of tuples of Arrays of length config.n_layers,
            with each tuple having 2 tensors of shape (batch_size, num_heads,
            sequence_length, embed_size_per_head) and 2 additional tensors of
            shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
            Contains pre-computed hidden-states for fast sequential decoding.
        decoder_hidden_states: Tuple of Arrays (one for the output of the
            embeddings + one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size). Hidden-states of the
            decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights of the decoder.
        cross_attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Attentions weights of the decoder's cross-attention layer.
        encoder_last_hidden_state: Sequence of hidden-states at the output of
            the last layer of the encoder. Shape:
            (batch_size, sequence_length, hidden_size).
        encoder_hidden_states: Tuple of Arrays (one for the output of the
            embeddings + one for the output of each layer). Hidden-states of
            the encoder at the output of each layer.
        encoder_attentions: Tuple of Arrays (one for each layer).
            Attentions weights of the encoder.
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = Seq2SeqModelOutput(
        ...     last_hidden_state=decoder_output,
        ...     encoder_last_hidden_state=encoder_output
        ... )
    """

    last_hidden_state: Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[Array] | None = None
    decoder_attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    encoder_last_hidden_state: Array | None = None
    encoder_hidden_states: tuple[Array] | None = None
    encoder_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """Output class for causal language models with cross-attention support.

    Used in decoder models that can attend to encoder outputs, such as
    in encoder-decoder architectures configured for causal generation.

    Attributes:
        logits: Prediction scores of the language modeling head (scores for
            each vocabulary token before SoftMax). Shape:
            (batch_size, sequence_length, config.vocab_size).
        past_key_values: Tuple of tuples of Arrays containing cached key-values
            for both self-attention and cross-attention blocks. Can be used
            to speed up sequential decoding.
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        cross_attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
            Cross attentions weights after the attention softmax.
        loss: Optional language modeling loss when labels are provided.

    Example:
        >>> output = CausalLMOutputWithCrossAttentions(
        ...     logits=jnp.zeros((2, 10, 50000)),
        ...     past_key_values=cache
        ... )
    """

    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class MaskedLMOutput(ModelOutput):
    """Output class for masked language models.

    Used for models trained with masked language modeling objectives,
    such as BERT, RoBERTa, and similar architectures.

    Attributes:
        logits: Prediction scores of the language modeling head (scores for
            each vocabulary token before SoftMax). Shape:
            (batch_size, sequence_length, config.vocab_size).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        last_hidden_state: Hidden states from the last layer. Shape:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        past_key_values: Cached key-values for efficient generation.
        loss: Optional masked language modeling loss when labels are provided.

    Example:
        >>> output = MaskedLMOutput(
        ...     logits=jnp.zeros((2, 10, 30000)),
        ...     loss=jnp.array(2.5)
        ... )
    """

    logits: Array | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None
    attentions: tuple[Array] | None = None
    past_key_values: TransformerCache | None = None
    loss: Array | None = None


@auto_pytree
class CausalLMOutput(MaskedLMOutput):
    """Output class for causal language models.

    Extends MaskedLMOutput for causal (autoregressive) language models
    like GPT, LLaMA, and similar decoder-only architectures.

    This class inherits all attributes from MaskedLMOutput and can be
    used interchangeably where appropriate. The main distinction is
    semantic - CausalLMOutput is used for models that generate text
    left-to-right.

    Attributes:
        logits: Prediction scores of the language modeling head (scores for
            each vocabulary token before SoftMax). Shape:
            (batch_size, sequence_length, config.vocab_size).
        hidden_states: Tuple of Arrays from all layers.
        last_hidden_state: Hidden states from the last layer.
        attentions: Tuple of attention weights from all layers.
        past_key_values: Cached key-values for efficient generation.
        loss: Optional causal language modeling loss.

    Example:
        >>> output = CausalLMOutput(
        ...     logits=jnp.zeros((2, 10, 50000)),
        ...     past_key_values=cache,
        ...     loss=jnp.array(3.2)
        ... )
    """

    ...


@auto_pytree
class Seq2SeqLMOutput(ModelOutput):
    """Output class for sequence-to-sequence language models.

    Used for encoder-decoder models configured for language modeling tasks,
    such as T5, BART, and machine translation models.

    Attributes:
        logits: Prediction scores of the language modeling head (scores for
            each vocabulary token before SoftMax). Shape:
            (batch_size, sequence_length, config.vocab_size).
        past_key_values: Tuple of tuples of Arrays of length config.n_layers,
            with each tuple having 2 tensors of shape (batch_size, num_heads,
            sequence_length, embed_size_per_head) and 2 additional tensors of
            shape (batch_size, num_heads, encoder_sequence_length, embed_size_per_head).
            Contains pre-computed hidden-states for fast sequential decoding.
        decoder_hidden_states: Tuple of Arrays (one for the output of the
            embeddings + one for the output of each layer). Hidden-states of
            the decoder at the output of each layer.
        decoder_attentions: Tuple of Arrays (one for each layer).
            Attentions weights of the decoder.
        cross_attentions: Tuple of Arrays (one for each layer).
            Attentions weights of the decoder's cross-attention layer.
        encoder_last_hidden_state: Sequence of hidden-states at the output
            of the last layer of the encoder.
        encoder_hidden_states: Tuple of Arrays. Hidden-states of the encoder
            at the output of each layer.
        encoder_attentions: Tuple of Arrays. Attentions weights of the encoder.
        loss: Optional sequence-to-sequence language modeling loss.

    Example:
        >>> output = Seq2SeqLMOutput(
        ...     logits=jnp.zeros((2, 20, 50000)),
        ...     encoder_last_hidden_state=jnp.zeros((2, 10, 768))
        ... )
    """

    logits: Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[Array] | None = None
    decoder_attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    encoder_last_hidden_state: Array | None = None
    encoder_hidden_states: tuple[Array] | None = None
    encoder_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class NextSentencePredictorOutput(ModelOutput):
    """Output class for next sentence prediction models.

    Used for models trained with next sentence prediction objectives,
    a common pretraining task in BERT-style models.

    Attributes:
        logits: Prediction scores of the next sequence prediction (classification)
            head (scores of True/False continuation before SoftMax). Shape:
            (batch_size, 2) where index 0 is "not next" and index 1 is "next".
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        loss: Optional next sentence prediction loss when labels are provided.

    Example:
        >>> output = NextSentencePredictorOutput(
        ...     logits=jnp.array([[0.2, 0.8], [0.9, 0.1]]),  # 2 samples
        ...     loss=jnp.array(0.35)
        ... )
    """

    logits: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class SequenceClassifierOutput(ModelOutput):
    """Output class for sequence classification models.

    Used for models performing classification tasks on sequences, such as
    sentiment analysis, text classification, and natural language inference.

    Attributes:
        logits: Classification (or regression if config.num_labels==1) scores
            before SoftMax. Shape: (batch_size, config.num_labels).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        past_key_values: Cached key-values for efficient generation.
        loss: Optional classification loss when labels are provided.
        aux_loss: Optional auxiliary loss for models with additional objectives
            (e.g., MoE routing loss).

    Example:
        >>> output = SequenceClassifierOutput(
        ...     logits=jnp.array([[0.1, 0.9], [0.8, 0.2]]),  # Binary classification
        ...     loss=jnp.array(0.42)
        ... )
    """

    logits: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    past_key_values: TransformerCache | None = None
    loss: Array | None = None
    aux_loss: Array | None = None


@auto_pytree
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    """Output class for sequence-to-sequence sequence classification models.

    Used for encoder-decoder models performing classification tasks,
    combining the full seq2seq architecture with classification heads.

    Attributes:
        logits: Classification (or regression if config.num_labels==1) scores
            before SoftMax. Shape: (batch_size, config.num_labels).
        past_key_values: Tuple of tuples of Arrays for cached key-values.
        decoder_hidden_states: Tuple of Arrays. Hidden-states of the decoder.
        decoder_attentions: Tuple of Arrays. Attentions weights of the decoder.
        cross_attentions: Tuple of Arrays. Cross-attention weights.
        encoder_last_hidden_state: Hidden-states at the output of the encoder's
            last layer. Shape: (batch_size, sequence_length, hidden_size).
        encoder_hidden_states: Tuple of Arrays. Hidden-states of the encoder.
        encoder_attentions: Tuple of Arrays. Attentions weights of the encoder.
        loss: Optional classification loss when labels are provided.

    Example:
        >>> output = Seq2SeqSequenceClassifierOutput(
        ...     logits=jnp.zeros((2, 3)),  # 3-class classification
        ...     encoder_last_hidden_state=jnp.zeros((2, 10, 768))
        ... )
    """

    logits: Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[Array] | None = None
    decoder_attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    encoder_last_hidden_state: Array | None = None
    encoder_hidden_states: tuple[Array] | None = None
    encoder_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class MultipleChoiceModelOutput(ModelOutput):
    """Output class for multiple choice models.

    Used for tasks where the model must select one answer from multiple
    options, such as SWAG, HellaSwag, or multiple choice reading comprehension.

    Attributes:
        logits: Classification scores before SoftMax. Shape:
            (batch_size, num_choices) where num_choices is the second dimension
            of the input tensors (see input_ids).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        loss: Optional multiple choice loss when labels are provided.

    Example:
        >>> output = MultipleChoiceModelOutput(
        ...     logits=jnp.zeros((2, 4)),  # 4 choices per sample
        ...     loss=jnp.array(1.2)
        ... )
    """

    logits: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class TokenClassifierOutput(ModelOutput):
    """Output class for token classification models.

    Used for tasks requiring per-token predictions, such as named entity
    recognition (NER), part-of-speech tagging, and chunking.

    Attributes:
        logits: Classification scores before SoftMax. Shape:
            (batch_size, sequence_length, config.num_labels).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        loss: Optional token classification loss when labels are provided.

    Example:
        >>> output = TokenClassifierOutput(
        ...     logits=jnp.zeros((2, 10, 9)),  # 9 NER tags
        ...     loss=jnp.array(0.85)
        ... )
    """

    logits: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class QuestionAnsweringModelOutput(ModelOutput):
    """Output class for extractive question answering models.

    Used for models that extract answer spans from context, such as
    SQuAD-style question answering.

    Attributes:
        start_logits: Span-start scores before SoftMax. Shape:
            (batch_size, sequence_length). Higher scores indicate more likely
            answer start positions.
        end_logits: Span-end scores before SoftMax. Shape:
            (batch_size, sequence_length). Higher scores indicate more likely
            answer end positions.
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        loss: Optional question answering loss when start/end positions are provided.

    Example:
        >>> output = QuestionAnsweringModelOutput(
        ...     start_logits=jnp.zeros((2, 512)),
        ...     end_logits=jnp.zeros((2, 512)),
        ...     loss=jnp.array(2.1)
        ... )
    """

    start_logits: Array = None
    end_logits: Array = None
    hidden_states: tuple[Array] | None = None
    attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """Output class for sequence-to-sequence question answering models.

    Used for encoder-decoder models performing extractive question answering,
    combining full seq2seq architecture with span extraction.

    Attributes:
        start_logits: Span-start scores before SoftMax. Shape:
            (batch_size, sequence_length).
        end_logits: Span-end scores before SoftMax. Shape:
            (batch_size, sequence_length).
        past_key_values: Tuple of tuples of Arrays for cached key-values.
        decoder_hidden_states: Tuple of Arrays. Hidden-states of the decoder.
        decoder_attentions: Tuple of Arrays. Attentions weights of the decoder.
        cross_attentions: Tuple of Arrays. Cross-attention weights.
        encoder_last_hidden_state: Hidden-states at the output of the encoder's
            last layer. Shape: (batch_size, sequence_length, hidden_size).
        encoder_hidden_states: Tuple of Arrays. Hidden-states of the encoder.
        encoder_attentions: Tuple of Arrays. Attentions weights of the encoder.
        loss: Optional question answering loss when start/end positions are provided.

    Example:
        >>> output = Seq2SeqQuestionAnsweringModelOutput(
        ...     start_logits=jnp.zeros((2, 512)),
        ...     end_logits=jnp.zeros((2, 512)),
        ...     encoder_last_hidden_state=jnp.zeros((2, 512, 768))
        ... )
    """

    start_logits: Array = None
    end_logits: Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[Array] | None = None
    decoder_attentions: tuple[Array] | None = None
    cross_attentions: tuple[Array] | None = None
    encoder_last_hidden_state: Array | None = None
    encoder_hidden_states: tuple[Array] | None = None
    encoder_attentions: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class MoeModelOutput(ModelOutput):
    """Output class for Mixture-of-Experts (MoE) models.

    Extends base model output with MoE-specific fields for router logits
    and auxiliary losses used in load balancing.

    Attributes:
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        past_key_values: Cached key-values for efficient generation.
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        router_logits: Tuple of Arrays (one for each MoE layer). Shape of each:
            (batch_size, sequence_length, num_experts). The logits output of
            the router network, used to compute the mixture of experts.
        all_router_losses: Tuple of Arrays containing per-layer router losses
            for load balancing.
        logits: Optional prediction scores for language modeling. Shape:
            (batch_size, sequence_length, vocab_size).
        loss: Optional loss value when labels are provided during training.

    Example:
        >>> output = MoeModelOutput(
        ...     last_hidden_state=jnp.zeros((2, 10, 768)),
        ...     router_logits=tuple(jnp.zeros((2, 10, 8)) for _ in range(12))
        ... )
    """

    last_hidden_state: Array = None
    hidden_states: tuple[Array] | None = None
    past_key_values: TransformerCache | None = None
    attentions: tuple[Array] | None = None
    router_logits: tuple[Array] | None = None
    all_router_losses: tuple[Array] | None = None
    logits: Array = None
    loss: Array | None = None


@auto_pytree
class MoeCausalLMOutput(MaskedLMOutput):
    """Output class for Mixture-of-Experts causal language models.

    Extends MaskedLMOutput with MoE-specific fields for models like
    Mixtral, Switch Transformer, and GLaM.

    Attributes:
        logits: Prediction scores of the language modeling head. Shape:
            (batch_size, sequence_length, config.vocab_size). Inherited from
            MaskedLMOutput.
        hidden_states: Tuple of Arrays from all layers. Inherited.
        last_hidden_state: Hidden states from the last layer. Inherited.
        attentions: Tuple of attention weights. Inherited.
        past_key_values: Cached key-values for generation. Inherited.
        aux_loss: Auxiliary loss used for training MoE models. Scalar value
            combining load balancing and other MoE-specific losses.
        router_logits: Tuple of Arrays (one for each MoE layer). Shape of each:
            (batch_size, sequence_length, num_experts). The logits output of
            the router network.
        all_router_losses: Tuple of Arrays containing per-layer router losses.
        loss: Optional language modeling loss when labels are provided.

    Example:
        >>> output = MoeCausalLMOutput(
        ...     logits=jnp.zeros((2, 10, 50000)),
        ...     aux_loss=jnp.array(0.01),
        ...     router_logits=tuple(jnp.zeros((2, 10, 8)) for _ in range(12))
        ... )
    """

    aux_loss: Array | None = None
    router_logits: tuple[Array] | None = None
    all_router_losses: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class VLMCausalLMOutput(ModelOutput):
    """Unified output class for Vision-Language Models (VLMs).

    Provides a standardized output structure for all VLM models including
    LLaVA, Qwen2-VL, Qwen3-VL, Gemma3, AyaVision, Mistral3, and Llama4.

    This class combines language model outputs with vision-specific features
    like image embeddings, video embeddings, and multi-dimensional position
    encoding deltas (mRoPE).

    Attributes:
        logits: Prediction scores of the language modeling head before SoftMax.
            Shape: (batch_size, sequence_length, config.vocab_size).
        past_key_values: Pre-computed hidden-states (key and values in attention
            blocks) for efficient autoregressive generation.
        hidden_states: Tuple of hidden-states at output of each layer plus
            embeddings. Shape of each: (batch_size, sequence_length, hidden_size).
        last_hidden_state: Hidden-state at output of the last layer.
            Shape: (batch_size, sequence_length, hidden_size).
        attentions: Attention weights after softmax. Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).
        image_hidden_states: Projected image features from the vision encoder
            after the multimodal projector. Shape varies by model architecture.
        video_hidden_states: Projected video features for models supporting
            video input (Qwen2-VL, Qwen3-VL, Llama4). Shape varies by model.
        rope_deltas: Position embedding deltas for multi-dimensional RoPE (mRoPE)
            used in Qwen2-VL and Qwen3-VL models for spatial-temporal encoding.
        router_logits: Router logits for MoE VLMs (Qwen3-VL-MoE). Shape:
            (batch_size, sequence_length, num_experts).
        aux_loss: Auxiliary loss for MoE load balancing.
        loss: Language modeling loss when labels are provided.

    Example:
        >>> output = VLMCausalLMOutput(
        ...     logits=jnp.zeros((2, 100, 50000)),
        ...     image_hidden_states=jnp.zeros((2, 576, 4096)),
        ...     rope_deltas=jnp.zeros((2, 3, 100))  # 3D position for mRoPE
        ... )
    """

    logits: Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[Array] | None = None
    last_hidden_state: Array | None = None
    attentions: tuple[Array] | None = None
    image_hidden_states: Array | None = None
    video_hidden_states: Array | None = None
    rope_deltas: Array | None = None
    router_logits: tuple[Array] | None = None
    aux_loss: Array | None = None
    loss: Array | None = None


@auto_pytree
class MambaOutput(BaseModelOutput):
    """Output class for Mamba state-space models.

    Contains the outputs from Mamba models which use selective state-space
    layers instead of attention for sequence modeling. Mamba models achieve
    linear complexity with sequence length while maintaining strong performance.

    Attributes:
        last_hidden_state: Final hidden states from the model. Shape:
            (batch_size, sequence_length, hidden_size). Inherited from
            BaseModelOutput.
        cache_params: Optional list of cached state-space parameters for
            efficient autoregressive generation. Each element contains the
            SSM state for a layer, enabling efficient incremental inference.
        hidden_states: Optional tuple of hidden states from all layers.
            Only returned when output_hidden_states=True. Inherited.
        loss: Optional loss value when labels are provided. Inherited.

    Example:
        >>> output = MambaOutput(
        ...     last_hidden_state=jnp.zeros((2, 10, 768)),
        ...     cache_params=[jnp.zeros((2, 16, 768)) for _ in range(24)]
        ... )
    """

    last_hidden_state: Array = None
    cache_params: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class MambaCausalLMOutput(BaseModelOutput):
    """Output class for Mamba causal language models.

    Contains the outputs from Mamba models configured for causal language
    modeling, including logits over the vocabulary for next-token prediction.

    Attributes:
        logits: Prediction scores over the vocabulary. Shape:
            (batch_size, sequence_length, vocab_size). Used for computing
            cross-entropy loss or sampling next tokens.
        cache_params: Optional list of cached state-space parameters for
            efficient autoregressive generation. Each element contains the
            SSM state for a layer.
        hidden_states: Optional tuple of hidden states from all layers.
            Only returned when output_hidden_states=True.
        loss: Optional language modeling loss when labels are provided.

    Example:
        >>> output = MambaCausalLMOutput(
        ...     logits=jnp.zeros((2, 10, 50000)),
        ...     cache_params=ssm_cache,
        ...     loss=jnp.array(2.8)
        ... )
    """

    logits: Array = None
    cache_params: list[Array] | None = None
    hidden_states: tuple[Array] | None = None
    loss: Array | None = None


@auto_pytree
class CLIPTextModelOutput(ModelOutput):
    """Output class for CLIP text encoder models.

    Contains the text embeddings and hidden states from CLIP's text encoder,
    used for computing text-image similarity.

    Attributes:
        text_embeds: The text embeddings obtained by applying the projection
            layer to the pooled output of the text model. Shape:
            (batch_size, output_dim). These are the final text representations
            used for computing similarity with image embeddings.
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size).
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape of each:
            (batch_size, sequence_length, hidden_size).
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).

    Example:
        >>> output = CLIPTextModelOutput(
        ...     text_embeds=jnp.zeros((2, 512)),
        ...     last_hidden_state=jnp.zeros((2, 77, 768))
        ... )
    """

    text_embeds: Array = None
    last_hidden_state: Array = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@auto_pytree
class ImageClassifierOutput(ModelOutput):
    """Output class for image classification models.

    Contains embeddings and hidden states for image classification tasks.
    Despite the name similarity to ImageClassifierOutputWithNoAttention,
    this class includes attention weights.

    Attributes:
        text_embeds: The embeddings obtained by applying the projection layer
            to the pooled output of the model. Shape: (batch_size, output_dim).
            Note: Named text_embeds for CLIP compatibility but contains image
            embeddings for classification.
        last_hidden_state: Sequence of hidden-states at the output of the last
            layer of the model. Shape: (batch_size, sequence_length, hidden_size)
            for ViT-style models.
        hidden_states: Tuple of Arrays (one for the output of the embeddings +
            one for the output of each layer). Shape varies by architecture.
        attentions: Tuple of Arrays (one for each layer). Shape of each:
            (batch_size, num_heads, sequence_length, sequence_length).

    Example:
        >>> output = ImageClassifierOutput(
        ...     text_embeds=jnp.zeros((2, 512)),  # Despite name, these are image embeds
        ...     last_hidden_state=jnp.zeros((2, 197, 768))  # ViT: 196 patches + CLS
        ... )
    """

    text_embeds: Array = None
    last_hidden_state: Array = None
    hidden_states: tuple[Array, ...] | None = None
    attentions: tuple[Array, ...] | None = None


@auto_pytree
class CLIPOutput(ModelOutput):
    """Output class for CLIP (Contrastive Language-Image Pre-training) models.

    Contains the combined outputs from both text and vision encoders,
    including similarity scores and individual model outputs.

    Attributes:
        loss: Contrastive training loss. Scalar value computed from the
            symmetric cross-entropy of the similarity matrices.
        logits_per_image: The scaled dot product scores between image_embeds
            and text_embeds. Shape: (image_batch_size, text_batch_size).
            This represents the image-text similarity scores.
        logits_per_text: The scaled dot product scores between text_embeds
            and image_embeds. Shape: (text_batch_size, image_batch_size).
            This represents the text-image similarity scores.
        text_embeds: The text embeddings obtained by applying the projection
            layer to the pooled output of the text model. Shape:
            (batch_size, output_dim).
        image_embeds: The image embeddings obtained by applying the projection
            layer to the pooled output of the vision model. Shape:
            (batch_size, output_dim).
        text_model_output: The full output of the text encoder including
            hidden states and attentions.
        vision_model_output: The full output of the vision encoder including
            hidden states and attentions.

    Example:
        >>> output = CLIPOutput(
        ...     logits_per_image=jnp.zeros((4, 8)),  # 4 images, 8 texts
        ...     logits_per_text=jnp.zeros((8, 4)),   # 8 texts, 4 images
        ...     text_embeds=jnp.zeros((8, 512)),
        ...     image_embeds=jnp.zeros((4, 512))
        ... )
    """

    loss: Array = None
    logits_per_image: Array = None
    logits_per_text: Array = None
    text_embeds: Array = None
    image_embeds: Array = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[tp.Any]:
        """Convert to tuple, recursively converting nested model outputs.

        Overrides the base to_tuple method to properly handle nested
        model outputs from the text and vision encoders.

        Returns:
            tuple[Any]: A tuple containing all values, with nested model
                outputs also converted to tuples.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@auto_pytree
class GreedySearchOutput(ModelOutput):
    """Output class for greedy search generation.

    Contains the generated sequences from decoder-only models using
    greedy search (selecting the highest probability token at each step).

    Attributes:
        sequences: The generated sequences. Shape: (batch_size, max_length).
            Contains token IDs including any prompt tokens that were provided.

    Example:
        >>> output = GreedySearchOutput(
        ...     sequences=jnp.array([[1, 2, 3, 4, 0], [1, 5, 6, 7, 8]])
        ... )
        >>> generated_ids = output.sequences
    """

    sequences: Array = None


@auto_pytree
class SampleOutput(ModelOutput):
    """Output class for sampling-based generation.

    Contains the generated sequences from decoder-only models using
    sampling (randomly selecting tokens according to their probabilities).

    Attributes:
        sequences: The generated sequences. Shape: (batch_size, max_length).
            Contains token IDs including any prompt tokens that were provided.
            Due to sampling, sequences may vary across runs with the same input.

    Example:
        >>> output = SampleOutput(
        ...     sequences=jnp.array([[1, 2, 8, 3, 0], [1, 5, 2, 9, 4]])
        ... )
        >>> generated_ids = output.sequences
    """

    sequences: Array = None


@auto_pytree
class BeamSearchOutput(ModelOutput):
    """Output class for beam search generation.

    Contains the generated sequences and their scores from decoder-only
    models using beam search (maintaining multiple hypotheses and selecting
    the best based on accumulated log probabilities).

    Attributes:
        sequences: The generated sequences. Shape: (batch_size, max_length)
            or (batch_size, num_beams, max_length) depending on configuration.
            Contains token IDs including any prompt tokens.
        scores: The scores (log probabilities) of the generated sequences.
            Shape: (batch_size,) or (batch_size, num_beams). Higher scores
            indicate more probable sequences.

    Example:
        >>> output = BeamSearchOutput(
        ...     sequences=jnp.array([[1, 2, 3, 4, 0], [1, 5, 6, 7, 8]]),
        ...     scores=jnp.array([-2.5, -3.1])
        ... )
        >>> best_sequence = output.sequences[jnp.argmax(output.scores)]
    """

    sequences: Array = None
    scores: Array = None
