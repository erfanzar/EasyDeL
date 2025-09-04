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

"""Model output classes for EasyDeL.

Defines standardized output structures for various model types and tasks.
These dataclasses provide consistent interfaces for model outputs while
maintaining compatibility with JAX pytrees.

Classes:
    ModelOutput: Base class for all model outputs
    CausalLMOutput: Output for causal language models
    MoeCausalLMOutput: Output for MoE causal language models
    SequenceClassifierOutput: Output for sequence classification
    ImageClassifierOutput: Output for image classification
    CLIPOutput: Output for CLIP models
    CLIPTextModelOutput: Output for CLIP text encoders
    GreedySearchOutput: Output for greedy generation
    SampleOutput: Output for sampling generation
    BeamSearchOutput: Output for beam search generation

Key Features:
    - Consistent interface across model types
    - JAX pytree compatibility
    - Optional fields with None defaults
    - Dictionary-like access patterns
    - Automatic validation

Example:
    >>> from easydel.infra.modeling_outputs import CausalLMOutput
    >>> output = CausalLMOutput(
    ...     logits=logits,
    ...     hidden_states=hidden_states,
    ...     attentions=attentions
    ... )
    >>> # Access as attribute or dictionary
    >>> logits = output.logits
    >>> logits = output["logits"]
"""

from __future__ import annotations

import typing as tp
from dataclasses import fields, is_dataclass

import chex
from eformer.pytree import auto_pytree
from jax.core import Tracer

if tp.TYPE_CHECKING:
    from easydel.layers.caching import TransformerCache, TransformerCacheView
else:
    TransformerCacheView = tp.Any
    TransformerCache = tp.Any


def _is_array(array):
    if isinstance(array, Tracer):
        return True
    return False


class ModelOutput(tp.OrderedDict):
    """Base class for all model outputs.

    Provides a consistent interface for model outputs that behaves like
    both a tuple (for positional access) and a dictionary (for named access).
    Automatically filters out None values and provides validation.

    Subclasses must use the @auto_pytree decorator to ensure JAX compatibility.

    Methods:
        to_tuple: Convert to tuple, excluding None values

    Note:
        All fields except the first should have None as default.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @auto_pytree decorator."
            )

    def to_tuple(self) -> tuple[tp.Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @auto_pytree decorator has been used.
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
                    if not isinstance(element, list | tuple) or not len(element) == 2 or not isinstance(element[0], str):
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
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        _fn, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return _fn, args, *remaining


@auto_pytree
class AttentionLayerOutput(ModelOutput):
    attention_output: chex.Array
    attention_weight: chex.Array | None = None
    attention_logits: chex.Array | None = None
    cache_view: TransformerCacheView | None = None


@auto_pytree
class EncoderLayerOutput(ModelOutput):
    hidden_states: chex.Array
    residual_states: chex.Array | None = None
    attention_weight: chex.Array | None = None
    attention_logits: chex.Array | None = None


@auto_pytree
class DecoderLayerOutput(ModelOutput):
    hidden_states: chex.Array
    residual_states: chex.Array | None = None
    cross_attention: chex.Array | None = None
    attention_weight: chex.Array | None = None
    attention_logits: chex.Array | None = None
    router_logits: chex.Array | None = None
    gate_loss: chex.Array | None = None
    cache_view: TransformerCacheView | None = None


@auto_pytree
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    past_key_values: dict[str, chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`chex.Array` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
            model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: chex.Array = None
    pooler_output: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        logits (`chex.Array` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(chex.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """

    logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tp.Dict[str, chex.Array]`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: chex.Array = None
    past_key_values: dict[str, chex.Array] | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`chex.Array` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by a
            Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
            prediction (classification) objective during pretraining.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: chex.Array = None
    pooler_output: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`chex.Array` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: chex.Array = None
    pooler_output: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    past_key_values: TransformerCache | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    cross_attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: chex.Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    cross_attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    last_hidden_state: chex.Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[chex.Array] | None = None
    decoder_attentions: tuple[chex.Array] | None = None
    decoder_attention_logits: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    cross_attention_logits: tuple[chex.Array] | None = None
    encoder_last_hidden_state: chex.Array | None = None
    encoder_hidden_states: tuple[chex.Array] | None = None
    encoder_attentions: tuple[chex.Array] | None = None
    encoder_attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        logits (`chex.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` tuples of length `config.n_layers`, with each tuple containing the cached key, value
            states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
            Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    logits: chex.Array = None
    past_key_values: TransformerCache | None = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    cross_attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        logits (`chex.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: chex.Array | None = None
    hidden_states: tuple[chex.Array] | None = None
    last_hidden_state: chex.Array | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    past_key_values: TransformerCache | None = None
    loss: chex.Array | None = None


CausalLMOutput = MaskedLMOutput


@auto_pytree
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        logits (`chex.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits: chex.Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[chex.Array] | None = None
    decoder_attentions: tuple[chex.Array] | None = None
    decoder_attention_logits: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    cross_attention_logits: tuple[chex.Array] | None = None
    encoder_last_hidden_state: chex.Array | None = None
    encoder_hidden_states: tuple[chex.Array] | None = None
    encoder_attentions: tuple[chex.Array] | None = None
    encoder_attention_logits: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class NextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`chex.Array` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        logits (`chex.Array` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    past_key_values: TransformerCache | None = None
    loss: chex.Array | None = None
    aux_loss: chex.Array | None = None


@auto_pytree
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    Args:
        logits (`chex.Array` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    logits: chex.Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[chex.Array] | None = None
    decoder_attentions: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    encoder_last_hidden_state: chex.Array | None = None
    encoder_hidden_states: tuple[chex.Array] | None = None
    encoder_attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        logits (`chex.Array` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        logits (`chex.Array` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (`chex.Array` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`chex.Array` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    start_logits: chex.Array = None
    end_logits: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    Args:
        start_logits (`chex.Array` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`chex.Array` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        past_key_values (`tuple(chex.Array | None)`):
            tp.Tuple of `tuple(chex.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    start_logits: chex.Array = None
    end_logits: chex.Array = None
    past_key_values: TransformerCache | None = None
    decoder_hidden_states: tuple[chex.Array] | None = None
    decoder_attentions: tuple[chex.Array] | None = None
    cross_attentions: tuple[chex.Array] | None = None
    encoder_last_hidden_state: chex.Array | None = None
    encoder_hidden_states: tuple[chex.Array] | None = None
    encoder_attentions: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class MoeModelOutput(ModelOutput):
    """
    Base class for MoE model outputs.

    Args:
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(chex.Array)`, *optional*):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            The logits output of the router network, which are used to compute the mixture of experts.
    """

    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array] | None = None
    past_key_values: TransformerCache | None = None
    attentions: tuple[chex.Array] | None = None
    attention_logits: tuple[chex.Array] | None = None
    router_logits: tuple[chex.Array] | None = None
    all_router_losses: tuple[chex.Array] | None = None
    logits: chex.Array = None
    loss: chex.Array | None = None


@auto_pytree
class MoeCausalLMOutput(MaskedLMOutput):
    """
    Base class for causal language modeling (CLM) outputs of MoE models.

    Args:
        aux_loss (`chex.Array`, *optional*):
            Auxiliary loss used for training MoE models.
        router_logits (`tuple(chex.Array)`, *optional*):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
            The logits output of the router network, which are used to compute the mixture of experts.
    """

    aux_loss: chex.Array | None = None
    router_logits: tuple[chex.Array] | None = None
    all_router_losses: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class MambaOutput(BaseModelOutput):
    last_hidden_state: chex.Array = None
    cache_params: list[chex.Array] | None = None
    hidden_states: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class MambaCausalLMOutput(BaseModelOutput):
    logits: chex.Array = None
    cache_params: list[chex.Array] | None = None
    hidden_states: tuple[chex.Array] | None = None
    loss: chex.Array | None = None


@auto_pytree
class CLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`chex.Array` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: chex.Array = None
    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


@auto_pytree
class ImageClassifierOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`chex.Array` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        last_hidden_state (`chex.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(chex.Array | None)`):
            tp.Tuple of `chex.Array` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    text_embeds: chex.Array = None
    last_hidden_state: chex.Array = None
    hidden_states: tuple[chex.Array, ...] | None = None
    attentions: tuple[chex.Array, ...] | None = None


@auto_pytree
class CLIPOutput(ModelOutput):
    """
    Args:
        loss:(`chex.Array`) training loss
        logits_per_image:(`chex.Array` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`chex.Array` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`chex.Array` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        image_embeds(`chex.Array` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPVisionModel`].
    """

    loss: chex.Array = None
    logits_per_image: chex.Array = None
    logits_per_text: chex.Array = None
    text_embeds: chex.Array = None
    image_embeds: chex.Array = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[tp.Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@auto_pytree
class GreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`chex.Array` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: chex.Array = None


@auto_pytree
class SampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (`chex.Array` of shape `(batch_size, max_length)`):
            The generated sequences.
    """

    sequences: chex.Array = None


@auto_pytree
class BeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`chex.Array` of shape `(batch_size, max_length)`):
            The generated sequences.
        scores (`chex.Array` of shape `(batch_size,)`):
            The scores (log probabilities) of the generated sequences.
    """

    sequences: chex.Array = None
    scores: chex.Array = None
