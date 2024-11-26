# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import flax
import jax.numpy as jnp
from jax.core import Tracer


def _is_array(array):
	if isinstance(array, Tracer):
		return True
	return False


class ModelOutput(OrderedDict):
	"""
	Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
	tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
	python dictionary.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		is_modeloutput_subclass = self.__class__ != ModelOutput

		if is_modeloutput_subclass and not is_dataclass(self):
			raise TypeError(
				f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
				" This is a subclass of ModelOutput and so must use the @dataclass decorator."
			)

	def __post_init__(self):
		"""Check the ModelOutput dataclass.

		Only occurs if @dataclass decorator has been used.
		"""
		class_fields = fields(self)

		# Safety and consistency checks
		if not len(class_fields):
			raise ValueError(f"{self.__class__.__name__} has no fields.")
		if not all(field.default is None for field in class_fields[1:]):
			raise ValueError(
				f"{self.__class__.__name__} should not have more than one required field."
			)

		first_field = getattr(self, class_fields[0].name)
		other_fields_are_none = all(
			getattr(self, field.name) is None for field in class_fields[1:]
		)

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
							raise ValueError(
								f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
							)
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
		raise Exception(
			f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
		)

	def setdefault(self, *args, **kwargs):
		raise Exception(
			f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
		)

	def pop(self, *args, **kwargs):
		raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

	def update(self, *args, **kwargs):
		raise Exception(
			f"You cannot use ``update`` on a {self.__class__.__name__} instance."
		)

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
		callable, _args, *remaining = super().__reduce__()
		args = tuple(getattr(self, field.name) for field in fields(self))
		return callable, args, *remaining

	def to_tuple(self) -> Tuple[Any]:
		"""
		Convert self to a tuple containing all the attributes/keys that are not `None`.
		"""
		return tuple(self[k] for k in self.keys())


@flax.struct.dataclass
class FlaxBaseModelOutput(ModelOutput):
	"""
	Base class for model's outputs, with potential hidden states and attentions.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	last_hidden_state: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithNoAttention(ModelOutput):
	"""
	Base class for model's outputs, with potential hidden states.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
	        for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
	        model at the output of each layer plus the optional initial embedding outputs.
	"""

	last_hidden_state: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndNoAttention(ModelOutput):
	"""
	Base class for model's outputs that also contains a pooling of the last hidden states.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
	        Last layer hidden-state after a pooling operation on the spatial dimensions.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
	        for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of the
	        model at the output of each layer plus the optional initial embedding outputs.
	"""

	last_hidden_state: jnp.ndarray = None
	pooler_output: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxImageClassifierOutputWithNoAttention(ModelOutput):
	"""
	Base class for outputs of image classification models.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
	        Classification (or regression if config.num_labels==1) scores (before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when
	    `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
	        for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
	        called feature maps) of the model at the output of each stage.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPast(ModelOutput):
	"""
	Base class for model's outputs, with potential hidden states and attentions.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    past_key_values (`Dict[str, jnp.ndarray]`):
	        Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
	        auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	last_hidden_state: jnp.ndarray = None
	past_key_values: Optional[Dict[str, jnp.ndarray]] = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPooling(ModelOutput):
	"""
	Base class for model's outputs that also contains a pooling of the last hidden states.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
	        Last layer hidden-state of the first token of the sequence (classification token) further processed by a
	        Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence
	        prediction (classification) objective during pretraining.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	last_hidden_state: jnp.ndarray = None
	pooler_output: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
	"""
	Base class for model's outputs that also contains a pooling of the last hidden states.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
	        Last layer hidden-state of the first token of the sequence (classification token) after further processing
	        through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
	        the classification token after processing through a linear layer and a tanh activation function. The linear
	        layer weights are trained from the next sentence prediction (classification) objective during pretraining.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
	        for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
	        `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
	        encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
	        `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
	        input) to speed up sequential decoding.
	"""

	last_hidden_state: jnp.ndarray = None
	pooler_output: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
	"""
	Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.

	        If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
	        hidden_size)` is output.
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
	        `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
	        encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
	        `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
	        input) to speed up sequential decoding.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	"""

	last_hidden_state: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqModelOutput(ModelOutput):
	"""
	Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
	decoding.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the decoder of the model.

	        If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
	        hidden_size)` is output.
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
	        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
	        blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
	    decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
	    decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	    encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
	        Sequence of hidden-states at the output of the last layer of the encoder of the model.
	    encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
	    encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	"""

	last_hidden_state: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None
	encoder_last_hidden_state: Optional[jnp.ndarray] = None
	encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxCausalLMOutputWithCrossAttentions(ModelOutput):
	"""
	Base class for causal language model (or autoregressive) outputs.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
	        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Cross attentions weights after the attention softmax, used to compute the weighted average in the
	        cross-attention heads.
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `jnp.ndarray` tuples of length `config.n_layers`, with each tuple containing the cached key, value
	        states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting.
	        Only relevant if `config.is_decoder = True`.

	        Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
	        `past_key_values` input) to speed up sequential decoding.
	"""

	logits: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxMaskedLMOutput(ModelOutput):
	"""
	Base class for masked language models outputs.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
	        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


FlaxCausalLMOutput = FlaxMaskedLMOutput


@flax.struct.dataclass
class FlaxSeq2SeqLMOutput(ModelOutput):
	"""
	Base class for sequence-to-sequence language models outputs.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
	        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
	        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
	        blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
	    decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
	    decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	    encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
	        Sequence of hidden-states at the output of the last layer of the encoder of the model.
	    encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
	    encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	"""

	logits: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None
	encoder_last_hidden_state: Optional[jnp.ndarray] = None
	encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxNextSentencePredictorOutput(ModelOutput):
	"""
	Base class for outputs of models predicting if two sentences are consecutive or not.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, 2)`):
	        Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
	        before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSequenceClassifierOutput(ModelOutput):
	"""
	Base class for outputs of sentence classification models.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
	        Classification (or regression if config.num_labels==1) scores (before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqSequenceClassifierOutput(ModelOutput):
	"""
	Base class for outputs of sequence-to-sequence sentence classification models.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
	        Classification (or regression if config.num_labels==1) scores (before SoftMax).
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
	        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
	        blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
	    decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
	    decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	    encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
	        Sequence of hidden-states at the output of the last layer of the encoder of the model.
	    encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
	    encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	"""

	logits: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None
	encoder_last_hidden_state: Optional[jnp.ndarray] = None
	encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxMultipleChoiceModelOutput(ModelOutput):
	"""
	Base class for outputs of multiple choice models.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, num_choices)`):
	        *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

	        Classification scores (before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxTokenClassifierOutput(ModelOutput):
	"""
	Base class for outputs of token classification models.

	Args:
	    logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.num_labels)`):
	        Classification scores (before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxQuestionAnsweringModelOutput(ModelOutput):
	"""
	Base class for outputs of question answering models.

	Args:
	    start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
	        Span-start scores (before SoftMax).
	    end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
	        Span-end scores (before SoftMax).
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	"""

	start_logits: jnp.ndarray = None
	end_logits: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class FlaxSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
	"""
	Base class for outputs of sequence-to-sequence question answering models.

	Args:
	    start_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
	        Span-start scores (before SoftMax).
	    end_logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
	        Span-end scores (before SoftMax).
	    past_key_values (`tuple(tuple(jnp.ndarray))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
	        Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape
	        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
	        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

	        Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
	        blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
	    decoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
	    decoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	    cross_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
	        weighted average in the cross-attention heads.
	    encoder_last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
	        Sequence of hidden-states at the output of the last layer of the encoder of the model.
	    encoder_hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
	        `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
	    encoder_attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
	        self-attention heads.
	"""

	start_logits: jnp.ndarray = None
	end_logits: jnp.ndarray = None
	past_key_values: Optional[Tuple[Tuple[jnp.ndarray]]] = None
	decoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	decoder_attentions: Optional[Tuple[jnp.ndarray]] = None
	cross_attentions: Optional[Tuple[jnp.ndarray]] = None
	encoder_last_hidden_state: Optional[jnp.ndarray] = None
	encoder_hidden_states: Optional[Tuple[jnp.ndarray]] = None
	encoder_attentions: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class MoeModelOutput(FlaxMaskedLMOutput):
	"""
	Base class for MoE model outputs.

	Args:
	    last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
	        Sequence of hidden-states at the output of the last layer of the model.
	    hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
	        Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer)
	        of shape `(batch_size, sequence_length, hidden_size)`.

	        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
	    attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
	        sequence_length)`.

	        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
	        heads.
	    router_logits (`tuple(jnp.ndarray)`, *optional*):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

	        The logits output of the router network, which are used to compute the mixture of experts.
	"""

	last_hidden_state: jnp.ndarray = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
	attentions: Optional[Tuple[jnp.ndarray]] = None
	router_logits: Optional[Tuple[jnp.ndarray]] = None
	all_router_losses: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class MoeCausalLMOutput(FlaxMaskedLMOutput):
	"""
	Base class for causal language modeling (CLM) outputs of MoE models.

	Args:
	    aux_loss (`jnp.ndarray`, *optional*):
	        Auxiliary loss used for training MoE models.
	    router_logits (`tuple(jnp.ndarray)`, *optional*):
	        Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

	        The logits output of the router network, which are used to compute the mixture of experts.
	"""

	aux_loss: Optional[jnp.ndarray] = None
	router_logits: Optional[Tuple[jnp.ndarray]] = None
	all_router_losses: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class MambaOutput(FlaxBaseModelOutput):
	last_hidden_state: jnp.ndarray = None
	cache_params: Optional[List[jnp.ndarray]] = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None


@flax.struct.dataclass
class MambaCausalLMOutput(FlaxBaseModelOutput):
	logits: jnp.ndarray = None
	cache_params: Optional[List[jnp.ndarray]] = None
	hidden_states: Optional[Tuple[jnp.ndarray]] = None
