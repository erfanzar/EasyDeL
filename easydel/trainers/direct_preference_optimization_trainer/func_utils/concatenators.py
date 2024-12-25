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
import typing as tp

import chex
import jax
from jax import numpy as jnp

from ..utils import pad_to_length


def concatenated_inputs(
	batch: tp.Dict[str, tp.Union[tp.List, chex.Array]],
	is_encoder_decoder: bool = False,
	label_pad_token_id: int = -100,
	padding_value: int = 0,
	fixed_max_length: int | None = None,
) -> tp.Dict[str, chex.Array]:
	"""The concatenated_inputs function takes a batch of chosen and rejected examples,
	and concatenates them together. This is useful for training the model to predict whether an example was chosen
	by the human annotator. The function also pads all inputs to
	the same length as the longest input in that batch.

	Args:
	    batch: tp.Dict[str,tp.Union[tp.List,chex.Array]]: Pass the batch of data
	        into the function,
	    is_encoder_decoder: bool: Determine whether the model is an
	        encoder-decoder model
	    label_pad_token_id: int: Pad the labels with a value of -100
	    padding_value: int: Pad the input_ids and attention_mask arrays
	        to the same length
	    truncation_mode: typing.Literal["keep_end", "keep_start"]: is
	        left padded or not should it keep start of the
	    fixed_max_length: int|None: by providing fixed_max_length the
	        func will always return a fixed sequence length and won't
	        use dynamic methods.
	Allow for the batch to be a list of arrays or just an array,
	Specify the type of data that is being passed in

	array or the end of the array?.

	Returns:
	    A dictionary of the concatenated inputs
	"""
	concatenated_batch = {}
	if fixed_max_length is None:
		if is_encoder_decoder:
			max_length = max(
				batch["chosen_labels"].shape[-1], batch["rejected_labels"].shape[-1]
			)
		else:
			max_length = max(
				batch["chosen_input_ids"].shape[-1],
				batch["rejected_input_ids"].shape[-1],
			)
	else:
		max_length = fixed_max_length
	for k in batch:
		if k.startswith("chosen") and isinstance(batch[k], jax.Array):
			if "labels" in k or is_encoder_decoder:
				pad_value = label_pad_token_id
			elif k.endswith("_input_ids"):
				pad_value = padding_value
			elif k.endswith("_attention_mask"):
				pad_value = 0
			else:
				raise KeyError("couldn't find pad_value [Dataset Issue]")
			concatenated_key = k.replace("chosen", "concatenated")
			concatenated_batch[concatenated_key] = pad_to_length(
				batch[k],
				max_length,
				pad_value=pad_value,
			)
	for k in batch:
		if k.startswith("rejected") and isinstance(batch[k], jax.Array):
			if "labels" in k or is_encoder_decoder:
				pad_value = label_pad_token_id
			elif k.endswith("_input_ids"):
				assert padding_value is not None, "`padding_value` can not be set as `None`"
				pad_value = padding_value
			elif k.endswith("_attention_mask"):
				pad_value = 0
			else:
				raise KeyError("couldn't find pad_value [Dataset Issue]")
			concatenated_key = k.replace("rejected", "concatenated")
			v2d = lambda ar: ar.reshape(ar.shape[0], -1)  # noqa
			concatenated_batch[concatenated_key] = jnp.concatenate(
				(
					v2d(concatenated_batch[concatenated_key]),
					pad_to_length(
						v2d(batch[k]),
						max_length,
						pad_value=pad_value,
					),
				),
				axis=0,
			)
	for k in list(concatenated_batch.keys()):
		val = concatenated_batch[k]
		if val.ndim == 3:
			# making 3d array 2d
			concatenated_batch[k] = val.reshape(val.shape[0], -1)
	if is_encoder_decoder:
		concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(
			2, 1
		)
		concatenated_batch["concatenated_attention_mask"] = batch[
			"prompt_attention_mask"
		].repeat(2, 1)

	return concatenated_batch
