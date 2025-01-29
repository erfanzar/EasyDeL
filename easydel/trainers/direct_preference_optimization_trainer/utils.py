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
from contextlib import contextmanager
from dataclasses import dataclass

import chex
import jax
from jax import numpy as jnp

import numpy as np


@dataclass
class DPODataCollatorWithPadding:
	r"""DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

	Args:
	    pad_token_id: int: The tokenizers pad_token_id.
	    label_pad_token_id: int: The label used for masking.
	    is_encoder_decoder: tp.Optional[bool]: Whether you model has an
	        encoder_decoder architecture
	"""

	max_prompt_length: int
	max_completion_length: int
	pad_token_id: int = 0
	label_pad_token_id: int = -100
	is_encoder_decoder: tp.Optional[bool] = False
	ids_to_pop_from_dataset: tp.Optional[dict] = None
	auto_fix_data: bool = True

	def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
		prompt_input_ids = [jnp.array(feature["prompt_input_ids"]) for feature in features]
		prompt_attention_mask = [jnp.ones_like(input_ids) for input_ids in prompt_input_ids]
		chosen_input_ids = [jnp.array(feature["chosen_input_ids"]) for feature in features]
		chosen_attention_mask = [jnp.ones_like(input_ids) for input_ids in chosen_input_ids]
		rejected_input_ids = [
			jnp.array(feature["rejected_input_ids"]) for feature in features
		]
		rejected_attention_mask = [
			jnp.ones_like(input_ids) for input_ids in rejected_input_ids
		]

		pixel_values = None
		pixel_attention_mask = None
		if "pixel_values" in features[0]:
			pixel_values = [jnp.array(feature["pixel_values"]) for feature in features]
		if "pixel_attention_mask" in features[0]:
			pixel_attention_mask = [
				jnp.array(feature["pixel_attention_mask"]) for feature in features
			]

		ref_chosen_logps = None
		ref_rejected_logps = None
		if "ref_chosen_logps" in features[0] and "ref_rejected_logps" in features[0]:
			ref_chosen_logps = jnp.array(
				[feature["ref_chosen_logps"] for feature in features]
			)
			ref_rejected_logps = jnp.array(
				[feature["ref_rejected_logps"] for feature in features]
			)

		# Pad sequences
		output = {
			"prompt_input_ids": pad(
				prompt_input_ids,
				self.max_prompt_length,
				padding_value=self.pad_token_id,
				padding_side="left",
			),
			"prompt_attention_mask": pad(
				prompt_attention_mask,
				self.max_prompt_length,
				padding_value=0,
				padding_side="left",
			),
			"chosen_input_ids": pad(
				chosen_input_ids,
				self.max_completion_length,
				padding_value=self.pad_token_id,
			),
			"chosen_attention_mask": pad(
				chosen_attention_mask,
				self.max_completion_length,
				padding_value=0,
			),
			"rejected_input_ids": pad(
				rejected_input_ids,
				self.max_completion_length,
				padding_value=self.pad_token_id,
			),
			"rejected_attention_mask": pad(
				rejected_attention_mask,
				self.max_completion_length,
				padding_value=0,
			),
		}

		# Add optional outputs
		if pixel_values is not None:
			output["pixel_values"] = pad(
				pixel_values,
				self.max_prompt_length,
				padding_value=0.0,
			)
		if pixel_attention_mask is not None:
			output["pixel_attention_mask"] = pad(
				pixel_attention_mask,
				self.max_prompt_length,
				padding_value=0,
			)
		if "image_sizes" in features[0]:
			output["image_sizes"] = jnp.array(
				[feature["image_sizes"] for feature in features]
			)
		if ref_chosen_logps is not None and ref_rejected_logps is not None:
			output["ref_chosen_logps"] = ref_chosen_logps
			output["ref_rejected_logps"] = ref_rejected_logps

		return output


def shift_and_pad(mask, *tensors):
	for i in range(mask.shape[0]):
		first_one_idx = np.nonzero(mask[i])[0][0].item()
		mask[i] = np.roll(mask[i], shift=-first_one_idx)
		for tensor in tensors:
			tensor[i] = np.roll(tensor[i], shift=-first_one_idx)

	if not tensors:
		return mask
	else:
		return mask, *tensors


def pad(
	tensors: list[jnp.ndarray],
	max_lenght: int,
	padding_value: int = 0,
	padding_side: str = "right",
) -> jnp.ndarray:
	"""
	Pads a list of tensors to the same shape along the first dimension.
	"""
	output_shape = tensors[0].shape[:-1]
	output_shape += (max_lenght,)
	output = jnp.full(
		(len(tensors), *output_shape),
		padding_value,
		dtype=tensors[0].dtype,
	)
	for i, t in enumerate(tensors):
		if padding_side == "left":
			seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
		elif padding_side == "right":
			seq_slice = slice(0, t.shape[0])
		else:
			raise ValueError("padding_side must be 'left' or 'right'")

		slices = (i,) + (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
		output = output.at[slices].set(t)
	return output


def pad_to_length(
	tensor: chex.Array,
	length: int,
	pad_value: tp.Union[int, float],
	axis: int = -1,
) -> chex.Array:
	if tensor.shape[axis] >= length:
		if tensor.ndim == 2:
			tensor = tensor[:, :length]
		return tensor
	else:
		pad_size = list(tensor.shape)
		pad_size[axis] = length - tensor.shape[axis]
		return jax.numpy.concatenate(
			[
				tensor,
				pad_value * jax.numpy.ones(pad_size, dtype=tensor.dtype),
			],
			axis=axis,
		)


def pad_sequence(
	sequences,
	batch_first=False,
	padding_value=0,
	max_len: int | None = None,
):
	max_len = max(seq.shape[-1] for seq in sequences) if max_len is None else max_len
	padding_value = jnp.array(padding_value).reshape(1)
	if batch_first:
		padded_seqs = [
			(
				jnp.concatenate(
					[
						seq.reshape(1, -1),
						jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
					],
					axis=1,
				)
				if seq.shape[-1] < max_len
				else seq.reshape(1, -1)
			)
			for seq in sequences
		]
	else:
		padded_seqs = [
			(
				jnp.concatenate(
					[
						jnp.ones((1, max_len - seq.shape[-1])) * padding_value,
						seq.reshape(1, -1),
					],
					axis=1,
				)
				if seq.shape[-1] < max_len
				else seq.reshape(1, -1)
			)
			for seq in sequences
		]

	return jnp.array(padded_seqs)


@contextmanager
def leave_alone_context_manager():
	# Perform setup actions (none in this case)
	yield
