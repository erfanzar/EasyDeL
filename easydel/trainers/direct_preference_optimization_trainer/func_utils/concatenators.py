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
from jax import numpy as jnp

from ..utils import pad_to_length


def concatenated_inputs(
	batch: tp.Dict[str, tp.Union[tp.List, chex.Array]],
	padding_value: int,
) -> tp.Dict[str, chex.Array]:
	"""The concatenated_inputs function takes a batch of chosen and rejected examples,
	and concatenates them together. This is useful for training the model to predict whether an example was chosen
	by the human annotator. The function also pads all inputs to
	the same length as the longest input in that batch.

	Args:
	    batch: tp.Dict[str,tp.Union[tp.List,chex.Array]]: Pass the batch of data
	        into the function
	    padding_value: int: Pad the input_ids and attention_mask arrays
	        to the same length
	Allow for the batch to be a list of arrays or just an array,
	Specify the type of data that is being passed in

	Returns:
	    A dictionary of the concatenated inputs
	"""
	output = {}

	output["prompt_input_ids"] = jnp.concatenate(
		[batch["prompt_input_ids"], batch["prompt_input_ids"]],
		axis=0,
	)
	output["prompt_attention_mask"] = jnp.concatenate(
		[batch["prompt_attention_mask"], batch["prompt_attention_mask"]],
		axis=0,
	)
	if "pixel_values" in batch:
		output["pixel_values"] = jnp.concatenate(
			[batch["pixel_values"], batch["pixel_values"]],
			axis=0,
		)

	if "pixel_attention_mask" in batch:
		output["pixel_attention_mask"] = jnp.concatenate(
			[batch["pixel_attention_mask"], batch["pixel_attention_mask"]],
			axis=0,
		)
	if "image_sizes" in batch:
		output["image_sizes"] = jnp.concatenate(
			[batch["image_sizes"], batch["image_sizes"]],
			axis=0,
		)

	max_completion_length = max(
		batch["chosen_input_ids"].shape[1],
		batch["rejected_input_ids"].shape[1],
	)
	output["completion_input_ids"] = jnp.concatenate(
		(
			pad_to_length(
				batch["chosen_input_ids"],
				max_completion_length,
				pad_value=padding_value,
			),
			pad_to_length(
				batch["rejected_input_ids"],
				max_completion_length,
				pad_value=padding_value,
			),
		),
	)
	output["completion_attention_mask"] = jnp.concatenate(
		(
			pad_to_length(
				batch["chosen_attention_mask"],
				max_completion_length,
				pad_value=0,
			),
			pad_to_length(
				batch["rejected_attention_mask"],
				max_completion_length,
				pad_value=0,
			),
		),
	)

	return output
