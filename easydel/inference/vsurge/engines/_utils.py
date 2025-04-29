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

import jax
import numpy as np
from jax import numpy as jnp

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any


class SlotData(tp.NamedTuple):
	"""Represents the output data for a single inference slot.

	This structure holds the generated tokens, their validity flags, and the
	current sequence length for one specific slot within a batch processed
	by the engine.

	Attributes:
	    tokens: The generated token IDs for the slot (JAX or NumPy array).
	            Shape typically (samples_per_slot, num_speculative_tokens).
	    valid: A boolean array indicating the validity of each generated token
	           (JAX or NumPy array). Shape matches `tokens`.
	    lengths: An array containing the current length(s) of the generated
	             sequence(s) for the slot (JAX or NumPy array). Shape
	             typically (samples_per_slot,).
	"""

	tokens: tp.Union[jax.Array, np.ndarray]
	valid: tp.Union[jax.Array, np.ndarray]
	lengths: tp.Union[jax.Array, np.ndarray]


class ResultTokens(tp.NamedTuple):
	"""Stores the results of a generation step (prefill or decode).

	This structure holds token data, validity flags, and sequence lengths
	concatenated into a single array (`data`) for efficient host transfer.
	Index tuples (`tokens_idx`, `valid_idx`, `length_idx`) specify the slices
	within `data` corresponding to each type of information. This is designed
	to minimize the number of device-to-host transfers.

	Attributes:
	    data: A single JAX or NumPy array containing concatenated token IDs,
	        validity flags, and lengths for the entire batch. Shape typically
	        (batch_size * samples_per_slot, concatenated_data_width).
	    tokens_idx: A tuple (start, end) indicating the column slice for token IDs
	                within the `data` array.
	    valid_idx: A tuple (start, end) indicating the column slice for validity flags
	               within the `data` array.
	    length_idx: A tuple (start, end) indicating the column slice for sequence lengths
	                within the `data` array.
	    samples_per_slot: The number of samples generated per inference slot (e.g., 1).
	                      Used by `get_result_at_slot` to extract data correctly.
	"""

	data: tp.Union[jax.Array, np.ndarray]
	tokens_idx: tp.Tuple[int, int]
	valid_idx: tp.Tuple[int, int]
	length_idx: tp.Tuple[int, int]
	samples_per_slot: int

	def copy_to_host_async(self: "ResultTokens") -> None:
		"""Initiates an asynchronous copy of the `data` array to the host CPU.

		If the data is already a NumPy array, this is a no-op.
		"""
		if isinstance(self.data, np.ndarray):
			return
		self.data.copy_to_host_async()

	def convert_to_numpy(self: "ResultTokens") -> "ResultTokens":
		"""Converts the internal `data` array to a NumPy array synchronously.

		Returns:
		    A new ResultTokens instance with the data as a NumPy array.
		"""
		return ResultTokens(
			np.array(self.data),
			self.tokens_idx,
			self.valid_idx,
			self.length_idx,
			self.samples_per_slot,
		)

	def get_result_at_slot(self, slot: int) -> SlotData:
		"""Extracts the generation results for a specific inference slot.

		Args:
		    slot: The index of the inference slot (0-based) for which to retrieve data.

		Returns:
		    A SlotData object containing the tokens, validity, and lengths for the
		    requested slot.

		Note:
		    This method correctly handles potential microbatching by using
		    `samples_per_slot` to calculate the correct indices within the `data` array.
		"""
		start_idx = slot * self.samples_per_slot
		end_idx = (slot + 1) * self.samples_per_slot
		return SlotData(
			tokens=self.data[
				start_idx:end_idx,
				self.tokens_idx[0] : self.tokens_idx[1],
			],
			valid=self.data[
				start_idx:end_idx,
				self.valid_idx[0] : self.valid_idx[1],
			],
			lengths=self.data[
				start_idx:end_idx,
				self.length_idx[0] : self.length_idx[1],
			][:, 0],
		)

	def __str__(self):
		return f"ResultTokens(data={self.data})"


def sample_top_p_efficient(
	logits: jax.Array,
	top_p: jax.Array,
	temperature: jax.Array,
	rng: jax.random.PRNGKey,
	top_k_for_computation: int = 50,
) -> jax.Array:
	vocab_size = logits.shape[-1]
	effective_k = min(top_k_for_computation, vocab_size)
	safe_temperature = jnp.where(temperature > 1e-6, temperature, 1.0)
	scaled_logits = logits / jnp.expand_dims(safe_temperature, axis=-1)
	top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, k=effective_k)
	top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)
	cumulative_probs_k = jnp.cumsum(top_k_probs, axis=-1)
	keep_mask_k = cumulative_probs_k <= jnp.expand_dims(top_p, axis=-1)
	keep_mask_k = keep_mask_k.at[..., 0].set(True)
	filtered_top_k_logits = jnp.where(keep_mask_k, top_k_logits, -jnp.inf)
	sampled_k_index = jax.random.categorical(rng, filtered_top_k_logits)
	next_token_index = jnp.take_along_axis(
		top_k_indices,
		jnp.expand_dims(sampled_k_index, axis=-1),
		axis=-1,
	).squeeze(-1)
	return next_token_index
