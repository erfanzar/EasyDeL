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


import functools

import chex
import jax
from jax import numpy as jnp


@functools.partial(
	jax.jit,
	static_argnames=[
		"average_log_prob",
		"is_encoder_decoder",
		"label_pad_token_id",
	],
)
def get_batch_log_probs(
	logits: chex.Array,
	labels: chex.Array,
	average_log_prob: bool = False,
	label_pad_token_id: int = -100,
	is_encoder_decoder: bool = False,
) -> chex.Array:
	"""
	The get_batch_log_probs function computes the log probability of a batch of sequences.

	Args:
	    logits: chex.Array: Compute the log_softmax of the input.
	    labels: chex.Array: Mask the logits.
	    average_log_prob: bool: Determine whether to average the log prob over the sequence length.
	    label_pad_token_id: int: Mask out the padding tokens in the labels.
	    is_encoder_decoder: bool: Indicate whether the model is an encoder-decoder model.

	Returns:
	    The log probability of the labels given the logits
	"""

	if logits.shape[:-1] != labels.shape:
		raise ValueError(
			"Logits (batch and sequence length dim) and labels must have the same shape."
		)

	if not is_encoder_decoder:
		labels = labels[:, 1:]
		logits = logits[:, :-1, :]

	batch, seq_len, dim = logits.shape
	loss_mask = labels != label_pad_token_id
	labels = jax.lax.select(
		labels == label_pad_token_id,
		jnp.zeros(labels.shape, dtype=labels.dtype),
		labels,
	)
	logits_log_s = jax.nn.log_softmax(logits, -1)
	per_token_log_probs = jnp.take_along_axis(
		logits_log_s, axis=2, indices=labels[:, :, None]
	).reshape(batch, seq_len)

	if average_log_prob:
		log_prob = jnp.sum((per_token_log_probs * loss_mask), axis=-1) / jnp.sum(
			loss_mask, axis=-1
		)
	else:
		log_prob = jnp.sum((per_token_log_probs * loss_mask), axis=-1)

	return log_prob
