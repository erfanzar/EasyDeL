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

import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import (
	LossConfig,
	LossMetrics,
)

from ..training_utils import (
	make_assertions_and_get_sizes,
	minibatch_call,
	update_metrics,
	update_state_respectfully,
)

RewardFunc = tp.Union[
	EasyDeLState,
	tp.Callable[[list, list], list[float]],
]


def get_per_token_logps(model, input_ids, attention_mask, prompt_length):
	"""
	Get per-token log probabilities using the model outputs.

	Args:
	    model: The language model
	    input_ids: Input token ids [batch_size, seq_len]
	    attention_mask: Input masks [batch_size, seq_len]
	    prompt_length: Length of the prompt
	"""

	logits = model(
		input_ids=input_ids,
		attention_mask=attention_mask,
	).logits[:, prompt_length - 1 :]
	logits = logits[:, :-1, :]
	token_log_probs = compute_per_token_logps(logits, input_ids, prompt_length)
	return token_log_probs


def compute_per_token_logps(logits, input_ids, prompt_length):
	"""
	Compute per-token log probabilities in a vectorized way.

	Args:
	    logits: Pre-trimmed logits [batch_size, seq_len, vocab_size]
	    input_ids: Input token ids [batch_size, seq_len]
	    prompt_length: Length of the prompt
	"""
	log_probs = jax.nn.log_softmax(logits, axis=-1)
	target_ids = input_ids[:, prompt_length:]
	token_log_probs = jnp.take_along_axis(
		log_probs,
		jnp.expand_dims(target_ids, axis=-1),
		axis=-1,
	)
	token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
	return token_log_probs


def grpo_step(
	state: EasyDeLState,
	batch: tp.Mapping[str, jax.Array],
	eos_token_id: int,
	num_generations: int,
	beta: float,
	prompt_length: int,
	loss_config: tp.Optional[LossConfig] = None,
	learning_rate_fn: optax.Schedule = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
	is_training: bool = True,
) -> tp.Tuple[EasyDeLState, LossMetrics]:
	# Determine batch size, minibatch size, and enforce partition spec.
	batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	assert eos_token_id is not None, "`eos_token_id` can not be None"

	def loss_fn(tree, minibatch):
		module = flax.nnx.merge(state.graphdef, tree, state.graphother)

		(
			prompt_ids,
			prompt_mask,
			completion_ids,
			completion_mask,
			advantages,
		) = (
			minibatch["prompt_ids"],
			minibatch["prompt_mask"],
			minibatch["completion_ids"],
			minibatch["completion_mask"],
			minibatch["advantages"],
		)

		input_ids = jnp.concatenate(
			[prompt_ids.repeat(num_generations, 0), completion_ids],
			axis=1,
		)
		attention_mask = jnp.concatenate(
			[prompt_mask.repeat(num_generations, 0), completion_mask],
			axis=1,
		)

		per_token_logps = get_per_token_logps(
			module,
			input_ids,
			attention_mask,
			prompt_ids.shape[-1],
		)

		ref_per_token_logps = minibatch["ref_per_token_logps"]
		per_token_kl = (
			jnp.exp(ref_per_token_logps - per_token_logps)
			- (ref_per_token_logps - per_token_logps)
			- 1
		)

		per_token_loss = jnp.exp(
			per_token_logps - jax.lax.stop_gradient(per_token_logps)
		) * jnp.expand_dims(advantages, 1)
		per_token_loss = -(per_token_loss - beta * per_token_kl)
		comps = jnp.sum(completion_mask, axis=1)
		loss = jnp.mean(jnp.sum(per_token_loss * completion_mask, axis=1) / comps)
		mean_kl = jnp.mean(jnp.sum(per_token_kl * completion_mask, axis=1) / comps)

		return loss, LossMetrics(
			loss=loss,
			accuracy=1,
			other_metrics={
				"mean_kl": mean_kl,
				"ref_per_token_logps": jnp.mean(ref_per_token_logps),
				"advantages": jnp.mean(advantages),
			},
		)

	# Compute gradients and metrics across minibatches.
	if is_training:
		gradients, metrics = minibatch_call(
			state=state,
			batch=batch,
			minibatch_size=minibatch_size,
			grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
		)
		state = update_state_respectfully(
			state=state,
			gradients=gradients,
			loss_config=loss_config,
			metrics=update_metrics(
				metrics=metrics,
				learning_rate_fn=learning_rate_fn,
				step=state.step,
				gradients=gradients,
			),
		)
		return state, metrics
	else:
		_, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
		return metrics
