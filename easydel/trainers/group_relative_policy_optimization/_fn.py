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


def get_per_token_logps(model, input_ids, prompt_length):
	"""
	Get per-token log probabilities using the model outputs.

	Args:
	    model: The language model
	    input_ids: Input token ids [batch_size, seq_len]
	    prompt_length: Length of the prompt
	"""

	logits = model(input_ids=input_ids).logits[:, prompt_length - 1 :]
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
	reward_funcs: RewardFunc,
	beta: float,
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
		input_ids = minibatch["input_ids"]
		attention_mask = minibatch["attention_mask"]
		prompt_completion_ids = minibatch["prompt_completion_ids"]
		prompt_completion_mask = minibatch["prompt_completion_mask"]
		ref_per_token_logps = minibatch["ref_per_token_logps"]

		batch_size, prompt_length = input_ids.shape
		completion_ids = prompt_completion_ids[:, prompt_length:]
		per_token_logps = get_per_token_logps(module, prompt_completion_ids, prompt_length)
		per_token_kl = (
			jnp.exp(ref_per_token_logps - per_token_logps)
			- (ref_per_token_logps - per_token_logps)
			- 1
		)
		is_eos = completion_ids == eos_token_id
		eos_idx = jnp.full((is_eos.shape[0],), is_eos.shape[1])
		has_eos = is_eos.any(axis=1)
		first_eos_positions = jnp.argmax(is_eos.astype(jnp.int32), axis=1)
		eos_idx = jnp.where(has_eos, first_eos_positions, eos_idx)
		sequence_indices = jnp.arange(is_eos.shape[1])[None, :].repeat(
			is_eos.shape[0],
			axis=0,
		)
		completion_mask = (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)

		rewards_per_func = jnp.zeros(
			(batch_size * num_generations, len(reward_funcs)),
			dtype="f4",
		)
		for i, reward_func in enumerate(reward_funcs):
			if isinstance(reward_func, EasyDeLState):
				reward_i = jax.lax.stop_gradient(
					reward_func.model(
						input_ids=prompt_completion_ids,
						attention_mask=prompt_completion_mask,
					).logits[:, 0]
				)

			else:
				reward_i = reward_func(
					prompt_completion_ids=prompt_completion_ids,
					prompt_completion_mask=prompt_completion_mask,
					completion_ids=completion_ids,
					completion_mask=completion_mask,
					input_ids=input_ids,
					attention_mask=attention_mask,
				)
			rewards_per_func = rewards_per_func.at[:, i].set(reward_i.reshape(-1))
		rewards = rewards_per_func.sum(axis=1)

		# Reshape rewards for grouping and compute statistics
		grouped_rewards = rewards.reshape(-1, num_generations)
		mean_grouped_rewards = jnp.mean(grouped_rewards, axis=1)
		std_grouped_rewards = jnp.std(grouped_rewards, axis=1)
		mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, num_generations)
		std_grouped_rewards = jnp.repeat(std_grouped_rewards, num_generations)
		advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
		per_token_loss = (
			jnp.exp(per_token_logps - jax.lax.stop_gradient(per_token_logps))
			* advantages[:, None]
		)
		per_token_loss = -(per_token_loss - beta * per_token_kl)

		masked_sum = (per_token_loss * completion_mask).sum(axis=1)
		mask_sum = completion_mask.sum(axis=1)
		loss = jnp.mean(masked_sum / mask_sum)
		mean_kl = (
			(per_token_kl * completion_mask).sum(axis=1) / completion_mask.sum(axis=1)
		).mean()
		return loss, LossMetrics(
			loss=loss,
			accuracy=1,
			other_metrics={
				"mean_kl": mean_kl.mean(),
				"reward": rewards.mean(),
				"reward_std": std_grouped_rewards.mean(),
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
