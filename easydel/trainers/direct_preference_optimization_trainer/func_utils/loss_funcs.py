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


def get_loss_function(
	loss_type: tp.Literal[
		"sigmoid",
		"hinge",
		"ipo",
		"exo_pair",
		"nca_pair",
		"robust",
		"bco_pair",
		"sppo_hard",
		"aot",
		"aot_pair",
		"apo_zero",
		"apo_down",
	],
	beta: float,
	label_smoothing: tp.Union[float, int],
):
	def _sigmoid_dpo_loss(
		logits: chex.Array,
		policy_chosen_log_probs: chex.Array = None,  # IGNORED
		reference_chosen_log_probs: chex.Array = None,  # IGNORED
		policy_rejected_log_probs: chex.Array = None,  # IGNORED
		reference_rejected_log_probs: chex.Array = None,  # IGNORED
	):
		losses = (
			-jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
			- jax.nn.log_sigmoid(-beta * logits) * label_smoothing
		)
		return losses

	def _hinge_dpo_loss(
		logits: chex.Array,
		policy_chosen_log_probs: chex.Array,  # IGNORED
		reference_chosen_log_probs: chex.Array,  # IGNORED
		policy_rejected_log_probs: chex.Array,  # IGNORED
		reference_rejected_log_probs: chex.Array,  # IGNORED
	):
		return jax.nn.relu(1 - beta * logits)

	def _ipo_dpo_loss(
		logits: chex.Array,
		policy_chosen_log_probs: chex.Array,  # IGNORED
		reference_chosen_log_probs: chex.Array,  # IGNORED
		policy_rejected_log_probs: chex.Array,  # IGNORED
		reference_rejected_log_probs: chex.Array,  # IGNORED
	):
		return (logits - 1 / (2 * beta)) ** 2

	def _kto_pair_dpo_loss(
		logits: chex.Array,  # IGNORED
		policy_chosen_log_probs: chex.Array,
		reference_chosen_log_probs: chex.Array,
		policy_rejected_log_probs: chex.Array,
		reference_rejected_log_probs: chex.Array,
	):
		x = jnp.mean(policy_chosen_log_probs - reference_chosen_log_probs)
		chosen_kl = jax.lax.clamp(
			min=jnp.array(0, dtype=x.dtype),
			x=x,
			max=jnp.array(1e9, dtype=x.dtype),
		)
		x = jnp.mean(policy_rejected_log_probs - reference_rejected_log_probs)
		rejected_kl = jax.lax.clamp(
			min=jnp.array(0, dtype=x.dtype),
			x=x,
			max=jnp.array(1e9, dtype=x.dtype),
		)

		chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
		rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
		losses = jnp.concatenate(
			(
				1 - jax.nn.sigmoid(beta * (chosen_log_ratios - rejected_kl)),
				1 - jax.nn.sigmoid(beta * (chosen_kl - rejected_log_ratios)),
			),
			0,
		)

		return losses

	def _robust_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array = None,  # IGNORED
		reference_chosen_log_probs: jax.Array = None,  # IGNORED
		policy_rejected_log_probs: jax.Array = None,  # IGNORED
		reference_rejected_log_probs: jax.Array = None,  # IGNORED
	):
		losses = (
			-jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
			+ jax.nn.log_sigmoid(-beta * logits) * label_smoothing
		) / (1 - 2 * label_smoothing)
		return losses

	def _exo_pair_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array = None,  # IGNORED
		reference_chosen_log_probs: jax.Array = None,  # IGNORED
		policy_rejected_log_probs: jax.Array = None,  # IGNORED
		reference_rejected_log_probs: jax.Array = None,  # IGNORED
	):
		nonlocal label_smoothing
		if label_smoothing == 0:
			label_smoothing = 1e-3
		losses = jax.nn.sigmoid(beta * logits) * (
			jax.nn.log_sigmoid(beta * logits) - jnp.log(1 - label_smoothing)
		) + jax.nn.sigmoid(-beta * logits) * (
			jax.nn.log_sigmoid(-beta * logits) - jnp.log(label_smoothing)
		)
		return losses

	def _bco_pair_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		chosen_logratios = policy_chosen_log_probs - reference_chosen_log_probs
		rejected_logratios = policy_rejected_log_probs - reference_rejected_log_probs

		# chosen_rewards = beta * chosen_logratios
		# rejected_rewards = beta * rejected_logratios
		# rewards = jnp.mean(jnp.concatenate((chosen_rewards, rejected_rewards), 0))
		delta = 0

		losses = -jax.nn.log_sigmoid(
			(beta * chosen_logratios) - delta
		) - jax.nn.log_sigmoid(-(beta * rejected_logratios - delta))
		return losses

	def _sppo_hard_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		a = policy_chosen_log_probs - reference_chosen_log_probs
		b = policy_rejected_log_probs - reference_rejected_log_probs

		losses = (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2
		return losses

	def _nca_pair_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		chosen_rewards = (policy_chosen_log_probs - reference_chosen_log_probs) * beta
		rejected_rewards = (policy_rejected_log_probs - reference_rejected_log_probs) * beta
		losses = (
			-jax.nn.log_sigmoid(chosen_rewards)
			- 0.5 * jax.nn.log_sigmoid(-chosen_rewards)
			- 0.5 * jax.nn.log_sigmoid(-rejected_rewards)
		)
		return losses

	def _aot_pair_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		chosen_logratios = policy_chosen_log_probs - reference_chosen_log_probs
		rejected_logratios = policy_rejected_log_probs - reference_rejected_log_probs

		chosen_logratios_sorted = jnp.sort(chosen_logratios, axis=0)
		rejected_logratios_sorted = jnp.sort(rejected_logratios, axis=0)

		delta = chosen_logratios_sorted - rejected_logratios_sorted

		losses = (
			-jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
			- jax.nn.log_sigmoid(-beta * delta) * label_smoothing
		)
		return losses

	def _aot_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		pi_logratios = policy_chosen_log_probs - policy_rejected_log_probs
		ref_logratios = reference_chosen_log_probs - reference_rejected_log_probs

		pi_logratios_sorted = jnp.sort(pi_logratios, axis=0)
		ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)

		delta = pi_logratios_sorted - ref_logratios_sorted

		losses = (
			-jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
			- jax.nn.log_sigmoid(-beta * delta) * label_smoothing
		)
		return losses

	def _apo_zero_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		chosen_logratios = policy_chosen_log_probs - reference_chosen_log_probs
		rejected_logratios = policy_rejected_log_probs - reference_rejected_log_probs

		losses_chosen = 1 - jax.nn.sigmoid(beta * chosen_logratios)
		losses_rejected = jax.nn.sigmoid(beta * rejected_logratios)

		losses = losses_chosen + losses_rejected
		return losses

	def _apo_down_dpo_loss(
		logits: jax.Array,
		policy_chosen_log_probs: jax.Array,
		reference_chosen_log_probs: jax.Array,
		policy_rejected_log_probs: jax.Array,
		reference_rejected_log_probs: jax.Array,
	):
		chosen_logratios = policy_chosen_log_probs - reference_chosen_log_probs
		rejected_logratios = policy_rejected_log_probs - reference_rejected_log_probs

		losses_chosen = jax.nn.sigmoid(beta * chosen_logratios)
		losses_rejected = 1 - jax.nn.sigmoid(beta * (chosen_logratios - rejected_logratios))

		losses = losses_chosen + losses_rejected
		return losses

	loss_function = {
		"ipo": _ipo_dpo_loss,
		"kto": _kto_pair_dpo_loss,
		"hinge": _hinge_dpo_loss,
		"sigmoid": _sigmoid_dpo_loss,
		"robust": _robust_dpo_loss,
		"exo_pair": _exo_pair_dpo_loss,
		"bco_pair": _bco_pair_dpo_loss,
		"sppo_hard": _sppo_hard_dpo_loss,
		"nca_pair": _nca_pair_dpo_loss,
		"aot_pair": _aot_pair_dpo_loss,
		"aot": _aot_dpo_loss,
		"apo_zero": _apo_zero_dpo_loss,
		"apo_down": _apo_down_dpo_loss,
	}.get(loss_type, None)
	assert loss_function is not None, f"given loss_type({loss_function}) is not valid"
	return loss_function
