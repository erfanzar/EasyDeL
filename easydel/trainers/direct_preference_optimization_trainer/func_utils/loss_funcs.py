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

from jax.nn import sigmoid, relu, log_sigmoid as logsigmoid


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
		"discopop",
	],
	beta: float,
	label_smoothing: tp.Union[float, int],
):
	def _base_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
		**kwargs,
	) -> chex.Array:
		"""Base DPO loss calculation."""
		logratios = chosen_logps - rejected_logps
		ref_logratios = ref_chosen_logps - ref_rejected_logps
		logits = logratios - ref_logratios
		return logits, logratios, ref_logratios

	# Update existing and add missing loss functions
	def _sigmoid_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
		**kwargs,
	) -> chex.Array:
		logits, _, _ = _base_dpo_loss(
			chosen_logps,
			rejected_logps,
			ref_chosen_logps,
			ref_rejected_logps,
			beta,
			label_smoothing,
		)
		return -(
			jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
			+ jax.nn.log_sigmoid(-beta * logits) * label_smoothing
		)

	def _nca_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
		**kwargs,
	) -> chex.Array:
		chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
		rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
		return -(
			jax.nn.log_sigmoid(chosen_rewards)
			+ 0.5 * jax.nn.log_sigmoid(-chosen_rewards)
			+ 0.5 * jax.nn.log_sigmoid(-rejected_rewards)
		)

	def _aot_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
		**kwargs,
	) -> chex.Array:
		logratios = chosen_logps - rejected_logps
		ref_logratios = ref_chosen_logps - ref_rejected_logps
		logratios_sorted = jnp.sort(logratios, axis=0)
		ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
		delta = logratios_sorted - ref_logratios_sorted
		return -(
			jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
			+ jax.nn.log_sigmoid(-beta * delta) * label_smoothing
		)

	def _discopop_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
		discopop_tau: float = 1.0,
		**kwargs,
	) -> chex.Array:
		logits, _, _ = _base_dpo_loss(
			chosen_logps,
			rejected_logps,
			ref_chosen_logps,
			ref_rejected_logps,
			beta,
			label_smoothing,
		)
		logits = logits * beta
		log_ratio_modulation = jax.nn.sigmoid(logits / discopop_tau)
		logistic_component = -jax.nn.log_sigmoid(logits)
		exp_component = jnp.exp(-logits)
		return (
			logistic_component * (1 - log_ratio_modulation)
			+ exp_component * log_ratio_modulation
		)

	def _hinge_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
		return relu(1 - beta * logits)

	def _ipo_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
		return (logits - 1 / (2 * beta)) ** 2

	def _kto_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
		return (
			-logsigmoid(beta * logits) * (1 - label_smoothing)
			- logsigmoid(-beta * logits) * label_smoothing
		)

	def _robust_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
		return (
			-logsigmoid(beta * logits) * (1 - label_smoothing)
			+ logsigmoid(-beta * logits) * label_smoothing
		) / (1 - 2 * label_smoothing)

	def _exo_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		import math

		logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
		label_smoothing = jnp.maximum(label_smoothing, 1e-3)
		return sigmoid(beta * logits) * (
			logsigmoid(beta * logits) - math.log(1 - label_smoothing)
		) + sigmoid(-beta * logits) * (
			logsigmoid(-beta * logits) - math.log(label_smoothing)
		)

	def _bco_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		chosen_logratios = chosen_logps - ref_chosen_logps
		rejected_logratios = rejected_logps - ref_rejected_logps
		chosen_rewards = beta * chosen_logratios
		rejected_rewards = beta * rejected_logratios
		delta = jnp.mean(jnp.concatenate([chosen_rewards, rejected_rewards]))
		return -logsigmoid((beta * chosen_logratios) - delta) - logsigmoid(
			-(beta * rejected_logratios - delta)
		)

	def _sppo_hard_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		a = chosen_logps - ref_chosen_logps
		b = rejected_logps - ref_rejected_logps
		return (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2

	def _nca_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
		rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
		return (
			-logsigmoid(chosen_rewards)
			- 0.5 * logsigmoid(-chosen_rewards)
			- 0.5 * logsigmoid(-rejected_rewards)
		)

	def _aot_pair_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		chosen_logratios = chosen_logps - ref_chosen_logps
		rejected_logratios = rejected_logps - ref_rejected_logps
		chosen_logratios_sorted = jnp.sort(chosen_logratios, axis=0)
		rejected_logratios_sorted = jnp.sort(rejected_logratios, axis=0)
		delta = chosen_logratios_sorted - rejected_logratios_sorted
		return (
			-logsigmoid(beta * delta) * (1 - label_smoothing)
			- logsigmoid(-beta * delta) * label_smoothing
		)

	def _aot_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		logratios = chosen_logps - rejected_logps
		ref_logratios = ref_chosen_logps - ref_rejected_logps
		logratios_sorted = jnp.sort(logratios, axis=0)
		ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
		delta = logratios_sorted - ref_logratios_sorted
		return (
			-logsigmoid(beta * delta) * (1 - label_smoothing)
			- logsigmoid(-beta * delta) * label_smoothing
		)

	def _apo_zero_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		chosen_logratios = chosen_logps - ref_chosen_logps
		rejected_logratios = rejected_logps - ref_rejected_logps
		losses_chosen = 1 - sigmoid(beta * chosen_logratios)
		losses_rejected = sigmoid(beta * rejected_logratios)
		return losses_chosen + losses_rejected

	def _apo_down_dpo_loss(
		chosen_logps: chex.Array,
		rejected_logps: chex.Array,
		ref_chosen_logps: chex.Array,
		ref_rejected_logps: chex.Array,
		beta: float,
		label_smoothing: float,
	):
		chosen_logratios = chosen_logps - ref_chosen_logps
		rejected_logratios = rejected_logps - ref_rejected_logps
		losses_chosen = sigmoid(beta * chosen_logratios)
		losses_rejected = 1 - sigmoid(beta * (chosen_logratios - rejected_logratios))
		return losses_chosen + losses_rejected

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
		"discopop": _discopop_dpo_loss,
	}.get(loss_type, None)
	assert loss_function is not None, f"given loss_type({loss_function}) is not valid"
	return loss_function
