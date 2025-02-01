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
import flax
import flax.nnx
import jax
from jax import numpy as jnp
from jax.nn import log_sigmoid as logsigmoid
from jax.nn import relu, sigmoid
from jax.sharding import PartitionSpec

from easydel.escale.partition.constraints import with_sharding_constraint
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
	make_assertions_and_get_sizes,
	minibatch_call,
	update_metrics,
	update_state_respectfully,
)
from ..utils import pad_to_length

LOSS_FN_VARIENTS = tp.Literal[
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
]


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


def get_loss_function(
	loss_type: LOSS_FN_VARIENTS,
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


def concatenated_forward(
	model: EasyDeLBaseModule,
	batch: tp.Dict[str, tp.Union[tp.List, chex.Array]],
	is_encoder_decoder: bool,
	label_pad_token_id: int,
	padding_value: int,
	max_length: int | None = None,
	truncation_mode: str = "keep_end",
	aux_loss_enabled: bool = False,
	loss_type: str = "sigmoid",
) -> tp.Dict[str, chex.Array]:
	"""Run model on concatenated chosen/rejected inputs for efficiency."""

	num_examples = batch["prompt_input_ids"].shape[0]
	concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

	model_kwargs = {}
	if aux_loss_enabled:
		model_kwargs["output_router_logits"] = True

	if "pixel_values" in concatenated_batch:
		model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
	if "pixel_attention_mask" in concatenated_batch:
		model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
	if "image_sizes" in concatenated_batch:
		model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

	prompt_input_ids = concatenated_batch["prompt_input_ids"]
	prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
	completion_input_ids = concatenated_batch["completion_input_ids"]
	completion_attention_mask = concatenated_batch["completion_attention_mask"]
	if is_encoder_decoder:
		labels = completion_input_ids
		labels = jnp.where(
			completion_attention_mask == 0,
			label_pad_token_id,
			completion_input_ids,
		)

		outputs = model(
			input_ids=prompt_input_ids,
			attention_mask=prompt_attention_mask,
			labels=labels,
			**model_kwargs,
		)
		logits = outputs.logits
		loss_mask = completion_attention_mask.astype(bool)
	else:
		input_ids = jnp.concatenate(
			[prompt_input_ids, completion_input_ids],
			axis=1,
		)
		attention_mask = jnp.concatenate(
			[prompt_attention_mask, completion_attention_mask],
			axis=1,
		)

		loss_mask = jnp.concatenate(
			[
				jnp.zeros_like(prompt_attention_mask),
				completion_attention_mask,
			],
			axis=1,
		)

		if max_length is not None:
			if truncation_mode == "keep_end":
				input_ids = input_ids[:, -max_length:]
				attention_mask = attention_mask[:, -max_length:]
				loss_mask = loss_mask[:, -max_length:]
			elif truncation_mode == "keep_start":
				input_ids = input_ids[:, :max_length]
				attention_mask = attention_mask[:, :max_length]
				loss_mask = loss_mask[:, :max_length]
			else:
				raise ValueError(
					f"Unknown truncation mode: '{truncation_mode}'. Should be one of ['keep_end', "
					"'keep_start']."
				)
		model_kwargs["input_ids"] = input_ids
		model_kwargs["attention_mask"] = attention_mask
		outputs = model(**model_kwargs)

		logits = outputs.logits
		labels = jnp.roll(input_ids, shift=-1, axis=1)
		loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype("bool")

	if logits.shape[:2] != labels.shape[:2]:
		seq_len = labels.shape[1]
		logits = logits[:, -seq_len:]

	labels = jnp.where(loss_mask, labels, 0)
	lsmax = jax.nn.log_softmax(logits, axis=-1)
	batch_size, seq_len = labels.shape
	per_token_logps = jnp.roll(
		jnp.where(
			loss_mask,
			lsmax[jnp.arange(batch_size)[:, None], jnp.arange(seq_len)[None, :], labels],
			0,
		),
		shift=1,
		axis=1,
	)

	all_logps = per_token_logps.sum(-1)

	if loss_type == "ipo":
		all_logps = all_logps / loss_mask.sum(-1)
	output = {}
	output["chosen_logps"] = all_logps[:num_examples]
	output["rejected_logps"] = all_logps[num_examples:]

	mean_chosen_logits = jnp.sum(
		jnp.where(
			loss_mask[:num_examples, :, None],
			logits[:num_examples],
			0,
		)
	) / jnp.sum(loss_mask[:num_examples])

	mean_rejected_logits = jnp.sum(
		jnp.where(
			loss_mask[num_examples:, :, None],
			logits[num_examples:],
			0,
		)
	) / jnp.sum(loss_mask[num_examples:])

	output["mean_chosen_logits"] = mean_chosen_logits
	output["mean_rejected_logits"] = mean_rejected_logits
	if aux_loss_enabled and hasattr(outputs, "aux_loss"):
		output["aux_loss"] = outputs.aux_loss
	return output


def training_step(
	state: EasyDeLState,
	batch: dict,
	reference_state: EasyDeLState,
	learning_rate_fn: tp.Callable,
	concatenated_forward: tp.Callable,
	beta: float = 0.1,
	label_smoothing: float = 0,
	loss_type: LOSS_FN_VARIENTS = "sigmoid",
	reference_free: bool = False,
	ref_precalculated: bool = True,
	loss_config: tp.Optional[LossConfig] = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
) -> tp.Tuple[EasyDeLState, LossMetrics]:
	batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=partition_spec,
	)

	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)
	_loss_func = get_loss_function(
		loss_type=loss_type,
		beta=beta,
		label_smoothing=label_smoothing,
	)

	def calculate_loss(tree: flax.nnx.GraphState, call_batch):
		model_output = concatenated_forward(state.merge(tree=tree), call_batch)

		if ref_precalculated:
			ref_chosen_logps = jax.lax.stop_gradient(call_batch["ref_chosen_logps"])
			ref_rejected_logps = jax.lax.stop_gradient(call_batch["ref_rejected_logps"])
		else:
			rfm = reference_state.model
			rfm.eval()
			out = jax.lax.stop_gradient(concatenated_forward(rfm, call_batch))
			ref_chosen_logps = out["chosen_logps"]
			ref_rejected_logps = out["rejected_logps"]

		chosen_logps = model_output["chosen_logps"]
		rejected_logps = model_output["rejected_logps"]
		losses = _loss_func(
			chosen_logps,
			rejected_logps,
			ref_chosen_logps,
			ref_rejected_logps,
			beta,
			label_smoothing,
		)

		chosen_rewards = beta * jax.lax.stop_gradient(chosen_logps - ref_chosen_logps)
		rejected_rewards = beta * jax.lax.stop_gradient(rejected_logps - ref_rejected_logps)
		if hasattr(model_output, "aux_loss"):
			losses += model_output["aux_loss"]

		metrics = LossMetrics(
			loss=losses.mean(),
			rejected_rewards=rejected_rewards,
			chosen_rewards=chosen_rewards,
		)
		return metrics.loss, metrics

	gradients, metrics = minibatch_call(
		state=state,
		batch=batch,
		minibatch_size=minibatch_size,
		grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
	)

	metrics = update_metrics(
		metrics=metrics,
		learning_rate_fn=learning_rate_fn,
		step=state.step,
		gradients=gradients,
	)
	state = update_state_respectfully(
		state=state,
		gradients=gradients,
		loss_config=loss_config,
		metrics=metrics,
	)
	return (state, metrics)


def evaluation_step(
	state: EasyDeLState,
	batch: dict,
	concatenated_forward: tp.Callable,
	reference_state: EasyDeLState = None,
	beta: float = 0.1,
	label_smoothing: float = 0,
	loss_type: LOSS_FN_VARIENTS = "sigmoid",
	ref_precalculated: bool = True,
	reference_free: bool = False,
	partition_spec: tp.Optional[PartitionSpec] = None,
) -> LossMetrics:
	*_, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=1,
		batch_partition_spec=partition_spec,
	)

	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)
	_loss_func = get_loss_function(
		loss_type=loss_type,
		beta=beta,
		label_smoothing=label_smoothing,
	)

	def calculate_loss(tree: flax.nnx.GraphState):
		(
			mean_chosen_logits,
			mean_rejected_logits,
			_,
			_,
		) = concatenated_forward(state.merge(tree), batch)

		if ref_precalculated:
			ref_chosen_logps = batch["ref_chosen_logps"]
			ref_rejected_logps = batch["ref_rejected_logps"]
		else:
			if reference_state is None:
				(
					ref_chosen_logps,
					ref_rejected_logps,
					_,
					_,
				) = concatenated_forward(state.model, batch)
			else:
				(
					ref_chosen_logps,
					ref_rejected_logps,
					_,
					_,
				) = concatenated_forward(reference_state.model, batch)

		pi_log_ratios = mean_chosen_logits - mean_rejected_logits

		if reference_free:
			ref_log_ratios = 0
		else:
			ref_log_ratios = ref_chosen_logps - ref_rejected_logps

		logits = pi_log_ratios - ref_log_ratios
		losses = _loss_func(
			logits,
			mean_chosen_logits,
			ref_chosen_logps,
			mean_rejected_logits,
			ref_rejected_logps,
		)
		chosen_rewards = beta * (mean_chosen_logits - ref_chosen_logps)
		rejected_rewards = beta * (mean_rejected_logits - ref_rejected_logps)
		metrics = LossMetrics(
			loss=losses.mean(),
			rejected_rewards=rejected_rewards,
			chosen_rewards=chosen_rewards,
		)
		return metrics

	metrics = calculate_loss(state.graphstate)
	return metrics
