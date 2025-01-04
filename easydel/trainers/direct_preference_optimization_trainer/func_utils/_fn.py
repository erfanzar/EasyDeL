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
import os
import typing as tp

import chex
import flax
import flax.nnx
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.escale.partition.constraints import with_sharding_constraint
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ...training_utils import (
	make_assertions_and_get_sizes,
	minibatch_call,
	update_metrics,
	update_state_respectfully,
)
from .concatenators import concatenated_inputs
from .log_probs import get_batch_log_probs
from .loss_funcs import get_loss_function

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


def concatenated_forward(
	model: EasyDeLBaseModule,
	batch: tp.Dict[str, tp.Union[tp.List, chex.Array]],
	is_encoder_decoder: bool = False,
	label_pad_token_id: int = -100,
	padding_value: int = 0,
	fixed_max_length: int | None = None,
) -> tp.Dict[str, chex.Array]:
	"""Run model on concatenated chosen/rejected inputs for efficiency."""

	# Get concatenated inputs
	concatenated_batch = concatenated_inputs(
		batch=batch,
		padding_value=padding_value,
		fixed_max_length=fixed_max_length,
	)

	num_examples = batch["prompt_input_ids"].shape[0]
	model_kwargs = {}

	# Add image-related features
	for k in ["pixel_values", "pixel_attention_mask", "image_sizes"]:
		if k in concatenated_batch:
			model_kwargs[k] = concatenated_batch[f"{k}"]

	if is_encoder_decoder:
		# Handle encoder-decoder models
		labels = concatenated_batch["completion_input_ids"]
		labels = jnp.where(
			concatenated_batch["completion_attention_mask"] == 0,
			label_pad_token_id,
			labels,
		)
		model_kwargs["labels"] = labels

		outputs = model(
			input_ids=concatenated_batch["prompt_input_ids"],
			attention_mask=concatenated_batch["prompt_attention_mask"],
			**model_kwargs,
		)
		logits = outputs.logits
		loss_mask = concatenated_batch["completion_attention_mask"].astype(bool)

	else:
		# Handle decoder-only models
		input_ids = jnp.concatenate(
			[
				concatenated_batch["prompt_input_ids"],
				concatenated_batch["completion_input_ids"],
			],
			axis=1,
		)

		attention_mask = jnp.concatenate(
			[
				concatenated_batch["prompt_attention_mask"],
				concatenated_batch["completion_attention_mask"],
			],
			axis=1,
		)

		loss_mask = jnp.concatenate(
			[
				jnp.zeros_like(concatenated_batch["prompt_attention_mask"]),
				concatenated_batch["completion_attention_mask"],
			],
			axis=1,
		)

		# Apply max length if specified
		if fixed_max_length is not None:
			input_ids = input_ids[:, :fixed_max_length]
			attention_mask = attention_mask[:, :fixed_max_length]
			loss_mask = loss_mask[:, :fixed_max_length]

		outputs = model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			**model_kwargs,
		)

		logits = outputs.logits[:, :-1, :]
		labels = input_ids[:, 1:].copy()
		loss_mask = loss_mask[:, 1:].astype(bool)

	# Calculate log probabilities
	per_token_logps = get_batch_log_probs(
		logits,
		labels,
		average_log_prob=False,
		is_encoder_decoder=is_encoder_decoder,
		label_pad_token_id=label_pad_token_id,
	)
	per_token_logps = jnp.where(
		jnp.broadcast_shapes(loss_mask, per_token_logps.shape),
		per_token_logps,
		0,
	)
	all_logps = per_token_logps.sum(-1)

	# Prepare output dictionary
	output = {
		"chosen_logps": all_logps[:num_examples],
		"rejected_logps": all_logps[num_examples:],
		"mean_chosen_logits": jnp.mean(logits[:num_examples][loss_mask[:num_examples]]),
		"mean_rejected_logits": jnp.mean(logits[num_examples:][loss_mask[num_examples:]]),
	}

	if os.environ("AUX_LOSS_ENABLED_DPO", "true") in [
		"true",
		"1",
		"os",
		"yes",
	] and hasattr(outputs, "aux_loss"):
		output["aux_loss"] = outputs.aux_loss

	return output


def training_step(
	state: EasyDeLState,
	batch: dict,
	learning_rate_fn: tp.Callable,
	concatenated_forward: tp.Callable,
	reference_state: EasyDeLState = None,
	beta: float = 0.1,
	label_smoothing: float = 0,
	loss_type: LOSS_FN_VARIENTS = "sigmoid",
	reference_free: bool = False,
	loss_config: tp.Optional[LossConfig] = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
) -> tuple[EasyDeLState, LossMetrics]:
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

		if "ref_chosen_logps" in call_batch and "ref_rejected_logps" in call_batch:
			ref_chosen_logps = call_batch["ref_chosen_logps"]
			ref_rejected_logps = call_batch["ref_rejected_logps"]
		else:
			if reference_state is None:
				out = concatenated_forward(state.model, call_batch)
			else:
				out = concatenated_forward(reference_state.model, call_batch)

			ref_chosen_logps = out["ref_chosen_logps"]
			ref_rejected_logps = out["ref_rejected_logps"]

		chosen_logps = model_output["chosen_logps"]
		rejected_logps = model_output["rejected_logps"]
		losses = _loss_func(
			chosen_logps,
			rejected_logps,
			ref_chosen_logps,
			ref_rejected_logps,
		)

		chosen_rewards = beta * jax.lax.stop_gradient(chosen_logps - ref_chosen_logps)
		rejected_rewards = beta * jax.lax.stop_gradient(rejected_logps - ref_rejected_logps)
		if hasattr(model_output, "aux_loss"):
			losses += model_output["aux_loss"]
		jax.debug.print("chosen_logps {}", chosen_logps)
		jax.debug.print("rejected_logps {}", rejected_logps)
		jax.debug.print("ref_chosen_logps {}", ref_chosen_logps)
		jax.debug.print("ref_rejected_logps {}", ref_rejected_logps)
		jax.debug.print("LOSS {}", losses)
		jax.debug.print("MEAN {}", jnp.mean(losses))
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

		if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
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
