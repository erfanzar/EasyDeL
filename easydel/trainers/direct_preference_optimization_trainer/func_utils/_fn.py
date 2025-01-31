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
