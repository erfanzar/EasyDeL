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

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple, Union

import chex
import fjformer
import flax
import jax

from easydel.etils.easystate import EasyDeLState
from easydel.trainers.direct_preference_optimization_trainer.func_utils.concatenators import (
	concatenated_dpo_inputs,
)
from easydel.trainers.direct_preference_optimization_trainer.func_utils.log_probs import (
	get_batch_log_probs,
)
from easydel.trainers.direct_preference_optimization_trainer.func_utils.loss_funcs import (
	get_loss_function,
)


@jax.tree_util.register_pytree_node_class
@dataclass
class DPOStepOut:
	loss: chex.Array
	chosen_rewards: chex.Array
	rejected_rewards: chex.Array

	@classmethod
	def tree_unflatten(cls, aux, children):
		return cls(*children)

	def tree_flatten(self):
		return (self.loss, self.chosen_rewards, self.rejected_rewards), {}


def create_dpo_concatenated_forward(
	is_encoder_decoder,
	label_pad_token_id,
	padding_value,
	truncation_mode: Literal["keep_end", "keep_start"] = "keep_end",
	fixed_max_length: int | None = None,
):
	"""The create_dpo_concatenated_forward function is a helper function that creates a forward pass function for the
	model. The forward pass function takes in an apply_fn, which is the model's apply_fn, and runs it on concatenated
	inputs. It returns chosen log probs, rejected log probs, chosen logits and rejected logits.

	Args:
			is_encoder_decoder: Determine whether the model is an encoder-
					decoder model or not
			label_pad_token_id: Pad the labels to the same length
			padding_value: Pad the inputs to the same length
			truncation_mode: typing.Literal["keep_end","keep_start"]: where
					to pad and where to keep.
			fixed_max_length: int|None: by providing fixed_max_length the
					func will always return a fixed sequence length
	and won't use dynamic methods.

	Returns:
			A function that takes in a apply_fn, params and a batch of
			inputs,
	"""

	# Will be Moved under jax.jit.
	@fjformer.core.implicit_compact
	def concatenated_forward(
		apply_fn: Callable,
		params: dict | flax.core.FrozenDict,
		batch: Dict[str, Union[List, chex.Array]],
	) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
		"""The concatenated_forward function is used to compute the log-probabilities of both chosen and rejected labels.

		Args:
				apply_fn: Callable: Pass in the model function
				params: dict | flax.core.FrozenDict: Pass the model
						parameters to the function
				batch: Dict[str, Union[List, chex.Array]] : Pass the batch
						of data to the concatenated_forward function

		Returns:
				The log_probs of the chosen and rejected labels, as well as
				their corresponding logits
		"""
		assert (
			padding_value is not None
		), "`padding_value` can not be set as `None` it must be an integer."
		concatenated_batch = concatenated_dpo_inputs(
			batch,
			is_encoder_decoder=is_encoder_decoder,
			label_pad_token_id=label_pad_token_id,
			padding_value=padding_value,
			fixed_max_length=fixed_max_length,
		)
		len_chosen = batch["chosen_labels"].shape[0]
		concatenated_batch["concatenated_input_ids"] = concatenated_batch[
			"concatenated_input_ids"
		].reshape(concatenated_batch["concatenated_input_ids"].shape[0], -1)
		concatenated_batch["concatenated_labels"] = concatenated_batch[
			"concatenated_labels"
		].reshape(concatenated_batch["concatenated_labels"].shape[0], -1)
		concatenated_batch["concatenated_attention_mask"] = concatenated_batch[
			"concatenated_attention_mask"
		].reshape(concatenated_batch["concatenated_attention_mask"].shape[0], -1)
		model_kwargs = (
			{
				"labels": concatenated_batch["concatenated_labels"],
				"decoder_input_ids": concatenated_batch.pop(
					"concatenated_decoder_input_ids", None
				),
			}
			if is_encoder_decoder
			else {}
		)
		all_logits = apply_fn(
			concatenated_batch["concatenated_input_ids"],
			attention_mask=concatenated_batch["concatenated_attention_mask"],
			params=params,
			**model_kwargs,
		).logits

		all_log_probs = get_batch_log_probs(
			all_logits,
			concatenated_batch["concatenated_labels"],
			average_log_prob=False,
			is_encoder_decoder=is_encoder_decoder,
			label_pad_token_id=label_pad_token_id,
		)
		chosen_log_probs = all_log_probs[:len_chosen]
		rejected_log_probs = all_log_probs[len_chosen:]

		chosen_logits = all_logits[:len_chosen]
		rejected_logits = all_logits[len_chosen:]

		return chosen_log_probs, rejected_log_probs, chosen_logits, rejected_logits

	return concatenated_forward


def create_dpo_train_function(
	concatenated_forward: Callable,
	ref_state: EasyDeLState = None,
	beta: float = 0.1,
	label_smoothing: float = 0,
	loss_type: Literal[
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
	] = "sigmoid",
	reference_free: bool = False,
):
	"""The create_dpo_train_function function is a helper function that creates the DPO training step.

	Args:
			concatenated_forward: Callable: Define the forward pass of the model
			ref_state: EasyDeLState: Specify the reference policy
			beta: float: Scale the logits
			label_smoothing: float: Smooth the labels
			loss_type (str): Determine the loss function
			reference_free: bool: Indicate whether the reference policy isused or not
	Returns:
			A function that takes in a state and a batch
	"""
	_loss_func = get_loss_function(
		loss_type=loss_type,
		beta=beta,
		label_smoothing=label_smoothing,
	)

	def dpo_step(state: EasyDeLState, batch: dict) -> tuple[EasyDeLState, DPOStepOut]:
		"""The dpo_step function is the core of DPO. It takes a state and a batch,
		and returns an updated state. The update is done by calculating the loss
		for each example in the batch, then taking its gradient with respect to
		the parameters of the policy network (which are stored in `state`). This
		gradient is then used to update `state`.

		Args:
				state: EasyDeLState: Store the parameters of the model
				batch: dict: Pass the data to the model

		Returns:
				A new state, which is a collection of the parameters and
				apply_fn
		"""

		def calculate_loss(params: dict | flax.core.FrozenDict):
			(
				policy_chosen_log_probs,
				policy_rejected_log_probs,
				policy_chosen_logits,
				policy_rejected_logits,
			) = concatenated_forward(state.apply_fn, params, batch)

			if (
				"reference_chosen_log_probs" in batch
				and "reference_rejected_log_probs" in batch
			):
				reference_chosen_log_probs = batch["reference_chosen_log_probs"]
				reference_rejected_log_probs = batch["reference_rejected_log_probs"]
			else:
				if ref_state is None:
					(
						reference_chosen_log_probs,
						reference_rejected_log_probs,
						_,
						_,
					) = concatenated_forward(
						state.apply_fn,
						state.params,
						batch,
					)
				else:
					(
						reference_chosen_log_probs,
						reference_rejected_log_probs,
						_,
						_,
					) = concatenated_forward(
						ref_state.apply_fn,
						ref_state.params,
						batch,
					)
			reference_chosen_log_probs = jax.lax.stop_gradient(reference_chosen_log_probs)
			reference_rejected_log_probs = jax.lax.stop_gradient(reference_rejected_log_probs)
			pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

			if reference_free:
				ref_log_ratios = 0
			else:
				ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

			logits = pi_log_ratios - ref_log_ratios
			losses = _loss_func(
				logits,
				policy_chosen_log_probs,
				reference_chosen_log_probs,
				policy_rejected_log_probs,
				reference_rejected_log_probs,
			)
			chosen_rewards = beta * jax.lax.stop_gradient(
				policy_chosen_log_probs - reference_chosen_log_probs
			)
			rejected_rewards = beta * jax.lax.stop_gradient(
				policy_rejected_log_probs - reference_rejected_log_probs
			)

			return losses.mean(), (chosen_rewards, rejected_rewards)

		grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
		(__loss, (__chosen_rewards, __rejected_rewards)), grads = grad_fn(state.params)
		new_state = state.apply_gradients(grads=grads)
		return new_state, DPOStepOut(
			loss=__loss,
			rejected_rewards=__rejected_rewards,
			chosen_rewards=__chosen_rewards,
		)

	return dpo_step


def create_dpo_eval_function(
	concatenated_forward: Callable,
	ref_state: EasyDeLState = None,
	beta: float = 0.1,
	label_smoothing: float = 0,
	loss_type: Literal[
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
	] = "sigmoid",
	reference_free: bool = False,
):
	"""The create_dpo_eval_function function is a helper function that creates the DPO evaluating step.

	Args:
			concatenated_forward: Callable: Define the forward pass of the
					model
			ref_state: EasyDeLState: Specify the reference policy
			beta: float: Scale the logits
			label_smoothing: float: Smooth the labels
			loss_type: Literal["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "aot_pair", "apo_zero", "apo_down"]: Determine
					the loss function
			reference_free: bool: Indicate whether the reference policy is
					used or not

	Returns:
			A function that takes in a state and a batch
	"""
	_loss_func = get_loss_function(
		loss_type=loss_type,
		beta=beta,
		label_smoothing=label_smoothing,
	)

	def dpo_step(state: EasyDeLState, batch: dict) -> DPOStepOut:
		"""The dpo_step function is the core of DPO. It takes a state and a batch,
		and returns an updated state. The update is done by calculating the loss
		for each example in the batch, then taking its gradient with respect to
		the parameters of the policy network (which are stored in `state`). This
		gradient is then used to update `state`.

		Args:
				state: EasyDeLState: Store the parameters of the model
				batch: dict: Pass the data to the model

		Returns:
				A `DPOStepOut` class
		"""

		def calculate_loss(params: dict | flax.core.FrozenDict):
			(
				policy_chosen_log_probs,
				policy_rejected_log_probs,
				policy_chosen_logits,
				policy_rejected_logits,
			) = concatenated_forward(state.apply_fn, params, batch)

			if (
				"reference_chosen_log_probs" in batch
				and "reference_rejected_log_probs" in batch
			):
				reference_chosen_log_probs = batch["reference_chosen_log_probs"]
				reference_rejected_log_probs = batch["reference_rejected_log_probs"]
			else:
				if ref_state is None:
					(
						reference_chosen_log_probs,
						reference_rejected_log_probs,
						_,
						_,
					) = concatenated_forward(state.apply_fn, state.params, batch)
				else:
					(
						reference_chosen_log_probs,
						reference_rejected_log_probs,
						_,
						_,
					) = concatenated_forward(ref_state.apply_fn, ref_state.params, batch)

			pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs

			if reference_free:
				ref_log_ratios = 0
			else:
				ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

			logits = pi_log_ratios - ref_log_ratios
			losses = _loss_func(
				logits,
				policy_chosen_log_probs,
				reference_chosen_log_probs,
				policy_rejected_log_probs,
				reference_rejected_log_probs,
			)
			chosen_rewards = beta * (policy_chosen_log_probs - reference_chosen_log_probs)
			rejected_rewards = beta * (
				policy_rejected_log_probs - reference_rejected_log_probs
			)
			return losses[0], (chosen_rewards, rejected_rewards)

		__loss, (__chosen_rewards, __rejected_rewards) = calculate_loss(state.params)

		return DPOStepOut(
			loss=__loss,
			rejected_rewards=__rejected_rewards,
			chosen_rewards=__chosen_rewards,
		)

	return dpo_step
