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

from typing import Optional

import jax
from fjformer.functions.loss_functions import cross_entropy_loss_and_accuracy
from fjformer.sharding import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec


def create_seq2seq_model_train_step(
	partition_spec: Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
):
	"""The create_seq2seq_model_train_step function is a training step function that takes in the current state
	of the model,and a batch of data. It then calculates the loss and accuracy for this batch, and returns
	an updated state with new parameters based on these gradients.

	Args:
	    partition_spec (PartitionSpec): Specify which devices the model will be split across
	    gradient_accumulation_steps (int) : gradient accumulation step size from arguments

	Returns:
	    A seq2seq_model_train_step function that takes in the
	    current state of the model,
	"""
	if partition_spec is None:
		partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
	assert (
		gradient_accumulation_steps > 0
	), "gradient_accumulation_steps must be greater than 0"  # Ignore

	def seq2seq_model_train_step(state, batch):
		"""The seq2seq_model_train_step function is a training step function that takes in the current state
		of the model and a batch of data. It then calculates the loss and accuracy for this batch,
		and returns an updated state with new parameters based on these gradients.

		Args:
		    state: Store the model parameters
		    batch: Pass the data to the model
		Returns:
		    A tuple of (state, loss, metrics)
		"""
		batch = with_sharding_constraint(batch, partition_spec)

		def calculate_loss(params):
			labels = batch.pop("labels", None)
			model_outputs = state.apply_fn(
				params=params,
				**batch,
				return_dict=True,
				deterministic=False,
				train=True,
			)
			logits = model_outputs.logits
			aux_loss = getattr(model_outputs, "aux_loss", None)

			loss, accuracy = cross_entropy_loss_and_accuracy(
				logits,
				labels,
				batch.get("decoder_attention_mask", None),
			)
			if aux_loss is not None:
				loss += aux_loss
			return loss, (accuracy, aux_loss)

		grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
		(loss__, (accuracy__, aux_loss__)), grad = grad_fn(state.params)
		state = state.apply_gradients(grads=grad)

		grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grad)
		max_grad_norm = jax.tree_util.tree_reduce(jnp.maximum, grad_norms)
		mean_grad_norm = jax.tree_util.tree_reduce(
			jnp.add, jax.tree_util.tree_map(jnp.sum, grad_norms)
		) / jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(jnp.size, grad_norms))
		metrics = {
			"accuracy": accuracy__,
			"max_grad_norm": max_grad_norm,
			"mean_grad_norm": mean_grad_norm,
			"grad_norms": grad_norms,
		}
		if aux_loss__ is not None:
			metrics.update({"aux_loss": aux_loss__})
		return state, loss__, metrics

	return seq2seq_model_train_step


def create_seq2seq_model_evaluation_step(
	partition_spec: Optional[PartitionSpec] = None,
):
	"""
	The create_seq2seq_model_evaluation_step function is used to create a function that calculates the loss
	and accuracy of a model. It takes in a set of parameters, which are then passed into the state.apply_fn function
	to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from these
	logits.

	Args:
	    partition_spec: Specify the partitioning of the model parameters

	Returns:
	    A function that can be used to calculate the loss and accuracy
	    of a model
	"""

	if partition_spec is None:
		partition_spec = PartitionSpec(("dp", "fsdp"), "sp")

	def seq2seq_model_evaluation_step(state, batch_eval):
		"""The seq2seq_model_evaluation_step function is used to calculate the loss and accuracy of a model.
		It takes in a set of parameters, which are then passed into the state.apply_fn function
		to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from
		these logits.

		Args:
		    state: Store the model parameters and other information
		        about the training process
		    batch_eval: Pass the batch of data to the function

		Returns:
		    The loss and accuracy of the model
		"""
		batch_eval = with_sharding_constraint(batch_eval, partition_spec)

		def calculate_loss(params):
			"""
			The calculate_loss function is used to calculate the loss and accuracy of a model.
			It takes in a set of parameters, which are then passed into the state.apply_fn function
			to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated
			from these logits.
			"""
			labels = batch_eval.pop("labels", None)
			model_outputs = state.apply_fn(
				params=params,
				**batch_eval,
				return_dict=True,
				train=False,
			)
			logits = model_outputs.logits
			aux_loss = getattr(model_outputs, "aux_loss", None)

			loss, accuracy = cross_entropy_loss_and_accuracy(
				logits,
				labels,
				batch_eval.get("decoder_attention_mask", None),
			)
			if aux_loss is not None:
				loss += aux_loss
			return loss, (accuracy, aux_loss)

		loss__, (accuracy__, aux_loss__) = calculate_loss(state.params)
		return loss__, accuracy__, aux_loss__

	return seq2seq_model_evaluation_step
