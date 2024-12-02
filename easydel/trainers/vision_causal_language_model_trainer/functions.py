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

from typing import List

import chex
import jax
from fjformer import with_sharding_constraint
from fjformer.functions.loss_functions import cross_entropy_loss_and_accuracy
from flax.struct import dataclass
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.easystate import EasyDeLState


@dataclass
class VisionCausalLanguageModelStepOutput:
	loss: chex.Array
	text_loss: chex.Array
	text_accuracy: chex.Array
	vision_loss: chex.Array
	vision_accuracy: chex.Array


def create_vision_casual_language_model_train_step(
	partition_spec=PartitionSpec(("dp", "fsdp"), "sp"),  # noqa:B008
):
	"""The create_vision_casual_language_model_train_step function is a training step function that takes in the current
	 state of the model,and a batch of data. It then calculates the loss and accuracy for this batch, and returns
	an updated state with new parameters based on these gradients.

	Args:
	    partition_spec: Specify which devices the model will be split
	        across

	Returns:
	    A casual_language_model_train_step function that takes in the
	    current state of the model,
	"""

	def vision_casual_language_model_train_step(
		state, batch
	) -> List[EasyDeLState, chex.Array, VisionCausalLanguageModelStepOutput]:
		"""The vision_casual_language_model_train_step function is a training step function that takes in the current state
		of the model and a batch of data. It then calculates the loss and accuracy for this batch,
		and returns an updated state with new parameters based on these gradients.

		Args:
		    state: Store the model parameters
		    batch: Pass the data to the model

		Returns:
		    A tuple of (state, loss,
		    VisionCausalLanguageModelStepOutput)
		"""
		batch = with_sharding_constraint(batch, partition_spec)

		def calculate_loss(params):
			labels = batch.get("labels", None)
			if labels is None:
				labels = batch["input_ids"][..., 1:]
			else:
				labels = labels[..., 1:]
			label_vision_mask = batch.pop("label_vision_mask")

			model_outputs = state.apply_fn(
				params=params,
				**batch,
				return_dict=True,
				train=True,
			)
			logits = model_outputs.logits
			aux_loss = getattr(model_outputs, "aux_loss", None)

			vision_loss, vision_accuracy = cross_entropy_loss_and_accuracy(
				logits[:, :-1, :],
				jnp.where(label_vision_mask, labels, 0),
				batch["attention_mask"].astype(jnp.float32)[:, 1:] * label_vision_mask,
			)
			text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
				logits[:, :-1, :],
				jnp.where(label_vision_mask, 0, labels),
				batch["attention_mask"].astype(jnp.float32)[:, 1:] * (1.0 - label_vision_mask),
			)

			loss = 0.5 * (
				vision_loss + text_loss + (aux_loss if aux_loss is not None else 0.0)
			)

			return loss, VisionCausalLanguageModelStepOutput(
				loss=loss,
				text_accuracy=text_accuracy,
				vision_accuracy=vision_accuracy,
				text_loss=text_loss,
				vision_loss=vision_loss,
			)

		grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
		(loss__, metrics), grad = grad_fn(state.params)
		state = state.apply_gradients(grads=grad)
		return state, loss__, metrics

	return vision_casual_language_model_train_step


def create_vision_casual_language_model_evaluation_step(
	partition_spec=PartitionSpec(("dp", "fsdp"), "sp"),  # noqa:B008
):
	"""The create_vision_casual_language_model_evaluation_step function is used to create a function that calculates the
	 loss and accuracy of a model. It takes in a set of parameters, which are then passed into the state.apply_fn function
	to generate logits for each token in the batch. The cross entropy loss and accuracy are then calculated from these
	logits.

	Args:
	    partition_spec: Specify the partitioning of the model parameters

	Returns:
	    A function that can be used to calculate the loss and accuracy
	    of a model
	"""

	def vision_casual_language_model_evaluation_step(
		state, batch
	) -> List[EasyDeLState, chex.Array, VisionCausalLanguageModelStepOutput]:
		"""The vision_casual_language_model_train_step function is a training step function that takes in the current state
		of the model and a batch of data. It then calculates the loss and accuracy for this batch,
		and returns an updated state with new parameters based on these gradients.

		Args:
		    state: Store the model parameters
		    batch: Pass the data to the model

		Returns:
		    A tuple of (state, loss,
		    VisionCausalLanguageModelStepOutput)
		"""
		batch = with_sharding_constraint(batch, partition_spec)

		def calculate_loss(params):
			labels = batch.get("labels", None)
			if labels is None:
				labels = batch["input_ids"][..., 1:]
			else:
				labels = labels[..., 1:]
			label_vision_mask = batch.pop("label_vision_mask")
			model_outputs = state.apply_fn(
				params=params,
				**batch,
				return_dict=True,
				train=False,
			)
			logits = model_outputs.logits
			aux_loss = getattr(model_outputs, "aux_loss", None)

			vision_loss, vision_accuracy = cross_entropy_loss_and_accuracy(
				logits[:, :-1, :],
				jnp.where(label_vision_mask, labels, 0),
				batch["attention_mask"].astype(jnp.float32)[:, 1:] * label_vision_mask,
			)
			text_loss, text_accuracy = cross_entropy_loss_and_accuracy(
				logits[:, :-1, :],
				jnp.where(label_vision_mask, 0, labels),
				batch["attention_mask"].astype(jnp.float32)[:, 1:] * (1.0 - label_vision_mask),
			)

			loss = 0.5 * (
				vision_loss + text_loss + (aux_loss if aux_loss is not None else 0.0)
			)

			return loss, VisionCausalLanguageModelStepOutput(
				loss=loss,
				text_accuracy=text_accuracy,
				vision_accuracy=vision_accuracy,
				text_loss=text_loss,
				vision_loss=vision_loss,
			)

		loss__, metrics = calculate_loss(state.params)
		return loss__, metrics

	return vision_casual_language_model_evaluation_step
