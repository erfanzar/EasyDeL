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

from typing import Any, Dict, Literal

import jax
import jax.tree_util as jtu
from fjformer.sharding import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec


def create_sequence_classification_model_train_step(
	partition_spec=PartitionSpec(("dp", "fsdp"), "sp"),  # noqa:B008
	gradient_accumulation_steps: int = 1,
	problem_type: Literal[
		"regression", "single_label_classification", "multi_label_classification"
	] = "regression",
	num_labels: int = 1,
):
	"""Creates an enhanced training step function for sequence classification models.

	Args:
	    partition_spec: Specification for model partitioning across devices
	    gradient_accumulation_steps: Number of steps to accumulate gradients
	    problem_type: Type of problem (regression, single_label_classification, or multi_label_classification)
	    num_labels: Number of labels for classification tasks

	Returns:
	    A training step function
	"""

	def mse_loss(preds, targets):
		"""Mean squared error loss."""
		return jnp.mean((preds - targets) ** 2)

	def cross_entropy_loss(preds, targets):
		"""Cross entropy loss with optional label smoothing."""
		logits = jax.nn.log_softmax(preds, axis=-1)
		return -jnp.mean(jnp.sum(jax.nn.one_hot(targets, num_labels) * logits, axis=-1))

	def binary_cross_entropy_loss(preds, targets):
		"""Binary cross entropy loss with logits."""
		log_probs = jax.nn.log_sigmoid(preds)
		log_not_probs = jax.nn.log_sigmoid(-preds)
		return -jnp.mean(targets * log_probs + (1.0 - targets) * log_not_probs)

	if problem_type == "regression":
		loss_fn = mse_loss
	elif problem_type == "single_label_classification":
		loss_fn = cross_entropy_loss
	elif problem_type == "multi_label_classification":
		loss_fn = binary_cross_entropy_loss
	else:
		raise ValueError(f"Unsupported problem type: {problem_type}")

	assert (
		gradient_accumulation_steps > 0
	), "gradient_accumulation_steps must be greater than 0"

	def compute_metrics(
		logits: jnp.ndarray, labels: jnp.ndarray
	) -> Dict[str, jnp.ndarray]:
		"""Compute problem-specific metrics.

		Args:
		    logits: Model output logits
		    labels: Ground truth labels

		Returns:
		    Dictionary of computed metrics
		"""
		metrics = {}

		if problem_type == "single_label_classification":
			predictions = jnp.argmax(logits, axis=-1)
			accuracy = jnp.mean(predictions == labels)
			metrics["accuracy"] = accuracy
		elif problem_type == "multi_label_classification":
			predictions = jnp.where(jax.nn.sigmoid(logits) > 0.5, 1, 0)
			accuracy = jnp.mean(predictions == labels)
			metrics["accuracy"] = accuracy
		elif problem_type == "regression":
			mse = jnp.mean((logits - labels) ** 2)
			mae = jnp.mean(jnp.abs(logits - labels))
			metrics.update(
				{
					"mse": mse,
					"mae": mae,
				}
			)

		return metrics

	def sequence_classification_model_train_step(
		state, batch
	) -> tuple[Any, float, Dict[str, float]]:
		"""Training step function for sequence classification models.

		Args:
		    state: Current model state
		    batch: Batch of training data

		Returns:
		    Tuple of (updated_state, loss, metrics)
		"""
		batch = with_sharding_constraint(batch, partition_spec)

		def calculate_loss_and_metrics(params):
			"""Calculate loss and metrics for the current batch."""
			labels = batch.pop("labels")
			batch_copy = dict(batch)  # Create a copy to avoid modifying the original

			# Forward pass
			model_outputs = state.apply_fn(
				params=params,
				**batch_copy,
				return_dict=True,
				train=True,
			)
			logits = model_outputs.logits[:, -1, :]

			# Calculate primary loss
			loss = loss_fn(logits, labels)

			# Calculate additional metrics
			metrics = compute_metrics(logits, labels)

			return loss, metrics

		# Calculate gradients with metrics
		grad_fn = jax.value_and_grad(calculate_loss_and_metrics, has_aux=True)
		(loss__, metrics__), grad = grad_fn(state.params)

		# Compute gradient statistics
		grad_norms = jtu.tree_map(jnp.linalg.norm, grad)
		max_grad_norm = jtu.tree_reduce(jnp.maximum, grad_norms)
		mean_grad_norm = jtu.tree_reduce(
			jnp.add, jtu.tree_map(jnp.sum, grad_norms)
		) / jtu.tree_reduce(jnp.add, jtu.tree_map(jnp.size, grad_norms))

		# Update model state
		state = state.apply_gradients(grads=grad)

		# Add gradient metrics to the metrics dictionary
		metrics__.update(
			{
				"max_grad_norm": max_grad_norm,
				"mean_grad_norm": mean_grad_norm,
				"loss": loss__,
			}
		)

		return state, loss__, metrics__

	return sequence_classification_model_train_step


def create_sequence_classification_model_eval_step(
	partition_spec=PartitionSpec(("dp", "fsdp"), "sp"),  # noqa:B008
	problem_type: Literal[
		"regression", "single_label_classification", "multi_label_classification"
	] = "regression",
	num_labels: int = 1,
):
	"""Creates an evaluation step function for sequence classification models.

	Args:
	    partition_spec: Specification for model partitioning across devices
	    problem_type: Type of problem (regression, single_label_classification, or multi_label_classification)
	    num_labels: Number of labels for classification tasks

	Returns:
	    An evaluation step function
	"""

	def mse_loss(preds, targets):
		"""Mean squared error loss."""
		return jnp.mean((preds - targets) ** 2)

	def cross_entropy_loss(preds, targets):
		"""Cross entropy loss with optional label smoothing."""
		logits = jax.nn.log_softmax(preds, axis=-1)
		return -jnp.mean(jnp.sum(jax.nn.one_hot(targets, num_labels) * logits, axis=-1))

	def binary_cross_entropy_loss(preds, targets):
		"""Binary cross entropy loss with logits."""
		log_probs = jax.nn.log_sigmoid(preds)
		log_not_probs = jax.nn.log_sigmoid(-preds)
		return -jnp.mean(targets * log_probs + (1.0 - targets) * log_not_probs)

	# Select appropriate loss function based on problem type
	if problem_type == "regression":
		loss_fn = mse_loss
	elif problem_type == "single_label_classification":
		loss_fn = cross_entropy_loss
	elif problem_type == "multi_label_classification":
		loss_fn = binary_cross_entropy_loss
	else:
		raise ValueError(f"Unsupported problem type: {problem_type}")

	def sequence_classification_model_eval_step(state, batch_eval):
		"""Evaluation step function for sequence classification models.

		Args:
		    state: Current model state
		    batch_eval: Batch of evaluation data

		Returns:
		    Tuple of (loss, metrics)
		"""
		batch_eval = with_sharding_constraint(batch_eval, partition_spec)

		def calculate_metrics(params):
			labels = batch_eval.pop("labels")

			# Forward pass
			model_outputs = state.apply_fn(
				params=params,
				**batch_eval,
				return_dict=True,
				train=False,  # Ensure dropout and other training-specific features are disabled
			)
			logits = model_outputs.logits[:, -1, :]

			# Calculate loss
			loss = loss_fn(logits, labels)

			# Calculate additional metrics based on problem type
			metrics = {"loss": loss}

			if problem_type == "single_label_classification":
				# Add accuracy for classification
				predictions = jnp.argmax(logits, axis=-1)
				accuracy = jnp.mean(predictions == labels)
				metrics["accuracy"] = accuracy
			elif problem_type == "multi_label_classification":
				# Add binary accuracy for multi-label classification
				predictions = jnp.where(jax.nn.sigmoid(logits) > 0.5, 1, 0)
				accuracy = jnp.mean(predictions == labels)
				metrics["accuracy"] = accuracy
			elif problem_type == "regression":
				# Add MSE and MAE for regression
				mse = jnp.mean((logits - labels) ** 2)
				mae = jnp.mean(jnp.abs(logits - labels))
				metrics.update(
					{
						"mse": mse,
						"mae": mae,
					}
				)

			return metrics

		# Compute metrics without gradient calculation
		metrics = calculate_metrics(state.params)

		return metrics

	return sequence_classification_model_eval_step
