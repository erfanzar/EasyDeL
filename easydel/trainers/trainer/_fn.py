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


def training_step(
	state: EasyDeLState,
	batch: tp.Mapping[str, jax.Array],
	loss_config: tp.Optional[LossConfig] = None,
	learning_rate_fn: optax.Schedule = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
) -> tp.Tuple[EasyDeLState, LossMetrics]:
	"""
	Performs a single training step by computing gradients via minibatch processing,
	updating the model state, and returning updated state and loss metrics.

	The function first determines the batch and minibatch sizes using assertions.
	It then applies sharding constraints to the batch. The loss function is defined
	as an inner function that merges the current model state with an updated tree,
	prepares the inputs, and computes the loss using the model's compute_loss method.
	Gradients are computed using `jax.value_and_grad` over minibatches. The state is updated
	respectfully using the computed gradients and updated metrics.

	Args:
	    state (EasyDeLState): The current model state, which includes parameters and model graph.
	    batch (tp.Mapping[str, jax.Array]): A mapping of input arrays for the current batch.
	    loss_config (tp.Optional[LossConfig], optional): Configuration settings for the loss
	        computation. Defaults to None.
	    learning_rate_fn (optax.Schedule, optional): A schedule function for the learning rate.
	        Defaults to None.
	    partition_spec (tp.Optional[PartitionSpec], optional): Specification for data sharding.
	        Defaults to None.
	    gradient_accumulation_steps (int, optional): Number of steps over which to accumulate gradients.
	        Defaults to 1.

	Returns:
	    tp.Tuple[EasyDeLState, LossMetrics]:
	        A tuple containing:
	            - The updated EasyDeLState after applying gradients.
	            - LossMetrics containing computed loss and other related metrics.
	"""
	# Determine batch size, minibatch size, and enforce partition spec.
	batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	def loss_fn(tree, minibatch):
		"""
		Computes the loss and additional metrics for a given minibatch and tree state.

		The inner function merges the current graph state with an updated tree,
		prepares inputs for a model call, pops the labels from the inputs, and calls
		the model's compute_loss method.

		Args:
		    tree: The current update to the model's graph state.
		    minibatch: A minibatch of input data.

		Returns:
		    A tuple containing:
		        - The computed loss (scalar).
		        - Additional metrics (LossMetrics) produced during loss computation.
		"""
		# Merge the state with the provided tree update.
		module = flax.nnx.merge(state.graphdef, tree, state.graphother)
		# Prepare inputs for the model call.
		call_batch = module.prepare_inputs_for_call(**minibatch)
		labels = call_batch.pop("labels", None)
		outputs, metrics = module.compute_loss(
			labels=labels,
			loss_config=loss_config,
			**call_batch,
		)
		return outputs.loss, metrics

	# Compute gradients and metrics across minibatches.
	gradients, metrics = minibatch_call(
		state=state,
		batch=batch,
		minibatch_size=minibatch_size,
		grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
	)
	# Update state using the computed gradients and updated metrics.
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


def evaluation_step(
	state: EasyDeLState,
	batch: tp.Mapping[str, jax.Array],
	loss_config: tp.Optional[LossConfig] = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
) -> tp.Tuple[tp.Any, LossMetrics]:
	"""
	Performs a single evaluation step by computing loss metrics for the input batch.

	The function determines the required partitioning for the batch, applies sharding constraints,
	and defines an inner loss function. This inner function merges the current state with the graph state,
	sets the model to evaluation mode, and computes loss and metrics via the model's compute_loss method.
	The computed LossMetrics are then returned.

	Args:
	    state (EasyDeLState): The current model state.
	    batch (tp.Mapping[str, jax.Array]): A mapping of input arrays for evaluation.
	    loss_config (tp.Optional[LossConfig], optional): Configuration for loss computation.
	        Defaults to None.
	    partition_spec (tp.Optional[PartitionSpec], optional): Specification for sharding the batch.
	        Defaults to None.

	Returns:
	    tp.Tuple[tp.Any, LossMetrics]:
	        A tuple containing:
	            - (Any): An additional output from loss computation (if any).
	            - LossMetrics: The computed loss metrics for the evaluation batch.
	"""
	# Enforce partitioning constraints and determine required sharding.
	*_, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=1,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	def loss_fn(tree):
		"""
		Computes loss metrics for the evaluation batch given a merged graph state.

		This inner function merges the provided tree with the current state,
		sets the module to evaluation mode, removes the labels from the batch,
		and computes the loss metrics via the module's compute_loss method.

		Args:
		    tree: The current update of the model's graph state.

		Returns:
		    LossMetrics: The computed metrics from the loss function.
		"""
		module = state.merge(tree)
		module.eval()
		labels = batch.pop("labels", None)
		outputs, metrics = module.compute_loss(
			labels=labels,
			loss_config=loss_config,
			**batch,  # Additional inputs passed directly to the model.
		)
		return metrics

	metrics = loss_fn(state.graphstate)
	return metrics
