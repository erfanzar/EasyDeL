import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.utils.helpers import check_bool_flag

SCAN_TRAINER = check_bool_flag("SCAN_TRAINER")
FAST_COMPILE = check_bool_flag("FAST_COMPILE")


def make_assertions_and_get_sizes(
	batch: tp.Dict,
	gradient_accumulation_steps: int,
	batch_partition_spec: tp.Optional[PartitionSpec] = None,
) -> tp.Tuple[int, int, PartitionSpec]:
	"""
	Validates the input parameters and computes the batch size, minibatch size, and batch partition specification.
	Args:
		batch (tp.Dict): A dictionary containing the batch data. The batch size is inferred from the first element's shape.
		gradient_accumulation_steps (int): The number of gradient accumulation steps. Must be greater than 0.
		batch_partition_spec (tp.Optional[PartitionSpec], optional): The partition specification for the batch. Defaults to None.
	Returns:
		tp.Tuple[int, int, PartitionSpec]: A tuple containing:
			- batch_size (int): The size of the batch.
			- minibatch_size (int): The size of the minibatch.
			- batch_partition_spec (PartitionSpec): The partition specification for the batch.
	Raises:
		ValueError: If `gradient_accumulation_steps` is not greater than 0.
		ValueError: If the batch size is not divisible by the gradient accumulation steps.
	"""

	batch_size = batch[list(batch.keys())[0]].shape[0]
	minibatch_size = batch_size // gradient_accumulation_steps
	if not gradient_accumulation_steps > 0:
		ValueError("`gradient_accumulation_steps` must be greater than 0.")
	if minibatch_size * gradient_accumulation_steps != batch_size:
		raise ValueError("Batch size must be divisible by gradient accumulation steps.")
	if batch_partition_spec is None:
		batch_partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
	return batch_size, minibatch_size, batch_partition_spec


def update_metrics(
	metrics: LossMetrics,
	learning_rate_fn: tp.Callable,
	step: int | jax.Array,
	gradients: tp.Optional[jax.Array],
) -> LossMetrics:
	"""
	Updates the given metrics with the current learning rate and gradient norms.

	Args:
		metrics (LossMetrics): An instance of LossMetrics to be updated.
		learning_rate_fn (tp.Callable): A callable that returns the learning rate given the current step.
		step (int | jax.Array): The current training step.
		gradients (Optional(jax.Array)): The gradients to compute norms from.

	Returns:
		LossMetrics: The updated metrics with learning rate and gradient norms.
	"""
	if learning_rate_fn is not None:
		metrics.learning_rate = learning_rate_fn(step)
	if gradients is not None:
		grad_norms = tu.tree_map(jnp.linalg.norm, gradients)
		metrics.max_grad_norm = tu.tree_reduce(jnp.maximum, grad_norms)
		grad_size = tu.tree_reduce(jnp.add, tu.tree_map(jnp.size, grad_norms))
		grad_sum = tu.tree_reduce(jnp.add, tu.tree_map(jnp.sum, grad_norms))
		metrics.mean_grad_norm = grad_sum / grad_size
		metrics.grad_norms = grad_norms
	return metrics


def update_state_respectfully(
	state: EasyDeLState,
	gradients: jax.Array,
	loss_config: LossConfig,
	metrics: LossMetrics,
) -> EasyDeLState:
	"""
	Updates the state of the model respectfully based on the provided gradients, loss configuration, and metrics.

	Args:
		state (EasyDeLState): The current state of the model.
		gradients (jax.Array): The gradients to be applied to the model's parameters.
		loss_config (LossConfig): Configuration for the loss, including conditions for breaking on NaN values.
		metrics (LossMetrics): Metrics containing the loss value.

	Returns:
		EasyDeLState: The updated state of the model.
	"""
	if FAST_COMPILE:
		return state.apply_gradients(grads=gradients)
	else:

		def update_fn(args):
			state, gradients = args
			return state.apply_gradients(grads=gradients)

		def skip_fn(args):
			state, _ = args
			return state

		should_update = True
		if loss_config is not None:
			should_update = lax.cond(
				loss_config.break_on_nan,
				lambda x: lax.cond(
					jnp.isnan(metrics.loss),
					lambda _: False,
					lambda _: True,
					None,
				),
				lambda x: True,
				None,
			)
		state = lax.cond(should_update, update_fn, skip_fn, (state, gradients))
		return state


def minibatch_call(
	state: EasyDeLState,
	batch: tp.Dict,
	minibatch_size: int,
	grad_fn: tp.Callable[[jax.Array, tp.Dict], tp.Tuple[jax.Array, LossMetrics]],
) -> tp.Tuple[jax.Array, LossMetrics]:
	"""
	Processes batch in smaller chunks for gradient accumulation using jax.lax.scan.
	Uses eval_shape to initialize accumulator structures efficiently.
	"""
	num_accum_steps = len(next(iter(batch.values()))) // minibatch_size
	if num_accum_steps > 1:

		def reshape_to_minibatches(arr):
			"""Reshape the batch into minibatches for accumulation."""
			batch_shape = (num_accum_steps, minibatch_size) + arr.shape[1:]
			return jnp.reshape(arr, batch_shape)

		batch = jax.tree_util.tree_map(reshape_to_minibatches, batch)

		(_, metrics_shape), grads_shape = jax.eval_shape(
			lambda: grad_fn(
				state.graphstate,
				jax.tree_util.tree_map(lambda x: x[0], batch),
			)
		)

		init_acc = {
			"grads": jax.tree_util.tree_map(
				lambda x: jnp.zeros(x.shape, x.dtype), grads_shape
			),
			"metrics": jax.tree_util.tree_map(
				lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape
			),
		}

		def accumulate_gradients(acc, minibatch):
			"""Accumulate gradients and metrics for each minibatch."""
			(_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
			new_acc = {
				"grads": jax.tree_util.tree_map(jnp.add, acc["grads"], step_grads),
				"metrics": jax.tree_util.tree_map(jnp.add, acc["metrics"], step_aux),
			}
			return new_acc, step_aux

		final_acc, aux = jax.lax.scan(
			accumulate_gradients,
			init_acc,
			batch,
			length=num_accum_steps,
		)
		gradients = jax.tree_util.tree_map(
			lambda x: x / num_accum_steps, final_acc["grads"]
		)
		metrics = jax.tree_util.tree_map(
			lambda x: x / num_accum_steps, final_acc["metrics"]
		)

	else:
		(_, metrics), gradients = grad_fn(state.graphstate, batch)

	return gradients, metrics
