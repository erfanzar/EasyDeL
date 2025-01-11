import os
import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tu
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

SCAN_TRAINER = os.environ.get("SCAN_TRAINER", "true").lower() in [
	"true",
	"1",
	"on",
	"yes",
]


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
	Processes a batch of data in smaller minibatches and accumulates gradients and metrics.

	Args:
		state (EasyDeLState): The current state of the model.
		batch (tp.Dict): The batch of data to be processed.
		minibatch_size (int): The size of each minibatch.
		grad_fn (tp.Callable[[jax.Array, tp.Dict], tp.Tuple[jax.Array, LossMetrics]]):
			A function that computes the gradients and metrics for a given minibatch.

	Returns:
		tp.Tuple[jax.Array, LossMetrics]: The accumulated gradients and metrics over all minibatches.
	"""

	def _minibatch_step(minibatch_idx: jax.Array | int):
		minibatch = jax.tree_map(
			lambda x: jax.lax.dynamic_slice_in_dim(
				x,
				start_index=minibatch_idx * minibatch_size,
				slice_size=minibatch_size,
				axis=0,
			),
			batch,
		)
		(_, step_aux), step_grads = grad_fn(state.graphstate, minibatch)
		return step_grads, step_aux

	def _scan_step(carry, minibatch_idx: jax.Array | int):
		step_grads, step_aux = _minibatch_step(minibatch_idx)
		carry = jax.tree_map(jnp.add, carry, (step_grads, step_aux))
		return carry, None

	grads_shapes, aux_shape = jax.eval_shape(_minibatch_step, 0)
	gradients = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
	aux = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), aux_shape)

	if SCAN_TRAINER and minibatch_size != 1:
		(gradients, aux), _ = jax.lax.scan(
			_scan_step,
			init=(gradients, aux),
			xs=jnp.arange(minibatch_size),
			length=minibatch_size,
		)
		gradients = jax.tree_util.tree_map(lambda g: g / minibatch_size, gradients)
		aux = jax.tree_util.tree_map(lambda m: m / minibatch_size, aux)
	else:
		for minibatch_idx in range(minibatch_size):
			(gradients, aux), _ = _scan_step((gradients, aux), minibatch_idx)
	return gradients, aux
