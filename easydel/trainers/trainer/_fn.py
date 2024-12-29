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

import jax
import optax
from fjformer.sharding import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.etils.easystate import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics


def create_training_step(
	loss_config: LossConfig = None,
	learning_rate_fn: optax.Schedule = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
	gradient_accumulation_steps: int = 1,
):
	if partition_spec is None:
		partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
	assert (
		gradient_accumulation_steps > 0
	), "`gradient_accumulation_steps` must be greater than 0."

	def train_step(
		state: EasyDeLState,
		batch: tp.Mapping[str, jax.Array],
	) -> tp.Tuple[EasyDeLState, jax.Array, LossMetrics]:
		batch_size = batch[list(batch.keys())[0]].shape[0]
		minibatch_size = batch_size // gradient_accumulation_steps

		assert minibatch_size * gradient_accumulation_steps == batch_size
		batch = with_sharding_constraint(batch, partition_spec)

		def loss_fn(tree, minibatch):
			module = state.merge(tree)
			module.train()
			call_batch = module.prepare_inputs_for_call(**minibatch)
			outputs, metrics = module.compute_loss(
				labels=call_batch.pop("labels", None),
				loss_config=loss_config,
				**call_batch,  # Passed directly to Model
			)

			return outputs.loss, metrics

		grad_fn = jax.value_and_grad(loss_fn, has_aux=True, allow_int=True)

		def _minibatch_step(minibatch_idx: jax.Array | int):
			minibatch = jax.tree_map(
				lambda x: jax.lax.dynamic_slice_in_dim(  # Slicing with variable index (jax.Array).
					x,
					start_index=minibatch_idx * minibatch_size,
					slice_size=minibatch_size,
					axis=0,
				),
				batch,
			)
			(_, step_metrics), step_grads = grad_fn(state.graphstate, minibatch)

			return step_grads, step_metrics

		def _scan_step(carry, minibatch_idx: jax.Array | int):
			"""Scan step function for looping over minibatches."""
			step_grads, step_metrics = _minibatch_step(minibatch_idx)

			carry = jax.tree_map(jnp.add, carry, (step_grads, step_metrics))

			return carry, None

		grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
		grads = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
		metrics = jax.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
		(grads, metrics), _ = jax.lax.scan(
			_scan_step,
			init=(grads, metrics),
			xs=jnp.arange(minibatch_size),
			length=minibatch_size,
		)
		# Average the accumulated gradients

		grads = jax.tree_map(lambda g: g / minibatch_size, grads)
		metrics = jax.tree_map(lambda m: m / minibatch_size, metrics)

		state = state.apply_gradients(grads=grads)

		if learning_rate_fn is not None:
			metrics.learning_rate = learning_rate_fn(state.step)

		grad_norms = jax.tree_util.tree_map(jnp.linalg.norm, grads)
		max_grad_norm = jax.tree_util.tree_reduce(jnp.maximum, grad_norms)

		mean_grad_norm = jax.tree_util.tree_reduce(
			jnp.add,
			jax.tree_util.tree_map(jnp.sum, grad_norms),
		) / jax.tree_util.tree_reduce(
			jnp.add,
			jax.tree_util.tree_map(jnp.size, grad_norms),
		)
		metrics.max_grad_norm = max_grad_norm
		metrics.mean_grad_norm = mean_grad_norm
		metrics.grad_norms = grad_norms

		return state, metrics.loss, metrics

	return train_step


def create_evaluation_step(
	loss_config: LossConfig = None,
	partition_spec: tp.Optional[PartitionSpec] = None,
):
	if partition_spec is None:
		partition_spec = PartitionSpec(("dp", "fsdp"), "sp")

	def train_step(
		state: EasyDeLState,
		batch: tp.Mapping[str, jax.Array],
	) -> tp.Tuple[tp.Any, LossMetrics]:
		batch = with_sharding_constraint(batch, partition_spec)

		def loss_fn(tree):
			module = state.merge(tree)
			module.eval()
			outputs, metrics = module.compute_loss(
				labels=batch.pop("labels", None),
				loss_config=loss_config,
				**batch,  # Passed directly to Model
			)

			return metrics

		metrics = loss_fn(state.graphstate)

		return metrics

	return train_step
