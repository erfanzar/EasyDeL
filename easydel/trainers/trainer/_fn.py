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
		batch = with_sharding_constraint(batch, partition_spec)

		def loss_fn(tree):
			module = state.merge(tree)
			module.train()
			call_batch = module.prepare_inputs_for_call(**batch)
			outputs, metrics = module.compute_loss(
				labels=call_batch.pop("labels", None),
				loss_config=loss_config,
				**call_batch,  # Passed directly to Model
			)

			return outputs.loss, metrics

		(
			(loss, metrics),
			grads,
		) = jax.value_and_grad(loss_fn, has_aux=True)(state.graphstate)
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

		return state, loss, metrics

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
