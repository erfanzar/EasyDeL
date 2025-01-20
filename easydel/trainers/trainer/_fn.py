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
from jax.sharding import PartitionSpec

from easydel.escale import with_sharding_constraint
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
	batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=gradient_accumulation_steps,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	def loss_fn(tree, minibatch):
		module = flax.nnx.merge(state.graphdef, tree, state.graphother)
		call_batch = module.prepare_inputs_for_call(**minibatch)
		labels = call_batch.pop("labels", None)
		outputs, metrics = module.compute_loss(
			labels=labels,
			loss_config=loss_config,
			**call_batch,
		)
		return outputs.loss, metrics

	gradients, metrics = minibatch_call(
		state=state,
		batch=batch,
		minibatch_size=minibatch_size,
		grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
	)
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
	*_, partition_spec = make_assertions_and_get_sizes(
		batch=batch,
		gradient_accumulation_steps=1,
		batch_partition_spec=partition_spec,
	)
	batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

	def loss_fn(tree):
		module = state.merge(tree)
		module.eval()
		labels = batch.pop("labels", None)
		outputs, metrics = module.compute_loss(
			labels=labels,
			loss_config=loss_config,
			**batch,
			# Passed directly to Model
		)

		return metrics

	metrics = loss_fn(state.graphstate)

	return metrics
