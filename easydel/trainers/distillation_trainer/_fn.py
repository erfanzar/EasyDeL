# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import chex
import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully


def distillation_loss(
    student_logits: chex.Array,
    teacher_logits: chex.Array,
    attention_mask: chex.Array | None = None,
    labels: chex.Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
):
    """
    Distillation loss that properly handles padding tokens using attention mask.
    """
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    kl_loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)

    if attention_mask is not None:
        kl_loss = kl_loss * attention_mask
        num_active_tokens = jnp.sum(attention_mask)
        kl_loss = jnp.sum(kl_loss) / jnp.maximum(num_active_tokens, 1.0)
    else:
        kl_loss = jnp.mean(kl_loss)
    kl_loss = kl_loss * (temperature**2)
    total_loss = alpha * kl_loss
    if use_hard_labels and labels is not None:
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(student_logits, labels)

        if attention_mask is not None:
            ce_loss = ce_loss * attention_mask
            ce_loss = jnp.sum(ce_loss) / jnp.maximum(num_active_tokens, 1.0)
        else:
            ce_loss = jnp.mean(ce_loss)

        total_loss += (1 - alpha) * ce_loss

    return total_loss


def distillation_step(
    student_state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> tuple[EasyDeLState, LossMetrics]:
    batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(student_state.graphdef, tree, student_state.graphother)
        student_outputs = module(**minibatch)
        teacher_outputs = teacher_state.model(**minibatch)
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)
        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
            labels=labels,
            use_hard_labels=(labels is not None),
            temperature=temperature,
            alpha=alpha,
        )
        return loss, LossMetrics(loss=loss)

    # Compute gradients and metrics across minibatches.
    if is_training:
        gradients, metrics = minibatch_call(
            state=student_state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        student_state = update_state_respectfully(
            state=student_state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=student_state.step,
                gradients=gradients,
            ),
        )
        return student_state, metrics
    else:
        _, metrics = loss_fn(tree=student_state.graphstate, minibatch=batch)
        return metrics
