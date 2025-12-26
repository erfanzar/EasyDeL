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

from __future__ import annotations

import typing as tp

import flax
import jax
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully


def _stop_gradient_tree(tree):
    """Detach a pytree by applying stop_gradient to every array leaf.

    Args:
        tree: PyTree to detach.

    Returns:
        PyTree with gradients stopped for all array leaves.
    """

    def _maybe_stop(x):
        if isinstance(x, jax.Array):
            return jax.lax.stop_gradient(x)
        return x

    return jax.tree_util.tree_map(_maybe_stop, tree)


def _kl_div(log_target: jax.Array, log_input: jax.Array) -> jax.Array:
    """Compute KL divergence KL(target || input) given log-probabilities.

    Args:
        log_target: Log probabilities of target distribution.
        log_input: Log probabilities of input distribution.

    Returns:
        Per-token KL divergence.
    """
    target_probs = jnp.exp(log_target)
    return jnp.sum(target_probs * (log_target - log_input), axis=-1)


def generalized_jsd_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    *,
    labels: jax.Array | None = None,
    mask: jax.Array | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> jax.Array:
    """Compute generalized Jensen-Shannon divergence for knowledge distillation.

    Implements the generalized JSD loss from Agarwal et al. (2024). When beta=0,
    reduces to KL(student || teacher); when beta=1, reduces to KL(teacher || student).

    Args:
        student_logits: Student model logits.
        teacher_logits: Teacher model logits.
        labels: Optional labels for masking (-100 positions are ignored).
        mask: Optional explicit mask for valid positions.
        beta: Interpolation factor between student and teacher KL.
        temperature: Temperature for softmax scaling.

    Returns:
        Scalar loss value.
    """
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)

    if beta <= 0.0:
        per_token = _kl_div(student_log_probs, teacher_log_probs)
    elif beta >= 1.0:
        per_token = _kl_div(teacher_log_probs, student_log_probs)
    else:
        beta_val = jnp.asarray(beta, dtype=student_logits.dtype)
        log_beta = jnp.log(beta_val)
        log_one_minus = jnp.log1p(-beta_val)
        mixture_log_probs = jax.scipy.special.logsumexp(
            jnp.stack(
                [
                    teacher_log_probs + log_one_minus,
                    student_log_probs + log_beta,
                ]
            ),
            axis=0,
        )
        kl_teacher = _kl_div(teacher_log_probs, mixture_log_probs)
        kl_student = _kl_div(student_log_probs, mixture_log_probs)
        per_token = beta_val * kl_teacher + (jnp.asarray(1.0, dtype=beta_val.dtype) - beta_val) * kl_student

    if mask is None and labels is not None:
        mask = (labels != -100).astype(student_logits.dtype)
    elif mask is not None:
        mask = mask.astype(student_logits.dtype)

    if mask is None:
        return jnp.mean(per_token)
    normalizer = jnp.maximum(mask.sum(), jnp.array(1.0, dtype=student_logits.dtype))
    return jnp.sum(per_token * mask) / normalizer


def gkd_step(
    student_state: EasyDeLState,
    batch: tp.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn=None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute GKD training or evaluation step.

    Args:
        student_state: Student model state.
        batch: Input batch.
        teacher_state: Teacher model state.
        loss_config: Optional loss configuration.
        learning_rate_fn: Function mapping step to learning rate.
        partition_spec: Sharding specification.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        is_training: Whether this is a training step.
        beta: Interpolation factor for generalized JSD.
        temperature: Temperature for softmax scaling.

    Returns:
        Updated student state and metrics (if training) or just metrics (if eval).
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(student_state.graphdef, tree, student_state.graphother)
        call_kwargs = dict(minibatch)
        labels = call_kwargs.pop("labels", None)
        student_outputs = module(**call_kwargs)
        teacher_outputs = teacher_state.model(**call_kwargs)
        teacher_outputs = _stop_gradient_tree(teacher_outputs)

        completion_mask = minibatch.get("completion_mask")
        attention_mask = minibatch.get("attention_mask")
        mask = completion_mask if completion_mask is not None else attention_mask

        loss_value = generalized_jsd_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_outputs.logits,
            labels=labels,
            mask=mask,
            beta=beta,
            temperature=temperature,
        )
        metrics = LossMetrics(
            loss=loss_value,
            other_metrics={"gkd_jsd_loss": jnp.asarray(loss_value)},
        )
        return loss_value, metrics

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
    _, metrics = loss_fn(tree=student_state.graphstate, minibatch=batch)
    return metrics
