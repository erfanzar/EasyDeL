# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Loss and step implementations for the GKD trainer.

Implements the generalised Jensen-Shannon divergence (GJSD) loss that
interpolates between forward and reverse KL by ``beta``, mixed with a
supervised cross-entropy term, and the scheduled-VJP variants for
MPMD pipeline parallelism.
"""

from __future__ import annotations

import collections.abc
import functools
import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    cached_scheduled_auxiliary,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    stop_gradient_tree,
    sync_module_schedule_config,
    update_metrics,
    update_state_respectfully,
)


def _stop_gradient_tree(tree):
    """Detach a pytree by applying stop_gradient to every array leaf.

    Args:
        tree: PyTree to detach.

    Returns:
        PyTree with gradients stopped for all array leaves.
    """

    def _maybe_stop(x):
        """Apply ``jax.lax.stop_gradient`` to JAX-array leaves only.

        Args:
            x: Pytree leaf (JAX array or Python value).

        Returns:
            ``x`` with ``stop_gradient`` applied if it is a JAX array,
            otherwise unchanged.
        """
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
    """Compute the generalized Jensen-Shannon divergence used by GKD.

    From Agarwal et al. 2024. Given the temperature-softened
    distributions ``p_s = softmax(student / T)`` and
    ``p_t = softmax(teacher / T)`` and a midpoint
    ``m = beta * p_s + (1 - beta) * p_t``, the loss is the convex
    combination
    ``beta * KL(p_t || m) + (1 - beta) * KL(p_s || m)``.
    The ``beta`` knob therefore interpolates the GKD objective between
    forward KL (``beta == 0``: ``KL(p_s || p_t)``) and reverse KL
    (``beta == 1``: ``KL(p_t || p_s)``); the canonical JSD corresponds
    to ``beta == 0.5``. The midpoint ``m`` is computed in log-space via
    ``logsumexp`` for numerical stability.

    Per-token contributions are masked by ``mask`` (preferred) or
    ``labels != -100`` and averaged over the masked positions.

    Args:
        student_logits: ``[batch, seq, vocab]`` raw student logits.
        teacher_logits: ``[batch, seq, vocab]`` raw teacher logits
            (caller should stop-gradient).
        labels: Optional ``[batch, seq]`` int labels; positions equal
            to ``-100`` are excluded.
        mask: Optional explicit ``[batch, seq]`` valid-position mask;
            takes priority over ``labels`` when both are provided.
        beta: Interpolation factor in ``[0, 1]``.
        temperature: Softmax temperature ``T``. Larger values produce
            softer distributions.

    Returns:
        Scalar mean per-token GJSD across the masked positions.
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


def _gkd_forward_logits(model, batch: collections.abc.Mapping[str, jax.Array]) -> jax.Array:
    """Run ``model`` on ``batch`` and return token logits for GKD.

    Drops loss-only fields (``labels``, ``completion_mask``,
    ``assistant_masks``, ``teacher_logits``) so they don't reach the
    model forward.

    Args:
        model: Student or teacher module.
        batch: Input batch dictionary.

    Returns:
        ``[batch, seq_len, vocab_size]`` logits.

    Raises:
        TypeError: If the model does not return logits.
    """
    call_kwargs = dict(batch)
    call_kwargs.pop("labels", None)
    call_kwargs.pop("completion_mask", None)
    call_kwargs.pop("assistant_masks", None)
    call_kwargs.pop("teacher_logits", None)
    call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
    call_kwargs = sanitize_model_call_kwargs(call_kwargs)
    outputs = model(**call_kwargs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits for GKD.")
    return logits


def gkd_step(
    student_state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn=None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    beta: float = 0.5,
    temperature: float = 1.0,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one GKD training or evaluation step.

    Forwards the batch through both student and teacher, computes the
    generalized Jensen-Shannon divergence (see
    :func:`generalized_jsd_loss`), and -- when ``is_training`` is
    ``True`` -- accumulates gradients via ``minibatch_call`` and
    applies an optimizer update. The teacher forward is wrapped in
    ``jax.checkpoint`` with ``nothing_saveable`` and stop-gradient'd
    so its activations are not retained for backprop and so its
    parameters can never receive gradient signal.

    On-policy and ``seq_kd`` batch swaps are performed by the trainer
    *before* this step is invoked; this function consumes whatever
    ``batch`` it receives.

    Args:
        student_state: Student ``EasyDeLState`` being differentiated.
        batch: Input batch with ``input_ids`` / ``attention_mask`` and
            optional ``labels``, ``completion_mask``, ``assistant_masks``.
        teacher_state: Frozen teacher ``EasyDeLState``.
        loss_config: ``LossConfig`` controlling NaN handling.
        learning_rate_fn: Schedule mapping step to learning rate.
        partition_spec: Sharding spec applied to the input batch.
        gradient_accumulation_steps: Gradient-accumulation factor.
        is_training: When ``False`` skips gradient computation and
            returns only ``LossMetrics``.
        beta: GJSD interpolation knob in ``[0, 1]``.
        temperature: GJSD softmax temperature.
        straight_through_emulator: Optional STE callable applied to
            the student graphstate before the forward pass (QAT path).

    Returns:
        ``(new_state, metrics)`` when ``is_training`` is ``True``;
        otherwise ``LossMetrics``. ``other_metrics["gkd_jsd_loss"]``
        records the raw scalar GJSD value.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=student_state.model.mesh, ignore_mpmd=True)

    def teacher_forward(minibatch: collections.abc.Mapping[str, jax.Array]) -> jax.Array:
        """Run the teacher in stop-gradient mode and return its logits.

        Args:
            minibatch: Input minibatch.

        Returns:
            ``[batch, seq_len, vocab_size]`` teacher logits with
            gradients detached.
        """
        teacher_call_kwargs = dict(minibatch)
        teacher_call_kwargs.pop("labels", None)
        teacher_call_kwargs.pop("completion_mask", None)
        teacher_call_kwargs.pop("assistant_masks", None)
        teacher_call_kwargs = filter_kwargs_for_callable(
            getattr(teacher_state.model, "forward", teacher_state.model), teacher_call_kwargs
        )
        teacher_call_kwargs = sanitize_model_call_kwargs(teacher_call_kwargs)
        teacher_static_kwargs = {
            key: teacher_call_kwargs.pop(key)
            for key in list(teacher_call_kwargs)
            if not hasattr(teacher_call_kwargs[key], "shape")
        }

        @functools.partial(
            jax.checkpoint,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        def _teacher_fwd(kw, t_graphstate):
            """Re-materializable teacher forward used inside ``teacher_forward``.

            Args:
                kw: Dynamic kwargs for the teacher module call.
                t_graphstate: Stop-gradient teacher graphstate.

            Returns:
                Stop-gradient teacher logits.
            """
            teacher_module = teacher_state.merge(t_graphstate)
            teacher_outputs = teacher_module(**kw, **teacher_static_kwargs)
            return jax.lax.stop_gradient(teacher_outputs.logits)

        return _teacher_fwd(
            teacher_call_kwargs,
            jax.lax.stop_gradient(teacher_state.graphstate),
        )

    def loss_fn(tree, minibatch):
        """Compute the GKD loss for one minibatch.

        Runs the student forward, the teacher forward (in
        ``stop_gradient`` mode), and then evaluates the generalized
        Jensen-Shannon divergence with optional masking from
        ``completion_mask`` / ``attention_mask``.

        Args:
            tree: Student graphstate to differentiate against.
            minibatch: One minibatch dict.

        Returns:
            ``(loss, metrics)`` where ``metrics`` records the GJSD
            value under ``other_metrics["gkd_jsd_loss"]``.
        """
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = student_state.merge(tree)
        call_kwargs = dict(minibatch)
        labels = call_kwargs.pop("labels", None)
        call_kwargs.pop("completion_mask", None)
        call_kwargs.pop("assistant_masks", None)
        teacher_logits = teacher_forward(minibatch)
        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        student_outputs = module(**call_kwargs)

        completion_mask = minibatch.get("completion_mask")
        attention_mask = minibatch.get("attention_mask")
        mask = completion_mask if completion_mask is not None else attention_mask

        loss_value = generalized_jsd_loss(
            student_logits=student_outputs.logits,
            teacher_logits=teacher_logits,
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


def _prepare_gkd_scheduled_batch(call) -> dict[str, tp.Any]:
    """Inject precomputed teacher logits into ``call.batch`` for GKD.

    Args:
        call: The :class:`ScheduledStepCall` being prepared.

    Returns:
        A copy of ``call.batch`` with ``teacher_logits`` populated.

    Raises:
        RuntimeError: If no teacher state is available.
    """
    batch = dict(call.batch)
    if "teacher_logits" in batch:
        return batch

    teacher_state = call.get("teacher_state")
    if teacher_state is None:
        raise RuntimeError("GKD scheduled MPMD training requires teacher_state.")

    teacher_model = teacher_state.model
    teacher_model.eval()
    sync_module_schedule_config(teacher_model, call.schedule)
    constrained_batch = constrain_scheduled_batch(teacher_model, batch, call.get("partition_spec"))
    teacher_forward = cached_scheduled_auxiliary(_gkd_forward_logits, teacher_model.mesh)
    batch["teacher_logits"] = stop_gradient_tree(teacher_forward(teacher_model, constrained_batch))
    return batch


def _gkd_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the GKD scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering ``beta``, ``temperature``, the partition spec,
        and the quantization emulator identity.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "temperature", "partition_spec"),
        object_fields=("straight_through_emulator",),
    )


def _make_gkd_scheduled_loss(call):
    """Build a SpectraX-scheduled GKD scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` evaluating the
        generalized Jensen-Shannon divergence against precomputed
        teacher logits.
    """
    beta = call.get("beta", 0.5)
    temperature = call.get("temperature", 1.0)
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar GKD loss inside the SpectraX scheduled VJP.

        Args:
            tree: Student graphstate to differentiate against.
            batch: Minibatch dict with precomputed ``teacher_logits``.

        Returns:
            The generalized JSD loss scalar.
        """
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        labels = call_batch.get("labels")
        teacher_logits = jax.lax.stop_gradient(call_batch["teacher_logits"])
        student_logits = _gkd_forward_logits(module, call_batch)
        completion_mask = call_batch.get("completion_mask")
        attention_mask = call_batch.get("attention_mask")
        mask = completion_mask if completion_mask is not None else attention_mask
        return generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            mask=mask,
            beta=beta,
            temperature=temperature,
        )

    return scheduled_loss


register_scheduled_loss_adapter(
    gkd_step,
    ScheduledLossAdapter(
        name="gkd",
        make_loss=_make_gkd_scheduled_loss,
        make_cache_key=_gkd_scheduled_loss_cache_key,
        prepare_batch=_prepare_gkd_scheduled_batch,
    ),
)
