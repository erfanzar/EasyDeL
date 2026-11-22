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

from ..distillation_trainer.loss_utils import fused_kl_vocab_parallel, resolve_kl_direction
from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_batch_sharding,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    make_scheduled_module_binder,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    sync_module_schedule_config,
    update_metrics,
    update_state_respectfully,
)


def _ejkernel_gjsd_loss(
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    *,
    labels: jax.Array | None,
    mask: jax.Array | None,
    beta: float,
    temperature: float,
    backend: str,
) -> jax.Array:
    """Generalized JSD via the ejkernel fused KL kernel (used when ``fused_backend`` is set).

    The fused kernel streams both softmaxes over the vocabulary without materializing a
    ``[..., V]`` log-softmax and supplies an analytic gradient w.r.t. the student
    (teacher detached), matching :func:`generalized_jsd_loss` numerically.

    Direction mapping (EasyDeL ``beta`` knob -> ejkernel ``direction``), matching the
    GKD paper / TRL convention and the ``beta -> 0/1`` limits of the JSD mixture
    ``m = beta * p_t + (1 - beta) * p_s`` itself:
      * ``beta <= 0`` -> ``"forward"`` == ``KL(p_teacher || p_student)``
      * ``beta >= 1`` -> ``"reverse"`` == ``KL(p_student || p_teacher)``
      * ``0 < beta < 1`` -> ``"jsd"`` (same convex JSD; symmetric at ``beta == 0.5``)

    ejkernel applies the Hinton ``T**2`` factor; ``generalized_jsd_loss`` does not, so the
    result is divided by ``temperature ** 2`` (which also rescales the gradient correctly).
    """
    weights = mask.astype(jnp.float32) if mask is not None else None
    if labels is not None:
        valid_label_mask = (labels != -100).astype(jnp.float32)
        weights = valid_label_mask if weights is None else weights * valid_label_mask

    direction = resolve_kl_direction(beta)

    out = fused_kl_vocab_parallel(
        student_logits,
        teacher_logits,
        weights,
        direction=direction,
        temperature=temperature,
        beta=float(beta),
        platform=(None if backend == "auto" else backend),
    )
    loss = out.loss.astype(jnp.float32)
    if temperature != 1.0:
        loss = loss / jnp.asarray(temperature**2, dtype=jnp.float32)
    return loss.astype(jnp.float32)


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
    ``m = beta * p_t + (1 - beta) * p_s``, the loss is the convex
    combination
    ``beta * KL(p_t || m) + (1 - beta) * KL(p_s || m)``.
    The ``beta`` knob therefore interpolates the GKD objective between
    forward KL ``KL(p_teacher || p_student)`` (``beta == 0``) and
    reverse KL ``KL(p_student || p_teacher)`` (``beta == 1``) — the
    GKD-paper / TRL convention and the rescaled limits of the mixture
    itself; the canonical JSD corresponds to ``beta == 0.5``.

    This is computed by the ejkernel fused KL kernel, which streams both
    softmaxes over the vocabulary without ever materializing a ``[..., V]``
    log-softmax and supplies an analytic gradient w.r.t. the student
    (teacher detached). It is numerically equivalent to the native
    ``log_softmax`` formulation. Per-token contributions are masked by
    ``mask`` (preferred) or ``labels != -100`` and averaged over the
    masked positions.

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
    return _ejkernel_gjsd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        mask=mask,
        beta=beta,
        temperature=temperature,
        backend="xla",  # XLA is the bandwidth-optimal KL backend on TPU (Pallas loses)
    )


def _gkd_resolve_mask(batch: collections.abc.Mapping[str, tp.Any]) -> jax.Array | None:
    """Resolve the GKD token mask: ``completion_mask`` > ``assistant_masks`` > ``attention_mask``."""
    completion_mask = batch.get("completion_mask")
    assistant_masks = batch.get("assistant_masks")
    attention_mask = batch.get("attention_mask")
    if completion_mask is not None:
        mask = completion_mask.astype(jnp.float32)
        if attention_mask is not None:
            mask = mask * attention_mask.astype(jnp.float32)
        return mask
    if assistant_masks is not None:
        mask = assistant_masks.astype(jnp.float32)
        if attention_mask is not None:
            mask = mask * attention_mask.astype(jnp.float32)
        return mask
    if attention_mask is not None:
        return attention_mask.astype(jnp.float32)
    return None


def _gkd_token_mass_from_batch(batch: collections.abc.Mapping[str, tp.Any], *, shift_tokens: bool) -> jax.Array:
    """Return total active token count for a GKD minibatch after optional causal shift."""
    mask = _gkd_resolve_mask(batch)
    if mask is None:
        ids = batch.get("input_ids", batch.get("inputs_embeds"))
        mask = jnp.ones(tuple(ids.shape[:2]), dtype=jnp.float32)
    labels = batch.get("labels")
    if labels is not None:
        mask = mask * (labels != -100).astype(jnp.float32)
    if shift_tokens:
        mask = mask[:, 1:]
    return jnp.sum(mask)


def _gkd_z_loss(
    student_logits: jax.Array,
    *,
    mask: jax.Array | None,
    labels: jax.Array | None,
    z_loss_coef: float,
) -> jax.Array:
    """Compute ``z_loss_coef * mean(logsumexp(logits)**2)`` over valid tokens in float32."""
    weights = mask.astype(jnp.float32) if mask is not None else jnp.ones(student_logits.shape[:2], dtype=jnp.float32)
    if labels is not None:
        weights = weights * (labels != -100).astype(jnp.float32)
    lse = jax.nn.logsumexp(student_logits.astype(jnp.float32), axis=-1)
    return (
        jnp.asarray(z_loss_coef, dtype=jnp.float32)
        * jnp.sum(jnp.square(lse) * weights)
        / jnp.maximum(jnp.sum(weights), jnp.float32(1.0))
    )


def _gkd_forward_outputs(model, batch: collections.abc.Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array | None]:
    """Run ``model`` on ``batch`` and return ``(logits, aux_loss)`` for GKD."""
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
    return logits, getattr(outputs, "aux_loss", None)


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
    logits, _ = _gkd_forward_outputs(model, batch)
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
    shift_tokens = bool(getattr(loss_config, "shift_tokens", True)) if loss_config is not None else True
    z_loss_coef = float(getattr(loss_config, "z_loss", 0.0) or 0.0) if loss_config is not None else 0.0
    scope_root = "easydel/trainer/gkd/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
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

        with jax.named_scope(scope_root + "/teacher_forward"):
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
        with jax.named_scope(scope_root + "/loss_fn"):
            if is_training and straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/merge_state"):
                module = student_state.merge(tree)
            call_kwargs = dict(minibatch)
            labels = call_kwargs.pop("labels", None)
            call_kwargs.pop("completion_mask", None)
            call_kwargs.pop("assistant_masks", None)
            teacher_logits = teacher_forward(minibatch)
            call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), call_kwargs)
            call_kwargs = sanitize_model_call_kwargs(call_kwargs)
            with jax.named_scope(scope_root + "/loss_fn/student_forward"):
                student_outputs = module(**call_kwargs)

            mask = _gkd_resolve_mask(minibatch)

            # Causal shift (LossConfig.shift_tokens default). Position t predicts token t+1, so the
            # JSD compares student/teacher distributions at [:, :-1] and masks by the completion /
            # attention mask[:, 1:] (and labels[:, 1:]) -- matching TRL GKD and ForCausalLMLoss, and
            # the UNSHIFTED -100/completion labels the trainer builds. ``generalized_jsd_loss`` stays
            # a pure (unshifted) reduction; the shift lives here.
            student_logits = student_outputs.logits
            if shift_tokens:
                student_logits = student_logits[:, :-1]
                teacher_logits = teacher_logits[:, :-1]
                if labels is not None:
                    labels = labels[:, 1:]
                if mask is not None:
                    mask = mask[:, 1:]

            with jax.named_scope(scope_root + "/loss_fn/compute_gjsd_loss"):
                jsd_loss = generalized_jsd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mask=mask,
                    beta=beta,
                    temperature=temperature,
                )
            total_loss = jsd_loss.astype(jnp.float32)
            other_metrics: dict[str, jax.Array] = {"gkd_jsd_loss": total_loss}
            z_value = None
            if z_loss_coef:
                z_value = _gkd_z_loss(
                    student_logits,
                    mask=mask,
                    labels=labels,
                    z_loss_coef=z_loss_coef,
                )
                total_loss = total_loss + z_value
                other_metrics["z_loss"] = z_value
            aux_loss = getattr(student_outputs, "aux_loss", None)
            if aux_loss is not None:
                aux_loss = aux_loss.astype(jnp.float32)
                total_loss = total_loss + aux_loss
                other_metrics["aux_loss"] = aux_loss
            metrics = LossMetrics(
                loss=total_loss,
                z_loss=z_value,
                other_metrics=other_metrics,
            )
            return total_loss, metrics

    def _gkd_microbatch_weight(_tree, minibatch) -> jax.Array:
        return _gkd_token_mass_from_batch(minibatch, shift_tokens=shift_tokens)

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=student_state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
                microbatch_weight_fn=_gkd_microbatch_weight if gradient_accumulation_steps > 1 else None,
            )
        with jax.named_scope(scope_root + "/update_state"):
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
    with jax.named_scope(scope_root + "/eval_call"):
        _, metrics = loss_fn(tree=student_state.graphstate, minibatch=batch)
    return metrics


def _prepare_gkd_scheduled_batch(call) -> dict[str, tp.Any]:
    """Return a mutable copy of ``call.batch`` for GKD scheduled MPMD training.

    The frozen teacher forward runs directly inside :func:`_make_gkd_scheduled_loss`
    under ``stop_gradient`` rather than invoking an uncached ``spx.jit`` + host
    ``jax.device_get`` on full ``[B, S, V]`` logits every step.
    """
    return dict(call.batch)


def _gkd_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the GKD scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering ``beta``, ``temperature``, the partition spec,
        and the quantization emulator / teacher state identities.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "temperature", "partition_spec", "loss_config"),
        object_fields=("straight_through_emulator", "teacher_state"),
    )


def _make_gkd_microbatch_weight(call):
    """Build the per-microbatch token-weight callable for GKD scheduled MPMD."""
    _loss_config = call.get("loss_config")
    shift_tokens = bool(getattr(_loss_config, "shift_tokens", True)) if _loss_config is not None else True

    def _weight(_tree, batch):
        return _gkd_token_mass_from_batch(batch, shift_tokens=shift_tokens)

    return _weight


def _make_gkd_scheduled_loss(call):
    """Build a SpectraX-scheduled GKD scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying loss config.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` evaluating the
        generalized Jensen-Shannon divergence (plus optional ``z_loss``
        and ``aux_loss``) against the frozen teacher forward.
    """
    beta = call.get("beta", 0.5)
    temperature = call.get("temperature", 1.0)
    partition_spec = call.get("partition_spec")
    teacher_state = call.get("teacher_state")
    _loss_config = call.get("loss_config")
    shift_tokens = bool(getattr(_loss_config, "shift_tokens", True)) if _loss_config is not None else True
    z_loss_coef = float(getattr(_loss_config, "z_loss", 0.0) or 0.0) if _loss_config is not None else 0.0
    bind_module = make_scheduled_module_binder(call, binder_fn=bind_scheduled_module)
    schedule = call.schedule

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar GKD loss inside the SpectraX scheduled VJP.

        Args:
            tree: Student graphstate to differentiate against.
            batch: Minibatch dict (may optionally include precomputed
                ``teacher_logits``; otherwise runs ``teacher_state`` in-graph).

        Returns:
            The generalized JSD loss scalar.
        """
        with jax.named_scope("easydel/trainer/gkd/scheduled_loss"):
            with jax.named_scope("easydel/trainer/gkd/scheduled_loss/bind_module"):
                module = bind_module(tree)
                model_batch = dict(batch)
                precomputed_teacher_logits = model_batch.pop("teacher_logits", None)
                call_batch = dict(
                    constrain_batch_sharding(
                        model_batch,
                        partition_spec,
                        mesh=getattr(module, "mesh", None),
                        ignore_mpmd=True,
                    )
                )
            labels = call_batch.get("labels")
            with jax.named_scope("easydel/trainer/gkd/scheduled_loss/student_forward"):
                student_logits, aux_loss = _gkd_forward_outputs(module, call_batch)
            with jax.named_scope("easydel/trainer/gkd/scheduled_loss/teacher_forward"):
                if precomputed_teacher_logits is not None:
                    teacher_logits = jax.lax.stop_gradient(jnp.asarray(precomputed_teacher_logits))
                else:
                    if teacher_state is None:
                        raise RuntimeError("GKD scheduled MPMD training requires teacher_state or teacher_logits.")
                    if hasattr(teacher_state, "merge") and hasattr(teacher_state, "graphstate"):
                        teacher_module = teacher_state.merge(jax.lax.stop_gradient(teacher_state.graphstate))
                    else:
                        teacher_module = teacher_state.model
                    sync_module_schedule_config(teacher_module, schedule)
                    teacher_logits = jax.lax.stop_gradient(_gkd_forward_logits(teacher_module, call_batch))
            mask = _gkd_resolve_mask(call_batch)
            # Causal shift (LossConfig.shift_tokens default), matching the eager ``gkd_step``:
            # position t predicts token t+1, so compare logits[:, :-1] and mask by [:, 1:].
            if shift_tokens:
                student_logits = student_logits[:, :-1]
                teacher_logits = teacher_logits[:, :-1]
                if labels is not None:
                    labels = labels[:, 1:]
                if mask is not None:
                    mask = mask[:, 1:]
            with jax.named_scope("easydel/trainer/gkd/scheduled_loss/compute_gjsd_loss"):
                total_loss = generalized_jsd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    mask=mask,
                    beta=beta,
                    temperature=temperature,
                ).astype(jnp.float32)
            if z_loss_coef:
                total_loss = total_loss + _gkd_z_loss(
                    student_logits,
                    mask=mask,
                    labels=labels,
                    z_loss_coef=z_loss_coef,
                )
            if aux_loss is not None:
                total_loss = total_loss + aux_loss.astype(jnp.float32)
            return total_loss

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=gkd_step,
    adapter=ScheduledLossAdapter(
        name="gkd",
        make_loss=_make_gkd_scheduled_loss,
        make_cache_key=_gkd_scheduled_loss_cache_key,
        prepare_batch=_prepare_gkd_scheduled_batch,
        make_microbatch_weight=_make_gkd_microbatch_weight,
    ),
)
