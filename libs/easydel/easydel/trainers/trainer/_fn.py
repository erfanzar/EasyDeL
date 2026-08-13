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

"""Core training and evaluation step functions.

This module provides the fundamental training and evaluation step implementations
used by the Trainer class. These functions handle:

- Training step: Gradient computation, model updates, and metrics tracking
- Evaluation step: Loss computation and metrics collection without updates
- Minibatch processing for gradient accumulation
- Distributed training with sharding constraints

The functions are designed to be JIT-compiled for optimal performance
and support various model architectures through the EasyDeLState abstraction.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import ForCausalLMLoss, LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_batch_sharding,
    constrain_scheduled_batch,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)


def _dft_causal_lm_metrics(logits: jax.Array, labels: jax.Array, ignore_index: int = -100) -> LossMetrics:
    """Compute the DFT SFT loss from logits and causal-LM labels."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    loss_mask = shift_labels != ignore_index
    safe_labels = jnp.where(loss_mask, shift_labels, 0)
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    token_logps = jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
    per_token_loss = -jax.lax.stop_gradient(jnp.exp(token_logps)) * token_logps
    weights = loss_mask.astype(logits.dtype)
    weight_sum = jnp.sum(weights)
    loss = jnp.sum(per_token_loss * weights) / jnp.maximum(weight_sum, 1.0)
    predictions = jnp.argmax(shift_logits, axis=-1)
    accuracy = jnp.sum((predictions == safe_labels).astype(logits.dtype) * weights) / jnp.maximum(weight_sum, 1.0)
    return LossMetrics(loss=loss, weight_sum=weight_sum, accuracy=accuracy)


def base_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    loss_type: str = "nll",
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run the shared base trainer loss path for train or eval.

    Args:
        state (EasyDeLState): Current parameter / optimizer state.
        batch (collections.abc.Mapping[str, jax.Array]): Input batch.
        loss_config (LossConfig | None): Optional loss configuration
            forwarded to ``module.compute_loss``.
        learning_rate_fn (optax.Schedule): Learning-rate schedule used
            for metric reporting.
        partition_spec (PartitionSpec | None): Sharding spec applied to
            the input batch.
        gradient_accumulation_steps (int): Number of microbatches
            whose gradients are accumulated before an update.
        is_training (bool): When True, compute gradients and apply an
            optimizer update; otherwise only compute metrics.
        straight_through_emulator (tp.Callable | None): Optional STE
            wrapping the parameter tree to emulate quantized forward
            passes.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: ``(state,
        metrics)`` in training mode; just ``metrics`` in evaluation
        mode.
    """
    scope_root = "easydel/trainer/base/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = constrain_batch_sharding(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute the base-trainer scalar loss and metrics for one microbatch.

        Steps performed inside the closure:

        1. Optionally pass ``tree`` through ``straight_through_emulator``
           so the forward pass runs under simulated quantisation while
           the gradient path stays differentiable.
        2. Rebind ``tree`` against the trainer state's graph definition
           via ``state.merge`` to recover a live module.
        3. When evaluating, switch the module to eval mode so dropout
           and other train-only paths short-circuit.
        4. Run ``module.prepare_inputs_for_call(**minibatch)`` so the
           module-specific input filter projects the raw batch onto the
           callable's signature, then pop ``"labels"`` (None when
           absent) and feed the rest to ``module.compute_loss``.

        ``compute_loss`` returns ``(outputs, metrics)`` where ``outputs``
        carries the scalar loss the trainer differentiates against; the
        metrics object is forwarded verbatim to
        :func:`update_state_respectfully` for logging.

        Args:
            tree: Differentiable parameter tree.
            minibatch: Mapping with the model-specific input fields and
                an optional ``labels`` entry.

        Returns:
            ``(loss, metrics)`` where ``loss`` is the scalar trainable
            objective and ``metrics`` is the :class:`LossMetrics`
            instance returned by ``module.compute_loss``.
        """
        with jax.named_scope(scope_root + "/loss_fn"):
            if is_training and straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/merge_state"):
                module = state.merge(tree)
                if not is_training:
                    module.eval()
            with jax.named_scope(scope_root + "/loss_fn/prepare_inputs"):
                call_batch = module.prepare_inputs_for_call(**minibatch)
                labels = call_batch.pop("labels", None)
            with jax.named_scope(scope_root + "/loss_fn/forward_and_loss"):
                if loss_type == "dft":
                    outputs = module(**call_batch)
                    metrics = _dft_causal_lm_metrics(
                        outputs.logits,
                        labels,
                        ignore_index=-100 if loss_config is None else loss_config.ignore_index,
                    )
                    outputs = outputs.replace(loss=metrics.loss)
                else:
                    outputs, metrics = module.compute_loss(
                        labels=labels,
                        loss_config=loss_config,
                        **call_batch,
                    )
            return outputs.loss, metrics

    if not is_training:
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(state.graphstate, batch)
        return metrics

    with jax.named_scope(scope_root + "/grad_and_minibatch"):
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
    with jax.named_scope(scope_root + "/update_state"):
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


def training_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    loss_type: str = "nll",
) -> tuple[EasyDeLState, LossMetrics]:
    """Perform one base trainer update step.

    Convenience wrapper over :func:`base_step` with ``is_training=True``.

    Args:
        state (EasyDeLState): Current state.
        batch (collections.abc.Mapping[str, jax.Array]): Input batch.
        loss_config (LossConfig | None): Optional loss configuration.
        learning_rate_fn (optax.Schedule): Learning-rate schedule for
            metric reporting.
        partition_spec (PartitionSpec | None): Sharding spec.
        gradient_accumulation_steps (int): Number of accumulation
            microbatches.
        straight_through_emulator (tp.Callable | None): Optional STE.

    Returns:
        tuple[EasyDeLState, LossMetrics]: Updated state and metrics.
    """
    return tp.cast(
        tuple[EasyDeLState, LossMetrics],
        base_step(
            state=state,
            batch=batch,
            loss_config=loss_config,
            learning_rate_fn=learning_rate_fn,
            partition_spec=partition_spec,
            gradient_accumulation_steps=gradient_accumulation_steps,
            is_training=True,
            straight_through_emulator=straight_through_emulator,
            loss_type=loss_type,
        ),
    )


def evaluation_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    loss_type: str = "nll",
) -> LossMetrics:
    """Perform one base trainer evaluation step.

    Convenience wrapper over :func:`base_step` with ``is_training=False``
    and a fixed accumulation factor of 1.

    Args:
        state (EasyDeLState): Current state.
        batch (collections.abc.Mapping[str, jax.Array]): Input batch.
        loss_config (LossConfig | None): Optional loss configuration.
        partition_spec (PartitionSpec | None): Sharding spec.

    Returns:
        LossMetrics: Metrics for the evaluated batch.
    """
    return tp.cast(
        LossMetrics,
        base_step(
            state=state,
            batch=batch,
            loss_config=loss_config,
            partition_spec=partition_spec,
            gradient_accumulation_steps=1,
            is_training=False,
            loss_type=loss_type,
        ),
    )


def _base_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the cache key for a scheduled base-trainer loss specialization.

    Args:
        call: Scheduled call descriptor with bound static arguments.

    Returns:
        tuple[tp.Any, ...]: Hashable identifier for the
        ``(partition_spec, loss_config, straight_through_emulator)``
        specialization.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("partition_spec", "loss_type"),
        object_fields=("loss_config", "straight_through_emulator"),
    )


def _make_base_scheduled_loss(call):
    """Build a scalar loss closure for the base-trainer scheduled-loss adapter.

    Args:
        call: Scheduled call descriptor providing ``loss_config`` and
            ``partition_spec``.

    Returns:
        tp.Callable: A function ``(tree, batch) -> Array`` that runs
        ``module.compute_loss(...)`` and returns its scalar loss.
    """
    loss_config = call.get("loss_config")
    partition_spec = call.get("partition_spec")
    loss_type = call.get("loss_type", "nll")

    def scheduled_loss(tree, batch):
        """Compute the base-trainer scheduled loss and auxiliary metrics.

        Args:
            tree: Current model parameter tree.
            batch (collections.abc.Mapping[str, jax.Array]): Input
                minibatch (with optional ``labels`` field).

        Returns:
            tuple: ``(loss, aux)`` where ``loss`` is the scalar loss from
            ``module.compute_loss`` and ``aux`` maps metric names
            (``z_loss``, ``accuracy``) to terminal-stage scalars.
        """
        with jax.named_scope("easydel/trainer/base/scheduled_loss"):
            with jax.named_scope("easydel/trainer/base/scheduled_loss/bind_module"):
                module = bind_scheduled_module(call, tree)
                batch = constrain_scheduled_batch(module, batch, partition_spec)
            with jax.named_scope("easydel/trainer/base/scheduled_loss/prepare_inputs"):
                call_batch = module.prepare_inputs_for_call(**batch)
                labels = call_batch.pop("labels", None)
            with jax.named_scope("easydel/trainer/base/scheduled_loss/forward_and_loss"):
                if loss_type == "dft":
                    outputs = module(**call_batch)
                    metrics = _dft_causal_lm_metrics(
                        outputs.logits,
                        labels,
                        ignore_index=-100 if loss_config is None else loss_config.ignore_index,
                    )
                    outputs = outputs.replace(loss=metrics.loss)
                else:
                    outputs, metrics = module.compute_loss(
                        labels=labels,
                        loss_config=loss_config,
                        **call_batch,
                    )
            aux = {
                name: value
                for name, value in (("z_loss", metrics.z_loss), ("accuracy", metrics.accuracy))
                if value is not None
            }
            if not aux:
                # The scheduled runtime requires at least one aux output when the
                # adapter declares has_aux; a detached copy of the loss keeps the
                # output structure stable for loss functions without metrics.
                aux = {"loss_detached": jax.lax.stop_gradient(outputs.loss)}
            return outputs.loss, aux

    return scheduled_loss


def _base_scheduled_weight_semantics(loss_config, loss_type: str) -> str | None:
    """Classify whether the base-trainer loss is a valid-token mean.

    The scheduled per-microbatch weighting reproduces the SPMD estimator
    only when each microbatch's loss is ``sum(token_losses) / n_valid`` —
    the default causal-LM normalization. Any other reduction keeps the
    historical uniform microbatch mean.

    Args:
        loss_config: The step's :class:`LossConfig` (or ``None``).
        loss_type: The base-trainer loss flavor (``"nll"`` or ``"dft"``).

    Returns:
        ``"valid_only"`` (mask = label validity), ``"masked"``
        (mask = attention/loss weights AND validity), or ``None`` when
        the loss is not a token-mean and weighting must stay uniform.
    """
    if loss_type == "dft":
        return "valid_only"
    if loss_config is None:
        return "masked"
    if getattr(loss_config, "mtp_only", False):
        return None
    reduction = getattr(loss_config, "reduction", None)
    if reduction == "mean":
        return "masked"
    if reduction is not None:
        return None
    if bool(getattr(loss_config, "divide_weight_sum", False)):
        return None
    factor = getattr(loss_config, "loss_normalizing_factor", None)
    name = getattr(factor, "value", factor)
    if isinstance(name, str) and name in ("NUM_REAL_TARGET_TOKENS", "NO_WEIGHT_NUM_REAL_TARGET_TOKENS"):
        return "masked"
    return None


def _model_uses_causal_lm_loss(call) -> bool:
    """Whether the call's model routes ``compute_loss`` to ``ForCausalLMLoss``.

    Resolved once at weight-fn build time (per adapter cache miss) from the
    live trainer state on the scheduled call. Anything that cannot be
    positively identified as the causal-LM loss — a non-CLM task head, a
    custom ``loss_type`` on the config, or a call carrying no state — fails
    closed so the scheduled reduction keeps the always-correct uniform
    microbatch weighting.

    Args:
        call: Scheduled call descriptor (``ScheduledStepCall``-like).

    Returns:
        bool: ``True`` only when the model's ``loss_function`` is
        ``ForCausalLMLoss``.
    """
    state = getattr(call, "state", None)
    if state is None:
        state = call.get("state")
    try:
        model = state.model if state is not None else None
        loss_fn = getattr(model, "loss_function", None)
    except Exception:
        return False
    # Identity first (robust to renames); the __name__ fallback keeps
    # functools.partial / thin wrappers that preserve __name__ working.
    if loss_fn is ForCausalLMLoss:
        return True
    return getattr(loss_fn, "__name__", None) == "ForCausalLMLoss"


def _make_base_scheduled_weight(call):
    """Build the per-microbatch valid-token weight for the base adapter.

    Mirrors the denominator used by ``ForCausalLMLoss`` /
    ``causal_lm_loss_chunked_lm_head``: shifted labels, ``ignore_index``
    validity, ANDed with the shifted attention mask when present.

    The weight only reproduces the causal-LM denominator: models whose
    ``compute_loss`` routes to a different head loss (question answering,
    token classification, ...) would get a silently WRONG weight through
    the ``input_ids`` fallback, so any non-``ForCausalLMLoss`` model keeps
    the historical uniform microbatch weighting (``None``).

    Args:
        call: Scheduled call descriptor providing ``loss_config``,
            ``loss_type``, and the live trainer ``state``.

    Returns:
        tp.Callable | None: ``(tree, batch) -> Array`` scalar weight
        callable, or ``None`` when the loss is not a token-mean (uniform
        microbatch weighting is kept).
    """
    loss_config = call.get("loss_config")
    loss_type = call.get("loss_type", "nll")
    semantics = _base_scheduled_weight_semantics(loss_config, loss_type)
    if semantics is None:
        return None
    if loss_type != "dft" and not _model_uses_causal_lm_loss(call):
        # ``dft`` computes its own causal-LM metrics and never consults the
        # model's loss function, so it keeps the valid-token weight.
        return None
    ignore_index = -100 if loss_config is None else int(loss_config.ignore_index)
    # ``dft`` always shifts inside ``_dft_causal_lm_metrics`` regardless of
    # ``loss_config.shift_tokens`` — the weight must shift with it.
    shift = True if (loss_type == "dft" or loss_config is None) else bool(loss_config.shift_tokens)

    def scheduled_weight(tree, batch):
        """Return this microbatch's valid-token count as a scalar array.

        Args:
            tree: Model parameter tree (unused).
            batch: Microbatch-sliced input mapping.

        Returns:
            jax.Array: Scalar float32 token count.
        """
        del tree
        labels = batch.get("labels")
        if labels is None:
            labels = batch.get("input_ids")
        if labels is None:
            return jnp.asarray(1.0, dtype=jnp.float32)
        labels = jnp.asarray(labels)
        attention_mask = batch.get("attention_mask")
        if shift and labels.ndim >= 2 and labels.shape[-1] > 1:
            shift_labels = labels[..., 1:]
            if attention_mask is not None and jnp.shape(attention_mask) == labels.shape:
                attention_mask = attention_mask[..., 1:]
        else:
            shift_labels = labels
        valid = (shift_labels != ignore_index).astype(jnp.float32)
        if semantics == "valid_only":
            return jnp.sum(valid)
        # NOTE: ``decoder_loss_weights`` is deliberately NOT consulted here.
        # The production dense CLM path (``ForCausalLMLoss``) derives its
        # denominator from the attention mask and label validity only — no
        # base-trainer batch producer emits ``decoder_loss_weights`` — so
        # honoring it here would diverge from the loss it must mirror.
        if attention_mask is not None and jnp.shape(attention_mask) == jnp.shape(shift_labels):
            base = jnp.asarray(attention_mask, jnp.float32)
        else:
            base = valid
        return jnp.sum(base * valid)

    return scheduled_weight


register_scheduled_loss_adapter(
    step_fn=training_step,
    adapter=ScheduledLossAdapter(
        name="base",
        make_loss=_make_base_scheduled_loss,
        make_cache_key=_base_scheduled_loss_cache_key,
        has_aux=True,
        make_microbatch_weight=_make_base_scheduled_weight,
    ),
)
