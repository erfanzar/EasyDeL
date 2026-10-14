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
from eformer.loggings import get_logger
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import CausalLMLossStrategy, ForCausalLMLoss, LossConfig, LossMetrics
from easydel.modules._base.causal_lm_module import BaseCausalLMModule
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
from easydel.modules.gemma.modeling_gemma import GemmaForCausalLM
from easydel.modules.gemma2.modeling_gemma2 import Gemma2ForCausalLM

from ..training_utils import (
    ScheduledLossAdapter,
    _infer_batch_size,
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

logger = get_logger(__name__)


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

    # CE/aux split: only needed when microbatches are averaged (see ``_split_ce_aux_objective``).
    split_count = (
        _ce_aux_split_count({"loss_config": loss_config, "loss_type": loss_type, "state": state})
        if is_training and gradient_accumulation_steps > 1
        else None
    )

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
                minibatch = dict(minibatch)
                ce_scale = minibatch.pop(_CE_TOKEN_SCALE_KEY, None)
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
            loss = outputs.loss
            if ce_scale is not None and split_count is not None:
                loss, metrics = _split_ce_aux_objective(loss, metrics, ce_scale, split_count, minibatch)
            return loss, metrics

    if not is_training:
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(state.graphstate, batch)
        return metrics

    with jax.named_scope(scope_root + "/grad_and_minibatch"):
        # Weight microbatches by valid-token count (same estimator as the scheduled MPMD
        # path) so a token-mean loss matches the full batch when microbatches hold
        # different numbers of valid tokens. ``None`` keeps the uniform average.
        microbatch_weight_fn = (
            _make_base_spmd_weight({"loss_config": loss_config, "loss_type": loss_type, "state": state, "batch": batch})
            if gradient_accumulation_steps > 1
            else None
        )
        if split_count is not None:
            # Added after the weight decision so the aux probe sees the original batch.
            batch = dict(batch)
            batch[_CE_TOKEN_SCALE_KEY] = _ce_token_scale_leaf(split_count, batch)
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
            microbatch_weight_fn=microbatch_weight_fn,
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
    split_count = _ce_aux_split_count(call)

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
                batch = dict(batch)
                ce_scale = batch.pop(_CE_TOKEN_SCALE_KEY, None)
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
            loss = outputs.loss
            if ce_scale is not None and split_count is not None:
                # Split before collecting the reported metrics so they match the objective (r11 S13-2).
                loss, metrics = _split_ce_aux_objective(loss, metrics, ce_scale, split_count, batch)
            aux = {
                name: value
                for name, value in (("z_loss", metrics.z_loss), ("accuracy", metrics.accuracy))
                if value is not None
            }
            if not aux:
                # The scheduled runtime requires at least one aux output when the
                # adapter declares has_aux; a detached copy of the loss keeps the
                # output structure stable for loss functions without metrics.
                aux = {"loss_detached": jax.lax.stop_gradient(loss)}
            return loss, aux

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
    closed so the scheduled reduction keeps the historical uniform
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
    except AttributeError:
        return False
    # Identity first (robust to renames); the __name__ fallback keeps
    # functools.partial / thin wrappers that preserve __name__ working.
    if loss_fn is ForCausalLMLoss:
        return True
    return getattr(loss_fn, "__name__", None) == "ForCausalLMLoss"


# These exact classes' forward bodies return only logits/hidden states/cache. Their
# backbones have no router/MTP aux output; Gemma2's softcap and Cohere's logit
# scaling act on logits, not on the training objective. Do not accept subclasses
# through this exception: they may override an auxiliary hook or backbone.
_CE_ONLY_CAUSAL_LM_CLASSES = frozenset((GemmaForCausalLM, Gemma2ForCausalLM, CohereForCausalLM))


def _has_reviewed_ce_only_forward(model_type: type) -> bool:
    """Accept the inherited CLM forward or an exactly reviewed CE-only head."""
    return model_type.forward is BaseCausalLMModule.forward or model_type in _CE_ONLY_CAUSAL_LM_CLASSES


def _make_base_spmd_weight(call):
    """Use token weights only when the objective provably has no unnormalized aux term.

    Used by both the SPMD base ``training_step`` (``minibatch_call``) and the
    scheduled MPMD adapter (``make_microbatch_weight``): both reduce the loss
    returned by ``module.compute_loss``, which adds ``outputs.aux_loss``
    without token normalization. Token-weighting microbatches would rescale
    that auxiliary term (and drop it entirely on zero-label microbatches).

    Weighting is enabled when the aux term is absent, decided in two steps:
    (1) statically, for the inherited ``BaseCausalLMModule.forward`` or a
    reviewed CE-only head with no router feature; (2) otherwise by abstractly
    tracing ``compute_loss`` (``jax.eval_shape``, no compute) and checking that
    no ``aux_loss``/``mtp_loss`` metric is produced. When aux is present, or
    cannot be ruled out (custom ``compute_loss``/loss strategy, trace failure,
    no batch), the result is ``None`` (uniform microbatch mean) and a one-time
    warning names the reason. DFT bypasses compute_loss and explicitly ignores
    the model's aux output, so it is always weighted.

    Args:
        call: A ``ScheduledStepCall`` (state on ``.state``) or a plain mapping
            with ``loss_config``, ``loss_type``, ``state`` and (for the aux
            probe) ``batch`` keys.

    Returns:
        tp.Callable | None: The valid-token weight callable, or ``None``.
    """
    if call.get("loss_type", "nll") == "dft":
        return _make_base_scheduled_weight(call)
    state = getattr(call, "state", None)
    if state is None:
        state = call.get("state")
    try:
        model = state.model if state is not None else None
    except AttributeError:
        model = None
    reason = _token_weight_blocker(model, call)
    if reason is not None:
        model_name = type(model).__name__
        logger.warn_once(
            f"Gradient accumulation / pipeline microbatches for {model_name} use the uniform microbatch mean "
            f"instead of valid-token weighting: {reason}."
        )
        return None
    return _make_base_scheduled_weight(call)


def _compute_loss_emits_aux(model, batch, loss_config) -> bool:
    """Abstractly trace ``compute_loss`` and report whether it adds an auxiliary loss.

    ``EasyDeLBaseModule.compute_loss`` adds ``outputs.aux_loss`` to the loss and records it in
    ``other_metrics["aux_loss"]`` only when the forward returns a non-None aux loss (router load
    balancing, MTP). The presence of that key in the traced metrics is therefore an exact,
    trace-time-static answer, with no numerical compute (``jax.eval_shape``).

    Args:
        model: Live module whose ``compute_loss`` is probed.
        batch: Mapping of batch arrays (only shapes/dtypes are used).
        loss_config: Loss configuration forwarded to ``compute_loss``.

    Returns:
        bool: ``True`` when the traced metrics carry an ``aux_loss`` or ``mtp_loss`` entry.
    """

    def _abstract(leaf):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return jax.ShapeDtypeStruct(tuple(leaf.shape), leaf.dtype)
        return leaf

    def probe(minibatch):
        minibatch = {k: v for k, v in minibatch.items() if k != _CE_TOKEN_SCALE_KEY}
        call_batch = model.prepare_inputs_for_call(**minibatch)
        labels = call_batch.pop("labels", None)
        _, metrics = model.compute_loss(labels=labels, loss_config=loss_config, **call_batch)
        return dict(metrics.other_metrics or {})

    shapes = jax.eval_shape(probe, jax.tree_util.tree_map(_abstract, dict(batch)))
    return "aux_loss" in shapes or "mtp_loss" in shapes


def _token_weight_blocker(model, call) -> str | None:
    """Explain why valid-token microbatch weighting is unsafe, or return ``None`` when it is safe.

    Token weighting rescales every term of the loss by the microbatch's valid-token share. That is
    exact for the token-mean cross-entropy but wrong for an auxiliary term (MoE router / MTP) that
    ``compute_loss`` adds without token normalization. Weighting is therefore allowed only when the
    objective provably has no such term: either statically (inherited or reviewed CE-only forward
    with no router feature) or by abstractly tracing ``compute_loss`` and finding no aux metric.

    Args:
        model: The trainer state's module (may be ``None``).
        call: Scheduled call or mapping; supplies ``batch`` and ``loss_config`` for the probe.

    Returns:
        str | None: Human-readable reason to fall back to uniform weighting, else ``None``.
    """
    structural = _structural_clm_blocker(model)
    if structural is not None:
        return structural
    model_type = type(model)
    if _has_reviewed_ce_only_forward(model_type) and getattr(model, "_router_aux_loss_feature", False) is None:
        return None
    batch = call.get("batch")
    if batch is None:
        batch = getattr(call, "batch", None)
    if batch is None:
        return "the forward may add an unnormalized auxiliary loss and no batch is available to prove otherwise"
    try:
        emits_aux = _compute_loss_emits_aux(model, batch, call.get("loss_config"))
    except (AttributeError, KeyError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        return (
            "the forward may add an unnormalized auxiliary loss and tracing compute_loss to check failed "
            f"({type(exc).__name__}: {exc})"
        )
    if emits_aux:
        return "compute_loss adds an unnormalized auxiliary loss (MoE router / MTP)"
    return None


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
    count = _valid_token_count_fn(loss_config, loss_type)
    if count is None:
        return None
    if loss_type != "dft" and not _model_uses_causal_lm_loss(call):
        # ``dft`` computes its own causal-LM metrics and never consults the
        # model's loss function, so it keeps the valid-token weight.
        return None

    def scheduled_weight(tree, batch):
        """Return this microbatch's valid-token count as a scalar array."""
        del tree
        return count(batch)

    return scheduled_weight


def _valid_token_count_fn(loss_config, loss_type: str):
    """Return ``batch -> scalar`` counting the tokens the causal-LM loss normalizes by.

    Mirrors the denominator of ``ForCausalLMLoss``: shifted labels, ``ignore_index``
    validity, ANDed with the shifted attention mask when present. ``None`` when the
    loss is not a valid-token mean (see :func:`_base_scheduled_weight_semantics`).
    """
    semantics = _base_scheduled_weight_semantics(loss_config, loss_type)
    if semantics is None:
        return None
    ignore_index = -100 if loss_config is None else int(loss_config.ignore_index)
    # ``dft`` always shifts inside ``_dft_causal_lm_metrics`` regardless of
    # ``loss_config.shift_tokens`` — the count must shift with it.
    shift = True if (loss_type == "dft" or loss_config is None) else bool(loss_config.shift_tokens)

    def count(batch):
        """Return the batch's valid-token count as a float32 scalar."""
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

    return count


# Per-row batch leaf carrying ``B / W`` (global rows over global valid tokens). It lets each
# microbatch rescale ONLY its cross-entropy term so the uniform microbatch mean reproduces the
# full-batch valid-token CE while the auxiliary (router / MTP) term keeps its uniform mean.
_CE_TOKEN_SCALE_KEY = "__easydel_ce_token_scale__"


def _structural_clm_blocker(model) -> str | None:
    """Reason the model's loss is not the standard causal-LM ``compute_loss``, else ``None``."""
    if not isinstance(model, BaseCausalLMModule):
        return "the model is not a BaseCausalLMModule"
    model_type = type(model)
    if model_type.compute_loss is not EasyDeLBaseModule.compute_loss:
        return "the model overrides compute_loss"
    if model_type.loss_strategy is not EasyDeLBaseModule.loss_strategy:
        return "the model overrides loss_strategy"
    try:
        if model.loss_function is not ForCausalLMLoss or type(model.loss_strategy) is not CausalLMLossStrategy:
            return "the model does not use the causal-LM loss"
    except AttributeError:
        return "the model does not expose a resolvable loss function"
    return None


def _ce_aux_split_count(call):
    """Valid-token count fn when the CE term can be token-weighted separately from aux, else ``None``.

    Applies only to the standard causal-LM ``compute_loss`` (loss = token-mean CE + aux) with a
    valid-token-mean loss config; DFT never adds aux, so it never needs the split.
    """
    loss_type = call.get("loss_type", "nll")
    if loss_type == "dft":
        return None
    state = getattr(call, "state", None)
    if state is None:
        state = call.get("state")
    try:
        model = state.model if state is not None else None
    except AttributeError:
        model = None
    if _structural_clm_blocker(model) is not None:
        return None
    return _valid_token_count_fn(call.get("loss_config"), loss_type)


def _ce_token_scale_leaf(count, batch) -> jax.Array:
    """Per-row ``B / W`` leaf for the full batch (``0`` when the batch has no valid tokens)."""
    rows = _infer_batch_size(batch)
    total = jnp.asarray(count(batch), jnp.float32)
    value = jnp.where(total > 0, jnp.float32(rows) / jnp.maximum(total, 1.0), jnp.float32(0.0))
    return jnp.full((rows,), value, dtype=jnp.float32)


def _split_ce_aux_objective(loss, metrics, scale_leaf, count, minibatch):
    """Rescale only the CE part of ``loss = CE + aux`` by ``w_i * B / (W * rows_i)``.

    Averaging these per-microbatch objectives uniformly over the ``B / rows_i`` equal
    microbatches gives ``sum_i w_i CE_i / W + mean_i aux_i``: the full-batch valid-token CE plus
    the uniform-mean auxiliary term. Zero-label microbatches keep their aux gradient.

    The reported metrics are split the same way (r11 S13-2): ``loss`` becomes the split objective
    and the valid-token-normalized ``accuracy`` / ``z_loss`` take the same CE factor, so their
    uniform microbatch mean is the full-batch token-weighted value. ``aux_loss`` stays uniform and
    ``weight_sum`` is untouched. Returns ``(loss, metrics)`` unchanged when no aux term was
    produced or no scale leaf is present.

    Returns:
        tuple: ``(objective, metrics)``.
    """
    aux = (getattr(metrics, "other_metrics", None) or {}).get("aux_loss")
    if scale_leaf is None or aux is None:
        return loss, metrics
    rows = scale_leaf.shape[0]
    ce = loss - aux
    scale = jnp.asarray(count(minibatch), jnp.float32) * jnp.mean(scale_leaf) / jnp.float32(rows)
    objective = ce * scale.astype(ce.dtype) + aux

    def _scaled(value):
        if value is None:
            return None
        value = jnp.asarray(value)
        return value * scale.astype(value.dtype)

    metrics = metrics.replace(
        loss=jax.lax.stop_gradient(objective),
        accuracy=_scaled(metrics.accuracy),
        z_loss=_scaled(metrics.z_loss),
    )
    return objective, metrics


def _base_scheduled_prepare_batch(call):
    """Attach the CE token-scale leaf when the scheduled loss may need the CE/aux split."""
    batch = dict(call.batch)
    batch.pop(_CE_TOKEN_SCALE_KEY, None)
    count = _ce_aux_split_count(call)
    if count is not None:
        batch[_CE_TOKEN_SCALE_KEY] = _ce_token_scale_leaf(count, batch)
    return batch


register_scheduled_loss_adapter(
    step_fn=training_step,
    adapter=ScheduledLossAdapter(
        name="base",
        make_loss=_make_base_scheduled_loss,
        make_cache_key=_base_scheduled_loss_cache_key,
        has_aux=True,
        # Adds the CE token-scale leaf so aux-producing models split CE (token-weighted) from aux.
        prepare_batch=_base_scheduled_prepare_batch,
        # Same CE-only guard as the SPMD step: models that can add an auxiliary loss use the
        # uniform microbatch mean, with the CE/aux split restoring valid-token CE weighting.
        make_microbatch_weight=_make_base_spmd_weight,
    ),
)
