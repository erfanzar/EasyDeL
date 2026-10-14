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

"""Compiled step functions for speculative-decoding drafter training.

The helpers in this file keep target and drafter execution in one step.  The
target path is frozen and stop-gradiented; the drafter path receives gradients
from a KL/CE objective over one or more draft-token positions.  The same loss
builder is registered for regular ``spx.jit`` trainer steps and SpectraX
scheduled MPMD steps.

Attributes:
    _TARGET_BATCH_KEYS: Tuple of batch keys that hold cached target outputs.
        These keys are stripped from the batch before forwarding through models.
"""

from __future__ import annotations

import collections.abc
import inspect
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
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

_TARGET_BATCH_KEYS = (
    "target_logits",
    "_target_logits",
    "target_hidden_states",
    "_target_hidden_states",
    "target_last_hidden_state",
    "_target_last_hidden_state",
)


def _first_present(*values: tp.Any) -> tp.Any:
    """Return the first value that is not ``None``.

    Args:
        *values: Values to check in order.

    Returns:
        The first non-``None`` value, or ``None`` if all values are ``None``.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _strip_forward_only_keys(batch: collections.abc.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
    """Return model-call kwargs with trainer-only tensors removed.

    Labels, completion masks, and cached target outputs are consumed by the
    loss code, not by ordinary model forwards.  Removing them here lets the
    trainer call different EasyDeL modules without every module accepting the
    full training batch schema.

    Args:
        batch: Training batch mapping that may contain ``labels``,
            ``completion_mask``, ``assistant_masks``, ``loss_mask``, and cached
            target outputs.

    Returns:
        A mutable ``dict`` copy of ``batch`` with trainer-only keys removed.
    """
    kwargs = dict(batch)
    kwargs.pop("labels", None)
    kwargs.pop("completion_mask", None)
    kwargs.pop("assistant_masks", None)
    kwargs.pop("loss_mask", None)
    for key in _TARGET_BATCH_KEYS:
        kwargs.pop(key, None)
    return kwargs


def _lookup_output_path(output: tp.Any, path: str | None) -> tp.Any:
    """Read a dotted attribute/key path from a module output.

    EasyDeL model outputs may be dataclasses, mapping-like outputs, or custom
    containers.  This helper lets config fields such as ``target_logits_attr``
    point at nested values without making the loss code know the concrete
    output type.

    Args:
        output: An object (dataclass, mapping, or custom container) to search.
        path: Dotted path such as ``"logits"`` or ``"block_logits.draft"``.
            May be ``None``.

    Returns:
        The resolved value, or ``None`` if ``path`` is ``None`` or any part of
        the path is missing.
    """
    if path is None:
        return None
    value = output
    for part in path.split("."):
        if value is None:
            return None
        if isinstance(value, collections.abc.Mapping):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
    return value


def _extract_logits(output: tp.Any, *, attr: str | None, role: str) -> jax.Array:
    """Resolve the logits tensor produced by a target or drafter call.

    Array returns are accepted directly.  Otherwise an explicit attribute path
    is honored first, followed by draft-oriented defaults
    (``draft_logits``, ``mtp_logits``) and the standard ``logits`` field.

    Args:
        output: Raw output from a model forward call.
        attr: Explicit dotted attribute path to read from ``output``.
            If ``None``, default paths are tried.
        role: Human-readable role string used in error messages
            (e.g., ``"target"`` or ``"drafter"``).

    Returns:
        The resolved logits array.

    Raises:
        TypeError: If ``attr`` is set but the path does not resolve, or if no
            logits field can be inferred from the output.
    """
    if isinstance(output, jax.Array) or (hasattr(output, "shape") and hasattr(output, "dtype")):
        return output
    if attr is not None:
        value = _lookup_output_path(output, attr)
        if value is None:
            raise TypeError(f"{role} output does not expose logits at `{attr}`.")
        return value
    for candidate in ("draft_logits", "mtp_logits", "logits"):
        value = _lookup_output_path(output, candidate)
        if value is not None:
            return value
    raise TypeError(f"Could not infer {role} logits; set the corresponding logits attribute in the config.")


def _call_module_or_method(
    module: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    *,
    method_name: str | None,
    output_hidden_states: bool = False,
    extra_kwargs: collections.abc.Mapping[str, tp.Any] | None = None,
) -> tp.Any:
    """Call a model forward path with only kwargs it can accept.

    The trainer supports normal ``__call__``/``forward`` execution and custom
    method names for drafters or targets.  Batch keys are filtered through the
    callable signature after optional target-output kwargs are injected.

    Args:
        module: The model module or object to call.
        batch: Full training batch mapping.
        method_name: Method name to invoke.  ``None``, ``""``, ``"__call__"``,
            or ``"forward"`` all map to ``module.__call__``.
        output_hidden_states: Whether to request hidden states from the call.
        extra_kwargs: Additional keyword arguments merged into the call kwargs
            after stripping.

    Returns:
        The raw output of the model call.
    """
    method = module.__call__ if method_name in (None, "", "__call__", "forward") else getattr(module, method_name)
    call_kwargs = _strip_forward_only_keys(batch)
    if output_hidden_states:
        call_kwargs["output_hidden_states"] = True
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)
    call_kwargs = filter_kwargs_for_callable(method, call_kwargs)
    call_kwargs = sanitize_model_call_kwargs(call_kwargs)
    return method(**call_kwargs)


def _module_accepts_block_loss_kwargs(module: tp.Any, method_name: str | None) -> bool:
    """Return whether the drafter forward should receive block-loss kwargs.

    Args:
        module: The model module to inspect.
        method_name: Method name to inspect.  ``None`` maps to ``__call__``.

    Returns:
        ``True`` if the method signature accepts ``loss_mask`` or
        ``return_block_outputs``, or if the method accepts ``**kwargs`` and the
        module config has ``block_size``.  ``False`` otherwise.
    """
    method = module.__call__ if method_name in (None, "", "__call__", "forward") else getattr(module, method_name)
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return getattr(getattr(module, "config", None), "block_size", None) is not None
    parameters = signature.parameters
    if "loss_mask" in parameters or "return_block_outputs" in parameters:
        return True
    has_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    return has_var_kwargs and getattr(getattr(module, "config", None), "block_size", None) is not None


def _deepspec_block_loss_requires_target_distribution(config: tp.Any) -> bool:
    """Return whether the native block loss needs target logits/probabilities."""
    if config is None:
        return True
    return (
        float(getattr(config, "l1_loss_alpha", 0.0)) > 0.0 or float(getattr(config, "confidence_head_alpha", 0.0)) > 0.0
    )


def _select_target_context_hidden_states(hidden_states: tp.Any, layer_ids: tp.Sequence[int] | None) -> tp.Any:
    """Keep only the target layers consumed by DSpark-style context projectors."""
    if hidden_states is None or layer_ids is None or not isinstance(hidden_states, list | tuple):
        return hidden_states
    selected = [hidden_states[0 if int(layer_id) == -1 else int(layer_id) + 1] for layer_id in layer_ids]
    return jnp.concatenate(selected, axis=-1)


def _target_num_hidden_layers(target_module: tp.Any) -> int | None:
    """Resolve a target module's decoder depth from text or multimodal config."""
    config = getattr(target_module, "config", None)
    if config is None:
        return None
    for candidate in (config, getattr(config, "text_config", None)):
        value = getattr(candidate, "num_hidden_layers", None)
        if value is not None:
            return int(value)
    return None


def _target_layer_ids_are_final(
    target_module: tp.Any,
    layer_ids: tp.Sequence[int] | None,
    target_num_hidden_layers: int | None = None,
) -> bool:
    """Return whether requested DSpark context is exactly the final decoder output."""
    if layer_ids is None or len(tuple(layer_ids)) != 1:
        return False
    num_hidden_layers = target_num_hidden_layers
    if num_hidden_layers is None:
        num_hidden_layers = _target_num_hidden_layers(target_module)
    if num_hidden_layers is None:
        return False
    return int(next(iter(layer_ids))) == num_hidden_layers - 1


def _chain_method_name(module: tp.Any, method_name: str | None, num_draft_tokens: int) -> str | None:
    """Resolve whether to use a multi-token draft-chain method.

    ``auto`` means: when more than one draft token is requested and the module
    implements ``compute_mtp_chain``, use it; otherwise fall back to logits
    returned by the normal drafter forward.

    Args:
        module: The model module to inspect.
        method_name: The configured chain method name.  ``None`` or ``""``
            disables chaining.  ``"auto"`` triggers heuristic selection.
        num_draft_tokens: Number of draft tokens requested.

    Returns:
        The resolved method name (e.g., ``"compute_mtp_chain"``), or ``None``
        if no chain method should be used.
    """
    if method_name in (None, ""):
        return None
    if method_name == "auto":
        if num_draft_tokens > 1 and hasattr(module, "compute_mtp_chain"):
            return "compute_mtp_chain"
        return None
    return method_name


def _call_chain_method(
    module: tp.Any,
    method_name: str,
    base_outputs: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    num_draft_tokens: int,
) -> jax.Array:
    """Call a drafter method that expands base outputs into draft-step logits.

    Methods such as ``compute_mtp_chain`` receive the normal forward output,
    ``input_ids``, and the requested number of draft steps.  ``attention_mask``
    is passed only when the method signature can accept it.

    Args:
        module: The model module containing the chain method.
        method_name: Name of the chain method to invoke.
        base_outputs: Base forward outputs to expand.
        batch: Training batch mapping (must contain ``input_ids``).
        num_draft_tokens: Number of draft tokens to generate.

    Returns:
        Draft logits returned by the chain method.

    Raises:
        ValueError: If ``input_ids`` is not present in ``batch``.
    """
    input_ids = batch.get("input_ids")
    if input_ids is None:
        raise ValueError(f"`{method_name}` requires `input_ids` in the batch.")
    method = getattr(module, method_name)
    kwargs: dict[str, tp.Any] = {}
    attention_mask = batch.get("attention_mask")
    try:
        signature = inspect.signature(method)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if attention_mask is not None and (accepts_kwargs or "attention_mask" in signature.parameters):
            kwargs["attention_mask"] = attention_mask
    except (TypeError, ValueError):
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
    return method(base_outputs, input_ids, int(num_draft_tokens), **kwargs)


def _resolve_draft_logits(
    module: tp.Any,
    drafter_outputs: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    *,
    target_logits: jax.Array,
    draft_logits_attr: str | None,
    draft_chain_method: str | None,
    num_draft_tokens: int,
) -> jax.Array:
    """Resolve drafter logits from either explicit outputs or chain methods.

    Multi-step heads may need the module method because the normal forward
    returns hidden states rather than final draft logits.  Single-step drafters
    usually resolve directly from ``draft_logits``, ``mtp_logits``, or
    ``logits``.

    Args:
        module: The drafter model module.
        drafter_outputs: Raw outputs from the drafter forward call.
        batch: Training batch mapping.
        target_logits: Frozen target logits used for shape normalization.
        draft_logits_attr: Explicit dotted attribute path for drafter logits.
        draft_chain_method: Configured chain method name.
        num_draft_tokens: Number of draft tokens requested.

    Returns:
        Normalized drafter logits shaped ``[draft_steps, batch, sequence, vocab]``.
    """
    chain_name = _chain_method_name(module, draft_chain_method, num_draft_tokens)
    if chain_name is not None:
        return _call_chain_method(module, chain_name, drafter_outputs, batch, num_draft_tokens)
    logits = _extract_logits(drafter_outputs, attr=draft_logits_attr, role="drafter")
    return _normalize_draft_logits(logits, target_logits, num_draft_tokens=num_draft_tokens)


def _normalize_draft_logits(
    draft_logits: jax.Array,
    target_logits: jax.Array,
    *,
    num_draft_tokens: int,
) -> jax.Array:
    """Convert drafter logits to ``[draft_steps, batch, sequence, vocab]``.

    The trainer accepts the common layouts returned by draft modules:
    ``[B, S, V]`` for one step, ``[K, B, S, V]``, ``[B, K, S, V]``, and
    ``[B, S, K, V]``.  The normalized leading axis is then sliced to the
    configured number of draft tokens.

    Args:
        draft_logits: Raw drafter logits array.
        target_logits: Target logits used as the shape reference for
            ``[batch, sequence, vocab]``.
        num_draft_tokens: Maximum number of draft steps to keep after
            normalization.

    Returns:
        Drafter logits with leading axis ``[draft_steps, ...]`` trimmed to
        ``num_draft_tokens``.

    Raises:
        ValueError: If the logits shape is incompatible with any known layout.
    """
    if draft_logits.ndim == target_logits.ndim:
        return draft_logits[None, ...]
    if draft_logits.ndim != target_logits.ndim + 1:
        raise ValueError(
            "Drafter logits must be [batch, sequence, vocab] or include one draft-step axis; "
            f"got {draft_logits.shape} vs target {target_logits.shape}."
        )

    batch, sequence, vocab = target_logits.shape
    if draft_logits.shape[1:] == (batch, sequence, vocab):
        chain = draft_logits
    elif draft_logits.shape[0] == batch and draft_logits.shape[2:] == (sequence, vocab):
        chain = jnp.swapaxes(draft_logits, 0, 1)
    elif draft_logits.shape[:2] == (batch, sequence) and draft_logits.shape[-1] == vocab:
        chain = jnp.transpose(draft_logits, (2, 0, 1, 3))
    else:
        raise ValueError(
            "Could not identify the draft-step axis in drafter logits; expected [K,B,S,V], [B,K,S,V], "
            f"or [B,S,K,V], got {draft_logits.shape}."
        )
    return chain[: int(num_draft_tokens)]


def _masked_sum(value: jax.Array, weights: jax.Array) -> jax.Array:
    """Sum a token-level metric under the active-token mask in float32.

    Args:
        value: Token-level values to sum.
        weights: Token-level weights (mask) matching ``value`` shape.

    Returns:
        Scalar sum of ``value * weights`` in float32.
    """
    return jnp.sum(value.astype(jnp.float32) * weights.astype(jnp.float32))


def _loss_mask_from_batch(batch: collections.abc.Mapping[str, tp.Any]) -> jax.Array | None:
    """Resolve the token mask used by speculative and block-drafter losses.

    Args:
        batch: Training batch mapping.

    Returns:
        The active-token mask, or ``None`` if no mask can be inferred.
    """
    loss_mask = _first_present(batch.get("completion_mask"), batch.get("loss_mask"), batch.get("assistant_masks"))
    if loss_mask is not None:
        return loss_mask
    labels = batch.get("labels")
    if labels is not None:
        return (labels != -100).astype(jnp.float32)
    return batch.get("attention_mask")


def _aligned_step_components(
    draft_logits: jax.Array,
    target_logits: jax.Array,
    *,
    labels: jax.Array | None,
    input_ids: jax.Array | None,
    attention_mask: jax.Array | None,
    loss_mask: jax.Array | None,
    temperature: float,
    target_offset: int,
) -> dict[str, jax.Array]:
    """Compute masked KL, CE, and acceptance numerators for one draft step.

    ``target_offset`` aligns drafter position ``t`` with the target
    distribution used to score the drafted token.  Attention, completion, and
    ``-100`` label masks are multiplied into one denominator so every reported
    numerator uses the same active-token set.  The acceptance-rate numerator is
    the distribution overlap ``sum(min(p_target, p_draft))`` at the loss
    temperature, which is the expected stochastic acceptance probability under
    the target/drafter distributions used by the KL term.

    Args:
        draft_logits: Drafter logits for this step ``[batch, sequence, vocab]``.
        target_logits: Frozen target logits ``[batch, sequence, vocab]``.
        labels: Optional hard labels ``[batch, sequence]``.
        input_ids: Optional input IDs used as labels when ``labels`` is ``None``.
        attention_mask: Optional attention mask.
        loss_mask: Optional loss mask.
        temperature: Softmax temperature for the KL and acceptance terms.
        target_offset: Position offset between drafter and target logits.
            Must be non-negative.

    Returns:
        Dictionary of numerators and denominator containing:
        ``weight_sum``, ``kl_num``, ``ce_num``, ``accuracy_num``,
        ``accept_rate_num``, ``acceptance_num``, ``target_prob_num``.

    Raises:
        ValueError: If ``target_offset`` is negative or the sequence is too
            short for the offset.
    """
    if target_offset < 0:
        raise ValueError("`target_offset` must be non-negative.")
    max_tokens = min(int(draft_logits.shape[1]), int(target_logits.shape[1]) - target_offset - 1)
    if max_tokens <= 0:
        raise ValueError(
            "Sequence length is too short for the configured draft-target offset "
            f"({target_offset}) and shapes {draft_logits.shape}, {target_logits.shape}."
        )

    student = draft_logits[:, :max_tokens, :].astype(jnp.float32)
    teacher = target_logits[:, target_offset : target_offset + max_tokens, :].astype(jnp.float32)
    weights = jnp.ones(student.shape[:2], dtype=jnp.float32)
    mask_slice = slice(target_offset + 1, target_offset + 1 + max_tokens)

    if attention_mask is not None:
        weights = weights * attention_mask[:, mask_slice].astype(jnp.float32)
    if loss_mask is not None:
        weights = weights * loss_mask[:, mask_slice].astype(jnp.float32)

    hard_labels = labels if labels is not None else input_ids
    aligned_labels = None
    if hard_labels is not None:
        aligned_labels = hard_labels[:, mask_slice]
        weights = weights * (aligned_labels != -100).astype(jnp.float32)

    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=jnp.float32))
    temp = jnp.asarray(temperature, dtype=jnp.float32)
    teacher_log_probs = jax.nn.log_softmax(teacher / temp, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    student_log_probs_t = jax.nn.log_softmax(student / temp, axis=-1)
    student_probs_t = jnp.exp(student_log_probs_t)
    kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs_t), axis=-1) * (temp**2)
    expected_acceptance = jnp.sum(jnp.minimum(teacher_probs, student_probs_t), axis=-1)

    target_top = jnp.argmax(teacher, axis=-1)
    draft_top = jnp.argmax(student, axis=-1)
    acceptance_match = (target_top == draft_top).astype(jnp.float32)
    target_prob_of_draft = jnp.take_along_axis(
        jax.nn.softmax(teacher, axis=-1),
        draft_top[..., None],
        axis=-1,
    ).squeeze(-1)

    if aligned_labels is None:
        ce = jnp.zeros_like(kl)
        correct = jnp.zeros_like(kl)
    else:
        safe_labels = jnp.where(aligned_labels != -100, aligned_labels, 0)
        student_log_probs = jax.nn.log_softmax(student, axis=-1)
        ce = -jnp.take_along_axis(student_log_probs, safe_labels[..., None], axis=-1).squeeze(-1)
        correct = (draft_top == safe_labels).astype(jnp.float32)

    return {
        "weight_sum": denom,
        "kl_num": _masked_sum(kl, weights),
        "ce_num": _masked_sum(ce, weights),
        "accuracy_num": _masked_sum(correct, weights),
        "accept_rate_num": _masked_sum(expected_acceptance, weights),
        "acceptance_num": _masked_sum(acceptance_match, weights),
        "target_prob_num": _masked_sum(target_prob_of_draft, weights),
    }


def _resolve_block_logits(output: tp.Any) -> jax.Array | None:
    """Return DeepSpec block logits from a drafter output when present.

    Args:
        output: Raw drafter output to inspect.

    Returns:
        Block logits array if found, otherwise ``None``.
    """
    block_logits = _lookup_output_path(output, "block_logits")
    if block_logits is not None:
        return block_logits
    draft_logits = _lookup_output_path(output, "draft_logits")
    if draft_logits is not None and getattr(draft_logits, "ndim", None) == 4:
        return draft_logits
    return None


def _binary_cross_entropy_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute stable elementwise BCE-with-logits.

    Args:
        logits: Logits array.
        targets: Target binary labels.

    Returns:
        Elementwise BCE loss.
    """
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _deepspec_block_loss(output: tp.Any, config: tp.Any) -> tuple[jax.Array, LossMetrics] | None:
    """Compute DSpark/DFlash native block loss when block outputs are available.

    DeepSpec block drafters return logits as ``[batch, anchors, block, vocab]``
    along with target IDs and an evaluation mask.  This loss mirrors the
    DeepSpec CE/L1/confidence objective while reporting EasyDeL's normal
    speculative accept-rate metrics.

    Args:
        output: Raw drafter output containing ``block_logits``, ``target_ids``,
            ``eval_mask``, and optionally ``aligned_target_logits``,
            ``confidence_logits``, ``block_keep_mask``.
        config: Drafter configuration object (may be ``None``).

    Returns:
        A tuple ``(total_loss, metrics)`` where ``total_loss`` is the weighted
        CE/L1/confidence sum and ``metrics`` is a ``LossMetrics`` object, or
        ``None`` if the required block outputs are absent.

    Raises:
        ValueError: If ``l1_loss_alpha > 0`` but ``aligned_target_logits`` is
            missing, or if confidence loss is requested without
            ``aligned_target_logits``.
    """
    block_logits = _resolve_block_logits(output)
    target_ids = _lookup_output_path(output, "target_ids")
    eval_mask = _lookup_output_path(output, "eval_mask")
    if block_logits is None or target_ids is None or eval_mask is None:
        return None

    block_logits = block_logits.astype(jnp.float32)
    target_ids = target_ids.astype(jnp.int32)
    eval_mask = eval_mask.astype(jnp.float32)
    block_size = int(block_logits.shape[-2])
    vocab_size = int(block_logits.shape[-1])

    loss_weight_mask = eval_mask
    loss_decay_gamma = getattr(config, "loss_decay_gamma", None)
    if loss_decay_gamma is not None and float(loss_decay_gamma) > 0.0:
        positions = jnp.arange(block_size, dtype=jnp.float32).reshape(1, 1, block_size)
        loss_weight_mask = loss_weight_mask * jnp.exp(-positions / float(loss_decay_gamma))

    safe_targets = jnp.clip(target_ids, 0, vocab_size - 1)
    log_probs = jax.nn.log_softmax(block_logits, axis=-1)
    ce_tokens = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    ce_den = jnp.maximum(jnp.sum(loss_weight_mask), jnp.asarray(1.0, dtype=jnp.float32))
    ce_loss = _masked_sum(ce_tokens, loss_weight_mask) / ce_den

    draft_top = jnp.argmax(block_logits, axis=-1).astype(jnp.int32)
    accuracy = _masked_sum((draft_top == safe_targets).astype(jnp.float32), eval_mask) / jnp.maximum(
        jnp.sum(eval_mask), jnp.asarray(1.0, dtype=jnp.float32)
    )

    aligned_target_logits = _lookup_output_path(output, "aligned_target_logits")
    l1_loss = jnp.asarray(0.0, dtype=jnp.float32)
    accept_rate = jnp.zeros_like(eval_mask, dtype=jnp.float32)
    l1_loss_alpha = float(getattr(config, "l1_loss_alpha", 0.0))
    if aligned_target_logits is not None:
        draft_probs = jax.nn.softmax(block_logits, axis=-1)
        target_probs = jax.nn.softmax(aligned_target_logits.astype(jnp.float32), axis=-1)
        l1_per_token = jnp.sum(jnp.abs(draft_probs - target_probs), axis=-1)
        l1_loss = _masked_sum(l1_per_token, loss_weight_mask) / ce_den
        accept_rate = jnp.clip(1.0 - 0.5 * l1_per_token, 0.0, 1.0)
    elif l1_loss_alpha > 0.0:
        raise ValueError("DeepSpec block loss requires `aligned_target_logits` when `l1_loss_alpha > 0`.")

    confidence_logits = _first_present(
        _lookup_output_path(output, "confidence_logits"),
        _lookup_output_path(output, "confidence_pred"),
    )
    confidence_loss = jnp.asarray(0.0, dtype=jnp.float32)
    confidence_head_alpha = float(getattr(config, "confidence_head_alpha", 0.0))
    confidence_metrics: dict[str, jax.Array] = {}
    if confidence_logits is not None and confidence_head_alpha > 0.0:
        if aligned_target_logits is None:
            raise ValueError("DeepSpec confidence loss requires `aligned_target_logits`.")
        confidence_targets = jax.lax.stop_gradient(accept_rate)
        confidence_errors = _binary_cross_entropy_with_logits(confidence_logits, confidence_targets)
        confidence_loss = _masked_sum(confidence_errors, loss_weight_mask) / ce_den
        confidence_probs = jax.nn.sigmoid(confidence_logits.astype(jnp.float32))
        confidence_delta = confidence_probs - confidence_targets
        confidence_metrics = {
            "confidence_abs_error": _masked_sum(jnp.abs(confidence_delta), loss_weight_mask) / ce_den,
            "confidence_bias": _masked_sum(confidence_delta, loss_weight_mask) / ce_den,
            "confidence_cumprod_bias": _masked_sum(
                (
                    jnp.cumprod(confidence_probs * eval_mask, axis=-1)
                    - jnp.cumprod(confidence_targets * eval_mask, axis=-1)
                ),
                loss_weight_mask,
            )
            / ce_den,
        }

    ce_loss_alpha = float(getattr(config, "ce_loss_alpha", 1.0))
    total_loss = (
        jnp.asarray(ce_loss_alpha, dtype=jnp.float32) * ce_loss
        + jnp.asarray(l1_loss_alpha, dtype=jnp.float32) * l1_loss
        + jnp.asarray(confidence_head_alpha, dtype=jnp.float32) * confidence_loss
    )

    block_keep_mask = _lookup_output_path(output, "block_keep_mask")
    valid_blocks = (jnp.sum(eval_mask, axis=-1) > 0).astype(jnp.float32)
    if block_keep_mask is not None:
        valid_blocks = valid_blocks * block_keep_mask.astype(jnp.float32)
    block_den = jnp.maximum(jnp.sum(valid_blocks), jnp.asarray(1.0, dtype=jnp.float32))
    valid_accept_rate = accept_rate * eval_mask
    expected_draft_accepted = jnp.sum(jnp.cumprod(valid_accept_rate, axis=-1), axis=-1)
    tau_probabilistic = _masked_sum(expected_draft_accepted + 1.0, valid_blocks) / block_den

    active_den = jnp.maximum(jnp.sum(eval_mask), jnp.asarray(1.0, dtype=jnp.float32))
    per_step = {}
    for step in range(block_size):
        step_mask = eval_mask[:, :, step]
        step_den = jnp.maximum(jnp.sum(step_mask), jnp.asarray(1.0, dtype=jnp.float32))
        per_step[f"draft_step{step + 1}_accept_rate"] = _masked_sum(accept_rate[:, :, step], step_mask) / step_den

    other_metrics = {
        "spec_ce_loss": ce_loss,
        "deepspec_block_ce_loss": ce_loss,
        "deepspec_block_l1_loss": l1_loss,
        "deepspec_block_confidence_loss": confidence_loss,
        "draft_accept_rate": _masked_sum(accept_rate, eval_mask) / active_den,
        "tau_probabilistic": tau_probabilistic,
        "draft_steps": jnp.asarray(block_size, dtype=jnp.int32),
        **per_step,
        **confidence_metrics,
    }
    return total_loss, LossMetrics(
        loss=total_loss,
        weight_sum=jnp.sum(eval_mask),
        accuracy=accuracy,
        other_metrics=other_metrics,
    )


def speculative_draft_loss(
    draft_logits: jax.Array,
    target_logits: jax.Array,
    *,
    labels: jax.Array | None = None,
    input_ids: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    loss_mask: jax.Array | None = None,
    temperature: float = 1.0,
    alpha: float = 1.0,
    num_draft_tokens: int = 1,
    draft_target_offset: int = 0,
    step_loss_decay: float | None = None,
) -> tuple[jax.Array, LossMetrics]:
    """Compute the full speculative-decoding objective and metrics.

    ``draft_logits`` may describe one or many draft steps.  Each step is
    aligned to the frozen target logits, accumulated under the same masking
    rules, and reported with per-step KL and acceptance-match metrics.  The
    returned scalar is ``alpha * KL + (1 - alpha) * CE``.

    Args:
        draft_logits: Drafter logits (may be multi-step).
        target_logits: Frozen target logits ``[batch, sequence, vocab]``.
        labels: Optional hard labels.
        input_ids: Optional input IDs used as labels fallback.
        attention_mask: Optional attention mask.
        loss_mask: Optional loss mask.
        temperature: Softmax temperature for KL and acceptance terms.
        alpha: Interpolation weight between KL and CE losses.
        num_draft_tokens: Number of draft tokens to evaluate.
        draft_target_offset: Offset between drafter and target positions.
        step_loss_decay: Optional geometric decay weight per draft step.

    Returns:
        Tuple of ``(loss, metrics)`` where ``loss`` is the scalar speculative
        loss and ``metrics`` is a ``LossMetrics`` object with per-step breakdowns.
    """
    chain_logits = _normalize_draft_logits(draft_logits, target_logits, num_draft_tokens=num_draft_tokens)
    steps = int(chain_logits.shape[0])
    totals: dict[str, jax.Array] = {
        "weight_sum": jnp.asarray(0.0, dtype=jnp.float32),
        "kl_num": jnp.asarray(0.0, dtype=jnp.float32),
        "ce_num": jnp.asarray(0.0, dtype=jnp.float32),
        "accuracy_num": jnp.asarray(0.0, dtype=jnp.float32),
        "accept_rate_num": jnp.asarray(0.0, dtype=jnp.float32),
        "acceptance_num": jnp.asarray(0.0, dtype=jnp.float32),
        "target_prob_num": jnp.asarray(0.0, dtype=jnp.float32),
    }
    per_step: dict[str, jax.Array] = {}
    weighted_loss = jnp.asarray(0.0, dtype=jnp.float32)
    use_weighted_steps = step_loss_decay is not None
    for step in range(steps):
        components = _aligned_step_components(
            chain_logits[step],
            target_logits,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            temperature=temperature,
            target_offset=int(draft_target_offset) + step,
        )
        denom = components["weight_sum"]
        step_kl_loss = components["kl_num"] / denom
        step_ce_loss = components["ce_num"] / denom
        per_step[f"draft_step{step + 1}_kl_loss"] = step_kl_loss
        per_step[f"draft_step{step + 1}_accept_rate"] = components["accept_rate_num"] / denom
        per_step[f"draft_step{step + 1}_acceptance_match"] = components["acceptance_num"] / denom
        if use_weighted_steps:
            step_weight = jnp.asarray(float(step_loss_decay) ** step, dtype=jnp.float32)
            alpha_value = jnp.asarray(alpha, dtype=jnp.float32)
            weighted_loss = weighted_loss + step_weight * (
                alpha_value * step_kl_loss + (1.0 - alpha_value) * step_ce_loss
            )
        for key, value in components.items():
            totals[key] = totals[key] + value

    denom = jnp.maximum(totals["weight_sum"], jnp.asarray(1.0, dtype=jnp.float32))
    kl_loss = totals["kl_num"] / denom
    ce_loss = totals["ce_num"] / denom
    alpha_value = jnp.asarray(alpha, dtype=jnp.float32)
    total_loss = weighted_loss if use_weighted_steps else alpha_value * kl_loss + (1.0 - alpha_value) * ce_loss
    metrics = LossMetrics(
        loss=total_loss,
        weight_sum=totals["weight_sum"],
        accuracy=totals["accuracy_num"] / denom,
        other_metrics={
            "spec_kl_loss": kl_loss,
            "spec_ce_loss": ce_loss,
            "draft_accept_rate": totals["accept_rate_num"] / denom,
            "draft_acceptance_match": totals["acceptance_num"] / denom,
            "target_prob_of_draft": totals["target_prob_num"] / denom,
            "draft_steps": jnp.asarray(steps, dtype=jnp.int32),
            "draft_step_loss_decay": jnp.asarray(step_loss_decay or 0.0, dtype=jnp.float32),
            **per_step,
        },
    )
    return total_loss, metrics


def _target_forward_bundle_from_module(
    target_module: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    *,
    target_logits_attr: str,
    target_forward_method: str | None,
    target_output_hidden_states: bool,
    pass_target_outputs: bool,
    require_logits: bool = True,
    target_layer_ids: tp.Sequence[int] | None = None,
    target_num_hidden_layers: int | None = None,
) -> dict[str, tp.Any]:
    """Build the frozen target-output bundle for one minibatch.

    If the batch already carries cached target logits or hidden states, those
    tensors are stop-gradiented and reused.  Otherwise the target module is
    executed with optional hidden-state output, and the requested target logits
    path is extracted for the drafter loss.

    Args:
        target_module: The target model module.
        batch: Training batch mapping.
        target_logits_attr: Dotted attribute path to extract target logits.
        target_forward_method: Optional custom forward method name.
        target_output_hidden_states: Whether to request hidden states.
        pass_target_outputs: Whether to include the full raw output in the
            bundle.

    Returns:
        Dictionary containing ``logits``, and optionally ``hidden_states``,
        ``last_hidden_state``, and ``outputs``.
    """
    cached_logits = batch.get("_target_logits", batch.get("target_logits"))
    if cached_logits is not None:
        result: dict[str, tp.Any] = {"logits": jax.lax.stop_gradient(cached_logits)}
        hidden_states = batch.get("_target_hidden_states", batch.get("target_hidden_states"))
        last_hidden_state = batch.get("_target_last_hidden_state", batch.get("target_last_hidden_state"))
        use_last_hidden_as_context = _target_layer_ids_are_final(
            target_module, target_layer_ids, target_num_hidden_layers
        )
        if last_hidden_state is None and isinstance(hidden_states, list | tuple) and hidden_states:
            last_hidden_state = hidden_states[-1]
        if hidden_states is not None:
            result["hidden_states"] = stop_gradient_tree(
                _select_target_context_hidden_states(hidden_states, target_layer_ids)
            )
        elif use_last_hidden_as_context and last_hidden_state is not None:
            result["hidden_states"] = jax.lax.stop_gradient(last_hidden_state)
        if last_hidden_state is not None and require_logits:
            result["last_hidden_state"] = jax.lax.stop_gradient(last_hidden_state)
        return result

    use_last_hidden_as_context = _target_layer_ids_are_final(target_module, target_layer_ids, target_num_hidden_layers)
    target_extra_kwargs = {"apply_lm_head": False} if not require_logits else None
    outputs = _call_module_or_method(
        target_module,
        batch,
        method_name=target_forward_method,
        output_hidden_states=target_output_hidden_states and not use_last_hidden_as_context,
        extra_kwargs=target_extra_kwargs,
    )
    result = {}
    if require_logits:
        result["logits"] = stop_gradient_tree(_extract_logits(outputs, attr=target_logits_attr, role="target"))
    hidden_states = _lookup_output_path(outputs, "hidden_states")
    last_hidden_state = _lookup_output_path(outputs, "last_hidden_state")
    if last_hidden_state is None and isinstance(hidden_states, list | tuple) and hidden_states:
        last_hidden_state = hidden_states[-1]
    if hidden_states is not None:
        result["hidden_states"] = stop_gradient_tree(
            _select_target_context_hidden_states(hidden_states, target_layer_ids)
        )
    elif use_last_hidden_as_context and last_hidden_state is not None:
        result["hidden_states"] = jax.lax.stop_gradient(last_hidden_state)
    if last_hidden_state is not None and require_logits:
        result["last_hidden_state"] = jax.lax.stop_gradient(last_hidden_state)
    if pass_target_outputs and require_logits:
        result["outputs"] = stop_gradient_tree(outputs)
    return result


def _target_forward_bundle(
    target_state: EasyDeLState,
    batch: collections.abc.Mapping[str, tp.Any],
    *,
    target_logits_attr: str,
    target_forward_method: str | None,
    target_output_hidden_states: bool,
    pass_target_outputs: bool,
    require_logits: bool = True,
    target_layer_ids: tp.Sequence[int] | None = None,
    target_num_hidden_layers: int | None = None,
) -> dict[str, tp.Any]:
    """Merge the frozen target state and collect stop-gradiented outputs.

    Args:
        target_state: Frozen target ``EasyDeLState``.
        batch: Training batch mapping.
        target_logits_attr: Dotted attribute path to extract target logits.
        target_forward_method: Optional custom forward method name.
        target_output_hidden_states: Whether to request hidden states.
        pass_target_outputs: Whether to include the full raw output in the
            bundle.

    Returns:
        Dictionary containing ``logits``, and optionally ``hidden_states``,
        ``last_hidden_state``, and ``outputs``.
    """
    target_module = target_state.merge(jax.lax.stop_gradient(target_state.graphstate))
    return _target_forward_bundle_from_module(
        target_module,
        batch,
        target_logits_attr=target_logits_attr,
        target_forward_method=target_forward_method,
        target_output_hidden_states=target_output_hidden_states,
        pass_target_outputs=pass_target_outputs,
        require_logits=require_logits,
        target_layer_ids=target_layer_ids,
        target_num_hidden_layers=target_num_hidden_layers,
    )


def _speculative_loss_for_module(
    module: tp.Any,
    batch: collections.abc.Mapping[str, tp.Any],
    target_bundle: collections.abc.Mapping[str, tp.Any],
    *,
    temperature: float,
    alpha: float,
    target_logits_attr: str,
    draft_logits_attr: str | None,
    draft_chain_method: str | None,
    drafter_forward_method: str | None,
    num_draft_tokens: int,
    draft_target_offset: int,
    pass_target_outputs: bool,
) -> tuple[jax.Array, LossMetrics]:
    """Run the drafter against a target bundle and compute the loss.

    Target logits, hidden states, last hidden state, or the full target output
    can be injected into target-conditioned drafters when
    ``pass_target_outputs`` is enabled.  Any drafter ``aux_loss`` is added to
    the speculative objective and surfaced as a metric.

    Args:
        module: The drafter model module.
        batch: Training batch mapping.
        target_bundle: Frozen target outputs mapping.
        temperature: Softmax temperature for KL and acceptance terms.
        alpha: Interpolation weight between KL and CE losses.
        target_logits_attr: Dotted attribute path for target logits (unused).
        draft_logits_attr: Dotted attribute path for drafter logits.
        draft_chain_method: Configured chain method name.
        drafter_forward_method: Optional custom drafter forward method name.
        num_draft_tokens: Number of draft tokens to evaluate.
        draft_target_offset: Offset between drafter and target positions.
        pass_target_outputs: Whether to pass target outputs to the drafter.

    Returns:
        Tuple of ``(loss, metrics)`` where ``loss`` is the scalar speculative
        loss and ``metrics`` is a ``LossMetrics`` object.

    Raises:
        ValueError: Propagated from alignment or normalization helpers if
            shapes are incompatible.
    """
    del target_logits_attr
    target_logits = target_bundle.get("logits")
    module_config = getattr(module, "config", None)
    if int(num_draft_tokens) == 1 and getattr(module_config, "ttt_length", None) is not None:
        num_draft_tokens = int(module_config.ttt_length)
    extra_kwargs: dict[str, tp.Any] = {}
    if "hidden_states" in target_bundle:
        extra_kwargs["target_hidden_states"] = target_bundle["hidden_states"]
    if pass_target_outputs:
        if target_logits is not None:
            extra_kwargs["target_logits"] = target_logits
        if "last_hidden_state" in target_bundle:
            extra_kwargs["target_last_hidden_state"] = target_bundle["last_hidden_state"]
        if "outputs" in target_bundle:
            extra_kwargs["target_outputs"] = target_bundle["outputs"]
    loss_mask = _loss_mask_from_batch(batch)
    if loss_mask is not None and _module_accepts_block_loss_kwargs(module, drafter_forward_method):
        extra_kwargs["loss_mask"] = loss_mask
        extra_kwargs["return_block_outputs"] = True

    drafter_outputs = _call_module_or_method(
        module,
        batch,
        method_name=drafter_forward_method,
        extra_kwargs=extra_kwargs,
    )
    block_loss = _deepspec_block_loss(drafter_outputs, getattr(module, "config", None))
    if block_loss is not None:
        return block_loss
    if target_logits is None:
        raise ValueError("Speculative fallback loss requires target logits; enable target logits for this path.")
    draft_logits = _resolve_draft_logits(
        module,
        drafter_outputs,
        batch,
        target_logits=target_logits,
        draft_logits_attr=draft_logits_attr,
        draft_chain_method=draft_chain_method,
        num_draft_tokens=num_draft_tokens,
    )
    loss, metrics = speculative_draft_loss(
        draft_logits,
        target_logits,
        labels=batch.get("labels"),
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        loss_mask=batch.get("completion_mask"),
        temperature=temperature,
        alpha=alpha,
        num_draft_tokens=num_draft_tokens,
        draft_target_offset=draft_target_offset,
        step_loss_decay=getattr(module_config, "step_loss_decay", None),
    )
    aux_loss = getattr(drafter_outputs, "aux_loss", None)
    if aux_loss is not None:
        aux_loss = aux_loss.astype(loss.dtype)
        loss = loss + aux_loss
        other_metrics = dict(metrics.other_metrics or {})
        other_metrics["aux_loss"] = aux_loss
        metrics = metrics.replace(loss=loss, other_metrics=other_metrics)
    return loss, metrics


def speculative_decoding_step(
    drafter_state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    target_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    temperature: float = 1.0,
    alpha: float = 1.0,
    target_logits_attr: str = "logits",
    draft_logits_attr: str | None = None,
    draft_chain_method: str | None = "auto",
    drafter_forward_method: str | None = None,
    target_forward_method: str | None = None,
    num_draft_tokens: int = 1,
    draft_target_offset: int = 0,
    target_output_hidden_states: bool = False,
    pass_target_outputs: bool = False,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Execute one compiled speculative-decoding train or eval step.

    The drafter state is the differentiable state.  The target state is merged
    inside the loss function and only contributes stop-gradiented teacher
    outputs.  Training uses minibatch gradient accumulation before applying
    optimizer updates; evaluation returns metrics without touching the state.

    Args:
        drafter_state: Differentiable drafter ``EasyDeLState``.
        batch: Training batch mapping of JAX arrays.
        target_state: Frozen target ``EasyDeLState``.
        loss_config: Optional loss configuration for gradient handling.
        learning_rate_fn: Optional learning-rate schedule for metrics.
        partition_spec: Optional sharding partition spec.
        gradient_accumulation_steps: Number of minibatch steps before applying
            gradients.
        is_training: ``True`` for training, ``False`` for evaluation.
        temperature: Softmax temperature for KL and acceptance terms.
        alpha: Interpolation weight between KL and CE losses.
        target_logits_attr: Dotted attribute path for target logits.
        draft_logits_attr: Dotted attribute path for drafter logits.
        draft_chain_method: Configured chain method name.
        drafter_forward_method: Optional custom drafter forward method.
        target_forward_method: Optional custom target forward method.
        num_draft_tokens: Number of draft tokens to evaluate.
        draft_target_offset: Offset between drafter and target positions.
        target_output_hidden_states: Whether to request target hidden states.
        pass_target_outputs: Whether to pass target outputs to the drafter.
        straight_through_emulator: Optional straight-through estimator applied
            to the drafter graphstate during training.

    Returns:
        During training: ``(updated_drafter_state, metrics)``.
        During evaluation: ``metrics`` only.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=drafter_state.model.mesh, ignore_mpmd=True)
    batch = dict(batch)

    def loss_fn(tree, minibatch):
        """Compute loss for a single minibatch.

        Args:
            tree: Drafter graphstate tree (may be emulated).
            minibatch: One minibatch of the training data.

        Returns:
            Tuple of ``(loss, metrics)``.
        """
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = drafter_state.merge(tree)
        loss_mask = _loss_mask_from_batch(minibatch)
        use_native_block_loss = loss_mask is not None and _module_accepts_block_loss_kwargs(
            module, drafter_forward_method
        )
        require_target_logits = (not use_native_block_loss) or _deepspec_block_loss_requires_target_distribution(
            getattr(module, "config", None)
        )
        target_layer_ids = getattr(getattr(module, "config", None), "target_layer_ids", None)
        target_num_hidden_layers = getattr(getattr(module, "config", None), "target_num_hidden_layers", None)
        target_bundle = _target_forward_bundle(
            target_state,
            minibatch,
            target_logits_attr=target_logits_attr,
            target_forward_method=target_forward_method,
            target_output_hidden_states=target_output_hidden_states,
            pass_target_outputs=pass_target_outputs,
            require_logits=require_target_logits,
            target_layer_ids=target_layer_ids,
            target_num_hidden_layers=target_num_hidden_layers,
        )
        return _speculative_loss_for_module(
            module,
            minibatch,
            target_bundle,
            temperature=temperature,
            alpha=alpha,
            target_logits_attr=target_logits_attr,
            draft_logits_attr=draft_logits_attr,
            draft_chain_method=draft_chain_method,
            drafter_forward_method=drafter_forward_method,
            num_draft_tokens=num_draft_tokens,
            draft_target_offset=draft_target_offset,
            pass_target_outputs=pass_target_outputs,
        )

    if is_training:
        gradients, metrics = minibatch_call(
            state=drafter_state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        drafter_state = update_state_respectfully(
            state=drafter_state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=drafter_state.step,
                gradients=gradients,
            ),
        )
        return drafter_state, metrics

    _, metrics = loss_fn(tree=drafter_state.graphstate, minibatch=batch)
    return metrics


def _prepare_speculative_scheduled_batch(call) -> dict[str, tp.Any]:
    """Materialize the SpectraX scheduled-loss batch as a mutable dict.

    Args:
        call: Scheduled loss call object containing ``call.batch``.

    Returns:
        Mutable dictionary copy of the batch.
    """
    return dict(call.batch)


def _speculative_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build the cache key fields that affect scheduled-loss compilation.

    Args:
        call: Scheduled loss call object.

    Returns:
        Tuple cache key built from the call's relevant fields.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "partition_spec",
            "temperature",
            "alpha",
            "target_logits_attr",
            "draft_logits_attr",
            "draft_chain_method",
            "drafter_forward_method",
            "target_forward_method",
            "num_draft_tokens",
            "draft_target_offset",
            "target_output_hidden_states",
            "pass_target_outputs",
        ),
        object_fields=("straight_through_emulator", "target_state"),
    )


def _make_speculative_scheduled_loss(call):
    """Build the scalar loss closure used by SpectraX MPMD scheduling.

    The closure binds the scheduled drafter module, constrains the batch using
    the trainer partition spec, merges the target state under the same schedule,
    and returns only the scalar loss expected by the generic scheduled
    backward/update path.

    Args:
        call: Scheduled loss call object.

    Returns:
        A callable ``scheduled_loss(tree, batch) -> loss`` that SpectraX will
        invoke during scheduled training.
    """
    partition_spec = call.get("partition_spec")
    temperature = call.get("temperature", 1.0)
    alpha = call.get("alpha", 1.0)
    target_logits_attr = call.get("target_logits_attr", "logits")
    draft_logits_attr = call.get("draft_logits_attr")
    draft_chain_method = call.get("draft_chain_method", "auto")
    drafter_forward_method = call.get("drafter_forward_method")
    target_forward_method = call.get("target_forward_method")
    num_draft_tokens = int(call.get("num_draft_tokens", 1))
    draft_target_offset = int(call.get("draft_target_offset", 0))
    target_output_hidden_states = bool(call.get("target_output_hidden_states", False))
    pass_target_outputs = bool(call.get("pass_target_outputs", False))
    target_state = call.get("target_state")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute scalar speculative loss for a scheduled training step.

        Args:
            tree: Scheduled drafter state tree.
            batch: Training batch dictionary.

        Returns:
            Scalar loss value.

        Raises:
            RuntimeError: If ``target_state`` is ``None`` in the bound call.
        """
        if target_state is None:
            raise RuntimeError("Speculative-decoding scheduled MPMD training requires target_state.")
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        target_module = target_state.merge(target_state.graphstate)
        sync_module_schedule_config(target_module, call.schedule)
        loss_mask = _loss_mask_from_batch(call_batch)
        use_native_block_loss = loss_mask is not None and _module_accepts_block_loss_kwargs(
            module, drafter_forward_method
        )
        require_target_logits = (not use_native_block_loss) or _deepspec_block_loss_requires_target_distribution(
            getattr(module, "config", None)
        )
        target_layer_ids = getattr(getattr(module, "config", None), "target_layer_ids", None)
        target_num_hidden_layers = getattr(getattr(module, "config", None), "target_num_hidden_layers", None)
        target_bundle = _target_forward_bundle_from_module(
            target_module,
            call_batch,
            target_logits_attr=target_logits_attr,
            target_forward_method=target_forward_method,
            target_output_hidden_states=target_output_hidden_states,
            pass_target_outputs=pass_target_outputs,
            require_logits=require_target_logits,
            target_layer_ids=target_layer_ids,
            target_num_hidden_layers=target_num_hidden_layers,
        )
        loss, _metrics = _speculative_loss_for_module(
            module,
            call_batch,
            target_bundle,
            temperature=temperature,
            alpha=alpha,
            target_logits_attr=target_logits_attr,
            draft_logits_attr=draft_logits_attr,
            draft_chain_method=draft_chain_method,
            drafter_forward_method=drafter_forward_method,
            num_draft_tokens=num_draft_tokens,
            draft_target_offset=draft_target_offset,
            pass_target_outputs=pass_target_outputs,
        )
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=speculative_decoding_step,
    adapter=ScheduledLossAdapter(
        name="speculative_decoding",
        make_loss=_make_speculative_scheduled_loss,
        make_cache_key=_speculative_scheduled_loss_cache_key,
        prepare_batch=_prepare_speculative_scheduled_batch,
    ),
)
