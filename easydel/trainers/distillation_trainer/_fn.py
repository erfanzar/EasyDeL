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

"""Internal functions for knowledge distillation training.

This module contains the core computational functions used by the distillation trainer,
including loss functions and training/evaluation step implementations. These functions
implement knowledge distillation as described by Hinton et al., where a student model
learns to mimic a teacher model's output distributions.

The distillation process uses temperature scaling to soften probability distributions,
allowing the student to learn from the teacher's confidence across all classes rather
than just the hard labels. The loss combines KL divergence between teacher and student
distributions with optional supervised learning loss.

All functions are designed for JAX/Flax models and support distributed training.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from eformer.escale import with_sharding_constraint
from jax import Array as JaxArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    sanitize_model_call_kwargs,
    update_metrics,
    update_state_respectfully,
)


def _compute_kl_and_ce(
    student_logits: Array,
    teacher_logits: Array,
    mask: Array,
    safe_labels: Array,
    use_hard_labels: bool,
    temperature: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array]:
    """Per-token KL and CE sums for one chunk of logits.

    Returns (kl_sum, ce_sum, mask_sum) — all scalar accumulators.
    """
    teacher_logits = jax.lax.stop_gradient(teacher_logits)
    t_probs = jax.nn.softmax(teacher_logits.astype(jnp.float32) / temperature, axis=-1)
    s_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32) / temperature, axis=-1)
    per_token_kl = -jnp.sum(t_probs * s_log_probs, axis=-1).astype(dtype)

    kl_sum = jnp.sum(per_token_kl * mask)
    mask_sum = jnp.sum(mask)

    ce_sum = jnp.zeros((), dtype=dtype)
    if use_hard_labels:
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        ).astype(dtype)
        ce_sum = jnp.sum(per_token_ce * mask)

    return kl_sum, ce_sum, mask_sum


def _finalize_distillation_metrics(
    kl_sum: Array,
    ce_sum: Array,
    mask_sum: Array,
    temperature: float,
    alpha: float,
    use_hard_labels: bool,
    dtype: jnp.dtype,
) -> tuple[Array, dict[str, Array]]:
    """Normalise accumulated KL/CE sums into the final scalar loss."""
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)
    normalizer = jnp.maximum(mask_sum, jnp.ones((), dtype=dtype))

    kl_loss = (kl_sum / normalizer) * temp_sq
    total_loss = alpha_s * kl_loss

    ce_loss = jnp.zeros((), dtype=dtype)
    if use_hard_labels:
        ce_loss = ce_sum / normalizer
        total_loss = total_loss + (jnp.ones((), dtype=dtype) - alpha_s) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


def _build_mask_and_labels(
    attention_mask: Array | None,
    loss_mask: Array | None,
    labels: Array | None,
    dtype: jnp.dtype,
    seq_len: int,
    batch_size: int,
) -> tuple[Array, Array, bool]:
    """Build a combined per-token mask and safe labels array.

    Returns (mask, safe_labels, has_labels).
    """
    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = jnp.ones((batch_size, seq_len), dtype=dtype)

    has_labels = labels is not None
    if has_labels:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = mask * valid_label_mask
        safe_labels = jnp.where(labels == -100, 0, labels)
    else:
        safe_labels = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    return mask, safe_labels, has_labels


def distillation_loss(
    student_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> tuple[Array, dict[str, Array]]:
    """Compute knowledge distillation loss between student and teacher models.

    This function implements the distillation loss as described in Hinton et al.'s
    "Distilling the Knowledge in a Neural Network". It combines KL divergence loss
    between temperature-scaled teacher and student distributions with optional
    supervised learning loss on hard labels.

    Args:
        student_logits (Array): Raw logits from the student model.
            Shape: [batch_size, sequence_length, vocab_size]
        teacher_logits (Array): Raw logits from the teacher model.
            Shape: [batch_size, sequence_length, vocab_size]
        attention_mask (Array | None): Mask indicating valid tokens.
            1 for valid tokens, 0 for padding. Shape: [batch_size, sequence_length]
        loss_mask (Array | None): Optional task-specific token mask used for loss
            computation. When provided, this takes priority over attention_mask.
            Useful for assistant-only objectives where prompt tokens are masked out.
        labels (Array | None): Ground truth labels for supervised loss.
            Shape: [batch_size, sequence_length]
        use_hard_labels (bool): Whether to include supervised loss with hard labels.
            If True, combines distillation loss with cross-entropy loss.
        temperature (float): Temperature for softening probability distributions.
            Higher values create softer distributions. Default: 4.0
        alpha (float): Weight for distillation loss vs supervised loss.
            1.0 means pure distillation, 0.0 means pure supervised. Default: 0.9

    Returns:
        tuple[Array, dict[str, Array]]: Scalar loss value combining distillation
        and optional supervised loss together with the individual components.

    Note:
        The loss is properly masked to ignore padding tokens when attention_mask
        is provided. The temperature scaling allows the student to learn from
        the teacher's relative confidence across all classes.
    """
    dtype = student_logits.dtype
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)

    # softmax / log_softmax need f32 for numerical stability over large vocab.
    # teacher_logits already has stop_gradient so f32 has no backward cost.
    # For student_logits, the explicit .astype(f32) means JAX's AD will
    # produce bf16 gradients (matching the primal dtype of student_logits),
    # so f32 stays contained in this loss function and does NOT leak into
    # the model's backward pass.
    teacher_probs = jax.nn.softmax(teacher_logits.astype(jnp.float32) / temperature, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32) / temperature, axis=-1)
    per_token_kl = -jnp.sum(teacher_probs * student_log_probs, axis=-1).astype(dtype)

    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = None

    if labels is not None:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = valid_label_mask if mask is None else mask * valid_label_mask

    if mask is not None:
        masked_kl = per_token_kl * mask
        normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
        kl_loss = jnp.sum(masked_kl) / normalizer
    else:
        kl_loss = jnp.mean(per_token_kl)
    kl_loss = kl_loss * temp_sq
    total_loss = alpha_s * kl_loss
    ce_loss = jnp.array(0.0, dtype=dtype)
    if use_hard_labels and labels is not None:
        safe_labels = jnp.where(labels == -100, 0, labels)
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        ).astype(dtype)

        if mask is not None:
            ce_loss = per_token_ce * mask
            normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
            ce_loss = jnp.sum(ce_loss) / normalizer
        else:
            ce_loss = jnp.mean(per_token_ce)

        total_loss = total_loss + (jnp.array(1.0, dtype=dtype) - alpha_s) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


def chunked_distillation_loss(
    student_hidden: Array,
    teacher_hidden: Array,
    student_lm_head_fn: tp.Callable[[Array], Array],
    teacher_lm_head_fn: tp.Callable[[Array], Array],
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
    chunk_size: int = 128,
) -> tuple[Array, dict[str, Array]]:
    """Memory-efficient distillation loss that avoids materialising full logits.

    Instead of receiving pre-computed ``[B, L, V]`` logits, this function takes
    the last hidden states from both models and their lm_head projection
    functions.  It processes the sequence in chunks of *chunk_size* tokens,
    projecting each chunk to vocab logits on-the-fly and immediately reducing
    to scalar KL / CE contributions.  Peak logit memory drops from
    ``O(B * L * V)`` to ``O(B * chunk_size * V)``.

    The scan body is wrapped in ``jax.checkpoint`` so that during the backward
    pass each chunk's logits are *recomputed* from the hidden states rather
    than stored, keeping memory constant regardless of sequence length.
    """
    dtype = student_hidden.dtype
    B, L = student_hidden.shape[:2]

    # Pad sequence length to a multiple of chunk_size.
    pad_len = (-L) % chunk_size
    if pad_len:
        student_hidden = jnp.pad(student_hidden, ((0, 0), (0, pad_len), (0, 0)))
        teacher_hidden = jnp.pad(teacher_hidden, ((0, 0), (0, pad_len), (0, 0)))
        if attention_mask is not None:
            attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_len)))
        if loss_mask is not None:
            loss_mask = jnp.pad(loss_mask, ((0, 0), (0, pad_len)))
        if labels is not None:
            labels = jnp.pad(labels, ((0, 0), (0, pad_len)), constant_values=-100)

    L_padded = L + pad_len
    n_chunks = L_padded // chunk_size

    mask, safe_labels, has_labels = _build_mask_and_labels(
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        dtype=dtype,
        seq_len=L_padded,
        batch_size=B,
    )

    # Reshape to [n_chunks, B, chunk_size, ...] for scanning.
    s_chunks = student_hidden.reshape(B, n_chunks, chunk_size, -1).transpose(1, 0, 2, 3)
    t_chunks = teacher_hidden.reshape(B, n_chunks, chunk_size, -1).transpose(1, 0, 2, 3)
    m_chunks = mask.reshape(B, n_chunks, chunk_size).transpose(1, 0, 2)
    l_chunks = safe_labels.reshape(B, n_chunks, chunk_size).transpose(1, 0, 2)

    _use_hard = use_hard_labels and has_labels

    @jax.checkpoint
    def _chunk_kl_ce(s_h, t_h, m, sl):
        s_logits = student_lm_head_fn(s_h)
        t_logits = teacher_lm_head_fn(t_h)
        return _compute_kl_and_ce(
            student_logits=s_logits,
            teacher_logits=t_logits,
            mask=m,
            safe_labels=sl,
            use_hard_labels=_use_hard,
            temperature=temperature,
            dtype=dtype,
        )

    def _scan_body(carry, xs):
        s_h, t_h, m, sl = xs
        kl, ce, ms = _chunk_kl_ce(s_h, t_h, m, sl)
        return (carry[0] + kl, carry[1] + ce, carry[2] + ms), None

    _zero = jnp.zeros((), dtype=dtype)
    (kl_sum, ce_sum, mask_sum), _ = jax.lax.scan(
        _scan_body,
        (_zero, _zero, _zero),
        (s_chunks, t_chunks, m_chunks, l_chunks),
    )

    return _finalize_distillation_metrics(
        kl_sum=kl_sum,
        ce_sum=ce_sum,
        mask_sum=mask_sum,
        temperature=temperature,
        alpha=alpha,
        use_hard_labels=_use_hard,
        dtype=dtype,
    )


def _resolve_indices(
    collection_length: int,
    indices: tuple[int, ...] | None,
    *,
    default_all: bool,
) -> tuple[int, ...]:
    if collection_length == 0:
        raise ValueError("Cannot select layers from an empty collection.")
    if not indices:
        if default_all:
            return tuple(range(collection_length))
        return (collection_length - 1,)
    resolved: list[int] = []
    for idx in indices:
        resolved_idx = idx if idx >= 0 else collection_length + idx
        if resolved_idx < 0 or resolved_idx >= collection_length:
            raise IndexError(f"Layer index {idx} is out of range for collection of length {collection_length}.")
        resolved.append(int(resolved_idx))
    return tuple(resolved)


def _masked_mse(values: jax.Array, targets: jax.Array, mask: jax.Array | None) -> jax.Array:
    if values.shape != targets.shape:
        raise ValueError(f"Mismatched tensor shapes for distillation: {values.shape} vs {targets.shape}.")
    diff = values - targets
    if mask is not None:
        mask = mask.astype(diff.dtype)
        while mask.ndim < diff.ndim:
            mask = mask[..., None]
        diff = diff * mask
        denom = jnp.maximum(mask.sum(), jnp.array(1.0, dtype=diff.dtype))
    else:
        denom = jnp.array(diff.size, dtype=diff.dtype)
    return jnp.sum(diff * diff) / denom


def _build_attention_mask(attention_mask: jax.Array | None, *, dtype: jnp.dtype) -> jax.Array | None:
    if attention_mask is None:
        return None
    mask = attention_mask.astype(dtype)
    return mask[:, None, :, None] * mask[:, None, None, :]


def _normalize_attention(tensor: jax.Array) -> jax.Array:
    denom = jnp.sum(tensor, axis=-1, keepdims=True)
    denom = jnp.maximum(denom, jnp.finfo(tensor.dtype).tiny)
    return tensor / denom


def _stop_gradient_tree(tree):
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x) if isinstance(x, JaxArray) else x, tree)


def distillation_step(
    student_state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    temperature: float = 4.0,
    alpha: float = 0.9,
    hidden_state_weight: float = 0.0,
    hidden_state_layers: tuple[int, ...] | None = None,
    hidden_state_loss: tp.Literal["mse"] = "mse",
    attention_weight: float = 0.0,
    attention_layers: tuple[int, ...] | None = None,
    attention_normalize: bool = False,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    logits_chunk_size: int = 0,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    if hidden_state_loss != "mse":
        raise ValueError(f"Unsupported hidden state loss '{hidden_state_loss}'. Only 'mse' is available.")

    request_hidden_states = hidden_state_weight != 0.0
    request_attentions = attention_weight != 0.0
    use_chunked = logits_chunk_size > 0

    def loss_fn(tree, minibatch):
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = student_state.merge(tree)

        # --- Teacher forward (per-minibatch, checkpointed) ---
        # Running teacher inside loss_fn ensures its intermediate activations
        # are created and freed per-minibatch rather than persisting across the
        # entire student forward+backward.  jax.checkpoint tells XLA it may
        # free ALL layer-internal activations once the output dict is produced.
        # Combined with stop_gradient (applied inside), no recomputation ever
        # occurs during backward — this purely reduces peak memory.
        teacher_kwargs = dict(minibatch)
        teacher_kwargs.pop("labels", None)
        teacher_kwargs.pop("completion_mask", None)
        teacher_kwargs.pop("assistant_masks", None)
        if use_chunked:
            teacher_kwargs["apply_lm_head"] = False
        if request_hidden_states:
            teacher_kwargs["output_hidden_states"] = True
        if request_attentions:
            teacher_kwargs["output_attentions"] = True
        teacher_kwargs = filter_kwargs_for_callable(teacher_state.model.__call__, teacher_kwargs)
        teacher_kwargs = sanitize_model_call_kwargs(teacher_kwargs)

        @jax.checkpoint
        def _teacher_fwd(kw):
            out = teacher_state.model(**kw)
            # Only return tensors we actually need — everything else is freed
            # by the checkpoint boundary (no backward since stop_gradient).
            results = {}
            if use_chunked:
                results["h"] = jax.lax.stop_gradient(out.last_hidden_state)
            else:
                results["l"] = jax.lax.stop_gradient(out.logits)
            if request_hidden_states:
                hs = getattr(out, "hidden_states", None)
                if hs is not None:
                    results["hs"] = jax.lax.stop_gradient(jnp.stack(hs, axis=1))
            if request_attentions:
                att = getattr(out, "attentions", None)
                if att is not None:
                    results["att"] = jax.lax.stop_gradient(jnp.stack(att, axis=1))
            return results

        teacher_out = _teacher_fwd(teacher_kwargs)
        if use_chunked:
            teacher_hidden_for_kl = teacher_out["h"]
        else:
            teacher_logits = teacher_out["l"]
        teacher_hidden_stacked = teacher_out.get("hs")
        teacher_attentions_stacked = teacher_out.get("att")

        # --- Student forward ---
        call_kwargs = dict(minibatch)
        call_kwargs.pop("labels", None)
        call_kwargs.pop("completion_mask", None)
        call_kwargs.pop("assistant_masks", None)
        if use_chunked:
            call_kwargs["apply_lm_head"] = False
        if request_hidden_states:
            call_kwargs["output_hidden_states"] = True
        if request_attentions:
            call_kwargs["output_attentions"] = True
        call_kwargs = filter_kwargs_for_callable(module.__call__, call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        student_outputs = module(**call_kwargs)
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)
        completion_mask = minibatch.get("completion_mask", None)

        if use_chunked:
            total_loss, loss_components = chunked_distillation_loss(
                student_hidden=student_outputs.last_hidden_state,
                teacher_hidden=teacher_hidden_for_kl,
                student_lm_head_fn=module.apply_lm_head,
                teacher_lm_head_fn=teacher_state.model.apply_lm_head,
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
                chunk_size=logits_chunk_size,
            )
        else:
            total_loss, loss_components = distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_logits,
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
            )
        metrics_map: dict[str, jax.Array] = dict(loss_components)

        if request_hidden_states:
            student_hidden = getattr(student_outputs, "hidden_states", None)
            if student_hidden is None or teacher_hidden_stacked is None:
                raise ValueError(
                    "Hidden-state distillation requested but models did not return hidden states. "
                    "Please ensure `output_hidden_states` is supported."
                )
            teacher_hidden = [teacher_hidden_stacked[:, i] for i in range(teacher_hidden_stacked.shape[1])]
            student_indices = _resolve_indices(len(student_hidden), hidden_state_layers, default_all=False)
            teacher_indices = _resolve_indices(len(teacher_hidden), hidden_state_layers, default_all=False)
            if len(student_indices) != len(teacher_indices):
                raise ValueError(
                    "Hidden-state layer selections for student and teacher have different lengths. "
                    "Please align the requested layers across both models."
                )
            hidden_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                hidden_losses.append(_masked_mse(student_hidden[s_idx], teacher_hidden[t_idx], attention_mask))
            hidden_loss_value = jnp.mean(jnp.stack(hidden_losses))
            hidden_loss_value = hidden_loss_value.astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(hidden_state_weight, dtype=total_loss.dtype) * hidden_loss_value
            metrics_map["hidden_state_loss"] = hidden_loss_value

        if request_attentions:
            student_attentions = getattr(student_outputs, "attentions", None)
            if student_attentions is None or teacher_attentions_stacked is None:
                raise ValueError(
                    "Attention distillation requested but models did not return attention probabilities. "
                    "Please ensure `output_attentions` is supported."
                )
            teacher_attentions = [teacher_attentions_stacked[:, i] for i in range(teacher_attentions_stacked.shape[1])]
            student_indices = _resolve_indices(len(student_attentions), attention_layers, default_all=True)
            teacher_indices = _resolve_indices(len(teacher_attentions), attention_layers, default_all=True)
            if len(student_indices) != len(teacher_indices):
                raise ValueError(
                    "Attention layer selections for student and teacher have different lengths. "
                    "Please align the requested layers across both models."
                )
            attn_mask = _build_attention_mask(attention_mask, dtype=total_loss.dtype)
            attention_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                s_attn = student_attentions[s_idx]
                t_attn = teacher_attentions[t_idx]
                if attention_normalize:
                    s_attn = _normalize_attention(s_attn)
                    t_attn = _normalize_attention(t_attn)
                attention_losses.append(_masked_mse(s_attn, t_attn, attn_mask))
            attention_loss_value = jnp.mean(jnp.stack(attention_losses))
            attention_loss_value = attention_loss_value.astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(attention_weight, dtype=total_loss.dtype) * attention_loss_value
            metrics_map["attention_loss"] = attention_loss_value

        metrics = LossMetrics(
            loss=total_loss,
            other_metrics={key: jnp.asarray(value) for key, value in metrics_map.items()},
        )
        return total_loss, metrics

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
