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

import typing as tp

import flax
import flax.nnx
import jax
import optax
from eformer.escale import with_sharding_constraint
from jax import Array as JaxArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import make_assertions_and_get_sizes, minibatch_call, update_metrics, update_state_respectfully


def distillation_loss(
    student_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
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
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    per_token_kl = -jnp.sum(teacher_probs * student_log_probs, axis=-1)

    if attention_mask is not None:
        mask = attention_mask.astype(dtype)
        masked_kl = per_token_kl * mask
        normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
        kl_loss = jnp.sum(masked_kl) / normalizer
    else:
        kl_loss = jnp.mean(per_token_kl)
    kl_loss = kl_loss * (temperature**2)
    total_loss = alpha * kl_loss
    ce_loss = jnp.array(0.0, dtype=dtype)
    if use_hard_labels and labels is not None:
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(student_logits, labels)

        if attention_mask is not None:
            mask = attention_mask.astype(dtype)
            ce_loss = ce_loss * mask
            normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
            ce_loss = jnp.sum(ce_loss) / normalizer
        else:
            ce_loss = jnp.mean(ce_loss)

        total_loss += (1 - alpha) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


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
    diff = (values - targets).astype(jnp.float32)
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
    batch: tp.Mapping[str, jax.Array],
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
) -> tuple[EasyDeLState, LossMetrics]:
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    if hidden_state_loss != "mse":
        raise ValueError(f"Unsupported hidden state loss '{hidden_state_loss}'. Only 'mse' is available.")

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(student_state.graphdef, tree, student_state.graphother)
        request_hidden_states = hidden_state_weight != 0.0
        request_attentions = attention_weight != 0.0
        call_kwargs = dict(minibatch)
        call_kwargs.pop("labels", None)
        if request_hidden_states:
            call_kwargs["output_hidden_states"] = True
        if request_attentions:
            call_kwargs["output_attentions"] = True
        student_outputs = module(**call_kwargs)
        teacher_outputs = teacher_state.model(**call_kwargs)
        teacher_outputs = _stop_gradient_tree(teacher_outputs)
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)
        total_loss, loss_components = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
            labels=labels,
            use_hard_labels=(labels is not None),
            temperature=temperature,
            alpha=alpha,
        )
        metrics_map: dict[str, jax.Array] = dict(loss_components)

        if request_hidden_states:
            student_hidden = getattr(student_outputs, "hidden_states", None)
            teacher_hidden = getattr(teacher_outputs, "hidden_states", None)
            if student_hidden is None or teacher_hidden is None:
                raise ValueError(
                    "Hidden-state distillation requested but models did not return hidden states. "
                    "Please ensure `output_hidden_states` is supported."
                )
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
            teacher_attentions = getattr(teacher_outputs, "attentions", None)
            if student_attentions is None or teacher_attentions is None:
                raise ValueError(
                    "Attention distillation requested but models did not return attention probabilities. "
                    "Please ensure `output_attentions` is supported."
                )
            student_indices = _resolve_indices(len(student_attentions), attention_layers, default_all=True)
            teacher_indices = _resolve_indices(len(teacher_attentions), attention_layers, default_all=True)
            if len(student_indices) != len(teacher_indices):
                raise ValueError(
                    "Attention layer selections for student and teacher have different lengths. "
                    "Please align the requested layers across both models."
                )
            attn_mask = _build_attention_mask(attention_mask, dtype=jnp.float32)
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
