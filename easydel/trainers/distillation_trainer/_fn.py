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

Additionally supports advanced distillation strategies:
- Attention transfer: Matching attention patterns between teacher and student
- Feature matching: Matching intermediate hidden representations

All functions are designed for JAX/Flax models and support distributed training.
"""

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
from .pooling import avg_pool_array_to_target_shape


def distillation_loss(
    student_logits: chex.Array,
    teacher_logits: chex.Array,
    attention_mask: chex.Array | None = None,
    labels: chex.Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
):
    """Compute knowledge distillation loss between student and teacher models.

    This function implements the distillation loss as described in Hinton et al.'s
    "Distilling the Knowledge in a Neural Network". It combines KL divergence loss
    between temperature-scaled teacher and student distributions with optional
    supervised learning loss on hard labels.

    Args:
        student_logits (chex.Array): Raw logits from the student model.
            Shape: [batch_size, sequence_length, vocab_size]
        teacher_logits (chex.Array): Raw logits from the teacher model.
            Shape: [batch_size, sequence_length, vocab_size]
        attention_mask (chex.Array | None): Mask indicating valid tokens.
            1 for valid tokens, 0 for padding. Shape: [batch_size, sequence_length]
        labels (chex.Array | None): Ground truth labels for supervised loss.
            Shape: [batch_size, sequence_length]
        use_hard_labels (bool): Whether to include supervised loss with hard labels.
            If True, combines distillation loss with cross-entropy loss.
        temperature (float): Temperature for softening probability distributions.
            Higher values create softer distributions. Default: 4.0
        alpha (float): Weight for distillation loss vs supervised loss.
            1.0 means pure distillation, 0.0 means pure supervised. Default: 0.9

    Returns:
        chex.Array: Scalar loss value combining distillation and optional supervised loss.

    Note:
        The loss is properly masked to ignore padding tokens when attention_mask
        is provided. The temperature scaling allows the student to learn from
        the teacher's relative confidence across all classes.
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


def attention_transfer_loss(
    student_attentions: tuple[chex.Array] | None,
    teacher_attentions: tuple[chex.Array] | None,
    match_layers: tuple[int, ...] | None = None,
    attention_mask: chex.Array | None = None,
) -> chex.Array:
    """Compute attention transfer loss between student and teacher attention maps.

    This function implements attention transfer distillation by minimizing the
    cosine distance between student and teacher attention patterns. It helps
    the student learn to focus on similar parts of the input as the teacher.

    Args:
        student_attentions (tuple[chex.Array] | None): Tuple of attention weight tensors
            from student model, one per layer. Shape per layer: [batch, num_heads, seq, seq]
        teacher_attentions (tuple[chex.Array] | None): Tuple of attention weight tensors
            from teacher model, one per layer. Shape per layer: [batch, num_heads, seq, seq]
        match_layers (tuple[int, ...] | None): Indices of layers to match. If None,
            matches all layers. Example: (6, 12, 18) to match layers 6, 12, and 18.
        attention_mask (chex.Array | None): Attention mask for valid positions.
            Shape: [batch_size, sequence_length]

    Returns:
        chex.Array: Scalar attention transfer loss (mean cosine distance across layers).

    Note:
        If teacher and student have different numbers of attention heads or sequence lengths,
        automatic pooling is applied to match dimensions. Cosine distance is used instead
        of MSE to be invariant to the scale of attention weights.
    """
    if student_attentions is None or teacher_attentions is None:
        return jnp.array(0.0)

    # Determine which layers to match
    if match_layers is None:
        # Match all layers up to the minimum number of layers
        num_layers = min(len(student_attentions), len(teacher_attentions))
        layers_to_match = range(num_layers)
    else:
        layers_to_match = match_layers

    total_loss = 0.0
    num_matched_layers = 0

    for layer_idx in layers_to_match:
        # Skip if layer doesn't exist in either model
        if layer_idx >= len(student_attentions) or layer_idx >= len(teacher_attentions):
            continue

        student_attn = student_attentions[layer_idx]
        teacher_attn = teacher_attentions[layer_idx]

        # Pool teacher attention if shapes don't match
        if student_attn.shape != teacher_attn.shape:
            teacher_attn = avg_pool_array_to_target_shape(teacher_attn, student_attn.shape)

        if attention_mask is not None:
            mask = attention_mask.astype(student_attn.dtype)
            key_mask = mask[:, None, None, :]
            student_attn = student_attn * key_mask
            teacher_attn = teacher_attn * key_mask

        # Compute cosine distance (optax.cosine_distance expects last dimension for dot product)
        # Average over batch, heads, and sequence dimensions
        cosine_dist = optax.cosine_distance(student_attn, teacher_attn, axis=-1)

        if attention_mask is not None:
            query_mask = mask[:, None, :]
            denom = jnp.sum(query_mask) * cosine_dist.shape[1]
            layer_loss = jnp.where(
                denom > 0,
                jnp.sum(cosine_dist * query_mask) / denom,
                0.0,
            )
        else:
            layer_loss = jnp.mean(cosine_dist)

        total_loss += layer_loss
        num_matched_layers += 1

    # Return average loss across matched layers
    if num_matched_layers == 0:
        return jnp.array(0.0)

    return total_loss / num_matched_layers


def feature_matching_loss(
    student_hidden_states: tuple[chex.Array] | None,
    teacher_hidden_states: tuple[chex.Array] | None,
    match_layers: tuple[int, ...] | None = None,
    attention_mask: chex.Array | None = None,
) -> chex.Array:
    """Compute feature matching loss between student and teacher hidden states.

    This function implements feature-based distillation by minimizing the mean
    squared error between student and teacher intermediate representations. It
    helps transfer the teacher's internal feature representations to the student.

    Args:
        student_hidden_states (tuple[chex.Array] | None): Tuple of hidden state tensors
            from student model, one per layer. Shape per layer: [batch, seq, hidden_dim]
        teacher_hidden_states (tuple[chex.Array] | None): Tuple of hidden state tensors
            from teacher model, one per layer. Shape per layer: [batch, seq, hidden_dim]
        match_layers (tuple[int, ...] | None): Indices of layers to match. If None,
            matches all layers. Example: (6, 12, 18) to match layers 6, 12, and 18.
        attention_mask (chex.Array | None): Attention mask for valid positions.
            Shape: [batch_size, sequence_length]

    Returns:
        chex.Array: Scalar feature matching loss (mean MSE across layers).

    Note:
        If teacher and student have different hidden dimensions or sequence lengths,
        automatic pooling is applied to resize teacher features to match student dimensions.
        MSE is used to ensure features are similar in both direction and magnitude.
    """
    if student_hidden_states is None or teacher_hidden_states is None:
        return jnp.array(0.0)

    # Determine which layers to match
    if match_layers is None:
        # Match all layers up to the minimum number of layers
        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
        layers_to_match = range(num_layers)
    else:
        layers_to_match = match_layers

    total_loss = 0.0
    num_matched_layers = 0

    for layer_idx in layers_to_match:
        # Skip if layer doesn't exist in either model
        if layer_idx >= len(student_hidden_states) or layer_idx >= len(teacher_hidden_states):
            continue

        student_hidden = student_hidden_states[layer_idx]
        teacher_hidden = teacher_hidden_states[layer_idx]

        # Pool teacher features if shapes don't match
        if student_hidden.shape != teacher_hidden.shape:
            teacher_hidden = avg_pool_array_to_target_shape(teacher_hidden, student_hidden.shape)

        diff = student_hidden - teacher_hidden

        # Compute MSE
        mse = jnp.mean(jnp.square(diff))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match hidden dims: [batch, seq] -> [batch, seq, 1]
            mask = attention_mask.astype(diff.dtype)
            mask_expanded = jnp.expand_dims(mask, axis=-1)
            masked_diff = jnp.square(diff) * mask_expanded
            num_valid_tokens = jnp.sum(mask)
            denom = num_valid_tokens * diff.shape[-1]
            mse = jnp.where(
                denom > 0,
                jnp.sum(masked_diff) / denom,
                0.0,
            )

        total_loss += mse
        num_matched_layers += 1

    # Return average loss across matched layers
    if num_matched_layers == 0:
        return jnp.array(0.0)

    return total_loss / num_matched_layers


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
    use_attention_transfer: bool = False,
    attention_loss_weight: float = 0.1,
    attention_match_layers: tuple[int, ...] | None = None,
    use_feature_matching: bool = False,
    feature_loss_weight: float = 0.1,
    feature_match_layers: tuple[int, ...] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Performs a single distillation training or evaluation step.

    Args:
        student_state: State of the student model being trained.
        batch: Batch of training data.
        teacher_state: State of the teacher model (frozen).
        loss_config: Configuration for loss computation.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for distributed training.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        is_training: Whether this is a training step (vs evaluation).
        temperature: Temperature for logit distillation.
        alpha: Weight for logit distillation loss vs supervised loss.
        use_attention_transfer: Whether to use attention transfer loss.
        attention_loss_weight: Weight for attention transfer loss.
        attention_match_layers: Layer indices to match for attention transfer.
        use_feature_matching: Whether to use feature matching loss.
        feature_loss_weight: Weight for feature matching loss.
        feature_match_layers: Layer indices to match for feature matching.

    Returns:
        Updated student state and loss metrics (if training), or just metrics (if eval).
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        module = flax.nnx.merge(student_state.graphdef, tree, student_state.graphother)

        # Get outputs with hidden states and attentions if needed
        output_hidden_states = use_feature_matching
        output_attentions = use_attention_transfer

        student_outputs = module(
            **minibatch,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        teacher_outputs = teacher_state.model(
            **minibatch,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)

        # Compute standard logit distillation loss
        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            attention_mask=attention_mask,
            labels=labels,
            use_hard_labels=(labels is not None),
            temperature=temperature,
            alpha=alpha,
        )

        # Add attention transfer loss if enabled
        if use_attention_transfer:
            attn_loss = attention_transfer_loss(
                student_attentions=student_outputs.attentions,
                teacher_attentions=teacher_outputs.attentions,
                match_layers=attention_match_layers,
                attention_mask=attention_mask,
            )
            loss = loss + attention_loss_weight * attn_loss

        # Add feature matching loss if enabled
        if use_feature_matching:
            feat_loss = feature_matching_loss(
                student_hidden_states=student_outputs.hidden_states,
                teacher_hidden_states=teacher_outputs.hidden_states,
                match_layers=feature_match_layers,
                attention_mask=attention_mask,
            )
            loss = loss + feature_loss_weight * feat_loss

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
