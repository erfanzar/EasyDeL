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

"""Loss computation utilities for EasyDeL models.

Provides configurable loss functions for various tasks including language modeling,
sequence classification, and custom objectives. Supports advanced features like
label smoothing, z-loss regularization, and flexible normalization strategies.

Classes:
    SpecialLossNormalizingFactor: Enum for dynamic loss normalization
    LossConfig: Configuration for loss computation behavior
    LossMetrics: Container for loss metrics and auxiliary information
    ForCausalLMLoss: Loss function for causal language modeling
    ForSequenceClassificationLoss: Loss function for sequence classification

Key Features:
    - Flexible loss normalization strategies
    - Label smoothing and z-loss regularization
    - Support for weighted losses
    - NaN detection and handling
    - Efficient JAX implementations

Example:
    >>> from easydel.infra import LossConfig
    >>> config = LossConfig(
    ...     label_smoothing=0.1,
    ...     z_loss=1e-4,
    ...     loss_normalizing_factor="NUM_REAL_TARGET_TOKENS"
    ... )
    >>> loss_fn = ForCausalLMLoss(config)
    >>> loss, metrics = loss_fn(
    ...     logits=model_output,
    ...     labels=targets
    ... )
"""

import enum
import typing as tp
from dataclasses import fields
from functools import reduce
from operator import mul

import chex
import flax
import flax.struct
import jax
import jax.numpy as jnp
from eformer.escale import PartitionAxis
from eformer.escale.partition.constraints import with_sharding_constraint
from eformer.pytree import auto_pytree
from jax import lax
from jax.sharding import PartitionSpec

from easydel.utils.compiling_utils import hash_fn


@enum.unique
class SpecialLossNormalizingFactor(enum.Enum):
    """
    Specifies special, dynamically calculated loss normalizing factors.

    These enums are used in loss functions to indicate how the loss should be
    normalized based on properties of the input batch, rather than using a fixed
    constant.

    Attributes:
        NO_WEIGHT_NUM_REAL_TARGET_TOKENS: Divides the loss by the number of non-padding target tokens,
            ignoring any provided loss weights.
        NUM_REAL_TARGET_TOKENS: Divides the loss by the number of non-padding target tokens,
            considering provided loss weights.
        NUM_TOTAL_TARGET_TOKENS: Divides the loss by the total number of target tokens, including padding.
        AVERAGE_PER_SEQUENCE: Computes the average loss per sequence in the batch.
    """

    NO_WEIGHT_NUM_REAL_TARGET_TOKENS = 0
    NUM_REAL_TARGET_TOKENS = 1
    NUM_TOTAL_TARGET_TOKENS = 2
    AVERAGE_PER_SEQUENCE = 3


SLNF = SpecialLossNormalizingFactor


FACTOR_TYPE = tp.Optional[float | int | str | SLNF]  # noqa


@auto_pytree
class LossConfig:
    """
    Configuration class for customizing loss computation behavior.

    Attributes:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the loss.
            Defaults to -100.
        label_smoothing (float): Amount of label smoothing to apply. 0.0 means no smoothing.
            Defaults to 0.0.
        z_loss (float): Coefficient for the z-loss regularization term, which encourages logits
            for non-target classes to be small. Defaults to 0.0.
        loss_normalizing_factor (FACTOR_TYPE): How to normalize the loss. Can be a constant float/int,
            a string representation of a `SpecialLossNormalizingFactor` enum, or the enum itself.
            Defaults to "NUM_REAL_TARGET_TOKENS".
        num_labels (tp.Optional[int]): The number of labels for classification tasks. Used in
            `ForSequenceClassificationLoss`. Defaults to None.
        problem_type (tp.Optional[str]): Specifies the problem type for sequence classification
            (e.g., "single_label_classification", "multi_label_classification").
            Defaults to None.
        divide_weight_sum (bool): If True, divides the loss by the sum of weights, in addition to
            the `loss_normalizing_factor`. Defaults to False.
        shift_tokens (bool): If True (typically for Causal LM), shifts the logits and labels
            so that the model predicts the next token. Defaults to True.
        break_on_nan (bool): If True, raises an `EasyDeLBreakRequest` if a NaN is encountered
            during loss computation. Defaults to True.
        reduction (tp.Optional[tp.Literal["none", "mean", "sum"]]): Specifies the reduction to apply
            to the loss. If None, the default reduction of the specific loss function is used.
            Defaults to None.
        num_classification_labels (tp.Optional[int]): Number of labels specifically for sequence
            classification. Alias for `num_labels`. Defaults to None.
        classification_problem_type (tp.Optional[tp.Literal["regression", "single_label_classification",
                "multi_label_classification"]]):
            Problem type specifically for sequence classification. Alias for `problem_type`.
            Defaults to None.
    """

    ignore_index: int = -100
    label_smoothing: float = 0.0
    z_loss: float = 0.0
    loss_normalizing_factor: FACTOR_TYPE = "NUM_REAL_TARGET_TOKENS"
    num_labels: str | None = None
    problem_type: str | None = None
    divide_weight_sum: bool = False
    shift_tokens: bool = True
    break_on_nan: bool = True
    reduction: tp.Literal["none", "mean", "sum"] | None = None
    num_classification_labels: int | None = None
    classification_problem_type: (
        tp.Literal["regression", "single_label_classification", "multi_label_classification"] | None
    ) = None

    def __post_init__(self):
        if self.z_loss <= 0:
            self.loss_normalizing_factor = SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS

    def __repr__(self):
        cls_name = self.__class__.__name__
        field_lines = [f"    {f.name}: {getattr(self, f.name)!r}".replace("\n", "\n    ") for f in fields(self)]
        return f"{cls_name}(\n" + "\n".join(field_lines) + "\n)"

    __str__ = __repr__
    __hash__ = hash_fn


@auto_pytree
class LossMetrics:
    """
    Container for various metrics related to loss computation and model training.

    Attributes:
        loss (tp.Optional[tp.Union[float, chex.Array]]): The primary computed loss value.
        z_loss (tp.Optional[tp.Union[float, chex.Array]]): The computed z-loss regularization term.
        weight_sum (tp.Optional[tp.Union[float, chex.Array]]): The sum of weights used in the loss calculation.
        accuracy (tp.Optional[tp.Union[float, chex.Array]]): Computed accuracy, if applicable.
        learning_rate (tp.Optional[tp.Union[float, chex.Array]]): The learning rate used for the current step.
        max_grad_norm (tp.Optional[flax.struct.PyTreeNode]): The maximum gradient norm observed.
        mean_grad_norm (tp.Optional[flax.struct.PyTreeNode]): The mean gradient norm observed.
        grad_norms (tp.Optional[flax.struct.PyTreeNode]): A pytree containing the norms of gradients for each parameter.
        chosen_rewards (tp.Optional[jax.Array]): Rewards for the chosen sequence in preference-based tasks.
        rejected_rewards (tp.Optional[jax.Array]): Rewards for the rejected sequence in preference-based tasks.
        other_metrics (tp.Optional[tp.Mapping[str, jax.Array]]): A dictionary for any additional custom metrics.
        execution_time (tp.Optional[float]): Time taken for the computation step.
    """

    loss: float | chex.Array | None = None
    z_loss: float | chex.Array | None = None
    weight_sum: float | chex.Array | None = None
    accuracy: float | chex.Array | None = None
    learning_rate: float | chex.Array | None = None
    max_grad_norm: flax.struct.PyTreeNode | None = None
    mean_grad_norm: flax.struct.PyTreeNode | None = None
    grad_norms: flax.struct.PyTreeNode | None = None
    chosen_rewards: jax.Array | None = None
    rejected_rewards: jax.Array | None = None
    other_metrics: tp.Mapping[str, jax.Array] | None = None
    execution_time: float | None = None


def dynamic_cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weight: jnp.ndarray | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the cross-entropy loss with optional label smoothing and ignore index,
    dynamically handling different reduction types.

    Args:
        logits (jnp.ndarray): The predicted logits from the model (batch_size, ..., num_classes).
        targets (jnp.ndarray): The target labels (batch_size, ...).
        weight (tp.Optional[jnp.ndarray]): Optional weights for each element (batch_size, ...).
            Defaults to None.
        ignore_index (int): Index in the target labels to ignore. Defaults to -100.
        reduction (str): Specifies the reduction method: 'mean', 'sum', or 'none'.
            Defaults to "mean".
        label_smoothing (float): The amount of label smoothing to apply (0.0 means no smoothing).
            Defaults to 0.0.

    Returns:
        tp.Tuple[jnp.ndarray, jnp.ndarray]:
            - The computed loss (scalar if reduction is 'mean' or 'sum', array otherwise).
            - The normalization factor (sum of weights or count of non-ignored elements).

    Raises:
        ValueError: If an invalid reduction method is specified.
    """
    num_classes = logits.shape[-1]
    targets = jax.nn.one_hot(targets, num_classes=num_classes)

    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + label_smoothing / num_classes

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1)

    # Create a mask for the ignore index
    ignore_mask = targets[..., ignore_index] == 1

    # Apply ignore_mask if ignore_index is used
    if ignore_index >= 0:
        loss = jnp.where(ignore_mask, 0.0, loss)

    # Apply weights if provided
    if weight is not None:
        loss = loss * weight
        # Adjust ignore_mask to consider weights as well
        ignore_mask = ignore_mask | (weight == 0)

    # Calculate the normalization factor (number of non-ignored elements or sum of weights)
    if weight is None:
        normalization_factor = jnp.sum(jnp.logical_not(ignore_mask))
    else:
        normalization_factor = jnp.sum(weight * jnp.logical_not(ignore_mask))

    # Apply reduction
    if reduction == "mean":
        loss = jnp.sum(loss) / jnp.maximum(normalization_factor, 1e-8)  # Avoid division by zero
    elif reduction == "sum":
        loss = jnp.sum(loss)
    elif reduction == "none":
        # Keep the loss per element
        pass
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")

    return loss, normalization_factor


def sigmoid_cross_entropy_with_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    label_smoothing: float = 0.0,
    axis: int | tuple | None = None,
) -> jnp.ndarray:
    """
    Computes sigmoid cross-entropy loss between logits and labels.

    Measures the probability error in discrete classification tasks in which each
    class is independent and not mutually exclusive. For instance, one could
    perform multilabel classification where a picture can contain both an elephant
    and a dog at the same time.

    Args:
        logits: The predicted logits from the model.
        labels: The target labels.
        weights (tp.Optional[jnp.ndarray]): Optional weights for the loss computation.
            Defaults to None.
        label_smoothing (float): Amount of label smoothing to apply (0.0 means no smoothing).
            Defaults to 0.0.
        axis (tp.Optional[tp.Union[int, tuple]]): The axis or axes along which to compute the mean.
            If None, the mean is computed over all elements. Defaults to None.

    Returns:
        jnp.ndarray: The computed sigmoid cross-entropy loss.
    """
    if label_smoothing > 0.0:
        labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    loss = -labels * log_p - (1.0 - labels) * log_not_p

    if weights is not None:
        loss *= weights

    if axis is None:
        return jnp.mean(loss)
    else:
        return jnp.mean(loss, axis=axis)


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    """
    Creates one-hot encoded versions of integer labels.

    Args:
        labels (jnp.ndarray): An array of integer labels.
        num_classes (int): The total number of classes.
        on_value (float): The value to use for the "on" state (corresponding to the label).
            Defaults to 1.0.
        off_value (float): The value to use for the "off" states.
            Defaults to 0.0.

    Returns:
        jnp.ndarray: The one-hot encoded array with shape `labels.shape + (num_classes,)`.
    """
    x = lax.eq(labels[..., None], jnp.arange(num_classes)[(None,) * labels.ndim])
    y = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return y.astype(jnp.float32)


@jax.custom_vjp
def cross_entropy_with_logits(
    logits: chex.Array,
    targets: chex.Array,
    z_loss: float,
) -> tuple[chex.Array, chex.Array]:
    """
    Computes cross-entropy loss with potential z-loss regularization.

    This function calculates the standard cross-entropy loss between logits and targets.
    It also includes an optional z-loss term, which penalizes large logits for non-target
    classes, encouraging the model to be less confident in incorrect predictions.
    A custom VJP (vector-Jacobian product) is defined for efficient gradient computation.

    Args:
        logits (chex.Array): The predicted logits from the model (batch_size, ..., num_classes).
        targets (chex.Array): The target labels, typically one-hot encoded (batch_size, ..., num_classes).
        z_loss (float): The coefficient for the z-loss regularization. If 0, z-loss is not computed.

    Returns:
        tp.Tuple[chex.Array, chex.Array]:
            - loss: The cross-entropy loss for each example (batch_size, ...).
            - z_loss: The z-loss value for each example (batch_size, ...). Returns zero if `z_loss` coeff is 0.
    """
    logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: chex.Array,
    targets: chex.Array,
    z_loss: float = 0.0,
) -> tuple[
    tuple[
        chex.Array,
        chex.Array,
    ],
    tuple[
        chex.Array,
        chex.Array,
        float,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
]:
    """
    Forward pass for cross_entropy_with_logits VJP.
    Calculates the loss and intermediates needed for the backward pass.
    """
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    loss = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    total_z_loss = z_loss * jax.lax.square(log_z)
    loss += total_z_loss
    return (loss, total_z_loss), (
        logits,
        targets,
        z_loss,
        exp_shifted,
        sum_exp,
        log_softmax,
        log_z,
    )


def _cross_entropy_with_logits_bwd(
    res: tuple[
        chex.Array,
        chex.Array,
        float,
        chex.Array,
        chex.Array,
        chex.Array,
        chex.Array,
    ],
    g: tuple[chex.Array, chex.Array],
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """Backward-mode of `cross_entropy_with_logits`."""
    g = g[0]
    logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
    deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
    g_logits = jnp.expand_dims(g, axis=-1) * deriv
    g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
    return (
        jnp.asarray(g_logits, logits.dtype),
        jnp.asarray(g_targets, targets.dtype),
        jnp.array(0.0),
    )


cross_entropy_with_logits.defvjp(
    _cross_entropy_with_logits_fwd,
    _cross_entropy_with_logits_bwd,
)


def compute_weighted_cross_entropy(
    logits: chex.Array,
    targets: chex.Array,
    weights: chex.Array | None = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: float | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Computes weighted cross-entropy loss, z-loss, and weight sum.

    Args:
        logits: The predicted logits.
        targets: The target class labels (integers).
        weights: tp.Optional weights for each example.
        label_smoothing: Label smoothing factor.
        z_loss: Coefficient for the auxiliary z-loss term.
        loss_normalizing_factor: A factor to normalize the loss.

    Returns:
        A tuple containing the total loss, z-loss, and sum of weights.
    """
    if not isinstance(logits, jax.Array):
        raise TypeError(f"logits must be a JAX array, got {type(logits)}")
    if not isinstance(targets, jax.Array):
        raise TypeError(f"targets must be a JAX array, got {type(targets)}")
    if weights is not None and not isinstance(weights, jax.Array):
        raise TypeError(f"weights must be a JAX array or None, got {type(weights)}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in range 0~1, got {label_smoothing}")
    if z_loss < 0.0:
        raise ValueError(f"z_loss must be non-negative, got {z_loss}")
    if logits.ndim != targets.ndim + 1:
        raise ValueError(f"Incorrect shapes. Got shape {logits.shape} logits and {targets.shape} targets")
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = onehot(
        targets,
        vocab_size,
        on_value=confidence,
        off_value=low_confidence,
    )
    total_loss, total_z_loss = cross_entropy_with_logits(
        logits,
        soft_targets,
        z_loss=z_loss,
    )
    total_loss = total_loss - normalizing_constant

    shape_dtype_struct = jax.eval_shape(lambda x: x, targets)
    weight_sum = reduce(mul, shape_dtype_struct.shape, 1)
    if weights is not None:
        total_loss = total_loss * weights
        total_z_loss = total_z_loss * weights
        weight_sum = jnp.sum(weights)

    if loss_normalizing_factor is not None:
        total_loss /= loss_normalizing_factor
        total_z_loss /= loss_normalizing_factor
    return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum


def compute_weighted_cross_entropy_and_accuracy(
    logits: chex.Array,
    targets: chex.Array,
    weights: chex.Array | None = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: float | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Computes weighted cross-entropy loss, z-loss, weight sum, and accuracy.

    Args:
        logits: The predicted logits.
        targets: The target class labels (integers).
        weights: tp.Optional weights for each example.
        label_smoothing: Label smoothing factor.
        z_loss: Coefficient for the auxiliary z-loss term.
        loss_normalizing_factor: A factor to normalize the loss.

    Returns:
        A tuple containing the total loss, z-loss, sum of weights, and accuracy.
    """
    total_loss, total_z_loss, weight_sum = compute_weighted_cross_entropy(
        logits=logits,
        targets=targets,
        weights=weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
    )

    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = jnp.equal(predictions, targets).astype(jnp.float32)
    accuracy = jnp.sum(correct_predictions * weights) / weight_sum

    return total_loss, total_z_loss, weight_sum, accuracy


def cross_entropy_loss_and_accuracy(source, target, valid=None):
    if valid is None:
        valid = jnp.ones(target.shape[:2])
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    source = source.astype(jnp.float32)
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(source, axis=-1),
            jnp.expand_dims(target, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(valid > 0.0, jnp.argmax(source, axis=-1) == target, jnp.array(False))
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy


def convert_special_loss_normalizing_factor_to_enum(x: str) -> SLNF:
    """
    Converts a stringified version of SpecialLossNormalizingFactor to an enum.

    Args:
        x: Stringified version of the enum value.

    Returns:
        The corresponding SpecialLossNormalizingFactor enum value.
    """
    x = x.upper()
    if x == "NUM_REAL_TARGET_TOKENS":
        return SLNF.NUM_REAL_TARGET_TOKENS
    if x == "NUM_TOTAL_TARGET_TOKENS":
        return SLNF.NUM_TOTAL_TARGET_TOKENS
    if x == "AVERAGE_PER_SEQUENCE":
        return SLNF.AVERAGE_PER_SEQUENCE
    if x == "NO_WEIGHT_NUM_REAL_TARGET_TOKENS":
        return SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS
    raise ValueError(f'Could not convert string "{x}" to SpecialLossNormalizingFactor')


@jax.vmap
def _sum_weights_per_segment(
    positions: chex.Array,
    segment_ids: chex.Array,
    weights: chex.Array,
) -> chex.Array:
    """Sums weights per packed segment to produce a normalizing vector."""

    def _repeat_last_nonnegative(xs, reverse=False):
        def fn(prev, x):
            y = jnp.where(x == 0, prev, x)
            return y, y

        return jax.lax.scan(fn, jnp.zeros_like(xs[0]), xs, reverse=reverse)[1]

    start_positions = positions == 0
    final_positions = jnp.concatenate([start_positions[1:], jnp.ones(1)])
    final_positions *= segment_ids != 0
    final_cumulative_weights = final_positions * jnp.cumsum(weights)
    final_total_weights = jnp.concatenate(
        [
            final_cumulative_weights[0:1],
            jnp.diff(_repeat_last_nonnegative(final_cumulative_weights)),
        ]
    )
    normalizer = _repeat_last_nonnegative(final_total_weights, reverse=True)
    return normalizer


def get_loss_normalizing_factor_and_weights(
    loss_normalizing_factor: FACTOR_TYPE,
    batch: tp.Mapping[str, chex.Array],
) -> tuple[float | None, chex.Array | None]:
    """
    Gets the loss normalizing factor and weights from a batch of data.

    Args:
        loss_normalizing_factor: The loss normalizing factor to use.
        batch: A dictionary containing the input batch of data.

    Returns:
        A tuple containing the loss normalizing factor and loss weights.
    """
    loss_weights = batch.get("decoder_loss_weights", None)
    if loss_normalizing_factor is None or not isinstance(loss_normalizing_factor, str | SLNF):
        return loss_normalizing_factor, loss_weights

    if isinstance(loss_normalizing_factor, str):
        loss_normalizing_factor = convert_special_loss_normalizing_factor_to_enum(loss_normalizing_factor)

    if loss_weights is None:
        loss_weights = jnp.asarray(batch["decoder_target_tokens"] > 0, jnp.float32)

    output_normalizing_factor = None
    if loss_normalizing_factor == SLNF.NUM_REAL_TARGET_TOKENS:
        output_normalizing_factor = jnp.sum(loss_weights)
    elif loss_normalizing_factor == SLNF.NUM_TOTAL_TARGET_TOKENS:
        output_normalizing_factor = jnp.prod(batch["decoder_target_tokens"].shape)
    elif loss_normalizing_factor == SLNF.AVERAGE_PER_SEQUENCE:
        if "decoder_segment_ids" in batch:
            norm_vec = _sum_weights_per_segment(
                batch["decoder_positions"],
                batch["decoder_segment_ids"],
                loss_weights,
            )
        else:
            norm_vec = jnp.sum(loss_weights, axis=-1, keepdims=True)
        loss_weights = jnp.nan_to_num(loss_weights / norm_vec, nan=0, posinf=0, neginf=0)
        output_normalizing_factor = jnp.sum(loss_weights)
    else:
        raise ValueError(f"Unsupported value of loss_normalizing_factor: {loss_normalizing_factor}")

    return output_normalizing_factor, loss_weights


def auxiliary_load_balancing_loss_func(
    gate_logits: chex.Array | tuple[chex.Array, ...],
    num_experts: int,
    top_k: int,
    attention_mask: chex.Array | None = None,
) -> jax.Array | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in JAX.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`. Should be a tuple/list of JAX arrays,
            with each array corresponding to a layer and having shape
            [batch_size * sequence_length, num_experts].
            Alternatively, can be a single stacked array of shape
            [num_layers * batch_size * sequence_length, num_experts].
        num_experts:
            Number of experts. Must be provided if `gate_logits` is not None.
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`jax.numpy.ndarray`, *optional*):
            The attention_mask used in forward function
            shape [batch_size, sequence_length] if not None.

    Returns:
        The auxiliary loss as a JAX scalar array, or 0 if gate_logits is None.
    """
    if gate_logits is None:
        return 0
    if num_experts is None:
        raise ValueError("num_experts must be specified if gate_logits is provided.")

    # If gate_logits is a tuple or list, concatenate them.
    # Assumes individual layer logits are already on the correct device.
    if isinstance(gate_logits, tuple | list):
        # Ensure all logits are JAX arrays before concatenation
        gate_logits_list = [jnp.asarray(layer_gate.reshape(-1, num_experts)) for layer_gate in gate_logits]
        concatenated_gate_logits = jnp.concatenate(gate_logits_list, axis=0)
    elif isinstance(gate_logits, jnp.ndarray):
        concatenated_gate_logits = gate_logits
    else:
        raise TypeError(f"gate_logits must be a JAX array, tuple/list of JAX arrays, or None. Got {type(gate_logits)}")

    routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)
    _, selected_experts = jax.lax.top_k(routing_weights, k=top_k)
    expert_mask = jax.nn.one_hot(selected_experts, num_classes=num_experts, dtype=jnp.float32)

    if attention_mask is None:
        tokens_per_expert = jnp.mean(expert_mask, axis=0)
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    else:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [batch_size, sequence_length], got {attention_mask.shape}")

        batch_size, sequence_length = attention_mask.shape
        total_tokens_per_layer = batch_size * sequence_length
        num_effective_tokens = concatenated_gate_logits.shape[0]

        if num_effective_tokens % total_tokens_per_layer != 0:
            raise ValueError(
                f"Total tokens in gate_logits ({num_effective_tokens}) is not divisible by "
                f"batch_size*sequence_length ({total_tokens_per_layer}). Ensure gate_logits are correctly concatenated."
            )

        num_hidden_layers = num_effective_tokens // total_tokens_per_layer
        mask_expanded = jnp.expand_dims(attention_mask, axis=(0, 3, 4))
        target_mask_shape = (
            num_hidden_layers,
            batch_size,
            sequence_length,
            top_k,
            num_experts,
        )
        expert_attention_mask_broadcast = jnp.broadcast_to(mask_expanded, target_mask_shape)
        expert_attention_mask = jnp.reshape(expert_attention_mask_broadcast, (-1, top_k, num_experts))
        masked_expert_contributions = expert_mask * expert_attention_mask
        tokens_per_expert_numerator = jnp.sum(masked_expert_contributions, axis=0)
        tokens_per_expert_denominator = jnp.sum(expert_attention_mask, axis=0)
        tokens_per_expert_denominator = jnp.where(tokens_per_expert_denominator == 0, 1.0, tokens_per_expert_denominator)
        tokens_per_expert = tokens_per_expert_numerator / tokens_per_expert_denominator
        mask_expanded_router = jnp.expand_dims(attention_mask, axis=(0, 3))

        target_router_mask_shape = (
            num_hidden_layers,
            batch_size,
            sequence_length,
            num_experts,
        )
        router_attention_mask_broadcast = jnp.broadcast_to(mask_expanded_router, target_router_mask_shape)
        router_per_expert_attention_mask = jnp.reshape(router_attention_mask_broadcast, (-1, num_experts))

        masked_routing_weights = routing_weights * router_per_expert_attention_mask

        router_prob_numerator = jnp.sum(masked_routing_weights, axis=0)
        router_prob_denominator = jnp.sum(router_per_expert_attention_mask, axis=0)
        router_prob_denominator = jnp.where(router_prob_denominator == 0, 1.0, router_prob_denominator)
        router_prob_per_expert = router_prob_numerator / router_prob_denominator
    router_prob_per_expert_expanded = jnp.expand_dims(router_prob_per_expert, axis=0)
    per_expert_loss_terms = tokens_per_expert * router_prob_per_expert_expanded
    overall_loss = jnp.sum(per_expert_loss_terms)
    final_loss = overall_loss * num_experts

    return jnp.asarray(final_loss, dtype=jnp.float32)


def fixed_cross_entropy(
    source: jax.Array,
    target: jax.Array,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    num_items_in_batch: int | None = None,
    batch: tp.Mapping[str, chex.Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """
    Jax implementation of fixed cross-entropy loss with z-loss, label smoothing, masking.

    Args:
        source: Predicted logits, shape (batch_size, num_classes) or (batch_size * seq_len, num_classes).
        target: True labels, shape (batch_size,) or (batch_size * seq_len,). Must be integers.
        num_items_in_batch: tp.Optional, used when reduction should be sum.
        attention_mask: tp.Optional, boolean mask applied to the loss.
        batch: tp.Optional batch for dynamic loss normalization
        **kwargs: Additional keyword arguments.

    Returns:
        The computed cross-entropy loss in LossMetrics.
    """
    if config is None:
        config = LossConfig()
    if source is None or target is None:
        raise ValueError("Logits and labels cannot be None")

    mask = attention_mask if attention_mask is not None else (target != config.ignore_index)
    nwn_cond = SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS
    nwv_cond = SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS.value
    loss_factor = config.loss_normalizing_factor
    if config.reduction is not None:
        (
            loss,
            accuracy,
        ) = dynamic_cross_entropy_loss(
            logits=source,
            targets=target,
            ignore_index=config.ignore_index,
            reduction=config.reduction,
            label_smoothing=config.label_smoothing,
        )
        total_z_loss = 0.0
        weight_sum = 1.0
    elif loss_factor is nwn_cond or loss_factor is nwv_cond:
        loss, accuracy = cross_entropy_loss_and_accuracy(source, target, mask)
        total_z_loss = 0.0
        weight_sum = 1.0
    else:
        if batch is None:
            if config.loss_normalizing_factor in str(SLNF.NUM_REAL_TARGET_TOKENS):
                batch = {
                    "decoder_target_tokens": target,
                    "decoder_loss_weights": mask,
                }
            else:
                batch = {}
        (
            loss_normalizing_factor,
            loss_weights,
        ) = get_loss_normalizing_factor_and_weights(config.loss_normalizing_factor, batch)

        (
            total_loss,
            total_z_loss,
            weight_sum,
            accuracy,
        ) = compute_weighted_cross_entropy_and_accuracy(
            logits=source,
            targets=target,
            weights=loss_weights,
            label_smoothing=config.label_smoothing,
            z_loss=config.z_loss,
            loss_normalizing_factor=loss_normalizing_factor,
        )

        loss = total_loss
        if num_items_in_batch is not None:
            loss = total_loss / num_items_in_batch
        elif config.divide_weight_sum:
            loss = total_loss / weight_sum
        loss = loss + total_z_loss * (config.z_loss**2)
    return LossMetrics(
        loss=loss,
        z_loss=total_z_loss,
        weight_sum=weight_sum,
        accuracy=accuracy,
    )


def ForCausalLMLoss(
    logits: jax.Array,
    labels: jax.Array,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    num_items_in_batch: int | None = None,
    batch: tp.Mapping[str, chex.Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """
    Jax implementation of loss function for causal language models.

    Args:
        logits: Predicted logits, shape (batch_size, seq_len, vocab_size).
        labels: True labels, shape (batch_size, seq_len). Must be integers.
        num_items_in_batch: tp.Optional, used when reduction should be sum.
        batch: tp.Optional batch for dynamic loss normalization
        **kwargs: Additional keyword arguments for the cross-entropy loss.

    Returns:
        The computed causal language modeling loss.
    """
    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")
    if paxis is not None:
        logits = with_sharding_constraint(
            logits,
            PartitionSpec(
                paxis.batch_axis,
                paxis.sequence_axis,
                paxis.hidden_state_axis,
            ),
        )
        labels = with_sharding_constraint(
            labels,
            PartitionSpec(
                paxis.batch_axis,
                paxis.sequence_axis,
            ),
        )
    shift_attn_m = attention_mask
    if config is None:
        config = LossConfig()
    if config.shift_tokens:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        if attention_mask is not None:
            shift_attn_m = attention_mask[:, 1:]
    else:
        shift_logits = logits
        shift_labels = labels

        if attention_mask is not None:
            shift_attn_m = attention_mask

    loss = fixed_cross_entropy(
        source=shift_logits,
        target=shift_labels,
        attention_mask=shift_attn_m,
        config=config,
        num_items_in_batch=num_items_in_batch,
        batch=batch,
        **kwargs,
    )
    return loss


def ForSequenceClassificationLoss(
    logits: jax.Array,
    labels: jax.Array,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: tp.Mapping[str, chex.Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """
    Jax implementation of loss function for sequence classification.

    Args:
        labels: True labels, shape (batch_size,) or (batch_size, num_labels) for multi label classification.
        logits: Predicted logits, shape (batch_size, num_labels) or (batch_size, 1) or (batch_size,) for regression.
        config: Configuration with problem_type and num_labels attributes.
        batch: tp.Optional batch for dynamic loss normalization
        **kwargs: Additional keyword arguments for the cross-entropy loss.

    Returns:
        The computed sequence classification loss.
    """

    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")

    num_labels = config.num_labels

    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == jnp.int32 or labels.dtype == jnp.int64):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    if config.problem_type == "regression":
        loss = jnp.mean((logits.squeeze() - labels.squeeze()) ** 2)
    elif config.problem_type == "single_label_classification":
        return fixed_cross_entropy(
            source=logits.reshape(-1, num_labels),
            target=labels.reshape(-1),
            attention_mask=attention_mask,
            config=config,
            batch=batch,
            **kwargs,
        )
    elif config.problem_type == "multi_label_classification":
        loss = jnp.mean(
            sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                label_smoothing=config.label_smoothing,
            )
        )
    else:
        raise ValueError(f"Invalid problem_type: {config.problem_type}")
    return LossMetrics(total_loss=loss)


def ForQuestionAnsweringLoss(
    start_logits: jax.Array,
    end_logits: jax.Array,
    start_positions: jax.Array,
    end_positions: jax.Array,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: tp.Mapping[str, chex.Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """
    Jax implementation of loss function for question answering.

    Args:
        start_logits: Predicted start logits, shape (batch_size, seq_len).
        end_logits: Predicted end logits, shape (batch_size, seq_len).
        start_positions: True start positions, shape (batch_size,).
        end_positions: True end positions, shape (batch_size,).
        batch: tp.Optional batch for dynamic loss normalization
        **kwargs: Additional keyword arguments for the cross-entropy loss.

    Returns:
        The computed question answering loss.
    """
    if start_logits is None or end_logits is None or start_positions is None or end_positions is None:
        raise ValueError("Logits and labels cannot be None")

    ignored_index = start_logits.shape[1]
    start_positions = jnp.clip(start_positions, 0, ignored_index)
    end_positions = jnp.clip(end_positions, 0, ignored_index)

    start_loss = fixed_cross_entropy(
        source=start_logits,
        target=start_positions,
        config=config,
        batch=batch,
        **kwargs,
    )
    end_loss = fixed_cross_entropy(
        source=end_logits,
        target=end_positions,
        config=config,
        batch=batch,
        **kwargs,
    )
    loss = (start_loss.loss + end_loss.loss) / 2
    accuracy = (start_loss.accuracy + end_loss.accuracy) / 2
    z_loss = (start_loss.z_loss + end_loss.z_loss) / 2
    weight_sum = (start_loss.weight_sum + end_loss.weight_sum) / 2
    return LossMetrics(
        loss=loss,
        accuracy=accuracy,
        z_loss=z_loss,
        weight_sum=weight_sum,
    )


def ForTokenClassification(
    logits: jax.Array,
    labels: jax.Array,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: tp.Mapping[str, chex.Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """
    Jax implementation of loss function for token classification.

    Args:
        logits: Predicted logits, shape (batch_size, seq_len, num_labels).
        labels: True labels, shape (batch_size, seq_len). Must be integers.
        config: Configuration with num_labels attribute.
        label_smoothing: Label smoothing factor.
        z_loss: Coefficient for the auxiliary z-loss term.
        loss_normalizing_factor: A factor to normalize the loss, can also be enum.
        batch: tp.Optional batch for dynamic loss normalization
        **kwargs: Additional keyword arguments for the cross-entropy loss.

    Returns:
        The computed token classification loss.
    """
    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")

    loss = fixed_cross_entropy(
        source=logits,
        target=labels,
        config=config,
        batch=batch,
        **kwargs,
    )
    return loss


LOSS_MAPPING = {
    "ForCausalLM": ForCausalLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
}
