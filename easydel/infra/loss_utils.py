import enum
import pprint
import typing as tp
from functools import reduce
from operator import mul

import chex
import flax
import flax.struct
import jax
import jax.numpy as jnp
from jax import lax


@enum.unique
class SpecialLossNormalizingFactor(enum.Enum):
	"""
	Specially calculated loss normalizing factors that are not constant.

	Attributes:
	    NUM_REAL_TARGET_TOKENS: Divide the loss by the number of real (non-padding) tokens.
	    NUM_TOTAL_TARGET_TOKENS: Divide the loss by the total number of target tokens.
	    AVERAGE_PER_SEQUENCE: Compute the average loss per sequence.
	"""

	NUM_REAL_TARGET_TOKENS = 1
	NUM_TOTAL_TARGET_TOKENS = 2
	AVERAGE_PER_SEQUENCE = 3


FACTOR_TYPE = tp.Optional[tp.Union[float, int, str, SpecialLossNormalizingFactor]]


@chex.dataclass
class LossConfig:
	ignore_index: int = -100
	label_smoothing: float = 0.0
	z_loss: float = 0.0
	loss_normalizing_factor: FACTOR_TYPE = "NUM_REAL_TARGET_TOKENS"
	num_labels: tp.Optional[str] = None
	problem_type: tp.Optional[str] = None
	divide_weight_sum: bool = False
	num_classification_labels: tp.Optional[int] = None
	classification_problem_type: tp.Optional[
		tp.Literal[
			"regression",
			"single_label_classification",
			"multi_label_classification",
		]
	] = None

	def __repr__(self):
		return pprint.pformat(
			{
				"ignore_index": self.ignore_index,
				"label_smoothing": self.label_smoothing,
				"z_loss": self.z_loss,
				"loss_normalizing_factor": self.loss_normalizing_factor,
				"num_labels": self.num_labels,
				"problem_type": self.problem_type,
				"divide_weight_sum": self.divide_weight_sum,
				"num_classification_labels": self.num_classification_labels,
				"classification_problem_type": self.classification_problem_type,
			}
		)


@chex.dataclass
class LossMetrics:
	loss: tp.Optional[tp.Union[float, chex.Array]] = None
	z_loss: tp.Optional[tp.Union[float, chex.Array]] = None
	weight_sum: tp.Optional[tp.Union[float, chex.Array]] = None
	accuracy: tp.Optional[tp.Union[float, chex.Array]] = None
	learning_rate: tp.Optional[tp.Union[float, chex.Array]] = None
	max_grad_norm: tp.Optional[flax.struct.PyTreeNode] = None
	mean_grad_norm: tp.Optional[flax.struct.PyTreeNode] = None
	grad_norms: tp.Optional[flax.struct.PyTreeNode] = None
	chosen_rewards: tp.Optional[jax.Array] = None
	rejected_rewards: tp.Optional[jax.Array] = None
	other_metrics: tp.Optional[tp.Mapping[str, jax.Array]] = None


def sigmoid_cross_entropy_with_logits(
	logits: jnp.ndarray,
	labels: jnp.ndarray,
	weights: tp.Optional[jnp.ndarray] = None,
	label_smoothing: float = 0.0,
	axis: tp.Optional[tp.Union[int, tuple]] = None,
) -> jnp.ndarray:
	"""
	Computes sigmoid cross entropy given logits and labels.

	Args:
	    logits: Input tensor
	    labels: Target tensor with the same shape as logits
	    weights: tp.Optional weights to apply to the loss
	    label_smoothing: Float in [0, 1]. Amount of smoothing to apply to labels
	    axis: The dimensions to reduce. If None, reduces all dimensions.

	Returns:
	    Sigmoid cross entropy loss, reduced according to axis if specified
	"""
	# Input validation
	if logits.shape != labels.shape:
		raise ValueError(
			f"Logits shape {logits.shape} must match labels shape {labels.shape}"
		)

	if label_smoothing < 0 or label_smoothing > 1:
		raise ValueError(f"Label smoothing must be in [0, 1], got {label_smoothing}")

	# Apply label smoothing if specified
	if label_smoothing > 0:
		labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

	# Compute stable sigmoid cross entropy
	zeros = jnp.zeros_like(logits)
	cond = logits >= zeros
	relu_logits = jnp.where(cond, logits, zeros)
	neg_abs_logits = jnp.where(cond, -logits, logits)

	loss = relu_logits - logits * labels + jnp.log1p(jnp.exp(neg_abs_logits))

	# Apply weights if specified
	if weights is not None:
		loss = loss * weights

	# Reduce if axis is specified
	if axis is not None:
		loss = jnp.mean(loss, axis=axis)

	return loss


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
	"""Create a dense one-hot version of an indexed array.

	NB: consider using the more standard ``jax.nn.one_hot`` instead.

	Args:
	  labels: an n-dim JAX array whose last dimension contains integer indices.
	  num_classes: the maximum possible index.
	  on_value: the "on" value for the one-hot array, defaults to 1.0.
	  off_value: the "off" value for the one-hot array, defaults to 0.0.
	Returns:
	  A (n+1)-dim array whose last dimension contains one-hot vectors of length
	  num_classes.
	"""
	x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
	x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
	return x.astype(jnp.float32)


@jax.custom_vjp
def cross_entropy_with_logits(
	logits: chex.Array,
	targets: chex.Array,
	z_loss: float,
) -> tp.Tuple[chex.Array, chex.Array]:
	"""
	Computes cross-entropy loss with a stable custom gradient and z-loss.

	This function computes the cross-entropy loss with an optional auxiliary loss
	term (`z_loss`) to prevent logits from drifting too far from zero.

	Args:
	    logits: The predicted logits.
	    targets: The one-hot encoded target labels.
	    z_loss: Coefficient for the auxiliary z-loss term.

	Returns:
	    A tuple containing the total loss and the z_loss.
	"""
	logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
	log_softmax = logits - logits_sum
	loss = -jnp.sum(targets * log_softmax, axis=-1)

	log_z = jnp.squeeze(logits_sum, axis=-1)
	total_z_loss = z_loss * jax.lax.square(log_z)
	loss += total_z_loss
	return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
	logits: chex.Array,
	targets: chex.Array,
	z_loss: float = 0.0,
) -> tp.Tuple[
	tp.Tuple[
		chex.Array,
		chex.Array,
	],
	tp.Tuple[
		chex.Array,
		chex.Array,
		float,
		chex.Array,
		chex.Array,
		chex.Array,
		chex.Array,
	],
]:
	"""Forward-mode of `cross_entropy_with_logits`."""
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
	res: tp.Tuple[
		chex.Array,
		chex.Array,
		float,
		chex.Array,
		chex.Array,
		chex.Array,
		chex.Array,
	],
	g: tp.Tuple[chex.Array, chex.Array],
) -> tp.Tuple[chex.Array, chex.Array, chex.Array]:
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
	weights: tp.Optional[chex.Array] = None,
	label_smoothing: float = 0.0,
	z_loss: float = 0.0,
	loss_normalizing_factor: tp.Optional[float] = None,
) -> tp.Tuple[chex.Array, chex.Array, chex.Array]:
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
		raise ValueError(
			f"Incorrect shapes. Got shape {logits.shape} logits and {targets.shape} targets"
		)
	vocab_size = logits.shape[-1]
	confidence = 1.0 - label_smoothing
	low_confidence = (1.0 - confidence) / (vocab_size - 1)
	normalizing_constant = -(
		confidence * jnp.log(confidence)
		+ (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
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
	weights: tp.Optional[chex.Array] = None,
	label_smoothing: float = 0.0,
	z_loss: float = 0.0,
	loss_normalizing_factor: tp.Optional[float] = None,
) -> tp.Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
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
		logits, targets, weights, label_smoothing, z_loss, loss_normalizing_factor
	)

	predictions = jnp.argmax(logits, axis=-1)
	correct_predictions = jnp.equal(predictions, targets).astype(jnp.float32)
	accuracy = jnp.sum(correct_predictions * weights) / weight_sum

	return total_loss, total_z_loss, weight_sum, accuracy


def convert_special_loss_normalizing_factor_to_enum(
	x: str,
) -> SpecialLossNormalizingFactor:
	"""
	Converts a stringified version of SpecialLossNormalizingFactor to an enum.

	Args:
	    x: Stringified version of the enum value.

	Returns:
	    The corresponding SpecialLossNormalizingFactor enum value.
	"""
	x = x.upper()
	if x == "NUM_REAL_TARGET_TOKENS":
		return SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
	if x == "NUM_TOTAL_TARGET_TOKENS":
		return SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS
	if x == "AVERAGE_PER_SEQUENCE":
		return SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE
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
) -> tp.Tuple[tp.Optional[float], tp.Optional[chex.Array]]:
	"""
	Gets the loss normalizing factor and weights from a batch of data.

	Args:
	    loss_normalizing_factor: The loss normalizing factor to use.
	    batch: A dictionary containing the input batch of data.

	Returns:
	    A tuple containing the loss normalizing factor and loss weights.
	"""
	loss_weights = batch.get("decoder_loss_weights", None)
	if loss_normalizing_factor is None or not isinstance(
		loss_normalizing_factor, (str, SpecialLossNormalizingFactor)
	):
		return loss_normalizing_factor, loss_weights

	if isinstance(loss_normalizing_factor, str):
		loss_normalizing_factor = convert_special_loss_normalizing_factor_to_enum(
			loss_normalizing_factor
		)

	if loss_weights is None:
		loss_weights = jnp.asarray(batch["decoder_target_tokens"] > 0, jnp.float32)

	output_normalizing_factor = None
	if loss_normalizing_factor == SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS:
		output_normalizing_factor = jnp.sum(loss_weights)
	elif loss_normalizing_factor == SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS:
		output_normalizing_factor = jnp.prod(batch["decoder_target_tokens"].shape)
	elif loss_normalizing_factor == SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE:
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
		raise ValueError(
			f"Unsupported value of loss_normalizing_factor: {loss_normalizing_factor}"
		)

	return output_normalizing_factor, loss_weights


def auxiliary_load_balancing_loss_func(
	gate_logits: tp.Union[chex.Array, tp.Tuple[chex.Array, ...]],
	num_experts: int,
	top_k: int,
	attention_mask: tp.Optional[chex.Array] = None,
) -> chex.Array:
	"""
	Computes auxiliary load balancing loss as in Switch Transformer.

	See Switch Transformer (https://arxiv.org/abs/2101.03961)

	Args:
	    gate_logits: The logits for the gating network, either as a single array
	        or tuple of arrays.
	    num_experts: The number of experts.
	    top_k: The number of experts to select.
	    attention_mask: An optional attention mask.

	Returns:
	    The auxiliary load balancing loss as a scalar array.

	Raises:
	    ValueError: If num_experts or top_k are invalid.
	"""
	# Input validation
	if num_experts < 1:
		raise ValueError(f"num_experts must be positive, got {num_experts}")
	if top_k < 1 or top_k > num_experts:
		raise ValueError(f"top_k must be between 1 and num_experts, got {top_k}")

	# Handle empty or invalid input
	if gate_logits is None:
		return jnp.array(0.0, dtype=jnp.float32)

	# Convert tuple of logits to single array
	if isinstance(gate_logits, tuple):
		concatenated_gate_logits = jnp.concatenate(gate_logits, axis=0)
	else:
		concatenated_gate_logits = gate_logits

	# Compute routing weights with improved numerical stability
	routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)

	# Get top-k expert selections
	_, selected_experts = jax.lax.top_k(routing_weights, top_k)
	expert_mask = jax.nn.one_hot(selected_experts, num_experts)

	if attention_mask is None:
		# Compute mean utilization per expert
		tokens_per_expert = jnp.mean(expert_mask.astype(jnp.float32), axis=0)
		router_prob_per_expert = jnp.mean(routing_weights, axis=0)
	else:
		# Handle masked version
		batch_size, sequence_length = attention_mask.shape
		num_hidden_layers = concatenated_gate_logits.shape[0] // (
			batch_size * sequence_length
		)

		expert_attention_mask = jnp.broadcast_to(
			attention_mask[jnp.newaxis, :, :, jnp.newaxis],
			(num_hidden_layers, batch_size, sequence_length, top_k),
		)

		# Compute masked means
		mask_sum = jnp.sum(attention_mask) + 1e-8  # Avoid division by zero
		tokens_per_expert = (
			jnp.sum(expert_mask * expert_attention_mask[..., jnp.newaxis], axis=(0, 1, 2))
			/ mask_sum
		)
		router_prob_per_expert = (
			jnp.sum(routing_weights * attention_mask[..., jnp.newaxis], axis=(0, 1))
			/ mask_sum
		)

	# Compute load balancing loss
	aux_loss = jnp.sum(tokens_per_expert * router_prob_per_expert) * num_experts
	return aux_loss


def fixed_cross_entropy(
	source: jax.Array,
	target: jax.Array,
	attention_mask: tp.Optional[jax.Array] = None,
	config: tp.Optional[LossConfig] = None,
	num_items_in_batch: tp.Optional[int] = None,
	batch: tp.Optional[tp.Mapping[str, chex.Array]] = None,
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

	mask = (
		attention_mask if attention_mask is not None else (target != config.ignore_index)
	)

	if batch is None:
		if config.loss_normalizing_factor in str(
			SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
		):
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

	return LossMetrics(
		loss=loss,
		z_loss=total_z_loss,
		weight_sum=weight_sum,
		accuracy=accuracy,
	)


def ForCausalLMLoss(
	logits: jax.Array,
	labels: jax.Array,
	attention_mask: tp.Optional[jax.Array] = None,
	config: tp.Optional[LossConfig] = None,
	num_items_in_batch: tp.Optional[int] = None,
	batch: tp.Optional[tp.Mapping[str, chex.Array]] = None,
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

	shift_logits = logits[:, :-1, :]
	shift_labels = labels[:, 1:]

	loss = fixed_cross_entropy(
		source=shift_logits,
		target=shift_labels,
		attention_mask=attention_mask,
		config=config,
		num_items_in_batch=num_items_in_batch,
		batch=batch,
		**kwargs,
	)
	return loss


def ForSequenceClassificationLoss(
	labels: jax.Array,
	logits: jax.Array,
	attention_mask: tp.Optional[jax.Array] = None,
	config: tp.Optional[LossConfig] = None,
	batch: tp.Optional[tp.Mapping[str, chex.Array]] = None,
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
			source=logits,
			target=labels,
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
	config: tp.Optional[LossConfig] = None,
	batch: tp.Optional[tp.Mapping[str, chex.Array]] = None,
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
	if (
		start_logits is None
		or end_logits is None
		or start_positions is None
		or end_positions is None
	):
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
	config: tp.Optional[LossConfig] = None,
	batch: tp.Optional[tp.Mapping[str, chex.Array]] = None,
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
