import jax
import jax.numpy as jnp
import numpy as np
import pytest

from .loss_utils import (
	SpecialLossNormalizingFactor,
	auxiliary_load_balancing_loss_func,
	compute_weighted_cross_entropy,
	compute_weighted_cross_entropy_and_accuracy,
	convert_special_loss_normalizing_factor_to_enum,
	cross_entropy_with_logits,
	fixed_cross_entropy,
	get_loss_normalizing_factor_and_weights,
	onehot,
)


@pytest.fixture
def sample_data():
	"""Fixture providing common test data."""
	rng = np.random.RandomState(0)
	batch_size = 4
	seq_length = 8
	vocab_size = 10

	logits = rng.randn(batch_size, seq_length, vocab_size)
	targets = rng.randint(0, vocab_size, (batch_size, seq_length))
	attention_mask = rng.randint(0, 2, (batch_size, seq_length)).astype(bool)
	weights = rng.rand(batch_size, seq_length)

	return {
		"logits": jnp.array(logits),
		"targets": jnp.array(targets),
		"attention_mask": jnp.array(attention_mask),
		"weights": jnp.array(weights),
	}


def test_onehot():
	"""Test one-hot encoding function."""
	labels = jnp.array([0, 1, 2])
	num_classes = 3
	expected = jnp.array(
		[
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
		]
	)

	result = onehot(labels, num_classes)
	np.testing.assert_array_almost_equal(result, expected)

	# Test custom values
	result = onehot(labels, num_classes, on_value=2.0, off_value=-1.0)
	expected = expected * 3.0 - 1.0
	np.testing.assert_array_almost_equal(result, expected)


def test_cross_entropy_with_logits():
	"""Test cross-entropy loss computation."""
	logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	z_loss = 0.1

	loss, z_loss_value = cross_entropy_with_logits(logits, targets, z_loss)

	assert loss.shape == (2,)
	assert z_loss_value.shape == (2,)
	assert jnp.all(loss >= 0)
	assert jnp.all(z_loss_value >= 0)


def test_compute_weighted_cross_entropy(sample_data):
	"""Test weighted cross-entropy computation."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	weights = sample_data["weights"]
	# Test without weights
	loss, z_loss, weight_sum = compute_weighted_cross_entropy(
		logits,
		targets,
		weights=None,
	)
	assert isinstance(loss, jnp.ndarray)
	assert isinstance(z_loss, jnp.ndarray)
	assert loss.shape == ()
	assert z_loss.shape == ()
	assert not jnp.isnan(loss)
	assert not jnp.isnan(z_loss)

	# Test with weights
	loss, z_loss, weight_sum = compute_weighted_cross_entropy(
		logits, targets, weights=weights
	)
	assert isinstance(weight_sum, jnp.ndarray)
	assert weight_sum.shape == ()
	assert jnp.all(weight_sum > 0)
	assert not jnp.isnan(loss)
	assert not jnp.isnan(z_loss)


def test_compute_weighted_cross_entropy_and_accuracy(sample_data):
	"""Test combined loss and accuracy computation."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	weights = sample_data["weights"]

	loss, z_loss, weight_sum, accuracy = compute_weighted_cross_entropy_and_accuracy(
		logits, targets, weights=weights
	)

	assert isinstance(accuracy, jnp.ndarray)
	assert accuracy.shape == ()
	assert 0 <= float(accuracy) <= 1
	assert not jnp.isnan(loss)
	assert not jnp.isnan(z_loss)
	assert not jnp.isnan(weight_sum)
	assert not jnp.isnan(accuracy)


def test_special_loss_normalizing_factor():
	"""Test special loss normalizing factor conversion."""
	# Test valid conversions
	assert (
		convert_special_loss_normalizing_factor_to_enum("NUM_REAL_TARGET_TOKENS")
		== SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
	)

	# Test invalid conversion
	with pytest.raises(ValueError):
		convert_special_loss_normalizing_factor_to_enum("INVALID")


def test_get_loss_normalizing_factor_and_weights():
	"""Test loss normalizing factor computation."""
	batch = {
		"decoder_target_tokens": jnp.array([[1, 2, 0], [3, 0, 0]]),
		"decoder_loss_weights": jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
	}

	# Test constant factor
	factor, weights = get_loss_normalizing_factor_and_weights(1.0, batch)
	assert factor == 1.0
	assert weights is batch["decoder_loss_weights"]

	# Test special factor
	factor, weights = get_loss_normalizing_factor_and_weights(
		SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS, batch
	)
	assert isinstance(factor, jnp.ndarray)
	assert weights is not None


def test_auxiliary_load_balancing_loss_func():
	"""Test auxiliary load balancing loss computation."""
	batch_size = 2
	seq_length = 4
	num_experts = 3
	top_k = 2

	# Fix: Reshape gate_logits to match expected dimensions
	gate_logits = jnp.array(
		[
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
			[0.7, 0.8, 0.9],
			[1.0, 1.1, 1.2],
		]
	)

	# Test without attention mask
	loss = auxiliary_load_balancing_loss_func((gate_logits,), num_experts, top_k)
	assert isinstance(loss, jnp.ndarray)
	assert loss.shape == ()
	assert not jnp.isnan(loss)

	# Test with attention mask
	attention_mask = jnp.ones((batch_size, seq_length))
	loss = auxiliary_load_balancing_loss_func(
		(gate_logits,), num_experts, top_k, attention_mask
	)
	assert isinstance(loss, jnp.ndarray)
	assert loss.shape == ()
	assert not jnp.isnan(loss)


def test_input_validation():
	"""Test input validation and error handling."""
	with pytest.raises(ValueError):
		auxiliary_load_balancing_loss_func(None, num_experts=0, top_k=1)

	with pytest.raises(ValueError):
		auxiliary_load_balancing_loss_func(None, num_experts=3, top_k=4)

	with pytest.raises(ValueError):
		compute_weighted_cross_entropy(
			jnp.zeros((2, 3)), jnp.zeros((2,)), label_smoothing=-0.1
		)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("z_loss", [0.0, 0.1])
@pytest.mark.parametrize("use_weights", [True, False])
def test_cross_entropy_integration(sample_data, label_smoothing, z_loss, use_weights):
	"""Integration test with different parameter combinations."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	# Fix: Handle None weights properly
	weights = (
		sample_data["weights"] if use_weights else jnp.ones_like(targets, dtype=jnp.float32)
	)

	loss, z_loss_val, weight_sum, accuracy = compute_weighted_cross_entropy_and_accuracy(
		logits=logits,
		targets=targets,
		weights=weights,
		label_smoothing=label_smoothing,
		z_loss=z_loss,
	)

	assert not jnp.isnan(loss)
	assert not jnp.isnan(z_loss_val)
	assert not jnp.isnan(weight_sum)
	assert not jnp.isnan(accuracy)
	assert 0 <= float(accuracy) <= 1


def test_gradient_computation():
	"""Test gradient computation for cross entropy loss."""

	@jax.jit
	def loss_fn(logits, targets):
		loss, _ = cross_entropy_with_logits(logits, targets, z_loss=0.1)
		return jnp.mean(loss)

	logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
	targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	grad_fn = jax.grad(loss_fn)
	grads = grad_fn(logits, targets)

	assert grads.shape == logits.shape
	assert not jnp.any(jnp.isnan(grads))


def test_special_loss_normalizing_factors():
	"""Test all special loss normalizing factors."""
	batch = {
		"decoder_target_tokens": jnp.array([[1, 2, 0], [3, 0, 0]]),
		"decoder_positions": jnp.array([[0, 1, 2], [0, 1, 2]]),
		"decoder_segment_ids": jnp.array([[1, 1, 1], [2, 2, 2]]),
	}

	special_factors = [
		SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
		SpecialLossNormalizingFactor.NUM_TOTAL_TARGET_TOKENS,
		SpecialLossNormalizingFactor.AVERAGE_PER_SEQUENCE,
	]

	for factor in special_factors:
		norm_factor, weights = get_loss_normalizing_factor_and_weights(factor, batch)
		assert norm_factor is not None
		assert weights is not None
		assert not jnp.any(jnp.isnan(weights))


def test_numerical_stability():
	"""Test numerical stability with extreme values."""

	# Test with very small probabilities
	small_logits = jnp.array([[-1e10, 0.0, -1e10]])
	targets = jnp.array([1])

	loss, z_loss, weight_sum = compute_weighted_cross_entropy(small_logits, targets)
	assert not jnp.isnan(loss)

	# Test with very large probabilities
	large_logits = jnp.array([[1e10, 0.0, 1e10]])
	loss, z_loss, weight_sum = compute_weighted_cross_entropy(large_logits, targets)
	assert not jnp.isnan(loss)


def test_batch_consistency():
	"""Test consistency across different batch sizes."""

	def compute_loss_for_batch(batch_size):
		logits = jnp.ones((batch_size, 3, 5))
		targets = jnp.zeros((batch_size, 3), dtype=jnp.int32)
		return compute_weighted_cross_entropy(logits, targets)[0]

	loss_1 = compute_loss_for_batch(1)
	loss_2 = compute_loss_for_batch(2)
	loss_4 = compute_loss_for_batch(4)

	# Per-example loss should be consistent
	np.testing.assert_allclose(loss_1 * 2, loss_2, rtol=1e-5)
	np.testing.assert_allclose(loss_1 * 4, loss_4, rtol=1e-5)


def test_label_smoothing_effects():
	"""Test the effects of label smoothing."""
	logits = jnp.array([[2.0, -1.0, -1.0]])  # Make the correct class more likely
	targets = jnp.array([0])

	# Compare losses with different label smoothing values
	loss_no_smooth = compute_weighted_cross_entropy(
		logits,
		targets,
		label_smoothing=0.0,
	)[0]
	loss_with_smooth = compute_weighted_cross_entropy(
		logits,
		targets,
		label_smoothing=0.1,
	)[0]

	assert float(loss_with_smooth) < float(loss_no_smooth)


def test_custom_vjp():
	"""Test custom vector-Jacobian product implementation."""

	@jax.jit
	def loss_fn(logits, targets):
		return cross_entropy_with_logits(logits, targets, z_loss=0.1)[0]

	logits = jnp.array([[1.0, 2.0, 3.0]])
	targets = jnp.array([[1.0, 0.0, 0.0]])

	# Test forward pass
	loss = loss_fn(logits, targets)
	assert not jnp.any(jnp.isnan(loss))

	# Test backward pass
	grad_fn = jax.grad(lambda x: jnp.sum(loss_fn(x, targets)))
	grads = grad_fn(logits)
	assert not jnp.any(jnp.isnan(grads))


def test_fixed_cross_entropy_no_mask(sample_data):
	"""Test fixed_cross_entropy with no mask, no smoothing, and no z-loss."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]

	loss = fixed_cross_entropy(logits, targets)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_mask(sample_data):
	"""Test fixed_cross_entropy with an attention mask."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	attention_mask = sample_data["attention_mask"]

	loss = fixed_cross_entropy(
		logits,
		targets,
		attention_mask=attention_mask,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_ignore_index(sample_data):
	"""Test fixed_cross_entropy with ignore_index."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	ignore_index = 0
	targets = targets.at[0, 0].set(ignore_index)

	loss = fixed_cross_entropy(
		logits,
		targets,
		ignore_index=ignore_index,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_label_smoothing(sample_data):
	"""Test fixed cross entropy with label smoothing"""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	label_smoothing = 0.1
	loss = fixed_cross_entropy(
		logits,
		targets,
		label_smoothing=label_smoothing,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_z_loss(sample_data):
	"""Test fixed cross entropy with z loss"""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	z_loss = 0.1
	loss = fixed_cross_entropy(
		logits,
		targets,
		z_loss=z_loss,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_num_items_in_batch(sample_data):
	"""Test fixed_cross_entropy when `num_items_in_batch` is set."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	num_items_in_batch = logits.shape[0]
	loss = fixed_cross_entropy(
		logits,
		targets,
		num_items_in_batch=num_items_in_batch,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_loss_normalizing_factor(sample_data):
	"""Test fixed_cross_entropy with loss_normalizing_factor (constant)."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	loss_normalizing_factor = 2.0

	loss = fixed_cross_entropy(
		logits,
		targets,
		loss_normalizing_factor=loss_normalizing_factor,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_with_special_loss_normalizing_factor(sample_data):
	"""Test fixed_cross_entropy with special loss normalizing factor."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	batch = {
		"decoder_target_tokens": targets,
		"decoder_loss_weights": jnp.ones_like(targets, dtype=jnp.float32),
	}

	loss = fixed_cross_entropy(
		logits,
		targets,
		loss_normalizing_factor=SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS,
		batch=batch,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("z_loss", [0.0, 0.1])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("use_num_items_in_batch", [True, False])
@pytest.mark.parametrize("loss_normalizing_factor_type", ["none", "constant"])
def test_fixed_cross_entropy_integration(
	sample_data,
	label_smoothing,
	z_loss,
	use_mask,
	use_num_items_in_batch,
	loss_normalizing_factor_type,
):
	"""Integration test with different parameter combinations."""
	logits = sample_data["logits"]
	targets = sample_data["targets"]
	attention_mask = sample_data["attention_mask"] if use_mask else None
	num_items_in_batch = logits.shape[0] if use_num_items_in_batch else None

	loss_normalizing_factor = None
	batch = {}
	if loss_normalizing_factor_type == "constant":
		loss_normalizing_factor = 2.0
	elif loss_normalizing_factor_type == "special":
		loss_normalizing_factor = SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
		batch = {
			"decoder_target_tokens": targets,
			"decoder_loss_weights": jnp.ones_like(targets, dtype=jnp.float32),
		}

	loss = fixed_cross_entropy(
		logits,
		targets,
		attention_mask=attention_mask if attention_mask is not None else None,
		label_smoothing=label_smoothing,
		z_loss=z_loss,
		num_items_in_batch=num_items_in_batch,
		loss_normalizing_factor=loss_normalizing_factor,
		batch=batch,
	)
	assert not jnp.isnan(loss)
	assert loss.shape == ()


def test_fixed_cross_entropy_input_validation():
	"""Test fixed cross entropy for value errors"""
	with pytest.raises(ValueError, match="Logits and labels cannot be None"):
		fixed_cross_entropy(None, jnp.array([1, 2, 3]))

	with pytest.raises(ValueError, match="Logits and labels cannot be None"):
		fixed_cross_entropy(jnp.array([[1, 2, 3], [1, 2, 3]]), None)


if __name__ == "__main__":
	pytest.main([__file__])
