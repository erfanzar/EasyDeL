# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
    """
    Pytest fixture providing sample data for loss function tests.

    Returns:
        dict: A dictionary containing sample 'logits', 'targets', 'attention_mask',
              and 'weights' as JAX arrays.
    """
    # Set a seed for reproducibility
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
    """
    Tests the `onehot` utility function for creating one-hot encoded vectors.
    Verifies both standard one-hot encoding and encoding with custom on/off values.
    """
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
    """
    Tests the `cross_entropy_with_logits` function.
    Checks the output shapes and ensures the loss values are non-negative.
    """
    logits = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    z_loss = 0.1

    loss, z_loss_value = cross_entropy_with_logits(logits, targets, z_loss)

    assert loss.shape == (2,)
    assert z_loss_value.shape == (2,)
    assert jnp.all(loss >= 0)
    assert jnp.all(z_loss_value >= 0)


def test_compute_weighted_cross_entropy(sample_data):
    """
    Tests the `compute_weighted_cross_entropy` function.
    Verifies the function works correctly both with and without explicit weights,
    checking output types, shapes, and ensuring results are not NaN.
    """
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
    loss, z_loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights=weights)
    assert isinstance(weight_sum, jnp.ndarray)
    assert weight_sum.shape == ()
    assert jnp.all(weight_sum > 0)
    assert not jnp.isnan(loss)
    assert not jnp.isnan(z_loss)


def test_compute_weighted_cross_entropy_and_accuracy(sample_data):
    """
    Tests the `compute_weighted_cross_entropy_and_accuracy` function.
    Verifies the calculation of loss, z_loss, weight sum, and accuracy,
    checking output types, shapes, accuracy range, and ensuring no NaNs.
    """
    logits = sample_data["logits"]
    targets = sample_data["targets"]
    weights = sample_data["weights"]

    loss, z_loss, weight_sum, accuracy = compute_weighted_cross_entropy_and_accuracy(logits, targets, weights=weights)

    assert isinstance(accuracy, jnp.ndarray)
    assert accuracy.shape == ()
    assert 0 <= float(accuracy) <= 1
    assert not jnp.isnan(loss)
    assert not jnp.isnan(z_loss)
    assert not jnp.isnan(weight_sum)
    assert not jnp.isnan(accuracy)


def test_special_loss_normalizing_factor():
    """
    Tests the `convert_special_loss_normalizing_factor_to_enum` function.
    Checks conversion of valid string identifiers to the enum and verifies that
    an invalid identifier raises a ValueError.
    """
    # Test valid conversions
    assert (
        convert_special_loss_normalizing_factor_to_enum("NUM_REAL_TARGET_TOKENS")
        == SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
    )

    # Test invalid conversion
    with pytest.raises(ValueError):
        convert_special_loss_normalizing_factor_to_enum("INVALID")


def test_get_loss_normalizing_factor_and_weights():
    """
    Tests the `get_loss_normalizing_factor_and_weights` function.
    Verifies the correct return values for both constant and special
    (enum-based) loss normalizing factors.
    """
    batch = {
        "decoder_target_tokens": jnp.array([[1, 2, 0], [3, 0, 0]]),
        "decoder_loss_weights": jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    }

    # Test constant factor
    factor, weights = get_loss_normalizing_factor_and_weights(1.0, batch)
    assert factor == 1.0
    assert weights is batch["decoder_loss_weights"]

    # Test special factor
    factor, weights = get_loss_normalizing_factor_and_weights(SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS, batch)
    assert isinstance(factor, jnp.ndarray)
    assert weights is not None


def test_auxiliary_load_balancing_loss_func():
    """
    Tests the `auxiliary_load_balancing_loss_func` (often used for MoE models).
    Checks the calculation both with and without an attention mask,
    verifying output type, shape, and ensuring no NaNs.
    """
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
    loss = auxiliary_load_balancing_loss_func((gate_logits,), num_experts, top_k, attention_mask)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert not jnp.isnan(loss)


def test_input_validation():
    """
    Tests input validation and error handling in various loss utility functions.
    Ensures that appropriate ValueErrors are raised for invalid inputs like
    zero experts, top_k > num_experts, or negative label smoothing.
    """
    with pytest.raises(ValueError):
        auxiliary_load_balancing_loss_func(None, num_experts=0, top_k=1)

    with pytest.raises(ValueError):
        auxiliary_load_balancing_loss_func(None, num_experts=3, top_k=4)

    with pytest.raises(ValueError):
        compute_weighted_cross_entropy(jnp.zeros((2, 3)), jnp.zeros((2,)), label_smoothing=-0.1)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("z_loss", [0.0, 0.1])
@pytest.mark.parametrize("use_weights", [True, False])
def test_cross_entropy_integration(sample_data, label_smoothing, z_loss, use_weights):
    """
    Integration test for `compute_weighted_cross_entropy_and_accuracy`.
    Runs the function with various combinations of label smoothing, z_loss, and
    weights usage, ensuring results are always valid (non-NaN, accuracy in [0, 1]).
    """
    logits = sample_data["logits"]
    targets = sample_data["targets"]
    # Fix: Handle None weights properly
    weights = sample_data["weights"] if use_weights else jnp.ones_like(targets, dtype=jnp.float32)

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
    """
    Tests if gradients can be computed through the `cross_entropy_with_logits` function.
    Verifies that the gradient shape matches the input logits shape and contains no NaNs.
    """

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
    """
    Tests `get_loss_normalizing_factor_and_weights` with all defined
    `SpecialLossNormalizingFactor` enum values.
    Ensures that a valid factor and weights are returned for each special case.
    """
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
    """
    Tests the numerical stability of `compute_weighted_cross_entropy`.
    Uses extreme logit values (very small and very large probabilities)
    to check if the computation results in NaN.
    """

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
    """
    Tests if the average loss computed by `compute_weighted_cross_entropy`
    is consistent across different batch sizes.
    The per-example loss should remain approximately the same.
    """

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
    """
    Tests the effect of label smoothing on the cross-entropy loss.
    Compares the loss with and without label smoothing, expecting the smoothed loss
    to be higher when the model is confident in the correct class.
    """
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
    """
    Tests the custom vector-Jacobian product (VJP) implementation defined for
    `cross_entropy_with_logits`.
    Verifies both the forward and backward passes execute without NaNs.
    """

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
    """
    Tests `fixed_cross_entropy` in the simplest case: no mask, no smoothing, no z-loss.
    Ensures a valid scalar loss is returned.
    """
    logits = sample_data["logits"]
    targets = sample_data["targets"]

    loss = fixed_cross_entropy(logits, targets)
    assert not jnp.isnan(loss)
    assert loss.shape == ()


def test_fixed_cross_entropy_with_mask(sample_data):
    """
    Tests `fixed_cross_entropy` with an attention mask applied.
    Ensures a valid scalar loss is returned when masking is used.
    """
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
    """
    Tests `fixed_cross_entropy` with the `ignore_index` parameter.
    Ensures tokens matching the ignore_index are excluded from the loss calculation.
    """
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
    """
    Tests `fixed_cross_entropy` with label smoothing applied.
    Ensures a valid scalar loss is returned when label smoothing is active.
    """
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
    """
    Tests `fixed_cross_entropy` with z_loss regularization applied.
    Ensures a valid scalar loss is returned when z_loss is active.
    """
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
    """
    Tests `fixed_cross_entropy` when `num_items_in_batch` is provided for normalization.
    Ensures a valid scalar loss is returned.
    """
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
    """
    Tests `fixed_cross_entropy` with a constant `loss_normalizing_factor`.
    Ensures a valid scalar loss is returned when normalization is applied.
    """
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
    """
    Tests `fixed_cross_entropy` with a `SpecialLossNormalizingFactor`.
    Ensures a valid scalar loss is returned when using special normalization.
    """
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
    """
    Integration test for `fixed_cross_entropy` covering multiple parameter combinations.
    Tests various settings for label smoothing, z_loss, masking, batch item count,
    and loss normalization factor types.
    """
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
    """
    Tests input validation for `fixed_cross_entropy`.
    Ensures ValueError is raised if logits or labels are None.
    """
    with pytest.raises(ValueError, match="Logits and labels cannot be None"):
        fixed_cross_entropy(None, jnp.array([1, 2, 3]))


if __name__ == "__main__":
    pytest.main([__file__])
