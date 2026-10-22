"""Independent numerical regressions for loss options (accelerator-only; not run on CPU)."""

import jax
import numpy as np
import pytest
from easydel.infra.loss_utils import LossConfig, cross_entropy_blockwise_logits, fixed_cross_entropy
from jax import numpy as jnp


def _inputs():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 5, 8)).astype(np.float32)
    targets = np.asarray([[1, 2, -100, 3, 4], [5, -100, 6, 7, 0]], dtype=np.int32)
    return logits, targets


def _dense_reference(logits, targets, smoothing, z_loss):
    """Dense KL(q||softmax(logits)) reference, including target-entropy correction."""
    logits = logits.astype(np.float64)
    valid = targets != -100
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + logits.max(axis=-1)
    log_probs = logits - log_z[..., None]
    vocab = logits.shape[-1]
    q = np.full_like(logits, smoothing / (vocab - 1))
    np.put_along_axis(q, np.maximum(targets, 0)[..., None], 1 - smoothing, axis=-1)
    entropy_terms = np.zeros_like(q)
    np.log(q, out=entropy_terms, where=q > 0)
    per_token = np.sum(q * (entropy_terms - log_probs), axis=-1) * valid
    z_terms = z_loss * log_z**2 * valid
    probs = np.exp(log_probs)
    gradient = (probs - q + 2 * z_loss * log_z[..., None] * probs) * valid[..., None]
    accuracy = np.sum((logits.argmax(-1) == targets) * valid) / max(valid.sum(), 1)
    return per_token + z_terms, z_terms, gradient, accuracy


@pytest.mark.parametrize("smoothing,z_loss", [(0.1, 0.0), (0.0, 0.01), (0.1, 0.01)])
@pytest.mark.parametrize("num_items,divide_weight_sum", [(None, False), (20, False), (None, True)])
def test_no_weight_mode_applies_regularization_and_normalizers(smoothing, z_loss, num_items, divide_weight_sum):
    logits, targets = _inputs()
    cfg = LossConfig(
        label_smoothing=smoothing,
        z_loss=z_loss,
        loss_normalizing_factor="NO_WEIGHT_NUM_REAL_TARGET_TOKENS",
        divide_weight_sum=divide_weight_sum,
    )
    values, z_terms, gradient, accuracy = _dense_reference(logits, targets, smoothing, z_loss)
    count = (targets != -100).sum()
    divisor = count * (num_items if num_items is not None else count if divide_weight_sum else 1)
    metrics = fixed_cross_entropy(jnp.asarray(logits), jnp.asarray(targets), config=cfg, num_items_in_batch=num_items)
    actual_grad = jax.grad(
        lambda x: fixed_cross_entropy(x, jnp.asarray(targets), config=cfg, num_items_in_batch=num_items).loss
    )(jnp.asarray(logits))
    np.testing.assert_allclose(metrics.loss, values.sum() / divisor, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.z_loss, z_terms.sum() / count, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_grad, gradient / divisor, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.accuracy, accuracy, atol=1e-6)
    np.testing.assert_allclose(metrics.weight_sum, count)


@pytest.mark.parametrize("option", ["chunk_block_size", "chunk_vocab_size"])
@pytest.mark.parametrize("size", [3, 128])
@pytest.mark.parametrize("reduction", [None, "none", "sum", "mean"])
def test_chunk_options_match_dense_reference_and_gradients(option, size, reduction):
    logits, targets = _inputs()
    values, z_terms, gradient, accuracy = _dense_reference(logits, targets, 0.1, 0.001)
    cfg = LossConfig(label_smoothing=0.1, z_loss=0.001, reduction=reduction, **{option: size})
    metrics = fixed_cross_entropy(jnp.asarray(logits), jnp.asarray(targets), config=cfg)
    actual_grad = jax.grad(lambda x: fixed_cross_entropy(x, jnp.asarray(targets), config=cfg).loss.sum())(
        jnp.asarray(logits)
    )
    if reduction in (None, "mean"):
        divisor = (targets != -100).sum()
        expected_loss, expected_z = values.sum() / divisor, z_terms.sum() / divisor
    elif reduction == "sum":
        divisor = 1
        expected_loss, expected_z = values.sum(), z_terms.sum()
    else:
        divisor = 1
        expected_loss, expected_z = values, z_terms
    np.testing.assert_allclose(metrics.loss, expected_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.z_loss, expected_z, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_grad, gradient / divisor, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.accuracy, accuracy, atol=1e-6)


@pytest.mark.parametrize("option", ["chunk_block_size", "chunk_vocab_size"])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_chunked_mean_normalizes_z_loss_with_fractional_weights(option, all_ignored):
    logits, targets = _inputs()
    if all_ignored:
        targets[:] = -100
    weights = np.asarray([[0.25, 1.5, 0.5, 0.0, 0.75], [1.0, 2.0, 0.5, 0.0, 0.25]], dtype=np.float32)
    values, z_terms, gradient, _ = _dense_reference(logits, targets, 0.1, 0.001)
    divisor = max(np.sum(weights * (targets != -100)), 1e-8)
    cfg = LossConfig(label_smoothing=0.1, z_loss=0.001, reduction="mean", **{option: 3})
    metrics = fixed_cross_entropy(jnp.asarray(logits), jnp.asarray(targets), jnp.asarray(weights), config=cfg)
    actual_grad = jax.grad(
        lambda x: fixed_cross_entropy(x, jnp.asarray(targets), jnp.asarray(weights), config=cfg).loss
    )(jnp.asarray(logits))
    np.testing.assert_allclose(metrics.loss, np.sum(values * weights) / divisor, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(metrics.z_loss, np.sum(z_terms * weights) / divisor, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual_grad, gradient * weights[..., None] / divisor, rtol=2e-5, atol=2e-6)


def test_public_blockwise_adapter_preserves_documented_sum_contract():
    logits, targets = _inputs()
    expected, expected_z, _, accuracy = _dense_reference(logits, targets, 0.1, 0.001)
    loss, z_loss, weight_sum, actual_accuracy = cross_entropy_blockwise_logits(
        jnp.asarray(logits), jnp.asarray(targets), label_smoothing=0.1, z_loss=0.001, block_size=3
    )
    np.testing.assert_allclose(loss, expected.sum(), rtol=2e-5)
    np.testing.assert_allclose(z_loss, expected_z.sum(), rtol=2e-5)
    np.testing.assert_allclose(weight_sum, (targets != -100).sum())
    np.testing.assert_allclose(actual_accuracy, accuracy)


def test_no_weight_mode_all_ignored_is_finite():
    logits, targets = _inputs()
    targets[:] = -100
    cfg = LossConfig(label_smoothing=0.1, z_loss=0.01, loss_normalizing_factor="NO_WEIGHT_NUM_REAL_TARGET_TOKENS")
    result = fixed_cross_entropy(jnp.asarray(logits), jnp.asarray(targets), config=cfg, num_items_in_batch=0)
    gradient = jax.grad(lambda x: fixed_cross_entropy(x, jnp.asarray(targets), config=cfg, num_items_in_batch=0).loss)(
        jnp.asarray(logits)
    )
    np.testing.assert_array_equal(result.loss, 0)
    np.testing.assert_array_equal(result.z_loss, 0)
    np.testing.assert_array_equal(gradient, 0)


@pytest.mark.parametrize("factor", ["NUM_REAL_TARGET_TOKENS", "NUM_TOTAL_TARGET_TOKENS", None, 2.0])
def test_explicit_batch_weights_obey_attention_and_ignore_masks_without_mutation(factor):
    from easydel.infra.loss_utils import causal_lm_loss_chunked_lm_head

    logits, targets = _inputs()
    attention = np.asarray([[1, 0, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=np.float32)
    weights = np.asarray([[1, 2, 3, 0.5, 1], [0.5, 1, 2, 1, 1]], dtype=np.float32)
    values, _, _, _ = _dense_reference(logits, targets, 0, 0)
    effective = weights * attention * (targets != -100)
    if factor == "NUM_REAL_TARGET_TOKENS":
        divisor = effective.sum()
    elif factor == "NUM_TOTAL_TARGET_TOKENS":
        divisor = targets.size
    else:
        divisor = factor or 1
    expected = np.sum(values * effective) / divisor
    batch_weights = jnp.asarray(weights)
    batch = {"decoder_target_tokens": jnp.asarray(targets), "decoder_loss_weights": batch_weights}
    config = LossConfig(
        shift_tokens=False,
        loss_normalizing_factor=factor,
        divide_weight_sum=False,
        compute_dtype="fp32",
        cross_entropy_platform="xla",
    )
    dense = fixed_cross_entropy(
        jnp.asarray(logits), jnp.asarray(targets), jnp.asarray(attention), config=config, batch=batch
    )
    chunked = causal_lm_loss_chunked_lm_head(
        jnp.asarray(logits),
        jnp.asarray(targets),
        lambda hidden: hidden,
        vocab_size=logits.shape[-1],
        attention_mask=jnp.asarray(attention),
        config=config,
        batch=batch,
        token_chunk_size=2,
    )
    np.testing.assert_allclose(dense.loss, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(chunked.loss, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(dense.weight_sum, effective.sum())
    np.testing.assert_allclose(chunked.weight_sum, effective.sum())
    assert batch["decoder_loss_weights"] is batch_weights
    np.testing.assert_array_equal(batch_weights, weights)
