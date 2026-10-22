"""Accelerator-only DFT parity against independent float64 NumPy math."""

import jax
import numpy as np
import pytest
from easydel.trainers.trainer._fn import _dft_causal_lm_metrics
from jax import numpy as jnp


@pytest.mark.parametrize("dtype,valid_tokens", [(jnp.bfloat16, 257), (jnp.float16, 65537)])
def test_dft_low_precision_logits_keep_exact_counts_and_fp32_loss(dtype, valid_tokens):
    logits_np = np.tile(np.array([1.25, -0.5, 0.75], dtype=np.float32), (1, valid_tokens + 1, 1))
    labels_np = np.zeros((1, valid_tokens + 1), dtype=np.int32)
    logits = jnp.asarray(logits_np, dtype=dtype)
    labels = jnp.asarray(labels_np)
    metrics = jax.jit(_dft_causal_lm_metrics)(logits, labels)
    probs = np.exp(logits_np[0, 0].astype(np.float64) - 1.25)
    probs /= probs.sum()
    expected = -probs[0] * np.log(probs[0])
    assert metrics.loss.dtype == jnp.float32
    assert metrics.accuracy.dtype == jnp.float32
    assert metrics.weight_sum.dtype == jnp.float32
    assert float(metrics.weight_sum) == valid_tokens
    assert float(metrics.accuracy) == 1.0
    np.testing.assert_allclose(metrics.loss, expected, rtol=1e-5, atol=1e-6)


def test_dft_detached_probability_gradient_and_ignored_tokens_match_reference():
    logits_np = np.array([[[1.25, -0.5, 0.75], [0.25, 1.0, -1.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
    labels = jnp.asarray([[0, 2, -100]], dtype=jnp.int32)
    logits = jnp.asarray(logits_np)
    loss, gradients = jax.jit(jax.value_and_grad(lambda x: _dft_causal_lm_metrics(x, labels).loss))(logits)
    probs = np.exp(logits_np[0, 0].astype(np.float64) - 1.25)
    probs /= probs.sum()
    expected_grad = np.zeros_like(logits_np)
    expected_grad[0, 0] = probs[2] * (probs - np.array([0, 0, 1]))
    np.testing.assert_allclose(loss, -probs[2] * np.log(probs[2]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradients, expected_grad, rtol=1e-5, atol=1e-6)
    empty = _dft_causal_lm_metrics(logits, jnp.full_like(labels, -100))
    assert float(empty.loss) == 0.0
    assert float(empty.accuracy) == 0.0
    assert float(empty.weight_sum) == 0.0
