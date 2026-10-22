"""BatchNorm forward: mode resolution and running-statistics update."""

import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.norms import BatchNorm


@pytest.mark.parametrize("axis", [-1, 1])
def test_batchnorm_training_normalizes_and_updates_running_stats(axis):
    bn = BatchNorm(num_features=4, axis=axis, momentum=0.5, rngs=spx.Rngs(0))
    host = np.arange(96, dtype=np.float32).reshape(8, 3, 4) * 0.5 + 3.0
    if axis == 1:
        host = np.swapaxes(host, 1, 2)
    reduction_axes = tuple(i for i in range(host.ndim) if i != axis % host.ndim)
    mean = host.mean(axis=reduction_axes, keepdims=True)
    var = host.var(axis=reduction_axes, keepdims=True)

    y = bn(jnp.asarray(host), use_running_average=False)

    assert y.shape == host.shape
    np.testing.assert_allclose(np.asarray(y), (host - mean) / np.sqrt(var + bn.epsilon), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(bn.mean.value), 0.5 * mean.reshape(-1), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(bn.var.value), 0.5 + 0.5 * var.reshape(-1), rtol=1e-4)
    running = bn(jnp.asarray(host), use_running_average=True)
    expected = (host - 0.5 * mean) / np.sqrt(0.5 + 0.5 * var + bn.epsilon)
    np.testing.assert_allclose(np.asarray(running), expected, rtol=1e-4, atol=1e-4)


def test_batchnorm_running_average_does_not_touch_stats():
    bn = BatchNorm(num_features=4, rngs=spx.Rngs(0))
    x = jnp.ones((8, 4), jnp.float32)
    before = np.asarray(bn.var.value).copy()

    y = bn(x, use_running_average=True)

    assert y.shape == x.shape
    np.testing.assert_array_equal(np.asarray(bn.var.value), before)


def test_batchnorm_without_any_mode_raises_value_error():
    bn = BatchNorm(num_features=4, rngs=spx.Rngs(0))
    bn.use_running_average = None
    with pytest.raises(ValueError, match="use_running_average"):
        bn(jnp.ones((2, 4), jnp.float32))
