"""Explicit activation precision must survive expert conversion and dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelMoELinear
from easydel.layers.quantization import QuantizationConfig, QuantizationType


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
def test_standalone_expert_precision(mode):
    layer = ColumnParallelMoELinear(
        num_experts=3,
        in_features=32,
        out_features=16,
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(12),
    )
    config = QuantizationConfig.for_matmul(mode)
    q = layer.to_quantized(config)
    x = np.random.default_rng(5).normal(size=(8, 32)).astype(np.float32)
    sizes = jnp.array([3, 0, 5], jnp.int32)
    quantized_x = x
    if config.activation_bits != 16:
        bound = 7 if config.activation_bits == 4 else 127
        scale = np.max(np.abs(x), axis=1, keepdims=True) / bound
        quantized_x = np.clip(np.round(x / np.where(scale == 0, 1, scale)), -bound, bound) * scale
    codes = np.asarray(q.quant_kernel.value).astype(np.float32)
    scales = np.asarray(q.quant_scales.value)
    want = np.concatenate([quantized_x[:3] @ codes[0] * scales[0], quantized_x[3:] @ codes[2] * scales[2]])
    for fn in [lambda a: q(a, sizes), jax.jit(lambda a: q(a, sizes))]:
        np.testing.assert_allclose(fn(jnp.asarray(x)), want, rtol=2e-5, atol=2e-5)
    view = q.kernel_view()
    assert isinstance(view, tuple) and len(view) == 2
    assert view.activation_bits == config.activation_bits
    leaves, tree = jax.tree.flatten(view)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.activation_bits == config.activation_bits


def test_legacy_expert_kernel_view_stays_plain_tuple():
    layer = ColumnParallelMoELinear(
        num_experts=2,
        in_features=16,
        out_features=16,
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(12),
    )
    q = layer.to_quantized(QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=8))
    assert type(q.kernel_view()) is tuple
