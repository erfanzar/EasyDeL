# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Broadcast bias heads must not constrain the returned query-head weights."""

import jax
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from jax import numpy as jnp

_HEAD_LAYOUTS = [(2, 2, 1), (2, 2, 2), (4, 2, 1), (4, 2, 2), (4, 2, 4), (4, 1, 1), (4, 1, 4)]


def _attention(q, k, v, bias, lazy, dtype):
    kernel = kernel_registry.get("attention", platform=Platform.XLA, backend=Backend.ANY)
    kwargs = {"init_bias": lambda: bias} if lazy else {"bias": bias}
    return kernel(q, k, v, dtype=dtype, softmax_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, **kwargs)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("heads", _HEAD_LAYOUTS)
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_registered_attention_broadcast_bias_abstract_return_contract(heads, lazy, dtype):
    """Abstract tracing checks the real registered wrapper without executing numerics."""
    hq, hkv, hb = heads
    q = jax.ShapeDtypeStruct((2, 3, hq, 8), jnp.float32)
    k = jax.ShapeDtypeStruct((2, 5, hkv, 8), jnp.float32)
    v = jax.ShapeDtypeStruct((2, 5, hkv, 6), jnp.float32)
    bias = jax.ShapeDtypeStruct((2, hb, 3, 5), jnp.float32)
    output, weights = jax.eval_shape(lambda q, k, v, b: _attention(q, k, v, b, lazy, dtype), q, k, v, bias)
    assert output.shape == (2, 3, hq, 6)
    assert weights.shape == (2, hq, 3, 5)
    assert output.dtype == weights.dtype == dtype


@pytest.mark.parametrize("heads", _HEAD_LAYOUTS)
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_registered_attention_broadcast_bias_dense_output_weights_and_gradients(heads, lazy, compiled):
    """MHA/GQA/MQA share the same independently expanded-head dense reference."""
    hq, hkv, hb = heads
    rng = np.random.default_rng(514)
    q = jnp.asarray(rng.normal(size=(2, 3, hq, 8)).astype(np.float32) * 0.3)
    k = jnp.asarray(rng.normal(size=(2, 5, hkv, 8)).astype(np.float32) * 0.3)
    v = jnp.asarray(rng.normal(size=(2, 5, hkv, 6)).astype(np.float32))
    bias = jnp.asarray(rng.normal(size=(2, hb, 3, 5)).astype(np.float32) * 0.2)
    cotangent = jnp.asarray(rng.normal(size=(2, 3, hq, 6)).astype(np.float32))

    def actual(q, k, v):
        return _attention(q, k, v, bias, lazy, jnp.float32)

    def reference(q, k, v):
        expanded_k = jnp.repeat(k, hq // hkv, axis=2)
        expanded_v = jnp.repeat(v, hq // hkv, axis=2)
        expanded_bias = jnp.repeat(bias, hq // hb, axis=1)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q * (8**-0.5), expanded_k, precision=jax.lax.Precision.HIGHEST)
        weights = jax.nn.softmax(logits + expanded_bias, axis=-1)
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, expanded_v, precision=jax.lax.Precision.HIGHEST)
        return output, weights

    actual = jax.jit(actual) if compiled else actual
    observed = actual(q, k, v)
    expected = reference(q, k, v)
    for result, target in zip(observed, expected, strict=True):
        np.testing.assert_allclose(result, target, rtol=2e-5, atol=2e-5)
    assert observed[1].shape == (2, hq, 3, 5)

    def loss(fn, q, k, v):
        return (fn(q, k, v)[0] * cotangent).sum()

    actual_grad = jax.grad(lambda q, k, v: loss(actual, q, k, v), argnums=(0, 1, 2))(q, k, v)
    expected_grad = jax.grad(lambda q, k, v: loss(reference, q, k, v), argnums=(0, 1, 2))(q, k, v)
    for result, target in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(result, target, rtol=2e-5, atol=2e-5)
