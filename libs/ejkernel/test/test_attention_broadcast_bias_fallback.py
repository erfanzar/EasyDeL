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

"""Independent bias axes must survive the default Pallas-to-standard-XLA path."""

import itertools

import jax
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ring_attention import ring_attention as pallas_ring
from ejkernel.kernels._xla.attention import attention
from jax import numpy as jnp

# All combinations of singleton/full batch, query and key axes; heads are
# independently singleton, KV heads or query heads (GQA).
_BIAS_AXES = list(itertools.product((1, 2), (1, 4), (1, 6)))


def _call(api, q, k, v, bias):
    if api == "pallas_default":
        # Intentionally no chunk_size, precision, metadata or sinks: these
        # would select a different fallback and hide the default-path defect.
        return pallas_ring(q, k, v, bias=bias)
    kwargs = {"bias": bias} if api == "xla" else {"init_bias": lambda: bias}
    return attention(q, k, v, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, **kwargs)[0]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("api", ["pallas_default", "xla", "xla_lazy"])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("axes", _BIAS_AXES)
def test_independent_bias_axes_abstract_output_and_vjp(api, heads, axes):
    """Trace public forward/VJP shapes without compiling or computing arrays."""
    batch, rows, columns = axes
    q = jax.ShapeDtypeStruct((2, 4, 4, 8), jnp.float32)
    kv = jax.ShapeDtypeStruct((2, 6, 2, 8), jnp.float32)
    bias = jax.ShapeDtypeStruct((batch, heads, rows, columns), jnp.float32)

    def forward(q, k, v, b):
        return _call(api, q, k, v, b)

    output = jax.eval_shape(forward, q, kv, kv, bias)
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    gradient = jax.grad(lambda q, k, v, b: forward(q, k, v, b).sum(), argnums=(0, 1, 2, 3))
    shapes = jax.eval_shape(gradient, q, kv, kv, bias)
    assert tuple(x.shape for x in shapes) == (q.shape, kv.shape, kv.shape, bias.shape)
    assert all(x.dtype == jnp.float32 for x in shapes)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("api", ["pallas_default", "xla", "xla_lazy"])
@pytest.mark.parametrize("shape", [(3, 1, 4, 6), (1, 1, 5, 6), (1, 1, 4, 7)])
def test_nonbroadcastable_bias_axes_rejected_abstract(api, shape):
    """Independent annotations do not silently accept nonbroadcastable axes."""
    q = jax.ShapeDtypeStruct((2, 4, 4, 8), jnp.float32)
    kv = jax.ShapeDtypeStruct((2, 6, 2, 8), jnp.float32)
    bias = jax.ShapeDtypeStruct(shape, jnp.float32)
    with pytest.raises(ValueError, match="broadcast"):
        jax.eval_shape(lambda q, k, v, b: _call(api, q, k, v, b), q, kv, kv, bias)


@pytest.mark.parametrize("api", ["pallas_default", "xla", "xla_lazy"])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("axes", _BIAS_AXES)
def test_independent_bias_axes_match_dense_reference(api, heads, axes):
    """Default fallback outputs and float32 XLA Q/K/V/bias gradients match dense GQA."""
    rng = np.random.default_rng(519)
    q = jnp.asarray(rng.normal(size=(2, 4, 4, 8)).astype(np.float32) * 0.25)
    k = jnp.asarray(rng.normal(size=(2, 6, 2, 8)).astype(np.float32) * 0.25)
    v = jnp.asarray(rng.normal(size=k.shape).astype(np.float32))
    batch, rows, columns = axes
    bias = jnp.asarray(rng.normal(size=(batch, heads, rows, columns)).astype(np.float32) * 0.125)
    cotangent = jnp.asarray(rng.normal(size=q.shape).astype(np.float32))

    def dense(q, k, v, bias):
        # The default Pallas standard-attention route retains its established
        # bf16 compute dtype and fp32 softmax, even with float32 inputs.
        dtype = jnp.bfloat16 if api == "pallas_default" else jnp.float32
        q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
        k, v = jnp.repeat(k, 2, axis=2), jnp.repeat(v, 2, axis=2)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q * (8**-0.5), k, precision=jax.lax.Precision.HIGHEST)
        if heads == 2:
            bias = jnp.repeat(bias, 2, axis=1)
        logits = logits + bias.astype(dtype)
        probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", probabilities, v, precision=jax.lax.Precision.HIGHEST).astype(jnp.float32)

    def actual(q, k, v, b):
        return _call(api, q, k, v, b)

    with jax.default_matmul_precision("highest"):
        observed = jax.jit(actual)(q, k, v, bias)
        # The default Pallas route computes in bf16 (ulp 2**-7 near 1), so its
        # accumulation order differs from the dense bf16 reference by up to about
        # half an ulp; the float32 XLA routes are held to float32 tolerance.
        tolerance = {"atol": 5e-3, "rtol": 1e-2} if api == "pallas_default" else {"atol": 2e-5, "rtol": 2e-5}
        np.testing.assert_allclose(observed, dense(q, k, v, bias), **tolerance)
        assert observed.dtype == jnp.float32
        # The default fallback deliberately quantizes contractions to bf16;
        # strict float32 gradient parity belongs to the explicit-float32 APIs.
        if api != "pallas_default":
            observed_grad = jax.grad(lambda *args: (actual(*args) * cotangent).sum(), argnums=(0, 1, 2, 3))
            expected_grad = jax.grad(lambda *args: (dense(*args) * cotangent).sum(), argnums=(0, 1, 2, 3))
            for got, want in zip(observed_grad(q, k, v, bias), expected_grad(q, k, v, bias), strict=True):
                np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)
