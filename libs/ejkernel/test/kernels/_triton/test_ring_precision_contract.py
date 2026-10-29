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

"""Direct registered Triton calls must honor precision before fused limitations."""

import jax
import numpy as np
import pytest
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._triton.ring_attention import ring_attention as triton_ring
from jax import numpy as jnp


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("per_head_sinks", [False, True])
def test_direct_registered_highest_precision_bias_sinks_and_qkv_gradients(compiled, per_head_sinks):
    """The XLA precision fallback accepts broadcast bias and per-head sink matrices."""
    rng = np.random.default_rng(517)
    q = jnp.asarray(rng.normal(size=(2, 4, 4, 8)).astype(np.float32) * 0.3)
    k = jnp.asarray(rng.normal(size=(2, 6, 2, 8)).astype(np.float32) * 0.3)
    v = jnp.asarray(rng.normal(size=(2, 6, 2, 8)).astype(np.float32))
    bias = jnp.asarray(rng.normal(size=(1, 1, 4, 6)).astype(np.float32) * 0.2)
    sink_shape = (4, 2) if per_head_sinks else (2,)
    sinks = jnp.asarray(rng.normal(size=sink_shape).astype(np.float32))
    cotangent = jnp.asarray(rng.normal(size=q.shape).astype(np.float32))
    registered = kernel_registry.get("ring_attention", platform=Platform.TRITON, backend=Backend.GPU)

    def actual(q, k, v):
        return registered(
            q,
            k,
            v,
            bias=bias,
            softmax_aux=sinks,
            chunk_size=2,
            causal=True,
            softmax_scale=0.5,
            precision=jax.lax.Precision.HIGHEST,
        )

    def reference(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, jnp.repeat(k, 2, axis=2), precision=jax.lax.Precision.HIGHEST)
        logits = logits * 0.5 + bias
        visible = jnp.arange(4)[:, None] >= jnp.arange(6)[None, :]
        logits = jnp.where(visible, logits, -jnp.inf)
        head_sinks = sinks[:, None, :] if per_head_sinks else sinks
        sink_logits = jnp.broadcast_to(head_sinks, (*logits.shape[:-1], 2))
        weights = jax.nn.softmax(jnp.concatenate((logits, sink_logits), axis=-1), axis=-1)[..., :6]
        return jnp.einsum("bhqk,bkhd->bqhd", weights, jnp.repeat(v, 2, axis=2), precision=jax.lax.Precision.HIGHEST)

    actual = jax.jit(actual) if compiled else actual
    np.testing.assert_allclose(actual(q, k, v), reference(q, k, v), atol=2e-5, rtol=2e-5)
    # Also exercise the exported entry point rather than just the registry wrapper.
    np.testing.assert_allclose(
        triton_ring(
            q,
            k,
            v,
            bias=bias,
            softmax_aux=sinks,
            chunk_size=2,
            causal=True,
            softmax_scale=0.5,
            precision=jax.lax.Precision.HIGHEST,
        ),
        reference(q, k, v),
        atol=2e-5,
        rtol=2e-5,
    )
    actual_grad = jax.grad(lambda q, k, v: (actual(q, k, v) * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    expected_grad = jax.grad(lambda q, k, v: (reference(q, k, v) * cotangent).sum(), argnums=(0, 1, 2))(q, k, v)
    for result, target in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(result, target, atol=2e-5, rtol=2e-5)
