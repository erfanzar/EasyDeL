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

"""Per-channel KDA chunk-local terms: Pallas TPU kernel vs the XLA reference (fwd + grads)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")


def _inputs(lead, cs, kd, seed, strong_decay):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    k = jax.random.normal(ks[0], (*lead, cs, kd))
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    q = jax.random.normal(ks[1], (*lead, cs, kd))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * kd**-0.5
    kb = k * jax.nn.sigmoid(jax.random.normal(ks[2], (*lead, cs, 1)))
    n = jax.random.normal(ks[3], (*lead, cs, kd))
    g = -5.0 * jax.nn.sigmoid(n) if strong_decay else -jax.nn.softplus(n) * 0.1
    ct = jax.random.normal(ks[4], (*lead, cs, cs))
    ct2 = jax.random.normal(ks[5], (*lead, cs, cs))
    return kb, k, q, jnp.cumsum(g, axis=-2), ct, ct2


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


@pytest.mark.parametrize(
    ("lead", "cs", "sb", "strong_decay"),
    [
        ((4, 2, 3), 64, 16, True),  # GLM-strength decay (overflow guard)
        ((4, 2, 3), 64, 32, False),
        ((6,), 32, 16, True),
        ((2, 2), 64, 64, True),  # a single diagonal block, no off-diagonal matmuls
        # Weak decay keeps every off-diagonal pair O(1): all key-block iterations (jb = 0, 1, 2, ...)
        # contribute at full scale, so an off-by-one in the block range / reference row is visible.
        ((2, 2), 64, 16, False),
        ((2,), 128, 16, False),
    ],
)
def test_pallas_matches_xla(lead, cs, sb, strong_decay):
    kb, k, q, g, ct, ct2 = _inputs(lead, cs, 128, seed=cs + sb, strong_decay=strong_decay)
    pallas = kernel_registry.get("kda_chunk_local", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("kda_chunk_local", platform=Platform.XLA, backend=Backend.ANY)

    def loss(fn):
        def f(kb, k, q, g):
            a, b = fn(kb, k, q, g)
            return jnp.sum(a * ct) + jnp.sum(b * ct2)

        return jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))

    a1, b1 = jax.jit(lambda *x: pallas(*x, sub_block=sb))(kb, k, q, g)
    a0, b0 = xla(kb, k, q, g)
    assert _rel(a1, a0) < 1e-5 and _rel(b1, b0) < 1e-5
    # Far-from-diagonal band (rows in the second half, columns in the first) checked against its
    # own scale: the global-max normalisation above would hide errors there under strong decay.
    half = cs // 2
    band1, band0 = np.asarray(a1)[..., half:, :half], np.asarray(a0)[..., half:, :half]
    band_scale = float(np.abs(band0).max())
    if band_scale > 1e-12:
        assert float(np.abs(band1 - band0).max()) / band_scale < 1e-4
    # strictly-lower / lower-with-diagonal structure is exact
    iu = np.triu_indices(cs, 0)
    assert float(np.abs(np.asarray(a1)[..., iu[0], iu[1]]).max()) == 0.0
    g1 = loss(lambda *x: pallas(*x, sub_block=sb))(kb, k, q, g)
    g0 = loss(xla)(kb, k, q, g)
    for got, want in zip(g1, g0, strict=True):
        assert np.isfinite(np.asarray(got)).all()
        assert _rel(got, want) < 1e-5


def test_bf16_inputs_match_xla_and_keep_grad_dtype():
    kb, k, q, g, ct, ct2 = _inputs((2, 2), 64, 128, seed=7, strong_decay=False)
    kb, k, q = (x.astype(jnp.bfloat16) for x in (kb, k, q))
    pallas = kernel_registry.get("kda_chunk_local", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("kda_chunk_local", platform=Platform.XLA, backend=Backend.ANY)

    def grads(fn):
        def f(kb, k, q, g):
            a, b = fn(kb, k, q, g)
            return jnp.sum(a * ct) + jnp.sum(b * ct2)

        return jax.jit(jax.grad(f, argnums=(0, 1, 2, 3)))(kb, k, q, g)

    a1, b1 = jax.jit(lambda *x: pallas(*x, sub_block=16))(kb, k, q, g)
    a0, b0 = xla(kb, k, q, g)
    assert a1.dtype == jnp.float32 and b1.dtype == jnp.float32
    assert _rel(a1, a0) < 1e-3 and _rel(b1, b0) < 1e-3
    g1 = grads(lambda *x: pallas(*x, sub_block=16))
    g0 = grads(xla)
    for got, want, primal in zip(g1, g0, (kb, k, q, g), strict=True):
        assert got.dtype == primal.dtype
        assert np.isfinite(np.asarray(got, np.float32)).all()
        assert _rel(np.asarray(got, np.float32), np.asarray(want, np.float32)) < 3e-2


@pytest.mark.parametrize(("cs", "sub_block"), [(64, 4), (64, 12), (24, 16), (64, 0)])
def test_unaligned_sub_block_routes_to_xla(cs, sub_block):
    """``sub_block`` that is not a multiple of 8 (or not dividing ``cs``) must fall back, not lower."""
    kb, k, q, g, _, _ = _inputs((3,), cs, 128, seed=2, strong_decay=True)
    pallas = kernel_registry.get("kda_chunk_local", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("kda_chunk_local", platform=Platform.XLA, backend=Backend.ANY)
    a1, b1 = pallas(kb, k, q, g, sub_block=sub_block)
    a0, b0 = xla(kb, k, q, g)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a0))
    np.testing.assert_array_equal(np.asarray(b1), np.asarray(b0))


def test_untileable_head_dim_routes_to_xla():
    kb, k, q, g, _, _ = _inputs((3,), 32, 48, seed=1, strong_decay=True)
    pallas = kernel_registry.get("kda_chunk_local", platform=Platform.PALLAS, backend=Backend.TPU)
    xla = kernel_registry.get("kda_chunk_local", platform=Platform.XLA, backend=Backend.ANY)
    a1, b1 = pallas(kb, k, q, g)
    a0, b0 = xla(kb, k, q, g)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a0))
    np.testing.assert_array_equal(np.asarray(b1), np.asarray(b0))
