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

"""Pallas mask-fused selected attention vs the dense masked reference (fwd + grads)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.sparse_selected_attention import dense_selected_attention_reference

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")


def _inputs(B, S, H, Hkv, Dk, Dv, seed, density):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(ks[0], (B, S, H, Dk), jnp.float32)
    k = jax.random.normal(ks[1], (B, S, Hkv, Dk), jnp.float32)
    v = jax.random.normal(ks[2], (B, S, Hkv, Dv), jnp.float32)
    mask = jax.random.uniform(ks[3], (B, S, S)) < density
    ct = jax.random.normal(ks[4], (B, S, H, Dv), jnp.float32)
    return q, k, v, mask, ct


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _grads(fn, q, k, v, ct):
    return jax.jit(jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v) * ct), argnums=(0, 1, 2)))(q, k, v)


@pytest.mark.parametrize(
    ("S", "H", "Hkv", "Dk", "Dv", "block", "density", "causal"),
    [
        (300, 4, 2, 192, 128, 128, 0.3, True),  # GQA, S not a tile multiple, Dk != Dv
        (512, 2, 2, 128, 128, 256, 0.5, True),
        (256, 4, 1, 128, 128, 128, 0.2, False),  # MQA, non-causal
        (200, 2, 2, 64, 64, 512, 0.4, True),  # block larger than S, small head dim
    ],
)
def test_pallas_matches_dense_reference(S, H, Hkv, Dk, Dv, block, density, causal):
    q, k, v, mask, ct = _inputs(2, S, H, Hkv, Dk, Dv, seed=S + H, density=density)
    scale = Dk**-0.5
    pallas = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)

    def sparse(q, k, v):
        return pallas(q, k, v, mask, scale, causal, block, block)

    def dense(q, k, v):
        return dense_selected_attention_reference(q, k, v, mask, scale, causal)

    assert _rel(jax.jit(sparse)(q, k, v), dense(q, k, v)) < 1e-4
    for got, want in zip(_grads(sparse, q, k, v, ct), _grads(dense, q, k, v, ct), strict=True):
        assert np.isfinite(np.asarray(got)).all()
        assert _rel(got, want) < 1e-4


@pytest.mark.parametrize(("block_q", "block_k"), [(256, 128), (128, 256), (128, 512)])
def test_unequal_tiles_match_dense_reference(block_q, block_k):
    """``block_q != block_k`` exercises the causal index-map clamps in the fwd / dq / dkv grids."""
    S, H, Hkv, Dk, Dv = 512, 2, 2, 128, 128
    q, k, v, mask, ct = _inputs(2, S, H, Hkv, Dk, Dv, seed=block_q + block_k, density=0.4)
    scale = Dk**-0.5
    pallas = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)

    def sparse(q, k, v):
        return pallas(q, k, v, mask, scale, True, block_q, block_k)

    def dense(q, k, v):
        return dense_selected_attention_reference(q, k, v, mask, scale, True)

    assert _rel(jax.jit(sparse)(q, k, v), dense(q, k, v)) < 1e-4
    for got, want in zip(_grads(sparse, q, k, v, ct), _grads(dense, q, k, v, ct), strict=True):
        assert np.isfinite(np.asarray(got)).all()
        assert _rel(got, want) < 1e-4


@pytest.mark.parametrize(("Dk", "Dv"), [(192, 128), (256, 256)])
def test_bf16_matches_f32_dense_reference(Dk, Dv):
    """Production dtype: bf16 q/k/v through the kernel vs the f32 dense reference (incl. glm5_next Dk=Dv=256)."""
    S, H, Hkv = 512, 4, 2
    q, k, v, mask, ct = _inputs(1, S, H, Hkv, Dk, Dv, seed=Dk + Dv, density=0.3)
    scale = Dk**-0.5
    pallas = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    qb, kb, vb = (x.astype(jnp.bfloat16) for x in (q, k, v))

    out = jax.jit(lambda q, k, v: pallas(q, k, v, mask, scale, True, 512, 512))(qb, kb, vb)
    ref = dense_selected_attention_reference(q, k, v, mask, scale, True)
    assert out.dtype == jnp.bfloat16
    assert _rel(np.asarray(out, np.float32), ref) < 3e-2

    grads = _grads(lambda q, k, v: pallas(q, k, v, mask, scale, True, 512, 512), qb, kb, vb, ct.astype(jnp.bfloat16))
    want = _grads(lambda q, k, v: dense_selected_attention_reference(q, k, v, mask, scale, True), q, k, v, ct)
    for got, w, primal in zip(grads, want, (qb, kb, vb), strict=True):
        assert got.dtype == primal.dtype
        assert np.isfinite(np.asarray(got, np.float32)).all()
        assert _rel(np.asarray(got, np.float32), w) < 5e-2


@pytest.mark.parametrize("platform", ["pallas", "xla"])
def test_valid_logits_below_old_sentinel_keep_their_mass(platform):
    """Finite logits far below -1e30 are still valid attention (r13 S01-1).

    Q = 1e15, K = -1e15 give dot products of about -1.28e32 (finite float32). With a finite -1e30
    running-max seed every probability underflowed to zero; the exact output is all ones for V = 1.
    """
    if platform == "pallas":
        fn = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    else:
        fn = kernel_registry.get("sparse_selected_attention", platform=Platform.XLA, backend=Backend.ANY)
    shape = (1, 128, 1, 128)
    q = jnp.full(shape, 1e15, jnp.float32)
    k = jnp.full(shape, -1e15, jnp.float32)
    v = jnp.ones(shape, jnp.float32)
    mask = jnp.ones((1, 128, 128), bool)
    out = np.asarray(fn(q, k, v, mask, 1.0, False, 128, 128))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.ones_like(out), rtol=1e-5, atol=1e-5)
    dv = jax.grad(lambda v: jnp.sum(fn(q, k, v, mask, 1.0, False, 128, 128)))(v)
    # Uniform attention over 128 keys: d(sum out)/dV[k] = sum_q 1/128 = 1 exactly. A float32
    # ``m + log(l)`` LSE rounded log(128) away and gave 128 here (r14 S01-1).
    np.testing.assert_allclose(np.asarray(dv), np.ones(shape, np.float32), rtol=1e-4, atol=1e-4)


def test_dense_reference_keeps_logits_below_old_sentinel():
    """The dense oracle must not let masked -1e30 entries dominate valid logits near -1e32 (r14 S04-2)."""
    q = jnp.full((1, 2, 1, 1), 1e16, jnp.float32)
    k = jnp.full((1, 2, 1, 1), -1e16, jnp.float32)
    v = jnp.ones((1, 2, 1, 1), jnp.float32)
    mask = jnp.eye(2, dtype=bool)[None]  # each row sees exactly one key
    out = np.asarray(dense_selected_attention_reference(q, k, v, mask, 1.0, False))
    np.testing.assert_allclose(out, np.ones_like(out), rtol=1e-6, atol=1e-6)
    empty = np.asarray(dense_selected_attention_reference(q, k, v, jnp.zeros((1, 2, 2), bool), 1.0, False))
    assert np.isfinite(empty).all() and float(np.abs(empty).max()) == 0.0


def test_rows_without_keys_are_zero():
    q, k, v, _, _ = _inputs(1, 256, 2, 2, 128, 128, seed=3, density=0.0)
    mask = jnp.zeros((1, 256, 256), bool).at[:, 200, 17].set(True)
    pallas = kernel_registry.get("sparse_selected_attention", platform=Platform.PALLAS, backend=Backend.TPU)
    out = pallas(q, k, v, mask, 128**-0.5, True, 128, 128)
    others = np.asarray(out)[:, np.arange(256) != 200]
    assert np.isfinite(np.asarray(out)).all() and float(np.abs(others).max()) == 0.0
    # The single selected pair is a softmax over one key: row 200 equals v[17] exactly.
    np.testing.assert_allclose(np.asarray(out)[:, 200], np.asarray(v)[:, 17], rtol=1e-4, atol=1e-5)
