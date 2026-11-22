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

"""Block-sparse selected attention: outputs and gradients vs a dense masked reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.sparse_selected_attention import (
    dense_selected_attention_reference,
    sparse_selected_attention,
)


def _inputs(B, S, H, Hkv, D, seed, density):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    q = jax.random.normal(ks[0], (B, S, H, D), jnp.float32)
    k = jax.random.normal(ks[1], (B, S, Hkv, D), jnp.float32)
    v = jax.random.normal(ks[2], (B, S, Hkv, D), jnp.float32)
    mask = jax.random.uniform(ks[3], (B, S, S)) < density
    return q, k, v, mask, ks[4]


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def _block_structured_mask(B, S, block):
    """Selection confined to a few key blocks so whole tiles are empty (exercises skipping)."""
    nb = S // block
    keep = (np.arange(nb)[None, :] % 3 == 0) | (np.arange(nb)[None, :] == np.arange(nb)[:, None])
    m = np.kron(keep, np.ones((block, block), bool))
    return jnp.asarray(np.broadcast_to(m, (B, S, S)))


def _grads(fn, q, k, v, ct):
    return jax.jit(jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v) * ct), argnums=(0, 1, 2)))(q, k, v)


@pytest.mark.parametrize(
    ("S", "H", "Hkv", "block_q", "block_k", "density", "causal"),
    [
        (64, 4, 4, 16, 16, 0.3, True),
        (67, 4, 2, 16, 32, 0.2, True),  # S not a tile multiple, GQA, bq != bk
        (96, 2, 1, 32, 16, 0.5, False),  # MQA, non-causal
        (40, 4, 4, 512, 512, 0.25, True),  # tiles larger than S
    ],
)
def test_forward_and_grads_match_dense_reference(S, H, Hkv, block_q, block_k, density, causal):
    B, D = 2, 32
    q, k, v, mask, ck = _inputs(B, S, H, Hkv, D, seed=S + H, density=density)
    scale = D**-0.5
    ct = jax.random.normal(ck, (B, S, H, D), jnp.float32)

    def sparse(q, k, v):
        return sparse_selected_attention(q, k, v, mask, scale, causal, block_q, block_k)

    def dense(q, k, v):
        return dense_selected_attention_reference(q, k, v, mask, scale, causal)

    assert _rel(jax.jit(sparse)(q, k, v), dense(q, k, v)) < 1e-4
    for got, want in zip(_grads(sparse, q, k, v, ct), _grads(dense, q, k, v, ct), strict=True):
        assert _rel(got, want) < 1e-4


def test_skipped_tiles_match_dense_reference():
    B, S, H, D, blk = 1, 128, 2, 16, 16
    q, k, v, _, ck = _inputs(B, S, H, H, D, seed=7, density=1.0)
    mask = _block_structured_mask(B, S, blk)
    ct = jax.random.normal(ck, (B, S, H, D), jnp.float32)

    def sparse(q, k, v):
        return sparse_selected_attention(q, k, v, mask, D**-0.5, True, blk, blk)

    def dense(q, k, v):
        return dense_selected_attention_reference(q, k, v, mask, D**-0.5, True)

    assert _rel(sparse(q, k, v), dense(q, k, v)) < 1e-4
    for got, want in zip(_grads(sparse, q, k, v, ct), _grads(dense, q, k, v, ct), strict=True):
        assert _rel(got, want) < 1e-4


def test_query_without_keys_outputs_zero_and_finite_grads():
    B, S, H, D = 1, 32, 2, 16
    q, k, v, _, _ = _inputs(B, S, H, H, D, seed=3, density=0.0)
    mask = jnp.zeros((B, S, S), bool).at[:, 20, 5].set(True)

    def sparse(q, k, v):
        return sparse_selected_attention(q, k, v, mask, D**-0.5, True, 8, 8)

    out = sparse(q, k, v)
    assert float(jnp.abs(out[:, jnp.arange(S) != 20]).max()) == 0.0
    for x in _grads(sparse, q, k, v, jnp.ones_like(q)):
        assert np.isfinite(np.asarray(x)).all()


def test_bf16_close_to_f32():
    B, S, H, D = 2, 256, 4, 64
    q, k, v, mask, _ = _inputs(B, S, H, H, D, seed=11, density=0.2)
    f32 = sparse_selected_attention(q, k, v, mask, D**-0.5, True, 64, 64)
    bf = sparse_selected_attention(
        q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v.astype(jnp.bfloat16), mask, D**-0.5, True, 64, 64
    )
    assert bf.dtype == jnp.bfloat16
    assert _rel(bf.astype(jnp.float32), f32) < 3e-2
