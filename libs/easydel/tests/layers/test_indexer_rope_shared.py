# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Indexer RoPE helpers agree with explicit reference rotations (accelerator test)."""

import numpy as np
import pytest
from easydel.layers.indexer import (
    IndexerSelection,
    apply_indexer_rope,
    apply_partial_rope,
    paged_block_topk,
)
from jax import numpy as jnp


def _tables(batch=1, seq=3, rot=4):
    ang = np.arange(batch * seq * rot, dtype=np.float32).reshape(batch, seq, rot) * 0.1
    return jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))


def _split_half_ref(x, cos, sin):
    rot = cos.shape[-1]
    half = rot // 2
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    out = np.concatenate(
        [x1 * cos[..., :half] - x2 * sin[..., :half], x2 * cos[..., half:] + x1 * sin[..., half:]], axis=-1
    )
    return np.concatenate([out, x_pass], axis=-1)


def test_split_half_partial_matches_reference():
    x = np.random.default_rng(0).normal(size=(1, 3, 8)).astype(np.float32)
    cos, sin = _tables()
    expected = _split_half_ref(x, np.asarray(cos), np.asarray(sin))
    np.testing.assert_allclose(np.asarray(apply_indexer_rope(jnp.asarray(x), cos, sin)), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(apply_partial_rope(jnp.asarray(x), cos, sin)), expected, rtol=1e-5, atol=1e-5)


def test_interleaved_matches_reference():
    x = np.random.default_rng(1).normal(size=(1, 3, 8)).astype(np.float32)
    cos, sin = _tables(rot=2)
    c, s = np.asarray(cos), np.asarray(sin)
    x1, x2 = x[..., :4:2], x[..., 1:4:2]
    rotated = np.stack([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).reshape(1, 3, 4)
    expected = np.concatenate([rotated, x[..., 4:]], axis=-1)
    got = apply_indexer_rope(jnp.asarray(x), cos, sin, style="interleaved")
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu_ok
def test_public_exports_present():
    assert callable(paged_block_topk)


def test_to_bias_ste_gradient_only_on_selected():
    import jax

    sel = IndexerSelection(jnp.asarray([[[0, 2]]], jnp.int32), score_proxy=jnp.ones((1, 1, 4), jnp.float32))

    def total(proxy):
        s = IndexerSelection(sel.indices, score_proxy=proxy)
        return jnp.sum(jnp.where(jnp.isfinite(s.to_bias(4)), s.to_bias(4), 0.0))

    grad = np.asarray(jax.grad(total)(jnp.ones((1, 1, 4), jnp.float32)))
    np.testing.assert_array_equal(grad, [[[1.0, 0.0, 1.0, 0.0]]])
