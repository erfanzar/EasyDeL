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

"""Paged token selection preserves both rotary conventions and physical cache writes."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.indexer import TokenIndexer, TokenIndexerConfig, paged_token_select
from jax import numpy as jnp


class IdentityNorm(spx.Module):
    """Leave projected keys unchanged so the reference can isolate RoPE and selection."""

    def forward(self, x):
        return x


@pytest.mark.parametrize("interleaved", [False, True])
def test_paged_token_rope_cache_and_selection_match_reference(interleaved):
    heads, dim, rotary = 2, 6, 4
    indexer = TokenIndexer(
        TokenIndexerConfig(6, 12, heads, dim, 2, rotary, rope_interleave=interleaved),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1)),
        norm_factory=IdentityNorm,
    )
    indexer.wq_b.weight.value = jnp.eye(12)
    indexer.wk.weight.value = jnp.eye(6)
    head_kernel = np.array([[0.3, -0.2], [0.2, 0.1], [0.0, 0.4], [0.1, 0.0], [0.2, 0.2], [0.1, 0.3]], np.float32)
    indexer.kernels_proj.weight.value = jnp.asarray(head_kernel)
    rng = np.random.default_rng(12)
    hidden = rng.normal(size=(3, dim)).astype(np.float32)
    q_resid = rng.normal(size=(3, heads * dim)).astype(np.float32)
    angles = np.arange(6, dtype=np.float32).reshape(3, 2) * 0.4
    cos, sin = np.cos(angles), np.sin(angles)

    def rotate(x):
        prefix, tail = x[..., :rotary], x[..., rotary:]
        c, s = cos, sin
        if x.ndim == 3:
            c, s = c[:, None], s[:, None]
        if interleaved:
            first, second = prefix[..., ::2], prefix[..., 1::2]
            result = np.stack([first * c - second * s, second * c + first * s], axis=-1).reshape(prefix.shape)
        else:
            first, second = np.split(prefix, 2, axis=-1)
            result = np.concatenate([first * c - second * s, second * c + first * s], axis=-1)
        return np.concatenate([result, tail], axis=-1)

    queries, keys = rotate(q_resid.reshape(3, heads, dim)), rotate(hidden)
    weights = hidden @ head_kernel / np.sqrt(heads)
    scores = np.maximum(np.einsum("thd,kd->thk", queries, keys) / np.sqrt(dim), 0)
    scores = np.einsum("thk,th->tk", scores, weights)
    expected_ids = np.full((3, 2), -1, np.int32)
    for row in range(3):
        picks = np.argsort(-scores[row, : row + 1], kind="stable")[:2]
        expected_ids[row, : len(picks)] = picks
    # Logical pages [0, 1] map to physical pages [1, 0].
    expected_pages = np.zeros((2, 2, dim), np.float32)
    expected_pages[1] = keys[:2]
    expected_pages[0, 0] = keys[2]

    selected, pages = paged_token_select(
        indexer,
        jnp.asarray(hidden),
        jnp.asarray(q_resid),
        jnp.asarray(np.concatenate([cos, sin], axis=-1)),
        jnp.zeros((2, 2, dim), jnp.float32),
        jnp.asarray([[1, 0]], jnp.int32),
        jnp.asarray([3], jnp.int32),
        jnp.asarray([0, 3], jnp.int32),
        jnp.asarray(1, jnp.int32),
        key_chunk=2,
    )
    np.testing.assert_array_equal(np.asarray(selected), expected_ids)
    np.testing.assert_allclose(np.asarray(pages), expected_pages, rtol=2e-5, atol=2e-5)
