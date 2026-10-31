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

"""KDA chunk-local dispatch: Pallas under a multi-device shard_map must match the XLA path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.operations.kernels.kda import _chunk_kda_per_channel_fwd
from jax.sharding import Mesh

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() < 4, reason="needs 4 TPU devices")

B, H, L, K, V = 2, 4, 256, 128, 128


def _inputs(seed):
    ks = jax.random.split(jax.random.PRNGKey(seed), 7)
    q = jax.random.normal(ks[0], (B, H, L, K))
    k = jax.random.normal(ks[1], (B, H, L, K))
    v = jax.random.normal(ks[2], (B, H, L, V))
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L)))
    g = -5.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (B, H, L, K)))
    s0 = jax.random.normal(ks[5], (B, H, K, V)) * 0.1
    ct = jax.random.normal(ks[6], (B, H, L, V))
    return q, k, v, b, g, s0, ct


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def test_non_dividing_heads_fall_back_to_xla_and_match():
    """H=3 does not divide tp=2: the dispatch must take the XLA path (no shard_map error), same result."""
    ks = jax.random.split(jax.random.PRNGKey(11), 5)
    Bx, Hx = 2, 3
    q = jax.random.normal(ks[0], (Bx, Hx, L, K))
    k = jax.random.normal(ks[1], (Bx, Hx, L, K))
    v = jax.random.normal(ks[2], (Bx, Hx, L, V))
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (Bx, Hx, L)))
    g = -5.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (Bx, Hx, L, K)))
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
    kw = dict(query=q, key=k, value=v, beta=b, decay=g, chunk_size=64, initial_state=None)
    o_x, s_x = jax.jit(lambda: _chunk_kda_per_channel_fwd(**kw))()
    with mesh:
        o_p, s_p = jax.jit(lambda: _chunk_kda_per_channel_fwd(**kw, kernel_mesh=mesh, kernel_lead_specs=("dp", "tp")))()
    assert _rel(o_p, o_x) < 1e-4 and _rel(s_p, s_x) < 1e-4


@pytest.mark.parametrize("packed", [False, True])
def test_sharded_pallas_matches_xla(packed):
    q, k, v, b, g, s0, ct = _inputs(seed=int(packed))
    seg = None
    if packed:
        seg = jnp.asarray([[0] * 100 + [1] * 150 + [-1] * 6, [0] * 256], jnp.int32)
        s0 = None
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))

    def run(**kw):
        def loss(q, k, v, b, g):
            o, s = _chunk_kda_per_channel_fwd(
                query=q, key=k, value=v, beta=b, decay=g, chunk_size=64, initial_state=s0, segment_ids=seg, **kw
            )
            return jnp.sum(o * ct) + jnp.sum(s), (o, s)

        return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))(q, k, v, b, g)

    (_, (o_x, s_x)), g_x = run()  # no mesh on a multi-device host -> XLA chunk-local
    with mesh:
        (_, (o_p, s_p)), g_p = run(kernel_mesh=mesh, kernel_lead_specs=("dp", "tp"))
    assert _rel(o_p, o_x) < 1e-4 and _rel(s_p, s_x) < 1e-4
    for got, want in zip(g_p, g_x, strict=True):
        assert np.isfinite(np.asarray(got)).all()
        assert _rel(got, want) < 1e-4
