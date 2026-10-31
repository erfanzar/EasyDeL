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

"""``chunk_impl="kda_chunk"`` (fused ejkernel kernel) vs the default per-channel chunk scan.

Values, final states and gradients on plain and packed rows (mixed segment lengths, a one-token segment, leading,
inner and trailing padding, both ``continue_initial_state`` modes), a 4-device ``shard_map`` case, and the
``KernelDeltaAttnOp`` seam.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.infra.sharding import CANONICAL_MESH_AXIS_NAMES
from easydel.operations import OperationMetadata
from easydel.operations.kernels.kda import (
    KernelDeltaAttnOp,
    _chunk_kda_per_channel_fused,
    _chunk_kda_per_channel_fwd,
    _recurrent_kda_per_channel_fwd,
)
from jax.sharding import Mesh

K = V = 128


def _inputs(B, H, L, seed, strong=True):
    ks = jax.random.split(jax.random.PRNGKey(seed), 6)
    q = jax.random.normal(ks[0], (B, H, L, K))
    k = jax.random.normal(ks[1], (B, H, L, K))
    v = jax.random.normal(ks[2], (B, H, L, V))
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (B, H, L)))
    n = jax.random.normal(ks[4], (B, H, L, K))
    g = -5.0 * jax.nn.sigmoid(n) if strong else -0.1 * jax.nn.softplus(n)
    s0 = jax.random.normal(ks[5], (B, H, K, V)) * 0.1
    return q, k, v, b, g, s0


def _segments(rows, L):
    out = []
    for spec in rows:
        ids = []
        for i, n in enumerate(spec):
            ids += [-1] * n if n < 0 else [i] * n
        out.append(ids + [-1] * (L - len(ids)))
    return jnp.asarray(out, jnp.int32)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


L = 200
# negative entries are padding runs: leading pad, mixed lengths incl. a 1-token segment, an inner pad, trailing pad
PACKED = [[-3, 5, 90, 3, 64, -6, 30], [1, 1, 130, 40]]


@pytest.mark.parametrize("continue_initial_state", [False, True], ids=["reset", "continue"])
@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_fused_matches_scan_values_and_grads(packed, continue_initial_state):
    q, k, v, b, g, s0 = _inputs(2, 2, L, seed=int(packed) + 2 * int(continue_initial_state))
    seg = _segments(PACKED, L) if packed else None
    w = jax.random.normal(jax.random.PRNGKey(9), (2, 2, L, V))
    kw = dict(chunk_size=64, segment_ids=seg, continue_initial_state=continue_initial_state)

    def run(fn):
        def loss(q, k, v, b, g, s0):
            o, s = fn(q, k, v, b, g, initial_state=s0, **kw)
            return jnp.sum(o * w) + jnp.sum(s), (o, s)

        return jax.jit(jax.value_and_grad(loss, argnums=range(6), has_aux=True))(q, k, v, b, g, s0)

    (_, (o_f, s_f)), g_f = run(_chunk_kda_per_channel_fused)
    (_, (o_s, s_s)), g_s = run(_chunk_kda_per_channel_fwd)
    assert _rel(o_f, o_s) < 1e-4 and _rel(s_f, s_s) < 1e-4
    o_r, s_r = _recurrent_kda_per_channel_fwd(
        q, k, v, b, g, initial_state=s0, segment_ids=seg, continue_initial_state=continue_initial_state
    )
    assert _rel(o_f, o_r) < 1e-4 and _rel(s_f, s_r) < 1e-4
    for a, c in zip(g_f, g_s, strict=True):
        assert np.isfinite(np.asarray(a)).all()
        assert _rel(a, c) < 1e-3


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "packed"])
def test_fused_with_gate_bound_matches_scan(packed):
    """``decay_lower_bound`` (factored MXU pairwise decay in the fused kernel) matches the scan's values and grads."""
    q, k, v, b, g, s0 = _inputs(2, 2, L, seed=20 + int(packed))
    assert float(g.min()) >= -5.0
    seg = _segments(PACKED, L) if packed else None
    w = jax.random.normal(jax.random.PRNGKey(19), (2, 2, L, V))

    def run(fn, **extra):
        def loss(q, k, v, b, g, s0):
            o, s = fn(q, k, v, b, g, initial_state=s0, chunk_size=64, segment_ids=seg, **extra)
            return jnp.sum(o * w) + jnp.sum(s), (o, s)

        return jax.jit(jax.value_and_grad(loss, argnums=range(6), has_aux=True))(q, k, v, b, g, s0)

    (_, (o_f, s_f)), g_f = run(_chunk_kda_per_channel_fused, decay_lower_bound=-5.0)
    (_, (o_s, s_s)), g_s = run(_chunk_kda_per_channel_fwd)
    assert _rel(o_f, o_s) < 1e-4 and _rel(s_f, s_s) < 1e-4
    for a, c in zip(g_f, g_s, strict=True):
        assert np.isfinite(np.asarray(a)).all()
        assert _rel(a, c) < 1e-3


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() < 4, reason="needs 4 TPU devices")
def test_fused_four_device_shard_map_matches_scan():
    q, k, v, b, g, s0 = _inputs(2, 4, L, seed=7)
    seg = _segments(PACKED, L)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("dp", "tp"))
    ct = jax.random.normal(jax.random.PRNGKey(3), (2, 4, L, V))

    def run(fn, **extra):
        def loss(q, k, v, b, g):
            o, s = fn(q, k, v, b, g, chunk_size=64, initial_state=s0, segment_ids=seg, **extra)
            return jnp.sum(o * ct) + jnp.sum(s), (o, s)

        return jax.jit(jax.value_and_grad(loss, argnums=range(5), has_aux=True))(q, k, v, b, g)

    (_, (o_s, s_s)), g_s = run(_chunk_kda_per_channel_fwd)
    with mesh:
        (_, (o_f, s_f)), g_f = run(_chunk_kda_per_channel_fused, kernel_mesh=mesh, kernel_lead_specs=("dp", "tp"))
    assert _rel(o_f, o_s) < 1e-4 and _rel(s_f, s_s) < 1e-4
    for a, c in zip(g_f, g_s, strict=True):
        assert np.isfinite(np.asarray(a)).all()
        assert _rel(a, c) < 1e-3


def test_op_seam_selects_fused_impl():
    q, k, v, b, g, s0 = _inputs(1, 2, 128, seed=11)
    seg = _segments([[40, -8, 80]], 128)
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1, 1, 1))
    mesh = Mesh(devices, CANONICAL_MESH_AXIS_NAMES)
    op = KernelDeltaAttnOp(
        OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, _stored_mesh=mesh)
    )
    # op layout is BTHD
    args = dict(
        query=q.transpose(0, 2, 1, 3),
        key=k.transpose(0, 2, 1, 3),
        value=v.transpose(0, 2, 1, 3),
        beta=b.transpose(0, 2, 1),
        decay=g.transpose(0, 2, 1, 3),
        recurrent_state=s0,
        segment_ids=seg,
        chunk_size=64,
    )
    fused = op(**args, chunk_impl="kda_chunk")
    scan = op(**args)
    assert _rel(fused.attention_outputs, scan.attention_outputs) < 1e-4
    assert _rel(fused.recurrent_state, scan.recurrent_state) < 1e-4
    bounded = op(**args, chunk_impl="kda_chunk", decay_lower_bound=-5.0)
    assert _rel(bounded.attention_outputs, scan.attention_outputs) < 1e-4
    assert _rel(bounded.recurrent_state, scan.recurrent_state) < 1e-4
    assert not np.array_equal(np.asarray(bounded.attention_outputs), np.asarray(fused.attention_outputs))
    with pytest.raises(ValueError, match="decay_lower_bound"):
        op(**args, chunk_impl="kda_chunk", decay_lower_bound=1.0)
    with pytest.raises(ValueError, match="chunk_impl"):
        op(**args, chunk_impl="twophase")
    with pytest.raises(ValueError, match="chunk_precision"):
        op(**args, chunk_impl="kda_chunk", chunk_precision="bf16")
