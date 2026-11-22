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

"""Pallas TPU short-row kernel of ``ragged_kda`` against the XLA short phase, and its in-place pool update.

Steps of short rows only (1..W raw tokens per row, conv fused) with read offsets, commit lengths, fresh rows, an empty
row and an out-of-range padding row: outputs, captured states and conv windows match the XLA reference
(``ragged_kda_xla``), every slot no row captures into stays bit for bit, and the compiled program of a step with
donated pools holds no pool-shaped copy.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.ragged_kda import ragged_kda_pallas
from ejkernel.kernels._xla.ragged_kda._xla_impl import ragged_kda_xla

K = V = 128
D_CONV = 4
LOWER = -5.0

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

# W = 4. Rows: lens, commit, read offset, init. Row 4 is empty, row 7 has an out-of-range base slot.
LENS = (4, 3, 1, 4, 0, 2, 4, 1)
COMMIT = (2, 3, 1, 1, 1, 1, 4, 1)
OFFSET = (1, 0, 0, 3, 0, 2, 0, 0)
INIT = (1, 1, 0, 1, 1, 1, 0, 1)
W = 4


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _case(heads: int, dtype, pool_dtype, seed: int = 0, pad: int = 3):
    rng = np.random.default_rng(seed)
    rows = len(LENS)
    num_slots = rows * W
    tokens = sum(LENS) + pad
    kd, vd = heads * K, heads * V
    perm = rng.permutation(rows)
    base = (perm * W).astype(np.int32)
    base[7] = num_slots + 5
    qsl = np.concatenate([[0], np.cumsum(LENS)]).astype(np.int32)

    def arr(shape, scale=1.0, dt=dtype):
        return jnp.asarray(scale * rng.normal(size=shape), jnp.float32).astype(dt)

    streams = dict(
        query=arr((tokens, heads, K)),
        key=arr((tokens, heads, K)),
        value=arr((tokens, heads, V)),
        beta=jax.nn.sigmoid(arr((tokens, heads), dt=jnp.float32)).astype(dtype),
        decay=LOWER * jax.nn.sigmoid(arr((tokens, heads, K), dt=jnp.float32)),
    )
    pools = dict(
        q_conv_state=arr((num_slots, kd, D_CONV), dt=pool_dtype),
        k_conv_state=arr((num_slots, kd, D_CONV), dt=pool_dtype),
        v_conv_state=arr((num_slots, vd, D_CONV), dt=pool_dtype),
        recurrent_state=arr((num_slots, heads, K, V), 0.1, dt=pool_dtype),
    )
    weights = dict(
        q_conv_weight=arr((kd, D_CONV), 0.5),
        k_conv_weight=arr((kd, D_CONV), 0.5),
        v_conv_weight=arr((vd, D_CONV), 0.5),
    )
    meta = dict(
        query_start_loc=jnp.asarray(qsl),
        state_indices=jnp.asarray(base),
        has_initial_state=jnp.asarray(INIT, bool),
        read_offset=jnp.asarray(OFFSET, jnp.int32),
        commit_lens=jnp.asarray(COMMIT, jnp.int32),
    )
    return streams, pools, weights, meta, base, num_slots


def _call(fn, streams, pools, weights, meta):
    return fn(
        streams["query"],
        streams["key"],
        streams["value"],
        streams["beta"],
        streams["decay"],
        pools["recurrent_state"],
        meta["query_start_loc"],
        meta["state_indices"],
        meta["has_initial_state"],
        meta["read_offset"],
        meta["commit_lens"],
        weights["q_conv_weight"],
        weights["k_conv_weight"],
        weights["v_conv_weight"],
        pools["q_conv_state"],
        pools["k_conv_state"],
        pools["v_conv_state"],
        short_row_max=W,
        decay_lower_bound=LOWER,
    )


def _captured_slots(base, num_slots):
    slots = set()
    for r, n in enumerate(LENS):
        if n == 0 or base[r] >= num_slots:
            continue
        c = min(max(COMMIT[r], 1), n)
        slots.update(int(base[r]) + j for j in range(n - c + 1))
    return slots


@pytest.mark.parametrize("heads", [4, 16])
@pytest.mark.parametrize(
    ("dtype", "pool_dtype", "tol"),
    [(jnp.float32, jnp.float32, 2e-5), (jnp.bfloat16, jnp.bfloat16, 2e-2)],
)
def test_short_rows_match_xla_short_phase(heads, dtype, pool_dtype, tol):
    streams, pools, weights, meta, base, num_slots = _case(heads, dtype, pool_dtype)
    got = jax.jit(lambda s, p, w, m: _call(ragged_kda_pallas, s, p, w, m))(streams, pools, weights, meta)
    want = jax.jit(lambda s, p, w, m: _call(ragged_kda_xla, s, p, w, m))(streams, pools, weights, meta)
    names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
    live = int(sum(LENS))
    out_g, out_w = np.asarray(got[0], np.float64), np.asarray(want[0], np.float64)
    assert _rel(out_g[:live], out_w[:live]) < tol
    assert not np.any(out_g[live:])
    captured = sorted(_captured_slots(base, num_slots))
    untouched = [s for s in range(num_slots) if s not in captured]
    for i, name in enumerate(names):
        g = np.asarray(got[1 + i].astype(jnp.float32), np.float64)
        w = np.asarray(want[1 + i].astype(jnp.float32), np.float64)
        assert _rel(g[captured], w[captured]) < tol, name
        before = np.asarray(pools[name].astype(jnp.float32), np.float64)
        np.testing.assert_array_equal(g[untouched], before[untouched], err_msg=name)


def test_step_with_donated_pools_has_no_pool_shaped_copy():
    streams, pools, weights, meta, _, _ = _case(4, jnp.bfloat16, jnp.bfloat16)

    def step(p, s, w, m):
        # The outputs keep the donated dict's structure, so each pool's output aliases that pool's buffer.
        _, cq, ck, cv, rec = _call(ragged_kda_pallas, s, p, w, m)
        return dict(q_conv_state=cq, k_conv_state=ck, v_conv_state=cv, recurrent_state=rec)

    text = jax.jit(step, donate_argnums=(0,)).lower(pools, streams, weights, meta).compile().as_text()
    shapes = {tuple(p.shape) for p in pools.values()} | {
        (p.shape[0], p.shape[2], p.shape[1]) for p in list(pools.values())[:3]
    }
    copies = []
    for line in text.splitlines():
        m = re.search(r"= \w+\[([\d,]+)\]\S* copy(-start)?\(", line)
        if m and tuple(int(d) for d in m.group(1).split(",")) in shapes:
            copies.append(line.strip()[:160])
    assert not copies, copies
