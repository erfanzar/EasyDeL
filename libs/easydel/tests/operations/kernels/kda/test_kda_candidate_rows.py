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

"""Speculative candidate rows of the packed-row KDA serving driver (``kda_packed_rows_forward``).

With a pool of ``rows * (1 + C)`` slots, a verify window must leave, for every speculative row and prefix ``p``, the
exact live state (recurrent + q/k/v conv windows) that the plain one-row-per-slot path reaches after the first ``p``
tokens, in prefix-major row ``rows + (p - 1) * rows + row``. A ``spec_recurrent_commit`` must copy the selected
candidate row into the live row before the next step; an inactive commit must leave it untouched.
"""

import jax
import jax.numpy as jnp
import numpy as np
from easydel.caching import KDACacheView
from easydel.caching.kda.cache import KDACacheConfig
from easydel.operations.kernels.kda import kda_packed_rows_forward
from spectrax import PartitionAxis

ROWS, H, HK, HV, DC, T = 2, 2, 8, 8, 4, 8
KEY_DIM, VALUE_DIM = H * HK, H * HV
CAND = 3
LEAVES = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")


def _view(pool, base_leaves):
    cfg = KDACacheConfig.create(
        num_hidden_layers=1,
        partition_axis=PartitionAxis(),
        batch_size=pool,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        d_conv=DC,
        recurrent_state_shape=(H, HK, HV),
    )
    leaves = {}
    for name, base in zip(LEAVES, base_leaves, strict=True):
        extra = pool - ROWS
        filler = jnp.full((extra, *base.shape[1:]), 7.0, base.dtype)  # garbage: must never leak into live rows
        leaves[name] = jnp.concatenate([base, filler], axis=0) if extra else base
    return KDACacheView(**leaves, positions=jnp.zeros((pool,), jnp.int32), metadata=cfg, layer_index=0)


def _inputs(seed):
    ks = jax.random.split(jax.random.PRNGKey(seed), 10)
    tok = {
        "q": jax.random.normal(ks[0], (ROWS, T, KEY_DIM)),
        "k": jax.random.normal(ks[1], (ROWS, T, KEY_DIM)),
        "v": jax.random.normal(ks[2], (ROWS, T, VALUE_DIM)),
        "b": jax.nn.sigmoid(jax.random.normal(ks[3], (ROWS, T, H))),
        "g": -5.0 * jax.nn.sigmoid(jax.random.normal(ks[4], (ROWS, T, H, HK))),
    }
    kernels = tuple(0.5 * jax.random.normal(ks[5 + i], (d, DC)) for i, d in enumerate((KEY_DIM, KEY_DIM, VALUE_DIM)))
    base = (
        jax.random.normal(ks[8], (ROWS, KEY_DIM, DC)),
        jax.random.normal(jax.random.fold_in(ks[8], 1), (ROWS, KEY_DIM, DC)),
        jax.random.normal(jax.random.fold_in(ks[8], 2), (ROWS, VALUE_DIM, DC)),
        0.1 * jax.random.normal(ks[9], (ROWS, H, HK, HV)),
    )
    return tok, kernels, base


def _pack(tok, lens):
    """Packed [1, T, ...] stream holding row r's first lens[r] tokens back to back."""
    idx_r, idx_t = [], []
    for r, n in enumerate(lens):
        idx_r += [r] * n
        idx_t += list(range(n))
    pad = T - len(idx_r)
    out = {}
    for name, x in tok.items():
        g = x[jnp.asarray(idx_r, jnp.int32), jnp.asarray(idx_t, jnp.int32)] if idx_r else x[0, :0]
        out[name] = jnp.concatenate([g, jnp.zeros((pad, *x.shape[2:]), x.dtype)], axis=0)[None]
    qsl = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)
    return out, qsl


def _run(view, tok, kernels, lens, context, commit=None):
    packed, qsl = _pack(tok, lens)
    return kda_packed_rows_forward(
        query=packed["q"],
        key=packed["k"],
        value=packed["v"],
        beta=packed["b"],
        decay=packed["g"],
        conv_kernels=kernels,
        cache_view=view,
        query_start_loc=qsl,
        context_lens=jnp.asarray(context, jnp.int32),
        conv_dtype=jnp.float32,
        chunk_size=16,
        spec_recurrent_commit=commit,
    )


def test_candidate_rows_equal_plain_prefix_states_and_commit_restores_them():
    tok, kernels, base = _inputs(0)
    lens = [3, 2]
    context = [n + 5 for n in lens]  # rows continue from carried context (speculative verify windows)
    pool_view = _view(ROWS * (1 + CAND), base)
    out_c, after = _run(pool_view, tok, kernels, lens, context)

    # Live rows and outputs match the candidate-free path exactly.
    plain_view = _view(ROWS, base)
    out_p, plain_after = _run(plain_view, tok, kernels, lens, context)
    np.testing.assert_allclose(np.asarray(out_c), np.asarray(out_p), rtol=0, atol=1e-6)
    for name in LEAVES:
        np.testing.assert_allclose(
            np.asarray(getattr(after, name)[:ROWS]), np.asarray(getattr(plain_after, name)), rtol=0, atol=1e-6
        )

    # Every candidate row == the live state of the plain path after that prefix of that row alone.
    for r, n in enumerate(lens):
        for p in range(1, n + 1):
            sub = [p if i == r else 0 for i in range(ROWS)]
            ctx = [context[i] - lens[i] + p if i == r else 1 for i in range(ROWS)]
            _, ref = _run(_view(ROWS, base), tok, kernels, sub, ctx)
            row = ROWS + (p - 1) * ROWS + r
            for name in LEAVES:
                np.testing.assert_allclose(
                    np.asarray(getattr(after, name)[row]),
                    np.asarray(getattr(ref, name)[r]),
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"{name} row={r} prefix={p}",
                )

    # Commit prefix 2 for row 0 and prefix 1 for row 1 on an empty step: live rows <- those candidate rows.
    commit = jnp.asarray([[1, 1], [2, 1]], jnp.int32)
    _, committed = _run(after, tok, kernels, [0, 0], [1, 1], commit=commit)
    for name in LEAVES:
        leaf = getattr(after, name)
        np.testing.assert_array_equal(np.asarray(getattr(committed, name)[0]), np.asarray(leaf[ROWS + 1 * ROWS + 0]))
        np.testing.assert_array_equal(np.asarray(getattr(committed, name)[1]), np.asarray(leaf[ROWS + 0 * ROWS + 1]))

    # Negative control: an inactive commit leaves the live rows untouched.
    _, untouched = _run(after, tok, kernels, [0, 0], [1, 1], commit=jnp.asarray([[0, 0], [2, 1]], jnp.int32))
    for name in LEAVES:
        np.testing.assert_array_equal(
            np.asarray(getattr(untouched, name)[:ROWS]), np.asarray(getattr(after, name)[:ROWS])
        )


def test_non_speculative_rows_keep_their_candidate_rows():
    """Rows without prior context (prefill) or longer than C must not overwrite their candidate rows."""
    tok, kernels, base = _inputs(1)
    pool_view = _view(ROWS * (1 + CAND), base)
    lens = [2, 5]  # row 0: fresh prefill (no context); row 1: longer than CAND
    _, after = _run(pool_view, tok, kernels, lens, [2, 9])
    for name in LEAVES:
        np.testing.assert_array_equal(np.asarray(getattr(after, name)[ROWS:]), 7.0)
