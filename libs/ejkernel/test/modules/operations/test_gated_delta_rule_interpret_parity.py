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

"""CPU (interpret-mode) parity lock for the packed grouped chunked GDR Pallas path.

Runs the real Pallas TPU kernels under ``force_tpu_interpret_mode`` against the
XLA recurrent reference. The shapes are chosen to be discriminating, not small:

* ``num_chunks == 8`` so the fused-group scheduling (``_N_FUSE``) is active —
  a segmented backward that consumes stale intra-group states passes at
  ``num_chunks < _N_FUSE`` and only fails here;
* grouped heads (``Hq != Hv``) so the grouped streaming path is exercised;
* segment boundaries that do NOT align with chunk boundaries, plus a ``-1``
  padding tail, so intra-chunk masking and tail handling are both crossed.

Forward-only checks cannot catch the stale-state class of bug (the forward is
exact either way); the gradient comparison is the load-bearing assertion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from ejkernel.kernels._pallas.tpu.gated_delta_rule._interface import gated_delta_rule
from jax.experimental.pallas import tpu as pltpu


def _make_packed_grouped_inputs(seg_period: int, batch=2, seq_len=1024, hq=2, hv=6, k_dim=16, v_dim=16):
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(keys[0], (batch, seq_len, hq, k_dim), jnp.float32)
    k = jax.random.normal(keys[1], (batch, seq_len, hq, k_dim), jnp.float32)
    v = jax.random.normal(keys[2], (batch, seq_len, hv, v_dim), jnp.float32)
    beta = jax.nn.sigmoid(jax.random.normal(keys[3], (batch, seq_len, hv), jnp.float32))
    decay = -jax.nn.softplus(jax.random.normal(keys[4], (batch, seq_len, hv), jnp.float32))
    # packed documents of seg_period tokens with the final 100 positions padding
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    seg_ids = jnp.where(positions < seq_len - 100, positions // seg_period, -1)
    seg_ids = jnp.broadcast_to(seg_ids, (batch, seq_len))
    return q, k, v, beta, decay, seg_ids


@pytest.mark.parametrize(
    "seg_period",
    [
        # documents span multiple chunks inside one fused group (historical
        # stale-state gradient bug: ~26% relative grad error before the fix)
        pytest.param(512, id="docs_span_chunks_in_group"),
        # boundaries fall mid-chunk (320 % 128 != 0): intra-chunk masking
        pytest.param(320, id="misaligned_boundaries"),
    ],
)
def test_packed_grouped_chunked_matches_recurrent_fwd_and_grad(seg_period):
    q, k, v, beta, decay, seg_ids = _make_packed_grouped_inputs(seg_period)
    chunk_size = 128

    def run(use_chunked, q, k, v, beta, decay):
        return gated_delta_rule(q, k, v, beta, decay, chunk_size=chunk_size, seg_ids=seg_ids, use_chunked=use_chunked)

    with pltpu.force_tpu_interpret_mode():
        out_pallas, state_pallas = run(True, q, k, v, beta, decay)
    out_ref, state_ref = run(False, q, k, v, beta, decay)

    assert jnp.max(jnp.abs(out_pallas - out_ref)) < 1e-5
    assert jnp.max(jnp.abs(state_pallas - state_ref)) < 1e-5

    def loss(use_chunked, *ops):
        out, _ = run(use_chunked, *ops)
        return jnp.mean(jnp.square(out))

    with pltpu.force_tpu_interpret_mode():
        grads_pallas = jax.grad(lambda *ops: loss(True, *ops), argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    grads_ref = jax.grad(lambda *ops: loss(False, *ops), argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)

    for name, gp, gr in zip(("dq", "dk", "dv", "dbeta", "ddecay"), grads_pallas, grads_ref, strict=True):
        scale = jnp.maximum(jnp.max(jnp.abs(gr)), 1e-6)
        rel = jnp.max(jnp.abs(gp - gr)) / scale
        assert rel < 5e-5, f"{name}: relative grad diff {float(rel):.3e} vs recurrent reference"
