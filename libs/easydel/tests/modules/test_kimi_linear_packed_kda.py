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

"""eSurge packed-row serving for Kimi Linear KDA layers.

eSurge feeds every layer one flattened token stream ``[1, T, D]`` with per-request
segments in ``cache_metadata.query_start_loc`` and one KDA cache slot per request.
Each request's outputs and final conv/recurrent state must equal running the layer on
that request alone; treating the stream as one sequence (the previous behaviour) mixes
requests through the shared recurrence.
"""

from types import SimpleNamespace

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.caching import KDACacheView

HIDDEN, HEADS, HEAD_DIM, D_CONV = 32, 2, 8, 4


def _layer():
    config = ed.KimiLinearConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
            "short_conv_kernel_size": D_CONV,
        },
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = ed.KimiLinearForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    layer = model.model.layers[0].self_attn
    # Non-trivial decay/gate parameters so the recurrence is exercised.
    layer.A_log.value = jnp.asarray(np.random.default_rng(1).normal(size=layer.A_log.value.shape), jnp.float32)
    return model, layer


def _empty_view(rows):
    width = HEADS * HEAD_DIM
    return KDACacheView(
        q_conv_state=jnp.zeros((rows, width, D_CONV), jnp.float32),
        k_conv_state=jnp.zeros((rows, width, D_CONV), jnp.float32),
        v_conv_state=jnp.zeros((rows, width, D_CONV), jnp.float32),
        recurrent_state=jnp.zeros((rows, HEADS, HEAD_DIM, HEAD_DIM), jnp.float32),
        positions=jnp.zeros((rows,), jnp.int32),
        metadata=None,
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_packed_rows_match_per_request():
    model, layer = _layer()
    lens = (7, 12)
    rng = np.random.default_rng(0)
    requests = [jnp.asarray(rng.normal(size=(1, n, HIDDEN)), jnp.float32) for n in lens]
    bucket = 32  # padded token bucket, as the runner compiles it
    packed = jnp.zeros((1, bucket, HIDDEN), jnp.float32)
    packed = packed.at[:, : lens[0]].set(requests[0][0]).at[:, lens[0] : sum(lens)].set(requests[1][0])
    metadata = SimpleNamespace(
        query_start_loc=jnp.asarray([0, lens[0], sum(lens)], jnp.int32),
        context_lens=jnp.asarray(lens, jnp.int32),
    )

    with model.mesh:
        out = layer(hidden_states=packed, cache_view=_empty_view(2), cache_metadata=metadata)
        dense = [layer(hidden_states=r, cache_view=_empty_view(1), cache_metadata=None) for r in requests]

    got = out.attention_output[0]
    assert _rel(got[: lens[0]], dense[0].attention_output[0]) < 1e-2
    assert _rel(got[lens[0] : sum(lens)], dense[1].attention_output[0]) < 1e-2
    for row, d in enumerate(dense):
        assert _rel(out.cache_view.recurrent_state[row], d.cache_view.recurrent_state[0]) < 1e-2
        assert _rel(out.cache_view.k_conv_state[row], d.cache_view.k_conv_state[0]) < 1e-2


def test_packed_decode_step_continues_each_request():
    """A pure decode step (one token per row) continues each row from its own state."""
    model, layer = _layer()
    rng = np.random.default_rng(3)
    prompts = [jnp.asarray(rng.normal(size=(1, n, HIDDEN)), jnp.float32) for n in (9, 5)]
    nxt = [jnp.asarray(rng.normal(size=(1, 1, HIDDEN)), jnp.float32) for _ in prompts]

    with model.mesh:
        pre = [layer(hidden_states=p, cache_view=_empty_view(1), cache_metadata=None) for p in prompts]
        ref = [
            layer(hidden_states=x, cache_view=p.cache_view, cache_metadata=None) for x, p in zip(nxt, pre, strict=True)
        ]
        stacked = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0) if hasattr(a, "ndim") and a.ndim >= 1 else a,
            pre[0].cache_view,
            pre[1].cache_view,
        )
        packed = jnp.zeros((1, 8, HIDDEN), jnp.float32).at[:, 0].set(nxt[0][0, 0]).at[:, 1].set(nxt[1][0, 0])
        metadata = SimpleNamespace(
            query_start_loc=jnp.asarray([0, 1, 2], jnp.int32),
            context_lens=jnp.asarray([10, 6], jnp.int32),
        )
        out = layer(hidden_states=packed, cache_view=stacked, cache_metadata=metadata)

    for row in range(2):
        assert _rel(out.attention_output[0, row], ref[row].attention_output[0, 0]) < 1e-2
        assert _rel(out.cache_view.recurrent_state[row], ref[row].cache_view.recurrent_state[0]) < 1e-2
