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

"""eSurge packed-row KDA serving path must reproduce the dense KDA layer.

The packed path (``Glm5NextLinearAttention._forward_packed_rows``) is a
separate implementation that splits one flattened token stream into per-request
rows. Each row's outputs and final recurrent state must equal running the
dense layer on that request alone.
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
    config = ed.Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_head_dim=HEAD_DIM,
        linear_num_heads=HEADS,
        linear_conv_kernel_dim=D_CONV,
        hc_mult=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = ed.Glm5NextForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    return model, model.model.layers[0].self_attn


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


def test_packed_rows_match_dense_per_request():
    model, layer = _layer()
    lens = (7, 12)
    rng = np.random.default_rng(0)
    requests = [jnp.asarray(rng.normal(size=(1, n, HIDDEN)), jnp.float32) for n in lens]
    # A padded token bucket, as the runner compiles it.
    bucket = 32
    packed = jnp.zeros((1, bucket, HIDDEN), jnp.float32)
    packed = packed.at[:, : lens[0]].set(requests[0][0]).at[:, lens[0] : sum(lens)].set(requests[1][0])
    metadata = SimpleNamespace(
        query_start_loc=jnp.asarray([0, lens[0], sum(lens)], jnp.int32),
        context_lens=jnp.asarray(lens, jnp.int32),
    )

    with model.mesh:
        out = layer(hidden_states=packed, cache_view=_empty_view(2), cache_metadata=metadata)
        dense = [layer(hidden_states=r, cache_view=_empty_view(1), cache_metadata=None) for r in requests]

    # TPU's default-precision dense conv differs from the packed path's exact
    # fp32 conv at the 1e-3 level; skipping the conv (the regression this
    # guards) is off by >100%.
    def rel(a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    got = out.attention_output[0]
    assert rel(got[: lens[0]], dense[0].attention_output[0]) < 1e-2
    assert rel(got[lens[0] : sum(lens)], dense[1].attention_output[0]) < 1e-2
    for row, d in enumerate(dense):
        assert rel(out.cache_view.recurrent_state[row], d.cache_view.recurrent_state[0]) < 1e-2
