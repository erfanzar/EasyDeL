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


def _layer(dtype=jnp.float32):
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
        dtype=dtype,
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


def test_multi_token_cached_continuation_matches_unsplit_prefill():
    """A second prefill chunk must reuse the first chunk's causal conv window."""
    model, layer = _layer()
    inputs = jnp.asarray(np.random.default_rng(19).normal(size=(1, 12, HIDDEN)), jnp.float32)
    split = 7  # Longer than the depthwise kernel; continuation is multi-token.

    with model.mesh:
        whole = layer(hidden_states=inputs, cache_view=_empty_view(1))
        prefix = layer(hidden_states=inputs[:, :split], cache_view=_empty_view(1))
        suffix = layer(hidden_states=inputs[:, split:], cache_view=prefix.cache_view)

    np.testing.assert_allclose(
        np.asarray(suffix.attention_output),
        np.asarray(whole.attention_output[:, split:]),
        rtol=1e-2,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(suffix.cache_view.recurrent_state),
        np.asarray(whole.cache_view.recurrent_state),
        rtol=1e-2,
        atol=1e-3,
    )


def test_mixed_dtype_cached_conv_preserves_projection_dtype():
    """Saved fp32 conv state must not promote a bf16 continuation's rolling window."""
    model, layer = _layer(dtype=jnp.bfloat16)
    inputs = jnp.asarray(np.random.default_rng(21).normal(size=(1, 12, HIDDEN)), jnp.bfloat16)
    with model.mesh:
        whole = layer(hidden_states=inputs, cache_view=_empty_view(1))
        prefix = layer(hidden_states=inputs[:, :7], cache_view=_empty_view(1))
        f32_state = prefix.cache_view.replace(
            q_conv_state=prefix.cache_view.q_conv_state.astype(jnp.float32),
            k_conv_state=prefix.cache_view.k_conv_state.astype(jnp.float32),
            v_conv_state=prefix.cache_view.v_conv_state.astype(jnp.float32),
        )
        suffix = layer(hidden_states=inputs[:, 7:], cache_view=f32_state)

    assert suffix.cache_view.q_conv_state.dtype == jnp.bfloat16
    assert suffix.cache_view.k_conv_state.dtype == jnp.bfloat16
    assert suffix.cache_view.v_conv_state.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(suffix.attention_output, dtype=np.float32),
        np.asarray(whole.attention_output[:, 7:], dtype=np.float32),
        rtol=0.05,
        atol=0.05,
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_packed_multi_token_continuation_matches_unsplit_prefill():
    """Packed rows with ``context_lens > lens`` and ``lens > 1`` must reuse the carried conv window."""
    model, layer = _layer()
    inputs = jnp.asarray(np.random.default_rng(23).normal(size=(1, 12, HIDDEN)), jnp.float32)
    split = 7
    n_suffix = 12 - split
    prefix_bucket = jnp.zeros((1, 16, HIDDEN), jnp.float32).at[:, :split].set(inputs[:, :split])
    suffix_bucket = jnp.zeros((1, 16, HIDDEN), jnp.float32).at[:, :n_suffix].set(inputs[:, split:])
    prefix_meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, split], jnp.int32), context_lens=jnp.asarray([split], jnp.int32)
    )
    suffix_meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, n_suffix], jnp.int32), context_lens=jnp.asarray([12], jnp.int32)
    )

    with model.mesh:
        whole = layer(hidden_states=inputs, cache_view=_empty_view(1))
        prefix = layer(hidden_states=prefix_bucket, cache_view=_empty_view(1), cache_metadata=prefix_meta)
        suffix = layer(hidden_states=suffix_bucket, cache_view=prefix.cache_view, cache_metadata=suffix_meta)

    assert _rel(suffix.attention_output[0, :n_suffix], whole.attention_output[0, split:]) < 1e-2
    assert _rel(suffix.cache_view.recurrent_state[0], whole.cache_view.recurrent_state[0]) < 1e-2


def test_packed_mixed_continuation_and_fresh_prefill_rows():
    """One continuation row (multi-token, has a conv prefix) beside a fresh prefill row."""
    model, layer = _layer()
    rng = np.random.default_rng(29)
    req0 = jnp.asarray(rng.normal(size=(1, 12, HIDDEN)), jnp.float32)
    req1 = jnp.asarray(rng.normal(size=(1, 6, HIDDEN)), jnp.float32)
    split = 7
    n0 = 12 - split
    bucket = 16
    prefix_bucket = jnp.zeros((1, bucket, HIDDEN), jnp.float32).at[:, :split].set(req0[:, :split])
    prefix_meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, split], jnp.int32), context_lens=jnp.asarray([split], jnp.int32)
    )
    packed = jnp.zeros((1, bucket, HIDDEN), jnp.float32)
    packed = packed.at[:, :n0].set(req0[:, split:]).at[:, n0 : n0 + 6].set(req1)
    meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, n0, n0 + 6], jnp.int32),
        context_lens=jnp.asarray([12, 6], jnp.int32),
    )

    with model.mesh:
        whole0 = layer(hidden_states=req0, cache_view=_empty_view(1))
        dense1 = layer(hidden_states=req1, cache_view=_empty_view(1))
        prefix0 = layer(hidden_states=prefix_bucket, cache_view=_empty_view(1), cache_metadata=prefix_meta)
        view = prefix0.cache_view
        both = view.replace(
            q_conv_state=jnp.concatenate([view.q_conv_state, jnp.zeros_like(view.q_conv_state)], axis=0),
            k_conv_state=jnp.concatenate([view.k_conv_state, jnp.zeros_like(view.k_conv_state)], axis=0),
            v_conv_state=jnp.concatenate([view.v_conv_state, jnp.zeros_like(view.v_conv_state)], axis=0),
            recurrent_state=jnp.concatenate([view.recurrent_state, jnp.zeros_like(view.recurrent_state)], axis=0),
            positions=jnp.concatenate([view.positions, jnp.zeros_like(view.positions)], axis=0),
        )
        out = layer(hidden_states=packed, cache_view=both, cache_metadata=meta)

    got = out.attention_output[0]
    assert _rel(got[:n0], whole0.attention_output[0, split:]) < 1e-2
    assert _rel(got[n0 : n0 + 6], dense1.attention_output[0]) < 1e-2
    assert _rel(out.cache_view.recurrent_state[0], whole0.cache_view.recurrent_state[0]) < 1e-2
    assert _rel(out.cache_view.recurrent_state[1], dense1.cache_view.recurrent_state[0]) < 1e-2


def test_rank_major_query_start_loc_is_rejected():
    """Per-channel KDA cannot advance rank-major DP rows; it must fail loudly rather than use rank 0."""
    import pytest

    model, layer = _layer()
    packed = jnp.zeros((1, 16, HIDDEN), jnp.float32)
    meta = SimpleNamespace(
        query_start_loc=jnp.asarray([[0, 4, 8], [0, 4, 8]], jnp.int32),
        context_lens=jnp.asarray([4, 4], jnp.int32),
    )
    with model.mesh, pytest.raises(ValueError, match="rank-major"):
        layer(hidden_states=packed, cache_view=_empty_view(2), cache_metadata=meta)
