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
import pytest
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


def _random_view(rows, seed):
    """A pool of ``rows`` slots with random nonzero windows and states, so untouched slots can be checked."""
    rng = np.random.default_rng(seed)
    width = HEADS * HEAD_DIM

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape), jnp.float32)

    return KDACacheView(
        q_conv_state=normal(rows, width, D_CONV),
        k_conv_state=normal(rows, width, D_CONV),
        v_conv_state=normal(rows, width, D_CONV),
        recurrent_state=0.1 * normal(rows, HEADS, HEAD_DIM, HEAD_DIM),
        positions=jnp.zeros((rows,), jnp.int32),
        metadata=None,
    )


def test_packed_routing_follows_the_speculative_commit_metadata():
    """eSurge sends ``spec_recurrent_commit`` exactly when the pool holds speculative candidate rows.

    Without it the flat-stream driver runs and never touches pool slots past the rows, whatever the pool size (eSurge
    keeps ``max_num_reqs`` slots, which can exceed twice the step's row cap). With it the row-padded driver runs and
    writes the candidate rows of a speculative verify row. Both drivers give the same live rows.
    """
    model, layer = _layer()
    rows, candidates = 2, 2
    pool = rows * (1 + candidates)
    names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
    # Host copy of the pool; each call gets its own identical pool, so no call can see another's writes.
    held = {name: np.asarray(getattr(_random_view(pool, seed=31), name)) for name in names}
    tokens = jnp.asarray(np.random.default_rng(37).normal(size=(1, 3, HIDDEN)), jnp.float32)
    packed = jnp.zeros((1, 8, HIDDEN), jnp.float32).at[:, :3].set(tokens)
    # Row 0: a 2-token verify window continuing its request; row 1: a decode step continuing its request.
    plain = SimpleNamespace(
        query_start_loc=jnp.asarray([0, 2, 3], jnp.int32), context_lens=jnp.asarray([9, 5], jnp.int32)
    )
    spec = SimpleNamespace(**vars(plain), spec_recurrent_commit=jnp.zeros((2, rows), jnp.int32))

    with model.mesh:
        flat = layer(hidden_states=packed, cache_view=_random_view(pool, seed=31), cache_metadata=plain)
        padded = layer(hidden_states=packed, cache_view=_random_view(pool, seed=31), cache_metadata=spec)

    for name in names:
        np.testing.assert_array_equal(np.asarray(getattr(flat.cache_view, name)[rows:]), held[name][rows:], err_msg=name)
    written = np.asarray(padded.cache_view.recurrent_state[rows:])
    assert not np.array_equal(written, held["recurrent_state"][rows:]), "the speculative step wrote no candidate row"
    assert _rel(flat.attention_output[0, :3], padded.attention_output[0, :3]) < 1e-2
    assert _rel(flat.cache_view.recurrent_state[:rows], padded.cache_view.recurrent_state[:rows]) < 1e-2


def test_packed_chunk_precision_applies_only_to_the_kda_chunk_impl():
    """``kda_chunk_precision`` is documented for ``kda_chunk_impl="kda_chunk"``; under ``"scan"`` it is ignored."""
    model, layer = _layer()
    n = 12
    tokens = jnp.asarray(np.random.default_rng(41).normal(size=(1, n, HIDDEN)), jnp.float32)
    packed = jnp.zeros((1, 16, HIDDEN), jnp.float32).at[:, :n].set(tokens)
    meta = SimpleNamespace(query_start_loc=jnp.asarray([0, n], jnp.int32), context_lens=jnp.asarray([n], jnp.int32))
    config = layer.config
    saved = (config.kda_chunk_impl, config.kda_chunk_precision)
    outputs = {}
    try:
        with model.mesh:
            for impl, precision in (("scan", "highest"), ("scan", "default"), ("kda_chunk", "default")):
                config.kda_chunk_impl, config.kda_chunk_precision = impl, precision
                out = layer(hidden_states=packed, cache_view=_empty_view(1), cache_metadata=meta)
                outputs[impl, precision] = np.asarray(out.attention_output)
    finally:
        config.kda_chunk_impl, config.kda_chunk_precision = saved
    np.testing.assert_array_equal(outputs["scan", "default"], outputs["scan", "highest"])
    assert not np.array_equal(outputs["kda_chunk", "default"], outputs["scan", "highest"])


def test_packed_rows_follow_recurrent_state_indices():
    """With ``cache_metadata.recurrent_state_indices`` row r reads and writes pool slot ``indices[r]``.

    eSurge names the slots when a step's row window does not start at slot 0. The same step on a pool holding only
    those slots, in row order and without indices, gives the same outputs and states; the other slots stay intact.
    """
    model, layer = _layer()
    names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
    slots = [3, 1]
    held = {name: np.asarray(getattr(_random_view(4, seed=43), name)) for name in names}
    window = KDACacheView(
        **{name: jnp.asarray(held[name][slots]) for name in names},
        positions=jnp.zeros((len(slots),), jnp.int32),
        metadata=None,
    )
    tokens = jnp.asarray(np.random.default_rng(47).normal(size=(1, 4, HIDDEN)), jnp.float32)
    packed = jnp.zeros((1, 8, HIDDEN), jnp.float32).at[:, :4].set(tokens)
    # Row 0: one decode token continuing its request; row 1: a 3-token continuation (shorter than the conv window).
    meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, 1, 4], jnp.int32), context_lens=jnp.asarray([9, 7], jnp.int32)
    )
    indexed = SimpleNamespace(**vars(meta), recurrent_state_indices=jnp.asarray([slots], jnp.int32))

    with model.mesh:
        want = layer(hidden_states=packed, cache_view=window, cache_metadata=meta)
        got = layer(hidden_states=packed, cache_view=_random_view(4, seed=43), cache_metadata=indexed)

    np.testing.assert_allclose(np.asarray(got.attention_output), np.asarray(want.attention_output), rtol=1e-5, atol=1e-6)
    for name in names:
        new = np.asarray(getattr(got.cache_view, name))
        np.testing.assert_allclose(
            new[slots], np.asarray(getattr(want.cache_view, name)), rtol=1e-5, atol=1e-6, err_msg=name
        )
        np.testing.assert_array_equal(new[[0, 2]], held[name][[0, 2]], err_msg=name)


def test_speculative_steps_refuse_explicit_recurrent_slots():
    """A speculative step cannot also name explicit pool slots: the row-padded driver keeps row r in slot r.

    Running it anyway would read and write other requests' states without any error, so the layer refuses the
    combination before it touches the pool.
    """
    model, layer = _layer()
    rows, candidates = 2, 2
    tokens = jnp.asarray(np.random.default_rng(53).normal(size=(1, 3, HIDDEN)), jnp.float32)
    packed = jnp.zeros((1, 8, HIDDEN), jnp.float32).at[:, :3].set(tokens)
    meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, 2, 3], jnp.int32),
        context_lens=jnp.asarray([9, 5], jnp.int32),
        spec_recurrent_commit=jnp.zeros((2, rows), jnp.int32),
        recurrent_state_indices=jnp.asarray([[3, 1]], jnp.int32),
    )

    with model.mesh, pytest.raises(ValueError, match="recurrent_state_indices"):
        layer(hidden_states=packed, cache_view=_random_view(rows * (1 + candidates), seed=53), cache_metadata=meta)
