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


def _layer(dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST):
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
        precision=precision,
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
    """Rank-major query_start_loc over a single-rank batch [1, T, D] fails loudly instead of advancing rank 0 only."""
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


def test_verify_step_equals_the_same_tokens_decoded_one_per_step():
    """A speculative verify step over slot groups equals decoding its tokens one per step from slot ``g + o``.

    Every request owns ``W`` slots ``g .. g + W - 1``. The verify step reads slot ``g + recurrent_read_offset`` and
    writes the windows and state after its ``j``-th token into ``g + j``. Decoding the same tokens one per step from the
    state of slot ``g + o`` must give the same outputs, and the state after the ``j``-th decode step must be the one
    captured in ``g + j``; slots past a row's tokens keep their contents.
    """
    model, layer = _layer()
    width = 4
    names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
    base, offset, lens, context = [0, 4], [1, 2], [3, 2], [9, 5]
    held = {name: np.asarray(getattr(_random_view(2 * width, seed=61), name)) for name in names}
    tokens = jnp.asarray(np.random.default_rng(67).normal(size=(sum(lens), HIDDEN)), jnp.float32)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    verify_meta = SimpleNamespace(
        query_start_loc=jnp.asarray(starts),
        context_lens=jnp.asarray(context, jnp.int32),
        recurrent_state_indices=jnp.asarray([base], jnp.int32),
        recurrent_read_offset=jnp.asarray(offset, jnp.int32),
        recurrent_commit_lens=jnp.ones((2,), jnp.int32),
        recurrent_slot_group_size=width,
    )
    packed = jnp.zeros((1, 8, HIDDEN), jnp.float32).at[0, : sum(lens)].set(tokens)
    read = [g + o for g, o in zip(base, offset, strict=True)]
    with model.mesh:
        verify = layer(hidden_states=packed, cache_view=_random_view(2 * width, seed=61), cache_metadata=verify_meta)
        view = KDACacheView(
            **{name: jnp.asarray(held[name][read]) for name in names},
            positions=jnp.zeros((2,), jnp.int32),
            metadata=None,
        )
        for j in range(max(lens)):
            live = [j < n for n in lens]
            step = jnp.zeros((1, 2, HIDDEN), jnp.float32)
            qsl, pos = [0], 0
            for r in range(2):
                if live[r]:
                    step = step.at[0, pos].set(tokens[starts[r] + j])
                    pos += 1
                qsl.append(pos)
            meta = SimpleNamespace(
                query_start_loc=jnp.asarray(qsl, jnp.int32),
                context_lens=jnp.asarray([c - n + j + 1 for c, n in zip(context, lens, strict=True)], jnp.int32),
            )
            decoded = layer(hidden_states=step, cache_view=view, cache_metadata=meta)
            view = decoded.cache_view
            for r in range(2):
                if not live[r]:
                    continue
                np.testing.assert_allclose(
                    np.asarray(verify.attention_output[0, starts[r] + j]),
                    np.asarray(decoded.attention_output[0, qsl[r]]),
                    rtol=1e-5,
                    atol=1e-6,
                )
                for name in names:
                    np.testing.assert_allclose(
                        np.asarray(getattr(verify.cache_view, name)[base[r] + j]),
                        np.asarray(getattr(view, name)[r]),
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=f"{name} row {r} token {j}",
                    )
    for name in names:
        new = np.asarray(getattr(verify.cache_view, name))
        for r in range(2):
            untouched = list(range(base[r] + lens[r], base[r] + width))
            np.testing.assert_array_equal(new[untouched], held[name][untouched], err_msg=name)


def test_candidate_row_commit_metadata_is_refused():
    """KDA layers serve speculative steps through slot groups; candidate-row commit metadata is a contract error."""
    model, layer = _layer()
    packed = jnp.zeros((1, 4, HIDDEN), jnp.float32)
    meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, 2, 3], jnp.int32),
        context_lens=jnp.asarray([9, 5], jnp.int32),
        spec_recurrent_commit=jnp.zeros((2, 2), jnp.int32),
    )
    with model.mesh, pytest.raises(ValueError, match="spec_recurrent_commit"):
        layer(hidden_states=packed, cache_view=_random_view(2, seed=53), cache_metadata=meta)


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


def test_packed_chunk_dots_follow_the_layer_matmul_precision():
    """eSurge's flat driver runs its chunk dots at the layer's matmul precision; an explicit ``kda_chunk`` keeps its own.

    ``from_pretrained`` builds layers at ``Precision("fastest")`` (``DEFAULT``): with ``kda_chunk_impl`` ``None`` (the
    default) or ``"scan"`` their multi-token rows take the single-pass chunk dots that ``kda_chunk_impl="kda_chunk"``
    with ``kda_chunk_precision="default"`` selects, close to the exact ones. A layer built at ``HIGHEST`` keeps exact
    chunk dots. Decode-only steps run no chunk dots, so their outputs and pools do not depend on the setting.
    """
    n, rows = 12, 4
    rng = np.random.default_rng(59)
    tokens = jnp.asarray(rng.normal(size=(1, n, HIDDEN)), jnp.float32)
    prefill = jnp.zeros((1, 16, HIDDEN), jnp.float32).at[:, :n].set(tokens)
    prefill_meta = SimpleNamespace(
        query_start_loc=jnp.asarray([0, n], jnp.int32), context_lens=jnp.asarray([n], jnp.int32)
    )
    # A decode bucket as eSurge compiles it: one token per row and as many tokens as rows.
    decode = jnp.asarray(rng.normal(size=(1, rows, HIDDEN)), jnp.float32)
    decode_meta = SimpleNamespace(
        query_start_loc=jnp.arange(rows + 1, dtype=jnp.int32), context_lens=jnp.asarray([9, 5, 13, 7], jnp.int32)
    )
    names = ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state")
    settings = {
        "auto": (None, "highest"),
        "scan": ("scan", "highest"),
        "exact": ("kda_chunk", "highest"),
        "fast": ("kda_chunk", "default"),
    }

    def serve(precision):
        model, layer = _layer(precision=precision)
        runs = {}
        with model.mesh:
            for label, (impl, chunk_precision) in settings.items():
                layer.config.kda_chunk_impl, layer.config.kda_chunk_precision = impl, chunk_precision
                p = layer(hidden_states=prefill, cache_view=_empty_view(1), cache_metadata=prefill_meta)
                d = layer(hidden_states=decode, cache_view=_random_view(rows, seed=61), cache_metadata=decode_meta)
                decoded = [np.asarray(d.attention_output)] + [np.asarray(getattr(d.cache_view, x)) for x in names]
                runs[label] = (np.asarray(p.attention_output), np.asarray(p.cache_view.recurrent_state), decoded)
        return runs

    for precision, same, other in (
        (jax.lax.Precision.DEFAULT, "fast", "exact"),
        (jax.lax.Precision.HIGHEST, "exact", "fast"),
    ):
        runs = serve(precision)
        for label in ("auto", "scan"):
            np.testing.assert_array_equal(runs[label][0], runs[same][0], err_msg=f"{precision} {label}")
            np.testing.assert_array_equal(runs[label][1], runs[same][1], err_msg=f"{precision} {label}")
        assert not np.array_equal(runs["scan"][0], runs[other][0]), precision
        assert _rel(runs["scan"][0], runs[other][0]) < 2e-2, precision
        assert _rel(runs["scan"][1], runs[other][1]) < 2e-2, precision
        for label in ("auto", "exact", "fast"):
            for got, want in zip(runs[label][2], runs["scan"][2], strict=True):
                np.testing.assert_array_equal(got, want, err_msg=f"{precision} {label}: decode step")
