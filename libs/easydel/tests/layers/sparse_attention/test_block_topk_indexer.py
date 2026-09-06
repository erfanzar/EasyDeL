# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``BlockTopKIndexer`` parity against the loop-based reference selection.

The ground truth below is a NumPy port of HF ``Qwen4ExpTextQSAIndexer``'s
per-batch/per-query loop: pool complete visible blocks of ``compress_ratio``
tokens, score with ``sum_h relu(q . k) / sqrt(d)`` after per-head norm + RoPE
(queries at their own position, block keys at the block start), take the top
``budget // ratio`` blocks, expand to member tokens, and append the incomplete
tail. The vectorized JAX module must reproduce the *exact* selection, under
causal-only, left-padded, vacuous-budget, and decode-shaped inputs.
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.sparse_attention import BlockTopKIndexer, apply_partial_rope
from jax import numpy as jnp

HIDDEN = 32
N_HEADS = 4
HEAD_DIM = 8
RATIO = 4
BUDGET = 8  # -> block_topk = 2
ROTARY = 4  # partial rotary width (even)
EPS = 1e-6


def _norm(x, w, eps=EPS):
    xf = x.astype(np.float32)
    out = xf / np.sqrt(np.mean(xf**2, axis=-1, keepdims=True) + eps)
    return out * (1.0 + w.astype(np.float32))


def _rope(x, cos, sin):
    """Reference apply_rotary_pos_emb: split-half on the leading rotary dims."""
    r = cos.shape[-1]
    h = r // 2
    xr, xn = x[..., :r], x[..., r:] if r < x.shape[-1] else None
    x1, x2 = xr[..., :h], xr[..., h:]
    out = np.concatenate([x1 * cos[..., :h] - x2 * sin[..., :h], x2 * cos[..., h:] + x1 * sin[..., h:]], axis=-1)
    return out if xn is None else np.concatenate([out, xn], axis=-1)


def _cos_sin_table(positions, rotary=ROTARY, base=10000.0):
    inv = 1.0 / (base ** (np.arange(0, rotary, 2, dtype=np.float64) / rotary))
    freqs = positions.astype(np.float64)[..., None] * inv  # [..., rotary/2]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _reference_select(q_proj, raw_k, w_qnorm, w_knorm, visible, budget=BUDGET, ratio=RATIO):
    """Loop-based reference. q_proj: [B,T,H,D] raw proj out; raw_k: [B,K,D]."""
    batch, q_len = q_proj.shape[:2]
    kv_len = raw_k.shape[1]
    block_topk = budget // ratio
    width = budget + ratio - 1
    out = np.full((batch, q_len, width), -1, dtype=np.int32)
    pos = np.arange(kv_len)
    cos_full, sin_full = _cos_sin_table(pos)  # [K, R]
    for b in range(batch):
        for t in range(q_len):
            q_idx = kv_len - q_len + t  # absolute index of the query
            vis = visible[b] & (pos <= q_idx)  # padding mask over the full kv range + causal
            vis_idx = np.nonzero(vis)[0]
            n_complete = vis_idx.shape[0] // ratio
            q = _norm(q_proj[b, t], w_qnorm)  # [H, D]
            qc, qs = cos_full[q_idx], sin_full[q_idx]
            q = _rope(q, qc[None, :], qs[None, :])
            selected = []
            if n_complete > 0:
                blocks = vis_idx[: n_complete * ratio].reshape(n_complete, ratio)
                pooled = raw_k[b][blocks].astype(np.float32).mean(axis=1)  # [nb, D]
                pooled = _norm(pooled, w_knorm)
                starts = blocks[:, 0]
                bk = _rope(pooled, cos_full[starts], sin_full[starts])
                scores = np.maximum(q.astype(np.float32) @ bk.T, 0.0).sum(axis=0) / np.sqrt(HEAD_DIM)  # [nb]
                k = min(block_topk, n_complete)
                top = np.argsort(-scores, kind="stable")[:k]
                selected.extend(blocks[top].flatten().tolist())
            selected.extend(vis_idx[n_complete * ratio :].tolist())
            out[b, t, : len(selected)] = np.asarray(selected, dtype=np.int32)
    return out


def _make_indexer(seed=0):
    rng = np.random.default_rng(seed)
    mod = BlockTopKIndexer(
        hidden_size=HIDDEN,
        index_n_heads=N_HEADS,
        index_kv_heads=1,
        index_head_dim=HEAD_DIM,
        indexer_budget=BUDGET,
        indexer_compress_ratio=RATIO,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(seed),
    )
    w = rng.standard_normal(((N_HEADS + 1) * HEAD_DIM, HIDDEN)).astype(np.float32) * 0.08
    mod.index_qk_proj.weight.value = jnp.asarray(w.T)  # HF [out,in] -> [in,out]
    wq = rng.standard_normal(HEAD_DIM).astype(np.float32) * 0.1
    wk = rng.standard_normal(HEAD_DIM).astype(np.float32) * 0.1
    mod.q_layernorm.weight.value = jnp.asarray(wq)
    mod.k_layernorm.weight.value = jnp.asarray(wk)
    return mod, w, wq, wk


def _run(mod, hidden, visible_pad, kv_len=None):
    """Run the module end to end and return selected indices."""
    batch, q_len = hidden.shape[:2]
    kv_len = kv_len or q_len
    q, raw_k = mod.project(jnp.asarray(hidden))
    pos = np.arange(kv_len)
    cos, sin = _cos_sin_table(pos)
    q_cos = jnp.asarray(cos[-q_len:][None].repeat(batch, 0))
    q_sin = jnp.asarray(sin[-q_len:][None].repeat(batch, 0))
    sel = mod.select(
        q,
        jnp.asarray(raw_k),
        q_cos=q_cos,
        q_sin=q_sin,
        k_cos=jnp.asarray(cos[None]),
        k_sin=jnp.asarray(sin[None]),
        visible=jnp.asarray(visible_pad),
    )
    return np.asarray(sel), np.asarray(raw_k)


def _as_mask_set(sel, kv_len):
    """Order-invariant set form: selections are consumed as a scatter mask, so
    block *order* is not semantic (HF topk vs lax.top_k tie-order differ)."""
    out = np.zeros((sel.shape[0], sel.shape[1], kv_len), dtype=bool)
    for b in range(sel.shape[0]):
        for t in range(sel.shape[1]):
            out[b, t, sel[b, t][sel[b, t] >= 0]] = True
    return out


@pytest.mark.parametrize("seq", [8, 9, 13])
def test_selection_matches_reference_causal(seq):
    mod, w, wq, wk = _make_indexer()
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((2, seq, HIDDEN)).astype(np.float32) * 0.5
    visible = np.ones((2, seq), bool)

    got, raw_k = _run(mod, hidden, visible)
    # reference ground truth from the same projections
    qk = hidden @ w.T
    q_proj = qk[..., : N_HEADS * HEAD_DIM].reshape(2, seq, N_HEADS, HEAD_DIM)
    want = _reference_select(q_proj, raw_k, wq, wk, visible)
    np.testing.assert_array_equal(_as_mask_set(got, seq), _as_mask_set(want, seq))


def test_selection_matches_reference_left_padded():
    mod, w, wq, wk = _make_indexer(seed=1)
    rng = np.random.default_rng(11)
    seq = 12
    hidden = rng.standard_normal((3, seq, HIDDEN)).astype(np.float32) * 0.5
    visible = np.ones((3, seq), bool)
    visible[0, :5] = False  # left padding on one row
    visible[1, :1] = False

    got, raw_k = _run(mod, hidden, visible)
    qk = hidden @ w.T
    q_proj = qk[..., : N_HEADS * HEAD_DIM].reshape(3, seq, N_HEADS, HEAD_DIM)
    want = _reference_select(q_proj, raw_k, wq, wk, visible)
    np.testing.assert_array_equal(_as_mask_set(got, seq), _as_mask_set(want, seq))


def test_vacuous_budget_selects_everything():
    """budget >= blocks: the selection must equal the full visible set."""
    mod = BlockTopKIndexer(
        hidden_size=HIDDEN,
        index_n_heads=N_HEADS,
        index_kv_heads=1,
        index_head_dim=HEAD_DIM,
        indexer_budget=64,
        indexer_compress_ratio=RATIO,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    seq = 6
    hidden = np.random.default_rng(3).standard_normal((1, seq, HIDDEN)).astype(np.float32)
    sel, _ = _run(mod, hidden, np.ones((1, seq), bool))
    picked = sel[0, -1]  # last query sees the whole prefix
    assert (picked >= 0).sum() == seq


def test_decode_shape_matches_prefill_last_position():
    """q=1 against a cached prefix must equal the prefill selection at that position."""
    mod, _w, _wq, _wk = _make_indexer(seed=2)
    rng = np.random.default_rng(5)
    seq = 10
    hidden = rng.standard_normal((1, seq, HIDDEN)).astype(np.float32) * 0.5
    visible = np.ones((1, seq), bool)

    sel_full, raw_k_full = _run(mod, hidden, visible)

    # decode the last position with the prefix cached
    q, _raw_k_new = mod.project(jnp.asarray(hidden[:, -1:]))
    pos = np.arange(seq)
    cos, sin = _cos_sin_table(pos)
    sel_dec = mod.select(
        q,
        jnp.asarray(raw_k_full),
        q_cos=jnp.asarray(cos[-1:][None]),
        q_sin=jnp.asarray(sin[-1:][None]),
        k_cos=jnp.asarray(cos[None]),
        k_sin=jnp.asarray(sin[None]),
        visible=jnp.asarray(visible),
    )
    # decode select sees the full kv range (cached prefix + current key)
    np.testing.assert_array_equal(
        _as_mask_set(np.asarray(sel_dec)[0:1, 0:1], seq)[0, 0], _as_mask_set(sel_full, seq)[0, -1]
    )


def test_build_mask_absorbs_padding():
    mod, *_ = _make_indexer()
    selected = jnp.asarray([[[0, 3, -1, -1], [1, -1, -1, -1]]], jnp.int32)
    mask = mod.build_mask(selected, kv_len=5)
    assert mask.shape == (1, 1, 2, 5)
    np.testing.assert_array_equal(np.asarray(mask)[0, 0, 0], [True, False, False, True, False])
    np.testing.assert_array_equal(np.asarray(mask)[0, 0, 1], [False, True, False, False, False])


def test_apply_partial_rope_matches_reference():
    x = np.random.default_rng(9).standard_normal((2, 3, HEAD_DIM)).astype(np.float32)
    pos = np.arange(3)
    cos, sin = _cos_sin_table(pos)
    got = apply_partial_rope(
        jnp.asarray(x[:, :, None, :]), jnp.asarray(cos[None, :, None, :]), jnp.asarray(sin[None, :, None, :])
    )
    want = _rope(x[:, :, None, :], cos[None, :, None, :], sin[None, :, None, :])
    np.testing.assert_allclose(np.asarray(got)[:, :, 0, :], want[:, :, 0, :], rtol=1e-5, atol=1e-6)


def test_decode_vacuous_budget_skips_topk(monkeypatch):
    """Prefixes within the token budget must not rank the full block buffer."""
    mod = BlockTopKIndexer(
        hidden_size=HIDDEN,
        index_n_heads=N_HEADS,
        index_kv_heads=1,
        index_head_dim=HEAD_DIM,
        indexer_budget=64,
        indexer_compress_ratio=RATIO,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    batch, kv_len = 1, 128
    max_blocks = kv_len // RATIO

    def fail_top_k(*_args, **_kwargs):
        raise AssertionError("vacuous selection must not execute top_k")

    monkeypatch.setattr(jax.lax, "top_k", fail_top_k)
    with jax.disable_jit():
        selected, _, _ = mod.select_step(
            jnp.zeros((batch, 1, N_HEADS, HEAD_DIM), jnp.float32),
            q_cos=jnp.ones((batch, 1, ROTARY), jnp.float32),
            q_sin=jnp.zeros((batch, 1, ROTARY), jnp.float32),
            key_buffer=jnp.zeros((batch, kv_len, HEAD_DIM), jnp.float32),
            block_keys=jnp.zeros((batch, max_blocks, HEAD_DIM), jnp.float32),
            blocks_complete=jnp.zeros((batch, max_blocks), jnp.bool_).at[:, 0].set(True),
            visible=jnp.ones((batch, kv_len), jnp.bool_),
            open_cos=jnp.ones((batch, 1, ROTARY), jnp.float32),
            open_sin=jnp.zeros((batch, 1, ROTARY), jnp.float32),
            write_at=jnp.asarray([7], jnp.int32),
        )

    picked = np.asarray(selected)[0, 0]
    np.testing.assert_array_equal(picked[picked >= 0], np.arange(8, dtype=np.int32))


def test_config_validation():
    with pytest.raises(ValueError, match="indexer_kv_heads"):
        BlockTopKIndexer(
            hidden_size=HIDDEN,
            index_n_heads=2,
            index_kv_heads=2,
            index_head_dim=8,
            indexer_budget=8,
            indexer_compress_ratio=4,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )
    with pytest.raises(ValueError, match="divisible"):
        BlockTopKIndexer(
            hidden_size=HIDDEN,
            index_n_heads=2,
            index_kv_heads=1,
            index_head_dim=8,
            indexer_budget=7,
            indexer_compress_ratio=4,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_select_does_not_materialize_quadratic_one_hot(monkeypatch):
    """Pooling must remain O(kv_len), not allocate [kv_len, n_blocks]."""
    mod, *_ = _make_indexer(seed=9)
    seq = 64
    hidden = np.random.default_rng(9).standard_normal((1, seq, HIDDEN)).astype(np.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError("quadratic one_hot pooling is forbidden")

    monkeypatch.setattr(jax.nn, "one_hot", forbidden)
    got, _ = _run(mod, hidden, np.ones((1, seq), bool))
    assert got.shape[:2] == (1, seq)


def test_nonzero_based_kv_positions_preserve_compact_block_membership():
    mod, *_ = _make_indexer(seed=10)
    seq = 12
    hidden = np.random.default_rng(10).standard_normal((1, seq, HIDDEN)).astype(np.float32)
    q, raw_k = mod.project(jnp.asarray(hidden))
    visible = jnp.ones((1, seq), jnp.bool_)

    cos0, sin0 = _cos_sin_table(np.arange(seq))
    got0 = mod.select(
        q,
        raw_k,
        q_cos=jnp.asarray(cos0[None]),
        q_sin=jnp.asarray(sin0[None]),
        k_cos=jnp.asarray(cos0[None]),
        k_sin=jnp.asarray(sin0[None]),
        visible=visible,
        kv_positions=jnp.arange(seq, dtype=jnp.int32)[None],
        q_indices=jnp.arange(seq, dtype=jnp.int32)[None],
    )

    offset = 100
    cos1, sin1 = _cos_sin_table(np.arange(offset, offset + seq))
    got1 = mod.select(
        q,
        raw_k,
        q_cos=jnp.asarray(cos1[None]),
        q_sin=jnp.asarray(sin1[None]),
        k_cos=jnp.asarray(cos1[None]),
        k_sin=jnp.asarray(sin1[None]),
        visible=visible,
        kv_positions=jnp.arange(offset, offset + seq, dtype=jnp.int32)[None],
        q_indices=jnp.arange(seq, dtype=jnp.int32)[None],
    )
    np.testing.assert_array_equal(_as_mask_set(np.asarray(got1), seq), _as_mask_set(np.asarray(got0), seq))


def test_training_score_proxy_preserves_hard_forward_and_carries_gradient():
    mod, *_ = _make_indexer(seed=12)
    seq = 12
    hidden = jnp.asarray(np.random.default_rng(12).standard_normal((1, seq, HIDDEN)).astype(np.float32))
    cos, sin = _cos_sin_table(np.arange(seq))
    cos, sin = jnp.asarray(cos[None]), jnp.asarray(sin[None])

    def loss(h):
        mask, _, score_proxy = mod(
            h,
            q_cos=cos,
            q_sin=sin,
            k_cos=cos,
            k_sin=sin,
            visible=jnp.ones((1, seq), jnp.bool_),
            return_score_proxy=True,
        )
        # Exact hard mask in the primal; selected logits are identically zero.
        ste_bias = jnp.where(mask, score_proxy - jax.lax.stop_gradient(score_proxy), -jnp.inf)
        finite = jnp.where(jnp.isfinite(ste_bias), ste_bias, -1e9)
        weights = jax.nn.softmax(finite, axis=-1)
        values = jnp.arange(seq, dtype=jnp.float32)
        return jnp.sum(weights * values)

    grad = jax.grad(loss)(hidden)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(jnp.abs(grad) > 0)


def test_packed_selection_is_rejected_until_block_semantics_are_preserved():
    mod, *_ = _make_indexer(seed=13)
    seq = 12
    hidden = jnp.asarray(np.random.default_rng(13).standard_normal((1, seq, HIDDEN)).astype(np.float32))
    q, raw_k = mod.project(hidden)
    cos, sin = _cos_sin_table(np.arange(seq))
    segments = jnp.array([[0] * 6 + [1] * 6], jnp.int32)
    with pytest.raises(NotImplementedError, match="packed-document QSA"):
        mod.select(
            q,
            raw_k,
            q_cos=jnp.asarray(cos[None]),
            q_sin=jnp.asarray(sin[None]),
            k_cos=jnp.asarray(cos[None]),
            k_sin=jnp.asarray(sin[None]),
            visible=jnp.ones((1, seq), jnp.bool_),
            q_indices=jnp.arange(seq, dtype=jnp.int32)[None],
            q_segment_ids=segments,
            kv_segment_ids=segments,
        )


def test_left_padding_score_proxy_tracks_actual_block_members():
    mod, *_ = _make_indexer(seed=19)
    seq, pad = 8, 2
    hidden = jnp.asarray(np.random.default_rng(19).standard_normal((1, seq, HIDDEN)).astype(np.float32))
    padded = jnp.pad(hidden, ((0, 0), (pad, 0), (0, 0)))
    cos = jnp.ones((1, seq, ROTARY), jnp.float32)
    sin = jnp.zeros_like(cos)
    pcos = jnp.ones((1, seq + pad, ROTARY), jnp.float32)
    psin = jnp.zeros_like(pcos)

    _, _, score = mod(
        hidden,
        q_cos=cos,
        q_sin=sin,
        k_cos=cos,
        k_sin=sin,
        visible=jnp.ones((1, seq), jnp.bool_),
        return_score_proxy=True,
    )
    _, _, padded_score = mod(
        padded,
        q_cos=pcos,
        q_sin=psin,
        k_cos=pcos,
        k_sin=psin,
        visible=jnp.concatenate([jnp.zeros((1, pad), jnp.bool_), jnp.ones((1, seq), jnp.bool_)], axis=1),
        return_score_proxy=True,
    )
    np.testing.assert_allclose(np.asarray(padded_score[:, :, pad:, pad:]), np.asarray(score), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(padded_score[..., :pad]), 0.0)
