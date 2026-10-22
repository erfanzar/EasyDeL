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

"""``indexer_distill_kl`` operation: float64 NumPy parity, lazy-vs-dense scores, gradient routing.

Reference: per query, the dense attention softmax over ``[window tokens | reachable entries | sink]`` (each part
optional) is summed over heads and L1-normalized over the entries; the loss is the mean over queries reaching at least
one entry of ``KL(target || softmax(scores over reachable entries))``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.indexer import IndexerScoreInputs, indexer_distillation_loss, shared_key_scores
from easydel.operations import OperationRegistry
from easydel.operations.kernels import IndexerDistillKLOp, indexer_distill_kl_op

HI = jax.lax.Precision.HIGHEST


def _reference_kl(q, entries, scores, scaling, *, kv=None, window_allowed=None, sinks=None) -> float:
    """float64 mean KL. ``window_allowed`` is ``[B, S, S]`` bool (token region), ``None`` = no token region."""
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[1]
    total, count = 0.0, 0
    for b in range(batch):
        for s in range(seq):
            reach = np.isfinite(scores[b, s])
            if not reach.any():
                continue
            summed = np.zeros(num_entries)
            for h in range(heads):
                parts = []
                if window_allowed is not None:
                    parts.append(np.where(window_allowed[b, s], (q[b, h, s] @ kv[b].T) * scaling, -np.inf))
                parts.append(np.where(reach, (q[b, h, s] @ entries[b].T) * scaling, -np.inf))
                if sinks is not None:
                    parts.append([sinks[h]])
                logits = np.concatenate(parts)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                off = 0 if window_allowed is None else seq
                summed += probs[off : off + num_entries]
            target = summed / summed.sum()
            row = np.where(reach, scores[b, s], -np.inf)
            log_pred = row - row.max()
            log_pred = log_pred - np.log(np.exp(log_pred).sum())
            total += sum(target[t] * (np.log(target[t]) - log_pred[t]) for t in range(num_entries) if target[t] > 0)
            count += 1
    return total / count


def test_registered_under_its_name():
    assert OperationRegistry.get("indexer_distill_kl") is IndexerDistillKLOp
    assert indexer_distill_kl_op() is indexer_distill_kl_op()


@pytest.mark.parametrize("block_q", [64, 4, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_window_and_sink_match_numpy_reference(block_q, padded):
    """Compressed-window target (window tokens + entries + sinks) equals a dense float64 reference."""
    rng = np.random.default_rng(0)
    batch, heads, seq, dim, ratio, window = 2, 3, 6, 4, 2, 3
    num_entries = seq // ratio
    q = rng.normal(size=(batch, heads, seq, dim))
    kv = rng.normal(size=(batch, seq, dim))
    entries = rng.normal(size=(batch, num_entries, dim))
    sinks = rng.normal(size=(heads,))
    positions = np.arange(seq)
    valid = np.ones((batch, seq), bool)
    if padded:
        valid[0, 2] = False
        valid[1, 4] = False
    allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - window)
    allowed = allowed[None] & valid[:, None, :]
    reach = np.arange(num_entries)[None, :] < ((positions + 1) // ratio)[:, None]  # query 0 reaches nothing
    scores = np.where(reach[None], rng.normal(size=(batch, seq, num_entries)), -np.inf)
    scaling = dim**-0.5
    expected = _reference_kl(q, entries, scores, scaling, kv=kv, window_allowed=allowed, sinks=sinks)
    assert expected > 1e-3

    def loss(q_, scores_):
        return indexer_distillation_loss(
            q_,
            jnp.asarray(entries, jnp.float32),
            scores_,
            scaling=scaling,
            window_kv=jnp.asarray(kv, jnp.float32),
            sinks=jnp.asarray(sinks, jnp.float32),
            token_valid=jnp.asarray(valid),
            window=window,
            precision=HI,
            block_q=block_q,
        )

    q32, scores32 = jnp.asarray(q, jnp.float32), jnp.asarray(scores, jnp.float32)
    np.testing.assert_allclose(float(loss(q32, scores32)), expected, rtol=1e-4)
    grad_q, grad_scores = jax.grad(loss, argnums=(0, 1))(q32, scores32)
    assert not np.any(np.asarray(grad_q))  # attention side is stop-gradient
    grad_scores = np.asarray(grad_scores)
    assert np.isfinite(grad_scores).all() and np.any(grad_scores != 0)
    assert not np.any(grad_scores[~np.broadcast_to(reach[None], grad_scores.shape)])


@pytest.mark.parametrize("block_q", [64, 5, 1])
@pytest.mark.parametrize("with_sink", [False, True])
def test_token_dsa_without_window_matches_numpy_reference(block_q, with_sink):
    """Token DSA (entries = tokens, ratio 1, causal reach, no window region), optionally with sinks."""
    rng = np.random.default_rng(2)
    batch, heads, seq, dim = 2, 2, 9, 4
    q = rng.normal(size=(batch, heads, seq, dim))
    tokens = rng.normal(size=(batch, seq, dim))
    sinks = rng.normal(size=(heads,)) if with_sink else None
    reach = np.arange(seq)[None, :] <= np.arange(seq)[:, None]
    scores = np.where(reach[None], rng.normal(size=(batch, seq, seq)), -np.inf)
    scaling = dim**-0.5
    expected = _reference_kl(q, tokens, scores, scaling, sinks=sinks)
    assert expected > 1e-3

    out = indexer_distill_kl_op()(
        jnp.asarray(q, jnp.float32),
        jnp.asarray(tokens, jnp.float32),
        jnp.asarray(scores, jnp.float32),
        scaling=scaling,
        sinks=None if sinks is None else jnp.asarray(sinks, jnp.float32),
        precision=HI,
        block_q=block_q,
    )
    np.testing.assert_allclose(float(out.loss), expected, rtol=1e-4)
    np.testing.assert_allclose(float(out.num_rows), batch * seq)
    np.testing.assert_allclose(float(out.kl_sum), expected * batch * seq, rtol=1e-4)


def test_per_head_entries_match_shared_entries_broadcast():
    """``[B, H, T, D]`` per-head entries equal the shared ``[B, T, D]`` form when every head sees the same keys."""
    rng = np.random.default_rng(3)
    batch, heads, seq, dim = 1, 3, 7, 4
    q = jnp.asarray(rng.normal(size=(batch, heads, seq, dim)), jnp.float32)
    keys = jnp.asarray(rng.normal(size=(batch, seq, dim)), jnp.float32)
    reach = np.arange(seq)[None, :] <= np.arange(seq)[:, None]
    scores = jnp.asarray(np.where(reach[None], rng.normal(size=(batch, seq, seq)), -np.inf), jnp.float32)
    shared = indexer_distillation_loss(q, keys, scores, scaling=0.5, precision=HI, block_q=3)
    per_head = indexer_distillation_loss(
        q, jnp.broadcast_to(keys[:, None], (batch, heads, seq, dim)), scores, scaling=0.5, precision=HI, block_q=3
    )
    np.testing.assert_allclose(float(per_head), float(shared), rtol=1e-6)


def test_no_reachable_entries_gives_zero_loss():
    q = jnp.ones((1, 2, 4, 3), jnp.float32)
    entries = jnp.ones((1, 2, 3), jnp.float32)
    scores = jnp.full((1, 4, 2), -jnp.inf, jnp.float32)
    out = indexer_distill_kl_op()(q, entries, scores, scaling=1.0)
    assert float(out.loss) == 0.0 and float(out.num_rows) == 0.0


def test_window_without_window_kv_is_rejected():
    q = jnp.ones((1, 1, 2, 2), jnp.float32)
    with pytest.raises(ValueError, match="window_kv"):
        indexer_distill_kl_op()(q, jnp.ones((1, 1, 2)), jnp.zeros((1, 2, 1)), scaling=1.0, window=2)


def _grouped_reference_kl(q, entries, scores, scaling, group, ent_vis_extra) -> float:
    """float64 mean KL for group-level scores ``[B, S, G]``; ``ent_vis_extra`` is a ``[B, S, T]`` token-level mask."""
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[1]
    num_groups = scores.shape[-1]
    total, count = 0.0, 0
    for b in range(batch):
        for s in range(seq):
            reach = np.isfinite(scores[b, s])
            vis = np.repeat(reach, group)[:num_entries] & ent_vis_extra[b, s]
            if not reach.any() or not vis.any():
                continue
            summed = np.zeros(num_entries)
            for h in range(heads):
                logits = np.where(vis, (q[b, h, s] @ entries[b].T) * scaling, -np.inf)
                probs = np.exp(logits - logits.max())
                summed += probs / probs.sum()
            padded = np.zeros(num_groups * group)
            padded[:num_entries] = summed
            target = padded.reshape(num_groups, group).sum(-1)
            target = target / target.sum()
            row = np.where(reach, scores[b, s], -np.inf)
            log_pred = row - row.max()
            log_pred = log_pred - np.log(np.exp(log_pred).sum())
            total += sum(target[g] * (np.log(target[g]) - log_pred[g]) for g in range(num_groups) if target[g] > 0)
            count += 1
    return total / count


@pytest.mark.parametrize("block_q", [64, 4, 1])
@pytest.mark.parametrize("group", [2, 3])
def test_grouped_scores_with_token_masks_match_numpy_reference(block_q, group):
    """Pooled / block indexers: group-level scores, token-level causality, segment ids and padding on the target."""
    rng = np.random.default_rng(4)
    batch, heads, seq, dim = 2, 2, 10, 4
    num_entries = seq
    num_groups = -(-num_entries // group)
    q = rng.normal(size=(batch, heads, seq, dim))
    tokens = rng.normal(size=(batch, num_entries, dim))
    positions = np.broadcast_to(np.arange(seq), (batch, seq))
    segments = np.stack([np.zeros(seq, int), (np.arange(seq) >= 6).astype(int)])
    valid = np.ones((batch, num_entries), bool)
    valid[0, 4] = False
    group_start = np.arange(num_groups) * group
    reach = group_start[None, :] <= np.arange(seq)[:, None]  # a group is reachable once its first token is
    scores = np.where(reach[None], rng.normal(size=(batch, seq, num_groups)), -np.inf)
    extra = (
        valid[:, None, :]
        & (positions[:, None, :] <= positions[:, :, None])
        & (segments[:, None, :] == segments[:, :, None])
    )
    scaling = dim**-0.5
    expected = _grouped_reference_kl(q, tokens, scores, scaling, group, extra)
    assert expected > 1e-3

    def loss(scores_):
        return indexer_distillation_loss(
            jnp.asarray(q, jnp.float32),
            jnp.asarray(tokens, jnp.float32),
            scores_,
            scaling=scaling,
            precision=HI,
            block_q=block_q,
            entry_group=group,
            entry_valid=jnp.asarray(valid),
            q_positions=jnp.asarray(positions),
            entry_positions=jnp.asarray(positions),
            q_segment_ids=jnp.asarray(segments),
            entry_segment_ids=jnp.asarray(segments),
        )

    scores32 = jnp.asarray(scores, jnp.float32)
    np.testing.assert_allclose(float(loss(scores32)), expected, rtol=1e-4)
    grad = np.asarray(jax.grad(loss)(scores32))
    assert np.isfinite(grad).all() and np.any(grad != 0)
    assert not np.any(grad[~np.broadcast_to(reach[None], grad.shape)])


def test_grouped_score_width_mismatch_is_rejected():
    q = jnp.ones((1, 1, 4, 2), jnp.float32)
    with pytest.raises(ValueError, match="candidates"):
        indexer_distill_kl_op()(q, jnp.ones((1, 4, 2)), jnp.zeros((1, 4, 4)), scaling=1.0, entry_group=2)


def test_half_of_a_mask_pair_is_rejected():
    q = jnp.ones((1, 1, 2, 2), jnp.float32)
    with pytest.raises(ValueError, match="q_positions"):
        indexer_distill_kl_op()(q, jnp.ones((1, 2, 2)), jnp.zeros((1, 2, 2)), scaling=1.0, q_positions=jnp.zeros((1, 2)))


@pytest.mark.parametrize("block_q", [64, 4, 3])
def test_lazy_score_inputs_match_dense_scores(block_q):
    """Per-tile scores from IndexerScoreInputs give the same loss and indexer gradients as dense scores."""
    rng = np.random.default_rng(1)
    batch, heads, seq, dim, ratio, window = 2, 3, 10, 4, 2, 3
    ih, idim = 2, 5
    num_entries = seq // ratio
    q = jnp.asarray(rng.normal(size=(batch, heads, seq, dim)), jnp.float32)
    kv = jnp.asarray(rng.normal(size=(batch, seq, dim)), jnp.float32)
    entries = jnp.asarray(rng.normal(size=(batch, num_entries, dim)), jnp.float32)
    sinks = jnp.asarray(rng.normal(size=(heads,)), jnp.float32)
    valid = jnp.ones((batch, seq), bool).at[0, 3].set(False)
    iq = jnp.asarray(rng.normal(size=(batch, seq, ih, idim)), jnp.float32)
    iw = jnp.asarray(rng.normal(size=(batch, seq, ih)), jnp.float32)
    ik = jnp.asarray(rng.normal(size=(batch, num_entries, idim)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None], (batch, seq))
    kwargs = dict(
        scaling=dim**-0.5, window_kv=kv, sinks=sinks, token_valid=valid, window=window, precision=HI, block_q=block_q
    )

    def lazy(a, b, c):
        return indexer_distillation_loss(q, entries, IndexerScoreInputs(a, b, c, positions, ratio, HI), **kwargs)

    def dense(a, b, c):
        return indexer_distillation_loss(q, entries, shared_key_scores(a, b, c, positions, ratio, HI), **kwargs)

    lv, lg = jax.value_and_grad(lazy, argnums=(0, 1, 2))(iq, iw, ik)
    dv, dg = jax.value_and_grad(dense, argnums=(0, 1, 2))(iq, iw, ik)
    assert float(dv) > 1e-3
    np.testing.assert_allclose(float(lv), float(dv), rtol=1e-5)
    for a, b in zip(lg, dg, strict=True):
        assert np.isfinite(np.asarray(a)).all() and np.any(np.asarray(b) != 0)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)
