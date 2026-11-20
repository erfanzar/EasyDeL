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

"""Tests for the batched speculative rejection sampler (accelerator only)."""

from __future__ import annotations

import jax
import numpy as np
from easydel.inference.esurge.runners.spec.rejection import (
    DRAFT_PROB_PENALTY,
    rejection_sample_from_uniforms,
    spec_rejection_sample,
)
from jax import numpy as jnp

# Uniforms closer than this to a decision boundary are redrawn, so float32 device math and the
# float64 reference cannot land on opposite sides of a boundary through rounding alone.
_MARGIN = 1e-3


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    return x - m - np.log(np.sum(np.exp(x - m), axis=-1, keepdims=True))


def _reference_row(target, draft, tokens, verified, u_acc, u_smp):
    """Independent float64 per-row loop implementing the acceptance rule."""
    target = target.astype(np.float64)
    draft = draft.astype(np.float64)
    accepted = 0
    final = None
    for j in range(verified):
        tok = int(tokens[j])
        t_lp, d_lp = target[j, tok], draft[j, tok]
        accept_lp = -np.inf if t_lp == -np.inf else min(t_lp - d_lp, 0.0)
        if u_acc[j] < np.exp(accept_lp):
            accepted += 1
            continue
        weights = np.maximum(np.exp(target[j]) - np.exp(draft[j] - DRAFT_PROB_PENALTY), 0.0)
        final = _ref_inverse_cdf(weights, u_smp[j], target[j])
        break
    if final is None:
        final = _ref_inverse_cdf(np.exp(target[accepted]), u_smp[accepted], target[accepted])
    k = tokens.shape[0]
    emitted = np.full((k + 1,), -1, dtype=np.int64)
    emitted[:accepted] = tokens[:accepted]
    emitted[accepted] = final
    return accepted, emitted


def _ref_inverse_cdf(weights, u, target_row):
    total = float(np.sum(weights))
    if total <= 0:
        return int(np.argmax(target_row))
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if w > 0 and u * total < acc:
            return i
    return int(np.max(np.nonzero(weights > 0)[0]))


def _boundaries(target, draft, tokens):
    """Every uniform value at which some decision of this row could flip."""
    acc_b = []
    for j in range(tokens.shape[0]):
        t_lp, d_lp = target[j, tokens[j]], draft[j, tokens[j]]
        acc_b.append(0.0 if t_lp == -np.inf else float(np.exp(min(t_lp - d_lp, 0.0))))
    smp_b = []
    for j in range(target.shape[0]):
        cands = [np.exp(target[j])]
        if j < draft.shape[0]:
            cands.append(np.maximum(np.exp(target[j]) - np.exp(draft[j] - DRAFT_PROB_PENALTY), 0.0))
        bounds = []
        for w in cands:
            if np.sum(w) > 0:
                bounds.extend((np.cumsum(w) / np.sum(w)).tolist())
        smp_b.append(np.asarray(bounds))
    return np.asarray(acc_b), smp_b


def _safe_uniforms(rng, target, draft, tokens):
    n_rows, k = tokens.shape
    u_acc = rng.uniform(size=(n_rows, k))
    u_smp = rng.uniform(size=(n_rows, k + 1))
    for r in range(n_rows):
        acc_b, smp_b = _boundaries(target[r].astype(np.float64), draft[r].astype(np.float64), tokens[r])
        for j in range(k):
            while abs(u_acc[r, j] - acc_b[j]) < _MARGIN:
                u_acc[r, j] = rng.uniform()
        for j in range(k + 1):
            while smp_b[j].size and np.min(np.abs(smp_b[j] - u_smp[r, j])) < _MARGIN:
                u_smp[r, j] = rng.uniform()
    return u_acc.astype(np.float32), u_smp.astype(np.float32)


def _random_case(rng, n_rows, k, vocab, mask_frac=0.3):
    t_logits = rng.normal(scale=2.0, size=(n_rows, k + 1, vocab))
    d_logits = t_logits[:, :k] + rng.normal(scale=1.0, size=(n_rows, k, vocab))
    # Emulate top-k/top-p filtering: masked target entries are -inf (keep at least the max).
    mask = rng.uniform(size=t_logits.shape) < mask_frac
    mask &= t_logits < np.max(t_logits, axis=-1, keepdims=True)
    t_logits = np.where(mask, -np.inf, t_logits)
    target = _log_softmax_np(t_logits).astype(np.float32)
    draft = _log_softmax_np(d_logits).astype(np.float32)
    tokens = np.zeros((n_rows, k), dtype=np.int32)
    for r in range(n_rows):
        for j in range(k):
            p = np.exp(draft[r, j].astype(np.float64))
            tokens[r, j] = rng.choice(vocab, p=p / p.sum())
    # Bias half the drafts toward the target argmax so long accepted prefixes occur.
    bias = rng.uniform(size=tokens.shape) < 0.5
    tokens = np.where(bias, np.argmax(target[:, :k], axis=-1), tokens).astype(np.int32)
    verified = rng.integers(0, k + 1, size=(n_rows,)).astype(np.int32)
    return target, draft, tokens, verified


def test_matches_numpy_reference_exactly():
    rng = np.random.default_rng(0)
    n_rows, k, vocab = 64, 4, 37
    target, draft, tokens, verified = _random_case(rng, n_rows, k, vocab)
    u_acc, u_smp = _safe_uniforms(rng, target, draft, tokens)

    accepted, emitted = rejection_sample_from_uniforms(
        jnp.asarray(target),
        jnp.asarray(draft),
        jnp.asarray(tokens),
        jnp.asarray(verified),
        jnp.asarray(u_acc),
        jnp.asarray(u_smp),
    )
    accepted, emitted = np.asarray(accepted), np.asarray(emitted)
    assert accepted.dtype == np.int32 and emitted.dtype == np.int32
    assert emitted.shape == (n_rows, k + 1)

    rejections = 0
    for r in range(n_rows):
        ref_acc, ref_emit = _reference_row(target[r], draft[r], tokens[r], verified[r], u_acc[r], u_smp[r])
        assert accepted[r] == ref_acc, r
        np.testing.assert_array_equal(emitted[r], ref_emit, err_msg=f"row {r}")
        rejections += int(ref_acc < verified[r])
    # The case mix must exercise both outcomes and the zero-verified rows.
    assert 0 < rejections < n_rows
    assert np.any(accepted == verified) and np.any(verified == 0)


def test_zero_residual_falls_back_to_target_argmax():
    # Force a zero residual: the draft probability exceeds the target probability wherever the target has mass.
    target = np.full((1, 2, 4), -np.inf, dtype=np.float32)
    target[0, :, 2] = 0.0
    draft = np.log(np.full((1, 1, 4), 0.25, dtype=np.float32))
    draft[0, 0, 2] = 10.0  # unnormalized on purpose: p_draft[2] >> p_target[2] empties the residual
    tokens = np.asarray([[1]], dtype=np.int32)  # target gives -inf => always rejected
    accepted, emitted = rejection_sample_from_uniforms(
        jnp.asarray(target),
        jnp.asarray(draft),
        jnp.asarray(tokens),
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([[0.0]], dtype=jnp.float32),
        jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
    )
    assert int(accepted[0]) == 0
    np.testing.assert_array_equal(np.asarray(emitted)[0], [2, -1])


def test_greedy_rows_reduce_to_argmax_comparison():
    rng = np.random.default_rng(1)
    n_rows, k, vocab = 48, 3, 29
    logits = rng.normal(size=(n_rows, k + 1, vocab))
    arg = np.argmax(logits, axis=-1)
    for fill in (-np.inf, -1e10):
        target = np.where(np.arange(vocab)[None, None, :] == arg[..., None], 0.0, fill).astype(np.float32)
        draft = _log_softmax_np(rng.normal(size=(n_rows, k, vocab))).astype(np.float32)
        tokens = np.where(rng.uniform(size=(n_rows, k)) < 0.7, arg[:, :k], rng.integers(0, vocab, (n_rows, k)))
        tokens = tokens.astype(np.int32)
        verified = rng.integers(0, k + 1, size=(n_rows,)).astype(np.int32)
        accepted, emitted = spec_rejection_sample(
            jnp.asarray(target),
            jnp.asarray(draft),
            jnp.asarray(tokens),
            jnp.asarray(verified),
            jax.random.PRNGKey(7),
            jnp.arange(n_rows, dtype=jnp.int32) * 13,
        )
        accepted, emitted = np.asarray(accepted), np.asarray(emitted)
        for r in range(n_rows):
            exp_acc = 0
            while exp_acc < verified[r] and tokens[r, exp_acc] == arg[r, exp_acc]:
                exp_acc += 1
            expected = np.full((k + 1,), -1)
            expected[:exp_acc] = tokens[r, :exp_acc]
            expected[exp_acc] = arg[r, exp_acc]
            assert accepted[r] == exp_acc, (fill, r)
            np.testing.assert_array_equal(emitted[r], expected, err_msg=f"fill={fill} row={r}")


def test_position_hashed_draws_are_batch_invariant():
    rng = np.random.default_rng(2)
    n_rows, k, vocab = 32, 3, 17
    target, draft, tokens, verified = _random_case(rng, n_rows, k, vocab)
    positions = rng.integers(0, 10**8, size=(n_rows,)).astype(np.int32)
    key = jax.random.PRNGKey(3)
    args = (target, draft, tokens, verified, positions)
    full = [np.asarray(x) for x in spec_rejection_sample(*map(jnp.asarray, args[:4]), key, jnp.asarray(positions))]
    perm = rng.permutation(n_rows)
    permuted = [jnp.asarray(a[perm]) for a in args]
    sub = [np.asarray(x) for x in spec_rejection_sample(*permuted[:4], key, permuted[4])]
    np.testing.assert_array_equal(sub[0], full[0][perm])
    np.testing.assert_array_equal(sub[1], full[1][perm])


def test_first_emitted_token_is_lossless():
    """The first emitted token must follow the target distribution at position 0.

    With N draws over V=6 bins the Pearson statistic is chi-square with 5 degrees of freedom; 35.9 is its
    1e-6 upper quantile. The draft penalty shifts each probability by at most ~5e-5 (about 10 expected
    counts at N=2e5), which adds well under 0.1 to the statistic, so the bound is not loosened by it.
    The draws are seeded, so the test is deterministic.
    """
    rng = np.random.default_rng(4)
    n, k, vocab = 200_000, 2, 6
    q0 = np.asarray([0.05, 0.30, 0.10, 0.25, 0.20, 0.10])
    q1 = np.asarray([0.40, 0.10, 0.10, 0.10, 0.20, 0.10])
    q2 = np.asarray([0.10, 0.10, 0.50, 0.10, 0.10, 0.10])
    p0 = np.asarray([0.30, 0.05, 0.25, 0.10, 0.10, 0.20])
    p1 = np.asarray([0.10, 0.40, 0.10, 0.20, 0.10, 0.10])
    target = np.broadcast_to(np.log(np.stack([q0, q1, q2]))[None], (n, k + 1, vocab)).astype(np.float32)
    draft = np.broadcast_to(np.log(np.stack([p0, p1]))[None], (n, k, vocab)).astype(np.float32)
    tokens = np.stack([rng.choice(vocab, size=n, p=p0), rng.choice(vocab, size=n, p=p1)], axis=1).astype(np.int32)
    verified = np.full((n,), k, dtype=np.int32)

    accepted, emitted = spec_rejection_sample(
        jnp.asarray(target),
        jnp.asarray(draft),
        jnp.asarray(tokens),
        jnp.asarray(verified),
        jax.random.PRNGKey(11),
        jnp.arange(n, dtype=jnp.int32),
    )
    first = np.asarray(emitted)[:, 0]
    assert np.all(first >= 0)
    counts = np.bincount(first, minlength=vocab)
    expected = n * q0
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < 35.9, (chi2, counts.tolist())

    # The acceptance rate at step 0 must equal sum(min(p0, q0)).
    acc_rate = float(np.mean(np.asarray(accepted) >= 1))
    expected_rate = float(np.sum(np.minimum(p0, q0)))
    # Binomial std at N=2e5 is < 1.2e-3; 6 sigma.
    assert abs(acc_rate - expected_rate) < 7e-3, (acc_rate, expected_rate)
