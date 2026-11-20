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

"""Batched, jitted speculative-decoding rejection sampling for every row at once.

One device call verifies a whole batch of draft windows. The acceptance rule:

* draft ``j`` of row ``r`` (``j < num_verified[r]``) is accepted when
  ``u_accept[r, j] < exp(min(target_lp[tok] - draft_lp[tok], 0))``; a token that
  the (filtered) target gives ``-inf`` is never accepted;
* acceptance is a prefix: verification stops at the first rejected draft;
* on rejection at position ``j`` the corrected token is drawn from the residual
  ``max(p_target - p_draft * exp(-DRAFT_PROB_PENALTY), 0)`` at ``j``; if that
  residual has zero mass the first maximum of the target log-probs is used;
* when every verified draft is accepted the bonus token is drawn from the target
  distribution at position ``num_verified[r]``.

The small ``DRAFT_PROB_PENALTY`` keeps the residual non-empty under rounding; it
biases each output probability by at most ``1 - exp(-DRAFT_PROB_PENALTY)``
(about ``5e-5``). Categorical draws use inverse-CDF sampling from one uniform,
so results are a deterministic function of the uniforms. Uniforms come either
from the caller (:func:`rejection_sample_from_uniforms`) or from a key folded
with position-hashed counters (:func:`spec_rejection_uniforms`), which makes a
row's draw depend only on ``(key, position, draft step, stream)`` and not on the
batch composition.

A greedy row, whose target is a point mass (``0`` at the argmax and ``-inf`` or a
huge negative value elsewhere), reduces exactly to ``argmax(target) == draft``
comparison with the argmax as the corrected/bonus token.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp

DRAFT_PROB_PENALTY = 5e-5
"""Log-space discount on the draft probability inside the rejection residual."""

DRAFT_POS_HASH_MUL = 10000019
"""Multiplier mixing the draft step into the per-row sample position."""

_STREAM_BONUS = 0
_STREAM_REJECT = 1
_STREAM_ACCEPT = 2


def _inverse_cdf_sample(weights: jax.Array, uniforms: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw one index per row from unnormalized non-negative weights.

    Args:
        weights: ``[R, V]`` float32 non-negative weights.
        uniforms: ``[R]`` float32 uniforms in ``[0, 1)``.

    Returns:
        Tuple ``(index, total)``: ``[R]`` int32 sampled index (the first index whose
        cumulative weight exceeds ``u * total``, clipped to the last positive
        weight), and ``[R]`` float32 total mass.
    """
    vocab = weights.shape[-1]
    cdf = jnp.cumsum(weights, axis=-1)
    total = cdf[:, -1]
    threshold = uniforms * total
    idx = jnp.sum((cdf <= threshold[:, None]).astype(jnp.int32), axis=-1)
    positions = jnp.arange(vocab, dtype=jnp.int32)[None, :]
    last_positive = jnp.max(jnp.where(weights > 0, positions, 0), axis=-1)
    return jnp.minimum(idx, last_positive).astype(jnp.int32), total


@jax.jit
def rejection_sample_from_uniforms(
    target_log_probs: jax.Array,
    draft_log_probs: jax.Array,
    draft_tokens: jax.Array,
    num_verified: jax.Array,
    accept_uniforms: jax.Array,
    sample_uniforms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Verify all rows' draft windows with caller-supplied uniforms.

    Args:
        target_log_probs: ``[R, k+1, V]`` filtered target log-probs; position ``j``
            is the target distribution for draft ``j`` and position ``k`` (or
            ``num_verified[r]``) the bonus distribution.
        draft_log_probs: ``[R, k, V]`` drafter log-probs the drafts were sampled from.
        draft_tokens: ``[R, k]`` int32 draft token ids.
        num_verified: ``[R]`` int32 number of drafts to verify per row, ``0..k``.
        accept_uniforms: ``[R, k]`` float32 uniforms in ``[0, 1)`` for acceptance tests.
        sample_uniforms: ``[R, k+1]`` float32 uniforms in ``[0, 1)``; entry ``j`` is
            used for the residual (``j < num_verified``) or bonus (``j == num_verified``)
            draw at position ``j``.

    Returns:
        Tuple ``(accepted, emitted)``: ``accepted`` is ``[R]`` int32 count of leading
        accepted drafts; ``emitted`` is ``[R, k+1]`` int32 holding the accepted drafts,
        then the corrected/bonus token, padded with ``-1``.
    """
    target = target_log_probs.astype(jnp.float32)
    draft = draft_log_probs.astype(jnp.float32)
    tokens = draft_tokens.astype(jnp.int32)
    n_rows, k_plus_1, _vocab = target.shape
    k = k_plus_1 - 1
    verified = jnp.clip(num_verified.astype(jnp.int32), 0, k)

    if k > 0:
        t_lp = jnp.take_along_axis(target[:, :k, :], tokens[:, :, None], axis=-1)[..., 0]
        d_lp = jnp.take_along_axis(draft, tokens[:, :, None], axis=-1)[..., 0]
        accept_lp = jnp.where(t_lp == -jnp.inf, -jnp.inf, jnp.minimum(t_lp - d_lp, 0.0))
        steps = jnp.arange(k, dtype=jnp.int32)[None, :]
        accept = (accept_uniforms.astype(jnp.float32) < jnp.exp(accept_lp)) & (steps < verified[:, None])
        accepted = jnp.sum(jnp.cumprod(accept.astype(jnp.int32), axis=-1), axis=-1).astype(jnp.int32)
    else:
        accepted = jnp.zeros((n_rows,), dtype=jnp.int32)
    rejected = accepted < verified

    row_ids = jnp.arange(n_rows, dtype=jnp.int32)
    target_row = target[row_ids, accepted]
    p_target = jnp.exp(target_row)
    if k > 0:
        draft_row = draft[row_ids, jnp.minimum(accepted, k - 1)]
        residual = jnp.maximum(p_target - jnp.exp(draft_row - DRAFT_PROB_PENALTY), 0.0)
        weights = jnp.where(rejected[:, None], residual, p_target)
    else:
        weights = p_target
    u_sample = sample_uniforms.astype(jnp.float32)[row_ids, accepted]
    sampled, mass = _inverse_cdf_sample(weights, u_sample)
    fallback = jnp.argmax(target_row, axis=-1).astype(jnp.int32)
    final = jnp.where(mass > 0, sampled, fallback)

    cols = jnp.arange(k_plus_1, dtype=jnp.int32)[None, :]
    padded_drafts = jnp.concatenate([tokens, jnp.full((n_rows, 1), -1, dtype=jnp.int32)], axis=1)
    emitted = jnp.where(
        cols < accepted[:, None],
        padded_drafts,
        jnp.where(cols == accepted[:, None], final[:, None], jnp.int32(-1)),
    )
    return accepted, emitted.astype(jnp.int32)


def spec_rejection_uniforms(
    rng_key: jax.Array,
    positions: jax.Array,
    num_verified: jax.Array,
    num_drafts: int,
) -> tuple[jax.Array, jax.Array]:
    """Derive position-hashed acceptance and sampling uniforms for every row.

    Row ``r`` at draft step ``j`` uses counter ``positions[r] + j * DRAFT_POS_HASH_MUL``
    (uint32 wraparound) folded into ``rng_key``, then one of three streams
    (bonus, reject, accept), so draws are reproducible per (key, position, step).

    Args:
        rng_key: JAX PRNG key (raw ``uint32[2]`` or typed).
        positions: ``[R]`` int32/uint32 per-row sample positions (e.g. the
            sequence position of the first verified token, or a per-request seed).
        num_verified: ``[R]`` int32 drafts verified per row; step ``num_verified``
            takes its sample uniform from the bonus stream, earlier steps from
            the reject stream.
        num_drafts: Static ``k``.

    Returns:
        Tuple ``(accept_uniforms [R, k], sample_uniforms [R, k+1])`` float32 in ``[0, 1)``.
    """
    steps = jnp.arange(num_drafts + 1, dtype=jnp.uint32)
    counters = positions.astype(jnp.uint32)[:, None] + steps[None, :] * jnp.uint32(DRAFT_POS_HASH_MUL)

    def _draw(counter: jax.Array, stream: int) -> jax.Array:
        sub = jax.random.fold_in(jax.random.fold_in(rng_key, counter), stream)
        return jax.random.uniform(sub, (), dtype=jnp.float32)

    def _grid(stream: int) -> jax.Array:
        return jax.vmap(jax.vmap(lambda c: _draw(c, stream)))(counters)

    accept_u = _grid(_STREAM_ACCEPT)[:, :num_drafts]
    reject_u = _grid(_STREAM_REJECT)
    bonus_u = _grid(_STREAM_BONUS)
    is_bonus = steps.astype(jnp.int32)[None, :] >= num_verified.astype(jnp.int32)[:, None]
    return accept_u, jnp.where(is_bonus, bonus_u, reject_u)


@jax.jit
def spec_rejection_sample(
    target_log_probs: jax.Array,
    draft_log_probs: jax.Array,
    draft_tokens: jax.Array,
    num_verified: jax.Array,
    rng_key: jax.Array,
    positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Batched speculative rejection sampling with position-hashed randomness.

    Args:
        target_log_probs: ``[R, k+1, V]`` filtered target log-probs.
        draft_log_probs: ``[R, k, V]`` drafter log-probs.
        draft_tokens: ``[R, k]`` int32 draft token ids.
        num_verified: ``[R]`` int32 drafts verified per row, ``0..k``.
        rng_key: JAX PRNG key shared by the batch.
        positions: ``[R]`` per-row sample positions hashed into the draws.

    Returns:
        Tuple ``(accepted [R] int32, emitted [R, k+1] int32 padded with -1)``; see
        :func:`rejection_sample_from_uniforms`.
    """
    accept_u, sample_u = spec_rejection_uniforms(rng_key, positions, num_verified, int(draft_tokens.shape[1]))
    return rejection_sample_from_uniforms(
        target_log_probs, draft_log_probs, draft_tokens, num_verified, accept_u, sample_u
    )


__all__ = (
    "DRAFT_POS_HASH_MUL",
    "DRAFT_PROB_PENALTY",
    "rejection_sample_from_uniforms",
    "spec_rejection_sample",
    "spec_rejection_uniforms",
)
