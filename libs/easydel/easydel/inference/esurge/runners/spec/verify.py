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
"""On-device resolution of a speculative step.

Every request row of a speculative step is verified, accepted and given its emitted tokens and its next-draft seed
inside the step's compiled programs (see :meth:`DrafterSpeculation.run_step`). This module holds the traceable pieces
of those programs and the host helpers for their packed output.

Rows are the window rows of the step (``P`` = the step's request bucket). ``rows[p]`` is an int32 vector of the
``ROW_*`` fields below and ``params[p]`` a float32 vector of the ``PARAM_*`` sampling fields.

Acceptance is standard speculative sampling with a point-mass proposal (every draft is its drafter's argmax): a
greedy row accepts a draft equal to the target's argmax; a sampled row accepts draft ``x`` when ``u < p_target(x)``
(``exp(min(log p_target - log q, 0))`` with ``q(x) = 1``), and on a rejection samples the residual
``max(p_target - q, 0)``, which is ``p_target`` without ``x``, or takes the argmax when that residual is empty. The
token after the last draft of a row that accepted every draft is the step sampler's own token, drawn from the target at
the row's last scheduled position.
"""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp
import numpy as np

from easydel.inference.sampling_processors import apply_sampling_constraints

from ...core.binary_search import apply_topk_mask, apply_topp_mask

ROW_LIVE = 0
"""1 when the row emits tokens this step (its scheduled tokens reach the end of its known tokens)."""
ROW_SEQ = 1
"""The request's sequence-buffer row (the packed output's row); -1 for a padding row."""
ROW_OFFSET = 2
"""First hidden row (and flat input id) of the row's scheduled tokens."""
ROW_REAL = 3
"""Real (non-draft) tokens the row scheduled."""
ROW_DRAFTS = 4
"""Drafts the row scheduled after its real tokens (verified this step)."""
ROW_START = 5
"""Position of the row's first scheduled token."""
ROW_DRAFTER = 6
"""The request's row in the drafter's pooled cache; -1 when it has none."""
ROW_FRESH = 7
"""1 when the drafter row starts over (it was just claimed by this request)."""
ROW_GREEDY = 8
"""1 for a greedy request (temperature <= 0)."""
ROW_SPEC = 9
"""1 when the row drafts in this step."""
NUM_ROW_FIELDS = 10

PARAM_TEMPERATURE = 0
PARAM_TOP_P = 1
PARAM_TOP_K = 2
PARAM_MIN_P = 3
NUM_PARAM_FIELDS = 4

_MASKED = -1e10


def filter_logits(
    logits: jax.Array,
    temperature: jax.Array,
    top_p: jax.Array,
    top_k: jax.Array,
    min_p: jax.Array,
) -> jax.Array:
    """Apply one request's top-k, top-p, temperature and min-p filters, in the sampler's order.

    Filters at their inactive value (``top_k <= 0``, ``top_p >= 1``, ``temperature <= 0``, ``min_p <= 0``) are
    skipped. Masked entries get a large negative value.

    Args:
        logits: ``[rows, vocab]`` logits of one request.
        temperature: Scalar temperature.
        top_p: Scalar nucleus threshold.
        top_k: Scalar top-k count.
        min_p: Scalar min-p threshold (probability space, relative to the top probability).

    Returns:
        Filtered float32 logits of the same shape.
    """
    x = logits.astype(jnp.float32)
    x = jax.lax.cond(top_k > 0, lambda v: apply_topk_mask(v, top_k.astype(jnp.int32), _MASKED), lambda v: v, x)
    x = jax.lax.cond(top_p < 1.0, lambda v: apply_topp_mask(v, top_p.astype(jnp.float32), _MASKED), lambda v: v, x)
    x = jax.lax.cond(temperature > 0.0, lambda v: v / temperature.astype(jnp.float32), lambda v: v, x)

    def _min_p(v: jax.Array) -> jax.Array:
        probs = jax.nn.softmax(v, axis=-1)
        keep = probs >= min_p.astype(jnp.float32) * jnp.max(probs, axis=-1, keepdims=True)
        return jnp.where(keep, v, jnp.full_like(v, _MASKED))

    return jax.lax.cond(min_p > 0.0, _min_p, lambda v: v, x)


def gather_rows(
    hidden: jax.Array,
    sampled: jax.Array,
    input_ids: jax.Array,
    rows: jax.Array,
    *,
    num_draft_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather each row's verify stripe, scheduled drafts and sampled token from a step's outputs.

    Stripe column ``j`` of a row is the hidden row that predicts the token after the row's ``j``-th draft: column 0
    predicts the first draft, column ``d`` (``d`` = the row's drafts) is its last scheduled token. Columns past ``d``
    repeat column ``d``.

    Args:
        hidden: ``[T, H]`` hidden rows of the step, one per scheduled token.
        sampled: ``[R >= P]`` tokens the step's sampler drew, in window-row order.
        input_ids: Flat input ids of the step (the token of hidden row ``i`` is ``input_ids[i]``).
        rows: ``[P, NUM_ROW_FIELDS]`` int32 row fields.
        num_draft_tokens: Static maximum drafts per row ``K``.

    Returns:
        ``(stripe [P, K + 1, H], drafts [P, K] (-1 past each row's drafts), sampled [P])``.
    """
    k = int(num_draft_tokens)
    p = int(rows.shape[0])
    num_hidden = int(hidden.shape[0])
    num_ids = int(input_ids.shape[0])
    col = jnp.arange(k + 1, dtype=jnp.int32)[None, :]
    count = rows[:, ROW_DRAFTS][:, None]
    first = rows[:, ROW_OFFSET] + rows[:, ROW_REAL]
    stripe = hidden[jnp.clip((first - 1)[:, None] + jnp.minimum(col, count), 0, num_hidden - 1)]
    draft_col = col[:, :k]
    draft_ids = input_ids[jnp.clip(first[:, None] + draft_col, 0, num_ids - 1)].astype(jnp.int32)
    drafts = jnp.where(draft_col < count, draft_ids, -1)
    return stripe, drafts, sampled[:p].astype(jnp.int32)


class Resolution(typing.NamedTuple):
    """Per-row result of :func:`resolve_rows`.

    Attributes:
        emitted: ``[P, K + 1]`` tokens the row emits (accepted drafts, then the corrected or bonus token), -1 padded;
            all -1 for rows that do not emit.
        accepted: ``[P]`` accepted drafts, -1 for rows that do not emit.
        seed_tokens: ``[P]`` the last emitted token (the next draft's input token).
        seed_positions: ``[P]`` position of the hidden row that seeds the next draft.
        seed_index: ``[P]`` stripe column of that hidden row (the accepted count; 0 for rows that do not emit).
        rng_key: The advanced key.
    """

    emitted: jax.Array
    accepted: jax.Array
    seed_tokens: jax.Array
    seed_positions: jax.Array
    seed_index: jax.Array
    rng_key: jax.Array


def resolve_rows(
    logits: jax.Array | None,
    drafts: jax.Array,
    sampled: jax.Array,
    rows: jax.Array,
    params: jax.Array,
    rng_key: jax.Array,
    logit_bias: jax.Array | None = None,
    allowed_token_ids_mask: jax.Array | None = None,
    bad_words_token_ids: jax.Array | None = None,
) -> Resolution:
    """Verify every row's drafts and choose its emitted tokens (the rule is in the module docstring).

    Args:
        logits: ``[P, K, V]`` target logits at the draft positions (stripe columns ``0..K-1``); ``None`` when
            ``K == 0``.
        drafts: ``[P, K]`` scheduled drafts, -1 past each row's draft count.
        sampled: ``[P]`` the step sampler's token per row (drawn at the row's last scheduled token).
        rows: ``[P, NUM_ROW_FIELDS]`` int32 row fields.
        params: ``[P, NUM_PARAM_FIELDS]`` float32 sampling fields.
        rng_key: Key for the acceptance draws and residual samples.

    Returns:
        The :class:`Resolution`.
    """
    p, k = int(drafts.shape[0]), int(drafts.shape[1])
    count = rows[:, ROW_DRAFTS]
    live = rows[:, ROW_LIVE] > 0
    greedy = rows[:, ROW_GREEDY] > 0
    if k > 0 and logits is not None:
        logits = apply_sampling_constraints(logits, logit_bias, allowed_token_ids_mask, bad_words_token_ids)
        logits = logits.astype(jnp.float32)
        vocab = int(logits.shape[-1])
        argmax = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        rng_key, key_u, key_r = jax.random.split(rng_key, 3)
        filtered = jax.vmap(filter_logits)(
            logits,
            params[:, PARAM_TEMPERATURE],
            params[:, PARAM_TOP_P],
            params[:, PARAM_TOP_K],
            params[:, PARAM_MIN_P],
        )
        log_p = jax.nn.log_softmax(filtered, axis=-1)
        x = jnp.clip(drafts, 0, vocab - 1)
        p_x = jnp.exp(jnp.take_along_axis(log_p, x[..., None], axis=-1)[..., 0])
        accept = jax.random.uniform(key_u, (p, k), dtype=jnp.float32) < p_x
        residual = jnp.where(jnp.arange(vocab, dtype=jnp.int32)[None, None, :] == x[..., None], -jnp.inf, log_p)
        resampled = jax.random.categorical(key_r, residual, axis=-1).astype(jnp.int32)
        # An empty residual (the target puts all its mass on the draft) cannot reject; keep the rule total anyway.
        resampled = jnp.where(p_x >= 1.0, jnp.argmax(filtered, axis=-1).astype(jnp.int32), resampled)
        target = jnp.where(greedy[:, None], argmax, jnp.where(accept, x, resampled))
        valid = jnp.arange(k, dtype=jnp.int32)[None, :] < count[:, None]
        match = jnp.logical_and(target == drafts, valid)
        accepted = jnp.sum(jnp.cumprod(match.astype(jnp.int32), axis=1), axis=1).astype(jnp.int32)
        target = jnp.concatenate([target, jnp.zeros((p, 1), jnp.int32)], axis=1)
    else:
        accepted = jnp.zeros((p,), jnp.int32)
        target = jnp.zeros((p, k + 1), jnp.int32)
    col = jnp.arange(k + 1, dtype=jnp.int32)[None, :]
    # Column j < count holds the target's token at draft j (the draft itself when accepted, the correction at the
    # first rejection); column count is the sampler's bonus token.
    tokens = jnp.where(col < count[:, None], target, sampled[:, None])
    emitted = jnp.where(jnp.logical_and(col <= accepted[:, None], live[:, None]), tokens, -1)
    seed_index = jnp.where(live, accepted, 0)
    seed_tokens = jnp.take_along_axis(tokens, seed_index[:, None], axis=1)[:, 0]
    seed_positions = jnp.maximum(rows[:, ROW_START] + rows[:, ROW_REAL] - 1 + seed_index, 0)
    return Resolution(
        emitted=emitted.astype(jnp.int32),
        accepted=jnp.where(live, accepted, -1).astype(jnp.int32),
        seed_tokens=seed_tokens.astype(jnp.int32),
        seed_positions=seed_positions.astype(jnp.int32),
        seed_index=seed_index.astype(jnp.int32),
        rng_key=rng_key,
    )


def pack_rows(
    rows: jax.Array,
    emitted: jax.Array,
    accepted: jax.Array,
    drafts: jax.Array | None,
    *,
    num_rows: int,
    num_draft_tokens: int,
) -> jax.Array:
    """Pack a step's results into the one int32 array the host reads, ``[num_rows, 2K + 2]``.

    Row ``r`` is sequence-buffer row ``r``. Columns ``[0, K + 1)`` are the emitted tokens, column ``K + 1`` the
    accepted count and ``[K + 2, 2K + 2)`` the next drafts; everything is -1 padded, and rows outside the window are
    all -1.
    """
    k = int(num_draft_tokens)
    p = int(emitted.shape[0])
    if drafts is None:
        drafts = jnp.full((p, k), -1, jnp.int32)
    elif int(drafts.shape[1]) < k:
        drafts = jnp.concatenate([drafts, jnp.full((p, k - int(drafts.shape[1])), -1, jnp.int32)], axis=1)
    packed = jnp.concatenate([emitted, accepted[:, None], drafts.astype(jnp.int32)], axis=1).astype(jnp.int32)
    seq = jnp.where(rows[:, ROW_SEQ] >= 0, rows[:, ROW_SEQ], int(num_rows))
    return jnp.full((int(num_rows), 2 * k + 2), -1, jnp.int32).at[seq].set(packed, mode="drop")


@functools.partial(jax.jit, static_argnames=("num_rows",))
def pack_pooled_drafts(
    seed_tokens: jax.Array,
    drafts: jax.Array,
    output_rows: jax.Array,
    *,
    num_rows: int,
) -> jax.Array:
    """Pack a pooled first draft as ``spec_out``: the previous token in emit[0], accepted=-1, then drafts."""
    k = int(drafts.shape[1])
    packed = jnp.full((int(num_rows), 2 * k + 2), -1, jnp.int32)
    packed = packed.at[output_rows, 0].set(seed_tokens.astype(jnp.int32))
    return packed.at[output_rows, k + 2 :].set(drafts.astype(jnp.int32))


def split_packed(packed: np.ndarray, num_draft_tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host views of a packed step result: ``(emitted [N, K + 1], accepted [N], drafts [N, K])``."""
    k = int(num_draft_tokens)
    return packed[:, : k + 1], packed[:, k + 1], packed[:, k + 2 :]


def merge_packed(parts: typing.Sequence[np.ndarray]) -> np.ndarray:
    """Merge the packed results of a step's windows (each covers its own rows, -1 elsewhere)."""
    out = np.array(parts[0], copy=True)
    for part in parts[1:]:
        mask = np.any(part >= 0, axis=1)
        out[mask] = part[mask]
    return out


__all__ = (
    "NUM_PARAM_FIELDS",
    "NUM_ROW_FIELDS",
    "Resolution",
    "filter_logits",
    "gather_rows",
    "merge_packed",
    "pack_pooled_drafts",
    "pack_rows",
    "resolve_rows",
    "split_packed",
)
