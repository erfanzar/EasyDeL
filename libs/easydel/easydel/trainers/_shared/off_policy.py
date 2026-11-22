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

"""Fail-closed off-policy contract between the rollout (behaviour) policy and the learner.

Online RL trainers sample completions with a rollout engine (eSurge) and then score them
with the learner. The two log-probabilities of the same sampled token can drift apart
(stale weights, quantisation, mismatched sampling filters). This module measures that
drift and turns it into an explicit contract:

* per-token log ratio ``lr_t = clip(logp_learner_t - logp_behavior_t, -20, 20)``;
  non-finite values are replaced by ``0`` and counted,
* per-token K3 estimate ``k3_t = expm1(lr_t) - lr_t`` (non-negative, ``KL(behavior || learner)``),
* per-sequence mean K3 over action tokens, used to reject drifting sequences,
* batch diagnostics (token-mean K3, ESS, chi-square, tail fraction, acceptance fraction).

Every function is a pure ``jnp`` function and is safe to call inside ``jit``. Batch
diagnostics are produced as additive sums (:data:`SUM_PREFIX` keys) so that gradient
accumulation, which averages per-microbatch metrics by ``1/K``, preserves every ratio;
:func:`summarize_behavior_sums` turns the sums into the final metrics.
"""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
from jax import numpy as jnp

LOG_RATIO_CLIP = 20.0
"""Absolute bound applied to the behaviour log ratio before any exponentiation."""

TAIL_LOG_RATIO = 0.25
"""``|lr_t|`` above which a token counts towards ``off_policy/tail_fraction``."""

SUM_PREFIX = "off_policy_sums/"
"""Metric-key prefix of the additive sums consumed by :func:`summarize_behavior_sums`."""

_SUM_KEYS = (
    "tokens",
    "k3_sum",
    "w_sum",
    "w2_sum",
    "chi2_sum",
    "tail_count",
    "nonfinite_count",
    "eligible_rows",
    "accepted_rows",
)


class BehaviorPolicyTerms(tp.NamedTuple):
    """Per-row behaviour-drift terms for one (micro)batch.

    Attributes:
        log_ratio: Clipped, sanitised ``lr_t`` of shape ``[batch, seq]`` (``0`` off the action mask).
        nonfinite: Boolean ``[batch, seq]``; action tokens whose raw log ratio was non-finite.
        sequence_k3: Per-sequence mean K3, shape ``[batch, 1]``.
        accept: Boolean ``[batch, 1]``; ``False`` for rejected rows. Rows without action
            tokens are reported as accepted because they carry no loss.
        sums: Additive batch sums (see :data:`SUM_PREFIX`).
    """

    log_ratio: jax.Array
    nonfinite: jax.Array
    sequence_k3: jax.Array
    accept: jax.Array
    sums: dict[str, jax.Array]


def behavior_log_ratio(
    learner_logps: jax.Array,
    behavior_logps: jax.Array,
    action_mask: jax.Array,
    clip: float = LOG_RATIO_CLIP,
) -> tuple[jax.Array, jax.Array]:
    """Sanitised learner/behaviour log ratio.

    ``lr_t = clip(logp_learner_t - logp_behavior_t, -clip, clip)`` on action tokens. Tokens
    whose raw difference is non-finite (``nan``/``inf``) get ``lr_t = 0`` and are flagged.
    The learner log-probabilities are treated as constants (``stop_gradient``).

    Args:
        learner_logps: Learner log-probs of the sampled tokens, ``[batch, seq]``.
        behavior_logps: Rollout-engine log-probs of the same tokens, ``[batch, seq]``.
        action_mask: Action-token mask, ``[batch, seq]``.
        clip: Absolute log-ratio bound.

    Returns:
        ``(log_ratio, nonfinite)``; ``log_ratio`` is ``0`` off the mask and ``nonfinite`` is
        a boolean array that is ``False`` off the mask.
    """
    mask = action_mask > 0
    raw = jax.lax.stop_gradient(learner_logps).astype(jnp.float32) - jax.lax.stop_gradient(behavior_logps).astype(
        jnp.float32
    )
    finite = jnp.isfinite(raw)
    nonfinite = mask & ~finite
    log_ratio = jnp.where(mask & finite, jnp.clip(jnp.where(finite, raw, 0.0), -clip, clip), 0.0)
    return log_ratio, nonfinite


def token_k3(log_ratio: jax.Array) -> jax.Array:
    """Per-token K3 estimate ``k3_t = expm1(lr_t) - lr_t`` (always ``>= 0``)."""
    return jnp.expm1(log_ratio) - log_ratio


def behavior_policy_terms(
    learner_logps: jax.Array,
    behavior_logps: jax.Array,
    action_mask: jax.Array,
    *,
    k3_threshold: float | None = None,
) -> BehaviorPolicyTerms:
    """Compute behaviour-drift terms, the per-row rejection decision and additive sums.

    With ``n_b`` action tokens in row ``b``:

    * ``K3_b = (1 / n_b) * sum_t k3_t``,
    * row ``b`` is *eligible* when ``n_b > 0``,
    * an eligible row is *accepted* when it has no non-finite token and, when
      ``k3_threshold`` is set, ``K3_b <= k3_threshold``.

    Sums (over action tokens, ``w_t = exp(lr_t)``): ``tokens = sum n_b``,
    ``k3_sum = sum k3_t``, ``w_sum = sum w_t``, ``w2_sum = sum w_t^2``,
    ``chi2_sum = sum (w_t - 1)^2``, ``tail_count = #{|lr_t| > 0.25}``,
    ``nonfinite_count``, ``eligible_rows`` and ``accepted_rows``.

    Args:
        learner_logps: Learner log-probs ``[batch, seq]``.
        behavior_logps: Behaviour log-probs ``[batch, seq]``.
        action_mask: Action-token mask ``[batch, seq]``.
        k3_threshold: Optional per-sequence K3 rejection bound.

    Returns:
        A :class:`BehaviorPolicyTerms`.
    """
    mask_f = (action_mask > 0).astype(jnp.float32)
    log_ratio, nonfinite = behavior_log_ratio(learner_logps, behavior_logps, action_mask)
    k3 = token_k3(log_ratio) * mask_f
    row_tokens = jnp.sum(mask_f, axis=-1, keepdims=True)
    sequence_k3 = jnp.sum(k3, axis=-1, keepdims=True) / jnp.maximum(row_tokens, 1.0)
    eligible = row_tokens > 0
    accepted = eligible & ~jnp.any(nonfinite, axis=-1, keepdims=True)
    if k3_threshold is not None:
        accepted = accepted & (sequence_k3 <= k3_threshold)
    weights = jnp.exp(log_ratio) * mask_f
    sums = {
        "tokens": jnp.sum(mask_f),
        "k3_sum": jnp.sum(k3),
        "w_sum": jnp.sum(weights),
        "w2_sum": jnp.sum(jnp.square(weights)),
        "chi2_sum": jnp.sum(jnp.square(jnp.exp(log_ratio) - 1.0) * mask_f),
        "tail_count": jnp.sum(((jnp.abs(log_ratio) > TAIL_LOG_RATIO) & (mask_f > 0)).astype(jnp.float32)),
        "nonfinite_count": jnp.sum(nonfinite.astype(jnp.float32)),
        "eligible_rows": jnp.sum(eligible.astype(jnp.float32)),
        "accepted_rows": jnp.sum(accepted.astype(jnp.float32)),
    }
    return BehaviorPolicyTerms(
        log_ratio=log_ratio,
        nonfinite=nonfinite,
        sequence_k3=sequence_k3,
        accept=accepted | ~eligible,
        sums=sums,
    )


def add_behavior_sums(
    accumulated: dict[str, jax.Array] | None,
    sums: collections.abc.Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
    """Element-wise add two sum dicts (``accumulated`` may be ``None``)."""
    if accumulated is None:
        return dict(sums)
    return {key: accumulated[key] + sums[key] for key in _SUM_KEYS}


def summarize_behavior_sums(
    sums: collections.abc.Mapping[str, jax.Array],
    *,
    accumulation_steps: int = 1,
) -> dict[str, jax.Array]:
    """Turn additive sums into the logged ``off_policy/*`` diagnostics.

    With ``N`` action tokens:

    * ``off_policy/behavior_k3 = sum k3_t / N``,
    * ``off_policy/ess = (sum w_t)^2 / (N * sum w_t^2)`` (``1`` means on-policy),
    * ``off_policy/chi2 = sum (w_t - 1)^2 / N``,
    * ``off_policy/tail_fraction = #{|lr_t| > 0.25} / N``,
    * ``off_policy/accepted_fraction = accepted_rows / eligible_rows`` (``1`` when no row is eligible),
    * ``off_policy/nonfinite_tokens``: the non-finite token count.

    Ratios are invariant to the ``1/K`` averaging applied by gradient accumulation; the
    count is rescaled by ``accumulation_steps``.

    Args:
        sums: Mapping with the keys produced by :func:`behavior_policy_terms`.
        accumulation_steps: Number of microbatches the sums were averaged over.

    Returns:
        Dict of scalar float32 metrics.
    """
    tokens = sums["tokens"]
    safe_tokens = jnp.maximum(tokens, 1e-12)
    has_tokens = tokens > 0
    ess = jnp.where(
        has_tokens,
        jnp.square(sums["w_sum"]) / jnp.maximum(tokens * sums["w2_sum"], 1e-30),
        1.0,
    )
    eligible = sums["eligible_rows"]
    return {
        "off_policy/behavior_k3": jnp.where(has_tokens, sums["k3_sum"] / safe_tokens, 0.0),
        "off_policy/ess": ess,
        "off_policy/chi2": jnp.where(has_tokens, sums["chi2_sum"] / safe_tokens, 0.0),
        "off_policy/tail_fraction": jnp.where(has_tokens, sums["tail_count"] / safe_tokens, 0.0),
        "off_policy/accepted_fraction": jnp.where(
            eligible > 0, sums["accepted_rows"] / jnp.maximum(eligible, 1e-12), 1.0
        ),
        "off_policy/nonfinite_tokens": sums["nonfinite_count"] * float(max(int(accumulation_steps), 1)),
    }


def split_behavior_sums(
    other_metrics: collections.abc.Mapping[str, jax.Array] | None,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Separate :data:`SUM_PREFIX` sums from the other metrics.

    Returns:
        ``(remaining_metrics, sums)``; ``sums`` is empty when no behaviour logprobs were seen.
    """
    remaining: dict[str, jax.Array] = {}
    sums: dict[str, jax.Array] = {}
    for key, value in (other_metrics or {}).items():
        if key.startswith(SUM_PREFIX):
            sums[key[len(SUM_PREFIX) :]] = value
        else:
            remaining[key] = value
    return remaining, sums


def acceptance_ok(accepted_fraction: jax.Array, min_acceptance: float | None) -> jax.Array:
    """``accepted / eligible >= min_acceptance`` (always ``True`` when ``min_acceptance`` is ``None``)."""
    if min_acceptance is None:
        return jnp.asarray(True)
    return accepted_fraction >= min_acceptance


def behavior_parity_ok(
    behavior_k3: jax.Array,
    tail_fraction: jax.Array,
    *,
    k3_max: float,
    tail_fraction_max: float,
) -> jax.Array:
    """First-step parity check: ``behavior_k3 <= k3_max`` and ``tail_fraction <= tail_fraction_max``."""
    return (behavior_k3 <= k3_max) & (tail_fraction <= tail_fraction_max)


def check_startup_behavior_parity(
    metrics: collections.abc.Mapping[str, tp.Any] | None,
    *,
    k3_max: float,
    tail_fraction_max: float,
) -> None:
    """Host-side first-step parity gate; raises when the rollout engine disagrees with the learner.

    Args:
        metrics: Step metrics carrying ``off_policy/behavior_k3`` and ``off_policy/tail_fraction``.
        k3_max: Bound on the token-mean behaviour K3.
        tail_fraction_max: Bound on the fraction of tokens with ``|lr_t| > 0.25``.

    Raises:
        RuntimeError: When the diagnostics are missing or exceed either bound.
    """
    metrics = metrics or {}
    if "off_policy/behavior_k3" not in metrics or "off_policy/tail_fraction" not in metrics:
        raise RuntimeError(
            "startup_behavior_k3_max is set but the first training step produced no behaviour diagnostics; "
            "the rollout engine did not return sampling log-probabilities."
        )
    behavior_k3 = float(jax.device_get(metrics["off_policy/behavior_k3"]))
    tail_fraction = float(jax.device_get(metrics["off_policy/tail_fraction"]))
    if not (behavior_k3 <= k3_max and tail_fraction <= tail_fraction_max):
        raise RuntimeError(
            "Behaviour-policy parity check failed on the first training step: "
            f"token-mean K3={behavior_k3:.6g} (max {k3_max:.6g}), "
            f"tail fraction={tail_fraction:.6g} (max {tail_fraction_max:.6g}). "
            "The rollout engine's log-probabilities do not match the learner's. Likely causes: rollout engine "
            "weights not in sync with the learner, a quantisation mismatch between rollout and learner, or "
            "sampling temperature / top-k / top-p filters that the learner's logprob computation does not apply. "
            "The first step's update was skipped."
        )
