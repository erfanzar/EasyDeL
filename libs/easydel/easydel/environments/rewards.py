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

"""Ordering-safe reward composition for group-relative policy optimization.

Group-relative trainers (GRPO and its relatives) only consume the *ordering* of
rewards inside one ``num_generations`` group: the advantage of a sample is its
reward minus the group baseline, so what matters is which samples rank above
which. Additive shaping terms (length, format, thinking budget, ...) are useful
tie-breakers but are dangerous when they can reorder the primary outcome: a
wrong answer with perfect formatting must never outrank a right answer with
sloppy formatting.

The helpers here make those guarantees explicit and testable:

* :func:`compose_outcome_reward` floors correct samples at
  :func:`correct_answer_floor` and caps incorrect samples at
  ``incorrect_ceiling``, so every correct sample outranks every incorrect one
  regardless of how the shaping terms add up.
* :func:`structural_validity` / :func:`gate_invalid` detect structurally broken
  completions (no EOS, truncation, restarted turns, malformed thinking blocks)
  and remove all of their upside while keeping negative shaping.
* :func:`lexicographic_blend` lets secondary criteria break ties without ever
  outweighing one unit of the primary score.
* :func:`think_budget_penalty` is a piecewise thinking-length penalty.
* :class:`RewardShaper` packages all of the above around any existing reward
  callable as a :class:`~easydel.trainers.reward_protocol.RewardProtocol`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import typing as tp

from easydel.trainers.reward_protocol import RewardProtocol
from easydel.trainers.training_utils import filter_kwargs_for_callable

__all__ = (
    "RewardShaper",
    "StructuralValidity",
    "compose_outcome_reward",
    "correct_answer_floor",
    "count_think_tokens",
    "gate_invalid",
    "lexicographic_blend",
    "structural_validity",
    "think_budget_penalty",
    "validate_structural_config",
)


def _check_outcome_bounds(incorrect_ceiling: float, correct_reward: float) -> None:
    """Raise when the incorrect ceiling does not sit strictly below the correct reward."""
    if not (math.isfinite(incorrect_ceiling) and math.isfinite(correct_reward)):
        raise ValueError("`incorrect_ceiling` and `correct_reward` must be finite.")
    if not incorrect_ceiling < correct_reward:
        raise ValueError(
            f"`incorrect_ceiling` ({incorrect_ceiling}) must be strictly below `correct_reward` ({correct_reward})."
        )


def correct_answer_floor(incorrect_ceiling: float, correct_reward: float) -> float:
    """Return the lowest reward a correct sample may receive after shaping.

    The floor is the midpoint ``incorrect_ceiling + (correct_reward - incorrect_ceiling) / 2``.
    It is strictly above ``incorrect_ceiling``, so a correct sample clamped to the
    floor still outranks the best possible incorrect sample, and leaves headroom
    on both sides for shaping terms to order samples within each class.

    Args:
        incorrect_ceiling: Highest reward an incorrect sample may receive.
        correct_reward: Base reward of a correct sample.

    Returns:
        The floor value.

    Raises:
        ValueError: If ``incorrect_ceiling >= correct_reward``.
    """
    _check_outcome_bounds(float(incorrect_ceiling), float(correct_reward))
    return float(incorrect_ceiling) + (float(correct_reward) - float(incorrect_ceiling)) / 2.0


def compose_outcome_reward(
    correct: bool,
    shaping_terms: collections.abc.Iterable[float | None] = (),
    *,
    correct_reward: float = 1.0,
    incorrect_ceiling: float = 0.5,
    incorrect_reward: float = 0.0,
) -> float:
    """Combine a binary outcome with additive shaping without reordering the outcome.

    The raw value is ``(correct_reward if correct else incorrect_reward) + sum(shaping_terms)``.
    A correct sample is then floored at :func:`correct_answer_floor` and an
    incorrect sample capped at ``incorrect_ceiling``. Because the floor is strictly
    above the ceiling, for any two samples ``c`` (correct) and ``w`` (incorrect)
    ``reward(c) >= floor > ceiling >= reward(w)`` holds whatever the shaping adds.

    Args:
        correct: Whether the sample's outcome is correct.
        shaping_terms: Additive shaping terms; ``None`` entries contribute nothing.
        correct_reward: Base reward of a correct sample.
        incorrect_ceiling: Upper bound for incorrect samples (must be below ``correct_reward``).
        incorrect_reward: Base reward of an incorrect sample. Values above the
            ceiling are capped like any other incorrect total.

    Returns:
        The composed reward.

    Raises:
        ValueError: If ``incorrect_ceiling >= correct_reward``, or a shaping term
            is NaN/infinite (a non-finite term would silently poison the floor
            and ceiling clamps, so it is rejected rather than guessed at).
    """
    floor = correct_answer_floor(incorrect_ceiling, correct_reward)
    terms = [float(term) for term in shaping_terms if term is not None]
    if not all(math.isfinite(term) for term in terms):
        raise ValueError(f"Shaping terms must be finite; got {terms}.")
    shaping = math.fsum(terms)
    if correct:
        return max(float(correct_reward) + shaping, floor)
    return min(float(incorrect_reward) + shaping, float(incorrect_ceiling))


@dataclasses.dataclass(frozen=True)
class StructuralValidity:
    """Structural flags of one completion's token ids.

    Attributes:
        has_eos: An EOS id occurs in the completion.
        truncated: The completion reached ``max_completion_length`` without an EOS.
        restarted: A forbidden id occurs (e.g. a new conversation/turn header
            emitted mid-completion, meaning the model restarted the dialogue).
        extra_turn_ends: More than ``max_turn_ends`` turn-end ids occur.
        think_ok: The thinking block is well formed (see :func:`structural_validity`).
        valid: All checks passed.
    """

    has_eos: bool
    truncated: bool
    restarted: bool
    extra_turn_ends: bool
    think_ok: bool
    valid: bool


def validate_structural_config(
    *,
    eos_ids: collections.abc.Collection[int],
    max_completion_length: int,
    forbidden_ids: collections.abc.Collection[int] = (),
    turn_end_ids: collections.abc.Collection[int] = (),
    max_turn_ends: int = 1,
    think_close_ids: collections.abc.Collection[int] = (),
    require_single_think_close: bool = False,
) -> None:
    """Validate a structural-validity configuration up front.

    Args:
        eos_ids: Token ids that terminate a completion. Must be non-empty.
        max_completion_length: Generation length cap. Must be positive.
        forbidden_ids: Ids that must not appear in a completion.
        turn_end_ids: Ids that end an assistant turn.
        max_turn_ends: Allowed number of turn-end ids. Must be non-negative.
        think_close_ids: Ids that close a thinking block.
        require_single_think_close: Strict thinking mode; requires ``think_close_ids``.

    Raises:
        ValueError: If a strict mode is requested without the ids it needs, or a
            bound is out of range.
    """
    del forbidden_ids
    if not eos_ids:
        raise ValueError("`eos_ids` must contain at least one token id.")
    if int(max_completion_length) < 1:
        raise ValueError(f"`max_completion_length` must be positive; got {max_completion_length}.")
    if int(max_turn_ends) < 0:
        raise ValueError(f"`max_turn_ends` must be non-negative; got {max_turn_ends}.")
    if require_single_think_close and not think_close_ids:
        raise ValueError("`require_single_think_close=True` needs non-empty `think_close_ids`.")
    del turn_end_ids


def structural_validity(
    token_ids: collections.abc.Sequence[int],
    *,
    eos_ids: collections.abc.Collection[int],
    max_completion_length: int,
    forbidden_ids: collections.abc.Collection[int] = (),
    turn_end_ids: collections.abc.Collection[int] = (),
    max_turn_ends: int = 1,
    think_close_ids: collections.abc.Collection[int] = (),
    require_single_think_close: bool = False,
) -> StructuralValidity:
    """Check a completion's token ids for structural problems.

    ``token_ids`` is the generated completion with padding removed. Thinking is
    ``think_ok`` when the number of think-close ids is exactly one in strict mode
    (``require_single_think_close=True``), and at most one otherwise (several
    closing tags are always malformed). A completion is ``valid`` when it has an
    EOS, is not truncated, did not restart, has no extra turn ends, and
    ``think_ok`` holds. An EOS id that is also listed in ``turn_end_ids`` counts
    as a turn end.

    Args:
        token_ids: Completion token ids.
        eos_ids: Ids that terminate a completion.
        max_completion_length: Generation length cap.
        forbidden_ids: Ids whose presence marks a restarted dialogue.
        turn_end_ids: Ids that end an assistant turn.
        max_turn_ends: Allowed number of turn-end ids.
        think_close_ids: Ids that close a thinking block.
        require_single_think_close: Require exactly one think-close id.

    Returns:
        The :class:`StructuralValidity` flags.

    Raises:
        ValueError: If the configuration is invalid (see :func:`validate_structural_config`).
    """
    validate_structural_config(
        eos_ids=eos_ids,
        max_completion_length=max_completion_length,
        forbidden_ids=forbidden_ids,
        turn_end_ids=turn_end_ids,
        max_turn_ends=max_turn_ends,
        think_close_ids=think_close_ids,
        require_single_think_close=require_single_think_close,
    )
    ids = [int(t) for t in token_ids]
    eos, forbidden, turn_end, think_close = (
        set(map(int, s)) for s in (eos_ids, forbidden_ids, turn_end_ids, think_close_ids)
    )
    has_eos = any(t in eos for t in ids)
    truncated = not has_eos and len(ids) >= int(max_completion_length)
    restarted = any(t in forbidden for t in ids)
    extra_turn_ends = sum(t in turn_end for t in ids) > int(max_turn_ends)
    n_close = sum(t in think_close for t in ids)
    think_ok = n_close == 1 if require_single_think_close else n_close <= 1
    valid = has_eos and not truncated and not restarted and not extra_turn_ends and think_ok
    return StructuralValidity(
        has_eos=has_eos,
        truncated=truncated,
        restarted=restarted,
        extra_turn_ends=extra_turn_ends,
        think_ok=think_ok,
        valid=valid,
    )


def gate_invalid(reward: float, valid: bool, *, cap: float = 0.0) -> float:
    """Remove all upside from a structurally invalid sample.

    Args:
        reward: The composed reward.
        valid: Whether the sample is structurally valid.
        cap: Highest reward an invalid sample may keep. :class:`RewardShaper`
            passes ``min(incorrect_reward, incorrect_ceiling)``, the reward of a
            valid incorrect sample with zero shaping.

    Returns:
        ``reward`` when valid, else ``min(cap, reward)`` (negative shaping is kept).
    """
    return float(reward) if valid else min(float(cap), float(reward))


def lexicographic_blend(
    primary: float,
    secondaries: collections.abc.Sequence[float],
    weights: collections.abc.Sequence[float] | None = None,
    *,
    primary_scale: float = 1.0,
    secondary_max: float = 1.0,
    band: float = 0.5,
) -> float:
    """Blend a primary score with secondary tie-breakers lexicographically.

    Each secondary is clipped to ``[0, secondary_max]`` and normalized to
    ``[0, 1]``; ``s`` is their weighted mean (also in ``[0, 1]``). The result is::

        primary + band * primary_scale * s,   with 0 <= band < 1

    ``primary_scale`` is the size of one primary *unit* (the smallest primary
    difference that should matter, e.g. ``1`` for an integer 0-10 grade). The
    secondary contribution lies in ``[0, band * primary_scale]``, a band strictly
    narrower than one unit, which gives the ordering guarantee: whenever
    ``primary_a >= primary_b + primary_scale``,

        blend_a >= primary_a >= primary_b + primary_scale > primary_b + band * primary_scale >= blend_b.

    So secondaries only reorder samples whose primaries are equal (or closer than
    one unit), and the result is never below the primary-only score.

    Args:
        primary: Primary score.
        secondaries: Secondary scores on ``[0, secondary_max]``.
        weights: Non-negative weights, one per secondary; ``None`` means equal weights.
        primary_scale: Size of one primary unit. Must be positive.
        secondary_max: Maximum of the secondary scale. Must be positive.
        band: Fraction of one primary unit the secondaries may span, in ``[0, 1)``.

    Returns:
        The blended score.

    Raises:
        ValueError: On mismatched weights, negative/all-zero weights, or out-of-range parameters.
    """
    if not primary_scale > 0:
        raise ValueError(f"`primary_scale` must be positive; got {primary_scale}.")
    if not secondary_max > 0:
        raise ValueError(f"`secondary_max` must be positive; got {secondary_max}.")
    if not 0.0 <= band < 1.0:
        raise ValueError(f"`band` must be in [0, 1); got {band}.")
    secondaries = [float(s) for s in secondaries]
    if not secondaries:
        return float(primary)
    weights = [1.0] * len(secondaries) if weights is None else [float(w) for w in weights]
    if len(weights) != len(secondaries):
        raise ValueError(f"Got {len(weights)} weights for {len(secondaries)} secondaries.")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be non-negative with a positive sum.")
    normalized = [min(max(s, 0.0), float(secondary_max)) / float(secondary_max) for s in secondaries]
    mean = math.fsum(w * s for w, s in zip(weights, normalized, strict=True)) / math.fsum(weights)
    mean = min(max(mean, 0.0), 1.0)
    return float(primary) + float(band) * float(primary_scale) * mean


def think_budget_penalty(
    n_think_tokens: int,
    *,
    budget: int,
    min_fraction: float = 0.1,
    tolerance: float = 0.5,
    weight: float = 0.2,
) -> float:
    """Piecewise penalty on the number of thinking tokens.

    With ``lo = min_fraction * budget``:

    * ``lo <= n <= budget``: ``0``.
    * ``n < lo``: quadratic, ``-weight * ((lo - n) / lo) ** 2`` (reaches ``-weight`` at ``n = 0``).
    * ``n > budget``: linear in the overshoot and capped,
      ``-weight * min(1, (n - budget) / (tolerance * budget))``; the full penalty
      is reached at ``budget * (1 + tolerance)`` tokens.

    Defaults (``min_fraction=0.1``, ``tolerance=0.5``, ``weight=0.2``) penalize
    only near-empty thinking and overshoots, and keep the maximum penalty small
    enough to act as shaping beside a unit-scale outcome reward. Pair it with
    :func:`compose_outcome_reward` so it can never reorder correct vs. incorrect.

    Args:
        n_think_tokens: Number of thinking tokens.
        budget: Thinking-token budget. Must be positive.
        min_fraction: Lower edge of the free zone as a fraction of ``budget``, in ``[0, 1]``.
        tolerance: Overshoot (fraction of ``budget``) at which the penalty saturates. Must be positive.
        weight: Maximum penalty magnitude. Must be non-negative.

    Returns:
        A non-positive penalty.

    Raises:
        ValueError: If a parameter is out of range.
    """
    if int(budget) < 1:
        raise ValueError(f"`budget` must be positive; got {budget}.")
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError(f"`min_fraction` must be in [0, 1]; got {min_fraction}.")
    if not tolerance > 0:
        raise ValueError(f"`tolerance` must be positive; got {tolerance}.")
    if weight < 0:
        raise ValueError(f"`weight` must be non-negative; got {weight}.")
    n = max(0.0, float(n_think_tokens))
    lo = float(min_fraction) * float(budget)
    if n < lo:
        return -float(weight) * ((lo - n) / lo) ** 2
    if n > budget:
        return -float(weight) * min(1.0, (n - float(budget)) / (float(tolerance) * float(budget)))
    return 0.0


def count_think_tokens(
    token_ids: collections.abc.Sequence[int], think_close_ids: collections.abc.Collection[int]
) -> int:
    """Count thinking tokens as the tokens before the first think-close id.

    Args:
        token_ids: Completion token ids.
        think_close_ids: Ids that close a thinking block.

    Returns:
        The index of the first think-close id, or ``len(token_ids)`` when the
        block is never closed (the whole completion was spent thinking).
    """
    close = set(map(int, think_close_ids))
    for index, token in enumerate(token_ids):
        if int(token) in close:
            return index
    return len(token_ids)


RewardCallable = tp.Callable[..., collections.abc.Sequence[float | None]]


def _call_reward(fn: RewardCallable, payload: collections.abc.Mapping[str, tp.Any], n: int) -> list[float | None]:
    """Call a trainer-style reward callable and check its output length."""
    values = list(fn(**filter_kwargs_for_callable(fn, payload)))
    if len(values) != n:
        raise ValueError(
            f"Reward function {getattr(fn, '__name__', fn)!r} returned {len(values)} values for {n} completions."
        )
    out: list[float | None] = []
    for v in values:
        value = None if v is None else float(v)  # float() first so numpy scalars are caught too
        out.append(value if value is not None and math.isfinite(value) else None)
    return out


class RewardShaper(RewardProtocol):
    """Wrap any reward callable with ordering-safe outcome shaping.

    For each completion:

    1. ``base_reward`` scores it; ``None`` (unscorable) propagates as ``None``.
    2. The sample is *correct* when its base score is ``>= correct_threshold``.
    3. Shaping terms from ``shaping_funcs`` (and the optional thinking-budget
       penalty) are combined with :func:`compose_outcome_reward`, so no shaping
       can make an incorrect sample outrank a correct one.
    4. With ``structural`` set, :func:`gate_invalid` caps structurally invalid
       completions at ``min(incorrect_reward, incorrect_ceiling)``, the reward of
       a valid incorrect sample with zero shaping. So an invalid sample never
       outranks a correct one, nor a valid incorrect one whose shaping is
       non-negative; it can only outrank a valid incorrect sample whose own
       negative shaping pushed it lower.

    Non-finite outputs (NaN/inf, including numpy scalars) of any wrapped callable
    are treated like ``None``: unscorable for ``base_reward``, ``0`` for shaping.

    ``base_reward`` and every entry in ``shaping_funcs`` use the trainer's reward
    calling convention (keyword batch payload in, one value per completion out),
    so plain functions and other :class:`RewardProtocol` instances both work.

    Args:
        base_reward: Outcome reward callable.
        shaping_funcs: Additive shaping callables; ``None`` outputs count as ``0``.
        correct_threshold: Base score at or above which a sample is correct.
        correct_reward: Base reward of a correct sample.
        incorrect_ceiling: Cap for incorrect samples (strictly below ``correct_reward``).
        incorrect_reward: Base reward of an incorrect sample.
        structural: Keyword arguments for :func:`structural_validity` (without
            ``token_ids``). Requires the trainer to forward ``completion_ids``.
        think_budget: Keyword arguments for :func:`think_budget_penalty` (without
            ``n_think_tokens``). Thinking tokens are counted with
            :func:`count_think_tokens`, so ``think_close_ids`` is required.
        weight: Weight in the trainer's multi-reward sum.
        name: Metric name; defaults to ``"shaped_<base name>"``.

    Raises:
        ValueError: On an invalid outcome, structural, or thinking-budget configuration.
    """

    def __init__(
        self,
        base_reward: RewardCallable,
        *,
        shaping_funcs: collections.abc.Sequence[RewardCallable] = (),
        correct_threshold: float = 0.5,
        correct_reward: float = 1.0,
        incorrect_ceiling: float = 0.5,
        incorrect_reward: float = 0.0,
        structural: collections.abc.Mapping[str, tp.Any] | None = None,
        think_budget: collections.abc.Mapping[str, tp.Any] | None = None,
        think_close_ids: collections.abc.Collection[int] = (),
        weight: float | None = None,
        name: str | None = None,
    ) -> None:
        _check_outcome_bounds(float(incorrect_ceiling), float(correct_reward))
        if structural is not None:
            validate_structural_config(**structural)
        if think_budget is not None:
            if not think_close_ids:
                raise ValueError("`think_budget` needs `think_close_ids` to count thinking tokens.")
            think_budget_penalty(0, **think_budget)  # validates the parameters
        base_name = getattr(base_reward, "__name__", type(base_reward).__name__)
        super().__init__(weight=weight, name=name or f"shaped_{base_name}")
        self.base_reward = base_reward
        self.shaping_funcs = tuple(shaping_funcs)
        self.correct_threshold = float(correct_threshold)
        self.correct_reward = float(correct_reward)
        self.incorrect_ceiling = float(incorrect_ceiling)
        self.incorrect_reward = float(incorrect_reward)
        self.structural = dict(structural) if structural is not None else None
        self.think_budget = dict(think_budget) if think_budget is not None else None
        self.think_close_ids = tuple(int(t) for t in think_close_ids)

    def compute_batch(
        self,
        *,
        prompts: tp.Sequence[tp.Any] | None = None,
        completions: tp.Sequence[tp.Any],
        completion_ids: tp.Sequence[tp.Sequence[int]] | None = None,
        **kwargs: tp.Any,
    ) -> list[float | None]:
        """Score a batch: base reward, shaping, outcome composition, structural gate.

        Args:
            prompts: Per-completion prompts.
            completions: Per-completion outputs.
            completion_ids: Per-completion generated token ids (needed for the
                structural gate and the thinking budget).
            **kwargs: Remaining trainer payload, forwarded to the wrapped callables.

        Returns:
            One shaped reward per completion, ``None`` where the base reward is unscorable.

        Raises:
            ValueError: If token ids are needed but missing, or a callable returns the wrong length.
        """
        n = len(completions)
        payload = {"prompts": prompts, "completions": completions, "completion_ids": completion_ids, **kwargs}
        base = _call_reward(self.base_reward, payload, n)
        shaping = [_call_reward(fn, payload, n) for fn in self.shaping_funcs]
        needs_ids = self.structural is not None or self.think_budget is not None
        if needs_ids and (completion_ids is None or len(completion_ids) != n):
            raise ValueError("RewardShaper needs per-completion `completion_ids` for structural/think-budget rules.")
        out: list[float | None] = []
        for i in range(n):
            if base[i] is None:
                out.append(None)
                continue
            terms = [column[i] for column in shaping]
            if self.think_budget is not None:
                assert completion_ids is not None
                n_think = count_think_tokens(completion_ids[i], self.think_close_ids)
                terms.append(think_budget_penalty(n_think, **self.think_budget))
            reward = compose_outcome_reward(
                base[i] >= self.correct_threshold,
                terms,
                correct_reward=self.correct_reward,
                incorrect_ceiling=self.incorrect_ceiling,
                incorrect_reward=self.incorrect_reward,
            )
            if self.structural is not None:
                assert completion_ids is not None
                reward = gate_invalid(
                    reward,
                    structural_validity(completion_ids[i], **self.structural).valid,
                    cap=min(self.incorrect_reward, self.incorrect_ceiling),
                )
            out.append(reward)
        return out
