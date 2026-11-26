# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How much of a host's disk Eyvan may own, and what it gives up first.

Two values and one enum, kept in the capacity layer rather than beside the
staging service that uses them, because process configuration has to validate
them and configuration cannot reach into the control plane -- the host-side
worker loads the same settings, and it is forbidden from importing the server.

Nothing here touches a disk or a record. It is the declaration an operator makes
about a cache; the decisions it licenses live in
``eyvan.server.service.stage``.
"""

from __future__ import annotations

from dataclasses import dataclass

from eyvan._internal.compat import StrEnum


class StageEvictionPolicy(StrEnum):
    """Which staged asset is given up first when a cell is over budget.

    Named and required rather than defaulted. "Evict something" is not an
    operational instruction, and a policy that arrives by default is one nobody
    chose: the two here fail in opposite directions, and which failure a fleet
    prefers is a property of its workloads rather than of this package.
    """

    #: Give up the asset nobody has used for longest. The right choice for a
    #: fleet running many small jobs; wrong for one where a rarely-used 400 GB
    #: dataset is the expensive thing to re-pull.
    LEAST_RECENTLY_USED = "least_recently_used"
    #: Give up the biggest asset first. Frees the budget in the fewest deletions
    #: and therefore the fewest re-pulls, at the cost of always sacrificing the
    #: asset that took longest to fetch.
    LARGEST_FIRST = "largest_first"


@dataclass(frozen=True, slots=True)
class CellStagePolicy:
    """The operator-declared bounds one fleet stages and evicts under.

    ``budget_bytes`` is a hard per-cell ceiling on charged stage bytes, not a
    target to hover around: nothing is admitted that would cross it. The two
    limits are separate because they answer different questions -- the budget is
    how much of a host's disk Eyvan may own, the per-action ceiling is how large
    a single asset may be before staging it is somebody's explicit decision
    rather than a cache's.
    """

    budget_bytes: int
    max_stage_bytes: int
    #: Required, never defaulted. See :class:`StageEvictionPolicy`.
    eviction: StageEvictionPolicy
    #: Staging attempts that may end without an answer before the stage is
    #: blocked. Bounds a permanently failing fetch without treating the first
    #: timeout as a verdict about the asset.
    max_attempts: int = 3

    def __post_init__(self) -> None:
        """Reject bounds that would silently disable or unbound the cache."""

        if not isinstance(self.eviction, StageEvictionPolicy):
            raise ValueError("a cell stage policy requires an explicit eviction policy")
        for name in ("budget_bytes", "max_stage_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"cell stage {name} must be a non-negative integer")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("cell stage max_attempts must be a positive integer")
        if self.max_stage_bytes > self.budget_bytes:
            raise ValueError("cell stage max_stage_bytes must not exceed the per-cell budget")

    @property
    def enabled(self) -> bool:
        """Return whether this policy permits staging anything at all.

        A zero budget is the off switch, and it is off by construction rather
        than by a second flag that could disagree with it.
        """

        return self.budget_bytes > 0 and self.max_stage_bytes > 0


__all__ = ("CellStagePolicy", "StageEvictionPolicy")
