# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What one run used, and what it was charged.

Pure arithmetic over lease documents, and it lives here rather than in the CLI
because two callers need the same answer: the control plane, which knows which
pools declare a price, and the CLI, which renders it. A second implementation
of "what did this cost" is the kind of duplication that drifts silently and is
noticed as a disagreement between two screens.

Nothing here infers a price. Eyvan carries no price catalogue and no
spot-discount constant, so an unpriced pool is reported as *unknown* rather than
as free -- a settled cost of zero from a pool that declares no unit price is the
absence of a price, not the absence of a bill, and the difference is what a team
puts in a budget.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from decimal import Decimal

from eyvan.domain import costing

#: Resource dimensions billed per accelerator-hour, in the order they are read.
_ACCELERATOR_DIMENSIONS: tuple[str, ...] = ("tpu", "gpu")


@dataclass(frozen=True, slots=True)
class RunSpend:
    """What one workload's capacity cost, and what it used.

    Attributes:
        accelerator_hours: Settled accelerator-hours per dimension. Always
            known, because it is measured rather than priced.
        settled: What the fleet charged, summed over the workload's leases.
        priced: Whether every contributing lease came from a pool that declares
            a price. ``settled`` is meaningless when this is false -- an
            unpriced pool settles to zero, which is not the same as free.
        reserved: What is currently reserved for leases that have not settled.
        leases: How many leases contributed.
    """

    accelerator_hours: dict[str, float]
    settled: float
    priced: bool
    reserved: float
    leases: int

    @property
    def dimension(self) -> str | None:
        """Return the accelerator dimension this run used, if exactly one."""

        used = [name for name in _ACCELERATOR_DIMENSIONS if self.accelerator_hours.get(name)]
        return used[0] if len(used) == 1 else None

    @property
    def hours(self) -> float:
        """Return the accelerator-hours of this run's single dimension."""

        dimension = self.dimension
        return float(self.accelerator_hours.get(dimension, 0.0)) if dimension else 0.0


def spend_of(leases: tp.Sequence[tp.Mapping[str, object]], *, priced_cells: tp.Container[str] | None = None) -> RunSpend:
    """Total what a workload's leases used and were charged.

    Args:
        leases: Lease documents belonging to one workload.
        priced_cells: Cell ids whose pool declares a unit price. ``None`` means
            "unknown", which is reported as unpriced rather than assumed.

    Returns:
        The totals, with prices flagged as trustworthy only when every lease
        came from a cell known to be priced. A settled lease's ``reserved_cost``
        is not counted: settlement moved its cost into ``settled_cost``, so the
        reservation it stood for is gone even though the record keeps the
        estimate.
    """

    # Sum persisted decimal values before returning to JSON floats. Rounding
    # each contribution would discard valid tiny measurements, while binary
    # addition would reintroduce drift already avoided during settlement.
    zero = Decimal(0)
    hours: dict[str, Decimal] = {}
    settled = zero
    reserved = zero
    priced = bool(leases) and priced_cells is not None
    for lease in leases:
        measured = lease.get("accelerator_hours_settled")
        if isinstance(measured, dict):
            for dimension, value in measured.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    hours[str(dimension)] = hours.get(str(dimension), zero) + costing.money(value)
        # Settlement moves a lease's cost into ``settled_cost`` but leaves the
        # original ``reserved_cost`` standing on the record. That estimate is no
        # longer money held, so only leases that have not settled count as
        # reserved; ``settled_at`` is the boundary the settlement itself uses.
        has_settled = lease.get("settled_at") is not None
        for key, target in (("settled_cost", "settled"), ("reserved_cost", "reserved")):
            value = lease.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                if target == "settled":
                    settled += costing.money(value)
                elif not has_settled:
                    reserved += costing.money(value)
        cell_id = lease.get("cell_id")
        if priced_cells is not None and (not isinstance(cell_id, str) or cell_id not in priced_cells):
            priced = False
    return RunSpend(
        accelerator_hours={name: float(value) for name, value in hours.items()},
        settled=float(settled),
        priced=priced,
        reserved=float(reserved),
        leases=len(leases),
    )


__all__ = ("RunSpend", "spend_of")
