# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rendering what one run cost, and what it would have cost on demand.

The sentence a spot-first scheduler exists to say, and none of Eyvan's
competitors prints it.

The arithmetic moved to :mod:`eyvan.domain.spend` when the control plane grew a
cost route of its own. Two implementations of "what did this run cost" drift,
and the disagreement surfaces as one number in a terminal and a different one in
a browser. Re-exported here because this is the name every caller already
imports.
"""

from __future__ import annotations

from eyvan.domain.spend import RunSpend, spend_of


def render(
    workload_id: str,
    state: str,
    spend: RunSpend,
    *,
    shape: str | None,
    capacity_class: str | None,
    on_demand_hourly: float | None,
    on_demand_source: str,
) -> tuple[str, ...]:
    """Render one run's spend, saying *unknown* wherever it is unknown."""

    lines = [f"  workload    {workload_id} ({state})"]
    if shape:
        capacity = f"{shape}" if not capacity_class else f"{shape}, {capacity_class}"
        lines.append(f"  capacity    {capacity}")
    if not spend.leases:
        lines.append("  used        nothing; this workload never held a lease")
        return tuple(lines)

    dimension = spend.dimension
    if dimension:
        lines.append(f"  used        {spend.hours:.1f} {dimension.upper()}-hours across {spend.leases} lease(s)")
    elif spend.accelerator_hours:
        measured = ", ".join(
            f"{value:.1f} {name.upper()}-hours" for name, value in sorted(spend.accelerator_hours.items())
        )
        lines.append(f"  used        {measured} across {spend.leases} lease(s)")
    else:
        lines.append(f"  used        {spend.leases} lease(s); no accelerator hours were settled")

    if spend.priced:
        lines.append(f"  cost        ${spend.settled:.2f}")
    else:
        # An unpriced pool settles to zero. Printing "$0.00" would present the
        # absence of a price as the absence of a bill.
        lines.append("  cost        unknown; no pool this run used declares a unit price")
    if spend.reserved:
        lines.append(f"  reserved    ${spend.reserved:.2f} still held by unsettled leases")

    if on_demand_hourly is None:
        lines.append(f"  on demand   unknown; {on_demand_source}")
        return tuple(lines)
    alternative = on_demand_hourly * spend.hours
    lines.append(f"  on demand   ${alternative:.2f} ({on_demand_source})")
    if spend.priced:
        saved = alternative - spend.settled
        share = f" ({saved / alternative:.0%} less)" if alternative > 0 else ""
        lines.append(f"  saved       ${saved:.2f}{share}")
    return tuple(lines)


__all__ = ("RunSpend", "render", "spend_of")
