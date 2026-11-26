# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The removed global fallback price, and the unpriced fleet it would have hidden.

``eyvan/domain/pricing.py`` carried ``configured_price()``: a constructor for a
``PriceSample`` scoped to nothing narrower than provider/accelerator/class, meant
as a fleet-wide fallback. Nothing in the package ever called it. Meanwhile a
terminated 46-cell soak reported 42 cells unpriced and $3.68 accrued -- a fleet
whose cost was mostly not measured, reporting a number that looked like a total.

The two facts are not the same defect and wiring the first does not fix the
second. Two reasons, both about what a price *means*, and both asserted below:

**The catalog is not on the costing path.** A quote is one hourly rate for one
accelerator dimension. ``CellRecord.hourly_cost`` sums a whole vector -- cpu,
memory_gib, gpu, tpu, and any custom dimension -- against ``unit_prices``.
Turning the first into the second means supplying dimensions the quote does not
mention, and the only available value is zero, which understates the bill
silently. That is the identical partial-pricing mistake
``NodeTypeBase._warn_about_unpriced_resources`` already warns about.

**A fallback is not an observation.** An operator who declares a price declares
it on a pool, beside the accelerator type, region and zone that make it true;
that mechanism is durable and is what ``catalog_from_pools`` reads. A number
attached to nothing narrower than a provider is a guess wearing a price's
clothes.

So the symbol is deleted, and the behaviour it would have papered over is pinned
instead: an undeclared price stays undeclared, the cell contributes zero, the
count of unpriced cells is reported beside that zero, and the spend-reconciled
verdict still refuses a fleet with any of them.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.domain import pricing
from eyvan.records import (
    CapacityClass,
    CellRecord,
    CellState,
    Principal,
    ResourceLimits,
    Role,
    RoleGrant,
    utcnow,
)
from eyvan.sdk.commands.soak_monitor import read_spend_reconciled
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

# ------------------------------------------------------------- the symbol is gone


def test_the_global_fallback_price_constructor_is_gone() -> None:
    """Removed from the module and from what the module says it exports."""

    assert not hasattr(pricing, "configured_price")
    assert "configured_price" not in pricing.__all__


def test_the_pricing_export_list_names_exactly_what_the_module_offers() -> None:
    """An export list that drifts is how a deleted symbol comes back by accident."""

    assert pricing.__all__ == ("CostProjection", "PriceKey", "PriceSample", "PricingCatalog")
    for name in pricing.__all__:
        assert hasattr(pricing, name), f"{name} is exported but not defined"


def test_nothing_in_the_catalog_answers_for_a_shape_no_record_priced() -> None:
    """With the fallback gone there is no path from "no declaration" to a number.

    The catalog raises rather than returning a rate, which is what keeps an
    undeclared price from entering admission as if someone had stated it.
    """

    with pytest.raises(LookupError, match="no price"):
        pricing.PricingCatalog().quote(pricing.PriceKey("gcp", "v5p-16", "spot"))


# --------------------------------------------------- what an unpriced fleet reports


def _admin() -> Principal:
    return Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _cell(cell_id: str, *, prices: dict[str, float]) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=4),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=32.0, tpu=4),
        unit_prices=prices,
        capacity_class=CapacityClass.ON_DEMAND,
        pool_id="pool-1",
    )


@pytest.mark.asyncio
async def test_an_entirely_unpriced_fleet_reports_zero_beside_the_count_that_explains_it() -> None:
    """The soak's shape: no price was declared, so no cost was measured.

    The zero is reported *with* the number of cells that produced it. Without
    that count the same document reads as a fleet that genuinely costs nothing,
    which is how $3.68 came to stand for 46 cells.
    """

    service = await _service()
    async with service.repository.transaction() as transaction:
        for index in range(3):
            await transaction.put_cell(_cell(f"cell-{index}", prices={}))
    start = utcnow()
    await service.accrue_cell_billing(now=start)
    await service.accrue_cell_billing(now=start + timedelta(hours=1))

    report = await service.cell_billing_report(_admin())

    assert report.cells == 3
    assert report.billed_seconds == pytest.approx(10800.0), "the clock ran; the price is what is missing"
    assert report.accrued_cost == 0.0
    assert report.idle_cost == 0.0
    assert report.unpriced_cells == 3, "a zero total must arrive with the count that makes it readable"


@pytest.mark.asyncio
async def test_a_declared_price_is_the_only_thing_that_produces_a_cost() -> None:
    """One priced cell among unpriced ones contributes exactly its own cost.

    No fallback fills in the others -- their share of the bill is absent, and the
    count says so.
    """

    service = await _service()
    async with service.repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-priced", prices={"tpu": 2.0}))
        await transaction.put_cell(_cell("cell-unpriced", prices={}))
    start = utcnow()
    await service.accrue_cell_billing(now=start)
    await service.accrue_cell_billing(now=start + timedelta(hours=1))

    report = await service.cell_billing_report(_admin())

    assert report.accrued_cost == pytest.approx(8.0), "4 TPU at 2.0/hour for one hour, and nothing else"
    assert report.unpriced_cells == 1


# ------------------------------------------ a fallback would not have been evidence


class _StubClient:
    def __init__(self, documents: dict[str, object]) -> None:
        self.documents = documents

    def request(self, method: str, path: str, **_kwargs: object) -> object:
        del method
        return self.documents[path]


def test_spend_reconciled_still_refuses_a_fleet_with_any_unpriced_cell() -> None:
    """The invariant the choice turns on: a declared fallback is not a measurement.

    Had the fallback been wired, the honest accounting would still have had to
    keep fallback-priced cells out of "observed", because a receipt that calls a
    fleet reconciled is asserting somebody measured it. With no fallback at all
    the rule is simply unchanged: one unpriced cell and the ledger is not
    reconciled.
    """

    documents = {
        "/api/v1/system/invariants": {
            "checked": [f"I{number}" for number in range(1, 10)],
            "violations": [],
            "violations_truncated": False,
        },
        "/api/v1/capacity/billing": {"cells": 46, "unpriced_cells": 42},
    }

    assert read_spend_reconciled(_StubClient(documents)) is False

    documents["/api/v1/capacity/billing"] = {"cells": 46, "unpriced_cells": 0}
    assert read_spend_reconciled(_StubClient(documents)) is True
