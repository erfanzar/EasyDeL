# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Versioned pricing samples used for admission and budget reservation.

There is deliberately **no fleet-wide fallback price constructor here.** One used
to exist -- ``configured_price``, which built a ``PriceSample`` scoped to nothing
but provider/accelerator/class -- and nothing ever called it. It is removed
rather than wired, for two reasons that are both about what a price *means*.

First, this catalog is not on the costing path and cannot be put on it by
declaring a fallback. A quote here is one hourly rate for one accelerator
dimension; ``CellRecord.hourly_cost`` sums a *vector* -- cpu, memory, gpu, tpu,
and any custom dimension -- against ``unit_prices``. Turning a single quote into
that vector means supplying the dimensions the quote does not mention, and the
only available value for them is zero, which understates the bill silently. That
is the same partial-pricing mistake ``NodeTypeBase._warn_about_unpriced_resources``
already warns about, and it would be introduced by the thing meant to fix it.

Second, a fallback is not an observation. An operator who declares a price
declares it on the pool, where the accelerator type, region and zone that make it
true are also declared; that mechanism exists, is durable, and is what
``catalog_from_pools`` reads. A number attached to nothing narrower than a
provider is a guess wearing a price's clothes, and this package refuses to write
"I could not find out" down as a fact.

So a cell whose pool declares no ``unit_prices`` stays unpriced. It contributes
zero, and it is *reported* as unpriced by ``CellBillingReport.unpriced_cells``,
by ``eyvan doctor``, and by the spend-reconciled proof verdict, which requires
zero unpriced cells before it will call a ledger reconciled. That is the correct
behaviour and the only honest one available without a price source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation


def _money(value: Decimal | int | float | str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("price must be a finite decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("price must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PriceKey:
    """Lookup dimensions for one accelerator price."""

    provider: str
    accelerator_type: str
    capacity_class: str
    project: str | None = None
    region: str | None = None
    zone: str | None = None


@dataclass(frozen=True, slots=True)
class PriceSample:
    """Observed or configured hourly price for one allocation unit."""

    key: PriceKey
    hourly_unit_usd: Decimal
    source: str
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "hourly_unit_usd", _money(self.hourly_unit_usd))
        if self.observed_at.tzinfo is None:
            raise ValueError("price samples require a timezone-aware observed_at")


@dataclass(frozen=True, slots=True)
class CostProjection:
    """Admission-time cost and accelerator-hour reservation."""

    hourly_cost_usd: Decimal
    reserved_cost_usd: Decimal
    accelerator_hours: Decimal
    price_observed_at: datetime
    price_source: str


class PricingCatalog:
    """In-memory view of persisted samples with deterministic specificity."""

    def __init__(self, samples: tuple[PriceSample, ...] = ()):
        self._samples: dict[PriceKey, PriceSample] = {}
        for sample in samples:
            self.put(sample)

    def put(self, sample: PriceSample) -> None:
        current = self._samples.get(sample.key)
        if current is None or sample.observed_at >= current.observed_at:
            self._samples[sample.key] = sample

    def quote(self, key: PriceKey) -> PriceSample:
        candidates = [
            sample
            for candidate_key, sample in self._samples.items()
            if candidate_key.provider == key.provider
            and candidate_key.accelerator_type == key.accelerator_type
            and candidate_key.capacity_class == key.capacity_class
            and (candidate_key.project is None or candidate_key.project == key.project)
            and (candidate_key.region is None or candidate_key.region == key.region)
            and (candidate_key.zone is None or candidate_key.zone == key.zone)
        ]
        if not candidates:
            raise LookupError(
                f"no price for {key.provider}/{key.accelerator_type}/{key.capacity_class} in {key.project or '*'}"
            )

        def rank(sample: PriceSample) -> tuple[int, datetime, str, str, str]:
            dimensions = (sample.key.project, sample.key.region, sample.key.zone)
            return (
                sum(value is not None for value in dimensions),
                sample.observed_at,
                # Equal-rank quotes resolve by scope tuple, never insertion order.
                sample.key.project or "",
                sample.key.region or "",
                sample.key.zone or "",
            )

        return max(candidates, key=rank)

    def project(
        self,
        key: PriceKey,
        *,
        accelerator_units: int,
        duration_seconds: int,
    ) -> CostProjection:
        if accelerator_units <= 0 or duration_seconds <= 0:
            raise ValueError("cost projection requires positive accelerator units and duration")
        sample = self.quote(key)
        hours = Decimal(accelerator_units) * Decimal(duration_seconds) / Decimal(3600)
        return CostProjection(
            hourly_cost_usd=sample.hourly_unit_usd * accelerator_units,
            reserved_cost_usd=sample.hourly_unit_usd * hours,
            accelerator_hours=hours,
            price_observed_at=sample.observed_at,
            price_source=sample.source,
        )


__all__ = (
    "CostProjection",
    "PriceKey",
    "PriceSample",
    "PricingCatalog",
)
