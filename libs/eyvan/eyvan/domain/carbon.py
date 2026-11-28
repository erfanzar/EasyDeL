# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shape a carbon number would have to arrive in to be worth reporting.

The contract only. No provider is implemented and none is planned here, because
the honest version of this feature is small and the dishonest version is easy:
publish a plausible-looking grams-per-kWh figure, let it influence placement,
and nobody can tell it apart from a real one.

Three properties make the difference, and all three are structural rather than
documented:

**Nullable, with a reason.** The absence of a reading is a first-class value,
not a zero and not ``None``. Zero is the most dangerous possible default -- it
says "this region is carbon-free", which is the strongest claim the type can
make, arrived at by having no data at all.

**Provenance-carrying.** A reading names where it came from and when it was
observed, and it cannot be constructed without them. A number with no source is
indistinguishable from a number someone made up.

**Freshness-gated.** Grid intensity moves hour to hour; a reading from last week
describes a grid that no longer exists. Past its window a reading becomes
unavailable rather than stale-but-used.

And the fourth, enforced by a test rather than by care: **nothing in the
placement path may import this module.** A wrong carbon number that silently
influences placement is worse than no carbon number, because the resulting
decisions look deliberate. Reporting is the only sanctioned consumer until
someone brings a source they trust.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from eyvan._internal.compat import StrEnum

#: How long a grid-intensity reading describes the grid it was read from.
#:
#: One hour because that is the resolution most grid operators publish at, not
#: because it is a safe amount of staleness -- a fast-moving grid invalidates a
#: reading sooner, and no default can know that.
DEFAULT_MAX_AGE_S = 3600.0

#: How far ahead of the control plane's clock a reading may be stamped.
#:
#: Not zero. A source that reads "now" is routinely a few hundred milliseconds
#: ahead of the moment the report is assembled, and rejecting that would make a
#: correctly configured provider fail about half the time, unpredictably. A
#: minute is far below the resolution grid data is published at, so nothing real
#: is admitted by the tolerance -- while an hour of skew, which is what actually
#: signals untrustworthy timestamps, is still refused.
FUTURE_TOLERANCE_S = 60.0


class CarbonUnavailableReason(StrEnum):
    """Why no intensity is being reported.

    A closed set so a caller can branch on it, and distinct members because they
    call for different responses: NO_PROVIDER is a deployment that never
    configured one, REGION_UNKNOWN is a provider that does not cover this place,
    STALE is a provider that has stopped answering, and PROVIDER_ERROR is one
    that answered badly. Only the last two are incidents.
    """

    NO_PROVIDER = "no_provider"
    REGION_UNKNOWN = "region_unknown"
    STALE = "stale"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True, slots=True)
class CarbonUnavailable:
    """No intensity for this region, and why.

    A value rather than an exception. Absence is the *ordinary* case here -- the
    shipped configuration has no provider -- and making the ordinary case raise
    would push every caller into a try block for a condition that is not an
    error.
    """

    reason: CarbonUnavailableReason
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.reason, CarbonUnavailableReason):
            raise ValueError("carbon unavailability requires a known reason code")


@dataclass(frozen=True, slots=True)
class CarbonIntensity:
    """One grid-intensity reading, with the provenance that makes it checkable."""

    region: str
    grams_co2e_per_kwh: float
    #: Who said so. Free-form because the set of grid-data providers is not
    #: Eyvan's to enumerate, but required, because a number with no attribution
    #: cannot be audited or disputed.
    source: str
    observed_at: datetime

    def __post_init__(self) -> None:
        if not self.region:
            raise ValueError("a carbon reading requires a region")
        if not self.source:
            # The whole point of the type. A reading that could be constructed
            # without a source would let an unattributed number look identical
            # to a measured one.
            raise ValueError("a carbon reading requires a source; an unattributed number is not a reading")
        if self.grams_co2e_per_kwh < 0:
            raise ValueError("carbon intensity cannot be negative")
        if not isfinite(self.grams_co2e_per_kwh):
            raise ValueError("carbon intensity must be finite")
        if self.observed_at.tzinfo is None:
            raise ValueError("a carbon reading requires a timezone-aware observation time")

    def age_seconds(self, now: datetime) -> float:
        return (now - self.observed_at).total_seconds()


#: What a source returns: a reading, or the reason there is none.
CarbonReading: tp.TypeAlias = "CarbonIntensity | CarbonUnavailable"


class CarbonSource(tp.Protocol):
    """Read the current grid intensity for one region."""

    def read(self, region: str) -> CarbonReading: ...


class NoCarbonSource:
    """The shipped default: nothing configured, and it says so.

    Not a source that returns zero, and not one that estimates. An operator who
    has configured nothing should see ``no_provider`` in every report, which is
    a true statement about their deployment rather than a number about their
    grid.
    """

    def read(self, region: str) -> CarbonReading:
        del region
        return CarbonUnavailable(
            CarbonUnavailableReason.NO_PROVIDER,
            "no carbon intensity provider is configured for this control plane",
        )


def fresh_reading(
    reading: CarbonReading,
    *,
    now: datetime,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> CarbonReading:
    """Return the reading, or ``CarbonUnavailable`` when it is too old to mean anything.

    Applied by the reporting path rather than by each source, so a source that
    forgets the gate cannot publish week-old numbers -- and so the window is one
    decision an operator can find, rather than one per provider.

    A reading from the future is refused too. Clock skew large enough to produce
    one is itself the signal that the timestamps cannot be trusted.
    """

    if isinstance(reading, CarbonUnavailable):
        return reading
    if max_age_s <= 0:
        raise ValueError("carbon freshness window must be positive")
    if not isfinite(max_age_s):
        raise ValueError("carbon freshness window must be finite")
    age = reading.age_seconds(now)
    if age < -FUTURE_TOLERANCE_S:
        return CarbonUnavailable(
            CarbonUnavailableReason.STALE,
            f"carbon reading for {reading.region!r} is timestamped {-age:.0f}s in the future",
        )
    if age > max_age_s:
        return CarbonUnavailable(
            CarbonUnavailableReason.STALE,
            f"carbon reading for {reading.region!r} is {timedelta(seconds=int(age))} old "
            f"(window is {timedelta(seconds=int(max_age_s))})",
        )
    return reading


__all__ = (
    "DEFAULT_MAX_AGE_S",
    "FUTURE_TOLERANCE_S",
    "CarbonIntensity",
    "CarbonReading",
    "CarbonSource",
    "CarbonUnavailable",
    "CarbonUnavailableReason",
    "NoCarbonSource",
    "fresh_reading",
)
