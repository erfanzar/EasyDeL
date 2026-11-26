# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Empirical capacity beliefs and bounded probing policy."""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from sys import float_info

from eyvan._internal.compat import UTC, StrEnum


def p95_latency(latencies_s: tp.Iterable[float]) -> float | None:
    """Return the nearest-rank p95 of observed latencies, or ``None`` when empty."""

    ordered = sorted(latencies_s)
    if not ordered:
        return None
    index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


class ProvisionOutcome(StrEnum):
    """Provider observations that update one capacity belief."""

    SUCCESS = "success"
    STOCKOUT = "stockout"
    QUOTA_ERROR = "quota_error"
    PROVIDER_ERROR = "provider_error"
    INTERRUPTED = "interrupted"
    #: The pool delivered a machine and the machine was broken -- an accelerator
    #: that never initialized, not a request the provider refused.
    #:
    #: Its own outcome rather than a PROVIDER_ERROR because the two say opposite
    #: things about what to do next. A provider error is usually about the
    #: request and clears in seconds; a bad host says something about the
    #: physical pool that the next allocation from the same rack or image is
    #: likely to repeat, and each repetition costs a full provision *and* a
    #: destroy. So it is evidence against the zone, and it backs off hardest.
    HOST_FAULT = "host_fault"


#: First cooldown, in seconds, after one failing observation of each kind. The
#: window doubles per consecutive failure and is capped by ``max_backoff_s``.
#: Anything not named here backs off from 15s, which is the historical default
#: for a provider error.
#: Keyed by outcome *value* so a declared policy and this default are the same
#: shape: an operator writes `stockout: 120` in YAML, and nothing has to
#: translate between a string and an enum on the way in.
_FAILURE_BACKOFF_BASE_S: dict[str, int] = {
    ProvisionOutcome.STOCKOUT.value: 60,
    ProvisionOutcome.HOST_FAULT.value: 300,
}


@dataclass(frozen=True, slots=True)
class CapacityPoolKey:
    """Provider-native capacity pool whose availability is correlated."""

    provider: str
    project: str
    zone: str
    accelerator_type: str
    capacity_class: str


@dataclass(frozen=True, slots=True)
class CapacityBelief:
    """Decayed evidence used for placement scoring and probe throttling."""

    key: CapacityPoolKey
    success_evidence: float = 1.0
    failure_evidence: float = 1.0
    provision_latencies_s: tuple[float, ...] = ()
    interruptions: float = 0.0
    successful_hours: float = 0.0
    consecutive_failures: int = 0
    cooldown_until: datetime | None = None
    last_observed_at: datetime | None = None

    @property
    def success_probability(self) -> float:
        return self.success_evidence / (self.success_evidence + self.failure_evidence)

    @property
    def uncertainty(self) -> float:
        alpha = self.success_evidence
        beta = self.failure_evidence
        total = alpha + beta
        # Factoring the beta standard deviation avoids underflow in the product
        # and squared denominator when both evidence values have decayed away.
        return math.sqrt(alpha / total) * math.sqrt(beta / total) / math.sqrt(total + 1)

    @property
    def interruption_rate_per_hour(self) -> float:
        return self.interruptions / max(self.successful_hours, 1.0)

    @property
    def provisioning_latency_p95_s(self) -> float | None:
        return p95_latency(self.provision_latencies_s)

    def observe(
        self,
        outcome: ProvisionOutcome,
        *,
        now: datetime | None = None,
        latency_s: float | None = None,
        successful_hours: float = 0.0,
        evidence_decay: float = 0.98,
        max_backoff_s: int = 3600,
        backoff_base_s: tp.Mapping[str, int] | None = None,
        default_backoff_s: int = 15,
    ) -> CapacityBelief:
        """Return a new belief after one provider observation.

        The three tuning arguments used to be module constants. They are how a
        pool reacts to being told no, and the right answer differs by zone and
        by purchase class -- a contended spot region is worth asking again in a
        minute, an account being rate-limited is not. Their defaults reproduce
        the constants exactly, so a caller that passes nothing is unchanged.
        """

        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("capacity observations require a timezone-aware timestamp")
        if not 0 < evidence_decay <= 1:
            raise ValueError("evidence_decay must be in (0, 1]")
        if latency_s is not None and not math.isfinite(latency_s):
            # One NaN sample would sit in the rolling window forever and turn
            # every later p95 into NaN.
            raise ValueError("latency_s must be a finite non-negative number")
        if latency_s is not None and latency_s < 0:
            raise ValueError("latency_s must be non-negative")
        # Persisted evidence must remain strictly positive even for a valid,
        # very small decay or a long run of one outcome. The smallest normal
        # float is a negligible prior floor, not a new success/failure sample.
        success = max(float_info.min, self.success_evidence * evidence_decay)
        failure = max(float_info.min, self.failure_evidence * evidence_decay)
        interruptions = self.interruptions * evidence_decay
        hours = self.successful_hours * evidence_decay + successful_hours
        failures = self.consecutive_failures
        cooldown = self.cooldown_until
        latencies = self.provision_latencies_s

        if outcome is ProvisionOutcome.SUCCESS:
            success += 1
            failures = 0
            cooldown = None
            if latency_s is not None:
                latencies = (*latencies, latency_s)[-128:]
        elif outcome is ProvisionOutcome.INTERRUPTED:
            interruptions += 1
        else:
            failure += 1
            failures += 1
            # A stockout is answered by asking again later; a host fault is
            # answered by provisioning a *replacement*, which costs a full
            # allocation and a destroy before anything can be learned. Paying
            # that every fifteen seconds against a zone handing out broken
            # machines is the loop this backoff exists to stop.
            curve = _FAILURE_BACKOFF_BASE_S if backoff_base_s is None else backoff_base_s
            base = curve.get(outcome.value, default_backoff_s)
            cooldown = now + timedelta(seconds=min(max_backoff_s, base * (2 ** (failures - 1))))

        return replace(
            self,
            success_evidence=success,
            failure_evidence=failure,
            provision_latencies_s=latencies,
            interruptions=interruptions,
            successful_hours=hours,
            consecutive_failures=failures,
            cooldown_until=cooldown,
            last_observed_at=now,
        )

    def should_probe(
        self,
        *,
        blocked_value_usd: float,
        probe_cost_usd: float,
        probe_budget_remaining_usd: float,
        now: datetime | None = None,
        uncertainty_threshold: float = 0.08,
    ) -> bool:
        """Decide whether bounded exploration is economically justified."""

        now = now or datetime.now(UTC)
        return (
            blocked_value_usd > probe_cost_usd
            and probe_cost_usd <= probe_budget_remaining_usd
            and self.uncertainty >= uncertainty_threshold
            and (self.cooldown_until is None or now >= self.cooldown_until)
        )
