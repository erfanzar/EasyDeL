# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observable contracts for dataclass methods omitted by the mutmut generator."""

from datetime import datetime, timedelta
from itertools import product
from math import sqrt

import pytest

from eyvan._internal.compat import UTC
from eyvan.domain.availability import CapacityBelief, CapacityPoolKey, ProvisionOutcome

NOW = datetime(2026, 9, 12, tzinfo=UTC)
KEY = CapacityPoolKey("gcp", "test-project", "test-zone", "test-accelerator", "spot")


def test_beta_scores_and_interruption_exposure_are_not_interchangeable() -> None:
    belief = CapacityBelief(KEY, success_evidence=2, failure_evidence=6, interruptions=3, successful_hours=4)
    assert belief.success_probability == 0.25
    assert belief.uncertainty == pytest.approx(sqrt(1 / 48))
    assert belief.interruption_rate_per_hour == 0.75
    assert CapacityBelief(KEY, interruptions=3, successful_hours=0.25).interruption_rate_per_hour == 3


def test_interruption_decays_evidence_without_clearing_failure_cooldown() -> None:
    original = CapacityBelief(
        KEY,
        success_evidence=4,
        failure_evidence=6,
        interruptions=4,
        successful_hours=4,
        consecutive_failures=3,
        cooldown_until=NOW + timedelta(seconds=90),
        provision_latencies_s=(8, 12),
    )
    observed = original.observe(
        ProvisionOutcome.INTERRUPTED,
        now=NOW,
        latency_s=99,
        successful_hours=2,
        evidence_decay=0.5,
    )
    assert observed.success_evidence == 2 and observed.failure_evidence == 3
    assert observed.interruptions == 3 and observed.successful_hours == 4
    assert observed.consecutive_failures == 3
    assert observed.cooldown_until == original.cooldown_until
    assert observed.provision_latencies_s == (8, 12)
    assert observed.last_observed_at == NOW and observed.key == KEY
    assert original.interruptions == original.successful_hours == 4
    assert original.last_observed_at is None


def test_success_retains_only_latest_128_latencies_and_resets_failure_streak() -> None:
    original = CapacityBelief(
        KEY,
        success_evidence=4,
        failure_evidence=6,
        consecutive_failures=8,
        cooldown_until=NOW + timedelta(hours=1),
        provision_latencies_s=tuple(range(128)),
    )
    observed = original.observe(ProvisionOutcome.SUCCESS, now=NOW, latency_s=128, evidence_decay=0.5)
    assert observed.provision_latencies_s == tuple(range(1, 129))
    assert observed.provisioning_latency_p95_s == 122
    assert observed.success_evidence == 3 and observed.failure_evidence == 3
    assert observed.consecutive_failures == 0 and observed.cooldown_until is None
    assert original.provision_latencies_s == tuple(range(128))
    assert original.consecutive_failures == 8


@pytest.mark.parametrize(
    ("outcome", "base"),
    [
        (ProvisionOutcome.STOCKOUT, 60),
        (ProvisionOutcome.HOST_FAULT, 300),
        (ProvisionOutcome.QUOTA_ERROR, 15),
        (ProvisionOutcome.PROVIDER_ERROR, 15),
    ],
)
def test_failure_backoff_caps_and_custom_policy_remain_distinct(outcome, base) -> None:
    original = CapacityBelief(KEY, provision_latencies_s=(4,), interruptions=2, successful_hours=8)
    first = original.observe(outcome, now=NOW, latency_s=999, evidence_decay=1, max_backoff_s=400)
    second = first.observe(outcome, now=NOW, evidence_decay=1, max_backoff_s=400)
    assert first.cooldown_until == NOW + timedelta(seconds=base)
    assert second.cooldown_until == NOW + timedelta(seconds=min(base * 2, 400))
    assert second.consecutive_failures == 2
    assert second.success_evidence == 1 and second.failure_evidence == 3
    assert second.interruptions == 2 and second.successful_hours == 8
    assert second.provision_latencies_s == (4,)
    custom = original.observe(outcome, now=NOW, backoff_base_s={outcome.value: 7}, default_backoff_s=11)
    fallback = original.observe(outcome, now=NOW, backoff_base_s={}, default_backoff_s=11)
    assert custom.cooldown_until == NOW + timedelta(seconds=7)
    assert fallback.cooldown_until == NOW + timedelta(seconds=11)


@pytest.mark.parametrize("value_ok,budget_ok,uncertainty_ok,cooldown_ok", list(product((False, True), repeat=4)))
def test_probe_requires_every_gate_including_exact_boundaries(value_ok, budget_ok, uncertainty_ok, cooldown_ok) -> None:
    belief = CapacityBelief(KEY, cooldown_until=NOW if cooldown_ok else NOW + timedelta(microseconds=1))
    decision = belief.should_probe(
        blocked_value_usd=11 if value_ok else 10,
        probe_cost_usd=10,
        probe_budget_remaining_usd=10 if budget_ok else 9,
        uncertainty_threshold=belief.uncertainty if uncertainty_ok else belief.uncertainty + 0.01,
        now=NOW,
    )
    assert decision is all((value_ok, budget_ok, uncertainty_ok, cooldown_ok))
