# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How hard a pool backs off, said out loud rather than compiled in.

Every one of these numbers was a module constant, and constants are somebody
else's answer. How long to wait after a zone refuses is a property of that zone
and that purchase class: a contended spot TPU region is worth asking again in a
minute, an account being rate-limited is not. An operator watching the wrong
behaviour had no way to say so and no lever except deleting the pool.

The tests that matter here are the ones about *not* changing anything: a fleet
that declares nothing must back off exactly as it always did, or this is a
silent behaviour change dressed as a feature.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.domain.availability import CapacityBelief, CapacityPoolKey, ProvisionOutcome
from eyvan.records import AvailabilityPolicy, CapacityPoolRecord

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def _belief() -> CapacityBelief:
    return CapacityBelief(
        key=CapacityPoolKey(
            provider="gcp",
            project="a-project",
            zone="us-east5-b",
            accelerator_type="v6e-8",
            capacity_class="spot",
        )
    )


def _cooldown(belief: CapacityBelief) -> timedelta:
    assert belief.cooldown_until is not None
    return belief.cooldown_until - NOW


# --------------------------------------------------------------------------- nothing declared, nothing changed


def test_a_fleet_that_declares_nothing_backs_off_as_it_always_did() -> None:
    """The floor under this change. Defaults reproduce the old constants."""

    policy = AvailabilityPolicy()

    assert policy.evidence_decay == 0.98
    assert policy.max_backoff_s == 3600
    assert policy.backoff_base_s == {"stockout": 60, "host_fault": 300}
    assert policy.default_backoff_s == 15


@pytest.mark.parametrize(
    ("outcome", "seconds"),
    [
        (ProvisionOutcome.STOCKOUT, 60),
        (ProvisionOutcome.HOST_FAULT, 300),
        (ProvisionOutcome.PROVIDER_ERROR, 15),
    ],
)
def test_the_declared_default_and_the_old_constant_agree(outcome: ProvisionOutcome, seconds: int) -> None:
    """Passing the defaults explicitly must be indistinguishable from passing nothing."""

    policy = AvailabilityPolicy()

    implicit = _belief().observe(outcome, now=NOW)
    explicit = _belief().observe(
        outcome,
        now=NOW,
        evidence_decay=policy.evidence_decay,
        max_backoff_s=policy.max_backoff_s,
        backoff_base_s=policy.backoff_base_s,
        default_backoff_s=policy.default_backoff_s,
    )

    assert _cooldown(implicit) == _cooldown(explicit) == timedelta(seconds=seconds)


# --------------------------------------------------------------------------- what declaring one changes


def test_a_declared_curve_replaces_the_compiled_one() -> None:
    """The point of the feature: an operator's number is the number used."""

    belief = _belief().observe(
        ProvisionOutcome.STOCKOUT,
        now=NOW,
        backoff_base_s={"stockout": 5},
        default_backoff_s=15,
    )

    assert _cooldown(belief) == timedelta(seconds=5)


def test_an_outcome_the_curve_does_not_name_uses_the_declared_default() -> None:
    """A partial curve is a curve, not an invitation to fall back to ours."""

    belief = _belief().observe(
        ProvisionOutcome.PROVIDER_ERROR,
        now=NOW,
        backoff_base_s={"stockout": 5},
        default_backoff_s=90,
    )

    assert _cooldown(belief) == timedelta(seconds=90)


def test_the_ceiling_still_bounds_a_declared_curve() -> None:
    """Doubling is what makes a ceiling necessary, whoever chose the base."""

    belief = _belief()
    for _ in range(10):
        belief = belief.observe(ProvisionOutcome.STOCKOUT, now=NOW, max_backoff_s=120, backoff_base_s={"stockout": 60})

    assert _cooldown(belief) == timedelta(seconds=120)


def test_a_base_above_the_ceiling_is_refused_rather_than_discarded() -> None:
    """Otherwise the number an operator wrote means nothing at all."""

    with pytest.raises(ValueError, match="between 1 and max_backoff_s"):
        AvailabilityPolicy(max_backoff_s=30, backoff_base_s={"stockout": 60})


def test_forgetting_faster_is_declarable() -> None:
    """A zone that was empty this morning is not evidence about this afternoon."""

    slow = _belief().observe(ProvisionOutcome.SUCCESS, now=NOW, evidence_decay=0.99)
    fast = _belief().observe(ProvisionOutcome.SUCCESS, now=NOW, evidence_decay=0.5)

    assert fast.failure_evidence < slow.failure_evidence


# --------------------------------------------------------------------------- how it reaches a cell


def _fleet(document: dict) -> EyvanFleetConfig:
    return EyvanFleetConfig.model_validate(document)


def _pools(fleet: EyvanFleetConfig) -> dict[str, CapacityPoolRecord]:
    """Compile the fleet's pool payloads into records, keyed by node type.

    The endpoint is filled in here for the same reason the config's own
    validation fills one in: fleet YAML may defer agent routing to the platform,
    and the durable record still needs an address.
    """

    records = {}
    for payload in fleet.pool_payloads():
        resolved = dict(payload)
        resolved["agent_endpoint"] = resolved["agent_endpoint"] or "http://agent.invalid"
        records[resolved["node_type"]] = CapacityPoolRecord.model_validate(resolved)
    return records


def _document(**overrides) -> dict:
    base = {
        "name": "research",
        "provider": {"type": "gcp", "project": "a-project", "region": "us-east5", "zone": "us-east5-b"},
        "pools": {
            "v6e-8": {
                "kind": "tpu",
                "accelerator_type": "v6e-8",
                "max_workers": 2,
                "provider_options": {"bootstrap_egress": "public-internet"},
            },
            "cpu-16": {
                "kind": "cpu",
                "machine_type": "n2-standard-16",
                "resources": {"cpu": 16, "memory_gib": 64},
                "max_workers": 2,
                "provider_options": {"bootstrap_egress": "public-internet"},
            },
        },
    }
    base.update(overrides)
    return base


def test_a_fleet_level_curve_reaches_every_pool() -> None:
    fleet = _fleet(_document(availability_policy={"backoff_base_s": {"stockout": 30}, "max_backoff_s": 600}))

    pools = _pools(fleet)

    assert pools["v6e-8"].availability_policy.backoff_base_s == {"stockout": 30}
    assert pools["cpu-16"].availability_policy.max_backoff_s == 600


def test_a_pool_overrides_the_fleet_whole_rather_than_field_by_field() -> None:
    """A backoff curve is one shape, and half of somebody else's is not one."""

    fleet = _fleet(
        _document(
            availability_policy={"backoff_base_s": {"stockout": 30}, "max_backoff_s": 600},
            pools={
                "v6e-8": {
                    "kind": "tpu",
                    "accelerator_type": "v6e-8",
                    "max_workers": 2,
                    "provider_options": {"bootstrap_egress": "public-internet"},
                    "availability_policy": {"backoff_base_s": {"stockout": 5}},
                },
                "cpu-16": {
                    "kind": "cpu",
                    "machine_type": "n2-standard-16",
                    "resources": {"cpu": 16, "memory_gib": 64},
                    "max_workers": 2,
                    "provider_options": {"bootstrap_egress": "public-internet"},
                },
            },
        )
    )

    pools = _pools(fleet)

    assert pools["v6e-8"].availability_policy.backoff_base_s == {"stockout": 5}
    # The fleet's ceiling is not inherited into the pool's own curve: it declared
    # a policy, and that policy's ceiling is its default.
    assert pools["v6e-8"].availability_policy.max_backoff_s == 3600
    assert pools["cpu-16"].availability_policy.max_backoff_s == 600


def test_a_cell_carries_the_curve_it_was_created_under() -> None:
    """So an observation needs nothing but the cell, and a retune is not retroactive."""

    fleet = _fleet(_document(availability_policy={"backoff_base_s": {"stockout": 30}}))
    pool = _pools(fleet)["v6e-8"]

    cell = pool.make_cell("cell-a", worker_count=1)

    assert cell.availability_policy.backoff_base_s == {"stockout": 30}
