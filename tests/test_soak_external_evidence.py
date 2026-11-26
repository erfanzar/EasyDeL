# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The closed P8 evidence sample, and the slot the control plane keeps it in.

The route exists because a control plane cannot measure itself honestly, so the
only interesting properties are the ones that stop it from *pretending* it can:
it stores exactly what it was given or nothing at all, it hands a sample back
only to the run that sample names, and it forgets a sample the moment it is old
enough that the monitor may already be dead.
"""

from __future__ import annotations

import json
import typing as tp

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app
from eyvan.soak_evidence import (
    EXTERNAL_EVIDENCE_FIELDS,
    EXTERNAL_EVIDENCE_SCHEMA,
    MAX_EVIDENCE_BODY_BYTES,
    ExternalEvidenceSample,
    SoakEvidenceStore,
)

PROOF_ID = "sha256:" + "d" * 64
OTHER_PROOF_ID = "sha256:" + "e" * 64
CONFIG_SHA256 = "sha256:" + "c" * 64

ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
TENANT = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "operator", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "research"}

EVIDENCE_PATH = "/api/v1/system/soak-evidence"
SOAK_PATH = "/api/v1/system/invariants/soak"


class _StepClock:
    """A clock the test advances by hand."""

    def __init__(self) -> None:
        self.monotonic_s = 1_000.0
        self.wall_s = 2_000.0

    def monotonic(self) -> float:
        return self.monotonic_s

    def wall(self) -> float:
        return self.wall_s


def _sample(**overrides: tp.Any) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": EXTERNAL_EVIDENCE_SCHEMA,
        "proof_id": PROOF_ID,
        "rss_bytes": 543_210,
        "fd_count": 91,
        "database_bytes": 1_048_576,
        "spend_reconciled": True,
        "provider_inventory_count": 3,
        "cleanup_complete": False,
    }
    document.update(overrides)
    return document


def _plane(tmp_path, store: SoakEvidenceStore | None = None) -> TestClient:
    return TestClient(
        create_app(
            settings=EyvanSettings(
                dev_auth_enabled=True,
                artifact_token_key="k" * 32,
                artifact_store_uri=str(tmp_path / "artifacts"),
            ),
            soak_evidence=store,
        )
    )


def _manifest(proof_id: str = PROOF_ID) -> dict[str, object]:
    return {
        "proof_id": proof_id,
        "config_sha256": CONFIG_SHA256,
        "participants": [{"workload_id": "wl-a", "lease_id": "lease-a", "cell_id": "cell-a"}],
    }


def _observed_evidence(plane: TestClient, proof_id: str = PROOF_ID) -> object:
    response = plane.post(SOAK_PATH, json=_manifest(proof_id), headers=ADMIN)
    assert response.status_code == 200, response.text
    return response.json()["snapshot"]["p8_evidence"]


# --------------------------------------------------------------------------- storing


def test_a_fresh_sample_is_republished_byte_for_byte_to_the_run_it_names(tmp_path) -> None:
    """The control plane is a courier here, not a source.

    Anything it added, rounded, or filled in would be evidence nobody measured,
    so the document the evaluator reads has to be the document the monitor sent.
    """

    plane = _plane(tmp_path)
    posted = _sample()

    accepted = plane.post(EVIDENCE_PATH, json=posted, headers=ADMIN)

    assert accepted.status_code == 200, accepted.text
    assert accepted.json() == {"accepted": True, "proof_id": PROOF_ID}
    assert _observed_evidence(plane) == posted


def test_only_the_most_recent_sample_is_kept(tmp_path) -> None:
    """A history would let an evaluator read a measurement that has been superseded."""

    plane = _plane(tmp_path)
    plane.post(EVIDENCE_PATH, json=_sample(rss_bytes=1), headers=ADMIN)
    plane.post(EVIDENCE_PATH, json=_sample(rss_bytes=2), headers=ADMIN)

    assert _observed_evidence(plane) == _sample(rss_bytes=2)


def test_with_no_monitor_the_soak_response_is_exactly_what_it_was(tmp_path) -> None:
    """Adding the route must not change one byte for a fleet that runs no monitor."""

    assert _observed_evidence(_plane(tmp_path)) is None


# --------------------------------------------------------------------------- refusing


@pytest.mark.parametrize("field", EXTERNAL_EVIDENCE_FIELDS)
def test_a_sample_missing_any_single_field_is_refused_and_stores_nothing(tmp_path, field: str) -> None:
    """The schema has eight keys and no optional ones.

    Accepting seven would mean the control plane decided what the eighth was.
    """

    plane = _plane(tmp_path)
    incomplete = {name: value for name, value in _sample().items() if name != field}

    refused = plane.post(EVIDENCE_PATH, json=incomplete, headers=ADMIN)

    assert refused.status_code == 400, refused.text
    assert _observed_evidence(plane) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "eyvan.soak.external-evidence.v2"),
        ("schema", ""),
        ("schema", None),
        ("proof_id", "sha256:not-hex"),
        ("proof_id", "SHA256:" + "d" * 64),
        ("proof_id", None),
        ("rss_bytes", "543210"),
        ("rss_bytes", -1),
        ("rss_bytes", 1.5),
        ("rss_bytes", True),
        ("rss_bytes", None),
        ("fd_count", "91"),
        ("fd_count", -1),
        ("fd_count", None),
        ("database_bytes", "1048576"),
        ("database_bytes", -1),
        ("database_bytes", None),
        ("provider_inventory_count", "3"),
        ("provider_inventory_count", -1),
        ("provider_inventory_count", None),
        ("spend_reconciled", 1),
        ("spend_reconciled", "true"),
        ("spend_reconciled", None),
        ("cleanup_complete", 0),
        ("cleanup_complete", "false"),
        ("cleanup_complete", None),
    ],
)
def test_a_mistyped_field_is_refused_rather_than_coerced(tmp_path, field: str, value: object) -> None:
    """``"12"`` is not ``12`` and ``0`` is not ``false``.

    Coercing either would mean the sample the evaluator reads is not the sample
    the monitor measured, and ``null`` is the one thing the schema cannot say.
    """

    plane = _plane(tmp_path)

    refused = plane.post(EVIDENCE_PATH, json=_sample(**{field: value}), headers=ADMIN)

    assert refused.status_code == 400, refused.text
    assert _observed_evidence(plane) is None


def test_an_extra_field_is_refused(tmp_path) -> None:
    """The consumer's parser rejects a widened document, so the store must too."""

    plane = _plane(tmp_path)

    refused = plane.post(EVIDENCE_PATH, json={**_sample(), "host": "monitor-1"}, headers=ADMIN)

    assert refused.status_code == 400, refused.text
    assert _observed_evidence(plane) is None


def test_a_refusal_never_replaces_the_sample_already_stored(tmp_path) -> None:
    """A broken monitor must not be able to erase a good measurement."""

    plane = _plane(tmp_path)
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)

    plane.post(EVIDENCE_PATH, json=_sample(rss_bytes=-1), headers=ADMIN)

    assert _observed_evidence(plane) == _sample()


def test_a_body_past_its_bound_is_refused(tmp_path) -> None:
    """A closed eight-key document is a few hundred bytes; anything else is not one."""

    plane = _plane(tmp_path)
    oversized = json.dumps(_sample()).encode() + b" " * MAX_EVIDENCE_BODY_BYTES

    refused = plane.request(
        "POST",
        EVIDENCE_PATH,
        content=oversized,
        headers={**ADMIN, "Content-Type": "application/json"},
    )

    assert refused.status_code == 413, refused.text
    assert _observed_evidence(plane) is None


def test_a_body_that_is_not_json_is_refused(tmp_path) -> None:
    plane = _plane(tmp_path)

    refused = plane.request(
        "POST",
        EVIDENCE_PATH,
        content=b"not json at all",
        headers={**ADMIN, "Content-Type": "application/json"},
    )

    assert refused.status_code == 400, refused.text
    assert _observed_evidence(plane) is None


# --------------------------------------------------------------------------- authority


def test_the_evidence_route_refuses_an_unauthenticated_caller(tmp_path) -> None:
    """An unauthenticated writer could forge the evidence a release gate reads."""

    plane = _plane(tmp_path)

    refused = plane.post(EVIDENCE_PATH, json=_sample())

    assert refused.status_code == 401, refused.text
    assert _observed_evidence(plane) is None


def test_the_evidence_route_refuses_a_tenant_without_fleet_authority(tmp_path) -> None:
    """The sample describes the whole plane, so writing it is fleet administration."""

    plane = _plane(tmp_path)

    refused = plane.post(EVIDENCE_PATH, json=_sample(), headers=TENANT)

    assert refused.status_code == 403, refused.text
    assert _observed_evidence(plane) is None


# --------------------------------------------------------------------------- binding


def test_a_sample_bound_to_another_run_is_not_evidence_for_this_one(tmp_path) -> None:
    """Evidence measured during someone else's soak says nothing about this one."""

    plane = _plane(tmp_path)
    accepted = plane.post(EVIDENCE_PATH, json=_sample(proof_id=OTHER_PROOF_ID), headers=ADMIN)

    assert accepted.status_code == 200, accepted.text
    assert _observed_evidence(plane, proof_id=PROOF_ID) is None
    assert _observed_evidence(plane, proof_id=OTHER_PROOF_ID) == _sample(proof_id=OTHER_PROOF_ID)


# --------------------------------------------------------------------------- freshness


def test_a_sample_past_the_freshness_bound_reads_as_absent(tmp_path) -> None:
    """A monitor that died an hour ago must read as missing.

    Republishing its last sample would let a soak pass on measurements taken
    before whatever killed the monitor happened.
    """

    clock = _StepClock()
    plane = _plane(tmp_path, SoakEvidenceStore(max_age_s=60.0, monotonic=clock.monotonic, wall=clock.wall))
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)

    clock.monotonic_s += 59.0
    clock.wall_s += 59.0
    assert _observed_evidence(plane) == _sample()

    clock.monotonic_s += 2.0
    clock.wall_s += 2.0
    assert _observed_evidence(plane) is None


def test_a_suspended_host_cannot_keep_a_stale_sample_fresh(tmp_path) -> None:
    """``CLOCK_MONOTONIC`` does not advance while a machine sleeps.

    Freshness must also use wall time so a sample taken before suspend cannot
    remain a current measurement after the machine wakes.
    """

    clock = _StepClock()
    plane = _plane(tmp_path, SoakEvidenceStore(max_age_s=60.0, monotonic=clock.monotonic, wall=clock.wall))
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)

    clock.wall_s += 31.5 * 3600.0

    assert _observed_evidence(plane) is None


def test_a_backwards_wall_clock_step_cannot_refresh_a_stale_sample(tmp_path) -> None:
    """The larger of the two ages wins, so an NTP step back cannot buy freshness."""

    clock = _StepClock()
    plane = _plane(tmp_path, SoakEvidenceStore(max_age_s=60.0, monotonic=clock.monotonic, wall=clock.wall))
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)

    clock.monotonic_s += 600.0
    clock.wall_s -= 3_600.0

    assert _observed_evidence(plane) is None


def test_a_new_sample_restarts_the_freshness_window(tmp_path) -> None:
    """A live monitor keeps its evidence current; only a dead one goes stale."""

    clock = _StepClock()
    plane = _plane(tmp_path, SoakEvidenceStore(max_age_s=60.0, monotonic=clock.monotonic, wall=clock.wall))
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)
    clock.monotonic_s += 59.0
    clock.wall_s += 59.0

    plane.post(EVIDENCE_PATH, json=_sample(rss_bytes=7), headers=ADMIN)
    clock.monotonic_s += 30.0
    clock.wall_s += 30.0

    assert _observed_evidence(plane) == _sample(rss_bytes=7)


# --------------------------------------------------------------------------- the shape itself


def test_the_stored_shape_is_exactly_the_shape_the_evaluator_parses(tmp_path) -> None:
    """The producer, the store and the consumer must agree on one closed key set.

    Parsed by the evaluator's own parser rather than by a copy of the field
    list, so a widened schema on either side fails here instead of at hour 70
    of a paid run.
    """

    from eyvan.sdk.commands.soak import _parse_p8_evidence

    plane = _plane(tmp_path)
    plane.post(EVIDENCE_PATH, json=_sample(), headers=ADMIN)

    republished = _observed_evidence(plane)
    parsed = _parse_p8_evidence(republished)

    assert parsed is not None
    assert parsed.proof_id == PROOF_ID
    assert parsed.rss_bytes == 543_210
    assert parsed.fd_count == 91
    assert parsed.database_bytes == 1_048_576
    assert parsed.spend_reconciled is True
    assert parsed.provider_inventory_count == 3
    assert parsed.cleanup_complete is False


def test_a_sample_round_trips_through_its_own_parser() -> None:
    sample = ExternalEvidenceSample(
        proof_id=PROOF_ID,
        rss_bytes=1,
        fd_count=2,
        database_bytes=3,
        spend_reconciled=False,
        provider_inventory_count=4,
        cleanup_complete=True,
    )

    assert ExternalEvidenceSample.from_dict(sample.as_dict()) == sample
    assert tuple(sample.as_dict()) == EXTERNAL_EVIDENCE_FIELDS


@pytest.mark.parametrize("bound", [0, -1, float("nan"), float("inf")])
def test_a_freshness_bound_that_would_discard_every_sample_is_refused(bound: float) -> None:
    """A zero or non-finite bound reads as "the monitor is down" forever."""

    with pytest.raises(ValueError):
        SoakEvidenceStore(max_age_s=bound)


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
