# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A network-bound head must supply everything the code says it requires.

This exists because one failure shape kept recurring and the suite could not see
it: a loopback development plane auto-configures something, a real head does not,
and the gap surfaces only when a live control plane is asked to do real work.
Every instance was caught by running a head rather than by testing one --
including two that appeared only after a TPU was provisioned, healthy, and
serving.

So the contract is derived from the source instead of being written down. Any
code path that refuses to work without a variable says so, in a message
containing "requires EYVAN_...", and this module reads those messages back and
asserts that a head either supplies the variable or is exempt for a reason
someone had to type. A new requirement automatically joins the contract; nobody
has to remember.

It deliberately does not assert a fixed list. A fixed list is a second copy of
the truth, and the whole class of bug this prevents is two copies disagreeing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eyvan.sdk.commands.head import HeadSecrets, HeadSpec

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "eyvan"

#: Matches the refusals the package raises when a variable is missing.
_REQUIREMENT = re.compile(r"requires? (EYVAN_[A-Z0-9_]+)")

#: Variables a head is not expected to supply, each with the reason it is not a
#: head's business. An entry here is a claim someone can be held to; an empty
#: reason is not accepted.
_NOT_A_HEAD_CONCERN: dict[str, str] = {
    "EYVAN_ALLOW_INSECURE_ARTIFACT_HTTP": (
        "a workload-side opt-in that weakens artifact transport; a head that set it for every "
        "workload would remove a protection nobody asked to remove"
    ),
    "EYVAN_AWS_REGION": (
        "required only by the aws-ec2 capacity backend, which a GCP head never selects. A head "
        "that wrote a region for a backend it does not run would be stating a fact about somebody "
        "else's fleet"
    ),
    "EYVAN_AWS_ACCESS_KEY_ID": (
        "same: the aws-ec2 backend's credential. It is also the kind of value a head must not "
        "invent -- a long-lived key written into every head's environment is a key in every backup"
    ),
    "EYVAN_NATIVE_WORKER_WHEELHOUSE": (
        "required only for private-google-access egress, where workers cannot reach PyPI and need "
        "their dependencies staged too; heads on cloud-nat or public-internet install normally. "
        "A head that selects private-google-access does NOT yet stage one -- see the test below, "
        "which pins that gap rather than hiding it"
    ),
}


def _declared_requirements() -> dict[str, set[str]]:
    """Return every variable the package refuses to run without, and where."""

    found: dict[str, set[str]] = {}
    for path in _PACKAGE_ROOT.rglob("*.py"):
        for match in _REQUIREMENT.finditer(path.read_text(encoding="utf-8")):
            found.setdefault(match.group(1), set()).add(path.relative_to(_PACKAGE_ROOT).as_posix())
    return found


def _head_environment(**overrides: object) -> dict[str, str]:
    """Render the environment a real GCP head writes for an eyvan-native fleet."""

    values: dict[str, object] = {
        "fleet": "research",
        "project": "example-project-1",
        "region": "us-central1",
        "zone": "us-central1-a",
        "egress": "public-internet",
        "artifact_store_uri": "gs://fleet-bucket/artifacts",
    }
    values.update(overrides)
    spec = HeadSpec(**values)  # type: ignore[arg-type]
    secrets = HeadSecrets.generate()
    from eyvan.sdk.commands.head import WorkerPackage

    package = WorkerPackage(
        uri="gs://fleet-bucket/artifacts/native-worker/" + "a" * 64 + "/eyvan-0.0.1-py3-none-any.whl",
        sha256="a" * 64,
        generation="1786489052345332",
    )
    return secrets.environment(spec, address="10.128.0.11", worker_package=package)


def test_the_source_still_declares_its_requirements_in_a_readable_way() -> None:
    """Guard the guard: a silent zero here would make every assertion below vacuous."""

    declared = _declared_requirements()

    assert len(declared) >= 5, f"only found {sorted(declared)}; the refusal wording may have changed"
    assert "EYVAN_NATIVE_WORKER_ROOT_SECRET" in declared
    assert "EYVAN_NATIVE_WORKER_PACKAGE" in declared


def test_a_head_supplies_every_variable_the_code_refuses_to_run_without() -> None:
    """The class-closing assertion.

    Five separate defects this session were one shape: a variable some code path
    refuses to run without, that `eyvan head up` never wrote. Each was found on
    live infrastructure, twice after real accelerators were already provisioned
    and billing.
    """

    environment = _head_environment()
    declared = _declared_requirements()

    missing = {
        name: sorted(paths)
        for name, paths in declared.items()
        if name not in environment and name not in _NOT_A_HEAD_CONCERN
    }

    assert not missing, (
        "a head does not supply variables the package refuses to run without: "
        + "; ".join(f"{name} (required in {', '.join(paths)})" for name, paths in sorted(missing.items()))
        + ". Either write it in HeadSecrets.environment, or add it to _NOT_A_HEAD_CONCERN with the reason."
    )


def test_every_exemption_states_a_reason_and_is_still_real() -> None:
    """An exemption that no longer matches a requirement is a stale claim."""

    declared = _declared_requirements()

    for name, reason in _NOT_A_HEAD_CONCERN.items():
        assert reason.strip(), f"{name} is exempt with no reason"
        assert name in declared, f"{name} is exempt from a requirement the package no longer declares"


def test_a_private_google_access_head_does_not_yet_stage_a_wheelhouse() -> None:
    """Pins a known gap in place rather than letting it read as covered.

    Workers on `private-google-access` cannot reach PyPI, so they need their
    dependencies staged alongside the wheel. A head that selects that egress
    publishes the wheel and not the wheelhouse, so its first TPU request is
    refused -- the same shape as the package gap, one branch over.

    Written as an assertion of the current behaviour so that fixing it fails this
    test and forces the exemption above to be revisited.
    """

    environment = _head_environment(egress="private-google-access")

    assert "EYVAN_NATIVE_WORKER_PACKAGE" in environment
    assert "EYVAN_NATIVE_WORKER_WHEELHOUSE" not in environment


@pytest.mark.parametrize(
    "name",
    [
        "EYVAN_FLEET_STATE_URI",
        "EYVAN_NATIVE_WORKER_ROOT_SECRET",
        "EYVAN_NATIVE_WORKER_PACKAGE",
        "EYVAN_NATIVE_WORKER_PACKAGE_SHA256",
        "EYVAN_NATIVE_WORKER_PACKAGE_GENERATION",
        "EYVAN_TRUSTED_CALLABLE_PROJECTS",
    ],
)
def test_each_variable_a_live_head_was_missing_is_now_written(name: str) -> None:
    """One case per defect actually observed on a live head this session.

    `EYVAN_FLEET_STATE_URI` is the one no refusal names: nothing raises without
    it, the agent simply keeps a private copy of the fleet state, and the cost is
    a cell that reaches `ready` publishing no worker endpoints while dispatch
    waits forever. A requirement nothing declares cannot be derived, so it is
    named here explicitly.
    """

    assert name in _head_environment()
