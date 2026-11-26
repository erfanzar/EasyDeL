# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which strings are evidence of a broken machine, and which only look like it.

The whole value of node health is that a match destroys a host. So the interesting
half of this file is the near misses: strings that contain a fault phrase as a
prefix, in the wrong case, or in a sentence that means something else entirely.
Each one that matched would take a healthy accelerator out of the fleet and hide
the user's real failure behind a replacement loop.

The other property under test is that a fault stays attached to the rank that
reported it. A gang's observation is every rank's text concatenated; matching it
as one blob would say "this workload saw a dead TPU" when the answerable question
is "which host".
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from eyvan._internal.compat import UTC
from eyvan.worker.runtime.node_health import (
    _DELIBERATELY_ABSENT,
    HOST_FAULT_REASON,
    HOST_FAULT_SIGNATURES,
    HostFault,
    HostFaultCode,
    HostReplacementBudget,
    format_host_fault_marker,
    host_faults_by_rank,
    host_replacement_scope,
    match_host_fault,
    parse_host_fault_marker,
    split_rank_reports,
)

FAULTS = [
    (
        "libtpu: Couldn't open iommu group /dev/vfio/12: No such device",
        HostFaultCode.TPU_IOMMU_GROUP_UNAVAILABLE,
    ),
    (
        "E0102 12:00:00.1 tpu_driver.cc:88] open(/dev/vfio/42) failed: Permission denied",
        HostFaultCode.TPU_VFIO_DEVICE_UNAVAILABLE,
    ),
    (
        "RuntimeError: Failed to initialize TPU system: 4 of 8 chips did not respond",
        HostFaultCode.TPU_SYSTEM_INIT_FAILED,
    ),
    ("F0102 tpu_initializer.cc:12] TPU initialization failed.", HostFaultCode.TPU_INITIALIZATION_FAILED),
    ("jax: No accelerator found; falling back is disabled", HostFaultCode.ACCELERATOR_ABSENT),
]

NEAR_MISSES = [
    # A prefix of a fault phrase continuing into a different sentence. This is
    # the single most dangerous shape: a telemetry sidecar failing to start is
    # not a dead accelerator.
    "Failed to initialize TPU system metrics exporter; continuing without telemetry",
    "Failed to initialize TPU system monitor connection, retrying in 5s",
    "Couldn't open iommu group configuration file, using defaults",
    "No accelerator found for the requested mesh axis; check your sharding",
    # Program-emitted text keeps its capitalisation; a user's prose does not.
    "tpu initialization failed because my flag was wrong",
    "traceback says no accelerator found in my unit test",
    # Considered for the table and deliberately excluded: each has a reading in
    # which the machine is fine.
    "RESOURCE_EXHAUSTED: Out of memory allocating 40.00G on device 0",
    "Failed to connect to coordinator at 10.0.0.4:8476 after 300s",
    "UNAVAILABLE: Socket closed",
    "RuntimeError: libtpu.so already in use by another process",
    "jaxlib.xla_extension.XlaRuntimeError: INTERNAL: something in my model",
    "Device or resource busy",
    # Nothing at all.
    "",
    'Traceback (most recent call last):\n  File "train.py", line 12, in <module>',
]


@pytest.mark.parametrize(("text", "code"), FAULTS, ids=lambda value: getattr(value, "value", None))
def test_each_signature_matches_the_message_it_was_written_for(text: str, code: HostFaultCode) -> None:
    fault = match_host_fault(text)

    assert fault is not None, f"{text!r} is a host fault and matched nothing"
    assert fault.code is code
    assert fault.evidence == text.strip()


@pytest.mark.parametrize("text", NEAR_MISSES)
def test_a_near_miss_is_not_evidence_of_a_broken_machine(text: str) -> None:
    """Each of these would destroy a healthy accelerator if it matched."""

    assert match_host_fault(text) is None


def test_every_signature_carries_a_code_and_a_diagnosis() -> None:
    """The fault record is the only account of why a machine was destroyed."""

    assert HOST_FAULT_SIGNATURES
    assert len({signature.code for signature in HOST_FAULT_SIGNATURES}) == len(HOST_FAULT_SIGNATURES)
    for signature in HOST_FAULT_SIGNATURES:
        assert signature.diagnosis.strip()


def test_a_fault_is_reported_by_its_cause_not_its_consequences() -> None:
    """A dead TPU logs one cause and then a cascade; the cause is the useful one."""

    output = "\n".join(
        (
            "libtpu: Couldn't open iommu group /dev/vfio/3",
            "F0102 tpu_initializer.cc:12] TPU initialization failed.",
            "RuntimeError: Failed to initialize TPU system",
        )
    )

    fault = match_host_fault(output)

    assert fault is not None
    assert fault.code is HostFaultCode.TPU_IOMMU_GROUP_UNAVAILABLE
    assert fault.evidence == "libtpu: Couldn't open iommu group /dev/vfio/3"


def test_the_evidence_is_one_bounded_line_of_the_output() -> None:
    """It has to fit a durable field beside a code, a workload, and a timestamp."""

    output = "noise\n" + "x" * 4000 + " TPU initialization failed\n" + "more noise"

    fault = match_host_fault(output)

    assert fault is not None
    assert len(fault.evidence) <= 320
    assert "\n" not in fault.evidence


# --------------------------------------------------------------------- attribution


def test_a_rank_report_is_split_at_its_headers_not_at_every_separator() -> None:
    """A rank's own diagnostic contains ``; `` constantly."""

    message = "rank 0=running; rank 1=failed: RuntimeError: chip 3; then chip 4 died; rank 2=succeeded"

    assert split_rank_reports(message) == (
        (0, ""),
        (1, "RuntimeError: chip 3; then chip 4 died"),
        (2, ""),
    )


def test_a_message_without_rank_structure_attributes_to_nothing() -> None:
    """Unknown is not "all of them"; a caller with two hosts has learned nothing."""

    assert split_rank_reports("cannot observe systemd service: transport failed") == ()
    assert host_faults_by_rank("Couldn't open iommu group /dev/vfio/1") == {}


def test_only_the_rank_that_reported_a_fault_carries_one() -> None:
    """The gang case: one dead TPU beside ranks that failed for ordinary reasons."""

    message = (
        "rank 0=failed: ValueError: bad shape; "
        "rank 1=failed: RuntimeError: Failed to initialize TPU system: chip 3; "
        "rank 2=running"
    )

    faults = host_faults_by_rank(message)

    assert set(faults) == {1}
    assert faults[1].code is HostFaultCode.TPU_SYSTEM_INIT_FAILED


# --------------------------------------------------------------------- the durable marker


def _fault() -> HostFault:
    return HostFault(code=HostFaultCode.TPU_SYSTEM_INIT_FAILED, evidence="Failed to initialize TPU system: chip 3")


def test_a_marker_round_trips_with_the_execution_it_was_observed_for() -> None:
    """Proof about one workload must not read as proof about the next one."""

    observed_at = datetime(2026, 8, 3, 9, 30, tzinfo=UTC)

    marker = format_host_fault_marker(_fault(), workload_id="wl-1", generation=4, observed_at=observed_at)
    proof = parse_host_fault_marker(HOST_FAULT_REASON, marker)

    assert proof is not None
    assert proof.code is HostFaultCode.TPU_SYSTEM_INIT_FAILED
    assert proof.evidence == "Failed to initialize TPU system: chip 3"
    assert proof.workload_id == "wl-1"
    assert proof.generation == 4
    assert proof.observed_at == observed_at


def test_a_marker_keeps_evidence_that_contains_separators_verbatim() -> None:
    """The evidence is quoted, so a message full of ``=`` and spaces survives."""

    fault = HostFault(
        code=HostFaultCode.TPU_VFIO_DEVICE_UNAVAILABLE,
        evidence='open(/dev/vfio/3) failed: errno=13 detail="denied" at=now',
    )

    proof = parse_host_fault_marker(
        HOST_FAULT_REASON,
        format_host_fault_marker(fault, workload_id="wl-1", generation=1, observed_at=datetime.now(UTC)),
    )

    assert proof is not None and proof.evidence == fault.evidence


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        (None, None),
        ("provider_error", "code=tpu_system_init_failed workload=wl-1 generation=1 at=2026-08-03T09:30:00+00:00 e"),
        (HOST_FAULT_REASON, None),
        (HOST_FAULT_REASON, "agent said something else entirely"),
        (HOST_FAULT_REASON, 'code=not_a_code workload=wl-1 generation=1 at=2026-08-03T09:30:00+00:00 evidence="x"'),
        (HOST_FAULT_REASON, 'code=tpu_system_init_failed workload=wl-1 generation=1 at=nonsense evidence="x"'),
    ],
)
def test_anything_that_is_not_a_readable_marker_is_no_proof(reason: str | None, detail: str | None) -> None:
    """``None`` costs capacity and never safety, so every doubt resolves to it."""

    assert parse_host_fault_marker(reason, detail) is None


def test_a_marker_requires_a_timezone_aware_observation_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        format_host_fault_marker(_fault(), workload_id="wl-1", generation=1, observed_at=datetime(2026, 8, 3, 9, 30))


# --------------------------------------------------------------------- the budget


def test_the_shipped_default_replaces_nothing() -> None:
    """This changes what happens to a failed attempt, so it is opted into."""

    assert HostReplacementBudget().max_replacements == 0
    assert HostReplacementBudget().enabled is False
    assert HostReplacementBudget(max_replacements=2).enabled is True


def test_the_budget_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYVAN_HOST_REPLACEMENT_BUDGET", "3")
    monkeypatch.setenv("EYVAN_HOST_REPLACEMENT_WINDOW_S", "900")

    budget = HostReplacementBudget.from_env()

    assert budget.max_replacements == 3
    assert budget.window_s == 900.0


def test_an_unset_environment_leaves_replacement_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_HOST_REPLACEMENT_BUDGET", raising=False)
    monkeypatch.delenv("EYVAN_HOST_REPLACEMENT_WINDOW_S", raising=False)

    assert HostReplacementBudget.from_env().enabled is False


@pytest.mark.parametrize(("budget", "window"), [(-1, 3600.0), (1, 0.0), (1, -60.0)])
def test_a_budget_that_bounds_nothing_is_refused(budget: int, window: float) -> None:
    with pytest.raises(ValueError):
        HostReplacementBudget(max_replacements=budget, window_s=window)


def test_the_budget_scope_is_the_pool_when_there_is_one() -> None:
    """A pool is the unit an operator sized and the unit a replacement lands in."""

    assert host_replacement_scope("pool-a", "gcp", "us-central2-b") == "pool:pool-a"
    assert host_replacement_scope(None, "gcp", "us-central2-b") == "unpooled:gcp/us-central2-b"
    assert host_replacement_scope(None, "gcp", "us-central2-b") != host_replacement_scope(None, "gcp", "us-east5-a")


def test_a_window_is_measured_from_the_observation_it_records() -> None:
    """Not from the row's mtime: draining and deleting both touch that."""

    old = datetime.now(UTC) - timedelta(hours=2)

    proof = parse_host_fault_marker(
        HOST_FAULT_REASON,
        format_host_fault_marker(_fault(), workload_id="wl-1", generation=1, observed_at=old),
    )

    assert proof is not None and proof.observed_at == old


def test_no_signature_matches_a_phrase_the_table_deliberately_excludes() -> None:
    """The rejected list is a decision, so it is enforced rather than admired.

    Each of these accompanies real bad-node incidents and each also has a reading
    in which the machine is fine. Adding one to the table would start replacing
    healthy hosts, and this is what notices.
    """

    assert _DELIBERATELY_ABSENT
    for phrase, why in _DELIBERATELY_ABSENT.items():
        assert match_host_fault(phrase) is None, f"{phrase!r} was excluded because {why}"
        assert why.strip()
