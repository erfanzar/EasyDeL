# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adversarial tests for the fail-closed P8 soak evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal, Inexact, Overflow, Rounded, Underflow, localcontext
from pathlib import Path

import pytest
from click.testing import CliRunner

from eyvan.sdk.commands.soak import (
    PERSISTENCE_THRESHOLD,
    AvailabilityPolicy,
    ConfigCapture,
    SoakRunInputs,
    SoakState,
    build_soak_receipt,
    execute_soak,
    observation_slot_count,
    run_soak,
    run_soak_for_duration,
)
from eyvan.soak_contract import (
    BackendIdentity,
    CapabilityIdentity,
    ExecutionIdentity,
    ExpectedExecution,
    ParticipantBinding,
    ProviderIdentity,
    participant_manifest_sha256,
    safe_period_id,
    safe_scope_id,
)

_PROOF_ID = "sha256:" + "d" * 64
_CONFIG_ID = "sha256:" + "c" * 64
_PERIOD_1 = "2026-01-01T00:00:00+00:00"
_PERIOD_2 = "2026-02-01T00:00:00+00:00"

_COMMAND_CAPABILITY = CapabilityIdentity(
    execution="command",
    checkpointable=False,
    coordination="none",
    required_executor="eyvan-native",
)
_CALLABLE_CAPABILITY = CapabilityIdentity(
    execution="callable",
    checkpointable=True,
    coordination="none",
    required_executor="eyvan-native",
)
_CPU_IDENTITY = ExecutionIdentity(
    capability=_COMMAND_CAPABILITY,
    backend=BackendIdentity(capacity_backend="gcp-compute", executor="eyvan-native"),
    provider=ProviderIdentity(provider="gcp", resource_class="cpu", capacity_class="spot"),
)
_TPU_IDENTITY = ExecutionIdentity(
    capability=_CALLABLE_CAPABILITY,
    backend=BackendIdentity(capacity_backend="gcp-tpu", executor="eyvan-native"),
    provider=ProviderIdentity(provider="gcp", resource_class="tpu", capacity_class="spot"),
)
_CPU_PARTICIPANT = ParticipantBinding(workload_id="wl-cpu", lease_id="lease-cpu", cell_id="cell-cpu")
_TPU_PARTICIPANT = ParticipantBinding(workload_id="wl-tpu", lease_id="lease-tpu", cell_id="cell-tpu")
_CPU_EXPECTED = ExpectedExecution(participant=_CPU_PARTICIPANT, identity=_CPU_IDENTITY)
_TPU_EXPECTED = ExpectedExecution(participant=_TPU_PARTICIPANT, identity=_TPU_IDENTITY)


def _violation(
    invariant: str = "I2",
    subject: str = "lease-a",
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "invariant": invariant,
        "subject": subject,
        "detail": detail if detail is not None else f"{subject} is stranded",
    }


def _budget(
    spent: float,
    revision: int,
    *,
    period_started_at: str | None = _PERIOD_1,
    scope: str = "acme/research",
) -> dict[str, object]:
    return {
        "scope_id": safe_scope_id(scope),
        "spent": spent,
        "revision": revision,
        "period_id": safe_period_id(period_started_at),
        "period_started_at": period_started_at,
    }


def _coverage(
    *,
    cpu: int | None = None,
    tpu: int | None = None,
    extra: list[dict[str, object]] | None = None,
    proof_id: str = _PROOF_ID,
    participants: tuple[ParticipantBinding, ...] = (_CPU_PARTICIPANT, _TPU_PARTICIPANT),
    truncated: bool = False,
) -> dict[str, object]:
    executions: list[dict[str, object]] = []
    if cpu is not None:
        executions.append(
            {
                "participant_id": _CPU_PARTICIPANT.opaque_id(proof_id),
                **_CPU_IDENTITY.as_dict(),
                "execution_count": cpu,
            }
        )
    if tpu is not None:
        executions.append(
            {
                "participant_id": _TPU_PARTICIPANT.opaque_id(proof_id),
                **_TPU_IDENTITY.as_dict(),
                "execution_count": tpu,
            }
        )
    executions.extend(extra or [])
    return {
        "proof_id": proof_id,
        "manifest_sha256": participant_manifest_sha256(proof_id, participants),
        "executions": executions,
        "truncated": truncated,
    }


def _p8_evidence(
    *,
    rss: int = 1_000,
    fds: int = 10,
    database: int = 10_000,
    spend_reconciled: bool = True,
    inventory: int = 2,
    cleanup_complete: bool = False,
    proof_id: str = _PROOF_ID,
) -> dict[str, object]:
    return {
        "schema": "eyvan.soak.external-evidence.v1",
        "proof_id": proof_id,
        "rss_bytes": rss,
        "fd_count": fds,
        "database_bytes": database,
        "spend_reconciled": spend_reconciled,
        "provider_inventory_count": inventory,
        "cleanup_complete": cleanup_complete,
    }


def _payload(
    *violations: dict[str, object],
    artifact_bytes: int = 0,
    budgets: list[dict[str, object]] | None = None,
    coverage: dict[str, object] | None = None,
    p8_evidence: dict[str, object] | None = None,
    violations_truncated: bool = False,
    workload_count: int = 2,
    cell_count: int = 2,
) -> dict[str, object]:
    return {
        "checked": [f"I{number}" for number in range(1, 10)],
        "violations": list(violations),
        "violations_truncated": violations_truncated,
        "snapshot": {
            "observed_at": "2026-01-01T00:00:00+00:00",
            "budget_spend": list(budgets or []),
            "budget_spend_truncated": False,
            "artifact_bytes": artifact_bytes,
            "workload_count": workload_count,
            "cell_count": cell_count,
            "coverage": coverage,
            "p8_evidence": p8_evidence,
        },
    }


def _run(payloads: list[dict[str, object]]) -> SoakState:
    remaining = list(payloads)
    return run_soak(
        lambda: remaining.pop(0),
        ticks=len(remaining),
        sleep=lambda _seconds: None,
        interval_s=0.0,
    )


def _inputs(
    *,
    duration_s: object = 3,
    policy: AvailabilityPolicy | None = None,
    expected: tuple[ExpectedExecution, ...] = (_CPU_EXPECTED, _TPU_EXPECTED),
    candidate_sha: str | None = "a" * 40,
    source_sha: str | None = "b" * 40,
    proof_id: str | None = _PROOF_ID,
    config_materialized: bool = True,
    config_sha256: str = _CONFIG_ID,
    rss_bound: int | None = 100,
    fd_bound: int | None = 10,
    database_bound: int | None = 1_000,
    artifact_bound: int | None = 10,
) -> SoakRunInputs:
    return SoakRunInputs.capture(
        candidate_sha=candidate_sha,
        source_sha=source_sha,
        config_sha256=config_sha256,
        config_file_count=1,
        config_snapshot_materialized=config_materialized,
        proof_id=proof_id,
        requested_duration_s=duration_s,
        interval_s=1,
        expected_executions=expected,
        max_rss_growth_bytes=rss_bound,
        max_fd_growth=fd_bound,
        max_database_growth_bytes=database_bound,
        max_artifact_net_growth_bytes=artifact_bound,
        availability_policy=policy,
    )


def _qualifying_payloads(*, violation: dict[str, object] | None = None) -> list[dict[str, object]]:
    findings = () if violation is None else (violation,)
    return [
        _payload(
            *findings,
            artifact_bytes=0,
            coverage=_coverage(),
            p8_evidence=_p8_evidence(),
        ),
        _payload(
            *findings,
            artifact_bytes=100,
            coverage=_coverage(cpu=1, tpu=1),
            p8_evidence=_p8_evidence(rss=1_025, fds=12, database=10_100),
        ),
        _payload(
            *findings,
            artifact_bytes=0,
            coverage=_coverage(cpu=1, tpu=1),
            p8_evidence=_p8_evidence(
                rss=1_020,
                fds=11,
                database=10_080,
                inventory=0,
                cleanup_complete=True,
            ),
        ),
    ]


def _build_receipt(
    state: SoakState,
    *,
    inputs: SoakRunInputs | None = None,
    actual_s: object | None = None,
    extra_reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    run_inputs = inputs or _inputs()
    actual = run_inputs.requested_duration_s if actual_s is None else actual_s
    started = datetime(2026, 1, 1, tzinfo=UTC)
    return build_soak_receipt(
        state,
        run_inputs,
        started_at=started,
        ended_at=started + timedelta(seconds=run_inputs.requested_duration_s),
        actual_duration_s=actual,
        status="completed",
        extra_reasons=extra_reasons,
    )


# --------------------------------------------------------------------------- invariant folding


def test_a_healthy_point_in_time_run_reports_healthy() -> None:
    state = _run([_payload() for _ in range(5)])
    assert state.report()["healthy"] is True
    assert state.report()["observations"] == 5


def test_transient_and_persistent_violations_use_only_opaque_subjects() -> None:
    transient = _run([_payload(_violation()), _payload(), _payload()])
    assert transient.report()["healthy"] is True
    assert transient.report()["violations"] == []

    persistent = _run([_payload(_violation(subject="tenant-secret")) for _ in range(PERSISTENCE_THRESHOLD)])
    finding = persistent.report()["violations"][0]
    assert finding["invariant"] == "I2"
    assert finding["subject_id"].startswith("sha256:")
    assert "subject" not in finding
    assert "detail" not in finding
    assert "tenant-secret" not in json.dumps(finding)


def test_flapping_violations_reset_instead_of_accumulating() -> None:
    state = _run([_payload(_violation()), _payload(_violation()), _payload()] * 5)
    assert state.report()["healthy"] is True
    assert state.report()["violations"] == []


def test_two_raw_subjects_remain_two_opaque_findings() -> None:
    state = _run([_payload(_violation(subject="lease-a"), _violation(subject="lease-b"))] * PERSISTENCE_THRESHOLD)
    ids = {item["subject_id"] for item in state.report()["violations"]}
    assert len(ids) == 2
    assert all(item.startswith("sha256:") for item in ids)


# --------------------------------------------------------------------------- availability taxonomy


def test_only_timeout_and_unreachable_failures_consume_availability_budget() -> None:
    calls = 0

    def observe() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("restart")
        return _payload()

    state = run_soak(observe, ticks=4, sleep=lambda _seconds: None, interval_s=0)
    strict = state.report()
    tolerated = state.report(AvailabilityPolicy(1, 1))
    assert strict["healthy"] is False
    assert tolerated["availability_within_policy"] is True
    assert tolerated["healthy"] is True
    assert tolerated["availability_budgeted_observations"] == 1
    assert tolerated["non_transient_failed_observations"] == 0


@pytest.mark.parametrize(
    "error",
    [
        ValueError("malformed"),
        PermissionError("denied"),
        RuntimeError("internal"),
    ],
)
def test_schema_authorization_filesystem_and_internal_failures_never_fit_availability_budget(error: Exception) -> None:
    state = SoakState()
    state.record_failure(error)
    report = state.report(AvailabilityPolicy(10, 10))
    assert report["availability_within_policy"] is True
    assert report["non_transient_failed_observations"] == 1
    assert report["healthy"] is False


@pytest.mark.parametrize(("status_code", "category"), [(403, "authorization"), (500, "server")])
def test_http_authorization_and_server_errors_never_fit_availability_budget(
    status_code: int,
    category: str,
) -> None:
    class HttpFailure(RuntimeError):
        pass

    error = HttpFailure("secret response")
    error.status_code = status_code  # type: ignore[attr-defined]
    state = SoakState()
    state.record_failure(error)
    report = state.report(AvailabilityPolicy(10, 10))
    assert report["failure_categories"] == {category: 1}
    assert report["healthy"] is False
    assert report["non_transient_failed_observations"] == 1


def test_counter_overflow_cannot_be_budgeted_into_a_false_pass() -> None:
    payloads = _qualifying_payloads()
    overflow = _payload(
        artifact_bytes=2**63,
        coverage=_coverage(),
        p8_evidence=_p8_evidence(),
    )
    state = _run([payloads[0], overflow, payloads[1], payloads[2]])
    receipt = _build_receipt(
        state,
        inputs=_inputs(duration_s=4, policy=AvailabilityPolicy(1, 1)),
        actual_s=4,
    )
    assert receipt["failure_categories"] == {"invalid_response": 1}
    assert receipt["availability_within_policy"] is True
    assert receipt["verdict"] == "inconclusive"
    assert receipt["exit_code"] == 2
    assert "non_transient_observation_error" in receipt["reasons"]


# --------------------------------------------------------------------------- budget period identity


def test_spend_decrease_across_an_ordinary_revision_increment_is_caught() -> None:
    state = _run(
        [
            _payload(budgets=[_budget(50, 1)]),
            _payload(budgets=[_budget(30, 2)]),
        ]
    )
    findings = state.report()["over_time"]
    receipt = _build_receipt(state, inputs=_inputs(duration_s=2), actual_s=2)
    assert receipt["verdict"] == "fail"
    assert receipt["exit_code"] == 1
    assert findings == [
        {
            "invariant": "I6",
            "subject_id": safe_scope_id("acme/research"),
            "code": "budget_spend_decreased",
        }
    ]


def test_only_a_proved_forward_period_transition_allows_spend_reset() -> None:
    rolled = _run(
        [
            _payload(budgets=[_budget(50, 7, period_started_at=_PERIOD_1)]),
            _payload(budgets=[_budget(0, 8, period_started_at=_PERIOD_2)]),
        ]
    )
    assert rolled.report()["over_time"] == []

    unproved = _run(
        [
            _payload(budgets=[_budget(50, 7, period_started_at=_PERIOD_1)]),
            _payload(budgets=[_budget(0, 7, period_started_at=_PERIOD_2)]),
        ]
    )
    assert unproved.report()["over_time"][0]["code"] == "budget_period_advance_unproved"


def test_budget_revision_regression_is_always_a_finding() -> None:
    state = _run([_payload(budgets=[_budget(50, 3)]), _payload(budgets=[_budget(60, 2)])])
    assert state.report()["over_time"][0]["code"] == "budget_revision_regressed"


# --------------------------------------------------------------------------- exact joint coverage


def test_a_complete_joint_cpu_tpu_receipt_can_pass_with_all_external_evidence() -> None:
    receipt = _build_receipt(_run(_qualifying_payloads()))
    assert receipt["schema_id"] == "eyvan.soak.receipt"
    assert receipt["verdict"] == "pass"
    assert receipt["exit_code"] == 0
    assert receipt["healthy"] is True
    assert len(receipt["expected_executions"]) == 2
    assert receipt["identity_requirements_met"] is True
    assert receipt["hardware_attested"] is False
    assert receipt["execution_evidence_scope"] == "durable_control_plane_state"


def test_separate_capability_backend_provider_projections_cannot_substitute_for_joint_identity() -> None:
    wrong = ExecutionIdentity(
        capability=_CPU_IDENTITY.capability,
        backend=_TPU_IDENTITY.backend,
        provider=_TPU_IDENTITY.provider,
    )
    wrong_counter = {
        "participant_id": _TPU_PARTICIPANT.opaque_id(_PROOF_ID),
        **wrong.as_dict(),
        "execution_count": 1,
    }
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["coverage"] = _coverage(cpu=1, extra=[wrong_counter])  # type: ignore[index]
    payloads[2]["snapshot"]["coverage"] = _coverage(cpu=1, extra=[wrong_counter])  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    assert receipt["verdict"] == "inconclusive"
    assert "expected_execution_not_exercised" in receipt["reasons"]


def test_unrelated_participant_cannot_satisfy_the_exact_manifest() -> None:
    unrelated = ParticipantBinding("wl-other", "lease-other", "cell-other")
    counter = {
        "participant_id": unrelated.opaque_id(_PROOF_ID),
        **_TPU_IDENTITY.as_dict(),
        "execution_count": 1,
    }
    # The server would bind a different manifest. A forged counter under the
    # expected manifest remains an unexpected participant and cannot match.
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["coverage"] = _coverage(cpu=1, extra=[counter])  # type: ignore[index]
    payloads[2]["snapshot"]["coverage"] = _coverage(cpu=1, extra=[counter])  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    assert receipt["verdict"] == "inconclusive"
    assert len(receipt["missing_expected_executions"]) == 1


def test_p8_inputs_require_both_cpu_and_tpu_joint_execution_identities() -> None:
    state = _run(
        [
            _payload(artifact_bytes=0, coverage=_coverage(), p8_evidence=_p8_evidence()),
            _payload(artifact_bytes=100, coverage=_coverage(cpu=1), p8_evidence=_p8_evidence()),
            _payload(
                artifact_bytes=0,
                coverage=_coverage(cpu=1),
                p8_evidence=_p8_evidence(inventory=0, cleanup_complete=True),
            ),
        ]
    )
    receipt = _build_receipt(state, inputs=_inputs(expected=(_CPU_EXPECTED,)))
    assert receipt["verdict"] == "inconclusive"
    assert "p8_tpu_mix_missing" in receipt["reasons"]


def test_coverage_proof_or_manifest_change_is_sticky_and_blocks_pass() -> None:
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["coverage"]["proof_id"] = "sha256:" + "e" * 64  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    assert receipt["verdict"] == "inconclusive"
    assert "coverage_binding_mismatch" in receipt["reasons"]


def test_duplicate_and_overflow_coverage_counters_are_non_qualifying() -> None:
    duplicate = {
        "participant_id": _CPU_PARTICIPANT.opaque_id(_PROOF_ID),
        **_CPU_IDENTITY.as_dict(),
        "execution_count": 1,
    }
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["coverage"]["executions"].append(dict(duplicate))  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    assert receipt["verdict"] == "inconclusive"
    assert "coverage_duplicate" in receipt["reasons"]

    overflow = _payload(
        coverage={
            **_coverage(),
            "executions": [{**duplicate, "execution_count": 2**63}],
        },
    )
    state = _run([overflow])
    assert state.failure_categories == {"invalid_response": 1}


# --------------------------------------------------------------------------- required P8 resource/cleanup boundary


def test_missing_resource_spend_and_provider_cleanup_evidence_has_explicit_exit_two_codes() -> None:
    payloads = _qualifying_payloads()
    for payload in payloads:
        payload["snapshot"]["p8_evidence"] = None  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    expected = {
        "p8_evidence_missing",
        "p8_evidence_binding_mismatch",
        "rss_growth_evidence_missing",
        "fd_growth_evidence_missing",
        "database_growth_evidence_missing",
        "spend_reconciliation_evidence_missing",
        "provider_inventory_evidence_missing",
        "cleanup_evidence_missing",
    }
    assert receipt["verdict"] == "inconclusive"
    assert receipt["exit_code"] == 2
    assert expected <= set(receipt["reasons"])


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("rss_bytes", 1_500, "rss_growth_exceeded"),
        ("fd_count", 50, "fd_growth_exceeded"),
        ("database_bytes", 12_000, "database_growth_exceeded"),
    ],
)
def test_resource_growth_over_a_required_bound_fails(field: str, value: int, reason: str) -> None:
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["p8_evidence"][field] = value  # type: ignore[index]
    receipt = _build_receipt(_run(payloads))
    assert receipt["verdict"] == "fail"
    assert reason in receipt["failure_reason_codes"]


def test_unreconciled_spend_or_residual_inventory_or_incomplete_cleanup_fails() -> None:
    for update, reason in (
        ({"spend_reconciled": False}, "spend_unreconciled"),
        ({"provider_inventory_count": 1}, "provider_inventory_not_empty"),
        ({"cleanup_complete": False}, "cleanup_incomplete"),
    ):
        payloads = _qualifying_payloads()
        payloads[-1]["snapshot"]["p8_evidence"].update(update)  # type: ignore[index]
        receipt = _build_receipt(_run(payloads))
        assert receipt["verdict"] == "fail"
        assert reason in receipt["failure_reason_codes"]


def test_artifact_creation_and_collection_do_not_hide_unbounded_net_growth() -> None:
    payloads = _qualifying_payloads()
    payloads[1]["snapshot"]["artifact_bytes"] = 100  # type: ignore[index]
    payloads[2]["snapshot"]["artifact_bytes"] = 50  # type: ignore[index]
    receipt = _build_receipt(_run(payloads), inputs=_inputs(artifact_bound=10))
    assert receipt["artifact"]["created_bytes"] == 100
    assert receipt["artifact"]["collected_bytes"] == 50
    assert receipt["artifact"]["net_growth_bytes"] == 50
    assert receipt["verdict"] == "fail"
    assert "artifact_net_growth_exceeded" in receipt["failure_reason_codes"]


def test_missing_growth_bounds_are_explicitly_inconclusive() -> None:
    receipt = _build_receipt(
        _run(_qualifying_payloads()),
        inputs=_inputs(rss_bound=None, fd_bound=None, database_bound=None, artifact_bound=None),
    )
    assert receipt["verdict"] == "inconclusive"
    assert {
        "rss_growth_bound_missing",
        "fd_growth_bound_missing",
        "database_growth_bound_missing",
        "artifact_growth_bound_missing",
    } <= set(receipt["reasons"])


# --------------------------------------------------------------------------- privacy and finite receipt values


def test_receipt_never_exports_raw_subject_detail_invalid_identity_or_arbitrary_reason() -> None:
    secret_values = (
        "org-secret",
        "project-secret",
        "supersecret",
        "private-bucket",
        "bad-candidate-secret",
        "sk-live-123",
    )
    violation = _violation(
        subject="org-secret/project-secret",
        detail="supersecret gs://private-bucket sk-live-123",
    )
    state = _run([_payload(violation) for _ in range(PERSISTENCE_THRESHOLD)])
    receipt = _build_receipt(
        state,
        inputs=_inputs(candidate_sha="bad-candidate-secret", source_sha="sk-live-123"),
        extra_reasons=("gs://private-bucket/supersecret",),
    )
    serialized = json.dumps(receipt, sort_keys=True, allow_nan=False)
    assert all(value not in serialized for value in secret_values)
    assert receipt["candidate_sha"] is None
    assert receipt["source_sha"] is None
    assert all(reason in receipt["reason_counts"] for reason in receipt["reasons"])
    assert all(" " not in reason for reason in receipt["reasons"])


def test_actual_duration_nan_inf_and_overflow_are_finite_inconclusive_receipts() -> None:
    state = _run(_qualifying_payloads())
    for invalid in (float("nan"), float("inf"), 1e308):
        receipt = _build_receipt(state, actual_s=invalid)
        assert receipt["verdict"] == "inconclusive"
        assert "duration_evidence_invalid" in receipt["reasons"]
        json.dumps(receipt, allow_nan=False)


def test_actual_duration_that_would_round_in_json_is_rejected_as_evidence() -> None:
    receipt = _build_receipt(
        _run(_qualifying_payloads()),
        actual_s=Decimal("1.0000000000000000000000000001"),
    )
    assert receipt["actual_duration_s"] == 0.0
    assert receipt["verdict"] == "inconclusive"
    assert "duration_evidence_invalid" in receipt["reasons"]
    json.dumps(receipt, allow_nan=False)


# --------------------------------------------------------------------------- fixed decimal scheduling and atomic files


class _FakeClock:
    def __init__(self, *, interrupt: bool = False):
        self.elapsed = 0.0
        self.interrupt = interrupt
        self.epoch = datetime(2026, 1, 1, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.elapsed

    def now(self) -> datetime:
        return self.epoch + timedelta(seconds=self.elapsed)

    def sleep(self, seconds: float) -> None:
        if self.interrupt:
            self.elapsed += min(seconds, 0.25)
            raise KeyboardInterrupt
        self.elapsed += seconds


def test_decimal_grid_two_point_one_over_point_three_is_exactly_seven() -> None:
    clock = _FakeClock()
    state = run_soak_for_duration(
        lambda: _payload(),
        duration_s=Decimal("2.1"),
        interval_s=Decimal("0.3"),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert observation_slot_count("2.1", "0.3") == 7
    assert state.attempts == 7
    assert clock.elapsed == pytest.approx(2.1)


def test_decimal_grid_rejects_extreme_exponents_before_integer_expansion() -> None:
    with localcontext() as context:
        context.prec = 1
        for signal in (Inexact, Rounded, Overflow, Underflow):
            context.traps[signal] = False
        for interval_s in ("1e-129", "1e-999999", "1e-1000000", "1e-999999999"):
            with pytest.raises(ValueError, match="bounded finite decimal"):
                observation_slot_count("252", interval_s)


def test_decimal_grid_rejects_decimal_objects_that_bypass_the_text_digit_bound() -> None:
    too_many_digits = Decimal((0, (1,) * 129, -128))
    with pytest.raises(ValueError, match="bounded finite decimal"):
        observation_slot_count(too_many_digits, "1")


@pytest.mark.parametrize(
    ("duration_s", "interval_s", "expected"),
    [
        ("0.001", "3", 1),
        ("2", "3", 1),
        ("252", "0.1", 2520),
    ],
)
def test_decimal_grid_is_independent_of_trapped_ambient_context_signals(
    duration_s: str,
    interval_s: str,
    expected: int,
) -> None:
    with localcontext() as context:
        context.prec = 1
        context.Emin = -2
        context.Emax = 2
        for signal in (Inexact, Rounded, Overflow, Underflow):
            context.traps[signal] = True
        assert observation_slot_count(duration_s, interval_s) == expected


def test_long_finite_decimal_ratio_uses_the_exact_integer_ceiling() -> None:
    # Exact seconds for 1.0000000000000000000000000001 hours. The final
    # observation is just beyond 3600 seconds, so a rounded Decimal quotient
    # incorrectly returns ten while the exact half-open grid contains eleven.
    duration_s = "3600.00000000000000000000000036"
    assert observation_slot_count(duration_s, "360") == 11


def test_grid_that_collapses_only_at_the_float_clock_boundary_is_rejected() -> None:
    duration_s = "0.3"
    interval_s = "0.09999999999999999"
    assert observation_slot_count(duration_s, interval_s) == 4
    with pytest.raises(ValueError, match="monotonic clock"):
        run_soak_for_duration(
            lambda: _payload(),
            duration_s=duration_s,
            interval_s=interval_s,
            sleep=lambda _seconds: None,
            monotonic=lambda: 0.0,
        )


def test_final_receipt_atomically_replaces_a_stale_pass(tmp_path: Path) -> None:
    clock = _FakeClock()
    payloads = _qualifying_payloads()
    output = tmp_path / "soak.json"
    output.write_text('{"verdict":"pass","stale":true}')
    receipt = execute_soak(
        lambda: payloads.pop(0),
        inputs=_inputs(),
        output=output,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
    )
    assert receipt["status"] == "completed"
    assert receipt["verdict"] == "pass"
    assert json.loads(output.read_text()) == receipt
    assert list(tmp_path.glob(".soak.json.*.tmp")) == []


def test_interruption_leaves_a_closed_atomic_exit_two_receipt(tmp_path: Path) -> None:
    clock = _FakeClock(interrupt=True)
    output = tmp_path / "soak.json"
    receipt = execute_soak(
        lambda: _qualifying_payloads()[0],
        inputs=_inputs(),
        output=output,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
    )
    assert json.loads(output.read_text()) == receipt
    assert receipt["status"] == "interrupted"
    assert receipt["verdict"] == "inconclusive"
    assert receipt["exit_code"] == 2


def test_frozen_config_snapshot_is_content_addressed_and_original_restore_is_irrelevant(tmp_path: Path) -> None:
    config = tmp_path / "fleet.yaml"
    config.write_text("fleet: before\n")
    captured = ConfigCapture.capture([config])
    snapshot = captured.materialize(tmp_path / "snapshots")
    frozen = (snapshot / "0000.config").read_bytes()

    config.write_text("fleet: changed\n")
    config.write_text("fleet: before\n")

    assert frozen == b"fleet: before\n"
    assert (snapshot / "0000.config").read_bytes() == frozen
    assert captured.materialize(tmp_path / "snapshots") == snapshot
    receipt = _build_receipt(_run(_qualifying_payloads()))
    assert receipt["configuration"]["frozen_content_addressed_snapshot"] is True
    assert receipt["verdict"] == "pass"


def test_changed_then_restored_source_config_cannot_change_frozen_execution_snapshot(tmp_path: Path) -> None:
    config = tmp_path / "fleet.yaml"
    config.write_text("fleet: frozen\n")
    capture = ConfigCapture.capture([config])
    capture.materialize(tmp_path / "snapshots")
    clock = _FakeClock()
    payloads = _qualifying_payloads()

    def mutate_source(tick: int, _state: SoakState) -> None:
        if tick == 0:
            config.write_text("fleet: transient-mutation\n")
        elif tick == 1:
            config.write_text("fleet: frozen\n")

    receipt = execute_soak(
        lambda: payloads.pop(0),
        inputs=_inputs(config_sha256=capture.combined_sha256),
        output=tmp_path / "soak.json",
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        now=clock.now,
        # Legacy endpoint-poll callbacks are intentionally not consulted.
        config_is_unchanged=lambda: False,
        on_tick=mutate_source,
    )
    assert receipt["verdict"] == "pass"
    assert config.read_text() == "fleet: frozen\n"
    snapshot = tmp_path / "snapshots" / capture.combined_sha256.removeprefix("sha256:") / "0000.config"
    assert snapshot.read_text() == "fleet: frozen\n"


# --------------------------------------------------------------------------- CLI preflight adversaries


def _execution_spec(participant: ParticipantBinding, identity: ExecutionIdentity) -> str:
    capability = identity.capability
    backend = identity.backend
    provider = identity.provider
    return ",".join(
        (
            participant.workload_id,
            participant.lease_id,
            participant.cell_id,
            capability.execution,
            str(capability.checkpointable).lower(),
            capability.coordination,
            capability.required_executor,
            backend.capacity_backend,
            backend.executor,
            provider.provider,
            provider.resource_class,
            provider.capacity_class,
        )
    )


def _cli_base(tmp_path: Path, *, candidate: str = "bad-candidate") -> tuple[list[str], Path]:
    config = tmp_path / "fleet.yaml"
    config.write_text("fleet: immutable\n")
    output = tmp_path / "soak.json"
    args = [
        "--user",
        "root",
        "--role",
        "fleet_admin",
        "soak",
        "--candidate-sha",
        candidate,
        "--source-sha",
        "b" * 40,
        "--proof-id",
        _PROOF_ID,
        "--config",
        str(config),
        "--config-snapshot-dir",
        str(tmp_path / "snapshots"),
        "--expect-execution",
        _execution_spec(_CPU_PARTICIPANT, _CPU_IDENTITY),
        "--expect-execution",
        _execution_spec(_TPU_PARTICIPANT, _TPU_IDENTITY),
        "--max-rss-growth-bytes",
        "100",
        "--max-fd-growth",
        "10",
        "--max-database-growth-bytes",
        "1000",
        "--max-artifact-net-growth-bytes",
        "10",
        "--output",
        str(output),
    ]
    return args, output


@pytest.mark.parametrize(
    "numeric_args",
    [
        ["--hours", "nan"],
        ["--hours", "inf"],
        ["--hours", "1e308"],
        ["--interval-s", "nan"],
        ["--interval-s", "1e308"],
        ["--max-failed-observations", str(2**63)],
    ],
)
def test_every_invalid_numeric_preflight_atomically_replaces_stale_pass(
    tmp_path: Path,
    numeric_args: list[str],
) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path)
    output.write_text('{"verdict":"pass","stale":true}')
    result = CliRunner().invoke(cli.main, [*args, *numeric_args])
    assert result.exit_code == 2, result.output
    persisted = json.loads(output.read_text())
    assert persisted["status"] == "not_started"
    assert persisted["verdict"] == "inconclusive"
    assert persisted["reasons"] == ["invalid_numeric_input"]
    assert "stale" not in persisted
    json.dumps(persisted, allow_nan=False)


def test_extreme_decimal_interval_replaces_stale_pass_without_traceback(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path)
    output.write_text('{"verdict":"pass","stale":true}')
    result = CliRunner().invoke(cli.main, [*args, "--hours", "0.07", "--interval-s", "1e-999999"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, ArithmeticError)
    persisted = json.loads(output.read_text())
    assert persisted["status"] == "not_started"
    assert persisted["verdict"] == "inconclusive"
    assert persisted["exit_code"] == 2
    assert persisted["reasons"] == ["invalid_numeric_input"]
    assert "stale" not in persisted
    assert list(tmp_path.glob(".soak.json.*.tmp")) == []
    json.dumps(persisted, allow_nan=False)


def test_click_preserves_decimal_hours_text_through_seconds_and_slot_count(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path, candidate="invalid-so-the-run-does-not-poll")
    result = CliRunner().invoke(cli.main, [*args, "--hours", "0.07", "--interval-s", "0.1"])
    assert result.exit_code == 2, result.output
    receipt = json.loads(output.read_text())
    assert receipt["requested_duration_s"] == 252.0
    assert receipt["observation_counts"]["requested"] == 2520
    assert receipt["candidate_sha"] is None


@pytest.mark.parametrize("precision", [1, 2])
def test_real_cli_ignores_untrapped_low_precision_sitecustomize_decimal_context(
    tmp_path: Path,
    precision: int,
) -> None:
    site_directory = tmp_path / "site"
    site_directory.mkdir()
    marker = tmp_path / "sitecustomize-loaded"
    (site_directory / "sitecustomize.py").write_text(
        f"""import os
from decimal import getcontext

context = getcontext()
context.prec = {precision}
for signal in context.traps:
    context.traps[signal] = False
with open(os.environ["EYVAN_TEST_DECIMAL_CONTEXT_MARKER"], "w", encoding="utf-8") as handle:
    handle.write(str(context.prec))
""",
        encoding="utf-8",
    )
    args, output = _cli_base(tmp_path, candidate="invalid-so-the-run-does-not-poll")
    output.write_text('{"verdict":"pass","stale":true}', encoding="utf-8")
    environment = os.environ.copy()
    environment["EYVAN_TEST_DECIMAL_CONTEXT_MARKER"] = str(marker)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(site_directory), environment.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-m", "eyvan.sdk.cli", *args, "--hours", "0.07", "--interval-s", "0.1"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert marker.read_text(encoding="utf-8") == str(precision)
    assert completed.returncode == 2, (completed.stdout, completed.stderr)
    assert "Traceback" not in completed.stdout
    assert "Traceback" not in completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "not_started"
    assert receipt["requested_duration_s"] == 252.0
    assert receipt["duration"]["requested_s"] == 252.0
    assert receipt["observation_counts"]["requested"] == 2520
    assert "stale" not in receipt
    json.dumps(receipt, allow_nan=False)


def test_cli_rejects_duration_that_cannot_survive_receipt_and_clock_float_conversion(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path)
    output.write_text('{"verdict":"pass","stale":true}', encoding="utf-8")
    result = CliRunner().invoke(
        cli.main,
        [*args, "--hours", "1.0000000000000000000000000001", "--interval-s", "360"],
    )
    assert result.exit_code == 2, result.output
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "not_started"
    assert receipt["reasons"] == ["invalid_numeric_input"]
    assert "requested_duration_s" not in receipt
    assert "stale" not in receipt


@pytest.mark.parametrize("mode", ["missing_config", "snapshot_path_is_file"])
def test_filesystem_preflight_errors_atomically_replace_stale_pass(tmp_path: Path, mode: str) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path)
    output.write_text('{"verdict":"pass","stale":true}')
    if mode == "missing_config":
        args.extend(["--config", str(tmp_path / "missing.yaml")])
    else:
        invalid_snapshot_root = tmp_path / "not-a-directory"
        invalid_snapshot_root.write_text("file")
        args.extend(["--config-snapshot-dir", str(invalid_snapshot_root)])
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "not_started"
    assert "invalid_configuration_input" in receipt["reasons"]
    assert "stale" not in receipt


def test_invalid_expected_identity_does_not_echo_secret_and_replaces_stale_pass(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    args, output = _cli_base(tmp_path)
    output.write_text('{"verdict":"pass","stale":true}')
    secret = "gs://private-bucket/supersecret?project=tenant"
    result = CliRunner().invoke(cli.main, [*args, "--expect-execution", secret])
    assert result.exit_code == 2
    receipt = json.loads(output.read_text())
    serialized = json.dumps(receipt)
    assert receipt["status"] == "not_started"
    assert "invalid_expected_execution_input" in receipt["reasons"]
    assert "private-bucket" not in serialized
    assert "supersecret" not in serialized
    assert "stale" not in serialized


def test_output_config_alias_still_replaces_a_stale_pass_with_closed_preflight(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    aliased = tmp_path / "same.json"
    aliased.write_text('{"verdict":"pass","stale":true}')
    result = CliRunner().invoke(
        cli.main,
        [
            "--user",
            "root",
            "--role",
            "fleet_admin",
            "soak",
            "--config",
            str(aliased),
            "--output",
            str(aliased),
        ],
    )
    assert result.exit_code == 2
    receipt = json.loads(aliased.read_text())
    assert receipt["status"] == "not_started"
    assert receipt["reasons"] == ["invalid_output_input_alias"]
    assert "stale" not in receipt


def test_soak_command_is_registered_and_zero_interval_is_closed_json(tmp_path: Path) -> None:
    from eyvan.sdk import cli

    assert "soak" in cli.main.commands
    args, output = _cli_base(tmp_path)
    result = CliRunner().invoke(cli.main, [*args, "--interval-s", "0"])
    assert result.exit_code == 2
    assert json.loads(output.read_text())["reasons"] == ["invalid_numeric_input"]
