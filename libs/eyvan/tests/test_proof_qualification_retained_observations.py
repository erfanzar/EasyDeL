# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression contracts for proof qualification and retained observations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eyvan.records import WorkloadDefinition
from eyvan.sdk.commands.proof import Verdict, _ProofRun, evaluate_secret_delivery


def _check(result, name):
    return next(check for check in result.checks if check.name == name)


def _events():
    return [
        {"action": "secret.issued", "payload": {"attempt": 1}},
        {"action": "secret.read", "payload": {"attempt": 1}},
    ]


@pytest.mark.parametrize("value", ["plain-token", "first\nsecond", "first\tsecond", "first\\second", 'first"second'])
@pytest.mark.parametrize("surface", ["record", "audit", "log", "key"])
def test_decoded_secret_leaves_are_detected(value, surface):
    events = _events()
    workload = {"id": "wl-1"}
    logs = ""
    if surface == "record":
        workload["spec"] = {"environment": {"LEAK_COPY": value}}
    elif surface == "audit":
        events[0]["payload"]["context"] = {"nested": [value]}
    elif surface == "key":
        workload[value] = "present"
    else:
        logs = value
    result = evaluate_secret_delivery(events, workload, logs, secret_value=value)
    assert _check(result, "value-never-recorded").verdict is Verdict.FAIL
    assert value not in _check(result, "value-never-recorded").detail


@pytest.mark.parametrize("value", ["first\tsecond", "first\\second"])
def test_leaked_environment_is_a_valid_workload_spec(value):
    spec = WorkloadDefinition(
        name="secret-proof",
        execution={"kind": "command", "command": "true"},
        environment={"LEAK_COPY": value},
        secret_mounts={"A": "file:///tmp/a"},
    )
    result = evaluate_secret_delivery(_events(), {"spec": spec.model_dump(mode="json")}, "", secret_value=value)
    assert _check(result, "value-never-recorded").verdict is Verdict.FAIL


@pytest.mark.parametrize("value", ["plain-token", "first\nsecond", "first\tsecond", "first\\second", 'first"second'])
def test_evidence_redacts_before_json_encoding(tmp_path: Path, value):
    target = tmp_path / "evidence.json"
    observations = {
        "workload": {"spec": {"environment": {"LEAK_COPY": value}}},
        "audit": [{"payload": {"nested": (value,)}}],
        "logs": f"prefix {value} suffix",
        "keys": {value: "present"},
    }
    original = repr(observations)
    run = _ProofRun("P4", evaluate_secret_delivery, evidence=str(target), secrets=(value,), observations=observations)
    run._write_evidence()
    decoded = json.loads(target.read_text())
    assert value not in decoded["logs"]
    assert value != decoded["workload"]["spec"]["environment"]["LEAK_COPY"]
    assert value != decoded["audit"][0]["payload"]["nested"][0]
    assert value not in decoded["keys"]
    assert target.stat().st_mode & 0o777 == 0o600
    assert repr(observations) == original


def _issued(attempt=1, generation=1, names=("A", "B")):
    return {"action": "secret.issued", "payload": {"attempt": attempt, "generation": generation, "names": list(names)}}


def _read(name="A", attempt=1, generation=1):
    return {"action": "secret.read", "payload": {"attempt": attempt, "generation": generation, "name": name}}


def _secret_result(events, state="succeeded"):
    return evaluate_secret_delivery(
        events,
        {"state": state, "spec": {"secret_mounts": {"A": "file:///tmp/a", "B": "file:///tmp/b"}}},
        "",
        secret_value="not-recorded",
    )


def test_one_read_per_issued_mount_passes():
    assert _secret_result([_issued(), _read("A"), _read("B")]).verdict is Verdict.PASS


@pytest.mark.parametrize("read", [_read("C"), _read("A", generation=2)])
def test_unissued_mount_or_generation_cannot_qualify(read):
    result = _secret_result([_issued(names=("A",)), read])
    assert _check(result, "read-matches-an-issuance").verdict is Verdict.FAIL


def test_duplicate_redemption_of_one_mount_fails():
    result = _secret_result([_issued(), _read("A"), _read("A"), _read("B")])
    assert _check(result, "one-read").verdict is Verdict.FAIL


@pytest.mark.parametrize("state, verdict", [("failed", Verdict.FAIL), ("running", Verdict.INCONCLUSIVE)])
@pytest.mark.parametrize("missing", ["attempt", "mount"])
def test_every_issued_mount_in_every_attempt_must_be_read(state, verdict, missing):
    if missing == "attempt":
        events = [_issued(names=("A",)), _read("A"), _issued(attempt=2, generation=2, names=("A",))]
    else:
        events = [_issued(), _read("A")]
    result = _secret_result(events, state=state)
    assert _check(result, "one-read").verdict is verdict
    assert result.verdict is verdict


def test_complete_two_attempt_two_mount_history_passes():
    events = [_issued(), _read("A"), _read("B"), _issued(2, 2), _read("A", 2, 2), _read("B", 2, 2)]
    result = _secret_result(events)
    assert result.verdict is Verdict.PASS
    assert _check(result, "redelivered").verdict is Verdict.PASS


def _log_record(rank=0, **overrides):
    return {"rank": rank, "timestamp": "2026-10-01T12:00:00+00:00", "stream": "stdout", "cursor": "good", **overrides}


@pytest.mark.parametrize(
    "field, value, check_name",
    [
        ("timestamp", None, "records-carry-a-timestamp"),
        ("timestamp", "not-a-time", "records-carry-a-timestamp"),
        ("stream", "unknown", "records-carry-a-stream"),
        ("cursor", "", "records-are-resumable"),
    ],
)
def test_healthy_rank_cannot_mask_another_rank_missing_metadata(field, value, check_name):
    from eyvan.sdk.commands.proof import evaluate_structured_logs

    result = evaluate_structured_logs(
        {"records": [_log_record(), _log_record(1, **{field: value})], "truncated_ranks": [2]}, expected_ranks=3
    )
    check = _check(result, check_name)
    assert check.verdict is Verdict.FAIL
    assert "1" in check.detail
    assert result.verdict is Verdict.FAIL


def test_unavailable_rank_does_not_need_a_synthetic_metadata_record():
    from eyvan.sdk.commands.proof import evaluate_structured_logs

    result = evaluate_structured_logs({"records": [_log_record()], "truncated_ranks": [1]}, expected_ranks=2)
    assert result.verdict is Verdict.PASS


def _receipt_identity_result(plane="matching", agreement=True, proof_id="P5", unreachable=()):
    from eyvan._internal.build_identity import build_identity
    from eyvan.sdk.commands.proof import _EVALUATORS, evaluate_receipt
    from eyvan.sdk.commands.receipts import build_proof_receipt, criterion_identity, receipt_digest_matches

    build = build_identity().as_dict()
    plane_build = build if plane == "matching" else None if plane is None else {"source_digest": plane}
    evaluator = _EVALUATORS[proof_id]
    criterion = criterion_identity(evaluator)
    document = build_proof_receipt(
        proof=proof_id,
        verdict="pass",
        exit_status=0,
        checks=[],
        criterion=criterion,
        plane_url="http://localhost",
        plane_build=plane_build,
        plane_build_detail="",
        started_at="2026-10-01T12:00:00+00:00",
        completed_at="2026-10-01T12:01:00+00:00",
        command=[],
        git={"commit": "a" * 40, "clean": True},
        fleet_builds={
            "plane": build,
            "agents": [{"build": build, "workers": []}],
            "unreachable": list(unreachable),
            "in_agreement": agreement,
        },
    )
    assert receipt_digest_matches(document) is True
    return evaluate_receipt(document, digest_intact=True, criterion_now=criterion, build_now=build)


@pytest.mark.parametrize(
    "plane, agreement, verdict",
    [
        (None, True, Verdict.INCONCLUSIVE),
        ("sha256:other", True, Verdict.FAIL),
        ("matching", False, Verdict.FAIL),
        ("matching", None, Verdict.INCONCLUSIVE),
        ("matching", True, Verdict.PASS),
    ],
)
def test_receipt_requires_exact_backend_identity(plane, agreement, verdict):
    assert _receipt_identity_result(plane, agreement).verdict is verdict


def test_receipt_cannot_qualify_unreached_agents_as_agreement():
    assert _receipt_identity_result(unreachable=({"endpoint": "missing"},)).verdict is Verdict.INCONCLUSIVE


def test_kubernetes_receipt_has_no_control_plane_or_fleet_requirement():
    assert _receipt_identity_result(plane=None, agreement=None, proof_id="P7").verdict is Verdict.PASS


@pytest.mark.parametrize("proof_id", ["P1", "P6"])
def test_plane_only_proof_does_not_require_execution_fleet(proof_id):
    assert _receipt_identity_result(agreement=None, proof_id=proof_id).verdict is Verdict.PASS


@pytest.mark.parametrize("alias", ["same", "canonical"])
def test_evidence_and_receipt_targets_cannot_alias(tmp_path, alias):
    target = tmp_path / "output.json"
    directory = tmp_path / "subdir"
    directory.mkdir()
    evidence = target if alias == "same" else directory / ".." / target.name
    run = _ProofRun("P4", evaluate_secret_delivery, receipt=str(target), evidence=str(evidence))
    with pytest.raises(SystemExit) as refused:
        run.preflight()
    assert refused.value.code == 3
    assert not target.exists()


def test_receipt_publication_cannot_clobber_file_created_after_preflight(tmp_path):
    from eyvan.sdk.commands.receipts import write_receipt_atomic

    target = tmp_path / "receipt.json"
    run = _ProofRun("P4", evaluate_secret_delivery, receipt=str(target))
    run.preflight()
    sentinel = b"another run's immutable receipt\n"
    target.write_bytes(sentinel)
    with pytest.raises(FileExistsError):
        write_receipt_atomic(target, {"schema": "new-receipt"}, replace_existing=False)
    assert target.read_bytes() == sentinel


@pytest.mark.parametrize("current_code", ["provider_error", None, "preempted"])
def test_unrelated_inventory_cannot_vouch_for_workload_preemption(current_code):
    from eyvan.sdk.commands.proof import evaluate_spot_preemption

    workload = {
        "id": "wl-proof",
        "preemption_count": 1,
        "capacity_loss_count": 0,
        "state": "running",
        "created_at": "2026-10-01T12:00:00+00:00",
        "spec": {"required_labels": {"eyvan_pool": "current"}},
    }
    current = {
        "id": "current-cell",
        "pool_id": "current",
        "provider_failure_code": current_code,
        "updated_at": "2026-10-01T12:01:00+00:00",
        "availability": {"interruptions": 1, "cooldown_until": None},
    }
    unrelated = {
        "id": "old-cell",
        "pool_id": "old",
        "provider_failure_code": "preempted",
        "updated_at": "2026-10-01T11:00:00+00:00",
        "availability": {"interruptions": 1, "cooldown_until": None},
    }
    pools = [{"id": "current"}, {"id": "old"}]
    without = evaluate_spot_preemption(workload, [current], pools)
    with_old = evaluate_spot_preemption(workload, [current, unrelated], pools)
    # Inventory fields and pool/time coincidence cannot link a provider event to
    # this workload's reclamation; counters alone establish only budget charging.
    for name in ("classified-as-preempted", "no-stockout-cooldown"):
        assert _check(without, name) == _check(with_old, name)
        assert _check(with_old, name).verdict is Verdict.INCONCLUSIVE
    assert with_old.verdict is Verdict.INCONCLUSIVE


def test_unlinked_inventory_does_not_hide_wrong_workload_budget():
    from eyvan.sdk.commands.proof import evaluate_spot_preemption

    result = evaluate_spot_preemption({"preemption_count": 0, "capacity_loss_count": 1, "state": "failed"}, [], [])
    assert result.verdict is Verdict.FAIL


@pytest.mark.parametrize("case, exit_code", [("leak", 1), ("missing-read", 1), ("unfinished", 2), ("complete", 0)])
def test_secret_cli_exit_tracks_decoded_leaks_and_mount_delivery(tmp_path, case, exit_code):
    from types import SimpleNamespace

    from click.testing import CliRunner

    from eyvan.sdk.commands.proof import proof

    value = "first\\second"
    value_file = tmp_path / "synthetic-value.txt"
    value_file.write_text(value)
    events = [_issued(), _read("A"), _read("B")]
    workload = {"state": "succeeded", "spec": {"secret_mounts": {"A": "file:///tmp/a", "B": "file:///tmp/b"}}}
    if case == "leak":
        workload["spec"]["environment"] = {"LEAK_COPY": value}
    elif case in {"missing-read", "unfinished"}:
        events.append(_issued(attempt=2, generation=2))
        workload["state"] = "running" if case == "unfinished" else "failed"

    class Client:
        def request(self, method, path, **kwargs):
            if path == "/api/v1/events":
                return events
            assert path == "/api/v1/workloads/wl-proof"
            return workload

        def request_text(self, method, path):
            assert path == "/api/v1/workloads/wl-proof/logs"
            return ""

    session = SimpleNamespace(
        client=Client(), workload_scope=lambda **kwargs: SimpleNamespace(org_id="o", project_id="p")
    )
    result = CliRunner().invoke(proof, ["secret", "wl-proof", "--value-file", str(value_file)], obj=session)
    assert result.exit_code == exit_code, result.output
    assert json.loads(result.output)["verdict"] == {0: "pass", 1: "fail", 2: "inconclusive"}[exit_code]
    assert value not in result.output
