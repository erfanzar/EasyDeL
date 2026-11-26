# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a hardware spend leaves behind, and what it is allowed to claim.

``docs/proofs.md`` records outcomes in prose, and prose cannot name the build a
run judged: every component of this system reports the same version string. So
"P5 passed" has been a claim about a candidate nobody could identify afterwards,
which is why that page says no historical run qualifies the current one.

These tests hold the properties that make the attribution executable -- the
receipt names the criterion and the builds, states what the run did *not*
establish, and refuses to qualify a build it was not minted against -- and the
one that keeps it honest: a run that was refused before it started never wears a
verdict's exit code.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from eyvan._internal import build_identity as identity_module
from eyvan._internal.build_identity import BuildIdentity
from eyvan.sdk.commands import receipts as receipts_module
from eyvan.sdk.commands.proof import (
    EXIT_CODES,
    EXIT_NOT_RUN,
    Check,
    ProofResult,
    Verdict,
    evaluate_receipt,
    evaluate_spot_preemption,
    proof,
)
from eyvan.sdk.commands.receipts import (
    RECEIPT_SCHEMA,
    build_proof_receipt,
    canonical_digest,
    criterion_identity,
    file_evidence,
    read_receipt,
    receipt_digest_matches,
    redact_command,
    unverified_dimensions,
)


@pytest.fixture(autouse=True)
def _isolated_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Receipt fixtures identify one candidate without depending on package scanning."""

    candidate = BuildIdentity(
        version="0.0.1",
        origin="source",
        origin_detail="test fixture",
        source_digest="sha256:plane",
        digest_detail="",
        file_count=1,
        python="3.11",
        platform="test",
    )
    monkeypatch.setattr(identity_module, "build_identity", lambda: candidate)
    monkeypatch.setattr(receipts_module, "build_identity", lambda: candidate)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "artifacts"))


def _receipt(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "proof": "P5",
        "verdict": "pass",
        "exit_status": 0,
        "checks": [{"name": "classified-as-preemption", "verdict": "pass", "detail": "preemption_count 1"}],
        "criterion": criterion_identity(evaluate_spot_preemption),
        "plane_url": "http://10.128.0.7:8080",
        "plane_build": {"version": "0.0.1", "source_digest": "sha256:plane"},
        "plane_build_detail": "",
        "fleet_builds": {
            "plane": {"source_digest": "sha256:plane"},
            "agents": [{"endpoint": "http://agent-a", "build": {"source_digest": "sha256:plane"}}],
            "unreachable": [],
            "unscanned_pools": [],
            "in_agreement": True,
        },
        "started_at": "2026-08-11T00:00:00+00:00",
        "completed_at": "2026-08-11T00:01:00+00:00",
        "command": ["eyvan", "proof", "preemption", "wl-1"],
        "git": {"commit": "a" * 40, "clean": True, "detail": ""},
    }
    fields.update(overrides)
    return build_proof_receipt(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the criterion


def test_the_criterion_is_identified_by_its_own_source() -> None:
    """A proof rerun after its evaluator was loosened is not the same proof."""

    identity = criterion_identity(evaluate_spot_preemption)

    assert identity["function"].endswith("evaluate_spot_preemption")
    assert str(identity["source_digest"]).startswith("sha256:")


def test_two_different_criteria_do_not_share_a_digest() -> None:
    from eyvan.sdk.commands.proof import evaluate_structured_logs

    assert (
        criterion_identity(evaluate_spot_preemption)["source_digest"]
        != (criterion_identity(evaluate_structured_logs)["source_digest"])
    )


def test_an_unreadable_criterion_reports_no_digest() -> None:
    """A built-in has no source; inventing one would attribute a verdict to it."""

    identity = criterion_identity(len)

    assert identity["source_digest"] is None
    assert identity["source_detail"]


# --------------------------------------------------------------------------- what a receipt says


def test_a_receipt_names_the_criterion_the_builds_and_the_run() -> None:
    document = _receipt()

    assert document["schema"] == RECEIPT_SCHEMA
    assert document["proof"] == "P5"
    assert document["verdict"] == "pass"
    assert str(document["criterion"]["source_digest"]).startswith("sha256:")
    assert document["candidate"]["build"]["source_digest"]
    assert document["plane"]["build"]["source_digest"] == "sha256:plane"
    assert document["run"]["started_at"] < document["run"]["completed_at"]


def test_the_digest_covers_every_other_field() -> None:
    """An edited verdict must stop verifying."""

    document = _receipt()
    assert receipt_digest_matches(document) is True

    tampered = dict(document)
    tampered["verdict"] = "pass"
    tampered["checks"] = [{"name": "invented", "verdict": "pass", "detail": "nothing observed this"}]

    assert receipt_digest_matches(tampered) is False


def test_a_receipt_without_a_digest_is_unknown_rather_than_forged() -> None:
    """Old receipts predate the field; calling them tampered helps nobody."""

    document = {key: value for key, value in _receipt().items() if key != "receipt_digest"}

    assert receipt_digest_matches(document) is None


def test_the_digest_is_independent_of_key_order() -> None:
    document = _receipt()
    reordered = dict(reversed(list(document.items())))

    assert canonical_digest({k: v for k, v in document.items() if k != "receipt_digest"}) == canonical_digest(
        {k: v for k, v in reordered.items() if k != "receipt_digest"}
    )


# --------------------------------------------------------------------------- what it does not claim


def test_a_plane_that_reported_no_build_is_named_as_unestablished() -> None:
    """The dimension nobody writes down at the end of a spend."""

    gaps = unverified_dimensions(plane_build=None, git={"commit": "a", "clean": True}, evidence={"path": "x"})

    assert any("control plane" in gap for gap in gaps)


def test_the_agent_and_worker_builds_are_always_unestablished() -> None:
    """A proof was once run against a stale agent and read as evidence.

    The client and the plane identify themselves; the thing that actually ran
    the workload does not, and a receipt that stayed silent about that would be
    read as covering it.
    """

    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a", "clean": True},
        evidence={"path": "x"},
    )

    assert any("cell agent" in gap and "worker" in gap for gap in gaps)


def test_a_dirty_tree_is_recorded_as_not_a_released_candidate() -> None:
    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a" * 40, "clean": False},
        evidence=None,
    )

    assert any("dirty" in gap for gap in gaps)
    assert any("raw observations" in gap for gap in gaps)


def test_a_receipt_never_implies_it_is_signed() -> None:
    document = _receipt()

    assert document["signature"] is None
    assert any("unsigned" in gap for gap in document["unverified"])


def test_a_fleet_that_agreed_leaves_no_build_dimension_open() -> None:
    """The dimension no receipt could close, closed.

    Agreement covers the whole chain: the client that submitted, the plane that
    scheduled, the agent that held the capacity, and the worker process that ran
    the callable. Nothing about builds is left for the reader to assume.
    """

    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a" * 40, "clean": True},
        evidence={"path": "x"},
        fleet_builds={"agents": [{"endpoint": "http://agent-a"}], "unreachable": [], "in_agreement": True},
    )

    assert not any("cell agent" in gap for gap in gaps)
    assert not any("worker" in gap for gap in gaps)


def test_an_unreachable_agent_leaves_the_gap_open() -> None:
    """A sweep that could not reach an agent has not established anything about it."""

    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a" * 40, "clean": True},
        evidence={"path": "x"},
        fleet_builds={
            "agents": [{"endpoint": "http://agent-a"}],
            "unreachable": [{"endpoint": "http://agent-b"}],
            "in_agreement": None,
        },
    )

    assert any("could not be reached" in gap for gap in gaps)


def test_a_run_against_a_stale_agent_says_so_rather_than_narrowing_anything() -> None:
    """The exact history this whole mechanism exists for.

    A proof was run against a stale agent and read as evidence. A receipt from
    such a run must not quietly report a narrower gap than one that could not
    check at all -- it has a positive finding, and it is bad news.
    """

    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a" * 40, "clean": True},
        evidence={"path": "x"},
        fleet_builds={"agents": [{"endpoint": "http://agent-a"}], "unreachable": [], "in_agreement": False},
    )

    assert any("different code than the control plane" in gap for gap in gaps)


def test_a_plane_that_cannot_sweep_its_agents_keeps_the_original_gap() -> None:
    """An older plane has no such route; the receipt reads as it always did."""

    gaps = unverified_dimensions(
        plane_build={"source_digest": "sha256:plane"},
        git={"commit": "a" * 40, "clean": True},
        evidence={"path": "x"},
        fleet_builds=None,
    )

    assert any("cell agent" in gap and "only the client" in gap for gap in gaps)


# --------------------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["eyvan", "--token", "secret-value", "proof"], ["eyvan", "--token", "<redacted>", "proof"]),
        (["eyvan", "--token=secret-value"], ["eyvan", "--token=<redacted>"]),
        (
            ["eyvan", "proof", "secret", "--value-file", "/tmp/v"],
            ["eyvan", "proof", "secret", "--value-file", "<redacted>"],
        ),
        (["eyvan", "--url", "http://host:8080"], ["eyvan", "--url", "http://host:8080"]),
    ],
)
def test_credential_arguments_never_reach_the_receipt(argv: list[str], expected: list[str]) -> None:
    """Both spellings, because redacting one and not the other reads as safe."""

    assert redact_command(argv) == expected


def test_a_trailing_credential_option_with_no_value_is_not_invented() -> None:
    assert redact_command(["eyvan", "--token"]) == ["eyvan", "--token"]


# --------------------------------------------------------------------------- evidence


def test_evidence_is_named_by_digest_and_size(tmp_path: Path) -> None:
    """The receipt points at evidence rather than embedding it.

    Raw plane documents carry fields this module cannot exhaustively redact, and
    a receipt is meant to be readable by people not entitled to all of them.
    """

    path = tmp_path / "evidence.json"
    path.write_text('{"cells": []}\n', encoding="utf-8")

    described = file_evidence(path)

    assert described["size_bytes"] == len('{"cells": []}\n')
    assert str(described["sha256"]).startswith("sha256:")
    assert described["media_type"] == "application/json"


# --------------------------------------------------------------------------- reading one back


def _verify_result(document, **kwargs) -> ProofResult:
    criterion = kwargs.pop("criterion", criterion_identity(evaluate_spot_preemption))
    build_now = kwargs.pop("build_now", None)
    if build_now is None:
        from eyvan._internal.build_identity import build_identity

        build_now = build_identity().as_dict()
    return evaluate_receipt(
        document,
        digest_intact=kwargs.pop("intact", True),
        criterion_now=criterion,
        build_now=build_now,
    )


def _check(result: ProofResult, name: str) -> Verdict:
    return next(check.verdict for check in result.checks if check.name == name)


def test_a_receipt_qualifies_the_build_it_was_minted_against() -> None:
    result = _verify_result(_receipt())

    assert result.verdict is Verdict.PASS
    assert _check(result, "candidate-current") is Verdict.PASS


def test_a_receipt_does_not_qualify_a_different_build() -> None:
    """The sentence ``docs/proofs.md`` states, as a decision."""

    result = _verify_result(_receipt(), build_now={"source_digest": "sha256:some-other-build"})

    assert result.verdict is Verdict.FAIL
    assert _check(result, "candidate-current") is Verdict.FAIL


def test_a_changed_criterion_disqualifies_a_passing_receipt() -> None:
    """The verdict was decided by code this build no longer runs."""

    result = _verify_result(_receipt(), criterion={"source_digest": "sha256:loosened", "function": "x"})

    assert _check(result, "criterion-current") is Verdict.FAIL
    assert result.verdict is Verdict.FAIL


def test_an_inconclusive_run_qualifies_nothing() -> None:
    """Verification never upgrades an outcome."""

    result = _verify_result(_receipt(verdict="inconclusive", exit_status=2))

    assert _check(result, "verdict") is Verdict.FAIL


def test_a_tampered_receipt_is_refused() -> None:
    result = _verify_result(_receipt(), intact=False)

    assert _check(result, "receipt-intact") is Verdict.FAIL
    assert result.verdict is Verdict.FAIL


def test_an_undigested_receipt_is_unknown_not_passing() -> None:
    result = _verify_result(_receipt(), intact=None)

    assert _check(result, "receipt-intact") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


def test_a_proof_this_build_cannot_evaluate_is_unknown() -> None:
    result = _verify_result(_receipt(proof="P9"), criterion=None)

    assert _check(result, "criterion-current") is Verdict.INCONCLUSIVE


def test_an_unknown_build_digest_compares_to_neither() -> None:
    result = _verify_result(_receipt(), build_now={"source_digest": None})

    assert _check(result, "candidate-current") is Verdict.INCONCLUSIVE
    assert result.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("overrides", [{"plane_build": None}, {"fleet_builds": None}])
def test_a_client_match_cannot_qualify_unknown_backend_builds(overrides: dict[str, object]) -> None:
    result = _verify_result(_receipt(**overrides))

    assert _check(result, "candidate-current") is Verdict.PASS
    assert result.verdict is Verdict.INCONCLUSIVE


# --------------------------------------------------------------------------- the command surface


def test_the_verify_command_reads_a_written_receipt(tmp_path: Path) -> None:
    """End to end: write one, read it back, and have it qualify this build."""

    from eyvan.sdk.commands.receipts import write_receipt_atomic

    path = tmp_path / "p5.json"
    write_receipt_atomic(path, _receipt())

    result = CliRunner().invoke(proof, ["verify", str(path)])

    assert result.exit_code == EXIT_CODES[Verdict.PASS], result.output
    assert json.loads(result.output)["verdict"] == "pass"


def test_an_unreadable_receipt_is_refused_rather_than_failed(tmp_path: Path) -> None:
    """Exit 1 would report a proof that failed; nothing was evaluated."""

    path = tmp_path / "not-a-receipt.json"
    path.write_text('{"schema": "something-else"}\n', encoding="utf-8")

    result = CliRunner().invoke(proof, ["verify", str(path)])

    assert result.exit_code == EXIT_NOT_RUN


def test_a_mistyped_option_does_not_wear_an_inconclusive_verdict() -> None:
    """Click answers a usage error with 2, which is this group's INCONCLUSIVE.

    A scheduled run would have recorded "the behaviour was not exercised, repeat
    the spend" for a command that can never work as typed.
    """

    result = CliRunner().invoke(proof, ["verify", "--nonexistent-option"])

    assert result.exit_code == EXIT_NOT_RUN


def test_a_receipt_path_that_exists_stops_the_run_before_it_spends(tmp_path: Path) -> None:
    """Checked before the plane is polled: a run that cannot record its result
    is stopped while it is still free."""

    existing = tmp_path / "taken.json"
    existing.write_text("{}\n", encoding="utf-8")

    result = CliRunner().invoke(
        proof,
        ["catalog", "--accelerator-type", "v5p-16", "--receipt", str(existing)],
        obj=object(),
    )

    assert result.exit_code == EXIT_NOT_RUN
    assert existing.read_text(encoding="utf-8") == "{}\n"


def test_an_output_directory_that_does_not_exist_is_refused_not_failed(tmp_path: Path) -> None:
    """Report a missing evidence directory as not-run rather than as a failed proof criterion."""

    result = CliRunner().invoke(
        proof,
        [
            "catalog",
            "--accelerator-type",
            "v5p-16",
            "--evidence",
            str(tmp_path / "absent" / "evidence.json"),
        ],
        obj=object(),
    )

    assert result.exit_code == EXIT_NOT_RUN


def test_reading_a_receipt_rejects_an_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema": "eyvan.proof-receipt/99"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown receipt schema"):
        read_receipt(path)


def test_a_verdict_algebra_receipt_reports_every_check() -> None:
    """The composed result keeps each question separate, not just the worst."""

    result = evaluate_receipt(
        _receipt(),
        digest_intact=True,
        criterion_now={"source_digest": "sha256:different"},
        build_now={"source_digest": "sha256:different-too"},
    )

    assert {check.name for check in result.checks} == {
        "receipt-intact",
        "verdict",
        "criterion-current",
        "candidate-current",
        "plane-current",
        "fleet-current",
    }
    assert isinstance(result.checks[0], Check)


# --------------------------------------------------------------------------- the wiring


class _StubPlane:
    """Answer a proof's routes, and record what it was asked for."""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def request(self, method: str, path: str, **kwargs: object) -> object:
        del method, kwargs
        self.calls.append(path)
        if path not in self.responses:
            raise click.ClickException(f"404 for {path}")
        return self.responses[path]

    def request_text(self, method: str, path: str, **kwargs: object) -> str:
        """The plain-text sibling, kept distinct the way the real client is.

        ``/logs`` is declared ``PlainTextResponse``, so a caller that decodes it
        as JSON fails against the real plane. A stub whose two accessors behaved
        identically would let that mistake pass here.
        """

        body = self.request(method, path, **kwargs)
        if not isinstance(body, str):
            raise AssertionError(f"{path} is not a plain-text route")
        return body

    def replace_identity_headers(self, headers: object) -> None:
        del headers


def _run_proof(monkeypatch, plane: _StubPlane, argv: list[str]):
    from eyvan.sdk import cli

    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: plane)
    return CliRunner().invoke(cli.main, ["--user", "alice", "--org", "acme", "--project", "research", *argv])


def _preempted_plane(**extra: object) -> _StubPlane:
    responses: dict[str, object] = {
        "/api/v1/workloads/wl-1": {
            "id": "wl-1",
            "state": "running",
            "preemption_count": 1,
            "capacity_loss_count": 0,
            "capacity_pool_id": "v5p-16",
            "attempt": 2,
        },
        "/api/v1/cells": [
            {"id": "cell-a", "capacity_pool_id": "v5p-16", "provider_failure_code": "preempted"},
        ],
        "/api/v1/capacity-pools": [{"id": "v5p-16", "stockout_cooldown_until": None}],
    }
    responses.update(extra)
    return _StubPlane(responses)


def test_a_proof_writes_a_receipt_naming_the_plane_it_polled(tmp_path: Path, monkeypatch) -> None:
    """The whole point of the option: the run becomes attributable afterwards."""

    receipt = tmp_path / "p5.json"
    plane = _preempted_plane(
        **{"/api/v1/build": {"version": "0.0.1", "source_digest": "sha256:the-plane", "origin": "wheel"}}
    )

    result = _run_proof(monkeypatch, plane, ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    assert result.exit_code in set(EXIT_CODES.values()), result.output
    document = read_receipt(receipt)
    assert document["proof"] == "P5"
    assert document["plane"]["build"]["source_digest"] == "sha256:the-plane"
    assert document["criterion"]["source_digest"] == criterion_identity(evaluate_spot_preemption)["source_digest"]
    assert receipt_digest_matches(document) is True


def test_a_plane_that_predates_the_build_route_is_recorded_as_unknown(tmp_path: Path, monkeypatch) -> None:
    """A 404 is not an error here: the proof still ran, and the receipt says so."""

    receipt = tmp_path / "p5.json"

    _run_proof(monkeypatch, _preempted_plane(), ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    document = read_receipt(receipt)
    assert document["plane"]["build"] is None
    assert document["plane"]["build_detail"]
    assert any("control plane" in gap for gap in document["unverified"])


def test_a_receipt_is_written_for_an_inconclusive_run_too(tmp_path: Path, monkeypatch) -> None:
    """An inconclusive spend is the one most worth being able to point at later."""

    receipt = tmp_path / "p5.json"
    plane = _StubPlane(
        {
            "/api/v1/workloads/wl-1": {
                "id": "wl-1",
                "state": "running",
                "preemption_count": 0,
                "capacity_loss_count": 0,
                "capacity_pool_id": "v5p-16",
            },
            "/api/v1/cells": [],
            "/api/v1/capacity-pools": [{"id": "v5p-16", "stockout_cooldown_until": None}],
        }
    )

    result = _run_proof(monkeypatch, plane, ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    assert result.exit_code == EXIT_CODES[Verdict.INCONCLUSIVE], result.output
    assert read_receipt(receipt)["verdict"] == "inconclusive"


def test_the_receipt_is_owner_only(tmp_path: Path, monkeypatch) -> None:
    receipt = tmp_path / "p5.json"

    _run_proof(monkeypatch, _preempted_plane(), ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    assert receipt.stat().st_mode & 0o077 == 0


def test_evidence_records_the_documents_the_verdict_was_read_from(tmp_path: Path, monkeypatch) -> None:
    evidence = tmp_path / "evidence.json"
    receipt = tmp_path / "p5.json"

    _run_proof(
        monkeypatch,
        _preempted_plane(),
        ["proof", "preemption", "wl-1", "--receipt", str(receipt), "--evidence", str(evidence)],
    )

    captured = json.loads(evidence.read_text(encoding="utf-8"))
    assert captured["workload"]["preemption_count"] == 1
    assert captured["cells"][0]["provider_failure_code"] == "preempted"
    described = read_receipt(receipt)["evidence"]
    assert described["size_bytes"] == evidence.stat().st_size
    assert evidence.stat().st_mode & 0o077 == 0


def test_evidence_is_written_without_a_receipt(tmp_path: Path, monkeypatch) -> None:
    """The two options are independent; asking for one must not require the other."""

    evidence = tmp_path / "evidence.json"

    _run_proof(monkeypatch, _preempted_plane(), ["proof", "preemption", "wl-1", "--evidence", str(evidence)])

    assert json.loads(evidence.read_text(encoding="utf-8"))["workload"]["id"] == "wl-1"


def test_a_leaked_secret_is_not_copied_into_the_evidence_file(tmp_path: Path, monkeypatch) -> None:
    """The run that exists to find a leak must not widen it to report one.

    The verdict records that the value appeared in the logs. Writing the value
    into a second durable file to prove that would be the same defect the proof
    is looking for.
    """

    value_file = tmp_path / "expected"
    value_file.write_text("super-secret-token\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    plane = _StubPlane(
        {
            "/api/v1/events": [
                {"action": "secret.issued", "payload": {"attempt": 1}},
                {"action": "secret.read", "payload": {"attempt": 1}},
            ],
            "/api/v1/workloads/wl-1": {"id": "wl-1", "state": "succeeded"},
            "/api/v1/workloads/wl-1/logs": "the token is super-secret-token\n",
        }
    )

    result = _run_proof(
        monkeypatch,
        plane,
        [
            "proof",
            "secret",
            "wl-1",
            "--value-file",
            str(value_file),
            "--evidence",
            str(evidence),
        ],
    )

    assert result.exit_code == EXIT_CODES[Verdict.FAIL], result.output
    assert "super-secret-token" not in evidence.read_text(encoding="utf-8")


def test_a_receipt_records_the_fleet_sweep_the_plane_could_answer(tmp_path: Path, monkeypatch) -> None:
    """End to end: the run asks, and what it learns lands in the receipt.

    Without this the gap-narrowing above is a pure function nobody calls.
    """

    receipt = tmp_path / "p5.json"
    plane = _preempted_plane(
        **{
            "/api/v1/build": {"version": "0.0.1", "source_digest": "sha256:the-plane", "origin": "wheel"},
            "/api/v1/capacity/builds": {
                "plane": {"source_digest": "sha256:the-plane"},
                "agents": [{"endpoint": "http://agent-a:8090", "build": {"source_digest": "sha256:the-plane"}}],
                "unreachable": [],
                "unscanned_pools": [],
                "in_agreement": True,
            },
        }
    )

    _run_proof(monkeypatch, plane, ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    document = read_receipt(receipt)
    assert document["fleet"]["in_agreement"] is True
    assert not any("worker" in gap for gap in document["unverified"])
    assert receipt_digest_matches(document) is True


def test_a_plane_without_the_sweep_route_still_writes_a_receipt(tmp_path: Path, monkeypatch) -> None:
    """A 404 here must cost the receipt one dimension, not the whole receipt."""

    receipt = tmp_path / "p5.json"

    _run_proof(monkeypatch, _preempted_plane(), ["proof", "preemption", "wl-1", "--receipt", str(receipt)])

    document = read_receipt(receipt)
    assert document["fleet"] is None
    assert any("cell agent" in gap and "only the client" in gap for gap in document["unverified"])
