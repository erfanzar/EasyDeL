# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reject malformed, oversized or inconsistent local release artifacts."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from test_oci_release import (
    LOCK,
    SCRIPT,
    _canonical,
    _gate_scan,
    _high_finding,
    _load_script,
    _receipt_fixture,
    _run,
    _scan_report,
    _sha,
)


def _receipt_commands(tmp_path: Path, receipt: dict, root: Path) -> tuple[list, list]:
    source = tmp_path / "input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    emit = ["emit-receipt", "--lock", LOCK, "--input", source, "--output", output, "--artifact-root", root]
    verify = [
        "verify-receipt",
        "--lock",
        LOCK,
        "--receipt",
        output,
        "--artifact-root",
        root,
        "--expected-version",
        "0.1.0",
    ]
    return emit, verify


def test_realistic_sbom_receipt_emission_and_replay(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    image = receipt["images"][0]
    document_record = image["sbom"]["document"]
    document_path = root / document_record["name"]
    document = json.loads(document_path.read_text())
    document["packages"] = [
        {"SPDXID": f"SPDXRef-Package-{index}", "name": f"library-{index}", "versionInfo": "1.2.3"}
        for index in range(150)
    ]
    _canonical(document_path, document)
    document_record["sha256"] = _sha(document_path)
    bundle_record = image["sbom"]["attestation"]["bundle"]
    bundle_path = root / bundle_record["name"]
    bundle = json.loads(bundle_path.read_text())
    statement = json.loads(base64.b64decode(bundle["dsseEnvelope"]["payload"]))
    statement["predicate"] = document
    payload = base64.b64encode(json.dumps(statement).encode()).decode()
    assert len(payload) > 4096
    bundle["dsseEnvelope"]["payload"] = payload
    _canonical(bundle_path, bundle)
    bundle_record["sha256"] = _sha(bundle_path)
    emit, verify = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 0, result.stderr
    result = _run(*verify)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("payload", ["!invalid!", base64.b64encode(b"{}").decode()])
def test_sbom_payload_still_rejects_malformed_input(tmp_path: Path, payload: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    bundle_record = receipt["images"][0]["sbom"]["attestation"]["bundle"]
    bundle_path = root / bundle_record["name"]
    bundle = json.loads(bundle_path.read_text())
    bundle["dsseEnvelope"]["payload"] = payload
    _canonical(bundle_path, bundle)
    bundle_record["sha256"] = _sha(bundle_path)
    emit, _ = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 1
    assert "DSSE payload" in result.stderr or "in-toto statement" in result.stderr


def test_bundle_file_size_is_bounded_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script(SCRIPT, "writer121_bundle_limit")
    monkeypatch.setattr(module, "MAX_ARTIFACT_BYTES", 32)
    bundle = tmp_path / "large.bundle.json"
    bundle.write_bytes(b" " * 33)
    with pytest.raises(module.VerificationError, match="byte limit"):
        module._attestation_predicate_from_bundle(bundle, {}, predicate_type="unused", context="bounded bundle")


def _render_check(tmp_path: Path, receipt: dict, root: Path, text: str):
    record = receipt["deployment"]["render"]
    path = root / record["name"]
    path.write_text(text)
    record["sha256"] = _sha(path)
    arguments = ["verify-rendered-digest-refs", "--render", path]
    for item in receipt["deployment"]["references"]:
        arguments.extend(["--reference", f"{item['role']}={item['reference']}"])
    return _run(*arguments)


@pytest.mark.parametrize("field", ["containers", "initContainers", "ephemeralContainers"])
def test_flow_container_cannot_evade_render_or_receipt_gate(tmp_path: Path, field: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    path = root / receipt["deployment"]["render"]["name"]
    text = path.read_text() + (
        f"\n---\napiVersion: v1\nkind: Pod\nspec:\n  {field}: [{{name: rogue, image: registry.example/rogue:latest}}]\n"
    )
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 1, result.stdout
    emit, verify = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 1, result.stdout
    _canonical(tmp_path / "receipt.json", receipt)
    result = _run(*verify)
    assert result.returncode == 1, result.stdout


def test_block_scalar_data_is_not_deployment_image_evidence(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    text = "apiVersion: v1\nkind: ConfigMap\ndata:\n  example: |\n"
    for item in receipt["deployment"]["references"]:
        text += f"    image: {item['reference']}\n"
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 1, result.stdout


def test_native_worker_image_env_rejects_foreign_mutable_repository(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    path = root / receipt["deployment"]["render"]["name"]
    text = path.read_text() + (
        "      env:\n        - name: EYVAN_KUBERNETES_WORKER_IMAGE\n          value: registry.example/rogue:latest\n"
    )
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 1, result.stdout


def test_block_render_native_worker_env_and_data_control(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    refs = receipt["deployment"]["references"]
    text = "apiVersion: apps/v1\nkind: Deployment\nspec:\n  template:\n    spec:\n      containers:\n"
    for item in refs[:2]:
        text += f"        - name: {item['role']}\n          image: {item['reference']}\n"
    text += (
        '          command: ["eyvan-agent"]\n          env:\n'
        "            - name: EYVAN_KUBERNETES_WORKER_IMAGE\n"
        f"              value: {refs[2]['reference']}\n"
        "---\napiVersion: v1\nkind: ConfigMap\ndata:\n  notes: |\n"
        "    image: registry.example/data-not-a-container:latest\n"
    )
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "malformed", ["empty", "missing-target", "missing-class", "missing-type", "wrong-class", "severity"]
)
def test_malformed_scan_is_rejected_before_passing_disposition(tmp_path: Path, malformed: str) -> None:
    report = _scan_report()
    target = report["Results"][0]
    target.update({"Class": "os-pkgs", "Type": "alpine"})
    if malformed == "empty":
        report["Results"] = [{}]
    elif malformed == "missing-target":
        del target["Target"]
    elif malformed == "missing-class":
        del target["Class"]
    elif malformed == "missing-type":
        del target["Type"]
    elif malformed == "wrong-class":
        target["Class"] = "config"
    else:
        finding = _high_finding()
        finding["Severity"] = "CRITCAL"
        target["Vulnerabilities"] = [finding]
    result, disposition = _gate_scan(tmp_path, report=report)
    assert result.returncode == 1, result.stdout
    assert not disposition.exists()


@pytest.mark.parametrize("severity", [None, "UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
def test_supported_clean_and_severity_scan_controls(tmp_path: Path, severity: str | None) -> None:
    report = _scan_report()
    target = report["Results"][0]
    target.update({"Class": "os-pkgs", "Type": "alpine"})
    if severity is None:
        del target["Vulnerabilities"]
    else:
        finding = _high_finding()
        finding["Severity"] = severity
        target["Vulnerabilities"] = [finding]
    result, disposition = _gate_scan(tmp_path, report=report)
    assert result.returncode == (1 if severity in {"HIGH", "CRITICAL"} else 0), result.stderr
    assert json.loads(disposition.read_text())["status"] == ("fail" if result.returncode else "pass")


@pytest.mark.parametrize("malformed", ["target", "severity"])
def test_receipt_replay_rejects_malformed_scan_with_rebound_hashes(tmp_path: Path, malformed: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    scan = receipt["images"][0]["scan"]
    raw_path = root / scan["raw"]["name"]
    report = json.loads(raw_path.read_text())
    if malformed == "target":
        report["Results"] = [{}]
    else:
        finding = _high_finding()
        finding["Severity"] = "CRITCAL"
        report["Results"][0]["Vulnerabilities"] = [finding]
    _canonical(raw_path, report)
    scan["raw"]["sha256"] = _sha(raw_path)
    disposition_path = root / scan["disposition"]["name"]
    disposition = json.loads(disposition_path.read_text())
    disposition["rawScanSha256"] = scan["raw"]["sha256"]
    _canonical(disposition_path, disposition)
    scan["disposition"]["sha256"] = _sha(disposition_path)
    emit, verify = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 1, result.stdout
    _canonical(tmp_path / "receipt.json", receipt)
    result = _run(*verify)
    assert result.returncode == 1, result.stdout


@pytest.mark.parametrize("wheel_name", ["eyvan-0.1.01-py3-none-any.whl", "eyvan-0.1.0.post1-py3-none-any.whl"])
def test_wheel_complete_version_mismatch_rejected_at_emit_and_replay(tmp_path: Path, wheel_name: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    wheel = receipt["packages"]["wheel"]
    (root / wheel_name).write_bytes((root / wheel["name"]).read_bytes())
    wheel["name"] = wheel_name
    emit, verify = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 1, result.stdout
    assert "wheel filename" in result.stderr
    _canonical(tmp_path / "receipt.json", receipt)
    result = _run(*verify)
    assert result.returncode == 1, result.stdout
    assert "wheel filename" in result.stderr


@pytest.mark.parametrize("wheel_name", ["eyvan-0.1.0-2build-py3-none-any.whl", "eyvan-0.1.0-py2.py3-none-any.whl"])
def test_wheel_build_and_compatibility_tags_still_accepted(tmp_path: Path, wheel_name: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    wheel = receipt["packages"]["wheel"]
    (root / wheel_name).write_bytes((root / wheel["name"]).read_bytes())
    wheel["name"] = wheel_name
    emit, verify = _receipt_commands(tmp_path, receipt, root)
    result = _run(*emit)
    assert result.returncode == 0, result.stderr
    result = _run(*verify)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("prefix", ["data:\n  example:\n    spec:\n      containers:\n", "spec:\n  containers:\n"])
def test_non_workload_mapping_is_not_deployment_evidence(tmp_path: Path, prefix: str) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    text = f"apiVersion: v1\nkind: ConfigMap\n{prefix}"
    last_line = prefix.splitlines()[-1]
    indent = " " * (len(last_line) - len(last_line.lstrip()) + 2)
    for item in receipt["deployment"]["references"]:
        text += f"{indent}- image: {item['reference']}\n"
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 1, result.stdout


def test_duplicate_yaml_mapping_cannot_publish_phantom_image_evidence(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    path = root / receipt["deployment"]["render"]["name"]
    text = path.read_text() + "spec:\n  containers: []\n"
    result = _render_check(tmp_path, receipt, root, text)
    assert result.returncode == 1, result.stdout
    assert "duplicate" in result.stderr
