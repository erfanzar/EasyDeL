# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adversarial fixtures for the immutable Eyvan OCI release gate.

The standalone repository is its own policy owner: the release workflow, the
lock, the verifier, and the Dockerfiles all live at this repository root, so
every contract here is checked against this checkout unconditionally.
"""

from __future__ import annotations

import base64
import email.message
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_eyvan_oci_release.py"
LOCK = ROOT / "oci" / "release-lock.json"
ALLOWLIST = ROOT / "oci" / "high-critical-allowlist.json"
TOOL_INSTALLER = ROOT / "oci" / "install_locked_tools.py"
ASSET_DOWNLOADER = ROOT / "oci" / "download_release_assets.py"
NOW = "2026-08-08T01:00:00Z"
ROLES = ("control-plane", "agent", "worker")
FULL_SHA = "a" * 40


def _canonical(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _policy(*, lock: Path = LOCK, allowlist: Path = ALLOWLIST, root: bool = False) -> subprocess.CompletedProcess[str]:
    arguments: list[str | Path] = [
        "validate-policy",
        "--lock",
        lock,
        "--allowlist",
        allowlist,
        "--now",
        NOW,
    ]
    if root:
        arguments.extend(("--repository-root", ROOT))
    return _run(*arguments)


def _exception(*, expires_at: str = "2026-08-20T00:00:00Z") -> dict[str, str]:
    return {
        "expiresAt": expires_at,
        "issue": "https://github.com/erfanzar/eyvan/issues/1234",
        "owner": "@eyvan-security-owner",
        "package": "openssl",
        "purl": "pkg:apk/alpine/openssl@3.5.2-r0",
        "reason": "Upstream patch is pending; network exposure is independently fenced.",
        "role": "control-plane",
        "version": "3.5.2-r0",
        "vulnerability": "CVE-2026-12345",
    }


def _scanner_metadata() -> dict[str, object]:
    lock = json.loads(LOCK.read_text())
    return {
        "artifactSha256": lock["tools"]["trivy"]["sha256"],
        "binarySha256": "b" * 64,
        "generatedAt": NOW,
        "name": "trivy",
        "schemaVersion": "eyvan.oci.scanner-metadata.v1",
        "version": lock["tools"]["trivy"]["version"],
    }


def _database_metadata() -> dict[str, object]:
    return {
        "digest": f"sha256:{'c' * 64}",
        "downloadedAt": NOW,
        "nextUpdate": "2026-08-08T06:00:00Z",
        "repository": "ghcr.io/aquasecurity/trivy-db:2",
        "schemaVersion": "eyvan.oci.vulnerability-db-metadata.v1",
        "updatedAt": "2026-08-08T00:00:00Z",
        "version": 2,
    }


def _scan_report(*, findings: list[dict[str, object]] | None = None, image: str = "fixture:sha") -> dict[str, object]:
    return {
        "ArtifactName": image,
        "ArtifactType": "container_image",
        "CreatedAt": NOW,
        "Results": [{"Class": "os-pkgs", "Target": "fixture", "Type": "alpine", "Vulnerabilities": findings or []}],
        "SchemaVersion": 2,
    }


def _high_finding(*, fixed_version: str = "") -> dict[str, object]:
    return {
        "FixedVersion": fixed_version,
        "InstalledVersion": "3.5.2-r0",
        "PkgIdentifier": {"PURL": "pkg:apk/alpine/openssl@3.5.2-r0"},
        "PkgName": "openssl",
        "Severity": "HIGH",
        "VulnerabilityID": "CVE-2026-12345",
    }


def _gate_scan(
    tmp_path: Path,
    *,
    report: dict[str, object],
    allowlist: dict[str, object] | None = None,
    scanner_metadata: dict[str, object] | None = None,
    database_metadata: dict[str, object] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    scan_path = tmp_path / "scan.json"
    scanner_path = tmp_path / "scanner.json"
    database_path = tmp_path / "database.json"
    allowlist_path = tmp_path / "allowlist.json"
    output = tmp_path / "disposition.json"
    _canonical(scan_path, report)
    _canonical(scanner_path, scanner_metadata if scanner_metadata is not None else _scanner_metadata())
    _canonical(database_path, database_metadata if database_metadata is not None else _database_metadata())
    _canonical(
        allowlist_path,
        allowlist if allowlist is not None else {"entries": [], "schemaVersion": "eyvan.oci.high-critical-allowlist.v1"},
    )
    result = _run(
        "gate-scan",
        "--lock",
        LOCK,
        "--allowlist",
        allowlist_path,
        "--role",
        "control-plane",
        "--scan",
        scan_path,
        "--scanner-metadata",
        scanner_path,
        "--database-metadata",
        database_path,
        "--image-ref",
        report["ArtifactName"],
        "--output",
        output,
        "--now",
        NOW,
    )
    return result, output


def _sigstore_bundle(log_index: int, statement: dict[str, object] | None = None) -> dict[str, object]:
    bundle: dict[str, object] = {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "verificationMaterial": {
            "tlogEntries": [
                {
                    "inclusionPromise": {"signedEntryTimestamp": "fixture"},
                    "integratedTime": str(1786150800 + log_index),
                    "logId": {"keyId": "3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3d0="},
                    "logIndex": str(log_index),
                }
            ]
        },
    }
    if statement is not None:
        payload = json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
        bundle["dsseEnvelope"] = {
            "payload": base64.b64encode(payload).decode(),
            "payloadType": "application/vnd.in-toto+json",
            "signatures": [{"keyid": "", "sig": base64.b64encode(b"fixture-signature").decode()}],
        }
    return bundle


def _artifact(root: Path, name: str, content: bytes | str | object) -> dict[str, str]:
    path = root / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif isinstance(content, str):
        path.write_text(content)
    else:
        _canonical(path, content)
    return {"name": name, "sha256": _sha(path)}


def _receipt_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    lock = json.loads(LOCK.read_text())
    root = tmp_path / "artifacts"
    root.mkdir()
    lock_artifact = _artifact(root, "release-lock.json", LOCK.read_bytes())
    allowlist_artifact = _artifact(root, "high-critical-allowlist.json", ALLOWLIST.read_bytes())
    images: list[dict[str, object]] = []
    rendered_references: list[str] = []
    for role_index, role in enumerate(ROLES, start=1):
        spec = lock["images"][role]
        index_digest = f"sha256:{str(role_index) * 64}"
        platform_digest = f"sha256:{str(role_index + 3) * 64}"
        image_ref = f"{spec['repository']}@{index_digest}"
        rendered_references.append(image_ref)
        sbom_value = {
            "SPDXID": "SPDXRef-DOCUMENT",
            "creationInfo": {"creators": ["Tool: syft-1.50.0"]},
            "dataLicense": "CC0-1.0",
            "documentNamespace": f"https://example.invalid/spdx/{role}/{FULL_SHA}",
            "name": f"eyvan-{role}",
            "packages": [{"SPDXID": "SPDXRef-Package", "name": "eyvan"}],
            "spdxVersion": "SPDX-2.3",
        }
        sbom = _artifact(root, f"{role}.spdx.json", sbom_value)
        sbom_statement = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": sbom_value,
            "predicateType": "https://spdx.dev/Document",
            "subject": [{"digest": {"sha256": index_digest.removeprefix("sha256:")}, "name": spec["repository"]}],
        }
        sbom_bundle = _artifact(root, f"{role}.sbom.bundle.json", _sigstore_bundle(role_index * 3 + 1, sbom_statement))
        raw_scan = _artifact(root, f"{role}.trivy.json", _scan_report(image=image_ref))
        scanner = _artifact(root, f"{role}.scanner.json", _scanner_metadata())
        database = _artifact(root, f"{role}.database.json", _database_metadata())
        disposition_value = {
            "allowlistSha256": allowlist_artifact["sha256"],
            "counts": {"allowed": 0, "critical": 0, "high": 0, "unmatched": 0},
            "databaseMetadataSha256": database["sha256"],
            "evaluatedAt": NOW,
            "image": image_ref,
            "rawScanSha256": raw_scan["sha256"],
            "role": role,
            "scannerMetadataSha256": scanner["sha256"],
            "schemaVersion": "eyvan.oci.scan-disposition.v1",
            "status": "pass",
        }
        disposition = _artifact(root, f"{role}.scan-disposition.json", disposition_value)
        signature_bundle = _artifact(root, f"{role}.image.bundle.json", _sigstore_bundle(role_index * 3))
        base_references = [
            {"argument": argument, "reference": reference} for argument, reference in sorted(spec["buildArgs"].items())
        ]
        materials = [{"digest": FULL_SHA, "uri": lock["sourceRepository"]}]
        for base in base_references:
            tagged, digest = base["reference"].split("@sha256:", 1)
            materials.append({"digest": digest, "uri": tagged})
        materials.sort(key=lambda item: item["uri"])
        resolved_dependencies = []
        for material in materials:
            algorithm = "sha1" if len(material["digest"]) == 40 else "sha256"
            resolved_dependencies.append({"digest": {algorithm: material["digest"]}, "uri": material["uri"]})
        provenance_value = {
            "buildDefinition": {
                "buildType": "https://github.com/Attestations/GitHubActionsWorkflow@v1",
                "externalParameters": {
                    "role": role,
                    "source": {"digest": {"sha1": FULL_SHA}, "uri": lock["sourceRepository"]},
                },
                "internalParameters": {"githubRunAttempt": 1, "githubRunId": 12345},
                "resolvedDependencies": resolved_dependencies,
            },
            "runDetails": {
                "builder": {"id": lock["identity"]["provenanceBuilderId"]},
                "metadata": {"invocationId": "https://github.com/erfanzar/eyvan/actions/runs/12345"},
            },
        }
        predicate = _artifact(root, f"{role}.provenance.json", provenance_value)
        provenance_statement = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": provenance_value,
            "predicateType": "https://slsa.dev/provenance/v1",
            "subject": [{"digest": {"sha256": index_digest.removeprefix("sha256:")}, "name": spec["repository"]}],
        }
        provenance_bundle = _artifact(
            root,
            f"{role}.provenance.bundle.json",
            _sigstore_bundle(role_index * 3 + 2, provenance_statement),
        )
        images.append(
            {
                "baseReferences": base_references,
                "indexDigest": index_digest,
                "labels": {
                    "description": spec["description"],
                    "licenses": "AGPL-3.0-or-later",
                    "revision": FULL_SHA,
                    "source": lock["sourceRepository"],
                    "title": spec["title"],
                    "version": "0.1.0",
                },
                "platform": "linux/amd64",
                "platformDigest": platform_digest,
                "provenance": {
                    "builderId": lock["identity"]["provenanceBuilderId"],
                    "bundle": provenance_bundle,
                    "issuer": lock["identity"]["issuer"],
                    "materials": materials,
                    "predicate": predicate,
                    "predicateType": "https://slsa.dev/provenance/v1",
                    "workflowIdentity": lock["identity"]["workflowIdentity"],
                },
                "repository": spec["repository"],
                "role": role,
                "sbom": {
                    "attestation": {
                        "bundle": sbom_bundle,
                        "issuer": lock["identity"]["issuer"],
                        "predicateType": "https://spdx.dev/Document",
                        "workflowIdentity": lock["identity"]["workflowIdentity"],
                    },
                    "document": sbom,
                    "format": "spdx-json",
                },
                "scan": {
                    "allowlistSha256": allowlist_artifact["sha256"],
                    "databaseMetadata": database,
                    "disposition": disposition,
                    "raw": raw_scan,
                    "scannerMetadata": scanner,
                },
                "signature": {
                    "bundle": signature_bundle,
                    "issuer": lock["identity"]["issuer"],
                    "rekor": {
                        "included": True,
                        "integratedTime": 1786150800 + role_index * 3,
                        "logId": "d" * 64,
                        "logIndex": role_index * 3,
                    },
                    "workflowIdentity": lock["identity"]["workflowIdentity"],
                },
            }
        )
    wheel = _artifact(root, "eyvan-0.1.0-py3-none-any.whl", b"qualified wheel fixture")
    sdist = _artifact(root, "eyvan-0.1.0.tar.gz", b"qualified sdist fixture")
    chart = _artifact(root, "eyvan-0.1.0.tgz", b"qualified chart fixture")
    overlay = _artifact(root, "eyvan-release-values.yaml", "release: fixture\n")
    render_text = "\n---\n".join(
        f'apiVersion: v1\nkind: Pod\nspec:\n  containers:\n    - image: "{reference}"\n'
        for reference in rendered_references
    )
    render = _artifact(root, "eyvan-rendered.yaml", render_text)
    receipt = {
        "deployment": {
            "chart": chart,
            "deploymentApplied": False,
            "overlay": overlay,
            "references": [
                {"reference": reference, "role": role}
                for role, reference in zip(ROLES, rendered_references, strict=True)
            ],
            "render": render,
        },
        "images": images,
        "packages": {"sdist": sdist, "wheel": wheel},
        "policy": {"allowlist": allowlist_artifact, "releaseLock": lock_artifact},
        "qualification": {
            "deployment": {"applied": False, "evidence": "not-applied"},
            "hardware": {"evidence": "not-run", "validated": False},
            "unverifiedPlatforms": ["linux/arm64", "linux/ppc64le", "linux/s390x"],
        },
        "schemaVersion": "eyvan.oci.release-receipt.v1",
        "source": {
            "fullSha": FULL_SHA,
            "ref": "refs/heads/main",
            "repository": lock["sourceRepository"],
            "version": "0.1.0",
        },
        "workflowApproval": {
            "actor": "release-operator",
            "approved": True,
            "environment": "eyvan-oci-release",
            "event": "workflow_dispatch",
            "reviewers": [{"login": "release-reviewer"}],
            "runAttempt": 1,
            "runId": 12345,
            "workflowIdentity": lock["identity"]["workflowIdentity"],
        },
    }
    return receipt, root


def test_repository_policy_is_canonical_and_dockerfiles_match_it() -> None:
    result = _policy(root=True)
    assert result.returncode == 0, result.stderr
    assert "0 exact High/Critical exception" in result.stdout


def test_repository_identity_is_the_standalone_eyvan_repository() -> None:
    # The workflow identity is what cosign --certificate-identity demands and
    # what every receipt, attestation, and provenance fact must repeat; a
    # monorepo leftover here would make every release unverifiable.
    lock = json.loads(LOCK.read_text())
    identity = "https://github.com/erfanzar/eyvan/.github/workflows/eyvan-oci-release.yaml@refs/heads/main"
    assert lock["sourceRepository"] == "https://github.com/erfanzar/eyvan"
    assert lock["identity"]["workflowIdentity"] == identity
    assert lock["identity"]["provenanceBuilderId"] == identity
    assert {spec["dockerfile"] for spec in lock["images"].values()} == {
        "Dockerfile",
        "Dockerfile.agent",
        "Dockerfile.worker",
    }
    for role in ROLES:
        assert (ROOT / lock["images"][role]["dockerfile"]).is_file(), role


def test_dockerfile_without_targetplatform_is_rejected(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text())
    relative_paths = [spec["dockerfile"] for spec in lock["images"].values()]
    relative_paths += [".github/workflows/eyvan-oci-release.yaml", ".github/workflows/publish.yaml"]
    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / relative).read_text())
    worker = tmp_path / lock["images"]["worker"]["dockerfile"]
    worker.write_text(worker.read_text().replace("FROM --platform=$TARGETPLATFORM", "FROM", 1))
    result = _run(
        "validate-policy",
        "--lock",
        LOCK,
        "--allowlist",
        ALLOWLIST,
        "--repository-root",
        tmp_path,
    )
    assert result.returncode == 1
    assert "FROM must use a required locked image ARG and TARGETPLATFORM" in result.stderr


def test_duplicate_json_key_is_rejected_before_policy_validation(tmp_path: Path) -> None:
    duplicate = tmp_path / "lock.json"
    duplicate.write_text('{"schemaVersion":"one","schemaVersion":"two"}\n')
    result = _policy(lock=duplicate)
    assert result.returncode == 1
    assert "duplicate JSON key" in result.stderr


def test_semantically_equal_but_noncanonical_lock_is_rejected(tmp_path: Path) -> None:
    pretty = tmp_path / "lock.json"
    pretty.write_text(json.dumps(json.loads(LOCK.read_text()), indent=2) + "\n")
    result = _policy(lock=pretty)
    assert result.returncode == 1
    assert "not canonical JSON" in result.stderr


def test_mutable_base_tag_is_rejected(tmp_path: Path) -> None:
    value = json.loads(LOCK.read_text())
    mutable = f"docker.io/library/python:latest@sha256:{'a' * 64}"
    value["baseImages"]["workerPython"]["reference"] = mutable
    value["images"]["worker"]["buildArgs"]["EYVAN_WORKER_BASE_IMAGE"] = mutable
    path = tmp_path / "lock.json"
    _canonical(path, value)
    result = _policy(lock=path)
    assert result.returncode == 1
    assert "moving tag" in result.stderr


def test_tool_download_must_be_the_exact_official_linux_amd64_asset(tmp_path: Path) -> None:
    value = json.loads(LOCK.read_text())
    value["tools"]["cosign"]["url"] = "https://attacker.invalid/releases/3.0.6/cosign-linux-amd64"
    path = tmp_path / "lock.json"
    _canonical(path, value)
    result = _policy(lock=path)
    assert result.returncode == 1
    assert "exact official linux/amd64 release asset" in result.stderr


def test_tool_lock_records_independently_verified_official_asset_sizes() -> None:
    lock = json.loads(LOCK.read_text())
    assert {name: spec["size"] for name, spec in lock["tools"].items()} == {
        "cosign": 135178161,
        "crane": 16532182,
        "helm": 17986975,
        "syft": 28722052,
        "trivy": 47748408,
    }


def test_tool_lock_rejects_missing_or_excessive_exact_size(tmp_path: Path) -> None:
    missing_value = json.loads(LOCK.read_text())
    del missing_value["tools"]["cosign"]["size"]
    missing = tmp_path / "missing-size.json"
    _canonical(missing, missing_value)
    result = _policy(lock=missing)
    assert result.returncode == 1
    assert "missing=['size']" in result.stderr

    excessive_value = json.loads(LOCK.read_text())
    excessive_value["tools"]["cosign"]["size"] = 512 * 1024 * 1024 + 1
    excessive = tmp_path / "excessive-size.json"
    _canonical(excessive, excessive_value)
    result = _policy(lock=excessive)
    assert result.returncode == 1
    assert "512 MiB hard limit" in result.stderr


class _FakeResponse:
    def __init__(self, content: bytes, *, content_length: str | None = None) -> None:
        self.status = 200
        self.headers = email.message.Message()
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self._stream = io.BytesIO(content)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_locked_tool_download_enforces_content_length_and_streaming_hard_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = _load_script(TOOL_INSTALLER, "eyvan_locked_tool_installer_test")
    destination = tmp_path / "tool"

    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"abc", content_length="4"),
    )
    with pytest.raises(RuntimeError, match="Content-Length mismatch"):
        installer._download("https://example.invalid/tool", destination, expected_size=3)

    destination.unlink(missing_ok=True)
    monkeypatch.setattr(
        installer.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"abcd"),
    )
    with pytest.raises(RuntimeError, match="exceeded locked size"):
        installer._download("https://example.invalid/tool", destination, expected_size=3)
    assert destination.stat().st_size <= 3


def test_release_asset_download_enforces_api_size_without_content_length(
    tmp_path: Path,
) -> None:
    downloader = _load_script(ASSET_DOWNLOADER, "eyvan_release_asset_downloader_test")
    response = _FakeResponse(b"abcd")

    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> _FakeResponse:
            return response

    with pytest.raises(downloader.DownloadError, match="exceeded its exact or hard byte limit"):
        downloader._download_asset(
            Opener(),
            {"name": "asset.json", "size": 3, "url": "https://api.github.com/asset"},
            destination=tmp_path,
            token="fixture-token",
            hard_limit=3,
            expected_sha256=None,
        )
    assert list(tmp_path.iterdir()) == []


def test_release_asset_download_refuses_a_provider_reported_mutable_release() -> None:
    downloader = _load_script(ASSET_DOWNLOADER, "eyvan_mutable_release_downloader_test")
    repository = "erfanzar/eyvan"
    assets = [
        {
            "id": index,
            "name": f"asset-{index}.json",
            "size": 1,
            "state": "uploaded",
            "url": f"https://api.github.com/repos/{repository}/releases/assets/{index}",
        }
        for index in range(1, downloader.MAX_RELEASE_ASSETS + 1)
    ]
    release = {
        "assets": assets,
        "draft": False,
        "immutable": False,
        "prerelease": False,
        "tag_name": "eyvan-v0.1.0",
        "target_commitish": FULL_SHA,
    }
    response = _FakeResponse(json.dumps(release).encode())

    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> _FakeResponse:
            return response

    with pytest.raises(downloader.DownloadError, match="provider-reported immutable"):
        downloader._release_metadata(
            Opener(),
            repository=repository,
            tag="eyvan-v0.1.0",
            token="fixture-token",
            expected_target=FULL_SHA,
        )


def test_release_download_rejects_a_branch_named_target_commitish() -> None:
    # The release workflow must create the release with --target <dispatch
    # SHA>: GitHub otherwise records target_commitish as the default branch
    # name, and the bootstrap download refuses anything but the exact SHA.
    downloader = _load_script(ASSET_DOWNLOADER, "eyvan_branch_target_downloader_test")
    repository = "erfanzar/eyvan"
    assets = [
        {
            "id": index,
            "name": f"asset-{index}.json",
            "size": 1,
            "state": "uploaded",
            "url": f"https://api.github.com/repos/{repository}/releases/assets/{index}",
        }
        for index in range(1, downloader.MAX_RELEASE_ASSETS + 1)
    ]
    release = {
        "assets": assets,
        "draft": False,
        "immutable": True,
        "prerelease": False,
        "tag_name": "eyvan-v0.1.0",
        "target_commitish": "main",
    }
    response = _FakeResponse(json.dumps(release).encode())

    class Opener:
        def open(self, *_args: object, **_kwargs: object) -> _FakeResponse:
            return response

    with pytest.raises(downloader.DownloadError, match="does not bind the exact tag and source SHA"):
        downloader._release_metadata(
            Opener(),
            repository=repository,
            tag="eyvan-v0.1.0",
            token="fixture-token",
            expected_target=FULL_SHA,
        )


def test_placeholder_action_sha_is_rejected(tmp_path: Path) -> None:
    value = json.loads(LOCK.read_text())
    value["actions"]["checkout"] = "0" * 40
    path = tmp_path / "lock.json"
    _canonical(path, value)
    result = _policy(lock=path)
    assert result.returncode == 1
    assert "non-placeholder full action commit SHA" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"expiresAt": "2026-08-08T00:59:59Z"}, "stale"),
        ({"expiresAt": "2026-09-08T01:00:01Z"}, "30-day"),
        ({"package": "openssl*"}, "wildcard"),
        ({"purl": "*"}, "wildcard"),
        ({"owner": "any"}, "owner"),
    ],
)
def test_allowlist_rejects_stale_long_or_wildcard_entries(
    tmp_path: Path,
    mutation: dict[str, str],
    message: str,
) -> None:
    entry = _exception()
    entry.update(mutation)
    path = tmp_path / "allowlist.json"
    _canonical(path, {"entries": [entry], "schemaVersion": "eyvan.oci.high-critical-allowlist.v1"})
    result = _policy(allowlist=path)
    assert result.returncode == 1
    assert message in result.stderr


@pytest.mark.parametrize("target_class,target_type", [("os-pkgs", "alpine"), ("lang-pkgs", "python-pkg")])
@pytest.mark.parametrize("severity", [None, "HIGH", "CRITICAL"])
def test_os_and_library_scans_keep_the_high_critical_gate(
    tmp_path: Path, target_class: str, target_type: str, severity: str | None
) -> None:
    findings = []
    if severity is not None:
        finding = _high_finding()
        finding["Severity"] = severity
        if target_class == "lang-pkgs":
            finding.update(
                InstalledVersion="1.2.3",
                PkgIdentifier={"PURL": "pkg:pypi/fixture-library@1.2.3"},
                PkgName="fixture-library",
            )
        findings.append(finding)
    report = _scan_report(findings=findings)
    report["Results"][0].update({"Class": target_class, "Type": target_type})
    result, output = _gate_scan(tmp_path, report=report)
    assert result.returncode == (0 if severity is None else 1), result.stderr
    disposition = json.loads(output.read_text())
    assert disposition["status"] == ("pass" if severity is None else "fail")
    assert disposition["counts"] == {
        "allowed": 0,
        "critical": int(severity == "CRITICAL"),
        "high": int(severity == "HIGH"),
        "unmatched": int(severity is not None),
    }
    if severity is not None:
        assert "unmatched High/Critical" in result.stderr


def test_clean_trivy_report_emits_canonical_passing_disposition(tmp_path: Path) -> None:
    result, output = _gate_scan(tmp_path, report=_scan_report())
    assert result.returncode == 0, result.stderr
    disposition = json.loads(output.read_text())
    assert disposition["status"] == "pass"
    assert disposition["evaluatedAt"] == NOW
    assert disposition["counts"] == {"allowed": 0, "critical": 0, "high": 0, "unmatched": 0}
    assert output.read_bytes() == (json.dumps(disposition, sort_keys=True, separators=(",", ":")) + "\n").encode()


@pytest.mark.parametrize("fixed_version", ["", "3.5.2-r1"])
def test_unmatched_high_finding_fails_whether_or_not_a_fix_exists(tmp_path: Path, fixed_version: str) -> None:
    result, output = _gate_scan(
        tmp_path,
        report=_scan_report(findings=[_high_finding(fixed_version=fixed_version)]),
    )
    assert result.returncode == 1
    assert "unmatched High/Critical" in result.stderr
    assert json.loads(output.read_text())["status"] == "fail"


def test_exact_allowlist_entry_can_match_one_high_finding(tmp_path: Path) -> None:
    allowlist = {
        "entries": [_exception()],
        "schemaVersion": "eyvan.oci.high-critical-allowlist.v1",
    }
    result, output = _gate_scan(
        tmp_path,
        report=_scan_report(findings=[_high_finding()]),
        allowlist=allowlist,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["counts"] == {
        "allowed": 1,
        "critical": 0,
        "high": 1,
        "unmatched": 0,
    }


def test_near_match_does_not_allow_a_different_installed_version(tmp_path: Path) -> None:
    finding = _high_finding()
    finding["InstalledVersion"] = "3.5.2-r2"
    allowlist = {
        "entries": [_exception()],
        "schemaVersion": "eyvan.oci.high-critical-allowlist.v1",
    }
    result, _output = _gate_scan(tmp_path, report=_scan_report(findings=[finding]), allowlist=allowlist)
    assert result.returncode == 1
    assert "unmatched High/Critical" in result.stderr


def test_scan_rejects_stale_database_and_unlocked_scanner(tmp_path: Path) -> None:
    database = _database_metadata()
    database["downloadedAt"] = "2026-08-06T00:00:00Z"
    stale, _ = _gate_scan(tmp_path, report=_scan_report(), database_metadata=database)
    assert stale.returncode == 1
    assert "downloaded in the last 24 hours" in stale.stderr

    scanner = _scanner_metadata()
    scanner["artifactSha256"] = "d" * 64
    unlocked, _ = _gate_scan(tmp_path, report=_scan_report(), scanner_metadata=scanner)
    assert unlocked.returncode == 1
    assert "scanner artifact hash does not match" in unlocked.stderr


def test_scan_rejects_mismatched_artifact_name(tmp_path: Path) -> None:
    report = _scan_report(image="fixture:one")
    scan = tmp_path / "scan.json"
    scanner = tmp_path / "scanner.json"
    database = tmp_path / "database.json"
    _canonical(scan, report)
    _canonical(scanner, _scanner_metadata())
    _canonical(database, _database_metadata())
    result = _run(
        "gate-scan",
        "--lock",
        LOCK,
        "--allowlist",
        ALLOWLIST,
        "--role",
        "control-plane",
        "--scan",
        scan,
        "--scanner-metadata",
        scanner,
        "--database-metadata",
        database,
        "--image-ref",
        "fixture:two",
        "--now",
        NOW,
    )
    assert result.returncode == 1
    assert "scanned 'fixture:one', expected 'fixture:two'" in result.stderr


def test_image_metadata_checks_labels_platform_entrypoint_and_oidc(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text())
    spec = lock["images"]["agent"]
    labels = {
        "org.opencontainers.image.description": spec["description"],
        "org.opencontainers.image.licenses": "AGPL-3.0-or-later",
        "org.opencontainers.image.revision": FULL_SHA,
        "org.opencontainers.image.source": lock["sourceRepository"],
        "org.opencontainers.image.title": spec["title"],
        "org.opencontainers.image.version": "0.1.0",
    }
    metadata = tmp_path / "metadata.json"
    _canonical(
        metadata,
        {
            "architecture": "amd64",
            "config": {"Entrypoint": ["eyvan-agent"], "Env": ["PATH=/bin"], "Labels": labels},
            "os": "linux",
        },
    )
    result = _run(
        "verify-image-metadata",
        "--lock",
        LOCK,
        "--role",
        "agent",
        "--metadata",
        metadata,
        "--revision",
        FULL_SHA,
        "--version",
        "0.1.0",
    )
    assert result.returncode == 0, result.stderr

    value = json.loads(metadata.read_text())
    value["config"]["Env"].append("EYVAN_OIDC_ISSUER=https://tenant.example")
    _canonical(metadata, value)
    result = _run(
        "verify-image-metadata",
        "--lock",
        LOCK,
        "--role",
        "agent",
        "--metadata",
        metadata,
        "--revision",
        FULL_SHA,
        "--version",
        "0.1.0",
    )
    assert result.returncode == 1
    assert "environment-specific OIDC" in result.stderr


@pytest.mark.parametrize("text", ["", "null\n", "- string\n", "plain string\n", "key: [unterminated\n"])
def test_render_rejects_empty_non_mapping_and_invalid_yaml_without_traceback(tmp_path: Path, text: str) -> None:
    render = tmp_path / "invalid-root.yaml"
    render.write_text(text)
    arguments: list[str | Path] = ["verify-rendered-digest-refs", "--render", render]
    for index, role in enumerate(ROLES, 1):
        reference = f"registry.example/eyvan-{role}@sha256:{str(index) * 64}"
        arguments.extend(("--reference", f"{role}={reference}"))
    result = _run(*arguments)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr
    assert "verified" not in result.stdout


def test_render_accepts_only_the_three_exact_distinct_digest_refs(tmp_path: Path) -> None:
    refs = [f"registry.example/eyvan-{role}@sha256:{str(index) * 64}" for index, role in enumerate(ROLES, 1)]
    render = tmp_path / "render.yaml"
    render.write_text(
        "\n---\n".join(
            f'apiVersion: v1\nkind: Pod\nspec:\n  containers:\n    - image: "{reference}"\n' for reference in refs
        )
    )
    arguments: list[str | Path] = ["verify-rendered-digest-refs", "--render", render]
    for role, reference in zip(ROLES, refs, strict=True):
        arguments.extend(("--reference", f"{role}={reference}"))
    result = _run(*arguments)
    assert result.returncode == 0, result.stderr

    render.write_text(render.read_text().replace(refs[2], "registry.example/eyvan-worker:latest"))
    result = _run(*arguments)
    assert result.returncode == 1
    assert "mutable image" in result.stderr


def test_emit_and_verify_canonical_closed_receipt(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    source.write_text(json.dumps(receipt, indent=2) + "\n")
    emit = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert emit.returncode == 0, emit.stderr
    assert output.read_bytes() == (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    asset_manifest = tmp_path / "asset-manifest.json"
    verify = _run(
        "verify-receipt",
        "--lock",
        LOCK,
        "--receipt",
        output,
        "--artifact-root",
        root,
        "--asset-manifest-output",
        asset_manifest,
        "--expected-sha",
        FULL_SHA,
        "--expected-ref",
        "refs/heads/main",
        "--expected-version",
        "0.1.0",
    )
    assert verify.returncode == 0, verify.stderr
    manifest = json.loads(asset_manifest.read_text())
    assert manifest["schemaVersion"] == "eyvan.oci.release-assets.v1"
    assert len(manifest["artifacts"]) == 34
    assert len({record["name"] for record in manifest["artifacts"]}) == 34


def test_receipt_rejects_unknown_diagnostics_field(tmp_path: Path) -> None:
    receipt, _root = _receipt_fixture(tmp_path)
    receipt["diagnostics"] = {"anything": "would make the schema open"}
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run("emit-receipt", "--lock", LOCK, "--input", source, "--output", output)
    assert result.returncode == 1
    assert "unknown=['diagnostics']" in result.stderr


def test_receipt_rejects_role_digest_or_label_mismatch(tmp_path: Path) -> None:
    receipt, _root = _receipt_fixture(tmp_path)
    receipt["images"][1]["labels"]["revision"] = "b" * 40
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run("emit-receipt", "--lock", LOCK, "--input", source, "--output", output)
    assert result.returncode == 1
    assert "labels do not exactly bind" in result.stderr


def test_receipt_artifact_hash_mismatch_fails(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    (root / receipt["packages"]["wheel"]["name"]).write_bytes(b"tampered")
    result = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert result.returncode == 1
    assert "hash mismatch" in result.stderr


def test_receipt_rejects_a_bundle_without_rekor_inclusion(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    bundle_record = receipt["images"][0]["signature"]["bundle"]
    bundle_path = root / bundle_record["name"]
    bundle = json.loads(bundle_path.read_text())
    del bundle["verificationMaterial"]["tlogEntries"][0]["inclusionPromise"]
    _canonical(bundle_path, bundle)
    bundle_record["sha256"] = _sha(bundle_path)
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert result.returncode == 1
    assert "no inclusion promise or proof" in result.stderr


def test_receipt_replays_the_scan_gate_instead_of_trusting_passing_counts(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    image = receipt["images"][0]
    raw_path = root / image["scan"]["raw"]["name"]
    raw = json.loads(raw_path.read_text())
    raw["Results"][0]["Vulnerabilities"].append(_high_finding())
    _canonical(raw_path, raw)
    image["scan"]["raw"]["sha256"] = _sha(raw_path)

    disposition_path = root / image["scan"]["disposition"]["name"]
    disposition = json.loads(disposition_path.read_text())
    disposition["rawScanSha256"] = image["scan"]["raw"]["sha256"]
    _canonical(disposition_path, disposition)
    image["scan"]["disposition"]["sha256"] = _sha(disposition_path)

    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert result.returncode == 1
    assert "unmatched High/Critical finding" in result.stderr


def test_receipt_replay_uses_the_recorded_scan_decision_time(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    image = receipt["images"][0]
    scanner_path = root / image["scan"]["scannerMetadata"]["name"]
    scanner = json.loads(scanner_path.read_text())
    scanner["generatedAt"] = "2026-08-07T02:00:00Z"
    _canonical(scanner_path, scanner)
    image["scan"]["scannerMetadata"]["sha256"] = _sha(scanner_path)

    disposition_path = root / image["scan"]["disposition"]["name"]
    disposition = json.loads(disposition_path.read_text())
    disposition["scannerMetadataSha256"] = image["scan"]["scannerMetadata"]["sha256"]
    assert disposition["evaluatedAt"] == NOW
    _canonical(disposition_path, disposition)
    image["scan"]["disposition"]["sha256"] = _sha(disposition_path)

    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert result.returncode == 0, result.stderr


def test_verify_rejects_noncanonical_receipt_even_when_values_match(tmp_path: Path) -> None:
    receipt, _root = _receipt_fixture(tmp_path)
    pretty = tmp_path / "receipt.json"
    pretty.write_text(json.dumps(receipt, indent=2) + "\n")
    result = _run("verify-receipt", "--lock", LOCK, "--receipt", pretty)
    assert result.returncode == 1
    assert "not canonical JSON" in result.stderr


def test_receipt_render_hash_can_be_valid_but_wrong_refs_still_fail(tmp_path: Path) -> None:
    receipt, root = _receipt_fixture(tmp_path)
    render_record = receipt["deployment"]["render"]
    render_path = root / render_record["name"]
    render_path.write_text(render_path.read_text().replace("@sha256:", ":latest#", 1))
    render_record["sha256"] = _sha(render_path)
    source = tmp_path / "receipt-input.json"
    output = tmp_path / "receipt.json"
    _canonical(source, receipt)
    result = _run(
        "emit-receipt",
        "--lock",
        LOCK,
        "--input",
        source,
        "--output",
        output,
        "--artifact-root",
        root,
    )
    assert result.returncode == 1
    assert "mutable image" in result.stderr


def test_workflow_separates_read_only_pr_and_environment_gated_release_lanes() -> None:
    path = ROOT / ".github" / "workflows" / "eyvan-oci-release.yaml"
    content = path.read_text()
    workflow = yaml.safe_load(content)
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert "pull_request_target" not in content

    pull_request = workflow["jobs"]["pull-request-qualification"]
    assert pull_request["permissions"] == {"contents": "read"}
    pull_text = json.dumps(pull_request)
    assert "docker/login-action" not in pull_text
    assert "id-token" not in pull_text
    assert pull_text.count('"push": false') == 3
    assert pull_text.count('"load": true') == 3
    assert "--help" in pull_text
    assert "spdx-json" in pull_text
    assert "gate-scan" in pull_text

    release = workflow["jobs"]["release"]
    assert release["environment"] == "eyvan-oci-release"
    assert release["permissions"] == {
        "actions": "read",
        "contents": "write",
        "id-token": "write",
        "packages": "write",
    }
    assert "attestations" not in content
    release_text = json.dumps(release)
    release_commands = "\n".join(step.get("run", "") for step in release["steps"])
    distribution_step = next(
        step
        for step in release["steps"]
        if step["name"] == "Build and smoke the exact wheel and sdist before OCI qualification"
    )
    assert "export UV_PYTHON_DOWNLOADS=never" in distribution_step["run"]
    # The standalone repository has no uv workspace or lockfile: the wheel and
    # sdist are built from the repo-root pyproject with the hash-pinned
    # requirements-build.txt backend.
    assert "--require-hashes -r requirements-build.txt" in distribution_step["run"]
    assert "uv build --no-build-isolation --wheel --sdist --out-dir dist ." in distribution_step["run"]
    assert "smoke_distribution dist/eyvan-*.whl" in distribution_step["run"]
    assert "smoke_distribution dist/eyvan-*.tar.gz" in distribution_step["run"]
    assert "--require-hashes -r requirements-build.txt -r requirements-docker.txt" in distribution_step["run"]
    assert "EXPECTED_EYVAN_VERSION" in distribution_step["run"]
    assert "eyvan-migrate" in distribution_step["run"]
    governance_step = next(
        step
        for step in release["steps"]
        if step["name"] == "Prove repository governance and approved environment review"
    )
    candidate_step = next(step for step in release["steps"] if step["name"].startswith("Prove candidate"))
    assert governance_step["env"]["GOVERNANCE_TOKEN"] == "${{ secrets.EYVAN_GITHUB_GOVERNANCE_TOKEN }}"
    assert "GOVERNANCE_TOKEN" not in candidate_step["env"]
    assert "APPROVAL_TOKEN" not in candidate_step["env"]
    assert "$VERIFIER" not in governance_step["run"]
    assert "verify_eyvan_release.py" not in governance_step["run"]
    assert "/branches/main" in release_commands
    assert "/immutable-releases" in release_commands
    assert 'branch.get("protected") is not True' in release_commands
    assert 'immutable.get("enabled") is not True' in release_commands
    assert "/approvals" in release_commands
    assert release_text.count('"push": true') == 3
    assert "sha-${{ steps.candidate.outputs.sha }}" in release_text
    assert "crane copy" in release_text
    assert "verify-rendered-digest-refs" in release_text
    assert "sign-blob" in release_text
    assert "gh release create" in release_commands and "--draft" in release_commands
    assert "gh release edit" in release_commands and "--draft=false" in release_commands
    assert "--jq .immutable" in release_commands
    step_names = [step["name"] for step in release["steps"]]
    promotion_index = step_names.index("Promote the verified and receipt-signed index digests to immutable version tags")
    assert promotion_index > step_names.index(
        "Package chart, hash digest overlay/render, and verify the M1.8 three-way split"
    )
    assert promotion_index > step_names.index("Emit and independently verify the closed canonical receipt")
    assert promotion_index > step_names.index("Keyless-sign and strictly verify the canonical receipt blob")
    logical_commands = content.replace(chr(92) + "\n", " ")
    image_verifications = [
        line.strip()
        for line in logical_commands.splitlines()
        if line.strip().startswith(("cosign verify ", "cosign verify-attestation "))
    ]
    assert len(image_verifications) == 3
    assert all("--new-bundle-format" in command for command in image_verifications)
    assert all("--bundle" not in command for command in image_verifications)
    assert "repository_dispatch" not in release_text  # dispatch uses the GitHub API event type
    assert ":latest" not in release_text


def _release_step(step_name_prefix: str) -> dict[str, object]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "eyvan-oci-release.yaml").read_text())
    return next(step for step in workflow["jobs"]["release"]["steps"] if step["name"].startswith(step_name_prefix))


def test_every_image_build_uses_the_repo_root_package_context() -> None:
    # The repository root is the package root: every Dockerfile copies
    # dashboard/, requirements-*.txt, and eyvan/ relative to it and demands it
    # as the build context; any other context kills all six builds at the
    # first COPY.
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "eyvan-oci-release.yaml").read_text())
    builds = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("docker/build-push-action")
    ]
    assert len(builds) == 6
    for step in builds:
        assert step["with"]["context"] == ".", step["name"]
        assert re.fullmatch(r"Dockerfile(\.agent|\.worker)?", step["with"]["file"]), step["name"]


def test_release_workflow_actions_match_the_locked_pins_exactly() -> None:
    # Every pinned action is addressed at exactly the commit the lock records,
    # and no other action appears anywhere in the release workflow.
    lock = json.loads(LOCK.read_text())
    expected = {
        "actions/checkout": lock["actions"]["checkout"],
        "actions/upload-artifact": lock["actions"]["uploadArtifact"],
        "astral-sh/setup-uv": lock["actions"]["setupUv"],
        "docker/build-push-action": lock["actions"]["buildPush"],
        "docker/login-action": lock["actions"]["login"],
        "docker/setup-buildx-action": lock["actions"]["setupBuildx"],
    }
    content = (ROOT / ".github" / "workflows" / "eyvan-oci-release.yaml").read_text()
    found: dict[str, set[str]] = {}
    for action, revision in re.findall(r"^\s*-?\s*uses:\s*([^@\s#]+)@([^\s#]+)", content, flags=re.MULTILINE):
        found.setdefault(action, set()).add(revision)
    assert found == {action: {revision} for action, revision in expected.items()}
    # The lock still records downloadArtifact so its pin cannot drift, but no
    # standalone workflow downloads Actions artifacts.
    assert "actions/download-artifact" not in content


def test_digest_overlay_sets_the_chart_read_image_keys(tmp_path: Path) -> None:
    # The chart helpers read image/agentImage/workerImage (repository plus a
    # ^sha256:[0-9a-f]{64}$ digest); controlPlane/agent/worker image keys are
    # never read and leave mutable defaults in the render.
    step = _release_step("Package chart, hash digest overlay/render")
    run = step["run"]
    assert "controlPlane:" not in run
    assert "deploy/helm/eyvan" in run
    lines = run.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == "{")
    end = next(
        index for index, line in enumerate(lines) if line.strip().startswith("}") and "eyvan-release-values.yaml" in line
    )
    overlay = tmp_path / "eyvan-release-values.yaml"
    refs = {
        "CONTROL_PLANE_REF": f"ghcr.io/erfanzar/eyvan-control-plane@sha256:{'1' * 64}",
        "AGENT_REF": f"ghcr.io/erfanzar/eyvan-agent@sha256:{'2' * 64}",
        "WORKER_REF": f"ghcr.io/erfanzar/eyvan-native-worker@sha256:{'3' * 64}",
    }
    subprocess.run(
        ["bash", "-c", "{\n" + "\n".join(lines[start + 1 : end]) + "\n} > '" + str(overlay) + "'"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **refs},
    )
    assert yaml.safe_load(overlay.read_text()) == {
        "image": {"repository": "ghcr.io/erfanzar/eyvan-control-plane", "digest": f"sha256:{'1' * 64}"},
        "agentImage": {"repository": "ghcr.io/erfanzar/eyvan-agent", "digest": f"sha256:{'2' * 64}"},
        "workerImage": {"repository": "ghcr.io/erfanzar/eyvan-native-worker", "digest": f"sha256:{'3' * 64}"},
    }
    # The agent and its Kubernetes-capacity worker image render only when the
    # cell agent is enabled, so the evidence render must exercise that path.
    template_command = run[run.index("helm template") : run.index("verify-rendered-digest-refs")]
    assert "--values evidence/eyvan-release-values.yaml" in template_command
    assert "--set cellAgent.enabled=true" in template_command
    assert "--set cellAgent.capacityBackend=kubernetes" in template_command


def test_receipt_step_takes_the_version_from_an_explicit_step_env() -> None:
    # GitHub injects INPUT_* only into composite actions, never run steps; the
    # receipt builder must read a step env var bound to the workflow input.
    step = _release_step("Emit and independently verify the closed canonical receipt")
    assert step["env"]["DISPATCH_VERSION"] == "${{ inputs.version }}"
    run = step["run"]
    assert 'os.environ["INPUT_VERSION"]' not in run
    assert 'os.environ["DISPATCH_VERSION"]' in run
    assert 'open("oci/release-lock.json"' in run


def test_release_creation_binds_target_commitish_to_the_qualified_sha() -> None:
    step = _release_step("Publish the draft-then-final immutable release")
    lines = step["run"].splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("gh release create"))
    end = start
    while lines[end].endswith("\\"):
        end += 1
    assert '--target "${GITHUB_SHA}"' in [line.rstrip("\\").strip() for line in lines[start : end + 1]]
    # The tag is derived from the dispatch version, never from a moving ref.
    assert 'tag="eyvan-v${{ inputs.version }}"' in step["run"]


def test_every_workflow_action_is_pinned_to_a_full_commit() -> None:
    for name in ("eyvan-oci-release.yaml", "publish.yaml"):
        content = (ROOT / ".github" / "workflows" / name).read_text()
        uses = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", content, flags=re.MULTILINE)
        assert uses
        for reference in uses:
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", reference), (name, reference)


def test_eyvan_publish_is_tag_gated_trusted_publishing_without_a_secret() -> None:
    # The standalone publish workflow is the entire PyPI surface: tag-driven
    # only, gated by the pypi environment, authenticated with OIDC trusted
    # publishing, and carrying no long-lived credential to steal.
    path = ROOT / ".github" / "workflows" / "publish.yaml"
    content = path.read_text()
    workflow = yaml.safe_load(content)
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["tags"] == ["v*"]

    assert workflow["permissions"] == {"contents": "read"}
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {"contents": "read", "id-token": "write", "attestations": "write"}

    checkout = publish["steps"][0]
    assert str(checkout["uses"]).startswith("actions/checkout@")
    assert checkout["with"]["persist-credentials"] is False

    agreement = next(step for step in publish["steps"] if step.get("name") == "The tag and the source must agree")
    agreement_commands = agreement["run"]
    assert "pyproject.toml" in agreement_commands
    assert "GITHUB_REF_NAME" in agreement_commands
    assert agreement["env"]["REQUESTED_VERSION"] == "${{ github.event.inputs.version }}"
    assert 'CLAIMED="$REQUESTED_VERSION"' in agreement_commands
    assert "but pyproject declares" in agreement_commands

    assert "uv publish --trusted-publishing always" in content
    assert "secrets." not in content
    assert "gh release download" not in content
    assert "--require-hashes -r requirements-build.txt" in content
    assert "eyvan-migrate" in content


def test_eyvan_release_downloader_stays_bounded_and_name_closed() -> None:
    downloader = ASSET_DOWNLOADER.read_text()
    assert "MAX_RELEASE_ASSETS = 36" in downloader
    assert "MAX_RELEASE_BYTES = 1024 * 1024 * 1024" in downloader
    assert "MAX_ARTIFACT_BYTES = 256 * 1024 * 1024" in downloader
    assert "Content-Length" in downloader
    assert "expected_names = {*records, RECEIPT_NAME, RECEIPT_BUNDLE_NAME}" in downloader
    assert "set(assets) != expected_names" in downloader
