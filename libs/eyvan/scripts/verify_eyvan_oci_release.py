#!/usr/bin/env python3
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fail-closed policy and receipt verifier for Eyvan OCI releases.

The verifier intentionally uses only the Python standard library.  It treats
all policy, scanner, image, render, and receipt documents as untrusted input:
duplicate JSON keys, non-canonical policy/receipt bytes, unknown schema fields,
and mutable references are errors rather than warnings.

Every default path is relative to the standalone Eyvan repository root (the
package root): the release workflow lives at
``.github/workflows/eyvan-oci-release.yaml``, the policy lock at
``oci/release-lock.json``, and the three Dockerfiles at the repository root.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

ROLES = ("control-plane", "agent", "worker")
ROLE_TITLES = {
    "control-plane": "Eyvan control plane",
    "agent": "Eyvan cell agent",
    "worker": "Eyvan native Kubernetes worker",
}
ROLE_DESCRIPTIONS = {
    "control-plane": "Policy-aware global fleet control plane for CPU, GPU, and TPU capacity",
    "agent": "Capacity and execution cell agent for the Eyvan fleet control plane",
    "worker": "Native execution worker for Eyvan Kubernetes capacity cells",
}
ROLE_ENTRYPOINTS = {
    "control-plane": "eyvan-server",
    "agent": "eyvan-agent",
    "worker": "eyvan-native-worker",
}
LOCK_SCHEMA = "eyvan.oci.release-lock.v1"
ALLOWLIST_SCHEMA = "eyvan.oci.high-critical-allowlist.v1"
SCAN_METADATA_SCHEMA = "eyvan.oci.scanner-metadata.v1"
DB_METADATA_SCHEMA = "eyvan.oci.vulnerability-db-metadata.v1"
DISPOSITION_SCHEMA = "eyvan.oci.scan-disposition.v1"
RECEIPT_SCHEMA = "eyvan.oci.release-receipt.v1"
ASSET_MANIFEST_SCHEMA = "eyvan.oci.release-assets.v1"
ISSUER = "https://token.actions.githubusercontent.com"
LICENSE = "AGPL-3.0-or-later"
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
ACTION_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
RFC3339_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z\Z")
VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9A-Za-z]+)+(?:[-+][0-9A-Za-z.-]+)?\Z")
ARTIFACT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\Z")
IMAGE_REF_RE = re.compile(
    r"(?P<name>[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+):(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})"
    r"@(?P<digest>sha256:[0-9a-f]{64})\Z"
)
FORBIDDEN_WILDCARD_RE = re.compile(r"[*?\[\]{}]")
# Match the release asset downloader's per-artifact hard limit. DSSE statements
# are artifact data, not small metadata strings; base64 adds one third in size.
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_DSSE_PAYLOAD_BYTES = 192 * 1024 * 1024


class VerificationError(ValueError):
    """Raised for a release-policy verification failure."""


def _fail(message: str) -> NoReturn:
    raise VerificationError(message)


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path, *, canonical: bool = False, maximum: int | None = None) -> Any:
    try:
        with path.open("rb") as handle:
            raw = handle.read() if maximum is None else handle.read(maximum + 1)
        if maximum is not None and len(raw) > maximum:
            _fail(f"{path} exceeds the {maximum}-byte limit")
    except OSError as error:
        _fail(f"cannot read {path}: {error}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        _fail(f"{path} is not strict UTF-8: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        _fail(f"invalid JSON in {path}: {error}")
    if canonical and raw != _canonical_bytes(value):
        _fail(f"{path} is not canonical JSON (sorted keys, compact separators, one trailing newline required)")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{context} must be an array")
    return value


def _exact_keys(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    obj = _mapping(value, context)
    actual = set(obj)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        _fail(f"{context} has the wrong fields ({', '.join(parts)})")
    return obj


def _string(value: Any, context: str, *, minimum: int = 1, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not (minimum <= len(value) <= maximum):
        _fail(f"{context} must be a string of length {minimum}..{maximum}")
    if value != value.strip() or "\x00" in value:
        _fail(f"{context} must not have surrounding whitespace or NUL bytes")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{context} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{context} must be a boolean")
    return value


def _https(value: Any, context: str) -> str:
    text = _string(value, context)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        _fail(f"{context} must be an absolute HTTPS URL without credentials or a fragment")
    return text


def _digest(value: Any, context: str) -> str:
    text = _string(value, context)
    if not DIGEST_RE.fullmatch(text):
        _fail(f"{context} must be sha256: followed by 64 lowercase hexadecimal characters")
    return text


def _hex(value: Any, context: str) -> str:
    text = _string(value, context)
    if not HEX_RE.fullmatch(text):
        _fail(f"{context} must be 64 lowercase hexadecimal characters")
    return text


def _full_sha(value: Any, context: str) -> str:
    text = _string(value, context)
    if not FULL_SHA_RE.fullmatch(text):
        _fail(f"{context} must be a full 40-character lowercase Git SHA")
    return text


def _version(value: Any, context: str) -> str:
    text = _string(value, context)
    if text.lower() in {"latest", "unknown", "dev", "main"} or not VERSION_RE.fullmatch(text):
        _fail(f"{context} must be a concrete dotted release version")
    return text


def _utc(value: Any, context: str) -> dt.datetime:
    text = _string(value, context)
    if not UTC_RE.fullmatch(text):
        _fail(f"{context} must be second-precision UTC in YYYY-MM-DDTHH:MM:SSZ form")
    try:
        parsed = dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.UTC)
    except ValueError as error:
        _fail(f"{context} is not a valid UTC timestamp: {error}")
    return parsed


def _rfc3339_utc(value: Any, context: str) -> dt.datetime:
    text = _string(value, context)
    if not RFC3339_UTC_RE.fullmatch(text):
        _fail(f"{context} must be a UTC RFC3339 timestamp")
    try:
        return dt.datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        _fail(f"{context} is not a valid UTC timestamp: {error}")


def _parse_now(value: str | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.UTC)
    return _utc(value, "--now")


def _immutable_image_ref(value: Any, context: str) -> str:
    text = _string(value, context)
    match = IMAGE_REF_RE.fullmatch(text)
    if not match:
        _fail(f"{context} must be an exact repository:fixed-tag@sha256 digest reference")
    if match.group("tag").lower() in {"latest", "stable", "main", "master", "edge", "nightly"}:
        _fail(f"{context} uses a moving tag: {match.group('tag')!r}")
    return text


def _artifact(value: Any, context: str) -> dict[str, str]:
    obj = _exact_keys(value, {"name", "sha256"}, context)
    name = _string(obj["name"], f"{context}.name", maximum=200)
    if not ARTIFACT_NAME_RE.fullmatch(name) or name in {".", ".."}:
        _fail(f"{context}.name must be a plain portable filename")
    return {"name": name, "sha256": _hex(obj["sha256"], f"{context}.sha256")}


def _logical_dockerfile_lines(text: str, context: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or (not pending and stripped.startswith("#")):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(pending)
        pending = ""
    if pending:
        _fail(f"{context} ends in an unterminated continuation")
    return lines


def _validate_release_lock(path: Path, *, repository_root: Path | None = None) -> dict[str, Any]:
    lock = _exact_keys(
        _load_json(path, canonical=True),
        {
            "actions",
            "baseImages",
            "identity",
            "images",
            "platform",
            "schemaVersion",
            "sourceRepository",
            "tools",
            "unverifiedPlatforms",
        },
        "release lock",
    )
    if lock["schemaVersion"] != LOCK_SCHEMA:
        _fail(f"release lock schemaVersion must be {LOCK_SCHEMA!r}")
    source = _https(lock["sourceRepository"], "release lock sourceRepository")
    if lock["platform"] != "linux/amd64":
        _fail("release lock platform must be exactly 'linux/amd64'")
    unverified = _sequence(lock["unverifiedPlatforms"], "release lock unverifiedPlatforms")
    expected_unverified = ["linux/arm64", "linux/ppc64le", "linux/s390x"]
    if unverified != expected_unverified:
        _fail(f"release lock unverifiedPlatforms must be {expected_unverified!r}")

    base_images = _exact_keys(lock["baseImages"], {"cloudSdk", "node", "uv", "workerPython"}, "release lock baseImages")
    for name, value in base_images.items():
        spec = _exact_keys(
            value,
            {"indexDigest", "platform", "reference", "resolvedAt"},
            f"release lock baseImages.{name}",
        )
        _immutable_image_ref(spec["reference"], f"release lock baseImages.{name}.reference")
        _digest(spec["indexDigest"], f"release lock baseImages.{name}.indexDigest")
        if spec["platform"] != "linux/amd64":
            _fail(f"release lock baseImages.{name}.platform must be linux/amd64")
        _utc(spec["resolvedAt"], f"release lock baseImages.{name}.resolvedAt")
    expected_build_references = {
        "control-plane": {
            "EYVAN_NODE_IMAGE": base_images["node"]["reference"],
            "EYVAN_RUNTIME_IMAGE": base_images["cloudSdk"]["reference"],
            "EYVAN_UV_IMAGE": base_images["uv"]["reference"],
        },
        "agent": {
            "EYVAN_RUNTIME_IMAGE": base_images["cloudSdk"]["reference"],
            "EYVAN_UV_IMAGE": base_images["uv"]["reference"],
        },
        "worker": {
            "EYVAN_UV_IMAGE": base_images["uv"]["reference"],
            "EYVAN_WORKER_BASE_IMAGE": base_images["workerPython"]["reference"],
        },
    }

    images = _exact_keys(lock["images"], set(ROLES), "release lock images")
    repositories: set[str] = set()
    dockerfiles: set[str] = set()
    for role in ROLES:
        spec = _exact_keys(
            images[role],
            {"buildArgs", "description", "dockerfile", "entrypoint", "repository", "title"},
            f"release lock images.{role}",
        )
        if spec["title"] != ROLE_TITLES[role]:
            _fail(f"release lock images.{role}.title is not the required title")
        if spec["description"] != ROLE_DESCRIPTIONS[role]:
            _fail(f"release lock images.{role}.description is not the required description")
        if spec["entrypoint"] != ROLE_ENTRYPOINTS[role]:
            _fail(f"release lock images.{role}.entrypoint is not the required entrypoint")
        repository = _string(spec["repository"], f"release lock images.{role}.repository")
        if "@" in repository or ":" in repository.rsplit("/", 1)[-1] or repository.lower().endswith("/latest"):
            _fail(f"release lock images.{role}.repository must be an untagged repository")
        if not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+", repository):
            _fail(f"release lock images.{role}.repository is invalid")
        if repository in repositories:
            _fail("the control-plane, agent, and worker repositories must be distinct")
        repositories.add(repository)
        dockerfile = _string(spec["dockerfile"], f"release lock images.{role}.dockerfile")
        expected_dockerfile = {
            "control-plane": "Dockerfile",
            "agent": "Dockerfile.agent",
            "worker": "Dockerfile.worker",
        }[role]
        if dockerfile != expected_dockerfile or dockerfile in dockerfiles:
            _fail(f"release lock images.{role}.dockerfile must be {expected_dockerfile!r}")
        dockerfiles.add(dockerfile)
        build_args = _mapping(spec["buildArgs"], f"release lock images.{role}.buildArgs")
        expected_args = {
            "control-plane": {"EYVAN_NODE_IMAGE", "EYVAN_RUNTIME_IMAGE", "EYVAN_UV_IMAGE"},
            "agent": {"EYVAN_RUNTIME_IMAGE", "EYVAN_UV_IMAGE"},
            "worker": {"EYVAN_WORKER_BASE_IMAGE", "EYVAN_UV_IMAGE"},
        }[role]
        if set(build_args) != expected_args:
            _fail(
                f"release lock images.{role}.buildArgs must have exactly {sorted(expected_args)!r}, "
                f"got {sorted(build_args)!r}"
            )
        if build_args != expected_build_references[role]:
            _fail(f"release lock images.{role}.buildArgs must reuse the named baseImages references exactly")
        for argument, reference in build_args.items():
            _immutable_image_ref(reference, f"release lock images.{role}.buildArgs.{argument}")

    tools = _exact_keys(lock["tools"], {"cosign", "crane", "helm", "syft", "trivy"}, "release lock tools")
    expected_tool_sources = {
        "cosign": lambda version: (
            f"https://github.com/sigstore/cosign/releases/download/v{version}/cosign-linux-amd64",
            "cosign",
        ),
        "crane": lambda version: (
            "https://github.com/google/go-containerregistry/releases/download/"
            f"v{version}/go-containerregistry_Linux_x86_64.tar.gz",
            "crane",
        ),
        "helm": lambda version: (f"https://get.helm.sh/helm-v{version}-linux-amd64.tar.gz", "linux-amd64/helm"),
        "syft": lambda version: (
            f"https://github.com/anchore/syft/releases/download/v{version}/syft_{version}_linux_amd64.tar.gz",
            "syft",
        ),
        "trivy": lambda version: (
            f"https://github.com/aquasecurity/trivy/releases/download/v{version}/trivy_{version}_Linux-64bit.tar.gz",
            "trivy",
        ),
    }
    for name, value in tools.items():
        spec = _exact_keys(
            value,
            {"archivePath", "sha256", "size", "url", "version"},
            f"release lock tools.{name}",
        )
        version = _version(spec["version"], f"release lock tools.{name}.version")
        url = _https(spec["url"], f"release lock tools.{name}.url")
        if version not in url or "/latest/" in url or url.endswith("/latest"):
            _fail(f"release lock tools.{name}.url must contain its fixed version")
        _hex(spec["sha256"], f"release lock tools.{name}.sha256")
        size = _integer(spec["size"], f"release lock tools.{name}.size", minimum=1)
        if size > 512 * 1024 * 1024:
            _fail(f"release lock tools.{name}.size exceeds the 512 MiB hard limit")
        archive_path = _string(spec["archivePath"], f"release lock tools.{name}.archivePath", maximum=200)
        if archive_path.startswith(("/", "../")) or "/../" in archive_path:
            _fail(f"release lock tools.{name}.archivePath must stay inside the archive")
        expected_url, expected_archive_path = expected_tool_sources[name](version)
        if url != expected_url or archive_path != expected_archive_path:
            _fail(f"release lock tools.{name} must use its exact official linux/amd64 release asset")

    actions = _exact_keys(
        lock["actions"],
        {
            "attestBuildProvenance",
            "buildPush",
            "checkout",
            "downloadArtifact",
            "login",
            "setupBuildx",
            "setupNode",
            "setupUv",
            "uploadArtifact",
        },
        "release lock actions",
    )
    for name, value in actions.items():
        sha = _string(value, f"release lock actions.{name}")
        if not ACTION_SHA_RE.fullmatch(sha) or sha == "0" * 40:
            _fail(f"release lock actions.{name} must be a non-placeholder full action commit SHA")

    identity = _exact_keys(
        lock["identity"],
        {"issuer", "provenanceBuilderId", "workflowIdentity"},
        "release lock identity",
    )
    if identity["issuer"] != ISSUER:
        _fail(f"release lock identity.issuer must be {ISSUER!r}")
    workflow_identity = _https(identity["workflowIdentity"], "release lock identity.workflowIdentity")
    expected_identity = f"{source}/.github/workflows/eyvan-oci-release.yaml@refs/heads/main"
    if workflow_identity != expected_identity:
        _fail(f"release lock workflow identity must be exact: {expected_identity}")
    if identity["provenanceBuilderId"] != workflow_identity:
        _fail("release lock provenanceBuilderId must equal the exact release workflow identity")

    if repository_root is not None:
        for role in ROLES:
            _validate_dockerfile(repository_root, role, lock)
        _validate_workflow_pins(repository_root, lock)
    return lock


def _validate_workflow_pins(repository_root: Path, lock: dict[str, Any]) -> None:
    expected = {
        "actions/attest-build-provenance": lock["actions"]["attestBuildProvenance"],
        "actions/checkout": lock["actions"]["checkout"],
        "actions/download-artifact": lock["actions"]["downloadArtifact"],
        "actions/upload-artifact": lock["actions"]["uploadArtifact"],
        # publish.yaml builds the dashboard into the wheel before the Python build.
        "actions/setup-node": lock["actions"]["setupNode"],
        "astral-sh/setup-uv": lock["actions"]["setupUv"],
        "docker/build-push-action": lock["actions"]["buildPush"],
        "docker/login-action": lock["actions"]["login"],
        "docker/setup-buildx-action": lock["actions"]["setupBuildx"],
    }
    found: dict[str, set[str]] = {name: set() for name in expected}
    for relative in (".github/workflows/eyvan-oci-release.yaml", ".github/workflows/publish.yaml"):
        path = repository_root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except OSError as error:
            _fail(f"cannot read release workflow {relative}: {error}")
        if "pull_request_target" in text:
            _fail(f"{relative} must never use pull_request_target")
        for action, revision in re.findall(r"^\s*-?\s*uses:\s*([^@\s#]+)@([^\s#]+)", text, flags=re.MULTILINE):
            if action not in expected:
                _fail(f"{relative} uses an action absent from the release lock: {action}")
            found[action].add(revision)
    for action, revision in expected.items():
        if not found[action]:
            # downloadArtifact stays recorded in the lock so its pinned commit
            # cannot drift, but no standalone workflow consumes Actions
            # artifacts; every other locked action must be exercised here.
            if action == "actions/download-artifact":
                continue
            _fail(f"release workflow never pins locked action {action}")
        if found[action] != {revision}:
            _fail(f"workflow pin for {action} must be exactly {revision}, got {sorted(found[action])!r}")


def _validate_dockerfile(repository_root: Path, role: str, lock: dict[str, Any]) -> None:
    spec = lock["images"][role]
    path = repository_root / spec["dockerfile"]
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        _fail(f"cannot read locked Dockerfile {path}: {error}")
    if any(token in text.lower() for token in (":latest", "=latest", "=unknown")):
        _fail(f"{spec['dockerfile']} contains a latest/unknown default")
    if "VITE_EYVAN_OIDC" in text or re.search(r"^\s*ENV\s+EYVAN_OIDC_", text, re.MULTILINE):
        _fail(f"{spec['dockerfile']} bakes environment-specific OIDC settings")
    lines = _logical_dockerfile_lines(text, spec["dockerfile"])
    args: dict[str, bool] = {}
    stages: set[str] = set()
    from_arguments: set[str] = set()
    for line in lines:
        upper = line.upper()
        if upper.startswith("ARG "):
            for token in line[4:].split():
                name, separator, _default = token.partition("=")
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                    _fail(f"{spec['dockerfile']} has malformed ARG {token!r}")
                args[name] = bool(separator)
        elif upper.startswith("FROM "):
            match = re.match(
                r"FROM\s+--platform=\$TARGETPLATFORM\s+\$\{([A-Z][A-Z0-9_]*)\}(?:\s+AS\s+([A-Za-z0-9_.-]+))?\Z",
                line,
                re.IGNORECASE,
            )
            if not match:
                _fail(f"{spec['dockerfile']} FROM must use a required locked image ARG and TARGETPLATFORM: {line}")
            from_arguments.add(match.group(1))
            if match.group(2):
                stages.add(match.group(2))
        elif upper.startswith("COPY ") and "--from=" in line:
            words = shlex.split(line)
            source = next((word.split("=", 1)[1] for word in words if word.startswith("--from=")), None)
            if source is None or source not in stages:
                _fail(f"{spec['dockerfile']} external COPY source is not a locked named stage: {line}")
    required = {
        *spec["buildArgs"],
        "EYVAN_REVISION",
        "EYVAN_SOURCE",
        "EYVAN_VERSION",
        "TARGETARCH",
        "TARGETOS",
        "TARGETPLATFORM",
    }
    missing = sorted(required - args.keys())
    defaulted = sorted(name for name in required if args.get(name))
    if missing or defaulted:
        _fail(
            f"{spec['dockerfile']} build args must be present without defaults; missing={missing}, defaulted={defaulted}"
        )
    if from_arguments != set(spec["buildArgs"]):
        _fail(
            f"{spec['dockerfile']} FROM arguments {sorted(from_arguments)!r} do not exactly match "
            f"locked build args {sorted(spec['buildArgs'])!r}"
        )
    required_labels = {
        'org.opencontainers.image.source="${EYVAN_SOURCE}"',
        'org.opencontainers.image.revision="${EYVAN_REVISION}"',
        'org.opencontainers.image.version="${EYVAN_VERSION}"',
        f'org.opencontainers.image.title="{spec["title"]}"',
        f'org.opencontainers.image.description="{spec["description"]}"',
        f'org.opencontainers.image.licenses="{LICENSE}"',
    }
    for label in required_labels:
        if label not in text:
            _fail(f"{spec['dockerfile']} is missing exact OCI label {label}")
    if 'test "$TARGETOS" = linux' not in text or 'test "$TARGETARCH" = amd64' not in text:
        _fail(f"{spec['dockerfile']} does not fail closed outside linux/amd64")
    expected_entrypoint = f'ENTRYPOINT ["{spec["entrypoint"]}"]'
    if expected_entrypoint not in text:
        _fail(f"{spec['dockerfile']} is missing {expected_entrypoint}")


def _validate_allowlist(path: Path, *, now: dt.datetime) -> dict[str, Any]:
    value = _exact_keys(_load_json(path, canonical=True), {"entries", "schemaVersion"}, "allowlist")
    if value["schemaVersion"] != ALLOWLIST_SCHEMA:
        _fail(f"allowlist schemaVersion must be {ALLOWLIST_SCHEMA!r}")
    entries = _sequence(value["entries"], "allowlist entries")
    identities: set[tuple[str, str, str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        context = f"allowlist entries[{index}]"
        item = _exact_keys(
            entry,
            {"expiresAt", "issue", "owner", "package", "purl", "reason", "role", "version", "vulnerability"},
            context,
        )
        role = _string(item["role"], f"{context}.role")
        if role not in ROLES:
            _fail(f"{context}.role must be one of {ROLES!r}")
        vulnerability = _string(item["vulnerability"], f"{context}.vulnerability", maximum=200)
        package = _string(item["package"], f"{context}.package", maximum=500)
        purl = _string(item["purl"], f"{context}.purl", maximum=1000)
        version = _string(item["version"], f"{context}.version", maximum=500)
        for field, text in (
            ("vulnerability", vulnerability),
            ("package", package),
            ("purl", purl),
            ("version", version),
        ):
            if FORBIDDEN_WILDCARD_RE.search(text) or text.lower() in {"all", "any", "unknown", "*"}:
                _fail(f"{context}.{field} contains a wildcard or non-exact placeholder")
        if not vulnerability.startswith(("CVE-", "GHSA-")):
            _fail(f"{context}.vulnerability must be an exact CVE or GHSA identifier")
        if not purl.startswith("pkg:"):
            _fail(f"{context}.purl must be an exact package URL")
        owner = _string(item["owner"], f"{context}.owner", maximum=200)
        if FORBIDDEN_WILDCARD_RE.search(owner) or owner.lower() in {"any", "unknown", "security-team"}:
            _fail(f"{context}.owner must name one accountable owner without wildcards")
        issue = _https(item["issue"], f"{context}.issue")
        if "/issues/" not in issue and "/security/" not in issue:
            _fail(f"{context}.issue must link a concrete issue or security advisory")
        reason = _string(item["reason"], f"{context}.reason", minimum=20, maximum=1000)
        if reason.lower() in {"accepted", "temporary exception", "false positive", "n/a"}:
            _fail(f"{context}.reason must be specific, not a placeholder")
        expires = _utc(item["expiresAt"], f"{context}.expiresAt")
        if expires <= now:
            _fail(f"{context}.expiresAt is stale")
        if expires > now + dt.timedelta(days=30):
            _fail(f"{context}.expiresAt exceeds the maximum 30-day exception window")
        identity = (role, vulnerability, package, purl, version)
        if identity in identities:
            _fail(f"{context} duplicates an existing exact exception")
        identities.add(identity)
        normalized.append(item)
    if normalized != sorted(
        normalized,
        key=lambda item: (item["role"], item["vulnerability"], item["purl"], item["version"]),
    ):
        _fail("allowlist entries must be sorted by role, vulnerability, purl, and version")
    return value


def _validate_scanner_metadata(value: Any, lock: dict[str, Any], *, now: dt.datetime) -> dict[str, Any]:
    obj = _exact_keys(
        value,
        {"artifactSha256", "binarySha256", "generatedAt", "name", "schemaVersion", "version"},
        "scanner metadata",
    )
    if obj["schemaVersion"] != SCAN_METADATA_SCHEMA or obj["name"] != "trivy":
        _fail("scanner metadata must identify the locked Trivy scanner schema")
    if obj["version"] != lock["tools"]["trivy"]["version"]:
        _fail("scanner metadata version does not match the release lock")
    if _hex(obj["artifactSha256"], "scanner metadata artifactSha256") != lock["tools"]["trivy"]["sha256"]:
        _fail("scanner artifact hash does not match the release lock")
    _hex(obj["binarySha256"], "scanner metadata binarySha256")
    generated = _utc(obj["generatedAt"], "scanner metadata generatedAt")
    if generated > now + dt.timedelta(minutes=5) or generated < now - dt.timedelta(hours=24):
        _fail("scanner metadata generatedAt is outside the 24-hour gate window")
    return obj


def _validate_db_metadata(value: Any, *, now: dt.datetime) -> dict[str, Any]:
    obj = _exact_keys(
        value,
        {"digest", "downloadedAt", "nextUpdate", "repository", "schemaVersion", "updatedAt", "version"},
        "vulnerability DB metadata",
    )
    if obj["schemaVersion"] != DB_METADATA_SCHEMA:
        _fail(f"vulnerability DB metadata schemaVersion must be {DB_METADATA_SCHEMA!r}")
    if obj["repository"] != "ghcr.io/aquasecurity/trivy-db:2":
        _fail("vulnerability DB metadata repository must identify schema-2 Trivy DB")
    if obj["version"] != 2:
        _fail("vulnerability DB metadata version must be integer 2")
    _digest(obj["digest"], "vulnerability DB metadata digest")
    downloaded = _utc(obj["downloadedAt"], "vulnerability DB metadata downloadedAt")
    updated = _utc(obj["updatedAt"], "vulnerability DB metadata updatedAt")
    next_update = _utc(obj["nextUpdate"], "vulnerability DB metadata nextUpdate")
    if downloaded > now + dt.timedelta(minutes=5) or downloaded < now - dt.timedelta(hours=24):
        _fail("vulnerability DB must have been downloaded in the last 24 hours")
    if updated > now + dt.timedelta(minutes=5) or updated < now - dt.timedelta(days=7):
        _fail("vulnerability DB updatedAt is outside the seven-day freshness window")
    if next_update <= updated:
        _fail("vulnerability DB nextUpdate must be later than updatedAt")
    return obj


def _finding_identity(finding: dict[str, Any], context: str) -> tuple[str, str, str, str]:
    vulnerability = _string(finding.get("VulnerabilityID"), f"{context}.VulnerabilityID", maximum=200)
    package = _string(finding.get("PkgName"), f"{context}.PkgName", maximum=500)
    version = _string(finding.get("InstalledVersion"), f"{context}.InstalledVersion", maximum=500)
    identifier = _mapping(finding.get("PkgIdentifier"), f"{context}.PkgIdentifier")
    purl = _string(identifier.get("PURL"), f"{context}.PkgIdentifier.PURL", maximum=1000)
    if not purl.startswith("pkg:"):
        _fail(f"{context}.PkgIdentifier.PURL is not an exact package URL")
    return vulnerability, package, purl, version


def _gate_scan(args: argparse.Namespace) -> None:
    # This single second-precision instant is both the policy decision clock and
    # the replay anchor recorded in the disposition.
    now = _parse_now(args.now).replace(microsecond=0)
    lock = _validate_release_lock(args.lock)
    allowlist = _validate_allowlist(args.allowlist, now=now)
    _validate_scanner_metadata(_load_json(args.scanner_metadata), lock, now=now)
    _validate_db_metadata(_load_json(args.database_metadata), now=now)
    report = _mapping(_load_json(args.scan), "Trivy report")
    schema_version = report.get("SchemaVersion")
    if schema_version != 2:
        _fail("Trivy report SchemaVersion must be integer 2")
    artifact_name = _string(report.get("ArtifactName"), "Trivy report ArtifactName")
    if args.image_ref is not None and artifact_name != args.image_ref:
        _fail(f"Trivy report scanned {artifact_name!r}, expected {args.image_ref!r}")
    if report.get("ArtifactType") not in {"container_image", "image"}:
        _fail("Trivy report ArtifactType must identify a container image")
    created = _rfc3339_utc(report.get("CreatedAt"), "Trivy report CreatedAt")
    if created > now + dt.timedelta(minutes=5) or created < now - dt.timedelta(hours=24):
        _fail("Trivy report CreatedAt is outside the 24-hour gate window")
    results = _sequence(report.get("Results"), "Trivy report Results")
    if not results:
        _fail("Trivy report Results must contain at least one analyzed target")

    applicable: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for entry in allowlist["entries"]:
        if entry["role"] == args.role:
            applicable[(entry["role"], entry["vulnerability"], entry["package"], entry["purl"], entry["version"])] = (
                entry
            )
    used: set[tuple[str, str, str, str, str]] = set()
    unmatched: list[tuple[str, str, str, str, str]] = []
    high = critical = allowed = 0
    seen_findings: set[tuple[str, str, str, str, str]] = set()
    for result_index, result_value in enumerate(results):
        result_context = f"Trivy report Results[{result_index}]"
        result = _mapping(result_value, result_context)
        _string(result.get("Target"), f"{result_context}.Target")
        # The locked workflow runs `trivy image --scanners vuln`: only OS and
        # language-package targets establish a supported vulnerability analysis.
        target_class = _string(result.get("Class"), f"{result_context}.Class")
        if target_class not in {"os-pkgs", "lang-pkgs"}:
            _fail(f"{result_context}.Class must identify an OS or language-package target")
        _string(result.get("Type"), f"{result_context}.Type")
        vulnerabilities = result.get("Vulnerabilities")
        if vulnerabilities is None:
            continue
        for finding_index, finding_value in enumerate(_sequence(vulnerabilities, "Trivy vulnerabilities")):
            context = f"Trivy report Results[{result_index}].Vulnerabilities[{finding_index}]"
            finding = _mapping(finding_value, context)
            severity = _string(finding.get("Severity"), f"{context}.Severity")
            if severity not in {"UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                _fail(f"{context}.Severity is not a supported Trivy severity")
            if severity not in {"HIGH", "CRITICAL"}:
                continue
            vulnerability, package, purl, version = _finding_identity(finding, context)
            identity = (args.role, vulnerability, package, purl, version)
            if identity in seen_findings:
                continue
            seen_findings.add(identity)
            if severity == "HIGH":
                high += 1
            else:
                critical += 1
            if identity in applicable:
                allowed += 1
                used.add(identity)
            else:
                unmatched.append(identity)
    unused = sorted(set(applicable) - used)
    disposition = {
        "allowlistSha256": _sha256(args.allowlist),
        "counts": {
            "allowed": allowed,
            "critical": critical,
            "high": high,
            "unmatched": len(unmatched),
        },
        "databaseMetadataSha256": _sha256(args.database_metadata),
        "evaluatedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image": artifact_name,
        "rawScanSha256": _sha256(args.scan),
        "role": args.role,
        "scannerMetadataSha256": _sha256(args.scanner_metadata),
        "schemaVersion": DISPOSITION_SCHEMA,
        "status": "pass" if not unmatched and not unused else "fail",
    }
    if args.output is not None:
        _write_canonical(args.output, disposition)
    if unmatched:
        rendered = ", ".join(f"{item[1]} {item[3]} {item[4]}" for item in unmatched[:20])
        _fail(
            f"{args.role} scan has {len(unmatched)} unmatched High/Critical finding(s), including unfixed findings: "
            f"{rendered}"
        )
    if unused:
        rendered = ", ".join(f"{item[1]} {item[3]} {item[4]}" for item in unused[:20])
        _fail(f"{args.role} scan has {len(unused)} unused allowlist entry/entries: {rendered}")
    print(f"{args.role}: scan gate passed ({high} High, {critical} Critical, {allowed} exact exception(s), 0 unmatched)")


def _image_config(value: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(value, list):
        if len(value) != 1:
            _fail("docker image metadata list must contain exactly one image")
        value = value[0]
    obj = _mapping(value, "image metadata")
    architecture = obj.get("architecture", obj.get("Architecture"))
    operating_system = obj.get("os", obj.get("Os"))
    config = obj.get("config", obj.get("Config"))
    return (
        _string(architecture, "image architecture"),
        _string(operating_system, "image OS"),
        _mapping(config, "image config"),
    )


def _verify_image_metadata(args: argparse.Namespace) -> None:
    lock = _validate_release_lock(args.lock)
    value = _load_json(args.metadata)
    architecture, operating_system, config = _image_config(value)
    if (operating_system, architecture) != ("linux", "amd64"):
        _fail(f"image platform is {operating_system}/{architecture}, expected linux/amd64")
    spec = lock["images"][args.role]
    labels = _mapping(config.get("Labels"), "image config Labels")
    required = {
        "org.opencontainers.image.description": spec["description"],
        "org.opencontainers.image.licenses": LICENSE,
        "org.opencontainers.image.revision": _full_sha(args.revision, "--revision"),
        "org.opencontainers.image.source": lock["sourceRepository"],
        "org.opencontainers.image.title": spec["title"],
        "org.opencontainers.image.version": _version(args.version, "--version"),
    }
    for name, expected in required.items():
        if labels.get(name) != expected:
            _fail(f"image label {name!r} is {labels.get(name)!r}, expected {expected!r}")
    entrypoint = config.get("Entrypoint")
    if entrypoint != [spec["entrypoint"]]:
        _fail(f"image entrypoint is {entrypoint!r}, expected {[spec['entrypoint']]!r}")
    env = config.get("Env", [])
    if not isinstance(env, list) or any(not isinstance(item, str) for item in env):
        _fail("image config Env must be an array of strings")
    forbidden = [item for item in env if item.startswith(("EYVAN_OIDC_", "VITE_EYVAN_OIDC_"))]
    if forbidden:
        _fail(f"image bakes environment-specific OIDC configuration: {forbidden!r}")
    print(f"{args.role}: image metadata and linux/amd64 platform verified")


def _rendered_image_values(text: str) -> list[str]:
    """Read image fields in block-style Helm YAML, refusing unsupported syntax.

    This standalone stdlib verifier is deliberately not a general YAML parser.
    Flow mappings, aliases, tags and multiline quoted scalars must be expanded
    to block mappings before verification. Scalar flow lists used for command
    arguments are supported; block-scalar data is never deployment evidence.
    Known workload kinds must precede their pod specs; workload Lists and
    duplicate mapping keys are refused rather than ambiguously interpreted.
    """
    found: list[str] = []
    stack: list[tuple[int, str]] = []
    seen_keys: set[tuple[int, tuple[str, ...], str]] = set()
    block_indent: int | None = None
    worker_env: tuple[int, bool] | None = None
    kind: str | None = None
    container_keys = {"containers", "initContainers", "ephemeralContainers"}
    pod_spec_paths = {
        "Pod": ("spec",),
        "PodTemplate": ("template", "spec"),
        "Deployment": ("spec", "template", "spec"),
        "DaemonSet": ("spec", "template", "spec"),
        "StatefulSet": ("spec", "template", "spec"),
        "ReplicaSet": ("spec", "template", "spec"),
        "ReplicationController": ("spec", "template", "spec"),
        "Job": ("spec", "template", "spec"),
        "CronJob": ("spec", "jobTemplate", "spec", "template", "spec"),
    }

    def scalar(value: str) -> str:
        match = re.fullmatch(r'("(?:[^"\\]|\\.)*"|\'(?:[^\']|\'\')*\'|[^\s].*?)(?:\s+#.*)?', value)
        if match is None:
            _fail("unsupported rendered YAML scalar; use single-line block mapping values")
        value = match[1]
        if value.startswith('"'):
            try:
                decoded = json.loads(value)
            except ValueError:
                _fail("unsupported rendered YAML quoted scalar")
            return _string(decoded, "rendered YAML scalar", minimum=0)
        if value.startswith("'"):
            return value[1:-1].replace("''", "'")
        if value[0] in "[{}&*!>|%@`" or ": " in value:
            _fail("unsupported rendered YAML syntax; expand flow mappings, aliases and tags to block mappings")
        return value

    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped or stripped.startswith("#"):
            continue
        if block_indent is not None:
            if indent > block_indent:
                continue
            block_indent = None
        if "\t" in line[: indent + 1]:
            _fail(f"unsupported rendered YAML indentation at line {line_number}")
        if stripped in {"---", "..."}:
            if worker_env is not None and not worker_env[1]:
                _fail("native-worker image environment must have a literal value")
            stack.clear()
            seen_keys.clear()
            worker_env = None
            kind = None
            continue
        sequence_item = stripped.startswith("- ")
        if sequence_item:
            stripped = stripped[2:].lstrip()
            indent += 2
            # Each sequence item is a fresh mapping, not a duplicate of the
            # preceding container/env/port item at the same indentation.
            seen_keys = {identity for identity in seen_keys if identity[0] < indent}
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [key for _, key in stack]
        pod_spec_path = pod_spec_paths.get(kind or "")
        in_container = (
            pod_spec_path is not None
            and tuple(path[: len(pod_spec_path)]) == pod_spec_path
            and len(path) > len(pod_spec_path)
            and path[len(pod_spec_path)] in container_keys
        )
        in_env = in_container and len(path) == len(pod_spec_path) + 2 and path[-1] == "env"
        if worker_env is not None and (not in_env or sequence_item):
            if not worker_env[1]:
                _fail("native-worker image environment must have a literal value")
            worker_env = None
        match = re.fullmatch(r'("(?:[^"\\]|\\.)*"|\'(?:[^\']|\'\')*\'|[A-Za-z0-9_.-]+):(?:\s+(.*)|\s*)', stripped)
        if match is None:
            # Plain scalar sequence entries (e.g. tolerations/arguments) cannot
            # introduce hidden container mappings. Anything else is refused.
            if not sequence_item or in_env or (path and path[-1] in container_keys):
                _fail(f"unsupported rendered YAML mapping at line {line_number}")
            scalar(stripped)
            continue
        key = scalar(match[1])
        identity = (indent, tuple(path), key)
        if identity in seen_keys:
            _fail(f"duplicate rendered YAML mapping key at line {line_number}: {key!r}")
        seen_keys.add(identity)
        value = (match[2] or "").strip()
        if key in container_keys and tuple(path) != pod_spec_path and kind not in {"ConfigMap", "Secret"}:
            _fail("unsupported rendered container location; use a known workload kind before its block-style pod spec")
        if not value or value.startswith("#"):
            if in_env and key == "valueFrom" and worker_env is not None:
                _fail("native-worker image environment must have a literal value")
            stack.append((indent, key))
            continue
        if re.fullmatch(r"[|>][+-]?(?:\s+#.*)?", value):
            if (key == "image" and in_container) or (in_env and key == "value"):
                _fail("deployment image references must be single-line literal scalars")
            block_indent = indent
            continue
        if value in {"{}", "[]"}:
            continue
        if value.startswith("["):
            try:
                items = json.loads(value)
            except ValueError:
                _fail("unsupported rendered YAML flow syntax; use block mappings")
            if (
                key in container_keys
                or not isinstance(items, list)
                or any(isinstance(item, (dict, list)) for item in items)
            ):
                _fail("unsupported rendered YAML flow mappings; use block mappings")
            continue
        parsed = scalar(value)
        if indent == 0 and key == "kind":
            if kind is not None or parsed.endswith("List"):
                _fail("unsupported duplicate kind or workload list; render separate block-style documents")
            kind = parsed
        if in_container and len(path) == len(pod_spec_path) + 1 and key == "image":
            found.append(parsed)
        if in_env and key == "name" and parsed == "EYVAN_KUBERNETES_WORKER_IMAGE":
            worker_env = (indent, False)
        elif in_env and key == "value" and worker_env is not None and worker_env[0] == indent:
            found.append(parsed)
            worker_env = (indent, True)
    if worker_env is not None and not worker_env[1]:
        _fail("native-worker image environment must have a literal value")
    return found


def _verify_rendered_refs(args: argparse.Namespace) -> None:
    expected: dict[str, str] = {}
    for value in args.reference:
        role, separator, reference = value.partition("=")
        if not separator or role not in ROLES or role in expected:
            _fail("--reference must occur once per role as role=repository@sha256:digest")
        if not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", reference):
            _fail(f"rendered reference for {role} is not an immutable digest reference")
        expected[role] = reference
    if tuple(expected) != ROLES:
        _fail(f"--reference order and roles must be exactly {ROLES!r}")
    if len(set(expected.values())) != len(ROLES):
        _fail("control-plane, agent, and worker deployment references must be distinct")
    try:
        text = args.render.read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        _fail(f"cannot read rendered Helm output: {error}")

    found = _rendered_image_values(text)
    if not found:
        _fail("rendered Helm output contains no deployment image references")
    mutable = sorted({item for item in found if "@sha256:" not in item or ":latest" in item.lower()})
    if mutable:
        _fail(f"rendered Helm output contains mutable image reference(s): {mutable!r}")
    found_set = set(found)
    expected_set = set(expected.values())
    missing = sorted(expected_set - found_set)
    unexpected = sorted(found_set - expected_set)
    if missing or unexpected:
        _fail(f"rendered digest references mismatch: missing={missing}, unexpected={unexpected}")
    print("rendered Helm output uses exactly the three qualified digest references")


def _validate_disposition(value: Any, context: str) -> dict[str, Any]:
    obj = _exact_keys(
        value,
        {
            "allowlistSha256",
            "counts",
            "databaseMetadataSha256",
            "evaluatedAt",
            "image",
            "rawScanSha256",
            "role",
            "scannerMetadataSha256",
            "schemaVersion",
            "status",
        },
        context,
    )
    if obj["schemaVersion"] != DISPOSITION_SCHEMA:
        _fail(f"{context}.schemaVersion must be {DISPOSITION_SCHEMA!r}")
    if obj["role"] not in ROLES:
        _fail(f"{context}.role is invalid")
    _string(obj["image"], f"{context}.image")
    for name in ("allowlistSha256", "databaseMetadataSha256", "rawScanSha256", "scannerMetadataSha256"):
        _hex(obj[name], f"{context}.{name}")
    _utc(obj["evaluatedAt"], f"{context}.evaluatedAt")
    counts = _exact_keys(obj["counts"], {"allowed", "critical", "high", "unmatched"}, f"{context}.counts")
    for name, count in counts.items():
        _integer(count, f"{context}.counts.{name}")
    if counts["unmatched"] != 0 or obj["status"] != "pass":
        _fail(f"{context} is not a passing zero-unmatched disposition")
    if counts["allowed"] > counts["high"] + counts["critical"]:
        _fail(f"{context}.counts.allowed exceeds High+Critical findings")
    return obj


def _validate_material(value: Any, context: str) -> dict[str, str]:
    obj = _exact_keys(value, {"digest", "uri"}, context)
    uri = _string(obj["uri"], f"{context}.uri", maximum=2000)
    if uri.startswith("https://"):
        _https(uri, f"{context}.uri")
    elif not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+:[A-Za-z0-9_.-]+", uri):
        _fail(f"{context}.uri must be the source HTTPS URL or an exact tagged base-image URI")
    digest = _string(obj["digest"], f"{context}.digest")
    if not (FULL_SHA_RE.fullmatch(digest) or HEX_RE.fullmatch(digest)):
        _fail(f"{context}.digest must be a full Git SHA-1 or SHA-256")
    return {"digest": digest, "uri": uri}


def _validate_rekor(value: Any, context: str) -> dict[str, Any]:
    obj = _exact_keys(value, {"included", "integratedTime", "logId", "logIndex"}, context)
    if _boolean(obj["included"], f"{context}.included") is not True:
        _fail(f"{context}.included must be true")
    _integer(obj["integratedTime"], f"{context}.integratedTime", minimum=1)
    _hex(obj["logId"], f"{context}.logId")
    _integer(obj["logIndex"], f"{context}.logIndex", minimum=0)
    return obj


def _protobuf_integer(value: Any, context: str) -> int:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    return _integer(value, context)


def _attestation_predicate_from_bundle(
    path: Path,
    image: dict[str, Any],
    *,
    predicate_type: str,
    context: str,
) -> Any:
    bundle = _mapping(_load_json(path, maximum=MAX_ARTIFACT_BYTES), context)
    if bundle.get("mediaType") != "application/vnd.dev.sigstore.bundle.v0.3+json":
        _fail(f"{context} must use the Sigstore bundle v0.3 media type")
    envelope = _exact_keys(
        bundle.get("dsseEnvelope"), {"payload", "payloadType", "signatures"}, f"{context} DSSE envelope"
    )
    if envelope["payloadType"] != "application/vnd.in-toto+json":
        _fail(f"{context} DSSE payloadType must be application/vnd.in-toto+json")
    signatures = _sequence(envelope["signatures"], f"{context} DSSE signatures")
    if not signatures:
        _fail(f"{context} DSSE envelope must contain a signature")
    for index, signature_value in enumerate(signatures):
        signature = _mapping(signature_value, f"{context} DSSE signatures[{index}]")
        encoded_signature = _string(signature.get("sig"), f"{context} DSSE signatures[{index}].sig")
        try:
            base64.b64decode(encoded_signature, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            _fail(f"{context} DSSE signature is not valid base64: {error}")
    encoded_payload = _string(
        envelope["payload"], f"{context} DSSE payload", maximum=4 * ((MAX_DSSE_PAYLOAD_BYTES + 2) // 3)
    )
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
        if len(payload) > MAX_DSSE_PAYLOAD_BYTES:
            _fail(f"{context} DSSE payload exceeds the {MAX_DSSE_PAYLOAD_BYTES}-byte limit")
        statement = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, json.JSONDecodeError, base64.binascii.Error) as error:
        _fail(f"{context} DSSE payload is not strict base64-encoded JSON: {error}")
    statement = _exact_keys(
        statement,
        {"_type", "predicate", "predicateType", "subject"},
        f"{context} in-toto statement",
    )
    if statement["_type"] != "https://in-toto.io/Statement/v0.1":
        _fail(f"{context} must contain an in-toto Statement v0.1")
    if statement["predicateType"] != predicate_type:
        _fail(f"{context} predicateType does not match the receipt")
    subjects = _sequence(statement["subject"], f"{context} subjects")
    if len(subjects) != 1:
        _fail(f"{context} must contain exactly one image subject")
    subject = _exact_keys(subjects[0], {"digest", "name"}, f"{context} subject")
    if subject["name"] != image["repository"]:
        _fail(f"{context} subject repository does not match the receipt")
    digest = _exact_keys(subject["digest"], {"sha256"}, f"{context} subject digest")
    if digest["sha256"] != image["indexDigest"].removeprefix("sha256:"):
        _fail(f"{context} subject digest does not match the receipt image")
    return statement["predicate"]


def _rekor_from_bundle(path: Path) -> dict[str, Any]:
    bundle = _mapping(_load_json(path, maximum=MAX_ARTIFACT_BYTES), "Sigstore bundle")
    verification = bundle.get("verificationMaterial")
    if isinstance(verification, dict):
        if bundle.get("mediaType") != "application/vnd.dev.sigstore.bundle.v0.3+json":
            _fail("Sigstore bundle must use the v0.3 media type")
        entries = _sequence(verification.get("tlogEntries"), "Sigstore bundle tlogEntries")
        if len(entries) != 1:
            _fail("Sigstore bundle must contain exactly one Rekor transparency-log entry")
        entry = _mapping(entries[0], "Sigstore bundle tlogEntries[0]")
        if not entry.get("inclusionPromise") and not entry.get("inclusionProof"):
            _fail("Sigstore bundle Rekor entry has no inclusion promise or proof")
        log_id_obj = _mapping(entry.get("logId"), "Sigstore bundle Rekor logId")
        encoded = _string(log_id_obj.get("keyId"), "Sigstore bundle Rekor logId.keyId")
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            log_id = base64.b64decode(padded, altchars=b"-_", validate=True).hex()
        except (ValueError, base64.binascii.Error) as error:
            _fail(f"Sigstore bundle Rekor log ID is not valid base64: {error}")
        if not HEX_RE.fullmatch(log_id):
            _fail("Sigstore bundle Rekor log ID must decode to 32 bytes")
        result = {
            "included": True,
            "integratedTime": _protobuf_integer(entry.get("integratedTime"), "Rekor integratedTime"),
            "logId": log_id,
            "logIndex": _protobuf_integer(entry.get("logIndex"), "Rekor logIndex"),
        }
        return _validate_rekor(result, "Sigstore bundle Rekor entry")
    payload = bundle.get("Payload")
    if isinstance(payload, dict):
        if not bundle.get("SignedEntryTimestamp"):
            _fail("legacy Cosign bundle has no signed Rekor entry timestamp")
        result = {
            "included": True,
            "integratedTime": _protobuf_integer(payload.get("integratedTime"), "Rekor integratedTime"),
            "logId": _hex(payload.get("logID"), "Rekor logID"),
            "logIndex": _protobuf_integer(payload.get("logIndex"), "Rekor logIndex"),
        }
        return _validate_rekor(result, "legacy Cosign bundle Rekor entry")
    _fail("Sigstore bundle contains no supported Rekor verification material")


def _extract_rekor(args: argparse.Namespace) -> None:
    value = _rekor_from_bundle(args.bundle)
    _write_canonical(args.output, value)
    print(f"verified Rekor inclusion at log index {value['logIndex']}")


def _validate_attestation(value: Any, context: str, identity: dict[str, Any], *, predicate_type: str) -> dict[str, Any]:
    obj = _exact_keys(value, {"bundle", "issuer", "predicateType", "workflowIdentity"}, context)
    _artifact(obj["bundle"], f"{context}.bundle")
    if obj["issuer"] != identity["issuer"] or obj["workflowIdentity"] != identity["workflowIdentity"]:
        _fail(f"{context} does not bind the exact keyless release workflow identity")
    if obj["predicateType"] != predicate_type:
        _fail(f"{context}.predicateType must be {predicate_type!r}")
    return obj


def _validate_receipt(value: Any, lock: dict[str, Any]) -> dict[str, Any]:
    receipt = _exact_keys(
        value,
        {
            "deployment",
            "images",
            "packages",
            "policy",
            "qualification",
            "schemaVersion",
            "source",
            "workflowApproval",
        },
        "receipt",
    )
    if receipt["schemaVersion"] != RECEIPT_SCHEMA:
        _fail(f"receipt schemaVersion must be {RECEIPT_SCHEMA!r}")
    source = _exact_keys(receipt["source"], {"fullSha", "ref", "repository", "version"}, "receipt source")
    full_sha = _full_sha(source["fullSha"], "receipt source.fullSha")
    version = _version(source["version"], "receipt source.version")
    if source["repository"] != lock["sourceRepository"]:
        _fail("receipt source.repository does not match the release lock")
    if source["ref"] != "refs/heads/main":
        _fail("receipt source.ref must be exactly refs/heads/main")

    approval = _exact_keys(
        receipt["workflowApproval"],
        {
            "actor",
            "approved",
            "environment",
            "event",
            "reviewers",
            "runAttempt",
            "runId",
            "workflowIdentity",
        },
        "receipt workflowApproval",
    )
    _string(approval["actor"], "receipt workflowApproval.actor", maximum=200)
    if _boolean(approval["approved"], "receipt workflowApproval.approved") is not True:
        _fail("receipt workflowApproval.approved must be true")
    if approval["environment"] != "eyvan-oci-release" or approval["event"] != "workflow_dispatch":
        _fail("receipt workflow approval must bind the manual eyvan-oci-release environment lane")
    _integer(approval["runId"], "receipt workflowApproval.runId", minimum=1)
    _integer(approval["runAttempt"], "receipt workflowApproval.runAttempt", minimum=1)
    reviewers = _sequence(approval["reviewers"], "receipt workflowApproval.reviewers")
    if not reviewers:
        _fail("receipt workflowApproval.reviewers must record at least one environment approval")
    normalized_reviewers = []
    for reviewer_index, reviewer_value in enumerate(reviewers):
        reviewer = _exact_keys(
            reviewer_value,
            {"login"},
            f"receipt workflowApproval.reviewers[{reviewer_index}]",
        )
        login = _string(reviewer["login"], f"receipt workflowApproval.reviewers[{reviewer_index}].login", maximum=200)
        normalized_reviewers.append({"login": login})
    if normalized_reviewers != sorted(normalized_reviewers, key=lambda reviewer: reviewer["login"]):
        _fail("receipt workflowApproval.reviewers must be sorted by login")
    if len({reviewer["login"] for reviewer in normalized_reviewers}) != len(normalized_reviewers):
        _fail("receipt workflowApproval.reviewers must not contain duplicates")
    if approval["workflowIdentity"] != lock["identity"]["workflowIdentity"]:
        _fail("receipt workflowApproval.workflowIdentity is not the exact locked identity")

    policy = _exact_keys(receipt["policy"], {"allowlist", "releaseLock"}, "receipt policy")
    _artifact(policy["releaseLock"], "receipt policy.releaseLock")
    allowlist_artifact = _artifact(policy["allowlist"], "receipt policy.allowlist")

    images = _sequence(receipt["images"], "receipt images")
    if len(images) != 3:
        _fail("receipt must contain exactly three images")
    normalized_images: list[dict[str, Any]] = []
    seen_index_digests: set[str] = set()
    for index, image_value in enumerate(images):
        context = f"receipt images[{index}]"
        image = _exact_keys(
            image_value,
            {
                "baseReferences",
                "indexDigest",
                "labels",
                "platform",
                "platformDigest",
                "provenance",
                "repository",
                "role",
                "sbom",
                "scan",
                "signature",
            },
            context,
        )
        role = image["role"]
        if role != ROLES[index]:
            _fail(f"receipt image roles must be ordered exactly as {ROLES!r}")
        spec = lock["images"][role]
        if image["repository"] != spec["repository"] or image["platform"] != lock["platform"]:
            _fail(f"{context} repository/platform does not match the release lock")
        index_digest = _digest(image["indexDigest"], f"{context}.indexDigest")
        _digest(image["platformDigest"], f"{context}.platformDigest")
        if index_digest in seen_index_digests:
            _fail("the three image index digests must be distinct")
        seen_index_digests.add(index_digest)

        base_references = _sequence(image["baseReferences"], f"{context}.baseReferences")
        expected_references = [
            {"argument": argument, "reference": reference} for argument, reference in sorted(spec["buildArgs"].items())
        ]
        normalized_references = []
        for reference_index, reference_value in enumerate(base_references):
            reference = _exact_keys(
                reference_value,
                {"argument", "reference"},
                f"{context}.baseReferences[{reference_index}]",
            )
            _string(reference["argument"], f"{context}.baseReferences[{reference_index}].argument")
            _immutable_image_ref(reference["reference"], f"{context}.baseReferences[{reference_index}].reference")
            normalized_references.append(reference)
        if normalized_references != expected_references:
            _fail(f"{context}.baseReferences does not exactly match the release lock")

        labels = _exact_keys(
            image["labels"],
            {"description", "licenses", "revision", "source", "title", "version"},
            f"{context}.labels",
        )
        expected_labels = {
            "description": spec["description"],
            "licenses": LICENSE,
            "revision": full_sha,
            "source": lock["sourceRepository"],
            "title": spec["title"],
            "version": version,
        }
        if labels != expected_labels:
            _fail(f"{context}.labels do not exactly bind the release source and role")

        sbom = _exact_keys(image["sbom"], {"attestation", "document", "format"}, f"{context}.sbom")
        if sbom["format"] != "spdx-json":
            _fail(f"{context}.sbom.format must be 'spdx-json'")
        _artifact(sbom["document"], f"{context}.sbom.document")
        _validate_attestation(
            sbom["attestation"],
            f"{context}.sbom.attestation",
            lock["identity"],
            predicate_type="https://spdx.dev/Document",
        )

        scan = _exact_keys(
            image["scan"],
            {"allowlistSha256", "databaseMetadata", "disposition", "raw", "scannerMetadata"},
            f"{context}.scan",
        )
        if _hex(scan["allowlistSha256"], f"{context}.scan.allowlistSha256") != allowlist_artifact["sha256"]:
            _fail(f"{context}.scan.allowlistSha256 does not match receipt policy.allowlist")
        for field in ("databaseMetadata", "disposition", "raw", "scannerMetadata"):
            _artifact(scan[field], f"{context}.scan.{field}")

        signature = _exact_keys(
            image["signature"],
            {"bundle", "issuer", "rekor", "workflowIdentity"},
            f"{context}.signature",
        )
        _artifact(signature["bundle"], f"{context}.signature.bundle")
        if (
            signature["issuer"] != lock["identity"]["issuer"]
            or signature["workflowIdentity"] != lock["identity"]["workflowIdentity"]
        ):
            _fail(f"{context}.signature does not bind the exact keyless release workflow identity")
        _validate_rekor(signature["rekor"], f"{context}.signature.rekor")

        provenance = _exact_keys(
            image["provenance"],
            {"builderId", "bundle", "issuer", "materials", "predicate", "predicateType", "workflowIdentity"},
            f"{context}.provenance",
        )
        _artifact(provenance["bundle"], f"{context}.provenance.bundle")
        _artifact(provenance["predicate"], f"{context}.provenance.predicate")
        if provenance["predicateType"] != "https://slsa.dev/provenance/v1":
            _fail(f"{context}.provenance.predicateType must be SLSA provenance v1")
        if (
            provenance["issuer"] != lock["identity"]["issuer"]
            or provenance["workflowIdentity"] != lock["identity"]["workflowIdentity"]
            or provenance["builderId"] != lock["identity"]["provenanceBuilderId"]
        ):
            _fail(f"{context}.provenance does not bind the locked builder and keyless identity")
        materials = [
            _validate_material(material, f"{context}.provenance.materials[{material_index}]")
            for material_index, material in enumerate(
                _sequence(provenance["materials"], f"{context}.provenance.materials")
            )
        ]
        expected_materials = [{"digest": full_sha, "uri": lock["sourceRepository"]}]
        for reference in expected_references:
            parsed = IMAGE_REF_RE.fullmatch(reference["reference"])
            assert parsed is not None
            expected_materials.append(
                {
                    "digest": parsed.group("digest").removeprefix("sha256:"),
                    "uri": reference["reference"].split("@", 1)[0],
                }
            )
        expected_materials = sorted(expected_materials, key=lambda item: item["uri"])
        if materials != expected_materials:
            _fail(f"{context}.provenance.materials does not exactly bind source and base images")
        normalized_images.append(image)

    packages = _exact_keys(receipt["packages"], {"sdist", "wheel"}, "receipt packages")
    wheel = _artifact(packages["wheel"], "receipt packages.wheel")
    sdist = _artifact(packages["sdist"], "receipt packages.sdist")
    normalized_version = version.replace("-", "_")
    compatibility_tag = r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    wheel_pattern = (
        rf"eyvan-{re.escape(normalized_version)}-(?:[0-9][A-Za-z0-9_]*-)?"
        rf"{compatibility_tag}-{compatibility_tag}-{compatibility_tag}\.whl"
    )
    if re.fullmatch(wheel_pattern, wheel["name"]) is None:
        _fail("receipt wheel filename does not match the release version and wheel field structure")
    if sdist["name"] != f"eyvan-{version}.tar.gz":
        _fail("receipt sdist filename does not match the release version")

    deployment = _exact_keys(
        receipt["deployment"],
        {"chart", "deploymentApplied", "overlay", "references", "render"},
        "receipt deployment",
    )
    chart = _artifact(deployment["chart"], "receipt deployment.chart")
    _artifact(deployment["overlay"], "receipt deployment.overlay")
    _artifact(deployment["render"], "receipt deployment.render")
    if chart["name"] != f"eyvan-{version}.tgz":
        _fail("receipt chart filename does not match the release version")
    if _boolean(deployment["deploymentApplied"], "receipt deployment.deploymentApplied") is not False:
        _fail("receipt deploymentApplied must be explicitly false")
    references = _sequence(deployment["references"], "receipt deployment.references")
    expected_deployment_references = [
        {
            "reference": f"{image['repository']}@{image['indexDigest']}",
            "role": image["role"],
        }
        for image in normalized_images
    ]
    normalized_deployment_references = []
    for index, value in enumerate(references):
        reference = _exact_keys(value, {"reference", "role"}, f"receipt deployment.references[{index}]")
        _string(reference["reference"], f"receipt deployment.references[{index}].reference")
        normalized_deployment_references.append(reference)
    if normalized_deployment_references != expected_deployment_references:
        _fail("receipt deployment references do not exactly match the three image index digests")

    qualification = _exact_keys(
        receipt["qualification"],
        {"deployment", "hardware", "unverifiedPlatforms"},
        "receipt qualification",
    )
    if qualification["unverifiedPlatforms"] != lock["unverifiedPlatforms"]:
        _fail("receipt qualification.unverifiedPlatforms must name every locked unverified platform")
    hardware = _exact_keys(qualification["hardware"], {"evidence", "validated"}, "receipt qualification.hardware")
    if _boolean(hardware["validated"], "receipt qualification.hardware.validated") is not False:
        _fail("receipt hardware validation must be explicitly false")
    if hardware["evidence"] != "not-run":
        _fail("receipt hardware evidence must be the closed value 'not-run'")
    deployment_qualification = _exact_keys(
        qualification["deployment"], {"applied", "evidence"}, "receipt qualification.deployment"
    )
    if _boolean(deployment_qualification["applied"], "receipt qualification.deployment.applied") is not False:
        _fail("receipt qualification deployment applied must be explicitly false")
    if deployment_qualification["evidence"] != "not-applied":
        _fail("receipt deployment evidence must be the closed value 'not-applied'")
    return receipt


def _artifact_records(receipt: dict[str, Any]) -> list[dict[str, str]]:
    records = [receipt["policy"]["releaseLock"], receipt["policy"]["allowlist"]]
    for image in receipt["images"]:
        records.extend(
            [
                image["sbom"]["document"],
                image["sbom"]["attestation"]["bundle"],
                image["scan"]["raw"],
                image["scan"]["scannerMetadata"],
                image["scan"]["databaseMetadata"],
                image["scan"]["disposition"],
                image["signature"]["bundle"],
                image["provenance"]["predicate"],
                image["provenance"]["bundle"],
            ]
        )
    records.extend(
        [
            receipt["packages"]["wheel"],
            receipt["packages"]["sdist"],
            receipt["deployment"]["chart"],
            receipt["deployment"]["overlay"],
            receipt["deployment"]["render"],
        ]
    )
    return records


def _verify_spdx(path: Path, context: str) -> None:
    value = _mapping(_load_json(path), context)
    if not isinstance(value.get("spdxVersion"), str) or not value["spdxVersion"].startswith("SPDX-2."):
        _fail(f"{context} is not an SPDX JSON document")
    _string(value.get("documentNamespace"), f"{context}.documentNamespace")
    packages = value.get("packages")
    if not isinstance(packages, list) or not packages:
        _fail(f"{context} contains no packages")


def _validate_provenance_predicate(path: Path, image: dict[str, Any], lock: dict[str, Any], full_sha: str) -> None:
    context = f"{image['role']} provenance predicate"
    value = _exact_keys(_load_json(path), {"buildDefinition", "runDetails"}, context)
    build = _exact_keys(
        value["buildDefinition"],
        {"buildType", "externalParameters", "internalParameters", "resolvedDependencies"},
        f"{context}.buildDefinition",
    )
    if build["buildType"] != "https://github.com/Attestations/GitHubActionsWorkflow@v1":
        _fail(f"{context} has an unexpected buildType")
    external = _exact_keys(
        build["externalParameters"], {"role", "source"}, f"{context}.buildDefinition.externalParameters"
    )
    if external != {
        "role": image["role"],
        "source": {"digest": {"sha1": full_sha}, "uri": lock["sourceRepository"]},
    }:
        _fail(f"{context} external parameters do not exactly bind source and role")
    internal = _exact_keys(
        build["internalParameters"], {"githubRunAttempt", "githubRunId"}, f"{context}.buildDefinition.internalParameters"
    )
    _integer(internal["githubRunAttempt"], f"{context}.githubRunAttempt", minimum=1)
    _integer(internal["githubRunId"], f"{context}.githubRunId", minimum=1)
    dependencies = _sequence(build["resolvedDependencies"], f"{context}.resolvedDependencies")
    predicate_materials = []
    for index, dependency in enumerate(dependencies):
        item = _exact_keys(dependency, {"digest", "uri"}, f"{context}.resolvedDependencies[{index}]")
        digest = _mapping(item["digest"], f"{context}.resolvedDependencies[{index}].digest")
        if set(digest) == {"sha1"}:
            normalized_digest = _full_sha(digest["sha1"], f"{context}.resolvedDependencies[{index}].digest.sha1")
        elif set(digest) == {"sha256"}:
            normalized_digest = _hex(digest["sha256"], f"{context}.resolvedDependencies[{index}].digest.sha256")
        else:
            _fail(f"{context}.resolvedDependencies[{index}].digest must have exactly sha1 or sha256")
        predicate_materials.append(
            {
                "digest": normalized_digest,
                "uri": _string(item["uri"], f"{context}.resolvedDependencies[{index}].uri"),
            }
        )
    if predicate_materials != image["provenance"]["materials"]:
        _fail(f"{context} resolved dependencies do not match receipt materials")
    run_details = _exact_keys(value["runDetails"], {"builder", "metadata"}, f"{context}.runDetails")
    builder = _exact_keys(run_details["builder"], {"id"}, f"{context}.runDetails.builder")
    if builder["id"] != image["provenance"]["builderId"]:
        _fail(f"{context} builder does not match the receipt")
    metadata = _exact_keys(run_details["metadata"], {"invocationId"}, f"{context}.runDetails.metadata")
    _string(metadata["invocationId"], f"{context}.runDetails.metadata.invocationId")


def _verify_receipt_artifacts(receipt: dict[str, Any], lock: dict[str, Any], root: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        _fail(f"artifact root does not exist: {error}")
    records = _artifact_records(receipt)
    names = [record["name"] for record in records]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        _fail(f"receipt artifact filenames must be unique: {duplicates!r}")
    paths: dict[str, Path] = {}
    for record in records:
        candidate = resolved_root / record["name"]
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            _fail(f"missing receipt artifact {record['name']!r}: {error}")
        if resolved.parent != resolved_root or not resolved.is_file() or candidate.is_symlink():
            _fail(f"receipt artifact {record['name']!r} must be a regular file directly under artifact root")
        actual = _sha256(resolved)
        if actual != record["sha256"]:
            _fail(f"receipt artifact {record['name']!r} hash mismatch: {actual} != {record['sha256']}")
        paths[record["name"]] = resolved
    if paths[receipt["policy"]["releaseLock"]["name"]].read_bytes() != _canonical_bytes(lock):
        _fail("receipt release-lock artifact differs from the validated release lock")
    full_sha = receipt["source"]["fullSha"]
    for image in receipt["images"]:
        _verify_spdx(paths[image["sbom"]["document"]["name"]], f"{image['role']} SPDX SBOM")
        disposition_path = paths[image["scan"]["disposition"]["name"]]
        disposition = _validate_disposition(
            _load_json(disposition_path, canonical=True), f"{image['role']} scan disposition"
        )
        if disposition["role"] != image["role"]:
            _fail(f"{image['role']} disposition role mismatch")
        if disposition["image"] != f"{image['repository']}@{image['indexDigest']}":
            _fail(f"{image['role']} disposition image mismatch")
        expected_hashes = {
            "allowlistSha256": image["scan"]["allowlistSha256"],
            "databaseMetadataSha256": image["scan"]["databaseMetadata"]["sha256"],
            "rawScanSha256": image["scan"]["raw"]["sha256"],
            "scannerMetadataSha256": image["scan"]["scannerMetadata"]["sha256"],
        }
        for name, expected in expected_hashes.items():
            if disposition[name] != expected:
                _fail(f"{image['role']} disposition {name} does not bind the receipt artifact")
        historical_scan_time = disposition["evaluatedAt"]
        with tempfile.TemporaryDirectory(prefix="eyvan-receipt-scan-") as temporary:
            regenerated_path = Path(temporary) / "disposition.json"
            _gate_scan(
                argparse.Namespace(
                    allowlist=paths[receipt["policy"]["allowlist"]["name"]],
                    database_metadata=paths[image["scan"]["databaseMetadata"]["name"]],
                    image_ref=f"{image['repository']}@{image['indexDigest']}",
                    lock=paths[receipt["policy"]["releaseLock"]["name"]],
                    now=historical_scan_time,
                    output=regenerated_path,
                    role=image["role"],
                    scan=paths[image["scan"]["raw"]["name"]],
                    scanner_metadata=paths[image["scan"]["scannerMetadata"]["name"]],
                )
            )
            if regenerated_path.read_bytes() != disposition_path.read_bytes():
                _fail(f"{image['role']} disposition does not exactly match a fresh scan evaluation")
        bundle_rekor = _rekor_from_bundle(paths[image["signature"]["bundle"]["name"]])
        if bundle_rekor != image["signature"]["rekor"]:
            _fail(f"{image['role']} receipt Rekor record does not match its Cosign bundle")
        sbom_bundle_path = paths[image["sbom"]["attestation"]["bundle"]["name"]]
        provenance_bundle_path = paths[image["provenance"]["bundle"]["name"]]
        _rekor_from_bundle(sbom_bundle_path)
        _rekor_from_bundle(provenance_bundle_path)
        sbom_document = _load_json(paths[image["sbom"]["document"]["name"]])
        if (
            _attestation_predicate_from_bundle(
                sbom_bundle_path,
                image,
                predicate_type="https://spdx.dev/Document",
                context=f"{image['role']} SBOM bundle",
            )
            != sbom_document
        ):
            _fail(f"{image['role']} SBOM bundle predicate does not match its SPDX document")
        provenance_path = paths[image["provenance"]["predicate"]["name"]]
        provenance_predicate = _load_json(provenance_path)
        if (
            _attestation_predicate_from_bundle(
                provenance_bundle_path,
                image,
                predicate_type="https://slsa.dev/provenance/v1",
                context=f"{image['role']} provenance bundle",
            )
            != provenance_predicate
        ):
            _fail(f"{image['role']} provenance bundle predicate does not match its predicate artifact")
        _validate_provenance_predicate(provenance_path, image, lock, full_sha)
    _verify_rendered_refs(
        argparse.Namespace(
            render=paths[receipt["deployment"]["render"]["name"]],
            reference=[f"{image['role']}={image['repository']}@{image['indexDigest']}" for image in receipt["images"]],
        )
    )


def _emit_receipt(args: argparse.Namespace) -> None:
    lock = _validate_release_lock(args.lock)
    value = _load_json(args.input)
    receipt = _validate_receipt(value, lock)
    if args.artifact_root is not None:
        _verify_receipt_artifacts(receipt, lock, args.artifact_root)
    _write_canonical(args.output, receipt)
    print(f"wrote canonical Eyvan OCI receipt: {args.output}")


def _verify_receipt(args: argparse.Namespace) -> None:
    lock = _validate_release_lock(args.lock)
    receipt = _validate_receipt(_load_json(args.receipt, canonical=True), lock)
    if args.expected_sha is not None and receipt["source"]["fullSha"] != args.expected_sha:
        _fail("receipt source SHA does not match --expected-sha")
    if args.expected_ref is not None and receipt["source"]["ref"] != args.expected_ref:
        _fail("receipt source ref does not match --expected-ref")
    if args.expected_version is not None and receipt["source"]["version"] != args.expected_version:
        _fail("receipt version does not match --expected-version")
    if args.expected_workflow_identity is not None and (
        receipt["workflowApproval"]["workflowIdentity"] != args.expected_workflow_identity
    ):
        _fail("receipt workflow identity does not match --expected-workflow-identity")
    if args.artifact_root is not None:
        _verify_receipt_artifacts(receipt, lock, args.artifact_root)
    if args.asset_manifest_output is not None:
        records = _artifact_records(receipt)
        if len(records) != 34 or len({record["name"] for record in records}) != 34:
            _fail("receipt must resolve to exactly 34 unique closed release artifacts")
        _write_canonical(
            args.asset_manifest_output,
            {"artifacts": records, "schemaVersion": ASSET_MANIFEST_SCHEMA},
        )
    print("canonical Eyvan OCI release receipt verified")


def _validate_policy(args: argparse.Namespace) -> None:
    now = _parse_now(args.now)
    repository_root = args.repository_root.resolve() if args.repository_root is not None else None
    _validate_release_lock(args.lock, repository_root=repository_root)
    allowlist = _validate_allowlist(args.allowlist, now=now)
    print(f"release policy verified; {len(allowlist['entries'])} exact High/Critical exception(s)")


def _capture_scanner_metadata(args: argparse.Namespace) -> None:
    lock = _validate_release_lock(args.lock)
    expected = lock["tools"]["trivy"]
    actual_hash = _sha256(args.binary)
    try:
        completed = subprocess.run(
            [str(args.binary), "--version"], check=True, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as error:
        _fail(f"cannot execute locked Trivy binary: {error}")
    first_line = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    match = re.fullmatch(r"Version:\s*v?([^\s]+)", first_line)
    if match is None or match.group(1) != expected["version"]:
        _fail(f"Trivy binary reports {first_line!r}, expected Version: {expected['version']}")
    now = _parse_now(args.now).replace(microsecond=0)
    value = {
        "artifactSha256": expected["sha256"],
        "binarySha256": actual_hash,
        "generatedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "name": "trivy",
        "schemaVersion": SCAN_METADATA_SCHEMA,
        "version": expected["version"],
    }
    _write_canonical(args.output, value)
    print(f"captured locked Trivy {expected['version']} metadata")


def _normalized_external_time(value: Any, context: str) -> str:
    parsed = _rfc3339_utc(value, context).replace(microsecond=0)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _capture_db_metadata(args: argparse.Namespace) -> None:
    raw = _mapping(_load_json(args.trivy_metadata), "Trivy DB cache metadata")
    required = {"Version", "NextUpdate", "UpdatedAt", "DownloadedAt"}
    if not required.issubset(raw):
        _fail(f"Trivy DB cache metadata is missing {sorted(required - set(raw))!r}")
    if raw["Version"] != 2:
        _fail("Trivy DB cache metadata Version must be integer 2")
    value = {
        "digest": _digest(args.digest, "--digest"),
        "downloadedAt": _normalized_external_time(raw["DownloadedAt"], "Trivy DB DownloadedAt"),
        "nextUpdate": _normalized_external_time(raw["NextUpdate"], "Trivy DB NextUpdate"),
        "repository": "ghcr.io/aquasecurity/trivy-db:2",
        "schemaVersion": DB_METADATA_SCHEMA,
        "updatedAt": _normalized_external_time(raw["UpdatedAt"], "Trivy DB UpdatedAt"),
        "version": 2,
    }
    _write_canonical(args.output, value)
    print(f"captured Trivy DB snapshot {value['digest']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-policy", help="validate canonical lock, allowlist, and Dockerfiles")
    validate.add_argument("--lock", type=Path, required=True)
    validate.add_argument("--allowlist", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path)
    validate.add_argument("--now", help="test-only/current UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form")
    validate.set_defaults(function=_validate_policy)

    scan = subparsers.add_parser("gate-scan", help="gate one raw Trivy JSON report")
    scan.add_argument("--lock", type=Path, required=True)
    scan.add_argument("--allowlist", type=Path, required=True)
    scan.add_argument("--role", choices=ROLES, required=True)
    scan.add_argument("--scan", type=Path, required=True)
    scan.add_argument("--scanner-metadata", type=Path, required=True)
    scan.add_argument("--database-metadata", type=Path, required=True)
    scan.add_argument("--image-ref")
    scan.add_argument("--output", type=Path)
    scan.add_argument("--now", help="test-only/current UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form")
    scan.set_defaults(function=_gate_scan)

    image = subparsers.add_parser(
        "verify-image-metadata", help="verify platform, entrypoint, labels, and OIDC neutrality"
    )
    image.add_argument("--lock", type=Path, required=True)
    image.add_argument("--role", choices=ROLES, required=True)
    image.add_argument("--metadata", type=Path, required=True)
    image.add_argument("--revision", required=True)
    image.add_argument("--version", required=True)
    image.set_defaults(function=_verify_image_metadata)

    render = subparsers.add_parser("verify-rendered-digest-refs", help="verify exact image refs in rendered Helm YAML")
    render.add_argument("--render", type=Path, required=True)
    render.add_argument("--reference", action="append", required=True)
    render.set_defaults(function=_verify_rendered_refs)

    scanner_metadata = subparsers.add_parser(
        "capture-scanner-metadata", help="capture the locked Trivy binary version and hashes"
    )
    scanner_metadata.add_argument("--lock", type=Path, required=True)
    scanner_metadata.add_argument("--binary", type=Path, required=True)
    scanner_metadata.add_argument("--output", type=Path, required=True)
    scanner_metadata.add_argument("--now", help="test-only/current UTC timestamp")
    scanner_metadata.set_defaults(function=_capture_scanner_metadata)

    db_metadata = subparsers.add_parser(
        "capture-db-metadata", help="normalize exact Trivy DB cache metadata and OCI digest"
    )
    db_metadata.add_argument("--trivy-metadata", type=Path, required=True)
    db_metadata.add_argument("--digest", required=True)
    db_metadata.add_argument("--output", type=Path, required=True)
    db_metadata.set_defaults(function=_capture_db_metadata)

    rekor = subparsers.add_parser("extract-rekor", help="extract and validate one Rekor entry from a Cosign bundle")
    rekor.add_argument("--bundle", type=Path, required=True)
    rekor.add_argument("--output", type=Path, required=True)
    rekor.set_defaults(function=_extract_rekor)

    emit = subparsers.add_parser("emit-receipt", help="validate and emit a canonical closed-schema receipt")
    emit.add_argument("--lock", type=Path, required=True)
    emit.add_argument("--input", type=Path, required=True)
    emit.add_argument("--output", type=Path, required=True)
    emit.add_argument("--artifact-root", type=Path)
    emit.set_defaults(function=_emit_receipt)

    verify = subparsers.add_parser("verify-receipt", help="verify a canonical closed-schema receipt")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path)
    verify.add_argument("--asset-manifest-output", type=Path)
    verify.add_argument("--expected-sha")
    verify.add_argument("--expected-ref")
    verify.add_argument("--expected-version")
    verify.add_argument("--expected-workflow-identity")
    verify.set_defaults(function=_verify_receipt)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
