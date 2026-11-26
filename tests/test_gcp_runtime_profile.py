# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for Eyvan's owner-only native GCP runtime profile."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from eyvan.capacity.providers.gcp.runtime.profile import (
    GCP_NATIVE_RUNTIME_PROFILE_SCHEMA,
    GCP_NATIVE_RUNTIME_PROFILE_VERSION,
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileError,
    GcpNativeRuntimeProfileStore,
)

_WHEEL_SHA256 = "a" * 64
_WHEEL_FILENAME = "eyvan-0.1.0-py3-none-any.whl"
_WHEEL_URI = f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{_WHEEL_SHA256}/{_WHEEL_FILENAME}"


def _wheel_uri(artifact_root_uri: str) -> str:
    """Return one valid staged wheel URI below a supplied artifact root."""

    return f"{artifact_root_uri.rstrip('/')}/runtime/wheels/{_WHEEL_SHA256}/{_WHEEL_FILENAME}"


def _profile(*, artifact_root_uri: str = "gs://eyvan-runtime-artifacts/eyvan") -> GcpNativeRuntimeProfile:
    """Build one explicit profile for replacement-write coverage."""

    return GcpNativeRuntimeProfile(
        project_id="example-project-1",
        artifact_root_uri=artifact_root_uri,
        agent_shared_secret="a" * 32,
        native_worker_root_secret="b" * 32,
        native_worker_wheel_uri=_wheel_uri(artifact_root_uri),
    )


def test_runtime_profile_round_trips_with_private_generated_secrets(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "gcp-native-runtime.json"
    store = GcpNativeRuntimeProfileStore(path)

    created = store.create(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan/",
        native_worker_wheel_uri=_WHEEL_URI,
    )
    loaded = store.load()
    document = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == created
    assert created.agent_shared_secret != created.native_worker_root_secret
    assert len(created.agent_shared_secret) >= 32
    assert len(created.native_worker_root_secret) >= 32
    assert created.agent_shared_secret not in repr(created)
    assert created.native_worker_root_secret not in repr(created)
    assert created.artifact_root_uri == "gs://eyvan-runtime-artifacts/eyvan"
    assert document["schema"] == GCP_NATIVE_RUNTIME_PROFILE_SCHEMA
    assert document["version"] == GCP_NATIVE_RUNTIME_PROFILE_VERSION
    assert document["native_worker_wheel_uri"] == created.native_worker_wheel_uri
    assert created.fleet_state_uri == "gs://eyvan-runtime-artifacts/eyvan/control-plane/fleet-state.json"


def test_runtime_profile_derives_a_stable_opaque_public_worker_ingress_tag() -> None:
    profile = _profile()
    rotated_worker_secret = GcpNativeRuntimeProfile(
        project_id=profile.project_id,
        artifact_root_uri=profile.artifact_root_uri,
        agent_shared_secret=profile.agent_shared_secret,
        native_worker_root_secret="c" * 32,
        native_worker_wheel_uri=profile.native_worker_wheel_uri,
    )

    assert profile.native_worker_public_ingress_tag == profile.native_worker_public_ingress_tag
    assert profile.native_worker_public_ingress_tag != rotated_worker_secret.native_worker_public_ingress_tag
    assert profile.native_worker_root_secret not in profile.native_worker_public_ingress_tag


def test_runtime_profile_save_replaces_only_after_validating_the_document(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "gcp-native-runtime.json"
    store = GcpNativeRuntimeProfileStore(path)
    store.create(project_id="example-project-1", artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan")
    replacement = _profile(artifact_root_uri="gs://eyvan-runtime-artifacts/next")

    saved_path = store.save(replacement)

    assert saved_path == path
    assert store.load() == replacement
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_runtime_profile_load_or_create_rejects_a_different_requested_runtime(tmp_path: Path) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "profile" / "gcp-native-runtime.json")
    store.load_or_create(project_id="example-project-1", artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan")

    with pytest.raises(GcpNativeRuntimeProfileError, match="does not match"):
        store.load_or_create(project_id="another-gcp-project", artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan")


def test_runtime_profile_store_uses_owner_only_file_and_directory_permissions(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "gcp-native-runtime.json"
    store = GcpNativeRuntimeProfileStore(path)

    store.create(project_id="example-project-1", artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("project_id", "artifact_root_uri"),
    (
        ("gs://eyvan-runtime-artifacts", "gs://eyvan-runtime-artifacts/eyvan"),
        ("EXAMPLE-RESEARCH-PROJECT", "gs://eyvan-runtime-artifacts/eyvan"),
        ("short", "gs://eyvan-runtime-artifacts/eyvan"),
        ("example-project-1", "https://eyvan-runtime-artifacts/eyvan"),
        ("example-project-1", "gs://Invalid_Bucket/eyvan"),
        ("example-project-1", "gs://ab/eyvan"),
        ("example-project-1", "gs://eyvan-runtime-artifacts/eyvan?credential=secret"),
        ("example-project-1", "gs://eyvan-runtime-artifacts//eyvan"),
    ),
)
def test_runtime_profile_rejects_invalid_gcp_project_ids_and_artifact_roots(
    tmp_path: Path,
    project_id: str,
    artifact_root_uri: str,
) -> None:
    store = GcpNativeRuntimeProfileStore(tmp_path / "profile" / "gcp-native-runtime.json")

    with pytest.raises(GcpNativeRuntimeProfileError):
        store.create(project_id=project_id, artifact_root_uri=artifact_root_uri)


@pytest.mark.parametrize(
    "native_worker_wheel_uri",
    (
        "https://eyvan-runtime-artifacts/releases/eyvan.whl",
        "gs://eyvan-runtime-artifacts/releases/",
        "gs://eyvan-runtime-artifacts/releases/eyvan.tar.gz",
        "gs://eyvan-runtime-artifacts/releases/eyvan.whl",
        f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'a' * 64}/eyvan-deadbeef.whl",
        f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/not-a-digest/{_WHEEL_FILENAME}",
        f"gs://eyvan-runtime-artifacts/eyvan/runtime/wheels/{'a' * 64}/../{_WHEEL_FILENAME}",
        f"gs://other-runtime-artifacts/eyvan/runtime/wheels/{'a' * 64}/{_WHEEL_FILENAME}",
        f"gs://eyvan-runtime-artifacts/other/runtime/wheels/{'a' * 64}/{_WHEEL_FILENAME}",
        "gs://eyvan-runtime-artifacts/releases/*.whl",
    ),
)
def test_runtime_profile_rejects_nonconcrete_native_worker_wheels(native_worker_wheel_uri: str) -> None:
    with pytest.raises(GcpNativeRuntimeProfileError):
        GcpNativeRuntimeProfile(
            project_id="example-project-1",
            artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
            agent_shared_secret="a" * 32,
            native_worker_root_secret="b" * 32,
            native_worker_wheel_uri=native_worker_wheel_uri,
        )


def test_runtime_profile_accepts_a_content_addressed_directory_with_a_valid_pip_wheel_filename() -> None:
    profile = GcpNativeRuntimeProfile(
        project_id="example-project-1",
        artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
        agent_shared_secret="a" * 32,
        native_worker_root_secret="b" * 32,
        native_worker_wheel_uri=_WHEEL_URI,
    )

    assert profile.native_worker_wheel_uri == _WHEEL_URI


def test_runtime_profile_store_rejects_a_symlinked_profile_file(tmp_path: Path) -> None:
    profile_directory = tmp_path / "profile"
    profile_directory.mkdir(mode=0o700)
    path = profile_directory / "gcp-native-runtime.json"
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    path.symlink_to(target)
    store = GcpNativeRuntimeProfileStore(path)

    with pytest.raises(GcpNativeRuntimeProfileError):
        store.load()
    with pytest.raises(GcpNativeRuntimeProfileError):
        store.save(_profile())


def test_runtime_profile_store_rejects_a_group_readable_profile_file(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "gcp-native-runtime.json"
    store = GcpNativeRuntimeProfileStore(path)
    store.create(project_id="example-project-1", artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan")
    os.chmod(path, 0o640)

    with pytest.raises(GcpNativeRuntimeProfileError, match="must not be group- or world-readable"):
        store.load()
    with pytest.raises(GcpNativeRuntimeProfileError, match="must not be group- or world-readable"):
        store.save(_profile())


def test_runtime_profile_document_rejects_unknown_or_secretless_schema_fields() -> None:
    document = _profile().as_document()
    document["unexpected"] = "field"

    with pytest.raises(GcpNativeRuntimeProfileError, match="unsupported field"):
        GcpNativeRuntimeProfile.from_document(document)

    document = _profile().as_document()
    document.pop("agent_shared_secret")

    with pytest.raises(GcpNativeRuntimeProfileError, match="agent_shared_secret"):
        GcpNativeRuntimeProfile.from_document(document)


def test_runtime_profile_rejects_a_noninteger_version_before_persistence() -> None:
    with pytest.raises(GcpNativeRuntimeProfileError, match="version"):
        GcpNativeRuntimeProfile(
            project_id="example-project-1",
            artifact_root_uri="gs://eyvan-runtime-artifacts/eyvan",
            agent_shared_secret="a" * 32,
            native_worker_root_secret="b" * 32,
            version=1.0,
        )
