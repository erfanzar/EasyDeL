# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local qualification of opt-in, version-pinned worker TLS bootstrap."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from eyvan.identity.mtls import MutualTlsMode, server_ssl_context
from eyvan.worker.runtime.native_bootstrap import build_native_worker_startup_script
from tests.test_identity_mtls import pki as _pki_fixture

pki = _pki_fixture

_RESOURCE = "projects/project-a/secrets/cell-a-worker-tls/versions/7"
_BOOTSTRAP = {
    "package": "gs://artifacts/eyvan/eyvan.whl",
    "package_sha256": "a" * 64,
    "package_generation": "1700000000000000",
    "port": 8123,
    "state_dir": "/var/lib/eyvan/worker",
    "worker_secret_resource": "projects/project-a/secrets/cell-a-bearer/versions/latest",
}


def test_opt_in_bootstrap_stages_a_pinned_bundle_before_requiring_client_certificates():
    script = build_native_worker_startup_script(**_BOOTSTRAP, worker_tls_secret_resource=_RESOURCE)
    assert _RESOURCE in script
    assert "eyvan.worker.runtime.native_tls_bootstrap" in script
    assert "--mtls-mode required" in script
    assert "--mtls-trust-bundle /run/eyvan-native/tls/trust_bundle.pem" in script
    assert "--mtls-certificate /run/eyvan-native/tls/certificate.pem" in script
    assert "--mtls-private-key /run/eyvan-native/tls/private_key.pem" in script
    assert script.index("eyvan.worker.runtime.native_tls_bootstrap") < script.index("ExecStart=")
    assert "Environment=EYVAN_MUTUAL_TLS" not in script


def test_disabled_tls_bootstrap_keeps_the_default_script_identical():
    assert build_native_worker_startup_script(**_BOOTSTRAP, worker_tls_secret_resource=None) == (
        build_native_worker_startup_script(**_BOOTSTRAP)
    )


@pytest.mark.parametrize(
    "resource",
    ["", "latest", _RESOURCE.replace("/7", "/latest"), _RESOURCE.replace("/7", "/0"), _RESOURCE + "\n"],
)
def test_bootstrap_refuses_unpinned_or_malformed_tls_references(resource):
    with pytest.raises(ValueError, match="immutable Secret Manager version"):
        build_native_worker_startup_script(**_BOOTSTRAP, worker_tls_secret_resource=resource)


def _bundle(pki):
    return {
        name: Path(getattr(pki["server"], name)).read_text() for name in ("trust_bundle", "certificate", "private_key")
    }


def _fake_gcloud(monkeypatch, tmp_path, payload, *, fail=False):
    executable = tmp_path / "gcloud"
    source = tmp_path / "payload"
    source.write_bytes(payload)
    executable.write_text(
        "#!/bin/sh\n" + f"cat '{source}'\n" + ("echo private-diagnostic >&2\nexit 1\n" if fail else "")
    )
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])


def test_real_fetch_installs_owner_only_validated_files(monkeypatch, tmp_path, pki):
    from eyvan.worker.runtime.native_tls_bootstrap import install_native_worker_tls

    bundle = _bundle(pki)
    _fake_gcloud(monkeypatch, tmp_path, json.dumps(bundle).encode())
    destination = tmp_path / "tls"
    files = install_native_worker_tls(_RESOURCE, destination)
    assert destination.stat().st_mode & 0o777 == 0o700
    for name, expected in bundle.items():
        path = Path(getattr(files, name))
        assert path.read_text() == expected
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_uid == os.geteuid()
    assert server_ssl_context(MutualTlsMode.REQUIRED, files) is not None
    assert sorted(path.name for path in destination.iterdir()) == [
        "certificate.pem",
        "private_key.pem",
        "trust_bundle.pem",
    ]


@pytest.mark.parametrize(
    "defect",
    ["missing", "extra", "broken-key", "mismatch", "duplicate", "oversized", "fetch", "encrypted", "utf8", "json"],
)
def test_bad_bundles_leave_no_tls_files_or_private_diagnostics(monkeypatch, tmp_path, pki, defect, capsys):
    from eyvan.worker.runtime.native_tls_bootstrap import install_native_worker_tls

    bundle = _bundle(pki)
    if defect == "missing":
        del bundle["private_key"]
    elif defect == "extra":
        bundle["unexpected"] = "private-diagnostic"
    elif defect == "broken-key":
        bundle["private_key"] = "private-diagnostic"
    elif defect == "mismatch":
        bundle["private_key"] = Path(pki["client"].private_key).read_text()
    elif defect == "encrypted":
        from cryptography.hazmat.primitives import serialization

        key = serialization.load_pem_private_key(bundle["private_key"].encode(), password=None)
        bundle["private_key"] = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(b"test-only-password"),
        ).decode()
    payload = json.dumps(bundle).encode()
    if defect == "utf8":
        payload = b"\xffprivate-diagnostic"
    elif defect == "json":
        payload = b"private-diagnostic"
    elif defect == "duplicate":
        payload = payload[:-1] + b', "private_key": "private-diagnostic"}'
    elif defect == "oversized":
        payload = b"x" * 65537
    _fake_gcloud(monkeypatch, tmp_path, payload, fail=defect == "fetch")
    with pytest.raises(ValueError) as caught:
        install_native_worker_tls(_RESOURCE, tmp_path / "tls")
    assert "private-diagnostic" not in str(caught.value)
    assert "PRIVATE KEY" not in str(caught.value)
    assert not (tmp_path / "tls").exists()
    assert not list(tmp_path.glob(".tls-*"))
    assert capsys.readouterr() == ("", "")


def test_unsafe_destination_refused_before_cloud_access(monkeypatch, tmp_path):
    from eyvan.worker.runtime.native_tls_bootstrap import install_native_worker_tls

    def forbidden(*args, **kwargs):
        pytest.fail("must not fetch before validating private destination")

    monkeypatch.setattr(subprocess, "run", forbidden)
    parent = tmp_path / "public"
    parent.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        install_native_worker_tls(_RESOURCE, parent / "tls")
    (tmp_path / "tls").symlink_to(parent, target_is_directory=True)
    with pytest.raises(ValueError):
        install_native_worker_tls(_RESOURCE, tmp_path / "tls")
    (tmp_path / "existing").mkdir(mode=0o700)
    with pytest.raises(ValueError):
        install_native_worker_tls(_RESOURCE, tmp_path / "existing")


def test_rendered_tls_lifecycle_is_explicit_and_version_changes_script():
    script = build_native_worker_startup_script(**_BOOTSTRAP, worker_tls_secret_resource=_RESOURCE)
    assert script.count(_RESOURCE) == 1
    assert "RuntimeDirectory=eyvan-native" in script
    assert "RuntimeDirectoryMode=0700" in script
    assert "Restart=always" in script
    assert script.index("eyvan.worker.runtime.native_tls_bootstrap") < script.index("gcloud secrets versions access")
    assert script != build_native_worker_startup_script(
        **_BOOTSTRAP, worker_tls_secret_resource=_RESOURCE.replace("/7", "/8")
    )
    disabled = build_native_worker_startup_script(**_BOOTSTRAP)
    assert "native_tls_bootstrap" not in disabled
    assert "--mtls-" not in disabled


@pytest.mark.parametrize("valid", [True, False])
def test_staging_cli_has_private_bounded_output(monkeypatch, tmp_path, pki, valid):
    import sys

    payload = json.dumps(_bundle(pki)).encode() if valid else b"private-diagnostic"
    _fake_gcloud(monkeypatch, tmp_path, payload)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eyvan.worker.runtime.native_tls_bootstrap",
            "--resource",
            _RESOURCE,
            "--directory",
            str(tmp_path / "tls"),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == (0 if valid else 1)
    assert result.stdout == b""
    assert "private-diagnostic" not in result.stderr.decode()
    assert "PRIVATE KEY" not in result.stderr.decode()
    assert (tmp_path / "tls").exists() is valid


@pytest.mark.parametrize("resource", [_RESOURCE + "x" * 1024, _RESOURCE + "%n", _RESOURCE + ";id"])
def test_unsafe_reference_never_reaches_fetch(monkeypatch, tmp_path, resource):
    from eyvan.worker.runtime.native_tls_bootstrap import install_native_worker_tls

    def forbidden(*args, **kwargs):
        pytest.fail("invalid reference reached gcloud")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="immutable Secret Manager version"):
        install_native_worker_tls(resource, tmp_path / "tls")


def test_bundle_fetch_scrubs_interpreter_steering_but_keeps_cloud_configuration(monkeypatch, tmp_path, pki):
    from eyvan._internal.subprocess_env import INTERPRETER_STEERING_ENV_NAMES
    from eyvan.worker.runtime.native_tls_bootstrap import install_native_worker_tls

    _fake_gcloud(monkeypatch, tmp_path, json.dumps(_bundle(pki)).encode())
    executable = tmp_path / "gcloud"
    checks = [f'test -z "${{{name}+x}}" || exit 88' for name in INTERPRETER_STEERING_ENV_NAMES]
    checks.append('test "$CLOUDSDK_CORE_PROJECT" = test-project || exit 89')
    executable.write_text(executable.read_text().replace("#!/bin/sh\n", "#!/bin/sh\n" + "\n".join(checks) + "\n"))
    for name in INTERPRETER_STEERING_ENV_NAMES:
        monkeypatch.setenv(name, "must-not-inherit")
    monkeypatch.setenv("CLOUDSDK_CORE_PROJECT", "test-project")
    files = install_native_worker_tls(_RESOURCE, tmp_path / "tls")
    assert Path(files.private_key).exists()
    assert all(os.environ[name] == "must-not-inherit" for name in INTERPRETER_STEERING_ENV_NAMES)
