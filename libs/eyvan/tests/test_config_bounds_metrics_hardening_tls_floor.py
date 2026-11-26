# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Config bounds, metrics hardening, the TLS version floor and workspace packaging."""

from __future__ import annotations

import inspect
import io
import tarfile
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from eyvan._internal.workspace import WorkspaceError, build_workspace
from eyvan.config import EyvanSettings
from eyvan.identity.mtls import MutualTlsFiles, MutualTlsMode, server_ssl_options
from eyvan.server.api.app import create_app
from eyvan.server.persistence.memory import InMemoryRepository

# --------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("scheduler_interval_s", float("nan")),
        ("scheduler_interval_s", -1.0),
        ("cell_reconcile_interval_s", float("inf")),
        ("soak_evidence_max_age_s", float("nan")),
        ("retention_interval_s", -1.0),
        ("lease_ttl_s", -5),
        ("lease_ttl_s", 0),
        ("native_worker_port", 70000),
        ("native_worker_port", 0),
        ("cell_reconcile_concurrency", 0),
    ],
)
def test_settings_reject_out_of_domain_knobs(field: str, bad: float | int) -> None:
    with pytest.raises(ValueError):
        EyvanSettings(**{field: bad})


def test_settings_still_accept_valid_knobs() -> None:
    settings = EyvanSettings(scheduler_interval_s=2.0, lease_ttl_s=120, native_worker_port=8100)
    assert settings.scheduler_interval_s == 2.0


@pytest.mark.parametrize("field", ["checkpoint_grace_s", "preemption_stop_grace_s", "indeterminate_execution_grace_s"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_settings_reject_nonfinite_grace_periods(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=field):
        EyvanSettings(**{field: bad})


@pytest.mark.parametrize("field", ["checkpoint_grace_s", "preemption_stop_grace_s", "indeterminate_execution_grace_s"])
@pytest.mark.parametrize("value", [0, 1, 2.5])
def test_settings_preserve_finite_grace_periods(field: str, value: float) -> None:
    assert getattr(EyvanSettings(**{field: value}), field) == value


# -------------------------------------------------------------------- metrics


def _metrics_app():
    repository = InMemoryRepository()
    settings = EyvanSettings(dev_auth_enabled=True, metrics_token="m" * 32)
    return create_app(settings=settings, repository=repository)


def test_metrics_refuses_a_non_ascii_bearer_with_401_not_500() -> None:
    client = TestClient(_metrics_app(), raise_server_exceptions=False)
    # Raw bytes: a real client can put non-ASCII on the wire even though the
    # httpx header builder refuses to encode it for us.
    response = client.get(
        "/metrics",
        headers={b"Authorization": b"Bearer t\xf6k\xe9n"},
    )
    assert response.status_code == 401
    assert response.headers["Cache-Control"] == "no-store"


def test_metrics_answers_carry_no_store() -> None:
    app = _metrics_app()
    client = TestClient(app, raise_server_exceptions=False)
    unauthorized = client.get("/metrics")
    assert unauthorized.status_code == 401
    assert unauthorized.headers["Cache-Control"] == "no-store"
    authorized = client.get("/metrics", headers={"Authorization": "Bearer " + "m" * 32})
    assert authorized.status_code == 200
    assert authorized.headers["Cache-Control"] == "no-store"


# ----------------------------------------------------------------- mTLS floor


def test_server_ssl_options_use_supported_uvicorn_tls_policy(tmp_path: Path) -> None:
    files = MutualTlsFiles(
        certificate=str(tmp_path / "cert.pem"),
        private_key=str(tmp_path / "key.pem"),
        trust_bundle=str(tmp_path / "bundle.pem"),
    )
    options = server_ssl_options(MutualTlsMode.REQUIRED, files)
    inspect.signature(uvicorn.run).bind(object(), **options)
    # The floor must not depend on the OpenSSL build's SSLContext default (the
    # Ubuntu CI runner's is MINIMUM_SUPPORTED); the factory sets it explicitly.
    assert callable(options["ssl_context_factory"])


# ------------------------------------------------------------------ workspace


def _write(root: Path, relative: str, content: str = "x") -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def _member_names(bundle) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(bundle.payload), mode="r:gz") as archive:
        return {member.name for member in archive.getmembers() if member.isfile()}


def test_workspace_excludes_are_case_insensitive_and_env_staged(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    _write(root, "src/main.py", "print('hi')")
    _write(root, "README.md")
    _write(root, ".env.production", "SECRET=1")
    _write(root, "FOO.PEM", "certificate")
    _write(root, "deploy.tfvars", "token=1")

    bundle = build_workspace(root, respect_gitignore=False)
    names = _member_names(bundle)

    assert f"{root.name}/src/main.py" in names or "src/main.py" in names
    assert not any(name.endswith(".env.production") for name in names)
    assert not any(name.endswith("FOO.PEM") for name in names)
    assert not any(name.endswith("deploy.tfvars") for name in names)


def test_a_git_failure_refuses_rather_than_ships_gitignored_secrets(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "ws"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    _write(root, ".gitignore", "tokens.txt\n")
    _write(root, "src/main.py", "print('hi')")
    _write(root, "tokens.txt", "secret-token")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)

    real_run = workspace_module.subprocess.run

    def flaky_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and any("ls-files" in str(part) for part in cmd):
            raise subprocess.TimeoutExpired(cmd="git", timeout=30.0)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(workspace_module.subprocess, "run", flaky_run)

    with pytest.raises(WorkspaceError, match="git could not enumerate"):
        build_workspace(root)


def test_a_non_repository_directory_still_falls_back_to_the_walk(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "plain"
    _write(root, "src/main.py", "print('hi')")

    real_run = workspace_module.subprocess.run

    def failing_git(cmd, *args, **kwargs):
        if isinstance(cmd, list) and any("git" == str(part) for part in cmd[:1]):
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repository")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_git)

    bundle = build_workspace(root, respect_gitignore=False)
    names = _member_names(bundle)
    assert any(name.endswith("src/main.py") for name in names)


@pytest.mark.parametrize("failure", ["timeout", "missing", "dubious", "corrupt", "ambiguous", "masked-nonrepo"])
def test_workspace_refuses_when_git_enumeration_and_repository_probe_both_fail(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    import subprocess

    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "ws"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _write(root, ".gitignore", "tokens.txt\n")
    _write(root, "src/main.py", "print('hi')")
    _write(root, "tokens.txt", "not-a-real-token")

    def failing_git(cmd, *args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30.0)
        if failure == "missing":
            raise FileNotFoundError("git is unavailable")
        diagnostic = {
            "dubious": "fatal: detected dubious ownership in repository",
            "corrupt": "fatal: bad object HEAD",
            "ambiguous": "",
            "masked-nonrepo": "fatal: not a git repository (or any of the parent directories): .git",
        }[failure]
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr=diagnostic)

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_git)

    with pytest.raises(WorkspaceError, match=r"git could not enumerate.*respect_gitignore=False"):
        build_workspace(root)


@pytest.mark.parametrize("failure", ["timeout", "missing", "dubious", "corrupt", "ambiguous", "bare", "nonrepo"])
def test_workspace_probe_requires_positive_nonrepository_evidence(tmp_path: Path, monkeypatch, failure: str) -> None:
    import subprocess

    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "plain"
    _write(root, "src/main.py", "print('hi')")
    monkeypatch.setenv("LC_ALL", "fa_IR.UTF-8")
    probes = []

    def failing_git(cmd, *args, **kwargs):
        if "--show-prefix" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="enumeration unavailable")
        if "check-ignore" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"")
        if "ls-files" in cmd:
            return subprocess.CompletedProcess(cmd, 128, stdout=b"", stderr=b"enumeration failed")
        assert "rev-parse" in cmd
        probes.append(cmd)
        assert kwargs["env"]["LC_ALL"] == "C"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30.0)
        if failure == "missing":
            raise FileNotFoundError("git is unavailable")
        diagnostic = {
            "dubious": "fatal: detected dubious ownership in repository",
            "corrupt": "fatal: bad object HEAD",
            "ambiguous": "",
            "bare": "",
            "nonrepo": "fatal: not a git repository (or any of the parent directories): .git",
        }[failure]
        return subprocess.CompletedProcess(cmd, 0 if failure == "bare" else 128, stdout="false\n", stderr=diagnostic)

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_git)
    if failure == "nonrepo":
        assert _member_names(build_workspace(root)) == {"src/main.py"}
    else:
        with pytest.raises(WorkspaceError, match=r"git could not enumerate.*respect_gitignore=False"):
            build_workspace(root)
    assert len(probes) == 1


@pytest.mark.parametrize("metadata", ["broken-file", "dangling-symlink"])
def test_workspace_refuses_broken_git_metadata_in_an_ancestor(tmp_path: Path, monkeypatch, metadata: str) -> None:
    import subprocess

    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "parent" / "ws"
    _write(root, "src/main.py", "print('hi')")
    if metadata == "broken-file":
        (root.parent / ".git").write_text("gitdir: missing\n")
    else:
        (root.parent / ".git").symlink_to("missing")

    def failing_git(cmd, *args, **kwargs):
        assert "--is-inside-work-tree" not in cmd
        return subprocess.CompletedProcess(cmd, 128, stdout=b"", stderr=b"not a git repository")

    monkeypatch.setattr(workspace_module.subprocess, "run", failing_git)
    with pytest.raises(WorkspaceError, match="git could not enumerate"):
        build_workspace(root)


@pytest.mark.parametrize("respect_gitignore", [True, False])
def test_plain_workspace_packages_with_git_available(tmp_path: Path, respect_gitignore: bool) -> None:
    root = tmp_path / "plain"
    _write(root, "src/main.py", "print('hi')")

    assert _member_names(build_workspace(root, respect_gitignore=respect_gitignore)) == {"src/main.py"}


def test_workspace_explicit_git_opt_out_does_not_require_git(tmp_path: Path, monkeypatch) -> None:
    from eyvan._internal import workspace as workspace_module

    root = tmp_path / "ws"
    _write(root, "src/main.py", "print('hi')")
    _write(root, "tokens.txt", "explicitly opted in")
    _write(root, ".env", "excluded")

    def unavailable_git(*args, **kwargs):
        raise AssertionError("respect_gitignore=False must not invoke git")

    monkeypatch.setattr(workspace_module.subprocess, "run", unavailable_git)
    assert _member_names(build_workspace(root, respect_gitignore=False)) == {"src/main.py", "tokens.txt"}
