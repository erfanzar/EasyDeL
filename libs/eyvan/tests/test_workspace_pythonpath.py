# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Monorepo and ``src/``-layout ``PYTHONPATH`` discovery across callable, command, and OCI workers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eyvan._internal.workspace import build_workspace, discover_workspace_python_paths, prepend_pythonpath
from eyvan.worker.runtime import workspace_entry
from eyvan.worker.runtime.callable_protocol import _materialize_workspace, execute_callable
from eyvan.worker.runtime.callables import dump_callable
from eyvan.worker.runtime.oci_callable_stager import stage_oci_callable_runner
from eyvan.worker.runtime.results import load_result


@pytest.fixture
def restore_process_state():
    """Undo cwd, ``sys.path``, and ``PYTHONPATH`` changes made by worker entrypoints."""

    cwd = Path.cwd()
    path = list(sys.path)
    pythonpath = os.environ.get("PYTHONPATH")
    yield
    os.chdir(cwd)
    sys.path[:] = path
    if pythonpath is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = pythonpath


def _write_files(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _build_easydel_style_monorepo(root: Path) -> Path:
    return _write_files(
        root,
        {
            "pyproject.toml": (
                '[project]\nname = "easydel-foundation"\n\n'
                '[tool.uv.workspace]\nmembers = ["libs/*"]\nexclude = ["libs/excluded_lib"]\n'
            ),
            "train.py": "ENTRY = 'root'\n",
            "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\n',
            "libs/easydel/easydel/__init__.py": 'ORIGIN = "workspace-easydel"\n',
            "libs/spectrax/pyproject.toml": '[project]\nname = "spectrax"\n',
            "libs/spectrax/spectrax/__init__.py": 'ORIGIN = "workspace-spectrax"\n',
            "libs/src_member/pyproject.toml": '[project]\nname = "src-member"\n',
            "libs/src_member/src/src_member/__init__.py": 'ORIGIN = "workspace-src-member"\n',
            "libs/flat_mod/pyproject.toml": '[project]\nname = "flat-mod"\n',
            "libs/flat_mod/flat_helper.py": 'ORIGIN = "workspace-flat-mod"\n',
            "libs/docs_only/README.md": "# docs only\n",
            "libs/excluded_lib/pyproject.toml": '[project]\nname = "excluded-lib"\n',
            "libs/excluded_lib/excluded_lib/__init__.py": 'ORIGIN = "excluded"\n',
            # Native/compiled crates must NOT be added to sys.path so compiled host .so extensions load cleanly.
            "libs/surgecore/pyproject.toml": '[project]\nname = "surgecore"\n',
            "libs/surgecore/Cargo.toml": '[package]\nname = "surgecore"\nversion = "0.1.0"\n',
            "libs/surgecore/src/lib.rs": "pub fn version() {}\n",
            "libs/surgecore/surgecore/__init__.py": 'ORIGIN = "broken-workspace-surgecore"\n',
            "libs/cpp_ext/CMakeLists.txt": "cmake_minimum_required(VERSION 3.20)\n",
            "libs/cpp_ext/cpp_ext/__init__.py": 'ORIGIN = "broken-workspace-cpp-ext"\n',
            # Eyvan itself must NOT shadow the worker's runtime eyvan (matched by name or package dir).
            "libs/eyvan/pyproject.toml": '[project]\nname = "eyvan"\n',
            "libs/eyvan/eyvan/__init__.py": 'ORIGIN = "shadow-eyvan"\n',
            "libs/eyvan_by_name/pyproject.toml": '[project]\nname = "eyvan"\n',
            "libs/eyvan_by_name/other_pkg/__init__.py": 'ORIGIN = "shadow-eyvan-name"\n',
        },
    )


def test_discover_workspace_python_paths_for_flat_src_and_monorepo_layouts(tmp_path: Path) -> None:
    flat = _write_files(tmp_path / "flat", {"helper.py": "X = 1\n"})
    assert discover_workspace_python_paths(flat) == [flat.resolve()]

    src_repo = _write_files(
        tmp_path / "src-repo",
        {
            "pyproject.toml": '[project]\nname = "demo"\n',
            "src/demo/__init__.py": 'ORIGIN = "src-layout"\n',
        },
    )
    assert discover_workspace_python_paths(src_repo) == [src_repo.resolve(), (src_repo / "src").resolve()]

    monorepo = _build_easydel_style_monorepo(tmp_path / "monorepo")
    discovered = discover_workspace_python_paths(monorepo)
    assert discovered == [
        monorepo.resolve(),
        (monorepo / "libs" / "easydel").resolve(),
        (monorepo / "libs" / "flat_mod").resolve(),
        (monorepo / "libs" / "spectrax").resolve(),
        (monorepo / "libs" / "src_member" / "src").resolve(),
    ]


def test_discover_workspace_python_paths_rejects_traversal_and_symlink_escapes(tmp_path: Path) -> None:
    outside = _write_files(tmp_path / "outside", {"secret_pkg/__init__.py": "LEAK = True\n"})
    repo = _write_files(
        tmp_path / "repo",
        {
            "pyproject.toml": (
                '[project]\nname = "malicious"\n\n'
                "[tool.uv.workspace]\n"
                'members = ["../../etc", "../outside", "/etc", "libs/../../outside", "libs/*"]\n'
            ),
            "libs/good/good_pkg/__init__.py": "OK = True\n",
        },
    )
    (repo / "libs" / "escape_link").symlink_to(outside)

    discovered = discover_workspace_python_paths(repo)
    assert discovered == [repo.resolve(), (repo / "libs" / "good").resolve()]
    assert all(path.is_relative_to(repo.resolve()) for path in discovered)


def test_prepend_pythonpath_preserves_order_and_deduplicates() -> None:
    merged = prepend_pythonpath(["/ws", "/ws/libs/easydel"], f"/ws/libs/easydel{os.pathsep}/opt/eyvan-native/multihost")
    assert merged == os.pathsep.join(["/ws", "/ws/libs/easydel", "/opt/eyvan-native/multihost"])


def _make_host_site(root: Path) -> Path:
    return _write_files(
        root,
        {
            "easydel/__init__.py": 'ORIGIN = "host-easydel"\n',
            "spectrax/__init__.py": 'ORIGIN = "host-spectrax"\n',
            "surgecore/__init__.py": 'ORIGIN = "host-surgecore-compiled"\n',
            "eyvan/__init__.py": 'ORIGIN = "host-eyvan-runtime"\n',
        },
    )


def test_callable_materialize_workspace_updates_sys_path_and_pythonpath_for_monorepo(
    tmp_path: Path,
    restore_process_state,
) -> None:
    monorepo = _build_easydel_style_monorepo(tmp_path / "monorepo")
    host_site = _make_host_site(tmp_path / "host-site")
    bundle = build_workspace(monorepo, respect_gitignore=False)

    os.environ["PYTHONPATH"] = f"{host_site}{os.pathsep}/opt/eyvan-native/multihost"
    destination = _materialize_workspace(
        bundle.payload,
        sha256=bundle.sha256,
        signature=None,
        signing_key=None,
        environment={"EYVAN_WORKSPACE_DIR": str(tmp_path / "unpacked")},
    )
    resolved = destination.resolve()
    expected_prefix = [
        str(resolved),
        str(resolved / "libs" / "easydel"),
        str(resolved / "libs" / "flat_mod"),
        str(resolved / "libs" / "spectrax"),
        str(resolved / "libs" / "src_member" / "src"),
    ]
    assert sys.path[: len(expected_prefix)] == expected_prefix
    pythonpath_entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries[: len(expected_prefix)] == expected_prefix
    assert pythonpath_entries[-2:] == [str(host_site), "/opt/eyvan-native/multihost"]

    # A child Python process spawned inside the callable sees local easydel & spectrax
    # shadowing host site-packages while surgecore & eyvan remain on the host!
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import easydel, spectrax, src_member, flat_helper, surgecore, eyvan; "
            "print("
            "easydel.ORIGIN, spectrax.ORIGIN, src_member.ORIGIN, "
            "flat_helper.ORIGIN, surgecore.ORIGIN, eyvan.ORIGIN)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == (
        "workspace-easydel workspace-spectrax workspace-src-member workspace-flat-mod "
        "host-surgecore-compiled host-eyvan-runtime"
    )


def test_execute_callable_imports_workspace_monorepo_modules_before_unpickling(
    tmp_path: Path,
    restore_process_state,
) -> None:
    monorepo = _write_files(
        tmp_path / "monorepo",
        {
            "pyproject.toml": '[project]\nname = "ws"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
            "libs/local_alpha/local_alpha/__init__.py": 'TAG = "from-monorepo-member"\n',
        },
    )
    bundle = build_workspace(monorepo, respect_gitignore=False)

    def job() -> tuple[str, str]:
        import importlib

        mod = importlib.import_module("local_alpha")
        return mod.TAG, os.environ.get("PYTHONPATH", "")

    artifact = dump_callable(job)

    class _MemoryTransfer:
        def __init__(self) -> None:
            self.written: dict[str, bytes] = {}

        def read(self, uri: str, *, token: str | None = None) -> bytes:
            del token
            return {"mem://callable": artifact.payload, "mem://workspace": bundle.payload}[uri]

        def write(self, uri: str, payload: bytes, *, token: str | None = None) -> None:
            del token
            self.written[uri] = payload

    transfer = _MemoryTransfer()
    exit_code = execute_callable(
        [
            "--uri",
            "mem://callable",
            "--sha256",
            artifact.sha256,
            "--python",
            artifact.python,
            "--workspace-uri",
            "mem://workspace",
            "--workspace-sha256",
            bundle.sha256,
            "--result-uri",
            "mem://result",
        ],
        transfer=transfer,
        environment={"EYVAN_WORKSPACE_DIR": str(tmp_path / "unpacked")},
    )
    assert exit_code == 0
    tag, pythonpath = load_result(transfer.written["mem://result"])  # type: ignore[misc]
    assert tag == "from-monorepo-member"
    assert any(entry.endswith("/libs/local_alpha") for entry in pythonpath.split(os.pathsep))


def test_workspace_entry_main_sets_monorepo_pythonpath_before_exec(
    tmp_path: Path,
    monkeypatch,
    restore_process_state,
) -> None:
    monorepo = _build_easydel_style_monorepo(tmp_path / "monorepo")
    host_site = _make_host_site(tmp_path / "host-site")
    bundle = build_workspace(monorepo, respect_gitignore=False)
    bundle_file = tmp_path / "ws.tar.gz"
    bundle_file.write_bytes(bundle.payload)

    captured: dict[str, object] = {}

    class _StopExec(BaseException):
        pass

    def fake_execvp(program: str, argv: list[str]) -> None:
        captured["program"] = program
        captured["argv"] = list(argv)
        captured["cwd"] = Path.cwd()
        captured["pythonpath"] = os.environ.get("PYTHONPATH", "")
        raise _StopExec

    monkeypatch.setattr(workspace_entry.os, "execvp", fake_execvp)
    monkeypatch.setenv("EYVAN_WORKSPACE_DIR", str(tmp_path / "ws-out"))
    monkeypatch.setenv("PYTHONPATH", f"{host_site}{os.pathsep}/opt/eyvan-native/multihost")
    monkeypatch.delenv("EYVAN_CALLABLE_SIGNING_KEY", raising=False)

    with pytest.raises(_StopExec):
        workspace_entry.main(
            [
                "--workspace-uri",
                str(bundle_file),
                "--workspace-sha256",
                bundle.sha256,
                "--",
                "python",
                "train.py",
            ]
        )

    cwd = captured["cwd"]
    assert isinstance(cwd, Path)
    entries = str(captured["pythonpath"]).split(os.pathsep)
    assert entries == [
        str(cwd),
        str(cwd / "libs" / "easydel"),
        str(cwd / "libs" / "flat_mod"),
        str(cwd / "libs" / "spectrax"),
        str(cwd / "libs" / "src_member" / "src"),
        str(host_site),
        "/opt/eyvan-native/multihost",
    ]


def test_staged_oci_workspace_entry_imports_monorepo_without_eyvan_package(tmp_path: Path) -> None:
    """In OCI standalone mode (``python -I`` without ``eyvan``), ``workspace_entry.py`` imports sibling ``workspace``."""

    staged_dir = tmp_path / "oci-staged"
    stage_oci_callable_runner(staged_dir)

    bare_venv = tmp_path / "bare-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(bare_venv)], check=True, timeout=120)
    bare_python = bare_venv / "bin" / "python"

    monorepo = _build_easydel_style_monorepo(tmp_path / "monorepo")
    host_site = _make_host_site(tmp_path / "host-site")
    bundle = build_workspace(monorepo, respect_gitignore=False)
    bundle_file = tmp_path / "ws.tar.gz"
    bundle_file.write_bytes(bundle.payload)
    digest = hashlib.sha256(bundle.payload).hexdigest()

    completed = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            f"PATH={bare_venv / 'bin'}{os.pathsep}/usr/bin{os.pathsep}/bin",
            f"EYVAN_WORKSPACE_DIR={tmp_path / 'extracted'}",
            f"PYTHONPATH={host_site}",
            str(bare_python),
            "-I",
            str(staged_dir / "workspace_entry.py"),
            "--workspace-uri",
            f"file://{bundle_file}",
            "--workspace-sha256",
            digest,
            "--",
            str(bare_python),
            "-c",
            "import easydel, spectrax, src_member, surgecore, eyvan; "
            "print(easydel.ORIGIN, spectrax.ORIGIN, src_member.ORIGIN, surgecore.ORIGIN, eyvan.ORIGIN)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == (
        "workspace-easydel workspace-spectrax workspace-src-member host-surgecore-compiled host-eyvan-runtime"
    )


def test_staged_oci_callable_runner_loads_monorepo_workspace_without_importing_eyvan(tmp_path: Path) -> None:
    """Staged ``callable_protocol.py`` imports sibling ``workspace.py`` and never touches a blocked ``eyvan``."""

    staged_dir = tmp_path / "oci-staged"
    runner = stage_oci_callable_runner(staged_dir)

    monorepo = _build_easydel_style_monorepo(tmp_path / "monorepo")
    bundle = build_workspace(monorepo, respect_gitignore=False)
    bundle_file = tmp_path / "ws.tar.gz"
    bundle_file.write_bytes(bundle.payload)

    def oci_job() -> tuple[str, str]:
        import importlib

        ed = importlib.import_module("easydel")
        spx = importlib.import_module("spectrax")
        return ed.ORIGIN, spx.ORIGIN

    artifact = dump_callable(oci_job)
    artifact_file = tmp_path / "callable.pkl"
    artifact_file.write_bytes(artifact.payload)
    result_file = tmp_path / "result.bin"

    # Plant a poisoned eyvan.py ahead on PYTHONPATH so any accidental `import eyvan` fails loudly!
    blocked = _write_files(
        tmp_path / "blocked",
        {
            "eyvan.py": 'raise RuntimeError("eyvan must not be imported by staged OCI callable runner")\n',
            "surgecore/__init__.py": 'ORIGIN = "host-surgecore"\n',
        },
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(part for part in (str(blocked), env.get("PYTHONPATH")) if part)
    env["EYVAN_WORKSPACE_DIR"] = str(tmp_path / "oci-ws")
    env["EYVAN_CALL_ARGS_JSON"] = "[]"
    env["EYVAN_CALL_KWARGS_JSON"] = "{}"

    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--uri",
            f"file://{artifact_file}",
            "--sha256",
            artifact.sha256,
            "--python",
            artifact.python,
            "--workspace-uri",
            f"file://{bundle_file}",
            "--workspace-sha256",
            bundle.sha256,
            "--result-uri",
            f"file://{result_file}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert load_result(result_file.read_bytes()) == ("workspace-easydel", "workspace-spectrax")


def test_workspace_entry_fetches_gcs_bundle_via_metadata_token(monkeypatch) -> None:
    """``workspace_entry._fetch`` downloads ``gs://`` bundles using GCE metadata credentials."""

    seen_urls: list[str] = []
    seen_headers: list[dict[str, str]] = []

    class _FakeResponse:
        def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
            self._body = body
            self.headers = headers or {}

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeOpener:
        def open(self, request, timeout: int = 10) -> _FakeResponse:
            del timeout
            seen_urls.append(request.full_url)
            seen_headers.append(dict(request.header_items()))
            return _FakeResponse(
                b'{"access_token":"tok-abc"}',
                headers={"Metadata-Flavor": "Google"},
            )

    def fake_urlopen(request, timeout: int = 60) -> _FakeResponse:
        del timeout
        seen_urls.append(request.full_url)
        seen_headers.append(dict(request.header_items()))
        return _FakeResponse(b"workspace-tar-bytes")

    monkeypatch.setattr(workspace_entry.urllib.request, "build_opener", lambda *a, **k: _FakeOpener())
    monkeypatch.setattr(workspace_entry.urllib.request, "urlopen", fake_urlopen)

    data = workspace_entry._fetch("gs://example-bucket/eyvan/workspaces/deadbeef.tar.gz", token=None)
    assert data == b"workspace-tar-bytes"
    assert seen_urls[0] == workspace_entry._METADATA_TOKEN_URL
    assert seen_urls[1] == (
        "https://storage.googleapis.com/storage/v1/b/example-bucket/o/eyvan%2Fworkspaces%2Fdeadbeef.tar.gz?alt=media"
    )
    assert seen_headers[1].get("Authorization") == "Bearer tok-abc"


def test_workspace_entry_rejects_spoofed_metadata_and_bad_gcs_uri(monkeypatch) -> None:
    with pytest.raises(ValueError, match="invalid GCS workspace URI"):
        workspace_entry._fetch("gs://bucket-only", token=None)

    class _SpoofedResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def read(self) -> bytes:
            return b'{"access_token":"bad"}'

        def __enter__(self) -> _SpoofedResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _SpoofedOpener:
        def open(self, request, timeout: int = 10) -> _SpoofedResponse:
            del request, timeout
            return _SpoofedResponse()

    monkeypatch.setattr(workspace_entry.urllib.request, "build_opener", lambda *a, **k: _SpoofedOpener())
    with pytest.raises(RuntimeError, match="Google metadata service"):
        workspace_entry._fetch("gs://example-bucket/workspaces/x.tar.gz", token=None)


def test_cli_upload_workspace_uses_gcs_for_remote_fleet_on_loopback(tmp_path: Path, monkeypatch) -> None:
    """Loopback CLI submitting to a remote GCP fleet stages workspaces in GCS, not 127.0.0.1."""

    from types import SimpleNamespace

    from eyvan.capacity.providers.gcp.runtime.profile import GcpNativeRuntimeProfile, GcpNativeRuntimeProfileStore
    from eyvan.sdk.commands.state import WorkloadScope
    from eyvan.sdk.commands.workloads import WorkloadCommandService

    src = _write_files(tmp_path / "repo", {"main.py": "print('ok')\n"})
    profile_path = tmp_path / "gcp-native-runtime.json"
    GcpNativeRuntimeProfileStore(profile_path).save(
        GcpNativeRuntimeProfile(
            project_id="example-project-a",
            artifact_root_uri="gs://example-bucket/eyvan",
            agent_shared_secret="s" * 32,
            native_worker_root_secret="n" * 32,
        )
    )

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def request(self, method: str, path: str, **kwargs: object) -> object:
            del kwargs
            self.calls.append((method, path))
            if path == "/api/v1/capacity":
                return {
                    "pools": [
                        {"fleet": "remote-v6e8", "pool": "v6e-8", "region": "us-central2"},
                        {"fleet": "laptop", "pool": "cpu", "region": "local"},
                    ]
                }
            return {}

    uploaded: list[tuple[str, str]] = []

    class _FakeStore:
        def put_workspace(self, payload: bytes, sha256: str) -> str:
            del payload
            uri = f"gs://example-bucket/eyvan/workspaces/{sha256}.tar.gz"
            uploaded.append((sha256, uri))
            return uri

    monkeypatch.setattr(
        "eyvan.capacity.providers.gcp.runtime.supervisor.runtime_profile_path_for",
        lambda _endpoint: profile_path,
    )
    monkeypatch.setattr("eyvan.artifacts.artifact_store_from_uri", lambda _root: _FakeStore())

    client = _FakeClient()
    session = SimpleNamespace(base_url="http://127.0.0.1:8080", client=client, workload_headers=lambda _scope: {})
    service = WorkloadCommandService(session)  # type: ignore[arg-type]
    scope = WorkloadScope(org_id="default", project_id="default")

    remote_ref = service._upload_workspace(src, scope, fleet_name="remote-v6e8")
    assert remote_ref.uri.startswith("gs://example-bucket/eyvan/workspaces/")
    assert len(uploaded) == 1
    assert not any(m == "PUT" for m, _ in client.calls)

    local_ref = service._upload_workspace(src, scope, fleet_name="laptop")
    assert local_ref.uri.startswith("eyvan://workspaces/")
    assert any(m == "PUT" and p.startswith("/api/v1/artifacts/workspaces/") for m, p in client.calls)
