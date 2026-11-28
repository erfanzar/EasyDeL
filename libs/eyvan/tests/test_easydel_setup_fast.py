# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Two-tier layered EasyDeL setup cache, bare git mirror, and Rust wheel cache tests."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from eyvan.records import MAX_SETUP_SCRIPT_BYTES
from eyvan.sdk.client import easydel_setup
from eyvan.sdk.client.easydel_setup import EasyDeLSetup
from eyvan.worker.runtime import dependency_entry


def _have_tools() -> bool:
    return shutil.which("bash") is not None and shutil.which("git") is not None


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=dev@example.com", "-c", "user.name=Dev", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _init_monorepo(root: Path, *, easydel_tag: str = "commit-A", rust_tag: str = "rust-v1") -> str:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    files = {
        "pyproject.toml": '[project]\nname = "easydel-foundation"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
        "uv.lock": "version = 1\n",
        "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\nversion = "0.2.0"\n',
        "libs/easydel/easydel/__init__.py": f'COMMIT_TAG = "{easydel_tag}"\n',
        "libs/eformer/pyproject.toml": '[project]\nname = "eformer"\nversion = "0.2.0"\n',
        "libs/eformer/eformer/__init__.py": 'TAG = "eformer-local"\n',
        "libs/surgecore/pyproject.toml": '[project]\nname = "surgecore"\nversion = "0.1.0"\n',
        "libs/surgecore/Cargo.toml": '[package]\nname = "surgecore"\nversion = "0.1.0"\n',
        "libs/surgecore/Cargo.lock": "version = 3\n",
        "libs/surgecore/src/lib.rs": f'pub const TAG: &str = "{rust_tag}";\n',
        "libs/surgecore/python/surgecore/__init__.py": f'RUST_TAG = "{rust_tag}"\n',
    }
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", f"easydel {easydel_tag}")
    return _git(root, "rev-parse", "HEAD")


def _write_real_wheel_uv(bin_dir: Path, *, install_log: Path, rust_build_log: Path, export_text: str) -> Path:
    """Create a sandbox-safe Python ``uv`` implementation that builds & installs real packages/wheels."""

    bin_dir.mkdir(parents=True, exist_ok=True)
    uv_path = bin_dir / "uv"
    script = textwrap.dedent(
        f"""\
        #!{sys.executable}
        import os
        import pathlib
        import shutil
        import subprocess
        import sys
        import tomllib
        import zipfile

        INSTALL_LOG = pathlib.Path({str(install_log)!r})
        RUST_BUILD_LOG = pathlib.Path({str(rust_build_log)!r})
        EXPORT_TEXT = {export_text!r}


        def _site_packages(python_path: str) -> pathlib.Path:
            venv = pathlib.Path(python_path).parent.parent
            return next((venv / "lib").glob("python*/site-packages"))


        def _write_dist(site: pathlib.Path, name: str, version: str, pkg_files: dict[str, bytes]) -> None:
            dist = site / f"{{name}}-{{version}}.dist-info"
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "METADATA").write_text(
                f"Metadata-Version: 2.1\\nName: {{name}}\\nVersion: {{version}}\\n",
                encoding="utf-8",
            )
            records = [f"{{dist.name}}/METADATA,,", f"{{dist.name}}/RECORD,,"]
            for rel, content in pkg_files.items():
                target = site / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                records.append(f"{{rel}},,")
            (dist / "RECORD").write_text("\\n".join(records) + "\\n", encoding="utf-8")


        def _install_local_dir(site: pathlib.Path, pkg_dir: pathlib.Path) -> None:
            meta = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))["project"]
            name = meta["name"]
            version = meta.get("version", "0.1.0")
            files: dict[str, bytes] = {{}}
            src_root = pkg_dir / "python" if (pkg_dir / "python").is_dir() else pkg_dir
            for py_file in sorted(src_root.rglob("*.py")):
                rel = py_file.relative_to(src_root).as_posix()
                files[rel] = py_file.read_bytes()
            _write_dist(site, name, version, files)


        def _install_wheel(site: pathlib.Path, wheel_path: pathlib.Path) -> None:
            with zipfile.ZipFile(wheel_path, "r") as zf:
                zf.extractall(site)


        def _install_third_party(site: pathlib.Path, req: str) -> None:
            name_ver = req.split(";", 1)[0].strip()
            if "==" in name_ver:
                name, version = name_ver.split("==", 1)
            else:
                name, version = name_ver, "1.0.0"
            mod = name.replace("-", "_")
            _write_dist(site, name, version, {{f"{{mod}}/__init__.py": f'VERSION = "{{version}}"\\n'.encode()}})
            # Also plant a stale decoy easydel in the base layer to verify Layer 2 always shadows Layer 1.
            if name == "heavy-thirdparty":
                decoy = site / "easydel"
                decoy.mkdir(exist_ok=True)
                (decoy / "__init__.py").write_text('COMMIT_TAG = "stale-in-base"\\n', encoding="utf-8")


        def main(argv: list[str]) -> int:
            if not argv:
                return 2
            if argv[0] == "--version":
                print("uv 0.9.test")
                return 0
            if argv[0] == "export":
                out = pathlib.Path(argv[argv.index("-o") + 1])
                out.write_text(EXPORT_TEXT, encoding="utf-8")
                return 0
            if argv[0] == "venv":
                target = argv[-1]
                return subprocess.call([sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", target])
            if argv[0] == "build":
                out_dir = pathlib.Path(argv[argv.index("--out-dir") + 1])
                pkg_dir = pathlib.Path(argv[-1])
                out_dir.mkdir(parents=True, exist_ok=True)
                meta = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))["project"]
                name = meta["name"]
                version = meta.get("version", "0.1.0")
                with RUST_BUILD_LOG.open("a", encoding="utf-8") as handle:
                    handle.write(f"{{name}}|{{pkg_dir}}\\n")
                wheel_path = out_dir / f"{{name}}-{{version}}-py3-none-any.whl"
                src_root = pkg_dir / "python" if (pkg_dir / "python").is_dir() else pkg_dir
                with zipfile.ZipFile(wheel_path, "w") as zf:
                    records = []
                    for py_file in sorted(src_root.rglob("*.py")):
                        rel = py_file.relative_to(src_root).as_posix()
                        zf.writestr(rel, py_file.read_bytes())
                        records.append(f"{{rel}},,")
                    dist = f"{{name}}-{{version}}.dist-info"
                    meta_txt = f"Metadata-Version: 2.1\\nName: {{name}}\\nVersion: {{version}}\\n"
                    wheel_txt = "Wheel-Version: 1.0\\nGenerator: test\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n"
                    zf.writestr(f"{{dist}}/METADATA", meta_txt)
                    zf.writestr(f"{{dist}}/WHEEL", wheel_txt)
                    records.extend([f"{{dist}}/METADATA,,", f"{{dist}}/WHEEL,,", f"{{dist}}/RECORD,,"])
                    zf.writestr(f"{{dist}}/RECORD", "\\n".join(records) + "\\n")
                return 0
            if argv[:2] == ["pip", "install"]:
                py_idx = argv.index("--python")
                target_py = argv[py_idx + 1]
                site = _site_packages(target_py)
                reqs: list[str] = []
                i = 2
                while i < len(argv):
                    tok = argv[i]
                    if tok in ("--python", "--index-url"):
                        i += 2
                        continue
                    if tok in ("--no-deps", "--no-sources", "-q"):
                        i += 1
                        continue
                    if tok == "-r":
                        req_file = pathlib.Path(argv[i + 1])
                        reqs.extend(
                            line.strip()
                            for line in req_file.read_text(encoding="utf-8").splitlines()
                            if line.strip() and not line.strip().startswith("#")
                        )
                        i += 2
                        continue
                    reqs.append(tok)
                    i += 1
                with INSTALL_LOG.open("a", encoding="utf-8") as handle:
                    handle.write(f"{{target_py}}|{{os.environ.get('UV_CACHE_DIR', '')}}|{{','.join(reqs)}}\\n")
                for item in reqs:
                    if item.endswith(".whl") and pathlib.Path(item).is_file():
                        _install_wheel(site, pathlib.Path(item))
                    elif item.startswith("./") or pathlib.Path(item).is_dir():
                        _install_local_dir(site, pathlib.Path(item))
                    else:
                        _install_third_party(site, item)
                return 0
            return 2


        raise SystemExit(main(sys.argv[1:]))
        """
    )
    uv_path.write_text(script, encoding="utf-8")
    uv_path.chmod(0o755)
    return uv_path


def _make_job_venv(venv_dir: Path, parent_site: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv_dir)], check=True)
    site = next((venv_dir / "lib").glob("python*/site-packages"))
    (site / dependency_entry.PARENT_PATH_FILE).write_text(
        f"import site; site.addsitedir({str(parent_site)!r})\n",
        encoding="utf-8",
    )
    return venv_dir


def test_setup_script_fits_workload_setup_byte_limit_and_passes_bash_syntax() -> None:
    raw = easydel_setup.SCRIPT_PATH.read_bytes()
    assert len(raw) <= MAX_SETUP_SCRIPT_BYTES, (
        f"easydel_setup.sh ({len(raw)} bytes) exceeds MAX_SETUP_SCRIPT_BYTES ({MAX_SETUP_SCRIPT_BYTES})"
    )
    if shutil.which("bash") is not None:
        check = subprocess.run(["bash", "-n", str(easydel_setup.SCRIPT_PATH)], capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_two_commits_share_base_venv_git_mirror_and_rust_wheel_with_full_isolation(tmp_path: Path) -> None:
    """Commit B skips rebuilding Layer 1 base deps AND skips Rust compilation while isolating commits A & B."""

    repo = tmp_path / "origin-repo"
    commit_a = _init_monorepo(repo, easydel_tag="commit-A", rust_tag="rust-v1")
    (repo / "libs" / "easydel" / "easydel" / "__init__.py").write_text('COMMIT_TAG = "commit-B"\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "easydel commit-B")
    commit_b = _git(repo, "rev-parse", "HEAD")

    # Parent site provides eyvan plus a stale decoy easydel that Layer 2 must shadow.
    parent_site = tmp_path / "parent-site"
    (parent_site / "eyvan_parent_marker").mkdir(parents=True)
    (parent_site / "eyvan_parent_marker" / "__init__.py").write_text('ORIGIN = "parent"\n', encoding="utf-8")
    (parent_site / "easydel").mkdir(parents=True)
    (parent_site / "easydel" / "__init__.py").write_text('COMMIT_TAG = "stale-in-parent"\n', encoding="utf-8")

    shared_root = tmp_path / "shared-eyvan"
    runtime_envs = shared_root / "runtime-envs"
    runtime_envs.mkdir(parents=True)

    install_log = tmp_path / "installs.log"
    rust_build_log = tmp_path / "rust-builds.log"
    bin_dir = tmp_path / "bin"
    _write_real_wheel_uv(
        bin_dir,
        install_log=install_log,
        rust_build_log=rust_build_log,
        export_text=".\n./libs/easydel\n./libs/eformer\n./libs/surgecore\nheavy-thirdparty==3.1.4\n",
    )
    # Cargo stub available only during Commit A's build (absent for Commit B to prove cache hit skips Rust).
    cargo_bin = tmp_path / "cargo-bin"
    cargo_bin.mkdir()
    cargo = cargo_bin / "cargo"
    cargo.write_text("#!/bin/sh\necho cargo 1.85.0\n", encoding="utf-8")
    cargo.chmod(0o755)

    job_a = _make_job_venv(runtime_envs / "job-a", parent_site)
    job_b = _make_job_venv(runtime_envs / "job-b", parent_site)

    # --- Run 1: Commit A (cold base venv + cold git mirror + cold Rust wheel) ---
    res_a = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit_a],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{cargo_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(job_a / "bin" / "python"),
            "EYVAN_RUNTIME_ENV_DIR": str(runtime_envs),
            "TMPDIR": str(tmp_path),
            "HOME": str(tmp_path / "home-a"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_a.returncode == 0, res_a.stderr
    assert "building shared base environment" in res_a.stdout
    assert "building and caching wheel for surgecore" in res_a.stdout
    assert list((shared_root / "git-mirrors").glob("*.git")), "bare git mirror was not created"
    assert (shared_root / "uv-cache").is_dir(), "shared uv-cache was not created"
    base_markers = list((shared_root / "base-envs").glob("*/.eyvan-complete"))
    assert len(base_markers) == 1
    wheel_markers = list((shared_root / "wheel-cache").glob("surgecore-*/.eyvan-complete"))
    assert len(wheel_markers) == 1
    assert len(rust_build_log.read_text(encoding="utf-8").splitlines()) == 1

    # --- Run 2: Commit B (warm base venv + incremental git mirror + cached Rust wheel, NO cargo on PATH!) ---
    res_b = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit_b],
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(job_b / "bin" / "python"),
            "EYVAN_RUNTIME_ENV_DIR": str(runtime_envs),
            "TMPDIR": str(tmp_path),
            "HOME": str(tmp_path / "home-b"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_b.returncode == 0, res_b.stderr
    assert "reusing cached base environment" in res_b.stdout
    assert "reusing cached wheel for surgecore" in res_b.stdout

    # Third-party dep installed ONCE total; Rust wheel built ONCE total!
    install_lines = install_log.read_text(encoding="utf-8").splitlines()
    heavy_installs = [line for line in install_lines if "heavy-thirdparty==3.1.4" in line]
    assert len(heavy_installs) == 1, f"expected 1 base third-party install, got: {heavy_installs}"
    assert str(shared_root / "uv-cache") in heavy_installs[0]
    assert len(rust_build_log.read_text(encoding="utf-8").splitlines()) == 1, "Commit B must not rebuild Rust"
    assert not list(tmp_path.glob("eyvan-easydel.*")), "temporary $WORK checkout leaked"

    # Both job environments run independently with exact commit isolation + shared Layer 1 + parent site!
    probe_code = (
        "import easydel, eformer, surgecore, heavy_thirdparty, eyvan_parent_marker; "
        "print(easydel.COMMIT_TAG, eformer.TAG, surgecore.RUST_TAG, "
        "heavy_thirdparty.VERSION, eyvan_parent_marker.ORIGIN)"
    )
    probe_a = subprocess.run([str(job_a / "bin" / "python"), "-c", probe_code], capture_output=True, text=True)
    probe_b = subprocess.run([str(job_b / "bin" / "python"), "-c", probe_code], capture_output=True, text=True)
    assert probe_a.returncode == 0, probe_a.stderr
    assert probe_b.returncode == 0, probe_b.stderr
    assert probe_a.stdout.strip() == "commit-A eformer-local rust-v1 3.1.4 parent"
    assert probe_b.stdout.strip() == "commit-B eformer-local rust-v1 3.1.4 parent"

    # --- Run 3: Rust crate source change invalidates ONLY the Rust wheel, STILL reuses Layer 1 base venv! ---
    (repo / "libs" / "surgecore" / "src" / "lib.rs").write_text('pub const TAG: &str = "rust-v2";\n', encoding="utf-8")
    (repo / "libs" / "surgecore" / "python" / "surgecore" / "__init__.py").write_text(
        'RUST_TAG = "rust-v2"\n', encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "surgecore rust-v2")
    commit_c = _git(repo, "rev-parse", "HEAD")

    job_c = _make_job_venv(runtime_envs / "job-c", parent_site)
    res_c = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit_c],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{cargo_bin}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(job_c / "bin" / "python"),
            "EYVAN_RUNTIME_ENV_DIR": str(runtime_envs),
            "TMPDIR": str(tmp_path),
            "HOME": str(tmp_path / "home-c"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_c.returncode == 0, res_c.stderr
    assert "reusing cached base environment" in res_c.stdout
    assert "building and caching wheel for surgecore" in res_c.stdout
    assert len(rust_build_log.read_text(encoding="utf-8").splitlines()) == 2
    heavy_after_c = [line for line in install_log.read_text(encoding="utf-8").splitlines() if "heavy-thirdparty" in line]
    assert len(heavy_after_c) == 1

    probe_c = subprocess.run([str(job_c / "bin" / "python"), "-c", probe_code], capture_output=True, text=True)
    assert probe_c.returncode == 0, probe_c.stderr
    assert probe_c.stdout.strip() == "commit-B eformer-local rust-v2 3.1.4 parent"

    # --- Run 4: --torch-cpu hashes into a separate isolated Layer 1 base venv ---
    job_d = _make_job_venv(runtime_envs / "job-d", parent_site)
    res_d = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit_c, "--torch-cpu"],
        env={
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(job_d / "bin" / "python"),
            "EYVAN_RUNTIME_ENV_DIR": str(runtime_envs),
            "TMPDIR": str(tmp_path),
            "HOME": str(tmp_path / "home-d"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_d.returncode == 0, res_d.stderr
    assert "building shared base environment" in res_d.stdout
    assert "reusing cached wheel for surgecore" in res_d.stdout
    assert len(list((shared_root / "base-envs").glob("*/.eyvan-complete"))) == 2


def test_easydel_setup_stays_focused_on_remote_env_bootstrap(tmp_path: Path) -> None:
    """``EasyDeLSetup`` handles remote git bootstrap; local code travels via ``workspace="."``."""

    assert not hasattr(EasyDeLSetup, "from_local")
    assert not hasattr(EasyDeLSetup, "has_local_changes")
    with pytest.raises(TypeError):
        EasyDeLSetup(local_path=".")  # type: ignore[call-arg]

    if shutil.which("bash") is not None:
        rejected = subprocess.run(
            [
                "bash",
                str(easydel_setup.SCRIPT_PATH),
                "--repo",
                "https://github.com/erfanzar/EasyDeL.git",
                "--ref",
                "0" * 40,
                "--patch-chunk",
                "abc",
            ],
            env={"PATH": "/usr/bin:/bin", "EYVAN_PYTHON": sys.executable, "TMPDIR": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert rejected.returncode == 2
        assert "unknown argument: --patch-chunk" in rejected.stderr


def test_dependency_entry_wires_shared_uv_cache_and_runtime_env_dir_into_setup(tmp_path: Path, monkeypatch) -> None:
    runtime_root = tmp_path / "eyvan-host" / "runtime-envs"
    runtime_root.mkdir(parents=True)
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("EYVAN_RUNTIME_ENV_DIR", raising=False)

    assert dependency_entry.shared_cache_root(runtime_root) == tmp_path / "eyvan-host"
    assert dependency_entry.uv_cache_dir(runtime_root) == tmp_path / "eyvan-host" / "uv-cache"

    seen = tmp_path / "setup-env.txt"
    monkeypatch.setattr(
        dependency_entry,
        "_install_commands",
        lambda environment, requirements: [
            [sys.executable, "-c", f"import pathlib; pathlib.Path({str(environment)!r}).mkdir(parents=True)"]
        ],
    )
    step = dependency_entry.SetupStep(
        script=f'printf "%s|%s" "$UV_CACHE_DIR" "$EYVAN_RUNTIME_ENV_DIR" > {shlex.quote(str(seen))}\n',
        args=(),
    )
    env_dir = dependency_entry.prepare([], root=runtime_root, setup=step)
    assert (env_dir / dependency_entry.COMPLETE_MARKER).exists()
    uv_cache, env_root_seen = seen.read_text(encoding="utf-8").split("|")
    assert uv_cache == str(tmp_path / "eyvan-host" / "uv-cache")
    assert env_root_seen == str(runtime_root)
