# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``EasyDeLSetup``: a built-in setup step that installs EasyDeL from a branch, tag or commit.

The branch is resolved to one commit at submission, so every host installs the
same code and a cached install is reused only for that commit. The bundled
script handles the monorepo layout (``libs/easydel``, ...) and a single package
at the repository root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eyvan.records import WorkloadSetup
from eyvan.sdk.client import easydel_setup
from eyvan.sdk.client.easydel_setup import EasyDeLSetup

MAIN = "090a03b2e0b3606bd38e0e3fcfdc640cd7e6e246"  # pragma: allowlist secret
BRANCH = "580a86a72ba7fe9c7ee6b040d2672c55089dfad6"  # pragma: allowlist secret
TAG_OBJECT = "1" * 40
TAG_COMMIT = "2" * 40


def _pkt(line: str) -> bytes:
    data = line.encode()
    return f"{len(data) + 4:04x}".encode() + data


#: The shape GitHub's smart-HTTP endpoint answers with: a service line, a flush,
#: the first ref carrying capabilities after a NUL, then refs, then a flush.
_ADVERTISEMENT = (
    _pkt("# service=git-upload-pack\n")
    + b"0000"
    + _pkt(f"{MAIN} HEAD\0multi_ack side-band-64k symref=HEAD:refs/heads/main\n")
    + _pkt(f"{MAIN} refs/heads/main\n")
    + _pkt(f"{BRANCH} refs/heads/ragged-kda-inference-optimizations\n")
    + _pkt(f"{TAG_OBJECT} refs/tags/v0.3.0\n")
    + _pkt(f"{TAG_COMMIT} refs/tags/v0.3.0^{{}}\n")
    + b"0000"
)


@pytest.fixture
def fake_remote(monkeypatch) -> list[str]:
    asked: list[str] = []

    def fetch(repository: str) -> bytes:
        asked.append(repository)
        return _ADVERTISEMENT

    monkeypatch.setattr(easydel_setup, "_fetch_advertised_refs", fetch)
    return asked


def test_the_ref_advertisement_is_parsed_including_the_capability_line() -> None:
    refs = easydel_setup.parse_advertised_refs(_ADVERTISEMENT)

    assert refs["HEAD"] == MAIN
    assert refs["refs/heads/ragged-kda-inference-optimizations"] == BRANCH
    assert "# service=git-upload-pack" not in refs


def test_a_non_git_answer_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="advertisement"):
        easydel_setup.parse_advertised_refs(b"<html>login</html>")


def test_defaults_are_main_of_erfanzar_easydel(fake_remote) -> None:
    setup = EasyDeLSetup()

    assert (setup.branch, setup.repo, setup.accelerator, setup.torch_cpu) == (
        "main",
        "erfanzar/EasyDeL",
        "tpu",
        False,
    )
    wire = setup.to_workload_setup()
    assert wire.args == ("--repo", "https://github.com/erfanzar/EasyDeL.git", "--ref", MAIN, "--accelerator", "tpu")
    assert wire.script == easydel_setup.SCRIPT_PATH.read_text(encoding="utf-8")
    assert fake_remote == ["https://github.com/erfanzar/EasyDeL.git"]


def test_a_branch_and_a_tag_resolve_to_commits(fake_remote) -> None:
    assert EasyDeLSetup("ragged-kda-inference-optimizations").to_workload_setup().args[3] == BRANCH
    # The peeled commit, not the annotated tag object, which git cannot check out as a commit.
    assert EasyDeLSetup("v0.3.0").to_workload_setup().args[3] == TAG_COMMIT


def test_a_full_commit_skips_the_lookup(fake_remote) -> None:
    assert EasyDeLSetup(BRANCH).to_workload_setup().args[3] == BRANCH
    assert fake_remote == []


def test_an_unknown_branch_says_so(fake_remote) -> None:
    with pytest.raises(ValueError, match="not a branch or tag"):
        EasyDeLSetup("no-such-branch").to_workload_setup()


def test_another_repo_and_extras_reach_the_script(fake_remote) -> None:
    args = EasyDeLSetup("main", repo="someone/EasyDeL-fork", accelerator="cpu", torch_cpu=True).to_workload_setup().args

    assert args == (
        "--repo",
        "https://github.com/someone/EasyDeL-fork.git",
        "--ref",
        MAIN,
        "--accelerator",
        "cpu",
        "--torch-cpu",
    )
    assert EasyDeLSetup("main", repo="https://git.example.com/x/easydel.git").repository_url == (
        "https://git.example.com/x/easydel.git"
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"branch": "--upload-pack=evil"},
        {"branch": "a..b"},
        {"branch": "has space"},
        {"repo": "http://github.com/a/b"},
        {"repo": "not a repo"},
        {"repo": "git@github.com:a/b.git"},
        {"accelerator": "gpu-ish"},
    ],
)
def test_bad_values_are_refused_at_definition(fields) -> None:
    with pytest.raises(ValueError):
        EasyDeLSetup(**fields)


def test_tpu_config_resolves_the_branch_at_submission_not_at_decoration(fake_remote) -> None:
    from eyvan import EasyDeLSetup as Exported
    from eyvan import TpuConfig

    config = TpuConfig("v6e-8", setup=Exported("ragged-kda-inference-optimizations"))

    assert fake_remote == [], "decorating a function must not touch the network"
    request = config.setup_request
    assert isinstance(request, WorkloadSetup)
    assert request.args[3] == BRANCH
    assert fake_remote == ["https://github.com/erfanzar/EasyDeL.git"]


# --------------------------------------------------------------------------- the bundled script


def _have_tools() -> bool:
    return shutil.which("bash") is not None and shutil.which("git") is not None


def _commit_repo(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    git = ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _run_script(tmp_path: Path, repo: Path, commit: str, *extra: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run the bundled script with a stub uv that records each install instead of performing it."""

    binaries = tmp_path / "bin"
    binaries.mkdir(exist_ok=True)
    log = tmp_path / "uv.log"
    uv = binaries / "uv"
    # `uv export` writes a lock export like EasyDeL's (root meta-package, members by
    # path, pinned third-party lines); `uv pip install -r` copies the file it was given.
    uv.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "--version" ] && exit 0\n'
        'printf "%s\\n" "$*" >> "$UV_LOG"\n'
        'prev=""; out=""; reqs=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; [ "$prev" = "-r" ] && reqs="$a"; prev="$a"; done\n'
        'if [ "$1" = "export" ]; then\n'
        "  printf '.\\n./libs/easydel\\n    # via easydel-foundation\\n./libs/eformer\\nnumpy==2.0.0\\n' > \"$out\"\n"
        "fi\n"
        '[ -n "$reqs" ] && cat "$reqs" > "$UV_LOG.requirements"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    # A Python that answers the version check without the packages being installed.
    python = binaries / "jobpython"
    python.write_text(
        '#!/bin/sh\nif [ "$2" = "-" ]; then cat >/dev/null; echo verified; exit 0; fi\n'
        # The interpreter running the tests, not whatever `python3` PATH finds:
        # a job's EYVAN_PYTHON is always >= 3.11, and a bare macOS python3 is 3.9.
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    environment = {
        "PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "UV_LOG": str(log),
        "EYVAN_PYTHON": str(python),
        "TMPDIR": str(tmp_path),
        "HOME": str(tmp_path),
    }
    completed = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit, *extra],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    installs = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return completed, installs


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_libs_without_a_workspace_lock_are_all_installed_from_one_checkout(tmp_path) -> None:
    libs = {"easydel": "easydel", "eformer": "eformer", "ejkernel": "ejkernel", "spectrax": "spectrax-lib"}
    repo = tmp_path / "repo"
    commit = _commit_repo(
        repo, {f"libs/{lib}/pyproject.toml": f'[project]\nname = "{name}"\n' for lib, name in libs.items()}
    )

    completed, installs = _run_script(tmp_path, repo, commit)

    assert completed.returncode == 0, completed.stderr
    assert len(installs) == 1, installs
    line = installs[0]
    assert "--no-sources" in line
    assert "easydel[tpu] @ file://" in line and "/libs/easydel" in line
    assert "ejkernel[tpu] @ file://" in line
    assert "eformer @ file://" in line and "spectrax-lib @ file://" in line
    assert not list(tmp_path.glob("eyvan-easydel.*")), "the checkout was left behind"


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_a_uv_workspace_is_installed_from_its_own_lock_into_the_job_python(tmp_path) -> None:
    """EasyDeL's layout: a root meta-package whose workspace members live under libs/.

    `pip install .` of that root would take the members from PyPI, so the lock is
    exported instead: every member by path, every other dependency at its pin.
    Even when eyvan is not stripped from the lock, `--no-deps` and `--no-sources`
    must be passed so uv never re-resolves `[tool.uv.sources]` into editable
    `.pth` pointers into the temporary checkout.
    """

    repo = tmp_path / "repo"
    commit = _commit_repo(
        repo,
        {
            "pyproject.toml": '[project]\nname = "easydel-foundation"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
            "uv.lock": "version = 1\n",
            "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\n',
            "libs/eformer/pyproject.toml": '[project]\nname = "eformer"\n',
        },
    )

    completed, calls = _run_script(tmp_path, repo, commit)

    assert completed.returncode == 0, completed.stderr
    export, install = calls
    assert export.startswith("export --frozen --package easydel --no-dev --no-editable")
    assert "--extra tpu" in export
    assert install.startswith("pip install --python ") and " -r " in install
    assert "--no-deps" in install
    assert "--no-sources" in install
    installed = (tmp_path / "uv.log.requirements").read_text(encoding="utf-8").splitlines()
    assert "." not in installed, "the empty root meta-package was installed"
    assert {"./libs/easydel", "./libs/eformer", "numpy==2.0.0"} <= set(installed)
    assert not list(tmp_path.glob("eyvan-easydel.*")), "the checkout was left behind"


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_the_script_installs_a_root_package_with_torch_first(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _commit_repo(repo, {"pyproject.toml": '[project]\nname = "easydel"\n'})

    completed, installs = _run_script(tmp_path, repo, commit, "--accelerator", "cpu", "--torch-cpu")

    assert completed.returncode == 0, completed.stderr
    assert "download.pytorch.org/whl/cpu torch" in installs[0]
    assert "easydel @ file://" in installs[1] and "[" not in installs[1].split("@")[0]


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
@pytest.mark.parametrize("ref", ["main", "0" * 40])
def test_the_script_refuses_anything_but_an_existing_full_commit(tmp_path, ref) -> None:
    repo = tmp_path / "repo"
    _commit_repo(repo, {"pyproject.toml": '[project]\nname = "easydel"\n'})

    completed, installs = _run_script(tmp_path, repo, ref)

    assert completed.returncode != 0
    assert installs == []


def test_the_script_ships_inside_the_package() -> None:
    """The wheel carries non-Python files under eyvan/; the script must be beside its module."""

    assert easydel_setup.SCRIPT_PATH.is_file()
    assert easydel_setup.SCRIPT_PATH.parent == Path(easydel_setup.__file__).parent


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_workspace_with_cargo_member_finds_home_cargo_or_bootstraps_rustup(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _commit_repo(
        repo,
        {
            "pyproject.toml": '[project]\nname = "easydel-foundation"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
            "uv.lock": "version = 1\n",
            "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\n',
            "libs/surgecore/pyproject.toml": '[project]\nname = "surgecore"\n',
            "libs/surgecore/Cargo.toml": '[package]\nname = "surgecore"\nversion = "0.1.0"\n',
        },
    )

    def _run_with_surgecore_export(run_dir: Path, curl_script: str) -> subprocess.CompletedProcess:
        binaries = run_dir / "bin"
        binaries.mkdir(exist_ok=True)
        uv = binaries / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            '[ "$1" = "--version" ] && exit 0\n'
            'prev=""; out=""\n'
            'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
            'if [ "$1" = "export" ]; then\n'
            "  printf '.\\n./libs/easydel\\n./libs/surgecore\\n' > \"$out\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        python = binaries / "jobpython"
        python.write_text(
            '#!/bin/sh\nif [ "$2" = "-" ]; then cat >/dev/null; echo verified; exit 0; fi\n'
            f'exec "{sys.executable}" "$@"\n',
            encoding="utf-8",
        )
        python.chmod(0o755)
        curl = binaries / "curl"
        curl.write_text(curl_script, encoding="utf-8")
        curl.chmod(0o755)
        return subprocess.run(
            ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit],
            env={
                "PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin",
                "EYVAN_PYTHON": str(python),
                "TMPDIR": str(run_dir),
                "HOME": str(run_dir),
            },
            capture_output=True,
            text=True,
            check=False,
        )

    # Case 1: cargo exists under $HOME/.cargo/bin (not initially on PATH).
    run1 = tmp_path / "run1"
    run1.mkdir()
    cargo_bin = run1 / ".cargo" / "bin"
    cargo_bin.mkdir(parents=True)
    cargo = cargo_bin / "cargo"
    cargo.write_text("#!/bin/sh\necho cargo 1.85.0\n", encoding="utf-8")
    cargo.chmod(0o755)
    curl_log1 = run1 / "curl.log"
    completed1 = _run_with_surgecore_export(run1, f'#!/bin/sh\necho "$*" >> "{curl_log1}"\nexit 1\n')
    assert completed1.returncode == 0, completed1.stderr
    assert not curl_log1.exists(), "rustup must not be fetched when $HOME/.cargo/bin/cargo already exists"

    # Case 2: cargo is absent everywhere; stub curl writes a tiny rustup installer to -o.
    run2 = tmp_path / "run2"
    run2.mkdir()
    curl_log2 = run2 / "curl.log"
    completed2 = _run_with_surgecore_export(
        run2,
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{curl_log2}"\n'
        'prev=""; out=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        "cat <<'EOF' > \"$out\"\n"
        "#!/bin/sh\n"
        'mkdir -p "$HOME/.cargo/bin"\n'
        'printf "#!/bin/sh\\necho cargo 1.85.0\\n" > "$HOME/.cargo/bin/cargo"\n'
        'chmod +x "$HOME/.cargo/bin/cargo"\n'
        "EOF\n",
    )
    assert completed2.returncode == 0, completed2.stderr
    assert "https://sh.rustup.rs" in curl_log2.read_text(encoding="utf-8")
    assert (run2 / ".cargo" / "bin" / "cargo").is_file()


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_workspace_lock_skips_eyvan_when_installed_and_scopes_metadata_check(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _commit_repo(
        repo,
        {
            "pyproject.toml": '[project]\nname = "easydel-foundation"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
            "uv.lock": "version = 1\n",
            "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\n',
            "libs/eformer/pyproject.toml": '[project]\nname = "eformer"\n',
            "libs/eyvan/pyproject.toml": '[project]\nname = "eyvan"\n',
        },
    )
    binaries = tmp_path / "bin"
    binaries.mkdir(exist_ok=True)
    log = tmp_path / "uv.log"
    verified_log = tmp_path / "verified.log"
    uv = binaries / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "--version" ] && exit 0\n'
        'printf "%s\\n" "$*" >> "$UV_LOG"\n'
        'prev=""; out=""; reqs=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; [ "$prev" = "-r" ] && reqs="$a"; prev="$a"; done\n'
        'if [ "$1" = "export" ]; then\n'
        "  printf '.\\n./libs/easydel\\n./libs/eformer\\n./libs/eyvan\\nnumpy==2.0.0\\n' > \"$out\"\n"
        "fi\n"
        '[ -n "$reqs" ] && cat "$reqs" > "$UV_LOG.requirements"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    python = binaries / "jobpython"
    probe_log = tmp_path / "eyvan-probe.log"
    python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-I" ] && [ "$2" = "-c" ] && printf "%s" "$3" | grep -q \'m.version("eyvan")\'; then\n'
        f'  echo probed >> "{probe_log}"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$2" = "-" ]; then\n'
        "  cat >/dev/null\n"
        "  shift 2\n"
        f'  printf "%s\\n" "$@" > "{verified_log}"\n'
        "  echo verified\n"
        "  exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    env = {
        "PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "UV_LOG": str(log),
        "EYVAN_PYTHON": str(python),
        "TMPDIR": str(tmp_path),
        "HOME": str(tmp_path),
    }
    completed = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert probe_log.read_text(encoding="utf-8").strip() == "probed"
    calls = log.read_text(encoding="utf-8").splitlines()
    install_call = next(line for line in calls if line.startswith("pip install "))
    assert "--no-deps" in install_call
    assert "--no-sources" in install_call
    installed = (tmp_path / "uv.log.requirements").read_text(encoding="utf-8").splitlines()
    assert "./libs/eyvan" not in installed
    assert {"./libs/easydel", "./libs/eformer", "numpy==2.0.0"} <= set(installed)
    checked_paths = verified_log.read_text(encoding="utf-8").splitlines()
    assert any(p.endswith("/libs/easydel/pyproject.toml") for p in checked_paths)
    assert any(p.endswith("/libs/eformer/pyproject.toml") for p in checked_paths)
    assert not any(p.endswith("/libs/eyvan/pyproject.toml") for p in checked_paths)


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_setup_script_succeeds_with_home_unset(tmp_path) -> None:
    repo = tmp_path / "repo"
    commit = _commit_repo(repo, {"pyproject.toml": '[project]\nname = "easydel"\n'})
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    python = binaries / "jobpython"
    python.write_text(
        f'#!/bin/sh\nif [ "$2" = "-" ]; then cat >/dev/null; echo verified; exit 0; fi\nexec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    python.chmod(0o755)
    # Deliberately omit HOME from the subprocess environment.
    completed = subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit],
        env={
            "PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(python),
            "TMPDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _run_real_verification_check(tmp_path: Path, *, mode: str) -> subprocess.CompletedProcess[str]:
    """Run the bundled script with its real post-install Python verifier against a temp venv."""

    repo = tmp_path / f"repo-{mode}"
    commit = _commit_repo(
        repo,
        {
            "pyproject.toml": '[project]\nname = "easydel-foundation"\n\n[tool.uv.workspace]\nmembers = ["libs/*"]\n',
            "uv.lock": "version = 1\n",
            "libs/easydel/pyproject.toml": '[project]\nname = "easydel"\n',
            "libs/eformer/pyproject.toml": '[project]\nname = "eformer"\n',
        },
    )
    run_dir = tmp_path / f"run-{mode}"
    run_dir.mkdir()
    venv = run_dir / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    site_packages = next((venv / "lib").glob("python*/site-packages"))
    binaries = run_dir / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    # Simulate uv writing installed metadata into the real venv's site-packages
    # while cwd is $SRC (inside $WORK), then let the script's real verification
    # snippet run unmodified against that venv.
    uv.write_text(
        "#!/bin/sh\n"
        '[ "$1" = "--version" ] && exit 0\n'
        'prev=""; out=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'if [ "$1" = "export" ]; then\n'
        "  printf '.\\n./libs/easydel\\n./libs/eformer\\n' > \"$out\"\n"
        "  exit 0\n"
        "fi\n"
        f'"{sys.executable}" - "$PWD" "{site_packages}" "{mode}" <<\'PY\'\n'
        "import json, pathlib, sys\n"
        "src = pathlib.Path(sys.argv[1]).resolve()\n"
        "sp = pathlib.Path(sys.argv[2])\n"
        "mode = sys.argv[3]\n"
        "for pkg in ('easydel', 'eformer'):\n"
        "    dist_info = sp / f'{pkg}-0.1.0.dist-info'\n"
        "    dist_info.mkdir(parents=True, exist_ok=True)\n"
        "    (dist_info / 'METADATA').write_text(f'Metadata-Version: 2.1\\nName: {pkg}\\nVersion: 0.1.0\\n')\n"
        "    record_lines = [f'{pkg}-0.1.0.dist-info/METADATA,,', f'{pkg}-0.1.0.dist-info/RECORD,,']\n"
        "    target = src / 'libs' / pkg\n"
        "    if mode == 'non_editable':\n"
        "        du = {'dir_info': {}, 'url': target.as_uri()}\n"
        "        (dist_info / 'direct_url.json').write_text(json.dumps(du))\n"
        "        record_lines.append(f'{pkg}-0.1.0.dist-info/direct_url.json,,')\n"
        "    elif mode == 'editable_direct_url' and pkg == 'eformer':\n"
        "        du = {'dir_info': {'editable': True}, 'url': target.as_uri()}\n"
        "        (dist_info / 'direct_url.json').write_text(json.dumps(du))\n"
        "        record_lines.append(f'{pkg}-0.1.0.dist-info/direct_url.json,,')\n"
        "    elif mode == 'pth_leak' and pkg == 'eformer':\n"
        "        (sp / 'eformer.pth').write_text(str(target) + '\\n')\n"
        "        record_lines.append('eformer.pth,,')\n"
        "    (dist_info / 'RECORD').write_text('\\n'.join(record_lines) + '\\n')\n"
        "PY\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return subprocess.run(
        ["bash", str(easydel_setup.SCRIPT_PATH), "--repo", str(repo), "--ref", commit],
        env={
            "PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "EYVAN_PYTHON": str(venv / "bin" / "python"),
            "TMPDIR": str(run_dir),
            "HOME": str(run_dir),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_post_install_verification_rejects_editable_or_pth_references_into_work_and_allows_non_editable(
    tmp_path: Path,
) -> None:
    """Catch editable direct_url.json or .pth pointers into $WORK before the EXIT trap deletes $WORK."""

    editable = _run_real_verification_check(tmp_path, mode="editable_direct_url")
    assert editable.returncode != 0
    assert "eformer" in editable.stderr
    assert "editable" in editable.stderr

    pth_leak = _run_real_verification_check(tmp_path, mode="pth_leak")
    assert pth_leak.returncode != 0
    assert "eformer" in pth_leak.stderr
    assert "eformer.pth" in pth_leak.stderr

    clean = _run_real_verification_check(tmp_path, mode="non_editable")
    assert clean.returncode == 0, clean.stderr
    assert "easydel setup: easydel 0.1.0" in clean.stdout
    assert "easydel setup: eformer 0.1.0" in clean.stdout
