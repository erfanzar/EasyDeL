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
VNEXT = "17513c559175ba0eb61ac98b1a8ce2f0d2a2a54f"  # pragma: allowlist secret
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
    + _pkt(f"{VNEXT} refs/heads/vnext-into-main\n")
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


def test_defaults_are_vnext_into_main_of_erfanzar_easydel(fake_remote) -> None:
    setup = EasyDeLSetup()

    assert (setup.branch, setup.repo, setup.accelerator, setup.torch_cpu) == (
        "vnext-into-main",
        "erfanzar/EasyDeL",
        "tpu",
        False,
    )
    wire = setup.to_workload_setup()
    assert wire.args == ("--repo", "https://github.com/erfanzar/EasyDeL.git", "--ref", VNEXT, "--accelerator", "tpu")
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
    assert "easydel[tpu] @ file://" in line and "/libs/easydel" in line
    assert "ejkernel[tpu] @ file://" in line
    assert "eformer @ file://" in line and "spectrax-lib @ file://" in line
    assert not list(tmp_path.glob("eyvan-easydel.*")), "the checkout was left behind"


@pytest.mark.skipif(not _have_tools(), reason="needs bash and git")
def test_a_uv_workspace_is_installed_from_its_own_lock_into_the_job_python(tmp_path) -> None:
    """EasyDeL's layout: a root meta-package whose workspace members live under libs/.

    `pip install .` of that root would take the members from PyPI, so the lock is
    exported instead: every member by path, every other dependency at its pin.
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
    assert export.startswith("export --frozen --all-packages --no-dev --no-editable")
    assert "--extra tpu" in export
    assert install.startswith("pip install --python ") and " -r " in install
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
