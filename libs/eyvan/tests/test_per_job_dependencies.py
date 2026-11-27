# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adding a package to one job without rebuilding an image.

The half of a per-job runtime environment Eyvan did not have. A workload could
already carry its working tree and its variables; its dependencies could only be
whatever the host image happened to hold, so adding one library meant building
an image and rolling a fleet -- which is exactly the friction people reach for
Ray to avoid.

Most of what is tested here is the failure side, because the two ways this goes
wrong are both silent. An install that fails and is ignored produces an
ImportError minutes later, on an accelerator, blamed on the user's code. And a
clean virtualenv on a TPU host hides `jax` and `libtpu`, so a job that asked for
one small package would find its accelerator stack gone.
"""

from __future__ import annotations

import os
import sys
import typing as tp
from pathlib import Path

import pytest

from eyvan.records import CallableExecution, CommandExecution, WorkloadDefinition
from eyvan.worker.runtime import dependency_entry

# --------------------------------------------------------------------------- what a job may declare


def _spec(*dependencies: str, callable_execution: bool = False) -> WorkloadDefinition:
    execution = (
        CallableExecution(artifact_uri="file:///tmp/x.pkl", sha256="0" * 64, python="3.13")
        if callable_execution
        else CommandExecution(command="python train.py")
    )
    return WorkloadDefinition(name="run", execution=execution, dependencies=dependencies)


def test_a_job_may_name_the_packages_it_needs() -> None:
    assert _spec("jax>=0.4.30", "flax").dependencies == ("jax>=0.4.30", "flax")


@pytest.mark.parametrize("option", ["-r requirements.txt", "--index-url https://pypi.example/simple", "-e ."])
def test_an_installer_option_is_not_a_dependency(option: str) -> None:
    """Both of these are honoured by pip and neither is a package.

    ``-r`` names a file on the *host*, which nothing shipped there, and
    ``--index-url`` points the install at somewhere the submitter chose. A
    dependency list that quietly accepts either is not a dependency list.
    """

    with pytest.raises(ValueError, match="installer option"):
        _spec(option)


def test_a_callable_carries_dependencies_too() -> None:
    """The first draft refused this, and the reasoning was wrong.

    The wrapper `exec`s, so the callable worker starts *under* the prepared
    environment: the interpreter is still the host's, with the job's packages
    layered over it, which is exactly what the pickled function's imports
    resolve against. A callable is the case that needs this most -- its
    dependencies are whatever the pickle happens to reference.
    """

    assert _spec("flax", callable_execution=True).dependencies == ("flax",)


# --------------------------------------------------------------------------- the environment that gets built


def test_the_same_packages_in_a_different_order_are_the_same_environment() -> None:
    """Order is not part of what was asked for, and two keys means two installs."""

    assert dependency_entry.environment_key(["flax", "jax"]) == dependency_entry.environment_key(["jax", "flax"])
    assert dependency_entry.environment_key(["jax"]) != dependency_entry.environment_key(["jax", "flax"])


def test_old_requirement_only_cache_is_not_reused(tmp_path, monkeypatch) -> None:
    import hashlib

    old = tmp_path / hashlib.sha256(b"flax").hexdigest()[:32]
    old.mkdir()
    (old / dependency_entry.COMPLETE_MARKER).write_text("flax")
    parent = tmp_path / "parent-site"
    parent.mkdir()
    monkeypatch.setattr(dependency_entry.site, "getsitepackages", lambda: [str(parent)])
    monkeypatch.setattr(dependency_entry.site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(
        dependency_entry, "_install_commands", lambda environment, requirements: _fake_install(environment)
    )
    built = dependency_entry.prepare(["flax"], root=tmp_path)
    assert built != old
    paths = list(built.rglob(dependency_entry.PARENT_PATH_FILE))
    assert len(paths) == 1
    assert paths[0].read_text() == f"import site; site.addsitedir({str(parent)!r})\n"
    assert (old / dependency_entry.COMPLETE_MARKER).read_text() == "flax"
    assert list(old.iterdir()) == [old / dependency_entry.COMPLETE_MARKER]


def test_dependency_cache_key_includes_interpreter_and_parent_paths(tmp_path, monkeypatch) -> None:
    original = dependency_entry.environment_key(["flax"])
    monkeypatch.setattr(dependency_entry.sys, "executable", str(tmp_path / "other-python"))
    other = dependency_entry.environment_key(["flax"])
    assert other != original
    parent = tmp_path / "parent-site"
    parent.mkdir()
    monkeypatch.setattr(dependency_entry.site, "getsitepackages", lambda: [str(parent)])
    assert dependency_entry.environment_key(["flax"]) != other
    assert dependency_entry.environment_key(["jax", " flax ", "jax"]) == dependency_entry.environment_key(
        ["flax", "jax"]
    )


def test_the_environment_can_see_the_host_accelerator_stack() -> None:
    """The decision this whole feature stands on.

    A clean virtualenv on a TPU host has no ``jax`` and no ``libtpu``, so a job
    that asked for one unrelated package would install a second, CPU-only JAX
    over a working one and fail on its first device call.
    """

    commands = dependency_entry._install_commands(Path("/tmp/env"), ["flax"])

    assert any("--system-site-packages" in command for command in commands)


def _fake_uv(directory: Path, *, exit_code: int) -> Path:
    """A `uv` on PATH whose `--version` exits with ``exit_code`` (a snap refusing to start is non-zero)."""

    directory.mkdir(parents=True, exist_ok=True)
    uv = directory / "uv"
    uv.write_text(f"#!/bin/sh\necho snap-confine refused >&2\nexit {exit_code}\n")
    uv.chmod(0o755)
    return uv


@pytest.mark.parametrize(("exit_code", "uses_uv"), [(0, True), (1, False)])
def test_a_uv_that_cannot_run_falls_back_to_the_interpreters_own_pip(tmp_path, monkeypatch, exit_code, uses_uv) -> None:
    """Fall back to interpreter-owned pip when uv exists on PATH but cannot run in the sandbox."""

    uv = _fake_uv(tmp_path / "bin", exit_code=exit_code)
    monkeypatch.setenv("PATH", str(uv.parent))

    commands = dependency_entry._install_commands(Path("/tmp/env"), ["flax"])

    assert (commands[0][0] == str(uv)) is uses_uv
    if not uses_uv:
        assert commands[0][:3] == [sys.executable, "-m", "venv"]
        assert commands[1][1:4] == ["-m", "pip", "install"]
    # Either path keeps the host accelerator stack visible.
    assert any("--system-site-packages" in command for command in commands)


def test_no_uv_on_path_uses_the_interpreters_own_pip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    commands = dependency_entry._install_commands(Path("/tmp/env"), ["flax"])

    assert commands[0][:3] == [sys.executable, "-m", "venv"]


def _fake_install(environment: Path) -> list[list[str]]:
    """Stand in for `uv venv`, which is a command that creates the directory."""

    return [[sys.executable, "-c", f"import pathlib; pathlib.Path({str(environment)!r}).mkdir(parents=True)"]]


def test_a_failed_install_fails_the_job_rather_than_starting_it(tmp_path, monkeypatch) -> None:
    """Running anyway produces an ImportError minutes later on an accelerator."""

    monkeypatch.setattr(
        dependency_entry,
        "_install_commands",
        lambda environment, requirements: [[sys.executable, "-c", "import sys; sys.exit(7)"]],
    )

    with pytest.raises(RuntimeError, match="failed"):
        dependency_entry.prepare(["nonexistent-package"], root=tmp_path)


def test_a_crashed_install_is_rebuilt_rather_than_trusted(tmp_path, monkeypatch) -> None:
    """The marker exists so a half-populated directory is not mistaken for one.

    A preemption during install leaves a virtualenv with some of the packages
    in it, which is indistinguishable from a good one by anything except the
    marker that is written last.
    """

    partial = tmp_path / dependency_entry.environment_key(["flax"])
    partial.mkdir(parents=True)
    (partial / "leftover").write_text("from an attempt that died")
    built: list[Path] = []

    def record(environment: Path, requirements):
        built.append(environment)
        return _fake_install(environment)

    monkeypatch.setattr(dependency_entry, "_install_commands", record)

    dependency_entry.prepare(["flax"], root=tmp_path)

    assert built == [partial]
    assert not (partial / "leftover").exists(), "the dead attempt's directory was reused"
    assert (partial / dependency_entry.COMPLETE_MARKER).exists()


def test_a_complete_environment_is_reused_rather_than_rebuilt(tmp_path, monkeypatch) -> None:
    """Eight ranks on one host declare the same packages; one of them installs."""

    calls: list[Path] = []
    monkeypatch.setattr(
        dependency_entry,
        "_install_commands",
        lambda environment, requirements: (calls.append(environment), _fake_install(environment))[1],
    )

    first = dependency_entry.prepare(["flax"], root=tmp_path)
    second = dependency_entry.prepare(["flax"], root=tmp_path)

    assert first == second
    assert len(calls) == 1


def test_activation_prepends_rather_than_replacing_the_path() -> None:
    """An accelerator host keeps vendor binaries outside any virtual environment."""

    activated = dependency_entry.activate(Path("/envs/abc"), {"PATH": "/usr/bin"})

    assert activated["PATH"] == f"/envs/abc/bin{os.pathsep}/usr/bin"
    assert activated["VIRTUAL_ENV"] == "/envs/abc"


# --------------------------------------------------------------------------- how it reaches the host


def _entrypoint(*dependencies: str, command: str = "python train.py") -> str:
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    return _ControllerWorkloadCommands._dependency_entrypoint(dependencies, command)


def test_a_job_declaring_nothing_runs_exactly_what_was_written() -> None:
    """No wrapper, no cache lookup, no second process for a job that asked for none."""

    assert _entrypoint() == "python train.py"


def test_a_version_specifier_survives_the_shell() -> None:
    """`jax>=0.4` unquoted is a redirect that truncates a file called `=0.4`."""

    wrapped = _entrypoint("jax>=0.4.30")

    assert "'jax>=0.4.30'" in wrapped
    assert wrapped.endswith(" python train.py")
    assert "eyvan.worker.runtime.dependency_entry" in wrapped


def test_the_wrapper_runs_before_the_workspace_entry() -> None:
    """Order matters: the workspace entry is itself `python -m`.

    It resolves through PATH, and the packages have to be in place by then --
    including `eyvan` itself, which the entry imports. Installing after would
    put the environment on PATH for nothing, because the command has already
    been exec'd by then. And the workload command is emitted verbatim after
    the wrapper's own `--`: a `--` inside the job's command is end-of-options
    to the job, not to this wrapper.
    """

    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    wrapped = _ControllerWorkloadCommands._dependency_entrypoint(
        ("flax",),
        "python -m eyvan.worker.runtime.workspace_entry --workspace-uri file:///w -- python train.py -- --fast",
    )

    assert wrapped.index("dependency_entry") < wrapped.index("workspace_entry")
    assert wrapped.endswith("-- python train.py -- --fast")


_FAKE_WRAPPER_PYTHON = """#!/bin/sh
# Stands in for `python -m <entry> ... -- COMMAND`: marks the environment, then
# execs what follows its `--`, as the real dependency/workspace entries do.
[ "$1" = -I ] && shift
[ "$1" = -m ] || exit 97
case "$2" in
  eyvan.worker.runtime.dependency_entry) export INSIDE_DEPENDENCIES=1 ;;
  eyvan.worker.runtime.workspace_entry) export INSIDE_WORKSPACE=1 ;;
  *) exit 98 ;;
esac
shift 2
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done
shift
exec "$@"
"""


def _run_through_host_shell(tmp_path: Path, entrypoint: str) -> str:
    """Run an entrypoint the way the native host does (`/bin/sh -c`), with the fake wrapper first on PATH."""

    import subprocess

    binaries = tmp_path / "bin"
    binaries.mkdir()
    fake = binaries / "python"
    fake.write_text(_FAKE_WRAPPER_PYTHON, encoding="utf-8")
    fake.chmod(0o755)
    environment = {"PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin"}
    completed = subprocess.run(
        ["/bin/sh", "-c", entrypoint], cwd=tmp_path, env=environment, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return (tmp_path / "seen.txt").read_text(encoding="utf-8")


#: Every line of a compound command must run inside its declared dependency
#: environment, including shell builtins and lines following the first one.
_COMPOUND_COMMANDS = (
    'set -e\necho "a=${INSIDE_DEPENDENCIES:-no}" > seen.txt\necho "b=${INSIDE_DEPENDENCIES:-no}" >> seen.txt',
    'echo "a=${INSIDE_DEPENDENCIES:-no}" > seen.txt && echo "b=${INSIDE_DEPENDENCIES:-no}" >> seen.txt',
    'MARK=1; echo "a=${INSIDE_DEPENDENCIES:-no}" > seen.txt; echo "b=${INSIDE_DEPENDENCIES:-no}" >> seen.txt',
    'cd . && echo "a=${INSIDE_DEPENDENCIES:-no}" > seen.txt | true; echo "b=${INSIDE_DEPENDENCIES:-no}" >> seen.txt',
)


@pytest.mark.parametrize("command", _COMPOUND_COMMANDS)
def test_every_part_of_a_compound_command_runs_with_its_packages(tmp_path, command) -> None:
    seen = _run_through_host_shell(tmp_path, _entrypoint("flax", command=command))

    assert "a=no" not in seen and "b=no" not in seen, seen
    assert "b=1" in seen


def test_a_compound_command_runs_inside_both_the_workspace_and_its_packages(tmp_path) -> None:
    """The workspace entry appended the command the same way and had the same flaw."""

    from eyvan.records import WorkspaceRef
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    command = (
        "set -e\n"
        'echo "ws=${INSIDE_WORKSPACE:-no} deps=${INSIDE_DEPENDENCIES:-no}" > seen.txt\n'
        'echo "ws=${INSIDE_WORKSPACE:-no} deps=${INSIDE_DEPENDENCIES:-no}" >> seen.txt'
    )
    workspace = WorkspaceRef(uri="file:///w.tar.gz", sha256="0" * 64)
    inner, _environment = _ControllerWorkloadCommands._workspace_entrypoint(
        tp_cast_any(object()), tp_cast_any(None), workspace, command, {}
    )
    seen = _run_through_host_shell(tmp_path, _ControllerWorkloadCommands._dependency_entrypoint(("flax",), inner))

    assert seen.splitlines() == ["ws=1 deps=1", "ws=1 deps=1"]


_MULTIHOST_HOOK = """import os
if os.environ.get("EYVAN_JAX_MULTIHOST") == "1":
    # Like eyvan_multihost: consume the marker, then join (here: record which process joined).
    os.environ.pop("EYVAN_JAX_MULTIHOST")
    with open(os.environ["HOOK_LOG"], "a") as handle:
        handle.write(os.environ.get("VIRTUAL_ENV", "outside-the-job-environment") + "\\n")
"""


def test_the_multihost_join_runs_in_the_job_not_in_the_dependency_wrapper(tmp_path) -> None:
    """Defer multihost sitecustomize initialization to the job interpreter, not the dependency wrapper.

    The wrapper must not consume ``EYVAN_JAX_MULTIHOST`` before the job enters
    its prepared environment and can import JAX.
    """

    import subprocess

    binaries = tmp_path / "bin"
    binaries.mkdir()
    shim = binaries / "python"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    hook_directory = tmp_path / "multihost"
    hook_directory.mkdir()
    (hook_directory / "sitecustomize.py").write_text(_MULTIHOST_HOOK, encoding="utf-8")
    (tmp_path / "job.py").write_text(
        "import os\nopen('job.txt', 'w').write(os.environ.get('EYVAN_JAX_MULTIHOST', 'consumed'))\n",
        encoding="utf-8",
    )
    environments = tmp_path / "envs"
    environments.mkdir()
    base = {"PATH": f"{binaries}{os.pathsep}/usr/bin{os.pathsep}/bin", "HOME": str(tmp_path)}
    # Prepared ahead, so nothing is installed: the key is computed by the same interpreter the wrapper runs.
    key = subprocess.run(
        [
            str(shim),
            "-I",
            "-c",
            "from eyvan.worker.runtime.dependency_entry import environment_key; "
            "print(environment_key(['eyvan-test-requirement']))",
        ],
        env=base,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    environment = environments / key
    environment.mkdir()
    (environment / dependency_entry.COMPLETE_MARKER).write_text("eyvan-test-requirement", encoding="utf-8")
    job_environment = {
        **base,
        "PYTHONPATH": str(hook_directory),
        "EYVAN_JAX_MULTIHOST": "1",
        "EYVAN_RUNTIME_ENV_DIR": str(environments),
        "HOOK_LOG": str(tmp_path / "hooks.txt"),
    }

    completed = subprocess.run(
        ["/bin/sh", "-c", _entrypoint("eyvan-test-requirement", command="python job.py")],
        cwd=tmp_path,
        env=job_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    joined = (tmp_path / "hooks.txt").read_text(encoding="utf-8").splitlines()
    assert joined == [str(environment)], "the hook must run once, inside the job's own environment"
    assert (tmp_path / "job.txt").read_text(encoding="utf-8") == "consumed"


def tp_cast_any(value: object) -> tp.Any:
    """Hand the unbound helper a stand-in where only file:// workspaces are exercised."""

    return value


@pytest.mark.parametrize(
    "command",
    ("python train.py", "python tool.py -- --fast", "'jax>=0.4' --flag"),
)
def test_a_plain_command_is_still_appended_verbatim(command) -> None:
    """One exec'd program keeps the PID systemd started; only shell syntax needs a shell."""

    assert _entrypoint("flax", command=command).endswith(" -- " + command)


def test_the_workload_command_keeps_its_own_end_of_options_marker(tmp_path, monkeypatch) -> None:
    """`python tool.py -- --fast` must not exec as `python tool.py --fast`.

    The wrapper consumes only its own end-of-options marker; stripping every
    bare `--` from the remainder silently rewrote the job's command line.
    """

    environment = tmp_path / "env"
    environment.mkdir()
    monkeypatch.setattr(dependency_entry, "prepare", lambda requirements, **_options: environment)

    class Executed(BaseException):
        """What `os.execvp` would have run, instead of replacing the process."""

        def __init__(self, command: list[str]) -> None:
            self.command = command

    def fake_execvp(program: str, arguments: list[str]) -> None:
        del program  # execvp(command[0], command): the full line is already in arguments
        raise Executed(list(arguments))

    monkeypatch.setattr(dependency_entry.os, "execvp", fake_execvp)
    saved = dict(os.environ)  # activate() rewrites the process environment
    try:
        for argv in (
            ["--require", "flax", "--"],
            ["--require", "flax"],
        ):
            with pytest.raises(Executed) as raised:
                dependency_entry.main([*argv, "python", "tool.py", "--", "--fast"])

            assert raised.value.command == ["python", "tool.py", "--", "--fast"]
    finally:
        os.environ.clear()
        os.environ.update(saved)


# --------------------------------------------------------------------------- what the CLI sends


def test_a_requirements_file_is_read_here_rather_than_on_the_host(tmp_path) -> None:
    """The whole reason the flag exists rather than passing `-r` through.

    That path names a file on the submitter's machine. The host has never seen
    it, so pip would either fail or read something else with the same name.
    """

    from eyvan.sdk.commands.workloads import _requirements

    listing = tmp_path / "requirements.txt"
    listing.write_text("# training\nflax==0.8.0\n\njax>=0.4.30  # pinned upstream\n", encoding="utf-8")

    assert _requirements(("optax",), listing) == ("optax", "flax==0.8.0", "jax>=0.4.30")


def test_a_requirements_file_that_redirects_the_install_is_refused(tmp_path) -> None:
    """`--index-url` inside the file does on the host what passing it would."""

    import click

    from eyvan.sdk.commands.workloads import _requirements

    listing = tmp_path / "requirements.txt"
    listing.write_text("--index-url https://pypi.example/simple\nflax\n", encoding="utf-8")

    with pytest.raises(click.UsageError, match="installer option"):
        _requirements((), listing)


def test_the_same_package_named_twice_is_installed_once(tmp_path) -> None:
    from eyvan.sdk.commands.workloads import _requirements

    listing = tmp_path / "requirements.txt"
    listing.write_text("flax\n", encoding="utf-8")

    assert _requirements(("flax", "optax"), listing) == ("flax", "optax")


def test_a_callable_dispatch_installs_before_the_worker_starts() -> None:
    """The wrapper is outermost on this path too.

    The callable worker resolves `python` through PATH and imports `eyvan`, so
    the environment has to exist before it runs -- and the pickled function's
    own imports resolve after that, inside the same process.
    """

    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    wrapped = _ControllerWorkloadCommands._dependency_entrypoint(
        ("flax",),
        "python -m eyvan.worker.runtime.callable_worker --sha256 abc --python 3.13",
    )

    assert wrapped.index("dependency_entry") < wrapped.index("callable_worker")


# --------------------------------------------------------------------------- where the packages land


def test_an_unwritable_host_path_prefers_user_home_over_cwd(tmp_path, monkeypatch, capsys) -> None:
    """Local cells without /var/lib write access cache under ~/.eyvan instead of polluting cwd."""

    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("EYVAN_RUNTIME_ENV_DIR", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(dependency_entry, "DEFAULT_CACHE_ROOT", Path("/proc/eyvan-cannot-write-here"))
    monkeypatch.chdir(repo)

    resolved = dependency_entry.cache_root()

    assert resolved == home / ".eyvan" / "runtime-envs"
    assert not (repo / dependency_entry.FALLBACK_CACHE_DIR).exists()
    assert str(resolved) in capsys.readouterr().err


def test_an_unwritable_host_path_falls_back_to_the_working_directory(tmp_path, monkeypatch, capsys) -> None:
    """When neither /var/lib nor HOME is usable, fall back to the working directory so the job still runs."""

    monkeypatch.delenv("EYVAN_RUNTIME_ENV_DIR", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(dependency_entry, "DEFAULT_CACHE_ROOT", Path("/proc/eyvan-cannot-write-here"))
    monkeypatch.chdir(tmp_path)

    assert dependency_entry.cache_root() == tmp_path / dependency_entry.FALLBACK_CACHE_DIR


def test_the_fallback_says_which_later_runs_reuse_the_install(tmp_path, monkeypatch, capsys) -> None:
    """The consequence, not just the fact.

    A silent fallback turns into minutes of reinstall on every attempt with no
    stated cause -- and the earlier wording claimed the install was never reused,
    which is wrong for the ordinary case of running twice from one directory.
    """

    monkeypatch.delenv("EYVAN_RUNTIME_ENV_DIR", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr(dependency_entry, "DEFAULT_CACHE_ROOT", Path("/proc/eyvan-cannot-write-here"))
    monkeypatch.chdir(tmp_path)

    dependency_entry.cache_root()
    reported = capsys.readouterr().err

    assert "is not writable" in reported
    assert dependency_entry.FALLBACK_CACHE_DIR in reported
    assert "same" in reported and "directory" in reported


def test_the_fallback_directory_is_ignored_by_the_repository() -> None:
    """It is written into whatever directory a job was submitted from.

    For anybody submitting out of a checkout -- which is how the examples read
    -- an un-ignored cache dirties `git status` with an installed virtualenv.
    """

    ignored = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")

    assert f"{dependency_entry.FALLBACK_CACHE_DIR}/" in ignored


def test_an_explicit_runtime_env_dir_wins_over_both(tmp_path, monkeypatch) -> None:
    """The declared path is the one a fleet points at a fast shared disk."""

    monkeypatch.setenv("EYVAN_RUNTIME_ENV_DIR", str(tmp_path / "declared"))

    assert dependency_entry.cache_root() == tmp_path / "declared"


def test_a_real_job_environment_can_import_eyvan_and_the_parent_environment(tmp_path, monkeypatch) -> None:
    """Expose the worker's virtual-environment packages to a real per-job dependency environment.

    ``--system-site-packages`` alone exposes only the base interpreter's packages,
    so query the prepared environment's interpreter to verify the inherited imports.
    """

    import subprocess
    import sys

    real = dependency_entry._install_commands

    def create_only(environment, requirements):
        creation, _install = real(environment, requirements)
        return [creation, [sys.executable, "-c", "pass"]]

    monkeypatch.setattr(dependency_entry, "_install_commands", create_only)

    environment = dependency_entry.prepare(["some-package==1.0"], root=tmp_path)
    python = environment / "bin" / "python"
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import sys, eyvan, pytest, pathlib; "
            "own = next(i for i, p in enumerate(sys.path) if p.endswith('site-packages') and p.startswith(sys.prefix)); "
            "parent = sys.path.index(str(pathlib.Path(pytest.__file__).resolve().parent.parent)); "
            "print(sys.version_info[:2]); print(sys.prefix); print(int(own < parent))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr
    version, prefix, own_first = probe.stdout.splitlines()
    assert version == str(tuple(sys.version_info[:2])), "the job must run the interpreter that pickled nothing else"
    assert Path(prefix).resolve() == environment.resolve()
    parents = (environment / "lib").glob(f"python*/site-packages/{dependency_entry.PARENT_PATH_FILE}")
    assert list(parents), "the parent environment is linked explicitly, not by accident"
    assert own_first == "1", "a package the job pins must win over the parent environment's copy"


def test_child_environment_inherits_pep660_editable_pth_packages(tmp_path, monkeypatch) -> None:
    """PEP 660 editable installs in the parent site-packages stay importable in child envs.

    Bare directory lines in a ``.pth`` file are appended to ``sys.path`` without
    evaluating ``.pth`` files inside them, breaking editable installs such as
    ``uv sync``. Emitting ``site.addsitedir(...)`` evaluates parent ``.pth``
    files while keeping packages pinned in the child environment ahead of them.
    """

    import subprocess

    editable_src = tmp_path / "editable-src"
    (editable_src / "editable_pkg").mkdir(parents=True)
    (editable_src / "editable_pkg" / "__init__.py").write_text("VALUE = 'from-editable'\n", encoding="utf-8")

    parent_site = tmp_path / "parent-site"
    parent_site.mkdir()
    (parent_site / "editable_pkg.pth").write_text(f"{editable_src}\n", encoding="utf-8")
    (parent_site / "pinned_pkg.py").write_text("ORIGIN = 'parent'\n", encoding="utf-8")

    child_venv = tmp_path / "child-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(child_venv)], check=True, timeout=120)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    child_site = child_venv / "lib" / version / "site-packages"
    (child_site / "pinned_pkg.py").write_text("ORIGIN = 'child'\n", encoding="utf-8")

    monkeypatch.setattr(dependency_entry.site, "getsitepackages", lambda: [str(parent_site)])
    monkeypatch.setattr(dependency_entry.site, "ENABLE_USER_SITE", False)
    dependency_entry._inherit_running_environment(child_venv)

    probe = subprocess.run(
        [
            str(child_venv / "bin" / "python"),
            "-I",
            "-c",
            "import editable_pkg, pinned_pkg; print(editable_pkg.VALUE, pinned_pkg.ORIGIN)",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "from-editable child"
