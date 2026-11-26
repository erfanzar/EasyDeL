# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coverage for interpreter-steering scrubbing at provider CLI boundaries.

A client-supplied ``PYTHONPATH`` must not steer gcloud's embedded Python into
checkout paths or trigger import-time failures in provider subprocesses.

The behavioral tests below pin the runners behind the busiest boundaries; a
source scan pins the invariant itself across every production gcloud spawn, so
a new boundary cannot land still inheriting the caller's interpreter.
"""

from __future__ import annotations

import ast
import subprocess
import typing as tp
from pathlib import Path

import pytest

from eyvan._internal.subprocess_env import INTERPRETER_STEERING_ENV_NAMES, provider_cli_env


def test_provider_cli_env_strips_interpreter_steering_and_keeps_the_rest() -> None:
    base = {
        "PATH": "/usr/bin",
        "HOME": "/home/operator",
        "PYTHONPATH": "libs/easydel:libs/eyvan",
        "PYTHONHOME": "/opt/custom-python",
        "PYTHONSTARTUP": "/home/operator/.pythonrc",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUSERBASE": "/home/operator/.local",
        "PYTHONEXECUTABLE": "/usr/bin/python3",
        "__PYVENV_LAUNCHER__": "/usr/bin/python3",
        "CLOUDSDK_PYTHON": "/usr/bin/python3",  # operator-chosen gcloud interpreter: respected
        "EYVAN_URL": "http://127.0.0.1:8080",
    }

    environment = provider_cli_env(base)

    for name in INTERPRETER_STEERING_ENV_NAMES:
        assert name not in environment
    assert environment == {
        "PATH": "/usr/bin",
        "HOME": "/home/operator",
        "CLOUDSDK_PYTHON": "/usr/bin/python3",
        "EYVAN_URL": "http://127.0.0.1:8080",
    }
    # The caller's mapping is never mutated, and two calls never share a dict.
    assert base["PYTHONPATH"] == "libs/easydel:libs/eyvan"
    assert provider_cli_env(base) is not environment


def test_provider_cli_env_defaults_to_the_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "libs/easydel")
    monkeypatch.setenv("EYVAN_PROJECT", "default")

    environment = provider_cli_env()

    assert "PYTHONPATH" not in environment
    assert environment["EYVAN_PROJECT"] == "default"


def _captured_env_run(
    captured: dict[str, tp.Any],
) -> tp.Callable[..., subprocess.CompletedProcess[str]]:
    """Return a subprocess.run stand-in that records the environment it was given."""

    def run(*args: tp.Any, **kwargs: tp.Any) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args[0], 0, "", "")

    return run


def test_gcloud_runners_never_inherit_pythonpath(monkeypatch) -> None:
    """Every module-level gcloud boundary must scrub the caller's interpreter steering."""

    monkeypatch.setenv("PYTHONPATH", "libs/easydel:libs/eyvan")

    from eyvan.capacity import fleet_state
    from eyvan.capacity.providers.gcp import compute_instances, native_worker_ingress, queued_resources

    for runner in (
        queued_resources._runner,
        compute_instances._runner,
        native_worker_ingress._gcloud_runner,
        fleet_state._gcloud,
    ):
        captured: dict[str, tp.Any] = {}
        monkeypatch.setattr(subprocess, "run", _captured_env_run(captured))
        runner(("version",))
        env = captured.get("env")
        assert env is not None, "the runner must pass an explicit scrubbed environment"
        assert "PYTHONPATH" not in env


def test_gcs_artifact_store_default_runner_scrubs_interpreter_steering(monkeypatch, tmp_path) -> None:
    """The SDK's own callable upload runs gcloud from the submitting process."""

    from eyvan.artifacts import GCSArtifactStore

    store = GCSArtifactStore("gs://eyvan-artifacts")
    captured: dict[str, tp.Any] = {}
    monkeypatch.setattr(subprocess, "run", _captured_env_run(captured))

    store.read("gs://eyvan-artifacts/callables/x.pkl")

    env = captured.get("env")
    assert env is not None
    assert "PYTHONPATH" not in env


# Track any production gcloud spawn sites that still inherit interpreter steering.
# Each entry requires a fix routing the spawn through ``provider_cli_env``; the
# scan rejects stale entries and any newly unscrubbed boundary.
PENDING_UNSCRUBBED_GCLOUD_SPAWNS = frozenset()

_SPAWN_CALL_NAMES = frozenset({"run", "Popen", "check_call", "check_output", "call"})


def _imports_of(tree: ast.Module, module_path: str, member: str | None = None) -> set[str]:
    """Return the local names imports bind from ``module_path``.

    With ``member``, only the local name bound to that one imported symbol;
    without it, every name imported from the module.
    """

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_path:
            for alias in node.names:
                if member is None or alias.name == member:
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import) and member is None:
            for alias in node.names:
                if alias.name == module_path:
                    bound.add(alias.asname or alias.name)
    return bound


def _constructs_gcloud_command(tree: ast.Module) -> bool:
    """Whether the module builds a gcloud argv: a sequence literal led by ``"gcloud"``."""

    return any(
        isinstance(node, (ast.List, ast.Tuple))
        and bool(node.elts)
        and isinstance(node.elts[0], ast.Constant)
        and node.elts[0].value == "gcloud"
        for node in ast.walk(tree)
    )


def _is_subprocess_spawn(call: ast.Call, module_aliases: set[str]) -> bool:
    """Whether ``call`` invokes ``subprocess.run``/``Popen``/``check_*``/``call``."""

    function = call.func
    if isinstance(function, ast.Name):
        return function.id in module_aliases
    return (
        isinstance(function, ast.Attribute)
        and function.attr in _SPAWN_CALL_NAMES
        and isinstance(function.value, ast.Name)
        and function.value.id in module_aliases
    )


def _subprocess_spawn_calls(tree: ast.Module) -> list[tuple[ast.Call, str]]:
    """Return every subprocess spawn call paired with its enclosing qualified name."""

    module_aliases = _imports_of(tree, "subprocess")
    calls: list[tuple[ast.Call, str]] = []
    enclosure: list[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosure.append(child.name)
                visit(child)
                enclosure.pop()
            else:
                if isinstance(child, ast.Call) and _is_subprocess_spawn(child, module_aliases):
                    calls.append((child, ".".join(enclosure)))
                visit(child)

    visit(tree)
    return calls


def _routes_through_provider_cli_env(call: ast.Call, env_names: set[str]) -> bool:
    """Whether the spawn pins an explicit environment built by ``provider_cli_env``."""

    return any(
        keyword.arg == "env"
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id in env_names
        for keyword in call.keywords
    )


def test_source_scan_every_gcloud_spawn_scrubs_interpreter_steering() -> None:
    """Every production gcloud spawn must route through ``provider_cli_env``.

    The behavioral tests pin individual runners; this scan pins the invariant
    itself, so a new gcloud boundary cannot land still inheriting the caller's
    interpreter steering. Sources are parsed, never executed: no network, no
    subprocesses.
    """

    import eyvan

    package_root = Path(eyvan.__file__).resolve().parent
    scrubbed: list[str] = []
    unscrubbed: dict[tuple[str, str], int] = {}
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not _constructs_gcloud_command(tree):
            continue
        relative = path.relative_to(package_root).as_posix()
        env_names = _imports_of(tree, "eyvan._internal.subprocess_env", "provider_cli_env")
        for call, qualname in _subprocess_spawn_calls(tree):
            if _routes_through_provider_cli_env(call, env_names):
                scrubbed.append(f"{relative}:{call.lineno}")
            else:
                unscrubbed[(relative, qualname)] = call.lineno

    unexpected = {
        f"{relative}:{qualname} at line {line}"
        for (relative, qualname), line in unscrubbed.items()
        if (relative, qualname) not in PENDING_UNSCRUBBED_GCLOUD_SPAWNS
    }
    assert not unexpected, (
        "gcloud spawn sites that inherit the caller's interpreter steering "
        f"(the outage-58 regression class): {sorted(unexpected)}; pass "
        "env=provider_cli_env() from eyvan._internal.subprocess_env, or add a "
        "commented entry to PENDING_UNSCRUBBED_GCLOUD_SPAWNS only with a "
        "tracked production fix"
    )
    stale = sorted(PENDING_UNSCRUBBED_GCLOUD_SPAWNS - set(unscrubbed))
    assert not stale, (
        "these sites now scrub interpreter steering; remove their pending "
        f"entries from PENDING_UNSCRUBBED_GCLOUD_SPAWNS: {stale}"
    )
    # Blind-spot tripwire: if the scan stops recognizing the population it
    # guards (renamed imports, a wrapper module around subprocess), it would
    # pass vacuously. This floor only trips when the scanner went blind.
    assert len(scrubbed) >= 5, (
        f"the scan recognized only {len(scrubbed)} scrubbed gcloud boundaries "
        f"({scrubbed}); extend the scanner before trusting this guard"
    )


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
