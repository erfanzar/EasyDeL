# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A per-job setup step: ``TpuConfig(..., setup=...)`` runs bash on every host before the job.

The step rides the dependency wrapper: it runs inside the job's environment
after its packages, with ``EYVAN_PYTHON`` naming the job interpreter, is cached
by script content and args, and a non-zero exit fails the job. Unset, nothing on
the wire or in the stored record changes.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path

import pytest

from eyvan.records import CallableExecution, CommandExecution, WorkloadDefinition, WorkloadSetup
from eyvan.worker.runtime import dependency_entry

# --------------------------------------------------------------------------- the wire form


def test_a_setup_step_is_exactly_one_of_inline_bash_or_a_url() -> None:
    assert WorkloadSetup(script="echo hi").script == "echo hi"
    assert WorkloadSetup(url="https://example.com/s.sh", args=("--ref", "abc")).args == ("--ref", "abc")
    with pytest.raises(ValueError, match="exactly one"):
        WorkloadSetup()
    with pytest.raises(ValueError, match="exactly one"):
        WorkloadSetup(script="echo", url="https://example.com/s.sh")


@pytest.mark.parametrize("url", ["http://example.com/s.sh", "file:///etc/passwd", "https://x.com/a b.sh"])
def test_a_setup_url_must_be_plain_https(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        WorkloadSetup(url=url)


def test_a_setup_hash_pins_only_a_fetched_script_and_must_be_hex() -> None:
    with pytest.raises(ValueError, match="already exact"):
        WorkloadSetup(script="echo", sha256="a" * 64)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        WorkloadSetup(url="https://example.com/s.sh", sha256="ABC")


def _spec(**fields) -> WorkloadDefinition:
    return WorkloadDefinition.model_validate(
        {
            "name": "job",
            "execution": CommandExecution(command="python train.py").model_dump(),
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
            **fields,
        }
    )


def test_a_spec_without_setup_serializes_exactly_as_before() -> None:
    document = _spec().model_dump(mode="json")

    assert "setup" not in document
    assert WorkloadDefinition.model_validate(document) == _spec()


def test_a_spec_with_setup_round_trips_and_drops_unset_parts() -> None:
    spec = _spec(setup={"url": "https://example.com/s.sh", "args": ["--ref", "abc"]})
    document = spec.model_dump(mode="json")

    assert document["setup"] == {"url": "https://example.com/s.sh", "args": ["--ref", "abc"]}
    assert WorkloadDefinition.model_validate(document).setup == spec.setup


# --------------------------------------------------------------------------- the SDK


def test_tpu_config_accepts_a_script_url_inline_bash_or_a_setup_script(tmp_path) -> None:
    from eyvan import SetupScript, TpuConfig

    assert TpuConfig("v6e-8").setup_request is None
    assert TpuConfig("v6e-8", setup="https://example.com/s.sh").setup_request == WorkloadSetup(
        url="https://example.com/s.sh"
    )
    assert TpuConfig("v6e-8", setup="pip list\necho done").setup_request == WorkloadSetup(script="pip list\necho done")
    configured = TpuConfig("v6e-8", setup=SetupScript("https://example.com/s.sh", args=["--ref", "580a86a"]))
    assert configured.setup_request == WorkloadSetup(url="https://example.com/s.sh", args=("--ref", "580a86a"))


def test_a_local_setup_file_ships_inline_and_honours_its_pin(tmp_path) -> None:
    from eyvan import SetupScript, TpuConfig

    local = tmp_path / "setup.sh"
    local.write_text("echo local\n", encoding="utf-8")
    digest = hashlib.sha256(b"echo local\n").hexdigest()

    shipped = TpuConfig("v6e-8", setup=SetupScript(str(local), args=["x"], sha256=digest)).setup_request
    assert shipped == WorkloadSetup(script="echo local\n", args=("x",))
    with pytest.raises(ValueError, match="does not match"):
        SetupScript(str(local), sha256="0" * 64).to_workload_setup()
    with pytest.raises(ValueError, match="https"):
        SetupScript("http://example.com/s.sh").to_workload_setup()


def test_gpu_and_cpu_configs_take_a_setup_step_too() -> None:
    from eyvan import CpuConfig, GpuConfig

    assert GpuConfig("nvidia-tesla-a100", setup="echo g").setup_request == WorkloadSetup(script="echo g")
    assert CpuConfig(setup="echo c").setup_request == WorkloadSetup(script="echo c")


def test_a_callable_request_without_setup_is_byte_identical() -> None:
    from eyvan.sdk.client.requests import pool_callable_request, pool_request_json

    common = dict(
        org_id="o",
        project_id="p",
        accelerator_type="v6e-8",
        pods=1,
        capacity_class="spot",
        execution=CallableExecution(
            artifact_uri="file:///c.pkl", sha256="0" * 64, result_uri="file:///r.pkl", python="3.12"
        ),
        name=None,
        environment=None,
        metadata=None,
        max_retries=3,
    )
    plain = pool_request_json(pool_callable_request(**common))
    with_setup = pool_request_json(pool_callable_request(**common, setup=WorkloadSetup(script="echo hi")))

    assert "setup" not in plain
    assert with_setup == {**plain, "setup": {"script": "echo hi"}}


# --------------------------------------------------------------------------- the dispatch wrapper


def _entrypoint(dependencies: tuple[str, ...], setup: WorkloadSetup | None) -> str:
    from eyvan.server.controller.workload_commands import _ControllerWorkloadCommands

    return _ControllerWorkloadCommands._dependency_entrypoint(dependencies, "python train.py", setup)


def test_a_job_without_setup_or_packages_runs_exactly_what_was_written() -> None:
    assert _entrypoint((), None) == "python train.py"


def test_a_setup_only_job_is_wrapped_and_carries_its_step_as_one_argument() -> None:
    setup = WorkloadSetup(url="https://example.com/s.sh", args=("--ref", "a b"))
    parts = shlex.split(_entrypoint((), setup))

    assert parts[:4] == ["python", "-I", "-m", "eyvan.worker.runtime.dependency_entry"]
    assert json.loads(parts[parts.index("--setup") + 1]) == setup.model_dump(mode="json")
    assert parts[-2:] == ["python", "train.py"]


# --------------------------------------------------------------------------- the worker step


def _fake_venv(monkeypatch) -> list[Path]:
    built: list[Path] = []

    def install(environment: Path, requirements):
        built.append(environment)
        return [[sys.executable, "-c", f"import pathlib; pathlib.Path({str(environment)!r}).mkdir(parents=True)"]]

    monkeypatch.setattr(dependency_entry, "_install_commands", install)
    return built


def test_the_step_runs_in_the_job_environment_with_its_args(tmp_path, monkeypatch) -> None:
    _fake_venv(monkeypatch)
    out = tmp_path / "seen.txt"
    step = dependency_entry.SetupStep(
        script=f'printf "%s|%s|%s|%s" "$EYVAN_PYTHON" "$VIRTUAL_ENV" "$1" "$2" > {shlex.quote(str(out))}\n',
        args=("--ref", "580a86a"),
    )

    environment = dependency_entry.prepare([], root=tmp_path / "envs", setup=step)

    python, venv, first, second = out.read_text(encoding="utf-8").split("|")
    assert python == str(environment / "bin" / "python")
    assert venv == str(environment)
    assert (first, second) == ("--ref", "580a86a")
    assert (environment / dependency_entry.COMPLETE_MARKER).exists()


def test_the_multihost_join_is_left_for_the_job_not_the_setup_step(tmp_path, monkeypatch) -> None:
    """A multi-host job's environment carries the JAX join marker.

    Any ``$EYVAN_PYTHON`` the step runs (``pip install`` is enough) would load
    the multi-host ``sitecustomize``, see the marker and try to join the gang
    from inside the setup step. The step must not see it; the job still must.
    """

    _fake_venv(monkeypatch)
    monkeypatch.setenv("EYVAN_JAX_MULTIHOST", "1")
    out = tmp_path / "seen.txt"
    step = dependency_entry.SetupStep(f'echo "${{EYVAN_JAX_MULTIHOST:-unset}}" > {shlex.quote(str(out))}\n', ())

    environment = dependency_entry.prepare([], root=tmp_path / "envs", setup=step)

    assert out.read_text(encoding="utf-8").strip() == "unset"
    assert dependency_entry.activate(environment)["EYVAN_JAX_MULTIHOST"] == "1"


def test_the_same_step_is_reused_and_a_changed_step_runs_again(tmp_path, monkeypatch) -> None:
    built = _fake_venv(monkeypatch)
    root = tmp_path / "envs"
    counter = tmp_path / "count"
    script = f"echo run >> {shlex.quote(str(counter))}\n"

    first = dependency_entry.prepare([], root=root, setup=dependency_entry.SetupStep(script, ("a",)))
    second = dependency_entry.prepare([], root=root, setup=dependency_entry.SetupStep(script, ("a",)))
    third = dependency_entry.prepare([], root=root, setup=dependency_entry.SetupStep(script, ("b",)))

    assert first == second != third
    assert len(built) == 2
    assert counter.read_text().count("run") == 2


def test_no_setup_keeps_the_environment_key_it_always_had() -> None:
    assert dependency_entry.environment_key(["flax"]) == dependency_entry.environment_key(["flax"], None)
    assert dependency_entry.environment_key(["flax"]) != dependency_entry.environment_key(["flax"], "x")


def test_a_failing_step_fails_the_job_and_leaves_nothing_trusted(tmp_path, monkeypatch) -> None:
    _fake_venv(monkeypatch)
    root = tmp_path / "envs"
    step = dependency_entry.SetupStep("exit 3\n", ())

    with pytest.raises(RuntimeError, match="status 3"):
        dependency_entry.prepare([], root=root, setup=step)

    assert not (root / dependency_entry.environment_key([], step.identity)).exists()


def test_a_fetched_script_is_checked_against_its_pin(monkeypatch) -> None:
    monkeypatch.setattr(dependency_entry, "_fetch_setup_script", lambda url: "echo fetched\n")
    good = hashlib.sha256(b"echo fetched\n").hexdigest()

    step = dependency_entry.resolve_setup({"url": "https://example.com/s.sh", "args": ["x"], "sha256": good})
    assert step == dependency_entry.SetupStep("echo fetched\n", ("x",))
    with pytest.raises(RuntimeError, match="pinned sha256"):
        dependency_entry.resolve_setup({"url": "https://example.com/s.sh", "sha256": "0" * 64})


def test_the_wrapper_runs_a_setup_only_job_then_execs_it(tmp_path, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        dependency_entry, "prepare", lambda requirements, **options: calls.append((requirements, options)) or tmp_path
    )

    class Executed(BaseException):
        pass

    def fake_execvp(program, arguments):
        raise Executed(list(arguments))

    monkeypatch.setattr(dependency_entry.os, "execvp", fake_execvp)
    monkeypatch.setattr(dependency_entry.os, "environ", dict(dependency_entry.os.environ))

    with pytest.raises(Executed) as raised:
        dependency_entry.main(["--setup", json.dumps({"script": "echo hi", "args": ["a"]}), "--", "python", "job.py"])

    assert raised.value.args[0] == ["python", "job.py"]
    assert calls == [([], {"setup": dependency_entry.SetupStep("echo hi", ("a",))})]


def test_a_malformed_setup_argument_fails_before_the_job(capsys) -> None:
    assert dependency_entry.main(["--setup", "not json", "--", "python", "job.py"]) == 1
    assert "not valid JSON" in capsys.readouterr().err
