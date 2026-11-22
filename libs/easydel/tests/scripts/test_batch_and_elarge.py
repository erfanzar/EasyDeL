# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the batch converter and the eLarge DP-serve router (no array computation)."""

from __future__ import annotations

import builtins
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import eformer.paths as paths
import pytest
from easydel.scripts import convert_hf_to_easydel_batch as batch
from easydel.scripts import elarge
from eformer.aparser import DataClassArgumentParser

pytestmark = pytest.mark.cpu_ok


def test_batch_converter_keeps_gs_out_root_as_uri(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = batch.main(["--out-root", "gs://bucket/models", "--source", "org/Model-1B", "--dry-run"])

    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if "--out" in line]
    argv = shlex.split(lines[0])
    assert argv[argv.index("--out") + 1] == "gs://bucket/models/Model-1B"
    assert not (tmp_path / "gs:").exists()


@pytest.mark.parametrize("contents", ["file", "nested_directory", "empty"])
def test_remote_skip_existing_uses_real_gcs_prefix_semantics(tmp_path, monkeypatch, capsys, contents):
    prefix = "models/Model-1B/"
    client = Mock()
    bucket = client.bucket.return_value
    # GCS directories are prefixes, not exact objects. Keep the real ePath/GCSPath
    # methods and mock only the storage client boundary (no cloud requests).
    bucket.blob.return_value.exists.return_value = False

    class _Page(list):
        prefixes = [prefix + "weights/"] if contents == "nested_directory" else []

    page = _Page([SimpleNamespace(name=prefix + "tensorstore_index.json")] if contents == "file" else [])
    client.list_blobs.return_value = SimpleNamespace(pages=[page])
    bucket.list_blobs.return_value = [] if contents == "empty" else [SimpleNamespace(name=prefix + "weights/0")]
    monkeypatch.setattr(paths.ePath, "_gcs_client", client)
    monkeypatch.chdir(tmp_path)

    remote = paths.ePath("gs://bucket/models/Model-1B")
    assert isinstance(remote, paths.GCSPath)
    assert remote.exists() is False
    assert remote.is_dir() is (contents != "empty")

    code = batch.main(
        [
            "--out-root",
            "gs://bucket/models/",
            "--source",
            "org/Model-1B",
            "--skip-existing",
            "--dry-run",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    if contents == "empty":
        assert "skipped=0" in output
        argv = shlex.split(next(line for line in output.splitlines() if "--out" in line))
        assert argv[argv.index("--out") + 1] == "gs://bucket/models/Model-1B"
    else:
        assert "skipped=1" in output
        assert "--out" not in output
    client.list_blobs.assert_called_once_with("bucket", prefix=prefix, delimiter="/")
    assert not (tmp_path / "gs:").exists()


def test_batch_converter_local_out_root_still_creates_directory(tmp_path):
    out_root = tmp_path / "out"

    code = batch.main(["--out-root", str(out_root), "--source", "org/Model-1B", "--dry-run"])

    assert code == 0
    assert (out_root / "Model-1B").is_dir()


def test_dp_router_recursion_guard_is_a_cli_flag_not_an_env_var():
    (args,) = DataClassArgumentParser(elarge.ElargeArgs).parse_args_into_dataclasses(
        args=["--dp-router-child"], look_for_args_file=False
    )
    assert args.dp_router_child is True
    assert "_ELARGE_DP_ROUTER" not in os.environ


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.returncode = None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        del timeout
        return 0

    def kill(self):
        pass

    def poll(self):
        return None


def test_failed_replica_spawn_terminates_already_started_replicas(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    started: list[_FakeProc] = []

    def fake_popen(*args, **kwargs):
        del args, kwargs
        if started:
            raise RuntimeError("second spawn failed")
        proc = _FakeProc()
        started.append(proc)
        return proc

    class _Elm:
        def to_dict(self):
            return {"sharding": {"axis_names": ["dp", "tp"], "axis_dims": [2, 1]}}

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(elarge, "_plan_replica_tpu_envs", lambda count: [{} for _ in range(count)])
    token = elarge._DP_ROUTER_CHILD.set(True)
    try:
        with pytest.raises(RuntimeError, match="second spawn failed"):
            elarge._serve_replicated(
                elm=_Elm(),
                replicas_params={"workdir": str(tmp_path)},
                server_kwargs={},
                host="127.0.0.1",
                port=0,
                log_level="info",
                ssl_keyfile=None,
                ssl_certfile=None,
            )
    finally:
        elarge._DP_ROUTER_CHILD.reset(token)

    assert len(started) == 1
    assert started[0].terminated


def test_dp_router_reexec_passes_internal_marker_without_adding_environment_knob(monkeypatch):
    captured = {}

    class _Reexec(BaseException):
        pass

    def execve(path, argv, environ):
        captured.update(path=path, argv=argv, environ=environ)
        raise _Reexec

    monkeypatch.setattr(os, "execve", execve)
    monkeypatch.setattr(os, "listdir", lambda path: [])
    monkeypatch.setattr(sys, "argv", ["elarge", "--config", "run.yaml"])
    token = elarge._DP_ROUTER_CHILD.set(False)
    try:
        with pytest.raises(_Reexec):
            elarge._serve_replicated(
                elm=None,
                replicas_params={},
                server_kwargs={},
                host="localhost",
                port=0,
                log_level="info",
                ssl_keyfile=None,
                ssl_certfile=None,
            )
    finally:
        elarge._DP_ROUTER_CHILD.reset(token)

    assert captured["argv"][-1] == "--dp-router-child"
    assert captured["argv"][1:5] == ["-m", "easydel.scripts.elarge", "--config", "run.yaml"]
    assert captured["environ"]["JAX_PLATFORMS"] == "cpu"
    assert "_ELARGE_DP_ROUTER" not in captured["environ"]


@pytest.mark.parametrize("failure", ["yaml", "import"])
def test_replica_startup_failures_close_processes_and_log_files(tmp_path, monkeypatch, failure):
    import yaml

    started = []
    logs = []
    real_import = builtins.__import__
    real_dump = yaml.safe_dump

    class _Elm:
        def to_dict(self):
            return {"sharding": {"axis_names": ["dp", "tp"], "axis_dims": [2, 1]}}

    def popen(*args, **kwargs):
        proc = _FakeProc()
        started.append(proc)
        logs.append(kwargs["stdout"])
        assert "--dp-router-child" not in args[0]
        assert "_ELARGE_DP_ROUTER" not in kwargs["env"]
        return proc

    def dump(*args, **kwargs):
        if started and failure == "yaml":
            raise RuntimeError("yaml failed")
        return real_dump(*args, **kwargs)

    def import_module(name, *args, **kwargs):
        if name == "easydel.inference.esurge.distributed.remote_engine" and failure == "import":
            raise RuntimeError("import failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(elarge, "_plan_replica_tpu_envs", lambda count: [{} for _ in range(count)])
    monkeypatch.setattr(yaml, "safe_dump", dump)
    monkeypatch.setattr(builtins, "__import__", import_module)
    # The handles in Popen kwargs are real files; check their closed state below.
    token = elarge._DP_ROUTER_CHILD.set(True)
    try:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            elarge._serve_replicated(
                elm=_Elm(),
                replicas_params={"workdir": str(tmp_path)},
                server_kwargs={},
                host="localhost",
                port=0,
                log_level="info",
                ssl_keyfile=None,
                ssl_certfile=None,
            )
    finally:
        elarge._DP_ROUTER_CHILD.reset(token)

    assert len(started) == (1 if failure == "yaml" else 2)
    assert all(proc.terminated for proc in started)
    assert all(handle.closed for handle in logs)


def test_replica_cleanup_reaps_forcibly_killed_processes_and_closes_logs(tmp_path):
    class _StalledProc(_FakeProc):
        def __init__(self):
            super().__init__()
            self.killed = False

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("fake-replica", timeout)
            self.returncode = -9
            return self.returncode

        def kill(self):
            self.killed = True

    proc = _StalledProc()
    log = (tmp_path / "replica.log").open("w", encoding="utf-8")
    try:
        elarge._stop_replicas([proc], [log])
        assert proc.terminated
        assert proc.killed
        assert proc.returncode == -9
        assert log.closed
    finally:
        log.close()
