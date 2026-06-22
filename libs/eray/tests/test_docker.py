# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for eray.docker — DockerConfig and command generation."""

from eray.docker.config import DockerConfig, make_docker_run_command


class TestDockerConfig:
    def test_defaults(self):
        dc = DockerConfig(image="python:3.11", command="echo hi")
        assert dc.image == "python:3.11"
        assert dc.network == "host"
        assert dc.remove is True
        assert dc.privileged is False
        assert dc.volumes is None
        assert dc.environment is None

    def test_full_config(self):
        dc = DockerConfig(
            image="my:latest",
            command=["python", "train.py"],
            volumes={"/data": "/data"},
            environment={"FOO": "bar"},
            network="bridge",
            privileged=True,
            gpus="all",
            shm_size="4g",
            remove=False,
            workdir="/app",
            user="1000:1000",
        )
        assert dc.privileged is True
        assert dc.gpus == "all"
        assert dc.shm_size == "4g"
        assert dc.workdir == "/app"


class TestMakeDockerRunCommand:
    def test_basic(self):
        dc = DockerConfig(image="python:3.11", command="echo hi")
        cmd = make_docker_run_command(dc)
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "--network" in cmd
        assert "host" in cmd
        assert "python:3.11" in cmd
        assert "echo hi" in cmd

    def test_no_remove(self):
        dc = DockerConfig(image="python:3.11", command="echo hi", remove=False)
        cmd = make_docker_run_command(dc)
        assert "--rm" not in cmd

    def test_volumes(self):
        dc = DockerConfig(
            image="python:3.11",
            command="echo hi",
            volumes={"/host/data": "/container/data"},
        )
        cmd = make_docker_run_command(dc)
        assert "-v" in cmd
        assert "/host/data:/container/data" in cmd

    def test_environment(self):
        dc = DockerConfig(
            image="python:3.11",
            command="echo hi",
            environment={"API_KEY": "secret", "DEBUG": "1"},
        )
        cmd = make_docker_run_command(dc)
        assert "-e" in cmd
        assert "API_KEY=secret" in cmd
        assert "DEBUG=1" in cmd

    def test_gpus(self):
        dc = DockerConfig(image="py:3.11", command="x", gpus="all")
        cmd = make_docker_run_command(dc)
        assert "--gpus" in cmd
        assert "all" in cmd

    def test_command_as_list(self):
        dc = DockerConfig(image="py:3.11", command=["python", "-c", "print(1)"])
        cmd = make_docker_run_command(dc)
        assert "python" in cmd
        assert "-c" in cmd
        assert "print(1)" in cmd
