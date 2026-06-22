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


"""Docker container configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DockerConfig:
    """Configuration for Docker container execution.

    Encapsulates all settings needed to run a Docker container in a
    distributed environment.

    Attributes:
        image (str): Docker image to use (e.g., "python:3.9").
        command (str | list[str]): Command to run in the container.
        volumes (dict[str, str] | None): Volume mappings from host to container
            paths (e.g., {"/host/data": "/container/data"}).
        environment (dict[str, str] | None): Environment variables to pass to
            the container.
        network (str): Network mode for the container. Defaults to "host".
        privileged (bool): Whether to run in privileged mode. Defaults to False.
        gpus (str | None): GPU configuration (e.g., "all", "0,1", or device IDs).
        shm_size (str | None): Shared memory size (e.g., "2g", "512m").
        remove (bool): Whether to remove the container after execution.
            Defaults to True.
        workdir (str | None): Working directory inside the container.
        user (str | None): User to run the container as (e.g., "1000:1000").

    Example:
        >>> config = DockerConfig(
        ...     image="tensorflow/tensorflow:latest-gpu",
        ...     command=["python", "train.py", "--epochs", "10"],
        ...     volumes={"/data": "/data", "/models": "/models"},
        ...     environment={"TF_CPP_MIN_LOG_LEVEL": "2"},
        ...     gpus="all",
        ...     shm_size="4g"
        ... )
    """

    image: str
    command: str | list[str]
    volumes: dict[str, str] = None
    environment: dict[str, str] = None
    network: str = "host"
    privileged: bool = False
    gpus: str | None = None
    shm_size: str | None = None
    remove: bool = True
    workdir: str | None = None
    user: str | None = None


def make_docker_run_command(config: DockerConfig) -> list[str]:
    """Construct a docker run command from configuration.

    Converts a DockerConfig object into a list of command-line arguments
    suitable for subprocess execution.

    Args:
        config (DockerConfig): Docker configuration object containing all
            container settings.

    Returns:
        list[str]: List of command arguments for subprocess execution.

    Example:
        >>> config = DockerConfig(image="python:3.9", command="python app.py")
        >>> cmd = make_docker_run_command(config)
        >>>
    """
    cmd = ["docker", "run"]

    if config.remove:
        cmd.append("--rm")

    if config.network:
        cmd.extend(["--network", config.network])

    if config.privileged:
        cmd.append("--privileged")

    if config.gpus:
        cmd.extend(["--gpus", config.gpus])

    if config.shm_size:
        cmd.extend(["--shm-size", config.shm_size])

    if config.workdir:
        cmd.extend(["--workdir", config.workdir])

    if config.user:
        cmd.extend(["--user", config.user])

    if config.volumes:
        for host_path, container_path in config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

    if config.environment:
        for key, value in config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])

    cmd.append(config.image)
    if isinstance(config.command, list):
        cmd.extend(config.command)
    else:
        cmd.append(config.command)

    return cmd
