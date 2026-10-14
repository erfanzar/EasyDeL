# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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


"""Docker execution utilities for Ray-based distributed computing.

This module provides utilities for running Docker containers on distributed
compute resources (TPUs, GPUs) using Ray. It supports single-pod and
multi-slice execution patterns, image building, and asynchronous execution.

Key Features:
    - Docker container configuration and command generation
    - Single-pod Docker execution with accelerator support
    - Multi-slice Docker execution for distributed workloads
    - Docker image building and registry push
    - Asynchronous Docker execution via Ray tasks

Example:
    Basic Docker execution on a TPU pod:

    >>> from eformer.executor.ray import DockerConfig, run_docker_on_pod
    >>> from eformer.executor.ray import TpuAcceleratorConfig
    >>>
    >>> config = DockerConfig(
    ...     image="my-ml-image:latest",
    ...     command="python train.py",
    ...     volumes={"/data": "/data"},
    ...     environment={"MODEL_NAME": "bert-base"}
    ... )
    >>>
    >>> tpu_config = TpuAcceleratorConfig(type="v4-8")
    >>> output = run_docker_on_pod(config, tpu_config)

    Multi-slice execution:

    >>> outputs = run_docker_multislice(
    ...     config,
    ...     tpu_config,
    ...     num_slices=4
    ... )
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import ray
from ray.remote_function import RemoteFunction

from .executor import RayExecutor
from .resource_manager import AcceleratorConfigType

logger = logging.getLogger("ray")


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


def run_docker_on_pod(
    docker_config: DockerConfig,
    accelerator_config: AcceleratorConfigType,
    capture_output: bool = True,
    **executor_kwargs,
) -> Any:
    """Run a Docker container on a compute pod (TPU/GPU).

    Executes a Docker container on a specific accelerator-enabled pod using
    Ray for resource allocation and fault tolerance.

    Args:
        docker_config (DockerConfig): Docker container configuration.
        accelerator_config (AcceleratorConfigType): Accelerator configuration
            specifying TPU or GPU resources.
        capture_output (bool): Whether to capture and return container output.
            Defaults to True.
        **executor_kwargs: Additional arguments passed to RayExecutor.execute_resumable(),
            such as max_retries, retry_exceptions, etc.

    Returns:
        Any: The stdout output from the container if capture_output is True,
            None otherwise.

    Raises:
        RuntimeError: If the Docker container exits with a non-zero status.

    Example:
        >>> from eformer.executor.ray import GpuAcceleratorConfig
        >>>
        >>> gpu_config = GpuAcceleratorConfig(count=2, type="v100")
        >>> output = run_docker_on_pod(
        ...     docker_config,
        ...     gpu_config,
        ...     max_retries=3
        ... )
    """

    def run_docker() -> tuple[int, str, str]:
        """Internal function to run Docker container.

        Returns:
            tuple[int, str, str]: Tuple of (return_code, stdout, stderr).
        """
        cmd = make_docker_run_command(docker_config)

        logger.info(f"Running Docker command: {' '.join(cmd)}")

        if capture_output:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=False)
            return result.returncode, "", ""

    if not isinstance(run_docker, RemoteFunction):
        run_docker = ray.remote(run_docker)

    result = RayExecutor.execute_resumable(
        remote_fn=run_docker,
        accelerator_config=accelerator_config,
        **executor_kwargs,
    )

    returncode, stdout, stderr = result

    if returncode != 0:
        logger.error(f"Docker container failed with exit code {returncode}")
        logger.error(f"stderr: {stderr}")
        raise RuntimeError(f"Docker container failed: {stderr}")

    return stdout if capture_output else None


def run_docker_multislice(
    docker_config: DockerConfig,
    accelerator_config: AcceleratorConfigType,
    capture_output: bool = True,
    **executor_kwargs,
) -> list[Any]:
    """Run Docker containers across multiple slices.

    Executes Docker containers in parallel across all hosts in the compute slices,
    typically used for distributed training or inference on TPU pods.

    Args:
        docker_config (DockerConfig): Base Docker container configuration.
            Each slice will receive a copy with slice-specific environment
            variables added.
        accelerator_config (AcceleratorConfigType): Accelerator configuration
            with multi-slice support (e.g., TPU v4-32 with 4 slices).
        capture_output (bool): Whether to capture and return container output.
            Defaults to True.
        **executor_kwargs: Additional arguments passed to
            RayExecutor.autoscale_execute_resumable().

    Returns:
        list[Any]: List of outputs from each host's Docker container.
            Length equals the number of hosts across all slices. Each element
            is stdout if capture_output is True, None otherwise.

    Raises:
        RuntimeError: If any Docker container exits with a non-zero status.

    Example:
        >>> from eformer.executor.ray import TpuAcceleratorConfig
        >>>
        >>> tpu_config = TpuAcceleratorConfig(type="v4-32", num_slices=4)
        >>> outputs = run_docker_multislice(
        ...     docker_config,
        ...     tpu_config,
        ...     max_retries=3
        ... )
        >>> print(f"Got {len(outputs)} outputs from all hosts")
    """
    executor_kwargs.setdefault("flatten", True)

    def run_docker_with_slice_env() -> tuple[int, str, str]:
        """Run Docker with slice-specific environment variables."""
        slice_id = os.environ.get("EXECUTOR_CALL_SLICE", "0")
        host_id = os.environ.get("EXECUTOR_CALL_INDEX", "0")
        slice_config = DockerConfig(
            image=docker_config.image,
            command=docker_config.command,
            volumes=docker_config.volumes,
            environment={
                **(docker_config.environment or {}),
                "EXECUTOR_CALL_SLICE": str(slice_id),
                "EXECUTOR_CALL_INDEX": str(host_id),
            },
            network=docker_config.network,
            privileged=docker_config.privileged,
            gpus=docker_config.gpus,
            shm_size=docker_config.shm_size,
            remove=docker_config.remove,
            workdir=docker_config.workdir,
            user=docker_config.user,
        )

        cmd = make_docker_run_command(slice_config)

        logger.info(f"Running Docker on slice {slice_id} host {host_id}: {' '.join(cmd)}")

        if capture_output:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=False)
            return result.returncode, "", ""

    if not isinstance(run_docker_with_slice_env, RemoteFunction):
        run_docker_with_slice_env = ray.remote(run_docker_with_slice_env)

    results = RayExecutor.autoscale_execute_resumable(
        remote_fn=run_docker_with_slice_env,
        accelerator_config=accelerator_config,
        **executor_kwargs,
    )

    if results and isinstance(results[0], list):
        results = [item for sublist in results for item in sublist]

    outputs = []
    for i, result in enumerate(results):
        try:
            returncode, stdout, stderr = result
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Unexpected docker result at host index {i}: {result!r}") from exc
        if returncode != 0:
            logger.error(f"Docker container on host {i} failed with exit code {returncode}")
            logger.error(f"stderr: {stderr}")
            raise RuntimeError(f"Docker container on host {i} failed: {stderr}")
        outputs.append(stdout if capture_output else None)

    return outputs


def build_and_push_docker_image(
    dockerfile_path: str,
    image_name: str,
    registry: str | None = None,
    build_args: dict[str, str] | None = None,
) -> str:
    """Build a Docker image and optionally push to a registry.

    Builds a Docker image from a Dockerfile and optionally pushes it to
    a container registry for use in distributed execution.

    Args:
        dockerfile_path (str): Path to the Dockerfile.
        image_name (str): Name for the Docker image (e.g., "my-app:v1.0").
        registry (str | None): Optional registry URL to push to
            (e.g., "gcr.io/my-project" or "docker.io/myuser").
        build_args (dict[str, str] | None): Optional build arguments to pass
            to docker build.

    Returns:
        str: Full image name with registry prefix if applicable
            (e.g., "gcr.io/my-project/my-app:v1.0").

    Raises:
        RuntimeError: If Docker build or push fails.

    Example:
        >>> image = build_and_push_docker_image(
        ...     "./Dockerfile",
        ...     "training-image:latest",
        ...     registry="gcr.io/my-project",
        ...     build_args={"PYTHON_VERSION": "3.9"}
        ... )
        >>> print(f"Built and pushed: {image}")
    """
    full_image_name = f"{registry}/{image_name}" if registry else image_name
    build_cmd = ["docker", "build", "-t", full_image_name]

    if build_args:
        for key, value in build_args.items():
            build_cmd.extend(["--build-arg", f"{key}={value}"])

    build_cmd.extend(["-f", dockerfile_path, os.path.dirname(dockerfile_path)])

    logger.info(f"Building Docker image: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed: {result.stderr}")

    if registry:
        push_cmd = ["docker", "push", full_image_name]
        logger.info(f"Pushing Docker image: {' '.join(push_cmd)}")
        result = subprocess.run(push_cmd, check=True, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Docker push failed: {result.stderr}")

    return full_image_name


@ray.remote
def run_docker_async(docker_config: DockerConfig) -> tuple[int, str, str]:
    """Asynchronous Docker container execution as a Ray task.

    Runs a Docker container asynchronously as a Ray remote task, allowing
    for parallel execution of multiple containers.

    Args:
        docker_config (DockerConfig): Docker configuration for the container.

    Returns:
        tuple[int, str, str]: Tuple containing:
            - return_code (int): Exit code from the Docker container.
            - stdout (str): Standard output from the container.
            - stderr (str): Standard error output from the container.

    Note:
        This function is decorated with @ray.remote, making it a Ray task
        that can be executed asynchronously.

    Example:
        >>> import ray
        >>>
        >>> ray.init()
        >>> config = DockerConfig(image="python:3.9", command="echo 'Hello'")
        >>> future = run_docker_async.remote(config)
        >>> return_code, stdout, stderr = ray.get(future)
        >>> print(f"Output: {stdout}")
    """
    cmd = make_docker_run_command(docker_config)

    logger.info(f"Running async Docker command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    return result.returncode, result.stdout, result.stderr
