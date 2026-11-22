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


"""Docker TPU execution utilities for managing Docker images and containers on Google Cloud Platform.

This module provides functionality for:
- Building and pushing Docker images to GCP Artifact Registry and GitHub Container Registry
- Configuring GCP Docker repositories with cleanup policies
- Managing Docker containers for TPU workloads
- Handling file operations and command execution with proper TTY support
"""

import json
import os
import pty
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

GCP_CLEANUP_POLICY: list[dict[str, Any]] = [
    {
        "name": "delete-stale",
        "action": {"type": "Delete"},
        "condition": {"olderThan": "86400s", "tagState": "ANY"},
    },
    {
        "name": "keep-latest",
        "action": {"type": "Keep"},
        "mostRecentVersions": {"keepCount": 5},
    },
]


def remove_path(path: Path) -> None:
    """Remove a file or directory at the specified path.

    Args:
        path: Path to the file or directory to remove.

    Raises:
        RuntimeError: If the path exists but is neither a file nor a directory.
    """
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        os.remove(path)
    elif path.exists():
        raise RuntimeError(f"Remove failed. Path ({path}) is neither a directory nor a file.")


def copy_path(src: Path, dst: Path) -> None:
    """Copy a file or directory from source to destination.

    Removes the destination if it exists before copying.

    Args:
        src: Source path to copy from.
        dst: Destination path to copy to.

    Raises:
        RuntimeError: If the source path is neither a file nor a directory.
    """

    remove_path(dst)

    if src.is_dir():
        shutil.copytree(src, dst)
    elif src.is_file():
        shutil.copy(src, dst)
    else:
        raise RuntimeError(f"Copy failed. Source path ({src}) is neither a directory nor a file. Check if it exists.")


def run_command(argv: list[str]) -> bytes:
    """Execute a command with proper TTY handling.

    Runs commands with pseudo-terminal allocation when stdout is a TTY,
    otherwise runs normally. Captures and returns output.

    Args:
        argv: Command and arguments to execute.

    Returns:
        Command output as bytes.

    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit code.
    """
    if sys.stdout.isatty():
        output = []

        def read(fd):
            data = os.read(fd, 1024)
            output.append(data)
            return data

        exit_code = pty.spawn(argv, master_read=read)
        if exit_code != 0:
            e = subprocess.CalledProcessError(exit_code, argv)
            e.output = b"".join(output)
            raise e

        return b"".join(output)
    else:
        try:
            return subprocess.check_output(argv, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
            raise e


def configure_gcp_artifact_registry(project_id: str, region: str, repository: str) -> None:
    """Set up GCP Artifact Registry repository with appropriate permissions for TPU access.

    Creates a new Docker repository in Artifact Registry if it doesn't exist,
    configures cleanup policies, and sets up public read access.

    Args:
        project_id: GCP project ID.
        region: GCP region for the repository.
        repository: Name of the Artifact Registry repository.

    Raises:
        subprocess.CalledProcessError: If GCP commands fail (except for expected errors).
    """

    try:
        run_command(["gcloud", "artifacts", "repositories", "describe", f"--location={region}", repository])
        print(f"Found existing artifact registry repository `{repository}`, skipping setup.")
        return
    except subprocess.CalledProcessError as e:
        if b"NOT_FOUND" not in e.output:
            raise

    run_command(["gcloud", "services", "enable", "artifactregistry.googleapis.com"])

    try:
        run_command(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                repository,
                f"--location={region}",
                "--repository-format=docker",
            ],
        )
    except subprocess.CalledProcessError as e:
        if b"ALREADY_EXISTS" not in e.output:
            print("Error creating repository: ", e.output)
            raise

    with open("/tmp/cleanup-policy.json", "w") as f:
        json.dump(GCP_CLEANUP_POLICY, f, indent=2)

    run_command(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "set-cleanup-policies",
            f"--location={region}",
            "--policy=/tmp/cleanup-policy.json",
            repository,
        ]
    )

    run_command(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            "--member=allUsers",
            "--role=roles/artifactregistry.reader",
            f"--location={region}",
            repository,
        ]
    )

    run_command(
        [
            "gcloud",
            "--project",
            project_id,
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            repository,
            "--location",
            region,
            "--member",
            "allUsers",
            "--role",
            "roles/artifactregistry.reader",
        ]
    )

    run_command(["gcloud", "auth", "configure-docker", "--quiet", f"{region}-docker.pkg.dev"])


@contextmanager
def manage_extra_context(extra_ctx: Path | None):
    """Context manager for handling temporary extra context directory.

    Copies extra context to a temporary mount point and cleans up after use.

    Args:
        extra_ctx: Optional path to extra context directory to copy.

    Yields:
        The extra context path if provided, None otherwise.
    """
    if extra_ctx is not None:
        mount_dst = Path(".mnt")
        copy_path(extra_ctx, mount_dst)
        try:
            yield extra_ctx
        finally:
            remove_path(mount_dst)
    else:
        yield None


def build_docker_image(
    docker_file: str | Path,
    image_name: str,
    tag: str,
    build_args: dict[str, str] | None = None,
) -> str:
    """Build a Docker image for Linux/AMD64 platform.

    Args:
        docker_file: Path to the Dockerfile.
        image_name: Name for the Docker image.
        tag: Tag for the Docker image.
        build_args: Optional dictionary of build arguments.

    Returns:
        Full image identifier (image_name:tag).

    Raises:
        subprocess.CalledProcessError: If Docker build fails.
    """
    args = [
        "docker",
        "buildx",
        "build",
        "--platform=linux/amd64",
        "-t",
        f"{image_name}:{tag}",
    ]

    if build_args:
        for key, value in build_args.items():
            args.extend(["--build-arg", f"{key}={value}"])

    args.extend(["-f", docker_file, "."])
    run_command(args)

    return f"{image_name}:{tag}"


def push_image_to_github(local_id: str, github_user: str, github_token: str | None = None) -> str:
    """Push a Docker image to GitHub Container Registry.

    Args:
        local_id: Local Docker image identifier.
        github_user: GitHub username.
        github_token: Optional GitHub personal access token for authentication.

    Returns:
        Full remote image identifier on ghcr.io.

    Raises:
        subprocess.CalledProcessError: If push operation fails.
    """
    if github_token:
        login_process = subprocess.Popen(
            ["docker", "login", "ghcr.io", "-u", github_user, "--password-stdin"],
            stdin=subprocess.PIPE,
        )
        print(login_process.communicate(input=github_token.encode(), timeout=10))

    remote_name = f"ghcr.io/{github_user}/{local_id}"

    run_command(["docker", "tag", local_id, remote_name])
    run_command(["docker", "push", remote_name])
    return remote_name


def push_image_to_gcp(
    local_id: str,
    project_id: str,
    region: str,
    repository: str,
) -> str:
    """Push a Docker image to Google Cloud Artifact Registry.

    Args:
        local_id: Local Docker image identifier.
        project_id: GCP project ID.
        region: GCP region for the repository.
        repository: Name of the Artifact Registry repository.

    Returns:
        Full remote image identifier in Artifact Registry.

    Raises:
        subprocess.CalledProcessError: If push operation fails.
    """
    configure_gcp_artifact_registry(project_id, region, repository)

    artifact_repo = f"{region}-docker.pkg.dev/{project_id}/{repository}"

    full_image_name = f"{artifact_repo}/{local_id}"
    run_command(["docker", "tag", local_id, full_image_name])
    run_command(["docker", "push", full_image_name])

    return f"{artifact_repo}/{local_id}"


def parse_image_name_and_tag(docker_base_image: str) -> tuple[str, str]:
    """Parse a Docker image string into image name and tag components.

    Args:
        docker_base_image: Full Docker image identifier (e.g., 'ubuntu:20.04').

    Returns:
        Tuple of (image_name, tag). Defaults to 'latest' if no tag specified.
    """
    if ":" in docker_base_image:
        base_image, base_tag = docker_base_image.rsplit(":", 1)
    else:
        base_image = docker_base_image
        base_tag = "latest"
    return base_image, base_tag


def create_docker_run_command(
    image_id: str,
    command: list[str],
    *,
    foreground: bool,
    env: dict[str, Any],
    name: str = "eformer",
) -> list[str]:
    """Construct a Docker run command with TPU-specific configuration.

    Creates a privileged container with host networking, shared memory,
    and environment variables for MegaScale TPU coordination.

    Args:
        image_id: Docker image identifier to run.
        command: Command and arguments to execute in the container.
        foreground: If True, run interactively; if False, run detached.
        env: Additional environment variables to set.
        name: Container name (default: 'eformer').

    Returns:
        Complete Docker run command as a list of arguments.
    """
    docker_command = [
        "docker",
        "run",
        "-t" if foreground else "-d",
        f"--name={name}",
        "--privileged",
        "--shm-size=32gb",
        "--net=host",
        "--init",
        "--mount",
        "type=volume,source=eformer,target=/home/eformer",
        "-v",
        "/tmp:/tmp",
    ]

    for v in ["MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_NUM_SLICES", "MEGASCALE_PORT", "MEGASCALE_SLICE_ID"]:
        docker_command.extend(["-e", str(v)])

    for k, v in env.items():
        docker_command.extend(["-e", f"{k}={v}"])

    docker_command.extend([image_id, *command])
    return docker_command
