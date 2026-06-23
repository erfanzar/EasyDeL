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


"""Ray actor for managing a single accelerator host."""

from __future__ import annotations

import logging
import os

import ray
import requests
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ..core.cluster import HostInfo

logger = logging.getLogger("ray")


@ray.remote
class DeviceHostActor:
    """Ray actor for managing a single TPU host within a slice.

    Handles task execution on a specific TPU host, managing TPU resources,
    environment variables, and task lifecycle. Supports cancellation and
    health monitoring. Each DeviceHostActor runs on a specific Ray node
    and manages TPU devices on that node.

    Attributes:
        host_id: Unique identifier for this host within its slice (0-based).
        slice_name: Name of the TPU slice this host belongs to.
        num_devices: Number of TPU devices on this host.
        _failed: Whether this host has encountered a failure.
        _awaitable: Current running task's ObjectRef.
        _node_id: Ray node ID where this actor is running.

    Environment Variables Set:
        TPU_HOST_ID: Host index within the slice.
        TPU_SLICE_NAME: Name of the parent slice.
        TPU_NUM_DEVICES: Number of devices on this host (if available).
    """

    def __init__(self, host_id: int, slice_name: str, num_devices: int | None = None):
        """Initialize a DeviceHostActor.

        Args:
            host_id: Unique identifier for this host within its slice.
            slice_name: Name of the TPU slice this host belongs to.
            num_devices: Optional number of TPU devices available on this host.
        """
        self.host_id = host_id
        self.slice_name = slice_name
        self.num_devices = num_devices or 0
        self._failed = False
        self._awaitable: ray.ObjectRef | None = None
        self._node_id = ray.get_runtime_context().get_node_id()
        logger.info(f"DeviceHostActor[{slice_name}#{host_id}] init; num_devices={num_devices}; node_id={self._node_id}")

    def healthy(self) -> bool:
        """Check if this host is healthy and operational.

        Returns:
            True if host is not failed and not being preempted.
        """
        return not self._failed and not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """Check if this GCP instance is being preempted.

        Queries the GCP metadata server to determine if the instance
        is scheduled for preemption.

        Returns:
            True if instance is being preempted, False otherwise.
        """
        try:
            r = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                headers={"Metadata-Flavor": "Google"},
                timeout=1.0,
            )
            return r.status_code == 200 and r.text.strip().upper() == "TRUE"
        except requests.RequestException:
            return False

    def get_info(self) -> HostInfo:
        """Get current information about this host.

        Returns:
            HostInfo object with host metadata and status.
        """
        return HostInfo(
            host_id=self.host_id,
            slice_name=self.slice_name,
            num_devices=self.num_devices,
            healthy=self.healthy(),
            failed=self._failed,
            node_id=self._node_id,
        )

    def _kill_vfio_holders(self):
        """Quietly kill processes holding /dev/vfio/*.

        Controlled by:
        - ERAY_KILL_VFIO=1 to enable (default 0 = disabled)
        - ERAY_INSTALL_LSOF=1 to attempt quiet, noninteractive lsof install (optional)

        All command outputs are suppressed; never prompts for sudo.
        """
        import os

        if os.getenv("ERAY_KILL_VFIO", "1") != "1":
            return
        try:
            import shutil
            import signal
            import subprocess

            def run_quiet(cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
                """Run a shell command quietly, suppressing output.

                Args:
                    cmd: Shell command to execute.
                    capture: If True, capture stdout instead of suppressing it.

                Returns:
                    CompletedProcess from subprocess.run.
                """
                return subprocess.run(
                    ["bash", "-lc", cmd],
                    check=False,
                    stdout=(subprocess.PIPE if capture else subprocess.DEVNULL),
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"),
                )

            if shutil.which("lsof") is None and os.getenv("ERAY_INSTALL_LSOF", "0") == "1":
                run_quiet("sudo -n apt-get -qq update || true")
                run_quiet("sudo -n apt-get -qq -y install lsof || true")

            if shutil.which("lsof") is None:
                return

            p = run_quiet("lsof -t /dev/vfio/* 2>/dev/null | sort -u", capture=True)
            pids = []
            if p and p.stdout:
                pids = [int(pid) for pid in p.stdout.split() if pid.isdigit() and int(pid) != os.getpid()]

            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        except Exception:
            pass

    def _merge_runtime_env(self, runtime_env: dict | None, env_vars: dict | None) -> dict:
        """Merge environment variables into a runtime environment dict.

        Args:
            runtime_env: Base runtime environment configuration.
            env_vars: Environment variables to merge in.

        Returns:
            Merged runtime environment dictionary.
        """
        re = dict(runtime_env or {})
        if env_vars:
            ev = dict(re.get("env_vars", {}))
            ev.update({str(k): str(v) for k, v in env_vars.items() if v is not None})
            re["env_vars"] = ev
        return re

    def _hacky_remove_tpu_lockfile(self):
        """Remove TPU lockfile that may prevent TPU initialization.

        Attempts to remove /tmp/libtpu_lockfile which can cause issues
        when reusing TPU resources. Falls back to sudo if needed.
        """
        try:
            if os.path.exists("/tmp/libtpu_lockfile"):
                os.unlink("/tmp/libtpu_lockfile")
        except FileNotFoundError:
            pass
        except PermissionError:
            try:
                os.system("sudo rm /tmp/libtpu_lockfile")
            except Exception:
                pass

    def _cancel_tasks_and_wait(self, tasks: list[ray.ObjectRef], timeout_s: float = 240.0) -> None:
        """Cancel Ray tasks and wait for them to complete.

        Forcefully cancels all provided tasks and waits for completion
        or timeout.

        Args:
            tasks: List of Ray ObjectRefs to cancel.
            timeout_s: Maximum time to wait for cancellation.
        """
        if not tasks:
            return
        try:
            for t in tasks:
                ray.cancel(t, force=True, recursive=True)
        except Exception as e:
            logger.warning(f"Failed to cancel some tasks: {e}")
        done, pending = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_s)
        if pending:
            logger.warning(f"Cancelled {len(done)} tasks; {len(pending)} still pending after {timeout_s}s.")

    def cancel_current(self):
        """Cancel the currently running task if any.

        Cancels and waits for the current task to complete,
        then clears the awaitable reference.
        """
        if self._awaitable:
            self._cancel_tasks_and_wait([self._awaitable])
            self._awaitable = None

    def run_remote_fn(
        self,
        remote_fn,
        *,
        f_args: tuple = (),
        f_kwargs: dict | None = None,
        runtime_env: dict | None = None,
        env: dict | None = None,
        num_cpus: float = 0.0,
        memory_bytes: float = 20e9,
        extra_resources: dict | None = None,
    ) -> ray.ObjectRef:
        """Launch a cancelable task on this host's node, reserving TPU resources.

        Executes a Ray remote function on this specific TPU host with proper
        resource allocation and node affinity. Automatically cancels any
        previously running task and manages TPU lockfiles.

        Args:
            remote_fn: Ray remote function or callable to execute. If not already
                a remote function, will be wrapped with @ray.remote(max_calls=1).
            f_args: Positional arguments to pass to the remote function.
            f_kwargs: Keyword arguments to pass to the remote function.
            runtime_env: Optional Ray runtime environment configuration for
                dependency management and environment setup.
            env: Additional environment variables to merge with host environment.
            num_cpus: Number of CPUs to reserve for the task (default: 0.0).
            memory_bytes: Memory to reserve in bytes (default: 20GB).
            extra_resources: Additional custom resources to request.

        Returns:
            ray.ObjectRef: Reference to the running task that can be used to
                retrieve results with ray.get() or cancel with ray.cancel().

        Raises:
            RuntimeError: If host is unhealthy or being preempted.

        Note:
            - Task runs with strict node affinity to this host's node.
            - TPU resources are automatically reserved based on num_devices.
            - Previous tasks are cancelled before starting new ones.
            - TPU lockfile is cleaned up before execution.
        """
        if not self.healthy():
            raise RuntimeError(f"Host {self.host_id} unhealthy or preempted")

        if self._awaitable:
            self._cancel_tasks_and_wait([self._awaitable])

        self._kill_vfio_holders()
        self._hacky_remove_tpu_lockfile()

        host_env = {"TPU_HOST_ID": str(self.host_id), "TPU_SLICE_NAME": self.slice_name}
        if self.num_devices:
            host_env["TPU_NUM_DEVICES"] = str(self.num_devices)
        merged_runtime_env = self._merge_runtime_env(runtime_env, {**host_env, **(env or {})})

        resources = dict(extra_resources or {})

        if self.num_devices and "TPU" not in resources:
            resources["TPU"] = self.num_devices

        try:
            from ray.remote_function import RemoteFunction as _RF

            py_fn = remote_fn._function if isinstance(remote_fn, _RF) else remote_fn
        except Exception:
            py_fn = remote_fn

        f_kwargs = f_kwargs or {}

        @ray.remote(max_calls=1)
        def _runner(fn, args, kwargs):
            """Run a function with JAX distributed shutdown on exit.

            Args:
                fn: The function to execute.
                args: Positional arguments for the function.
                kwargs: Keyword arguments for the function.

            Returns:
                The result of fn(*args, **kwargs).
            """
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    import jax.distributed as jdist

                    jdist.shutdown()
                except Exception:
                    pass

        self._awaitable = _runner.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
            resources=resources or None,
            num_cpus=num_cpus,
            num_gpus=0,
            memory=int(memory_bytes),
            runtime_env=merged_runtime_env,
            max_retries=0,
        ).remote(py_fn, f_args, f_kwargs)

        return self._awaitable

    def shutdown(self) -> None:
        """Gracefully shut down this host actor.

        Cancels any running task and marks the host as failed.
        """
        try:
            self.cancel_current()
        finally:
            self._failed = True
            logger.info(f"Shut down DeviceHostActor[{self.slice_name}#{self.host_id}]")
