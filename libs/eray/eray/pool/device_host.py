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
from ..resources.topology import DEFAULT_SPLIT_BASE_PORT, SplitModeT, plan_host_partition, plan_host_split

logger = logging.getLogger("ray")


def _assert_supported_tpu_subset(assigned) -> None:
    """Refuse TPU chip subsets that libtpu cannot form into a sub-slice.

    Validated on a v5p-8: 2-chip sub-slices only initialize for aligned
    pairs ({0,1}, {2,3}); any other pair (columns or diagonals of the 2x2
    chip grid) crashes libtpu's SliceBuilder with a firmware core dump and
    kills the process with no Python traceback. Ray assigns arbitrary chip
    ids to fractional ``TPU`` requests, so the assignment must be checked
    in-task before JAX initializes.

    Only enforced on real TPU hosts (VFIO/accel device files present) so
    CPU test clusters with fake ``TPU`` resources are unaffected.

    Args:
        assigned: Chip ids Ray assigned to this task.

    Raises:
        RuntimeError: If two chips were assigned that are not an aligned
            ``{2k, 2k+1}`` pair.
    """
    import glob

    if len(assigned) != 2:
        return
    if not (os.path.exists("/dev/vfio") or glob.glob("/dev/accel*")):
        return
    ids = sorted(int(i) for i in assigned)
    if ids[0] % 2 != 0 or ids[1] != ids[0] + 1:
        raise RuntimeError(
            f"TPU chips {ids} cannot form a 2-chip sub-slice: libtpu only accepts aligned pairs "
            f"({{0,1}}, {{2,3}}, ...) — misaligned pairs crash the TPU runtime with a firmware core "
            f"dump. Ray assigned a misaligned pair; relaunch the swarm/split, or use 1-chip runs "
            f"for guaranteed placement."
        )


@ray.remote(max_calls=1)
def _fn_runner(fn, args, kwargs):
    """Run a function with JAX distributed shutdown on exit.

    Refuses unsupported 2-chip TPU subsets before user code initializes JAX
    (see _assert_supported_tpu_subset).

    Args:
        fn: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.

    Returns:
        The result of fn(*args, **kwargs).
    """
    _assert_supported_tpu_subset(ray.get_runtime_context().get_accelerator_ids().get("TPU") or [])
    try:
        return fn(*args, **kwargs)
    finally:
        try:
            import jax.distributed as jdist

            jdist.shutdown()
        except Exception:
            pass


@ray.remote(max_calls=1)
def _split_fn_runner(fn, args, kwargs, cooperative_base_port):
    """Run one split of a host, binding chip-dependent env from Ray's assignment.

    For cooperative splits, reads the TPU chip Ray assigned to this task and
    binds the chip-dependent process-topology variables before user code
    initializes any runtime. The chip cannot be pinned at plan time: Ray
    resolves its assignment through the process-visible chip ids at task
    start, so the assignment must be read back, not dictated.

    Args:
        fn: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        cooperative_base_port: Base port for cooperative process endpoints;
            None for isolated splits (no chip-dependent env to bind).

    Returns:
        The result of fn(*args, **kwargs).

    Raises:
        RuntimeError: If a cooperative split was not assigned exactly one
            chip, or an isolated split was assigned an unsupported 2-chip
            subset (see _assert_supported_tpu_subset).
    """
    assigned = ray.get_runtime_context().get_accelerator_ids().get("TPU") or []
    _assert_supported_tpu_subset(assigned)
    if cooperative_base_port is not None:
        if len(assigned) != 1:
            raise RuntimeError(f"cooperative split expected exactly one assigned TPU chip, got {assigned!r}")
        chip = int(assigned[0])
        os.environ["TPU_VISIBLE_CHIPS"] = str(chip)
        os.environ["CLOUD_TPU_TASK_ID"] = str(chip)
        os.environ["TPU_PROCESS_PORT"] = str(cooperative_base_port + chip)
        os.environ["ERAY_SPLIT_ID"] = str(chip)
    try:
        return fn(*args, **kwargs)
    finally:
        try:
            import jax.distributed as jdist

            jdist.shutdown()
        except Exception:
            pass


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
        _awaitables: ObjectRefs of the currently running task(s) — a single
            whole-host task or one task per split.
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
        self._awaitables: list[ray.ObjectRef] = []
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
        for t in tasks:
            # Per-ref: ray.cancel(force=True) raises on not-yet-finished
            # actor tasks (e.g. a class run's __ray_ready__ ref); one failure
            # must not abort cancellation of the remaining task refs.
            try:
                ray.cancel(t, force=True, recursive=True)
            except Exception as e:
                logger.warning(f"Failed to cancel a task: {e}")
        done, pending = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_s)
        if pending:
            logger.warning(f"Cancelled {len(done)} tasks; {len(pending)} still pending after {timeout_s}s.")

    def cancel_current(self):
        """Cancel the currently running task(s) if any.

        Cancels and waits for all current tasks to complete (a whole-host
        task or the split tasks launched by run_split_remote_fn), then
        clears the awaitable references.
        """
        if self._awaitables:
            self._cancel_tasks_and_wait(list(self._awaitables))
            self._awaitables = []

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

        self.cancel_current()
        self._kill_vfio_holders()
        self._hacky_remove_tpu_lockfile()

        host_env = {"TPU_HOST_ID": str(self.host_id), "TPU_SLICE_NAME": self.slice_name}
        if self.num_devices:
            host_env["TPU_NUM_DEVICES"] = str(self.num_devices)
        merged_runtime_env = self._merge_runtime_env(runtime_env, {**host_env, **(env or {})})

        resources = dict(extra_resources or {})

        if self.num_devices and "TPU" not in resources:
            resources["TPU"] = self.num_devices

        py_fn = self._unwrap_remote_fn(remote_fn)
        f_kwargs = f_kwargs or {}

        awaitable = _fn_runner.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
            resources=resources or None,
            num_cpus=num_cpus,
            num_gpus=0,
            memory=int(memory_bytes),
            runtime_env=merged_runtime_env,
            max_retries=0,
        ).remote(py_fn, f_args, f_kwargs)
        self._awaitables = [awaitable]

        return awaitable

    @staticmethod
    def _unwrap_remote_fn(remote_fn):
        """Extract the plain Python function from a Ray remote function.

        Args:
            remote_fn: A Ray RemoteFunction or plain callable.

        Returns:
            The underlying Python callable.
        """
        try:
            from ray.remote_function import RemoteFunction as _RF

            return remote_fn._function if isinstance(remote_fn, _RF) else remote_fn
        except Exception:
            return remote_fn

    def run_split_remote_fn(
        self,
        remote_fn,
        num_splits: int,
        *,
        mode: SplitModeT = "isolated",
        base_port: int = DEFAULT_SPLIT_BASE_PORT,
        chip_grid: tuple[int, int, int] | None = None,
        f_args: tuple = (),
        f_kwargs: dict | None = None,
        runtime_env: dict | None = None,
        env: dict | None = None,
        num_cpus: float = 0.0,
        memory_bytes: float = 20e9,
        extra_resources: dict | None = None,
    ) -> list[ray.ObjectRef]:
        """Split this host's chips across multiple concurrent processes.

        Instead of one task owning every chip on the host (run_remote_fn),
        launches ``num_splits`` tasks on this node, each reserving a
        fractional ``TPU`` resource and receiving the chip-partitioning
        environment computed by :func:`eray.resources.topology.plan_host_split`.
        The function runs once per split; user code can read ``ERAY_SPLIT_ID``
        / ``ERAY_NUM_SPLITS`` to identify its split.

        In ``isolated`` mode each split is an independent runtime whose chips
        are assigned disjointly by Ray's TPU accelerator manager (1 or 2
        chips per split). In ``cooperative`` mode the splits form one
        multi-process runtime, one process per chip, with explicit libtpu
        process topology.

        Args:
            remote_fn: Ray remote function or callable to execute once per split.
            num_splits: Number of processes to carve the host into. Must
                divide the host's chip count evenly.
            mode: ``"isolated"`` (default) or ``"cooperative"``.
            base_port: First port for cooperative process endpoints.
            chip_grid: Optional explicit host chip layout ``(x, y, z)`` for
                cooperative mode.
            f_args: Positional arguments passed to every split invocation.
            f_kwargs: Keyword arguments passed to every split invocation.
            runtime_env: Optional Ray runtime environment configuration.
            env: Additional environment variables merged into every split.
            num_cpus: CPUs reserved per split task (default: 0.0).
            memory_bytes: Memory reserved per split task (default: 20GB).
            extra_resources: Additional custom resources requested per split.

        Returns:
            List of ``num_splits`` ObjectRefs, index-aligned with split id.

        Raises:
            RuntimeError: If the host is unhealthy or its chip count is unknown.
            ValueError: If the split is invalid for this host (see plan_host_split).
            NotImplementedError: For unsupported cooperative layouts.

        Note:
            Splitting a host is validated on hardware only by running it on a
            TPU host; CPU-side tests cover the planned environment and task
            fan-out, not libtpu behavior.
        """
        if not self.healthy():
            raise RuntimeError(f"Host {self.host_id} unhealthy or preempted")
        if not self.num_devices:
            raise RuntimeError(
                f"Host {self.host_id} has an unknown chip count (num_devices={self.num_devices}); cannot split"
            )

        plan = plan_host_split(self.num_devices, num_splits, mode=mode, base_port=base_port, chip_grid=chip_grid)

        self.cancel_current()
        self._kill_vfio_holders()
        self._hacky_remove_tpu_lockfile()

        host_env = {
            "TPU_HOST_ID": str(self.host_id),
            "TPU_SLICE_NAME": self.slice_name,
            "TPU_NUM_DEVICES": str(self.num_devices),
        }

        base_resources = dict(extra_resources or {})
        base_resources.setdefault("TPU", plan.resources_per_split()["TPU"])

        py_fn = self._unwrap_remote_fn(remote_fn)
        f_kwargs = f_kwargs or {}
        cooperative_base_port = plan.base_port if mode == "cooperative" else None

        # Track on self incrementally: if a later launch in this loop raises,
        # cancel_current must still be able to reach the already-launched runs.
        self._awaitables = []
        awaitables = self._awaitables
        for split_env in plan.env_per_split:
            merged_runtime_env = self._merge_runtime_env(runtime_env, {**host_env, **split_env, **(env or {})})
            awaitables.append(
                _split_fn_runner.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
                    resources=dict(base_resources),
                    num_cpus=num_cpus,
                    num_gpus=0,
                    memory=int(memory_bytes),
                    runtime_env=merged_runtime_env,
                    max_retries=0,
                ).remote(py_fn, f_args, f_kwargs, cooperative_base_port)
            )

        return list(awaitables)

    def run_swarm_remote_fn(
        self,
        run_payloads: list[dict],
        *,
        runtime_env: dict | None = None,
        env: dict | None = None,
        num_cpus: float = 0.0,
        memory_bytes: float = 20e9,
        namespace: str | None = None,
        actor_name_suffix: str = "",
    ) -> list[ray.ObjectRef]:
        """Run a swarm of heterogeneous workloads on this host, split by chips.

        Unlike run_split_remote_fn (one function, uniform splits), each run in
        the swarm has its own function and chip count — e.g. ``(2, 1, 1)`` on
        a 4-chip host. All runs are isolated runtimes whose chips are assigned
        disjointly by Ray's fractional ``TPU`` scheduling. Chip validity is
        checked by :func:`eray.resources.topology.plan_host_partition`.

        A run whose ``fn`` is a class is instantiated as a **detached named
        Ray actor** on this node instead of a task: it holds its chips and
        stays addressable via ``ray.get_actor(name + suffix, namespace=...)``
        until explicitly killed (it survives this host actor and the pool).
        Its ObjectRef resolves when the actor has finished ``__init__``.

        Args:
            run_payloads: One dict per run with keys:
                ``fn`` (callable or class, required), ``is_class`` (bool,
                optional — inferred when absent), ``chips`` (int or None —
                None on every run means an even split of this host's chips),
                ``name`` (str or None, surfaced as ``ERAY_RUN_NAME``;
                required for class runs), ``f_args`` (tuple), ``f_kwargs``
                (dict or None), and ``env`` (dict or None, per-run extra
                environment).
            runtime_env: Optional Ray runtime environment configuration.
            env: Environment variables merged into every run.
            num_cpus: CPUs reserved per run task/actor (default: 0.0).
            memory_bytes: Memory reserved per run task/actor (default: 20GB).
            namespace: Ray namespace for class runs' named actors. Required
                when any run is a class.
            actor_name_suffix: Suffix appended to class runs' actor names to
                keep them unique when the swarm layout repeats across hosts.

        Returns:
            List of ObjectRefs, index-aligned with ``run_payloads``. Function
            runs resolve to the function's return value; class runs resolve
            (to True) once the actor is ready.

        Raises:
            RuntimeError: If the host is unhealthy or its chip count is unknown.
            ValueError: If payloads are malformed or the chip plan is invalid
                for this host (see plan_host_partition), if only some runs
                specify ``chips``, or if a class run lacks a name/namespace.
        """
        import inspect

        if not self.healthy():
            raise RuntimeError(f"Host {self.host_id} unhealthy or preempted")
        if not self.num_devices:
            raise RuntimeError(
                f"Host {self.host_id} has an unknown chip count (num_devices={self.num_devices}); cannot swarm"
            )
        if not run_payloads:
            raise ValueError("run_payloads must not be empty")
        for i, payload in enumerate(run_payloads):
            if not callable(payload.get("fn")):
                raise ValueError(f"run_payloads[{i}] has no callable 'fn'")

        explicit = [p.get("chips") for p in run_payloads]
        if all(c is None for c in explicit):
            if self.num_devices % len(run_payloads) != 0:
                raise ValueError(
                    f"{len(run_payloads)} runs do not divide {self.num_devices} chips evenly; "
                    f"give each run an explicit 'chips' count"
                )
            counts = [self.num_devices // len(run_payloads)] * len(run_payloads)
        elif any(c is None for c in explicit):
            raise ValueError("either every run or no run may specify 'chips'")
        else:
            # Do not int() here — plan_host_partition validates wholeness and
            # converts; pre-truncating would silently grant different hardware
            # than requested (e.g. chips=1.5 becoming 1).
            counts = list(explicit)

        plan = plan_host_partition(
            self.num_devices,
            counts,
            run_names=[p.get("name") for p in run_payloads],
        )

        self.cancel_current()
        self._kill_vfio_holders()
        self._hacky_remove_tpu_lockfile()

        host_env = {
            "TPU_HOST_ID": str(self.host_id),
            "TPU_SLICE_NAME": self.slice_name,
            "TPU_NUM_DEVICES": str(self.num_devices),
        }

        # Track on self incrementally: a mid-loop synchronous failure (e.g. a
        # duplicate actor name) must leave earlier launches reachable by
        # cancel_current instead of orphaned.
        self._awaitables = []
        awaitables = self._awaitables
        for payload, run_env, resources in zip(run_payloads, plan.env_per_run, plan.resources_per_run(), strict=True):
            merged_runtime_env = self._merge_runtime_env(
                runtime_env,
                {**host_env, **run_env, **(env or {}), **(payload.get("env") or {})},
            )
            is_class = payload.get("is_class")
            if is_class is None:
                is_class = inspect.isclass(payload["fn"])
            run_num_cpus = payload.get("num_cores")
            run_num_cpus = num_cpus if run_num_cpus is None else run_num_cpus
            if is_class:
                name = payload.get("name")
                if not name:
                    raise ValueError("class runs need a name (it becomes the actor name)")
                if not namespace:
                    raise ValueError("class runs need a namespace so the named actors are addressable")
                # Detached: the runtime must survive this host actor and the
                # swarm's pool drain; it is torn down via shutdown_swarm or
                # ray.kill(ray.get_actor(...)).
                actor = (
                    ray.remote(payload["fn"])
                    .options(
                        name=f"{name}{actor_name_suffix}",
                        namespace=namespace,
                        lifetime="detached",
                        scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
                        resources=resources,
                        num_cpus=run_num_cpus,
                        num_gpus=0,
                        memory=int(memory_bytes),
                        runtime_env=merged_runtime_env,
                    )
                    .remote(*(payload.get("f_args") or ()), **(payload.get("f_kwargs") or {}))
                )
                awaitables.append(actor.__ray_ready__.remote())
            else:
                awaitables.append(
                    _fn_runner.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(self._node_id, soft=False),
                        resources=resources,
                        num_cpus=run_num_cpus,
                        num_gpus=0,
                        memory=int(memory_bytes),
                        runtime_env=merged_runtime_env,
                        max_retries=0,
                    ).remote(
                        self._unwrap_remote_fn(payload["fn"]),
                        payload.get("f_args") or (),
                        payload.get("f_kwargs") or {},
                    )
                )

        return list(awaitables)

    def shutdown(self) -> None:
        """Gracefully shut down this host actor.

        Cancels any running task and marks the host as failed.
        """
        try:
            self.cancel_current()
        finally:
            self._failed = True
            logger.info(f"Shut down DeviceHostActor[{self.slice_name}#{self.host_id}]")
