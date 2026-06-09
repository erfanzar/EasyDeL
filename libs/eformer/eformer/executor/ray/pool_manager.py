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


"""Resource pool management for distributed Ray actors.

This module provides comprehensive abstractions for managing pools of Ray actors,
with specialized focus on TPU/GPU slice management for distributed computing.
It includes health monitoring, automatic scaling, resource lifecycle management,
and placement group coordination for optimal resource allocation.

Key Components:
    - **ActorPoolMember**: Wrapper for actor handles with metadata
    - **ResourcePoolManager**: Abstract base for managing actor pools
    - **SlicePoolManager**: Specialized manager for TPU/GPU slices with placement groups
    - **SliceActor**: Ray actor for managing individual compute slices
    - **DeviceHostActor**: Ray actor for managing individual TPU hosts within slices

Resource Management Features:
    - Placement group coordination with STRICT_SPREAD strategy
    - Automatic resource request handling through Ray autoscaler
    - Health monitoring with graceful shutdown sequences
    - Robust error handling with actor restart capabilities
    - Slot-based actor allocation for deterministic placement

Environment Variables:
    - **EFORMER_SCALE_POLL_S**: Scaling operation polling interval (default: "30")
    - **EFORMER_SCALE_ADD_TIMEOUT_S**: Timeout for adding new actors (default: "604800")

Example:
    Managing a multi-slice TPU configuration with placement groups:

    >>> from eformer.executor.ray import SlicePoolManager
    >>>
    >>>
    >>> manager = SlicePoolManager(tpu_type="v4-8")
    >>> manager.scale_multislice(num_slices=4)
    >>> actors = manager.get_all_actors_in_pool()
    >>>
    >>>
    >>> manager.prepare_all_slices()
    >>> manager.drain_actor_pool()
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar
from uuid import uuid4

import ray
import requests
from ray.actor import ActorHandle
from ray.autoscaler.sdk import request_resources
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from .types import HostInfo, SliceInfo

logger = logging.getLogger("ray")

HEALTH_CHECK_TIMEOUT_S = 60
SLICE_ACTOR_START_TIMEOUT_S = 4 * 60 * 60
SCALE_POLL_S = int(os.getenv("EFORMER_SCALE_POLL_S", "30"))
SCALE_ADD_TIMEOUT_S = int(os.getenv("EFORMER_SCALE_ADD_TIMEOUT_S", "604800"))
ActorInfoT = TypeVar("ActorInfoT")


class InsufficientSlicesError(RuntimeError):
    """Raised when the requested number of TPU slices cannot be allocated.

    This exception is raised by SlicePoolManager.scale_multislice when
    none of the requested slice counts can be satisfied, typically due to:
    - Insufficient TPU resources in the cluster
    - Preemption of TPU nodes during scaling
    - Ray autoscaler unable to provision required nodes

    The exception message includes details about requested vs available slices.

    Example:
        >>> manager = SlicePoolManager(tpu_type="v4-32")
        >>> try:
        ...     manager.scale_multislice([4, 8])
        ... except InsufficientSlicesError as e:
        ...     print(f"Could not allocate TPU slices: {e}")
        ...
    """

    pass


@dataclass(frozen=True)
class ActorPoolMember(Generic[ActorInfoT]):
    """Container for an actor handle and its associated metadata.

    Attributes:
        actor: Ray actor handle for remote execution.
        actor_info: Metadata about the actor (type depends on ActorInfoT).
    """

    actor: ActorHandle
    actor_info: ActorInfoT


class ResourcePoolManager(Generic[ActorInfoT]):
    """Abstract base class for managing pools of Ray actors.

    Provides common functionality for scaling, health monitoring, and
    lifecycle management of actor pools. Subclasses should implement
    create_actor() to define how actors are created.

    Attributes:
        _actor_pool: List of active actor pool members.
    """

    def __init__(self) -> None:
        """Initialize an empty actor pool."""
        self._actor_pool: list[ActorPoolMember[ActorInfoT]] = []

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all actor handles in the pool.

        Returns:
            List of Ray actor handles.
        """
        return [m.actor for m in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[ActorInfoT]]:
        """Get a copy of all pool members with their metadata.

        Returns:
            List of ActorPoolMember objects containing actors and their info.
        """
        return self._actor_pool.copy()

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        return self.__class__.__name__

    def get_actor_name_from_actor_info(self, actor_info: ActorInfoT) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return str(actor_info)

    def create_actor(self) -> ActorHandle:
        """Create a new actor instance.

        Must be implemented by subclasses to define actor creation logic.

        Returns:
            Ray actor handle for the newly created actor.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy actors from the pool.

        Performs health checks on all actors and removes those that are
        unresponsive, dead, or unhealthy. Attempts to kill removed actors.
        """
        if not self._actor_pool:
            return

        ref_map = {m: m.actor.healthy.remote() for m in self._actor_pool}
        refs = list(ref_map.values())

        done, _ = ray.wait(refs, num_returns=len(refs), timeout=HEALTH_CHECK_TIMEOUT_S)

        done_set = set(done)
        healthy: list[ActorPoolMember[HostInfo]] = []

        for member, ref in ref_map.items():
            name = self.get_actor_name_from_actor_info(member.actor_info)
            if ref in done_set:
                try:
                    if ray.get(ref, timeout=0):
                        healthy.append(member)
                    else:
                        logger.warning(f"Actor {name} reported unhealthy; killing")
                        try:
                            ray.kill(member.actor, no_restart=True)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Actor {name} health check exception ({e}); killing")
                    try:
                        ray.kill(member.actor, no_restart=True)
                    except Exception:
                        pass
            else:
                logger.warning(f"Actor {name} health timeout; killing")
                try:
                    ray.kill(member.actor, no_restart=True)
                except Exception:
                    pass

        self._actor_pool = healthy

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new actors to the pool to reach desired size.

        Creates new actors asynchronously and waits for them to start.
        Actors that fail to start within the timeout are killed.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        current = len(self._actor_pool)
        if current >= desired_num_actors:
            return
        num_to_add = desired_num_actors - current
        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        actors = [self.create_actor() for _ in range(num_to_add)]
        awaitables = [(actor, actor.get_info.remote()) for actor in actors]

        logger.info(f"Waiting up to {SLICE_ACTOR_START_TIMEOUT_S}s for {num_to_add} slice actors to start...")
        ray.wait([a for _, a in awaitables], num_returns=len(awaitables), timeout=SLICE_ACTOR_START_TIMEOUT_S)

        started = 0
        for actor, info_ref in awaitables:
            try:
                info = ray.get(info_ref, timeout=0)
                self._actor_pool.append(ActorPoolMember(actor, info))
                started += 1
                logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)}")
            except Exception as e:
                logger.warning(f"SliceActor failed to start in time: {e}; killing actor")
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass

        logger.info(f"Started {started}/{num_to_add} slice actors")

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove actors to reach the desired pool size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Removed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        """Scale the actor pool to the desired size.

        First removes unhealthy actors, then adds or removes actors
        as needed to reach the target size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current = len(self._actor_pool)
        if current < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif current > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        if not self._actor_pool:
            return

        shutdown_refs = []
        for member in self._actor_pool:
            try:
                shutdown_refs.append(member.actor.shutdown.remote())
            except Exception:
                pass

        try:
            ray.wait(shutdown_refs, num_returns=len(shutdown_refs), timeout=5.0)
        except Exception:
            pass

        for member in self._actor_pool:
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Killed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

        self._actor_pool = []


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
        - EFORMER_KILL_VFIO=1 to enable (default 0 = disabled)
        - EFORMER_INSTALL_LSOF=1 to attempt quiet, noninteractive lsof install (optional)

        All command outputs are suppressed; never prompts for sudo.
        """
        import os

        if os.getenv("EFORMER_KILL_VFIO", "1") != "1":
            return
        try:
            import shutil
            import signal
            import subprocess

            def run_quiet(cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
                return subprocess.run(
                    ["bash", "-lc", cmd],
                    check=False,
                    stdout=(subprocess.PIPE if capture else subprocess.DEVNULL),
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"),
                )

            if shutil.which("lsof") is None and os.getenv("EFORMER_INSTALL_LSOF", "0") == "1":
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
            runtime_env: Optional Ray runtime environment configuration for
                dependency management and environment setup.
            env: Additional environment variables to merge with host environment.
            num_cpus: Number of CPUs to reserve for the task (default: 8.0).
            memory_bytes: Memory to reserve in bytes (default: 20GB).
            extra_resources: Additional custom resources to request.

        Returns:
            ray.ObjectRef: Reference to the running task that can be used to
                retrieve results with ray.get() or cancel with ray.cancel().

        Raises:
            RuntimeError: If host is unhealthy or being preempted.
            ValueError: If remote_fn doesn't have max_calls=1 set.

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


@ray.remote
class SliceActor:
    """Ray actor for managing a TPU slice with multiple hosts.

    Coordinates multiple TPU hosts within a single slice, handling
    placement groups, resource allocation, and distributed task execution.
    Each SliceActor manages a complete TPU pod/slice and ensures hosts
    are properly distributed across nodes using placement groups.

    Attributes:
        _actor_pool: List of DeviceHostActor pool members for this slice.
        _failed: Whether this slice has failed or been preempted.
        _slice_info: Detailed information about the TPU slice configuration.
        _host_placement_group: Ray placement group for STRICT_SPREAD host distribution.
        _host_infos: Node and device information for each host in the slice.

    Lifecycle:
        1. Created by SlicePoolManager with TPU head resource requirement.
        2. Discovers slice configuration from TPU environment.
        3. Creates placement group for host distribution.
        4. Spawns DeviceHostActors on each host node.
        5. Manages task execution across all hosts.
        6. Cleans up resources on shutdown.
    """

    def __init__(self):
        """Initialize a slice actor.

        Creates an empty actor pool and prepares to manage TPU hosts
        within a single slice. Discovers slice information from the
        TPU environment during initialization.
        """
        self._actor_pool: list[ActorPoolMember[HostInfo]] = []
        self._failed = False
        self._slice_info: SliceInfo | None = None
        self._host_placement_group = None
        self._host_infos: list[dict] | None = None
        self._initialize_slice_info()

    @staticmethod
    @ray.remote(num_cpus=0)
    def discover_node_info():
        """Discover information about the current Ray node.

        Static remote function to gather node metadata including IP,
        node ID, pod name, and TPU count.

        Returns:
            Dictionary with node information.
        """
        import ray

        pod_name = None
        ray_tpu = None
        try:
            from ray.util.accelerators import tpu as ray_tpu

            pod_name = ray_tpu.get_current_pod_name()
        except Exception:
            ray_tpu = None
        num_devices = None
        try:
            from ray._private.accelerators import TPUAcceleratorManager

            num_devices = TPUAcceleratorManager.get_current_node_num_accelerators()
        except Exception:
            pass
        num_hosts = 1
        if ray_tpu is not None:
            try:
                num_hosts = int(ray_tpu.get_current_pod_worker_count())
            except Exception:
                num_hosts = 1
        if os.getenv("EFORMER_MODERATE", "1") == "1" and pod_name:
            available_hosts = ray.cluster_resources().get(pod_name, None)
            if available_hosts is not None and num_hosts > available_hosts:
                available_hosts = int(available_hosts)
                num_devices = int(available_hosts)
                real_num_devices = 4
                print(
                    f"auto-discovered to set num_hosts from {num_hosts} to {available_hosts} and "
                    f"num_devices from {num_devices} to {real_num_devices}"
                )
                num_hosts = available_hosts
                num_devices = real_num_devices
        return {
            "ip": ray.util.get_node_ip_address(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "pod_name": pod_name,
            "num_devices": num_devices,
            "num_hosts": num_hosts,
        }

    def _create_actor_for_host_id(self, host_id: int) -> ActorHandle:
        """Create a DeviceHostActor for a specific host ID.

        Creates an actor with node affinity to ensure it runs on the
        correct TPU host. The actor is created as detached to persist
        beyond the parent's lifetime.

        Args:
            host_id: Zero-based index of the host within the slice.

        Returns:
            Ray actor handle for the DeviceHostActor.

        Raises:
            RuntimeError: If slice/host info not initialized or host_id invalid.
        """
        if not self._slice_info or not self._host_infos:
            raise RuntimeError("Slice or host info not initialized")
        if host_id >= len(self._host_infos):
            raise RuntimeError(f"Missing host info for host_id={host_id}")

        info = self._host_infos[host_id]
        node_id = info["node_id"]
        num_devices_for_host = info.get("num_devices")

        return DeviceHostActor.options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            name=f"{self._slice_info.slice_name}-host-{host_id}-{uuid4().hex[:8]}",
        ).remote(host_id, self._slice_info.slice_name, num_devices_for_host)

    def _initialize_slice_info(self) -> None:
        """Initialize slice information from TPU environment.

        Discovers slice name, host count, and TPU configuration.
        Sets _failed flag if initialization fails.
        """
        try:
            from ray.util.accelerators import tpu as ray_tpu

            num_accelerators_per_host = None
            try:
                from ray._private.accelerators import TPUAcceleratorManager

                num_accelerators_per_host = TPUAcceleratorManager.get_current_node_num_accelerators()
            except Exception:
                pass
            slice_name = ray_tpu.get_current_pod_name()
            num_hosts = int(ray_tpu.get_current_pod_worker_count())
            if os.getenv("EFORMER_MODERATE", "1") == "1":
                available_hosts = ray.cluster_resources().get(slice_name, None)
                if available_hosts is not None and num_hosts > available_hosts:
                    available_hosts = int(available_hosts)
                    real_accelerators_per_host = 4
                    print(
                        f"setting {num_hosts=} to {available_hosts=} and "
                        f"{num_accelerators_per_host=} to {real_accelerators_per_host=}"
                    )
                    num_hosts = available_hosts
                    num_accelerators_per_host = real_accelerators_per_host
            ip_address = ray.util.get_node_ip_address()

            self._slice_info = SliceInfo(
                slice_name=slice_name,
                num_hosts=num_hosts,
                ip_address=ip_address,
                num_accelerators_per_host=num_accelerators_per_host or 0,
            )
            logger.info(f"Initialized SliceActor: {self._slice_info}")
        except Exception as e:
            logger.error(f"Failed to initialize slice info: {e}")
            self._failed = True

    def healthy(self) -> bool:
        """Check if the slice is healthy and operational.

        Verifies that the slice has not failed and is not being preempted
        by the cloud provider.

        Returns:
            True if slice is healthy and available for task execution,
            False if failed or being preempted.
        """
        if self._failed:
            return False
        return not self.is_being_preempted()

    def is_being_preempted(self) -> bool:
        """Check if this GCP instance is being preempted.

        Queries GCP metadata server to determine preemption status.

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

    def get_info(self) -> SliceInfo:
        """Get current information about this slice.

        Returns:
            SliceInfo object containing slice configuration, host count,
            IP addresses, and TPU device information.

        Raises:
            RuntimeError: If slice information has not been initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        return self._slice_info

    def get_all_actors_in_pool(self) -> list[ActorHandle]:
        """Get all actor handles in the pool.

        Returns:
            List of Ray actor handles.
        """
        return [m.actor for m in self._actor_pool]

    def get_all_pool_members(self) -> list[ActorPoolMember[HostInfo]]:
        """Get a copy of all pool members with their metadata.

        Returns:
            List of ActorPoolMember objects containing actors and their info.
        """
        return self._actor_pool.copy()

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        if self._slice_info:
            return f"SliceActor({self._slice_info.slice_name})"
        return "SliceActor(uninitialized)"

    def get_actor_name_from_actor_info(self, actor_info: HostInfo) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return f"{actor_info.slice_name}-host-{actor_info.host_id}"

    def _ensure_host_placement_group(self) -> None:
        """Ensure placement group exists for distributing hosts.

        Creates a placement group with STRICT_SPREAD strategy to ensure
        hosts are distributed across different nodes. Also discovers
        host information for each placement group bundle.

        Raises:
            RuntimeError: If slice info is not initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        if self._host_placement_group is not None:
            return

        slice_label = self._slice_info.slice_name
        bundles = [{"CPU": 0, slice_label: 1} for _ in range(self._slice_info.num_hosts)]
        request_resources(bundles=bundles)
        self._host_placement_group = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(self._host_placement_group.ready())

        futures = [
            SliceActor.discover_node_info.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    self._host_placement_group,
                    placement_group_bundle_index=i,
                    placement_group_capture_child_tasks=False,
                )
            ).remote()
            for i in range(self._slice_info.num_hosts)
        ]
        self._host_infos = ray.get(futures)
        self._slice_info = SliceInfo(
            slice_name=self._slice_info.slice_name,
            num_hosts=self._slice_info.num_hosts,
            ip_address=self._slice_info.ip_address,
            num_accelerators_per_host=self._slice_info.num_accelerators_per_host,
            node_ids=[h.get("node_id") for h in self._host_infos],
            host_infos=self._host_infos,
        )
        logger.info(f"Prepared host placement group for slice {self._slice_info.slice_name};")

    def prepare_hosts(self) -> None:
        """Prepare host placement group for this slice.

        Ensures the placement group is created and ready for host actors.
        """
        self._ensure_host_placement_group()

    def create_actor(self) -> ActorHandle:
        """Create a new TPU host actor within this slice.

        Creates a DeviceHostActor with proper node affinity to ensure it runs
        on the correct host within the slice. Assigns TPU resources if
        available on the target node.

        Returns:
            Ray actor handle for the newly created DeviceHostActor.

        Raises:
            RuntimeError: If slice is not initialized or host info is missing.
        """
        if not self._slice_info:
            raise RuntimeError("Cannot create host actor: slice not initialized")
        self._ensure_host_placement_group()

        host_id = len(self._actor_pool)
        if not self._host_infos or host_id >= len(self._host_infos):
            raise RuntimeError(f"Missing host info for host_id={host_id}")

        info = self._host_infos[host_id]
        node_id = info["node_id"]
        num_devices_for_host = info.get("num_devices")

        return DeviceHostActor.options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(host_id, self._slice_info.slice_name, num_devices_for_host)

    def ensure_host_pool(self, desired_hosts: int | None = None) -> None:
        """Ensure the host actor pool has the desired number of hosts.

        Args:
            desired_hosts: Target number of hosts, defaults to slice's host count.

        Raises:
            RuntimeError: If slice info is not initialized.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        self._ensure_host_placement_group()
        target = desired_hosts if desired_hosts is not None else self._slice_info.num_hosts
        self._scale_actor_pool(target)

    def _remove_unhealthy_members_from_actor_pool(self) -> None:
        """Remove unhealthy actors from the pool.

        Performs health checks on all actors and removes those that are
        unresponsive, dead, or unhealthy. Attempts to kill removed actors.
        """
        if not self._actor_pool:
            return

        ref_map = {m: m.actor.healthy.remote() for m in self._actor_pool}
        refs = list(ref_map.values())

        done, _ = ray.wait(refs, num_returns=len(refs), timeout=HEALTH_CHECK_TIMEOUT_S)

        done_set = set(done)
        healthy: list[ActorPoolMember[ActorInfoT]] = []

        for member, ref in ref_map.items():
            name = self.get_actor_name_from_actor_info(member.actor_info)
            if ref in done_set:
                try:
                    if ray.get(ref, timeout=0):
                        healthy.append(member)
                    else:
                        logger.warning(f"Actor {name} reported unhealthy; killing")
                        try:
                            ray.kill(member.actor, no_restart=True)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Actor {name} health check exception ({e}); killing")
                    try:
                        ray.kill(member.actor, no_restart=True)
                    except Exception:
                        pass
            else:
                logger.warning(f"Actor {name} health timeout; killing")
                try:
                    ray.kill(member.actor, no_restart=True)
                except Exception:
                    pass

        self._actor_pool = healthy

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new host actors to reach the desired pool size.

        Creates host actors sequentially to ensure correct host_id assignment.
        Each actor is given a timeout to start, after which it's killed.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        current = len(self._actor_pool)

        if current >= desired_num_actors:
            return
        to_add = desired_num_actors - current
        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        info_ref_to_actor: dict[ray.ObjectRef, ActorHandle] = {}
        for host_id in range(current, current + to_add):
            actor = self._create_actor_for_host_id(host_id)
            info_ref = actor.get_info.remote()
            info_ref_to_actor[info_ref] = actor

        pending = list(info_ref_to_actor.keys())
        started = 0
        poll_s = 2.0
        deadline = time.time() + HEALTH_CHECK_TIMEOUT_S

        while pending and time.time() < deadline:
            done, pending = ray.wait(pending, num_returns=len(pending), timeout=poll_s)
            if not done:
                continue
            for info_ref in done:
                actor = info_ref_to_actor.pop(info_ref, None)
                if not actor:
                    continue
                try:
                    info = ray.get(info_ref, timeout=0)
                    self._actor_pool.append(ActorPoolMember(actor, info))
                    started += 1
                    logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)}")
                except Exception as e:
                    logger.error(f"Failed to start host actor: {e}")
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass

    def _remove_members_from_actor_pool(self, desired_num_actors: int) -> None:
        """Remove actors to reach the desired pool size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        while len(self._actor_pool) > desired_num_actors:
            member = self._actor_pool.pop()
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                try:
                    ray.get(member.actor.shutdown.remote(), timeout=5)
                except Exception:
                    pass
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Removed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

    def _scale_actor_pool(self, desired_num_actors: int) -> None:
        """Scale the actor pool to the desired size.

        First removes unhealthy actors, then adds or removes actors
        as needed to reach the target size.

        Args:
            desired_num_actors: Target number of actors in the pool.
        """
        self._remove_unhealthy_members_from_actor_pool()
        current = len(self._actor_pool)
        if current < desired_num_actors:
            self._add_members_to_actor_pool(desired_num_actors)
        elif current > desired_num_actors:
            self._remove_members_from_actor_pool(desired_num_actors)

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        if not self._actor_pool:
            return

        shutdown_refs = []
        for member in self._actor_pool:
            try:
                shutdown_refs.append(member.actor.shutdown.remote())
            except Exception:
                pass

        try:
            ray.wait(shutdown_refs, num_returns=len(shutdown_refs), timeout=5.0)
        except Exception:
            pass

        for member in self._actor_pool:
            name = self.get_actor_name_from_actor_info(member.actor_info)
            try:
                ray.kill(member.actor, no_restart=True)
                logger.info(f"Killed actor {name}")
            except Exception as e:
                logger.error(f"Failed to kill actor {name}: {e}")

        self._actor_pool = []

    def _await_all_hosts_healthy(self, timeout_s: int = 60, poll_s: float = 2.0) -> bool:
        """Wait for all hosts in the pool to become healthy.

        Polls the health status of all host actors until they all report
        healthy or the timeout is reached.

        Args:
            timeout_s: Maximum time to wait for hosts to become healthy (default: 60).
            poll_s: Interval between health checks in seconds (default: 2.0).

        Returns:
            True if all hosts became healthy within the timeout, False otherwise.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            statuses = ray.get([m.actor.healthy.remote() for m in self._actor_pool])
            if all(statuses):
                return True
            time.sleep(poll_s)
        return False

    def run_remote_fn(
        self,
        remote_fn,
        runtime_env: dict | None = None,
        env: dict | None = None,
        f_args: tuple = (),
        f_kwargs: dict | None = None,
    ):
        """Execute a remote function on all hosts in this slice.

        Ensures all hosts are ready, then launches the function on each host
        in parallel. The function runs with TPU resources reserved. This is
        the primary method used by RayExecutor.autoscale_execute to run
        workloads across the slice.

        Args:
            remote_fn: Ray remote function or callable to execute. Will be
                executed once per host in the slice.
            runtime_env: Optional Ray runtime environment configuration for
                dependencies and environment setup.
            env: Optional environment variables to set on all hosts.

        Returns:
            List[ray.ObjectRef]: One ObjectRef per host in the slice,
                ordered by host_id. Results can be retrieved with ray.get().

        Raises:
            RuntimeError: If slice info is not initialized.

        Note:
            - Automatically ensures host pool is at full capacity.
            - Each host runs the function with proper TPU resource allocation.
            - Functions run in parallel across all hosts.
            - Environment variables include TPU_HOST_ID and TPU_SLICE_NAME.
        """
        if not self._slice_info:
            raise RuntimeError("Slice info not initialized")
        self.ensure_host_pool(self._slice_info.num_hosts)

        try:
            self._await_all_hosts_healthy(timeout_s=int(os.getenv("EFORMER_HOST_HEALTH_WAIT_S", "60")))
        except Exception:
            pass

        futures = [
            member.actor.run_remote_fn.remote(
                remote_fn,
                f_args=f_args,
                f_kwargs=f_kwargs,
                runtime_env=runtime_env,
                env=env,
            )
            for member in self._actor_pool
        ]
        return futures

    def shutdown(self):
        """Gracefully shut down this slice actor.

        Removes the placement group, marks the slice as failed,
        and prevents any new task execution on this slice.
        """
        try:
            self.drain_actor_pool()
        except Exception:
            pass
        if self._host_placement_group:
            try:
                remove_placement_group(self._host_placement_group)
            except Exception:
                pass
            self._host_placement_group = None
        self._failed = True


class SlicePoolManager(ResourcePoolManager[SliceInfo]):
    """Manager for multiple TPU slices in multi-slice configurations.

    Coordinates multiple SliceActors to manage multi-slice TPU configurations.
    Handles scaling, health monitoring, and distributed task execution across
    multiple TPU slices. This is the top-level manager used by RayExecutor
    for multi-slice workloads.

    Attributes:
        _tpu_type: Type of TPU (e.g., "v4-8", "v5e-16").
        _last_scale_ts: Timestamp of last scaling operation for rate limiting.
        _last_scale_check_ts: Timestamp of last scale check.
        _actor_pool: List of SliceActor pool members.

    Hierarchy:
        SlicePoolManager -> SliceActors -> DeviceHostActors -> Tasks

    Resource Requirements:
        - Each SliceActor requires a TPU-{type}-head resource.
        - Each slice requires placement group bundles for host distribution.
        - Automatically requests resources from Ray autoscaler.
    """

    def __init__(self, tpu_type: str | None):
        """Initialize a slice pool manager.

        Args:
            tpu_type: Type of TPU to manage (e.g., "v4-8", "v5e-16").
                     Used for resource labeling and identification.
        """
        super().__init__()
        self._tpu_type = tpu_type
        self._last_scale_ts: float | None = None
        self._last_scale_check_ts: float | None = None
        self._head_pg = None
        self._head_pg_target = 0

    def get_actor_pool_name(self) -> str:
        """Get a human-readable name for this actor pool.

        Returns:
            String identifier for the pool, defaults to class name.
        """
        return f"SlicePool({self._tpu_type})"

    def get_actor_name_from_actor_info(self, actor_info: SliceInfo) -> str:
        """Generate a human-readable name from actor info.

        Args:
            actor_info: Metadata about the actor.

        Returns:
            String representation of the actor for logging.
        """
        return actor_info.slice_name

    def create_actor(self) -> ActorHandle:
        """Create a new SliceActor to manage a TPU slice.

        Creates a SliceActor with appropriate resource requirements
        based on the TPU type. The actor will manage all hosts within
        its assigned slice.

        Returns:
            Ray actor handle for the newly created SliceActor.
        """
        return SliceActor.options(num_cpus=0, resources={f"TPU-{self._tpu_type}-head": 1}).remote()

    def scale_multislice(self, num_slices: int | Sequence[int]) -> None:
        """Scale the pool to the desired number of slices.

        Supports flexible scaling with multiple valid sizes. Will scale
        to the largest feasible size from the provided options. This method
        is typically called by RayExecutor.autoscale_execute to set up
        the required number of slices.

        Args:
            num_slices: Target number of slices or list of valid sizes.
                If int: exact number of slices required.
                If sequence: will try largest first, falling back to smaller.

        Raises:
            ValueError: If target is invalid or empty list provided.
            InsufficientSlicesError: If none of the requested sizes can be achieved.

        Example:
            >>> manager.scale_multislice(4)
            >>> manager.scale_multislice([2, 4, 8])

        Note:
            - Requests TPU head resources from Ray autoscaler.
            - Removes unhealthy actors before scaling.
            - Falls back to smaller sizes if larger ones unavailable.
        """
        self._last_scale_ts = time.time()

        if isinstance(num_slices, int):
            valid = [int(num_slices)]
        else:
            valid = sorted({int(x) for x in num_slices})
            if not valid:
                raise ValueError("valid sizes list is empty")

        target = valid[-1]
        if target <= 0:
            raise ValueError(f"Target slice count must be > 0, got {target}")

        head_bundles = [{"CPU": 0, f"TPU-{self._tpu_type}-head": 1} for _ in range(target)]
        request_resources(bundles=head_bundles)

        self._scale_actor_pool(target)
        current = len(self._actor_pool)
        if current not in valid:
            feasible = [v for v in valid if v <= current]
            if not feasible:
                raise InsufficientSlicesError(f"Requested one of {valid}, but only {current} slices available")
            self._scale_actor_pool(feasible[-1])

    def prepare_all_slices(self) -> None:
        """Prepare all slices by ensuring host placement groups.

        Pre-requests resources for all slices and prepares their host
        placement groups for distributed execution. This ensures that
        all nodes are ready before task execution begins.

        This method:
        1. Fetches slice information from all SliceActors.
        2. Requests host resources for each slice from autoscaler.
        3. Creates placement groups with STRICT_SPREAD strategy.
        4. Ensures all hosts are discovered and ready.

        Note:
            Called automatically by autoscale_execute before running tasks.
            Essential for proper multi-host coordination within each slice.
        """

        if os.getenv("EFORMER_SAFE_GATHER", "1") == "1":
            slice_infos = []
            good_members = []
            for m in self._actor_pool:
                try:
                    si = ray.get(m.actor.get_info.remote(), timeout=30)
                    slice_infos.append(si)
                    good_members.append(m)
                except Exception as e:
                    try:
                        ray.kill(m.actor, no_restart=True)
                    except Exception:
                        pass
                    logger.warning(f"Pruned dead SliceActor during prepare_all_slices: {e}")

            if len(good_members) != len(self._actor_pool):
                self._actor_pool = good_members
                if not self._actor_pool:
                    raise RuntimeError("No SliceActors available after pruning.")

            all_bundles = []
            for info in slice_infos:
                all_bundles.extend([{"CPU": 0, info.slice_name: 1}] * info.num_hosts)
            if all_bundles:
                request_resources(bundles=all_bundles)

            ray.get([m.actor.prepare_hosts.remote() for m in self._actor_pool])
        else:
            slice_infos: list[SliceInfo] = ray.get([m.actor.get_info.remote() for m in self._actor_pool])
            all_bundles = []

            for info in slice_infos:
                all_bundles.extend([{"CPU": 0, info.slice_name: 1}] * info.num_hosts)
            if all_bundles:
                request_resources(bundles=all_bundles)

            ray.get([m.actor.prepare_hosts.remote() for m in self._actor_pool])

    def should_scale_up_multislice(self, valid_sizes: Sequence[int]) -> bool:
        """Check if pool should scale up to a larger size.

        Implements rate limiting to prevent frequent scaling operations.

        Args:
            valid_sizes: List of valid pool sizes.

        Returns:
            True if scaling up is recommended.
        """
        self._last_scale_check_ts = time.time()
        current = len(self._actor_pool)
        larger = [size for size in valid_sizes if size > current]
        if not larger:
            return False
        if self._last_scale_ts and (time.time() - self._last_scale_ts) < 60:
            return False
        return True

    def execute_on_each_slice(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Execute a function on all hosts across all slices.

        Prepares all slices and runs the function on every host in parallel.
        Returns results grouped by slice.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            List of lists where outer list represents slices and inner lists
            contain ObjectRefs for each host in that slice.
        """
        self.prepare_all_slices()
        per_slice_futures = ray.get(
            [m.actor.run_remote_fn.remote(remote_fn, runtime_env=runtime_env, env=env) for m in self._actor_pool]
        )
        return per_slice_futures

    def execute_on_each_host_flat(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Execute a function on all hosts, returning a flat list.

        Similar to execute_on_each_slice but flattens the nested result
        structure into a single list of ObjectRefs.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            Flat list of ObjectRefs from all hosts across all slices.
        """
        per_slice = self.execute_on_each_slice(remote_fn, env=env, runtime_env=runtime_env)
        return [f for sub in per_slice for f in sub]

    def execute_on_each_host(self, fn, *args, env: dict | None = None, **kwargs):
        """Execute a function on each host across all slices.

        Prepares all slices and runs the function on every host actor
        in parallel across all slices.

        Args:
            fn: Function to execute on each host.
            *args: Positional arguments for fn.
            env: Optional environment variables.
            **kwargs: Keyword arguments for fn.

        Returns:
            Nested list of results (outer list for slices, inner for hosts).
        """
        self.prepare_all_slices()
        ray.get([member.actor.ensure_host_pool.remote() for member in self._actor_pool])

        @ray.remote(max_calls=1)
        def _runner():
            return fn(*args, **kwargs)

        return ray.get([member.actor.run_remote_fn.remote(_runner, env=env) for member in self._actor_pool])

    def schedule_on_each_host(self, remote_fn, env: dict | None = None, runtime_env: dict | None = None):
        """Schedule a function on all hosts without waiting for results.

        Ensures all slices and hosts are ready, then schedules the function
        execution on all hosts. Returns immediately with ObjectRefs.

        Args:
            remote_fn: Ray remote function or callable to execute.
            env: Optional environment variables to set.
            runtime_env: Optional Ray runtime environment configuration.

        Returns:
            Flat list of ObjectRefs that can be waited on later.
        """
        self.prepare_all_slices()
        ray.get([m.actor.ensure_host_pool.remote() for m in self._actor_pool])
        per_slice_futures = ray.get(
            [m.actor.run_remote_fn.remote(remote_fn, runtime_env=runtime_env, env=env) for m in self._actor_pool]
        )
        return [f for sl in per_slice_futures for f in sl]

    def _add_members_to_actor_pool(self, desired_num_actors: int) -> None:
        """Add new SliceActors to the pool with timeout and retry logic.

        Creates and starts SliceActors asynchronously, waiting for them to
        initialize with configurable timeout. Implements a polling strategy
        that periodically nudges the Ray autoscaler to provision resources.

        This method differs from the base class implementation by:
        - Creating all actors immediately (non-blocking)
        - Polling with timeout instead of blocking wait
        - Periodically requesting resources from autoscaler
        - Processing actors as they become ready

        Args:
            desired_num_actors: Target number of SliceActors in the pool.

        Behavior:
            1. Creates actor handles immediately (scheduling is deferred)
            2. Polls every SCALE_POLL_S seconds for actors to start
            3. Nudges autoscaler each poll to request TPU head resources
            4. Processes actors as they complete initialization
            5. Kills actors that don't start within SCALE_ADD_TIMEOUT_S

        Environment Variables:
            EFORMER_SCALE_POLL_S: Poll interval in seconds (default: 30)
            EFORMER_SCALE_ADD_TIMEOUT_S: Total timeout in seconds (default: 604800/7 days)

        Note:
            - Actors are added to pool as soon as they're ready
            - Partial success is supported (some actors may start)
            - Unstarted actors are killed after timeout
            - Progress is logged throughout the scaling process
        """
        current = len(self._actor_pool)

        if current >= desired_num_actors:
            return

        if current != 0:
            logger.info("Recreating head PG due to non-zero current pool (align bundles with slots)")
            self.drain_actor_pool()
            current = 0

        logger.info(f"Scaling up pool {self.get_actor_pool_name()} from {current} to {desired_num_actors}")

        self._ensure_head_pg(desired_num_actors)

        slot_to_actor: dict[int, ActorHandle] = {}
        slot_to_info_ref: dict[int, ray.ObjectRef] = {}

        def _start_for_slot(slot: int):
            actor = SliceActor.options(
                num_cpus=0,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    self._head_pg,
                    placement_group_bundle_index=slot,
                    placement_group_capture_child_tasks=False,
                ),
            ).remote()
            slot_to_actor[slot] = actor
            slot_to_info_ref[slot] = actor.get_info.remote()

        for slot in range(current, desired_num_actors):
            _start_for_slot(slot)

        deadline = time.time() + SCALE_ADD_TIMEOUT_S
        started = 0

        while slot_to_info_ref and time.time() < deadline:
            remaining = len(slot_to_info_ref)
            head_bundles = [{"CPU": 0, f"TPU-{self._tpu_type}-head": 1} for _ in range(remaining)]
            try:
                request_resources(bundles=head_bundles)
            except Exception:
                pass

            pending_refs = list(slot_to_info_ref.values())
            done, _ = ray.wait(pending_refs, num_returns=1, timeout=SCALE_POLL_S)
            if not done:
                continue

            for ref in done:
                slot = next((s for s, r in slot_to_info_ref.items() if r == ref), None)
                if slot is None:
                    continue
                actor = slot_to_actor.get(slot)

                slot_to_info_ref.pop(slot, None)

                try:
                    info = ray.get(ref, timeout=0)

                    self._actor_pool.append(ActorPoolMember(actor, info))
                    started += 1
                    logger.info(f"Added actor {self.get_actor_name_from_actor_info(info)} (slot {slot})")
                except Exception as e:
                    logger.warning(f"SliceActor for slot {slot} failed to start: {e}; killing and re-queuing")
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
                    _start_for_slot(slot)

            logger.info(f"Started {started}/{desired_num_actors - current} slice actors so far")

        if slot_to_info_ref:
            for _, actor in slot_to_actor.items():
                try:
                    if not any(m.actor == actor for m in self._actor_pool):
                        ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            logger.info(f"Started {started}/{desired_num_actors - current} slice actors (timed out for the rest)")

    def _ensure_head_pg(self, desired_num_actors: int) -> None:
        """Ensure head placement group exists with the correct number of bundles.

        Creates or recreates the head placement group to match the desired number of actors.
        The placement group uses STRICT_SPREAD strategy to ensure SliceActors are distributed
        across different nodes for optimal resource utilization and fault tolerance.

        Each bundle in the placement group reserves one TPU head resource for a SliceActor.
        If the current placement group doesn't match the target size, it's destroyed and
        recreated with the correct number of bundles.

        Args:
            desired_num_actors: Target number of SliceActors (and placement group bundles)

        Side Effects:
            - Requests resources from Ray autoscaler for the new placement group
            - Destroys existing placement group if size mismatch
            - Updates _head_pg and _head_pg_target instance variables
            - Blocks until placement group is ready

        Note:
            The placement group uses TPU-{tpu_type}-head resource labels for each bundle
            to ensure proper resource allocation by the Ray autoscaler.
        """
        label = f"TPU-{self._tpu_type}-head"
        if self._head_pg and self._head_pg_target == desired_num_actors:
            return
        if self._head_pg:
            try:
                remove_placement_group(self._head_pg)
            except Exception:
                pass
            self._head_pg = None
            self._head_pg_target = 0

        bundles = [{"CPU": 0, label: 1} for _ in range(desired_num_actors)]
        request_resources(bundles=bundles)
        self._head_pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(self._head_pg.ready())
        self._head_pg_target = desired_num_actors

    def _destroy_head_pg(self) -> None:
        """Destroy the current head placement group and reset tracking variables.

        Removes the head placement group from Ray's cluster state and resets the
        instance variables that track placement group state. This is typically
        called during pool shutdown or when recreating placement groups with
        different sizes.

        The method uses best-effort cleanup - if the placement group removal fails
        (e.g., due to Ray cluster issues), the error is logged but not propagated
        to avoid disrupting the overall shutdown process.

        Side Effects:
            - Removes placement group from Ray cluster
            - Sets _head_pg to None
            - Resets _head_pg_target to 0
            - Frees up cluster resources reserved by the placement group

        Note:
            This is a cleanup operation that should be called when the placement
            group is no longer needed, such as during pool draining or before
            recreating with a different size.
        """
        if self._head_pg:
            try:
                remove_placement_group(self._head_pg)
            except Exception:
                pass
            self._head_pg = None
            self._head_pg_target = 0

    def drain_actor_pool(self) -> None:
        """Shut down and remove all actors from the pool.

        Attempts graceful shutdown first, then forcefully kills actors.
        Clears the actor pool after draining.
        """
        super().drain_actor_pool()
        self._destroy_head_pg()
