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


"""Cluster utilities for distributed execution with Ray and JAX.

This module provides utilities for managing distributed clusters, particularly
focused on SLURM environments and Ray cluster initialization. It handles
automatic discovery of cluster topology, coordinator selection, and proper
resource allocation across distributed nodes.

Note:
    This implementation is adapted from the Levanter project
    (https://github.com/stanford-crfm/levanter/blob/main/src/levanter/distributed.py)
    with modifications for the eFormer/EasyDeL framework.
"""

import atexit
import itertools
import logging
import os
import re
import socket
import subprocess
from dataclasses import dataclass

import jax
import ray
from jax._src import clusters, distributed

logger = logging.getLogger("eray-executor")


_JOBID_PARAM = "SLURM_JOB_ID"
_NODE_LIST_CHOICES = ["SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"]
_TASKS_PER_NODE = "SLURM_STEP_TASKS_PER_NODE"
_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_NODE_NAME = "SLURMD_NODENAME"


class eSlurmCluster(clusters.SlurmCluster):
    """Extended SLURM cluster implementation for Ray executor.

    This class extends JAX's SlurmCluster to provide additional functionality
    for Ray-based distributed execution in SLURM environments. It handles
    automatic coordinator address discovery, device ID assignment, and
    local process counting.

    Attributes:
        Inherits all attributes from jax.clusters.SlurmCluster.
    """

    @classmethod
    def get_coordinator_address(cls) -> str:
        """Get the coordinator address for the SLURM cluster.

        Automatically determines the coordinator node and port based on
        SLURM environment variables. The coordinator is typically the
        first node in the node list.

        Returns:
            str: The coordinator address in the format "hostname:port".

        Raises:
            ValueError: If node list cannot be found in environment variables.
        """
        _id = os.environ[_JOBID_PARAM]
        port = _choose_port(_id)
        node_list = eSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )
        delims = {",", "["}
        ind = next((i for i, ch in enumerate(node_list) if ch in delims), len(node_list))
        if ind == len(node_list) or node_list[ind] == ",":
            return f"{node_list[:ind]}:{port}"
        else:
            prefix = node_list[:ind]
            suffix = node_list[ind + 1 :]
            delims2 = {",", "-"}
            ind2 = next((i for i, ch in enumerate(suffix) if ch in delims2), None)
            return f"{prefix}{suffix[:ind2]}:{port}"

    @classmethod
    def _node_list(cls):
        """Get the SLURM node list from environment variables.

        Searches through multiple possible SLURM environment variables
        to find the node list.

        Returns:
            str | None: The node list string if found, None otherwise.
        """
        return next((os.environ[o] for o in _NODE_LIST_CHOICES if o in os.environ), None)

    @classmethod
    def get_local_device_ids_for_process(cls) -> list[int] | None:
        """Get the device IDs assigned to the current local process.

        Determines which CUDA devices should be used by this process based
        on the SLURM task configuration and CUDA_VISIBLE_DEVICES.

        Returns:
            list[int] | None: List of device IDs for this process, or None
                if device assignment cannot be determined.

        Raises:
            ValueError: If the number of visible devices is not evenly
                divisible by the number of local tasks.
        """
        local_process_id = cls.get_local_process_id()

        if local_process_id is None:
            return None

        if _VISIBLE_DEVICES not in os.environ:
            return None

        local_process_count = cls._infer_local_process_count()

        all_visible_devices = [int(x) for x in os.environ[_VISIBLE_DEVICES].split(",")]

        if len(all_visible_devices) % local_process_count != 0:
            raise ValueError(
                f"Number of visible devices ({len(all_visible_devices)}) is not divisible by the number "
                f"of local tasks ({local_process_count})"
            )

        num_devices_per_local_process = len(all_visible_devices) // local_process_count

        begin = local_process_id * num_devices_per_local_process
        return all_visible_devices[begin : begin + num_devices_per_local_process]

    @classmethod
    def _infer_local_process_count(cls):
        """Infer the number of processes on the local node.

        Parses SLURM environment variables to determine how many tasks
        are running on the current node.

        Returns:
            int: The number of local processes/tasks.

        Raises:
            ValueError: If node list cannot be found in environment variables.
        """
        node_list = eSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )

        node_list = _square_brace_expand(node_list)
        local_node = os.environ[_NODE_NAME]
        local_node_index = node_list.index(local_node)
        unrolled_tasks_per_node = []
        multi_match = re.compile(r"(\d+)\(x(\d+)\)")
        for x in os.environ[_TASKS_PER_NODE].split(","):
            match = multi_match.match(x)
            if match:
                unrolled_tasks_per_node.extend([int(match.group(1))] * int(match.group(2)))
            else:
                unrolled_tasks_per_node.append(int(x))

        tasks_on_local_node = unrolled_tasks_per_node[local_node_index]
        return tasks_on_local_node


def _square_brace_expand(node_list):
    """Expand SLURM node list notation with square brackets.

    Expands compressed SLURM node list notation (e.g., "node[01-03,05]")
    into a full list of node names.

    Args:
        node_list (str): Compressed node list string.

    Returns:
        list[str]: Expanded list of individual node names.

    Example:
        >>> _square_brace_expand("node[01-03,05]")
        ['node01', 'node02', 'node03', 'node05']
    """
    parts = re.findall(r"(\[.*?\]|[^\[\]]+)", node_list)

    def generate_numbers(number_string):
        if "-" in number_string:
            start, end = map(int, number_string.split("-"))
            return [str(i).zfill(len(number_string.split("-")[0])) for i in range(start, end + 1)]
        else:
            return [number_string]

    processed_parts = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            number_sequences = part.strip("[]").split(",")
            processed_parts.append(
                list(itertools.chain.from_iterable(generate_numbers(seq) for seq in number_sequences))
            )
        else:
            processed_parts.append([part])

    expanded_nodes = ["".join(combination) for combination in itertools.product(*processed_parts)]

    return expanded_nodes


def logical_cpu_core_count():
    """Get the number of logical CPU cores available to the process.

    First checks SLURM environment variables, then falls back to OS CPU count.

    Returns:
        int: Number of logical CPU cores available.
    """
    num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
    if num_cpus is not None:
        return int(num_cpus)

    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1


def _remove_if_possible(path):
    """Remove a file if possible, ignoring errors.

    Args:
        path (str): Path to the file to remove.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def _touch(file_path):
    """Create or update the timestamp of a file.

    Args:
        file_path (str): Path to the file to touch.
    """
    with open(file_path, "a"):
        os.utime(file_path, None)


def _choose_port(_id):
    """Choose a port number based on a job ID.

    Generates a deterministic port number in the range 53248-65535
    based on the provided ID.

    Args:
        _id (str | int): Job or process ID.

    Returns:
        int: Port number.
    """
    port = int(_id) % 2**12 + (65535 - 2**12 + 1)
    return port


def _is_this_machine(host):
    """Check if the given host identifies this machine.

    Determines whether a hostname or IP address refers to the current machine.

    Args:
        host (str): Hostname or IP address to check.

    Returns:
        bool: True if the host refers to this machine, False otherwise.
    """
    if host == "localhost" or host == "0.0.0.0":
        return True
    try:
        machine_ips = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]
        host_ip = socket.gethostbyname(host)
    except socket.gaierror:
        return False
    return any(host_ip == machine_ip for machine_ip in machine_ips)


def _is_local_leader():
    """Determine if this process is the local leader.

    Uses file locking to elect a single leader process among all processes
    running on the same node. The leader is responsible for starting the
    Ray head node.

    Returns:
        bool: True if this process is the local leader, False otherwise.
    """
    import atexit

    import filelock
    from jax.experimental.multihost_utils import broadcast_one_to_all

    if jax.process_count() == 1:
        return True

    import random

    random_id = random.randint(0, 1000000)
    random_id = broadcast_one_to_all(random_id)

    lock = filelock.FileLock(f"/tmp/eray_executor_local_process_zero_lock.{random_id}")
    action_performed_file = f"/tmp/eray_executor_local_process_zero_action_performed.{random_id}"

    try:
        with lock.acquire(timeout=0.1):
            if not os.path.exists(action_performed_file):
                _touch(action_performed_file)
                atexit.register(_remove_if_possible, lock.lock_file)
                atexit.register(_remove_if_possible, action_performed_file)
                return True
            return False
    except filelock.Timeout:
        return False


_already_initialized = False


def auto_ray_cluster(
    address: str | None = None,
    namespace: str | None = "eray-executor",
    start_workers: bool = True,
    fail_if_cluster_already_initialized: bool = False,
    **kwargs,
):
    """Automatically initialize a Ray cluster.

    Handles automatic discovery and initialization of Ray clusters in various
    environments. Can start both head and worker nodes as needed, with special
    support for SLURM clusters.

    Args:
        address (str | None): Ray cluster address. If None, attempts auto-discovery
            from RAY_ADDRESS environment variable or JAX coordinator.
        namespace (str | None): Ray namespace to use. Defaults to "eray-executor".
        start_workers (bool): Whether to start Ray workers on non-head nodes.
            Defaults to True.
        fail_if_cluster_already_initialized (bool): Whether to fail if a cluster
            is already running. Defaults to False.
        **kwargs: Additional arguments passed to ray.init().

    Raises:
        RuntimeError: If Ray head or worker fails to start.

    Note:
        This function can only be called once per process. Subsequent calls
        will be ignored with a warning.
    """
    global _already_initialized

    if _already_initialized:
        logger.warning("auto_ray_cluster has already been called. Ignoring subsequent calls.")
        return

    def _munge_address_port(address: str):
        host, port_str = address.split(":")
        port = int(port_str)
        return host, port

    if address is None:
        if os.getenv("RAY_ADDRESS") is not None:
            address = os.getenv("RAY_ADDRESS")
            logger.info("Auto-discovered ray address using RAY_ADDRESS: %s", address)
        else:
            coord_address = getattr(distributed.global_state, "coordinator_address", None)
            if coord_address is None:
                logger.info("No auto-discovered ray address found. Using ray.init('local').")
                address = "local"
            else:
                logger.info(f"Auto-discovered ray address using JAX coordinator address: {coord_address}")
                host, port = _munge_address_port(coord_address)

                ray_port = _choose_port(port + 240)
                address = f"{host}:{ray_port}"
                num_cpus = logical_cpu_core_count()

                if _is_local_leader():
                    if _is_this_machine(host):
                        logger.info(f"Starting ray head on port {ray_port}. We are process the coordinator {host}.")
                        logger.info(f"Starting ray head with num_cpus set to {num_cpus}.")
                        ret = subprocess.run(
                            [
                                "ray",
                                "start",
                                "--head",
                                "--port",
                                str(ray_port),
                                "--num-cpus",
                                str(num_cpus),
                                "--dashboard-host=0.0.0.0",
                            ],
                        ).returncode
                        if ret != 0:
                            if not fail_if_cluster_already_initialized:
                                logger.warning(
                                    f"Failed to start ray head with exit code {ret}. Checking if we can connect to"
                                    " the head..."
                                )
                                ret = subprocess.run(["ray", "status"]).returncode
                                if ret != 0:
                                    raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                                else:
                                    logger.info(f"Ray head already running on port {ray_port}. Connecting to it.")
                            else:
                                raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                        else:
                            logger.info(f"Successfully started ray head on port {ray_port}.")

                        atexit.register(
                            lambda: subprocess.run(
                                ["ray", "stop", "-g", "10", "--force"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        )
                    elif start_workers:
                        logger.info(
                            f"Starting ray worker and connecting to {address}. We are process {jax.process_index()}."
                        )
                        logger.info(f"Starting ray worker with num_cpus set to {num_cpus}.")
                        ret = subprocess.run(
                            ["ray", "start", "--address", address, "--num-cpus", str(num_cpus)],
                        ).returncode
                        if ret != 0:
                            raise RuntimeError(f"Failed to start ray head with exit code {ret}")
                        else:
                            logger.info(f"Successfully started ray worker and connected to {address}.")

    logger.info(f"ray.init(address={address!r}, namespace={namespace!r}, **{kwargs!r})")

    for i in range(0, 5):
        try:
            ray.init(address=address, namespace=namespace, **kwargs)
            break
        except Exception as e:
            if i == 4:
                raise e
            else:
                logger.warning(f"Failed to initialize ray with address {address}. Retrying...")
                continue

    def do_shutdown():
        logger.info("Shutting down ray...")
        ray.shutdown()

    atexit.register(do_shutdown)
    _already_initialized = True


@dataclass(frozen=True)
class DistributedConfig:
    """Configuration for distributed JAX execution.

    Encapsulates all settings needed to initialize JAX in a distributed
    environment, with automatic detection of SLURM clusters.

    Attributes:
        coordinator_address (str | None): Address of the coordinator process.
            Auto-detected from SLURM if None.
        num_processes (int | None): Total number of processes in the cluster.
        process_id (int | None): ID of this process (0-indexed).
        local_device_ids (int | list[int] | None): Device IDs to use on this node.
            Auto-detected from SLURM if None.
    """

    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: int | list[int] | None = None

    def _is_distributed(self):
        """Check if this is a distributed configuration.

        Returns:
            bool: True if any distributed settings are specified or a
                cluster environment is detected.
        """
        if (
            (self.coordinator_address is not None)
            or (self.num_processes is not None)
            or (self.process_id is not None)
            or (self.local_device_ids is not None)
        ):
            return True

        if any(env.is_env_present() for env in clusters.ClusterEnv._cluster_types):
            return True

        return False

    def initialize(self):
        """Initialize JAX distributed execution.

        Sets up JAX for distributed execution based on the configuration,
        with automatic detection of SLURM cluster settings if needed.

        Note:
            If no distributed configuration is provided and no cluster is
            detected, JAX will run in single-process mode.
        """
        if self._is_distributed():
            device_ids = self.local_device_ids
            coordinator_address = self.coordinator_address

            if eSlurmCluster.is_env_present():
                if device_ids is None:
                    device_ids = eSlurmCluster.get_local_device_ids_for_process()

                if coordinator_address is None:
                    coordinator_address = eSlurmCluster.get_coordinator_address()

            jax.distributed.initialize(
                coordinator_address,
                self.num_processes,
                self.process_id,
                device_ids,
                initialization_timeout=30 * 60,
            )
            logger.info(
                f"Initialized jax.distributed with {jax.device_count()} devices, {jax.process_count()} processes,"
                f" coordinator_address={coordinator_address}, process_id={self.process_id}, my"
                f" device_ids={device_ids}."
            )
        else:
            logger.info(
                "Not initializing jax.distributed because no distributed config "
                "was provided, and no cluster was detected."
            )


@dataclass
class RayClusterConfig:
    """Configuration for Ray cluster initialization.

    Controls how Ray clusters are started and connected to.

    Attributes:
        address (str | None): Ray cluster address. If None, uses auto-discovery.
        start_workers (bool): Whether to start Ray workers on non-head nodes.
            Defaults to True.
        auto_start_cluster (bool): Whether to automatically start the Ray cluster.
            Defaults to True.
    """

    address: str | None = None
    start_workers: bool = True
    auto_start_cluster: bool = True

    def initialize(self):
        """Initialize the Ray cluster based on configuration.

        Calls auto_ray_cluster() with the configured settings if
        auto_start_cluster is True.
        """
        if self.auto_start_cluster:
            auto_ray_cluster(address=self.address, start_workers=self.start_workers)
