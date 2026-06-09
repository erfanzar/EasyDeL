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


from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import typing as tp
from functools import wraps
from typing import Any

import requests
import yaml

TPU_VERSION = os.getenv("TPU_VERSION", "v4")
TPU_SLICE_SIZE = int(os.getenv("TPU_SLICE_SIZE", "8"))
TPU_CORES_PER_HOST = int(os.getenv("TPU_CORES_PER_HOST", "4"))
SSH_USER = os.getenv("PATCHER_USER")
INTERNAL_IPS = ["0.0.0.0"]
EXTERNAL_IPS = ["0.0.0.0"]


COLORS: dict[str, str] = {
    "PURPLE": "\033[95m",
    "BLUE": "\033[94m",
    "CYAN": "\033[96m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "ORANGE": "\033[38;5;208m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
    "RESET": "\033[0m",
    "BLUE_PURPLE": "\033[38;5;99m",
}

LEVEL_COLORS: dict[str, str] = {
    "DEBUG": COLORS["ORANGE"],
    "INFO": COLORS["BLUE_PURPLE"],
    "WARNING": COLORS["YELLOW"],
    "ERROR": COLORS["RED"],
    "CRITICAL": COLORS["RED"] + COLORS["BOLD"],
    "FATAL": COLORS["RED"] + COLORS["BOLD"],
}

_LOGGING_LEVELS: dict[str, int] = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
    "critical": 50,
    "fatal": 50,
    "error": 40,
    "warning": 30,
    "warn": 30,
    "info": 20,
    "debug": 10,
    "notset": 0,
}


def _resources_json(d: dict) -> str:
    """Convert a dictionary to a compact JSON string for Ray resource specification.

    Args:
        d: Dictionary containing resource key-value pairs.

    Returns:
        str: JSON string with no extra whitespace, suitable for command-line arguments.

    Example:
        >>> _resources_json({"TPU": 4, "CPU": 8})
        '{"TPU":4,"CPU":8}'
    """
    return json.dumps(d, separators=(",", ":"))


class ColorFormatter(logging.Formatter):
    """Custom log formatter that adds color coding to log messages based on level.

    Formats log messages with ANSI color codes to make different log levels
    visually distinct in terminal output. Colors are defined in the LEVEL_COLORS
    dictionary and include support for DEBUG, INFO, WARNING, ERROR, and CRITICAL levels.

    Attributes:
        Inherits all attributes from logging.Formatter.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(ColorFormatter())
        >>> logger.addHandler(handler)
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with color coding.

        Args:
            record: The log record to format.

        Returns:
            str: Formatted log message with ANSI color codes.
        """
        orig_levelname = record.levelname
        color = LEVEL_COLORS.get(record.levelname, COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{COLORS['RESET']}"
        current_time = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        formatted_name = f"{color}({current_time} {record.name}){COLORS['RESET']}"
        message = f"{formatted_name} {record.getMessage()}"
        record.levelname = orig_levelname
        return message


class LazyLogger:
    """A lazily-initialized logger that defers setup until first use.

    This class provides a logger that is not initialized until the first
    log message is emitted, reducing startup overhead when logging may
    not be used. It uses ColorFormatter for colorized terminal output.

    Attributes:
        _name: Name for the logger.
        _level: Logging level (defaults to LOGGING_LEVEL_ED environment variable).
        _logger: The underlying logging.Logger instance, initialized on first use.

    Example:
        >>> logger = LazyLogger("my_module")
        >>> logger.info("This message triggers initialization")
    """

    def __init__(self, name: str, level: int | None = None):
        """Initialize a LazyLogger.

        Args:
            name: Name for the logger, typically the module name.
            level: Optional logging level. Defaults to LOGGING_LEVEL_ED
                environment variable or INFO if not set.
        """
        self._name = name
        self._level = level or _LOGGING_LEVELS[os.getenv("LOGGING_LEVEL_ED", "INFO")]
        self._logger: logging.Logger | None = None

    def _ensure_initialized(self) -> None:
        """Initialize the underlying logger if not already done.

        Creates a logging.Logger with a StreamHandler and ColorFormatter.
        Disables propagation to avoid duplicate messages.
        """
        if self._logger is not None:
            return

        logger = logging.getLogger(self._name)
        logger.propagate = False
        logger.setLevel(self._level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self._level)
        formatter = ColorFormatter()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self._logger = logger

    def __getattr__(self, name: str) -> tp.Callable:
        """Proxy attribute access to the underlying logger.

        Intercepts calls to logging methods (debug, info, warning, etc.)
        and ensures the logger is initialized before delegating.

        Args:
            name: Name of the attribute being accessed.

        Returns:
            Callable: A wrapped logging method.

        Raises:
            AttributeError: If the attribute is not a valid logging method.
        """
        if name in _LOGGING_LEVELS or name.upper() in _LOGGING_LEVELS or name in ("exception", "log"):

            @wraps(getattr(logging.Logger, name))
            def wrapped_log_method(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
                self._ensure_initialized()
                return getattr(self._logger, name)(*args, **kwargs)

            return wrapped_log_method
        raise AttributeError(f"'LazyLogger' object has no attribute '{name}'")


def get_logger(name: str, level: int | None = None) -> LazyLogger:
    """Create a LazyLogger instance with the specified name and level.

    Factory function for creating LazyLogger instances. Provides a simple
    interface for obtaining loggers throughout the application.

    Args:
        name: Name for the logger, typically the module name (__name__).
        level: Optional logging level override. If None, uses the
            LOGGING_LEVEL_ED environment variable or defaults to INFO.

    Returns:
        LazyLogger: A lazily-initialized logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return LazyLogger(name, level)


class RayManager:
    """Manages Ray operations including starting, stopping, and verifying clusters.

    This class provides comprehensive functionality for managing Ray clusters
    on TPU infrastructure, including:

    - Finding and validating the Ray executable
    - Starting head and worker nodes with proper resource configuration
    - Stopping individual nodes or entire clusters
    - Verifying cluster health and TPU resource availability
    - Running commands locally or remotely via SSH

    Attributes:
        logger: Logger instance for this class.
        ray_path: Path to the Ray executable.

    Example:
        >>> manager = RayManager()
        >>> manager.start_head_node("10.0.0.1", {"TPU": 4})
        >>> manager.start_worker_node("10.0.0.1", "10.0.0.2", {"TPU": 4})
        >>> manager.verify_cluster("10.0.0.1", expected_tpu_count=8)
    """

    def __init__(self):
        """Initialize the RayManager.

        Locates the Ray executable and sets up logging. Raises FileNotFoundError
        if Ray cannot be found in the system.
        """
        self.logger = logging.getLogger(__name__)
        self.ray_path = self._find_ray_executable()

    def _find_ray_executable(self) -> pathlib.Path:
        """Return the absolute path to 'ray' inside the current venv or system.

        Searches for the Ray executable in the following order:
        1. RAY_EXECUTABLE_PATH environment variable
        2. Same directory as the Python executable
        3. System PATH

        Returns:
            pathlib.Path: Absolute path to the Ray executable.

        Raises:
            FileNotFoundError: If Ray executable cannot be located.
        """
        if ray_path := os.getenv("RAY_EXECUTABLE_PATH"):
            path = pathlib.Path(ray_path).expanduser().resolve()
            self.logger.debug(f"Using RAY_EXECUTABLE_PATH: {path}")
            return path

        bin_dir = pathlib.Path(sys.executable).parent
        ray_path = bin_dir / "ray"
        if ray_path.is_file():
            self.logger.debug(f"Found ray in bin directory: {ray_path}")
            return ray_path

        if ray_path := shutil.which("ray"):
            path = pathlib.Path(ray_path)
            self.logger.debug(f"Found ray in PATH: {path}")
            return path

        self.logger.error("Ray executable could not be located")
        raise FileNotFoundError("ray executable could not be located")

    def run_command(
        self,
        command: str | list[str],
        use_sudo: bool = False,
        check: bool = True,
        capture_output: bool = False,
        remote_ip: str | None = None,
    ) -> bool | str:
        """Run a command locally or remotely with optional output capture.

        Executes shell commands either on the local machine or on a remote
        machine via SSH. Supports sudo elevation and output capture.

        Args:
            command: Command to execute, as a string or list of arguments.
            use_sudo: If True, prepend 'sudo' to the command.
            check: If True, raise exception on non-zero exit code.
            capture_output: If True, return stdout; otherwise return success bool.
            remote_ip: If provided, execute command on this remote host via SSH.

        Returns:
            Union[bool, str]: If capture_output is True, returns stdout as string.
                Otherwise, returns True if command succeeded, False if it failed.

        Example:
            >>> manager.run_command(["ls", "-la"])
            True
            >>> manager.run_command("hostname", capture_output=True, remote_ip="10.0.0.1")
            'worker-1'
        """
        if isinstance(command, list):
            cmd_str = " ".join([str(c) for c in command])
        else:
            cmd_str = command

        self.logger.info(f"Running command: {cmd_str} {'on ' + remote_ip if remote_ip else 'locally'}")

        if use_sudo:
            if isinstance(command, list):
                full_command = ["sudo", *command]
            else:
                full_command = f"sudo {command}"
        else:
            full_command = command

        try:
            if remote_ip:
                if isinstance(full_command, list):
                    remote_cmd = shlex.join([str(c) for c in full_command])
                else:
                    remote_cmd = full_command
                ssh_command = [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=5",
                    f"{SSH_USER}@{remote_ip}",
                    remote_cmd,
                ]
                result = subprocess.run(ssh_command, check=check, capture_output=capture_output, text=True)
                return result.stdout if capture_output else (result.returncode == 0)
            else:
                if isinstance(full_command, list):
                    result = subprocess.run(full_command, check=check, capture_output=capture_output, text=True)
                else:
                    result = subprocess.run(
                        full_command,
                        shell=True,
                        check=check,
                        capture_output=capture_output,
                        text=True,
                    )
                return result.stdout if capture_output else (result.returncode == 0)
        except subprocess.CalledProcessError as e:
            location = f"on {remote_ip}" if remote_ip else "locally"
            self.logger.error(f"Command failed {location}: {cmd_str}. Error: {e.stderr}")
            return False if not capture_output else ""
        except FileNotFoundError:
            self.logger.error(f"Command not found: {cmd_str}")
            return False if not capture_output else ""

    def is_ray_running(self, ip: str) -> bool:
        """Check if Ray is already running on a node.

        Determines if Ray processes are running on the specified IP address
        by checking for 'ray::IDLE' processes via ps command.

        Args:
            ip: IP address of the node to check.

        Returns:
            bool: True if Ray is running on the node, False otherwise.
        """
        local_ip = IPManager.get_local_ip()

        if local_ip == ip:
            try:
                subprocess.check_output("ps aux | grep -v grep | grep 'ray::IDLE'", shell=True)
                self.logger.debug(f"Ray is running locally on {ip}")
                return True
            except subprocess.CalledProcessError:
                self.logger.debug(f"Ray is not running locally on {ip}")
                return False
        else:
            result = self.run_command("ps aux | grep -v grep | grep 'ray::IDLE'", remote_ip=ip)
            if result:
                self.logger.debug(f"Ray is running remotely on {ip}")
            else:
                self.logger.debug(f"Ray is not running remotely on {ip}")
            return result

    def start_head_node(self, head_ip: str, resources: dict[str, Any]) -> bool:
        """Start Ray head node with specified resources.

        Initializes the Ray head node on the specified IP address. If Ray is
        already running on the node, it will be stopped first.

        Args:
            head_ip: IP address for the head node.
            resources: Dictionary of custom resources to advertise (e.g., {"TPU": 4}).

        Returns:
            bool: True if the head node started successfully, False otherwise.

        Note:
            The head node will listen on port 6379 and start the dashboard
            on 0.0.0.0 to allow remote access.
        """
        self.logger.info(f"Starting Ray head node on {head_ip}")
        local_ip = IPManager.get_local_ip()

        if self.is_ray_running(head_ip):
            self.logger.info(f"Ray is already running on head node {head_ip}. Stopping it first...")
            self.stop_node(head_ip)
            time.sleep(3)

        resources_str = _resources_json(resources)
        cmd = [
            self.ray_path,
            "start",
            "--head",
            "--port=6379",
            f"--resources={resources_str}",
            f"--node-ip-address={head_ip}",
            "--dashboard-host=0.0.0.0",
        ]

        success = self.run_command(cmd, remote_ip=None if local_ip == head_ip else head_ip)

        if not success:
            self.logger.error("Failed to start Ray head node")
            return False

        self.logger.info("Ray head node started successfully")
        return True

    def start_worker_node(self, head_ip: str, worker_ip: str, resources: dict[str, Any]) -> bool:
        """Start Ray worker node and connect it to the head node.

        Initializes a Ray worker node on the specified IP and connects it
        to the existing head node. If Ray is already running on the worker,
        it will be stopped first.

        Args:
            head_ip: IP address of the Ray head node to connect to.
            worker_ip: IP address for the worker node.
            resources: Dictionary of custom resources to advertise (e.g., {"TPU": 4}).

        Returns:
            bool: True if the worker node started and connected successfully,
                False otherwise.
        """
        self.logger.info(f"Starting Ray worker on {worker_ip}")
        local_ip = IPManager.get_local_ip()

        if self.is_ray_running(worker_ip):
            self.logger.info(f"Ray is already running on worker node {worker_ip}. Stopping it first...")
            self.stop_node(worker_ip)
            time.sleep(3)

        resources_str = _resources_json(resources)
        cmd = [
            str(self.ray_path),
            "start",
            f"--address={head_ip}:6379",
            f"--resources={resources_str}",
            f"--node-ip-address={worker_ip}",
        ]
        success = self.run_command(cmd, remote_ip=None if local_ip == worker_ip else worker_ip)

        if success:
            self.logger.info(f"Successfully started Ray worker on {worker_ip}")
            return True
        else:
            self.logger.error(f"Failed to start Ray worker on {worker_ip}")
            return False

    def stop_node(self, ip: str) -> bool:
        """Stop Ray on a specific node.

        Stops all Ray processes on the specified IP address.

        Args:
            ip: IP address of the node to stop Ray on.

        Returns:
            bool: True if Ray was stopped successfully, False otherwise.
        """
        self.logger.info(f"Stopping Ray on {ip}...")
        local_ip = IPManager.get_local_ip()

        if local_ip == ip:
            return self.run_command([self.ray_path, "stop"])
        else:
            return self.run_command(f"{self.ray_path} stop", remote_ip=ip)

    def stop_cluster(self, ips: list[str]) -> None:
        """Stop Ray cluster on all nodes.

        Stops Ray processes on all nodes in the provided list.

        Args:
            ips: List of IP addresses of nodes in the cluster.
        """
        self.logger.info("Stopping Ray on all nodes...")
        for ip in ips:
            self.stop_node(ip)
        self.logger.info("Ray cluster stopped on all nodes.")

    def verify_cluster(self, head_ip: str, expected_tpu_count: int, allow_head_only: bool = False) -> bool:
        """Verify Ray cluster setup and TPU resource availability.

        Connects to the Ray cluster and verifies that the expected number
        of TPU cores are registered. Allows a 10% tolerance for TPU detection.

        Args:
            head_ip: IP address of the Ray head node.
            expected_tpu_count: Expected number of TPU cores across all nodes.
            allow_head_only: If True, allow verification to pass for head-only
                nodes with zero TPU resources.

        Returns:
            bool: True if verification passed, False otherwise.

        Note:
            Creates a temporary Python script to perform verification,
            which is cleaned up after execution.
        """
        self.logger.info("Verifying Ray cluster setup...")
        if allow_head_only and expected_tpu_count == 0:
            self.logger.info("Head-only node detected (no TPU resources)")
            return True

        script_content = """
import ray
import time
import sys
import os


for i in range(5):
    try:
        ray.init(address='auto')
        break
    except Exception as e:
        print(f"Connection attempt {i+1} failed: {e}")
        time.sleep(5)
        if i == 4:
            print("Could not connect to Ray cluster")
            sys.exit(1)


try:
    resources = ray.cluster_resources()
    print("\\nCLUSTER RESOURCES:\\n==================")
    for k, v in sorted(resources.items()):
        print(f"{k}: {v}")


    tpu_count = resources.get('TPU', 0)
    print(f"\\nTOTAL TPU COUNT: {tpu_count}")


    expected_tpu = int(os.environ.get('EXPECTED_TPU_COUNT', 64))
    if tpu_count < expected_tpu * 0.9:
        print(f"WARNING: Not all TPU cores are detected! Expected ~{expected_tpu}")
        sys.exit(2)
    else:
        print(f"SUCCESS: TPU cores detected ({tpu_count}/{expected_tpu})")
except Exception as e:
    print(f"Error getting cluster resources: {e}")
    sys.exit(3)

ray.shutdown()
"""
        script_path = "/tmp/verify_ray_tpu.py"
        with open(script_path, "w") as f:
            f.write(script_content)
        self.logger.debug(f"Created verification script at {script_path}")

        env = os.environ.copy()
        env["EXPECTED_TPU_COUNT"] = str(expected_tpu_count)

        try:
            subprocess.run(["python", script_path], env=env, check=True)
            self.logger.info("Verification successful!")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Verification failed with status {e.returncode}")
            return False
        finally:
            try:
                os.remove(script_path)
                self.logger.debug("Removed verification script")
            except FileNotFoundError:
                self.logger.warning("Verification script not found for removal")


class IPManager:
    """Manages IP address operations and configurations.

    Provides static methods for retrieving local and external IP addresses,
    reading IP configurations from YAML files, and testing SSH connectivity
    to cluster nodes.

    This class is designed for managing network configurations in distributed
    TPU/Ray clusters where nodes need to communicate with each other.

    Example:
        >>> local_ip = IPManager.get_local_ip()
        >>> external_ip = IPManager.get_external_ip()
        >>> IPManager.read_ips_from_yaml("cluster_config.yaml")
        >>> IPManager.test_ssh_connectivity(["10.0.0.1", "10.0.0.2"])
    """

    @staticmethod
    def get_external_ip() -> str:
        """Get the external IP address of the machine.

        Queries multiple public IP detection services to determine the
        machine's external IP address.

        Returns:
            str: External IP address, or an error message if detection fails.

        Note:
            Tries ipify.org, ipinfo.io, and checkip.amazonaws.com in sequence,
            using the first successful response.
        """
        urls = ["https://api.ipify.org", "https://ipinfo.io/ip", "https://checkip.amazonaws.com"]

        logger = logging.getLogger(__name__)

        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    ip = response.text.strip()
                    logger.debug(f"Retrieved external IP from {url}: {ip}")
                    return ip
            except Exception as e:
                logger.debug(f"Failed to get external IP from {url}: {e}")
                continue

        logger.warning("Could not determine external IP")
        return "Could not determine external IP"

    @staticmethod
    def get_local_ip() -> str:
        """Get the local IP address of the machine.

        Retrieves the primary local IP address using the 'hostname -I' command.

        Returns:
            str: Local IP address, or '127.0.0.1' if detection fails.
        """
        logger = logging.getLogger(__name__)

        try:
            hostname_output = subprocess.check_output(["hostname", "-I"]).decode().strip()
            ip = hostname_output.split()[0]
            logger.debug(f"Retrieved local IP: {ip}")
            return ip
        except Exception as e:
            logger.error(f"Error getting local IP: {e}")
            return "127.0.0.1"

    @staticmethod
    def read_ips_from_yaml(yaml_file: str) -> bool:
        """Read IP addresses from a YAML configuration file.

        Parses a YAML file containing 'internal_ips' and 'external_ips' lists
        and updates the global INTERNAL_IPS and EXTERNAL_IPS variables.

        Args:
            yaml_file: Path to the YAML configuration file.

        Returns:
            bool: True if IPs were successfully read, False on error.

        Expected YAML format:
            internal_ips:
              - 10.0.0.1
              - 10.0.0.2
            external_ips:
              - 35.192.1.1
              - 35.192.1.2
        """
        global INTERNAL_IPS, EXTERNAL_IPS
        logger = logging.getLogger(__name__)

        try:
            with open(yaml_file, "r") as file:
                data = yaml.safe_load(file)

            INTERNAL_IPS = data.get("internal_ips", [])
            EXTERNAL_IPS = data.get("external_ips", [])

            if not INTERNAL_IPS:
                logger.error("No internal IPs found in the YAML file")
                return False

            logger.info(f"Read {len(INTERNAL_IPS)} internal IPs and {len(EXTERNAL_IPS)} external IPs from {yaml_file}")
            return True
        except Exception as e:
            logger.error(f"Error reading YAML file: {e}")
            return False

    @staticmethod
    def test_ssh_connectivity(ips: list[str]) -> bool:
        """Test SSH connectivity to all nodes in the cluster.

        Attempts to connect to each IP address via SSH and execute a simple
        command to verify connectivity.

        Args:
            ips: List of IP addresses to test.

        Returns:
            bool: True if all connections succeeded, False if any failed.

        Note:
            Uses the SSH_USER global variable for authentication. Connections
            use StrictHostKeyChecking=no and a 5-second timeout.
        """
        logger = logging.getLogger(__name__)
        logger.info("Testing SSH connectivity to all nodes...")
        all_good = True

        for ip in ips:
            logger.info(f"Testing connection to {ip}...")
            ssh_command = [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                f"{SSH_USER}@{ip}",
                "echo OK",
            ]

            try:
                subprocess.run(
                    ssh_command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Connection to {ip}: SUCCESS")
            except subprocess.CalledProcessError:
                logger.error(f"Connection to {ip}: FAILED")
                all_good = False

        if all_good:
            logger.info("All SSH connections successful!")
            return True
        else:
            logger.error("Some SSH connections failed. Please check your SSH setup.")
            logger.error("Make sure your SSH key is added to the authorized_keys file on all nodes.")
            logger.error(f"You may need to run: ssh-copy-id {SSH_USER}@<node-ip>")
            return False


class ClusterManager:
    """Manages TPU cluster setup and configuration.

    Provides high-level cluster management operations including setup of
    multi-node Ray clusters on TPU infrastructure, handling both head
    and worker nodes, and managing self-job mode for individual machines.

    Attributes:
        logger: Logger instance for this class.
        ray_manager: RayManager instance for low-level Ray operations.

    Example:
        >>> ray_manager = RayManager()
        >>> cluster_manager = ClusterManager(ray_manager)
        >>> cluster_manager.setup_cluster(use_external=False)
    """

    def __init__(self, ray_manager: RayManager):
        """Initialize the ClusterManager.

        Args:
            ray_manager: RayManager instance to use for Ray operations.
        """
        self.logger = logging.getLogger(__name__)
        self.ray_manager = ray_manager

    def setup_cluster(self, use_external: bool) -> bool:
        """Set up a complete Ray cluster with head and worker nodes.

        Initializes a Ray cluster across multiple nodes, starting the head
        node on the first IP and worker nodes on remaining IPs. Stops any
        existing Ray processes before starting.

        Args:
            use_external: If True, use external IPs; otherwise use internal IPs.

        Returns:
            bool: True if cluster was set up successfully, False otherwise.

        Note:
            Uses global TPU_VERSION, TPU_SLICE_SIZE, and TPU_CORES_PER_HOST
            variables for resource configuration.
        """
        ips = EXTERNAL_IPS if use_external else INTERNAL_IPS
        head_ip = ips[0]
        worker_ips = ips[1:]

        self.logger.info(f"Setting up Ray cluster with {'external' if use_external else 'internal'} IPs")
        self.logger.info(f"TPU Version: {TPU_VERSION}")
        self.logger.info(f"TPU Slice Size: {TPU_SLICE_SIZE}")
        self.logger.info(f"TPU Cores per Host: {TPU_CORES_PER_HOST}")
        self.logger.info(f"SSH User: {SSH_USER}")
        self.logger.info(f"Head node: {head_ip}")
        self.logger.info(f"Worker nodes: {', '.join(worker_ips)}")

        self.ray_manager.stop_cluster(ips)
        self.logger.info("Waiting for processes to fully stop...")
        time.sleep(5)

        head_resources = {
            "TPU": TPU_CORES_PER_HOST,
            f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head": 1,
            f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
        }

        if not self.ray_manager.start_head_node(head_ip, head_resources):
            self.logger.error("Failed to start Ray head node. Exiting.")
            return False

        self.logger.info("Waiting for head node to initialize...")
        time.sleep(10)

        for i, worker_ip in enumerate(worker_ips):
            self.logger.info(f"Starting worker node at {worker_ip}")
            worker_resources = {
                "TPU": TPU_CORES_PER_HOST,
                f"TPU-{TPU_VERSION}-worker-{i}": 1,
                f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
            }
            self.ray_manager.start_worker_node(head_ip, worker_ip, worker_resources)

        self.logger.info("Ray cluster setup complete!")
        self.logger.info(f"Total expected TPU cores: {TPU_SLICE_SIZE}")
        self.logger.info("Waiting for workers to register resources...")
        time.sleep(15)

        self.ray_manager.verify_cluster(head_ip, TPU_SLICE_SIZE)

        self.logger.info("IMPORTANT: To use Ray in your applications, initialize with:")
        self.logger.info(f"ray.init(address='{head_ip}:6379')")

        return True

    def setup_self_job_node(self, args) -> int:
        """Set up a single node in self-job mode.

        Configures the current machine as either a head or worker node based
        on its IP address and provided arguments. Useful for distributed setups
        where each node independently joins the cluster.

        Args:
            args: Namespace containing command-line arguments including:
                - external: Whether to use external IPs
                - internal_ips: Comma-separated internal IPs
                - head_node_ip: IP of external head node (optional)
                - head_only: Run as head-only without TPU resources
                - num_slices: Number of TPU slices
                - verify: Whether to verify cluster after setup
                - stop: Stop Ray instead of starting

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        using_external = args.external or args.internal_ips is None
        local_ip = IPManager.get_external_ip() if using_external else IPManager.get_local_ip()
        external_head_ip = args.head_node_ip
        is_head_only = args.head_only

        self.logger.info(f"Running in self-job mode on {local_ip}")
        if external_head_ip:
            self.logger.info(f"Using external head node at: {external_head_ip}")
        if is_head_only:
            self.logger.info("Running as head-only node (no TPU resources)")
        self.logger.info(f"Using Ray command: {self.ray_manager.ray_path}")

        try:
            result = subprocess.run([self.ray_manager.ray_path, "--version"], capture_output=True, check=True, text=True)
            self.logger.info(f"Ray version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error("Ray command not found or not working.")
            self.logger.error("Please install Ray with: pip install -U ray")
            self.logger.error("Make sure ray is in your PATH or in ~/.local/bin/")
            self.logger.error(f"Exec {e!s}")
            return 1

        self.logger.info("Stopping any existing Ray processes...")
        try:
            subprocess.run([self.ray_manager.ray_path, "stop"], stderr=subprocess.DEVNULL)
        except Exception:
            self.logger.warning("No Ray process was running or could not stop Ray")
        time.sleep(3)

        if args.stop:
            self.ray_manager.run_command(f"{self.ray_manager.ray_path} stop")
            return 0

        is_external_head = external_head_ip and local_ip == external_head_ip
        if external_head_ip:
            head_ip = external_head_ip
        else:
            ips_to_use = EXTERNAL_IPS if using_external else INTERNAL_IPS
            if not ips_to_use:
                self.logger.error("No IPs provided")
                return 1
            head_ip = ips_to_use[0]

        is_head_node = is_external_head or (local_ip == head_ip and not external_head_ip)

        if is_head_node:
            self.logger.info(f"This machine ({local_ip}) is the HEAD node")

            if is_head_only:
                resources = {"head-node": 1, "control-plane": 1, "ray-cluster-head": 1}
                self.logger.info("Starting as head-only node (no TPU resources)")
            else:
                self.logger.info("Starting as head node with TPU resources")
                resources = {
                    "TPU": TPU_CORES_PER_HOST,
                    f"TPU-{TPU_VERSION}": TPU_CORES_PER_HOST,
                    f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE}-head": 1,
                    f"TPU-{TPU_VERSION}-{TPU_SLICE_SIZE * args.num_slices}-global-head": 1,
                    f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
                    "head-node": 1,
                    "ray-cluster-head": 1,
                }

            if not self.ray_manager.start_head_node(local_ip, resources):
                self.logger.error("Failed to start Ray head")
                return 1

            self.logger.info("Ray head started successfully!")
            self.logger.info(f"Dashboard available at: http://{local_ip}:8265")
            self.logger.info(f"Ray cluster address: {local_ip}:6379")

            if is_head_only:
                self.logger.info("Waiting for TPU workers to connect...")
                self.logger.info(f"Run the script on TPU nodes with --head-node-ip {local_ip}")
        else:
            self.logger.info(f"This machine ({local_ip}) is a WORKER node")
            self.logger.info(f"Will connect to head node at: {head_ip}")

            resources = {
                "TPU": TPU_CORES_PER_HOST,
                f"TPU-{TPU_VERSION}": TPU_CORES_PER_HOST,
                f"accelerator_type:TPU-{TPU_VERSION.upper()}": 1,
            }

            if not self.ray_manager.start_worker_node(head_ip, local_ip, resources):
                self.logger.error("Failed to start Ray worker")
                return 1

            self.logger.info("Ray worker started successfully!")
            self.logger.info(f"Connected to head at: {head_ip}:6379")
            self.logger.info(f"Worker resources: {resources}")

        if is_head_node and args.verify:
            self.logger.info("Waiting for cluster to stabilize...")
            time.sleep(10)
            expected_tpu = 0 if is_head_only else TPU_SLICE_SIZE * args.num_slices
            if self.ray_manager.verify_cluster(head_ip, expected_tpu):
                self.logger.info("Cluster verification successful!")
            else:
                self.logger.error("Cluster verification failed!")
                return 1

        return 0


class ArgumentParser:
    """Handles command-line argument parsing for the Ray TPU patcher.

    Provides static methods to parse and validate command-line arguments
    for setting up and managing Ray clusters on TPU infrastructure.

    The parser supports multiple argument groups:
    - General options (logging level, log file)
    - Cluster configuration (external IPs, TPU version, slice size)
    - IP configuration (config file, IP lists, head node)
    - Operation modes (stop, verify, test-ssh, self-job)
    """

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments for Ray TPU cluster setup.

        Returns:
            argparse.Namespace: Parsed arguments containing all configuration
                options for cluster setup and management.

        Note:
            Uses argparse.ArgumentDefaultsHelpFormatter to display default
            values in the help text.
        """
        parser = argparse.ArgumentParser(
            description="Ray TPU Cluster Setup for TPU Slices",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        general_group = parser.add_argument_group("General Options")
        general_group.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Set the logging level",
        )
        general_group.add_argument("--log-file", help="Path to log file (if not specified, logs to console only)")

        cluster_group = parser.add_argument_group("Cluster Configuration")
        cluster_group.add_argument("--external", action="store_true", help="Use external IPs instead of internal IPs")
        cluster_group.add_argument(
            "--tpu-version",
            default=TPU_VERSION,
            help=f"Set TPU version (default: {TPU_VERSION})",
        )
        cluster_group.add_argument(
            "--tpu-slice",
            type=int,
            default=TPU_SLICE_SIZE,
            help=f"Set TPU slice size (default: {TPU_SLICE_SIZE})",
        )
        cluster_group.add_argument(
            "--num-slices",
            type=int,
            default=1,
            help="Number of TPU slices to combine (default: 1)",
        )
        cluster_group.add_argument("--ssh-user", default=SSH_USER, help=f"SSH username to use (default: {SSH_USER})")

        ip_group = parser.add_argument_group("IP Configuration")
        ip_group.add_argument("--config", help="Path to YAML config file with IP addresses")
        ip_group.add_argument("--internal-ips", help="Comma-separated list of internal IPs for all slices")
        ip_group.add_argument("--external-ips", help="Comma-separated list of external IPs for all slices")
        ip_group.add_argument("--slice-config", help="Path to YAML config file with slice configurations")
        ip_group.add_argument("--head-node-ip", help="IP address of external head node (if not using first IP in list)")

        operation_group = parser.add_argument_group("Operation Modes")
        operation_group.add_argument("--stop", action="store_true", help="Stop the Ray cluster")
        operation_group.add_argument("--verify", action="store_true", help="Verify the Ray cluster setup")
        operation_group.add_argument("--test-ssh", action="store_true", help="Test SSH connectivity to all nodes")
        operation_group.add_argument("--self-job", action="store_true", help="Run only on the current machine")
        operation_group.add_argument(
            "--head-only",
            action="store_true",
            help="Run this node as head only (no TPU resources)",
        )

        return parser.parse_args()


def main():
    """Main entry point for the Ray TPU cluster patcher.

    Parses command-line arguments and performs the requested operation:
    - Set up a complete cluster across multiple nodes
    - Set up a single node in self-job mode
    - Test SSH connectivity
    - Stop an existing cluster
    - Verify cluster health

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = ArgumentParser.parse_arguments()

    logger = get_logger("RayPatcher")

    logger.debug(f"Command line arguments: {vars(args)}")

    global TPU_VERSION, TPU_SLICE_SIZE, SSH_USER, INTERNAL_IPS, EXTERNAL_IPS
    TPU_VERSION = args.tpu_version
    TPU_SLICE_SIZE = args.tpu_slice
    SSH_USER = args.ssh_user

    if args.internal_ips:
        INTERNAL_IPS = args.internal_ips.split(",")
        logger.info(f"Using internal IPs from command line: {INTERNAL_IPS}")

    if args.external_ips:
        EXTERNAL_IPS = args.external_ips.split(",")
        logger.info(f"Using external IPs from command line: {EXTERNAL_IPS}")

    if args.config:
        if not IPManager.read_ips_from_yaml(args.config):
            logger.error("Failed to read configuration from YAML file.")
            return 1

    ray_manager = RayManager()
    cluster_manager = ClusterManager(ray_manager)

    if args.self_job:
        return cluster_manager.setup_self_job_node(args)
    else:
        using_external = args.external or args.internal_ips is None
        ips = EXTERNAL_IPS if using_external else INTERNAL_IPS
        total_tpu_cores = TPU_SLICE_SIZE * args.num_slices

        if args.test_ssh:
            return 0 if IPManager.test_ssh_connectivity(ips) else 1

        if args.stop:
            ray_manager.stop_cluster(ips)
            return 0

        if args.verify:
            head_ip = ips[0] if ips else "0.0.0.0"
            return 0 if ray_manager.verify_cluster(head_ip, total_tpu_cores) else 1

        return 0 if cluster_manager.setup_cluster(using_external) else 1


if __name__ == "__main__":
    sys.exit(main())
