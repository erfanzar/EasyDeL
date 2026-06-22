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


"""Cluster topology types for multi-slice execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultisliceInfo:
    """Information about a multi-slice configuration for distributed execution.

    This class stores configuration data for multi-slice TPU/GPU clusters where
    computation is distributed across multiple slices that coordinate through
    a central coordinator node.

    Attributes:
        coordinator_ip: IP address of the coordinator node.
        slice_id: Unique identifier for this slice within the multi-slice setup.
        num_slices: Total number of slices in the multi-slice configuration.
        port: Port number for multi-slice coordination communication.

    Example:
        >>> multi_slice_config = MultisliceInfo(
        ...     coordinator_ip="10.0.0.1",
        ...     slice_id=0,
        ...     num_slices=4,
        ...     port=8081
        ... )
        >>> print(f"Slice {multi_slice_config.slice_id} of {multi_slice_config.num_slices}")
    """

    coordinator_ip: str
    slice_id: int
    num_slices: int
    port: int = 8081


@dataclass
class SliceInfo:
    """Information about a single compute slice in a distributed cluster.

    This class represents the configuration and metadata for a single compute
    slice, which typically consists of multiple hosts with accelerators (TPUs/GPUs).
    Used in multi-slice configurations for large-scale distributed training.

    Attributes:
        slice_name: Unique name identifier for this slice.
        num_hosts: Number of host machines in this slice.
        ip_address: IP address of the slice head node.
        num_accelerators_per_host: Number of accelerators (TPUs/GPUs) per host machine.

    Example:
        >>> slice_config = SliceInfo(
        ...     slice_name="slice-0",
        ...     num_hosts=8,
        ...     ip_address="10.0.1.10",
        ...     num_accelerators_per_host=8
        ... )
        >>> total_accelerators = slice_config.num_hosts * slice_config.num_accelerators_per_host
        >>> print(f"Slice has {total_accelerators} total accelerators")
    """

    slice_name: str
    num_hosts: int
    ip_address: str
    num_accelerators_per_host: int
    node_ids: list[str] | None = None
    host_infos: list[dict] | None = None


@dataclass(frozen=True)
class HostInfo:
    """Information about a TPU host within a slice.

    Attributes:
        host_id: Unique identifier for the host within its slice.
        slice_name: Name of the TPU slice this host belongs to.
        num_devices: Number of TPU devices available on this host.
        healthy: Whether the host is currently healthy and operational.
        failed: Whether the host has encountered a failure.
    """

    host_id: int
    slice_name: str
    num_devices: int | None
    healthy: bool
    failed: bool
    node_id: str | None = None
