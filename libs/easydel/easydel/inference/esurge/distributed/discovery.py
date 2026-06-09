# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""DNS-based service discovery for multi-host eSurge clusters.

This module resolves a DNS service name (typically a Kubernetes headless service)
into a deterministic, de-duplicated list of host addresses.  The resolved list is
used by :class:`~.controller.DistributedController` to establish the cluster
topology — each host's position in the sorted list determines its rank.

Classes:
    DiscoveryResult: Immutable container holding the resolved host list.

Functions:
    resolve_service_hosts: Resolves a DNS name into a :class:`DiscoveryResult`.

Example:
    Resolving a Kubernetes headless service::

        >>> from easydel.inference.esurge.distributed.discovery import resolve_service_hosts
        >>> result = resolve_service_hosts("esurge-workers.default.svc.cluster.local", world_size=4)
        >>> result.world_size
        4
        >>> result.rank_to_host
        {0: '10.0.0.1', 1: '10.0.0.2', 2: '10.0.0.3', 3: '10.0.0.4'}
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscoveryResult:
    """Immutable result of DNS service discovery containing the resolved host list.

    Hosts are sorted deterministically (IP addresses numerically first, then
    hostnames lexicographically) so that every node in the cluster derives the
    same rank assignment from an identical DNS query.

    Attributes:
        hosts: Sorted, de-duplicated list of resolved host addresses.
    """

    hosts: list[str]

    @property
    def world_size(self) -> int:
        """Return the total number of hosts in the cluster."""
        return len(self.hosts)

    @property
    def rank_to_host(self) -> dict[int, str]:
        """Return a mapping from rank index to host address."""
        return {idx: host for idx, host in enumerate(self.hosts)}


def _host_sort_key(host: str) -> tuple[int, str]:
    """Return a sort key that orders IP addresses before hostnames.

    IP addresses are converted to zero-padded integers so they sort
    numerically.  Hostnames sort lexicographically after all IPs.

    Args:
        host: An IP address string or hostname.

    Returns:
        A ``(priority, sort_string)`` tuple where *priority* is ``0`` for IPs
        and ``1`` for hostnames.
    """
    try:
        ip = ipaddress.ip_address(host)
        return (0, f"{int(ip):039d}")
    except ValueError:
        return (1, host)


def resolve_service_hosts(service_name: str, world_size: int | None = None) -> DiscoveryResult:
    """Resolve a DNS service name into a sorted, de-duplicated host list.

    Uses :func:`socket.getaddrinfo` to look up *service_name* and collects
    unique IP addresses from the results.  The addresses are sorted via
    :func:`_host_sort_key` to produce a deterministic ordering across hosts.

    Args:
        service_name: DNS name to resolve (e.g. a Kubernetes headless service).
        world_size: If provided, the number of resolved hosts must match this
            value exactly; otherwise a :class:`ValueError` is raised.

    Returns:
        A :class:`DiscoveryResult` containing the sorted host list.

    Raises:
        ValueError: If *service_name* is empty, DNS resolution fails, no hosts
            are found, or the resolved count does not match *world_size*.
    """

    if not service_name or not str(service_name).strip():
        raise ValueError("`distributed_service_name` must be a non-empty DNS name.")

    try:
        entries = socket.getaddrinfo(str(service_name), None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Failed to resolve distributed service '{service_name}': {exc}") from exc

    hosts: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        host = sockaddr[0]
        if host in seen:
            continue
        seen.add(host)
        hosts.append(host)

    hosts.sort(key=_host_sort_key)

    if not hosts:
        raise ValueError(f"No hosts resolved for distributed service '{service_name}'.")

    if world_size is not None and int(world_size) != len(hosts):
        raise ValueError(
            "Distributed world size mismatch: "
            f"resolved_hosts={len(hosts)} expected_world_size={int(world_size)} "
            f"service={service_name!r} hosts={hosts}"
        )

    return DiscoveryResult(hosts=hosts)
