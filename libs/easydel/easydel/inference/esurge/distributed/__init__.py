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

"""Distributed control-plane helpers for multi-host eSurge serving.

This package implements the leader/worker coordination layer that allows the
eSurge inference engine to run in lockstep across multiple hosts.  The
control-plane is built on top of ZeroMQ and uses DNS-based discovery to
resolve the cluster topology.

Modules:
    discovery: DNS-based cluster discovery utilities.
    protocol: Wire-protocol constants and config / sampling fingerprint helpers.

Exports:
    DiscoveryResult: Immutable container holding the resolved host list.
    compute_sampled_digest: Hash sampled tokens for cross-host verification.
    make_config_fingerprint: Build a stable SHA-256 fingerprint of an engine
        config so the leader can verify every worker runs the same configuration.
    resolve_service_hosts: Resolve a DNS service name into a sorted host list.
"""

from .coordinator import (
    LocalCoordinator,
    StepCoordinationError,
    StepCoordinator,
    StepHandle,
    create_step_coordinator,
)
from .discovery import DiscoveryResult, resolve_service_hosts
from .protocol import compute_sampled_digest, make_config_fingerprint
from .request_plane import (
    OriginRequestPlane,
    OwnerRequestPlane,
    RequestPlaneError,
    create_request_plane,
)

__all__ = (
    "DiscoveryResult",
    "LocalCoordinator",
    "OriginRequestPlane",
    "OwnerRequestPlane",
    "RequestPlaneError",
    "StepCoordinationError",
    "StepCoordinator",
    "StepHandle",
    "compute_sampled_digest",
    "create_request_plane",
    "create_step_coordinator",
    "make_config_fingerprint",
    "resolve_service_hosts",
)
