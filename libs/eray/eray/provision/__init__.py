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

"""Spot TPU capacity provisioning: queued resources, fleet registry, watcher.

The library layer under ``eray qr`` / ``eray fleet`` / ``eray autoscale``.
Talks to GCP exclusively through the gcloud CLI (subprocess), consistent with
the rest of eray; no google-api client dependencies.
"""

from .fleet import ensure_tpu, fleet_status, head_reachable
from .qr import (
    PENDING_STATES,
    RUNTIME_VERSION_BY_FAMILY,
    TERMINAL_STATES,
    QrSpec,
    QueuedResource,
    create_queued_resource,
    default_runtime_version,
    delete_queued_resource,
    describe_queued_resource,
    list_queued_resources,
    qr_create_args,
    wait_for_active,
)
from .registry import ClusterRecord, ClusterRegistry, ConflictError, GcsBackend, LocalBackend

__all__ = (
    "PENDING_STATES",
    "RUNTIME_VERSION_BY_FAMILY",
    "TERMINAL_STATES",
    "ClusterRecord",
    "ClusterRegistry",
    "ConflictError",
    "GcsBackend",
    "LocalBackend",
    "QrSpec",
    "QueuedResource",
    "create_queued_resource",
    "default_runtime_version",
    "delete_queued_resource",
    "describe_queued_resource",
    "ensure_tpu",
    "fleet_status",
    "head_reachable",
    "list_queued_resources",
    "qr_create_args",
    "wait_for_active",
)
