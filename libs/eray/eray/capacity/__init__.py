# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Portions adapted from the Iris cluster manager in marin-community/marin
# (lib/iris — Copyright The Marin Authors, Apache-2.0).
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

"""eray.capacity — acquire and hold TPU capacity via GCP Queued Resources.

The API-client capacity layer under ``eray tpu provision`` / ``eray tpu
reclaim``: typed queued-resource operations (:class:`CloudGcpQrService`),
a multi-zone acquire+hold+reclaim loop (:class:`CapacityPool`), and an
in-memory fake for tests (:class:`FakeQrService`). Adapted from the Iris
cluster manager (marin-community/marin, Apache-2.0) and extended with spot
semantics — see the module docstrings for what changed and why.
"""

from .fake import FakeClock, FakeQrService
from .gcp_qr import CloudGcpQrService, GcpQrService, build_queued_resource, capacity_mode
from .pool import CapacityPool, PoolAction, PoolReport, PoolSpec, plan_pool
from .state import load_pool_specs, remove_pool_spec, save_pool_spec
from .types import (
    CAPACITY_POOL_LABEL,
    CAPACITY_TYPE_LABEL,
    KNOWN_GCP_ZONES,
    KNOWN_TPU_TYPES,
    QR_DEAD_STATES,
    QR_DYING_STATES,
    QR_LIVE_STATES,
    QR_PENDING_STATES,
    TPU_TOPOLOGIES,
    CapacityType,
    InfraError,
    InfraUnavailableError,
    QueuedResourceInfo,
    QuotaExhaustedError,
    ResourceNotFoundError,
    TpuCreateRequest,
    TpuTopologyInfo,
    get_tpu_topology,
)

__all__ = [
    "CAPACITY_POOL_LABEL",
    "CAPACITY_TYPE_LABEL",
    "KNOWN_GCP_ZONES",
    "KNOWN_TPU_TYPES",
    "QR_DEAD_STATES",
    "QR_DYING_STATES",
    "QR_LIVE_STATES",
    "QR_PENDING_STATES",
    "TPU_TOPOLOGIES",
    "CapacityPool",
    "CapacityType",
    "CloudGcpQrService",
    "FakeClock",
    "FakeQrService",
    "GcpQrService",
    "InfraError",
    "InfraUnavailableError",
    "PoolAction",
    "PoolReport",
    "PoolSpec",
    "QueuedResourceInfo",
    "QuotaExhaustedError",
    "ResourceNotFoundError",
    "TpuCreateRequest",
    "TpuTopologyInfo",
    "build_queued_resource",
    "capacity_mode",
    "get_tpu_topology",
    "load_pool_specs",
    "plan_pool",
    "remove_pool_spec",
    "save_pool_spec",
]
