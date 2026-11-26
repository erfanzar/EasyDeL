# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Google Cloud capacity primitives."""

from eyvan.capacity.providers.gcp.compute_instances import (
    ComputeInstance,
    ComputeInstanceSpec,
    ComputeProviderConfig,
    GcloudComputeProvider,
    GcloudDefaultComputeServiceAccountResolver,
    GcloudNativeWorkerArtifactAccessStore,
    GcloudNativeWorkerSecretStore,
    GcpComputeInstanceStatus,
    GcsArtifactBucket,
    compute_instance_name,
    native_worker_secret_id,
    normalize_compute_config,
)
from eyvan.capacity.providers.gcp.native_worker_ingress import (
    FirewallNetworkReference,
    FirewallRuleConflictError,
    GcloudFirewallRuleError,
    GcloudNativeWorkerPublicIngress,
    HttpPublicIpv4Resolver,
    NativeWorkerEndpointMode,
    NativeWorkerIngressError,
    NativeWorkerPublicIngress,
    PublicIpv4ResolutionError,
    endpoint_mode_from_metadata,
)
from eyvan.capacity.providers.gcp.queued_resources import (
    GcloudTpuProvider,
    GcpQueuedResourceState,
    GcpTpuNodeState,
    ProviderObservation,
    QueuedResource,
    QueuedResourceSpec,
    TpuProviderConfig,
)
from eyvan.capacity.providers.gcp.quota_discovery import GcloudQuotaError, GcpTpuQuotaDiscovery, TpuQuotaAvailability
from eyvan.capacity.providers.gcp.quota_usage import GcpQuotaUsageError, GcpTpuQuotaUsageDiscovery, TpuQuotaUsage

__all__ = (
    "ComputeInstance",
    "ComputeInstanceSpec",
    "ComputeProviderConfig",
    "FirewallNetworkReference",
    "FirewallRuleConflictError",
    "GcloudComputeProvider",
    "GcloudDefaultComputeServiceAccountResolver",
    "GcloudFirewallRuleError",
    "GcloudNativeWorkerArtifactAccessStore",
    "GcloudNativeWorkerPublicIngress",
    "GcloudNativeWorkerSecretStore",
    "GcloudQuotaError",
    "GcloudTpuProvider",
    "GcpComputeInstanceStatus",
    "GcpQueuedResourceState",
    "GcpQuotaUsageError",
    "GcpTpuNodeState",
    "GcpTpuQuotaDiscovery",
    "GcpTpuQuotaUsageDiscovery",
    "GcsArtifactBucket",
    "HttpPublicIpv4Resolver",
    "NativeWorkerEndpointMode",
    "NativeWorkerIngressError",
    "NativeWorkerPublicIngress",
    "ProviderObservation",
    "PublicIpv4ResolutionError",
    "QueuedResource",
    "QueuedResourceSpec",
    "TpuProviderConfig",
    "TpuQuotaAvailability",
    "TpuQuotaUsage",
    "compute_instance_name",
    "endpoint_mode_from_metadata",
    "native_worker_secret_id",
    "normalize_compute_config",
)
