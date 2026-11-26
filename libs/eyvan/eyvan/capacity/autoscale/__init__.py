# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate, persist, and discover Spot TPU fleet configurations for Eyvan."""

from eyvan.capacity.autoscale.cache import (
    AutoscaleConfigCache,
    AutoscaleDefaultsRepository,
    GeneratedAutoscaleCache,
    QueueAdmissionDenialCache,
)
from eyvan.capacity.autoscale.cluster_capacity import (
    DEFAULT_MAX_OBSERVATION_AGE_S,
    MAX_OBSERVATION_SKEW_S,
    ClusterCapacityAdvice,
    ClusterCapacityPolicy,
    advise_launch,
    observation_age,
)
from eyvan.capacity.autoscale.compiler import (
    generate_autoscale_config,
    generate_zone_autoscale_config,
    render_autoscale_yaml,
)
from eyvan.capacity.autoscale.models import (
    DEFAULT_AUTOSCALE_CACHE_ROOT,
    DEFAULT_AUTOSCALE_DEFAULTS_PATH,
    AutoscaleDefaults,
    AutoscaleGenerationRequest,
    AutoscaleRequest,
    AutoscaleSkipReason,
    AutoscaleZoneRequest,
    CachedAutoscaleConfig,
    GeneratedAutoscaleConfig,
    GeneratedZoneAutoscaleConfig,
    GenerationQuotaSnapshot,
    QueueAdmissionDenial,
    QueueAdmissionDenialKey,
    QuotaSnapshot,
    SkippedAutoscaleCandidate,
    TpuProfile,
    validate_gcp_project_id,
)
from eyvan.capacity.autoscale.pool_capacity import (
    CLUSTER_CAPACITY_MAX_AGE_ENV,
    CLUSTER_CAPACITY_PLANNING_ENV,
    DEFAULT_NAMESPACE,
    KUBERNETES_BACKENDS,
    KUBERNETES_NAMESPACE_ENV,
    ClusterCapacityPass,
    PoolCapacityTarget,
    cluster_capacity_policy_from_env,
    kubernetes_capacity_target,
)
from eyvan.capacity.autoscale.profiles import (
    DEFAULT_TPU_PROFILES,
    QUEUED_TPU_PROFILE_CATALOG,
    QueuedTpuProfileCatalog,
    TpuProfileResolution,
    resolve_queued_tpu_profile,
    select_profiles,
    tpu_profile_from_accelerator_type,
)
from eyvan.capacity.autoscale.queue_admission import QueueAdmissionDenialRecorder
from eyvan.capacity.autoscale.service import AutoscaleGenerationResult, AutoscaleGenerationService, TpuQuotaDiscovery

__all__ = (
    "CLUSTER_CAPACITY_MAX_AGE_ENV",
    "CLUSTER_CAPACITY_PLANNING_ENV",
    "DEFAULT_AUTOSCALE_CACHE_ROOT",
    "DEFAULT_AUTOSCALE_DEFAULTS_PATH",
    "DEFAULT_MAX_OBSERVATION_AGE_S",
    "DEFAULT_NAMESPACE",
    "DEFAULT_TPU_PROFILES",
    "KUBERNETES_BACKENDS",
    "KUBERNETES_NAMESPACE_ENV",
    "MAX_OBSERVATION_SKEW_S",
    "QUEUED_TPU_PROFILE_CATALOG",
    "AutoscaleConfigCache",
    "AutoscaleDefaults",
    "AutoscaleDefaultsRepository",
    "AutoscaleGenerationRequest",
    "AutoscaleGenerationResult",
    "AutoscaleGenerationService",
    "AutoscaleRequest",
    "AutoscaleSkipReason",
    "AutoscaleZoneRequest",
    "CachedAutoscaleConfig",
    "ClusterCapacityAdvice",
    "ClusterCapacityPass",
    "ClusterCapacityPolicy",
    "GeneratedAutoscaleCache",
    "GeneratedAutoscaleConfig",
    "GeneratedZoneAutoscaleConfig",
    "GenerationQuotaSnapshot",
    "PoolCapacityTarget",
    "QueueAdmissionDenial",
    "QueueAdmissionDenialCache",
    "QueueAdmissionDenialKey",
    "QueueAdmissionDenialRecorder",
    "QueuedTpuProfileCatalog",
    "QuotaSnapshot",
    "SkippedAutoscaleCandidate",
    "TpuProfile",
    "TpuProfileResolution",
    "TpuQuotaDiscovery",
    "advise_launch",
    "cluster_capacity_policy_from_env",
    "generate_autoscale_config",
    "generate_zone_autoscale_config",
    "kubernetes_capacity_target",
    "observation_age",
    "render_autoscale_yaml",
    "resolve_queued_tpu_profile",
    "select_profiles",
    "tpu_profile_from_accelerator_type",
    "validate_gcp_project_id",
)
