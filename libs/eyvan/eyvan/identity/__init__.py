# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider workload identity: rotation, metadata identity, and mutual TLS.

Deliberately low in the import graph -- standard library and
:mod:`eyvan._internal` only -- because all three of its concerns are shared by
the control plane, the cell agent, and the native worker, and a module those
three import must not drag the control plane into the worker.
"""

from __future__ import annotations

from eyvan.identity.metadata import (
    GCP_METADATA_IDENTITY_URL,
    METADATA_FLAVOR_GOOGLE,
    METADATA_FLAVOR_HEADER,
    MetadataResponse,
    WorkloadIdentity,
    WorkloadIdentitySource,
    resolve_workload_identity,
)
from eyvan.identity.mtls import (
    MUTUAL_TLS_EDGES,
    FleetMutualTls,
    MutualTlsConfigurationError,
    MutualTlsError,
    MutualTlsFiles,
    MutualTlsMode,
    MutualTlsRolloutError,
    SurfaceMutualTls,
    TlsSurface,
    client_ssl_context,
    enablement_plan,
    parse_mutual_tls_mode,
    server_ssl_context,
    server_ssl_options,
    validate_fleet_state,
    validate_rollout_step,
)
from eyvan.identity.rotation import (
    MINIMUM_CREDENTIAL_BYTES,
    CapabilityKeyring,
    CredentialEpoch,
    CredentialRotationBlocked,
    CredentialRotationError,
    CredentialRotationFenceMismatch,
    CredentialRotationInterleaved,
    CredentialRotationState,
    RotatingCredential,
    StaticCredential,
)

__all__ = (
    "GCP_METADATA_IDENTITY_URL",
    "METADATA_FLAVOR_GOOGLE",
    "METADATA_FLAVOR_HEADER",
    "MINIMUM_CREDENTIAL_BYTES",
    "MUTUAL_TLS_EDGES",
    "CapabilityKeyring",
    "CredentialEpoch",
    "CredentialRotationBlocked",
    "CredentialRotationError",
    "CredentialRotationFenceMismatch",
    "CredentialRotationInterleaved",
    "CredentialRotationState",
    "FleetMutualTls",
    "MetadataResponse",
    "MutualTlsConfigurationError",
    "MutualTlsError",
    "MutualTlsFiles",
    "MutualTlsMode",
    "MutualTlsRolloutError",
    "RotatingCredential",
    "StaticCredential",
    "SurfaceMutualTls",
    "TlsSurface",
    "WorkloadIdentity",
    "WorkloadIdentitySource",
    "client_ssl_context",
    "enablement_plan",
    "parse_mutual_tls_mode",
    "resolve_workload_identity",
    "server_ssl_context",
    "server_ssl_options",
    "validate_fleet_state",
    "validate_rollout_step",
)
