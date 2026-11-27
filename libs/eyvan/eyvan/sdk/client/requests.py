# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared typed request construction for Eyvan SDK clients."""

from __future__ import annotations

import typing as tp

from pydantic import BaseModel

from eyvan._internal.json_types import JsonObject, model_json
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    CrossRegionTransferPolicy,
    PriorityBand,
    TpuChipRequest,
    WorkloadSetup,
    WorkloadState,
    WorkspaceRef,
)
from eyvan.sdk.client.jobs import reject_remote_callable_signature, reject_remote_callable_signing_environment
from eyvan.sdk.client.types import QueryParams
from eyvan.server.api.schemas import PoolCallableSubmitRequest, PoolWorkloadSubmitRequest


def pool_request_json(request: BaseModel) -> JsonObject:
    """Serialize a pool request, omitting only server-defaulted top-level fields."""

    document = model_json(request)
    for field_name in set(type(request).model_fields) - request.model_fields_set:
        document.pop(field_name, None)
    return document


def pool_workload_request(
    *,
    org_id: str,
    project_id: str,
    command: str,
    name: str | None,
    environment: tp.Mapping[str, str] | None,
    secret_mounts: tp.Mapping[str, str] | None = None,
    metadata: tp.Mapping[str, str] | None,
    max_retries: int,
    max_failure_retries: int = 0,
    max_preemption_retries: int = 10,
    pods: int = 1,
    zone: str | None = None,
    workspace: WorkspaceRef | None = None,
    checkpointable: bool = False,
    checkpoint_hook: str | None = None,
    checkpoint_output_uri: str | None = None,
    can_submit_children: bool = False,
    priority_band: PriorityBand = PriorityBand.INTERACTIVE,
    intra_band_priority: int = 0,
    checkpoint_grace_seconds: int = 0,
    estimated_duration_seconds: int = 3600,
    named_assets: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    tpu_chips: TpuChipRequest | None = None,
    setup: WorkloadSetup | None = None,
    reclaim_policy: tp.Literal["requeue", "stop"] | None = None,
    cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
) -> PoolWorkloadSubmitRequest:
    """Build the shared pool-command request payload."""

    optional_fields: dict[str, tp.Any] = {"tpu_chips": tpu_chips} if tpu_chips is not None else {}
    if setup is not None:
        optional_fields["setup"] = setup
    if reclaim_policy is not None and "reclaim_policy" in PoolWorkloadSubmitRequest.model_fields:
        optional_fields["reclaim_policy"] = reclaim_policy
    if cross_region_transfer_policy is not None:
        optional_fields["cross_region_transfer_policy"] = cross_region_transfer_policy
    return PoolWorkloadSubmitRequest(
        **optional_fields,
        org_id=org_id,
        project_id=project_id,
        command=command,
        name=name,
        environment=dict(environment or {}),
        secret_mounts=dict(secret_mounts or {}),
        metadata=dict(metadata or {}),
        dependencies=dependencies,
        max_retries=max_retries,
        max_failure_retries=max_failure_retries,
        max_preemption_retries=max_preemption_retries,
        pods=pods,
        zone=zone,
        workspace=workspace,
        checkpointable=checkpointable,
        checkpoint_hook=checkpoint_hook,
        checkpoint_output_uri=checkpoint_output_uri,
        can_submit_children=can_submit_children,
        priority_band=priority_band,
        intra_band_priority=intra_band_priority,
        checkpoint_grace_seconds=checkpoint_grace_seconds,
        estimated_duration_seconds=estimated_duration_seconds,
        named_assets=named_assets,
    )


def pool_callable_request(
    *,
    org_id: str,
    project_id: str,
    accelerator_type: str,
    pods: int,
    capacity_class: CapacityClass,
    execution: CallableExecution,
    name: str | None,
    environment: tp.Mapping[str, str] | None,
    secret_mounts: tp.Mapping[str, str] | None = None,
    metadata: tp.Mapping[str, str] | None,
    max_retries: int,
    max_failure_retries: int = 0,
    max_preemption_retries: int = 10,
    checkpointable: bool = False,
    checkpoint_hook: str | None = None,
    checkpoint_output_uri: str | None = None,
    can_submit_children: bool = False,
    priority_band: PriorityBand = PriorityBand.INTERACTIVE,
    intra_band_priority: int = 0,
    checkpoint_grace_seconds: int = 0,
    estimated_duration_seconds: int = 3600,
    named_assets: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
    atomic: bool = False,
    tpu_chips: TpuChipRequest | None = None,
    setup: WorkloadSetup | None = None,
    reclaim_policy: tp.Literal["requeue", "stop"] | None = None,
    cross_region_transfer_policy: CrossRegionTransferPolicy | None = None,
) -> PoolCallableSubmitRequest:
    """Build the shared pool-callable request payload.

    ``tpu_chips`` is passed to the model only when set, so ``pool_request_json``
    (which drops fields the caller never set) keeps whole-slice requests
    byte-identical to what they were before chip sharing existed.
    """

    reject_remote_callable_signature(execution.signature)
    reject_remote_callable_signing_environment(environment)
    optional_fields: dict[str, tp.Any] = {"tpu_chips": tpu_chips} if tpu_chips is not None else {}
    if setup is not None:
        optional_fields["setup"] = setup
    if reclaim_policy is not None and "reclaim_policy" in PoolCallableSubmitRequest.model_fields:
        optional_fields["reclaim_policy"] = reclaim_policy
    if cross_region_transfer_policy is not None:
        optional_fields["cross_region_transfer_policy"] = cross_region_transfer_policy
    return PoolCallableSubmitRequest(
        **optional_fields,
        org_id=org_id,
        project_id=project_id,
        accelerator_type=accelerator_type,
        pods=pods,
        atomic=atomic,
        capacity_class=capacity_class,
        execution=execution,
        name=name,
        environment=dict(environment or {}),
        secret_mounts=dict(secret_mounts or {}),
        metadata=dict(metadata or {}),
        max_retries=max_retries,
        max_failure_retries=max_failure_retries,
        max_preemption_retries=max_preemption_retries,
        checkpointable=checkpointable,
        checkpoint_hook=checkpoint_hook,
        checkpoint_output_uri=checkpoint_output_uri,
        can_submit_children=can_submit_children,
        priority_band=priority_band,
        intra_band_priority=intra_band_priority,
        checkpoint_grace_seconds=checkpoint_grace_seconds,
        estimated_duration_seconds=estimated_duration_seconds,
        named_assets=named_assets,
        dependencies=dependencies,
    )


def workload_list_params(*, org_id: str, project_id: str, state: WorkloadState | None) -> QueryParams:
    """Build workload-list query parameters without serializing empty state."""

    return {"org_id": org_id, "project_id": project_id, **({"state": state.value} if state else {})}
