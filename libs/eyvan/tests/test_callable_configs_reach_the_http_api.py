# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every public hardware config is usable by the shared HTTP submitter."""

import json

import httpx
import pytest

from eyvan.artifacts import LocalArtifactStore, StoredCallableArtifact
from eyvan.records import CallableExecution, WorkloadDefinition, WorkloadRecord
from eyvan.sdk.client.callables import EyvanCallableSubmitter
from eyvan.sdk.client.jobs import CallableSubmission, CpuConfig, GpuConfig, TpuConfig
from eyvan.sdk.client.sync import EyvanClient


@pytest.mark.parametrize(
    ("config", "atomic"),
    [
        (CpuConfig(), False),
        (GpuConfig(type="nvidia-tesla-a100"), False),
        (TpuConfig(type="v6e-8"), False),
        (TpuConfig(type="v6e-8", atomic=True), True),
    ],
)
def test_hardware_configs_submit_a_callable_and_return_a_durable_handle(tmp_path, config, atomic) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/v1/fleets/research/callables"
        assert payload["atomic"] is atomic
        assert payload["accelerator_type"] == config.type
        workload = WorkloadRecord(
            id="submitted",
            org_id="acme",
            project_id="research",
            created_by="alice",
            spec=WorkloadDefinition(name="probe", execution=CallableExecution.model_validate(payload["execution"])),
        )
        return httpx.Response(201, json=workload.model_dump(mode="json"))

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://plane") as raw:
        submitter = EyvanCallableSubmitter(
            EyvanClient(base_url="http://plane", client=raw), LocalArtifactStore(tmp_path)
        )
        job = submitter.submit_callable(
            CallableSubmission(
                invocation_id="probe",
                fleet="research",
                org_id="acme",
                project_id="research",
                callable_name="probe",
                tpu=config,
                artifact=StoredCallableArtifact("gs://trusted/probe", "a" * 64, "3.12", 1),
                result_uri="gs://trusted/result",
            )
        )
        assert job.id == "submitted"
        assert job.result_uri == "gs://trusted/result"
