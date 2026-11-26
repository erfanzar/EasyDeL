# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tracking configuration and the exact opt-in dispatch envelope."""

from __future__ import annotations

import json

import pytest

from eyvan.config import EyvanSettings
from eyvan.records import CallableExecution, CommandExecution, WorkloadDefinition, WorkloadRecord
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService


def test_tracking_settings_are_absent_by_default(monkeypatch) -> None:
    for name in ("EYVAN_TRACKER_URL", "EYVAN_TRACKER_TOKEN", "EYVAN_TRACKING_STORE_URI"):
        monkeypatch.delenv(name, raising=False)
    settings = EyvanSettings.from_env()
    assert (settings.tracker_url, settings.tracker_token, settings.tracking_store_uri) == (None, None, None)


def test_tracking_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EYVAN_TRACKER_URL", "https://tracker.example")
    monkeypatch.setenv("EYVAN_TRACKER_TOKEN", "private-token")
    monkeypatch.setenv("EYVAN_TRACKING_STORE_URI", "gs://runs/tracking")
    settings = EyvanSettings.from_env()
    assert (settings.tracker_url, settings.tracker_token, settings.tracking_store_uri) == (
        "https://tracker.example",
        "private-token",
        "gs://runs/tracking",
    )


@pytest.mark.parametrize("url", ["ftp://tracker.example", "file:///tmp/tracker", "tracker.example", "http://"])
def test_tracking_settings_refuse_invalid_url(url) -> None:
    with pytest.raises(ValueError, match="tracker_url"):
        EyvanSettings(tracker_url=url)


@pytest.mark.parametrize("callable_job", [False, True])
def test_tracking_dispatch_is_scoped_and_unset_is_byte_identical(callable_job, monkeypatch) -> None:
    monkeypatch.delenv("EYVAN_CONTROL_PLANE_CA_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    execution = (
        CallableExecution(artifact_uri="gs://artifacts/job.pkl", sha256="0" * 64, python="3.11")
        if callable_job
        else CommandExecution(command="python train.py")
    )
    workload = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        created_by="alice",
        spec=WorkloadDefinition(name="train", execution=execution, environment={"USER_VALUE": "unchanged"}),
    )
    controller = EyvanController(repository, service, settings=EyvanSettings())
    command, environment = controller._render_execution(workload, runtime_environment={"RUNTIME": "preserved"})
    expected = {
        "RUNTIME": "preserved",
        "USER_VALUE": "unchanged",
        "EYVAN_WORKLOAD_ID": "wl-1",
        "EYVAN_ATTEMPT": "1",
        "EYVAN_RESUME_CHECKPOINT_URI": "",
    }
    assert json.dumps(environment).encode() == json.dumps(expected).encode()
    controller.settings = EyvanSettings(tracking_store_uri="gs://runs/tracking/")
    enabled_command, enabled_environment = controller._render_execution(
        workload, runtime_environment={"RUNTIME": "preserved"}
    )
    assert enabled_command == command
    assert enabled_environment == {**expected, "EYVAN_TRACKING_URI": "gs://runs/tracking/acme/research"}
    assert "EYVAN_TRACKER_TOKEN" not in enabled_environment
    controller.settings = EyvanSettings(tracking_store_uri=None)
    assert controller._render_execution(workload, runtime_environment={"RUNTIME": "preserved"}) == (command, environment)
