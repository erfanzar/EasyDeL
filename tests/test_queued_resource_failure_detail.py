# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A FAILED queued resource carries GCP's own reason into the observation.

Preserving ``failedData.error`` lets operators distinguish provider-side faults
from stockouts or configuration problems even while the resource is suspending.
"""

from __future__ import annotations

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.capacity.providers.gcp.queued_resources import (
    GcloudTpuProvider,
    GcpQueuedResourceState,
    QueuedResource,
)


def _qr(state: GcpQueuedResourceState, raw_state: dict) -> QueuedResource:
    return QueuedResource("cell-r1", state, (), "v5litepod-8", {"state": raw_state})


def test_a_failed_queued_resource_names_the_provider_reason() -> None:
    qr = _qr(
        GcpQueuedResourceState.FAILED,
        {
            "state": "FAILED",
            "failedData": {"error": {"code": 10, "message": "Failed to perform tenant project\ncreation"}},
        },
    )

    observation = GcloudTpuProvider._queued_resource_observation(qr)

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert observation.detail == (
        "queued resource is terminal (FAILED): Failed to perform tenant project creation (code 10)"
    )
    assert observation.failure_code is None, "a tenant-project fault is not a stockout or a quota halt"


def test_a_failed_stockout_is_classified_from_the_providers_words() -> None:
    qr = _qr(
        GcpQueuedResourceState.FAILED,
        {"failedData": {"error": {"code": 8, "message": "There is no more capacity in the zone; stockout"}}},
    )

    observation = GcloudTpuProvider._queued_resource_observation(qr)

    assert observation.failure_code is CapacityProviderFailureCode.STOCKOUT
    assert "no more capacity" in observation.detail


@pytest.mark.parametrize("raw_state", ({}, {"failedData": {}}, {"failedData": {"error": {"message": "  "}}}))
def test_a_terminal_resource_without_a_reason_keeps_the_plain_sentence(raw_state: dict) -> None:
    observation = GcloudTpuProvider._queued_resource_observation(_qr(GcpQueuedResourceState.FAILED, raw_state))

    assert observation.detail == "queued resource is terminal (FAILED)"
    assert observation.failure_code is None
