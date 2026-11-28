# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional correlation metadata must not alter legacy mutation payloads."""

import json

import pytest
from pydantic import ValidationError

from eyvan.capacity.drivers.base import MutationContext
from eyvan.worker.agent.models import CapacityRequest, MutationModel

BASE = {"idempotency_key": "command", "generation": 1, "fencing_token": "fence"}


def test_a_mutation_without_correlation_retains_its_exact_wire_shape():
    request = CapacityRequest(mutation=MutationModel(**BASE), desired={})
    assert request.model_dump(mode="json") == {"mutation": BASE, "desired": {}}
    assert json.loads(request.model_dump_json()) == {"mutation": BASE, "desired": {}}
    assert MutationModel(**BASE, correlation_id=None).model_dump() == BASE


def test_a_mutation_can_carry_a_bounded_correlation_identity():
    mutation = MutationModel(**BASE, correlation_id="wl_example")
    assert mutation.model_dump(mode="json")["correlation_id"] == "wl_example"
    context = MutationContext(**BASE, correlation_id=mutation.correlation_id)
    assert context.correlation_id == "wl_example"


def test_correlation_does_not_change_mutation_identity_or_hash():
    original = MutationContext(**BASE)
    correlated = MutationContext(**BASE, correlation_id="wl_example")
    assert correlated == original
    assert hash(correlated) == hash(original)


def test_correlation_survives_agent_and_native_worker_adapters():
    from eyvan.worker.agent.models import WorkloadRequest
    from eyvan.worker.agent.service import _context as agent_context
    from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
    from eyvan.worker.runtime.native_worker import _context as worker_context

    mutation = MutationModel(**BASE, correlation_id="wl_example")
    agent = agent_context(CapacityRequest(mutation=mutation, desired={}))
    assert agent.correlation_id == "wl_example"
    forwarded = NativeRemoteExecutor._mutation(agent)
    worker = worker_context(WorkloadRequest(mutation=forwarded, spec={}))
    assert worker.correlation_id == "wl_example"


@pytest.mark.parametrize("value", ["", "bad\nidentity", "has spaces", "x" * 129])
def test_invalid_correlation_identity_is_rejected_at_the_wire_boundary(value):
    with pytest.raises(ValidationError):
        MutationModel(**BASE, correlation_id=value)
