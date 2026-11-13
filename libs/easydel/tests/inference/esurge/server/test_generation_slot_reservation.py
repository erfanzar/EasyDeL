"""Pre-flight generation-slot check: an overloaded server rejects before the SSE stream starts."""

import asyncio
from types import SimpleNamespace

import pytest
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from fastapi import HTTPException

pytestmark = pytest.mark.cpu_ok


def _server(slots: int) -> SimpleNamespace:
    queue = asyncio.Queue(slots)
    for token in range(slots):
        queue.put_nowait(token)
    return SimpleNamespace(
        _generation_slots=queue,
        _max_generation_slots=slots,
        _overload_message="Server is busy, please try again later",
        _active_requests=set(),
    )


def _check(server) -> None:
    BaseInferenceApiServer._check_generation_slot_capacity(
        server, endpoint="/v1/chat/completions", request_id="r1", model="m", raw_request=None, stream=True
    )


def test_a_free_slot_passes_the_pre_flight_check_without_consuming_it():
    server = _server(2)
    _check(server)
    assert server._generation_slots.qsize() == 2, "the pre-flight check must not consume a slot"


def test_a_saturated_server_is_rejected_with_503_before_streaming():
    server = _server(1)
    server._generation_slots.get_nowait()
    with pytest.raises(HTTPException) as rejected:
        _check(server)
    assert rejected.value.status_code == 503
    assert rejected.value.detail == server._overload_message


def test_an_unlimited_or_partially_built_server_never_rejects():
    for server in (
        SimpleNamespace(
            _generation_slots=None, _max_generation_slots=0, _overload_message="busy", _active_requests=set()
        ),
        SimpleNamespace(_overload_message="busy"),
    ):
        _check(server)
