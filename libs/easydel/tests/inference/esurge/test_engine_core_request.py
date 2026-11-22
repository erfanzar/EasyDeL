"""The documented engine wire request remains usable without model execution."""

import msgspec
import pytest
from easydel.inference.esurge.engine_types import EngineCoreRequest

pytestmark = pytest.mark.cpu_ok


def test_engine_core_request_wire_roundtrip():
    request = EngineCoreRequest(
        request_id="wire-request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=None,
        eos_token_id=2,
        arrival_time=42.0,
        data_parallel_rank=1,
        client_index=3,
        current_wave=4,
        priority=5,
    )
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=EngineCoreRequest)
    assert restored == request
    assert restored.prompt_token_ids == [1, 2, 3]
    assert restored.data_parallel_rank == 1
    assert restored.current_wave == 4
