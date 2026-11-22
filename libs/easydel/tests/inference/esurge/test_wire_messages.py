# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wire-protocol regression tests (pure serialization; no array computation)."""

from __future__ import annotations

import msgspec
import pytest
from easydel.inference.esurge.distributed import wire


@pytest.mark.cpu_ok
def test_wire_message_round_trip():
    messages = [
        wire.Hello(rank=0, world_size=1, config_fingerprint="abc", auth="token"),
        wire.HelloOk(rank=0, engine_spec=wire.EngineSpec(esurge_name="e", max_model_len=8)),
        wire.Ready(rank=1),
        wire.Step(step_id=3, mode=wire.STEP_MODE_SYNC),
        wire.Drain(step_id=3, want_digest=True),
        wire.StepAck(rank=1, step_id=3, phase=wire.ACK_PHASE_SYNC_DONE, ok=True, num_reqs=2),
        wire.Heartbeat(rank=0, ts=1.5),
        wire.Shutdown(reason="done"),
        wire.Control(kind="ping", args={"x": 1}),
        wire.Admit(request_id="r0", origin_rank=-1, origin_counter=0),
        wire.AdmitAck(request_id="r0", ok=True),
        wire.OutputUpdate(seq=7),
        wire.AbortReq(request_id="r0"),
        wire.StopHit(hits={"r0": "</s>"}),
        wire.QueueStats(num_waiting=1, num_running=2, pending_prefill_tokens=3),
    ]

    for message in messages:
        decoded = wire.decode_message(wire.encode_message(message))
        assert type(decoded) is type(message)
        assert decoded == message


@pytest.mark.cpu_ok
def test_unknown_tag_is_rejected_at_decode_time():
    """Tags outside the ``WireMessage`` union must fail decoding loudly."""
    frame = msgspec.msgpack.encode({"type": "NotARealMessage", "kind": "noop"})

    with pytest.raises(msgspec.DecodeError):
        wire.decode_message(frame)
