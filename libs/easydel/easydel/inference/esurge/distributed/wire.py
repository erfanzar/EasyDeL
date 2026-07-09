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

"""Wire protocol for the eSurge multi-host step-coordination plane.

Messages travel as ZeroMQ multipart frames: a msgspec-encoded header (one of
the tagged structs below) optionally followed by a payload frame carrying a
pickled ``SchedulerOutput``. Authentication and routing decisions are made
from the header alone — a peer that never authenticated has its payload
dropped *before* any unpickling happens.

The payload stays pickle-encoded (``PAYLOAD_CODEC_PICKLE``): per-step decode
payloads are small parallel lists whose serialization cost is microseconds
against a multi-millisecond device step, and a full msgspec schema would
have to mirror SamplingParams and numpy arrays for no measurable gain. The
``payload_codec`` field on :class:`Step` is the seam for a future migration.
"""

from __future__ import annotations

import pickle
import typing

import msgspec

PAYLOAD_CODEC_PICKLE = "pickle.v1"

STEP_MODE_SYNC = 0
STEP_MODE_ASYNC = 1

ACK_PHASE_SYNC_DONE = 0
ACK_PHASE_DRAINED = 1


class Hello(msgspec.Struct, tag=True):
    """Worker's first message: identity + config proof.

    Attributes:
        rank: The worker's ``jax.process_index()``.
        world_size: The worker's ``jax.process_count()``.
        config_fingerprint: SHA-256 fingerprint of the engine config.
        auth: Shared auth token.
    """

    rank: int
    world_size: int
    config_fingerprint: str
    auth: str


class HelloOk(msgspec.Struct, tag=True):
    """Leader's acceptance of a worker's :class:`Hello`."""

    rank: int


class Ready(msgspec.Struct, tag=True):
    """Worker signal that its runner is compiled and KV cache allocated."""

    rank: int


class Step(msgspec.Struct, tag=True):
    """One step of the replicated program (payload frame carries the plan).

    Attributes:
        step_id: Monotonic step index.
        mode: ``STEP_MODE_SYNC`` or ``STEP_MODE_ASYNC``.
        payload_codec: Encoding of the payload frame.
    """

    step_id: int
    mode: int
    payload_codec: str = PAYLOAD_CODEC_PICKLE
    want_digest: bool = False


class Drain(msgspec.Struct, tag=True):
    """Materialize a previously async-dispatched step.

    Attributes:
        step_id: The step to drain.
        want_digest: Whether the ack must carry a sampled-token digest.
    """

    step_id: int
    want_digest: bool = False


class StepAck(msgspec.Struct, tag=True):
    """Worker acknowledgement (or failure report) for a step phase.

    Attributes:
        rank: Acking worker rank.
        step_id: The acknowledged step.
        phase: ``ACK_PHASE_SYNC_DONE`` or ``ACK_PHASE_DRAINED``.
        ok: ``False`` turns this into a NACK.
        num_reqs: Requests in the worker's model output (cross-check).
        digest: Sampled-token digest when requested.
        error: Error message on NACK.
        traceback: Formatted traceback on NACK.
    """

    rank: int
    step_id: int
    phase: int
    ok: bool
    num_reqs: int = -1
    digest: str | None = None
    error: str | None = None
    traceback: str | None = None


class Heartbeat(msgspec.Struct, tag=True):
    """Liveness beacon (both directions).

    Attributes:
        rank: Sender rank.
        ts: Sender wall-clock timestamp.
    """

    rank: int
    ts: float


class Shutdown(msgspec.Struct, tag=True):
    """Orderly teardown notice.

    Attributes:
        reason: Human-readable teardown cause.
    """

    reason: str = ""


class Control(msgspec.Struct, tag=True):
    """Reserved control channel (weight swaps, profiling, future ingress).

    Attributes:
        kind: Operation discriminator.
        args: Operation arguments.
    """

    kind: str
    args: dict[str, typing.Any] = {}


WireMessage = Hello | HelloOk | Ready | Step | Drain | StepAck | Heartbeat | Shutdown | Control

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(WireMessage)


def encode_message(message: WireMessage) -> bytes:
    """Encode a wire header to bytes."""
    return _encoder.encode(message)


def decode_message(data: bytes) -> WireMessage:
    """Decode a wire header from bytes.

    Raises:
        msgspec.DecodeError: If the bytes are not a valid wire message.
    """
    return _decoder.decode(data)


def encode_payload(scheduler_output) -> bytes:
    """Serialize a scheduler output for the payload frame."""
    return pickle.dumps(scheduler_output, protocol=pickle.HIGHEST_PROTOCOL)


def decode_payload(data: bytes):
    """Deserialize a payload frame produced by :func:`encode_payload`.

    Only call this for peers that already passed :class:`Hello`
    authentication — unpickling is the trust boundary.
    """
    return pickle.loads(data)
