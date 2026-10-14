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


ROLE_WORKER = "worker"
ROLE_CLIENT = "client"


class Hello(msgspec.Struct, tag=True):
    """A peer's first message: identity + config proof.

    Attributes:
        rank: The worker's ``jax.process_index()`` (``-1`` for clients).
        world_size: The worker's ``jax.process_count()`` (``0`` for clients).
        config_fingerprint: SHA-256 fingerprint of the engine config. Clients
            send ``""`` — they attach to whatever engine the owner runs.
        auth: Shared auth token.
        role: ``ROLE_WORKER`` for in-group step-replay ranks (counted toward
            the Ready barrier), ``ROLE_CLIENT`` for request-plane clients
            (remote engine handles, routers) that only admit requests and
            receive outputs.
        client_id: Stable identifier for ``ROLE_CLIENT`` peers (unused for
            workers).
    """

    rank: int
    world_size: int
    config_fingerprint: str
    auth: str
    role: str = ROLE_WORKER
    client_id: str = ""


class EngineSpec(msgspec.Struct):
    """Owner-engine facts a request-plane client needs to admit remotely.

    Attributes:
        esurge_name: The owner engine's display name.
        max_model_len: Context limit (prompt + generation).
        max_num_seqs: Concurrent-request limit.
        reserve_tokens: Tokens reserved for generation at admission.
        eos_token_ids: Combined engine EOS ids, in priority order.
        tokenizer_source: Id/path clients can load a matching tokenizer from.
        page_size: KV page size in tokens (prefix-affinity block alignment).
    """

    esurge_name: str = ""
    max_model_len: int = 0
    max_num_seqs: int = 0
    reserve_tokens: int = 0
    eos_token_ids: list[int] = []
    tokenizer_source: str = ""
    page_size: int = 0


class HelloOk(msgspec.Struct, tag=True):
    """Leader's acceptance of a peer's :class:`Hello`.

    Attributes:
        rank: Echo of the accepted worker rank (``-1`` for clients).
        engine_spec: Owner-engine facts for request-plane clients.
    """

    rank: int
    engine_spec: EngineSpec | None = None


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


class Admit(msgspec.Struct, tag=True):
    """Origin → owner: admit a request group into the owner's scheduler.

    The payload frame carries the origin-built ``EngineRequest`` objects
    (pickled) — post-truncation token ids, finalized sampling params, and the
    ``n>1`` child fan-out — so the owner schedules exactly what a local
    admission would have produced. The owner renames the request ids into its
    own id space and streams raw token deltas back to the origin.

    Attributes:
        request_id: The origin's parent request id (its local registry key).
        origin_rank: Origin process rank (identifies the return path).
        origin_counter: Origin-plane monotonic admission counter (all
            admissions; keeps owner-side ids unique per rank).
        content_hash: :func:`compute_admission_key` over (tokens, params, n);
            the coalescing key component and owner-id suffix source.
        coalescable: Whether identical cross-rank admissions may share one
            scheduled request (deterministic-counter trainer traffic).
        coalesce_counter: Per-rank sequence counting only *coalescable*
            admissions — the coalescing key's counter component. Kept
            separate from ``origin_counter`` so interleaved non-coalescable
            traffic on one rank cannot desynchronize the key sequence.
        n_samples: Parallel-sampling fan-out (validation aid).
        payload_codec: Encoding of the payload frame.
    """

    request_id: str
    origin_rank: int
    origin_counter: int
    content_hash: str = ""
    coalescable: bool = False
    coalesce_counter: int = 0
    n_samples: int = 1
    payload_codec: str = PAYLOAD_CODEC_PICKLE


class AdmitAck(msgspec.Struct, tag=True):
    """Owner → origin: acceptance (or rejection) of an :class:`Admit`.

    Attributes:
        request_id: Echo of the origin's parent request id.
        ok: ``False`` turns this into a NACK.
        error: Failure description on NACK.
        already_finished: Coalesced attach to a group that already finished.
        payload_codec: Encoding of the optional payload frame. A coalesced
            attach to a group that already produced tokens carries a
            catch-up payload: the ``(outputs, finished_requests)`` tuple
            replaying the group's token log in the subscriber's id space.
    """

    request_id: str
    ok: bool
    error: str | None = None
    already_finished: bool = False
    payload_codec: str = PAYLOAD_CODEC_PICKLE


class OutputUpdate(msgspec.Struct, tag=True):
    """Owner → origin: one step's raw token deltas for remotely-admitted requests.

    The payload frame carries a pickled ``(outputs, finished_requests)``
    tuple: a list of ``EngineCoreOutput`` renamed into the origin's id space,
    and the origin-mapped scheduler-side finished ids (terminations that
    carry no terminal output). The origin re-stamps the bundle with its own
    clock — ``perf_counter`` is process-local — and feeds its unchanged
    :class:`OutputPipeline`.

    Attributes:
        seq: Owner-plane monotonic update sequence (gap detection aid).
        payload_codec: Encoding of the payload frame.
    """

    seq: int
    payload_codec: str = PAYLOAD_CODEC_PICKLE


class AbortReq(msgspec.Struct, tag=True):
    """Origin → owner: cancel a remotely-admitted request.

    Attributes:
        request_id: Origin-side request id (parent aborts every ``n>1``
            child; a child id aborts one sample).
    """

    request_id: str


class StopHit(msgspec.Struct, tag=True):
    """Origin → owner: parser-detected stop-string matches.

    The origin renders text locally, so stop strings are discovered there;
    the owner must still free the scheduled rows. Mirrors the engine's
    internal ``on_stop_strings`` payload.

    Attributes:
        hits: Origin-side sample-level request id → matched stop string.
    """

    hits: dict[str, str] = {}


class QueueStats(msgspec.Struct, tag=True):
    """Owner → clients: scheduler load snapshot for join-shortest-queue routing.

    Piggybacked on the leader's heartbeat cadence to request-plane clients.
    Values are best-effort reads of live scheduler state (no lock taken on
    the owner) — routing tolerates slight staleness.

    Attributes:
        num_waiting: Requests queued but not yet running.
        num_running: Requests currently scheduled.
        pending_prefill_tokens: Prompt tokens not yet computed across the
            waiting/running sets — the JSQ cost signal.
    """

    num_waiting: int = 0
    num_running: int = 0
    pending_prefill_tokens: int = 0


WireMessage = (
    Hello
    | HelloOk
    | Ready
    | Step
    | Drain
    | StepAck
    | Heartbeat
    | Shutdown
    | Control
    | Admit
    | AdmitAck
    | OutputUpdate
    | AbortReq
    | StopHit
    | QueueStats
)

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


def encode_payload(obj) -> bytes:
    """Serialize a payload-frame object (scheduler outputs, admits, deltas)."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def decode_payload(data: bytes):
    """Deserialize a payload frame produced by :func:`encode_payload`.

    Only call this for peers that already passed :class:`Hello`
    authentication — unpickling is the trust boundary.
    """
    return pickle.loads(data)
