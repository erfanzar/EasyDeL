# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Versioned wire models for control-plane to cell-agent operations."""

from __future__ import annotations

import re
import typing as tp
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic import JsonValue as PydanticJsonValue

from eyvan._internal.compat import StrEnum
from eyvan._internal.correlation_ids import CORRELATION_ID_WIRE_PATTERN
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode, ExecutionWorkloadStatus
from eyvan.records import MAX_CELL_STAGE_ASSET_LENGTH

WireJsonObject: tp.TypeAlias = dict[str, PydanticJsonValue]

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class AgentModel(BaseModel):
    """Strict base for every cell-agent request and response."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MutationConflictCode(StrEnum):
    """Machine-readable mutation conflicts returned by the cell agent."""

    DRIVER_BUSY = "driver_busy"
    STALE_MUTATION = "stale_mutation"


class MutationConflict(AgentModel):
    """Validated HTTP 409 response from a cell agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(min_length=1)
    code: MutationConflictCode


class CapacityProviderFailure(AgentModel):
    """Validated provider failure response from a cell agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detail: str = Field(min_length=1)
    code: CapacityProviderFailureCode


class AgentCapabilities(AgentModel):
    """Versioned execution and capacity capabilities exposed by one agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    executor: str
    accelerator_types: tuple[str, ...]
    capacity_classes: tuple[str, ...]
    #: Backends this agent can apply capacity for. Empty means the agent predates
    #: the field and must be skipped, not rejected.
    capacity_backends: tuple[str, ...] = ()
    supports_checkpoint: bool
    supports_callables: bool
    supports_exec: bool
    max_gang_replicas: int | None
    resource_enforcement: str


class AgentBuild(AgentModel):
    """What code one agent is made of.

    A separate route rather than a field on :class:`AgentCapabilities`, which
    forbids extras: an added field there would make a control plane one version
    behind unable to decode any capabilities response at all. A route an old
    caller never asks for costs it nothing, and a new caller meeting an old
    agent gets 404 -- which it records as unknown rather than as a guess.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    origin: str
    origin_detail: str = ""
    source_digest: str | None = None
    digest_detail: str = ""
    file_count: int = 0
    python: str
    platform: str


class NativeWorkerHealth(AgentModel):
    """Authenticated health response returned by an Eyvan native worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: tp.Literal["ok"]
    runtime: tp.Literal["native-worker"]


class MutationModel(AgentModel):
    idempotency_key: str = Field(min_length=1, max_length=256)
    generation: int = Field(ge=0)
    fencing_token: str = Field(min_length=1, max_length=512)
    correlation_id: str | None = Field(default=None, pattern=CORRELATION_ID_WIRE_PATTERN)

    @model_serializer(mode="wrap")
    def _wire_document(self, handler):
        document = handler(self)
        if self.correlation_id is None:
            document.pop("correlation_id", None)
        return document


class CapacityRequest(AgentModel):
    mutation: MutationModel
    desired: WireJsonObject


class WorkloadRequest(AgentModel):
    mutation: MutationModel
    spec: WireJsonObject


class GangWorkloadMember(AgentModel):
    """One independently fenced cell participating in a native workload gang."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1, max_length=96)
    agent_endpoint: str = Field(min_length=1, max_length=2048)
    mutation: MutationModel

    @model_validator(mode="after")
    def validate_values(self) -> GangWorkloadMember:
        if self.cell_id != self.cell_id.strip():
            raise ValueError("gang member cell_id must be trimmed")
        if self.agent_endpoint != self.agent_endpoint.strip():
            raise ValueError("gang member agent_endpoint must be trimmed")
        try:
            endpoint = urlsplit(self.agent_endpoint)
        except ValueError as exc:
            raise ValueError("gang member agent_endpoint must be a valid URL") from exc
        if not endpoint.scheme or not endpoint.netloc:
            raise ValueError("gang member agent_endpoint must be an absolute URL with a scheme and host")
        return self


class GangWorkloadRequest(AgentModel):
    """One workload submission coordinated across multiple native execution cells."""

    model_config = ConfigDict(extra="forbid")

    members: tuple[GangWorkloadMember, ...] = Field(min_length=2)
    spec: WireJsonObject

    @model_validator(mode="after")
    def validate_members(self) -> GangWorkloadRequest:
        cell_ids = tuple(member.cell_id for member in self.members)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("gang member cell IDs must be unique")
        endpoints = {member.agent_endpoint for member in self.members}
        if len(endpoints) != 1:
            raise ValueError("native workload gangs require every member to use the same agent_endpoint")
        generations = {member.mutation.generation for member in self.members}
        if len(generations) != 1:
            raise ValueError("native workload gang member generations must match")
        return self


class WorkloadMutationRequest(AgentModel):
    mutation: MutationModel


class GangWorkloadMutationRequest(AgentModel):
    """Fenced stop or retirement request for an existing native workload gang."""

    model_config = ConfigDict(extra="forbid")

    members: tuple[GangWorkloadMember, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_members(self) -> GangWorkloadMutationRequest:
        cell_ids = tuple(member.cell_id for member in self.members)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("gang member cell IDs must be unique")
        endpoints = {member.agent_endpoint for member in self.members}
        if len(endpoints) != 1:
            raise ValueError("native workload gangs require every member to use the same agent_endpoint")
        generations = {member.mutation.generation for member in self.members}
        if len(generations) != 1:
            raise ValueError("native workload gang member generations must match")
        return self


class CheckpointRequest(WorkloadMutationRequest):
    hook: str = Field(min_length=1)
    output_uri: str | None = None


class ExecRequest(AgentModel):
    """One diagnostic command to run beside a live workload.

    A list of arguments, never a shell string. A string would be split by a
    shell on the host, which turns a diagnostic into an injection surface the
    moment any part of it is templated.
    """

    command: tuple[str, ...] = Field(min_length=1)
    #: Enforced by systemd as the unit's RuntimeMaxSec, so the bound survives
    #: this agent dying -- a subprocess timeout would leave the unit running
    #: with nobody waiting for it.
    timeout_s: float = Field(default=30.0, gt=0, le=600)
    runtime_rank: int | None = Field(default=None, ge=0, le=9_999, strict=True)


class AgentObservation(AgentModel):
    cell_id: str
    state: CapacityStatus
    generation: int
    execution_endpoint: str | None = None
    detail: str = ""
    metadata: WireJsonObject = Field(default_factory=dict)
    observed_tpu_vms: int | None = Field(default=None, ge=1)
    # Optional for rolling compatibility with agents that predate durable log
    # rank width. The controller retains the last exact value when it is absent.
    runtime_process_count: int | None = Field(default=None, ge=1, le=10_000)
    #: Typed reason for a non-ready state, carried beside the human-readable
    #: ``detail``. Optional so an agent that predates it still validates.
    failure_code: CapacityProviderFailureCode | None = None


class WorkloadObservation(AgentModel):
    remote_id: str
    status: ExecutionWorkloadStatus
    message: str = ""
    checkpoint_uri: str | None = None


class WorkloadReference(AgentModel):
    """Minimal response returned after submission or checkpoint creation."""

    remote_id: str = Field(min_length=1)


class CellStageAction(StrEnum):
    """Which of the three staging operations one request asks for.

    One RPC with a named action rather than three routes, because all three
    share the same fence, the same asset and the same byte bound, and an agent
    that implements one implements all three. A control plane that finds the
    route missing therefore learns one fact -- this agent has no staging
    executor -- instead of three, which is what makes the rolling upgrade a
    single question.
    """

    #: Report what the asset would occupy, fetching nothing.
    MEASURE = "measure"
    #: Materialize the asset and report the bytes that are provably present.
    STAGE = "stage"
    #: Remove every copy of the asset from this host.
    EVICT = "evict"


class CellStageVerdict(StrEnum):
    """What a staging action established, in the only three shapes there are.

    Carried in a 200 body rather than mapped onto status codes, because the two
    failures mean opposite things to the control plane -- terminal versus
    transient -- and every status code that already carries meaning on this
    wire (409 stale, 501 unsupported, 5xx unreachable) is spoken for. A
    refusal is the agent answering, not the agent failing.
    """

    #: The action completed; the reported bytes are a fact about that disk.
    OK = "ok"
    #: This request can never succeed as written; do not retry it unchanged.
    REFUSED = "refused"
    #: Nothing was established. Presence is unproven in both directions.
    INCONCLUSIVE = "inconclusive"


class CellStageActionRequest(AgentModel):
    """One fenced, byte-bounded staging instruction for one cell.

    ``max_bytes`` is a term of the request, not a hint: it is the smaller of the
    control plane's per-action ceiling and the headroom left in this cell's
    budget, and an agent that honours it cannot overrun the budget.
    """

    model_config = ConfigDict(extra="forbid")

    mutation: MutationModel
    action: CellStageAction
    asset: str = Field(min_length=1, max_length=MAX_CELL_STAGE_ASSET_LENGTH)
    max_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_values(self) -> CellStageActionRequest:
        """Reject an asset name that is untrimmed or carries a NUL byte."""

        if self.asset != self.asset.strip():
            raise ValueError("a staged asset must be trimmed")
        if "\x00" in self.asset:
            raise ValueError("a staged asset must not contain NUL bytes")
        return self


class NativeStageActionRequest(AgentModel):
    """One byte-bounded staging instruction for the native worker that owns a disk.

    The same three terms as :class:`CellStageActionRequest` and deliberately not
    its fence. A native worker holds no cell record, so it has no generation to
    compare against, and carrying one it could only ignore would make the hop
    look fenced twice when it is fenced once. The fence is the agent's: it
    checks the request against its own durable capacity record before
    forwarding. The bearer credential on this hop is a second, narrower guard --
    :func:`~eyvan.worker.runtime.native_bootstrap.derive_native_worker_token`
    derives it per provider generation, so a worker rebuilt under a new one does
    not accept a credential minted for the old.
    """

    model_config = ConfigDict(extra="forbid")

    action: CellStageAction
    asset: str = Field(min_length=1, max_length=MAX_CELL_STAGE_ASSET_LENGTH)
    max_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_values(self) -> NativeStageActionRequest:
        """Reject an asset name that is untrimmed or carries a NUL byte."""

        if self.asset != self.asset.strip():
            raise ValueError("a staged asset must be trimmed")
        if "\x00" in self.asset:
            raise ValueError("a staged asset must not contain NUL bytes")
        return self


class CellStageActionResult(AgentModel):
    """What one staging action established on one host.

    ``size_bytes`` and ``digest`` describe bytes read back off that host's disk.
    Neither is ever the size or digest the asset *declared*: the control plane
    charges its budget against this number and scores placement on it, so an
    expectation reported as an observation is a lie it would then act on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset: str = Field(min_length=1, max_length=MAX_CELL_STAGE_ASSET_LENGTH)
    action: CellStageAction
    verdict: CellStageVerdict
    size_bytes: int = Field(default=0, ge=0)
    #: ``sha256:<hex>`` over the observed bytes, or ``None`` when the action did
    #: not read them. ``None`` is unknown content, never "no content".
    digest: str | None = None
    detail: str = Field(default="", max_length=2048)
    #: Where the asset lives on that host, when it is present.
    path: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def validate_values(self) -> CellStageActionResult:
        """Reject a malformed digest and bytes claimed without a verdict to hold them."""

        if self.digest is not None and _SHA256_DIGEST.fullmatch(self.digest) is None:
            raise ValueError("a stage digest must be sha256:<64 lowercase hex characters>")
        if self.verdict is not CellStageVerdict.OK and (self.size_bytes or self.digest is not None):
            raise ValueError("only a completed stage action may report observed bytes")
        return self
