# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Getting a secret value to a running attempt without persisting it anywhere.

Three pieces have to line up, and each of them is a place where this could be
built, tested, and still never happen to a real workload:

* dispatch mints one capability per mount and injects the address to redeem it
  at -- if it does not, the route is reachable by nothing;
* the route resolves a mount only for the workload's **current** attempt -- if it
  does not, the credential is a bearer token with a TTL and nothing more;
* the cell agent redeems the capabilities and writes the values into the
  owner-only environment file it already writes -- if it does not, the job sees
  a URL and a token instead of its secret.

Every test here asserts one of those seams rather than the pieces in isolation.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import timedelta

import pytest
from test_secret_dispatch_transactions import repository as _dispatch_repository_fixture

from eyvan.artifacts._capability_tokens import InvalidCapabilityToken, sign_claims
from eyvan.artifacts.secret_capability import (
    SECRET_TOKEN_VERSION,
    InvalidSecretToken,
    SecretCapabilitySigner,
)
from eyvan.artifacts.workload_capability import WorkloadCapabilitySigner
from eyvan.secrets import MAX_SECRET_BYTES
from eyvan.worker.runtime.secret_materialization import (
    SecretMaterializationError,
    materialize_secrets,
    secret_environment_keys,
)

SECRET = b"k" * 32
DISPATCH_FENCE = "dispatch_test_fence"
dispatch_repository = _dispatch_repository_fixture


def _signer(*, ttl_seconds: float = 3600, clock=time.time) -> SecretCapabilitySigner:
    return SecretCapabilitySigner(SECRET, ttl_seconds=ttl_seconds, clock=clock)


def _issue(
    signer: SecretCapabilitySigner,
    *,
    attempt: int = 1,
    generation: int = 1,
    dispatch_fence: str = DISPATCH_FENCE,
    name: str = "HF_TOKEN",
) -> str:
    return signer.issue(
        workload_id="wl-1",
        org_id="acme",
        project_id="research",
        attempt=attempt,
        generation=generation,
        dispatch_fence=dispatch_fence,
        name=name,
    )


# --------------------------------------------------------------------------- the token


def test_a_secret_token_cannot_verify_as_a_submit_capability() -> None:
    """Audience separation, and the reason the prefix is inside the signed material.

    Same deployment key, so a claim check is the only thing between the two --
    and a claim check is exactly what should not be load-bearing here.
    """

    token = _issue(_signer())

    # The base class, not the secret-specific one: the prefix is checked in the
    # shared primitive, before any signer's claim rules run. That is the point --
    # the separation does not depend on either signer being careful.
    with pytest.raises(InvalidCapabilityToken, match="invalid capability token"):
        WorkloadCapabilitySigner(SECRET).verify(token, workload_id="wl-1", generation=1)


def test_a_submit_capability_cannot_verify_as_a_secret_token() -> None:
    """The same separation in the direction that would leak a value."""

    token = WorkloadCapabilitySigner(SECRET).issue(
        workload_id="wl-1", org_id="acme", project_id="research", generation=1
    )

    with pytest.raises(InvalidCapabilityToken, match="invalid capability token"):
        _signer().verify(token, workload_id="wl-1", org_id="acme", project_id="research")


def test_a_secret_token_expires() -> None:
    """The one capability that must expire, because a read cannot be undone.

    A submit capability is safe unexpiring: the control plane re-authorizes the
    action on every use. A secret capability hands back a value, and no later
    revocation un-reads it.
    """

    now = [1_000.0]
    signer = _signer(ttl_seconds=60, clock=lambda: now[0])
    token = _issue(signer)

    signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research")
    now[0] += 61

    with pytest.raises(InvalidSecretToken, match="expired"):
        signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research")


def test_a_token_for_another_project_is_refused() -> None:
    """Scope is checked against the request, not merely carried in the claims."""

    token = _issue(_signer())

    with pytest.raises(InvalidSecretToken, match="different project"):
        _signer().verify(token, workload_id="wl-1", org_id="acme", project_id="platform")


def test_an_added_claim_is_refused_rather_than_ignored() -> None:
    """Honoring the claims a build recognizes is how a capability quietly widens."""

    signer = _signer()
    payload = _issue(signer).split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["scope"] = "everything"
    # Signed with the real key, so the signature check passes and the claim-set
    # check is what has to refuse it. An unsigned forgery would prove only that
    # HMAC works.
    forged = sign_claims(SECRET, SECRET_TOKEN_VERSION, claims)

    with pytest.raises(InvalidSecretToken, match="unexpected claims"):
        signer.verify(forged, workload_id="wl-1", org_id="acme", project_id="research")


def test_the_token_names_a_mount_and_not_a_reference() -> None:
    """A reference in the token would outlive the spec that authorized it.

    An operator who corrected a mount would find the old reference still
    readable until the capability expired.
    """

    token = _issue(_signer())
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))

    assert claims["name"] == "HF_TOKEN"
    assert not any("gcp-secret" in str(value) for value in claims.values())
    assert token.startswith(f"{SECRET_TOKEN_VERSION}.")


# --------------------------------------------------------------------------- dispatch


async def test_a_workload_with_mounts_receives_capabilities_at_dispatch(tmp_path) -> None:
    """The seam that makes the route reachable from a real cell.

    Without it the capability is mintable, the route accepts it, and nothing
    running has ever been handed one.
    """

    environment, workload, signer = await _dispatch_with_mounts(tmp_path)

    assert environment["EYVAN_SECRET_COUNT"] == "1"
    assert environment["EYVAN_SECRET_0_NAME"] == "HF_TOKEN"
    assert environment["EYVAN_SECRET_0_URL"].startswith("https://eyvan.test/api/v1/secrets/content?")
    assert f"workload_id={workload.id}" in environment["EYVAN_SECRET_0_URL"]
    claims = signer.verify(
        environment["EYVAN_SECRET_0_TOKEN"],
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
    )
    assert claims.attempt == workload.attempt
    assert claims.generation == workload.generation
    assert workload.dispatch_reservation is not None
    assert claims.dispatch_fence == workload.dispatch_reservation.fence


async def test_the_capability_carries_the_attempt_being_dispatched(tmp_path) -> None:
    """Not attempt 1, and not a constant.

    Written with an attempt that is *not* the default on purpose: a test whose
    fixture uses the default value of the field under test cannot tell "read the
    field" from "hardcode the default", and would pass against a capability that
    was permanently pinned to the first attempt.
    """

    environment, workload, signer = await _dispatch_with_mounts(tmp_path, attempt=4)

    assert workload.attempt == 4
    claims = signer.verify(
        environment["EYVAN_SECRET_0_TOKEN"],
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
    )
    assert claims.attempt == 4


async def test_no_secret_value_reaches_the_dispatch(tmp_path) -> None:
    """The property the whole design exists for.

    The agent request is persisted in an outbox and echoed into logs; a value in
    it would be a value in both.
    """

    environment, _workload, _signer = await _dispatch_with_mounts(tmp_path)

    assert not any("hf_abcdef" in value for value in environment.values())
    assert "gcp-secret://my-project/training-hf-token" not in json.dumps(environment)


async def test_a_workload_without_mounts_receives_no_scaffolding(tmp_path) -> None:
    """Every existing workload's dispatch environment stays as it was."""

    environment, _workload, _signer = await _dispatch_with_mounts(tmp_path, mounts={})

    assert not [key for key in environment if key.startswith("EYVAN_SECRET_")]


async def test_dispatch_records_which_mounts_were_granted(tmp_path) -> None:
    """One issuance event per fenced dispatch, before the agent RPC.

    It is the pairing that carries the signal: a ``secret.read`` for a dispatch
    with no matching ``secret.issued`` is a capability being used outside the
    reservation that minted it.
    """

    _environment, workload, _signer = await _dispatch_with_mounts(tmp_path)
    events = await _events(workload.id)
    issued = [event for event in events if event.action == "secret.issued"]

    assert len(issued) == 1
    assert issued[0].payload["names"] == ["HF_TOKEN"]
    assert issued[0].payload["attempt"] == workload.attempt


@pytest.mark.parametrize("dispatch_repository", ["memory", "postgres", "local"], indirect=True)
@pytest.mark.parametrize("gang", [False, True])
async def test_accepted_but_unobserved_dispatch_keeps_its_secret_readable(dispatch_repository, tmp_path, gang):
    """A lost response is not evidence that the admitted worker stopped.

    No absence observation is invented: the worker still owns the accepted
    execution while the outbox is FAILED and its delivery claim is released.
    """
    from test_secret_dispatch_transactions import _plane as dispatch_plane
    from test_secret_dispatch_transactions import _seed_dispatch, _snapshot

    from eyvan.records import LeaseState, OutboxState, WorkloadState
    from eyvan.server.persistence.local import LocalJsonRepository

    service, signer, agent, controller = await dispatch_plane(dispatch_repository, tmp_path, fail=True)
    seeded, _leases, _command = await _seed_dispatch(dispatch_repository, gang=gang)
    assert await controller.dispatch_outbox_once() == 1
    workload, leases, commands, events = await _snapshot(dispatch_repository, seeded.id)
    assert workload.state is WorkloadState.LEASED and workload.remote_id is None
    assert workload.dispatch_reservation is not None
    assert all(lease.state is LeaseState.PENDING for lease in leases)
    dispatch = next(command for command in commands if command.topic == "workload.dispatch")
    assert dispatch.state is OutboxState.FAILED
    assert dispatch.claimed_by is None and dispatch.claimed_until is None
    assert len(agent.executions) == len(agent.values) == 1
    assert not any(event.action == "workload.dispatch_reservation_failed" for event in events)
    if isinstance(dispatch_repository, LocalJsonRepository):
        await dispatch_repository.close()
        await dispatch_repository.open()
    claims = signer.verify(agent.tokens[0], workload_id=seeded.id, org_id="acme", project_id="research")
    assert claims.dispatch_fence == workload.dispatch_reservation.fence
    assert await service.authorize_secret_mount(
        org_id="acme",
        project_id="research",
        workload_id=seeded.id,
        attempt=claims.attempt,
        generation=claims.generation,
        dispatch_fence=claims.dispatch_fence,
        name=claims.name,
    ) == ("file:///operator/secrets/hf-token", "file:///operator/secrets")


# --------------------------------------------------------------------------- the cell


def test_the_agent_resolves_declared_mounts_into_the_launch_environment() -> None:
    """What the job actually ends up seeing."""

    resolved = materialize_secrets(
        {
            "EYVAN_SECRET_COUNT": "2",
            "EYVAN_SECRET_0_NAME": "HF_TOKEN",
            "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content?name=HF_TOKEN",
            "EYVAN_SECRET_0_TOKEN": "eyvs1.a.b",
            "EYVAN_SECRET_1_NAME": "WANDB_KEY",
            "EYVAN_SECRET_1_URL": "https://eyvan.test/api/v1/secrets/content?name=WANDB_KEY",
            "EYVAN_SECRET_1_TOKEN": "eyvs1.c.d",
        },
        fetch=lambda url, _token: b"value-for-" + url.rsplit("=", 1)[1].encode(),
    )

    assert resolved == {"HF_TOKEN": "value-for-HF_TOKEN", "WANDB_KEY": "value-for-WANDB_KEY"}


def test_the_worker_accepts_exactly_one_mibibyte_and_rejects_one_byte_more() -> None:
    environment = {
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": "HF_TOKEN",
        "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
        "EYVAN_SECRET_0_TOKEN": "eyvs1.capability.signature",
    }

    exact = materialize_secrets(environment, fetch=lambda _url, _token: b"x" * MAX_SECRET_BYTES)
    assert exact["HF_TOKEN"] == "x" * MAX_SECRET_BYTES

    with pytest.raises(SecretMaterializationError, match="exceeds") as captured:
        materialize_secrets(environment, fetch=lambda _url, _token: b"x" * (MAX_SECRET_BYTES + 1))
    assert "x" * 64 not in str(captured.value)


def test_the_worker_reads_one_byte_past_the_limit_instead_of_slicing(monkeypatch) -> None:
    from eyvan.worker.runtime import secret_materialization

    requested: list[int] = []

    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, amount):
            requested.append(amount)
            return b"s" * amount

    monkeypatch.setattr(secret_materialization.urllib.request, "urlopen", lambda *_args, **_kwargs: OversizedResponse())

    with pytest.raises(SecretMaterializationError, match="exceeds"):
        materialize_secrets(
            {
                "EYVAN_SECRET_COUNT": "1",
                "EYVAN_SECRET_0_NAME": "HF_TOKEN",
                "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
                "EYVAN_SECRET_0_TOKEN": "eyvs1.capability.signature",
            }
        )

    assert requested == [MAX_SECRET_BYTES + 1]


def test_resolver_exceptions_cannot_reflect_a_value_or_capability() -> None:
    value = "hf_adversarial_value"
    capability = "eyvs1.adversarial.signature"

    def fetch(_url, _token):
        raise RuntimeError(f"provider reflected {value} and {capability}")

    with pytest.raises(SecretMaterializationError) as captured:
        materialize_secrets(
            {
                "EYVAN_SECRET_COUNT": "1",
                "EYVAN_SECRET_0_NAME": "HF_TOKEN",
                "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
                "EYVAN_SECRET_0_TOKEN": capability,
            },
            fetch=fetch,
        )

    assert value not in str(captured.value)
    assert capability not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    "name",
    [
        "BAD-NAME",
        "PATH",
        "JAX_PROCESS_ID",
        "EYVAN_SECRET_9_TOKEN",
        "EYVAN_CONTAINER_RUNTIME",
        "EYVAN_CHECKPOINT_URI",
        "EYVAN_PROFILE_URI",
        "EYVAN_SOURCE_JOB_ID",
        "EYVAN_MANAGED_CALLABLE_TOKEN",
        "EYVAN_MANAGED_WORKSPACE_TOKEN",
        "EYVAN_MANAGED_RESULT_TOKEN",
        "EYVAN_WORKLOAD_TOKEN",
        "EYVAN_FUTURE_RUNTIME_FIELD",
    ],
)
def test_worker_defensively_rejects_invalid_or_reserved_mount_names(name: str) -> None:
    environment = {
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": name,
        "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
        "EYVAN_SECRET_0_TOKEN": "eyvs1.name.capability",
    }

    with pytest.raises(SecretMaterializationError):
        materialize_secrets(environment, fetch=lambda _url, _token: b"value")


@pytest.mark.parametrize("payload", [b"value\nINJECTED=1", b"value\rINJECTED=1", b"value\x00suffix"])
def test_worker_rejects_env_file_injection_bytes(payload: bytes) -> None:
    environment = {
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": "HF_TOKEN",
        "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
        "EYVAN_SECRET_0_TOKEN": "eyvs1.injection.capability",
    }

    with pytest.raises(SecretMaterializationError, match="cannot be represented") as captured:
        materialize_secrets(environment, fetch=lambda _url, _token: payload)

    assert payload.decode(errors="ignore") not in str(captured.value)


def test_worker_rejects_a_secret_name_colliding_with_the_launch_environment() -> None:
    environment = {
        "HF_TOKEN": "ordinary-value",
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": "HF_TOKEN",
        "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
        "EYVAN_SECRET_0_TOKEN": "eyvs1.collision.capability",
    }

    with pytest.raises(SecretMaterializationError, match="collides"):
        materialize_secrets(environment, fetch=lambda _url, _token: b"secret-value")


def test_the_capability_does_not_survive_into_the_job() -> None:
    """A job receives the value it asked for, not the means to ask again.

    Otherwise a leaked process environment hands over a working credential for
    the rest of the attempt rather than one secret.
    """

    environment = {
        "EYVAN_SECRET_COUNT": "1",
        "EYVAN_SECRET_0_NAME": "HF_TOKEN",
        "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
        "EYVAN_SECRET_0_TOKEN": "eyvs1.a.b",
        "EYVAN_WORKLOAD_ID": "wl-1",
    }

    dropped = secret_environment_keys(environment)

    assert set(dropped) == {
        "EYVAN_SECRET_COUNT",
        "EYVAN_SECRET_0_NAME",
        "EYVAN_SECRET_0_URL",
        "EYVAN_SECRET_0_TOKEN",
    }
    assert "EYVAN_WORKLOAD_ID" not in dropped


def test_an_unresolvable_mount_refuses_the_launch() -> None:
    """A job that starts without the credential it declared fails further away.

    Usually as an authentication error against someone else's service, which is
    a much longer path to the actual cause.
    """

    def fetch(_url, _token):
        raise RuntimeError("403")

    with pytest.raises(SecretMaterializationError, match="HF_TOKEN"):
        materialize_secrets(
            {
                "EYVAN_SECRET_COUNT": "1",
                "EYVAN_SECRET_0_NAME": "HF_TOKEN",
                "EYVAN_SECRET_0_URL": "https://eyvan.test/x",
                "EYVAN_SECRET_0_TOKEN": "eyvs1.a.b",
            },
            fetch=fetch,
        )


def test_a_launch_with_no_mounts_fetches_nothing() -> None:
    """Every existing launch keeps its behaviour, including making no requests."""

    def fetch(_url, _token):
        raise AssertionError("a launch with no declared mounts must not call the control plane")

    assert materialize_secrets({"EYVAN_WORKLOAD_ID": "wl-1"}, fetch=fetch) == {}


def test_the_systemd_launcher_writes_resolved_values_and_not_capabilities(tmp_path, monkeypatch) -> None:
    """The wiring, not the helper.

    ``materialize_secrets`` passing its own tests proves nothing about whether a
    launch calls it -- which is the shape half of this package's defects have
    taken.
    """

    from eyvan.worker.runtime import systemd_transient

    monkeypatch.setattr(
        systemd_transient,
        "materialize_secrets",
        lambda environment: {"HF_TOKEN": "hf_resolved"} if environment.get("EYVAN_SECRET_COUNT") else {},
    )
    executor = systemd_transient.SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=lambda *_args, **_kwargs: None,
        boot_id_reader=lambda: "boot-a",
    )

    executor._prepare_launch(
        {"remote_id": "eyvan-wl-1"},
        {
            "environment": {
                "EYVAN_SECRET_COUNT": "1",
                "EYVAN_SECRET_0_NAME": "HF_TOKEN",
                "EYVAN_SECRET_0_URL": "https://eyvan.test/x",
                "EYVAN_SECRET_0_TOKEN": "eyvs1.a.b",
                "EYVAN_WORKLOAD_ID": "wl-1",
            }
        },
    )

    environment_file = executor._environment_file_path("eyvan-wl-1")
    written = environment_file.read_text(encoding="utf-8")
    assert "HF_TOKEN=hf_resolved\n" in written
    assert "EYVAN_WORKLOAD_ID=wl-1\n" in written
    assert "EYVAN_SECRET_" not in written and "eyvs1.a.b" not in written
    assert environment_file.stat().st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------- helpers

_LAST_REPOSITORY: list[object] = []


async def _dispatch_with_mounts(tmp_path, mounts: dict[str, str] | None = None, *, attempt: int = 1):
    """Dispatch one workload through the real controller and return its environment."""

    from test_controller import FakeAgent, principal

    from eyvan.artifacts.managed import ArtifactTokenSigner, LocalManagedArtifactStore
    from eyvan.artifacts.transfers import ManagedArtifactAccess
    from eyvan.config import EyvanSettings
    from eyvan.records import (
        CellRecord,
        CellState,
        CommandExecution,
        PolicyRecord,
        ResourceVector,
        Role,
        WorkloadDefinition,
    )
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence.memory import InMemoryRepository
    from eyvan.server.service.control_plane import ControlPlaneService
    from eyvan.server.service.scheduling import CoreSchedulingService

    declared = {"HF_TOKEN": "gcp-secret://my-project/training-hf-token"} if mounts is None else mounts
    repository = InMemoryRepository()
    _LAST_REPOSITORY[:] = [repository]
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    signer = SecretCapabilitySigner(SECRET, ttl_seconds=3600)
    fake_agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(),
        agent_factory=lambda _cell: fake_agent,
        artifact_access=ManagedArtifactAccess(
            store=LocalManagedArtifactStore(tmp_path / "artifacts"),
            signer=ArtifactTokenSigner(SECRET, default_ttl_seconds=3600, max_ttl_seconds=3600),
            public_url="https://eyvan.test",
            token_ttl_seconds=3600,
        ),
        secret_capability=signer,
    )
    await service.put_policy(
        principal(Role.FLEET_ADMIN),
        PolicyRecord(
            org_id="acme",
            project_id="research",
            allowed_secret_references=("gcp-secret://my-project/training-hf-token",),
            updated_by="root",
        ),
        idempotency_key="policy",
    )
    await service.upsert_cell(
        principal(Role.FLEET_ADMIN),
        CellRecord(
            id="cell-a",
            provider="local",
            region="local",
            state=CellState.REGISTERING,
            capacity=ResourceVector(cpu=8, memory_gib=64),
            allocatable=ResourceVector(cpu=8, memory_gib=64),
            endpoint="http://agent:8090",
        ),
        idempotency_key="cell",
    )
    assert await controller.dispatch_outbox_once() == 1
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="trainer",
            execution=CommandExecution(command="python train.py"),
            secret_mounts=declared,
        ),
        idempotency_key="trainer",
    )
    if attempt != 1:
        async with repository.transaction() as tx:
            workload = workload.model_copy(update={"attempt": attempt})
            await tx.put_workload(workload)
    await controller.reconcile_once()
    assert await controller.dispatch_outbox_once() == 1
    async with repository.transaction() as tx:
        dispatched = await tx.get_workload(workload.id)
    assert dispatched is not None
    return fake_agent.submissions[0][1].spec["environment"], dispatched, signer


async def _events(workload_id: str):
    repository = _LAST_REPOSITORY[0]
    async with repository.transaction() as tx:  # type: ignore[attr-defined]
        return [event for event in await tx.list_events(limit=200) if event.entity_id == workload_id]


# --------------------------------------------------------------------------- the route


def _plane(tmp_path):
    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path / "artifacts"),
        )
    )
    return TestClient(app), app


async def _live_workload(app, mounts: dict[str, str], prefixes: tuple[str, ...] = ()):
    """Submit a workload, allow its mounts, and put it in a state that can read."""

    from eyvan.records import (
        CommandExecution,
        LeaseRecord,
        LeaseState,
        PolicyRecord,
        Principal,
        Role,
        RoleGrant,
        WorkloadDefinition,
        WorkloadDispatchReservation,
        WorkloadState,
        utcnow,
    )

    service = app.state.service
    admin = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    operator = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    await service.put_policy(
        admin,
        PolicyRecord(
            org_id="acme",
            project_id="research",
            allowed_secret_references=prefixes,
            updated_by="root",
        ),
        idempotency_key="policy",
    )
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="trainer",
            execution=CommandExecution(command="python train.py"),
            secret_mounts=mounts,
        ),
        idempotency_key="trainer",
    )
    lease_id = f"lease-{workload.id}"
    reservation = WorkloadDispatchReservation(
        fence=DISPATCH_FENCE,
        outbox_id=f"dispatch-{workload.id}",
        outbox_attempt=1,
        generation=workload.generation,
        workload_attempt=workload.attempt,
        lease_ids=(lease_id,),
    )
    async with service.repository.transaction() as tx:
        await tx.put_lease(
            LeaseRecord(
                id=lease_id,
                workload_id=workload.id,
                cell_id="cell-test",
                org_id=workload.org_id,
                project_id=workload.project_id,
                state=LeaseState.ACTIVE,
                resources=workload.spec.gang,
                generation=workload.generation,
                fencing_token="test-fence",
                expires_at=utcnow() + timedelta(hours=1),
            )
        )
        running = workload.model_copy(
            update={
                "state": WorkloadState.RUNNING,
                "lease_id": lease_id,
                "lease_ids": (lease_id,),
                "dispatch_reservation": reservation,
                "remote_id": f"remote-{workload.id}",
            }
        )
        await tx.put_workload(running)
    return running


def _read(client, app, workload, *, name="HF_TOKEN", token=None, attempt=None):
    capability = token or app.state.secret_capability.issue(
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
        attempt=workload.attempt if attempt is None else attempt,
        generation=workload.generation,
        dispatch_fence=(
            workload.dispatch_reservation.fence if workload.dispatch_reservation is not None else DISPATCH_FENCE
        ),
        name=name,
    )
    return client.get(
        "/api/v1/secrets/content",
        params={"org_id": "acme", "project_id": "research", "workload_id": workload.id, "name": name},
        headers={"X-Eyvan-Secret-Token": capability},
    )


def test_injected_artifact_access_does_not_disable_secret_delivery(tmp_path, monkeypatch) -> None:
    import anyio
    from fastapi.testclient import TestClient

    from eyvan.artifacts.managed import ArtifactTokenSigner, LocalManagedArtifactStore
    from eyvan.artifacts.transfers import ManagedArtifactAccess
    from eyvan.config import EyvanSettings
    from eyvan.server.api import app as app_module
    from eyvan.server.api.app import create_app

    key = b"z" * 32
    access = ManagedArtifactAccess(
        store=LocalManagedArtifactStore(tmp_path / "injected-artifacts"),
        signer=ArtifactTokenSigner(key, default_ttl_seconds=3600, max_ttl_seconds=3600),
        public_url="https://eyvan.test",
        token_ttl_seconds=3600,
    )
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_token_key=key.decode(),
            artifact_store_uri=str(tmp_path / "unused-artifacts"),
            public_url="https://eyvan.test",
        ),
        artifact_access=access,
    )
    assert app.state.artifact_access is access
    assert app.state.secret_capability is not None
    assert app.state.controller.secret_capability is app.state.secret_capability
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf"},
        ("file:///run/secrets",),
    )
    monkeypatch.setattr(app_module, "read_secret", lambda _reference, **_kwargs: b"resolved")

    response = _read(TestClient(app), app, workload)

    assert response.status_code == 200
    assert response.content == b"resolved"


def test_a_live_attempt_reads_its_declared_secret(tmp_path, monkeypatch) -> None:
    """The happy path, end to end over HTTP."""

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.from_thread.BlockingPortal
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"hf_abcdef")

    response = _read(client, app, workload)

    assert response.status_code == 200, response.text
    assert response.content == b"hf_abcdef"
    assert response.headers["Cache-Control"] == "no-store"


def test_api_passes_the_matched_file_root_into_the_race_safe_open(tmp_path) -> None:
    import anyio

    root = tmp_path / "allowed"
    root.mkdir()
    secret = root / "token"
    secret.write_bytes(b"from-file-store")
    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": f"file://{secret}"},
        (f"file://{root}",),
    )

    response = _read(client, app, workload)

    assert response.status_code == 200
    assert response.content == b"from-file-store"


def test_the_api_accepts_exactly_one_mibibyte(tmp_path, monkeypatch) -> None:
    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda _reference, **_kwargs: b"x" * MAX_SECRET_BYTES)

    response = _read(client, app, workload)

    assert response.status_code == 200, response.text
    assert len(response.content) == MAX_SECRET_BYTES


def test_the_api_rejects_one_byte_over_the_limit_without_emitting_it(tmp_path, monkeypatch, caplog) -> None:
    import anyio

    from eyvan.server.api import app as app_module

    marker = b"oversized-secret-marker"
    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    value = marker + b"x" * (MAX_SECRET_BYTES + 1 - len(marker))
    capability = app.state.secret_capability.issue(
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
        attempt=workload.attempt,
        generation=workload.generation,
        dispatch_fence=DISPATCH_FENCE,
        name="HF_TOKEN",
    )
    monkeypatch.setattr(app_module, "read_secret", lambda _reference, **_kwargs: value)

    response = _read(client, app, workload, token=capability)

    assert response.status_code == 413, response.text
    assert marker not in response.content
    assert capability not in response.text
    assert marker.decode() not in caplog.text
    assert capability not in caplog.text

    async def secret_events():
        async with app.state.service.repository.transaction() as tx:
            return [event for event in await tx.list_events(limit=200) if event.action.startswith("secret.")]

    events = anyio.run(secret_events)
    assert any(event.action == "secret.read" for event in events)
    serialized = json.dumps([event.payload for event in events])
    assert marker.decode() not in serialized
    assert capability not in serialized


def test_a_token_from_a_finished_attempt_is_refused(tmp_path, monkeypatch) -> None:
    """What makes the credential per-attempt rather than merely short-lived.

    A retry or a preemption resume bumps ``attempt``, and the capability minted
    for the previous one stops working immediately -- without a revocation list,
    and without waiting out the TTL.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"hf_abcdef")

    stale = _read(client, app, workload, attempt=workload.attempt + 1)

    assert stale.status_code == 403, stale.text
    assert b"hf_abcdef" not in stale.content


def test_a_mount_the_policy_no_longer_allows_is_refused(tmp_path, monkeypatch) -> None:
    """Narrowing the allow-list after a leak has to be a real revocation.

    An allow-list checked only at submission would bar future specs while every
    already-running workload kept reading the secret that leaked.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"hf_abcdef")
    assert _read(client, app, workload).status_code == 200

    async def narrow():
        from eyvan.records import PolicyRecord, Principal, Role, RoleGrant

        await app.state.service.put_policy(
            Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),)),
            PolicyRecord(
                org_id="acme",
                project_id="research",
                allowed_secret_references=("file:///run/secrets/other-",),
                updated_by="root",
            ),
            idempotency_key="narrow",
        )

    anyio.run(narrow)

    assert _read(client, app, workload).status_code == 403


def test_a_capability_cannot_be_redirected_to_another_mount(tmp_path, monkeypatch) -> None:
    """A token names one mount, and the request names one mount.

    Without the comparison, a job holding a capability for its least sensitive
    mount could read its most sensitive one.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf", "DB_PASSWORD": "file:///run/secrets/db"},
        ("file:///run/secrets",),
    )
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"value")
    capability = app.state.secret_capability.issue(
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
        attempt=workload.attempt,
        generation=workload.generation,
        dispatch_fence=DISPATCH_FENCE,
        name="HF_TOKEN",
    )

    response = client.get(
        "/api/v1/secrets/content",
        params={
            "org_id": "acme",
            "project_id": "research",
            "workload_id": workload.id,
            "name": "DB_PASSWORD",
        },
        headers={"X-Eyvan-Secret-Token": capability},
    )

    assert response.status_code == 403, response.text


def test_an_undeclared_mount_is_refused(tmp_path, monkeypatch) -> None:
    """A capability is not a licence to name anything.

    The mount has to still be in the spec, so removing one takes effect at once.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"value")

    assert _read(client, app, workload, name="NOT_DECLARED").status_code == 403


def test_both_outcomes_of_a_secret_read_are_audited(tmp_path, monkeypatch) -> None:
    """One event per fetch, granted and refused, in the authorization transaction.

    A granted read that could be committed unaudited would make the ledger a
    record of what the control plane noticed rather than of what it did.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"hf_abcdef")

    forged = "eyvs1.forged.signature"
    assert _read(client, app, workload).status_code == 200
    assert _read(client, app, workload, name="NOT_DECLARED").status_code == 403
    rejected = _read(client, app, workload, token=forged)
    assert rejected.status_code == 403
    assert forged not in rejected.text

    async def secret_events():
        async with app.state.service.repository.transaction() as tx:
            return [event for event in await tx.list_events(limit=200) if event.action.startswith("secret.")]

    events = anyio.run(secret_events)
    recorded = [event.action for event in events]

    assert "secret.read" in recorded
    assert "secret.read_denied" in recorded
    assert "secret.token_rejected" in recorded
    assert forged not in json.dumps([event.payload for event in events])
    rejected_event = next(event for event in events if event.action == "secret.token_rejected")
    assert rejected_event.payload["reason"] == "invalid_capability"


def test_invalid_capability_audit_uses_only_a_resolved_workload_scope(tmp_path) -> None:
    import anyio

    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf"},
        ("file:///run/secrets",),
    )
    attacker = "attacker\ncontrolled"
    forged = "eyvs1.attacker.controlled"

    response = client.get(
        "/api/v1/secrets/content",
        params={
            "org_id": attacker,
            "project_id": attacker,
            "workload_id": workload.id,
            "name": attacker,
        },
        headers={"X-Eyvan-Secret-Token": forged},
    )
    assert response.status_code == 403

    async def rejection_events():
        async with app.state.service.repository.transaction() as tx:
            return [event for event in await tx.list_events(limit=200) if event.action == "secret.token_rejected"]

    events = anyio.run(rejection_events)
    event = events[-1]
    serialized = json.dumps(event.model_dump(mode="json"))
    assert event.org_id == "acme"
    assert event.project_id == "research"
    assert event.entity_id == workload.id
    assert event.payload == {"workload_id": workload.id, "reason": "invalid_capability"}
    assert attacker not in serialized
    assert forged not in serialized


def test_nonexistent_workload_does_not_create_an_attacker_owned_audit_event(tmp_path) -> None:
    import anyio

    client, app = _plane(tmp_path)
    marker = "attacker-controlled-audit-field"
    response = client.get(
        "/api/v1/secrets/content",
        params={"org_id": marker, "project_id": marker, "workload_id": marker, "name": marker},
        headers={"X-Eyvan-Secret-Token": "eyvs1.attacker.controlled"},
    )
    assert response.status_code == 403

    async def events():
        async with app.state.service.repository.transaction() as tx:
            return await tx.list_events(limit=200)

    assert marker not in json.dumps([event.model_dump(mode="json") for event in anyio.run(events)])


def test_mount_mismatch_audit_uses_the_verified_claim_name(tmp_path) -> None:
    import anyio

    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf"},
        ("file:///run/secrets",),
    )
    token = app.state.secret_capability.issue(
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
        attempt=workload.attempt,
        generation=workload.generation,
        dispatch_fence=DISPATCH_FENCE,
        name="HF_TOKEN",
    )
    attacker_name = "ATTACKER_QUERY_NAME"
    response = client.get(
        "/api/v1/secrets/content",
        params={
            "org_id": "acme",
            "project_id": "research",
            "workload_id": workload.id,
            "name": attacker_name,
        },
        headers={"X-Eyvan-Secret-Token": token},
    )
    assert response.status_code == 403

    async def rejection():
        async with app.state.service.repository.transaction() as tx:
            events = [event for event in await tx.list_events(limit=200) if event.action == "secret.token_rejected"]
            return events[-1]

    event = anyio.run(rejection)
    assert event.payload["reason"] == "mount_mismatch"
    assert event.payload["name"] == "HF_TOKEN"
    assert attacker_name not in json.dumps(event.payload)
    assert token not in json.dumps(event.payload)


def test_query_scope_mismatch_is_audited_under_the_verified_workload(tmp_path) -> None:
    import anyio

    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf"},
        ("file:///run/secrets",),
    )
    token = app.state.secret_capability.issue(
        workload_id=workload.id,
        org_id="acme",
        project_id="research",
        attempt=workload.attempt,
        generation=workload.generation,
        dispatch_fence=DISPATCH_FENCE,
        name="HF_TOKEN",
    )
    marker = "attacker-scope"
    response = client.get(
        "/api/v1/secrets/content",
        params={"org_id": marker, "project_id": marker, "workload_id": workload.id, "name": "HF_TOKEN"},
        headers={"X-Eyvan-Secret-Token": token},
    )
    assert response.status_code == 403

    async def rejection():
        async with app.state.service.repository.transaction() as tx:
            events = [event for event in await tx.list_events(limit=200) if event.action == "secret.token_rejected"]
            return events[-1]

    event = anyio.run(rejection)
    assert event.org_id == "acme"
    assert event.project_id == "research"
    assert event.payload == {"workload_id": workload.id, "reason": "out_of_scope", "name": "HF_TOKEN"}
    assert marker not in json.dumps(event.model_dump(mode="json"))


def test_no_secret_value_appears_in_the_audit_trail(tmp_path, monkeypatch) -> None:
    """The ledger is read by more people than the spec is.

    So it carries the mount *name*, never the reference and never the value.
    """

    import anyio

    from eyvan.server.api import app as app_module

    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))
    monkeypatch.setattr(app_module, "read_secret", lambda reference, **_kwargs: b"hf_abcdef")
    assert _read(client, app, workload).status_code == 200

    async def payloads():
        async with app.state.service.repository.transaction() as tx:
            return json.dumps(
                [event.payload for event in await tx.list_events(limit=200) if event.action.startswith("secret.")]
            )

    recorded = anyio.run(payloads)

    assert "HF_TOKEN" in recorded
    assert "hf_abcdef" not in recorded
    assert "/run/secrets/hf" not in recorded


def test_an_unreachable_store_is_retryable_without_reflecting_provider_output(tmp_path, monkeypatch, caplog) -> None:
    """A store outage stays retryable, but its exception is not a safe response."""

    import anyio

    from eyvan.secrets import SecretUnavailable
    from eyvan.server.api import app as app_module

    value = "hf_provider_reflected_value"
    capability = "eyvs1.provider.reflection"
    client, app = _plane(tmp_path)
    workload = anyio.run(_live_workload, app, {"HF_TOKEN": "file:///run/secrets/hf"}, ("file:///run/secrets",))

    def unavailable(_reference, **_kwargs):
        raise SecretUnavailable(f"provider reflected {value} and {capability}")

    monkeypatch.setattr(app_module, "read_secret", unavailable)

    response = _read(client, app, workload)

    assert response.status_code == 503, response.text
    assert response.headers["Retry-After"] == "5"
    assert response.json()["detail"] == "secret mount could not be resolved"
    assert value not in response.text
    assert capability not in response.text
    assert value not in caplog.text
    assert capability not in caplog.text


def test_unknown_resolver_exception_is_the_same_constant_retryable_503(tmp_path, monkeypatch, caplog) -> None:
    import anyio

    from eyvan.server.api import app as app_module

    marker = "unknown-resolver-reflected-secret"
    client, app = _plane(tmp_path)
    workload = anyio.run(
        _live_workload,
        app,
        {"HF_TOKEN": "file:///run/secrets/hf"},
        ("file:///run/secrets",),
    )

    def fail(_reference, **_kwargs):
        raise RuntimeError(marker)

    monkeypatch.setattr(app_module, "read_secret", fail)
    response = _read(client, app, workload)

    assert response.status_code == 503
    assert response.json()["detail"] == "secret mount could not be resolved"
    assert marker not in response.text
    assert marker not in caplog.text


# --------------------------------------------------------------------------- the SDK


def test_a_callable_config_carries_its_mounts_to_the_submission(tmp_path) -> None:
    """``@execute`` is the SDK's primary interface.

    A ``secret_mounts`` field that only ``eyvan workload submit`` can set would
    leave every callable with ``environment`` -- the persisted path -- as its
    only way to pass a credential, which is the practice this replaces.
    """

    import httpx

    from eyvan.artifacts import LocalArtifactStore
    from eyvan.records import CallableExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.sdk import EyvanCallableSubmitter, EyvanClient
    from eyvan.sdk.execution import CallableSubmission, TpuConfig
    from eyvan.worker.runtime.callables import dump_callable

    store = LocalArtifactStore(tmp_path)
    stored_artifact = store.put(dump_callable(lambda: None))
    execution = CallableExecution(
        artifact_uri=stored_artifact.uri,
        sha256=stored_artifact.sha256,
        python=stored_artifact.python,
        result_uri="gs://artifacts/results/run-1.result",
    )
    pending = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=execution),
        created_by="alice",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=pending.model_dump(mode="json"))

    client = EyvanClient(client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"))
    submitter = EyvanCallableSubmitter(client, store)

    submitter.submit_callable(
        CallableSubmission(
            invocation_id="call-1",
            fleet="research",
            org_id="acme",
            project_id="research",
            callable_name="tests.train",
            tpu=TpuConfig(type="v5p-16", secret_mounts={"HF_TOKEN": "gcp-secret://my-project/training-hf-token"}),
            artifact=stored_artifact,
            result_uri="gs://artifacts/results/run-1.result",
        )
    )

    assert json.loads(requests[0].content)["secret_mounts"] == {"HF_TOKEN": "gcp-secret://my-project/training-hf-token"}
    client.close()


def test_a_secret_is_not_delivered_in_clear_text_without_explicit_consent(tmp_path) -> None:
    """A secret value is worse to expose than the capability that fetches it.

    The managed-artifact path already refuses a plain-HTTP public URL unless an
    operator accepts the risk. Secret delivery used the same URL and had no such
    gate, so on a `head up` plane -- which terminates no TLS -- the capability and
    the secret it redeems both travelled unencrypted.

    Refused as a permanent dispatch refusal, not a bare error, so the attempt
    settles and returns its accelerator instead of retrying forever against a
    scheme that will not change.
    """

    from eyvan.artifacts.managed import ArtifactTokenSigner, LocalManagedArtifactStore
    from eyvan.artifacts.transfers import ManagedArtifactAccess
    from eyvan.config import EyvanSettings
    from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord
    from eyvan.server.controller import EyvanController
    from eyvan.server.controller.workload_commands import InsecureManagedArtifactTransport
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    def _controller(*, allow_insecure: bool) -> EyvanController:
        repository = InMemoryRepository()
        return EyvanController(
            repository,
            ControlPlaneService(repository),
            settings=EyvanSettings(dev_auth_enabled=False, allow_insecure_artifact_http=allow_insecure),
            artifact_access=ManagedArtifactAccess(
                store=LocalManagedArtifactStore(tmp_path / f"artifacts-{allow_insecure}"),
                signer=ArtifactTokenSigner(b"k" * 32, default_ttl_seconds=3600, max_ttl_seconds=3600),
                # What `eyvan head up` actually writes: internal address, no TLS.
                public_url="http://10.128.0.11:8080",
                token_ttl_seconds=3600,
            ),
        )

    workload = WorkloadRecord(
        id="wl_secret",
        org_id="acme",
        project_id="research",
        created_by="operator",
        spec=WorkloadDefinition(
            name="needs-a-secret",
            execution=CommandExecution(command="python train.py"),
            secret_mounts={"TOKEN": "gcp-secret://project/token"},
        ),
    )

    with pytest.raises(InsecureManagedArtifactTransport, match="clear text"):
        _controller(allow_insecure=False)._secret_url(workload, "TOKEN")

    # Explicit consent is what makes a VPC-only plane usable at all; the point is
    # that it is typed, not inferred.
    assert _controller(allow_insecure=True)._secret_url(workload, "TOKEN").startswith("http://10.128.0.11:8080")
