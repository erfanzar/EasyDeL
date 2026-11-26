# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rotating a signing credential without restarting anything that is running.

Rotation is easy to implement in a way that passes a unit test and fails in
production, so every test here asserts one of the four things that actually go
wrong:

* a capability minted before the rotation is still redeemable after it -- if it
  is not, rotation restarts every workload and nobody will run one;
* retired material stops at an **absolute instant**, not when the last token
  expires, not at the next restart;
* two rotations cannot both be in flight, across replicas and across a restart,
  because the fence is a durable row rather than a process variable; and
* a rotation nobody could observe stops the credential instead of guessing.

The durable half runs against the in-memory repository *and* PostgreSQL: the
fence is a locked read followed by an insert, and that is a database behaviour,
not a Python one.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from eyvan.artifacts._capability_tokens import InvalidCapabilityToken
from eyvan.artifacts.secret_capability import SecretCapabilitySigner
from eyvan.identity.rotation import (
    CredentialRotationBlocked,
    CredentialRotationFenceMismatch,
    CredentialRotationInterleaved,
    CredentialRotationState,
    RotatingCredential,
)
from eyvan.records import CredentialRotationRecord
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service.credential_rotation import (
    SECRET_CAPABILITY_CREDENTIAL,
    CredentialRotationService,
    new_rotation_fence,
    new_rotation_id,
)

OLD_MATERIAL = b"old-material-that-is-long-enough"
NEW_MATERIAL = b"new-material-that-is-long-enough"
ORIGIN = datetime(2026, 8, 8, 12, 0, 0)


class _Clock:
    """A clock a test can move, because "a definite point" is a claim about time."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _issue(signer: SecretCapabilitySigner, *, name: str = "HF_TOKEN") -> str:
    return signer.issue(
        workload_id="wl-1",
        org_id="acme",
        project_id="research",
        attempt=1,
        generation=1,
        dispatch_fence="dispatch_fence",
        name=name,
    )


def _verify(signer: SecretCapabilitySigner, token: str) -> str:
    return signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research").name


# --------------------------------------------------------------------------- the definite point


def test_a_capability_minted_before_a_rotation_survives_it() -> None:
    """The property that makes rotation something an operator will actually do.

    If rotating invalidated everything in flight, every rotation would be a
    fleet-wide restart, and the credential would never be rotated at all.
    """

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)

    credential.commit(_begin(credential), NEW_MATERIAL, overlap_s=300)

    assert _verify(signer, token) == "HF_TOKEN"
    assert credential.generation == 2


def test_retired_material_stops_verifying_at_its_absolute_instant() -> None:
    """Old material dies on the clock, not when the token holding it expires.

    The token below has 3000 seconds of TTL left at the moment its generation is
    retired. "Rotation completes when the last token expires" is the design this
    asserts against: it makes the end of a compromised key depend on the longest
    TTL anyone ever issued.
    """

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)
    retired = credential.commit(_begin(credential), NEW_MATERIAL, overlap_s=300)

    assert retired.not_after == pytest.approx(clock.now + 300)
    clock.now = retired.not_after - 1
    assert _verify(signer, token) == "HF_TOKEN"

    clock.now = retired.not_after
    with pytest.raises(InvalidCapabilityToken):
        _verify(signer, token)

    # And the token itself is still well inside its own validity window, so the
    # refusal above came from the retirement and from nothing else.
    fresh = _issue(signer)
    assert _verify(signer, fresh) == "HF_TOKEN"


def test_a_zero_overlap_rotation_retires_the_previous_generation_immediately() -> None:
    """``overlap_s=0`` is a real answer for a credential known to be compromised."""

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)

    credential.commit(_begin(credential), NEW_MATERIAL, overlap_s=0)

    with pytest.raises(InvalidCapabilityToken):
        _verify(signer, token)


# --------------------------------------------------------------------------- interleaving


def test_a_second_rotation_is_refused_while_one_is_in_flight() -> None:
    """Two operators reacting to one alert produce one rotation and one refusal."""

    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.begin("fence-a")

    with pytest.raises(CredentialRotationInterleaved):
        credential.begin("fence-b")


def test_only_the_fence_that_began_a_rotation_can_resolve_it() -> None:
    """A fence nobody else holds is what makes "cannot interleave" mean anything."""

    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.begin("fence-a")

    with pytest.raises(CredentialRotationFenceMismatch):
        credential.commit("fence-b", NEW_MATERIAL, overlap_s=60)
    with pytest.raises(CredentialRotationFenceMismatch):
        credential.abandon("fence-b")

    credential.commit("fence-a", NEW_MATERIAL, overlap_s=60)
    assert credential.generation == 2


def test_an_abandoned_rotation_changes_nothing_and_frees_the_credential() -> None:
    """Explicitly-did-not-happen is a different answer from could-not-tell."""

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)

    credential.abandon(_begin(credential, "fence-a"))

    assert credential.generation == 1
    assert _verify(signer, token) == "HF_TOKEN"
    assert credential.begin("fence-b") == 2


# --------------------------------------------------------------------------- failing closed


def test_an_inconclusive_rotation_stops_issuing_but_not_verifying() -> None:
    """Fail closed means stop minting, not invalidate what is already out there.

    Refusing to verify would turn one operator's network timeout into a fleet
    outage; minting under a key whose distribution is unknown would turn it into
    a silent one. Only the second is a decision this process is entitled to make
    on its own.
    """

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)

    credential.mark_inconclusive(_begin(credential, "fence-a"))

    assert credential.rotation_state is CredentialRotationState.INCONCLUSIVE
    assert _verify(signer, token) == "HF_TOKEN"
    with pytest.raises(CredentialRotationBlocked):
        credential.signing_material()
    with pytest.raises(CredentialRotationBlocked):
        _issue(signer)


def test_an_inconclusive_rotation_refuses_another_rotation_until_resolved() -> None:
    """No automatic retry, and no second unknown stacked on the first.

    The refusal is bounded rather than a loop: it ends when a human supplies an
    outcome, and nothing in the package converts it on its own.
    """

    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.mark_inconclusive(_begin(credential, "fence-a"))

    with pytest.raises(CredentialRotationBlocked):
        credential.begin("fence-b")

    credential.resolve("fence-a", material=None)
    assert credential.rotation_state is CredentialRotationState.ABANDONED
    assert credential.generation == 1
    assert credential.begin("fence-b") == 2


def test_resolving_an_inconclusive_rotation_as_committed_installs_the_material() -> None:
    """The other half of the resolution: it turned out to have happened."""

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)
    credential.mark_inconclusive(_begin(credential, "fence-a"))

    credential.resolve("fence-a", material=NEW_MATERIAL, overlap_s=120)

    assert credential.generation == 2
    assert _verify(signer, token) == "HF_TOKEN"
    clock.now += 120
    with pytest.raises(InvalidCapabilityToken):
        _verify(signer, token)


def test_an_inconclusive_rotation_cannot_be_resolved_by_another_fence() -> None:
    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.mark_inconclusive(_begin(credential, "fence-a"))

    with pytest.raises(CredentialRotationFenceMismatch):
        credential.resolve("fence-b", material=NEW_MATERIAL, overlap_s=1)


# --------------------------------------------------------------------------- no material anywhere


def test_no_material_reaches_a_repr_an_error_or_an_epoch() -> None:
    """Every surface that a traceback, a log formatter or a metric can reach.

    A default dataclass ``__repr__`` over a keyring puts the signing key in every
    traceback that happens to include one, which is how a credential ends up in
    an exception reporter.
    """

    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.begin("fence-a")
    credential.mark_inconclusive("fence-a")

    observed = [repr(credential), str(credential.epochs()), repr(credential.epochs())]
    for error_type, call in (
        (CredentialRotationBlocked, credential.signing_material),
        (CredentialRotationBlocked, lambda: credential.begin("fence-b")),
        (CredentialRotationFenceMismatch, lambda: credential.resolve("wrong", material=NEW_MATERIAL)),
    ):
        with pytest.raises(error_type) as raised:
            call()
        observed.append(str(raised.value))

    for text in observed:
        assert OLD_MATERIAL.decode() not in text
        assert NEW_MATERIAL.decode() not in text


def _begin(credential: RotatingCredential, fence: str = "fence-a") -> str:
    credential.begin(fence)
    return fence


# --------------------------------------------------------------------------- the durable fence


@pytest.fixture(params=["memory", "local", "sql"])
async def rotation_repository(request: pytest.FixtureRequest, tmp_path):
    """One fixture per durable backend, because the fence is a database behaviour.

    The in-memory backend proves the rule, the local JSON store proves it
    survives a restart, and the SQLAlchemy one proves the row and its
    constraints exist. PostgreSQL's own locking is exercised separately below,
    where a real database is available.
    """

    from sqlalchemy.ext.asyncio import create_async_engine

    if request.param == "memory":
        yield InMemoryRepository()
        return
    if request.param == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
        await repository.open()
        try:
            yield repository
        finally:
            await repository.close()
        return
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
    await create_schema(engine)
    repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        yield repository
    finally:
        await engine.dispose()


class _StepClock:
    """A datetime clock a test moves, so a retirement instant can be asserted."""

    def __init__(self) -> None:
        self.now = ORIGIN

    def __call__(self) -> datetime:
        return self.now


async def test_a_durable_rotation_refuses_a_second_replica(rotation_repository) -> None:
    """Two services, one repository: exactly one of them gets to rotate.

    This is the case a process-local fence does not cover, and it is the ordinary
    one -- a control plane runs more than one replica precisely so that no single
    process is the fence.
    """

    replica_a = CredentialRotationService(rotation_repository)
    replica_b = CredentialRotationService(rotation_repository)

    started = await replica_a.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=300)

    assert started.generation == 2
    with pytest.raises(CredentialRotationInterleaved):
        await replica_b.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-b", overlap_s=300)


async def test_a_committed_rotation_records_one_agreed_retirement_instant(rotation_repository) -> None:
    """Every replica reads the same point rather than each deriving its own.

    A ``not_after`` computed per process is not a definite point: two replicas
    with a second of clock skew disagree about whether a token is still good, and
    the disagreement is invisible until an audit asks which one was right.
    """

    clock = _StepClock()
    service = CredentialRotationService(rotation_repository, clock=clock)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=300)

    clock.now = ORIGIN + timedelta(seconds=42)
    committed = await service.commit(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
    )

    assert committed.state is CredentialRotationState.COMMITTED
    assert committed.resolved_at == ORIGIN + timedelta(seconds=42)
    assert committed.previous_not_after == ORIGIN + timedelta(seconds=342)
    async with rotation_repository.transaction() as transaction:
        stored = await transaction.get_credential_rotation(SECRET_CAPABILITY_CREDENTIAL, 2)
    assert stored == committed
    assert await service.current(SECRET_CAPABILITY_CREDENTIAL) == committed


async def test_a_rotation_can_only_be_resolved_by_its_own_fence(rotation_repository) -> None:
    service = CredentialRotationService(rotation_repository)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)

    for call in (service.commit, service.abandon, service.record_inconclusive):
        with pytest.raises(CredentialRotationFenceMismatch):
            await call(
                credential=SECRET_CAPABILITY_CREDENTIAL,
                generation=started.generation,
                fence="fence-b",
            )

    async with rotation_repository.transaction() as transaction:
        assert (await transaction.get_credential_rotation(SECRET_CAPABILITY_CREDENTIAL, 2)).is_in_flight


async def test_an_inconclusive_rotation_blocks_the_next_one_until_a_human_resolves_it(
    rotation_repository,
) -> None:
    """Durably fail closed, and end at a resolution rather than at a retry.

    The refusal is what stops a scheduled rotation job from grinding against a
    provider that will never answer -- which would hold the fence forever.
    """

    service = CredentialRotationService(rotation_repository)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)
    unknown = await service.record_inconclusive(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
        detail="provider did not answer",
    )

    assert unknown.state is CredentialRotationState.INCONCLUSIVE
    # Nothing was inferred about the previous generation: an unobserved rotation
    # does not get to claim it retired anything.
    assert unknown.previous_not_after is None
    assert unknown.blocks_further_rotation

    with pytest.raises(CredentialRotationBlocked):
        await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-b", overlap_s=60)

    resolved = await service.resolve(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
        committed=False,
        detail="operator confirmed the provider never received it",
    )

    assert resolved.state is CredentialRotationState.ABANDONED
    assert resolved.previous_not_after is None
    following = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-c", overlap_s=60)
    assert following.generation == 3


async def test_an_unresolved_rotation_survives_reopening_the_backend(rotation_repository) -> None:
    """A restart must not release the fence.

    If it did, "two rotations cannot interleave" would mean "unless one of the
    replicas restarted", which is the state a rotation is most likely to be in
    when a process dies mid-rotation.
    """

    service = CredentialRotationService(rotation_repository)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)
    await service.record_inconclusive(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
    )

    await rotation_repository.close()
    await rotation_repository.open()

    reopened = CredentialRotationService(rotation_repository)
    current = await reopened.current(SECRET_CAPABILITY_CREDENTIAL)
    assert current is not None
    assert current.state is CredentialRotationState.INCONCLUSIVE
    with pytest.raises(CredentialRotationBlocked):
        await reopened.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-b", overlap_s=60)


async def test_a_persisted_rotation_row_carries_no_material_or_digest_of_it(rotation_repository) -> None:
    """The ledger is read by more people than the key is, and it is backed up.

    A digest counts as material for this purpose: a credential is short enough
    that a stored digest is an offline cracking target carried by every backup
    and support bundle.
    """

    service = CredentialRotationService(rotation_repository)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)
    committed = await service.commit(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
    )

    encoded = json.dumps(committed.model_dump(mode="json"), sort_keys=True)
    import hashlib

    assert OLD_MATERIAL.decode() not in encoded
    assert NEW_MATERIAL.decode() not in encoded
    for material in (OLD_MATERIAL, NEW_MATERIAL):
        assert hashlib.sha256(material).hexdigest() not in encoded
    assert set(committed.model_dump()) == {
        "id",
        "credential",
        "generation",
        "fence",
        "state",
        "overlap_s",
        "started_at",
        "resolved_at",
        "previous_not_after",
        "detail",
    }


async def test_a_rotation_detail_never_carries_a_provider_message_verbatim(rotation_repository) -> None:
    """An operator note is bounded and stripped, because it is pasted by hand.

    Control characters in a durable row break the local JSON document's own
    reader long before anyone reads the note.
    """

    service = CredentialRotationService(rotation_repository)
    started = await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)

    recorded = await service.record_inconclusive(
        credential=SECRET_CAPABILITY_CREDENTIAL,
        generation=started.generation,
        fence="fence-a",
        detail="timeout\n" + "x" * 900,
    )

    assert "\n" not in recorded.detail
    assert len(recorded.detail) == 512


async def test_a_rotation_ledger_row_cannot_claim_an_impossible_outcome() -> None:
    """The record refuses shapes the state machine can never produce.

    Two of them matter: an in-flight rotation with an outcome, and a rotation
    that retired the previous generation without committing. Both would read as
    "the old key is dead" to anything auditing the table.
    """

    with pytest.raises(ValueError):
        CredentialRotationRecord(
            id=new_rotation_id(),
            credential=SECRET_CAPABILITY_CREDENTIAL,
            generation=2,
            fence=new_rotation_fence(),
            state=CredentialRotationState.PENDING,
            overlap_s=60,
            resolved_at=ORIGIN,
        )
    with pytest.raises(ValueError):
        CredentialRotationRecord(
            id=new_rotation_id(),
            credential=SECRET_CAPABILITY_CREDENTIAL,
            generation=2,
            fence=new_rotation_fence(),
            state=CredentialRotationState.INCONCLUSIVE,
            overlap_s=60,
            resolved_at=ORIGIN,
            previous_not_after=ORIGIN,
        )
    with pytest.raises(ValueError):
        CredentialRotationRecord(
            id=new_rotation_id(),
            credential=SECRET_CAPABILITY_CREDENTIAL,
            generation=1,
            fence=new_rotation_fence(),
            state=CredentialRotationState.PENDING,
            overlap_s=60,
        )


def test_a_rotation_identifier_fits_the_ledger_column() -> None:
    """32 characters, because that is what the migration declares.

    ``uuid4().hex`` with a prefix is 36 and would fail at the first insert on
    PostgreSQL and silently truncate on a database that is less strict.
    """

    assert len(new_rotation_id()) <= 32
    assert len(new_rotation_fence()) <= 64


# --------------------------------------------------------------------------- on a running control plane


async def test_a_live_control_plane_rotates_without_restarting_anything(monkeypatch, tmp_path) -> None:
    """The end-to-end claim, on an assembled application rather than a keyring.

    The control plane that minted a capability keeps honouring it across a
    rotation -- so an operator rotating a compromised key does not have to stop
    the workloads that are holding capabilities minted under it -- and stops
    honouring it at the instant the ledger recorded, not at a restart.

    ``app.state`` is the seam deliberately: new material never crosses the API,
    because a signing key in a request body is a signing key in every proxy log.
    """

    from fastapi.testclient import TestClient

    from eyvan.config import EyvanSettings, RepositoryBackend
    from eyvan.server.api.app import create_app

    monkeypatch.setenv("EYVAN_ARTIFACT_TOKEN_KEY", "a" * 64)
    monkeypatch.setenv("HOME", str(tmp_path))
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            repository_backend=RepositoryBackend.MEMORY,
            controller_enabled=False,
            artifact_store_uri=str(tmp_path / "artifacts"),
        )
    )
    with TestClient(app):
        credential = app.state.secret_capability_credential
        signer = app.state.secret_capability
        ledger = app.state.credential_rotation
        assert credential.generation == 1

        token = signer.issue(
            workload_id="wl-1",
            org_id="acme",
            project_id="research",
            attempt=1,
            generation=1,
            dispatch_fence="dispatch_fence",
            name="HF_TOKEN",
        )

        started = await ledger.begin(
            credential=SECRET_CAPABILITY_CREDENTIAL,
            fence="fence-a",
            overlap_s=300,
        )
        retired = credential.commit(_begin(credential, "fence-a"), NEW_MATERIAL, overlap_s=300)
        committed = await ledger.commit(
            credential=SECRET_CAPABILITY_CREDENTIAL,
            generation=started.generation,
            fence="fence-a",
        )

        assert committed.state is CredentialRotationState.COMMITTED
        assert credential.generation == 2
        # Still redeemable: nothing that was running had to be restarted.
        assert signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research").name == "HF_TOKEN"

        monkeypatch.setattr(credential, "_clock", lambda: retired.not_after)
        with pytest.raises(InvalidCapabilityToken):
            signer.verify(token, workload_id="wl-1", org_id="acme", project_id="research")
        fresh = signer.issue(
            workload_id="wl-1",
            org_id="acme",
            project_id="research",
            attempt=1,
            generation=1,
            dispatch_fence="dispatch_fence",
            name="HF_TOKEN",
        )
        assert signer.verify(fresh, workload_id="wl-1", org_id="acme", project_id="research").name == "HF_TOKEN"


def test_a_retired_generation_is_refused_indistinguishably_from_a_forgery() -> None:
    """Which generation a token failed against would date the rotation for a prober."""

    clock = _Clock()
    credential = RotatingCredential(OLD_MATERIAL, clock=clock)
    signer = SecretCapabilitySigner(credential, ttl_seconds=3600, clock=clock)
    token = _issue(signer)
    credential.commit(_begin(credential), NEW_MATERIAL, overlap_s=60)
    clock.now += 60

    forged = SecretCapabilitySigner(b"unrelated-material-long-enough!!", ttl_seconds=3600, clock=clock)
    foreign = forged.issue(
        workload_id="wl-1",
        org_id="acme",
        project_id="research",
        attempt=1,
        generation=1,
        dispatch_fence="dispatch_fence",
        name="HF_TOKEN",
    )

    with pytest.raises(InvalidCapabilityToken) as retired:
        _verify(signer, token)
    with pytest.raises(InvalidCapabilityToken) as unrelated:
        _verify(signer, foreign)

    assert str(retired.value) == str(unrelated.value)


async def test_a_fleet_that_never_rotated_keeps_writing_the_older_local_document(tmp_path) -> None:
    """The rolling-upgrade property: a new version appears only when it must.

    A build that always emitted v4 would make every local state file unreadable
    to a rolled-back replica the first time it saved, whether or not anything had
    ever been rotated.
    """

    from eyvan.server.persistence.local import (
        LOCAL_CONTROL_PLANE_SCHEMA_VERSION,
        LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION,
        LocalControlPlaneStore,
        local_control_plane_document,
    )
    from eyvan.server.persistence.state import ControlPlaneState

    store = LocalControlPlaneStore(tmp_path / "control-plane.json")
    empty = ControlPlaneState()
    store.save(empty)
    assert json.loads((tmp_path / "control-plane.json").read_text())["version"] == LOCAL_CONTROL_PLANE_SCHEMA_VERSION

    repository = LocalJsonRepository(tmp_path / "rotated.json")
    await repository.open()
    try:
        service = CredentialRotationService(repository)
        await service.begin(credential=SECRET_CAPABILITY_CREDENTIAL, fence="fence-a", overlap_s=60)
    finally:
        await repository.close()

    document = json.loads((tmp_path / "rotated.json").read_text())
    assert document["version"] == LOCAL_CONTROL_PLANE_V4_SCHEMA_VERSION
    assert [row["credential"] for row in document["credential_rotations"]] == [SECRET_CAPABILITY_CREDENTIAL]
    assert local_control_plane_document(empty).version == LOCAL_CONTROL_PLANE_SCHEMA_VERSION


def test_a_malformed_resolution_leaves_the_credential_closed() -> None:
    """Failing closed has to survive the resolution attempt as well.

    Reopening the rotation before validating the new material would leave the
    credential PENDING -- signing again, under exactly the material the operator
    was in the middle of retiring -- because a typo in the resolution silently
    undid the fail-closed state.
    """

    credential = RotatingCredential(OLD_MATERIAL, clock=_Clock())
    credential.mark_inconclusive(_begin(credential, "fence-a"))

    with pytest.raises(ValueError):
        credential.resolve("fence-a", material=b"too-short", overlap_s=60)

    assert credential.rotation_state is CredentialRotationState.INCONCLUSIVE
    with pytest.raises(CredentialRotationBlocked):
        credential.signing_material()
    with pytest.raises(CredentialRotationBlocked):
        credential.begin("fence-b")
