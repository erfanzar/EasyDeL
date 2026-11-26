# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fencing one credential rotation against every other replica.

:mod:`eyvan.identity.rotation` holds the in-process keyring; this holds the row
that makes two processes agree about it. Both halves are needed and neither is
sufficient: an in-process fence does not survive a restart or reach a second
replica, and a row does not decide which material a signature is made with.

**No provider call happens inside a transaction.** The shape of this API is the
enforcement: :meth:`CredentialRotationService.begin` commits a row and returns,
the caller installs the new material at the provider with nothing held, and
:meth:`commit`, :meth:`abandon` or :meth:`record_inconclusive` opens a second
short transaction. A single ``rotate()`` that did the provider call in the
middle would hold a row lock across the network, which is how one slow secret
manager becomes a control plane that cannot schedule.

**An unobserved outcome is recorded as unobserved.** ``record_inconclusive``
writes ``INCONCLUSIVE``; nothing in this module ever promotes that to committed
or abandoned, and nothing retries it. It stops the credential -- ``begin``
refuses while it stands -- until :meth:`resolve` is called with an outcome a
human established. That is a bounded refusal rather than a retry loop, which
matters because a retry here would hold a fence forever.
"""

from __future__ import annotations

import typing as tp
import uuid
from datetime import datetime, timedelta

from eyvan.identity.rotation import (
    CredentialRotationBlocked,
    CredentialRotationFenceMismatch,
    CredentialRotationInterleaved,
    CredentialRotationState,
)
from eyvan.records import CredentialRotationRecord, utcnow
from eyvan.server.persistence.base import Repository, RepositorySession

#: The credential every capability in this package is signed with. Named so the
#: ledger stays readable when a second one is added rather than becoming a table
#: with one implicit row.
SECRET_CAPABILITY_CREDENTIAL: tp.Final = "secret-capability"

#: ``rot_`` plus 28 hex characters. Sized to the 32-character identifier column
#: rather than to ``uuid4().hex``, which would be four characters too long and
#: would fail at the first insert rather than here.
_ID_HEX_LENGTH: tp.Final = 28


def new_rotation_id() -> str:
    """Return a rotation identifier that fits the ledger's identifier column."""

    return f"rot_{uuid.uuid4().hex[:_ID_HEX_LENGTH]}"


def new_rotation_fence() -> str:
    """Return an unguessable fence for one rotation.

    Unguessable rather than merely unique: the fence is the only thing that
    decides who may resolve a rotation, so a predictable one would let a second
    operator commit the first one's rotation.
    """

    return uuid.uuid4().hex


class CredentialRotationService:
    """Durable, fenced lifecycle for one deployment's signing credentials."""

    def __init__(
        self,
        repository: Repository,
        *,
        clock: tp.Callable[[], datetime] = utcnow,
        id_factory: tp.Callable[[], str] = new_rotation_id,
    ):
        self.repository = repository
        self._clock = clock
        self._id_factory = id_factory

    async def current(self, credential: str) -> CredentialRotationRecord | None:
        """Return the newest recorded rotation of one credential, if any."""

        async with self.repository.transaction() as tx:
            return await tx.latest_credential_rotation(credential)

    async def begin(
        self,
        *,
        credential: str,
        fence: str,
        overlap_s: float,
    ) -> CredentialRotationRecord:
        """Claim the exclusive right to rotate ``credential``.

        Args:
            credential: Logical credential name.
            fence: Unguessable token the caller must present to resolve this
                rotation. Use :func:`new_rotation_fence`.
            overlap_s: Seconds the superseded generation keeps verifying once
                this rotation commits. Fixed here rather than at commit time so
                a commit cannot widen a window an operator already approved.

        Returns:
            The in-flight row, whose ``generation`` is what a commit installs.

        Raises:
            CredentialRotationInterleaved: If another rotation is in flight.
            CredentialRotationBlocked: If an inconclusive rotation is unresolved.
        """

        started_at = self._clock()
        async with self.repository.transaction() as tx:
            latest = await tx.latest_credential_rotation(credential)
            if latest is not None and latest.state is CredentialRotationState.PENDING:
                raise CredentialRotationInterleaved(
                    f"credential {credential!r} already has a rotation to generation {latest.generation} in flight"
                )
            if latest is not None and latest.state is CredentialRotationState.INCONCLUSIVE:
                raise CredentialRotationBlocked(
                    f"credential {credential!r} has an unresolved rotation to generation {latest.generation} whose "
                    "outcome is unknown; resolve it before rotating again"
                )
            # Derived after the locked read, never before it: a generation
            # computed from a snapshot taken outside the lock is exactly how two
            # replicas both decide they are producing generation N.
            generation = (1 if latest is None else latest.generation) + 1
            record = CredentialRotationRecord(
                id=self._id_factory(),
                credential=credential,
                generation=generation,
                fence=fence,
                state=CredentialRotationState.PENDING,
                overlap_s=overlap_s,
                started_at=started_at,
            )
            await tx.put_credential_rotation(record)
            return record

    async def commit(self, *, credential: str, generation: int, fence: str) -> CredentialRotationRecord:
        """Record that the new material is installed and retire the old one.

        The retirement instant is computed here, from this transaction's clock
        and the overlap fixed at ``begin``, so every replica reads one agreed
        point rather than each deriving its own.
        """

        resolved_at = self._clock()
        async with self.repository.transaction() as tx:
            record = await self._require_in_flight(tx, credential, generation, fence)
            committed = record.model_copy(
                update={
                    "state": CredentialRotationState.COMMITTED,
                    "resolved_at": resolved_at,
                    "previous_not_after": resolved_at + timedelta(seconds=record.overlap_s),
                }
            )
            await tx.put_credential_rotation(committed)
            return committed

    async def abandon(
        self,
        *,
        credential: str,
        generation: int,
        fence: str,
        detail: str = "",
    ) -> CredentialRotationRecord:
        """Record that the rotation definitely did not happen.

        Only for a caller that established it: a provider that refused the write,
        a validation that failed before anything was sent. Everything else --
        every timeout, every dropped connection, every ambiguous 5xx -- is
        :meth:`record_inconclusive`.
        """

        resolved_at = self._clock()
        async with self.repository.transaction() as tx:
            record = await self._require_in_flight(tx, credential, generation, fence)
            abandoned = record.model_copy(
                update={
                    "state": CredentialRotationState.ABANDONED,
                    "resolved_at": resolved_at,
                    "detail": _safe_detail(detail),
                }
            )
            await tx.put_credential_rotation(abandoned)
            return abandoned

    async def record_inconclusive(
        self,
        *,
        credential: str,
        generation: int,
        fence: str,
        detail: str = "",
    ) -> CredentialRotationRecord:
        """Record that the rotation's outcome could not be observed.

        This is the fail-closed path. The row stands until a human resolves it,
        :meth:`begin` refuses while it does, and no code path here converts it to
        either outcome. ``previous_not_after`` is deliberately left unset: the
        superseded generation cannot be given a retirement instant by a rotation
        that does not know whether it superseded anything.
        """

        resolved_at = self._clock()
        async with self.repository.transaction() as tx:
            record = await self._require_in_flight(tx, credential, generation, fence)
            inconclusive = record.model_copy(
                update={
                    "state": CredentialRotationState.INCONCLUSIVE,
                    "resolved_at": resolved_at,
                    "detail": _safe_detail(detail),
                }
            )
            await tx.put_credential_rotation(inconclusive)
            return inconclusive

    async def resolve(
        self,
        *,
        credential: str,
        generation: int,
        fence: str,
        committed: bool,
        detail: str = "",
    ) -> CredentialRotationRecord:
        """Close an inconclusive rotation with an outcome someone established.

        Args:
            credential: Logical credential name.
            generation: The inconclusive rotation's generation.
            fence: The fence that rotation was begun under.
            committed: Whether the new material turned out to be installed.
            detail: Operator-safe note. Never a provider message or a path.

        Returns:
            The now-terminal row.

        Raises:
            CredentialRotationFenceMismatch: If no inconclusive rotation of this
                credential and generation is held under this fence.
        """

        resolved_at = self._clock()
        async with self.repository.transaction() as tx:
            record = await tx.get_credential_rotation(credential, generation)
            if record is None or record.state is not CredentialRotationState.INCONCLUSIVE or record.fence != fence:
                raise CredentialRotationFenceMismatch(
                    f"credential {credential!r} has no inconclusive rotation to generation {generation} "
                    "held under this fence"
                )
            update: dict[str, object] = {
                "state": CredentialRotationState.COMMITTED if committed else CredentialRotationState.ABANDONED,
                "resolved_at": resolved_at,
                "detail": _safe_detail(detail),
            }
            if committed:
                update["previous_not_after"] = resolved_at + timedelta(seconds=record.overlap_s)
            resolved = record.model_copy(update=update)
            await tx.put_credential_rotation(resolved)
            return resolved

    @staticmethod
    async def _require_in_flight(
        tx: RepositorySession,
        credential: str,
        generation: int,
        fence: str,
    ) -> CredentialRotationRecord:
        record = await tx.get_credential_rotation(credential, generation)
        if record is None or record.state is not CredentialRotationState.PENDING or record.fence != fence:
            raise CredentialRotationFenceMismatch(
                f"credential {credential!r} has no rotation to generation {generation} in flight under this fence"
            )
        return record


def _safe_detail(detail: str) -> str:
    """Bound an operator note without echoing anything structured.

    Callers pass their own sentence; this only makes sure a provider message
    someone pasted in cannot make the row unwritable or unreadable.
    """

    text = "" if detail is None else str(detail)
    text = "".join(character for character in text if character.isprintable())
    return text[:512]


__all__ = (
    "SECRET_CAPABILITY_CREDENTIAL",
    "CredentialRotationService",
    "new_rotation_fence",
    "new_rotation_id",
)
