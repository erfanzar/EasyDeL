# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fenced rotation of the credential material Eyvan signs and authenticates with.

Three properties, and each of them is a separate failure this file exists to
make impossible.

**A running workload is not restarted.** Rotation replaces the material a
process *signs* with while keeping the material it *verifies* with wider for a
bounded overlap, so a capability minted a minute before the rotation is still
honoured a minute after it. Nothing in flight is invalidated by the act of
rotating; that is what makes rotation something an operator will actually do.

**Old material stops at a definite point, not "eventually".** The retiring
epoch carries an absolute ``not_after``. It is not derived from the longest
token TTL, not from a restart, and not from the last observed use -- all three
are ways of saying "we hope nobody is still holding one". After ``not_after``
the retired material verifies nothing, whatever a token's own expiry claims.

**Two rotations cannot interleave.** A rotation is begun under a fence and can
only be resolved by the holder of that fence. Two operators, or two replicas
reacting to the same alert, produce one rotation and one refusal rather than two
generations racing to be the active one.

The fourth property is the one that is easy to get wrong: a rotation whose
outcome could not be observed **fails closed**. It is recorded as
:attr:`CredentialRotationState.INCONCLUSIVE` -- not committed, not abandoned --
and the credential stops issuing new material until a human resolves it.
Verification of already-issued material continues, because refusing that would
turn an unknown into a fleet outage; minting under a key whose distribution is
unknown would turn it into a silent one. There is no automatic retry: an
inconclusive rotation is terminal until resolved, so it cannot spin forever
against a provider that will never answer.

Nothing here logs, reprs, formats, or compares material. The exception messages
in this module name generations and states only, and :class:`RotatingCredential`
carries an explicit ``__repr__`` for the same reason: the default one would put
the signing key in every traceback that happened to include a keyring.
"""

from __future__ import annotations

import math
import time
import typing as tp
from dataclasses import dataclass

from eyvan._internal.compat import StrEnum

#: Shortest material accepted for any Eyvan signing credential. Matches the
#: capability-token floor so a rotation cannot quietly weaken what it replaces.
MINIMUM_CREDENTIAL_BYTES: tp.Final = 32


class CredentialRotationState(StrEnum):
    """How one rotation of one credential ended, or that it has not ended."""

    #: Begun under a fence; the new material has not been confirmed anywhere.
    PENDING = "pending"
    #: The new material is in effect and the previous generation has a definite
    #: ``not_after``.
    COMMITTED = "committed"
    #: Explicitly known not to have happened. Nothing changed; the previous
    #: generation keeps signing.
    ABANDONED = "abandoned"
    #: The outcome could not be observed. Never inferred to be either of the
    #: two above, and never retried automatically.
    INCONCLUSIVE = "inconclusive"


#: States in which a rotation is finished and a new one may begin.
RESOLVED_ROTATION_STATES: tp.Final = frozenset({CredentialRotationState.COMMITTED, CredentialRotationState.ABANDONED})


class CredentialRotationError(RuntimeError):
    """Base error for every refusal in this module."""


class CredentialRotationInterleaved(CredentialRotationError):
    """A second rotation was begun while one was still in flight."""


class CredentialRotationFenceMismatch(CredentialRotationError):
    """A rotation was resolved by something other than the fence that began it."""


class CredentialRotationBlocked(CredentialRotationError):
    """An unresolved inconclusive rotation is holding this credential closed.

    Raised both when new material is asked for and when a further rotation is
    attempted. Both are the same refusal: nobody knows what the last rotation
    did, so this process will not add a second unknown on top of it.
    """


@dataclass(frozen=True, slots=True)
class CredentialEpoch:
    """One generation's validity window, without its material.

    Deliberately materialless. Diagnostics, metrics and operator output all want
    "which generations are live and until when", and every one of those call
    sites is somewhere a key must never reach.
    """

    generation: int
    #: ``None`` for the generation currently signing; otherwise the absolute
    #: epoch second after which this generation verifies nothing.
    not_after: float | None = None

    def is_live(self, now: float) -> bool:
        """Return whether this generation still verifies at ``now``."""

        return self.not_after is None or now < self.not_after


def _require_material(material: bytes) -> bytes:
    """Reject material too short to be a signing credential.

    The value is never named in the error, only its length requirement.
    """

    if not isinstance(material, bytes) or len(material) < MINIMUM_CREDENTIAL_BYTES:
        raise ValueError(f"credential material must contain at least {MINIMUM_CREDENTIAL_BYTES} bytes")
    return material


def _require_fence(fence: str) -> str:
    """Reject a fence that cannot identify one rotation."""

    if not isinstance(fence, str) or not fence or fence != fence.strip():
        raise ValueError("a credential rotation requires a non-empty trimmed fence")
    if len(fence) > 64:
        raise ValueError("a credential rotation fence must be at most 64 characters")
    return fence


def _require_overlap(overlap_s: float) -> float:
    """Reject an overlap that would not produce a definite point."""

    if isinstance(overlap_s, bool) or not isinstance(overlap_s, (int, float)):
        raise ValueError("a credential rotation overlap must be a number of seconds")
    value = float(overlap_s)
    if not math.isfinite(value) or value < 0:
        raise ValueError("a credential rotation overlap must be finite and non-negative")
    return value


class CapabilityKeyring(tp.Protocol):
    """What a capability signer needs from a credential, rotating or not."""

    def signing_material(self) -> bytes:
        """Return the material new tokens must be signed with."""
        ...

    def verification_materials(self) -> tuple[bytes, ...]:
        """Return every material a presented token may still be verified against."""
        ...


class StaticCredential:
    """One material that never rotates -- the behaviour before this module.

    Kept as a real class rather than a special case inside the signer so that
    "this deployment does not rotate" and "this deployment rotates" travel the
    same code path, and the rotating path is therefore exercised by every
    existing test of the non-rotating one.
    """

    __slots__ = ("_material",)

    def __init__(self, material: bytes):
        self._material = _require_material(material)

    def signing_material(self) -> bytes:
        """Return the single material."""

        return self._material

    def verification_materials(self) -> tuple[bytes, ...]:
        """Return the single material."""

        return (self._material,)

    def __repr__(self) -> str:
        """Describe the credential without its material."""

        return "StaticCredential()"


class RotatingCredential:
    """A credential whose material can be replaced under a fence.

    The state machine, in the order an operator drives it::

        begin(fence)                     -> the generation this will produce
        ... install the new material at the provider, out of band ...
        commit(fence, material, overlap) -> new material signs; old dies at a point
        abandon(fence)                   -> nothing changed
        mark_inconclusive(fence)         -> outcome unknown; signing stops

    The out-of-band step is deliberately outside this object and outside any
    transaction: installing material talks to a provider, and holding a lock
    across that call is how a rotation takes a control plane down with it.
    """

    __slots__ = ("_clock", "_epochs", "_pending_fence", "_pending_generation", "_state")

    def __init__(
        self,
        material: bytes,
        *,
        generation: int = 1,
        clock: tp.Callable[[], float] = time.time,
    ):
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ValueError("a credential generation must be a positive integer")
        self._clock = clock
        self._epochs: list[tuple[CredentialEpoch, bytes]] = [
            (CredentialEpoch(generation=generation), _require_material(material))
        ]
        self._state: CredentialRotationState | None = None
        self._pending_fence: str | None = None
        self._pending_generation: int | None = None

    @property
    def generation(self) -> int:
        """Return the generation currently signing."""

        return self._epochs[0][0].generation

    @property
    def rotation_state(self) -> CredentialRotationState | None:
        """Return the state of the most recent rotation, or ``None`` if never rotated."""

        return self._state

    @property
    def pending_generation(self) -> int | None:
        """Return the generation an in-flight rotation will produce, if any."""

        return self._pending_generation if self._state is CredentialRotationState.PENDING else None

    def epochs(self) -> tuple[CredentialEpoch, ...]:
        """Return every generation still able to verify, newest first, without material."""

        now = self._clock()
        return tuple(epoch for epoch, _ in self._epochs if epoch.is_live(now))

    def signing_material(self) -> bytes:
        """Return the material new tokens are signed with.

        Raises:
            CredentialRotationBlocked: If the last rotation's outcome is unknown.
                Failing closed here is the whole point: a token signed under a
                key that may or may not have reached the other replicas is a
                token nobody can classify afterwards.
        """

        if self._state is CredentialRotationState.INCONCLUSIVE:
            raise CredentialRotationBlocked(
                "credential rotation outcome is unknown; resolve the rotation before issuing new material"
            )
        return self._epochs[0][1]

    def verification_materials(self) -> tuple[bytes, ...]:
        """Return every material a presented token may still verify against.

        Newest first, and filtered by the absolute ``not_after`` rather than by
        anything a token carries. Retired material stops verifying at that
        instant even when the token presenting it has not expired.
        """

        now = self._clock()
        return tuple(material for epoch, material in self._epochs if epoch.is_live(now))

    def begin(self, fence: str) -> int:
        """Claim the right to rotate and return the generation it will produce.

        Args:
            fence: Identifier only the caller resolving this rotation holds.

        Returns:
            The generation a successful commit will install.

        Raises:
            CredentialRotationInterleaved: If a rotation is already in flight.
            CredentialRotationBlocked: If an inconclusive rotation is unresolved.
        """

        fence = _require_fence(fence)
        if self._state is CredentialRotationState.PENDING:
            raise CredentialRotationInterleaved(
                f"credential rotation to generation {self._pending_generation} is already in flight"
            )
        if self._state is CredentialRotationState.INCONCLUSIVE:
            raise CredentialRotationBlocked(
                "credential rotation outcome is unknown; resolve the rotation before beginning another"
            )
        self._state = CredentialRotationState.PENDING
        self._pending_fence = fence
        self._pending_generation = self.generation + 1
        return self._pending_generation

    def commit(self, fence: str, material: bytes, *, overlap_s: float) -> CredentialEpoch:
        """Install new material and give the previous generation a definite end.

        Args:
            fence: The fence returned to the caller of :meth:`begin`.
            material: The new signing material.
            overlap_s: Seconds the previous generation keeps verifying. ``0``
                retires it immediately.

        Returns:
            The retired generation's epoch, carrying its absolute ``not_after``.

        Raises:
            CredentialRotationFenceMismatch: If no rotation is in flight under
                this fence.
        """

        self._require_pending(fence)
        overlap = _require_overlap(overlap_s)
        material = _require_material(material)
        assert self._pending_generation is not None
        retired = CredentialEpoch(generation=self.generation, not_after=self._clock() + overlap)
        previous = self._epochs[0][1]
        self._epochs = [
            (CredentialEpoch(generation=self._pending_generation), material),
            (retired, previous),
            *[(epoch, held) for epoch, held in self._epochs[1:]],
        ]
        self._state = CredentialRotationState.COMMITTED
        self._pending_fence = None
        self._pending_generation = None
        return retired

    def abandon(self, fence: str) -> None:
        """Record that the rotation definitely did not happen.

        Nothing changes: the previous generation keeps signing, and a new
        rotation may begin. Only a caller that *knows* the new material was not
        installed may call this -- everything else is
        :meth:`mark_inconclusive`.
        """

        self._require_pending(fence)
        self._state = CredentialRotationState.ABANDONED
        self._pending_fence = None
        self._pending_generation = None

    def mark_inconclusive(self, fence: str) -> None:
        """Record that the rotation's outcome could not be observed.

        Fails closed: :meth:`signing_material` and :meth:`begin` both refuse
        until :meth:`resolve` is called with the same fence. Verification is
        left alone, so nothing already issued is invalidated by an operator's
        network timeout.
        """

        self._require_pending(fence)
        self._state = CredentialRotationState.INCONCLUSIVE

    def resolve(self, fence: str, *, material: bytes | None = None, overlap_s: float = 0.0) -> None:
        """Close an inconclusive rotation with an outcome a human established.

        Args:
            fence: The fence the inconclusive rotation was begun under.
            material: The new material, if the rotation is now known to have
                happened. ``None`` records that it did not.
            overlap_s: Seconds the previous generation keeps verifying when
                ``material`` is supplied.

        Raises:
            CredentialRotationFenceMismatch: If the credential is not blocked on
                this fence.
        """

        fence = _require_fence(fence)
        if self._state is not CredentialRotationState.INCONCLUSIVE or self._pending_fence != fence:
            raise CredentialRotationFenceMismatch("no inconclusive credential rotation is held under this fence")
        # Validated before anything is mutated. Reopening the rotation and then
        # discovering the new material is malformed would leave the credential
        # in PENDING -- issuing again, under the material an operator was in the
        # middle of retiring. Failing closed has to survive the resolution too.
        if material is not None:
            _require_material(material)
            _require_overlap(overlap_s)
        self._state = CredentialRotationState.PENDING
        if material is None:
            self.abandon(fence)
        else:
            self.commit(fence, material, overlap_s=overlap_s)

    def _require_pending(self, fence: str) -> None:
        fence = _require_fence(fence)
        if self._state is not CredentialRotationState.PENDING or self._pending_fence != fence:
            raise CredentialRotationFenceMismatch("no credential rotation is in flight under this fence")

    def __repr__(self) -> str:
        """Describe generations and state, never material."""

        state = "never-rotated" if self._state is None else str(self._state)
        return f"RotatingCredential(generation={self.generation}, state={state}, epochs={len(self._epochs)})"


__all__ = (
    "MINIMUM_CREDENTIAL_BYTES",
    "RESOLVED_ROTATION_STATES",
    "CapabilityKeyring",
    "CredentialEpoch",
    "CredentialRotationBlocked",
    "CredentialRotationError",
    "CredentialRotationFenceMismatch",
    "CredentialRotationInterleaved",
    "CredentialRotationState",
    "RotatingCredential",
    "StaticCredential",
)
