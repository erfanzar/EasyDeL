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
"""Versioned learner-to-sampler weight transport for disaggregated asynchronous RL.

The learner owns a :class:`WeightPublisher`; every sampler owns a subscriber.
The protocol is pull-based and versioned:

1. A subscriber calls ``request()``; it may hold at most one outstanding
   request token (or one unsettled offer) at a time.
2. ``publish(version, snapshot)`` stages the snapshot. The publisher keeps at
   most one staged snapshot: staging a newer one drops the older one, and a
   snapshot is released (``backend.release`` and the caller's ``release``
   callback, exactly once) as soon as nothing references it any more.
3. The newest staged version is offered only to subscribers with a request
   outstanding (backpressure: a busy sampler never receives weights it did not
   ask for). Each :class:`WeightOffer` names its ``predecessor`` -- the version
   the publisher believes the subscriber has installed -- so the subscriber
   verifies the chain and detects divergence. Version jumps are allowed but
   explicit (:attr:`WeightOffer.skipped_versions`).
4. An offer that was not yet pulled is *superseded* when a newer version is
   published; the request token is re-armed and the newer version offered.
5. The subscriber pulls the payload, installs it, and ``ack()``\\s, or
   ``decline()``\\s it. A pull that does not settle within ``pull_timeout``
   seconds trips the watchdog: the offer is settled as ``STALLED``, its
   snapshot reference dropped, and :class:`WeightTransferStalled` raised.

Every offer is settled exactly once with one :class:`OfferOutcome` (acked,
declined, superseded, stalled or closed), on every path including ``close()``.

Backends decide how a staged snapshot reaches a sampler:

* :class:`InProcessWeightTransport` hands the pytree itself to samplers living
  in the learner process.
* :class:`JaxTransferServerBackend` + :class:`JaxTransferClient` move device
  arrays between processes with ``jax.experimental.transfer``; the control
  messages (request/offer/begin/ack) travel over a :class:`SocketMessageChannel`
  between a :class:`RemoteSubscriberBridge` (learner side) and a
  :class:`RemoteWeightSubscriber` (sampler side).

This module is opt-in for disaggregated rollouts: nothing in the default trainer path imports it. When the learner and
eSurge share chips, ``eSurge.update_model_weights`` swaps weights in place and no transport is needed.

Security: the control channel is authenticated (``auth_token``, a role-bound HMAC challenge-response with capped
pre-authentication frames and a handshake deadline; without a token only Unix-domain or loopback peers are accepted).
The payload served by :class:`JaxTransferServerBackend` travels over ``jax.experimental.transfer``, which has no
authentication or encryption, and its pull ids are derived from the offer id. Anyone who can reach the data-plane
addresses can pull the weights, so bind them only on a trusted network.
"""

from __future__ import annotations

import base64
import collections
import contextlib
import dataclasses
import enum
import itertools
import json
import logging
import math
import selectors
import socket
import struct
import threading
import time
import types
import typing as tp

logger = logging.getLogger(__name__)

__all__ = (
    "InProcessWeightTransport",
    "JaxTransferClient",
    "JaxTransferServerBackend",
    "LeafSpec",
    "OfferOutcome",
    "RemoteSubscriberBridge",
    "RemoteWeightSubscriber",
    "SocketMessageChannel",
    "WeightOffer",
    "WeightOfferSettled",
    "WeightPublisher",
    "WeightSubscriber",
    "WeightTransferStalled",
    "WeightTransportBackend",
    "WeightTransportClosed",
    "WeightTransportError",
    "WeightVersionChainError",
    "verify_offer_chain",
)


class WeightTransportError(RuntimeError):
    """Base class for weight-transport failures."""


class WeightTransportClosed(WeightTransportError):
    """The publisher, subscriber or channel was closed."""


class WeightVersionChainError(WeightTransportError):
    """An offer does not extend the version the subscriber has installed."""


class WeightTransferStalled(WeightTransportError):
    """A pull did not settle within the configured ``pull_timeout``."""


class OfferOutcome(enum.StrEnum):
    """Final settlement of a :class:`WeightOffer`."""

    ACKED = "acked"
    DECLINED = "declined"
    SUPERSEDED = "superseded"
    STALLED = "stalled"
    CLOSED = "closed"


class WeightOfferSettled(WeightTransportError):
    """An operation targeted an offer that has already been settled."""

    def __init__(self, message: str, outcome: OfferOutcome | None = None):
        super().__init__(message)
        self.outcome = outcome


@dataclasses.dataclass(frozen=True, slots=True)
class WeightOffer:
    """One offer of a staged snapshot to one subscriber.

    Args:
        offer_id: Publisher-unique offer identifier.
        subscriber: Name of the subscriber the offer was made to.
        version: Offered snapshot version.
        predecessor: Version the publisher believes the subscriber has
            installed (``None`` for a subscriber that never installed one).
        descriptor: Backend metadata needed to fetch the payload (JSON-able).
        metadata: Caller metadata attached at ``publish`` time.
    """

    offer_id: int
    subscriber: str
    version: int
    predecessor: int | None
    descriptor: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)
    metadata: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)

    @property
    def skipped_versions(self) -> int:
        """Number of versions this offer jumps over (0 for a consecutive version)."""
        if self.predecessor is None:
            return 0
        return self.version - self.predecessor - 1

    def to_wire(self) -> dict[str, tp.Any]:
        """Return a JSON-serializable representation."""
        return {
            "offer_id": self.offer_id,
            "subscriber": self.subscriber,
            "version": self.version,
            "predecessor": self.predecessor,
            "descriptor": dict(self.descriptor),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_wire(cls, data: tp.Mapping[str, tp.Any]) -> WeightOffer:
        """Inverse of :meth:`to_wire`."""
        return cls(
            offer_id=int(data["offer_id"]),
            subscriber=str(data["subscriber"]),
            version=int(data["version"]),
            predecessor=None if data["predecessor"] is None else int(data["predecessor"]),
            descriptor=types.MappingProxyType(dict(data.get("descriptor") or {})),
            metadata=types.MappingProxyType(dict(data.get("metadata") or {})),
        )


def verify_offer_chain(offer: WeightOffer, installed_version: int | None) -> None:
    """Check that ``offer`` extends ``installed_version``.

    Raises:
        WeightVersionChainError: If the offer's predecessor differs from the
            installed version (a gap: the two sides disagree about what the
            sampler holds) or the offer does not move the version forward.
    """
    if offer.predecessor != installed_version:
        raise WeightVersionChainError(
            f"offer {offer.offer_id} for version {offer.version} names predecessor {offer.predecessor}, "
            f"but subscriber {offer.subscriber!r} has version {installed_version} installed"
        )
    if installed_version is not None and offer.version <= installed_version:
        raise WeightVersionChainError(
            f"offer {offer.offer_id} does not advance subscriber {offer.subscriber!r}: "
            f"offered version {offer.version} <= installed version {installed_version}"
        )


class WeightTransportBackend(tp.Protocol):
    """Publisher-side staging backend."""

    def stage(self, version: int, snapshot: tp.Any) -> tp.Any:
        """Prepare ``snapshot`` for transfer and return an opaque handle."""
        ...

    def describe(self, handle: tp.Any) -> tp.Mapping[str, tp.Any]:
        """Return JSON-able metadata a remote subscriber needs to fetch ``handle``."""
        ...

    def serve(self, handle: tp.Any, offer_id: int, timeout: float | None) -> None:
        """Make ``handle`` fetchable for one pull of offer ``offer_id``."""
        ...

    def release(self, handle: tp.Any) -> None:
        """Drop every resource held for ``handle``."""
        ...


class InProcessWeightTransport:
    """Backend for samplers in the learner process: the payload is the pytree itself.

    Nothing is copied; the subscriber receives the very object passed to
    :meth:`WeightPublisher.publish` (e.g. a graphstate fed to an eSurge
    engine's ``update_model_weights``). The caller's ``release`` callback runs
    once no offer references the snapshot any more.
    """

    def stage(self, version: int, snapshot: tp.Any) -> tp.Any:
        del version
        return snapshot

    def describe(self, handle: tp.Any) -> tp.Mapping[str, tp.Any]:
        del handle
        return {"kind": "in_process"}

    def serve(self, handle: tp.Any, offer_id: int, timeout: float | None) -> None:
        del handle, offer_id, timeout

    def release(self, handle: tp.Any) -> None:
        del handle


@dataclasses.dataclass(eq=False)
class _Staged:
    version: int
    snapshot: tp.Any
    handle: tp.Any
    descriptor: tp.Mapping[str, tp.Any]
    metadata: tp.Mapping[str, tp.Any]
    release: tp.Callable[[tp.Any], None] | None
    refs: int = 1
    released: bool = False


@dataclasses.dataclass(eq=False)
class _OfferRecord:
    offer: WeightOffer
    staged: _Staged
    state: str = "offered"  # offered -> pulling -> settled
    outcome: OfferOutcome | None = None
    pulling_since: float | None = None


@dataclasses.dataclass(eq=False)
class _SubscriberRecord:
    name: str
    installed: int | None
    requested: bool = False
    active: _OfferRecord | None = None
    closed: bool = False


class WeightPublisher:
    """Learner-side publisher of versioned weight snapshots (thread-safe).

    Args:
        backend: Staging backend; :class:`InProcessWeightTransport` by default.
        pull_timeout: Seconds a subscriber may spend between ``pull`` and
            ``ack``/``decline`` before the watchdog settles the offer as
            ``STALLED``. ``None`` disables the watchdog.
        watchdog_interval: If set (and ``pull_timeout`` is set), a daemon thread
            checks for stalled pulls every ``watchdog_interval`` seconds so a
            stalled snapshot is released without waiting for the next call.
            The error is raised by the next ``publish``/``check_watchdog``.
        clock: Monotonic clock (injectable for tests).
        history_size: Number of recent settlements kept in :attr:`settlements`.
    """

    def __init__(
        self,
        backend: WeightTransportBackend | None = None,
        *,
        pull_timeout: float | None = None,
        watchdog_interval: float | None = None,
        clock: tp.Callable[[], float] = time.monotonic,
        history_size: int = 1024,
    ):
        if pull_timeout is not None and (not math.isfinite(pull_timeout) or pull_timeout <= 0):
            raise ValueError(f"pull_timeout must be finite and positive, got {pull_timeout}")
        if watchdog_interval is not None and (not math.isfinite(watchdog_interval) or watchdog_interval <= 0):
            raise ValueError(f"watchdog_interval must be finite and positive, got {watchdog_interval}")
        self.backend: WeightTransportBackend = backend if backend is not None else InProcessWeightTransport()
        self.pull_timeout = pull_timeout
        self._clock = clock
        self._cond = threading.Condition(threading.RLock())
        self._staged: _Staged | None = None
        self._latest_version: int | None = None
        self._subscribers: dict[str, _SubscriberRecord] = {}
        self._offer_ids = itertools.count(1)
        self._to_release: list[_Staged] = []
        self._stalled: list[WeightOffer] = []
        self._history: collections.deque[tuple[int, int, str, OfferOutcome]] = collections.deque(maxlen=history_size)
        self._closed = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        if pull_timeout is not None and watchdog_interval is not None:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                args=(float(watchdog_interval),),
                name="weight-publisher-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    # ------------------------------------------------------------------ public API

    @property
    def latest_version(self) -> int | None:
        """Most recently published version."""
        with self._cond:
            return self._latest_version

    @property
    def staged_version(self) -> int | None:
        """Version of the staged snapshot, or ``None`` if none is staged."""
        with self._cond:
            return None if self._staged is None else self._staged.version

    @property
    def settlements(self) -> tuple[tuple[int, int, str, OfferOutcome], ...]:
        """Recent settlements as ``(offer_id, version, subscriber, outcome)`` tuples."""
        with self._cond:
            return tuple(self._history)

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def installed_version(self, name: str) -> int | None:
        """Version the publisher believes subscriber ``name`` has installed."""
        with self._cond:
            return self._subscribers[name].installed

    def subscribe(self, name: str, *, installed_version: int | None = None) -> WeightSubscriber:
        """Register a subscriber.

        Args:
            name: Unique subscriber name.
            installed_version: Version the sampler already holds (e.g. the
                weights it was built from), or ``None``.
        """
        with self._locked():
            self._ensure_open()
            if name in self._subscribers:
                raise ValueError(f"subscriber {name!r} is already registered")
            self._subscribers[name] = _SubscriberRecord(name=name, installed=installed_version)
        return WeightSubscriber(self, name, installed_version)

    def publish(
        self,
        version: int,
        snapshot: tp.Any,
        *,
        release: tp.Callable[[tp.Any], None] | None = None,
        metadata: tp.Mapping[str, tp.Any] | None = None,
    ) -> None:
        """Stage ``snapshot`` as ``version`` and offer it to every requesting subscriber.

        Versions must strictly increase. The previously staged snapshot is
        dropped, and unpulled offers of it are superseded.

        Args:
            version: New, strictly larger version.
            snapshot: Payload (typically an inference-only pytree).
            release: Called once with ``snapshot`` when it is no longer referenced.
            metadata: JSON-able metadata attached to every offer.

        Raises:
            WeightTransferStalled: A pull stalled; raised *before* staging, so
                ``snapshot`` stays owned by the caller.
        """
        version = int(version)
        with self._locked():
            self._ensure_open()
            self._raise_stalled_locked()
            if self._latest_version is not None and version <= self._latest_version:
                raise ValueError(f"version must increase: {version} <= {self._latest_version}")
            handle = self.backend.stage(version, snapshot)
            try:
                descriptor = types.MappingProxyType(dict(self.backend.describe(handle)))
                frozen_metadata = types.MappingProxyType(dict(metadata or {}))
            except BaseException:
                with contextlib.suppress(Exception):
                    self.backend.release(handle)
                raise
            staged = _Staged(
                version=version,
                snapshot=snapshot,
                handle=handle,
                descriptor=descriptor,
                metadata=frozen_metadata,
                release=release,
            )
            previous, self._staged, self._latest_version = self._staged, staged, version
            if previous is not None:
                self._unref_locked(previous)
            for record in self._subscribers.values():
                active = record.active
                if active is not None and active.state == "offered":
                    self._settle_locked(active, OfferOutcome.SUPERSEDED)
                    record.requested = True
                self._dispatch_locked(record)
            self._cond.notify_all()

    def check_watchdog(self) -> None:
        """Settle stalled pulls and raise :class:`WeightTransferStalled` if any were found."""
        with self._locked():
            self._raise_stalled_locked()

    def close(self) -> None:
        """Settle every open offer as ``CLOSED``, release the staged snapshot. Idempotent."""
        with self._locked():
            if self._closed:
                return
            self._closed = True
            for record in self._subscribers.values():
                if record.active is not None:
                    self._settle_locked(record.active, OfferOutcome.CLOSED)
                record.requested = False
                record.closed = True
            if self._staged is not None:
                self._unref_locked(self._staged)
                self._staged = None
            self._cond.notify_all()
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def __enter__(self) -> WeightPublisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- subscriber ops

    def _request(self, name: str) -> None:
        with self._locked():
            record = self._record(name)
            if record.requested or record.active is not None:
                raise WeightTransportError(
                    f"subscriber {name!r} already has an outstanding request or unsettled offer; "
                    "settle it before requesting again"
                )
            record.requested = True
            self._dispatch_locked(record)
            self._cond.notify_all()

    def _receive(self, name: str, timeout: float | None) -> WeightOffer | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._locked():
            while True:
                record = self._record(name)
                active = record.active
                if active is not None and active.state == "offered":
                    return active.offer
                if active is not None:
                    raise WeightTransportError(
                        f"subscriber {name!r} has an active offer {active.offer.offer_id} "
                        f"in state {active.state!r}; settle it before receiving again"
                    )
                if not record.requested:
                    raise WeightTransportError(f"subscriber {name!r} has no outstanding request to receive on")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)

    def _begin_pull(self, name: str, offer: WeightOffer) -> _Staged:
        with self._locked():
            active = self._active_for(name, offer)
            if active.state != "offered":
                raise WeightTransportError(f"offer {offer.offer_id} is already being pulled")
            self.backend.serve(active.staged.handle, offer.offer_id, self.pull_timeout)
            active.state = "pulling"
            active.pulling_since = self._clock()
            return active.staged

    def _ack(self, name: str, offer: WeightOffer) -> None:
        with self._locked():
            active = self._active_for(name, offer)
            if active.state != "pulling":
                raise WeightTransportError(f"offer {offer.offer_id} must be pulled before it is acknowledged")
            self._settle_locked(active, OfferOutcome.ACKED)
            self._dispatch_locked(self._record(name))

    def _decline(self, name: str, offer: WeightOffer) -> bool:
        with self._locked():
            record = self._record(name)
            active = record.active
            if active is None or active.offer.offer_id != offer.offer_id:
                return False
            self._settle_locked(active, OfferOutcome.DECLINED)
            return True

    def _unsubscribe(self, name: str) -> None:
        with self._locked():
            record = self._subscribers.pop(name, None)
            if record is None:
                return
            if record.active is not None:
                self._settle_locked(record.active, OfferOutcome.CLOSED)
            record.requested = False
            record.closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------------ internals

    @contextlib.contextmanager
    def _locked(self) -> tp.Iterator[None]:
        """Hold the lock; run snapshot releases after it is dropped."""
        try:
            with self._cond:
                yield
        finally:
            self._drain_releases()

    def _drain_releases(self) -> None:
        with self._cond:
            pending, self._to_release = self._to_release, []
        errors: list[BaseException] = []
        for staged in pending:
            try:
                self.backend.release(staged.handle)
            except BaseException as err:
                errors.append(err)
            if staged.release is not None:
                try:
                    staged.release(staged.snapshot)
                except BaseException as err:
                    errors.append(err)
        if errors:
            raise errors[0]

    def _ensure_open(self) -> None:
        if self._closed:
            raise WeightTransportClosed("weight publisher is closed")

    def _record(self, name: str) -> _SubscriberRecord:
        record = self._subscribers.get(name)
        if record is None or record.closed:
            raise WeightTransportClosed(f"subscriber {name!r} is closed or unknown")
        return record

    def _active_for(self, name: str, offer: WeightOffer) -> _OfferRecord:
        self._expire_stalled_locked()
        if self._closed:
            raise WeightOfferSettled(f"offer {offer.offer_id} was closed with the publisher", OfferOutcome.CLOSED)
        record = self._subscribers.get(name)
        active = None if record is None else record.active
        if active is None or active.offer.offer_id != offer.offer_id:
            outcome = self._outcome_of(offer.offer_id)
            if outcome is OfferOutcome.STALLED:
                raise WeightTransferStalled(
                    f"offer {offer.offer_id} (version {offer.version}) to {name!r} was settled as stalled "
                    f"after exceeding pull_timeout={self.pull_timeout}s"
                )
            raise WeightOfferSettled(
                f"offer {offer.offer_id} (version {offer.version}) to {name!r} is already settled ({outcome})",
                outcome,
            )
        return active

    def _outcome_of(self, offer_id: int) -> OfferOutcome | None:
        for settled_id, _, _, outcome in reversed(self._history):
            if settled_id == offer_id:
                return outcome
        return None

    def _dispatch_locked(self, record: _SubscriberRecord) -> None:
        staged = self._staged
        if record.closed or not record.requested or record.active is not None or staged is None:
            return
        if record.installed is not None and staged.version <= record.installed:
            return
        offer = WeightOffer(
            offer_id=next(self._offer_ids),
            subscriber=record.name,
            version=staged.version,
            predecessor=record.installed,
            descriptor=staged.descriptor,
            metadata=staged.metadata,
        )
        staged.refs += 1
        record.requested = False
        record.active = _OfferRecord(offer=offer, staged=staged)

    def _settle_locked(self, active: _OfferRecord, outcome: OfferOutcome) -> bool:
        if active.state == "settled":
            return False
        active.state = "settled"
        active.outcome = outcome
        record = self._subscribers.get(active.offer.subscriber)
        if record is not None and record.active is active:
            record.active = None
            if outcome is OfferOutcome.ACKED:
                record.installed = active.offer.version
        self._history.append((active.offer.offer_id, active.offer.version, active.offer.subscriber, outcome))
        self._unref_locked(active.staged)
        self._cond.notify_all()
        return True

    def _unref_locked(self, staged: _Staged) -> None:
        staged.refs -= 1
        if staged.refs == 0 and not staged.released:
            staged.released = True
            self._to_release.append(staged)

    def _expire_stalled_locked(self) -> None:
        if self.pull_timeout is None:
            return
        now = self._clock()
        for record in self._subscribers.values():
            active = record.active
            if (
                active is not None
                and active.state == "pulling"
                and active.pulling_since is not None
                and now - active.pulling_since > self.pull_timeout
            ):
                self._settle_locked(active, OfferOutcome.STALLED)
                self._stalled.append(active.offer)

    def _raise_stalled_locked(self) -> None:
        self._expire_stalled_locked()
        if not self._stalled:
            return
        stalled, self._stalled = self._stalled, []
        detail = ", ".join(f"{offer.subscriber!r} (version {offer.version})" for offer in stalled)
        raise WeightTransferStalled(
            f"weight pull stalled for more than pull_timeout={self.pull_timeout}s: {detail}; "
            "the offers were settled as stalled and their snapshots released"
        )

    def _watchdog_loop(self, interval: float) -> None:
        while not self._watchdog_stop.wait(interval):
            try:
                with self._locked():
                    if self._closed:
                        return
                    self._expire_stalled_locked()
            except Exception:
                logger.warning("WeightPublisher watchdog iteration failed", exc_info=True)


class WeightSubscriber:
    """Sampler-side handle for a subscriber living in the publisher's process.

    Typical loop::

        subscriber.request()
        offer = subscriber.receive(timeout=...)
        weights = subscriber.pull(offer)
        engine.update_model_weights(graphstate=weights)
        subscriber.ack(offer)
    """

    def __init__(self, publisher: WeightPublisher, name: str, installed_version: int | None):
        self._publisher = publisher
        self.name = name
        self._installed_version = installed_version

    @property
    def installed_version(self) -> int | None:
        """Version this subscriber has acknowledged (its own view)."""
        return self._installed_version

    def request(self) -> None:
        """Arm the single request token; the newest version newer than the installed one is offered."""
        self._publisher._request(self.name)

    def receive(self, timeout: float | None = None) -> WeightOffer | None:
        """Wait for an offer; returns ``None`` on timeout.

        Raises:
            WeightVersionChainError: If the offer does not extend the installed version.
        """
        offer = self._publisher._receive(self.name, timeout)
        if offer is not None:
            verify_offer_chain(offer, self._installed_version)
        return offer

    def pull(self, offer: WeightOffer) -> tp.Any:
        """Start pulling ``offer`` and return its payload (the published snapshot).

        Raises:
            WeightOfferSettled: If the offer was superseded/declined/closed meanwhile.
        """
        return self._publisher._begin_pull(self.name, offer).snapshot

    def ack(self, offer: WeightOffer) -> None:
        """Confirm ``offer`` is installed; the subscriber's version becomes ``offer.version``.

        Raises:
            WeightTransferStalled: If the watchdog already settled the pull.
            WeightOfferSettled: If the offer was settled another way.
        """
        self._publisher._ack(self.name, offer)
        self._installed_version = offer.version

    def decline(self, offer: WeightOffer) -> bool:
        """Settle ``offer`` as declined; returns ``False`` if it was already settled."""
        return self._publisher._decline(self.name, offer)

    def close(self) -> None:
        """Unsubscribe; an open offer is settled as ``CLOSED``."""
        self._publisher._unsubscribe(self.name)


# ---------------------------------------------------------------------- remote control channel

_MAX_FRAME_BYTES = 64 << 20
_HANDSHAKE_FRAME_BYTES = 1024
_HANDSHAKE_TIMEOUT = 10.0
_BRIDGE_RECEIVE_POLL_INTERVAL = 0.05
_CLOSE_RPC_TIMEOUT = 2.0
_LEARNER_ROLE = "learner"
_SAMPLER_ROLE = "sampler"


class _PeerDisconnected(Exception):
    """Internal signal raised when the bridge detects peer EOF/RST or local close."""


def _peer_is_local(sock: socket.socket) -> bool:
    """Whether ``sock`` is a Unix-domain socket or a TCP socket whose peer is a loopback address."""
    af_unix = getattr(socket, "AF_UNIX", None)
    if af_unix is not None and sock.family == af_unix:
        return True
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return False
    import ipaddress

    try:
        host = sock.getpeername()[0]
        addr = ipaddress.ip_address(str(host).split("%", 1)[0])
        if addr.is_loopback:
            return True
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            return addr.ipv4_mapped.is_loopback
        return False
    except (OSError, ValueError):
        return False


def _handshake_mac(token: bytes, role: str, own_nonce: bytes, peer_nonce: bytes) -> str:
    import hashlib
    import hmac

    return hmac.new(token, b"|".join((role.encode("ascii"), own_nonce, peer_nonce)), hashlib.sha256).hexdigest()


class SocketMessageChannel:
    """Length-prefixed JSON messages over a stream socket (thread-safe send/recv).

    The channel itself carries no trust. :class:`RemoteSubscriberBridge` and :class:`RemoteWeightSubscriber`
    run :meth:`authenticate` before any RPC: with a shared ``auth_token`` both ends prove they hold it through a
    role-bound HMAC challenge-response, and without one only Unix-domain or loopback TCP peers are accepted.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    @classmethod
    def connect(cls, host: str, port: int, *, timeout: float | None = 30.0) -> SocketMessageChannel:
        """Open a TCP connection to ``host:port``."""
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)
        return cls(sock)

    def authenticate(self, auth_token: str | bytes | None, *, role: str) -> None:
        """Mutually authenticate the peer before any RPC runs on this channel.

        Each side sends a random nonce, then ``HMAC-SHA256(token, role | own_nonce | peer_nonce)``. The MAC is bound
        to the sender's role (learner or sampler), so a reply captured from one side cannot be reflected back, and
        both nonces are fresh, so it cannot be replayed. Handshake frames are capped at a few hundred bytes and the
        exchange has a fixed deadline, so an unauthenticated peer cannot make this side buffer large messages.

        Args:
            auth_token: Shared secret. ``None`` accepts only Unix-domain sockets and loopback TCP peers.
            role: ``"learner"`` or ``"sampler"``; the two ends must use different roles.

        Raises:
            WeightTransportError: If the peer is not local and no token is given, the peer's MAC is wrong, or the
                handshake is malformed. The channel is closed before raising.
        """
        if role not in (_LEARNER_ROLE, _SAMPLER_ROLE):
            raise ValueError(f"role must be {_LEARNER_ROLE!r} or {_SAMPLER_ROLE!r}, got {role!r}")
        if auth_token is None:
            if not _peer_is_local(self._sock):
                self.close()
                raise WeightTransportError("weight-transport control channel to a non-loopback peer requires auth_token")
            return
        import hmac
        import secrets

        token = auth_token.encode("utf-8") if isinstance(auth_token, str) else bytes(auth_token)
        if not token:
            self.close()
            raise WeightTransportError("auth_token must not be empty")
        peer_role = _SAMPLER_ROLE if role == _LEARNER_ROLE else _LEARNER_ROLE
        previous_timeout = self._sock.gettimeout()
        deadline = time.monotonic() + _HANDSHAKE_TIMEOUT
        try:
            own_nonce = secrets.token_hex(32).encode("ascii")
            self.send({"op": "hello", "nonce": own_nonce.decode("ascii")}, deadline=deadline)
            hello = self.recv(max_bytes=_HANDSHAKE_FRAME_BYTES, deadline=deadline)
            peer_nonce = str(hello.get("nonce", "")).encode("ascii")
            if hello.get("op") != "hello" or len(peer_nonce) != 64 or peer_nonce == own_nonce:
                raise WeightTransportError("malformed weight-transport handshake")
            self.send({"op": "auth", "mac": _handshake_mac(token, role, own_nonce, peer_nonce)}, deadline=deadline)
            proof = self.recv(max_bytes=_HANDSHAKE_FRAME_BYTES, deadline=deadline)
            expected = _handshake_mac(token, peer_role, peer_nonce, own_nonce)
            if proof.get("op") != "auth" or not hmac.compare_digest(str(proof.get("mac", "")), expected):
                raise WeightTransportError("weight-transport peer failed authentication")
        except (WeightTransportError, OSError, ValueError, UnicodeError) as err:
            self.close()
            if isinstance(err, WeightTransportError):
                raise
            raise WeightTransportError(f"weight-transport handshake failed: {err}") from err
        self._sock.settimeout(previous_timeout)

    def send(self, message: tp.Mapping[str, tp.Any], *, deadline: float | None = None) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        if len(payload) > _MAX_FRAME_BYTES:
            raise WeightTransportError(f"control message of {len(payload)} bytes exceeds {_MAX_FRAME_BYTES}")
        frame = struct.pack(">I", len(payload)) + payload
        with self._send_lock:
            previous_timeout: float | None = None
            restore_timeout = False
            try:
                if deadline is None:
                    self._sock.sendall(frame)
                else:
                    previous_timeout = self._sock.gettimeout()
                    restore_timeout = True
                    view = memoryview(frame)
                    sent = 0
                    while sent < len(view):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("control channel send timed out")
                        self._sock.settimeout(remaining)
                        written = self._sock.send(view[sent:])
                        if written <= 0:
                            raise WeightTransportClosed("control channel closed by peer")
                        sent += written
            except WeightTransportClosed:
                raise
            except OSError as err:
                raise WeightTransportClosed(f"control channel send failed: {err}") from err
            finally:
                if restore_timeout:
                    with contextlib.suppress(OSError):
                        self._sock.settimeout(previous_timeout)

    def recv(self, *, max_bytes: int = _MAX_FRAME_BYTES, deadline: float | None = None) -> dict[str, tp.Any]:
        with self._recv_lock:
            previous_timeout: float | None = None
            restore_timeout = False
            try:
                if deadline is not None:
                    try:
                        previous_timeout = self._sock.gettimeout()
                        restore_timeout = True
                    except OSError as err:
                        raise WeightTransportClosed(f"control channel recv failed: {err}") from err
                (size,) = struct.unpack(">I", self._recv_exact(4, deadline=deadline))
                if size > max_bytes:
                    raise WeightTransportError(f"control message of {size} bytes exceeds {max_bytes}")
                raw = self._recv_exact(size, deadline=deadline)
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeError, ValueError) as err:
                    raise WeightTransportError(f"malformed control message: {err}") from err
                if not isinstance(message, dict):
                    raise WeightTransportError("control message is not a JSON object")
                return message
            finally:
                if restore_timeout:
                    with contextlib.suppress(OSError):
                        self._sock.settimeout(previous_timeout)

    def _recv_exact(self, size: int, *, deadline: float | None = None) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WeightTransportClosed("control channel recv timed out")
                self._sock.settimeout(remaining)
            try:
                chunk = self._sock.recv(size - len(chunks))
            except OSError as err:
                raise WeightTransportClosed(f"control channel recv failed: {err}") from err
            if not chunk:
                raise WeightTransportClosed("control channel closed by peer")
            chunks.extend(chunk)
        return bytes(chunks)

    def is_peer_closed(self) -> bool:
        """Return whether the socket is closed locally or the peer has sent EOF/RST."""
        with self._recv_lock:
            try:
                fd = self._sock.fileno()
            except OSError:
                return True
            if fd < 0:
                return True
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(fd, selectors.EVENT_READ)
                    events = selector.select(timeout=0)
            except (OSError, ValueError):
                return True
            if not events:
                return False
            flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
            try:
                peeked = self._sock.recv(1, flags)
                return len(peeked) == 0
            except BlockingIOError:
                return False
            except OSError:
                return True

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._sock.close()


_REMOTE_ERRORS: dict[str, type[WeightTransportError]] = {
    cls.__name__: cls
    for cls in (
        WeightTransportError,
        WeightTransportClosed,
        WeightVersionChainError,
        WeightTransferStalled,
        WeightOfferSettled,
    )
}


class RemoteSubscriberBridge:
    """Learner-side endpoint serving one remote subscriber over a :class:`SocketMessageChannel`.

    It registers a local subscriber named ``name`` on ``publisher`` and answers
    the remote side's RPCs (``request``, ``receive``, ``begin``, ``ack``,
    ``decline``, ``close``) on a daemon thread. ``begin`` makes the backend
    serve the payload for that offer (e.g. registers transfer-server pulls).

    Before serving any RPC the thread runs :meth:`SocketMessageChannel.authenticate` as the learner: with
    ``auth_token`` the sampler must prove it holds the same token, and without one only Unix-domain or loopback
    peers are accepted. On failure the channel is closed, :attr:`auth_error` is set and nothing is served;
    :attr:`authenticated` is set on success.

    Security note: the control channel is authenticated, but the payload of :class:`JaxTransferServerBackend`
    travels over ``jax.experimental.transfer``, which has no authentication of its own. Bind its data-plane
    addresses only on a trusted network.
    """

    def __init__(
        self,
        publisher: WeightPublisher,
        channel: SocketMessageChannel,
        name: str,
        *,
        installed_version: int | None = None,
        auth_token: str | bytes | None = None,
    ):
        self._publisher = publisher
        self._channel = channel
        self._auth_token = auth_token
        self.authenticated = threading.Event()
        self.auth_error: WeightTransportError | None = None
        self._closed = False
        self._subscriber = publisher.subscribe(name, installed_version=installed_version)
        self._offers: dict[int, WeightOffer] = {}
        self._thread = threading.Thread(target=self._serve, name=f"weight-bridge-{name}", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def close(self, timeout: float | None = 5.0) -> None:
        """Close the bridge, unsubscribe its subscriber, close its channel, and wait for its serving thread."""
        self._closed = True
        self._subscriber.close()
        self._channel.close()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _serve(self) -> None:
        try:
            try:
                # Authenticate on the serving thread so a bridge and its sampler can be built from one thread.
                self._channel.authenticate(self._auth_token, role=_LEARNER_ROLE)
            except WeightTransportError as err:
                self.auth_error = err
                return
            self.authenticated.set()
            while True:
                try:
                    message = self._channel.recv()
                except WeightTransportClosed:
                    return
                except (WeightTransportError, ValueError, UnicodeError) as err:
                    logger.warning(
                        "Closing weight-transport bridge %r after invalid frame: %s",
                        self._subscriber.name,
                        err,
                    )
                    return
                op = message.get("op")
                try:
                    reply = self._handle(op, message)
                except _PeerDisconnected:
                    return
                except WeightTransportError as err:
                    reply = {"ok": False, "error": type(err).__name__, "message": str(err)}
                    if isinstance(err, WeightOfferSettled) and err.outcome is not None:
                        reply["outcome"] = str(err.outcome)
                except Exception as err:
                    reply = {"ok": False, "error": "WeightTransportError", "message": f"{type(err).__name__}: {err}"}
                try:
                    self._channel.send(reply)
                except WeightTransportClosed:
                    return
                if op == "close":
                    return
        finally:
            self._offers.clear()
            self._subscriber.close()
            self._channel.close()

    def _handle(self, op: str | None, message: tp.Mapping[str, tp.Any]) -> dict[str, tp.Any]:
        if op == "request":
            self._subscriber.request()
            return {"ok": True}
        if op == "receive":
            timeout = message.get("timeout")
            if timeout is not None and float(timeout) <= 0:
                if self._closed or self._channel.is_peer_closed():
                    raise _PeerDisconnected()
                offer = self._publisher._receive(self._subscriber.name, 0.0)
            else:
                deadline = None if timeout is None else time.monotonic() + float(timeout)
                offer = None
                while True:
                    if self._closed or self._channel.is_peer_closed():
                        raise _PeerDisconnected()
                    if deadline is None:
                        step_timeout = _BRIDGE_RECEIVE_POLL_INTERVAL
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        step_timeout = min(remaining, _BRIDGE_RECEIVE_POLL_INTERVAL)
                    offer = self._publisher._receive(self._subscriber.name, step_timeout)
                    if offer is not None:
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        break
            if offer is None:
                return {"ok": True, "offer": None}
            # A subscriber can hold at most one active offer; drop any superseded/stalled descriptor immediately.
            self._offers.clear()
            self._offers[offer.offer_id] = offer
            return {"ok": True, "offer": offer.to_wire()}
        if op in ("begin", "ack", "decline"):
            offer_id = int(message["offer_id"])
            offer = self._offers.get(offer_id)
            if offer is None:
                if op == "decline":
                    with self._publisher._locked():
                        self._publisher._record(self._subscriber.name)
                    return {"ok": True, "declined": False}
                with self._publisher._cond:
                    outcome = self._publisher._outcome_of(offer_id)
                if outcome is OfferOutcome.STALLED:
                    raise WeightTransferStalled(f"offer {offer_id} was settled as stalled")
                raise WeightOfferSettled(f"unknown offer {offer_id}", outcome)
            if op == "begin":
                try:
                    self._subscriber.pull(offer)
                except (WeightOfferSettled, WeightTransferStalled):
                    self._offers.pop(offer.offer_id, None)
                    raise
                return {"ok": True}
            if op == "ack":
                try:
                    self._subscriber.ack(offer)
                except (WeightOfferSettled, WeightTransferStalled):
                    self._offers.pop(offer.offer_id, None)
                    raise
                self._offers.pop(offer.offer_id, None)
                return {"ok": True}
            declined = self._subscriber.decline(offer)
            self._offers.pop(offer.offer_id, None)
            return {"ok": True, "declined": declined}
        if op == "close":
            return {"ok": True}
        raise WeightTransportError(f"unknown op {op!r}")


class _PayloadFetcher(tp.Protocol):
    def fetch(self, offer: WeightOffer, *, timeout: float | None) -> tp.Any: ...


class RemoteWeightSubscriber:
    """Sampler-side subscriber in another process, talking to a :class:`RemoteSubscriberBridge`.

    It keeps its own installed version and verifies every offer's chain
    against it, so divergence between the two processes is detected.

    Args:
        channel: Control channel connected to the bridge.
        fetcher: Payload fetcher (e.g. :class:`JaxTransferClient`); only needed for :meth:`pull`.
        installed_version: Version the sampler already holds.
        pull_timeout: Watchdog for :meth:`pull` on this side (seconds).
        auth_token: Shared secret checked against the bridge's before any RPC. ``None`` is accepted only on
            Unix-domain or loopback channels.

    Raises:
        WeightTransportError: If authentication fails (the channel is closed).
    """

    def __init__(
        self,
        channel: SocketMessageChannel,
        fetcher: _PayloadFetcher | None = None,
        *,
        installed_version: int | None = None,
        pull_timeout: float | None = None,
        auth_token: str | bytes | None = None,
    ):
        channel.authenticate(auth_token, role=_SAMPLER_ROLE)
        self._channel = channel
        self._fetcher = fetcher
        self._installed_version = installed_version
        self._pull_timeout = pull_timeout
        self._lock = threading.Lock()
        self._closed = False

    @property
    def installed_version(self) -> int | None:
        return self._installed_version

    def _call(self, message: dict[str, tp.Any]) -> dict[str, tp.Any]:
        with self._lock:
            if self._closed:
                raise WeightTransportClosed("remote subscriber is closed")
            self._channel.send(message)
            reply = self._channel.recv()
        if not reply.get("ok"):
            cls = _REMOTE_ERRORS.get(str(reply.get("error")), WeightTransportError)
            if cls is WeightOfferSettled:
                outcome = reply.get("outcome")
                raise WeightOfferSettled(str(reply.get("message")), None if outcome is None else OfferOutcome(outcome))
            raise cls(str(reply.get("message")))
        return reply

    def request(self) -> None:
        self._call({"op": "request"})

    def receive(self, timeout: float | None = None) -> WeightOffer | None:
        reply = self._call({"op": "receive", "timeout": timeout})
        if reply.get("offer") is None:
            return None
        offer = WeightOffer.from_wire(reply["offer"])
        verify_offer_chain(offer, self._installed_version)
        return offer

    def pull(self, offer: WeightOffer) -> tp.Any:
        """Begin the pull on the learner side, then fetch the payload with ``fetcher``."""
        if self._fetcher is None:
            raise WeightTransportError("RemoteWeightSubscriber.pull requires a fetcher")
        self._call({"op": "begin", "offer_id": offer.offer_id})
        return self._fetcher.fetch(offer, timeout=self._pull_timeout)

    def ack(self, offer: WeightOffer) -> None:
        self._call({"op": "ack", "offer_id": offer.offer_id})
        self._installed_version = offer.version

    def decline(self, offer: WeightOffer) -> bool:
        return bool(self._call({"op": "decline", "offer_id": offer.offer_id}).get("declined"))

    def close(self) -> None:
        if self._lock.acquire(blocking=False):
            try:
                if self._closed:
                    return
                self._closed = True
                deadline = time.monotonic() + _CLOSE_RPC_TIMEOUT
                with contextlib.suppress(WeightTransportError):
                    self._channel.send({"op": "close"}, deadline=deadline)
                    self._channel.recv(deadline=deadline)
            finally:
                self._channel.close()
                self._lock.release()
        else:
            # Another thread holds _lock (e.g. blocked in receive(timeout=None)):
            # mark closed and shut down the socket immediately so the blocked recv() wakes up.
            self._closed = True
            self._channel.close()


# ---------------------------------------------------------------------- jax transfer-server backend


@dataclasses.dataclass(frozen=True, slots=True)
class LeafSpec:
    """Shape, dtype and sharding description of one transferred leaf."""

    shape: tuple[int, ...]
    dtype: str
    sharding: tp.Mapping[str, tp.Any] | None

    def to_wire(self) -> dict[str, tp.Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "sharding": self.sharding}

    @classmethod
    def from_wire(cls, data: tp.Mapping[str, tp.Any]) -> LeafSpec:
        return cls(shape=tuple(int(d) for d in data["shape"]), dtype=str(data["dtype"]), sharding=data.get("sharding"))


_DEFAULT_TRANSFER_PULL_TIMEOUT = 300.0


def _describe_sharding(sharding: tp.Any) -> dict[str, tp.Any] | None:
    if sharding is None:
        return None
    from jax.sharding import NamedSharding, PartitionSpec, SingleDeviceSharding

    if isinstance(sharding, NamedSharding):
        spec: list[tp.Any] = []
        for entry in sharding.spec:
            if entry is PartitionSpec.UNCONSTRAINED:
                raise WeightTransportError("PartitionSpec.UNCONSTRAINED is not supported for weight transfer")
            if isinstance(entry, tuple):
                if any(sub is PartitionSpec.UNCONSTRAINED or not isinstance(sub, str) for sub in entry):
                    raise WeightTransportError(f"unsupported PartitionSpec axis tuple {entry!r} for weight transfer")
                spec.append(list(entry))
            elif entry is None or isinstance(entry, str):
                spec.append(entry)
            else:
                raise WeightTransportError(f"unsupported PartitionSpec entry {entry!r} for weight transfer")
        return {
            "kind": "named",
            "axis_names": list(sharding.mesh.axis_names),
            "mesh_shape": [int(size) for size in sharding.mesh.devices.shape],
            "spec": spec,
        }
    if isinstance(sharding, SingleDeviceSharding):
        return None
    device_set = getattr(sharding, "device_set", None)
    if device_set is not None and len(device_set) == 1:
        return None
    raise WeightTransportError(
        f"unsupported multi-device sharding {type(sharding).__name__}; "
        "only NamedSharding and single-device shardings are supported for weight transfer"
    )


def _uuid(offer_id: int, leaf_index: int) -> int:
    return (int(offer_id) << 24) | int(leaf_index)


@dataclasses.dataclass(eq=False)
class _JaxStaged:
    leaves: list[tp.Any]
    descriptor: dict[str, tp.Any]


class JaxTransferServerBackend:
    """Publisher backend serving snapshot leaves through ``jax.experimental.transfer``.

    Every leaf must be a ``jax.Array``. :meth:`describe` produces the small
    metadata message (treedef, shapes, dtypes, shardings, server address)
    shipped to the subscriber over the control channel; :meth:`serve`
    registers one transfer per leaf so the subscriber can pull them one at a
    time and only one leaf's destination buffer is in flight at once.

    Args:
        server: An existing transfer server; started on ``device``'s client if omitted.
        address: Bind address of the started server.
        transport_addresses: Data-plane addresses of the started server.
        device: Device whose client hosts the server (default: first local device).
        default_pull_timeout: Finite fallback timeout (seconds) passed to
            ``server.await_pull`` whenever the publisher's ``pull_timeout`` is
            ``None``, ensuring unpulled leaves never stay pinned in HBM forever.
    """

    def __init__(
        self,
        server: tp.Any = None,
        *,
        address: str = "127.0.0.1:0",
        transport_addresses: tp.Sequence[str] = ("127.0.0.1:0",),
        device: tp.Any = None,
        default_pull_timeout: float = _DEFAULT_TRANSFER_PULL_TIMEOUT,
    ):
        if not math.isfinite(default_pull_timeout) or default_pull_timeout <= 0:
            raise ValueError(f"default_pull_timeout must be finite and positive, got {default_pull_timeout}")
        if server is None:
            server = _start_transfer_server(address, transport_addresses, device)
        self.server = server
        self._address: str = server.address()
        self.default_pull_timeout = float(default_pull_timeout)
        self._closed = False

    @property
    def address(self) -> str:
        if self._closed or self.server is None:
            raise WeightTransportClosed("JaxTransferServerBackend is closed")
        return self._address

    def stage(self, version: int, snapshot: tp.Any) -> _JaxStaged:
        import jax

        del version
        server = self.server
        if self._closed or server is None:
            raise WeightTransportClosed("JaxTransferServerBackend is closed")
        leaves, treedef = jax.tree.flatten(snapshot)
        if len(leaves) >= 1 << 24:
            raise ValueError(f"snapshot has too many leaves ({len(leaves)})")
        for index, leaf in enumerate(leaves):
            if not isinstance(leaf, jax.Array):
                raise TypeError(f"snapshot leaf {index} is {type(leaf).__name__}; every leaf must be a jax.Array")
        try:
            encoded_treedef: str | None = base64.b64encode(treedef.serialize_using_proto()).decode("ascii")
        except Exception:
            encoded_treedef = None
        specs = [
            LeafSpec(tuple(int(d) for d in leaf.shape), str(leaf.dtype), _describe_sharding(leaf.sharding))
            for leaf in leaves
        ]
        descriptor = {
            "kind": "jax_transfer",
            "address": self._address,
            "treedef": encoded_treedef,
            "leaves": [spec.to_wire() for spec in specs],
        }
        return _JaxStaged(leaves=list(leaves), descriptor=descriptor)

    def describe(self, handle: _JaxStaged) -> tp.Mapping[str, tp.Any]:
        return handle.descriptor

    def serve(self, handle: _JaxStaged, offer_id: int, timeout: float | None) -> None:
        import datetime

        server = self.server
        if self._closed or server is None:
            raise WeightTransportClosed("JaxTransferServerBackend is closed")
        effective_timeout = self.default_pull_timeout if timeout is None else float(timeout)
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise ValueError(f"pull timeout must be finite and positive, got {effective_timeout}")
        deadline = datetime.datetime.now() + datetime.timedelta(seconds=effective_timeout)
        for index, leaf in enumerate(handle.leaves):
            if self._closed:
                raise WeightTransportClosed("JaxTransferServerBackend is closed")
            server.await_pull(_uuid(offer_id, index), [leaf], timeout=deadline)

    def release(self, handle: _JaxStaged) -> None:
        handle.leaves.clear()

    def close(self) -> None:
        """Close the backend and drop its transfer-server reference. Idempotent."""
        if self._closed:
            return
        self._closed = True
        server, self.server = self.server, None
        if server is not None:
            for method_name in ("close", "shutdown"):
                method = getattr(server, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method()
                    break

    def __enter__(self) -> JaxTransferServerBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _start_transfer_server(address: str, transport_addresses: tp.Sequence[str], device: tp.Any) -> tp.Any:
    import jax
    from jax.experimental import transfer

    device = device if device is not None else jax.local_devices()[0]
    return transfer.start_transfer_server(device.client, address, list(transport_addresses))


class JaxTransferClient:
    """Subscriber-side fetcher pulling leaves served by a :class:`JaxTransferServerBackend`.

    Leaves are pulled one at a time (each pull completes before the next one
    starts) under a watchdog: a leaf that does not arrive within the remaining
    ``timeout`` raises :class:`WeightTransferStalled`.

    Args:
        server: This process's transfer server; started if omitted.
        mesh: Mesh used to rebuild ``NamedSharding`` leaves (axis names must match).
        sharding_resolver: ``(index, LeafSpec) -> Sharding`` override; takes precedence over ``mesh``.
        treedef: Treedef override for snapshots whose treedef cannot be serialized.
        address, transport_addresses, device: Used when starting the server.
    """

    def __init__(
        self,
        server: tp.Any = None,
        *,
        mesh: tp.Any = None,
        sharding_resolver: tp.Callable[[int, LeafSpec], tp.Any] | None = None,
        treedef: tp.Any = None,
        address: str = "127.0.0.1:0",
        transport_addresses: tp.Sequence[str] = ("127.0.0.1:0",),
        device: tp.Any = None,
    ):
        if server is None:
            server = _start_transfer_server(address, transport_addresses, device)
        self.server = server
        self._mesh = mesh
        self._resolver = sharding_resolver
        self._treedef = treedef
        self._device = device
        self._lock = threading.Lock()
        self._closed = False
        self._connections: dict[str, tp.Any] = {}

    def _connection(self, address: str) -> tp.Any:
        with self._lock:
            if self._closed or self.server is None:
                raise WeightTransportClosed("JaxTransferClient is closed")
            connection = self._connections.get(address)
            if connection is None:
                connection = self.server.connect(address)
                self._connections[address] = connection
            return connection

    def _evict_connection(self, address: str, connection: tp.Any) -> None:
        with self._lock:
            if self._connections.get(address) is connection:
                self._connections.pop(address, None)
        close_fn = getattr(connection, "close", None)
        if callable(close_fn):
            with contextlib.suppress(Exception):
                close_fn()

    def close(self) -> None:
        """Close the client, dropping cached connections and transfer-server references. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = list(self._connections.values())
            self._connections.clear()
            server, self.server = self.server, None
        for connection in connections:
            close_fn = getattr(connection, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()
        if server is not None:
            for method_name in ("close", "shutdown"):
                method = getattr(server, method_name, None)
                if callable(method):
                    with contextlib.suppress(Exception):
                        method()
                    break

    def __enter__(self) -> JaxTransferClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _sharding(self, index: int, spec: LeafSpec) -> tp.Any:
        import jax
        from jax.sharding import NamedSharding, PartitionSpec, SingleDeviceSharding

        if self._resolver is not None:
            return self._resolver(index, spec)
        described = spec.sharding
        if described is not None and described.get("kind") == "named" and self._mesh is not None:
            if list(self._mesh.axis_names) != list(described["axis_names"]):
                raise WeightTransportError(
                    f"leaf {index}: receiver mesh axes {list(self._mesh.axis_names)} differ from "
                    f"sender axes {described['axis_names']}"
                )
            entries = [tuple(entry) if isinstance(entry, list) else entry for entry in described["spec"]]
            return NamedSharding(self._mesh, PartitionSpec(*entries))
        if described is not None and described.get("kind") == "named":
            raise WeightTransportError(f"leaf {index} is mesh-sharded; pass mesh= or sharding_resolver=")
        return SingleDeviceSharding(self._device if self._device is not None else jax.local_devices()[0])

    def fetch(self, offer: WeightOffer, *, timeout: float | None) -> tp.Any:
        """Pull every leaf of ``offer`` and return the reassembled pytree."""
        import jax
        import jax.numpy as jnp

        descriptor = offer.descriptor
        if descriptor.get("kind") != "jax_transfer":
            raise WeightTransportError(f"offer {offer.offer_id} is not a jax-transfer offer: {descriptor.get('kind')}")
        treedef = self._treedef
        if treedef is None:
            encoded = descriptor.get("treedef")
            if encoded is None:
                raise WeightTransportError("sender treedef is not serializable; pass treedef= to JaxTransferClient")
            treedef = jax.tree_util.PyTreeDef.deserialize_using_proto(
                jax.tree_util.default_registry, base64.b64decode(encoded)
            )
        address = str(descriptor["address"])
        connection = self._connection(address)
        deadline = None if timeout is None else time.monotonic() + timeout
        leaves: list[tp.Any] = []
        try:
            for index, wire in enumerate(descriptor["leaves"]):
                spec = LeafSpec.from_wire(wire)
                target = jax.ShapeDtypeStruct(spec.shape, jnp.dtype(spec.dtype), sharding=self._sharding(index, spec))
                leaves.append(self._pull_leaf(connection, offer, index, target, deadline))
        except BaseException:
            leaves.clear()
            self._evict_connection(address, connection)
            raise
        return jax.tree.unflatten(treedef, leaves)

    def _pull_leaf(self, connection: tp.Any, offer: WeightOffer, index: int, target: tp.Any, deadline) -> tp.Any:
        import datetime

        import jax

        pull_uuid = _uuid(offer.offer_id, index)
        if deadline is None:
            try:
                (array,) = connection.pull(pull_uuid, [target])
                return jax.block_until_ready(array)
            except Exception as err:
                raise WeightTransportError(f"pull of leaf {index} of offer {offer.offer_id} failed") from err

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WeightTransferStalled(
                f"pull of leaf {index} of offer {offer.offer_id} (version {offer.version}) did not finish "
                "within the pull timeout"
            )
        pull_deadline = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
        result: list[tp.Any] = []
        failure: list[BaseException] = []

        def run() -> None:
            try:
                (array,) = connection.pull(pull_uuid, [target], timeout=pull_deadline)
                result.append(jax.block_until_ready(array))
            except BaseException as err:
                failure.append(err)

        worker = threading.Thread(target=run, name=f"weight-pull-{offer.offer_id}-{index}", daemon=True)
        worker.start()
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            raise WeightTransferStalled(
                f"pull of leaf {index} of offer {offer.offer_id} (version {offer.version}) did not finish "
                "within the pull timeout"
            )
        if failure:
            if isinstance(failure[0], TimeoutError) or time.monotonic() >= deadline:
                raise WeightTransferStalled(
                    f"pull of leaf {index} of offer {offer.offer_id} (version {offer.version}) did not finish "
                    "within the pull timeout"
                ) from failure[0]
            raise WeightTransportError(f"pull of leaf {index} of offer {offer.offer_id} failed") from failure[0]
        return result[0]
