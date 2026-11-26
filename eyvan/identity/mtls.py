# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mutual TLS between the control plane, the cell agent, and the native worker.

Default off, everywhere, in every constructor in this package. Turning it on is
an operator decision with a wire consequence, and a security feature that
silently changes what a rolling upgrade can talk to is a security feature that
gets rolled back.

Enablement order
----------------

Three surfaces, two directions of trust::

    control plane  --client-->  cell agent  --client-->  native worker

The control plane is a client and never a server here; the worker is a server
and never a client; the agent is both. Run the rollout **outward-in**: every
client must already be presenting a certificate before any server starts
requiring one.

1. **Distribute the trust anchor.** Put the CA bundle on all three surfaces with
   ``mode`` still ``off``. Nothing changes on the wire; a surface that cannot
   verify a peer cannot safely be asked to require one.
2. **Install key pairs.** Issue a certificate and key per surface, still with
   ``mode`` off. Restarts in this step are ordinary restarts.
3. **Servers to** ``permissive``, **innermost first**: worker, then agent. A
   permissive server accepts a client certificate and still accepts a client
   without one, so every already-running control plane and agent keeps working
   through the restart.
4. **Clients on.** The agent starts presenting its certificate to the worker,
   then the control plane starts presenting its certificate to the agent.
   Verify -- do not assume -- that every connection is now authenticated before
   continuing.
5. **Servers to** ``required``, **innermost first**: worker, then agent.

Step 5 is the one that locks a fleet out, and it is why
:func:`validate_rollout_step` refuses two specific transitions rather than only
checking the final state:

* a server that requires certificates while its client presents none -- the
  control plane can no longer reach its own agents, and the agents it cannot
  reach are the ones holding the running workloads; and
* a server that starts requiring in the *same step* that its client starts
  presenting. That step has no safe order: whichever process restarts first,
  there is a window with a requiring server and a certificate-less client, and
  a rollout is not atomic.

Reversal is always safe in the opposite order -- servers back to ``permissive``
first, then clients off -- and needs no validation, because loosening a server
never strands anyone.
"""

from __future__ import annotations

import ssl
import typing as tp
from dataclasses import dataclass, replace

from eyvan._internal.compat import StrEnum


class MutualTlsMode(StrEnum):
    """What a server does with a peer certificate."""

    #: Today's behaviour: no client certificate is requested or verified.
    OFF = "off"
    #: Request and verify a client certificate, but serve clients without one.
    PERMISSIVE = "permissive"
    #: Refuse any peer that does not present a certificate this surface trusts.
    REQUIRED = "required"


class TlsSurface(StrEnum):
    """The three processes this contract holds between."""

    CONTROL_PLANE = "control_plane"
    AGENT = "agent"
    WORKER = "worker"


#: Client-to-server edges, ordered outermost first. The control plane never
#: serves an agent and the worker never calls one, so these two are the whole
#: graph.
MUTUAL_TLS_EDGES: tp.Final[tuple[tuple[TlsSurface, TlsSurface], ...]] = (
    (TlsSurface.CONTROL_PLANE, TlsSurface.AGENT),
    (TlsSurface.AGENT, TlsSurface.WORKER),
)


class MutualTlsError(RuntimeError):
    """Base error for mutual-TLS configuration and rollout refusals."""


class MutualTlsRolloutError(MutualTlsError):
    """A rollout step would strand a client that is still running."""


class MutualTlsConfigurationError(MutualTlsError, ValueError):
    """Mutual TLS is enabled but its files do not form a usable context.

    Also a :class:`ValueError`, so a settings object that validates this at
    construction refuses in the same category as every other malformed setting
    rather than in one a caller has to know about separately.
    """


@dataclass(frozen=True, slots=True)
class MutualTlsFiles:
    """Paths, never material.

    Eyvan does not read a private key: the path is handed to :mod:`ssl`, which
    keeps the key inside OpenSSL. Nothing in this package ever holds key bytes,
    so nothing in this package can log them.
    """

    trust_bundle: str = ""
    certificate: str = ""
    private_key: str = ""

    @property
    def has_trust_anchor(self) -> bool:
        """Return whether this surface can verify a peer at all."""

        return bool(self.trust_bundle)

    @property
    def has_identity(self) -> bool:
        """Return whether this surface can present a certificate."""

        return bool(self.certificate and self.private_key)


@dataclass(frozen=True, slots=True)
class SurfaceMutualTls:
    """One surface's position in the rollout."""

    #: What this surface requires of peers that connect *to* it.
    server_mode: MutualTlsMode = MutualTlsMode.OFF
    #: Whether this surface presents a certificate when it connects *out*.
    client_certificate: bool = False
    #: Whether this surface holds the CA bundle. Step 1 of the order.
    trust_anchor: bool = False


@dataclass(frozen=True, slots=True)
class FleetMutualTls:
    """The rollout position of all three surfaces at one moment.

    The default is the whole fleet off, which is exactly today's deployment.
    """

    control_plane: SurfaceMutualTls = SurfaceMutualTls()
    agent: SurfaceMutualTls = SurfaceMutualTls()
    worker: SurfaceMutualTls = SurfaceMutualTls()

    def surface(self, surface: TlsSurface) -> SurfaceMutualTls:
        """Return one surface's position."""

        return {
            TlsSurface.CONTROL_PLANE: self.control_plane,
            TlsSurface.AGENT: self.agent,
            TlsSurface.WORKER: self.worker,
        }[surface]

    def with_surface(self, surface: TlsSurface, position: SurfaceMutualTls) -> FleetMutualTls:
        """Return a copy with one surface moved."""

        field = {
            TlsSurface.CONTROL_PLANE: "control_plane",
            TlsSurface.AGENT: "agent",
            TlsSurface.WORKER: "worker",
        }[surface]
        return replace(self, **{field: position})

    @property
    def enabled_anywhere(self) -> bool:
        """Return whether any surface has left the default off position."""

        return any(
            self.surface(name).server_mode is not MutualTlsMode.OFF or self.surface(name).client_certificate
            for name in TlsSurface
        )


def validate_fleet_state(state: FleetMutualTls) -> None:
    """Refuse a fleet position in which some client cannot reach its server.

    Args:
        state: The position to check.

    Raises:
        MutualTlsRolloutError: If a server requires certificates its client does
            not present, or a surface is verifying without a trust anchor.
    """

    for name in TlsSurface:
        position = state.surface(name)
        if position.server_mode is not MutualTlsMode.OFF and not position.trust_anchor:
            raise MutualTlsRolloutError(f"{name} cannot verify peer certificates before it holds the trust anchor")
        if position.client_certificate and not position.trust_anchor:
            raise MutualTlsRolloutError(f"{name} cannot present a certificate before it holds the trust anchor")
    for client, server in MUTUAL_TLS_EDGES:
        if state.surface(server).server_mode is MutualTlsMode.REQUIRED and not state.surface(client).client_certificate:
            raise MutualTlsRolloutError(
                f"{server} would require a certificate that {client} does not present, stranding it"
            )


def validate_rollout_step(before: FleetMutualTls, after: FleetMutualTls) -> None:
    """Refuse a transition that has no safe restart order.

    The end state being valid is not enough. A step that both starts requiring
    on a server and starts presenting on its client is unsafe in one of its two
    possible orders, and an operator does not get to choose which process
    restarts first.

    Args:
        before: The fleet position now.
        after: The proposed fleet position.

    Raises:
        MutualTlsRolloutError: If ``after`` strands a client, or if the step
            cannot be applied in any order without a window that does.
    """

    validate_fleet_state(after)
    for client, server in MUTUAL_TLS_EDGES:
        became_required = (
            before.surface(server).server_mode is not MutualTlsMode.REQUIRED
            and after.surface(server).server_mode is MutualTlsMode.REQUIRED
        )
        client_started = not before.surface(client).client_certificate and after.surface(client).client_certificate
        if became_required and client_started:
            raise MutualTlsRolloutError(
                f"{server} must not begin requiring certificates in the same step that {client} begins presenting one; "
                "roll the client out first and confirm it before requiring"
            )


def enablement_plan() -> tuple[FleetMutualTls, ...]:
    """Return the documented rollout, from today's fully-off fleet to required.

    Every consecutive pair satisfies :func:`validate_rollout_step`; that is the
    property the order exists to have, and it is asserted rather than described.
    """

    anchored = SurfaceMutualTls(trust_anchor=True)
    off = FleetMutualTls()
    step_anchor = FleetMutualTls(control_plane=anchored, agent=anchored, worker=anchored)
    step_worker_permissive = step_anchor.with_surface(
        TlsSurface.WORKER, SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE, trust_anchor=True)
    )
    step_agent_permissive = step_worker_permissive.with_surface(
        TlsSurface.AGENT, SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE, trust_anchor=True)
    )
    step_agent_client = step_agent_permissive.with_surface(
        TlsSurface.AGENT,
        SurfaceMutualTls(server_mode=MutualTlsMode.PERMISSIVE, client_certificate=True, trust_anchor=True),
    )
    step_control_plane_client = step_agent_client.with_surface(
        TlsSurface.CONTROL_PLANE, SurfaceMutualTls(client_certificate=True, trust_anchor=True)
    )
    step_worker_required = step_control_plane_client.with_surface(
        TlsSurface.WORKER, SurfaceMutualTls(server_mode=MutualTlsMode.REQUIRED, trust_anchor=True)
    )
    step_agent_required = step_worker_required.with_surface(
        TlsSurface.AGENT,
        SurfaceMutualTls(server_mode=MutualTlsMode.REQUIRED, client_certificate=True, trust_anchor=True),
    )
    return (
        off,
        step_anchor,
        step_worker_permissive,
        step_agent_permissive,
        step_agent_client,
        step_control_plane_client,
        step_worker_required,
        step_agent_required,
    )


def parse_mutual_tls_mode(value: str | None) -> MutualTlsMode:
    """Resolve an operator-supplied mode, defaulting to off.

    An unrecognized spelling is refused rather than treated as off: a typo in
    ``required`` that silently disables mutual TLS is the failure this whole
    module is meant to prevent.
    """

    if value is None or not value.strip():
        return MutualTlsMode.OFF
    try:
        return MutualTlsMode(value.strip().lower())
    except ValueError:
        choices = ", ".join(mode.value for mode in MutualTlsMode)
        raise MutualTlsConfigurationError(f"mutual TLS mode must be one of: {choices}") from None


def client_ssl_context(files: MutualTlsFiles) -> ssl.SSLContext | None:
    """Build the context a client uses, or ``None`` when mutual TLS is off.

    ``None`` rather than a default context, so a caller that is not configured
    for mutual TLS keeps its transport untouched instead of acquiring a
    different -- possibly stricter, possibly looser -- TLS posture as a side
    effect of this module existing.
    """

    if not files.has_trust_anchor and not files.has_identity:
        return None
    if not files.has_trust_anchor:
        raise MutualTlsConfigurationError("a mutual TLS client requires a trust bundle to verify its server")
    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=files.trust_bundle)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    if files.has_identity:
        _load_identity(context, files)
    return context


def server_ssl_context(mode: MutualTlsMode, files: MutualTlsFiles) -> ssl.SSLContext | None:
    """Build the context a server listens with, or ``None`` when mutual TLS is off.

    ``PERMISSIVE`` maps to ``CERT_OPTIONAL`` and ``REQUIRED`` to
    ``CERT_REQUIRED``. That single difference is the whole of step 3 versus step
    5 of the enablement order: a permissive listener completes a handshake with a
    client that has no certificate, which is what keeps every already-running
    control plane and agent connected while the rollout is half done.
    """

    if mode is MutualTlsMode.OFF:
        return None
    if not files.has_identity:
        raise MutualTlsConfigurationError("a mutual TLS server requires a certificate and private key")
    if not files.has_trust_anchor:
        raise MutualTlsConfigurationError("a mutual TLS server requires a trust bundle to verify its clients")
    context = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH, cafile=files.trust_bundle)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED if mode is MutualTlsMode.REQUIRED else ssl.CERT_OPTIONAL
    _load_identity(context, files)
    return context


def server_ssl_options(mode: MutualTlsMode, files: MutualTlsFiles) -> dict[str, object]:
    """Return the ``uvicorn.run`` keyword arguments for one server surface.

    Empty when ``mode`` is off, which is what keeps the default listener byte
    for byte what it is today.
    """

    if mode is MutualTlsMode.OFF:
        return {}
    if not files.has_identity:
        raise MutualTlsConfigurationError("a mutual TLS server requires a certificate and private key")
    if not files.has_trust_anchor:
        raise MutualTlsConfigurationError("a mutual TLS server requires a trust bundle to verify its clients")

    def context_factory(_config: object, _default: object) -> ssl.SSLContext:
        # Uvicorn's own context is SSLContext(ssl_version), whose minimum is
        # whatever the OpenSSL build defaults to: TLS 1.2 on some builds,
        # MINIMUM_SUPPORTED (TLS 1.0) on the Ubuntu CI runner. Building the
        # context here applies the explicit TLS 1.2 floor everywhere.
        context = server_ssl_context(mode, files)
        assert context is not None  # mode is not OFF
        return context

    return {
        "ssl_certfile": files.certificate,
        "ssl_keyfile": files.private_key,
        "ssl_ca_certs": files.trust_bundle,
        "ssl_cert_reqs": int(ssl.CERT_REQUIRED if mode is MutualTlsMode.REQUIRED else ssl.CERT_OPTIONAL),
        "ssl_version": int(ssl.PROTOCOL_TLS_SERVER),
        # Requires Uvicorn >=0.48, where ssl_context_factory was added.
        "ssl_context_factory": context_factory,
    }


def _load_identity(context: ssl.SSLContext, files: MutualTlsFiles) -> None:
    """Attach a certificate and key, normalizing the failure.

    The path is kept because it is the fix; the OpenSSL detail is dropped
    because it varies by build and is not something an operator acts on.
    """

    try:
        context.load_cert_chain(certfile=files.certificate, keyfile=files.private_key)
    except (OSError, ssl.SSLError):
        raise MutualTlsConfigurationError(
            f"mutual TLS certificate {files.certificate!r} and key could not be loaded"
        ) from None


__all__ = (
    "MUTUAL_TLS_EDGES",
    "FleetMutualTls",
    "MutualTlsConfigurationError",
    "MutualTlsError",
    "MutualTlsFiles",
    "MutualTlsMode",
    "MutualTlsRolloutError",
    "SurfaceMutualTls",
    "TlsSurface",
    "client_ssl_context",
    "enablement_plan",
    "parse_mutual_tls_mode",
    "server_ssl_context",
    "server_ssl_options",
    "validate_fleet_state",
    "validate_rollout_step",
)
