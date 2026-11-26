# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Node-local discovery of the control plane a machine already belongs to.

The question this answers is "which control plane am I part of", asked by a
process that is *running on a control-plane host*. It is the same question Ray
answers when ``ray.init()`` finds a cluster without being told where it is, and
it is answered the same way: the thing that stood the node up wrote the answer
down somewhere well known on that node, and the client reads it.

Two boundaries make this safe, and both are deliberate.

**Node-local, never operator-local.** A head binds a private VPC address with
VPC-only ingress -- ``sdk/commands/head.py`` refuses to report anything else --
so the URL written here is correct on the node and unroutable from a laptop.
Writing it into an operator's ``~/.config/eyvan/config.json`` would silently
repoint every later ``eyvan`` command at an address that cannot be reached,
which is why the head writes *this* file and not that one.

**The discovery file structurally cannot hold a credential.** It is an ordinary
:class:`~eyvan.sdk.contexts.EyvanContextConfig` document, and that model forbids
extra fields, so there is no field a token could be written into even by
mistake. The token is discovered separately by :func:`discover_node_token`,
which *reads the credential where the installer already put it* and copies it
nowhere -- see that function for why copying is refused.
"""

from __future__ import annotations

import os
import tempfile
import typing as tp
from pathlib import Path

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.contexts import (
    DEFAULT_ORG_ID,
    DEFAULT_PROJECT_ID,
    ContextConfigError,
    EyvanContext,
    EyvanContextConfig,
    FileContextStore,
    _encoded_config,
)

#: Where a head records the control plane it serves, for its own processes.
#:
#: Under ``/etc`` rather than a home directory because it describes the machine
#: rather than a user: every account on the head gets the same answer, including
#: the ones a workload runs as.
NODE_CONTEXT_PATH = Path("/etc/eyvan/node.json")

#: Where the head keeps the environment its server and migration both read.
HEAD_ENVIRONMENT_PATH = "/etc/eyvan/head.env"

#: The variable that repoints discovery, for tests and for a second head on one
#: machine. Deliberately not an :class:`~eyvan.config.EyvanSettings` field: it
#: configures a *client*, and nothing the server reads.
NODE_FILE_ENVIRONMENT_VARIABLE = "EYVAN_NODE_FILE"


class NodeContextError(RuntimeError):
    """Raised when node-local discovery finds something it must not guess about."""


def node_context_path(environ: tp.Mapping[str, str] | None = None) -> Path:
    """Resolve the node discovery file from the environment or the machine default."""

    environment = os.environ if environ is None else environ
    configured = environment.get(NODE_FILE_ENVIRONMENT_VARIABLE)
    return Path(configured or NODE_CONTEXT_PATH).expanduser()


def head_environment_path(environ: tp.Mapping[str, str] | None = None) -> Path:
    """Resolve the head environment file beside whichever node file is in use.

    Kept together so a test that redirects one redirects both, and so a second
    head on one machine cannot read the first head's credential.
    """

    environment = os.environ if environ is None else environ
    configured = environment.get(NODE_FILE_ENVIRONMENT_VARIABLE)
    if configured is None:
        return Path(HEAD_ENVIRONMENT_PATH)
    return Path(configured).expanduser().parent / "head.env"


def _environment_value(path: Path, name: str) -> str | None:
    """Read one ``NAME=value`` line without sourcing the file.

    Parsed rather than sourced for the reason the installer refuses shell
    metacharacters in these values in the first place: this file is read by
    processes that must never execute its contents.
    """

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                return value.strip()
    return None


def write_node_context(
    environment_path: Path | str,
    node_path: Path | str,
    *,
    fleet: str,
) -> Path:
    """Record the control plane this node serves, derived from the head's own environment.

    ``EYVAN_PUBLIC_URL`` is the single source of truth: it is the address the
    server advertises for itself, so it stays correct on a re-run that adopts an
    existing head and regenerates no secrets.

    Args:
        environment_path: The head's systemd ``EnvironmentFile``.
        node_path: Where to write the discovery document.
        fleet: The fleet this head serves, used as the context name.

    Returns:
        The path written.

    Raises:
        NodeContextError: If the environment names no public URL, or names a
            loopback one.
    """

    source = Path(environment_path)
    try:
        public_url = _environment_value(source, "EYVAN_PUBLIC_URL")
    except OSError as exc:
        raise NodeContextError(f"could not read the head environment {source}: {exc}") from exc
    if not public_url:
        raise NodeContextError(f"{source} names no EYVAN_PUBLIC_URL, so this node cannot describe its control plane")
    endpoint = ControlPlaneEndpoint.parse(public_url)
    if endpoint.is_loopback:
        # The whole value of this file is that it carries a *non-loopback* URL.
        # A loopback endpoint is the signal the SDK uses to decide it is in local
        # development and should start a control plane of its own -- so a node
        # file saying ``127.0.0.1`` would re-arm exactly the hazard it exists to
        # disarm, while looking like the fix.
        raise NodeContextError(
            f"refusing to record {public_url!r} as this node's control plane: a loopback URL is the signal that "
            "starts a local development plane, and this node already runs the real one"
        )
    context = EyvanContext(
        fleet=fleet,
        org_id=DEFAULT_ORG_ID,
        project_id=DEFAULT_PROJECT_ID,
        base_url=endpoint.url,
    )
    config = EyvanContextConfig(active_context=fleet, contexts={fleet: context})
    target = Path(node_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        # 0644, unlike the operator's own context file: this describes the
        # machine, and every account that runs a workload here has to be able to
        # read it. It contains no secret, by construction.
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(_encoded_config(config, target))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o644)
        return target
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def discover_node_context(environ: tp.Mapping[str, str] | None = None) -> EyvanContext | None:
    """Return the control plane this node serves, or ``None`` when it serves none.

    ``None`` for an ordinary machine is the whole point: discovery fills a
    vacuum and never overrides a choice. A malformed document is *not* ``None``
    -- it is a node that was configured and configured wrongly, which the caller
    must not silently treat as an unconfigured laptop.
    """

    target = node_context_path(environ)
    if not target.exists():
        return None
    return FileContextStore(target).active()


def discover_node_token(
    endpoint: ControlPlaneEndpoint,
    environ: tp.Mapping[str, str] | None = None,
) -> str | None:
    """Return this node's bootstrap credential, read where the installer put it.

    Read in place and copied nowhere. A control-plane bearer token is a fixed
    fleet administrator on a plane that provisions accelerators, with no expiry
    and no revocation short of rotating every credential the head holds; writing
    a second copy into a home directory would create a durable secret in a
    location nothing in this package treats as one. The credential already
    exists on this machine, readable by the group the installer created, so
    there is nothing to copy.

    ``None`` unless the requested endpoint is *exactly* the plane this node
    serves. A bearer token sent to a plane that did not issue it is a
    disclosure, so the comparison is equality on the normalized URL rather than
    any host or prefix heuristic.

    Raises:
        NodeContextError: If the credential exists but this process may not read
            it. Returning ``None`` there would surface as a 401 at submission
            time, naming neither the file nor the group that guards it.
    """

    node_context = discover_node_context(environ)
    if node_context is None or node_context.base_url != endpoint.url:
        return None
    source = head_environment_path(environ)
    try:
        return _environment_value(source, "EYVAN_BOOTSTRAP_TOKEN") or None
    except PermissionError as exc:
        raise NodeContextError(
            f"this node's control-plane credential in {source} is not readable by this account; it is readable by "
            "group 'eyvan-head', so log in again to pick up that membership, or run 'newgrp eyvan-head'"
        ) from exc
    except FileNotFoundError:
        # An identity-provider head keeps no bootstrap token, and neither does a
        # machine that is not a head. Both are ordinary.
        return None
    except OSError as exc:
        raise NodeContextError(f"could not read this node's control-plane credential {source}: {exc}") from exc


def discover_node_token_or_warn(
    endpoint: ControlPlaneEndpoint,
    environ: tp.Mapping[str, str] | None = None,
    *,
    warn: tp.Callable[[str], None] | None = None,
) -> str | None:
    """Discover this node's credential, reporting rather than raising when it cannot.

    :func:`discover_node_token` raises on an unreadable credential because
    silence there becomes an unexplained 401 much later. That is right for a
    caller that needs the credential and wrong as a *startup* step: most commands
    need no credential at all, and failing `eyvan context` because a token file
    is unreadable turns one degraded capability into a completely unusable CLI.

    So the explanation is delivered where it is discovered, and the process
    continues without a token. A command that genuinely needed one still fails --
    but the operator has already been told exactly why, and by then it reads as
    the cause rather than a mystery.
    """

    try:
        return discover_node_token(endpoint, environ)
    except NodeContextError as error:
        if warn is not None:
            warn(str(error))
        return None


__all__ = (
    "HEAD_ENVIRONMENT_PATH",
    "NODE_CONTEXT_PATH",
    "NODE_FILE_ENVIRONMENT_VARIABLE",
    "ContextConfigError",
    "NodeContextError",
    "discover_node_context",
    "discover_node_token",
    "discover_node_token_or_warn",
    "head_environment_path",
    "node_context_path",
    "write_node_context",
)
