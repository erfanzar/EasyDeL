# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Turning a dispatch's secret capabilities into values, on the cell.

The control plane hands a cell one short-lived capability per declared mount and
the address to redeem it at. This redeems them, and it runs on the **agent**
rather than inside the job for two reasons.

The value can then be written into the launch environment file the agent already
writes at ``0600``, so it never appears on a command line, in a unit definition,
or in any durable record -- whereas a job that fetched its own secrets would have
to be handed the capability, and a capability that reaches the job is a
capability the job can log.

And it means every execution kind is covered by one implementation. Wrapping the
entrypoint would have needed a shim per kind: callable, command, workspace,
container.

Standard library only, like its neighbours here. This runs on a TPU worker where
the agent's dependencies are the ones that had to be installed, and one JSON
fetch is not worth adding to that list.
"""

from __future__ import annotations

import http.client
import re
import ssl
import time
import typing as tp
import urllib.error
import urllib.request

from eyvan._internal.plane_tls import control_plane_urlopen_options
from eyvan.secrets import MAX_SECRET_BYTES
from eyvan.worker.distributed_environment import is_runtime_owned_environment_name
from eyvan.worker.runtime.http_retry import HTTP_ATTEMPTS, RETRYABLE_HTTP_STATUS, retry_delay

#: Written by the control plane at dispatch, one triple per mount.
COUNT_ENV = "EYVAN_SECRET_COUNT"

_TIMEOUT_S = 30.0
# Kept as the local spelling used by the bounded read below. The public constant
# lives with the server-side store so both transfer boundaries share one limit.
_MAX_SECRET_BYTES = MAX_SECRET_BYTES
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class SecretMaterializationError(RuntimeError):
    """Raised when a declared secret could not be resolved for a launch."""


def secret_environment_keys(environment: tp.Mapping[str, str]) -> tuple[str, ...]:
    """Return the scaffolding variables that must not survive into the job.

    The capabilities in particular. A job receives the *value* it asked for and
    not the means to fetch it again -- otherwise a leaked process environment
    would hand an attacker a working credential for the rest of the attempt,
    rather than one secret.
    """

    keys = [key for key in environment if isinstance(key, str) and key.startswith("EYVAN_SECRET_")]
    return tuple(sorted(keys))


def materialize_secrets(
    environment: tp.Mapping[str, str],
    *,
    fetch: tp.Callable[[str, str], bytes] | None = None,
) -> dict[str, str]:
    """Resolve every declared mount and return the variables the job should see.

    Raises rather than skipping a mount it could not resolve. A job that starts
    without the credential it declared fails later, further from the cause, and
    often as an authentication error against someone else's service -- so the
    launch is refused here, where the reason is still in hand.
    """

    count = _count(environment)
    if not count:
        return {}
    if fetch is not None:
        resolve = fetch
    else:
        # The redemption runs in the native worker, before the job exists, so the
        # plane's CA is read from the job's environment rather than this process's.
        options = control_plane_urlopen_options(environment)

        def resolve(url: str, token: str) -> bytes:
            return _fetch(url, token, options=options)

    resolved: dict[str, str] = {}
    for index in range(count):
        name = environment.get(f"EYVAN_SECRET_{index}_NAME", "")
        url = environment.get(f"EYVAN_SECRET_{index}_URL", "")
        token = environment.get(f"EYVAN_SECRET_{index}_TOKEN", "")
        if not isinstance(name, str) or not isinstance(url, str) or not isinstance(token, str):
            raise SecretMaterializationError(f"secret mount {index} has invalid dispatch fields")
        if not name or not url or not token:
            raise SecretMaterializationError(f"secret mount {index} is missing its name, address, or capability")
        _validate_mount_name(name, environment=environment, resolved=resolved)
        if any(character in url or character in token for character in ("\x00", "\n", "\r")):
            raise SecretMaterializationError(f"secret mount {index} has invalid dispatch fields")
        try:
            value = resolve(url, token)
        except SecretMaterializationError:
            raise
        except Exception:
            # A resolver exception is an untrusted diagnostic boundary. Custom
            # stores and HTTP clients have been observed to include request
            # headers or response fragments in their exception text; either can
            # be the capability or the value. Preserve the mount name only.
            raise SecretMaterializationError(f"secret mount {name!r} could not be fetched") from None
        if not isinstance(value, bytes):
            raise SecretMaterializationError(f"secret mount {name!r} returned an invalid payload")
        if len(value) > _MAX_SECRET_BYTES:
            raise SecretMaterializationError(
                f"secret mount {name!r} exceeds the {_MAX_SECRET_BYTES}-byte environment limit"
            )
        try:
            decoded = value.decode()
        except UnicodeDecodeError:
            # A binary secret in an environment variable would be silently
            # mangled by the env-file encoding. Refusing names the real
            # constraint instead of delivering something subtly wrong.
            #
            # ``from None``: the decode error's message quotes the offending
            # bytes, and those bytes are the secret.
            raise SecretMaterializationError(
                f"secret mount {name!r} is not text and cannot be an environment variable"
            ) from None
        if "\x00" in decoded or "\n" in decoded or "\r" in decoded:
            # NUL is invalid in execve environments; CR/LF would create a second
            # assignment in both systemd and Docker env-file grammars.
            raise SecretMaterializationError(f"secret mount {name!r} cannot be represented in an environment file")
        resolved[name] = decoded
    return resolved


def _validate_mount_name(
    name: str,
    *,
    environment: tp.Mapping[str, str],
    resolved: tp.Mapping[str, str],
) -> None:
    """Defensively repeat the model's environment-name/collision contract."""

    if not _ENVIRONMENT_NAME.fullmatch(name):
        raise SecretMaterializationError("secret mount name is not a valid environment identifier")
    if is_runtime_owned_environment_name(name):
        raise SecretMaterializationError("secret mount name is reserved by Eyvan")
    if name in environment or name in resolved:
        raise SecretMaterializationError("secret mount name collides with an existing environment variable")


def _count(environment: tp.Mapping[str, str]) -> int:
    raw = environment.get(COUNT_ENV, "")
    if not raw:
        return 0
    try:
        if not isinstance(raw, str):
            raise ValueError
        count = int(raw)
    except (TypeError, ValueError):
        # ``raw`` arrived beside capabilities. Never echo an attacker-chosen
        # value from that environment into the agent response or journal.
        raise SecretMaterializationError(f"{COUNT_ENV} is not a valid count") from None
    if count < 0:
        raise SecretMaterializationError(f"{COUNT_ENV} is not a valid count")
    return count


def _fetch(url: str, token: str, *, options: tp.Mapping[str, ssl.SSLContext] | None = None) -> bytes:
    """Redeem one capability, retrying only what is worth retrying.

    A 403 is the control plane saying this attempt is over; retrying it delays a
    launch that is already doomed and adds refusals to the audit trail that look
    like an attack. The shared policy already distinguishes the two.
    """

    request = urllib.request.Request(url, method="GET", headers={"X-Eyvan-Secret-Token": token})
    for attempt in range(HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S, **(options or {})) as response:
                payload = response.read(_MAX_SECRET_BYTES + 1)
                if len(payload) > _MAX_SECRET_BYTES:
                    # Never turn an oversized credential into a different,
                    # apparently valid credential by silently slicing it.
                    raise SecretMaterializationError(f"secret exceeds the {_MAX_SECRET_BYTES}-byte environment limit")
                return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in RETRYABLE_HTTP_STATUS
            delay = retry_delay(exc, attempt)
            # The body is deliberately not read: on the one route in the API that
            # returns a secret, an error body is the only thing standing between
            # a failure message and a value.
            exc.close()
            if not retryable or attempt == HTTP_ATTEMPTS - 1:
                raise SecretMaterializationError(f"secret fetch refused with HTTP {exc.code}") from None
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError):
            if attempt == HTTP_ATTEMPTS - 1:
                # urllib errors can include a request URL. The token belongs in a
                # header today, but preserving that invariant is safer than
                # depending on every future transport exception format.
                raise SecretMaterializationError("secret fetch could not reach the control plane") from None
            delay = retry_delay(None, attempt)
        time.sleep(delay)
    raise SecretMaterializationError("secret fetch failed")


__all__ = (
    "COUNT_ENV",
    "SecretMaterializationError",
    "materialize_secrets",
    "secret_environment_keys",
)
