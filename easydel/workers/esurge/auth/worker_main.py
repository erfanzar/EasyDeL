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

"""Subprocess entry point for the eSurge auth ZMQ worker.

Hosts a single :class:`EnhancedApiKeyManager` instance behind a ZMQ REP
socket and dispatches one auth command per request
(``generate_api_key``, ``validate_key``, ``authorize_request``,
``record_usage``, key lifecycle commands, audit-log fetch, statistics,
and ``shutdown``). The serving loop translates manager exceptions back
into a wire payload (``status="error"`` with ``exception_type`` set)
that :class:`AuthWorkerClient` re-raises locally as
:class:`PermissionDenied`, :class:`RateLimitExceeded` or
:class:`QuotaExceeded`.

Designed to be spawned by :class:`AuthWorkerManager` under a CPU-only
JAX configuration (the worker does no model work) so the API server
can keep authentication state, audit logs and rate-limit windows in a
single process even when multiple FastAPI workers serve requests.

Invocation:
    The script is invoked via ``python -m
    easydel.workers.esurge.auth.worker_main`` with the following CLI
    options::

        --endpoint <zmq-endpoint>         (required)
        --require-api-key                 (flag)
        --admin-key <secret>
        --enable-audit-logging / --disable-audit-logging
        --max-audit-entries <int>         (default 10000)
        --storage-dir <path>
        --enable-persistence / --disable-persistence
        --auto-save-interval <seconds>    (default 60.0)

    Before importing JAX-aware modules ``main()`` forces a CPU-only
    placement and disables distributed init so the auth worker stays
    out of the way of the model workers.
"""

from __future__ import annotations

import argparse
import os

import zmq

from .auth_manager import EnhancedApiKeyManager


def _auth_worker(
    endpoint: str,
    require_api_key: bool,
    admin_key: str | None,
    enable_audit_logging: bool,
    max_audit_entries: int,
    storage_dir: str | None,
    enable_persistence: bool,
    auto_save_interval: float,
) -> None:
    """Run the auth worker REQ/REP loop until a ``shutdown`` command arrives.

    Constructs a single :class:`EnhancedApiKeyManager`, binds a ZMQ
    ``REP`` socket on ``endpoint``, then serves one Pyobj-encoded
    command per iteration. Each incoming message must be a dict with a
    ``cmd`` key whose value selects one of the supported manager
    operations; the reply is always a dict with a ``status`` of
    ``"ok"`` or ``"error"`` plus operation-specific payload fields.

    Exceptions raised by the manager during ``authorize_request`` are
    forwarded with their ``type(e).__name__`` in ``exception_type`` so
    :class:`AuthWorkerClient` can re-raise the matching local class.

    Args:
        endpoint: ZeroMQ endpoint to bind to (e.g. ``ipc:///tmp/auth.sock``
            or ``tcp://127.0.0.1:5555``).
        require_api_key: When ``True``, the wrapped manager rejects
            unauthenticated calls.
        admin_key: Optional bootstrap admin key passed straight to the
            manager constructor; refreshed on every restart.
        enable_audit_logging: Whether the manager records mutation /
            authorisation events.
        max_audit_entries: Capacity of the in-memory audit-log ring.
        storage_dir: Filesystem path for :class:`AuthStorage`. ``None``
            falls back to ``~/.cache/esurge-auth``.
        enable_persistence: Whether to attach the on-disk
            :class:`AuthStorage` and hydrate from it.
        auto_save_interval: Minimum seconds between auto-saves; also
            controls how often the dirty flag triggers a flush.
    """
    # Initialize auth manager
    auth_manager = EnhancedApiKeyManager(
        require_api_key=require_api_key,
        admin_key=admin_key,
        enable_audit_logging=enable_audit_logging,
        max_audit_entries=max_audit_entries,
        storage_dir=storage_dir,
        enable_persistence=enable_persistence,
        auto_save=enable_persistence,
        save_interval=auto_save_interval,
    )

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(endpoint)

    try:
        while True:
            message = socket.recv_pyobj()
            cmd = message.get("cmd")

            if cmd == "generate_api_key":
                try:
                    raw_key, metadata = auth_manager.generate_api_key(
                        name=message["name"],
                        role=message.get("role"),
                        description=message.get("description"),
                        created_by=message.get("created_by"),
                        expires_in_days=message.get("expires_in_days"),
                        rate_limits=message.get("rate_limits"),
                        quota=message.get("quota"),
                        permissions=message.get("permissions"),
                        tags=message.get("tags"),
                        metadata=message.get("metadata"),
                    )
                    socket.send_pyobj(
                        {
                            "status": "ok",
                            "raw_key": raw_key,
                            "metadata": metadata.as_dict(include_sensitive=False),
                        }
                    )
                except Exception as e:
                    socket.send_pyobj({"status": "error", "message": str(e)})

            elif cmd == "validate_key":
                metadata = auth_manager.validate_key(message["raw_key"])
                socket.send_pyobj(
                    {
                        "status": "ok",
                        "metadata": metadata.as_dict(include_sensitive=False) if metadata else None,
                    }
                )

            elif cmd == "authorize_request":
                try:
                    metadata = auth_manager.authorize_request(
                        raw_key=message["raw_key"],
                        ip_address=message.get("ip_address"),
                        endpoint=message.get("endpoint"),
                        model=message.get("model"),
                        requested_tokens=message.get("requested_tokens", 0),
                    )
                    socket.send_pyobj(
                        {
                            "status": "ok",
                            "metadata": metadata.as_dict(include_sensitive=False),
                        }
                    )
                except Exception as e:
                    socket.send_pyobj({"status": "error", "message": str(e), "exception_type": type(e).__name__})

            elif cmd == "record_usage":
                auth_manager.record_usage(
                    raw_key=message["raw_key"],
                    prompt_tokens=message["prompt_tokens"],
                    completion_tokens=message["completion_tokens"],
                )
                socket.send_pyobj({"status": "ok"})

            elif cmd == "revoke_key":
                success = auth_manager.revoke_key(
                    key_id=message["key_id"],
                    revoked_by=message.get("revoked_by"),
                )
                socket.send_pyobj({"status": "ok", "success": success})

            elif cmd == "suspend_key":
                success = auth_manager.suspend_key(
                    key_id=message["key_id"],
                    suspended_by=message.get("suspended_by"),
                )
                socket.send_pyobj({"status": "ok", "success": success})

            elif cmd == "reactivate_key":
                success = auth_manager.reactivate_key(
                    key_id=message["key_id"],
                    reactivated_by=message.get("reactivated_by"),
                )
                socket.send_pyobj({"status": "ok", "success": success})

            elif cmd == "delete_key":
                success = auth_manager.delete_key(
                    key_id=message["key_id"],
                    deleted_by=message.get("deleted_by"),
                )
                socket.send_pyobj({"status": "ok", "success": success})

            elif cmd == "get_key_by_id":
                metadata = auth_manager.get_key_by_id(message["key_id"])
                socket.send_pyobj(
                    {
                        "status": "ok",
                        "metadata": metadata.as_dict(include_sensitive=False) if metadata else None,
                    }
                )

            elif cmd == "list_keys":
                keys = auth_manager.list_keys(
                    role=message.get("role"),
                    status=message.get("status"),
                    tags=message.get("tags"),
                )
                socket.send_pyobj(
                    {
                        "status": "ok",
                        "keys": [k.as_dict(include_sensitive=False) for k in keys],
                    }
                )

            elif cmd == "update_key":
                success = auth_manager.update_key(
                    key_id=message["key_id"],
                    name=message.get("name"),
                    description=message.get("description"),
                    role=message.get("role"),
                    expires_in_days=message.get("expires_in_days"),
                    rate_limits=message.get("rate_limits"),
                    quota=message.get("quota"),
                    permissions=message.get("permissions"),
                    tags=message.get("tags"),
                    metadata=message.get("metadata"),
                    updated_by=message.get("updated_by"),
                )
                socket.send_pyobj({"status": "ok", "success": success})

            elif cmd == "rotate_key":
                result = auth_manager.rotate_key(
                    key_id=message["key_id"],
                    rotated_by=message.get("rotated_by"),
                )
                if result:
                    new_raw_key, metadata = result
                    socket.send_pyobj(
                        {
                            "status": "ok",
                            "raw_key": new_raw_key,
                            "metadata": metadata.as_dict(include_sensitive=False),
                        }
                    )
                else:
                    socket.send_pyobj({"status": "error", "message": "Key not found"})

            elif cmd == "get_audit_logs":
                logs = auth_manager.get_audit_logs(
                    limit=message.get("limit", 100),
                    key_id=message.get("key_id"),
                    action=message.get("action"),
                )
                socket.send_pyobj(
                    {
                        "status": "ok",
                        "logs": [log.as_dict() for log in logs],
                    }
                )

            elif cmd == "get_statistics":
                stats = auth_manager.get_statistics()
                socket.send_pyobj({"status": "ok", "statistics": stats})

            elif cmd == "shutdown":
                socket.send_pyobj({"status": "ok"})
                break

            else:
                socket.send_pyobj({"status": "error", "message": f"Unknown cmd {cmd}"})

    finally:
        socket.close(0)
        ctx.term()


def main():
    """CLI entry point: parse arguments, force CPU-only JAX, run the loop.

    Parses ``sys.argv`` for the worker configuration described in the
    module docstring, sets ``JAX_PLATFORMS=cpu`` (plus
    ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` and
    ``ENABLE_DISTRIBUTED_INIT=0``) so spawning this script never
    competes with the model workers for accelerators, then hands off to
    :func:`_auth_worker`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--require-api-key", action="store_true")
    parser.add_argument("--admin-key", default=None)
    audit_group = parser.add_mutually_exclusive_group()
    audit_group.add_argument("--enable-audit-logging", dest="enable_audit_logging", action="store_true")
    audit_group.add_argument("--disable-audit-logging", dest="enable_audit_logging", action="store_false")
    parser.add_argument("--max-audit-entries", type=int, default=10000)
    parser.add_argument("--storage-dir", default=None)
    persistence_group = parser.add_mutually_exclusive_group()
    persistence_group.add_argument("--enable-persistence", dest="enable_persistence", action="store_true")
    persistence_group.add_argument("--disable-persistence", dest="enable_persistence", action="store_false")
    parser.add_argument("--auto-save-interval", type=float, default=60.0)
    parser.set_defaults(enable_audit_logging=True, enable_persistence=True)
    args = parser.parse_args()

    # Disable JAX initialization
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

    _auth_worker(
        endpoint=args.endpoint,
        require_api_key=args.require_api_key,
        admin_key=args.admin_key,
        enable_audit_logging=args.enable_audit_logging,
        max_audit_entries=args.max_audit_entries,
        storage_dir=args.storage_dir,
        enable_persistence=args.enable_persistence,
        auto_save_interval=args.auto_save_interval,
    )


if __name__ == "__main__":
    main()
