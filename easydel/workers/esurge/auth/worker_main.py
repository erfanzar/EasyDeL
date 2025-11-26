# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Auth worker process for handling authentication operations via ZMQ."""

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
    """Run the auth worker process.

    This function starts a ZeroMQ server that handles authentication requests.

    Args:
            endpoint: ZeroMQ endpoint to bind to.
            require_api_key: If True, all requests must provide a valid API key.
            admin_key: Optional admin key for initial setup.
            enable_audit_logging: Enable audit logging.
            max_audit_entries: Maximum audit log entries to keep.
            storage_dir: Directory for persistent storage.
            enable_persistence: Enable persistent storage.
            auto_save_interval: Auto-save interval in seconds.
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
    """Main entry point for auth worker process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--require-api-key", action="store_true")
    parser.add_argument("--admin-key", default=None)
    parser.add_argument("--enable-audit-logging", action="store_true", default=True)
    parser.add_argument("--max-audit-entries", type=int, default=10000)
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument("--enable-persistence", action="store_true", default=True)
    parser.add_argument("--auto-save-interval", type=float, default=60.0)
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
