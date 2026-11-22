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

"""Auth worker <-> ``AuthWorkerClient`` round-trip tests (pure Python, no array computation)."""

import os
import threading
import time

import pytest
from easydel.workers.esurge.auth import (
    ApiKeyPermissions,
    AuditLogEntry,
    AuthWorkerClient,
    PermissionDenied,
    QuotaConfig,
    QuotaExceeded,
    RateLimitConfig,
    RateLimitExceeded,
)
from easydel.workers.esurge.auth.worker_main import _auth_worker

pytestmark = pytest.mark.cpu_ok


@pytest.fixture
def client(tmp_path):
    endpoint = f"ipc://{tmp_path / 'auth.sock'}"
    thread = threading.Thread(
        target=_auth_worker,
        kwargs={
            "endpoint": endpoint,
            "require_api_key": False,
            "admin_key": None,
            "enable_audit_logging": True,
            "max_audit_entries": 100,
            "storage_dir": None,
            "enable_persistence": False,
            "auto_save_interval": 60.0,
        },
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 10
    while not os.path.exists(tmp_path / "auth.sock") and time.time() < deadline:
        time.sleep(0.02)
    cli = AuthWorkerClient(endpoint)
    yield cli
    cli.shutdown()
    thread.join(timeout=10)


def test_permission_denied_crosses_the_wire_as_the_shared_class(client):
    with pytest.raises(PermissionDenied):
        client.authorize_request("sk-this-key-does-not-exist")


def test_rate_limit_and_quota_errors_cross_the_wire_as_shared_classes(client):
    raw, _ = client.generate_api_key(name="limited", rate_limits=RateLimitConfig(requests_per_minute=1))
    client.authorize_request(raw)
    with pytest.raises(RateLimitExceeded):
        client.authorize_request(raw)
    raw, _ = client.generate_api_key(name="quota", quota=QuotaConfig(max_total_tokens=1))
    with pytest.raises(QuotaExceeded):
        client.authorize_request(raw, requested_tokens=2)


def test_get_audit_logs_returns_entries_with_as_dict(client):
    raw, _ = client.generate_api_key(name="k")
    client.authorize_request(raw)
    logs = client.get_audit_logs()
    assert logs and all(isinstance(entry, AuditLogEntry) for entry in logs)
    assert {entry.as_dict()["action"] for entry in logs} >= {"key_created", "request_authorized"}


def test_reservations_reconcile_and_cancel_across_the_wire(client):
    raw, _ = client.generate_api_key(name="pending", rate_limits=RateLimitConfig(tokens_per_minute=10))
    client.authorize_request(raw, requested_tokens=10, reservation_id="first")
    with pytest.raises(RateLimitExceeded):
        client.authorize_request(raw, requested_tokens=1, reservation_id="second")
    assert client.release_reservation(raw, "first")
    client.authorize_request(raw, requested_tokens=10, reservation_id="second")
    client.record_usage(raw, 1, 2, reservation_id="second")
    client.record_usage(raw, 1, 2, reservation_id="second")
    assert client.validate_key(raw).total_requests == 1
    client.authorize_request(raw, requested_tokens=7, reservation_id="third")
    assert client.release_reservation(raw, "third")
    client.flush()


@pytest.mark.parametrize("expires_in_days", [None, 7])
def test_update_key_clears_expiry_across_the_wire(client, expires_in_days):
    raw, metadata = client.generate_api_key(name="expired", expires_in_days=-1)
    assert client.validate_key(raw) is None
    original_expiry = client.get_key_by_id(metadata.key_id).expires_at
    assert client.update_key(metadata.key_id, name="same-expiry")
    assert client.get_key_by_id(metadata.key_id).expires_at == original_expiry
    # Older clients omit clear_expiry entirely; the worker must retain the TTL.
    assert client._request({"cmd": "update_key", "key_id": metadata.key_id, "description": "legacy"})["success"]
    assert client.get_key_by_id(metadata.key_id).expires_at == original_expiry

    assert client.update_key(metadata.key_id, expires_in_days=expires_in_days, clear_expiry=True, updated_by="admin")
    assert client.validate_key(raw).expires_at is None
    event = client.get_audit_logs(key_id=metadata.key_id, action="key_updated")[0]
    assert event.actor == "admin"
    assert event.details == {"expires_at": None}
    assert client.update_key(metadata.key_id, expires_in_days=2)
    assert client.validate_key(raw).expires_at > time.time()


def test_rotate_unknown_key_returns_none(client):
    assert client.rotate_key("key_missing") is None


def test_rotate_known_key_returns_new_secret(client):
    raw, meta = client.generate_api_key(name="k")
    rotated = client.rotate_key(meta.key_id)
    assert rotated is not None
    new_raw, new_meta = rotated
    assert new_raw != raw
    assert new_meta.key_id == meta.key_id


def test_malformed_command_does_not_kill_the_worker(client):
    with pytest.raises(RuntimeError):
        client._request({"cmd": "validate_key"})  # missing ``raw_key``
    with pytest.raises(RuntimeError):
        client._request({"cmd": "revoke_key"})  # missing ``key_id``
    raw, _ = client.generate_api_key(name="still-alive")
    assert client.validate_key(raw) is not None


def test_streaming_and_tool_restrictions_cross_the_wire(client):
    raw, _ = client.generate_api_key(
        name="k", permissions=ApiKeyPermissions(enable_streaming=False, enable_function_calling=False)
    )
    client.authorize_request(raw)
    with pytest.raises(PermissionDenied):
        client.authorize_request(raw, stream=True)
    with pytest.raises(PermissionDenied):
        client.authorize_request(raw, uses_tools=True)
