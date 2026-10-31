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

"""Regression tests for ``EnhancedApiKeyManager`` (pure Python, no array computation)."""

import pytest
from easydel.workers.esurge.auth import (
    ApiKeyPermissions,
    EnhancedApiKeyManager,
    PermissionDenied,
    QuotaConfig,
    QuotaExceeded,
    RateLimitConfig,
    RateLimitExceeded,
)

pytestmark = pytest.mark.cpu_ok


def _manager(**kwargs) -> EnhancedApiKeyManager:
    return EnhancedApiKeyManager(enable_persistence=False, **kwargs)


@pytest.mark.parametrize("admin_key", ["", "sk-admin-secret"])
def test_short_admin_key_is_rejected_loudly(admin_key):
    with pytest.raises(ValueError, match="at least 16"):
        _manager(admin_key=admin_key)


def test_valid_admin_key_is_created():
    mgr = _manager(admin_key="sk-admin-secret-long-enough")
    assert mgr.validate_key("sk-admin-secret-long-enough") is not None


def test_unknown_keys_are_not_cached():
    mgr = _manager()
    for i in range(50):
        assert mgr.validate_key(f"sk-unknown-token-{i:04d}-xxxxxxxx") is None
    assert len(mgr._raw_to_hash_cache) == 0


def test_rotated_key_invalidates_old_secret():
    mgr = _manager()
    old_raw, meta = mgr.generate_api_key(name="k")
    assert mgr.validate_key(old_raw) is not None
    new_raw, _ = mgr.rotate_key(meta.key_id)
    assert mgr.validate_key(old_raw) is None
    assert mgr.validate_key(new_raw) is not None


def test_rotate_unknown_key_returns_none():
    assert _manager().rotate_key("key_missing") is None


def test_rejected_requests_do_not_consume_rate_budget():
    mgr = _manager()
    raw, _ = mgr.generate_api_key(
        name="k",
        rate_limits=RateLimitConfig(requests_per_minute=2),
        permissions=ApiKeyPermissions(max_tokens_per_request=5),
    )
    for _ in range(6):
        with pytest.raises(PermissionDenied):
            mgr.authorize_request(raw, requested_tokens=10)
    # Both admitted requests still fit in the per-minute budget.
    mgr.authorize_request(raw, requested_tokens=1)
    mgr.authorize_request(raw, requested_tokens=1)


def test_streaming_permission_is_enforced():
    mgr = _manager()
    raw, _ = mgr.generate_api_key(name="k", permissions=ApiKeyPermissions(enable_streaming=False))
    mgr.authorize_request(raw, stream=False)
    with pytest.raises(PermissionDenied, match="Streaming"):
        mgr.authorize_request(raw, stream=True)


def test_function_calling_permission_is_enforced():
    mgr = _manager()
    raw, _ = mgr.generate_api_key(name="k", permissions=ApiKeyPermissions(enable_function_calling=False))
    mgr.authorize_request(raw, uses_tools=False)
    with pytest.raises(PermissionDenied, match="Function calling"):
        mgr.authorize_request(raw, uses_tools=True)


def test_persisted_statistics_roundtrip_and_clear(tmp_path):
    mgr = EnhancedApiKeyManager(storage_dir=tmp_path, enable_persistence=True, auto_save=True, save_interval=0)
    raw, _ = mgr.generate_api_key(name="k")
    mgr.record_usage(raw, prompt_tokens=2, completion_tokens=3)
    restarted = EnhancedApiKeyManager(storage_dir=tmp_path, enable_persistence=True, auto_save=False)
    stats = restarted.get_persisted_statistics()
    assert stats["total_keys"] == 1
    assert stats["total_tokens_all_keys"] == 5
    assert restarted.validate_key(raw) is not None
    assert list(tmp_path.glob("*.bak"))
    mgr.clear_persisted_data()
    assert mgr.get_persisted_statistics() == {}
    assert not list(tmp_path.iterdir())
    empty = EnhancedApiKeyManager(storage_dir=tmp_path, enable_persistence=True, auto_save=False)
    assert empty.validate_key(raw) is None


@pytest.mark.parametrize(
    ("rate_limits", "quota", "error"),
    [
        (RateLimitConfig(requests_per_minute=1, tokens_per_minute=5), None, RateLimitExceeded),
        (RateLimitConfig(requests_per_minute=1), QuotaConfig(max_total_tokens=5), QuotaExceeded),
    ],
)
def test_late_rejections_preserve_request_rate_budget(rate_limits, quota, error):
    mgr = _manager()
    raw, _ = mgr.generate_api_key(name="k", rate_limits=rate_limits, quota=quota)
    for _ in range(3):
        with pytest.raises(error):
            mgr.authorize_request(raw, requested_tokens=6)
    assert mgr.authorize_request(raw, requested_tokens=1) is not None
    with pytest.raises(RateLimitExceeded):
        mgr.authorize_request(raw, requested_tokens=1)


def test_hour_limit_rejections_do_not_charge_minute_window(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("easydel.workers.esurge.auth.auth_manager.time.time", lambda: now[0])
    mgr = _manager()
    raw, _ = mgr.generate_api_key(name="k", rate_limits=RateLimitConfig(requests_per_minute=1, requests_per_hour=1))
    mgr.authorize_request(raw)
    now[0] = 3590.0
    with pytest.raises(RateLimitExceeded):
        mgr.authorize_request(raw)
    now[0] = 3601.0
    assert mgr.authorize_request(raw) is not None


@pytest.mark.parametrize("expires_in_days", [None, 7])
def test_clear_expiry_persists_and_audits_removal(tmp_path, expires_in_days):
    mgr = EnhancedApiKeyManager(storage_dir=tmp_path, auto_save=False)
    raw, meta = mgr.generate_api_key(name="expired", expires_in_days=-1)
    assert mgr.validate_key(raw) is None
    original_expiry = mgr.get_key_by_id(meta.key_id).expires_at
    assert mgr.update_key(meta.key_id, name="unchanged-expiry")
    assert mgr.get_key_by_id(meta.key_id).expires_at == original_expiry

    assert mgr.update_key(meta.key_id, expires_in_days=expires_in_days, clear_expiry=True, updated_by="admin")
    assert mgr.validate_key(raw).expires_at is None
    event = mgr.get_audit_logs(key_id=meta.key_id, action="key_updated")[0]
    assert event.actor == "admin"
    assert event.details == {"expires_at": None}
    mgr.close()
    reopened = EnhancedApiKeyManager(storage_dir=tmp_path, auto_save=False)
    assert reopened.validate_key(raw).expires_at is None
    assert not mgr.update_key("missing", clear_expiry=True)
