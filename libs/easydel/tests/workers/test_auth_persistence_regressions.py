"""Public-behavior regressions for auth budgets, persistence and pipeline workers."""

import json
import threading

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
from easydel.workers.esurge.auth import worker_manager as auth_workers
from easydel.workers.esurge.pipeline import worker_main
from easydel.workers.esurge.pipeline.zmq_workers import TokenizerWorkerClient
from easydel.workers.response_store.file_store import FileResponseStore

pytestmark = pytest.mark.cpu_ok


def manager():
    return EnhancedApiKeyManager(enable_persistence=False)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"rate_limits": RateLimitConfig(requests_per_minute=0)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(requests_per_hour=0)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(requests_per_day=0)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_minute=0)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_hour=0)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_day=0)}, RateLimitExceeded),
        ({"quota": QuotaConfig(max_total_requests=0)}, QuotaExceeded),
        ({"quota": QuotaConfig(max_total_tokens=0)}, QuotaExceeded),
        ({"quota": QuotaConfig(monthly_request_limit=0)}, QuotaExceeded),
        ({"quota": QuotaConfig(monthly_token_limit=0)}, QuotaExceeded),
        ({"permissions": ApiKeyPermissions(max_tokens_per_request=0)}, PermissionDenied),
    ],
)
def test_explicit_zero_budgets_are_enforced(options, error):
    mgr = manager()
    raw, _ = mgr.generate_api_key(name="zero", **options)
    with pytest.raises(error):
        mgr.authorize_request(raw, requested_tokens=1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RateLimitConfig(tokens_per_minute=-1),
        lambda: QuotaConfig(monthly_request_limit=-1),
        lambda: ApiKeyPermissions(max_tokens_per_request=-1),
    ],
)
def test_negative_configuration_is_rejected(factory):
    with pytest.raises(ValueError, match="nonnegative"):
        factory()


@pytest.mark.parametrize("allowed", [None, [], ["203.0.113.5"], ["203.0.113.6"]])
def test_ip_allowlists_distinguish_empty_from_unrestricted(allowed):
    mgr = manager()
    raw, _ = mgr.generate_api_key(name="ips", permissions=ApiKeyPermissions(allowed_ip_addresses=allowed))
    if allowed is None or "203.0.113.5" in allowed:
        assert mgr.authorize_request(raw, ip_address="203.0.113.5") is not None
    else:
        with pytest.raises(PermissionDenied):
            mgr.authorize_request(raw, ip_address="203.0.113.5")


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"quota": QuotaConfig(max_total_requests=1)}, QuotaExceeded),
        ({"quota": QuotaConfig(max_total_tokens=10)}, QuotaExceeded),
        ({"quota": QuotaConfig(monthly_request_limit=1)}, QuotaExceeded),
        ({"quota": QuotaConfig(monthly_token_limit=10)}, QuotaExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_minute=10)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_hour=10)}, RateLimitExceeded),
        ({"rate_limits": RateLimitConfig(tokens_per_day=10)}, RateLimitExceeded),
    ],
)
def test_pending_admissions_reserve_budgets_and_cancellation_releases(options, error):
    mgr = manager()
    raw, _ = mgr.generate_api_key(name="pending", **options)
    mgr.authorize_request(raw, requested_tokens=10, reservation_id="a")
    with pytest.raises(error):
        mgr.authorize_request(raw, requested_tokens=10, reservation_id="b")
    assert mgr.release_reservation(raw, "a")
    assert not mgr.release_reservation(raw, "a")
    mgr.authorize_request(raw, requested_tokens=10, reservation_id="b")


def test_completion_reconciles_estimate_exactly_once_and_out_of_order():
    mgr = manager()
    raw, _ = mgr.generate_api_key(name="tokens", rate_limits=RateLimitConfig(tokens_per_minute=20))
    mgr.authorize_request(raw, requested_tokens=10, reservation_id="a")
    mgr.authorize_request(raw, requested_tokens=10, reservation_id="b")
    mgr.record_usage(raw, 1, 2, reservation_id="b")
    mgr.record_usage(raw, 1, 2, reservation_id="b")
    meta = mgr.validate_key(raw)
    assert meta.total_requests == 1
    assert meta.monthly_tokens == 3
    mgr.authorize_request(raw, requested_tokens=7, reservation_id="c")
    with pytest.raises(RateLimitExceeded):
        mgr.authorize_request(raw, requested_tokens=1)
    assert mgr.release_reservation(raw, "a")
    assert mgr.release_reservation(raw, "c")


def test_rotation_cannot_resurrect_a_concurrently_deleted_key(monkeypatch):
    mgr = manager()
    raw, meta = mgr.generate_api_key(name="concurrent")
    lookup_complete = threading.Event()
    resume_rotation = threading.Event()
    delete_started = threading.Event()
    delete_complete = threading.Event()
    original_lookup = mgr.get_key_by_id
    rotated = []

    def paused_lookup(key_id):
        result = original_lookup(key_id)
        lookup_complete.set()
        assert resume_rotation.wait(5)
        return result

    def delete():
        delete_started.set()
        assert mgr.delete_key(meta.key_id)
        delete_complete.set()

    monkeypatch.setattr(mgr, "get_key_by_id", paused_lookup)
    rotation = threading.Thread(target=lambda: rotated.append(mgr.rotate_key(meta.key_id)))
    rotation.start()
    assert lookup_complete.wait(5)
    deletion = threading.Thread(target=delete)
    deletion.start()
    assert delete_started.wait(5)
    delete_complete.wait(0.1)
    resume_rotation.set()
    rotation.join(5)
    deletion.join(5)
    assert not rotation.is_alive() and not deletion.is_alive()
    assert mgr.validate_key(raw) is None
    assert mgr.validate_key(rotated[0][0]) is None
    assert mgr.list_keys() == []


def test_rotation_preserves_inflight_accounting_without_allowing_new_admission():
    mgr = manager()
    raw, meta = mgr.generate_api_key(name="rotate-inflight", rate_limits=RateLimitConfig(tokens_per_minute=10))
    mgr.authorize_request(raw, requested_tokens=10, reservation_id="work")
    new_raw, _ = mgr.rotate_key(meta.key_id)
    with pytest.raises(PermissionDenied):
        mgr.authorize_request(raw, requested_tokens=1)
    mgr.record_usage(raw, 1, 2, reservation_id="work")
    assert mgr.validate_key(new_raw).monthly_tokens == 3
    mgr.authorize_request(new_raw, requested_tokens=7, reservation_id="next")
    assert mgr.release_reservation(new_raw, "next")


def test_negative_usage_components_cannot_create_token_credit():
    mgr = manager()
    raw, _ = mgr.generate_api_key(name="usage", rate_limits=RateLimitConfig(tokens_per_minute=10))
    mgr.record_usage(raw, -100, 5)
    meta = mgr.validate_key(raw)
    assert meta.total_prompt_tokens == 0
    assert meta.total_completion_tokens == meta.monthly_tokens == 5
    with pytest.raises(RateLimitExceeded):
        mgr.authorize_request(raw, requested_tokens=6)


def test_lifecycle_changes_are_durable_before_ack_and_close_flushes_usage(tmp_path):
    mgr = EnhancedApiKeyManager(storage_dir=tmp_path, save_interval=60)
    raw, meta = mgr.generate_api_key(name="persist")
    assert EnhancedApiKeyManager(storage_dir=tmp_path).validate_key(raw) is not None
    mgr.record_usage(raw, 2, 3)
    mgr.close()
    assert EnhancedApiKeyManager(storage_dir=tmp_path).validate_key(raw).monthly_tokens == 5
    assert mgr.revoke_key(meta.key_id)
    assert EnhancedApiKeyManager(storage_dir=tmp_path).validate_key(raw) is None


def test_auth_worker_shutdown_flushes_pending_usage(tmp_path):
    from easydel.workers.esurge.auth import AuthWorkerClient
    from easydel.workers.esurge.auth.worker_main import _auth_worker

    endpoint = f"ipc://{tmp_path / 'auth.sock'}"
    thread = threading.Thread(
        target=_auth_worker,
        args=(endpoint, False, None, True, 100, str(tmp_path / "storage"), True, 60.0),
        daemon=True,
    )
    thread.start()
    client = AuthWorkerClient(endpoint)
    try:
        raw, _ = client.generate_api_key(name="shutdown-persistence")
        client.authorize_request(raw, requested_tokens=8, reservation_id="r")
        client.record_usage(raw, 3, 5, reservation_id="r")
    finally:
        client.shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive()
    restarted = EnhancedApiKeyManager(storage_dir=tmp_path / "storage")
    assert restarted.validate_key(raw).monthly_tokens == 8


def test_auth_missing_primary_recovers_backup_and_restores_primary(tmp_path):
    mgr = EnhancedApiKeyManager(storage_dir=tmp_path)
    raw, _ = mgr.generate_api_key(name="recovery")
    primary = tmp_path / "keys.json"
    primary.replace(tmp_path / "keys.bak")
    restored = EnhancedApiKeyManager(storage_dir=tmp_path)
    assert restored.validate_key(raw) is not None
    assert primary.exists()


def test_response_missing_primary_recovers_backup(tmp_path):
    store = FileResponseStore(tmp_path)
    store.put_response("a", {"v": 1})
    store.put_conversation("c", [{"content": "hello"}])
    store.close()
    store.index_file.replace(tmp_path / "index.json.bak")
    restored = FileResponseStore(tmp_path)
    assert restored.get_response("a") == {"v": 1}
    assert restored.get_conversation("c") == [{"content": "hello"}]
    assert restored.index_file.exists()


@pytest.mark.parametrize("limit", [0, 1])
def test_reopening_store_applies_reduced_capacity_immediately(tmp_path, limit):
    store = FileResponseStore(tmp_path)
    for number in range(3):
        store.put_response(str(number), {"v": number})
        store.put_conversation(str(number), [{"v": number}])
    store.close()
    restored = FileResponseStore(tmp_path, max_stored_responses=limit, max_stored_conversations=limit)
    assert restored.stats()["responses"] == restored.stats()["conversations"] == limit
    assert len(list(restored.responses_dir.iterdir())) == limit
    assert len(list(restored.conversations_dir.iterdir())) == limit
    reopened = FileResponseStore(tmp_path)
    assert reopened.stats()["responses"] == reopened.stats()["conversations"] == limit


@pytest.mark.parametrize("response_id", ["../../escaped", "/tmp/absolute-response-id", "safe"])
def test_response_ids_are_data_not_filesystem_paths(tmp_path, response_id):
    store = FileResponseStore(tmp_path / "store")
    store.put_response(response_id, {"v": 1})
    index = json.loads(store.index_file.read_text())
    filename = index["responses"]["items"][response_id]["file"]
    assert "/" not in filename
    assert store.get_response(response_id) == {"v": 1}
    assert store.delete_response(response_id)
    assert not list(store.responses_dir.iterdir())


def test_safe_legacy_response_records_remain_readable(tmp_path):
    store = FileResponseStore(tmp_path)
    store.put_response("legacy", {"v": 1})
    store.close()
    assert (store.responses_dir / "legacy.bin").exists()
    reopened = FileResponseStore(tmp_path)
    assert reopened.get_response("legacy") == {"v": 1}
    reopened.put_response("legacy", {"v": 2})
    assert reopened.get_response("legacy") == {"v": 2}


def test_bootstrap_credential_is_never_logged(monkeypatch):
    messages = []
    secret = "sk-complete-bootstrap-secret"
    monkeypatch.setattr(auth_workers.logger, "info", lambda text: messages.append(text))
    from io import StringIO
    from types import SimpleNamespace

    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdin=StringIO())

    monkeypatch.setattr(auth_workers.subprocess, "Popen", spawn)
    auth_workers.AuthWorkerManager(admin_key=secret)._spawn_auth_worker("ipc:///tmp/unused")
    assert all(secret not in argument for command in commands for argument in command)
    assert messages
    assert all(secret not in message for message in messages)


def test_tokenizer_exception_replies_and_continues_serving(monkeypatch, tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")), unk_token="[UNK]"
    )
    monkeypatch.setattr(worker_main.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    endpoint = f"ipc://{tmp_path / 'tokenizer.sock'}"
    thread = threading.Thread(target=worker_main._tokenizer_worker, args=(endpoint, "unused", {}), daemon=True)
    thread.start()
    client = TokenizerWorkerClient(endpoint, request_timeout=5)
    try:
        with pytest.raises(RuntimeError):
            client.tokenize("bad", "\ud800")
        assert client.tokenize("valid", "hello") == [1]
    finally:
        client.shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_tokenizer_timeout_does_not_poison_req_socket(tmp_path):
    endpoint = f"ipc://{tmp_path / 'absent.sock'}"
    client = TokenizerWorkerClient(endpoint, request_timeout=0.02)
    try:
        with pytest.raises(TimeoutError):
            client.tokenize("first", "hello")
        # Another request must time out rather than raise ZMQ's EFSM error.
        with pytest.raises(TimeoutError):
            client.tokenize("second", "hello")
    finally:
        client.close()
