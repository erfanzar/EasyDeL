"""HTTP admission enforces direct and worker-backed key capabilities without model execution."""

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import CompletionRequest
from easydel.workers.esurge.auth import (
    ApiKeyPermissions,
    ApiKeyRole,
    AuthWorkerClient,
    EnhancedApiKeyManager,
    QuotaConfig,
    RateLimitConfig,
)
from easydel.workers.esurge.auth.worker_main import _auth_worker
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_ok


@pytest.fixture(params=["direct", "worker"])
def auth_manager(request, tmp_path):
    if request.param == "direct":
        yield EnhancedApiKeyManager(require_api_key=True, enable_persistence=False)
        return
    socket_path = tmp_path / "auth.sock"
    endpoint = f"ipc://{socket_path}"
    thread = threading.Thread(
        target=_auth_worker,
        kwargs={
            "endpoint": endpoint,
            "require_api_key": True,
            "admin_key": None,
            "enable_audit_logging": False,
            "max_audit_entries": 100,
            "storage_dir": None,
            "enable_persistence": False,
            "auto_save_interval": 60.0,
        },
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not os.path.exists(socket_path) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), "Auth worker did not bind"
    client = AuthWorkerClient(endpoint)
    try:
        yield client
    finally:
        client.shutdown()
        thread.join(timeout=10)
        assert not thread.is_alive()


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions", "/v1/responses"])
@pytest.mark.parametrize(
    "capability",
    [
        {"stream": True},
        {"tools": [{"type": "function", "function": {"name": "f"}}]},
        {"functions": [{"name": "f"}]},
        {"tool_choice": {"type": "function", "function": {"name": "f"}}},
        {"function_call": {"name": "f"}},
    ],
)
def test_restricted_key_rejected_before_inference(auth_manager, endpoint, capability):
    key, _ = auth_manager.generate_api_key(
        name="restricted", permissions=ApiKeyPermissions(enable_streaming=False, enable_function_calling=False)
    )
    server = object.__new__(eSurgeApiServer)
    server.auth_manager = auth_manager
    server._require_api_key = True
    server._refine_chat_request_callback = None
    server._default_store_responses = False
    server._get_adapter = lambda _: SimpleNamespace(esurge=SimpleNamespace())
    server._ensure_request_max_tokens = lambda *_: 8
    server._parse_responses_max_tokens = lambda *_: (8, 8)
    app = FastAPI()
    handlers = {
        "/v1/chat/completions": server.chat_completions,
        "/v1/completions": server.completions,
        "/v1/responses": server.responses,
    }
    app.add_api_route(endpoint, handlers[endpoint], methods=["POST"])
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "prompt": "hello",
        "input": "hello",
        **capability,
    }
    with TestClient(app) as client:
        response = client.post(endpoint, json=payload, headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403, response.text
    assert (
        "Streaming" in response.json()["detail"]
        if capability.get("stream")
        else ("Function calling" in response.json()["detail"])
    )


@pytest.mark.parametrize("restriction", ["ip", "endpoint"])
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/v1/admin/keys", None),
        ("GET", "/v1/admin/keys/stats", None),
        ("GET", "/v1/admin/keys/target", None),
        ("POST", "/v1/admin/keys", {"name": "new-key"}),
        ("PATCH", "/v1/admin/keys/target", {"name": "changed"}),
        ("DELETE", "/v1/admin/keys/target/revoke", None),
        ("POST", "/v1/admin/keys/target/suspend", None),
        ("POST", "/v1/admin/keys/target/reactivate", None),
        ("POST", "/v1/admin/keys/target/rotate", None),
        ("DELETE", "/v1/admin/keys/target", None),
        ("GET", "/v1/admin/audit-logs", None),
    ],
)
def test_admin_routes_enforce_ip_and_endpoint_policies(auth_manager, restriction, method, path, payload):
    permissions = ApiKeyPermissions(
        allowed_ip_addresses=["192.0.2.1"] if restriction == "ip" else None,
        allowed_endpoints=["/v1/completions"] if restriction == "endpoint" else None,
    )
    key, _ = auth_manager.generate_api_key(name="restricted-admin", role=ApiKeyRole.ADMIN, permissions=permissions)
    server = object.__new__(eSurgeApiServer)
    server.auth_manager = auth_manager
    app = FastAPI()
    for endpoint in server._endpoints:
        app.add_api_route(endpoint.path, endpoint.handler, methods=endpoint.methods)
    with TestClient(app, client=("203.0.113.10", 50000)) as client:
        response = client.request(method, path, json=payload, headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403, response.text
    assert ("IP address" if restriction == "ip" else "endpoint") in response.json()["detail"]


def test_admin_authorization_preserves_401_403_and_429(auth_manager):
    key, _ = auth_manager.generate_api_key(
        name="admin",
        role=ApiKeyRole.ADMIN,
        permissions=ApiKeyPermissions(allowed_ip_addresses=["203.0.113.10"], allowed_endpoints=["/v1/admin/keys/stats"]),
        rate_limits=RateLimitConfig(requests_per_minute=1),
    )
    user_key, _ = auth_manager.generate_api_key(name="user", role=ApiKeyRole.USER)
    server = object.__new__(eSurgeApiServer)
    server.auth_manager = auth_manager
    app = FastAPI()
    app.add_api_route("/v1/admin/keys/stats", server.get_api_key_stats_endpoint, methods=["GET"])
    with TestClient(app, client=("203.0.113.10", 50000)) as client:
        assert client.get("/v1/admin/keys/stats").status_code == 401
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": "invalid"}).status_code == 401
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": user_key}).status_code == 403
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": key}).status_code == 200
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": key}).status_code == 429


def test_admin_requests_complete_zero_token_reservations_and_denied_roles_release(auth_manager):
    key, metadata = auth_manager.generate_api_key(
        name="admin", role=ApiKeyRole.ADMIN, quota=QuotaConfig(max_total_requests=2)
    )
    user_key, user_metadata = auth_manager.generate_api_key(name="user", quota=QuotaConfig(max_total_requests=1))
    server = _inference_server(auth_manager)
    app = FastAPI()
    app.add_api_route("/v1/admin/keys/stats", server.get_api_key_stats_endpoint, methods=["GET"])
    with TestClient(app) as client:
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": key}).status_code == 200
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": key}).status_code == 200
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": key}).status_code == 429
        assert client.get("/v1/admin/keys/stats", headers={"X-API-Key": user_key}).status_code == 403
    assert auth_manager.get_key_by_id(metadata.key_id).total_requests == 2
    assert auth_manager.get_key_by_id(user_metadata.key_id).total_requests == 0
    retry = _raw_request(user_key)
    server._authorize_request(retry, model="m", requested_tokens=1)
    server._record_api_key_usage(retry, 0, 1)
    assert auth_manager.get_key_by_id(user_metadata.key_id).total_requests == 1


def _inference_server(auth_manager):
    server = object.__new__(eSurgeApiServer)
    server.auth_manager = auth_manager
    server._require_api_key = True
    server._ensure_request_max_tokens = lambda request, _: request.max_tokens
    server._get_adapter = lambda _: SimpleNamespace(esurge=SimpleNamespace())
    return server


def _raw_request(key):
    return Request({"type": "http", "headers": [(b"x-api-key", key.encode())]})


def test_out_of_order_completion_reconciles_its_own_reservation_once(auth_manager):
    key, metadata = auth_manager.generate_api_key(name="overlap", quota=QuotaConfig(max_total_tokens=100))
    server = _inference_server(auth_manager)

    async def run():
        gates = {"first": asyncio.Event(), "second": asyncio.Event()}
        admitted = {"first": asyncio.Event(), "second": asyncio.Event()}
        raw_requests = {name: _raw_request(key) for name in gates}

        async def complete(request, esurge, prompt, request_id, raw_request):
            admitted[prompt].set()
            await gates[prompt].wait()
            tokens = 7 if prompt == "first" else 5
            server._record_api_key_usage(raw_request, 0, tokens)
            server._record_api_key_usage(raw_request, 0, tokens)
            return {"tokens": tokens}

        server._handle_completion_response = complete
        first = asyncio.create_task(
            server.completions(CompletionRequest(model="m", prompt="first", max_tokens=80), raw_requests["first"])
        )
        await admitted["first"].wait()
        second = asyncio.create_task(
            server.completions(CompletionRequest(model="m", prompt="second", max_tokens=20), raw_requests["second"])
        )
        try:
            await admitted["second"].wait()
            assert raw_requests["first"].state.auth_reservation_id != raw_requests["second"].state.auth_reservation_id
            gates["second"].set()
            assert await second == {"tokens": 5}
            with pytest.raises(HTTPException) as error:
                await server.completions(
                    CompletionRequest(model="m", prompt="blocked", max_tokens=70), _raw_request(key)
                )
            assert error.value.status_code == 429
        finally:
            gates["first"].set()
            gates["second"].set()
            await asyncio.gather(first, second)

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert current.total_requests == 2
    assert current.total_completion_tokens == 12


@pytest.mark.parametrize("failure", ["invalid", "exception", "cancelled", "stream-close", "stream-error"])
def test_failed_request_releases_only_its_own_admission(auth_manager, failure):
    key, metadata = auth_manager.generate_api_key(name="retry", quota=QuotaConfig(max_total_tokens=8))
    server = _inference_server(auth_manager)

    async def failed(*_):
        if failure == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError("generation failed")

    async def streaming(*_):
        async def body():
            yield "partial"
            raise RuntimeError("stream failed")

        return StreamingResponse(body())

    server._handle_completion_response = failed
    server._handle_completion_streaming = streaming

    async def run():
        raw_request = _raw_request(key)
        request = CompletionRequest(
            model="m", prompt="" if failure == "invalid" else "p", max_tokens=8, stream=failure.startswith("stream")
        )
        if failure in ("invalid", "cancelled"):
            expected = HTTPException if failure == "invalid" else asyncio.CancelledError
            with pytest.raises(expected):
                await server.completions(request, raw_request)
        elif failure.startswith("stream"):
            response = await server.completions(request, raw_request)
            assert await anext(response.body_iterator) == "partial"
            if failure == "stream-error":
                with pytest.raises(RuntimeError, match="stream failed"):
                    await anext(response.body_iterator)
            await response.body_iterator.aclose()
        else:
            assert (await server.completions(request, raw_request)).status_code == 500
        # A fresh full-size admission succeeds only if the failed request released its own budget.
        retry = _raw_request(key)
        server._authorize_request(retry, model="m", requested_tokens=8)
        server._record_api_key_usage(retry, 1, 2)

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert current.total_requests == 1
    assert current.total_prompt_tokens == 1
    assert current.total_completion_tokens == 2


def test_batch_stream_accounts_all_prompts_as_one_http_request(auth_manager):
    key, metadata = auth_manager.generate_api_key(name="batch", quota=QuotaConfig(max_total_requests=1))
    server = _inference_server(auth_manager)
    server.metrics = SimpleNamespace(total_tokens_generated=0)
    server._prepare_sampling_params = lambda *_: None
    server._close_stream_queue = lambda *_: None

    @asynccontextmanager
    async def slot(**_):
        yield

    server._acquire_generation_slot = slot

    def stream(prompt, sampling_params, request_id):
        yield RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="out", token_ids=[4, 5], finish_reason="stop")],
            finished=True,
            accumulated_text="out",
            num_generated_tokens=2,
            processing_time=1.0,
        )

    def start_task(generate, abort):
        queue = asyncio.Queue()
        for output in generate():
            queue.put_nowait(("data", output))
        queue.put_nowait(("end", None))
        return queue

    server._start_stream_task = start_task
    server._get_adapter = lambda _: SimpleNamespace(
        esurge=SimpleNamespace(
            tokenizer=lambda _: {"input_ids": [1, 2, 3]}, stream=stream, abort_request=lambda *_: None
        )
    )
    app = FastAPI()
    app.add_api_route("/v1/completions", server.completions, methods=["POST"])
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "m", "prompt": ["a", "b"], "stream": True, "max_tokens": 8},
            headers={"X-API-Key": key},
        )
    assert response.status_code == 200, response.text
    assert response.text.count("data: [DONE]") == 1
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert current.total_requests == 1
    assert current.total_prompt_tokens == 6
    assert current.total_completion_tokens == 4


def test_shutdown_flushes_auth_without_closing_shared_client(auth_manager):
    # The same manager/client must remain usable after an embedded server shuts down.
    key, metadata = auth_manager.generate_api_key(name="shared")
    server = _inference_server(auth_manager)
    asyncio.run(server.on_shutdown())
    assert auth_manager.validate_key(key).key_id == metadata.key_id


def test_shutdown_persists_dirty_state_and_propagates_flush_failure(tmp_path, monkeypatch):
    from easydel.workers.esurge.auth.auth_storage import AuthStorage

    manager = EnhancedApiKeyManager(storage_dir=str(tmp_path), auto_save=False)
    server = _inference_server(manager)
    key, metadata = manager.generate_api_key(name="persisted")
    raw_request = _raw_request(key)
    server._authorize_request(raw_request, model="m", requested_tokens=8)
    server._record_api_key_usage(raw_request, 2, 3)
    asyncio.run(server.on_shutdown())
    reloaded = EnhancedApiKeyManager(storage_dir=str(tmp_path), auto_save=False)
    assert reloaded.get_key_by_id(metadata.key_id).total_completion_tokens == 3

    def fail_save(*_, **__):
        raise OSError("storage unavailable")

    monkeypatch.setattr(AuthStorage, "save_keys", fail_save)
    with pytest.raises(OSError, match="storage unavailable"):
        asyncio.run(server.on_shutdown())


def test_http_patch_expiry_distinguishes_omission_null_and_positive_ttl(auth_manager):
    admin_key, _ = auth_manager.generate_api_key(name="admin", role=ApiKeyRole.ADMIN)
    _, target = auth_manager.generate_api_key(name="target", expires_in_days=1)
    initial_expiry = target.expires_at
    assert initial_expiry is not None
    server = _inference_server(auth_manager)
    app = FastAPI()
    app.add_api_route("/v1/admin/keys/{key_id}", server.update_api_key_endpoint, methods=["PATCH"])
    with TestClient(app) as client:
        path = f"/v1/admin/keys/{target.key_id}"
        headers = {"X-API-Key": admin_key}
        omitted = client.patch(path, json={"name": "renamed"}, headers=headers)
        assert omitted.status_code == 200, omitted.text
        assert omitted.json()["expires_at"] == initial_expiry
        assert auth_manager.get_key_by_id(target.key_id).expires_at == initial_expiry
        cleared = client.patch(path, json={"expires_in_days": None}, headers=headers)
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["expires_at"] is None
        assert auth_manager.get_key_by_id(target.key_id).expires_at is None
        before = time.time()
        renewed = client.patch(path, json={"expires_in_days": 2}, headers=headers)
        after = time.time()
        assert renewed.status_code == 200, renewed.text
        deadline = renewed.json()["expires_at"]
        assert before + 2 * 86400 <= deadline <= after + 2 * 86400
        assert auth_manager.get_key_by_id(target.key_id).expires_at == deadline


@pytest.mark.parametrize(
    "failure", ["pre-close", "start-error", "start-cancel", "body-error", "body-cancel", "receive-disconnect"]
)
def test_stream_response_lifecycle_releases_unused_admission(auth_manager, failure):
    from starlette.requests import ClientDisconnect

    key, metadata = auth_manager.generate_api_key(name="lifecycle", quota=QuotaConfig(max_total_tokens=8))
    server = _inference_server(auth_manager)

    async def run():
        source_closed = asyncio.Event()
        body_sent = asyncio.Event()

        async def streaming(*_):
            async def body():
                try:
                    yield "partial"
                    await asyncio.Event().wait()
                finally:
                    source_closed.set()

            return StreamingResponse(body(), headers={"X-Custom": "preserved"})

        server._handle_completion_streaming = streaming
        raw_request = _raw_request(key)
        response = await server.completions(
            CompletionRequest(model="m", prompt="p", max_tokens=8, stream=True), raw_request
        )
        assert response.headers["X-Custom"] == "preserved"
        scope = {"type": "http", "asgi": {"spec_version": "2.0" if failure == "receive-disconnect" else "2.4"}}

        async def receive():
            await body_sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            stage = "start" if message["type"] == "http.response.start" else "body"
            if failure == f"{stage}-error":
                raise OSError("client went away")
            if failure == f"{stage}-cancel":
                raise asyncio.CancelledError
            if stage == "body":
                body_sent.set()

        if failure == "pre-close":
            await response.body_iterator.aclose()
        elif failure.endswith("error"):
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
        elif failure.endswith("cancel"):
            with pytest.raises(asyncio.CancelledError):
                await response(scope, receive, send)
        else:
            await response(scope, receive, send)
        if failure.startswith("body") or failure == "receive-disconnect":
            assert source_closed.is_set()
        # No forced GC or extra iterator-close is needed after an ASGI failure.
        retry = _raw_request(key)
        server._authorize_request(retry, model="m", requested_tokens=8)
        server._record_api_key_usage(retry, 1, 2)
        await response.body_iterator.aclose()  # repeated cleanup must not consume another request

    asyncio.run(run())
    current = auth_manager.get_key_by_id(metadata.key_id)
    assert current.total_requests == 1
    assert current.total_prompt_tokens == 1
    assert current.total_completion_tokens == 2


@pytest.mark.parametrize("completed_prompts", [1, 2])
@pytest.mark.parametrize("disconnect", ["body-close", "send-error", "send-cancel"])
def test_batch_disconnect_accounts_completed_prompts(auth_manager, completed_prompts, disconnect):
    from starlette.requests import ClientDisconnect

    key, metadata = auth_manager.generate_api_key(name="batch-disconnect", quota=QuotaConfig(max_total_requests=1))
    server = _inference_server(auth_manager)
    server.metrics = SimpleNamespace(total_tokens_generated=0)
    server._prepare_sampling_params = lambda *_: None
    server._close_stream_queue = lambda *_: None

    @asynccontextmanager
    async def slot(**_):
        yield

    server._acquire_generation_slot = slot
    started_prompts = []

    def stream(prompt, sampling_params, request_id):
        started_prompts.append(prompt)
        yield RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="out", token_ids=[4, 5], finish_reason="stop")],
            finished=True,
            accumulated_text="out",
            num_generated_tokens=2,
            processing_time=1.0,
        )

    def start_task(generate, abort):
        queue = asyncio.Queue()
        for output in generate():
            queue.put_nowait(("data", output))
        queue.put_nowait(("end", None))
        return queue

    server._start_stream_task = start_task
    server._get_adapter = lambda _: SimpleNamespace(
        esurge=SimpleNamespace(
            tokenizer=lambda _: {"input_ids": [1, 2, 3]}, stream=stream, abort_request=lambda *_: None
        )
    )

    async def run():
        raw_request = _raw_request(key)

        async def connected():
            return False

        raw_request.is_disconnected = connected
        response = await server.completions(
            CompletionRequest(model="m", prompt=["a", "b"], max_tokens=8, stream=True), raw_request
        )
        seen_finishes = 0

        def is_disconnect_chunk(event):
            nonlocal seen_finishes
            event = event.decode() if isinstance(event, bytes) else event
            payload = json.loads(event.removeprefix("data: "))
            if any(choice.get("finish_reason") for choice in payload.get("choices", [])):
                seen_finishes += 1
                if seen_finishes == completed_prompts:
                    if completed_prompts == 2:
                        assert payload["usage"]["total_tokens"] == 10
                    return True
            return False

        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body" and is_disconnect_chunk(message["body"]):
                if disconnect == "send-cancel":
                    raise asyncio.CancelledError
                raise OSError("client closed after completed prompt")

        if disconnect == "body-close":
            async for event in response.body_iterator:
                if is_disconnect_chunk(event):
                    break
            await response.body_iterator.aclose()
        else:
            expected = asyncio.CancelledError if disconnect == "send-cancel" else ClientDisconnect
            with pytest.raises(expected):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
        current = auth_manager.get_key_by_id(metadata.key_id)
        assert current.total_requests == 1
        assert current.total_prompt_tokens == 3 * completed_prompts
        assert current.total_completion_tokens == 2 * completed_prompts
        assert server.metrics.total_tokens_generated == 2 * completed_prompts
        assert len(started_prompts) == completed_prompts
        with pytest.raises(HTTPException) as error:
            server._authorize_request(_raw_request(key), model="m", requested_tokens=1)
        assert error.value.status_code == 429
        await response.body_iterator.aclose()
        assert auth_manager.get_key_by_id(metadata.key_id).total_requests == 1

    asyncio.run(run())
