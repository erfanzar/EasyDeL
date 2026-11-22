"""Check real engine stream bridges retain partial usage without running models."""

import pytest
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.workers.esurge.auth import EnhancedApiKeyManager, QuotaConfig, QuotaExceeded
from starlette.requests import Request

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("engine_cls", [eSurge, RemoteEngineHandle])
@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_real_engine_partial_stream_consumes_actual_quota(engine_cls, endpoint):
    manager = EnhancedApiKeyManager(require_api_key=True, enable_persistence=False)
    key, metadata = manager.generate_api_key(name="partial", quota=QuotaConfig(max_total_tokens=10))
    manager.authorize_request(key, requested_tokens=10, reservation_id="partial")
    raw_request = Request({"type": "http"})
    raw_request.state.api_key = key
    raw_request.state.auth_reservation_id = "partial"
    server = object.__new__(eSurgeApiServer)
    server.auth_manager = manager
    engine = object.__new__(engine_cls)

    def snapshots(**kwargs):
        yield RequestOutput(
            request_id="partial",
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="observed", token_ids=[4, 5])],
            finished=False,
            accumulated_text="observed",
            delta_text="observed",
            num_generated_tokens=2,
        )
        pytest.fail("An interrupted stream must not need a terminal snapshot to account usage")

    # Keep the production protocol bridges; replace only model execution.
    engine.chat = snapshots
    stream_fn = engine.iter_chat_completion_stream if endpoint == "chat" else engine.iter_responses_stream
    kwargs = {"model": "m", "messages": []}
    if endpoint == "responses":
        kwargs["response_id"] = "partial"
    stream = stream_fn(**kwargs, **server._stream_usage_callback_kwargs(stream_fn, raw_request))
    try:
        for frame in stream:
            if endpoint == "chat":
                observed_delta = any(choice.delta.content == "observed" for choice in frame.choices)
            else:
                observed_delta = frame.event == "response.output_text.delta"
            if observed_delta:
                break
        else:
            pytest.fail("The real engine bridge did not expose the generated content")
    finally:
        stream.close()
        server._release_api_key_reservation(raw_request)

    current = manager.get_key_by_id(metadata.key_id)
    assert (current.total_requests, current.total_prompt_tokens, current.total_completion_tokens) == (1, 3, 2)
    with pytest.raises(QuotaExceeded):
        manager.authorize_request(key, requested_tokens=6, reservation_id="too-large")
    manager.authorize_request(key, requested_tokens=5, reservation_id="remaining")
    assert manager.release_reservation(key, "remaining")
