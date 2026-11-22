"""Exercise real remote/routed protocol bridges without models, arrays or networking."""

import inspect
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.distributed.routed_engine import RoutedEngine
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput

pytestmark = pytest.mark.cpu_ok


def _snapshot(count, text, *, finished=False):
    return RequestOutput(
        request_id="progress",
        prompt="prompt",
        prompt_token_ids=[1, 2, 3],
        outputs=[
            CompletionOutput(
                index=0,
                text="visible" if count >= 2 else "",
                token_ids=list(range(count)),
                finish_reason="length" if finished else None,
            )
        ],
        finished=finished,
        accumulated_text="visible" if count >= 2 else "",
        delta_text=text,
        num_generated_tokens=count,
    )


def _engine(monkeypatch, *, routed, snapshots):
    remote = object.__new__(RemoteEngineHandle)
    remote._channel = SimpleNamespace(alive=True)
    remote._fatal = None
    monkeypatch.setattr(remote, "chat", lambda **kwargs: snapshots())
    return RoutedEngine([remote]) if routed else remote


def _stream(engine, endpoint, callback=None):
    kwargs = {"model": "test", "messages": []}
    method = engine.iter_chat_completion_stream if endpoint == "chat" else engine.iter_responses_stream
    assert inspect.signature(method).parameters["usage_callback"].default is None
    if endpoint == "responses":
        kwargs["response_id"] = "progress"
    if callback is not None:
        kwargs["usage_callback"] = callback
    return method(**kwargs)


def _is_content(frame, endpoint):
    if endpoint == "chat":
        return any(choice.delta.content == "visible" for choice in frame.choices)
    return frame.event == "response.output_text.delta"


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("routed", [False, True])
def test_real_remote_bridges_report_invisible_usage_before_content(monkeypatch, endpoint, routed):
    def snapshots():
        yield _snapshot(1, "")
        yield _snapshot(2, "visible")
        pytest.fail("Closing after content must not consume another snapshot")

    engine = _engine(monkeypatch, routed=routed, snapshots=snapshots)
    observed = []
    stream = _stream(engine, endpoint, lambda prompt, completion: observed.append((prompt, completion)))
    try:
        for frame in stream:
            if _is_content(frame, endpoint):
                assert observed == [(3, 1), (3, 2)]
                if endpoint == "chat":
                    assert frame.usage is None
                break
        else:
            pytest.fail("No visible content was emitted")
    finally:
        stream.close()
    if routed:
        assert engine._inflight == [0]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("routed", [False, True])
def test_real_remote_bridges_keep_terminal_usage_with_optional_progress(monkeypatch, endpoint, routed):
    def snapshots():
        yield _snapshot(1, "")
        yield _snapshot(2, "visible")
        yield _snapshot(3, "", finished=True)

    engine = _engine(monkeypatch, routed=routed, snapshots=snapshots)
    observed = []
    frames = list(_stream(engine, endpoint, lambda prompt, completion: observed.append((prompt, completion))))
    assert observed == [(3, 1), (3, 2), (3, 3)]
    if endpoint == "chat":
        assert all(frame.usage is None for frame in frames[:-1])
        assert frames[-1].usage.prompt_tokens == 3
        assert frames[-1].usage.completion_tokens == 3
    else:
        assert frames[-1].event == "response.completed"
        assert frames[-1].payload.response.usage.input_tokens == 3
        assert frames[-1].payload.response.usage.output_tokens == 3
    if routed:
        assert engine._inflight == [0]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("routed", [False, True])
def test_progress_errors_propagate_before_content_and_release_routes(monkeypatch, endpoint, routed):
    def snapshots():
        yield _snapshot(1, "visible")
        pytest.fail("Callback error must stop the protocol bridge")

    def failed_callback(prompt, completion):
        raise RuntimeError("progress consumer failed")

    engine = _engine(monkeypatch, routed=routed, snapshots=snapshots)
    stream = _stream(engine, endpoint, failed_callback)
    with pytest.raises(RuntimeError, match="progress consumer failed"):
        for frame in stream:
            assert not _is_content(frame, endpoint)
    if routed:
        assert engine._inflight == [0]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("with_callback", [False, True])
@pytest.mark.parametrize("with_adapter", [False, True])
def test_routed_legacy_member_stream_signatures_remain_compatible(endpoint, with_callback, with_adapter):
    def legacy(*, messages, model, response_id=None):
        yield (messages, model, response_id)

    def adapter(**kwargs):
        return legacy(**kwargs)

    stream_fn = adapter if with_adapter else legacy
    member = SimpleNamespace(iter_chat_completion_stream=stream_fn, iter_responses_stream=stream_fn)
    engine = RoutedEngine([member])
    callback = (lambda prompt, completion: pytest.fail("Legacy member has no progress API")) if with_callback else None
    assert list(_stream(engine, endpoint, callback)) == [([], "test", "progress" if endpoint == "responses" else None)]
    assert engine._inflight == [0]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_routed_eager_stream_failure_releases_acquired_route(endpoint):
    def failed(**kwargs):
        raise RuntimeError("member failed before iterator creation")

    member = SimpleNamespace(iter_chat_completion_stream=failed, iter_responses_stream=failed)
    engine = RoutedEngine([member])
    with pytest.raises(RuntimeError, match="before iterator creation"):
        list(_stream(engine, endpoint))
    assert engine._inflight == [0]
