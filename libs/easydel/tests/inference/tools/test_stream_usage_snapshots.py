"""Raw snapshot usage survives protocol filtering and stream interruption."""

import asyncio
import uuid

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.stream_protocol import (
    iter_chat_completion_stream_responses,
    iter_responses_stream_frames,
)

pytestmark = pytest.mark.cpu_ok


def _snapshot(tokens, text="", *, prompt_ids=None, finished=False):
    return RequestOutput(
        request_id="usage",
        prompt="hello",
        prompt_token_ids=[1, 2, 3] if prompt_ids is None else prompt_ids,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=list(range(tokens)),
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
        accumulated_text=text,
        delta_text=text,
        num_generated_tokens=tokens,
    )


def _convert(adapter, snapshots, callback=None):
    if adapter == "chat":
        return iter_chat_completion_stream_responses(
            snapshots, model="m", completion_id="chat-usage", created=1, usage_callback=callback
        )
    return iter_responses_stream_frames(snapshots, response_id="usage", model="m", created_at=1, usage_callback=callback)


def _is_text(adapter, frame):
    if adapter == "chat":
        return any(choice.delta.content == "hello" for choice in frame.choices)
    return frame.event == "response.output_text.delta"


@pytest.mark.parametrize("adapter", ["chat", "responses"])
@pytest.mark.parametrize("segmented", [False, True])
def test_usage_observes_every_snapshot_once_without_changing_wire_frames(adapter, segmented, monkeypatch):
    # Item IDs are generated independently of usage observation. Pin the UUID
    # source so both conversions can be compared byte-for-byte structurally.
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=1))
    prompt_ids = [[1], [2, 3]] if segmented else [1, 2, 3]
    snapshots = [
        _snapshot(0, prompt_ids=prompt_ids),
        _snapshot(2, prompt_ids=prompt_ids),
        _snapshot(5, "hello", prompt_ids=prompt_ids),
        _snapshot(7, prompt_ids=prompt_ids, finished=True),
    ]

    def wire_frame(frame):
        if adapter == "chat":
            return frame.model_dump()
        return {"event": frame.event, "data": frame.payload.model_dump()}

    observations = []
    frames = []
    wire_frames = []
    for frame in _convert(adapter, iter(snapshots), lambda p, c: observations.append((p, c))):
        frames.append(frame)
        # Serialize at yield time just like SSE transport: accumulator items
        # remain mutable while the stream is in progress.
        wire_frames.append(wire_frame(frame))
        if _is_text(adapter, frame):
            # The visible frame must not race ahead of its usage, and invisible
            # snapshots must not disappear from accounting.
            assert observations == [(3, 0), (3, 2), (3, 5)]
    assert observations == [(3, 0), (3, 2), (3, 5), (3, 7)]
    assert wire_frames == [wire_frame(frame) for frame in _convert(adapter, iter(snapshots))]
    if adapter == "chat":
        assert all(frame.usage is None for frame in frames[:-1])
        usage = frames[-1].usage
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 7, 10)
    else:
        assert frames[-1].event == "response.completed"
        payloads = [frame.payload.model_dump() for frame in frames]
        assert all(payload.get("response", {}).get("usage") is None for payload in payloads[:-1])
        usage = payloads[-1]["response"]["usage"]
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (3, 7, 10)


@pytest.mark.parametrize("adapter", ["chat", "responses"])
@pytest.mark.parametrize("interruption", ["close", "cancel"])
def test_usage_is_retained_before_close_or_cancellation_without_read_ahead(adapter, interruption):
    observations = []
    consumed = []

    def snapshots():
        for snapshot in [_snapshot(2), _snapshot(5, "hello")]:
            consumed.append(snapshot.num_generated_tokens)
            yield snapshot
        pytest.fail("An interrupted protocol stream must not consume another snapshot")

    stream = _convert(adapter, snapshots(), lambda p, c: observations.append((p, c)))
    try:
        for frame in stream:
            if _is_text(adapter, frame):
                break
        else:
            pytest.fail("Expected a visible text delta")
        assert observations == [(3, 2), (3, 5)]
        if interruption == "cancel":
            with pytest.raises(asyncio.CancelledError):
                stream.throw(asyncio.CancelledError())
    finally:
        stream.close()
    assert consumed == [2, 5]
    assert observations == [(3, 2), (3, 5)]


@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_usage_survives_producer_failure_after_invisible_snapshot(adapter):
    observations = []
    failure = RuntimeError("producer failed")

    def snapshots():
        yield _snapshot(2)
        yield _snapshot(5)
        raise failure

    stream = _convert(adapter, snapshots(), lambda p, c: observations.append((p, c)))
    with pytest.raises(RuntimeError) as caught:
        list(stream)
    assert caught.value is failure
    assert observations == [(3, 2), (3, 5)]


@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_usage_callback_errors_propagate_before_snapshot_frames(adapter):
    observations = []
    failure = ValueError("observer failed")

    def observe(prompt, completion):
        observations.append((prompt, completion))
        raise failure

    frames = []
    stream = _convert(adapter, iter([_snapshot(5, "hello")]), observe)
    with pytest.raises(ValueError) as caught:
        for frame in stream:
            frames.append(frame)
    assert caught.value is failure
    assert observations == [(3, 5)]
    # Responses can emit its initial skeleton before consuming any snapshot.
    assert frames == [] if adapter == "chat" else [frame.event for frame in frames] == ["response.created"]


@pytest.mark.parametrize("adapter", ["chat", "responses"])
def test_empty_stream_has_no_synthetic_usage_callback(adapter):
    observations = []
    with pytest.raises(RuntimeError, match="without any output"):
        list(_convert(adapter, iter([]), lambda p, c: observations.append((p, c))))
    assert observations == []
