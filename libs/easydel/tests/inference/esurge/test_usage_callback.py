"""Array-free coverage of actual engine bridges and protocol usage observers."""

import pytest
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge

pytestmark = pytest.mark.cpu_ok


def snapshot(generated, text="", *, finished=False):
    return RequestOutput(
        request_id="usage-progress",
        prompt="hello",
        prompt_token_ids=[[11, 12], [13]],
        outputs=[
            CompletionOutput(index=0, text=text, token_ids=[21] * generated, finish_reason="stop" if finished else None)
        ],
        finished=finished,
        accumulated_text=text,
        delta_text=text,
        num_generated_tokens=generated,
    )


def stream_for(endpoint, outputs, **kwargs):
    # Retain the real public methods and converters, replacing only model work.
    engine = eSurge.__new__(eSurge)

    def chat(**chat_kwargs):
        assert chat_kwargs["stream"] is True
        return outputs

    engine.chat = chat
    if endpoint == "chat":
        return engine.iter_chat_completion_stream(model="test", messages=[], **kwargs)
    return engine.iter_responses_stream(response_id="resp_usage", model="test", messages=[], **kwargs)


def has_text(endpoint, frame):
    if endpoint == "chat":
        return any(choice.delta.content == "visible" for choice in frame.choices)
    return frame.event == "response.output_text.delta" and frame.payload.delta == "visible"


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_usage_observed_before_visible_event_and_retained_on_close(endpoint):
    observed = []

    def outputs():
        yield snapshot(1)  # Parser-buffered tokens must count even without text.
        yield snapshot(4, "visible")
        pytest.fail("Closing at a delta must not require consuming terminal output")

    stream = stream_for(
        endpoint, outputs(), usage_callback=lambda prompt, completion: observed.append((prompt, completion))
    )
    try:
        for frame in stream:
            if has_text(endpoint, frame):
                assert observed == [(3, 1), (3, 4)]
                break
        else:
            pytest.fail("Missing visible output")
    finally:
        stream.close()
    assert observed == [(3, 1), (3, 4)]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_invisible_progress_survives_source_error(endpoint):
    observed = []

    def outputs():
        yield snapshot(2)
        raise RuntimeError("decoder failed")

    stream = stream_for(
        endpoint, outputs(), usage_callback=lambda prompt, completion: observed.append((prompt, completion))
    )
    with pytest.raises(RuntimeError, match="decoder failed"):
        list(stream)
    assert observed == [(3, 2)]


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
def test_usage_observer_failure_is_not_swallowed_or_emitted_after_content(endpoint):
    def callback(prompt, completion):
        assert (prompt, completion) == (3, 4)
        raise ValueError("accounting failed")

    stream = stream_for(endpoint, iter([snapshot(4, "visible")]), usage_callback=callback)
    with pytest.raises(ValueError, match="accounting failed"):
        for frame in stream:
            assert not has_text(endpoint, frame)


@pytest.mark.parametrize("endpoint", ["chat", "responses"])
@pytest.mark.parametrize("callback_mode", ["omitted", "none", "observer"])
def test_terminal_wire_usage_unchanged_with_optional_observer(endpoint, callback_mode):
    observed = []
    kwargs = {}
    if callback_mode == "none":
        kwargs["usage_callback"] = None
    elif callback_mode == "observer":
        kwargs["usage_callback"] = lambda prompt, completion: observed.append((prompt, completion))
    frames = list(stream_for(endpoint, iter([snapshot(1), snapshot(4, "visible", finished=True)]), **kwargs))
    assert observed == ([(3, 1), (3, 4)] if callback_mode == "observer" else [])
    if endpoint == "chat":
        assert all(frame.usage is None for frame in frames[:-1])
        usage = frames[-1].usage
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 4, 7)
        assert frames[-1].choices[0].finish_reason == "stop"
    else:
        assert frames[-1].event == "response.completed"
        usage = frames[-1].payload.response.usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (3, 4, 7)
        assert all(frame.payload.model_dump().get("response", {}).get("usage") is None for frame in frames[:-1])
