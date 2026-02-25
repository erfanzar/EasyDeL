from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import DeltaMessage


def test_stream_debug_context_captures_key_shapes():
    output = RequestOutput(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=[1, 2],
        outputs=[CompletionOutput(index=0, text="abc", token_ids=[11, 12, 13])],
        accumulated_text="abcdef",
        delta_text="def",
        num_generated_tokens=3,
        delta_tool_calls=[{"index": 0, "function": {"arguments": "{}"}}],
    )
    delta_message = DeltaMessage(role="assistant", content="def", tool_calls=[{"index": 0}])

    context = eSurgeApiServer._build_stream_debug_context(
        endpoint="/v1/chat/completions",
        request_id="req-1",
        model="test-model",
        queue_kind="data",
        disconnected=False,
        output=output,
        previous_text="abc",
        current_text="abcdef",
        delta_text="def",
        previous_token_ids=[11, 12],
        current_token_ids=[11, 12, 13],
        delta_token_ids=[13],
        raw_delta_message=delta_message,
        delta_message=delta_message,
        delta_tool_calls_raw=[{"index": 0}],
        saw_tool_call_delta=True,
    )

    assert context["endpoint"] == "/v1/chat/completions"
    assert context["request_id"] == "req-1"
    assert context["model"] == "test-model"
    assert context["delta_text_len"] == 3
    assert context["previous_text_len"] == 3
    assert context["current_text_len"] == 6
    assert context["output_num_generated_tokens"] == 3
    assert context["output_primary_token_ids_len"] == 3
    assert context["delta_message_tool_calls_len"] == 1


def test_stream_debug_context_truncates_preview():
    text = "x" * 200
    preview = eSurgeApiServer._stream_debug_preview(text, max_chars=50)
    assert preview is not None
    assert preview.endswith("...")
    assert len(preview) == 53


def test_stream_debug_context_includes_stream_error_and_tools_shape():
    err = RuntimeError("boom")
    err.__stream_producer_traceback__ = "Traceback (most recent call last):\n  ...\nRuntimeError: boom\n"
    tools = [{"name": "lookup"}]

    context = eSurgeApiServer._build_stream_debug_context(
        endpoint="/v1/chat/completions",
        request_id="req-err",
        model="test-model",
        queue_kind="error",
        disconnected=False,
        stream_error=err,
        tools=tools,
    )

    assert context["queue_kind"] == "error"
    assert context["stream_error_type"] == "RuntimeError"
    assert context["stream_error_message"] == "boom"
    assert context["stream_error_producer_traceback"] is not None
    assert context["tools_type"] == "list"
    assert context["tools_len"] == 1
    assert context["first_tool_type"] == "dict"


def test_tool_protocol_text_detection_handles_partial_markers():
    assert eSurgeApiServer._looks_like_tool_protocol_text("<tool_call><arg_key>name</arg_key>") is True
    assert eSurgeApiServer._looks_like_tool_protocol_text("<tool name") is True
    assert eSurgeApiServer._looks_like_tool_protocol_text("normal assistant text") is False
