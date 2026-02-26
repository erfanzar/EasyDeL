from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.openai_api_modules import DeltaMessage
from easydel.inference.tools.tool_calling_mixin import ToolCallingMixin


def test_jsonify_tool_calls_filters_non_mapping_entries():
    payload = BaseInferenceApiServer._jsonify_tool_calls(
        [
            "bad-entry",
            {"index": 0, "function": {"name": "lookup", "arguments": "{}"}},
            123,
        ]
    )
    assert payload == [{"index": 0, "function": {"name": "lookup", "arguments": "{}"}}]


def test_coerce_stream_delta_message_normalizes_tool_calls_and_fallback():
    delta = DeltaMessage(content="ok", tool_calls=["bad", {"index": 0, "function": {"arguments": "{}"}}])
    normalized = BaseInferenceApiServer._coerce_stream_delta_message(
        delta, fallback_text="fallback", default_role="assistant"
    )

    assert normalized is not None
    assert normalized.role == "assistant"
    assert normalized.content == "ok"
    assert normalized.tool_calls == [{"index": 0, "function": {"arguments": "{}"}}]

    normalized_bad = BaseInferenceApiServer._coerce_stream_delta_message(object(), fallback_text="fallback")
    assert normalized_bad is not None
    assert normalized_bad.content == "fallback"


class _BrokenStreamingParser:
    def extract_tool_calls_streaming(self, **_kwargs):
        raise AttributeError("'str' object has no attribute 'items'")


class _StringStreamingParser:
    def extract_tool_calls_streaming(self, **_kwargs):
        return "delta-from-parser"


class _Server(ToolCallingMixin):
    pass


def test_tool_calling_mixin_streaming_falls_back_on_parser_exception():
    server = _Server()
    server.tool_parsers = {"test-model": _BrokenStreamingParser()}

    delta = server.extract_tool_calls_streaming(
        model_name="test-model",
        previous_text="",
        current_text="abc",
        delta_text="abc",
        request=None,
    )
    assert isinstance(delta, DeltaMessage)
    assert delta.content == "abc"


def test_tool_calling_mixin_streaming_coerces_string_delta():
    server = _Server()
    server.tool_parsers = {"test-model": _StringStreamingParser()}

    delta = server.extract_tool_calls_streaming(
        model_name="test-model",
        previous_text="old",
        current_text="new",
        delta_text="raw",
        request=None,
    )
    assert isinstance(delta, DeltaMessage)
    assert delta.content == "delta-from-parser"


def test_compute_delta_text_handles_overlap_and_shrink_without_warnings():
    # Overlap recovery.
    overlap_delta = BaseInferenceApiServer._compute_delta_text(
        current_text="world!",
        previous_text="hello world",
        fallback_delta="!",
    )
    assert overlap_delta == "!"

    # Snapshot rewrite shrink: prefer fallback if it is not a replay.
    shrink_delta = BaseInferenceApiServer._compute_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="answer",
    )
    assert shrink_delta == "answer"

    # Snapshot rewrite shrink without fallback should emit nothing.
    shrink_empty = BaseInferenceApiServer._compute_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="",
    )
    assert shrink_empty == ""


def test_compute_snapshot_delta_text_handles_shrink_without_reset_noise():
    delta = EngineUtilsMixin._compute_snapshot_delta_text(
        current_text="answer",
        previous_text="very long prior content",
        fallback_delta="answer",
    )
    assert delta == "answer"
