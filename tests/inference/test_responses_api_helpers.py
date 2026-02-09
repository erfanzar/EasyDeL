import json

from easydel.inference.inference_engine_interface import BaseInferenceApiServer


def test_responses_reasoning_summary_requested_default_on_with_explicit_opt_out():
    assert BaseInferenceApiServer._responses_reasoning_summary_requested({"reasoning": {"summary": "auto"}}) is True
    assert BaseInferenceApiServer._responses_reasoning_summary_requested({"include": ["reasoning.summary"]}) is True
    assert BaseInferenceApiServer._responses_reasoning_summary_requested({}) is True
    assert BaseInferenceApiServer._responses_reasoning_summary_requested({"reasoning": False}) is False
    assert BaseInferenceApiServer._responses_reasoning_summary_requested({"reasoning": {"summary": "none"}}) is False


def test_responses_payload_to_messages_handles_function_call_output_items():
    payload = {
        "model": "dummy",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "calling tool"}],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"ok": True},
            },
        ],
    }

    messages = BaseInferenceApiServer._responses_payload_to_messages(payload)
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert isinstance(messages[0]["content"], list)
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][0]["text"] == "calling tool"

    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[1]["content"] == json.dumps({"ok": True}, ensure_ascii=False)


def test_build_responses_output_items_are_canonical():
    tool_calls = [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    items = BaseInferenceApiServer._build_responses_output_items(
        output_text="Paris is sunny.",
        tool_calls=tool_calls,
        reasoning_text="I should call weather first.",
        include_reasoning_summary=True,
    )

    assert [item["type"] for item in items] == ["reasoning", "function_call", "message"]
    assert items[0]["summary"][0]["type"] == "summary_text"
    assert items[1]["call_id"] == "call_weather_1"
    assert items[1]["name"] == "get_weather"
    assert items[2]["content"][0]["type"] == "output_text"
    assert items[2]["content"][0]["annotations"] == []
    assert items[2]["content"][0]["logprobs"] == []
    assert items[2]["content"][0]["text"] == "Paris is sunny."


def test_responses_assistant_message_from_output_items_preserves_tool_calls():
    output_items = [
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"abc"}',
            "status": "completed",
        },
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Done."}],
            "status": "completed",
        },
    ]

    assistant_message = BaseInferenceApiServer._responses_assistant_message_from_output_items(output_items)
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Done."
    assert assistant_message["tool_calls"][0]["id"] == "call_1"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "lookup"
