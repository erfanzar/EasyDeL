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

"""Tests for ``easydel.inference.typed_models``.

This module defines the Pydantic models that mirror the OpenAI Responses
API wire format (``/v1/responses``). We test:

* Default-factory ID prefixes (``rs_``, ``fc_``, ``msg_``, ``call_``)
* Pydantic discriminator dispatch on ``ResponsesOutputItem``
* ``model_validate`` round-trip through dict for each item type
* ``assistant_message_from_output_items`` collapses output items to ChatMessage
* Stream-state dataclasses' ``item_id`` property delegation
* JSON round-trip via ``model_dump`` / ``model_validate``
"""

from __future__ import annotations

import json

import pytest

from easydel.inference.typed_models import (
    FunctionCallStreamState,
    MessageStreamState,
    ReasoningStreamState,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseOutputTextPart,
    ResponseReasoningItem,
    ResponseSummaryText,
    ResponsesFinalizationOptions,
    ResponsesResponse,
    ResponsesTextConfig,
    ResponsesTextFormat,
    ResponsesUsage,
    StreamEventFrame,
    assistant_message_from_output_items,
)


def test_response_summary_text_default_type():
    item = ResponseSummaryText()
    assert item.type == "summary_text"
    assert item.text == ""


def test_response_output_text_part_default_type_and_lists():
    part = ResponseOutputTextPart()
    assert part.type == "output_text"
    assert part.annotations == []
    assert part.logprobs == []
    assert part.text == ""


def test_response_reasoning_item_id_has_rs_prefix():
    item = ResponseReasoningItem()
    assert item.id.startswith("rs_")
    assert item.type == "reasoning"
    assert item.summary == []


def test_response_function_call_item_ids_have_correct_prefixes():
    """``id`` is prefixed ``fc_``; ``call_id`` is prefixed ``call_``."""
    item = ResponseFunctionCallItem()
    assert item.id.startswith("fc_")
    assert item.call_id.startswith("call_")
    assert item.type == "function_call"
    assert item.status == "completed"


def test_response_message_item_id_has_msg_prefix():
    item = ResponseMessageItem()
    assert item.id.startswith("msg_")
    assert item.type == "message"
    assert item.role == "assistant"
    assert item.content == []
    assert item.status == "completed"


def test_response_function_call_item_unique_ids():
    """Default-factory IDs are uuid-based -- two instances must differ."""
    a = ResponseFunctionCallItem()
    b = ResponseFunctionCallItem()
    assert a.id != b.id
    assert a.call_id != b.call_id


def test_responses_response_discriminator_dispatches_message_item():
    """A dict with type='message' is parsed as ``ResponseMessageItem``."""
    response = ResponsesResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 1700000000,
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_a",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            ],
        }
    )
    assert isinstance(response.output[0], ResponseMessageItem)
    assert response.output[0].content[0].text == "hi"


def test_responses_response_discriminator_dispatches_function_call():
    response = ResponsesResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 1700000000,
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_a",
                    "call_id": "call_a",
                    "name": "search",
                    "arguments": '{"q": "x"}',
                },
            ],
        }
    )
    assert isinstance(response.output[0], ResponseFunctionCallItem)
    assert response.output[0].name == "search"


def test_responses_response_discriminator_dispatches_reasoning():
    response = ResponsesResponse.model_validate(
        {
            "id": "resp_1",
            "created_at": 1700000000,
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_a",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                },
            ],
        }
    )
    assert isinstance(response.output[0], ResponseReasoningItem)
    assert response.output[0].summary[0].text == "thinking"


def test_responses_response_unknown_output_type_raises():
    """Pydantic's discriminator rejects an unrecognized type."""
    with pytest.raises(Exception):
        ResponsesResponse.model_validate(
            {
                "id": "resp_1",
                "created_at": 1700000000,
                "model": "test-model",
                "status": "completed",
                "output": [{"type": "not_a_known_type"}],
            }
        )


def test_responses_response_default_text_config():
    """``ResponsesResponse.text`` defaults to ``ResponsesTextConfig`` with format='text'."""
    response = ResponsesResponse(
        id="r",
        created_at=0,
        model="m",
        status="completed",
    )
    assert isinstance(response.text, ResponsesTextConfig)
    assert response.text.format.type == "text"


def test_responses_usage_zero_defaults():
    usage = ResponsesUsage()
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_responses_text_format_default_value():
    fmt = ResponsesTextFormat()
    assert fmt.type == "text"


def test_responses_finalization_options_all_none_by_default():
    """All fields default to None so model_copy(update=…) only overwrites set fields."""
    opts = ResponsesFinalizationOptions()
    assert opts.error is None
    assert opts.incomplete_details is None
    assert opts.instructions is None
    assert opts.max_output_tokens is None
    assert opts.previous_response_id is None
    assert opts.store is None
    assert opts.temperature is None
    assert opts.top_p is None
    assert opts.truncation is None
    assert opts.tool_choice is None


def test_response_message_item_json_round_trip():
    original = ResponseMessageItem(
        id="msg_test",
        content=[ResponseOutputTextPart(text="hello")],
    )
    serialized = original.model_dump_json()
    revived = ResponseMessageItem.model_validate_json(serialized)
    assert revived.id == original.id
    assert revived.content[0].text == "hello"


def test_responses_response_json_round_trip():
    original = ResponsesResponse(
        id="resp_test",
        created_at=1700000000,
        model="test",
        status="completed",
        output=[
            ResponseMessageItem(content=[ResponseOutputTextPart(text="hi there")]),
            ResponseFunctionCallItem(name="search", arguments='{"q":"x"}'),
        ],
        usage=ResponsesUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    serialized = original.model_dump_json()
    parsed = json.loads(serialized)
    assert parsed["id"] == "resp_test"
    assert parsed["status"] == "completed"
    assert len(parsed["output"]) == 2
    revived = ResponsesResponse.model_validate_json(serialized)
    assert isinstance(revived.output[0], ResponseMessageItem)
    assert isinstance(revived.output[1], ResponseFunctionCallItem)


def test_reasoning_stream_state_item_id_property_delegates():
    item = ResponseReasoningItem(id="rs_explicit")
    state = ReasoningStreamState(item=item, output_index=0)
    assert state.item_id == "rs_explicit"
    assert state.done is False


def test_function_call_stream_state_item_id_property_delegates():
    item = ResponseFunctionCallItem(id="fc_explicit")
    state = FunctionCallStreamState(item=item, output_index=2)
    assert state.item_id == "fc_explicit"


def test_message_stream_state_default_content_index_is_zero():
    item = ResponseMessageItem(id="msg_explicit")
    state = MessageStreamState(item=item, output_index=1)
    assert state.content_index == 0
    assert state.item_id == "msg_explicit"
    assert state.done is False


def test_stream_event_frame_pairs_event_name_and_payload():
    frame = StreamEventFrame(event="response.created", payload={"foo": "bar"})
    assert frame.event == "response.created"
    assert frame.payload == {"foo": "bar"}


def test_assistant_message_from_output_items_collapses_text_parts():
    items = [
        ResponseMessageItem(content=[ResponseOutputTextPart(text="hello "), ResponseOutputTextPart(text="world")]),
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.role == "assistant"
    assert "hello" in (msg.content or "")
    assert "world" in (msg.content or "")


def test_assistant_message_includes_function_calls_as_tool_calls():
    items = [
        ResponseFunctionCallItem(name="search", arguments='{"q":"a"}', call_id="call_1"),
        ResponseFunctionCallItem(name="translate", arguments='{"text":"b"}', call_id="call_2"),
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2
    names = {tc.function.name for tc in msg.tool_calls}
    assert names == {"search", "translate"}


def test_assistant_message_skips_reasoning_items():
    """Reasoning items aren't part of the visible chat history."""
    items = [
        ResponseReasoningItem(summary=[ResponseSummaryText(text="thinking")]),
        ResponseMessageItem(content=[ResponseOutputTextPart(text="hello")]),
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.content is not None
    assert "hello" in msg.content
    assert "thinking" not in msg.content
    assert not msg.tool_calls


def test_assistant_message_accepts_dict_items():
    """Raw dicts (e.g. from JSON-roundtripped storage) are accepted alongside model instances."""
    items = [
        {
            "type": "message",
            "id": "msg_dict",
            "content": [{"type": "output_text", "text": "from dict"}],
        },
        {
            "type": "function_call",
            "id": "fc_dict",
            "call_id": "call_dict",
            "name": "search",
            "arguments": "{}",
        },
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.content and "from dict" in msg.content
    assert msg.tool_calls and msg.tool_calls[0].function.name == "search"


def test_assistant_message_skips_unknown_dict_types():
    """Unknown ``type`` in a dict item is silently skipped (per the function contract)."""
    items = [
        {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        {"type": "mystery_type", "data": "ignored"},
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.content and "ok" in msg.content


def test_assistant_message_skips_non_dict_non_model_items():
    """A list or string in the items list is silently skipped."""
    items = [
        "a stray string",
        ResponseMessageItem(content=[ResponseOutputTextPart(text="kept")]),
        42,
    ]
    msg = assistant_message_from_output_items(items)
    assert msg.content and "kept" in msg.content
