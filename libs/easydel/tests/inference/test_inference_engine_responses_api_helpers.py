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

import json

from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.openai_api_modules import ResponsesRequest


def test_responses_reasoning_summary_requested_default_on_with_explicit_opt_out():
    assert (
        BaseInferenceApiServer._responses_reasoning_summary_requested(
            ResponsesRequest.model_validate({"model": "dummy", "reasoning": {"summary": "auto"}})
        )
        is True
    )
    assert (
        BaseInferenceApiServer._responses_reasoning_summary_requested(
            ResponsesRequest.model_validate({"model": "dummy", "include": ["reasoning.summary"]})
        )
        is True
    )
    assert BaseInferenceApiServer._responses_reasoning_summary_requested(ResponsesRequest(model="dummy")) is True
    assert (
        BaseInferenceApiServer._responses_reasoning_summary_requested(
            ResponsesRequest.model_validate({"model": "dummy", "reasoning": False})
        )
        is False
    )
    assert (
        BaseInferenceApiServer._responses_reasoning_summary_requested(
            ResponsesRequest.model_validate({"model": "dummy", "reasoning": {"summary": "none"}})
        )
        is False
    )


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

    messages = BaseInferenceApiServer._responses_payload_to_messages(ResponsesRequest.model_validate(payload))
    assert len(messages) == 2
    assert messages[0].role == "assistant"
    assert isinstance(messages[0].content, list)
    assert messages[0].content[0]["type"] == "text"
    assert messages[0].content[0]["text"] == "calling tool"

    assert messages[1].role == "tool"
    assert messages[1].tool_call_id == "call_1"
    assert messages[1].content == json.dumps({"ok": True}, ensure_ascii=False)


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

    assert [item.type for item in items] == ["reasoning", "function_call", "message"]
    assert items[0].summary[0].type == "summary_text"
    assert items[1].call_id == "call_weather_1"
    assert items[1].name == "get_weather"
    assert items[2].content[0].type == "output_text"
    assert items[2].content[0].annotations == []
    assert items[2].content[0].logprobs == []
    assert items[2].content[0].text == "Paris is sunny."


def test_build_responses_output_items_skip_empty_message_for_tool_only_output():
    tool_calls = [
        {
            "id": "call_weather_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]

    items = BaseInferenceApiServer._build_responses_output_items(output_text="", tool_calls=tool_calls)

    assert [item.type for item in items] == ["function_call"]


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
    assert assistant_message.role == "assistant"
    assert assistant_message.content == "Done."
    assert assistant_message.tool_calls is not None
    assert assistant_message.tool_calls[0].id == "call_1"
    assert assistant_message.tool_calls[0].function.name == "lookup"


def test_responses_assistant_message_from_tool_only_items_uses_null_content():
    output_items = [
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"abc"}',
            "status": "completed",
        }
    ]

    assistant_message = BaseInferenceApiServer._responses_assistant_message_from_output_items(output_items)
    assert assistant_message.role == "assistant"
    assert assistant_message.content is None
    assert assistant_message.tool_calls is not None
    assert assistant_message.tool_calls[0].id == "call_1"


def test_build_responses_output_items_text_only():
    """Text-only output without tools or reasoning."""
    items = BaseInferenceApiServer._build_responses_output_items(
        output_text="Hello world",
        tool_calls=None,
        reasoning_text=None,
        include_reasoning_summary=False,
    )
    assert len(items) == 1
    assert items[0].type == "message"
    assert items[0].content[0].text == "Hello world"


def test_build_responses_output_items_empty_output():
    """Empty output without tools should produce a message with empty text."""
    items = BaseInferenceApiServer._build_responses_output_items(
        output_text="",
        tool_calls=None,
    )
    assert len(items) == 1
    assert items[0].type == "message"


def test_build_responses_output_items_reasoning_only():
    """Reasoning without content text."""
    items = BaseInferenceApiServer._build_responses_output_items(
        output_text="",
        tool_calls=None,
        reasoning_text="Let me think...",
        include_reasoning_summary=True,
    )
    types = [i.type for i in items]
    assert "reasoning" in types
    reasoning_item = next(i for i in items if i.type == "reasoning")
    assert reasoning_item.summary[0].text == "Let me think..."


def test_responses_payload_to_messages_handles_string_input():
    """String input should be converted to a user message."""
    payload = {"model": "dummy", "input": "Hello!"}
    messages = BaseInferenceApiServer._responses_payload_to_messages(ResponsesRequest.model_validate(payload))
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello!"


def test_responses_payload_to_messages_handles_system_instructions():
    """System instructions should be converted to a system message when include_instructions=True."""
    payload = {
        "model": "dummy",
        "input": "Hello!",
        "instructions": "You are helpful.",
    }
    request = ResponsesRequest.model_validate(payload)
    messages_default = BaseInferenceApiServer._responses_payload_to_messages(request)
    assert len(messages_default) == 1
    assert messages_default[0].role == "user"

    messages_with = BaseInferenceApiServer._responses_payload_to_messages(request, include_instructions=True)
    assert len(messages_with) == 2
    assert messages_with[0].role == "system"
    assert messages_with[0].content == "You are helpful."
    assert messages_with[1].role == "user"


def test_lru_set_respects_max_size():
    """LRU store should evict oldest entries when exceeding max_size."""
    from collections import OrderedDict

    store = OrderedDict()
    BaseInferenceApiServer._lru_set(store, "a", 1, max_size=2)
    BaseInferenceApiServer._lru_set(store, "b", 2, max_size=2)
    assert len(store) == 2
    BaseInferenceApiServer._lru_set(store, "c", 3, max_size=2)
    assert len(store) == 2
    assert "a" not in store
    assert "c" in store


def test_lru_set_updates_existing_key():
    """Updating an existing key should move it to the end (most recent)."""
    from collections import OrderedDict

    store = OrderedDict()
    BaseInferenceApiServer._lru_set(store, "a", 1, max_size=3)
    BaseInferenceApiServer._lru_set(store, "b", 2, max_size=3)
    BaseInferenceApiServer._lru_set(store, "a", 10, max_size=3)  # Update a
    BaseInferenceApiServer._lru_set(store, "c", 3, max_size=3)
    BaseInferenceApiServer._lru_set(store, "d", 4, max_size=3)
    assert "b" not in store
    assert "a" in store
    assert store["a"] == 10
