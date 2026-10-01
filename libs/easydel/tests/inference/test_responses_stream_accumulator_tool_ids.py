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

"""Regression tests for ``ResponsesStreamAccumulator`` tool-call item keying.

Tool-call streaming deltas carry the tool id only on the first chunk while
final batch-extracted ``ToolCall`` objects mint fresh ids; the accumulator
must key function-call state positionally so one tool call maps to exactly
one Responses output item.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.inference.openai_api_modules import DeltaFunctionCall, DeltaToolCall, ToolCall
from easydel.inference.stream_protocol import ResponsesStreamAccumulator


@dataclass
class _FakeOutput:
    index: int = 0
    text: str = ""
    reasoning_content: str | None = None
    tool_calls: list | None = None
    token_ids: list = field(default_factory=list)


@dataclass
class _FakeRequestOutput:
    outputs: list
    delta_text: str = ""
    delta_reasoning_content: str | None = None
    delta_tool_calls: list | None = None
    reasoning_content: str | None = None
    accumulated_text: str = ""
    tool_calls: list | None = None
    prompt_token_ids: list = field(default_factory=list)
    finished: bool = False
    num_generated_tokens: int | None = None
    usage: object | None = None
    finish_reason: str | None = None


def _function_items(response):
    return [item for item in response.output if getattr(item, "type", None) == "function_call"]


def test_finalize_reuses_streamed_item_when_batch_id_differs():
    """R2PA-08: a fresh batch ToolCall.id must not duplicate a streamed item."""
    accumulator = ResponsesStreamAccumulator(response_id="resp1", model="m")

    accumulator.add_output(
        _FakeRequestOutput(
            outputs=[_FakeOutput()],
            delta_tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="chatcmpl-tool-streamed",
                    type="function",
                    function=DeltaFunctionCall(name="get_weather").model_dump(exclude_none=True),
                )
            ],
        )
    )
    accumulator.add_output(
        _FakeRequestOutput(
            outputs=[_FakeOutput()],
            delta_tool_calls=[
                DeltaToolCall(
                    index=0,
                    function=DeltaFunctionCall(arguments='{"location": "Paris"}').model_dump(exclude_none=True),
                )
            ],
        )
    )

    frames, final_response = accumulator.finalize(
        last_output=_FakeRequestOutput(
            outputs=[_FakeOutput()],
            tool_calls=[
                ToolCall(type="function", function={"name": "get_weather", "arguments": '{"location": "Paris"}'})
            ],
            finished=True,
        ),
        prompt_tokens=1,
    )

    items = _function_items(final_response)
    assert len(items) == 1, f"expected one function item, got {len(items)}"
    item = items[0]
    assert item.call_id == "chatcmpl-tool-streamed"
    assert item.name == "get_weather"
    assert item.arguments == '{"location": "Paris"}'
    assert not [frame for frame in frames if frame.event == "response.output_item.added"], (
        "finalize must not add a duplicate output item"
    )


def test_streaming_deltas_after_first_chunk_stay_on_one_item():
    """Deltas without an id (the OpenAI shape) extend the same item as the first."""
    accumulator = ResponsesStreamAccumulator(response_id="resp2", model="m")
    accumulator.add_output(
        _FakeRequestOutput(
            outputs=[_FakeOutput()],
            delta_tool_calls=[
                DeltaToolCall(
                    index=0,
                    id="chatcmpl-tool-x",
                    type="function",
                    function=DeltaFunctionCall(name="get_weather").model_dump(exclude_none=True),
                )
            ],
        )
    )
    accumulator.add_output(
        _FakeRequestOutput(
            outputs=[_FakeOutput()],
            delta_tool_calls=[
                DeltaToolCall(
                    index=0, function=DeltaFunctionCall(arguments='{"location":').model_dump(exclude_none=True)
                )
            ],
        )
    )
    accumulator.add_output(
        _FakeRequestOutput(
            outputs=[_FakeOutput()],
            delta_tool_calls=[
                DeltaToolCall(index=0, function=DeltaFunctionCall(arguments=' "Paris"}').model_dump(exclude_none=True))
            ],
        )
    )

    added_frames = [
        frame
        for frame in accumulator.add_output(
            _FakeRequestOutput(
                outputs=[_FakeOutput()],
                delta_tool_calls=[
                    DeltaToolCall(index=0, function=DeltaFunctionCall(arguments="!").model_dump(exclude_none=True))
                ],
            )
        )
        if frame.event == "response.output_item.added"
    ]
    assert not added_frames, "a continuation delta must not open a second item"
    assert len(accumulator.function_states) == 1
    assert len(accumulator.function_order) == 1


def test_finalize_batch_only_tools_still_creates_items():
    """No streamed deltas: finalize still creates one item per batch tool call."""
    accumulator = ResponsesStreamAccumulator(response_id="resp3", model="m")
    frames, final_response = accumulator.finalize(
        last_output=_FakeRequestOutput(
            outputs=[_FakeOutput()],
            tool_calls=[
                ToolCall(type="function", function={"name": "f1", "arguments": '{"a": 1}'}),
                ToolCall(type="function", function={"name": "f2", "arguments": '{"b": 2}'}),
            ],
            finished=True,
        ),
        prompt_tokens=3,
    )
    items = _function_items(final_response)
    assert [item.name for item in items] == ["f1", "f2"]
    assert [item.arguments for item in items] == ['{"a": 1}', '{"b": 2}']
    assert items[0].call_id and items[1].call_id
    assert len([frame for frame in frames if frame.event == "response.output_item.added"]) == 2
