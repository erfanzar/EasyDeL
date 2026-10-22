# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only public parser-ID and shared decorator contract regressions."""

import json

import pytest
from easydel.inference import logits_process
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from easydel.inference.parsing import DelegatingParser
from easydel.inference.tools.abstract_tool import ToolParser
from easydel.infra.utils import add_start_docstrings

pytestmark = pytest.mark.cpu_ok


class _FreshBatchIDToolParser(ToolParser):
    """Simulate independent streaming and batch extraction of multiple calls."""

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, **kwargs):
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=f"stream_{index}" if not previous_text else None,
                    type="function" if not previous_text else None,
                    function=DeltaFunctionCall(
                        name=f"tool_{index}" if not previous_text else None,
                        arguments="{" if not previous_text else "}",
                    ),
                )
                for index in range(2)
            ]
        )

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(id=f"batch_{index}", function=FunctionCall(name=f"tool_{index}", arguments="{}"))
                for index in range(3)
            ],
            content=None,
        )


def _parser():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "tool_0", "parameters": {"type": "object"}}}],
        }
    )
    return DelegatingParser(tool_parser=_FreshBatchIDToolParser(object()), tool_request=request)


@pytest.mark.parametrize("complete_stream", [False, True])
def test_streamed_ids_match_final_calls_and_catch_up_deltas(complete_stream):
    parser = _parser()
    assert parser.streamed_tool_call_id(0) is None
    assert parser.streamed_tool_call_id(-1) is None

    streamed = parser.process_delta("first", "first", [], "", [])
    deltas = list(streamed.delta_tool_calls)
    if complete_stream:
        deltas.extend(parser.process_delta("firstlast", "last", [], "first", []).delta_tool_calls)

    for index in range(2):
        assert parser.streamed_tool_call_id(index) == f"stream_{index}"
    assert parser.streamed_tool_call_id(2) is None

    final = parser.process_final("firstlast", [])
    assert [call.id for call in final.tool_calls] == ["stream_0", "stream_1", "batch_2"]
    deltas.extend(final.delta_tool_calls or [])
    for index, call in enumerate(final.tool_calls):
        assert parser.streamed_tool_call_id(index) == call.id
        announced_ids = [delta.id for delta in deltas if delta.index == index and delta.id]
        assert announced_ids == [call.id]
        arguments = "".join(delta.function.arguments or "" for delta in deltas if delta.index == index)
        assert json.loads(arguments) == json.loads(call.function.arguments) == {}
    assert parser.streamed_tool_call_id(99) is None


def test_docstring_decorator_has_one_implementation_at_both_public_paths():
    assert logits_process.add_start_docstrings is add_start_docstrings
    assert logits_process.LOGITS_PROCESSOR_INPUTS_DOCSTRING in logits_process.LogitsProcessor.__call__.__doc__


@pytest.mark.parametrize("decorator", [add_start_docstrings, logits_process.add_start_docstrings])
@pytest.mark.parametrize("existing_doc", [None, "", "Original documentation."])
def test_docstring_decorator_preserves_function_and_prepends_all_fragments(decorator, existing_doc):
    def function():
        return "result"

    function.__doc__ = existing_doc
    decorated = decorator("First. ", "Second. ")(function)
    assert decorated is function
    assert decorated.__doc__ == "First. Second. " + (existing_doc or "")
    assert decorated() == "result"


@pytest.mark.parametrize("decorator", [add_start_docstrings, logits_process.add_start_docstrings])
def test_docstring_decorator_accepts_no_fragments(decorator):
    def function():
        """Existing docstring."""

    assert decorator()(function) is function
    assert function.__doc__ == "Existing docstring."
