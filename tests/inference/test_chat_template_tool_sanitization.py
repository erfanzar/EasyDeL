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

from easydel.inference.esurge.mixins.io import EngineIOMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.sampling_params import SamplingParams


class _DummyTokenizer:
    def __init__(self):
        self.tools_calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.tools_calls.append(tools)
        if tools is not None:
            # Mimic tokenizer internals that expect dict entries.
            for tool in tools:
                _ = tool.items()
        return "PROMPT"


class _DummyEngine(EngineUtilsMixin):
    def __init__(self):
        self.tokenizer = _DummyTokenizer()


class _ChatCaptureEngine(EngineIOMixin, EngineUtilsMixin):
    def __init__(self):
        self.tokenizer = _ToolAwareTokenizer()
        self.generate_calls = []

    def _messages_have_multimodal_content(self, _messages):
        return False

    def _format_chat_prompt(self, *args, **kwargs):
        del args, kwargs
        return "PROMPT"

    def _build_tool_parser_request(self, *args, **kwargs):
        del args, kwargs
        return None

    def generate(self, prompt, sampling_params=None, request_id=None, use_tqdm=False, tool_parser_request=None):
        self.generate_calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
                "use_tqdm": use_tqdm,
                "tool_parser_request": tool_parser_request,
            }
        )
        return [object()]


def test_format_chat_prompt_retries_with_sanitized_tools_on_items_error():
    engine = _DummyEngine()
    prompt = engine._format_chat_prompt(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "valid"}, "broken-tool-entry"],
    )
    assert prompt == "PROMPT"
    assert len(engine.tokenizer.tools_calls) >= 1
    final_tools = engine.tokenizer.tools_calls[-1]
    assert len(final_tools) == 1
    assert final_tools[0]["name"] == "valid"
    assert final_tools[0]["parameters"] == {}


class _NestedItemsTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        if tools:
            for tool in tools:
                parameters = tool.get("parameters", {})
                # Simulate templates that require mapping fields.
                _ = parameters.items()
                properties = parameters.get("properties", {})
                _ = properties.items()
        return "PROMPT"


class _StrictMessageTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        for msg in messages:
            content = msg.get("content")
            # Simulate templates that iterate over content parts and call .items().
            for part in content:
                _ = part.items()
        return "PROMPT"


def test_format_chat_prompt_normalizes_stringified_tool_schemas():
    engine = _DummyEngine()
    engine.tokenizer = _NestedItemsTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "name": "read_file",
                "parameters": '{"type":"object","properties":"{\\"path\\":{\\"type\\":\\"string\\"}}"}',
            }
        ],
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) >= 1
    _, used_tools = engine.tokenizer.calls[-1]
    assert used_tools is not None
    assert isinstance(used_tools[0]["parameters"], dict)
    assert isinstance(used_tools[0]["parameters"].get("properties"), dict)


def test_format_chat_prompt_retries_with_structured_messages_when_template_expects_parts():
    engine = _DummyEngine()
    engine.tokenizer = _StrictMessageTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[{"role": "user", "content": "hello world"}],
        tools=[{"name": "noop", "parameters": {"type": "object", "properties": {}}}],
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) >= 2
    retry_messages, _ = engine.tokenizer.calls[-1]
    assert isinstance(retry_messages[0]["content"], list)
    assert retry_messages[0]["content"][0]["type"] == "text"


class _ToolCallArgsItemsTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        for msg in messages:
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                function_payload = call.get("function", {})
                arguments = function_payload.get("arguments", {})
                _ = arguments.items()
        return "PROMPT"


def test_format_chat_prompt_normalizes_tool_call_arguments_in_messages():
    engine = _DummyEngine()
    engine.tokenizer = _ToolCallArgsItemsTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"/tmp/a.txt"}'},
                    }
                ],
            }
        ],
        tools=None,
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) == 1
    used_messages, _ = engine.tokenizer.calls[0]
    arguments = used_messages[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["path"] == "/tmp/a.txt"


class _FunctionWrappedToolsTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        if tools:
            for tool in tools:
                _ = tool["function"]["name"]
                _ = tool["function"]["parameters"].items()
        return "PROMPT"


def test_format_chat_prompt_retries_with_wrapped_tools_when_template_expects_function_wrapper():
    engine = _DummyEngine()
    engine.tokenizer = _FunctionWrappedToolsTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "name": "lookup",
                "description": "Find a result",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) >= 2
    _, final_tools = engine.tokenizer.calls[-1]
    assert final_tools is not None
    assert final_tools[0]["type"] == "function"
    assert final_tools[0]["function"]["name"] == "lookup"


class _NoNoneContentTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        for msg in messages:
            assert msg.get("content") is not None
        return "PROMPT"


def test_format_chat_prompt_normalizes_none_content_for_tool_only_messages():
    engine = _DummyEngine()
    engine.tokenizer = _NoNoneContentTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"abc"}'},
                    }
                ],
            }
        ],
        tools=None,
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) == 1
    used_messages, _ = engine.tokenizer.calls[0]
    assert used_messages[0]["content"] == ""


class _StrictSystemPositionTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        tools=None,
        add_generation_prompt=True,
        chat_template=None,
        **_kwargs,
    ):
        self.calls.append((messages, tools))
        seen_non_system = False
        for msg in messages:
            if msg.get("role") == "system" and seen_non_system:
                raise RuntimeError("System message must be at the beginning.")
            if msg.get("role") != "system":
                seen_non_system = True
        return "PROMPT"


def test_format_chat_prompt_collapses_late_system_messages_to_front():
    engine = _DummyEngine()
    engine.tokenizer = _StrictSystemPositionTokenizer()

    prompt = engine._format_chat_prompt(
        messages=[
            {"role": "system", "content": "base instructions"},
            {"role": "user", "content": "first turn"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"abc"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "system", "content": "follow the schema strictly"},
        ],
        tools=None,
    )

    assert prompt == "PROMPT"
    assert len(engine.tokenizer.calls) == 1
    used_messages, _ = engine.tokenizer.calls[0]
    assert used_messages[0]["role"] == "system"
    assert used_messages[0]["content"] == "base instructions\n\nfollow the schema strictly"
    assert [msg["role"] for msg in used_messages[1:]] == ["user", "assistant", "tool"]


class _ToolAwareTokenizer(_DummyTokenizer):
    def get_vocab(self):
        return {
            "<|tool_call>": 48,
            "<tool_call|>": 49,
            "<|tool>": 50,
            "<tool|>": 51,
            "<|tool_response>": 52,
            "<tool_response|>": 53,
        }


def test_prepare_chat_sampling_params_hides_special_tokens_for_plain_chat():
    engine = _DummyEngine()
    engine.tokenizer = _ToolAwareTokenizer()

    params = engine._prepare_chat_sampling_params(
        None,
        tools=None,
        tool_choice=None,
    )

    assert params.skip_special_tokens is True
    assert params.logit_bias is not None
    assert params.logit_bias[48] == -100.0
    assert params.logit_bias[52] == -100.0


def test_prepare_chat_sampling_params_preserves_tool_protocol_when_tools_enabled():
    engine = _DummyEngine()
    engine.tokenizer = _ToolAwareTokenizer()

    params = engine._prepare_chat_sampling_params(
        SamplingParams(max_tokens=32, skip_special_tokens=True),
        tools=[{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        tool_choice="auto",
    )

    assert params.skip_special_tokens is False
    assert params.logit_bias is None


def test_chat_uses_prepared_sampling_params_for_plain_chat():
    engine = _ChatCaptureEngine()

    engine.chat(
        [{"role": "user", "content": "write fibonacci in c"}],
        sampling_params=SamplingParams(max_tokens=32),
    )

    assert len(engine.generate_calls) == 1
    used_params = engine.generate_calls[0]["sampling_params"]
    assert used_params.skip_special_tokens is True
    assert used_params.logit_bias is not None
    assert used_params.logit_bias[48] == -100.0
    assert used_params.logit_bias[52] == -100.0
