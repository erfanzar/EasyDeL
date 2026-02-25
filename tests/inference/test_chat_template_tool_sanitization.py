from easydel.inference.esurge.mixins.utils import EngineUtilsMixin


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
