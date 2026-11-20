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

"""A remote handle (the dp front's view of a replica) runs the tool and reasoning parsers the engine would.

Pure Python: no workers, networking or array computation.
"""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import remote_engine, wire
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


def _offline_handle(monkeypatch, **kwargs):
    """A handle with the transport, workers and output pipeline stubbed; admission is real."""

    class Tokenizer:
        eos_token_id = None
        chat_template = None

        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) for c in text]}

        def decode(self, ids, **kw):
            return "".join(chr(t) for t in ids)

        def get_vocab(self):
            return {}

    tokenizer = Tokenizer()
    admitted = []

    class Channel:
        alive = True
        queue_stats = wire.QueueStats()
        engine_spec = wire.EngineSpec(
            esurge_name="m", max_model_len=4096, max_num_seqs=4, reserve_tokens=1, tokenizer_source="t", page_size=16
        )

        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def set_plane_handler(self, handler):
            self.handler = handler

        def send_to_leader(self, message, payload=None):
            admitted.append(message.request_id)
            self.handler(wire.AdmitAck(request_id=message.request_id, ok=True), None)
            handle._registry.outputs[message.request_id].finished = True
            handle._registry.signal(message.request_id)

    class Workers:
        def __init__(self, *a):
            pass

        def start(self, **kw):
            return SimpleNamespace(tokenize=lambda rid, prompt: tokenizer(prompt)["input_ids"]), SimpleNamespace()

        def shutdown(self):
            pass

    class Pipeline:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(remote_engine, "_ClientChannel", Channel)
    monkeypatch.setattr(remote_engine, "WorkerManager", Workers)
    monkeypatch.setattr(remote_engine, "OutputPipeline", Pipeline)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.jax.process_count", lambda: 1)
    handle = remote_engine.RemoteEngineHandle("unused", 1234, auth_token="test", **kwargs)
    return handle, admitted


class _ToolParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _ReasoningParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def adjust_request(self, sampling_params):
        return sampling_params


def test_handle_runs_the_configured_parsers_and_hands_the_tool_request_to_them(monkeypatch):
    from easydel.inference.reasoning import ReasoningParserManager
    from easydel.inference.tools import ToolParserManager

    monkeypatch.setattr(ToolParserManager, "get_tool_parser", classmethod(lambda cls, name: _ToolParser))
    monkeypatch.setattr(ReasoningParserManager, "get_reasoning_parser", classmethod(lambda cls, name: _ReasoningParser))
    handle, _admitted = _offline_handle(monkeypatch, tool_parser="glm47", reasoning_parser="glm45")
    try:
        assert handle.tool_parser == "glm47" and handle.reasoning_parser_name == "glm45"
        seen = {}
        original = handle._admission.add_request

        def spy(rid, prompt, params, **kw):
            seen[rid] = kw.get("tool_parser_request")
            return original(rid, prompt, params, **kw)

        monkeypatch.setattr(handle._admission, "add_request", spy)
        monkeypatch.setattr(
            "easydel.inference.esurge.engine.chat_templating.format_chat_prompt", lambda *a, **kw: "prompt"
        )
        handle.chat(
            [{"role": "user", "content": "weather?"}],
            tools=TOOLS,
            tool_choice="auto",
            sampling_params=SamplingParams(max_tokens=1),
            request_id="r1",
        )
        request = seen["r1"]
        assert request is not None and request.tools[0].function.name == "get_weather"
        parser = handle._registry.records["r1"].delegating_parser
        assert isinstance(parser.tool_parser, _ToolParser)
        assert isinstance(parser.reasoning_parser, _ReasoningParser)
    finally:
        handle.close()


def test_handle_without_parser_names_auto_detects_from_model_type(monkeypatch):
    calls = {}

    def fake_tool(*, tokenizer, model_type):
        calls["tool"] = model_type
        return None

    def fake_reasoning(*, tokenizer, model_type):
        calls["reasoning"] = model_type
        return None

    monkeypatch.setattr("easydel.inference.esurge.engine.bootstrap.auto_detect_tool_parser", fake_tool)
    monkeypatch.setattr("easydel.inference.esurge.engine.bootstrap.auto_detect_reasoning_parser_name", fake_reasoning)
    handle, _ = _offline_handle(monkeypatch, model_type="glm5_next_text")
    try:
        assert calls == {"tool": "glm5_next_text", "reasoning": "glm5_next_text"}
        assert handle.tool_parser is None and handle.reasoning_parser_name is None
    finally:
        handle.close()


def test_handle_told_nothing_about_the_model_runs_no_parsers(monkeypatch):
    handle, _ = _offline_handle(monkeypatch)
    try:
        assert handle.tool_parser is None and handle.reasoning_parser_name is None
    finally:
        handle.close()


def test_unknown_parser_names_disable_parsing_instead_of_failing():
    name, cls, rname, rcls = remote_engine.resolve_handle_parsers(
        tokenizer=object(), tool_parser="no-such-parser", reasoning_parser="no-such-reasoning", model_type=None
    )
    assert (name, cls, rname, rcls) == (None, None, None, None)
