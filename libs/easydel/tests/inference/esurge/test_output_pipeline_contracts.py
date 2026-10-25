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

"""Host-only regressions for parser IDs, protocol tokens and monitoring ownership."""

import json
import socket
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from easydel.inference.esurge import monitoring
from easydel.inference.esurge.engine.monitoring_stack import MonitoringStack
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.parsing.delegating_parser import DelegatingParser
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.tools.parsers.hermes_tool_parser import HermesToolParser

pytestmark = pytest.mark.cpu_ok


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)

    def get_vocab(self):
        return {chr(value): value for value in range(128)}


class PublicParserView:
    """Expose the parser's public API without any private parser state."""

    __slots__ = ("delegate",)

    def __init__(self, delegate):
        self.delegate = delegate

    @property
    def reasoning_parser(self):
        return self.delegate.reasoning_parser

    def process_delta(self, *args):
        return self.delegate.process_delta(*args)

    def process_final(self, *args):
        return self.delegate.process_final(*args)

    def streamed_tool_call_id(self, index):
        return self.delegate.streamed_tool_call_id(index)


@pytest.mark.parametrize("call_count", [1, 2])
@pytest.mark.parametrize("stream_first", [False, True])
def test_output_pipeline_preserves_public_streamed_ids_through_finalization(call_count, stream_first):
    tokenizer = CharacterTokenizer()
    parser = PublicParserView(DelegatingParser(tool_parser=HermesToolParser(tokenizer)))
    text = "".join(
        "<tool_call>" + json.dumps({"name": f"lookup_{index}", "arguments": {"id": index}}) + "</tool_call>"
        for index in range(call_count)
    )
    tokens = tokenizer.encode(text)
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="request",
        prompt="hello",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    registry.register(
        "request",
        RequestRecord(
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=512, skip_special_tokens=False),
            delegating_parser=parser,
        ),
        output,
    )
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=SimpleNamespace(
            decode=lambda request_id, tokens, **kwargs: SimpleNamespace(
                accumulated_text=tokenizer.decode(tokens),
                delta_text=tokenizer.decode(tokens),
                last_decoded_index=len(tokens),
            ),
            reset=lambda request_id: None,
        ),
        eos_token_ids=[],
        decode_interval_tokens=1 if stream_first else len(tokens) + 1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda matches: None,
        on_activity=lambda: None,
        on_fatal=lambda *args: pytest.fail("Output pipeline unexpectedly failed"),
    )
    pipeline.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("request", tokens)], timestamp=1.0)})
    streamed_ids = {delta.index: delta.id for delta in output.delta_tool_calls or [] if delta.id}
    assert len(streamed_ids) == (call_count if stream_first else 0)
    pipeline.submit(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("request", [], finish_reason=FinishReason.STOP)], timestamp=2.0)}
    )
    assert output.finished
    assert output.outputs[0].finish_reason == "tool_calls"
    assert len(output.tool_calls) == call_count
    assert [call.function.name for call in output.tool_calls] == [f"lookup_{index}" for index in range(call_count)]
    final_ids = [call.id for call in output.tool_calls]
    assert all(final_ids)
    assert len(set(final_ids)) == call_count
    for index, streamed_id in streamed_ids.items():
        assert final_ids[index] == streamed_id
    if not stream_first:
        assert {delta.index: delta.id for delta in output.delta_tool_calls if delta.id} == dict(enumerate(final_ids))


@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "lookup"}}]])
def test_kimi_protocol_markers_are_suppressed_only_when_tools_are_disabled(tools):
    markers = (
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
    )
    ordinary = ("tool", "stool", "toolbar", "<toolbar>", "tool_calls_section_begin", "<toolbox>")
    vocab = {token: index for index, token in enumerate((*markers, *ordinary))}
    from tests.inference.esurge.test_engine_stub_behaviors import _ChatStub

    engine = _ChatStub(SimpleNamespace(get_vocab=lambda: vocab))
    params = engine.chat([{"role": "user", "content": "hello"}], tools=tools)
    expected = {vocab[marker]: -100.0 for marker in markers} if not tools else None
    assert params.logit_bias == expected
    assert params.skip_special_tokens is (not bool(tools))


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scrape(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        return response.read().decode()


@pytest.fixture
def clean_monitoring():
    if not monitoring.PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client not installed")
    monitoring.stop_monitoring()
    try:
        yield
    finally:
        monitoring.stop_monitoring()


def test_disabled_stack_cannot_stop_unrelated_global_exporter(clean_monitoring):
    port = _free_port()
    server = monitoring.start_monitoring_server(prometheus_port=port)
    stack = MonitoringStack(info=lambda *args: None)
    stack.start_monitoring(enable_prometheus=False, enable_console=False, start_grafana=False)
    stack.stop_monitoring()
    assert server.running
    assert "esurge_" in _scrape(port)
    assert monitoring.start_monitoring_server(prometheus_port=port) is server


def test_stack_releases_only_its_lease_on_preexisting_exporter(clean_monitoring):
    port = _free_port()
    server = monitoring.start_monitoring_server(prometheus_port=port)
    stack = MonitoringStack(info=lambda *args: None)
    stack.start_monitoring(prometheus_port=port, enable_console=False, start_grafana=False)
    stack.stop_monitoring()
    assert server.running
    assert "esurge_" in _scrape(port)


def test_second_stack_keeps_shared_exporter_until_last_owner_stops_and_can_restart(clean_monitoring):
    port = _free_port()
    first = MonitoringStack(info=lambda *args: None)
    second = MonitoringStack(info=lambda *args: None)
    try:
        for stack in (first, second):
            stack.start_monitoring(prometheus_port=port, enable_console=False, start_grafana=False)
        first.stop_monitoring()
        assert "esurge_" in _scrape(port)
        second.stop_monitoring()
        with pytest.raises(urllib.error.URLError):
            _scrape(port)
        first.start_monitoring(prometheus_port=port, enable_console=False, start_grafana=False)
        assert "esurge_" in _scrape(port)
    finally:
        first.stop_monitoring()
        second.stop_monitoring()


def test_console_and_exporter_leases_are_independent(clean_monitoring, monkeypatch):
    class Console:
        def __init__(self, **kwargs):
            self.running = False

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    monkeypatch.setattr(monitoring, "RichConsoleMonitor", Console)
    external_console = monitoring.start_console_monitor()
    console_stack = MonitoringStack(info=lambda *args: None)
    exporter_stack = MonitoringStack(info=lambda *args: None)
    try:
        console_stack.start_monitoring(enable_prometheus=False, enable_console=True, start_grafana=False)
        port = _free_port()
        exporter_stack.start_monitoring(prometheus_port=port, enable_console=False, start_grafana=False)
        exporter_stack.stop_monitoring()
        assert external_console.running
        console_stack.stop_monitoring()
        assert external_console.running
        monitoring.stop_monitoring()
        assert not external_console.running
    finally:
        console_stack.stop_monitoring()
        exporter_stack.stop_monitoring()
