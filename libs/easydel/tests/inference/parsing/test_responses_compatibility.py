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

"""Public API compatibility and terminal parser regressions; no array computation."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from easydel.inference.inference_engine_interface import BaseInferenceApiServer
from easydel.inference.oai_proxies import InferenceApiRouter
from easydel.inference.openai_api_modules import ResponsesRequest
from easydel.inference.parsing import DelegatingParser
from easydel.inference.reasoning import ReasoningMixin, abstract_reasoning
from easydel.inference.reasoning.parsers import GraniteReasoningParser
from easydel.inference.sampling_funcs import dynamic_sample_tokens, vmaped_sample_top_p_efficient
from easydel.inference.sampling_params import BeamSearchParams
from easydel.utils import _parser_utils

pytestmark = pytest.mark.cpu_ok


def test_responses_preserve_sdk_request_controls_outside_json_body():
    router = object.__new__(InferenceApiRouter)
    request = ResponsesRequest(
        model="m",
        input="hi",
        extra_headers={"X-Trace": "trace"},
        extra_query={"api-version": "preview"},
        timeout=13.5,
        extra_body={"vendor_flag": True},
        top_k=7,
    )
    params = router.build_oai_params_from_responses_request(request)
    assert params["extra_headers"] == {"X-Trace": "trace"}
    assert params["extra_query"] == {"api-version": "preview"}
    assert params["timeout"] == 13.5
    assert params["extra_body"] == {"vendor_flag": True, "top_k": 7}


@pytest.mark.parametrize(
    "suffix",
    ["H", "Here", "Here's my res", "Here is my response", "Here's my thought process", "Here is my thought pro"],
)
@pytest.mark.parametrize("channel", ["content", "reasoning", "post_reasoning"])
@pytest.mark.parametrize("finish", ["empty_final_snapshot", "final_flag"])
def test_public_mixin_flushes_terminal_prefixes_once_in_the_correct_channel(suffix, channel, finish):
    mixin = ReasoningMixin()
    mixin.reasoning_parsers = {"m": GraniteReasoningParser(object())}
    if channel == "reasoning":
        text = "Here's my thought process:reason " + suffix
        expected = ("reason " + suffix, "")
    elif channel == "post_reasoning":
        text = "Here's my thought process:reasonHere's my response:Answer " + suffix
        expected = ("reason", "Answer " + suffix)
    else:
        text = "Answer " + suffix
        expected = ("", text)
    reasoning, content = "", ""
    for index, char in enumerate(text):
        delta = mixin.extract_reasoning_streaming(
            "m",
            text[:index],
            text[: index + 1],
            char,
            final=finish == "final_flag" and index == len(text) - 1,
        )
        if delta:
            reasoning += delta.reasoning_content or ""
            content += delta.content or ""
    tail = mixin.extract_reasoning_streaming("m", text, text, "", final=True)
    if tail:
        reasoning += tail.reasoning_content or ""
        content += tail.content or ""
    assert (reasoning, content) == expected
    assert mixin.extract_reasoning_streaming("m", text, text, "", final=True) is None


@pytest.mark.parametrize(
    "suffix",
    ["H", "Here", "Here's my res", "Here is my response", "Here's my thought process", "Here is my thought pro"],
)
@pytest.mark.parametrize("channel", ["content", "reasoning", "post_reasoning"])
@pytest.mark.parametrize("stream_last", [False, True])
def test_delegating_final_reconciles_prefixes_and_final_unstreamed_text(suffix, channel, stream_last):
    parser = DelegatingParser(reasoning_parser=GraniteReasoningParser(object()))
    if channel == "reasoning":
        text = "Here's my thought process:\nreason " + suffix
        expected = ("\nreason " + suffix, "")
    elif channel == "post_reasoning":
        text = "Here's my thought process:reason\nHere's my response:\nAnswer " + suffix
        expected = ("reason\n", "\nAnswer " + suffix)
    else:
        text = "Answer " + suffix
        expected = ("", text)
    reasoning, content = "", ""
    streamed = text if stream_last else text[:-2]
    for index, char in enumerate(streamed):
        delta = parser.process_delta(streamed[: index + 1], char, [], streamed[:index], [])
        reasoning += delta.delta_reasoning or ""
        content += delta.delta_content or ""
    final = parser.process_final(text, [])
    reasoning += final.delta_reasoning or ""
    content += final.delta_content or ""
    assert (reasoning, content) == expected
    assert (final.accumulated_reasoning, final.accumulated_content) == expected
    repeated = parser.process_final(text, [])
    assert not repeated.delta_reasoning
    assert not repeated.delta_content


def test_delegating_still_buffers_delimiters_after_visible_preamble():
    parser = DelegatingParser(reasoning_parser=GraniteReasoningParser(object()))
    text = "Preamble.Here's my thought process:reasonHere is my response:answer"
    reasoning, content = "", ""
    for index, char in enumerate(text):
        delta = parser.process_delta(text[: index + 1], char, [], text[:index], [])
        reasoning += delta.delta_reasoning or ""
        content += delta.delta_content or ""
    final = parser.process_final(text, [])
    reasoning += final.delta_reasoning or ""
    content += final.delta_content or ""
    assert (reasoning, content) == ("reason", "Preamble.answer")


@pytest.mark.parametrize(
    "text", ["Here's my thought process:", "Here is my response:", "Here's my thought process:Here's my response:"]
)
def test_delegating_final_does_not_reemit_complete_delimiters(text):
    parser = DelegatingParser(reasoning_parser=GraniteReasoningParser(object()))
    for index, char in enumerate(text):
        delta = parser.process_delta(text[: index + 1], char, [], text[:index], [])
        assert not delta.delta_content
        assert not delta.delta_reasoning
    final = parser.process_final(text, [])
    assert not final.delta_content
    assert not final.delta_reasoning
    assert final.accumulated_content == ""
    assert final.accumulated_reasoning == ""


def test_reasoning_helpers_alias_shared_implementations_and_preserve_plugin_loading(tmp_path, monkeypatch):
    name = "responses_compat_plugin"
    monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(abstract_reasoning.ReasoningParserManager, "reasoning_parsers", {})
    plugin = tmp_path / (name + ".py")
    plugin.write_text(
        "from easydel.inference.reasoning.abstract_reasoning import ReasoningParserManager\n"
        "from easydel.inference.reasoning.basic_parsers import BaseThinkingReasoningParser\n"
        "@ReasoningParserManager.register_module('responses_compat_parser')\n"
        "class Plugin(BaseThinkingReasoningParser):\n"
        "    start_token = '[reason]'\n"
        "    end_token = '[/reason]'\n"
    )
    abstract_reasoning.ReasoningParserManager.import_reasoning_parser(str(plugin))
    parser_cls = abstract_reasoning.ReasoningParserManager.get_reasoning_parser("responses_compat_parser")
    assert parser_cls(object()).extract_reasoning("[reason]secret[/reason]answer") == ("secret", "answer")
    assert sys.modules[name].Plugin is parser_cls
    with pytest.raises(FileNotFoundError):
        _parser_utils.import_from_path(name, tmp_path / "missing.py")


@pytest.mark.parametrize(
    ("value", "check", "expected"),
    [
        ([], "first", True),
        ([], "all", True),
        (["a", 1], "first", True),
        (["a", 1], "all", False),
        ([1], "first", False),
        (("a",), "all", False),
    ],
)
def test_reasoning_shared_list_checker_preserves_compatibility(value, check, expected):
    assert _parser_utils.is_list_of(value, str, check=check) is expected


def test_reasoning_shared_list_checker_preserves_invalid_check_exception():
    with pytest.raises(AssertionError):
        _parser_utils.is_list_of([], str, check="invalid")


def test_beam_search_configuration_preserves_fields_defaults_and_roundtrip():
    default = BeamSearchParams(beam_width=4, max_tokens=100)
    assert dataclasses.asdict(default) == {
        "beam_width": 4,
        "max_tokens": 100,
        "ignore_eos": False,
        "temperature": 0.0,
        "length_penalty": 1.0,
        "include_stop_str_in_output": False,
    }
    custom = BeamSearchParams(4, 100, True, 0.7, 1.2, True)
    assert BeamSearchParams(**dataclasses.asdict(custom)) == custom


def test_public_sampling_entry_points_preserve_argument_names_without_running_numerics():
    assert list(inspect.signature(dynamic_sample_tokens).parameters) == [
        "tokens",
        "length",
        "logits",
        "top_p",
        "temperature",
        "random_sampling",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "rngs",
    ]
    assert list(inspect.signature(vmaped_sample_top_p_efficient).parameters) == [
        "logits",
        "top_p",
        "temperature",
        "rng",
        "top_k_for_computation",
    ]


class _HelperServer(BaseInferenceApiServer):
    """Minimal non-generating server to exercise inherited extension contracts."""

    def __init__(self, executor):
        self.thread_pool = executor
        self.tokenization_threads = []

    async def chat_completions(self, *args, **kwargs):
        raise NotImplementedError

    completions = chat_completions
    health_check = chat_completions
    get_metrics = chat_completions
    list_models = chat_completions
    get_model = chat_completions
    list_tools = chat_completions
    execute_tool = chat_completions

    def _create_sampling_params(self, request):
        raise NotImplementedError

    def _count_tokens(self, content, model_name=None):
        self.tokenization_threads.append(threading.get_ident())
        if model_name == "bad":
            raise ValueError("unknown tokenizer")
        return len(content.split()) + (1 if model_name == "with_bos" else 0)


def test_async_token_count_offloads_to_worker_and_preserves_model_and_errors():
    with ThreadPoolExecutor(max_workers=1) as pool:
        server = _HelperServer(pool)

        async def run():
            event_loop_thread = threading.get_ident()
            count = await server._count_tokens_async("two words", "with_bos")
            assert count == 3
            assert server.tokenization_threads == [pool.submit(threading.get_ident).result()]
            assert server.tokenization_threads[0] != event_loop_thread
            with pytest.raises(ValueError, match="unknown tokenizer"):
                await server._count_tokens_async("hello", "bad")

        asyncio.run(run())


def test_finish_reason_helper_retains_budget_rule_and_subclass_override():
    with ThreadPoolExecutor(max_workers=1) as pool:
        server = _HelperServer(pool)
        assert server._determine_finish_reason(4, 4, "answer") == "length"
        assert server._determine_finish_reason(3, 4, "answer") == "stop"

        class ToolServer(_HelperServer):
            def _determine_finish_reason(self, tokens_generated, max_tokens, text):
                return (
                    "tool_calls"
                    if text == "call"
                    else super()._determine_finish_reason(tokens_generated, max_tokens, text)
                )

        assert ToolServer(pool)._determine_finish_reason(1, 4, "call") == "tool_calls"
