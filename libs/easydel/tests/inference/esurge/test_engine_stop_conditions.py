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

import threading
import time

from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.chat_templating import normalize_stop_sequences
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.utils import check_stop
from easydel.inference.parsing import DelegatingParser
from easydel.inference.reasoning.parsers import DeepSeekR1ReasoningParser
from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import DetokenizerResult


class _DetokenizerStub:
    def reset(self, request_id: str):
        return None


class _ScriptedDetokenizer(_DetokenizerStub):
    """Detokenizer-client stub returning a fixed decode result."""

    def __init__(self, decoded_text: str, delta_text: str):
        self._decoded_text = decoded_text
        self._delta_text = delta_text

    def decode(
        self,
        request_id,
        tokens,
        *,
        finished=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
        prompt_context=None,
    ):
        return DetokenizerResult(
            accumulated_text=self._decoded_text,
            delta_text=self._delta_text,
            last_decoded_index=len(tokens),
            finished=finished,
            detoktook=0.0,
        )


def _make_pipeline(
    decoded_text: str = "",
    delta_text: str = "",
    *,
    on_stop_strings=None,
    decode_interval_tokens: int = 1,
    decode_interval_secs: float = 0.0,
) -> OutputPipeline:
    """Build an OutputPipeline over a scripted detokenizer and inert callbacks."""
    return OutputPipeline(
        registry=RequestRegistry(),
        detokenizer_client=_ScriptedDetokenizer(decoded_text, delta_text),
        eos_token_ids=[],
        decode_interval_tokens=decode_interval_tokens,
        decode_interval_secs=decode_interval_secs,
        on_stop_strings=on_stop_strings if on_stop_strings is not None else (lambda stops: None),
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: None,
    )


def _make_admission(extra_stops=None, callback=None, generation_config=None, primary_eos_token_id=None):
    """Build a RequestAdmission with inert fakes for sampling-params tests."""
    return RequestAdmission(
        registry=RequestRegistry(),
        scheduler_submit=lambda requests: None,
        tokenizer_client=None,
        tokenizer=None,
        context_config=None,
        reserve_tokens=0,
        max_model_len=4096,
        tool_parser_class=None,
        reasoning_parser_class=None,
        sampling_params_callback=lambda: callback,
        generation_config_dict=generation_config or {},
        primary_eos_token_id=primary_eos_token_id,
        eos_token_ids=[],
        extra_stops=normalize_stop_sequences(extra_stops),
        ignore_stop_strings_in_reasoning=False,
        on_activity=lambda: None,
        info=lambda *args, **kwargs: None,
        callback_engine=None,
    )


class _DummyTokenizer:
    def __init__(self):
        self._vocab = {"<think>": 1, "</think>": 2}

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._vocab:
            return [self._vocab[text]]
        return [99]

    def decode(self, token_ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self._vocab.items()}
        return "".join(reverse.get(i, "") for i in token_ids)


def test_check_stop_with_custom_stop_token_id():
    sampling_params = SamplingParams(max_tokens=16, stop_token_ids=[42], ignore_eos=True)
    request = EngineRequest(
        request_id="req-stop-token",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        eos_token_id=0,
    )
    request.status = EngineRequestStatus.RUNNING
    request.append_output_token_ids(42)

    assert check_stop(request, max_model_len=4096) is True
    assert request.status == EngineRequestStatus.FINISHED_STOPPED
    assert request.stop_reason == 42


def test_check_stop_ignores_eos_when_ignore_eos_true():
    sampling_params = SamplingParams(max_tokens=16, stop_token_ids=[2], ignore_eos=True)
    request = EngineRequest(
        request_id="req-ignore-eos",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sampling_params,
        eos_token_id=2,
    )
    request.status = EngineRequestStatus.RUNNING
    request.append_output_token_ids(2)

    assert check_stop(request, max_model_len=4096) is False
    assert request.status == EngineRequestStatus.RUNNING


def test_stop_string_policy_trims_on_match():
    pipeline = _make_pipeline()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"])
    rd = RequestRecord(**{"sampling_params": sampling_params, "decoder_visible_text": "Hello "})

    visible_text, visible_delta, stop_triggered, stop_reason = pipeline._apply_stop_string_policy(
        rd,
        accumulated_text="Hello world<user>ignored",
        fallback_delta="world<user>ignored",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "Hello world"
    assert visible_delta == "world"


def test_stop_string_policy_passes_through_without_match():
    pipeline = _make_pipeline()
    sampling_params = SamplingParams(max_tokens=32, stop=["abcd"])
    rd = RequestRecord(**{"sampling_params": sampling_params, "decoder_visible_text": ""})

    visible_text, visible_delta, stop_triggered, stop_reason = pipeline._apply_stop_string_policy(
        rd,
        accumulated_text="abcx",
        fallback_delta="abcx",
    )

    assert stop_triggered is False
    assert stop_reason is None
    assert visible_text == "abcx"
    assert visible_delta == "abcx"


def test_stop_string_policy_can_include_stop_string_when_requested():
    pipeline = _make_pipeline()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"], include_stop_str_in_output=True)
    rd = RequestRecord(**{"sampling_params": sampling_params, "decoder_visible_text": ""})

    visible_text, visible_delta, stop_triggered, stop_reason = pipeline._apply_stop_string_policy(
        rd,
        accumulated_text="ans<user>tail",
        fallback_delta="ans<user>tail",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "ans<user>"
    assert visible_delta == "ans<user>"


def test_snapshot_delta_handles_empty_reset_without_fallback():
    delta = OutputPipeline._compute_snapshot_delta_text(
        current_text="",
        previous_text="tool markup before parser normalization",
        fallback_delta="",
    )

    assert delta == ""


def test_prepare_sampling_params_for_request_merges_engine_extra_stops():
    admission = _make_admission(extra_stops=["<user>", "DONE"])
    template = SamplingParams(max_tokens=64, stop=["DONE", "</assistant>"])

    prepared = admission.prepare_sampling_params_for_request(
        template,
        request_id="req-extra-stops",
        prompt="hello",
    )

    assert prepared.stop == ["DONE", "</assistant>", "<user>"]
    assert template.stop == ["DONE", "</assistant>"]


def test_prepare_sampling_params_for_request_applies_callback_then_extra_stops():
    def _callback(params: SamplingParams, _metadata):
        params.stop = ["CALLBACK_STOP"]
        return params

    admission = _make_admission(extra_stops="<user>", callback=_callback)
    template = SamplingParams(max_tokens=64, stop=["INITIAL"])

    prepared = admission.prepare_sampling_params_for_request(
        template,
        request_id="req-extra-stops-callback",
        prompt="hello",
    )

    assert prepared.stop == ["CALLBACK_STOP", "<user>"]


def test_prepare_sampling_params_for_request_merges_generation_config_eos_ids():
    admission = _make_admission(
        generation_config={"eos_token_id": [154820, 154827, 154829]},
        primary_eos_token_id=154820,
    )
    template = SamplingParams(max_tokens=64, stop_token_ids=[777])

    prepared = admission.prepare_sampling_params_for_request(
        template,
        request_id="req-generation-config-eos",
        prompt="hello",
    )

    assert set(prepared.stop_token_ids) == {777, 154827, 154829}
    assert prepared.all_stop_token_ids == {777, 154820, 154827, 154829}
    assert template.stop_token_ids == [777]


def test_prepare_sampling_params_respects_ignore_eos_for_generation_config_ids():
    admission = _make_admission(
        generation_config={"eos_token_id": [154820, 154827]},
        primary_eos_token_id=154820,
    )
    template = SamplingParams(max_tokens=64, stop_token_ids=[777], ignore_eos=True)

    prepared = admission.prepare_sampling_params_for_request(
        template,
        request_id="req-generation-config-ignore-eos",
        prompt="hello",
    )

    assert set(prepared.stop_token_ids) == {777}
    assert prepared.all_stop_token_ids == {777, 154820}


def test_process_engine_outputs_keeps_raw_text_before_reasoning_split():
    pipeline = _make_pipeline(
        decoded_text="<think>plan</think><tool_call>{}</tool_call>",
        delta_text="<think>plan</think><tool_call>{}</tool_call>",
    )
    reasoning_parser = DeepSeekR1ReasoningParser(_DummyTokenizer())
    request_id = "req-raw-before-parse"
    pipeline._active_requests[request_id] = RequestRecord(**{
        "parent_request_id": request_id,
        "sample_index": 0,
        "generated_tokens": [],
        "last_decoded_index": 0,
        "last_decode_time": 0.0,
        "start_time": time.perf_counter(),
        "first_token_time": None,
        "reported_generated_count": 0,
        "sampling_params": SamplingParams(max_tokens=16),
        "prompt_token_ids": [1, 2],
        "delegating_parser": DelegatingParser(reasoning_parser=reasoning_parser),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })
    pipeline._request_outputs[request_id] = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[[1, 2]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    pipeline._process_engine_outputs(
        {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=[11],
                    )
                ]
            )
        }
    )

    output = pipeline._request_outputs[request_id]
    completion = output.outputs[0]
    assert completion.raw_text == "<think>plan</think><tool_call>{}</tool_call>"
    assert output.raw_accumulated_text == "<think>plan</think><tool_call>{}</tool_call>"
    assert completion.reasoning_content == "plan"
    assert completion.text == "<tool_call>{}</tool_call>"


def test_process_engine_outputs_uses_engine_timestamp_for_generation_metrics():
    pipeline = _make_pipeline(decoded_text="xy", delta_text="y")
    request_id = "req-engine-timestamp-metrics"
    start_time = 100.0
    pipeline._active_requests[request_id] = RequestRecord(**{
        "parent_request_id": request_id,
        "sample_index": 0,
        "generated_tokens": [],
        "decodable_tokens": [],
        "last_decoded_index": 0,
        "last_decode_time": start_time,
        "start_time": start_time,
        "first_token_time": None,
        "reported_generated_count": 0,
        "sampling_params": SamplingParams(max_tokens=16),
        "prompt_token_ids": [1, 2],
        "delegating_parser": DelegatingParser(),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })
    pipeline._request_outputs[request_id] = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[[1, 2]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    pipeline._process_engine_outputs(
        {
            0: EngineCoreOutputs(
                outputs=[EngineCoreOutput(request_id=request_id, new_token_ids=[11])],
                timestamp=start_time + 1.0,
            )
        }
    )
    pipeline._process_engine_outputs(
        {
            0: EngineCoreOutputs(
                outputs=[EngineCoreOutput(request_id=request_id, new_token_ids=[12])],
                timestamp=start_time + 2.0,
            )
        }
    )

    output = pipeline._request_outputs[request_id]
    assert output.time_spent_generating == 2.0
    assert output.first_token_time == 1.0
    assert output.num_generated_tokens == 2
    assert output.tokens_per_second == 2.0


def test_process_engine_outputs_queues_parser_stop_without_scheduler_lock():
    queued_stops = []
    pipeline = _make_pipeline(
        decoded_text="hello STOP tail",
        delta_text="hello STOP tail",
        on_stop_strings=lambda stops: queued_stops.append(dict(stops)),
    )
    request_id = "req-parser-stop-queued"
    pipeline._active_requests[request_id] = RequestRecord(**{
        "parent_request_id": request_id,
        "sample_index": 0,
        "generated_tokens": [],
        "decodable_tokens": [],
        "last_decoded_index": 0,
        "last_decode_time": 0.0,
        "start_time": time.perf_counter(),
        "first_token_time": None,
        "reported_generated_count": 0,
        "sampling_params": SamplingParams(max_tokens=16, stop=["STOP"]),
        "prompt_token_ids": [1, 2],
        "delegating_parser": DelegatingParser(),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })
    pipeline._request_outputs[request_id] = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[[1, 2]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    pipeline._process_engine_outputs(
        {
            0: EngineCoreOutputs(
                outputs=[EngineCoreOutput(request_id=request_id, new_token_ids=[11])],
            )
        }
    )

    output = pipeline._request_outputs[request_id]
    assert queued_stops == [{request_id: "STOP"}]
    assert output.finished is True
    assert output.outputs[0].finish_reason == "stop"


def test_process_engine_outputs_marks_finished_requests_without_token_output():
    pipeline = _make_pipeline(decoded_text="", delta_text="")
    request_id = "req-finished-only"
    event = threading.Event()
    pipeline._request_events[request_id] = event
    pipeline._active_requests[request_id] = RequestRecord(**{
        "parent_request_id": request_id,
        "sample_index": 0,
        "generated_tokens": [],
        "last_decoded_index": 0,
        "last_decode_time": 0.0,
        "start_time": time.perf_counter(),
        "first_token_time": None,
        "reported_generated_count": 0,
        "sampling_params": SamplingParams(max_tokens=16),
        "prompt_token_ids": [1, 2],
        "delegating_parser": DelegatingParser(),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
    })
    pipeline._request_outputs[request_id] = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[[1, 2]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    pipeline._process_engine_outputs({0: EngineCoreOutputs(finished_requests={request_id})})

    output = pipeline._request_outputs[request_id]
    assert event.is_set()
    assert output.finished is True
    assert output.outputs[0].finish_reason == "abort"
    assert request_id not in pipeline._active_requests


def test_find_first_stop_string_picks_earliest_match():
    """Stop string matching should pick the earliest occurrence."""
    result = OutputPipeline._find_first_stop_string("hello\nworld\nstop", ["\nstop", "\nworld"])
    assert result is not None
    idx, stop = result
    assert stop == "\nworld"
    assert idx == 5  # position of first \n before "world"


def test_find_first_stop_string_prefers_longer_at_same_position():
    """When two stop strings match at the same position, prefer the longer one."""
    result = OutputPipeline._find_first_stop_string("hello\nworld", ["\n", "\nworld"])
    assert result is not None
    idx, stop = result
    assert stop == "\nworld"  # longer match at same position
    assert idx == 5


def test_find_first_stop_string_ignores_empty():
    """Empty stop strings should be skipped."""
    result = OutputPipeline._find_first_stop_string("hello world", ["", "world"])
    assert result is not None
    _idx, stop = result
    assert stop == "world"


def test_find_first_stop_string_returns_none_when_no_match():
    result = OutputPipeline._find_first_stop_string("hello world", ["xyz", "abc"])
    assert result is None


def test_apply_stop_string_policy_with_include_stop():
    """When include_stop_str_in_output=True, the stop string should be included."""
    pipeline = _make_pipeline()
    sp = SamplingParams(max_tokens=16, stop=["\nstop"])
    sp.include_stop_str_in_output = True
    rd = RequestRecord(**{"sampling_params": sp, "decoder_visible_text": ""})

    visible, _delta, stop_hit, stop_reason = pipeline._apply_stop_string_policy(
        rd, accumulated_text="hello\nstop world", fallback_delta="hello\nstop world"
    )
    assert stop_hit is True
    assert stop_reason == "\nstop"
    assert visible == "hello\nstop"  # includes the stop string


def test_apply_stop_string_policy_without_include_stop():
    """Default: stop string should NOT be included in output."""
    pipeline = _make_pipeline()
    sp = SamplingParams(max_tokens=16, stop=["\nstop"])
    rd = RequestRecord(**{"sampling_params": sp, "decoder_visible_text": ""})

    visible, _delta, stop_hit, stop_reason = pipeline._apply_stop_string_policy(
        rd, accumulated_text="hello\nstop world", fallback_delta="hello\nstop world"
    )
    assert stop_hit is True
    assert stop_reason == "\nstop"
    assert visible == "hello"  # excludes the stop string


def test_decode_and_parse_skips_when_interval_not_reached():
    """_decode_and_parse should return None when decode interval hasn't been reached."""
    pipeline = _make_pipeline(
        decoded_text="test",
        delta_text="test",
        decode_interval_tokens=100,  # Very high threshold
        decode_interval_secs=100.0,  # Very high timeout
    )
    rd = RequestRecord(**{
        "last_decoded_index": 0,
        "last_decode_time": time.perf_counter(),
        "sampling_params": SamplingParams(max_tokens=16),
        "delegating_parser": DelegatingParser(),
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
        "decoder_visible_text": "",
    })
    parsed, _raw, _raw_delta, _stop_hit, _stop_reason = pipeline._decode_and_parse(
        "req-1", rd, [1], time.perf_counter(), finished=False
    )
    assert parsed is None  # Should skip because interval not reached
