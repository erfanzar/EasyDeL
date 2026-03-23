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

from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.mixins.parsing import EngineParsingMixin
from easydel.inference.esurge.mixins.utils import EngineUtilsMixin
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.utils import check_stop
from easydel.inference.reasoning.parsers import DeepSeekR1ReasoningParser
from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import DetokenizerResult


class _StopPolicyHarness(EngineParsingMixin, EngineUtilsMixin):
    pass


class _SamplingParamsHarness(EngineUtilsMixin):
    def __init__(self, extra_stops=None, callback=None, generation_config=None, primary_eos_token_id=None):
        self.extra_stops = self._normalize_stop_sequences(extra_stops)
        self._sampling_params_callback = callback
        self._generation_config_dict = generation_config or {}
        self._primary_eos_token_id = primary_eos_token_id


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


class _DetokenizerStub:
    def reset(self, request_id: str):
        return None


class _ProcessHarness(EngineParsingMixin, EngineUtilsMixin):
    def __init__(self, decoded_text: str, delta_text: str):
        self.decode_interval_tokens = 1
        self.decode_interval_secs = 0.0
        self._decoded_text = decoded_text
        self._delta_text = delta_text
        self._request_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._scheduler_lock = threading.Lock()
        self._request_events = {}
        self._output_event = threading.Event()
        self._active_requests = {}
        self._request_outputs = {}
        self._detokenizer_client = _DetokenizerStub()
        self.scheduler = type("Sched", (), {"requests": {}})()

    def _touch_activity(self):
        return None

    def _decode_with_pipeline(
        self,
        request_id,
        decodable_tokens,
        finished,
        skip_special_tokens,
        spaces_between_special_tokens,
        prompt_context=None,
    ):
        return DetokenizerResult(
            accumulated_text=self._decoded_text,
            delta_text=self._delta_text,
            last_decoded_index=len(decodable_tokens),
            finished=finished,
            detoktook=0.0,
        )


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
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"])
    rd = {"sampling_params": sampling_params, "decoder_visible_text": "Hello "}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="Hello world<user>ignored",
        fallback_delta="world<user>ignored",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "Hello world"
    assert visible_delta == "world"


def test_stop_string_policy_passes_through_without_match():
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["abcd"])
    rd = {"sampling_params": sampling_params, "decoder_visible_text": ""}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="abcx",
        fallback_delta="abcx",
    )

    assert stop_triggered is False
    assert stop_reason is None
    assert visible_text == "abcx"
    assert visible_delta == "abcx"


def test_stop_string_policy_can_include_stop_string_when_requested():
    harness = _StopPolicyHarness()
    sampling_params = SamplingParams(max_tokens=32, stop=["<user>"], include_stop_str_in_output=True)
    rd = {"sampling_params": sampling_params, "decoder_visible_text": ""}

    visible_text, visible_delta, stop_triggered, stop_reason = harness._apply_stop_string_policy(
        rd,
        accumulated_text="ans<user>tail",
        fallback_delta="ans<user>tail",
    )

    assert stop_triggered is True
    assert stop_reason == "<user>"
    assert visible_text == "ans<user>"
    assert visible_delta == "ans<user>"


def test_snapshot_delta_handles_empty_reset_without_fallback():
    harness = _StopPolicyHarness()

    delta = harness._compute_snapshot_delta_text(
        current_text="",
        previous_text="tool markup before parser normalization",
        fallback_delta="",
    )

    assert delta == ""


def test_prepare_sampling_params_for_request_merges_engine_extra_stops():
    harness = _SamplingParamsHarness(extra_stops=["<user>", "DONE"])
    template = SamplingParams(max_tokens=64, stop=["DONE", "</assistant>"])

    prepared = harness._prepare_sampling_params_for_request(
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

    harness = _SamplingParamsHarness(extra_stops="<user>", callback=_callback)
    template = SamplingParams(max_tokens=64, stop=["INITIAL"])

    prepared = harness._prepare_sampling_params_for_request(
        template,
        request_id="req-extra-stops-callback",
        prompt="hello",
    )

    assert prepared.stop == ["CALLBACK_STOP", "<user>"]


def test_prepare_sampling_params_for_request_merges_generation_config_eos_ids():
    harness = _SamplingParamsHarness(
        generation_config={"eos_token_id": [154820, 154827, 154829]},
        primary_eos_token_id=154820,
    )
    template = SamplingParams(max_tokens=64, stop_token_ids=[777])

    prepared = harness._prepare_sampling_params_for_request(
        template,
        request_id="req-generation-config-eos",
        prompt="hello",
    )

    assert set(prepared.stop_token_ids) == {777, 154827, 154829}
    assert prepared.all_stop_token_ids == {777, 154820, 154827, 154829}
    assert template.stop_token_ids == [777]


def test_prepare_sampling_params_respects_ignore_eos_for_generation_config_ids():
    harness = _SamplingParamsHarness(
        generation_config={"eos_token_id": [154820, 154827]},
        primary_eos_token_id=154820,
    )
    template = SamplingParams(max_tokens=64, stop_token_ids=[777], ignore_eos=True)

    prepared = harness._prepare_sampling_params_for_request(
        template,
        request_id="req-generation-config-ignore-eos",
        prompt="hello",
    )

    assert set(prepared.stop_token_ids) == {777}
    assert prepared.all_stop_token_ids == {777, 154820}


def test_process_engine_outputs_keeps_raw_text_before_reasoning_split():
    harness = _ProcessHarness(
        decoded_text="<think>plan</think><tool_call>{}</tool_call>",
        delta_text="<think>plan</think><tool_call>{}</tool_call>",
    )
    reasoning_parser = DeepSeekR1ReasoningParser(_DummyTokenizer())
    request_id = "req-raw-before-parse"
    harness._active_requests[request_id] = {
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
        "reasoning_parser_instance": reasoning_parser,
        "tool_parser_instance": None,
        "tool_parser_request": None,
        "parser_previous_text": "",
        "parser_previous_token_ids": [],
        "accumulated_reasoning": "",
        "accumulated_content": "",
    }
    harness._request_outputs[request_id] = RequestOutput(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[[1, 2]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    harness._process_engine_outputs(
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

    output = harness._request_outputs[request_id]
    completion = output.outputs[0]
    assert completion.raw_text == "<think>plan</think><tool_call>{}</tool_call>"
    assert output.raw_accumulated_text == "<think>plan</think><tool_call>{}</tool_call>"
    assert completion.reasoning_content == "plan"
    assert completion.text == "<tool_call>{}</tool_call>"
