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

"""Native streaming detokenization and stop scanning stay bounded per output step."""

import pytest
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import Detokenizer

pytestmark = pytest.mark.cpu_ok

_STOP = "<|user|>"


class _CharacterTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class _Client:
    """Recording wrapper; all decoding still executes in the native object."""

    def __init__(self):
        self.native = Detokenizer(_CharacterTokenizer(), context_window=0, max_states=8)
        self.batch_sizes = []

    def decode_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return self.native.decode_batch(requests)

    def reset(self, request_id):
        self.native.reset(request_id)

    def shutdown(self):
        self.native.shutdown()


@pytest.fixture
def client():
    detokenizer = _Client()
    yield detokenizer
    detokenizer.shutdown()


def _stream(client, text: str, *, include_stop: bool = False):
    """Stream one character token per step; return output, result text sizes and stop-scan lengths."""
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="r",
        prompt="p",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    params = SamplingParams(max_tokens=len(text) + 1, stop=[_STOP], include_stop_str_in_output=include_stop)
    registry.register("r", RequestRecord(prompt_token_ids=[1], sampling_params=params), output)
    stops = []
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=client,
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=stops.append,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: pytest.fail(f"output pipeline failed: {exc}"),
    )
    result_text_sizes: list[int] = []
    scanned: list[int] = []
    decode_batch = client.decode_batch
    find = OutputPipeline._find_first_stop_string

    def recording_decode_batch(requests):
        results = decode_batch(requests)
        for request, result in zip(requests, results, strict=True):
            if not request["finished"]:
                # Normal native results contain only their delta. A resync snapshot is allowed but remains bounded here.
                result_text_sizes.append(len(result.accumulated_text or result.delta_text))
        return results

    def recording_find(text, stop_sequences):
        scanned.append(len(text))
        return find(text, stop_sequences)

    client.decode_batch = recording_decode_batch
    pipeline._find_first_stop_string = recording_find
    try:
        for step, char in enumerate(text):
            if output.finished:
                break
            outputs = [EngineCoreOutput("r", [ord(char)])]
            pipeline.submit({0: EngineCoreOutputs(outputs=outputs, timestamp=float(step + 1))})
        if not output.finished:
            outputs = [EngineCoreOutput("r", [], finish_reason=FinishReason.LENGTH)]
            pipeline.submit({0: EngineCoreOutputs(outputs=outputs, timestamp=float(len(text) + 1))})
    finally:
        client.decode_batch = decode_batch
    return output, result_text_sizes, scanned, stops


def _letters(n: int) -> str:
    return "".join(chr(97 + i % 26) for i in range(n))


def test_per_step_result_text_and_stop_scan_do_not_grow_with_the_output(client):
    peaks = {}
    for n in (2048, 32768):
        text = _letters(n)
        output, result_text_sizes, scanned, stops = _stream(client, text)
        completion = output.outputs[0]
        assert output.finished and completion.finish_reason == "length" and not stops
        assert completion.text == text
        assert completion.token_ids == [ord(c) for c in text]
        assert isinstance(completion.token_ids, list)
        # The terminal step returns and scans the whole text once; every streaming step is bounded.
        peaks[n] = (max(result_text_sizes), max(scanned[:-1]))
    assert peaks[32768][0] <= peaks[2048][0] + 1
    assert peaks[32768][1] <= peaks[2048][1]
    assert peaks[32768][1] <= 2 * len(_STOP)


def test_output_bundle_uses_one_native_batch_for_all_due_decodes(client):
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="parent",
        prompt="p",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=index, text="", token_ids=[]) for index in range(2)],
    )
    registry.outputs["parent"] = output
    for index, request_id in enumerate(("r0", "r1")):
        registry.register(
            request_id,
            RequestRecord(
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=2),
                sample_index=index,
                parent_request_id="parent",
            ),
            output,
        )
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=client,
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: pytest.fail(f"output pipeline failed: {exc}"),
    )
    pipeline.submit(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("r0", [65]), EngineCoreOutput("r1", [66])], timestamp=1.0)}
    )
    assert client.batch_sizes == [2]
    assert [completion.text for completion in output.outputs] == ["A", "B"]

    pipeline.submit(
        {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput("r0", [], finish_reason=FinishReason.LENGTH),
                    EngineCoreOutput("r1", [], finish_reason=FinishReason.LENGTH),
                ],
                timestamp=2.0,
            )
        }
    )
    assert client.batch_sizes == [2, 2]
    assert output.finished
    assert [completion.text for completion in output.outputs] == ["A", "B"]


@pytest.mark.parametrize("include_stop", [False, True])
def test_stop_string_split_across_steps_is_found_at_its_first_position(client, include_stop):
    text = _letters(3000) + _STOP + "tail" + _STOP
    output, _, _, stops = _stream(client, text, include_stop=include_stop)
    completion = output.outputs[0]
    cutoff = 3000 + (len(_STOP) if include_stop else 0)
    assert completion.text == text[:cutoff]
    assert completion.finish_reason == "stop"
    assert stops == [{"r": _STOP}]
