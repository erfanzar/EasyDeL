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

"""Per-step cost of streaming through the output pipeline and the detokenizer worker stays bounded."""

import pickle
import threading

import pytest
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.sampling_params import SamplingParams
from easydel.workers.esurge.pipeline import worker_main
from easydel.workers.esurge.pipeline.zmq_workers import DetokenizerWorkerClient

pytestmark = pytest.mark.cpu_ok

_STOP = "<|user|>"


class _CharacterTokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_main.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _CharacterTokenizer())
    uri = f"ipc://{tmp_path / 'd.sock'}"
    thread = threading.Thread(target=worker_main._detokenizer_worker, args=(uri, "unused", {}, 8), daemon=True)
    thread.start()
    client = DetokenizerWorkerClient(uri, request_timeout=30)
    yield client
    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _stream(client, text: str, *, include_stop: bool = False):
    """Stream ``text`` one character token per step; return the output and per-step reply bytes and scan lengths."""
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
    reply_bytes: list[int] = []
    scanned: list[int] = []
    request = client._request
    find = OutputPipeline._find_first_stop_string

    def recording_request(payload):
        reply = request(payload)
        if payload.get("cmd") == "decode":
            reply_bytes.append(len(pickle.dumps(reply)))
        return reply

    def recording_find(text, stop_sequences):
        scanned.append(len(text))
        return find(text, stop_sequences)

    client._request = recording_request
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
        client._request = request
    return output, reply_bytes, scanned, stops


def _letters(n: int) -> str:
    return "".join(chr(97 + i % 26) for i in range(n))


def test_per_step_reply_bytes_and_stop_scan_do_not_grow_with_the_output(client):
    peaks = {}
    for n in (2048, 32768):
        text = _letters(n)
        output, reply_bytes, scanned, stops = _stream(client, text)
        completion = output.outputs[0]
        assert output.finished and completion.finish_reason == "length" and not stops
        assert completion.text == text
        assert completion.token_ids == [ord(c) for c in text]
        assert isinstance(completion.token_ids, list)
        # The terminal step returns and scans the whole text once; every streaming step is bounded.
        peaks[n] = (max(reply_bytes[:-1]), max(scanned[:-1]))
    assert peaks[32768][0] <= peaks[2048][0] + 16
    assert peaks[32768][1] <= peaks[2048][1]
    assert peaks[32768][1] <= 2 * len(_STOP)


@pytest.mark.parametrize("include_stop", [False, True])
def test_stop_string_split_across_steps_is_found_at_its_first_position(client, include_stop):
    text = _letters(3000) + _STOP + "tail" + _STOP
    output, _, _, stops = _stream(client, text, include_stop=include_stop)
    completion = output.outputs[0]
    cutoff = 3000 + (len(_STOP) if include_stop else 0)
    assert completion.text == text[:cutoff]
    assert completion.finish_reason == "stop"
    assert stops == [{"r": _STOP}]
