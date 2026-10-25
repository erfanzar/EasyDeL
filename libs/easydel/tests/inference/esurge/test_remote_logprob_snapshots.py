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

"""Remote wire, rendering and streaming snapshots, without workers or JAX arrays."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import remote_engine, wire
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine_types import EngineCoreOutput, FinishReason
from easydel.inference.esurge.outputs import LogprobsLists
from easydel.inference.sampling_params import SamplingParams


@pytest.mark.cpu_ok
@pytest.mark.parametrize("with_logprobs", [True, False])
def test_remote_stream_preserves_owned_probabilities_across_wire_updates(monkeypatch, with_logprobs):
    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text]}

        def decode(self, ids, **kwargs):
            return "".join(chr(token) for token in ids)

    tokenizer = Tokenizer()

    class Detokenizer:
        def __init__(self):
            self.previous = {}

        def decode(self, request_id, tokens, **kwargs):
            text = tokenizer.decode(tokens)
            previous = self.previous.get(request_id, "")
            self.previous[request_id] = text
            return SimpleNamespace(
                accumulated_text=text, delta_text=text[len(previous) :], last_decoded_index=len(tokens)
            )

        def reset(self, request_id):
            self.previous.pop(request_id, None)

    class Channel:
        alive = True
        queue_stats = wire.QueueStats()
        engine_spec = wire.EngineSpec(
            esurge_name="test",
            max_model_len=32,
            max_num_seqs=1,
            reserve_tokens=1,
            tokenizer_source="test-tokenizer",
            page_size=16,
        )

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def set_plane_handler(self, handler):
            self.handler = handler

        def send_to_leader(self, message, payload=None):
            if isinstance(message, wire.AbortReq):
                return
            assert isinstance(message, wire.Admit)
            [request] = wire.decode_payload(payload)
            self.request_id = request.request_id
            self.handler(wire.AdmitAck(request_id=message.request_id, ok=True), None)
            self.emit(97, -0.25, finished=False)

        def emit(self, token, logprob, *, finished):
            probabilities = (
                LogprobsLists(logprob_token_ids=[[token, 99]], logprobs=[[logprob, -2.0]], sampled_token_ranks=[0])
                if with_logprobs
                else None
            )
            update = EngineCoreOutput(
                request_id=self.request_id,
                new_token_ids=[token],
                new_logprobs=probabilities,
                finish_reason=FinishReason.LENGTH if finished else None,
            )
            self.handler(wire.OutputUpdate(seq=2 if finished else 1), wire.encode_payload(([update], None)))

    class Workers:
        def __init__(self, *args):
            pass

        def start(self, **kwargs):
            return SimpleNamespace(tokenize=lambda rid, text: tokenizer(text)["input_ids"]), Detokenizer()

        def shutdown(self):
            pass

    monkeypatch.setattr(remote_engine, "_ClientChannel", Channel)
    monkeypatch.setattr(remote_engine, "WorkerManager", Workers)
    # Real rendering and registry updates run inline; no threads or processes.
    monkeypatch.setattr(OutputPipeline, "start", lambda self: None)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.jax.process_count", lambda: 1)

    handle = remote_engine.RemoteEngineHandle("unused", 1234, auth_token="test", decode_interval_tokens=1)
    stream = handle.stream("prompt", SamplingParams(max_tokens=2, logprobs=2 if with_logprobs else None), request_id="r")
    try:
        first = next(stream)
        live = handle._registry.outputs["r"].outputs[0]
        assert first.outputs[0].token_ids == [97]
        assert first.outputs[0].text == "a"
        if with_logprobs:
            assert first.outputs[0].logprobs == [{97: -0.25, 99: -2.0}]
            assert first.outputs[0].cumulative_logprob == -0.25
            assert first.outputs[0].logprobs is not live.logprobs
            assert first.outputs[0].logprobs[0] is not live.logprobs[0]
        else:
            assert first.outputs[0].logprobs is None
            assert first.outputs[0].cumulative_logprob is None

        handle._channel.emit(98, -0.5, finished=True)
        final = next(stream)
        assert final.finished
        assert final.outputs[0].token_ids == [97, 98]
        assert final.outputs[0].text == "ab"
        assert final.outputs[0].finish_reason == "length"
        assert first.outputs[0].token_ids == [97]
        if with_logprobs:
            expected = [{97: -0.25, 99: -2.0}, {98: -0.5, 99: -2.0}]
            assert final.outputs[0].logprobs == expected
            assert final.outputs[0].cumulative_logprob == -0.75
            assert live.logprobs == expected
            assert live.cumulative_logprob == -0.75
            # Mutations in either direction must not leak into another snapshot.
            live.logprobs[0][97] = -9.0
            live.logprobs.append({100: -3.0})
            assert first.outputs[0].logprobs == [{97: -0.25, 99: -2.0}]
            assert final.outputs[0].logprobs == expected
            final.outputs[0].logprobs[1][98] = -8.0
            assert live.logprobs[1][98] == -0.5
        else:
            assert final.outputs[0].logprobs is None
            assert final.outputs[0].cumulative_logprob is None
        with pytest.raises(StopIteration):
            next(stream)
    finally:
        stream.close()
        handle.close()
