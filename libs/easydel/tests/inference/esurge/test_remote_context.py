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

"""Pure-Python remote admission tests with no workers, networking or array computation."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.distributed import remote_engine, wire
from easydel.inference.sampling_params import SamplingParams


@pytest.mark.cpu_ok
def test_remote_generate_reports_prompt_actually_seen_after_truncation(monkeypatch):
    """Remote admissions use the same default truncated-prompt metadata policy as local engines."""

    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text]}

        def decode(self, ids, **kwargs):
            return "".join(chr(token) for token in ids)

    tokenizer = Tokenizer()
    scheduled = []

    class Channel:
        alive = True
        queue_stats = wire.QueueStats()
        engine_spec = wire.EngineSpec(
            esurge_name="test",
            max_model_len=6,
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
            assert isinstance(message, wire.Admit)
            scheduled.extend(wire.decode_payload(payload))
            self.handler(wire.AdmitAck(request_id=message.request_id, ok=True), None)
            # Stand in for the owner's completed generation. Admission and
            # origin-plane bookkeeping remain real; no numerical model runs.
            handle._registry.outputs[message.request_id].finished = True
            handle._registry.signal(message.request_id)

    class Workers:
        def __init__(self, *args):
            pass

        def start(self, **kwargs):
            client = SimpleNamespace(tokenize=lambda request_id, prompt: tokenizer(prompt)["input_ids"])
            return client, SimpleNamespace()

        def shutdown(self):
            pass

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(remote_engine, "_ClientChannel", Channel)
    monkeypatch.setattr(remote_engine, "WorkerManager", Workers)
    monkeypatch.setattr(remote_engine, "OutputPipeline", Pipeline)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.jax.process_count", lambda: 1)

    handle = remote_engine.RemoteEngineHandle("unused", 1234, auth_token="test")
    try:
        [output] = handle.generate("abcdefgh", SamplingParams(max_tokens=1), request_id="prompt")
        assert output.prompt == "defgh"
        assert output.prompt_token_ids == [[ord(char) for char in "defgh"]]
        assert scheduled[0].prompt_token_ids == [ord(char) for char in "defgh"]
    finally:
        handle.close()
