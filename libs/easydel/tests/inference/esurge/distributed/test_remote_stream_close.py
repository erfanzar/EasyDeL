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

"""Pure-Python client-side request cleanup on interrupted remote streams."""

from threading import RLock
from types import SimpleNamespace

from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.sampling_params import SamplingParams


def test_closing_remote_stream_aborts_unfinished_request():
    handle = object.__new__(RemoteEngineHandle)
    output = RequestOutput(
        request_id="remote-request",
        prompt="hello",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=0, text="hi", token_ids=[2])],
        num_generated_tokens=1,
        accumulated_text="hi",
        update_seq=1,
    )
    notified = []
    registry = SimpleNamespace(
        request_lock=RLock(),
        output_lock=RLock(),
        records={"remote-request": SimpleNamespace(parent_request_id="remote-request")},
        outputs={"remote-request": output},
        events={"remote-request": object()},
        wait_request=lambda *_args, **_kwargs: None,
        signal=lambda *_args: None,
    )
    handle._registry = registry
    handle._plane = SimpleNamespace(notify_abort=notified.append)
    handle._admission = SimpleNamespace(
        prepare_sampling_params_for_request=lambda params, **_kwargs: params,
        add_request=lambda *_args: None,
    )
    handle._raise_if_unusable = lambda: None
    stream = handle.stream("hello", SamplingParams(max_tokens=10), request_id="remote-request")

    assert next(stream).accumulated_text == "hi"
    stream.close()

    assert notified == ["remote-request"]
    assert output.finished
    assert output.outputs[0].finish_reason == "abort"
    assert "remote-request" not in registry.records
    assert "remote-request" not in registry.events
