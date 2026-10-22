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

"""eSurge engine behaviors exercised through lightweight stubs (no model, no array compute)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.engine.registry import RequestRegistry
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge

pytestmark = pytest.mark.cpu_ok


class _VocabTokenizer:
    calls = 0

    def get_vocab(self):
        type(self).calls += 1
        return {
            "<|tool_call>": 1,
            "<tool_response|>": 2,
            "[TOOL_CALLS]": 3,
            "tool": 4,
            "stool": 5,
            "toolbar": 6,
            "hello": 7,
            "<toolbar>": 8,
            "[stool]": 9,
            "<｜tool▁calls▁begin｜>": 10,
        }


class _ChatStub(eSurge):
    """Capture the public chat-to-generation boundary without model computation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _format_chat_prompt(self, messages, **kwargs):
        return "hello"

    def _build_tool_parser_request(self, **kwargs):
        return None

    def generate(self, prompts, sampling_params=None, **kwargs):
        return [sampling_params]


def test_plain_chat_suppresses_only_protocol_tokens():
    engine = _ChatStub(_VocabTokenizer())
    params = engine.chat([{"role": "user", "content": "hello"}])
    assert set(params.logit_bias) == {1, 2, 3, 10}
    assert all(v == -100.0 for v in params.logit_bias.values())


def test_plain_chat_refreshes_cached_protocol_ids_after_tokenizer_replacement():
    class ReplacementTokenizer:
        def get_vocab(self):
            return {"<|tool_call>": 101, "tool": 102}

    engine = _ChatStub(_VocabTokenizer())
    engine.chat([{"role": "user", "content": "hello"}])
    engine.tokenizer = ReplacementTokenizer()
    params = engine.chat([{"role": "user", "content": "hello"}])
    assert params.logit_bias == {101: -100.0}


def test_tool_chat_keeps_protocol_tokens_available():
    engine = _ChatStub(_VocabTokenizer())
    params = engine.chat([{"role": "user", "content": "hello"}], tools=[{"name": "weather"}])
    assert params.skip_special_tokens is False
    assert params.logit_bias is None


class _SchedulerStub(eSurge):
    def __init__(self):
        self.runner = SimpleNamespace(
            num_draft_tokens=3,
            update_model_weights=lambda **kwargs: None,
            executor_manager=SimpleNamespace(has_compiled_variants=lambda: True, kv_pages=object()),
        )
        self.cache_config = SimpleNamespace(enable_prefix_caching=True)
        self._scheduler_max_num_batched_tokens = 512
        self._runner_async_scheduling = False
        self._long_prefill_token_threshold = 128
        self._registry = RequestRegistry()
        self._request_lock = self._registry.request_lock
        self._output_lock = self._registry.output_lock
        self._active_requests = self._registry.records
        self._request_outputs = self._registry.outputs
        self._finished_request_ids = self._registry.finished_ids
        self._request_events = self._registry.events

    @property
    def num_running_requests(self):
        return 0

    @property
    def num_pending_requests(self):
        return 0

    @property
    def _generation_alive(self):
        return False

    def _drain_pipeline_workers(self, reason):
        return None

    def _log_cache_event(self, event, details):
        return None


def test_scheduler_rebuild_keeps_construction_settings(monkeypatch):
    def from_runner(runner, **kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(esurge_engine, "Scheduler", SimpleNamespace(from_runner=from_runner))
    engine = _SchedulerStub()
    engine.update_model_weights(model=object(), graphdef="graph", graphstate={}, graphother={})
    assert engine.scheduler.max_num_batched_tokens == 512
    assert engine.scheduler.enable_prefix_caching is True
    assert engine.scheduler.async_scheduling is False
    assert engine.scheduler.long_prefill_token_threshold == 128
    assert engine.scheduler.num_draft_tokens == 3


class _MultimodalStreamStub(eSurge):
    def __init__(self, max_outputs):
        self._registry = RequestRegistry(max_outputs=max_outputs)
        self._request_lock = self._registry.request_lock
        self._output_lock = self._registry.output_lock
        self._request_events = self._registry.events
        self._request_outputs = self._registry.outputs
        self._finished_request_ids = self._registry.finished_ids
        self._max_request_outputs = max_outputs

    @property
    def _generation_alive(self):
        return True

    def _raise_if_scheduler_failed(self):
        return None

    def _recover_orphaned_request(self, request_id):
        return False

    def abort_request(self, request_id):
        raise AssertionError("finished stream must not abort")

    def admit_finished(self, request_id):
        event = threading.Event()
        event.set()
        self._request_events[request_id] = event
        self._request_outputs[request_id] = RequestOutput(
            request_id=request_id,
            prompt="p",
            prompt_token_ids=[1],
            outputs=[CompletionOutput(index=0, text="x", token_ids=[2], finish_reason="stop")],
            finished=True,
            accumulated_text="x",
            delta_text="x",
            num_generated_tokens=1,
            update_seq=1,
        )


def test_multimodal_stream_registers_finished_output_for_retention():
    engine = _MultimodalStreamStub(max_outputs=1)
    for rid in ("a", "b"):
        engine.admit_finished(rid)
        outputs = list(engine._stream_multimodal_request(rid))
        assert outputs[-1].finished
    assert list(engine._request_outputs) == ["b"]
    assert engine._request_events == {}


@pytest.mark.parametrize("max_outputs", [0, -1])
def test_multimodal_stream_nonpositive_retention_discards_finished_output(max_outputs):
    engine = _MultimodalStreamStub(max_outputs=max_outputs)
    engine.admit_finished("request")
    snapshots = list(engine._stream_multimodal_request("request"))
    assert snapshots[-1].accumulated_text == "x"
    assert engine._request_outputs == {}
    assert engine._request_events == {}


def test_abandoned_multimodal_stream_aborts_unfinished_request():
    class AbortableStream(_MultimodalStreamStub):
        def abort_request(self, request_id):
            self._request_outputs[request_id].finished = True
            self._request_events.pop(request_id, None)
            self._registry.track_finished(request_id)

    engine = AbortableStream(max_outputs=0)
    engine.admit_finished("request")
    engine._request_outputs["request"].finished = False
    stream = engine._stream_multimodal_request("request")
    assert next(stream).finished is False
    stream.close()
    assert engine._request_outputs == {}
    assert engine._request_events == {}
