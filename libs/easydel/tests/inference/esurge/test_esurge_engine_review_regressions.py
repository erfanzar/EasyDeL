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

"""Comprehensive regression tests for eSurge Core Engine, Scheduler, KV Cache, Multimodal & Metrics review findings."""

from __future__ import annotations

import gc
import logging
import signal
import socket
import subprocess
import threading
import time
import weakref
from collections import UserDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.config import (
    eSurgeCacheRuntimeConfig,
    eSurgeContextConfig,
    eSurgeRuntimeConfig,
)
from easydel.inference.esurge.core.binary_search import (
    apply_topk_mask,
    apply_topk_mask_bf16,
    apply_topp_mask,
    apply_topp_mask_bf16,
    int32_bsearch,
)
from easydel.inference.esurge.core.interface import (
    CacheGroupSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    create_kv_cache_specs_from_config,
)
from easydel.inference.esurge.core.manager import (
    CacheManager,
    page_hash_extra_key,
    request_page_hash_extra_keys,
)
from easydel.inference.esurge.distributed.zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator
from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.core_driver import FINISH_ABORT, EngineCoreDriver
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.engine.monitoring_stack import MonitoringStack
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput, eSurge
from easydel.inference.esurge.metrics import MetricsCollector, initialize_metrics
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.multimodal.types import MultiModalFeature
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.page_table import PageTable
from easydel.inference.esurge.request import AdmissionRecord
from easydel.inference.openai_api_modules import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from easydel.inference.reasoning.parsers.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from easydel.inference.reasoning.parsers.mistral_reasoning_parser import MistralReasoningParser
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.tools.parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from easydel.inference.tools.parsers.gemma4_tool_parser import Gemma4ToolParser
from easydel.infra.utils import AttnMaskDetail, AttnMaskType

pytestmark = pytest.mark.cpu_ok


class _StubTokenizer:
    eos_token_id = 0

    def __init__(self, *, vocab: dict[str, int] | None = None, special_tokens: list[str] | None = None):
        self._vocab = dict(vocab or {"<eos>": 0, "hello": 1, "world": 2})
        self.all_special_tokens = list(special_tokens or [])
        self.added_tokens_decoder = {}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(t) for t in token_ids if 0 <= t < 0x110000)

    def __call__(self, text: str, add_special_tokens: bool = False, return_attention_mask: bool = False):
        del add_special_tokens, return_attention_mask
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools=None,
        chat_template=None,
        **kwargs,
    ):
        del tokenize
        parts = [f"{m.get('role')}:{m.get('content')}" for m in messages]
        if chat_template:
            parts.append(f"tpl={chat_template}")
        if tools:
            parts.append(f"tools={len(tools)}")
        if kwargs:
            parts.append("kw=" + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())))
        if add_generation_prompt:
            parts.append("<gen>")
        return "|".join(parts)


class _DummyDetokenizer:
    def __init__(self):
        self._buffers: dict[str, list[int]] = {}

    def decode(self, request_id: str, tokens: list[int], finished: bool = False, **kwargs):
        del finished, kwargs
        prev = self._buffers.get(request_id, [])
        delta_tokens = tokens[len(prev) :]
        self._buffers[request_id] = list(tokens)
        accumulated = "".join(chr(t) for t in tokens if t > 0)
        delta = "".join(chr(t) for t in delta_tokens if t > 0)
        return SimpleNamespace(
            accumulated_text=accumulated,
            delta_text=delta,
            last_decoded_index=len(tokens),
        )

    def decode_batch(self, requests: list[dict]):
        results = []
        for req in requests:
            rid = req["request_id"]
            token_offset = int(req.get("token_offset", 0))
            text_offset = int(req.get("text_offset", 0))
            tokens_delta = list(req.get("tokens_delta") or [])
            prev = self._buffers.get(rid, [])
            full_tokens = [*prev[:token_offset], *tokens_delta]
            self._buffers[rid] = full_tokens
            accumulated = "".join(chr(t) for t in full_tokens if t > 0)
            delta = accumulated[text_offset:] if 0 <= text_offset <= len(accumulated) else accumulated
            results.append(
                SimpleNamespace(
                    accumulated_text=accumulated,
                    delta_text=delta,
                    last_decoded_index=len(full_tokens),
                )
            )
        return results

    def reset(self, request_id: str) -> None:
        self._buffers.pop(request_id, None)


def _full_cache_group(page_size: int = 4) -> list[CacheGroupSpec]:
    return [
        CacheGroupSpec(
            kv_cache_spec=FullAttentionSpec(
                page_size=page_size,
                num_kv_heads=1,
                head_size=4,
                dtype=jnp.float32,
                use_mla=False,
            ),
            layer_names=["layer.0"],
        )
    ]


def _make_admission(
    registry: RequestRegistry,
    submit_fn,
    *,
    tokenizer=None,
    reasoning_parser_class=None,
    tool_parser_class=None,
) -> RequestAdmission:
    tok = tokenizer or _StubTokenizer()
    return RequestAdmission(
        registry=registry,
        scheduler_submit=submit_fn,
        tokenizer_client=SimpleNamespace(tokenize=lambda rid, text: [1, 2, 3]),
        tokenizer=tok,
        context_config=eSurgeContextConfig.coerce_config({}),
        reserve_tokens=1,
        max_model_len=64,
        tool_parser_class=tool_parser_class,
        reasoning_parser_class=reasoning_parser_class,
        sampling_params_callback=lambda: None,
        generation_config_dict={},
        primary_eos_token_id=0,
        eos_token_ids=[0],
        extra_stops=[],
        ignore_stop_strings_in_reasoning=False,
        on_activity=lambda: None,
        info=lambda *args: None,
    )


def _make_minimal_engine(core: EngineCoreDriver, *, destroy_pages_on_pause: bool = True, num_pages: int = 16):
    engine = eSurge.__new__(eSurge)
    engine._closed = False
    engine._paused = False
    engine._kv_cache_valid = True
    engine._profiling_active = False
    engine._scheduler_lock = threading.RLock()
    engine._engine_loop = None
    engine._scheduler_running_fallback = True
    engine._worker_replay_thread = None
    engine._worker_stop_event = threading.Event()
    engine._step_coordinator = None
    engine._request_plane = None
    engine._monitoring = None
    engine._worker_manager = None
    engine._heap_frozen = False
    engine._scheduler_max_num_batched_tokens = 32
    engine._runner_async_scheduling = False
    engine._long_prefill_token_threshold = None
    engine.runtime_config = eSurgeRuntimeConfig.from_dict(max_model_len=64, max_num_seqs=4)
    engine.cache_config = eSurgeCacheRuntimeConfig.from_dict(
        enable_prefix_caching=True,
        destroy_pages_on_pause=destroy_pages_on_pause,
        page_size=4,
    )
    engine._registry = RequestRegistry()
    engine._request_lock = engine._registry.request_lock
    engine._output_lock = engine._registry.output_lock
    engine._active_requests = engine._registry.records
    engine._request_outputs = engine._registry.outputs
    engine._request_events = engine._registry.events
    engine._finished_request_ids = engine._registry.finished_ids
    engine._output_event = engine._registry.output_event
    engine._idle_monitor = SimpleNamespace(start=lambda: None, stop=lambda: None, touch=lambda: None)
    engine._output_pipeline = SimpleNamespace(start=lambda: None, stop=lambda: None)
    engine.core = core
    engine._info = lambda *args, **kwargs: None
    engine._log_cache_event = lambda *args, **kwargs: None
    engine._drain_pipeline_workers = lambda reason: None
    engine._drop_heap_freeze = lambda: None
    engine._freeze_startup_heap = lambda: None
    engine._install_signal_diagnostics = lambda: None
    engine._uninstall_signal_diagnostics = lambda: None

    class _FakeRunner:
        def __init__(self):
            self.metadata = SimpleNamespace(num_pages=num_pages, page_size=4, data_parallel_size=1)
            self.executor_manager = SimpleNamespace(kv_pages=object(), has_compiled_variants=lambda: True)
            self.model = SimpleNamespace(config=SimpleNamespace())
            self.kv_cache_groups = _full_cache_group(page_size=4)
            self.max_num_seqs = 4
            self.max_model_len = 64
            self.max_num_seq_buckets = [4]
            self.num_draft_tokens = 0
            self.destroy_calls = 0
            self.init_calls = 0
            self.reset_calls = 0
            self.shutdown_calls = 0
            self.release_calls = 0

        def destroy_kv_cache(self):
            self.destroy_calls += 1
            self.executor_manager.kv_pages = None

        def initialize_kv_cache(self):
            self.init_calls += 1
            self.executor_manager.kv_pages = object()

        def reset_state(self):
            self.reset_calls += 1

        def shutdown(self):
            self.shutdown_calls += 1

        def release_model_state(self, *, clear_compiled_cache=False):
            del clear_compiled_cache
            self.release_calls += 1

    engine.runner = _FakeRunner()
    return engine


# ---------------------------------------------------------------------------
# S1-C1 & S1-H21: pause()/resume()/initiate() prefix-cache reset & OOM shrink
# ---------------------------------------------------------------------------


def test_s1_c1_pause_resets_prefix_cache_when_destroying_kv_pages():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    engine = _make_minimal_engine(core, destroy_pages_on_pause=True, num_pages=16)
    try:
        # Populate prefix cache with a completed request of 8 prompt tokens (2 full pages).
        prompt = [10, 11, 12, 13, 14, 15, 16, 17]
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="r1",
                    prompt_token_ids=prompt,
                    sampling_params=SamplingParams(max_tokens=1),
                    eos_token_id=0,
                )
            ]
        )
        so = core.schedule()
        core.commit(
            so,
            ModelRunnerOutput(
                req_ids=["r1"],
                req_id_to_index={"r1": 0},
                sampled_token_ids=[[0]],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        # Confirm a second request WOULD hit before pause.
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="r2",
                    prompt_token_ids=prompt,
                    sampling_params=SamplingParams(max_tokens=1),
                    eos_token_id=0,
                )
            ]
        )
        so2 = core.schedule()
        assert so2.scheduled_new_reqs[0].num_computed_tokens == 4
        core.commit(
            so2,
            ModelRunnerOutput(
                req_ids=["r2"],
                req_id_to_index={"r2": 0},
                sampled_token_ids=[[0]],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        # Drain the finished_req_ids housekeeping step so all pages are freed.
        drain_so = core.schedule()
        if drain_so.finished_req_ids:
            core.commit(
                drain_so,
                ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                    spec_token_ids=None,
                    logprobs=None,
                    prompt_logprobs_dict={},
                ),
            )

        # Pause destroys KV pages -> MUST reset prefix cache.
        engine.pause()
        assert engine.runner.destroy_calls == 1
        assert engine.runner.executor_manager.kv_pages is None

        # After pause/re-init, identical prompt MUST NOT get phantom prefix-cache hits.
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="r3",
                    prompt_token_ids=prompt,
                    sampling_params=SamplingParams(max_tokens=1),
                    eos_token_id=0,
                )
            ]
        )
        so3 = core.schedule()
        assert so3.scheduled_new_reqs[0].num_computed_tokens == 0
    finally:
        engine.core.shutdown()


def test_s1_h21_resume_rebuilds_core_when_kv_reinit_shrinks_num_pages():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=64,
        enable_prefix_caching=True,
        async_scheduling=False,
    )
    engine = _make_minimal_engine(core, destroy_pages_on_pause=True, num_pages=64)
    engine._scheduler_running_fallback = False
    engine._kv_cache_valid = False
    engine.runner.executor_manager.kv_pages = None

    def _shrink_init():
        engine.runner.init_calls += 1
        engine.runner.metadata.num_pages = 48  # Simulated PP OOM retry shrink
        engine.runner.executor_manager.kv_pages = object()

    engine.runner.initialize_kv_cache = _shrink_init
    initiated = []
    engine.initiate = lambda: initiated.append(engine.core.num_pages)

    try:
        assert engine.core.num_pages == 64
        engine.resume()
        assert engine.core.num_pages == 48
        assert initiated == [48]
    finally:
        engine.core.shutdown()


# ---------------------------------------------------------------------------
# S1-C2: Empty-prompt requests emit abort EngineCoreOutput & finish cleanly
# ---------------------------------------------------------------------------


def test_s1_c2_empty_prompt_emits_abort_output_when_include_finished_set_false():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
        include_finished_set=False,
    )
    try:
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="req-empty",
                    prompt_token_ids=[],
                    sampling_params=SamplingParams(max_tokens=8),
                    eos_token_id=0,
                )
            ]
        )
        so = core.schedule()
        assert "req-empty" in so.finished_req_ids

        outputs = core.commit(
            so,
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        assert 0 in outputs
        assert len(outputs[0].outputs) == 1
        out = outputs[0].outputs[0]
        assert out.request_id == "req-empty"
        assert out.new_token_ids == []
        assert out.finish_reason == FinishReason.ABORT
        assert out.finished is True

        # Feed into OutputPipeline and verify the consumer RequestOutput terminates.
        registry = RequestRegistry()
        ro = RequestOutput(
            request_id="req-empty",
            prompt="",
            prompt_token_ids=[[]],
            outputs=[CompletionOutput(index=0, text="", token_ids=[])],
        )
        registry.register("req-empty", RequestRecord(sampling_params=SamplingParams(max_tokens=8)), ro)
        pipeline = OutputPipeline(
            registry=registry,
            detokenizer_client=_DummyDetokenizer(),
            eos_token_ids=[0],
            decode_interval_tokens=1,
            decode_interval_secs=0.0,
            on_stop_strings=lambda stops: None,
            on_activity=lambda: None,
            on_fatal=lambda exc, tb: (_ for _ in ()).throw(exc),
        )
        pipeline.submit(outputs)
        assert ro.finished is True
        assert ro.outputs[0].finish_reason == "abort"
        assert "req-empty" not in registry.records

        # Also test finish_requests() removes queued _empty requests before schedule().
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="req-empty-abort",
                    prompt_token_ids=[],
                    sampling_params=SamplingParams(max_tokens=8),
                    eos_token_id=0,
                )
            ]
        )
        assert core.is_live("req-empty-abort")
        freed = core.finish_requests(["req-empty-abort"], FINISH_ABORT)
        assert freed == ["req-empty-abort"]
        assert not core.is_live("req-empty-abort")
    finally:
        core.shutdown()


# ---------------------------------------------------------------------------
# S1-C3, S1-H4, S1-M4, S1-M16, S3-M6: Admission n>1 media/seeds/hooks/arrival
# ---------------------------------------------------------------------------


def test_s1_c3_multimodal_n_gt_1_shares_zero_copy_media_across_all_siblings():
    registry = RequestRegistry()
    submitted: list[AdmissionRecord] = []
    adm = _make_admission(registry, submitted.extend)

    pv = np.ones((4, 16), dtype=np.float32)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    vpv = np.ones((8, 16), dtype=np.float32)
    vgrid = np.array([[2, 2, 2]], dtype=np.int64)
    feats = [MultiModalFeature.from_image(pv, grid), MultiModalFeature.from_video(vpv, vgrid)]

    adm.add_request(
        "mm-req",
        "describe",
        SamplingParams(n=3, max_tokens=8),
        prompt_token_ids=[1, 2, 3, 4],
        pixel_values=pv,
        image_grid_thw=grid,
        pixel_values_videos=vpv,
        video_grid_thw=vgrid,
        mm_features=feats,
    )

    assert len(submitted) == 3
    for idx, rec in enumerate(submitted):
        assert rec.sample_index == idx
        assert rec.pixel_values is pv
        assert rec.image_grid_thw is grid
        assert rec.pixel_values_videos is vpv
        assert rec.video_grid_thw is vgrid
        assert rec.mm_features is not feats
        assert len(rec.mm_features) == len(feats)
        for copied, orig in zip(rec.mm_features, feats, strict=True):
            assert copied is not orig
            assert copied.pixel_values is orig.pixel_values
            assert copied.grid_thw is orig.grid_thw
    assert len({id(rec.mm_features[0]) for rec in submitted}) == 3


def test_s1_h4_and_s1_m16_seeded_n_gt_1_clones_params_derives_distinct_seeds_and_adjusts_once():
    adjust_calls: list[int] = []

    class _CountingReasoningParser:
        def __init__(self, tokenizer):
            del tokenizer

        def adjust_request(self, sp):
            adjust_calls.append(id(sp))
            sp.stop = [*list(getattr(sp, "stop", None) or []), "</think>"]
            return sp

        def configure_prompt_context(self, **kwargs):
            del kwargs

    registry = RequestRegistry()
    submitted: list[AdmissionRecord] = []
    adm = _make_admission(registry, submitted.extend, reasoning_parser_class=_CountingReasoningParser)

    adm.add_request(
        "seeded-req",
        "hello",
        SamplingParams(n=4, temperature=0.8, seed=42, max_tokens=8),
        prompt_token_ids=[1, 2, 3],
    )

    assert len(submitted) == 4
    param_ids = [id(r.sampling_params) for r in submitted]
    # Every child has an isolated SamplingParams clone
    assert len(set(param_ids)) == 4
    # Distinct deterministic seeds (child 0 keeps 42, children 1..3 derived & distinct)
    seeds = [r.sampling_params.seed for r in submitted]
    assert seeds[0] == 42
    assert len(set(seeds)) == 4
    assert all(isinstance(s, int) and s >= 0 for s in seeds)
    # adjust_request ran exactly once per child SamplingParams (NOT n+1 times on one shared object)
    assert len(adjust_calls) == 4
    assert set(adjust_calls) == set(param_ids)
    for r in submitted:
        assert r.sampling_params.stop == ["</think>"]

    # Re-running with the same base seed yields identical child seeds (deterministic)
    submitted2: list[AdmissionRecord] = []
    adm2 = _make_admission(RequestRegistry(), submitted2.extend)
    adm2.add_request(
        "seeded-req-2",
        "hello",
        SamplingParams(n=4, temperature=0.8, seed=42, max_tokens=8),
        prompt_token_ids=[1, 2, 3],
    )
    assert [r.sampling_params.seed for r in submitted2] == seeds


def test_s1_m4_multihost_arrival_time_uses_monotonic_counter_even_with_explicit_ids(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    registry = RequestRegistry()
    submitted: list[AdmissionRecord] = []
    adm = _make_admission(registry, submitted.extend)

    adm.add_request("explicit-1", "a", SamplingParams(max_tokens=4), prompt_token_ids=[1, 2])
    adm.add_request("explicit-2", "b", SamplingParams(max_tokens=4), prompt_token_ids=[1, 2])
    adm.add_request("explicit-3", "c", SamplingParams(n=2, max_tokens=4), prompt_token_ids=[1, 2])

    arrivals = [r.arrival_time for r in submitted]
    assert len(arrivals) == 4
    assert arrivals[0] > 0.0
    assert arrivals == sorted(arrivals)
    assert len(set(arrivals)) == 4


def test_s3_m6_admission_invokes_tool_parser_adjust_request():
    tok = _StubTokenizer(vocab={"<|tool_call>": 10, "<tool_call|>": 11, "<|dsml|>": 12})
    tool_req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
        }
    )
    tool_req_none = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "none",
        }
    )

    for parser_cls in (Gemma4ToolParser, DeepSeekV32ToolParser):
        submitted: list[AdmissionRecord] = []
        adm = _make_admission(RequestRegistry(), submitted.extend, tokenizer=tok, tool_parser_class=parser_cls)
        adm.add_request(
            "req-tools",
            "hi",
            SamplingParams(max_tokens=8, skip_special_tokens=True),
            prompt_token_ids=[1, 2],
            tool_parser_request=tool_req,
        )
        assert submitted[0].sampling_params.skip_special_tokens is False

        submitted_none: list[AdmissionRecord] = []
        adm_none = _make_admission(
            RequestRegistry(), submitted_none.extend, tokenizer=tok, tool_parser_class=parser_cls
        )
        adm_none.add_request(
            "req-no-tools",
            "hi",
            SamplingParams(max_tokens=8, skip_special_tokens=True),
            prompt_token_ids=[1, 2],
            tool_parser_request=tool_req_none,
        )
        assert submitted_none[0].sampling_params.skip_special_tokens is True


# ---------------------------------------------------------------------------
# S1-H5: Admission failure rollback + lazy _chat_multimodal(stream=True)
# ---------------------------------------------------------------------------


def test_s1_h5_failed_scheduler_submit_rolls_back_registry_and_metrics(monkeypatch):
    collector = MetricsCollector()
    monkeypatch.setattr("easydel.inference.esurge.engine.admission.get_metrics_collector", lambda: collector)

    registry = RequestRegistry()

    def _boom(records):
        raise RuntimeError(f"submit failed for {len(records)} records")

    adm = _make_admission(registry, _boom)
    for n_samples, req_id in ((1, "fail-single"), (3, "fail-multi")):
        registry.retain_output(req_id)
        with pytest.raises(RuntimeError, match="submit failed"):
            adm.add_request(req_id, "hi", SamplingParams(n=n_samples, max_tokens=4), prompt_token_ids=[1, 2])
        assert req_id not in registry.records
        assert req_id not in registry.outputs
        assert req_id not in registry.events
        assert not registry.output_is_retained(req_id)
        for i in range(n_samples):
            assert f"{req_id}-{i}" not in registry.records
            assert f"{req_id}-{i}" not in registry.events
        assert req_id not in collector.request_metrics
        assert collector.counters["total_requests"] == 0


def test_s1_h5_multimodal_chat_stream_defers_admission_until_iterated():
    engine = eSurge.__new__(eSurge)
    engine._registry = RequestRegistry()
    engine._reasoning_parser_class = None
    engine.tokenizer = _StubTokenizer()
    admitted = []

    class _MM:
        def extract_media_from_messages(self, messages):
            del messages
            return [object()], []

        def process_images(self, images):
            del images
            return np.zeros((1, 4), dtype=np.float32), None

        def process_videos(self, videos):
            del videos
            return None, None

        def process_images_to_features(self, images, processed=None):
            del images, processed
            return []

        def tokenize_multimodal(self, **kwargs):
            del kwargs
            return [1, 2, 3]

    engine._multimodal_manager = _MM()
    engine._prepare_sampling_params_for_request = lambda sp, **kwargs: sp
    engine._add_request = lambda **kwargs: admitted.append(kwargs["request_id"])
    engine._ensure_scheduler_running = lambda **kwargs: None

    gen = engine._chat_multimodal(
        [{"role": "user", "content": [{"type": "image", "image": object()}]}],
        request_id="lazy-mm",
        stream=True,
    )
    # Merely creating the stream iterator MUST NOT admit the request or leak leases.
    assert admitted == []
    assert not engine._registry.output_is_retained("lazy-mm")
    gen.close()
    assert admitted == []


# ---------------------------------------------------------------------------
# S1-H1: Raw pixel_values prefix-cache hash collision + error nonce fallback
# ---------------------------------------------------------------------------


def test_s1_h1_raw_pixel_values_without_mm_features_do_not_collide_in_cache_manager():
    img_a = np.zeros((1, 3, 4, 4), dtype=np.float32)
    img_b = np.ones((1, 3, 4, 4), dtype=np.float32)

    rec_text = AdmissionRecord(request_id="text", prompt_token_ids=[1, 2, 3, 4, 5], sampling_params=SamplingParams(max_tokens=2), eos_token_id=0)
    rec_a1 = AdmissionRecord(
        request_id="img-a1",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=2),
        eos_token_id=0,
        pixel_values=img_a,
    )
    rec_a2 = AdmissionRecord(
        request_id="img-a2",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=2),
        eos_token_id=0,
        pixel_values=img_a.copy(),
    )
    rec_b = AdmissionRecord(
        request_id="img-b",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=2),
        eos_token_id=0,
        pixel_values=img_b,
    )

    assert page_hash_extra_key(rec_text) == 0
    assert page_hash_extra_key(rec_a1) != 0
    assert page_hash_extra_key(rec_a1) == page_hash_extra_key(rec_a2)
    assert page_hash_extra_key(rec_a1) != page_hash_extra_key(rec_b)

    # Erroring multimodal feature extraction must NOT collapse onto a shared ("multimodal",) key.
    class _BrokenFeatures:
        def __bool__(self):
            return True

        def __iter__(self):
            raise RuntimeError("corrupt feature list")

    bad1 = SimpleNamespace(request_id="bad-1", has_vision=True, mm_features=_BrokenFeatures())
    bad2 = SimpleNamespace(request_id="bad-2", has_vision=True, mm_features=_BrokenFeatures())
    assert request_page_hash_extra_keys(bad1) != request_page_hash_extra_keys(bad2)
    assert page_hash_extra_key(bad1) != page_hash_extra_key(bad2)

    # End-to-end verification in CacheManager: img_a caches pages, img_b and text must NOT hit.
    cm = CacheManager(num_pages=16, kv_cache_groups=_full_cache_group(page_size=4), max_model_len=32, enable_caching=True)

    def _kv_req(rid: str, pv):
        return SimpleNamespace(
            request_id=rid,
            all_token_ids=[1, 2, 3, 4, 5],
            num_tokens=5,
            num_computed_tokens=0,
            sampling_params=SamplingParams(max_tokens=2),
            has_vision=pv is not None,
            mm_features=[],
            pixel_values=pv,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
        )

    r_a1 = _kv_req("a1", img_a)
    hits, n = cm.get_computed_pages(r_a1)
    assert n == 0
    assert cm.allocate_slots(r_a1, num_new_tokens=5, num_new_computed_tokens=0, new_computed_pages=hits) is not None
    cm.cache_pages(r_a1, num_computed_tokens=4)

    _, n_same = cm.get_computed_pages(_kv_req("a2", img_a.copy()))
    _, n_diff = cm.get_computed_pages(_kv_req("b1", img_b))
    _, n_text = cm.get_computed_pages(_kv_req("t1", None))
    assert n_same == 4
    assert n_diff == 0
    assert n_text == 0


# ---------------------------------------------------------------------------
# S1-H9 & S1-H10 & S1-M12: CacheManager stale keys, Hybrid EAGLE, LINEAR merge
# ---------------------------------------------------------------------------


def test_s1_h9_cache_manager_refreshes_stale_keys_after_token_growth_without_publish():
    sw_group = [
        CacheGroupSpec(
            kv_cache_spec=SlidingWindowSpec(
                page_size=4,
                num_kv_heads=1,
                head_size=4,
                dtype=jnp.float32,
                use_mla=False,
                sliding_window=8,
            ),
            layer_names=["layer.0"],
        )
    ]
    cm = CacheManager(num_pages=32, kv_cache_groups=sw_group, max_model_len=64, enable_caching=True)

    # First publish 3 pages (12 tokens) via a donor request.
    donor = SimpleNamespace(
        request_id="donor",
        all_token_ids=list(range(10, 26)),
        num_tokens=16,
        num_computed_tokens=0,
        sampling_params=SamplingParams(max_tokens=4),
        has_vision=False,
        mm_features=[],
    )
    hits, _ = cm.get_computed_pages(donor)
    assert cm.allocate_slots(donor, 16, 0, hits) is not None
    cm.cache_pages(donor, 16)
    cm.free(donor)

    # Now probe/lookup a request initially at 5 tokens (1 cached key), then grow to 16 tokens without publish.
    req = SimpleNamespace(
        request_id="grower",
        all_token_ids=list(range(10, 15)),
        num_tokens=5,
        num_computed_tokens=0,
        sampling_params=SamplingParams(max_tokens=4),
        has_vision=False,
        mm_features=[],
    )
    assert len(cm.request_page_hashes(req)) == 1

    req.all_token_ids = list(range(10, 26))
    req.num_tokens = 16

    # Previously probe and lookup raised IndexError on SlidingWindowSpec and truncated hashes!
    assert cm.probe_num_computed_tokens(req) == 12
    hashes = cm.request_page_hashes(req)
    assert len(hashes) == 4
    _, hit_tokens = cm.get_computed_pages(req)
    assert hit_tokens == 12


def test_s1_h10_hybrid_full_sliding_eagle_reserves_only_one_page():
    groups = [
        CacheGroupSpec(
            kv_cache_spec=FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False),
            layer_names=["layer.0"],
        ),
        CacheGroupSpec(
            kv_cache_spec=SlidingWindowSpec(
                page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False, sliding_window=4
            ),
            layer_names=["layer.1"],
        ),
    ]
    cm = CacheManager(num_pages=32, kv_cache_groups=groups, max_model_len=64, enable_caching=True, use_eagle=True)

    # Publish 2 pages (8 tokens) — exactly cdiv(window-1, ps) + 1 = 1 + 1 = 2 pages.
    donor = SimpleNamespace(
        request_id="donor",
        all_token_ids=list(range(1, 10)),
        num_tokens=9,
        num_computed_tokens=0,
        sampling_params=SamplingParams(max_tokens=4),
        has_vision=False,
        mm_features=[],
    )
    hits, _ = cm.get_computed_pages(donor)
    assert cm.allocate_slots(donor, 9, 0, hits) is not None
    cm.cache_pages(donor, 8)
    cm.free(donor)

    # With 2 cached pages and EAGLE reserving 1 page, hit MUST be 4 tokens (1 page), NOT 0!
    probe_req = SimpleNamespace(
        request_id="probe",
        all_token_ids=list(range(1, 10)),
        num_tokens=9,
        num_computed_tokens=0,
        sampling_params=SamplingParams(max_tokens=4),
        has_vision=False,
        mm_features=[],
    )
    assert cm.probe_num_computed_tokens(probe_req) == 4
    pages, hit_tokens = cm.get_computed_pages(probe_req)
    assert hit_tokens == 4
    assert len(pages.pages[0]) == 1 and len(pages.pages[1]) == 1


def test_s1_m12_linear_layers_merge_into_full_group_even_with_global_head_dim():
    class _HybridCfg:
        layer_types = ("full_attention", "linear_attention", "sliding_attention", "linear_attention")
        global_head_dim = 512
        attention_k_eq_v = True
        num_global_key_value_heads = 2

        def get_mask_details(self):
            return {
                0: AttnMaskDetail(mask_type=AttnMaskType.FULL, size=None),
                1: AttnMaskDetail(mask_type=AttnMaskType.LINEAR, size=None),
                2: AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=128),
                3: AttnMaskDetail(mask_type=AttnMaskType.LINEAR, size=None),
            }

    specs = create_kv_cache_specs_from_config(
        config=_HybridCfg(),
        page_size=16,
        num_kv_heads=8,
        head_size=256,
        dtype=jnp.float32,
    )
    full_specs = [s for s in specs if isinstance(s.kv_cache_spec, FullAttentionSpec)]
    sliding_specs = [s for s in specs if isinstance(s.kv_cache_spec, SlidingWindowSpec)]
    assert len(full_specs) == 1
    assert len(sliding_specs) == 1
    assert full_specs[0].layer_names == ["layer.0", "layer.1", "layer.3"]
    assert full_specs[0].kv_cache_spec.head_size == 512
    assert full_specs[0].kv_cache_spec.num_kv_heads == 2


def test_s1_m12_cache_pages_clamps_to_allocated_pages():
    cm = CacheManager(num_pages=16, kv_cache_groups=_full_cache_group(page_size=4), max_model_len=32, enable_caching=True)
    req = SimpleNamespace(
        request_id="clamp-req",
        all_token_ids=list(range(1, 13)),
        num_tokens=12,
        num_computed_tokens=0,
        sampling_params=SamplingParams(max_tokens=4),
        has_vision=False,
        mm_features=[],
    )
    hits, _ = cm.get_computed_pages(req)
    # Allocate only 1 page (4 tokens)
    assert cm.allocate_slots(req, 4, 0, hits) is not None
    # Publishing with num_computed_tokens=12 (> 1 allocated page) must not overshoot req.cached
    cm.cache_pages(req, 12)
    # Allocate 2 more pages and publish again cleanly without KeyError/ValueError
    req.num_computed_tokens = 4
    assert cm.allocate_slots(req, 8, 0, None) is not None
    cm.cache_pages(req, 12)


# ---------------------------------------------------------------------------
# S1-H2 & S1-M3: OutputPipeline delta_tool_calls preservation & stop policy
# ---------------------------------------------------------------------------


def test_s1_h2_delta_tool_calls_preserved_on_final_step_and_coalesced_reads():
    registry = RequestRegistry()
    ro = RequestOutput(
        request_id="tc-req",
        prompt="call tool",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )

    chunk_a = DeltaToolCall(index=0, id="call_1", type="function", function=DeltaFunctionCall(name="get_weather", arguments=""))
    chunk_b = DeltaToolCall(index=0, function=DeltaFunctionCall(arguments='{"city":"NYC"}'))
    final_call = ToolCall(id="call_1", type="function", function=FunctionCall(name="get_weather", arguments='{"city":"NYC"}'))

    class _ToolDelegatingParser:
        reasoning_parser = None
        tool_parser = object()

        def __init__(self):
            self.step = 0

        def process_delta(self, accumulated, delta, token_ids, prev_text, prev_ids):
            del accumulated, delta, token_ids, prev_text, prev_ids
            self.step += 1
            emitted = [chunk_a] if self.step == 1 else [chunk_b]
            return SimpleNamespace(
                to_dict=lambda: {
                    "delta_reasoning": None,
                    "delta_content": "",
                    "accumulated_reasoning": "",
                    "accumulated_content": "",
                    "tool_calls": None,
                    "delta_tool_calls": emitted,
                }
            )

        def process_final(self, accumulated, token_ids):
            del accumulated, token_ids
            return SimpleNamespace(
                tool_calls=[final_call],
                to_dict=lambda: {
                    "delta_reasoning": None,
                    "delta_content": "",
                    "accumulated_reasoning": "",
                    "accumulated_content": "",
                    "tool_calls": [final_call],
                    "delta_tool_calls": None,  # process_delta already emitted the chunks!
                },
            )

        def streamed_tool_call_id(self, index):
            return "call_1" if index == 0 else None

    rd = RequestRecord(
        prompt="call tool",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=8),
        delegating_parser=_ToolDelegatingParser(),
    )
    registry.register("tc-req", rd, ro)
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=_DummyDetokenizer(),
        eos_token_ids=[0],
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: (_ for _ in ()).throw(exc),
    )

    # Submit step 1 (non-finished) AND step 2 (tokens + finished=True) BEFORE consumer reads!
    pipeline.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("tc-req", [65])])})
    pipeline.submit(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("tc-req", [66], finish_reason=FinishReason.STOP)])}
    )

    # (a) Final step must NOT have clobbered ro.delta_tool_calls to None!
    assert ro.delta_tool_calls == [chunk_b]
    assert ro.tool_call_delta_history == [chunk_a, chunk_b]

    # (b) Consumer reading via _iter_request_snapshots after coalescing must receive BOTH chunks!
    engine = eSurge.__new__(eSurge)
    engine._registry = registry
    engine._request_lock = registry.request_lock
    engine._output_lock = registry.output_lock
    engine._request_events = registry.events
    engine._request_outputs = registry.outputs
    engine._max_request_outputs = None
    engine._scheduler_running_fallback = True
    engine._engine_loop = None
    engine._worker_replay_thread = None
    engine._raise_if_scheduler_failed = lambda: None
    engine._recover_orphaned_request = lambda rid: False

    snapshots = list(engine._iter_request_snapshots("tc-req", stopped_message="stopped"))
    assert len(snapshots) == 1
    assert snapshots[0].finished is True
    assert snapshots[0].delta_tool_calls == [chunk_a, chunk_b]


def test_s1_m3_stop_string_parsed_domain_retraction_one_char_rewrite_and_finish_reason():
    registry = RequestRegistry()
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=_DummyDetokenizer(),
        eos_token_ids=[0],
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: (_ for _ in ()).throw(exc),
    )

    # 1. Parsed-domain retraction must NOT leak raw delta_text when parsed["delta_content"] is empty.
    class _RetractingParser:
        reasoning_parser = object()

        def process_delta(self, accumulated, delta, *args):
            del accumulated, delta, args
            return SimpleNamespace(
                to_dict=lambda: {
                    "delta_reasoning": "secret",
                    "delta_content": "",
                    "accumulated_reasoning": "secret",
                    "accumulated_content": "visible",
                    "tool_calls": None,
                    "delta_tool_calls": None,
                }
            )

    rd = RequestRecord(
        sampling_params=SamplingParams(max_tokens=8, stop=["END"], ignore_stop_strings_in_reasoning=True),
        delegating_parser=_RetractingParser(),
        decoder_visible_text="visible-old",
    )
    parsed, visible, vis_delta, stop_hit, _ = pipeline._parse_with_stop_string_policy(
        rd,
        accumulated_text="<think>secret</think>visible",
        delta_text="<think>secret",
        token_ids=[1, 2],
        finished=False,
    )
    assert "<think>" not in parsed["delta_content"]
    assert "<think>" not in vis_delta
    assert visible == "visible" and not stop_hit

    # 2. 1-char stop string must detect rewritten accumulated_text (overlap=0 previously compared "" == "").
    rd_1char = RequestRecord(sampling_params=SamplingParams(max_tokens=8, stop=["\n"]))
    v1, _, hit1, _ = pipeline._apply_stop_string_policy(rd_1char, accumulated_text="abc", fallback_delta="abc", finished=False)
    assert v1 == "abc" and not hit1
    # Now rewrite earlier text to contain "\n" at index 1 while growing length to 4.
    v2, _, hit2, stop_r = pipeline._apply_stop_string_policy(rd_1char, accumulated_text="a\nbd", fallback_delta="d", finished=False)
    assert hit2 is True and stop_r == "\n" and v2 == "a"

    # 3. Stop-string finish_reason must not be overwritten by post-stop tool_calls in _finalize_request.
    post_stop_call = ToolCall(id="call_post", type="function", function=FunctionCall(name="bad", arguments="{}"))

    class _PostStopToolParser:
        reasoning_parser = object()

        def process_delta(self, accumulated, delta, *args):
            del delta, args
            return SimpleNamespace(
                tool_calls=[post_stop_call],
                to_dict=lambda: {
                    "delta_reasoning": None,
                    "delta_content": accumulated,
                    "accumulated_reasoning": "",
                    "accumulated_content": accumulated,
                    "tool_calls": [post_stop_call],
                    "delta_tool_calls": None,
                },
            )

        def process_final(self, accumulated, *args):
            return self.process_delta(accumulated, accumulated)

        def streamed_tool_call_id(self, index):
            del index
            return None

    ro_stop = RequestOutput(
        request_id="stop-req",
        prompt="p",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    rd_stop = RequestRecord(
        sampling_params=SamplingParams(max_tokens=8, stop=["STOP"], ignore_stop_strings_in_reasoning=True),
        delegating_parser=_PostStopToolParser(),
    )
    registry.register("stop-req", rd_stop, ro_stop)
    tokens = [ord(c) for c in "ok STOP <tool_call>"]
    pipeline.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("stop-req", tokens)])})
    assert ro_stop.finished is True
    assert ro_stop.outputs[0].finish_reason == "stop"
    assert ro_stop.outputs[0].text == "ok "


# ---------------------------------------------------------------------------
# S1-H3, S1-H13, S1-L5: MultiModalManager templating, bucketing & edge cases
# ---------------------------------------------------------------------------


def test_s1_h3_and_s1_h13_tokenize_multimodal_threads_chat_kwargs_and_buckets_images():
    class _FakePILImage:
        def __init__(self, size):
            self.size = size
            self.resized_to = None

        def convert(self, mode):
            assert mode == "RGB"
            return self

        def resize(self, size, resample=None):
            del resample
            out = _FakePILImage(size)
            out.resized_to = size
            return out

    seen = {}

    class _CapturingProcessor:
        def __init__(self):
            self.tokenizer = _StubTokenizer()

        def __call__(self, text=None, images=None, videos=None, return_tensors=None, padding=False):
            del videos, return_tensors, padding
            seen["text"] = text
            seen["image_sizes"] = [img.size for img in (images or [])]
            return UserDict({"input_ids": np.array([[ord(c) for c in text]], dtype=np.int32)})

    proc = _CapturingProcessor()
    manager = MultiModalManager(processor=proc, model=None, resolution_buckets=[(32, 32), (64, 64)])
    img = _FakePILImage((40, 40))  # Nearest bucket is (32, 32)
    tools = [{"type": "function", "function": {"name": "search"}}]

    ids = manager.tokenize_multimodal(
        messages=[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "find"}]}],
        images=[img],
        tools=tools,
        chat_template="custom_tpl",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert "tools=1" in seen["text"]
    assert "tpl=custom_tpl" in seen["text"]
    assert "enable_thinking=False" in seen["text"]
    assert seen["image_sizes"] == [(32, 32)]
    assert ids == [ord(c) for c in seen["text"]]


def test_s1_l5_multimodal_none_placeholder_and_multi_media_without_grid_slicing():
    cfg = SimpleNamespace(
        image_token_id=100,
        vision_start_token_id=101,
        vision_end_token_id=102,
        video_token_id=103,
    )
    tok = _StubTokenizer()
    tok.convert_ids_to_tokens = lambda tid: {100: "<IMG>", 101: "<VS>", 102: "<VE>", 103: "<VID>"}.get(tid)
    manager = MultiModalManager(processor=tok, model=SimpleNamespace(config=cfg), enable_cache=False)

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "video", "video": None},
                {"type": "text", "text": "plain text"},
            ],
        }
    ]
    extracted_imgs, extracted_vids = manager.extract_media_from_messages(msgs)
    assert extracted_imgs == [] and extracted_vids == []
    normalized = manager._normalize_messages_for_chat_template(msgs, cfg, tok)
    assert normalized[0]["content"] == "plain text"

    # Multi-image 3D pixel_values without grid_thw must slice per image instead of duplicating.
    pv_3d = np.stack([np.full((4, 8), 1.0, dtype=np.float32), np.full((4, 8), 2.0, dtype=np.float32)], axis=0)
    feats = manager.process_images_to_features([object(), object()], processed=(pv_3d, None))
    assert len(feats) == 2
    assert feats[0].pixel_values.shape == (1, 4, 8)
    assert feats[1].pixel_values.shape == (1, 4, 8)
    assert feats[0].mm_hash != feats[1].mm_hash

    # Multi-video 4D pixel_values without grid_thw must slice per video instead of duplicating.
    vpv_4d = np.stack([np.full((2, 4, 8), 3.0, dtype=np.float32), np.full((2, 4, 8), 4.0, dtype=np.float32)], axis=0)
    vfeats = manager.process_videos_to_features([object(), object()], processed=(vpv_4d, None))
    assert len(vfeats) == 2
    assert vfeats[0].pixel_values.shape == (1, 2, 4, 8)
    assert vfeats[1].pixel_values.shape == (1, 2, 4, 8)
    assert vfeats[0].mm_hash != vfeats[1].mm_hash


# ---------------------------------------------------------------------------
# S1-H6, S1-H7, S1-H20: ZMQ leader/worker lifecycle, update_model_weights, close
# ---------------------------------------------------------------------------


def test_s1_h6_and_s1_h20_zmq_leader_and_worker_pause_terminate_and_release():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    try:
        # 1. Leader in distributed_mode: IdleMonitor skips reset; default pause() shuts down coordinator
        # symmetrically, while explicit pause(shutdown_coordinator=False) keeps coordinator alive.
        leader = _make_minimal_engine(core, destroy_pages_on_pause=True)
        coord_shutdowns: list[str] = []
        leader._step_coordinator = SimpleNamespace(
            is_leader=True,
            world_size=2,
            shutdown=lambda reason="": coord_shutdowns.append(reason),
            check_health=lambda: None,
            start=lambda: None,
        )
        leader._perform_idle_reset()
        assert leader.runner.destroy_calls == 0  # skipped in distributed_mode

        leader.pause(shutdown_coordinator=False)
        assert coord_shutdowns == []
        assert leader.runner.destroy_calls == 1

        leader._scheduler_running_fallback = True
        leader._kv_cache_valid = True
        leader.pause()
        assert len(coord_shutdowns) == 1
        assert leader.runner.destroy_calls == 2

        # 2. Worker rank: pause() stops replay thread AND destroys KV cache; terminate() runs full cleanup.
        worker = _make_minimal_engine(core, destroy_pages_on_pause=True)
        worker._scheduler_running_fallback = False
        worker._profiling_active = True
        prof_stopped = []
        worker.stop_profiling = lambda: prof_stopped.append(True)
        plane_failed = []
        worker._request_plane = SimpleNamespace(
            owns_scheduling=False,
            fail_pending=lambda reason: plane_failed.append(reason),
        )
        worker_coord_shutdowns = []
        worker._step_coordinator = SimpleNamespace(
            is_leader=False,
            world_size=2,
            shutdown=lambda reason="": worker_coord_shutdowns.append(reason),
        )

        # Spawn a live dummy worker replay thread waiting on _worker_stop_event.
        t = threading.Thread(target=lambda: worker._worker_stop_event.wait(5.0), daemon=True)
        t.start()
        worker._worker_replay_thread = t

        worker.pause()
        assert not t.is_alive()
        assert worker._worker_replay_thread is None
        assert worker.runner.destroy_calls == 1
        assert worker._kv_cache_valid is False
        assert prof_stopped == [True]
        assert worker.runner.shutdown_calls == 1
        assert len(worker_coord_shutdowns) == 1
        assert len(plane_failed) >= 1

        # 2b. Worker rank when leader Shutdown already exited replay thread BEFORE worker.pause() runs:
        # pause() must still run full cleanup (clear _worker_replay_thread, stop output worker/profiling,
        # shut down runner/coordinator, fail pending request_plane, destroy KV cache).
        worker_exited = _make_minimal_engine(core, destroy_pages_on_pause=True)
        worker_exited._scheduler_running_fallback = False
        worker_exited._profiling_active = True
        exited_prof = []
        exited_out_stop = []
        exited_plane = []
        exited_coord = []
        worker_exited.stop_profiling = lambda: exited_prof.append(True)
        worker_exited._output_pipeline = SimpleNamespace(
            start=lambda: None,
            stop=lambda: exited_out_stop.append(True),
        )
        worker_exited._request_plane = SimpleNamespace(
            owns_scheduling=False,
            fail_pending=lambda reason: exited_plane.append(reason),
        )
        worker_exited._step_coordinator = SimpleNamespace(
            is_leader=False,
            world_size=2,
            shutdown=lambda reason="": exited_coord.append(reason),
        )
        dead_t = threading.Thread(target=lambda: None, daemon=True)
        dead_t.start()
        dead_t.join(timeout=2.0)
        assert not dead_t.is_alive()
        worker_exited._worker_replay_thread = dead_t
        assert worker_exited._generation_alive is False

        worker_exited.pause()
        assert worker_exited._worker_replay_thread is None
        assert worker_exited.runner.destroy_calls == 1
        assert worker_exited.runner.shutdown_calls == 1
        assert exited_prof == [True]
        assert len(exited_out_stop) >= 1
        assert len(exited_coord) == 1
        assert len(exited_plane) >= 1

        # 3. Worker release_model_state() stops active replay thread and destroys KV cache before releasing weights.
        worker2 = _make_minimal_engine(core, destroy_pages_on_pause=True)
        worker2._scheduler_running_fallback = False
        worker2._step_coordinator = SimpleNamespace(
            is_leader=False,
            world_size=2,
            shutdown=lambda reason="": None,
        )
        t2 = threading.Thread(target=lambda: worker2._worker_stop_event.wait(5.0), daemon=True)
        t2.start()
        worker2._worker_replay_thread = t2
        worker2.release_model_state()
        assert not t2.is_alive()
        assert worker2._worker_replay_thread is None
        assert worker2.runner.destroy_calls == 1
        assert worker2.runner.release_calls == 1
    finally:
        core.shutdown()


def _free_local_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.parametrize("pause_order", ["leader_first", "worker_first"])
@pytest.mark.parametrize("resume_order", ["leader_first", "worker_first"])
def test_s1_h6_and_s1_h20_real_zmq_lockstep_pause_resume_across_heartbeat_timeout(pause_order, resume_order):
    port = _free_local_tcp_port()
    leader_core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    worker_core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    leader = _make_minimal_engine(leader_core, destroy_pages_on_pause=True)
    worker = _make_minimal_engine(worker_core, destroy_pages_on_pause=True)
    leader._scheduler_running_fallback = False
    worker._scheduler_running_fallback = False
    leader.runtime_config = eSurgeRuntimeConfig.from_dict(
        max_model_len=64,
        max_num_seqs=4,
        overlap_execution=False,
    )
    worker.runtime_config = eSurgeRuntimeConfig.from_dict(
        max_model_len=64,
        max_num_seqs=4,
        overlap_execution=False,
    )
    worker_out_stops: list[bool] = []
    worker._output_pipeline = SimpleNamespace(start=lambda: None, stop=lambda: worker_out_stops.append(True))

    leader_coord = ZmqLeaderCoordinator(
        leader.runner,
        world_size=2,
        bind_host="127.0.0.1",
        control_port=port,
        auth_token="test-token",
        config_fingerprint="fp-lockstep",
        ready_timeout_s=5.0,
        step_timeout_s=2.0,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.25,
    )
    worker_coord = ZmqWorkerCoordinator(
        worker.runner,
        rank=1,
        world_size=2,
        leader_addr="127.0.0.1",
        control_port=port,
        auth_token="test-token",
        config_fingerprint="fp-lockstep",
        connect_timeout_s=5.0,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.25,
    )
    leader._step_coordinator = leader_coord
    worker._step_coordinator = worker_coord

    def _run_concurrent(fn_leader, fn_worker, *, leader_first: bool, stagger_s: float = 0.0) -> None:
        errors: list[tuple[str, BaseException]] = []

        def _wrap(label, fn, delay):
            try:
                if delay > 0:
                    time.sleep(delay)
                fn()
            except BaseException as exc:
                errors.append((label, exc))

        t_leader = threading.Thread(target=_wrap, args=("leader", fn_leader, 0.0 if leader_first else stagger_s))
        t_worker = threading.Thread(target=_wrap, args=("worker", fn_worker, stagger_s if leader_first else 0.0))
        t_leader.start()
        t_worker.start()
        t_leader.join(timeout=8.0)
        t_worker.join(timeout=8.0)
        assert not t_leader.is_alive(), "leader thread timed out"
        assert not t_worker.is_alive(), "worker thread timed out"
        assert errors == [], f"unexpected errors: {errors}"

    try:
        # Initial handshake
        _run_concurrent(leader.initiate, worker.initiate, leader_first=True)
        assert leader._engine_loop is not None and leader._engine_loop.running
        assert worker._worker_replay_thread is not None and worker._worker_replay_thread.is_alive()
        leader_coord.check_health()

        # Lockstep pause in requested order
        if pause_order == "leader_first":
            leader.pause()
            # Wait for leader's wire.Shutdown to cause worker replay thread to exit BEFORE worker.pause()
            # runs, exercising the exited-replay-thread cleanup path.
            replay_t = worker._worker_replay_thread
            if replay_t is not None:
                replay_t.join(timeout=3.0)
                assert not replay_t.is_alive()
            worker.pause()
        else:
            worker.pause()
            leader.pause()

        assert leader.runner.destroy_calls == 1
        assert worker.runner.destroy_calls == 1
        assert leader._kv_cache_valid is False
        assert worker._kv_cache_valid is False
        assert leader_coord._io_thread is None
        assert worker_coord._sock is None
        assert worker._worker_replay_thread is None
        assert len(worker_out_stops) >= 1

        # Wait strictly longer than heartbeat_timeout_s (0.25s) to simulate training phase between rollouts.
        time.sleep(0.4)

        # Lockstep resume in requested order (staggered so either leader or worker enters initiate() first).
        _run_concurrent(
            leader.resume,
            worker.resume,
            leader_first=(resume_order == "leader_first"),
            stagger_s=0.05,
        )

        assert leader.runner.init_calls == 1
        assert worker.runner.init_calls == 1
        assert leader._kv_cache_valid is True
        assert worker._kv_cache_valid is True
        assert leader._engine_loop is not None and leader._engine_loop.running
        assert worker._worker_replay_thread is not None and worker._worker_replay_thread.is_alive()
        assert leader._scheduler_exception is None
        assert worker._scheduler_exception is None
        leader_coord.check_health()
        assert leader_coord._next_step_id == 0
    finally:
        for eng in (worker, leader):
            try:
                eng.terminate()
            except Exception:
                pass
        worker_coord.shutdown("test cleanup")
        leader_coord.shutdown("test cleanup")
        worker_core.shutdown()
        leader_core.shutdown()


def test_s1_h7_close_shuts_down_core_pipeline_plane_and_update_weights_handles_partial_kwargs_and_leases(monkeypatch):
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    engine = _make_minimal_engine(core)
    engine._scheduler_running_fallback = False  # Already stopped!

    closed_events = []
    engine._output_pipeline = SimpleNamespace(stop=lambda: closed_events.append("output_pipeline"))
    engine._request_plane = SimpleNamespace(
        owns_scheduling=True,
        shutdown=lambda: closed_events.append("request_plane"),
    )
    orig_shutdown = core.shutdown

    def _tracked_core_shutdown():
        closed_events.append("core")
        orig_shutdown()

    core.shutdown = _tracked_core_shutdown

    engine.close()
    assert "output_pipeline" in closed_events
    assert "request_plane" in closed_events
    assert "core" in closed_events

    # Partial graph kwargs with unresolvable missing components must raise ValueError (not AttributeError),
    # and _schedule_requests_direct must reject admissions while _updating_weights is True.
    core2 = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    try:
        eng2 = _make_minimal_engine(core2)
        eng2._scheduler_running_fallback = False
        eng2._registry.retain_output("leaked-lease")

        # Direct fence check: _schedule_requests_direct rejects admission while _updating_weights=True.
        eng2._updating_weights = True
        with pytest.raises(RuntimeError, match="model weights are being updated"):
            eng2._schedule_requests_direct(
                [AdmissionRecord(request_id="r-fence", prompt_token_ids=[1, 2], sampling_params=SamplingParams(max_tokens=2), eos_token_id=0)]
            )
        eng2._updating_weights = False

        with pytest.raises(ValueError, match="missing"):
            eng2.update_model_weights(graphstate=object())
        # _updating_weights flag must be cleared even when update_model_weights raises.
        assert not getattr(eng2, "_updating_weights", False)

        # Partial graphstate-only update SUCCEEDS when runner.executor_manager supplies graphdef & graphother,
        # rejects concurrent _schedule_requests_direct while updating, rebuilds self.core, and clears _output_leases.
        class _HostState:
            def __init__(self, tag: str):
                self.tag = tag

            def overlay(self, other):
                return (self.tag, other)

        new_state = _HostState("new-weights")
        eng2.runner.executor_manager.graphdef = "saved-graphdef"
        eng2.runner.executor_manager.graphother = "saved-graphother"
        monkeypatch.setattr(
            esurge_engine.spx,
            "bind",
            lambda gd, combined: SimpleNamespace(graphdef=gd, combined=combined),
        )

        runner_updates = []
        fence_rejected_during_update = []

        def _fake_update_weights(**kwargs):
            runner_updates.append(kwargs)
            # Confirm admissions are fenced while update_model_weights is actively running.
            try:
                eng2._schedule_requests_direct(
                    [
                        AdmissionRecord(
                            request_id="r-mid-update",
                            prompt_token_ids=[1, 2],
                            sampling_params=SamplingParams(max_tokens=2),
                            eos_token_id=0,
                        )
                    ]
                )
            except RuntimeError as exc:
                fence_rejected_during_update.append(str(exc))

        eng2.runner.update_model_weights = _fake_update_weights
        old_core = eng2.core
        eng2.update_model_weights(graphstate=new_state)

        assert len(runner_updates) == 1
        assert runner_updates[0]["graphdef"] == "saved-graphdef"
        assert runner_updates[0]["graphstate"] is new_state
        assert runner_updates[0]["graphother"] == "saved-graphother"
        assert runner_updates[0]["reset_state"] is True
        assert len(fence_rejected_during_update) == 1
        assert "model weights are being updated" in fence_rejected_during_update[0]
        assert not getattr(eng2, "_updating_weights", False)
        assert eng2.core is not old_core
        assert not eng2._registry.output_is_retained("leaked-lease")
    finally:
        eng2.core.shutdown()
        core2.shutdown()


# ---------------------------------------------------------------------------
# S3-H10: eSurge.chat(..., tools=None) preserves special reasoning delimiters
# ---------------------------------------------------------------------------


def test_s3_h10_prepare_chat_sampling_params_preserves_special_reasoning_delimiters():
    # DeepSeekR1 / Mistral with special <think> / [THINK] tokens
    for parser_cls, special_toks in (
        (DeepSeekR1ReasoningParser, ["<think>", "</think>"]),
        (MistralReasoningParser, ["[THINK]", "[/THINK]"]),
    ):
        engine = eSurge.__new__(eSurge)
        engine.tokenizer = _StubTokenizer(
            vocab={tok: i + 10 for i, tok in enumerate(special_toks)} | {"<tool_call>": 99},
            special_tokens=special_toks,
        )
        engine._reasoning_parser_class = parser_cls

        sp = engine._prepare_chat_sampling_params(SamplingParams(max_tokens=16), tools=None)
        assert sp.skip_special_tokens is False
        # Tool protocol tokens are still suppressed via logit_bias
        assert sp.logit_bias[99] <= -100.0

    # Without a reasoning parser, tools=None still sets skip_special_tokens=True
    plain_engine = eSurge.__new__(eSurge)
    plain_engine.tokenizer = _StubTokenizer(vocab={"<tool_call>": 99}, special_tokens=["<eos>"])
    plain_engine._reasoning_parser_class = None
    sp_plain = plain_engine._prepare_chat_sampling_params(SamplingParams(max_tokens=16), tools=None)
    assert sp_plain.skip_special_tokens is True


# ---------------------------------------------------------------------------
# S1-M1: SIGTERM diagnostic handler non-blocking, weakref, restored on close
# ---------------------------------------------------------------------------


def test_s1_m1_signal_diagnostics_uses_weakref_restores_sig_dfl_and_does_not_deadlock(monkeypatch):
    orig = signal.getsignal(signal.SIGTERM)
    if orig not in (signal.SIG_DFL, None):
        pytest.skip("Non-default SIGTERM handler already installed in test runner")

    engine = eSurge.__new__(eSurge)
    engine._closed = False
    engine._scheduler_lock = threading.RLock()
    engine._scheduler_running_fallback = True
    engine._engine_loop = None
    engine._scheduler_heartbeat = time.monotonic()
    engine.core = SimpleNamespace(counts=lambda: [1, 2, 3, 4])

    engine._install_signal_diagnostics()
    installed = signal.getsignal(signal.SIGTERM)
    assert installed is not signal.SIG_DFL

    # Invoke handler while _scheduler_lock is HELD by another thread (must NOT deadlock!)
    locked = threading.Event()
    release = threading.Event()

    def _hold_lock():
        with engine._scheduler_lock:
            locked.set()
            release.wait(2.0)

    holder = threading.Thread(target=_hold_lock, daemon=True)
    holder.start()
    assert locked.wait(1.0)
    killed = []
    written = []
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("os.write", lambda fd, data: written.append(data) or len(data))
    try:
        installed(signal.SIGTERM, None)
    finally:
        release.set()
        holder.join(timeout=2.0)
    assert killed and killed[0][1] == signal.SIGTERM
    assert any(b"<locked>" in chunk for chunk in written)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    # Re-install and verify weakref allows GC + close() restores SIG_DFL
    engine._install_signal_diagnostics()
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    engine.close()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    engine2 = eSurge.__new__(eSurge)
    engine2._closed = False
    engine2._install_signal_diagnostics()
    ref = weakref.ref(engine2)
    del engine2
    gc.collect()
    try:
        assert ref() is None
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# S1-M2: EngineLoop._run_overlap aborts & best-effort drains on execute_async error
# ---------------------------------------------------------------------------


def test_s1_m2_overlap_loop_aborts_and_drains_inflight_when_next_execute_async_raises():
    events: list[str] = []
    aborts: list[BaseException] = []

    def _so(tag: str):
        return SimpleNamespace(
            tag=tag,
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={f"{tag}-r": 1},
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(num_reqs=0),
            async_scheduling=True,
        )

    script = [_so("S1"), _so("S2"), _so("S3")]

    class _Core:
        num_running = 1
        num_waiting = 0

        def schedule(self):
            out = script.pop(0)
            events.append(f"schedule:{out.tag}")
            return out

        def commit(self, so, mo):
            events.append(f"commit:{so.tag}")
            return {}

    class _Runner:
        def execute_model_async(self, so):
            events.append(f"dispatch:{so.tag}")
            if so.tag == "S2":
                raise RuntimeError("execute_async boom on S2")
            return SimpleNamespace(tag=so.tag)

        def wait_for_execution(self, handle):
            events.append(f"drain:{handle.tag}")
            return SimpleNamespace(tag=handle.tag)

        def can_dispatch_next_before_async_drain(self, so):
            del so
            return True

        def log_it(self, *args, **kwargs):
            pass

    loop = EngineLoop(
        core=_Core(),
        runner=_Runner(),
        coordinator=None,
        scheduler_lock=threading.RLock(),
        emit_outputs=lambda out: None,
        drain_parser_stops=lambda: None,
        on_fatal=lambda exc: aborts.append(exc),
        heartbeat=lambda: None,
        handle_profiling_step=lambda: None,
        is_nonrecoverable=lambda exc: False,
        overlap_execution=True,
        async_scheduling=True,
        runner_verbose=False,
        info=lambda *args, **kwargs: None,
    )
    loop.start()
    loop.join(timeout=5.0)
    assert loop.thread is None
    assert len(aborts) == 1
    assert "execute_async boom on S2" in str(aborts[0])
    # S1 was best-effort drained & committed; loop did NOT retry S3!
    assert "drain:S1" in events
    assert "commit:S1" in events
    assert "schedule:S3" not in events


# ---------------------------------------------------------------------------
# S1-M5: Versioned output-change notification prevents stolen wakeups
# ---------------------------------------------------------------------------


def test_s1_m5_concurrent_waiters_do_not_steal_output_wakeups():
    registry = RequestRegistry()
    v0 = registry.output_version()
    results: list[tuple[bool, int]] = []
    ready = threading.Barrier(3)

    def _waiter():
        ready.wait()
        results.append(registry.wait_for_output_change(v0, timeout=2.0))

    t1 = threading.Thread(target=_waiter, daemon=True)
    t2 = threading.Thread(target=_waiter, daemon=True)
    t1.start()
    t2.start()
    ready.wait()
    time.sleep(0.02)
    t_start = time.perf_counter()
    registry.notify_output_change()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)
    elapsed = time.perf_counter() - t_start

    assert len(results) == 2
    assert all(woke and ver > v0 for woke, ver in results)
    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# S1-M6: EngineCoreDriver.commit() preserves spec_drafts when spec_token_ids is None
# ---------------------------------------------------------------------------


def test_s1_m6_commit_preserves_spec_drafts_when_spec_token_ids_is_none():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_draft_tokens=2,
    )
    try:
        core.add_requests(
            [
                AdmissionRecord(
                    request_id="spec-req",
                    prompt_token_ids=[1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=8),
                    eos_token_id=0,
                )
            ]
        )
        so = core.schedule()
        core.commit(
            so,
            ModelRunnerOutput(
                req_ids=["spec-req"],
                req_id_to_index={"spec-req": 0},
                req_id_to_row_index={"spec-req": 0},
                sampled_token_ids=[[42]],
                spec_token_ids=None,  # None! Drafts come exclusively via spec_drafts
                spec_drafts=np.array([[100, 101]], dtype=np.int32),
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        state = core.request_state("spec-req")
        assert state is not None
        assert state[3] == [100, 101]
    finally:
        core.shutdown()


# ---------------------------------------------------------------------------
# S1-M10: int32_bsearch / topk / topp work cleanly under jax_enable_x64=True
# ---------------------------------------------------------------------------


def test_s1_m10_binary_search_masks_work_under_jax_enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        bits = int32_bsearch((2,), lambda mid: mid >= jnp.int32(7))
        assert bits.dtype == jnp.int32
        assert bits.shape == (2,)
        np.testing.assert_array_equal(np.asarray(bits), np.array([6, 6], dtype=np.int32))

        logits = jnp.array([[1.0, 4.0, 2.0, 3.0], [-5.0, -1.0, -3.0, -2.0]], dtype=jnp.float32)
        topk = apply_topk_mask(logits, jnp.array([2, 2], dtype=jnp.int32), replace_val=-1e9)
        assert topk.shape == logits.shape
        assert int(jnp.sum(topk[0] > -1e8)) == 2

        topp = apply_topp_mask(logits, jnp.array([0.8, 0.8], dtype=jnp.float32), replace_val=-1e9)
        assert topp.shape == logits.shape

        logits_bf16 = logits.astype(jnp.bfloat16)
        assert apply_topk_mask_bf16(logits_bf16, jnp.array([2, 2], dtype=jnp.int32), -1e9).dtype == jnp.bfloat16
        assert apply_topp_mask_bf16(logits_bf16, jnp.array([0.8, 0.8], dtype=jnp.float32), -1e9).dtype == jnp.bfloat16
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# S1-L4: Atomic PageTable.append_row on overflow
# ---------------------------------------------------------------------------


def test_s1_l4_page_table_append_row_is_atomic_on_overflow():
    pt = PageTable(max_num_reqs=2, max_num_pages_per_req=3, max_num_batched_tokens=16)
    pt.append_row([1, 2], row_idx=0)
    v_before = pt.cpu_version
    with pytest.raises(ValueError, match="exceeds max_num_pages_per_req"):
        pt.append_row([3, 4], row_idx=0)
    assert int(pt.num_pages_per_row[0]) == 2
    assert pt.cpu_version == v_before
    assert pt.page_table_cpu[0, :2].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# S1-L6: MetricsCollector & MonitoringStack housekeeping
# ---------------------------------------------------------------------------


def test_s1_l6_metrics_failed_count_and_file_handler_cleanup(tmp_path):
    collector = MetricsCollector()
    collector.start_request("ok", prompt_tokens=4)
    collector.add_generated_tokens("ok", 3)
    collector.complete_request("ok", finish_reason="stop")

    collector.start_request("err", prompt_tokens=4)
    collector.complete_request("err", finish_reason="abort", error="boom")

    sys_metrics = collector.get_system_metrics(window_seconds=60.0)
    assert sys_metrics.total_requests_completed == 1
    assert sys_metrics.total_requests_failed == 1
    assert sys_metrics.total_tokens_generated == 3

    # Repeated initialize_metrics(log_file=...) must not leak FileHandlers on eSurge.metrics logger.
    metrics_logger = logging.getLogger("eSurge.metrics")
    before_handlers = list(metrics_logger.handlers)
    try:
        c1 = initialize_metrics(log_file=str(tmp_path / "m1.log"))
        c2 = initialize_metrics(log_file=str(tmp_path / "m2.log"))
        file_handlers = [h for h in metrics_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert c1._file_handler is None
        c2.close()
        assert [h for h in metrics_logger.handlers if isinstance(h, logging.FileHandler)] == []
    finally:
        for h in list(metrics_logger.handlers):
            if h not in before_handlers:
                metrics_logger.removeHandler(h)
                h.close()


def test_s1_l6_monitoring_stack_grafana_kill_prom_cleanup_and_reused_port(monkeypatch):
    stack = MonitoringStack(info=lambda *args, **kwargs: None)

    # 1. Grafana terminate timeout falls back to .kill()
    events = []

    class _StubbornProc:
        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            del timeout
            if "kill" not in events:
                raise subprocess.TimeoutExpired(cmd="grafana", timeout=2)
            events.append("wait_after_kill")
            return 0

        def kill(self):
            events.append("kill")

    stack._grafana_process = _StubbornProc()
    stack._stop_grafana_service()
    assert events == ["terminate", "kill", "wait_after_kill"]
    assert stack._grafana_process is None

    # 2. Failed Grafana autostart cleans up newly spawned Prometheus server
    prom_stopped = []
    stack2 = MonitoringStack(info=lambda *args, **kwargs: None)

    def _fake_start_prom(scrape_target, prometheus_server_port=9090):
        del scrape_target
        stack2._prometheus_process = object()
        return f"http://localhost:{prometheus_server_port}"

    stack2._start_prometheus_server = _fake_start_prom
    stack2._stop_prometheus_server = lambda: prom_stopped.append(True) or setattr(stack2, "_prometheus_process", None)
    stack2._provision_running_grafana = lambda **kwargs: None
    stack2._prepare_grafana_provisioning = lambda **kwargs: "/tmp/nonexistent-esurge-test-dir"
    stack2._start_local_grafana_service = lambda **kwargs: None

    assert (
        stack2._start_grafana_service(
            prometheus_url="http://localhost:11184/metrics",
            grafana_host="localhost",
            grafana_port=3000,
            grafana_image="grafana/grafana:latest",
            grafana_admin_user="admin",
            grafana_admin_password="admin",
            allow_anonymous=True,
            datasource_name="eSurge",
            datasource_uid="esurge-prometheus",
            datasource_url=None,
            use_docker=False,
        )
        is None
    )
    assert prom_stopped == [True]

    # 3. Reported Prometheus URL uses the reused singleton's actual port
    stack3 = MonitoringStack(info=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "easydel.inference.esurge.monitoring.start_monitoring_server",
        lambda prometheus_port, update_interval, owner: SimpleNamespace(prometheus_port=11184),
    )
    monkeypatch.setattr("easydel.inference.esurge.engine.monitoring_stack.get_metrics_collector", lambda: object())
    urls = stack3.start_monitoring(prometheus_port=19999, enable_prometheus=True, start_grafana=False)
    assert urls["prometheus"] == "http://0.0.0.0:11184/metrics"


# ---------------------------------------------------------------------------
# Leak Audit #1: MultiModalManager weakref/metadata + release/update/close
# ---------------------------------------------------------------------------


def test_leak_1_multimodal_manager_does_not_pin_model_and_clears_across_lifecycle(monkeypatch):
    class _HeavyVLM:
        supports_flat_patch_inputs = True

        def __init__(self, tag: str):
            self.tag = tag
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1)
            )
            self.weights = np.ones((8, 8), dtype=np.float32)

    m1 = _HeavyVLM("v1")
    m1_ref = weakref.ref(m1)
    manager = MultiModalManager(processor=None, model=m1, resolution_buckets=[(28, 28)], enable_cache=True)
    manager.cache.put("img-key", np.zeros((2, 4), dtype=np.float32))
    assert manager._supports_flat_patch_inputs() is True
    assert manager._get_vision_config().patch_size == 14

    # Dropping external strong references must immediately free m1 even before release_model_state(),
    # while lightweight config/capability metadata remains usable.
    del m1
    gc.collect()
    assert m1_ref() is None
    assert manager._supports_flat_patch_inputs() is True
    assert manager._get_vision_config().patch_size == 14

    # Engine release_model_state() must clear manager.model and manager.cache.
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    try:
        engine = _make_minimal_engine(core)
        engine._scheduler_running_fallback = False
        engine._multimodal_manager = manager
        engine.runner.release_model_state = lambda clear_compiled_cache=False: None
        engine.release_model_state()
        assert manager.model is None
        assert manager.cache.get("img-key") is None

        # update_model_weights() refreshes manager metadata from the new model and clears stale cache.
        m2_holder: list = [_HeavyVLM("v2")]
        m2_ref = weakref.ref(m2_holder[0])
        manager.cache.put("stale-key", np.ones((2, 4), dtype=np.float32))

        class _State:
            def overlay(self, other):
                return (self, other)

        engine.runner.executor_manager.graphdef = "gd2"
        engine.runner.executor_manager.graphother = "go2"
        monkeypatch.setattr(esurge_engine.spx, "bind", lambda gd, combined: m2_holder[0])

        def _fake_update(**kwargs):
            del kwargs
            engine.runner.model = m2_holder[0]

        engine.runner.update_model_weights = _fake_update
        engine.update_model_weights(graphstate=_State())
        assert manager.cache.get("stale-key") is None
        assert manager._supports_flat_patch_inputs() is True

        # Releasing strong refs to m2 (and clearing runner.model) proves manager holds no strong ref.
        engine.runner.model = None
        m2_holder.clear()
        gc.collect()
        assert m2_ref() is None
    finally:
        engine.core.shutdown()
        core.shutdown()


# ---------------------------------------------------------------------------
# Leak Audit #2: eSurge.close() releases HBM/KV/caches/registry & breaks gc.freeze cycles
# ---------------------------------------------------------------------------


def test_leak_2_close_releases_all_resources_and_breaks_cycles_under_gc_freeze():
    from easydel.inference.esurge.engine.idle import IdleMonitor

    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    events: list[str] = []

    class _HeavyRunner:
        def __init__(self):
            self.kv_pages = object()
            self.model = object()

        def reset_state(self):
            events.append("reset_state")

        def destroy_kv_cache(self):
            self.kv_pages = None
            events.append("destroy_kv_cache")

        def release_model_state(self, *, clear_compiled_cache: bool = False):
            self.model = None
            events.append(f"release_model_state:{clear_compiled_cache}")

        def shutdown(self):
            events.append("runner_shutdown")

    class _FakeDrafter:
        def __init__(self):
            self.target = object()

        def release_target_model(self):
            self.target = None
            events.append("drafter_release")

    class _FakePlane:
        owns_scheduling = True

        def __init__(self, cb):
            self._cb = cb

        def shutdown(self):
            events.append("plane_shutdown")

    class _FakeCoord:
        is_leader = True

        def __init__(self):
            self.stats_provider = None

        def set_stats_provider(self, fn):
            self.stats_provider = fn

        def shutdown(self, reason=""):
            del reason
            events.append("coord_shutdown")

    # Simulate a second live engine holding gc.freeze() so gc.unfreeze() cannot run on engine1.close().
    esurge_engine._freeze_heap()
    try:
        def _build_cyclic_engine():
            eng = _make_minimal_engine(core)
            eng._scheduler_running_fallback = False
            eng.runner = _HeavyRunner()
            eng.drafter = drafter
            eng._multimodal_manager = mm
            eng._monitoring = MonitoringStack(info=eng._info)
            eng._idle_monitor = IdleMonitor(
                on_idle_reset=eng._perform_idle_reset,
                idle_reset_seconds=None,
                min_interval=10.0,
                is_running=lambda: eng._scheduler_running,
                is_busy=eng._has_active_work,
                info=eng._info,
            )
            eng._output_pipeline = OutputPipeline(
                registry=eng._registry,
                detokenizer_client=_DummyDetokenizer(),
                eos_token_ids=[0],
                decode_interval_tokens=1,
                decode_interval_secs=0.0,
                on_stop_strings=eng._enqueue_parser_stop_requests,
                on_activity=eng._touch_activity,
                on_fatal=eng._on_output_worker_fatal,
            )
            eng._admission = RequestAdmission(
                registry=eng._registry,
                scheduler_submit=eng._submit_scheduler_requests,
                tokenizer_client=SimpleNamespace(tokenize=lambda rid, p: [1, 2]),
                tokenizer=_StubTokenizer(),
                context_config=eSurgeContextConfig.coerce_config(None),
                reserve_tokens=4,
                max_model_len=64,
                tool_parser_class=None,
                reasoning_parser_class=None,
                sampling_params_callback=lambda: eng._sampling_params_callback,
                generation_config_dict=None,
                primary_eos_token_id=0,
                eos_token_ids=[0],
                extra_stops=[],
                ignore_stop_strings_in_reasoning=False,
                on_activity=eng._touch_activity,
                info=eng._info,
                callback_engine=eng,
            )
            eng._sampling_params_callback = None
            coord.set_stats_provider(eng._queue_stats_snapshot)
            eng._step_coordinator = coord
            eng._request_plane = _FakePlane(eng.abort_request)
            eng._engine_loop = EngineLoop(
                core=core,
                runner=eng.runner,
                coordinator=coord,
                scheduler_lock=eng._scheduler_lock,
                emit_outputs=eng._enqueue_engine_outputs,
                drain_parser_stops=eng._drain_parser_stop_requests_locked,
                on_fatal=eng._abort_scheduler_due_to_error,
                heartbeat=eng._update_scheduler_heartbeat,
                handle_profiling_step=eng._handle_profiling_step,
                is_nonrecoverable=eng._is_nonrecoverable_scheduler_error,
                overlap_execution=False,
                async_scheduling=False,
                runner_verbose=False,
                info=eng._info,
            )
            eng._registry.register(
                "inflight-1",
                RequestRecord(prompt="hi", prompt_token_ids=[1]),
                RequestOutput(
                    request_id="inflight-1",
                    prompt="hi",
                    prompt_token_ids=[[1]],
                    outputs=[CompletionOutput(index=0, text="", token_ids=[])],
                ),
            )
            return eng

        drafter = _FakeDrafter()
        mm = MultiModalManager(processor=object(), model=SimpleNamespace(config=SimpleNamespace()), enable_cache=True)
        mm.cache.put("k", np.zeros((2, 2), dtype=np.float32))
        coord = _FakeCoord()
        engine = _build_cyclic_engine()

        # Freeze engine1 while it is still alive with all cycles present (matches initiate()).
        gc.freeze()
        eng_ref = weakref.ref(engine)
        runner_ref = engine.runner
        registry_ref = engine._registry

        engine.close()

        assert "reset_state" in events
        assert "destroy_kv_cache" in events
        assert "release_model_state:True" in events
        assert "runner_shutdown" in events
        assert "drafter_release" in events
        assert "plane_shutdown" in events
        assert "coord_shutdown" in events
        assert runner_ref.kv_pages is None
        assert runner_ref.model is None
        assert drafter.target is None
        assert mm.model is None
        assert mm.cache.get("k") is None
        assert len(registry_ref.outputs) == 0
        assert len(registry_ref.records) == 0
        assert coord.stats_provider is None
        assert engine.monitoring_active is False

        # Pure refcounting must reclaim engine immediately even though gc.freeze() is still active!
        del engine
        assert eng_ref() is None
    finally:
        esurge_engine._release_heap_freeze()
        gc.unfreeze()


def test_leak_2_close_breaks_real_zmq_request_plane_and_runner_cycles_under_gc_freeze():
    from easydel.inference.esurge.distributed.request_plane import create_request_plane
    from easydel.inference.esurge.distributed.zmq_coordinator import ZmqLeaderCoordinator, ZmqWorkerCoordinator
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.executors.compile_orchestrator import CompileOrchestrator
    from easydel.inference.esurge.runners.executors.pipeline_microbatch import PipelineMicrobatchExecutor
    from easydel.inference.esurge.runners.executors.sampler_executor import SamplerRuntime
    from easydel.inference.esurge.runners.model_runner import eSurgeRunner
    from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
    from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper

    class _DummyDrafter:
        def __init__(self):
            self.weights = np.ones((4, 4), dtype=np.float32)
            self.target_model = object()

        def reset(self, batch_size: int = 1):
            del batch_size

        def release_target_model(self):
            self.target_model = None

    def _build_real_runner_stack():
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("x",))
        sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        sampler_rt = SamplerRuntime(
            sampler_executor=SimpleNamespace(),
            sampler_sharding=sh,
            empty_sharding=sh,
            max_num_reqs=4,
            vocab_size=16,
            min_input_pad=1,
        )
        em = ExecutionManager.__new__(ExecutionManager)
        em.kv_pages = object()
        em.drop_recurrent_snapshots = lambda: None
        em.last_batch_metadata = None
        em.last_spec_window_delta = None
        em._batch_preparer = None
        em.sampler_runtime = sampler_rt
        em._greedy_total_tokens_cache = {}
        em._model_executor = SimpleNamespace(
            model=object(),
            graphdef=object(),
            clear_runtime_graph_args=lambda: None,
            refresh_kv_pages_template=lambda val: None,
            _pipeline_kv_carry_map_cache={},
            pipeline_runtime=None,
            shutdown=lambda: None,
        )
        em._sampler_executor = SimpleNamespace(model=object())
        em.mesh = None
        em.graphdef = None
        em.graphstate = None
        em.graphother = None
        em.clear_cache = lambda: None
        em._pp_microbatch = PipelineMicrobatchExecutor(em)
        em._compile_orchestrator = CompileOrchestrator(em)

        runner = eSurgeRunner.__new__(eSurgeRunner)
        runner.model = object()
        runner.requests = {}
        runner.sequence_buffer = SimpleNamespace(clear=lambda: None)
        runner.slot_pool = None
        runner._executor = None
        runner.max_num_reqs = 4
        runner._side_input_provider = None
        runner._side_input_tokenizer_state = None
        runner.executor_manager = em
        drafter = _DummyDrafter()
        runner.drafter = drafter
        runner.spec = DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=2)
        runner_ref = weakref.ref(runner)
        runner.vlm = VlmPrefillHelper(
            model_getter=lambda _ref=runner_ref: getattr(_ref(), "model", None),
            metadata=SimpleNamespace(data_parallel_size=1),
            compile_vision_encoder=False,
            vision_patch_buckets=None,
            max_num_batched_tokens=32,
            max_num_tokens=32,
        )
        return runner, em, em._pp_microbatch, em._compile_orchestrator, sampler_rt, runner.vlm, runner.spec, drafter

    esurge_engine._freeze_heap()
    try:
        # 1. Real ZmqLeaderCoordinator + OwnerRequestPlane + real runner/VLM/spec/PP/orchestrator cycles under gc.freeze()
        def _setup_leader():
            core = EngineCoreDriver(
                kv_cache_groups=_full_cache_group(page_size=4),
                max_num_seqs=4,
                max_num_batched_tokens=32,
                max_model_len=64,
                num_pages=16,
                enable_prefix_caching=False,
            )
            eng = _make_minimal_engine(core)
            eng._scheduler_running_fallback = False
            runner, em, pp_mb, orch, srt, vlm, spec, drafter = _build_real_runner_stack()
            eng.runner = runner
            eng.drafter = drafter
            eng._output_pipeline = OutputPipeline(
                registry=eng._registry,
                detokenizer_client=_DummyDetokenizer(),
                eos_token_ids=[0],
                decode_interval_tokens=1,
                decode_interval_secs=0.0,
                on_stop_strings=eng._enqueue_parser_stop_requests,
                on_activity=eng._touch_activity,
                on_fatal=eng._on_output_worker_fatal,
            )
            coord = ZmqLeaderCoordinator(
                runner,
                world_size=1,
                bind_host="127.0.0.1",
                control_port=0,
                auth_token="tok",
                config_fingerprint="fp",
            )
            coord.set_stats_provider(eng._queue_stats_snapshot)
            plane = create_request_plane(
                coord,
                submit_outputs=eng._output_pipeline.submit,
                scheduler_submit=eng._schedule_requests_direct,
                abort_request=eng.abort_request,
                apply_stop_strings=eng._enqueue_parser_stop_requests,
                next_arrival_stamp=lambda: 1.0,
                alive=lambda: eng._generation_alive,
                admit_timeout_s=5.0,
                info=eng._info,
            )
            eng._step_coordinator = coord
            eng._request_plane = plane
            refs = {
                "engine": weakref.ref(eng),
                "runner": weakref.ref(runner),
                "em": weakref.ref(em),
                "pp_mb": weakref.ref(pp_mb),
                "orch": weakref.ref(orch),
                "srt": weakref.ref(srt),
                "vlm": weakref.ref(vlm),
                "spec": weakref.ref(spec),
                "drafter": weakref.ref(drafter),
                "coord": weakref.ref(coord),
                "plane": weakref.ref(plane),
            }
            return eng, refs

        leader_eng, leader_refs = _setup_leader()
        gc.freeze()
        leader_eng.close()
        del leader_eng
        gc.collect()
        for label, ref in leader_refs.items():
            assert ref() is None, f"Leader object {label!r} leaked under gc.freeze()"

        # 2. Real ZmqWorkerCoordinator + OriginRequestPlane under gc.freeze()
        def _setup_worker():
            core = EngineCoreDriver(
                kv_cache_groups=_full_cache_group(page_size=4),
                max_num_seqs=4,
                max_num_batched_tokens=32,
                max_model_len=64,
                num_pages=16,
                enable_prefix_caching=False,
            )
            eng = _make_minimal_engine(core)
            eng._scheduler_running_fallback = False
            runner, em, pp_mb, orch, srt, vlm, spec, drafter = _build_real_runner_stack()
            eng.runner = runner
            eng.drafter = drafter
            eng._output_pipeline = OutputPipeline(
                registry=eng._registry,
                detokenizer_client=_DummyDetokenizer(),
                eos_token_ids=[0],
                decode_interval_tokens=1,
                decode_interval_secs=0.0,
                on_stop_strings=eng._enqueue_parser_stop_requests,
                on_activity=eng._touch_activity,
                on_fatal=eng._on_output_worker_fatal,
            )
            coord = ZmqWorkerCoordinator(
                runner,
                rank=1,
                world_size=2,
                leader_addr="127.0.0.1:19999",
                auth_token="tok",
                config_fingerprint="fp",
            )
            plane = create_request_plane(
                coord,
                submit_outputs=eng._output_pipeline.submit,
                scheduler_submit=eng._schedule_requests_direct,
                abort_request=eng.abort_request,
                apply_stop_strings=eng._enqueue_parser_stop_requests,
                next_arrival_stamp=lambda: 1.0,
                alive=lambda: eng._generation_alive,
                admit_timeout_s=5.0,
                info=eng._info,
            )
            eng._step_coordinator = coord
            eng._request_plane = plane
            refs = {
                "engine": weakref.ref(eng),
                "runner": weakref.ref(runner),
                "em": weakref.ref(em),
                "pp_mb": weakref.ref(pp_mb),
                "orch": weakref.ref(orch),
                "srt": weakref.ref(srt),
                "vlm": weakref.ref(vlm),
                "spec": weakref.ref(spec),
                "coord": weakref.ref(coord),
                "plane": weakref.ref(plane),
                "drafter": weakref.ref(drafter),
            }
            return eng, refs

        worker_eng, worker_refs = _setup_worker()
        gc.freeze()
        worker_eng.close()
        del worker_eng
        gc.collect()
        for label, ref in worker_refs.items():
            assert ref() is None, f"Worker object {label!r} leaked under gc.freeze()"
    finally:
        esurge_engine._release_heap_freeze()
        gc.unfreeze()


# ---------------------------------------------------------------------------
# Leak Audit #5 coord & #6: ZMQ include_finished_set & MetricsCollector cleanup
# ---------------------------------------------------------------------------


def test_leak_5_and_6_zmq_forces_include_finished_set_and_close_completes_metrics():
    from easydel.inference.esurge.config import eSurgeDistributedConfig, eSurgeRuntimeConfig

    eng = eSurge.__new__(eSurge)
    eng.runtime_config = eSurgeRuntimeConfig.coerce_config(None)
    assert eng.runtime_config.include_finished_set is False
    eng.distributed_config = eSurgeDistributedConfig(coordination="replicated")
    assert eng._effective_include_finished_set() is False
    eng.distributed_config = eSurgeDistributedConfig(coordination="zmq")
    assert eng._effective_include_finished_set() is True

    # #6: eSurge.close() completes unfinished registry requests in MetricsCollector.
    collector = initialize_metrics()
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    engine = _make_minimal_engine(core)
    engine._scheduler_running_fallback = False
    collector.start_request("unfinished-close-req", prompt_tokens=4)
    engine._registry.register(
        "unfinished-close-req",
        RequestRecord(prompt="p", prompt_token_ids=[1, 2, 3, 4]),
        RequestOutput(
            request_id="unfinished-close-req",
            prompt="p",
            prompt_token_ids=[[1, 2, 3, 4]],
            outputs=[CompletionOutput(index=0, text="", token_ids=[])],
            finished=False,
        ),
    )
    assert "unfinished-close-req" in collector.request_metrics
    engine.close()
    assert "unfinished-close-req" not in collector.request_metrics

    # abort_request() also cleans MetricsCollector when rd is present even if ro was evicted/missing.
    core2 = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    try:
        engine2 = _make_minimal_engine(core2)
        engine2._detokenizer_client = _DummyDetokenizer()
        engine2._failed_detokenizer_resets = set()
        engine2._detokenizer_cleanup_threshold = 100
        collector.start_request("evicted-ro-req", prompt_tokens=2)
        engine2._active_requests["evicted-ro-req"] = RequestRecord(prompt="p", prompt_token_ids=[1, 2])
        engine2.abort_request("evicted-ro-req")
        assert "evicted-ro-req" not in collector.request_metrics
    finally:
        core2.shutdown()


# ---------------------------------------------------------------------------
# Leak Audit #9 & #23: Weakref monitoring leases, partial-init rollback, zombie/urllib/REGISTRY
# ---------------------------------------------------------------------------


def test_leak_9_monitoring_weakref_leases_and_partial_init_rollback(monkeypatch):
    from easydel.inference.esurge import monitoring

    monitoring.stop_monitoring()
    created_servers = []

    class _FakeServer:
        def __init__(self, prometheus_port=8000, dashboard_port=None, update_interval=1.0):
            del dashboard_port, update_interval
            self.prometheus_port = prometheus_port
            self.running = False
            self.stopped = 0
            created_servers.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False
            self.stopped += 1

    monkeypatch.setattr(monitoring, "eSurgeMonitoringServer", _FakeServer)
    try:
        # 1. Owner lease uses weakref.finalize: abandoning an engine + MonitoringStack without
        # calling close()/stop_monitoring() must NOT pin the engine and must auto-stop the server.
        class _DummyEngine:
            def _info(self, *args, **kwargs):
                del args, kwargs

        eng = _DummyEngine()
        stack = MonitoringStack(info=eng._info)
        eng._monitoring = stack
        monitoring.start_monitoring_server(prometheus_port=18888, owner=stack)
        assert monitoring._monitoring_server is created_servers[-1]
        assert created_servers[-1].running is True

        eng_ref = weakref.ref(eng)
        stack_ref = weakref.ref(stack)
        del stack
        del eng
        gc.collect()

        assert eng_ref() is None
        assert stack_ref() is None
        assert monitoring._monitoring_server is None
        assert created_servers[-1].running is False
        assert created_servers[-1].stopped == 1

        # 2. Partial-init failure in MonitoringStack.start_monitoring() rolls back the Prometheus lease.
        failing_stack = MonitoringStack(info=lambda *a, **kw: None)
        monkeypatch.setattr("easydel.inference.esurge.engine.monitoring_stack.get_metrics_collector", lambda: object())

        def _boom_grafana(**kwargs):
            del kwargs
            raise RuntimeError("grafana startup crashed")

        failing_stack._start_grafana_service = _boom_grafana
        with pytest.raises(RuntimeError, match="grafana startup crashed"):
            failing_stack.start_monitoring(
                prometheus_port=18889,
                enable_prometheus=True,
                start_grafana=True,
                enable_console=False,
            )
        assert failing_stack.monitoring_active is False
        assert failing_stack.has_active_resources is False
        assert monitoring._monitoring_server is None
        assert created_servers[-1].running is False
        assert len(monitoring._monitoring_server_owners) == 0
    finally:
        monitoring.stop_monitoring()


def test_leak_23_prometheus_reap_urllib_close_and_partial_registry_rollback(monkeypatch):
    from easydel.inference.esurge import monitoring

    # 1. _stop_prometheus_server calls wait() after kill() to reap zombie subprocesses.
    prom_events: list[str] = []

    class _StubbornProm:
        def terminate(self):
            prom_events.append("terminate")

        def wait(self, timeout=None):
            del timeout
            if "kill" not in prom_events:
                raise subprocess.TimeoutExpired(cmd="prometheus", timeout=5)
            prom_events.append("wait_after_kill")
            return 0

        def kill(self):
            prom_events.append("kill")

    stack = MonitoringStack(info=lambda *a, **kw: None)
    stack._prometheus_process = _StubbornProm()
    stack._stop_prometheus_server()
    assert prom_events == ["terminate", "kill", "wait_after_kill"]
    assert stack._prometheus_process is None

    # 2. All 4 urllib.request.urlopen calls in _provision_running_grafana close their responses via `with`.
    opened_responses = []

    class _TrackedResp:
        def __init__(self, body: bytes = b"{}"):
            self._body = body
            self.entered = False
            self.exited = False
            opened_responses.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            self.exited = True
            return False

        def read(self):
            return self._body

    import urllib.error
    import urllib.request

    call_idx = 0

    def _fake_urlopen(req, timeout=5):
        nonlocal call_idx
        del timeout
        call_idx += 1
        url = req.full_url
        if url.endswith("/api/health"):
            return _TrackedResp(b'{"database":"ok"}')
        if url.endswith("/api/datasources") and req.get_method() == "POST":
            raise urllib.error.HTTPError(url, 409, "Conflict", hdrs=None, fp=None)
        if "/api/datasources/uid/" in url:
            return _TrackedResp(b"{}")
        if url.endswith("/api/dashboards/db"):
            return _TrackedResp(b'{"url":"/d/esurge-overview"}')
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    grafana_url = stack._provision_running_grafana(
        grafana_host="127.0.0.1",
        grafana_port=3000,
        grafana_admin_user="admin",
        grafana_admin_password="admin",
        datasource_name="eSurge",
        datasource_uid="esurge-uid",
        datasource_url="http://127.0.0.1:9090",
    )
    assert grafana_url == "http://127.0.0.1:3000"
    # Health + PUT update + dashboard POST all entered and exited context managers (3 responses;
    # also test non-409 POST path for the 4th urlopen call).
    assert len(opened_responses) == 3
    assert all(r.entered and r.exited for r in opened_responses)

    opened_responses.clear()

    def _fake_urlopen_create(req, timeout=5):
        del timeout
        if req.full_url.endswith("/api/dashboards/db"):
            return _TrackedResp(b'{"url":"/d/esurge-overview"}')
        return _TrackedResp(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen_create)
    assert (
        stack._provision_running_grafana(
            grafana_host="127.0.0.1",
            grafana_port=3000,
            grafana_admin_user="admin",
            grafana_admin_password="admin",
            datasource_name="eSurge",
            datasource_uid="esurge-uid",
            datasource_url="http://127.0.0.1:9090",
        )
        == "http://127.0.0.1:3000"
    )
    assert len(opened_responses) == 3
    assert all(r.entered and r.exited for r in opened_responses)

    # 3. Partial PrometheusMetrics.__init__ failure unregisters earlier collectors from global REGISTRY.
    if monitoring.PROMETHEUS_AVAILABLE:
        prefix = f"leak23_rollback_{time.monotonic_ns()}_"

        def _failing_info(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("boom during Info registration")

        with monkeypatch.context() as patch:
            patch.setattr(monitoring, "Info", _failing_info)
            with pytest.raises(RuntimeError, match="boom during Info registration"):
                monitoring.PrometheusMetrics(prefix=prefix)

        # Retry with the exact same prefix must succeed (zero leaked collectors in global REGISTRY!).
        recovered = monitoring.PrometheusMetrics(prefix=prefix)
        try:
            assert len(recovered._collectors) == 16
        finally:
            recovered.close()


# ---------------------------------------------------------------------------
# Leak Audit #16: RequestOutput.tool_call_delta_history bounded & drained/cleared
# ---------------------------------------------------------------------------


def test_leak_16_tool_call_delta_history_skipped_non_streaming_and_drained_on_stream():
    from easydel.inference.esurge.engine import output_pipeline as op_mod

    # 1. Non-streaming admission sets tool_call_delta_history=None; OutputPipeline accumulates 0 items
    #    while still surfacing delta_tool_calls on the step.
    registry = RequestRegistry()
    adm = _make_admission(registry, lambda recs: None)
    adm.add_request(
        "non-stream-req",
        "call tool",
        SamplingParams(max_tokens=8),
        prompt_token_ids=[1],
        track_tool_call_delta_history=False,
    )
    ro_ns = registry.get_output("non-stream-req")
    assert ro_ns is not None
    assert ro_ns.tool_call_delta_history is None

    step_counter = {"n": 0}

    class _RepeatingToolParser:
        reasoning_parser = None
        tool_parser = object()

        def process_delta(self, accumulated, delta, token_ids, prev_text, prev_ids):
            del accumulated, delta, token_ids, prev_text, prev_ids
            step_counter["n"] += 1
            c = DeltaToolCall(index=0, function=DeltaFunctionCall(arguments=f"chunk_{step_counter['n']}"))
            return SimpleNamespace(
                to_dict=lambda: {
                    "delta_reasoning": None,
                    "delta_content": "",
                    "accumulated_reasoning": "",
                    "accumulated_content": "",
                    "tool_calls": None,
                    "delta_tool_calls": [c],
                }
            )

        def process_final(self, accumulated, token_ids):
            del accumulated, token_ids
            return SimpleNamespace(
                tool_calls=None,
                to_dict=lambda: {
                    "delta_reasoning": None,
                    "delta_content": "",
                    "accumulated_reasoning": "",
                    "accumulated_content": "",
                    "tool_calls": None,
                    "delta_tool_calls": None,
                },
            )

    registry.get_record("non-stream-req").delegating_parser = _RepeatingToolParser()
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=_DummyDetokenizer(),
        eos_token_ids=[0],
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: (_ for _ in ()).throw(exc),
    )
    for tok in range(1, 20):
        pipeline.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("non-stream-req", [64 + tok])])})
    pipeline.submit(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("non-stream-req", [85], finish_reason=FinishReason.STOP)])}
    )
    assert ro_ns.tool_call_delta_history is None
    assert ro_ns.delta_tool_calls is not None and len(ro_ns.delta_tool_calls) == 1

    # 2. Streaming drains tool_call_delta_history on every snapshot read (O(1) memory) while
    #    preserving coalesced multi-step deltas, and clears on _release_output_consumer().
    reg2 = RequestRegistry()
    adm2 = _make_admission(reg2, lambda recs: None)
    adm2.add_request(
        "stream-req",
        "call tool",
        SamplingParams(max_tokens=16),
        prompt_token_ids=[1],
        track_tool_call_delta_history=True,
    )
    ro_s = reg2.get_output("stream-req")
    assert ro_s is not None and ro_s.tool_call_delta_history == []
    reg2.get_record("stream-req").delegating_parser = _RepeatingToolParser()
    pipe2 = OutputPipeline(
        registry=reg2,
        detokenizer_client=_DummyDetokenizer(),
        eos_token_ids=[0],
        decode_interval_tokens=1,
        decode_interval_secs=0.0,
        on_stop_strings=lambda stops: None,
        on_activity=lambda: None,
        on_fatal=lambda exc, tb: (_ for _ in ()).throw(exc),
    )

    eng = eSurge.__new__(eSurge)
    eng._registry = reg2
    eng._request_lock = reg2.request_lock
    eng._output_lock = reg2.output_lock
    eng._request_events = reg2.events
    eng._request_outputs = reg2.outputs
    eng._max_request_outputs = None
    eng._scheduler_running_fallback = True
    eng._engine_loop = None
    eng._worker_replay_thread = None
    eng._raise_if_scheduler_failed = lambda: None
    eng._recover_orphaned_request = lambda rid: False

    # Coalesce 2 steps before first read, then emit 2 more steps and finish.
    pipe2.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("stream-req", [65])])})
    pipe2.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("stream-req", [66])])})
    assert len(ro_s.tool_call_delta_history) == 2

    gen = eng._iter_request_snapshots("stream-req", stopped_message="stopped")
    snap1 = next(gen)
    assert len(snap1.delta_tool_calls) == 2
    assert len(ro_s.tool_call_delta_history) == 0  # drained immediately on read!

    pipe2.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("stream-req", [67])])})
    pipe2.submit(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("stream-req", [68], finish_reason=FinishReason.STOP)])}
    )
    snap2 = next(gen)
    assert snap2.finished is True
    assert len(snap2.delta_tool_calls) == 2
    assert len(ro_s.tool_call_delta_history) == 0
    gen.close()

    # 3. Stalled un-drained consumer history is capped at _MAX_TOOL_CALL_DELTA_HISTORY.
    ro_cap = RequestOutput(
        request_id="cap-req",
        prompt="p",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    comp = ro_cap.outputs[0]
    many = [DeltaToolCall(index=0, function=DeltaFunctionCall(arguments=str(i))) for i in range(op_mod._MAX_TOOL_CALL_DELTA_HISTORY + 50)]
    OutputPipeline._update_outputs(
        comp,
        ro_cap,
        0,
        {
            "delta_reasoning": None,
            "delta_content": "",
            "accumulated_reasoning": "",
            "accumulated_content": "",
            "tool_calls": None,
            "delta_tool_calls": many,
        },
        "",
        "",
    )
    assert len(ro_cap.tool_call_delta_history) == op_mod._MAX_TOOL_CALL_DELTA_HISTORY

    # 4. Direct ro.delta_tool_calls fallback (when tool_call_delta_history is empty/unused) delivers
    #    every distinct step 1, 2, ... without dropping step 2+ or replaying stale identical lists.
    reg4 = RequestRegistry()
    ro_fb = RequestOutput(
        request_id="fb-req",
        prompt="p",
        prompt_token_ids=[[1]],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    ev_fb = reg4.register("fb-req", RequestRecord(prompt="p", prompt_token_ids=[1]), ro_fb)
    eng4 = eSurge.__new__(eSurge)
    eng4._registry = reg4
    eng4._request_lock = reg4.request_lock
    eng4._output_lock = reg4.output_lock
    eng4._request_events = reg4.events
    eng4._request_outputs = reg4.outputs
    eng4._max_request_outputs = None
    eng4._scheduler_running_fallback = True
    eng4._engine_loop = None
    eng4._worker_replay_thread = None
    eng4._raise_if_scheduler_failed = lambda: None
    eng4._recover_orphaned_request = lambda rid: False

    c1 = [DeltaToolCall(index=0, function=DeltaFunctionCall(arguments="a"))]
    c2 = [DeltaToolCall(index=0, function=DeltaFunctionCall(arguments="b"))]
    ro_fb.delta_tool_calls = c1
    ro_fb.update_seq = 1
    ev_fb.set()
    gen4 = eng4._iter_request_snapshots("fb-req", stopped_message="stopped")
    s1 = next(gen4)
    assert s1.delta_tool_calls == c1

    # Step 2 with a new delta_tool_calls list must NOT be dropped.
    ro_fb.delta_tool_calls = c2
    ro_fb.update_seq = 2
    ev_fb.set()
    s2 = next(gen4)
    assert s2.delta_tool_calls == c2

    # Step 3 with unchanged stale list object must NOT replay c2.
    ro_fb.finished = True
    ro_fb.update_seq = 3
    ev_fb.set()
    s3 = next(gen4)
    assert s3.finished is True
    assert s3.delta_tool_calls is None
    gen4.close()


# ---------------------------------------------------------------------------
# Leak Audit #17 & #25: EngineCoreDriver media drop on _KIND_NEW & shutdown cleanup
# ---------------------------------------------------------------------------


def test_leak_17_and_25_core_driver_drops_raw_media_on_schedule_and_clears_on_shutdown():
    core = EngineCoreDriver(
        kv_cache_groups=_full_cache_group(page_size=4),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    pv = np.ones((4, 8), dtype=np.float32)
    grid = np.array([[1, 2, 2]], dtype=np.int64)
    vpv = np.ones((2, 8), dtype=np.float32)
    vgrid = np.array([[1, 1, 2]], dtype=np.int64)
    feat = MultiModalFeature.from_image(pv, grid)

    core.add_requests(
        [
            AdmissionRecord(
                request_id="vlm-live",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams(max_tokens=8),
                eos_token_id=0,
                pixel_values=pv,
                image_grid_thw=grid,
                pixel_values_videos=vpv,
                video_grid_thw=vgrid,
                mm_features=[feat],
            ),
            AdmissionRecord(
                request_id="vlm-wait-abort",
                prompt_token_ids=[4, 5, 6],
                sampling_params=SamplingParams(max_tokens=8),
                eos_token_id=0,
                pixel_values=np.ones((2, 8), dtype=np.float32),
                image_grid_thw=grid,
                mm_features=[MultiModalFeature.from_image(np.ones((2, 8), dtype=np.float32), grid)],
            ),
        ]
    )
    # Aborting a waiting request before schedule() immediately drops its heavy media refs.
    rec_wait = core.record_of("vlm-wait-abort")
    assert rec_wait is not None and rec_wait.pixel_values is not None
    core.finish_requests(["vlm-wait-abort"], FINISH_ABORT)
    assert rec_wait.pixel_values is None
    assert rec_wait.mm_features == []

    # Scheduling _KIND_NEW transfers media into NewRequestData and clears it on _records[h].
    so = core.schedule()
    assert len(so.scheduled_new_reqs) == 1
    new_req = so.scheduled_new_reqs[0]
    assert new_req.req_id == "vlm-live"
    assert new_req.pixel_values is pv
    assert new_req.pixel_values_videos is vpv
    assert new_req.mm_features == [feat]

    live_rec = core.record_of("vlm-live")
    assert live_rec is not None
    assert live_rec.pixel_values is None
    assert live_rec.image_grid_thw is None
    assert live_rec.pixel_values_videos is None
    assert live_rec.video_grid_thw is None
    assert live_rec.mm_features == []

    # #25: core.shutdown() clears all host bookkeeping dicts.
    core.image_expander = object()
    core.shutdown()
    assert core._handles == {}
    assert core._records == {}
    assert core._names == {}
    assert core._reports_due == {}
    assert core._steps == {}
    assert core._empty == []
    assert core._empty_freed == []
    assert core.image_expander is None
