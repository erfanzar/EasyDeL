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

"""Host-only admission, scheduling and composition contracts (no JAX arrays)."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.config import eSurgeContextConfig, eSurgeRuntimeConfig
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.engine.registry import RequestRegistry
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.reasoning.parsers.gemma4_reasoning_parser import Gemma4ReasoningParser
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


class Tokenizer:
    eos_token_id = 0

    def get_vocab(self):
        return {"<|channel>": 1, "<channel|>": 2, "hello": 3}

    def __call__(self, text, **kwargs):
        return {"input_ids": [3]}


@pytest.mark.parametrize("n", [1, 3])
def test_admission_preserves_reasoning_channel_tokens_for_every_sample(n):
    registry = RequestRegistry()
    scheduled = []
    admission = RequestAdmission(
        registry=registry,
        scheduler_submit=scheduled.extend,
        tokenizer_client=SimpleNamespace(tokenize=lambda *args: [3]),
        tokenizer=Tokenizer(),
        context_config=eSurgeContextConfig.coerce_config(None),
        reserve_tokens=1,
        max_model_len=64,
        tool_parser_class=None,
        reasoning_parser_class=Gemma4ReasoningParser,
        sampling_params_callback=lambda: None,
        generation_config_dict={},
        primary_eos_token_id=0,
        eos_token_ids=[0],
        extra_stops=[],
        ignore_stop_strings_in_reasoning=True,
        on_activity=lambda: None,
        info=lambda *args, **kwargs: None,
    )
    admission.add_request("request", "hello", SamplingParams(max_tokens=4, n=n, skip_special_tokens=True))
    assert len(scheduled) == n
    for request in scheduled:
        assert request.sampling_params.skip_special_tokens is False
        assert registry.get_record(request.request_id).sampling_params.skip_special_tokens is False
    assert registry.get_record("request").sampling_params.skip_special_tokens is False


def test_public_alive_tracks_local_loop_and_worker_replay():
    engine = eSurge.__new__(eSurge)
    assert engine.alive is False
    engine._scheduler_running = True
    assert engine.alive is True
    engine._scheduler_running = False
    replay = SimpleNamespace(is_alive=lambda: True)
    engine._worker_replay_thread = replay
    assert engine.alive is True
    replay.is_alive = lambda: False
    assert engine.alive is False


@pytest.mark.parametrize("work", ["finished", "preempted", "new", "cached", None])
def test_zero_token_schedule_retains_all_bookkeeping_cases(work):
    output = SimpleNamespace(
        finished_req_ids={"r"} if work == "finished" else set(),
        preempted_req_ids={"r"} if work == "preempted" else set(),
        scheduled_new_reqs=["r"] if work == "new" else [],
        scheduled_cached_reqs=SimpleNamespace(num_reqs=int(work == "cached")),
    )
    assert EngineLoop._zero_schedule_needs_update(output) is (work is not None)


def test_engine_scheduler_rebuild_retains_optional_capabilities():
    engine = eSurge.__new__(eSurge)
    engine.runtime_config = eSurgeRuntimeConfig.from_dict(
        chunked_prefill_enabled=True, token_safety_margin=999, include_finished_set=True
    )
    engine.cache_config = SimpleNamespace(enable_prefix_caching=False)
    engine._scheduler_max_num_batched_tokens = 6
    engine._runner_async_scheduling = False
    engine._long_prefill_token_threshold = None
    engine.runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False
                ),
                layer_names=None,
            )
        ],
        metadata=SimpleNamespace(num_pages=64, page_size=4, data_parallel_size=1),
        max_num_seqs=4,
        max_model_len=64,
        max_num_seq_buckets=[4],
        num_draft_tokens=0,
    )
    # Both initial construction and a rebuild use the same real factory.
    for _ in range(2):
        scheduler = engine._build_scheduler()
        try:
            for rid in ("a", "b"):
                scheduler.add_request(EngineRequest(rid, [1, 2, 3, 4], SamplingParams(max_tokens=4), eos_token_id=0))
            assert scheduler.schedule().num_scheduled_tokens == {"a": 4, "b": 2}
            # Safety reserves govern fresh admission, not progress in pages
            # already owned by b's unfinished prefill. Blocking that progress
            # would strand the request without any way to release its pages.
            assert scheduler.schedule().num_scheduled_tokens == {"b": 2}
            assert scheduler.schedule().total_num_scheduled_tokens == 0
            scheduler.finish_requests(["a", "b"], EngineRequestStatus.FINISHED_ABORTED)
            outputs = scheduler.update_from_output(
                scheduler.schedule(),
                ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                    spec_token_ids=None,
                    logprobs=None,
                    prompt_logprobs_dict={},
                ),
            )
            assert outputs[0].finished_requests == {"a", "b"}
        finally:
            scheduler.shutdown()


class MultimodalChat(eSurge):
    def __init__(self, manager):
        self._multimodal_manager = manager
        self.tokenizer = Tokenizer()
        self.admitted = None

    def _format_chat_prompt(self, *args, **kwargs):
        return "hello"

    def _prepare_sampling_params_for_request(self, params, **kwargs):
        return params

    def _build_tool_parser_request(self, **kwargs):
        return None

    def _add_request(self, **kwargs):
        self.admitted = kwargs

    def _ensure_scheduler_running(self, **kwargs):
        pass

    def _wait_for_request(self, request_id):
        return self.admitted


class NoCacheMediaManager:
    def __init__(self):
        self.image_calls = 0
        self.video_calls = 0
        self.image_pixels, self.image_grid = object(), object()
        self.video_pixels, self.video_grid = object(), object()

    def extract_media_from_messages(self, messages):
        return ["image"], ["video"]

    def process_images(self, images):
        self.image_calls += 1
        return self.image_pixels, self.image_grid

    def process_videos(self, videos):
        self.video_calls += 1
        return self.video_pixels, self.video_grid

    def process_images_to_features(self, images, *, processed=None):
        pixels, grid = self.process_images(images) if processed is None else processed
        return [("image", pixels, grid)]

    def process_videos_to_features(self, videos, *, processed=None):
        pixels, grid = self.process_videos(videos) if processed is None else processed
        return [("video", pixels, grid)]

    def tokenize_multimodal(self, **kwargs):
        return [3]


def test_multimodal_chat_reuses_pixels_without_any_cache():
    manager = NoCacheMediaManager()
    engine = MultimodalChat(manager)
    result = engine._chat_multimodal([], request_id="media")
    assert result["pixel_values"] is manager.image_pixels
    assert result["pixel_values_videos"] is manager.video_pixels
    assert result["mm_features"] == [
        ("image", manager.image_pixels, manager.image_grid),
        ("video", manager.video_pixels, manager.video_grid),
    ]
    assert manager.image_calls == manager.video_calls == 1


class RunnerOptions(Exception):
    def __init__(self, options):
        self.options = options


@pytest.mark.parametrize(
    "route", ["constructor", "runtime", "aggregate", "elarge_flat", "elarge_runtime", "override_zero"]
)
def test_public_engine_forwards_speculation_options(monkeypatch, route):
    assets = SimpleNamespace(
        model=SimpleNamespace(esurge_compatible_model=object()),
        processor=None,
        tokenizer=Tokenizer(),
        tokenizer_source="test",
        tool_parser_name=None,
        reasoning_parser_name=None,
        tool_parser_class=None,
        reasoning_parser_class=None,
        drafter=None,
        model_name="test",
    )
    monkeypatch.setattr(esurge_engine, "build_engine_assets", lambda **kwargs: assets)
    monkeypatch.setattr(eSurge, "_apply_data_parallel_axis_to_model", lambda *args: None)
    monkeypatch.setattr(eSurge, "_warn_if_data_parallel_axis_shards_the_model", lambda *args: None)
    monkeypatch.setattr(
        esurge_engine,
        "WorkerManager",
        lambda *args, **kwargs: SimpleNamespace(
            start=lambda **kwargs: (None, None), tokenizer_endpoint=None, detokenizer_endpoint=None
        ),
    )

    def runner(**kwargs):
        raise RunnerOptions(kwargs)

    monkeypatch.setattr(esurge_engine, "eSurgeRunner", runner)
    options = {"spec_debug_max_traces": 7, "spec_adaptive_conf_threshold": 0.75}
    runtime = {"max_num_batched_tokens": 32, **options}
    expected = options
    if route == "constructor":
        kwargs = {"runtime": {"max_num_batched_tokens": 32}, **options}
    elif route == "runtime":
        kwargs = {"runtime": eSurgeRuntimeConfig.from_dict(runtime)}
    elif route == "aggregate":
        kwargs = {"config": {"runtime": runtime}}
    elif route == "override_zero":
        expected = {"spec_debug_max_traces": 0, "spec_adaptive_conf_threshold": 0.0}
        kwargs = {"runtime": runtime, **expected}
    else:
        from easydel.infra.elarge.builders import to_esurge_kwargs

        esurge = runtime if route == "elarge_flat" else {"runtime": runtime}
        kwargs = to_esurge_kwargs({"model": {"name_or_path": "test"}, "esurge": esurge})
    with pytest.raises(RunnerOptions) as captured:
        eSurge(model="test", **kwargs)
    for key, value in expected.items():
        assert captured.value.options[key] == value


@pytest.mark.parametrize(
    "options",
    [
        {"spec_debug_max_traces": -1},
        {"spec_debug_max_traces": 0.5},
        {"spec_debug_max_traces": True},
        {"spec_adaptive_conf_threshold": -0.1},
        {"spec_adaptive_conf_threshold": 1.1},
        {"spec_adaptive_conf_threshold": float("nan")},
        {"spec_adaptive_conf_threshold": float("inf")},
    ],
)
def test_public_engine_rejects_invalid_speculation_options_before_loading(options):
    with pytest.raises(ValueError, match="spec_"):
        eSurge(model="not-loaded", **options)
