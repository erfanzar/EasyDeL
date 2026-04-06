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

import hashlib
import pprint
import threading
from inspect import signature
from types import SimpleNamespace

import jax.numpy as jnp
from transformers.generation.configuration_utils import GenerationConfig

from easydel.inference.esurge.mixins.lifecycle import EngineLifecycleMixin
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import model_runner as model_runner_module
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.scheduler.output import CachedRequestData, SchedulerOutput
from easydel.inference.logits_process import (
    FrequencyPenaltyLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)
from easydel.inference.sampling_params import SamplingParams
from easydel.infra.mixins import generation as generation_module
from easydel.infra.mixins.generation import EasyGenerationMixin


def test_esurge_generation_surface_includes_required_engine_kwargs():
    get_esurge_params = signature(EasyGenerationMixin.get_esurge).parameters
    esurge_generate_params = signature(EasyGenerationMixin.esurge_generate).parameters
    pause_esurge_params = signature(EasyGenerationMixin.pause_esurge).parameters

    for name in ("data_parallelism_axis", "enable_prefix_caching", "max_num_seq_buckets"):
        assert name in get_esurge_params
        assert name in esurge_generate_params
    for name in ("release_model_state", "clear_compiled_cache"):
        assert name in pause_esurge_params


def test_get_esurge_refreshes_model_state_before_auto_resume(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self._paused = True
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.update_calls = 0
            self.resume_calls = 0

        def update_model_weights(self, _model):
            self.update_calls += 1

        def resume(self):
            # Regression guard: resume must never happen before state refresh.
            assert self.update_calls > 0
            self.resume_calls += 1
            self._paused = False

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})
    model = DummyModel()
    engine = DummyEngine()

    kwargs = dict(
        tokenizer="tok",
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    model_hash = model._esurge_cache_scope()
    extra_dict_str = pprint.pformat(kwargs)
    bytes_in = hashlib.md5(extra_dict_str.encode("utf-8")).digest()
    extra_dict_hash = int.from_bytes(bytes_in, byteorder="big", signed=True)
    generation_module._ESURGE_MAP_CACHE[f"{model_hash}-{extra_dict_hash}"] = engine

    resolved = model.get_esurge(**kwargs)

    assert resolved is engine
    assert engine.update_calls == 1
    assert engine.resume_calls == 1


def test_get_esurge_default_cache_refreshes_legacy_engine_before_auto_resume(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self._paused = True
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.update_calls = 0
            self.resume_calls = 0
            self.runner = type("Runner", (), {"model": None})()

        def update_model_weights(self, _model):
            self.update_calls += 1
            self.runner.model = object()

        def resume(self):
            assert self.update_calls > 0
            self.resume_calls += 1
            self._paused = False

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})
    model = DummyModel()
    engine = DummyEngine()

    generation_module._ESURGE_MAP_CACHE[f"{model._esurge_cache_scope()}-cached"] = engine

    resolved = model.get_esurge()

    assert resolved is engine
    assert engine.update_calls == 1
    assert engine.resume_calls == 1


def test_refresh_esurge_engine_weights_reuses_cached_graphdef_only_for_matching_source_layout():
    class DummyGraphDef:
        def __init__(self, fingerprint):
            self._fingerprint = fingerprint

        def __hash__(self):
            return self._fingerprint

    class DummyConfig:
        def __init__(self, marker):
            self.marker = marker

        def to_dict(self):
            return {"marker": self.marker}

    class DummyModel(EasyGenerationMixin):
        def __init__(self, graphdef):
            self.graphdef = graphdef
            self.config = DummyConfig("baseline-layout")

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self.calls = []
            self.runner = type(
                "Runner",
                (),
                {"executor_manager": type("ExecMgr", (), {"graphdef": "cached-engine-graphdef"})()},
            )()

        def update_model_weights(self, _model, *, restart_scheduler=True, graphdef=None):
            self.calls.append(
                {
                    "restart_scheduler": restart_scheduler,
                    "graphdef": graphdef,
                }
            )

    model = DummyModel(DummyGraphDef(11))
    engine = DummyEngine()
    model._remember_esurge_engine_source_graphdef(engine, model.graphdef)

    model._refresh_esurge_engine_weights(engine)
    assert engine.calls[-1]["graphdef"] == "cached-engine-graphdef"

    model.graphdef = DummyGraphDef(22)
    model.config = DummyConfig("changed-layout")
    model._refresh_esurge_engine_weights(engine)
    assert engine.calls[-1]["graphdef"] is None


def test_refresh_esurge_engine_weights_reuses_cached_graphdef_when_layout_signature_matches():
    class DummyGraphDef:
        def __init__(self, fingerprint):
            self._fingerprint = fingerprint

        def __hash__(self):
            return self._fingerprint

    class DummyConfig:
        def __init__(self, marker):
            self.marker = marker

        def to_dict(self):
            return {"marker": self.marker}

    class DummyModel(EasyGenerationMixin):
        def __init__(self, *, graphdef_fingerprint, config_marker):
            self.graphdef = DummyGraphDef(graphdef_fingerprint)
            self.config = DummyConfig(config_marker)

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self.calls = []
            self.runner = type(
                "Runner",
                (),
                {"executor_manager": type("ExecMgr", (), {"graphdef": "cached-engine-graphdef"})()},
            )()

        def update_model_weights(self, _model, *, restart_scheduler=True, graphdef=None):
            self.calls.append(
                {
                    "restart_scheduler": restart_scheduler,
                    "graphdef": graphdef,
                }
            )

    model = DummyModel(graphdef_fingerprint=11, config_marker="same-layout")
    engine = DummyEngine()
    model._remember_esurge_engine_source_graphdef(engine, model.graphdef)

    model.graphdef = DummyGraphDef(22)
    model._refresh_esurge_engine_weights(engine)
    assert engine.calls[-1]["graphdef"] == "cached-engine-graphdef"

    model.config = DummyConfig("changed-layout")
    model.graphdef = DummyGraphDef(33)
    model._refresh_esurge_engine_weights(engine)
    assert engine.calls[-1]["graphdef"] is None


def test_source_layout_signature_does_not_touch_esurge_compatible_model():
    class DummyConfig:
        def to_dict(self):
            return {"marker": "baseline-layout"}

    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = DummyConfig()

        def static_hash(self, _ignored):
            return "dummy-model"

        @property
        def graphstate_type(self):
            return int

        @property
        def esurge_compatible_model(self):
            raise AssertionError("layout signature must not construct esurge_compatible_model")

    model = DummyModel()

    signature = model._source_layout_signature_for_esurge_metadata()

    assert isinstance(signature, str)
    assert signature


def test_lifecycle_update_graphdef_prefers_overridden_esurge_graphdef():
    class WrapperLike:
        graphdef = "wrapper-graphdef"

        @property
        def esurge_graphdef(self):
            return "delegated-esurge-graphdef"

        def _esurge_graphdef_from_graphdef(self, _graphdef):
            raise AssertionError("wrapper helper should not be used when esurge_graphdef is overridden")

    resolved = EngineLifecycleMixin._resolve_graphdef_for_weight_update(
        WrapperLike(),
        split_graphdef="split-wrapper-graphdef",
    )

    assert resolved == "delegated-esurge-graphdef"


def test_lifecycle_split_graph_components_prefers_compatible_model_for_wrapper_delegate():
    class CompatibleModel:
        def split_module(self):
            return ("compatible-graphdef", "compatible-graphstate", "compatible-graphother")

    class WrapperLike:
        def __init__(self):
            self.model = CompatibleModel()

        def split_module(self):
            return ("wrapper-graphdef", "wrapper-graphstate", "wrapper-graphother")

        @property
        def esurge_graphdef(self):
            return "delegated-esurge-graphdef"

        @property
        def esurge_compatible_model(self):
            return self.model

    split_model, graphdef, graphstate, graphother = EngineLifecycleMixin._split_graph_components_for_weight_update(
        WrapperLike()
    )

    assert split_model.__class__.__name__ == "CompatibleModel"
    assert (graphdef, graphstate, graphother) == (
        "compatible-graphdef",
        "compatible-graphstate",
        "compatible-graphother",
    )


def test_lifecycle_prefetch_waits_for_async_decode_batches():
    request = EngineRequest(
        request_id="req-decode",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        eos_token_id=2,
    )
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1
    scheduler = SimpleNamespace(requests={request.request_id: request})
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        num_common_prefix_pages=[],
        finished_req_ids=set(),
        async_scheduling=True,
    )

    assert not EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, scheduler_output)


def test_lifecycle_prefetch_allows_pure_prefill_batches():
    request = EngineRequest(
        request_id="req-prefill",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=4),
        eos_token_id=2,
    )
    request.num_computed_tokens = request.num_tokens - 1
    scheduler = SimpleNamespace(requests={request.request_id: request})
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        num_common_prefix_pages=[],
        finished_req_ids=set(),
        async_scheduling=True,
    )

    assert EngineLifecycleMixin._can_prefetch_scheduler_output(scheduler, scheduler_output)


def test_lifecycle_aborts_after_prefetched_overlap_drain_failure():
    request = EngineRequest(
        request_id="req-prefill",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(max_tokens=4),
        eos_token_id=2,
    )
    request.num_computed_tokens = request.num_tokens - 2

    current_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        num_common_prefix_pages=[],
        finished_req_ids=set(),
        async_scheduling=True,
    )
    prefetched_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        num_common_prefix_pages=[],
        finished_req_ids=set(),
        async_scheduling=True,
    )

    class DummyScheduler:
        def __init__(self):
            self.requests = {request.request_id: request}
            self.running = []
            self.waiting = []
            self.schedule_calls = 0

        def schedule(self):
            self.schedule_calls += 1
            if self.schedule_calls == 1:
                return current_output
            if self.schedule_calls == 2:
                return prefetched_output
            raise AssertionError("scheduler loop should stop after aborting the prefetched overlap path")

    class DummyRunner:
        def __init__(self):
            self.executor_manager = SimpleNamespace(kv_pages=object())
            self.async_dispatches = 0

        def execute_model_async(self, scheduler_output):
            del scheduler_output
            self.async_dispatches += 1
            return "future"

        def wait_for_execution(self, future):
            del future
            raise RuntimeError("drain failed")

        def shutdown(self):
            pass

    class DummyEngine(EngineLifecycleMixin):
        def __init__(self):
            self._scheduler_lock = threading.Lock()
            self._request_lock = threading.Lock()
            self._output_lock = threading.Lock()
            self._output_event = threading.Event()
            self._request_events = {}
            self._active_requests = {}
            self._request_outputs = {}
            self._finished_request_ids = set()
            self._scheduler_running = False
            self._scheduler_thread = None
            self._scheduler_exception = None
            self._scheduler_exception_tb = None
            self._scheduler_heartbeat = None
            self._profiling_active = False
            self._profiling_steps_remaining = 0
            self._profiling_output_dir = None
            self._profiling_host_level = None
            self._profiling_python_level = None
            self._paused = True
            self._overlap_execution = True
            self._distributed_controller = None
            self._kv_cache_valid = True
            self.runner = DummyRunner()
            self.scheduler = DummyScheduler()

        def _touch_activity(self):
            pass

        def _start_idle_monitor(self):
            pass

        def _stop_idle_monitor(self):
            pass

        def _info(self, *_args, **_kwargs):
            pass

        def _update_scheduler_heartbeat(self):
            pass

        def _process_engine_outputs(self, _outputs):
            raise AssertionError("engine outputs should not be processed after the drain failure")

        def _handle_profiling_step(self):
            pass

        def _install_signal_diagnostics(self):
            pass

        def _is_nonrecoverable_scheduler_error(self, _exc):
            return False

        def _reset_runner_state_if_idle(self, _reason):
            pass

    engine = DummyEngine()
    engine.initiate()
    assert engine._scheduler_thread is not None
    engine._scheduler_thread.join(timeout=1.0)

    assert engine._scheduler_thread is not None
    assert not engine._scheduler_thread.is_alive()
    assert not engine._scheduler_running
    assert isinstance(engine._scheduler_exception, RuntimeError)
    assert "drain failed" in str(engine._scheduler_exception)
    assert engine.scheduler.schedule_calls == 2
    assert engine.runner.async_dispatches == 1


def test_model_runner_update_model_weights_replaces_explicit_graphdef_with_compatible_graphdef(monkeypatch):
    raw_graphdef = object()
    compatible_graphdef = object()

    class CompatibleModel:
        graphdef = compatible_graphdef

    class RawModel:
        @property
        def esurge_compatible_model(self):
            return CompatibleModel()

    monkeypatch.setattr(model_runner_module.flax.nnx, "merge", lambda graphdef, graphstate, graphother: RawModel())

    class DummyExecutorManager:
        def __init__(self):
            self.calls = []

        def update_graphs(self, **kwargs):
            self.calls.append(kwargs)

    class DummyRunner:
        def __init__(self):
            self.requests = []
            self.executor_manager = DummyExecutorManager()
            self.model = None
            self.setup_calls = 0

        def _setup_variables(self):
            self.setup_calls += 1

    runner = DummyRunner()

    eSurgeRunner.update_model_weights(
        runner,
        model=None,
        graphdef=raw_graphdef,
        graphstate="graphstate",
        graphother="graphother",
        reset_state=True,
    )

    assert isinstance(runner.model, CompatibleModel)
    assert runner.executor_manager.calls[0]["graphdef"] is compatible_graphdef
    assert runner.executor_manager.calls[0]["graphstate"] == "graphstate"
    assert runner.executor_manager.calls[0]["graphother"] == "graphother"


def test_get_esurge_preserves_existing_positional_argument_order():
    params = list(signature(EasyGenerationMixin.get_esurge).parameters)
    assert params.index("max_num_batched_tokens") < params.index("max_num_seq_buckets")
    assert params.index("silent_mode") < params.index("data_parallelism_axis")


def test_resume_esurge_refreshes_model_state_before_resuming(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self._paused = True
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.update_calls = 0
            self.resume_calls = 0
            self.runner = type("Runner", (), {"model": None})()

        def update_model_weights(self, _model, *, restart_scheduler=True):
            assert restart_scheduler is False
            self.update_calls += 1
            self.runner.model = object()

        def resume(self):
            assert self.runner.model is not None
            self.resume_calls += 1
            self._paused = False

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})
    model = DummyModel()
    engine = DummyEngine()

    generation_module._ESURGE_MAP_CACHE[f"{model._esurge_cache_scope()}-resume"] = engine
    model.resume_esurge()

    assert engine.update_calls == 1
    assert engine.resume_calls == 1


def test_get_esurge_does_not_inherit_buckets_when_max_num_seqs_is_explicit(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self, cached_engine):
            self._cached_engine = cached_engine
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

        def get_relevant_esurge(self, tokenizer=None, max_num_seqs=None):
            return self._cached_engine

    class CachedEngine:
        def __init__(self):
            self.runner = type("Runner", (), {"max_num_seq_buckets": [1, 2, 4, 8]})()

    class ExpectedEngine:
        def __init__(self):
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self._paused = False
            self.update_calls = 0

        def update_model_weights(self, _model):
            self.update_calls += 1

    def _unexpected_esurge_ctor(*_args, **_kwargs):
        raise AssertionError("Unexpected eSurge construction; expected cached key lookup to succeed.")

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})
    monkeypatch.setattr("easydel.inference.eSurge", _unexpected_esurge_ctor)

    model = DummyModel(CachedEngine())
    expected_engine = ExpectedEngine()

    kwargs = dict(
        tokenizer="tok",
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=16,
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    model_hash = model._esurge_cache_scope()
    extra_dict = dict(
        tokenizer=kwargs["tokenizer"],
        max_model_len=kwargs["max_model_len"],
        min_input_pad=kwargs["min_input_pad"],
        max_num_seqs=kwargs["max_num_seqs"],
        max_num_seq_buckets=None,
        max_num_batched_tokens=kwargs["max_num_batched_tokens"],
        hbm_utilization=kwargs["hbm_utilization"],
        page_size=kwargs["page_size"],
        enable_prefix_caching=kwargs["enable_prefix_caching"],
        data_parallelism_axis=kwargs["data_parallelism_axis"],
        runner_verbose=kwargs["runner_verbose"],
        decode_truncated_prompt=kwargs["decode_truncated_prompt"],
        destroy_pages_on_pause=kwargs["destroy_pages_on_pause"],
        silent_mode=kwargs["silent_mode"],
    )
    extra_dict_str = pprint.pformat(extra_dict)
    bytes_in = hashlib.md5(extra_dict_str.encode("utf-8")).digest()
    extra_dict_hash = int.from_bytes(bytes_in, byteorder="big", signed=True)
    generation_module._ESURGE_MAP_CACHE[f"{model_hash}-{extra_dict_hash}"] = expected_engine

    resolved = model.get_esurge(**kwargs)

    assert resolved is expected_engine
    assert expected_engine.update_calls == 1


def test_get_esurge_skips_redundant_refresh_for_fresh_engine(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self):
            self._paused = False
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.update_calls = 0
            self.runner = type("Runner", (), {"model": object()})()

        def update_model_weights(self, _model, *, restart_scheduler=True):
            del restart_scheduler
            self.update_calls += 1

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})

    created_engines: list[DummyEngine] = []

    def _fake_esurge_ctor(*_args, **_kwargs):
        engine = DummyEngine()
        created_engines.append(engine)
        return engine

    monkeypatch.setattr("easydel.inference.eSurge", _fake_esurge_ctor)

    model = DummyModel()
    engine = model.get_esurge(
        tokenizer="tok",
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    assert engine is created_engines[0]
    assert engine.update_calls == 0


def test_get_esurge_new_engine_does_not_require_graphdef_for_fingerprint(monkeypatch):
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

        @property
        def graphdef(self):
            raise AttributeError("graphdef unavailable")

    class DummyEngine:
        def __init__(self):
            self._paused = False
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})

    created_engines: list[DummyEngine] = []

    def _fake_esurge_ctor(*_args, **_kwargs):
        engine = DummyEngine()
        created_engines.append(engine)
        return engine

    monkeypatch.setattr("easydel.inference.eSurge", _fake_esurge_ctor)

    model = DummyModel()
    engine = model.get_esurge(
        tokenizer="tok",
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    assert engine is created_engines[0]
    assert not hasattr(engine, "_easydel_source_graphdef_fingerprint")


def test_get_esurge_reuses_cached_engine_for_equivalent_tokenizer_instances(monkeypatch):
    class DummyProcessor:
        def __init__(self, name_or_path, *, padding_side="right"):
            self.name_or_path = name_or_path
            self.padding_side = padding_side

    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self, processor):
            self._paused = False
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.update_calls = 0
            self.processor = processor
            self.tokenizer = processor
            self.runner = type("Runner", (), {"model": object()})()

        def update_model_weights(self, _model, *, restart_scheduler=True):
            del restart_scheduler
            self.update_calls += 1

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})

    created_engines: list[DummyEngine] = []

    def _fake_esurge_ctor(*_args, **kwargs):
        engine = DummyEngine(kwargs["tokenizer"])
        created_engines.append(engine)
        return engine

    monkeypatch.setattr("easydel.inference.eSurge", _fake_esurge_ctor)

    model = DummyModel()
    first_processor = DummyProcessor("tok")
    second_processor = DummyProcessor("tok")

    first = model.get_esurge(
        tokenizer=first_processor,
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )
    second = model.get_esurge(
        tokenizer=second_processor,
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    assert first is second
    assert len(created_engines) == 1
    assert second.processor is second_processor
    assert second.tokenizer is second_processor


def test_get_esurge_reuses_cached_engine_when_tokenizer_padding_side_changes(monkeypatch):
    class DummyProcessor:
        def __init__(self, name_or_path, *, padding_side):
            self.name_or_path = name_or_path
            self.padding_side = padding_side

    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "granted_freq_max_position_embedding": 1024,
                },
            )()

        def static_hash(self, _ignored):
            return "dummy-model"

    class DummyEngine:
        def __init__(self, processor):
            self._paused = False
            self.silent_mode = True
            self.num_running_requests = 0
            self.num_pending_requests = 0
            self.processor = processor
            self.tokenizer = processor
            self.runner = type("Runner", (), {"model": object()})()

        def update_model_weights(self, _model, *, restart_scheduler=True):
            del restart_scheduler

    monkeypatch.setattr(generation_module, "_ESURGE_MAP_CACHE", {})

    created_engines: list[DummyEngine] = []

    def _fake_esurge_ctor(*_args, **kwargs):
        engine = DummyEngine(kwargs["tokenizer"])
        created_engines.append(engine)
        return engine

    monkeypatch.setattr("easydel.inference.eSurge", _fake_esurge_ctor)

    model = DummyModel()
    first = model.get_esurge(
        tokenizer=DummyProcessor("tok", padding_side="right"),
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )
    second_processor = DummyProcessor("tok", padding_side="left")
    second = model.get_esurge(
        tokenizer=second_processor,
        max_model_len=512,
        min_input_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[1, 2, 4, 8],
        max_num_batched_tokens=64,
        hbm_utilization=0.5,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        runner_verbose=False,
        decode_truncated_prompt=True,
        destroy_pages_on_pause=True,
        silent_mode=True,
    )

    assert first is second
    assert len(created_engines) == 1
    assert second.processor is second_processor
    assert second.tokenizer is second_processor


def test_generate_accepts_penalty_kwargs_for_compiled_generation():
    class DummyModel(EasyGenerationMixin):
        def __init__(self):
            self.config = type(
                "Cfg",
                (),
                {
                    "is_encoder_decoder": False,
                    "max_position_embeddings": 128,
                },
            )()
            self.generation_config = GenerationConfig(
                do_sample=True,
                max_new_tokens=1,
                pad_token_id=0,
                eos_token_id=2,
            )

        def _sample(
            self,
            input_ids,
            max_length,
            pad_token_id,
            eos_token_id,
            prng_key,
            *,
            logits_warper=None,
            logits_processor=None,
            trace=True,
            model_kwargs=None,
        ):
            del input_ids, max_length, pad_token_id, eos_token_id, prng_key, logits_warper, trace, model_kwargs
            return logits_processor

    model = DummyModel()
    processors = model.generate(
        jnp.asarray([[1, 2]], dtype=jnp.int32),
        do_sample=True,
        max_new_tokens=1,
        presence_penalty=0.4,
        frequency_penalty=0.2,
        repetition_penalty=1.3,
        trace=False,
    )

    assert any(isinstance(processor, PresencePenaltyLogitsProcessor) for processor in processors)
    assert any(isinstance(processor, FrequencyPenaltyLogitsProcessor) for processor in processors)
    assert any(isinstance(processor, RepetitionPenaltyLogitsProcessor) for processor in processors)
