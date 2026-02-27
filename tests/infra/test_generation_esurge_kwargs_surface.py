import hashlib
import pprint
from inspect import signature

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
