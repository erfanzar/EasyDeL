"""Host-only engine-to-runner cache ownership regression (no JAX numerics)."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper

pytestmark = pytest.mark.cpu_ok


class Model:
    graphdef = object()

    @property
    def esurge_compatible_model(self):
        return self


class RunnerReady(Exception):
    def __init__(self, runner):
        self.runner = runner


@pytest.mark.parametrize("policy", ["enabled", "disabled", "undersized", "no_manager"])
def test_actual_engine_constructor_shares_manager_cache_and_invalidation(monkeypatch, policy):
    model = Model()
    tokenizer = SimpleNamespace(get_vocab=lambda: {}, eos_token_id=0)
    processor = None if policy == "no_manager" else object()
    assets = SimpleNamespace(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
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

    def manager_factory(**kwargs):
        # The engine has no enable_cache flag; exercise both manager policies at its construction seam.
        kwargs["enable_cache"] = policy != "disabled"
        return MultiModalManager(**kwargs)

    monkeypatch.setattr(esurge_engine, "MultiModalManager", manager_factory)

    def runner_factory(**kwargs):
        # Capture the real engine constructor kwargs; only device allocation
        # and graph execution are stubbed. Refresh uses the real runner method.
        runner = eSurgeRunner.__new__(eSurgeRunner)
        runner.model = kwargs["model"]
        runner.requests = {}
        runner.executor_manager = SimpleNamespace(update_graphs=lambda **kwargs: None)
        runner.vlm = VlmPrefillHelper(
            model_getter=lambda: runner.model,
            metadata=SimpleNamespace(data_parallel_size=1),
            compile_vision_encoder=False,
            vision_patch_buckets=None,
            max_num_batched_tokens=32,
            max_num_tokens=32,
            vision_cache=kwargs["vision_cache"],
        )
        raise RunnerReady(runner)

    monkeypatch.setattr(esurge_engine, "eSurgeRunner", runner_factory)
    engine = eSurge.__new__(eSurge)
    with pytest.raises(RunnerReady) as captured:
        engine.__init__(
            model="test",
            runtime={"max_num_batched_tokens": 32},
            vision={"vision_cache_capacity_mb": 0 if policy == "undersized" else 1},
        )
    runner = captured.value.runner
    manager = engine._multimodal_manager
    expected_cache = manager.cache if manager is not None else None
    assert runner.vlm.vision_cache is expected_cache
    if policy in ("disabled", "no_manager"):
        assert expected_cache is None
    else:
        # Empty caches are falsy, so truthiness-based forwarding is incorrect.
        assert expected_cache is not None and not expected_cache
        assert expected_cache.capacity_bytes == (0 if policy == "undersized" else 1024 * 1024)

    # Byte views, hashing, bookkeeping and cache round trips only: no JAX
    # arrays, numerical kernels or model forward are executed in this test.
    pixels = np.frombuffer(b"pixels", dtype=np.uint8)
    grid = np.frombuffer(b"grid", dtype=np.uint8)
    encoded = (np.frombuffer(b"embeds", dtype=np.uint8), (np.frombuffer(b"deepstack", dtype=np.uint8),))
    calls = []

    def encode():
        calls.append(None)
        return encoded

    def request(modality):
        return runner.vlm._encoded_vision_features(pixels, grid, modality=modality, encode=encode)

    for modality in ("image", "video"):
        for _ in range(2):
            result = request(modality)
            assert result[0].tobytes() == b"embeds"
            assert result[1][0].tobytes() == b"deepstack"
    assert len(calls) == (2 if policy == "enabled" else 4)
    if policy == "enabled":
        assert manager.get_cache_stats()["hits"] == 2
        manager.clear_cache()
        assert len(expected_cache) == 0
        request("image")
        assert len(calls) == 3
        # In-place weight refresh, same object: old encoder entries cannot hit.
        runner.update_model_weights(model=model, reset_state=False)
        request("image")
        assert runner.model is model
        assert len(calls) == 4
