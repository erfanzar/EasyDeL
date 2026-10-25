"""CPU-only orchestration regressions for compile diagnostics controls and cache-page handoff.

Executors and device transfers are host stubs: these tests exercise compile
bookkeeping and configuration, not JAX array creation or compiled numerics.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.executors import batch_preparer, compile_orchestrator

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("debug_mode", [False, True])
@pytest.mark.parametrize("use_aot", [False, True])
@pytest.mark.parametrize("variant", ["model_step", "pipeline_model_step", "backbone", "sampler"])
def test_compile_diagnostics_are_opt_in(monkeypatch, debug_mode, use_aot, variant):
    compiled = []
    runtime_pages = object()
    inputs = SimpleNamespace(batch_metadata=object(), rng_key=np.array([0, 1], dtype=np.uint32))

    def compile_model(**kwargs):
        compiled.append(kwargs)
        return SimpleNamespace(kv_pages=runtime_pages)

    def compile_sampler(**kwargs):
        compiled.append(kwargs)

    model_executor = SimpleNamespace(
        has_model_step=lambda *args: False,
        supports_pipeline_model_step=True,
        compile_model_step=compile_model,
        compile_pipeline_model_step=compile_model,
        compile_backbone=compile_model,
    )
    sampler_executor = SimpleNamespace(
        cache_key=lambda **kwargs: bool(kwargs.get("greedy", False)),
        has=lambda key: False,
        compile=compile_sampler,
    )
    manager = SimpleNamespace(
        _debug_mode=debug_mode,
        _debug_baselines={},
        use_aot_forward=use_aot,
        kv_pages=object(),
        rng_key=inputs.rng_key,
        model=SimpleNamespace(
            mesh=SimpleNamespace(is_mpmd=False),
            config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=11)),
            dtype=np.float32,
        ),
        _model_executor=model_executor,
        _sampler_executor=sampler_executor,
        _has_backbone_variant=lambda *args, **kwargs: False,
        get_compile_configurations=lambda *args: (object(), {"weights": np.zeros((2, 3), dtype=np.float32)}, {}, inputs),
        _sampler_sharding=None,
        _sampler_zero_token_counts=None,
    )

    original_hash = compile_orchestrator._tree_hash

    def enabled_hash(tree):
        assert debug_mode and use_aot, "disabled diagnostics must not hash metadata"
        return original_hash(tree)

    monkeypatch.setattr(compile_orchestrator, "_tree_hash", enabled_hash)
    monkeypatch.setattr(compile_orchestrator.jax, "device_put", lambda value, sharding: value)
    orchestrator = compile_orchestrator.CompileOrchestrator(manager)
    kwargs = dict(num_tokens=8, padded_num_reqs=4, max_num_reqs=4, metadata=None)
    if variant == "backbone":
        kwargs.pop("padded_num_reqs")
    getattr(orchestrator, f"_compile_{variant}_variant")(**kwargs)

    # Disabling diagnostics does not disable compilation or cache-page handoff.
    assert len(compiled) == (2 if variant == "sampler" else 1)
    if variant != "sampler":
        assert manager.kv_pages is runtime_pages
    if debug_mode and use_aot:
        suffix = "pp_model_step" if variant == "pipeline_model_step" else variant
        prefix = "8" if variant == "backbone" else "8_4"
        assert list(manager._debug_baselines) == [f"{prefix}_hash_in_{suffix}"]
        # Real metadata hashing remains available for enabled diagnostics.
        assert "float32" in str(manager._debug_baselines)
        assert ("(4,)" if variant == "sampler" else "(2, 3)") in str(manager._debug_baselines)
        if variant == "sampler":
            # Runtime compares a compact host window, not packed device args.
            i32 = np.arange(4, dtype=np.int32)
            f32 = np.ones(4, dtype=np.float32)
            runtime_inputs = (
                4,
                3,
                7,
                i32,
                i32,
                i32,
                i32,
                i32,
                i32,
                np.ones(4, dtype=np.bool_),
                f32,
                f32,
                i32,
                f32,
                f32,
                f32,
                f32,
            )
            differences = compile_orchestrator._tree_hash_diff(
                manager._debug_baselines["8_4_hash_in_sampler"], compile_orchestrator._tree_hash(runtime_inputs)
            )
            assert len(differences) == 17
            assert all(differences[index] for index in range(3, 17))
    else:
        assert manager._debug_baselines == {}


@pytest.mark.parametrize("setting,enabled", [(None, True), ("0", False), ("false", False), ("yes", True)])
def test_dp_local_page_configuration_preserves_default_and_overrides(monkeypatch, setting, enabled):
    name = "EASYDEL_ENABLE_DP_LOCAL_PAGE_PATH"
    if setting is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, setting)
    # Construction normally transfers placeholders. Keep this test host-only.
    monkeypatch.setattr(batch_preparer.jax, "device_put", lambda value, sharding: value)
    preparer = batch_preparer.BatchMetadataPreparer(
        metadata=SimpleNamespace(
            version="v3",
            data_parallel_size=2,
            get_max_num_seqs=lambda: 4,
            max_num_pages_per_req=2,
            num_pages=8,
            page_size=16,
        ),
        empty_sharding=None,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=16,
        min_input_pad=1,
    )
    # Request row 0 belongs to shard 0 (pages 1..3), not shard 1's page 4.
    kwargs = dict(
        num_requests=1,
        scheduled=np.array([1]),
        num_computed_tokens_cpu=np.array([0]),
        page_table_cpu=np.array([[4, 0]], dtype=np.int32),
    )
    if enabled:
        with pytest.raises(ValueError, match="Non-DP-local page IDs"):
            preparer._enforce_dp_local_page_tables(**kwargs)
    else:
        preparer._enforce_dp_local_page_tables(**kwargs)
