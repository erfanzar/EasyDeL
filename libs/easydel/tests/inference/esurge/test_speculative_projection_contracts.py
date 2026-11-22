"""CPU-only regressions for speculative projection and token-bucket contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.runners import model_runner, vlm_prefill
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.utils import flags

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "model,expected",
    [
        (SimpleNamespace(), True),
        (SimpleNamespace(get_lm_logit_postprocess=lambda: lambda logits: logits * 2), False),
        (SimpleNamespace(get_lm_logit_postprocess=lambda: None), True),
        # An authoritative identity hook supersedes stale legacy attributes.
        (SimpleNamespace(get_lm_logit_postprocess=lambda: None, logit_scale=2), True),
        # Scalar attributes alone are not an executable transform contract:
        # family hooks encode the correct scale/clip/soft-cap order.
        (SimpleNamespace(logit_scale=2), True),
        (SimpleNamespace(config=SimpleNamespace(final_logit_softcapping=30)), True),
    ],
    ids=["raw", "custom-hook", "identity", "authoritative-identity", "legacy-scale", "legacy-softcap"],
)
def test_fused_verify_uses_authoritative_postprocessing_contract(monkeypatch, model, expected):
    monkeypatch.setattr(flags, "get_bool", lambda flag: False)
    drafter = SimpleNamespace(supports_draft_in_verify=True, verify_draft_batched=lambda **kwargs: None)
    strategy = DrafterSpeculation(runner=SimpleNamespace(model=model), drafter=drafter, num_draft_tokens=2)
    assert strategy.can_draft_in_verify() is expected


@pytest.mark.parametrize("family", ["cohere", "gemma3"])
def test_real_family_logit_hook_disables_raw_fused_projection(monkeypatch, family):
    from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
    from easydel.modules.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    # Call real scalar-only hook construction without instantiating model arrays.
    model = SimpleNamespace(
        get_base_lm_logit_postprocess=lambda: None,
        logit_scale=2.0,
        config=SimpleNamespace(final_logit_softcapping=30.0),
    )
    model_class = CohereForCausalLM if family == "cohere" else Gemma3ForCausalLM
    model.get_lm_logit_postprocess = lambda: model_class.get_lm_logit_postprocess(model)
    monkeypatch.setattr(flags, "get_bool", lambda flag: False)
    drafter = SimpleNamespace(supports_draft_in_verify=True, verify_draft_batched=lambda **kwargs: None)
    strategy = DrafterSpeculation(runner=SimpleNamespace(model=model), drafter=drafter, num_draft_tokens=2)
    assert not strategy.can_draft_in_verify()


def test_explicit_projection_keeps_fused_verify_available_with_custom_postprocessing(monkeypatch):
    monkeypatch.setattr(flags, "get_bool", lambda flag: False)

    class CustomProjectionSpeculation(DrafterSpeculation):
        def draft_in_verify_project_fn(self):
            return projection

    def projection(hidden):
        return hidden

    model = SimpleNamespace(get_lm_logit_postprocess=lambda: lambda logits: logits * 3)
    drafter = SimpleNamespace(supports_draft_in_verify=True, verify_draft_batched=lambda **kwargs: None)
    strategy = CustomProjectionSpeculation(runner=SimpleNamespace(model=model), drafter=drafter, num_draft_tokens=2)
    assert strategy.can_draft_in_verify()


@pytest.fixture
def make_host_runner(monkeypatch):
    class HostRunner(model_runner.eSurgeRunner):
        def _build_kv_cache_groups(self):
            return []

        def _apply_window_aware_runtime_cap(self, max_num_batched_tokens):
            return None

        def _setup_variables(self):
            pass  # Deliberately do not allocate device arrays.

        def _log_startup_summary(self):
            pass

    class HostManager:
        def __init__(self, **kwargs):
            self.full_hidden_state_max_tokens = kwargs["full_hidden_state_max_tokens"]
            self.host_payloads_replicated = kwargs["host_payloads_replicated"]
            self.main_hidden_enabled = False

        def set_main_hidden_enabled(self, enabled=True):
            self.main_hidden_enabled = enabled

        def update_graphs(self, **kwargs):
            pass  # No compiled programs exist in this host-only fixture.

    class HostModel:
        """Weak-referenceable stand-in for the production model object."""

    text_config = SimpleNamespace(attn_mechanism="ragged_page_attention_v3", layer_types=())
    model = HostModel()
    model.mesh = None
    model.config = SimpleNamespace(get_text_config=lambda: text_config)
    model.graphdef = object()
    model.esurge_compatible_model = model
    model.create_ragged_page_cache_config = lambda **kwargs: SimpleNamespace(
        num_pages=64,
        page_size=2,
        max_num_pages_per_req=8,
        data_parallel_size=1,
        get_max_num_seqs=lambda: 2,
    )
    monkeypatch.setattr(model_runner, "ExecutionManager", HostManager)

    def make(**kwargs):
        runner = HostRunner(model, max_model_len=16, max_num_seqs=2, min_input_pad=1, **kwargs)
        runner.requests = {}
        return runner

    return make


def test_runner_construction_and_late_coordination_keep_instance_policy(make_host_runner):
    driver = make_host_runner()
    coordinated = make_host_runner(host_payloads_replicated=True)
    assert not driver.host_payloads_replicated
    assert not driver.executor_manager.host_payloads_replicated
    assert coordinated.host_payloads_replicated
    assert coordinated.executor_manager.host_payloads_replicated
    coordinated.host_payloads_replicated = False
    driver.host_payloads_replicated = True
    assert driver.executor_manager.host_payloads_replicated
    assert not coordinated.executor_manager.host_payloads_replicated


@pytest.mark.parametrize("draft_count,expected_bucket", [(0, 2), (2, 3), (19, 16)])
def test_speculative_hidden_window_selects_sufficient_bucket_and_clamps_overflow(
    make_host_runner, draft_count, expected_bucket
):
    runner = make_host_runner(drafter=SimpleNamespace(num_draft_tokens=draft_count))
    assert runner.executor_manager.full_hidden_state_max_tokens == expected_bucket


def test_runner_shares_vision_cache_and_invalidates_on_same_object_weight_refresh(make_host_runner, monkeypatch):
    # The encoder is a NumPy fake: this exercises runner/helper/cache plumbing,
    # not JAX array computation or vision numerical parity.
    monkeypatch.setattr(vlm_prefill.jnp, "asarray", np.asarray)
    shared_cache = VisionEncoderCache(capacity_mb=1)
    runner = make_host_runner(vision_cache=shared_cache)
    model = runner.model
    model.compute_embedding = lambda input_ids, image_embeds=None: image_embeds
    model.gain = 2

    def encode(pixels, **kwargs):
        return pixels * model.gain, ()

    runner.vlm.image_features_jit = encode
    request = SimpleNamespace(
        pixel_values=np.asarray([[1.0, 3.0]], dtype=np.float32),
        image_grid_thw=np.asarray([[1, 1, 1]], dtype=np.int32),
        pixel_values_videos=None,
        mm_features=[],
    )
    first = runner.vlm.compiled_vision_embed_kwargs(request)
    assert first is not None
    np.testing.assert_array_equal(first["image_embeds"], [[2, 6]])
    assert len(shared_cache) == 1
    second = runner.vlm.compiled_vision_embed_kwargs(request)
    np.testing.assert_array_equal(second["image_embeds"], [[2, 6]])
    assert shared_cache.hits == 1

    shared_cache.clear()
    model.gain = 3
    after_clear = runner.vlm.compiled_vision_embed_kwargs(request)
    np.testing.assert_array_equal(after_clear["image_embeds"], [[3, 9]])

    # Refreshing weights in place must invalidate the encoder namespace even
    # though model identity (and all raw image/grid bytes) remains unchanged.
    model.gain = 4
    runner.update_model_weights(model, reset_state=False)
    runner.vlm.image_features_jit = encode
    after_refresh = runner.vlm.compiled_vision_embed_kwargs(request)
    np.testing.assert_array_equal(after_refresh["image_embeds"], [[4, 12]])


def test_dspark_runner_enables_main_hidden_through_manager_api(make_host_runner):
    runner = make_host_runner(drafter=SimpleNamespace(num_draft_tokens=2, needs_main_hidden=True))
    assert runner.executor_manager.main_hidden_enabled


@pytest.fixture
def host_execution_manager():
    # Use the real public manager APIs with NumPy-only executable stand-ins.
    # Construction of device buffers and compiled JAX programs is out of scope.
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager

    class HostModelExecutor:
        def __init__(self):
            self.model = object()
            self.graphdef = object()
            self.main_hidden_enabled = False
            self.projections = {}
            self.runtime_weights = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            self.runtime_bias = np.asarray([7, 8, 9], dtype=np.float32)

        def cache_keys(self):
            return list(self.projections)

        def get_lm_head(self, *, padded_num_reqs):
            return self.projections[padded_num_reqs]

        def compile_lm_head(self, *, padded_num_reqs, **kwargs):
            self.projections[padded_num_reqs] = lambda weights, bias, rows: rows @ weights + bias

        def runtime_graph_args(self):
            return self.runtime_weights, self.runtime_bias

        def clear_runtime_graph_args(self):
            self.runtime_weights = None
            self.runtime_bias = None

        def clear_cache(self):
            self.projections.clear()

    executor = HostModelExecutor()
    manager = ExecutionManager.__new__(ExecutionManager)
    manager.model = executor.model
    manager.graphdef = executor.graphdef
    # Live runtime graph args deliberately differ from trace-state placeholders.
    manager.graphstate = object()
    manager.graphother = object()
    manager._model_executor = executor
    manager._sampler_executor = SimpleNamespace(
        model=manager.model, clear_cache=lambda: None, cache_keys=lambda: ["sample"]
    )
    manager._debug_baselines = {}
    manager._batch_preparer = SimpleNamespace(_cache_group_preparers={})
    manager._pp_microbatch = None
    return manager


@pytest.mark.parametrize("needs_main_hidden", [False, True])
def test_spec_projection_uses_public_manager_with_live_weights(host_execution_manager, needs_main_hidden):
    model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=2)))
    runner = SimpleNamespace(model=model, executor_manager=host_execution_manager)
    strategy = DrafterSpeculation(
        runner=runner, drafter=SimpleNamespace(needs_main_hidden=needs_main_hidden), num_draft_tokens=2
    )
    rows = np.asarray([[2, 3], [5, 7]], dtype=np.float32)
    supplied = np.concatenate([rows, np.full_like(rows, -100)], axis=1) if needs_main_hidden else rows
    expected = np.asarray([[21, 27, 33], [40, 53, 66]], dtype=np.float32)
    np.testing.assert_array_equal(strategy.project_hidden_rows(supplied), expected)
    # A different row count compiles on demand without a dummy model batch.
    np.testing.assert_array_equal(strategy.project_hidden_rows(supplied[:1]), expected[:1])
    with pytest.raises(RuntimeError, match="before compiling"):
        host_execution_manager.set_main_hidden_enabled(True)


@pytest.mark.parametrize("clear_compiled_cache", [False, True])
def test_runner_release_validates_lifecycle_and_releases_manager_state(
    make_host_runner, host_execution_manager, clear_compiled_cache
):
    runner = make_host_runner()
    runner.executor_manager = host_execution_manager
    hidden_rows = np.asarray([[1, 2]], dtype=np.float32)
    # Populate a projection bucket to check retention independently of live state.
    host_execution_manager.project_hidden_rows(hidden_rows)
    runner.requests = {"active": object()}
    with pytest.raises(RuntimeError, match="requests are active"):
        runner.release_model_state(clear_compiled_cache=clear_compiled_cache)
    assert runner.model is not None
    assert host_execution_manager.model is not None

    runner.requests.clear()
    # The constructor fixture deliberately owns no device-backed sequence state.
    runner.reset_state = lambda: None
    runner.release_model_state(clear_compiled_cache=clear_compiled_cache)
    assert runner.model is None
    assert host_execution_manager.model is None
    assert host_execution_manager.graphdef is None
    assert host_execution_manager.graphstate is None
    assert host_execution_manager.graphother is None
    assert host_execution_manager.has_compiled_variants() is not clear_compiled_cache
    with pytest.raises(RuntimeError, match="Restore model state"):
        host_execution_manager.project_hidden_rows(hidden_rows)


def test_deferred_recurrent_commit_preserves_pending_request_state(make_host_runner):
    runner = make_host_runner()
    pending = {}
    runner._pending_spec_recurrent_commit_by_req = pending
    strategy = DrafterSpeculation(runner=runner, drafter=SimpleNamespace(), num_draft_tokens=2)
    strategy.recurrent_candidate_count = lambda: 3
    assert strategy.queue_or_commit_recurrent_candidate_state(req_id="a", row_pos=0, prefix_len=2, defer=True)
    assert pending == {"a": 2}
    # Invalid candidate indices must not overwrite a queued accepted prefix.
    assert not strategy.queue_or_commit_recurrent_candidate_state(req_id="a", row_pos=0, prefix_len=4, defer=True)
    assert pending == {"a": 2}
    with pytest.raises(ValueError, match="positive prefix"):
        runner.queue_speculative_recurrent_commit("b", 0)
    assert pending == {"a": 2}


def test_token_handoff_placement_uses_execution_manager_sharding(make_host_runner, monkeypatch):
    runner = make_host_runner()
    runner.executor_manager._empty_sharding = "replicated"
    monkeypatch.setattr(model_runner.jax, "device_put", lambda value, sharding: np.asarray(value))
    runner._handoff_positions_cache = {}
    runner._handoff_scalar_cache = {}
    np.testing.assert_array_equal(runner._get_handoff_positions(3), [0, 1, 2])
    assert int(runner._get_handoff_scalar(2)) == 2


@pytest.mark.parametrize("extra_execute", [False, True], ids=["main-step", "replay-step"])
def test_rank_major_side_inputs_keep_request_types_and_rank_padding(extra_execute):
    runner = model_runner.eSurgeRunner.__new__(model_runner.eSurgeRunner)
    runner.model = SimpleNamespace(esurge_requires_token_types=True)
    runner.metadata = SimpleNamespace(data_parallel_size=2)
    runner.num_reqs_max_model_len = 4
    runner._side_input_provider = None
    runner._side_input_tokenizer_state = None
    # Deliberately collect rank 1 before rank 0: side-input order must follow
    # the batch preparer's global-row ownership, not compact window order.
    token_ids = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    runner.sequence_buffer = SimpleNamespace(
        req_ids=["a", None, "b", "c"],
        token_ids=token_ids,
        num_computed_tokens=np.asarray([0, 0, 1, 0]),
        num_tokens_no_spec=np.asarray([3, 0, 3, 3]),
    )
    runner.requests = {
        "a": SimpleNamespace(token_types=np.asarray([0, 1, 2], dtype=np.int32)),
        "b": SimpleNamespace(token_types=np.asarray([-1, 3, -1], dtype=np.int32)),
        "c": SimpleNamespace(token_types=None),
    }
    row_indices = np.asarray([2, 0, 3], dtype=np.int32)
    if extra_execute:
        payload = runner.side_inputs_for_execute(
            token_ids_cpu=token_ids[row_indices],
            num_computed_tokens_cpu=np.asarray([1, 0, 0]),
            scheduled_full_cpu=np.asarray([1, 2, 1]),
            window_row_indices_cpu=row_indices,
            num_tokens_static=8,
        )
    else:
        payload = runner._build_model_side_inputs(
            req_ids_window=["b", "a", "c"],
            scheduled_list=[1, 2, 1],
            window_row_indices=row_indices,
            num_tokens_static=8,
        )
    np.testing.assert_array_equal(payload["token_types"], [0, 1, -1, -1, 3, -1, -1, -1])
