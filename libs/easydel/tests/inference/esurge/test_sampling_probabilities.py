"""Position-major output transport (host) and independent policy-score parity (hardware)."""

from functools import partial

import numpy as np
import pytest
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.runners.async_types import AsyncWindowResult
from easydel.inference.esurge.runners.model_runner import _AsyncExecutionHandle, _materialize_logprobs
from easydel.inference.esurge.runners.spec.sampling import score_sampled_tokens
from jax import numpy as jnp


@pytest.mark.cpu_ok
@pytest.mark.parametrize("deferred", [False, True])
def test_requested_scores_keep_sparse_rows_and_position_offsets_without_metrics(deferred):
    payloads = {
        "scored": (
            np.asarray([[7, 7, 3], [5, 3, 5]]),
            np.asarray([[-0.2, -0.2, -1.8], [-0.9, -0.6, -0.9]]),
            np.asarray([0, 1]),
        ),
    }
    req_ids = ["unrequested", "prefill", "scored"]
    emitted = [[4], [], [7, 5]]
    if deferred:
        # Async sampling has one token per request. Sparse model row indices
        # cannot be replaced with dense enumeration by deferred materialization.
        payloads["scored"] = tuple(array[:1] for array in payloads["scored"])
        emitted[-1] = [7]
        output = ModelRunnerOutput(req_ids, dict(zip(req_ids, range(3), strict=True)), [], None, None, {})
        handle = _AsyncExecutionHandle(
            output,
            [AsyncWindowResult(req_ids, [2, 0, 4], np.asarray([99, 99, 4, 99, 7]), [True, False, True])],
            resolve_logprobs=partial(_materialize_logprobs, req_ids, payloads=payloads, requested={"scored"}),
        )
        resolved = handle.get_output()
        assert resolved is handle.get_output()
        assert resolved.sampled_token_ids == emitted
        assert resolved.token_logprobs is None  # Metrics stayed disabled.
        scores = resolved.logprobs
    else:
        scores = _materialize_logprobs(req_ids, emitted, payloads, {"scored"})
    assert scores.logprob_token_ids[0] == []
    assert scores.logprob_token_ids[1] == [7, 3]
    assert scores.logprobs[1] == [-0.2, -1.8]
    assert len(scores.logprobs) == sum(map(len, emitted))
    if not deferred:
        assert scores.logprob_token_ids[2] == [5, 3]
        assert scores.sampled_token_ranks[2] == 1


@pytest.mark.cpu_ok
@pytest.mark.parametrize("payload", [{}, {"a": (np.asarray([[8]]), np.asarray([[-0.5]]), np.asarray([0]))}])
def test_missing_or_misaligned_requested_scores_fail_instead_of_fabricating_probabilities(payload):
    with pytest.raises(RuntimeError):
        _materialize_logprobs(["a"], [[7]], payload, {"a"})


def run_host_runner_with_vision_payload(monkeypatch, deferred, dp_size, extra_vision):
    """Exercise the public runner with host-only executable boundaries and retain its payload."""
    from types import SimpleNamespace

    from easydel.inference.esurge.runners import model_runner
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.slot_pool import RecurrentSlotPool
    from easydel.inference.esurge.runners.spec.interface import NullSpeculation
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.esurge.scheduler.output import CachedRequestData, SchedulerOutput
    from easydel.inference.sampling_params import SamplingParams

    # Only the device executable/scorer are stand-ins; exercise both public
    # runner entry points, validity handling and actual output constructors.
    monkeypatch.setattr(model_runner.jax, "copy_to_host_async", lambda array: array)
    monkeypatch.setattr(model_runner, "host_payload_broadcast_needed", lambda host_payloads_replicated=None: False)
    monkeypatch.setattr(model_runner, "get_metrics_collector", lambda: None)
    runner = model_runner.eSurgeRunner.__new__(model_runner.eSurgeRunner)
    runner.metadata = SimpleNamespace(data_parallel_size=dp_size, num_pages=32, page_size=2)
    runner.model = SimpleNamespace()
    rows = [0, 1] if dp_size == 1 else [2, 0]
    monkeypatch.setattr(model_runner, "model_uses_mrope", lambda model: True)
    runner.sequence_buffer = SequenceBuffer(4, 8, 8, 16, [2])
    runner.requests = {}
    for row, (name, prompt) in zip(rows, [("prefill", [1, 2, 3]), ("emit", [4])], strict=True):
        page_ids = ([17, 18] if row == 2 else [1, 2],)
        state = CachedRequestState(name, prompt, SamplingParams(logprobs=0), None, page_ids, 0, [])
        runner.requests[name] = state
        runner.sequence_buffer.add_request(state, row)
    if dp_size > 1:
        vision = runner.requests["prefill"]
        vision.prefill_inputs_embeds = np.asarray([[11, 12], [13, 14], [15, 16]])
        vision.prefill_position_ids = np.asarray([[5, 8, 11], [6, 9, 12], [7, 10, 13]])
        vision.prefill_visual_pos_masks = np.asarray([True, False, False])
        vision.prefill_deepstack_visual_embeds = [np.asarray([[21, 22]])]
        vision.mm_features = [object()]
        vision._vision_processed = True
        if extra_vision:
            video = runner.requests["emit"]
            video.mm_features = [object()]
            video._vision_processed = True
            video.prefill_inputs_embeds = np.asarray([[31, 32]])
            video.prefill_position_ids = np.asarray([[9], [10], [11]])
            video.prefill_visual_pos_masks = np.asarray([True])
            video.prefill_deepstack_visual_embeds = [np.asarray([[41, 42]])]
        runner._get_vlm_cpu_buffers = lambda **kwargs: (
            np.zeros((4, 2)),
            np.zeros(4, dtype=bool),
            np.zeros((3, 4), dtype=np.int32),
            np.zeros(4, dtype=bool),
            [np.zeros((4, 2))],
        )
    runner.slot_pool = RecurrentSlotPool(metadata=runner.metadata, max_num_reqs=4, slot_indexed_state=False)
    runner.drafter = None
    runner.spec = NullSpeculation()
    runner._has_recurrent_layers = False
    runner._slot_indexed_state = False
    runner.spec_decode_recurrent_candidates = False
    runner.spec_decode_recurrent_replay = False
    runner.enable_sampler_metrics = False
    runner.async_scheduling = False
    runner._pre_async_results = None
    runner._allow_sparse_window_packing = False
    runner.max_model_len = 8
    runner.num_reqs_max_model_len = 4
    runner.num_tokens_paddings = [4]
    runner.window_planner = SimpleNamespace(get_token_bucket=lambda *args, **kwargs: 4)
    runner._get_current_bucket = lambda count: 4
    runner._update_states = lambda schedule: None
    runner._reorder_decode_first = lambda schedule: None
    runner._reorder_decode_first_per_shard = lambda *args: None
    runner._model_uses_vlm_inputs = lambda: dp_size > 1
    runner._collect_schedulable_window_rows = lambda **kwargs: (
        np.asarray(rows),
        ["prefill", "emit"],
        [1, 1],
        runner.sequence_buffer.num_slots,
        False,
    )
    for name in (
        "_scheduled_full_cpu",
        "_req_num_tokens_cpu",
        "_window_row_indices_cpu",
        "_window_recurrent_slot_indices_cpu",
    ):
        setattr(runner, name, np.zeros(4, dtype=np.int32))
    runner._active_mask_full_cpu = np.zeros(4, dtype=bool)
    runner.input_ids_buf = runner.position_ids_buf = None
    runner._get_full_attention_page_table_index = lambda: 0
    runner.cache_group_inputs_for_execute = lambda **kwargs: ()
    runner._get_window_state_views = lambda **kwargs: (
        runner.sequence_buffer.token_ids[rows],
        np.zeros(2, dtype=np.int32),
        np.ones(2),
        np.ones(2),
        np.zeros(2, dtype=np.int32),
        np.zeros(2),
        runner.sequence_buffer.page_table[0].get_cpu_tensor(),
        np.zeros(2),
        np.zeros(2),
        np.ones(2),
        0,
    )
    executed = []

    def execute(**kwargs):
        executed.append(kwargs)
        return np.asarray([9, 7, -1, -1]), None, None, None, None, np.zeros((4, 16)), {}

    runner.executor_manager = SimpleNamespace(execute=execute)
    runner._score_output_row = lambda **kwargs: (np.asarray([[7]]), np.asarray([[-0.75]]), np.asarray([1]))
    runner.perf = model_runner.RunnerPerfTracker(history_maxlen=2)
    runner.log_it = lambda *args: None
    schedule = SchedulerOutput(
        [], CachedRequestData.make_empty(), {"prefill": 1, "emit": 1}, 2, {}, [0], set(), async_scheduling=False
    )
    output = runner.execute_model_async(schedule).get_output() if deferred else runner.execute_model(schedule)
    assert output.req_ids == ["prefill", "emit"]
    assert output.sampled_token_ids == [[], [7]]
    assert output.logprobs.logprob_token_ids == [[7]]
    assert output.logprobs.logprobs == [[-0.75]]
    assert output.token_logprobs is None
    assert runner.requests["emit"].output_token_ids == [7]
    if dp_size > 1:
        inputs = executed[0]
        first_embed = [31, 32] if extra_vision else [0, 0]
        np.testing.assert_array_equal(inputs["prefill_embeds_cpu"], [first_embed, [0, 0], [11, 12], [0, 0]])
        np.testing.assert_array_equal(inputs["prefill_embeds_mask_cpu"], [extra_vision, False, True, False])
        np.testing.assert_array_equal(inputs["mrope_position_ids_cpu"][:, 2], [5, 6, 7])
        np.testing.assert_array_equal(inputs["visual_pos_masks_cpu"], [extra_vision, False, True, False])
        residuals = [[41, 42], [21, 22], [0, 0], [0, 0]] if extra_vision else [[21, 22], [0, 0], [0, 0], [0, 0]]
        np.testing.assert_array_equal(inputs["deepstack_visual_embeds_cpu"][0], residuals)
        # The actual model injects from one globally compact table using the
        # flattened mask's cumulative count. A rank-local gap would inject zero
        # into the last-rank vision token, despite correct mask/embedding shapes.
        from easydel.modules._base import qwen_vl

        monkeypatch.setattr(qwen_vl, "jnp", np)
        injected = qwen_vl.DeepStackVisualMixin()._deepstack_process(
            np.zeros((dp_size, 4 // dp_size, 2), dtype=np.float32),
            inputs["visual_pos_masks_cpu"].reshape(dp_size, 4 // dp_size),
            inputs["deepstack_visual_embeds_cpu"][0],
        )
        expected = [[41, 42] if extra_vision else [0, 0], [0, 0], [21, 22], [0, 0]]
        np.testing.assert_array_equal(injected.reshape(4, 2), expected)
    return executed[0]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2, 4])
@pytest.mark.parametrize("extra_vision", [False, True])
def test_public_runner_outputs_requested_scores_with_metrics_disabled(monkeypatch, deferred, dp_size, extra_vision):
    run_host_runner_with_vision_payload(monkeypatch, deferred, dp_size, extra_vision)


@pytest.mark.parametrize("greedy,top_k", [(False, 0), (False, 2), (True, 0)])
def test_behavior_scores_match_independent_temperature_penalty_and_topk_reference(greedy, top_k):
    logits = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    history = np.asarray([[1, 1, 2, 0]], dtype=np.int32)
    token = np.asarray([3], dtype=np.int32)
    values = dict(
        logits=jnp.asarray(logits),
        sampled_tokens=jnp.asarray(token),
        token_history=jnp.asarray(history),
        seq_lens=jnp.asarray([3]),
        temperature=jnp.asarray([0.0 if greedy else 0.5]),
        top_p=jnp.asarray([1.0]),
        top_k=jnp.asarray([top_k]),
        min_p=jnp.asarray([0.0]),
        frequency_penalty=jnp.asarray([0.25]),
        presence_penalty=jnp.asarray([0.5]),
        repetition_penalty=jnp.asarray([2.0]),
        num_logprobs=2,
    )
    ids, scores, ranks = (np.asarray(array) for array in score_sampled_tokens(**values))
    reference = logits[0].astype(np.float64)
    for index in (1, 2):
        count = int((history[0, :3] == index).sum())
        reference[index] = (reference[index] - 0.5 - 0.25 * count) / 2.0
    if top_k:
        reference[np.argsort(reference)[:-top_k]] = -np.inf
    reference /= 0.5
    expected = 0.0 if greedy else reference[3] - np.log(np.exp(reference - reference.max()).sum()) - reference.max()
    assert ids[0, 0] == 3
    np.testing.assert_allclose(scores[0, 0], expected, atol=2e-6)
    assert ranks[0] == 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("min_p", [0.1, 0.2])
def test_top_p_precedes_temperature_and_min_p_in_sampled_distribution(dtype, min_p):
    ids, scores, _ranks = score_sampled_tokens(
        jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=dtype),
        jnp.asarray([3]),
        jnp.asarray([[0]]),
        jnp.asarray([1]),
        jnp.asarray([0.5]),
        jnp.asarray([0.8]),
        jnp.asarray([0]),
        jnp.asarray([min_p]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        num_logprobs=0,
    )
    # Nucleus keeps logits 2 and 3 BEFORE temperature. Temperature 0.5 gives
    # probabilities [sigmoid(-2), sigmoid(2)]; min-p 0.2 then removes logit 2.
    expected = -np.log1p(np.exp(-2.0)) if min_p == 0.1 else 0.0
    np.testing.assert_array_equal(ids, [[3]])
    np.testing.assert_allclose(scores, [[expected]], atol=2e-6)


def test_logprobs_zero_still_reports_the_sampled_token():
    ids, scores, ranks = score_sampled_tokens(
        jnp.asarray([[0.0, 0.0]]),
        jnp.asarray([1]),
        jnp.asarray([[0]]),
        jnp.asarray([1]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
        jnp.asarray([0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        num_logprobs=0,
    )
    np.testing.assert_array_equal(ids, [[1]])
    np.testing.assert_allclose(scores, [[-np.log(2.0)]], atol=2e-6)
    np.testing.assert_array_equal(ranks, [0])


def test_batched_score_output_window_matches_per_row_reference_and_skips_penalties():
    """Window-batched logprob scoring must match per-row scoring across mixed k, filters, and metrics."""
    from types import SimpleNamespace

    from easydel.inference.esurge.runners import model_runner
    from easydel.inference.sampling_params import SamplingParams

    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.normal(size=(4, 16)), dtype=jnp.float32)
    sampled = jnp.asarray([3, 7, 1, -1], dtype=jnp.int32)
    history = np.zeros((4, 8), dtype=np.int32)

    params_a = SamplingParams(temperature=0.8, top_k=5, top_p=1.0, min_p=0.0, logprobs=0)
    params_b = SamplingParams(temperature=0.6, top_k=0, top_p=0.9, min_p=0.05, logprobs=3)
    params_c = SamplingParams(temperature=0.0, logprobs=1)  # greedy point mass

    req_a = SimpleNamespace(sampling_params=params_a)
    req_b = SimpleNamespace(sampling_params=params_b)
    req_c = SimpleNamespace(sampling_params=params_c)
    window_entries = [
        (0, "r0", req_a, 0, 4, True),
        (1, "r1", req_b, 1, 5, True),
        (2, "r2", req_c, 2, 3, True),
        (3, "r3", req_a, 3, 2, False),  # partial prefill -> must be ignored
    ]

    runner = object.__new__(model_runner.eSurgeRunner)
    runner.enable_sampler_metrics = True

    batched_payloads, metric_scores = runner._score_output_window(
        logits=logits,
        sampled_tokens=sampled,
        token_ids_window_cpu=history,
        window_entries=window_entries,
        requested_logprobs={"r0", "r1", "r2"},
        sampling_constraints=None,
        padded_num_reqs=4,
    )
    assert set(batched_payloads.keys()) == {"r0", "r1", "r2"}
    batched_lp = model_runner._materialize_logprobs(
        ["r0", "r1", "r2", "r3"],
        [[3], [7], [1], []],
        batched_payloads,
        {"r0", "r1", "r2"},
    )
    assert batched_lp is not None

    ref_payloads = {}
    for row_pos, rid, req_state, _req_idx, seq_len, is_valid in window_entries:
        if not is_valid:
            continue
        ref_payloads[rid] = runner._score_output_row(
            logits=logits[row_pos : row_pos + 1],
            sampled_tokens=sampled[row_pos : row_pos + 1],
            history=history[row_pos],
            seq_len=seq_len,
            sampling_params=req_state.sampling_params,
            sampling_constraints=None,
            row_pos=row_pos,
        )
    ref_lp = model_runner._materialize_logprobs(
        ["r0", "r1", "r2", "r3"],
        [[3], [7], [1], []],
        ref_payloads,
        {"r0", "r1", "r2"},
    )
    assert ref_lp is not None
    assert batched_lp.logprob_token_ids == ref_lp.logprob_token_ids
    assert batched_lp.sampled_token_ranks == ref_lp.sampled_token_ranks
    for got_scores, exp_scores in zip(batched_lp.logprobs, ref_lp.logprobs, strict=True):
        np.testing.assert_allclose(got_scores, exp_scores, atol=1e-5)
    metric_np = np.asarray(metric_scores)
    np.testing.assert_allclose(metric_np[0], ref_lp.logprobs[0][0], atol=1e-5)
    np.testing.assert_allclose(metric_np[1], ref_lp.logprobs[1][0], atol=1e-5)
    np.testing.assert_allclose(metric_np[2], 0.0, atol=1e-6)
    np.testing.assert_allclose(metric_np[3], 0.0, atol=1e-6)


@pytest.mark.cpu_ok
def test_score_output_window_caches_replicated_params_and_skips_redundant_assert_equal(monkeypatch):
    """Steady-state decode scoring without penalties must broadcast/place once and hit device cache thereafter."""
    from types import SimpleNamespace

    from easydel.inference.esurge.runners import model_runner
    from easydel.inference.sampling_params import SamplingParams
    from jax.experimental import multihost_utils

    mesh = object()

    class _FakeSharding:
        def __init__(self, m=mesh, spec=None):
            del spec
            self.mesh = m
            self.device_set = frozenset({0})

    class _FakeLogits:
        def __init__(self, shape):
            self.shape = shape
            self.sharding = _FakeSharding(mesh)

        def __getitem__(self, item):
            raise AssertionError(f"Unexpected eager logits slice on exact bucket shape: {item!r}")

    class _FakeTokens:
        def __init__(self, shape):
            self.shape = shape
            self.sharding = _FakeSharding(mesh)

        def __getitem__(self, item):
            raise AssertionError(f"Unexpected eager tokens slice on exact bucket shape: {item!r}")

    broadcast_calls = 0
    put_calls: list[tuple[int, bool | None]] = []
    score_calls = 0

    def _fake_broadcast(vals):
        nonlocal broadcast_calls
        broadcast_calls += 1
        return vals

    def _fake_put(values, sharding, *, host_payloads_replicated=None):
        del sharding
        put_calls.append((len(values), host_payloads_replicated))
        return tuple(np.asarray(v) for v in values)

    def _fake_score(logits_w, tokens_w, *args, **kwargs):
        nonlocal score_calls
        del logits_w, tokens_w, args, kwargs
        score_calls += 1
        return (
            np.zeros((4, 1), dtype=np.int32),
            np.full((4, 1), -0.5, dtype=np.float32),
            np.zeros((4,), dtype=np.int32),
        )

    monkeypatch.setattr(model_runner.jax.sharding, "NamedSharding", _FakeSharding)
    monkeypatch.setattr(model_runner.jax.sharding, "PartitionSpec", lambda *a, **k: ())
    monkeypatch.setattr(model_runner.jax, "copy_to_host_async", lambda a: a)
    monkeypatch.setattr(model_runner, "host_payload_broadcast_needed", lambda rep=None: not bool(rep))
    monkeypatch.setattr(model_runner, "put_host_payload", _fake_put)
    monkeypatch.setattr(model_runner, "score_sampled_tokens", _fake_score)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", _fake_broadcast)
    fake_logits = _FakeLogits((4, 16))
    fake_tokens = _FakeTokens((4,))

    runner = object.__new__(model_runner.eSurgeRunner)
    runner.enable_sampler_metrics = False
    runner.executor_manager = SimpleNamespace(host_payloads_replicated=False)
    params = SamplingParams(temperature=1.0, top_k=64, top_p=1.0, logprobs=0)
    req = SimpleNamespace(sampling_params=params)
    history = np.zeros((4, 8), dtype=np.int32)

    # Step 1 (seq_len=10): 1 broadcast + 1 put_host_payload(..., host_payloads_replicated=True) -> zero assert_equal!
    entries_step1 = [(0, "r0", req, 0, 10, True), (1, "r1", req, 1, 10, True)]
    runner._score_output_window(
        logits=fake_logits,
        sampled_tokens=fake_tokens,
        token_ids_window_cpu=history,
        window_entries=entries_step1,
        requested_logprobs={"r0", "r1"},
        sampling_constraints=None,
        padded_num_reqs=4,
    )
    assert broadcast_calls == 1
    assert put_calls == [(9, True)]
    assert score_calls == 1

    # Step 2 (seq_len=11, penalties off): 100% cache hit -> ZERO broadcasts and ZERO puts!
    entries_step2 = [(0, "r0", req, 0, 11, True), (1, "r1", req, 1, 11, True)]
    runner._score_output_window(
        logits=fake_logits,
        sampled_tokens=fake_tokens,
        token_ids_window_cpu=history,
        window_entries=entries_step2,
        requested_logprobs={"r0", "r1"},
        sampling_constraints=None,
        padded_num_reqs=4,
    )
    assert broadcast_calls == 1
    assert put_calls == [(9, True)]
    assert score_calls == 2
