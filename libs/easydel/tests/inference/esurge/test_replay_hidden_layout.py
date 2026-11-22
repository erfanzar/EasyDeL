"""Host-only regressions for token-major versus gathered replay hidden states."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.pipeline_execution_manager import RunnerExecutionManager
from easydel.inference.esurge.runners.spec import strategy as speculation

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("row_pos,prefix_len", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("token_bucket", [4, 8])
@pytest.mark.parametrize("layout", ["tokens", "requests"])
@pytest.mark.parametrize("operation", ["prefix", "sequential", "suffix"])
def test_prefix_replay_preserves_active_request_hidden_and_sampling_seed(
    monkeypatch, row_pos, prefix_len, token_bucket, layout, operation
):
    """Replay uses token zero only for full outputs, never for gathered rows."""
    manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    manager.kv_pages = object()
    manager.full_hidden_state_max_tokens = token_bucket if layout == "tokens" else 2
    manager._model_executor = SimpleNamespace(_uses_mpmd_mesh=lambda: False, _uses_spmd_dp=lambda: False)
    calls = []
    expected = np.asarray([303, 909])
    hidden = np.zeros((token_bucket if layout == "tokens" else 4, 2), dtype=np.int32)
    token_index = 0 if operation == "sequential" else prefix_len - 1
    hidden[token_index if layout == "tokens" else row_pos] = expected
    # A different live request's row must not seed either the drafter or a
    # sampled verifier's next target distribution.
    if layout == "requests":
        hidden[prefix_len - 1] = [202, 808]

    def execute(**kwargs):
        calls.append(kwargs)
        return np.asarray([11, 22, 33, 44]), None, None, None, hidden, None, None

    manager.execute = execute
    runner = SimpleNamespace(
        executor_manager=manager,
        scheduled_token_counts_cpu=np.asarray([2, 2, 2, 2]),
        active_request_mask_cpu=np.ones(4, dtype=bool),
        cache_group_inputs_for_execute=lambda **kwargs: (),
        side_inputs_for_execute=lambda **kwargs: None,
        input_ids_buf=None,
        position_ids_buf=None,
        num_tokens_paddings=[token_bucket],
        record_speculative_execution_time=lambda **kwargs: None,
    )
    monkeypatch.setattr(speculation, "_restore_recurrent_state", lambda live, snapshot: live)
    strategy = speculation.DrafterSpeculation(runner=runner, drafter=SimpleNamespace(), num_draft_tokens=2)
    params = dict.fromkeys(
        (
            "temperature_window_cpu",
            "top_p_window_cpu",
            "top_k_window_cpu",
            "min_p_window_cpu",
            "frequency_penalties_window_cpu",
            "presence_penalties_window_cpu",
            "repetition_penalties_window_cpu",
            "page_table_window_cpu",
            "page_table_window_version",
            "mrope_position_ids_cpu",
            "prefill_embeds_cpu",
            "prefill_embeds_mask_cpu",
            "visual_pos_masks_cpu",
            "deepstack_visual_embeds_cpu",
        )
    )
    params.update(
        num_computed_tokens_window_cpu=np.asarray([2, 4, 6, 8]),
        window_row_indices_cpu=np.asarray([3, 1, 0, 2]),
        recurrent_slot_indices_cpu=None,
        padded_num_reqs=4,
        token_ids_window_cpu=np.zeros((4, 16), dtype=np.int32),
    )
    if operation == "prefix":
        token, replay_hidden = strategy.replay_prefix_sample(
            row_pos,
            prefix_len,
            pre_step_kv_pages=object(),
            num_tokens_static=token_bucket,
            window_token_offsets=np.asarray([0, 2, 4, 6]),
            total_scheduled=8,
            rng_after_verify="preserved-rng",
            **params,
        )
    elif operation == "sequential":
        token, replay_hidden = strategy.replay_prefix_sequential(
            row_pos, prefix_len, pre_step_kv_pages=object(), rng_after_verify="preserved-rng", **params
        )
    else:
        token, replay_hidden = strategy.run_suffix_sample(row_pos, 0, suffix_len=prefix_len, **params)
    assert token == [11, 22, 33, 44][row_pos]
    np.testing.assert_array_equal(replay_hidden, expected)
    if operation != "suffix":
        assert manager.rng_key == "preserved-rng"
    assert calls[0]["num_tokens"] == token_bucket
    scheduled = 1 if operation == "sequential" else prefix_len
    np.testing.assert_array_equal(calls[0]["scheduled_full_cpu"], np.eye(4, dtype=int)[row_pos] * scheduled)


@pytest.mark.parametrize(
    "mpmd,dp,fused,limit,expected",
    [
        (False, False, False, 4, "tokens"),
        (False, False, False, 2, "requests"),
        (False, True, False, 4, "tokens"),
        (True, False, False, 0, "tokens"),
        (True, False, True, 0, "requests"),
    ],
)
def test_equal_token_and_request_buckets_use_executor_layout_contract(mpmd, dp, fused, limit, expected):
    manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    manager.full_hidden_state_max_tokens = limit
    manager._model_executor = SimpleNamespace(
        _uses_mpmd_mesh=lambda: mpmd,
        _uses_spmd_dp=lambda: dp,
        _compile_mode=lambda: "mpmd",
        _cache={(4, 4, "model_step", "mpmd"): object()} if fused else {},
    )
    assert manager.hidden_state_layout(np.zeros((4, 2)), num_tokens=4, padded_num_reqs=4) == expected


@pytest.mark.parametrize("model_request_bucket,expected", [(2, "requests"), (8, "tokens")])
def test_pp_compaction_and_unrelated_cached_variant_do_not_change_layout(monkeypatch, model_request_bucket, expected):
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager

    hidden = np.zeros((2, 3))
    output = SimpleNamespace(hidden_states=hidden)
    inputs = object()
    calls = []

    def execute_model(self, **kwargs):
        calls.append(kwargs)
        return output

    monkeypatch.setattr(ExecutionManager, "execute_model", execute_model)
    manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    manager._model_executor = SimpleNamespace(
        _uses_mpmd_mesh=lambda: True,
        _compile_mode=lambda: "mpmd",
        # A fused (2 tokens, 2 requests) executable can coexist with a split
        # (2 tokens, 8 requests) variant. Only the executed key is authoritative.
        _cache={(2, 2, "model_step", "mpmd"): object()},
    )
    assert manager.execute_model(2, model_request_bucket, inputs, False) is output
    assert calls == [dict(num_tokens=2, padded_num_reqs=model_request_bucket, inputs=inputs, use_pipeline_runtime=False)]
    assert manager.hidden_state_layout(hidden, num_tokens=2, padded_num_reqs=8) == expected


def test_explicit_hidden_layout_rejects_invalid_rows_instead_of_clamping():
    hidden = np.asarray([[10], [20]])
    with pytest.raises(IndexError, match="outside"):
        speculation.window_hidden_row(
            hidden,
            row_pos=2,
            token_offset=0,
            token_index=0,
            total_window_tokens=1,
            padded_reqs=2,
            output_layout="requests",
        )
    manager = RunnerExecutionManager.__new__(RunnerExecutionManager)
    manager._model_executor = SimpleNamespace(_uses_mpmd_mesh=lambda: False)
    with pytest.raises(ValueError, match="bucket"):
        manager.hidden_state_layout(hidden, num_tokens=8, padded_num_reqs=4)
