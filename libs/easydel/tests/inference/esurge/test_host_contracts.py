"""Host-only regressions for cache planning and documented sequence APIs."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.pipeline_plan import build_pipeline_inference_plan, cap_metadata_pages
from easydel.inference.esurge.runners.sequence_buffer import DecodeRowInfo, ModelRunBatch, SequenceBuffer
from easydel.inference.esurge.runners.spec import SpecDecodeStats, build_target_kv_pairs
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("dp_size,expected", [(1, 17), (4, 16)])
def test_spmd_margin_and_cap_update_mixed_metadata(dp_size, expected):
    plan = build_pipeline_inference_plan(
        model=SimpleNamespace(mesh=None), max_cache_tokens=40, cache_capacity_margin=0.85
    )
    assert not plan.is_enabled
    mixed = SimpleNamespace(num_pages=100)
    metadata = SimpleNamespace(num_pages=100, page_size=2, data_parallel_size=dp_size, _mixed_layer_configs={0: mixed})
    assert cap_metadata_pages(metadata, plan) is metadata
    assert metadata.num_pages == expected
    assert mixed.num_pages == expected


def test_spmd_margin_without_explicit_token_cap():
    plan = build_pipeline_inference_plan(model=SimpleNamespace(mesh=None), cache_capacity_margin=0.75)
    metadata = SimpleNamespace(num_pages=100, page_size=16)
    cap_metadata_pages(metadata, plan)
    assert metadata.num_pages == 75


def make_buffer():
    # Page tables remain host-only until an explicit device access/commit.
    return SequenceBuffer(4, 8, 6, 32, [2])


def test_documented_begin_sequence_and_prepare_batch():
    buffer = make_buffer()
    assert buffer.begin_sequence("a", [1, 2, 3]) == 0
    assert buffer.begin_sequence("b", [4, 5, 6, 7]) == 1
    batch = buffer.prepare_batch()
    assert isinstance(batch, ModelRunBatch)
    assert batch.req_ids == ("a", "b")
    assert batch.scheduled_token_counts == (3, 3)
    np.testing.assert_array_equal(batch.input_ids, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(batch.position_ids, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(batch.request_indices, [0, 0, 0, 1, 1, 1])
    assert batch.decode_rows == ()
    np.testing.assert_array_equal(buffer.num_computed_tokens, [0, 0, 0, 0])
    buffer.token_ids[0, 0] = 31
    assert int(batch.input_ids[0]) == 1  # Independent batch snapshot.
    with pytest.raises(ValueError, match="already present"):
        buffer.begin_sequence("a", [9])
    with pytest.raises(ValueError, match="nonnegative"):
        buffer.prepare_batch(-1)
    assert buffer.prepare_batch(0).input_ids.shape == (0,)


def test_prepare_batch_skips_holes_and_reports_decode_rows():
    buffer = make_buffer()
    buffer.begin_sequence("a", [1, 2])
    buffer.begin_sequence("b", [3, 4])
    buffer.remove_request("a")
    buffer.num_computed_tokens[1] = 2
    buffer.token_ids[1, 2] = 5
    buffer.num_tokens[1] = 3
    batch = buffer.prepare_batch()
    assert batch.req_ids == ("b",)
    np.testing.assert_array_equal(batch.input_ids, [5])
    np.testing.assert_array_equal(batch.position_ids, [2])
    np.testing.assert_array_equal(batch.request_indices, [1])
    assert batch.decode_rows == (DecodeRowInfo("b", 1, 0, 2),)


def test_public_sampling_accessors_track_request_lifecycle():
    buffer = make_buffer()
    assert buffer.max_num_logprobs is None
    assert buffer.no_prompt_logprob
    assert buffer.no_allowed_token_ids
    buffer.begin_sequence(
        "a", [1], SamplingParams(temperature=0.8, top_k=4, frequency_penalty=0.5, logprobs=3, prompt_logprobs=2)
    )
    assert buffer.max_num_logprobs == 3
    assert not buffer.no_prompt_logprob
    params = buffer.get_active_sampling_params(0)
    assert params["temperature"] == pytest.approx(0.8)
    assert params["top_k"] == 4
    assert params["frequency_penalty"] == pytest.approx(0.5)
    assert "presence_penalty" not in params
    assert buffer.get_active_sampling_params(-1) == {}
    assert buffer.get_active_sampling_params(99) == {}
    buffer.remove_request("a")
    assert buffer.get_active_sampling_params(0) == {}
    assert buffer.max_num_logprobs is None
    assert buffer.no_prompt_logprob


def test_documented_cached_request_token_access():
    state = CachedRequestState("a", [1, 2, 3], SamplingParams(), None, ([],), 0, [4, 5])
    assert [state.get_token_id(i) for i in range(state.num_tokens)] == [1, 2, 3, 4, 5]
    with pytest.raises(IndexError):
        state.get_token_id(5)


def test_public_spec_decode_stats():
    stats = SpecDecodeStats()
    assert stats.acceptance_rate == stats.tokens_per_second == stats.speedup_vs_baseline == 0
    assert stats.mean_accepted_per_step == 0
    stats = SpecDecodeStats(5, 4, 12, 9, 2.0, 14)
    assert stats.acceptance_rate == 0.75
    assert stats.tokens_per_second == 7
    assert stats.speedup_vs_baseline == 2.8
    assert stats.mean_accepted_per_step == 2.25


def test_public_dense_kv_helper_preserves_mapping_and_request_row():
    keys = np.arange(24).reshape(2, 3, 4)
    values = keys + 100
    cache = SimpleNamespace(views=[SimpleNamespace(key=keys, value=values), SimpleNamespace()])
    pairs = build_target_kv_pairs(cache, [1, 0, 8, -1], batch_index=1, kv_len=2)
    assert pairs[0] is pairs[2] is pairs[3] is None
    np.testing.assert_array_equal(pairs[1][0], keys[1:2, :2])
    np.testing.assert_array_equal(pairs[1][1], values[1:2, :2])
    with pytest.raises(ValueError, match="views"):
        build_target_kv_pairs(object(), [0])
