"""Host-only rank ownership and retained multimodal request-state regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.model_runner import eSurgeRunner
from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
from easydel.inference.esurge.runners.slot_pool import RecurrentSlotPool
from easydel.inference.esurge.runners.state_sync import SequenceStateSync
from easydel.inference.esurge.scheduler.output import CachedRequestData, NewRequestData, SchedulerOutput
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("dp_size", [2, 4])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_rank_major_vision_admission_preserves_features_grids_expanded_prompt_and_slots(dp_size, modality):
    rank = dp_size - 1
    row = 2 * rank
    capacity = 2 * dp_size
    metadata = SimpleNamespace(data_parallel_size=dp_size, num_pages=16 * dp_size)
    buffer = SequenceBuffer(capacity, 16, 16, 32, [2])
    pool = RecurrentSlotPool(metadata=metadata, max_num_reqs=capacity, slot_indexed_state=False)
    requests = {}
    sync = SequenceStateSync(
        sequence_buffer=buffer,
        requests=requests,
        slot_pool=pool,
        metadata=metadata,
        executor_manager=SimpleNamespace(clear_recurrent_slots=lambda slots: None),
        max_model_len=16,
        spec_decode_recurrent_candidates=False,
        spec_recurrent_commit_cpu=np.zeros((2, capacity), dtype=np.int32),
        pending_spec_recurrent_commit_by_req={},
    )
    pixels = np.asarray([[1.0, 2.0]], dtype=np.float32)
    grids = np.asarray([[1, 2, 2]], dtype=np.int32)
    feature = SimpleNamespace(embeddings=np.asarray([[3.0, 4.0]]))
    pixel_key, grid_key = (
        ("pixel_values", "image_grid_thw") if modality == "image" else ("pixel_values_videos", "video_grid_thw")
    )
    pages = ([16 * rank + 1, 16 * rank + 2, 16 * rank + 3],)
    vision = NewRequestData(
        "vision",
        [1, 9, 9, 9, 9, 2],
        SamplingParams(),
        pages,
        0,
        dp_rank=rank,
        mm_features=[feature],
        **{pixel_key: pixels, grid_key: grids},
    )
    text = NewRequestData("text", [1, 2], SamplingParams(), ([1],), 0, dp_rank=0)
    schedule = SchedulerOutput(
        [vision, text], CachedRequestData.make_empty(), {"vision": 6, "text": 2}, 8, {}, [0], set()
    )
    assert sync.update_states(schedule)
    assert buffer.req_id_to_index == {"vision": row, "text": 0}
    assert pool.dp_rank_by_req == {"vision": rank, "text": 0}
    assert pool.slot_by_req == {"vision": row, "text": 0}
    np.testing.assert_array_equal(buffer.token_ids[row, :6], [1, 9, 9, 9, 9, 2])
    state = requests["vision"]
    assert getattr(state, pixel_key) is pixels
    assert getattr(state, grid_key) is grids
    assert state.mm_features[0] is feature
    assert state.num_prompt_tokens == 6
    # Host precompute supplies these after admission; lifecycle updates must
    # retain them across chunked prefill and preemption, not re-admit a blank state.
    embeds = np.arange(12, dtype=np.float32).reshape(6, 2)
    positions = np.arange(18, dtype=np.int32).reshape(3, 6)
    visual_mask = np.asarray([False, True, True, True, True, False])
    deepstack = [np.asarray([[3.0, 4.0]] * 4)]
    token_types = np.asarray([-1, 0, 1, 2, 3, -1], dtype=np.int32)
    state.prefill_inputs_embeds = embeds
    state.prefill_position_ids = positions
    state.prefill_visual_pos_masks = visual_mask
    state.prefill_deepstack_visual_embeds = deepstack
    state.token_types = token_types
    keep = SchedulerOutput(
        [], CachedRequestData(["vision"], [False], [[]], [([],)], [2], [rank]), {"vision": 2}, 2, {}, [0], set()
    )
    assert not sync.update_states(keep)
    preempt = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"vision"})
    assert sync.update_states(preempt)
    assert "vision" not in buffer.req_id_to_index
    resume = SchedulerOutput(
        [], CachedRequestData(["vision"], [True], [[]], [pages], [2], [rank]), {"vision": 2}, 2, {}, [0], set()
    )
    assert sync.update_states(resume)
    assert requests["vision"] is state
    assert buffer.req_id_to_index["vision"] == row
    assert pool.slot_by_req["vision"] == row
    assert pool.dp_rank_by_req["vision"] == rank
    assert getattr(state, pixel_key) is pixels
    assert getattr(state, grid_key) is grids
    assert state.mm_features[0] is feature
    assert state.prefill_inputs_embeds is embeds
    assert state.prefill_position_ids is positions
    assert state.prefill_visual_pos_masks is visual_mask
    assert state.prefill_deepstack_visual_embeds is deepstack
    assert state.token_types is token_types
    np.testing.assert_array_equal(buffer.token_ids[row, :6], [1, 9, 9, 9, 9, 2])

    # Use the live synchronized request map with the real token-type provider.
    # Rank 0 text, empty middle ranks and last-rank media must stay disjoint.
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.model = SimpleNamespace(esurge_requires_token_types=True)
    runner.metadata = metadata
    runner.num_reqs_max_model_len = capacity
    runner.sequence_buffer = buffer
    runner.requests = requests
    runner._side_input_provider = None
    runner._side_input_tokenizer_state = None
    payload = runner.side_inputs_for_execute(
        token_ids_cpu=buffer.token_ids[[row, 0]],
        num_computed_tokens_cpu=np.asarray([2, 0]),
        scheduled_full_cpu=np.asarray([2, 2]),
        window_row_indices_cpu=np.asarray([row, 0]),
        num_tokens_static=4 * dp_size,
    )
    expected = np.full(4 * dp_size, -1, dtype=np.int32)
    expected[4 * rank : 4 * rank + 2] = [1, 2]
    np.testing.assert_array_equal(payload["token_types"], expected)
    assert payload["token_types"].dtype == np.int32


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_token_window_offsets_preserve_rank_padding_and_compact_request_order(dp_size):
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.metadata = SimpleNamespace(data_parallel_size=dp_size)
    runner.num_reqs_max_model_len = 8
    rows = [6, 0, 7, 2]
    offsets, bases = runner.token_offsets_for_window(
        scheduled_list=[2, 3, 1, 1], window_row_indices=rows, num_tokens_static=16
    )
    expected = {1: ([0, 2, 5, 6], [0, 0, 0, 0]), 2: ([8, 0, 10, 3], [8, 0, 8, 0]), 4: ([12, 0, 14, 4], [12, 0, 12, 4])}
    np.testing.assert_array_equal(offsets, expected[dp_size][0])
    np.testing.assert_array_equal(bases, expected[dp_size][1])


def test_unbalanced_vision_window_cannot_spill_into_another_rank():
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.metadata = SimpleNamespace(data_parallel_size=2)
    runner.num_reqs_max_model_len = 4
    with pytest.raises(ValueError, match="rank-local token bucket"):
        runner.token_offsets_for_window(scheduled_list=[3, 2], window_row_indices=[0, 1], num_tokens_static=8)
