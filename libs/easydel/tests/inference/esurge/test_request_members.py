"""Documented EngineRequest helpers keep working (conversion from the wire request, vision lifecycle, NaN flag)."""

import numpy as np
import pytest
from easydel.inference.esurge.engine_types import EngineCoreRequest
from easydel.inference.esurge.request import EngineRequest

pytestmark = pytest.mark.cpu_ok


def test_from_engine_core_request_copies_request_fields():
    pixels = np.ones((4, 8), dtype=np.float32)
    video_pixels = np.zeros((2, 4, 8), dtype=np.float32)
    core = EngineCoreRequest(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=None,
        eos_token_id=2,
        arrival_time=12.5,
        data_parallel_rank=None,
        priority=3,
        client_index=5,
        pixel_values=pixels,
        pixel_values_videos=video_pixels,
    )
    request = EngineRequest.from_engine_core_request(core)
    assert request.request_id == "r1"
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.arrival_time == 12.5 and request.priority == 3 and request.client_index == 5
    assert request.pixel_values is pixels and request.pixel_values_videos is video_pixels and request.has_vision
    # The wire struct carries no n>1 parent metadata; the engine assigns those at admission.
    assert request.parent_request_id is None and request.sample_index == 0


def test_clear_vision_data_marks_request_processed():
    request = EngineRequest("r", [1], None, 2, pixel_values=np.ones((1, 1), dtype=np.float32))
    assert request.has_vision and not request.vision_processed
    request.clear_vision_data()
    assert request.pixel_values is None and request.vision_processed and not request.has_vision


def test_is_output_corrupted_tracks_nan_count():
    request = EngineRequest("r", [1], None, 2)
    assert not request.is_output_corrupted
    request.num_nans_in_logits = 2
    assert request.is_output_corrupted
