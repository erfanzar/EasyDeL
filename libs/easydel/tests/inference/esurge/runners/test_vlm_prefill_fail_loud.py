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

"""VLM prefill must fail loudly on a vision-encode error, not serve garbage.

``eSurgeRunner._precompute_vlm_prefill`` previously swallowed every exception
during vision encoding and returned, leaving ``prefill_inputs_embeds`` unset.
The compiled step then ran against the raw prompt token ids (image placeholder
tokens never replaced by visual embeddings), silently producing garbage. The
fix re-raises so the request is rejected instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easydel.inference.esurge.runners.model_runner import eSurgeRunner


def _runner_that_fails_vision(exc: Exception) -> eSurgeRunner:
    runner = object.__new__(eSurgeRunner)
    # model_uses_mrope tolerates a config-less model (returns False).
    runner.model = SimpleNamespace()

    def _raise(*args, **kwargs):
        raise exc

    runner._compute_vlm_prefill_with_compiled_vision = _raise
    runner._compute_embedding_with_info_single_pass = _raise
    return runner


def _vision_req_state() -> SimpleNamespace:
    return SimpleNamespace(
        req_id="vlm-req-0",
        vision_processed=False,
        _vision_processed=False,
        pixel_values=object(),  # non-None -> real vision input to encode
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        prefill_inputs_embeds=None,
        prompt_token_ids=[1, 2, 3, 4],
    )


def test_vision_encode_failure_raises_instead_of_serving_garbage():
    """A vision-encode error must surface as a RuntimeError, not be swallowed."""
    original = ValueError("mosaic lowering failed for vision tower")
    runner = _runner_that_fails_vision(original)
    req_state = _vision_req_state()

    with pytest.raises(RuntimeError, match="vision encode failed"):
        runner._precompute_vlm_prefill(req_state)

    # Guard against silent degradation: embeddings must not have been produced.
    assert req_state.prefill_inputs_embeds is None


def test_vision_encode_failure_chains_original_cause():
    """The original encode error is preserved as the exception cause."""
    original = RuntimeError("kernel dispatch error")
    runner = _runner_that_fails_vision(original)
    req_state = _vision_req_state()

    with pytest.raises(RuntimeError) as excinfo:
        runner._precompute_vlm_prefill(req_state)

    assert excinfo.value.__cause__ is original


def test_no_raw_vision_inputs_is_still_a_clean_skip():
    """The genuinely-recoverable path (cached features only) must not raise."""
    runner = _runner_that_fails_vision(ValueError("should not be called"))
    req_state = _vision_req_state()
    req_state.pixel_values = None
    req_state.pixel_values_videos = None

    # No raw pixels to encode: mark processed and return without touching the
    # (failing) vision encode path.
    runner._precompute_vlm_prefill(req_state)
    assert req_state._vision_processed is True
