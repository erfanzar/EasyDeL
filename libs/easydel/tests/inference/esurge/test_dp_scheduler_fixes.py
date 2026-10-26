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

"""DPScheduler regressions: pending-structured flag, image expansion, start-method default."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler
from easydel.inference.esurge.scheduler.output import SchedulerOutput
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

# Host bookkeeping only: the fake expander uses NumPy patches, never JAX array computation.
pytestmark = pytest.mark.cpu_ok


def _kv_config() -> CacheGroupsConfig:
    return CacheGroupsConfig(
        num_pages=64,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False
                ),
                layer_names=None,
            )
        ],
    )


def _make_scheduler() -> DPScheduler:
    return DPScheduler(
        kv_cache_config=_kv_config(),
        dp_size=2,
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        use_worker_processes=False,
    )


def _request(req_id: str, *, images=None) -> EngineRequest:
    return EngineRequest(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=4),
        eos_token_id=0,
        images=images,
    )


def test_scheduler_output_declares_pending_structured_output_tokens() -> None:
    scheduler = _make_scheduler()
    try:
        output = scheduler.schedule()
        assert output.pending_structured_output_tokens is False
    finally:
        scheduler.shutdown()


def test_dp_schedule_propagates_pending_structured_output_flag_from_any_rank() -> None:
    scheduler = _make_scheduler()
    try:
        assert scheduler.schedule().pending_structured_output_tokens is False

        rank1 = scheduler.schedulers[1]
        original = rank1.schedule

        def flagged() -> SchedulerOutput:
            out = original()
            out.pending_structured_output_tokens = True
            return out

        rank1.schedule = flagged
        assert scheduler.schedule().pending_structured_output_tokens is True
    finally:
        scheduler.shutdown()


class _FakeExpander:
    def expand(self, prompt_token_ids, images):
        token_ids = [*prompt_token_ids, 900, 901]
        return SimpleNamespace(
            token_ids=token_ids,
            token_types=[-1] * len(token_ids),
            spans=[(4, 6)],
            features=[
                MultiModalFeature.from_image(
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                    np.array([[1, 1, 2]], dtype=np.int32),
                    PlaceholderRange(4, 2),
                )
            ],
        )


def test_dp_scheduler_expands_images_before_routing() -> None:
    scheduler = _make_scheduler()
    try:
        scheduler.image_expander = _FakeExpander()
        request = _request("img0", images=[object()])
        scheduler.add_request(request)

        assert request.images is None
        assert request.prompt_token_ids == [1, 2, 3, 4, 900, 901]
        assert request.image_spans == [(4, 6)]
        rank = scheduler.req_id_to_dp_rank["img0"]
        assert "img0" in scheduler.schedulers[rank].requests
    finally:
        scheduler.shutdown()


def test_dp_scheduler_rejects_images_without_expander_and_does_not_register() -> None:
    scheduler = _make_scheduler()
    try:
        request = _request("img1", images=[object()])
        with pytest.raises(ValueError, match="does not declare image support"):
            scheduler.add_request(request)
        assert "img1" not in scheduler.req_id_to_dp_rank
    finally:
        scheduler.shutdown()


def test_start_method_defaults_to_spawn_and_ignores_environment(monkeypatch) -> None:
    monkeypatch.setenv("EASYDEL_DP_SCHEDULER_START_METHOD", "fork")
    scheduler = _make_scheduler()
    try:
        assert scheduler.worker_start_method == "spawn"
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("use_worker_processes", [False, True])
def test_dp_from_runner_attaches_model_image_expander(use_worker_processes) -> None:
    class ImageModel:
        config = SimpleNamespace()

        @staticmethod
        def esurge_image_expander_builder(model):
            return _FakeExpander()

    class InProcessDPScheduler(DPScheduler):
        def __init__(self, **kwargs):
            super().__init__(use_worker_processes=False, **kwargs)

    runner = SimpleNamespace(
        model=ImageModel(),
        kv_cache_groups=_kv_config().kv_cache_groups,
        metadata=SimpleNamespace(num_pages=64, page_size=4, data_parallel_size=2),
        max_num_seqs=4,
        max_model_len=64,
        max_num_seq_buckets=[4],
    )
    factory = Scheduler if use_worker_processes else InProcessDPScheduler
    scheduler = factory.from_runner(
        runner,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    try:
        request = _request("from-runner-image", images=[object()])
        scheduler.add_request(request)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"from-runner-image": 6}
        new_request = output.scheduled_new_reqs[0]
        assert new_request.prompt_token_ids == [1, 2, 3, 4, 900, 901]
        assert new_request.has_vision
        assert len(new_request.mm_features) == 1
        feature = new_request.mm_features[0]
        assert feature.modality == "image"
        assert feature.placeholder_range == PlaceholderRange(4, 2)
        np.testing.assert_array_equal(feature.pixel_values, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(feature.grid_thw, [[1, 1, 2]])
        assert request.image_spans == [(4, 6)]
    finally:
        scheduler.shutdown()
