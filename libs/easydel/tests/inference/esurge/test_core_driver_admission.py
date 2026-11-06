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

"""Engine-core admission: image expansion before routing, the registry, and the runner's step view."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.engine.core_driver import FINISH_ABORT, EngineCoreDriver
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.request import AdmissionRecord, EngineRequest
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

# Host bookkeeping only: the core is numpy-in/numpy-out and the fake expander uses NumPy patches.
pytestmark = pytest.mark.cpu_ok


def _groups() -> list[CacheGroupSpec]:
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    return [CacheGroupSpec(kv_cache_spec=spec, layer_names=None)]


def _core(image_expander=None, dp: int = 2) -> EngineCoreDriver:
    return EngineCoreDriver(
        kv_cache_groups=_groups(),
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        num_pages=64,
        data_parallel_size=dp,
        enable_prefix_caching=False,
        async_scheduling=False,
        image_expander=image_expander,
    )


def _record(req_id: str, *, images=None, parent=None) -> AdmissionRecord:
    return AdmissionRecord(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=4),
        eos_token_id=0,
        images=images,
        parent_request_id=parent,
    )


class _FakeExpander:
    def expand(self, prompt_token_ids, images):
        token_ids = [*prompt_token_ids, 900, 901]
        return SimpleNamespace(
            token_ids=token_ids,
            token_types=np.asarray([-1, -1, -1, -1, 0, 0], np.int64),
            spans=[(4, 6)],
            features=[
                MultiModalFeature.from_image(
                    np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
                    np.array([[1, 1, 2]], dtype=np.int32),
                    PlaceholderRange(4, 2),
                )
            ],
        )


def test_images_are_expanded_before_routing():
    core = _core(_FakeExpander())
    try:
        record = _record("img0", images=[object()])
        core.add_requests([record])
        assert record.images is None
        assert record.prompt_token_ids == [1, 2, 3, 4, 900, 901]
        assert record.image_spans == [(4, 6)]
        assert core.is_live("img0")
        out = core.schedule()
        assert out.num_scheduled_tokens == {"img0": 6}
        new = out.scheduled_new_reqs[0]
        assert new.prompt_token_ids == [1, 2, 3, 4, 900, 901]
        assert new.has_vision and new.mm_features[0].placeholder_range == PlaceholderRange(4, 2)
        assert new.dp_rank in (0, 1)
    finally:
        core.shutdown()


def test_images_without_an_expander_are_rejected_and_not_registered():
    core = _core()
    try:
        with pytest.raises(ValueError, match="does not declare image support"):
            core.add_requests([_record("img1", images=[object()])])
        assert not core.is_live("img1")
        assert core.num_waiting == 0
    finally:
        core.shutdown()


def test_image_span_longer_than_the_budget_is_rejected():
    class _Long(_FakeExpander):
        def expand(self, prompt_token_ids, images):
            out = super().expand(prompt_token_ids, images)
            out.token_ids = [*prompt_token_ids, *([900] * 20)]
            out.spans = [(4, 24)]
            return out

    core = _core(_Long())
    try:
        with pytest.raises(ValueError, match="exceeds max_num_batched_tokens"):
            core.add_requests([_record("img2", images=[object()])])
        assert not core.is_live("img2")
    finally:
        core.shutdown()


def test_from_runner_attaches_the_model_image_expander():
    class ImageModel:
        config = SimpleNamespace()

        @staticmethod
        def esurge_image_expander_builder(model):
            return _FakeExpander()

    runner = SimpleNamespace(
        model=ImageModel(),
        kv_cache_groups=_groups(),
        metadata=SimpleNamespace(num_pages=64, page_size=4, data_parallel_size=2),
        max_num_seqs=4,
        max_model_len=64,
        max_num_seq_buckets=[4],
        _has_recurrent_layers=False,
    )
    core = EngineCoreDriver.from_runner(
        runner, max_num_batched_tokens=16, enable_prefix_caching=False, async_scheduling=False
    )
    try:
        core.add_requests([_record("from-runner-image", images=[object()])])
        out = core.schedule()
        assert out.num_scheduled_tokens == {"from-runner-image": 6}
        feature = out.scheduled_new_reqs[0].mm_features[0]
        np.testing.assert_array_equal(feature.pixel_values, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(feature.grid_thw, [[1, 1, 2]])
    finally:
        core.shutdown()


def test_registry_resolves_children_and_abort_frees_them():
    core = _core(dp=1)
    try:
        core.add_requests([_record("p-0", parent="p"), _record("p-1", parent="p"), _record("q")])
        assert sorted(core.children_of("p")) == ["p-0", "p-1"]
        assert core.finish_requests(["p-0", "p-1", "missing"], FINISH_ABORT) == ["p-0", "p-1"]
        assert not core.is_live("p-0") and core.is_live("q")
        out = core.schedule()
        assert out.finished_req_ids == {"p-0", "p-1"}
        assert list(out.num_scheduled_tokens) == ["q"]
    finally:
        core.shutdown()


def test_duplicate_live_id_is_rejected():
    core = _core(dp=1)
    try:
        core.add_requests([_record("dup")])
        with pytest.raises(ValueError, match="already scheduled"):
            core.add_requests([_record("dup")])
    finally:
        core.shutdown()


@pytest.mark.parametrize("name", [Scheduler, EngineRequest])
def test_removed_names_raise_and_point_to_the_replacement(name):
    with pytest.raises(RuntimeError, match="EngineCore"):
        name()
