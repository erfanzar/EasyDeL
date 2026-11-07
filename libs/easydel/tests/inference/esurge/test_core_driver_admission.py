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
from easydel.inference.esurge.config import eSurgeContextConfig
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.engine.admission import RequestAdmission
from easydel.inference.esurge.engine.core_driver import FINISH_ABORT, EngineCoreDriver
from easydel.inference.esurge.engine.registry import RequestRegistry
from easydel.inference.esurge.engine_types import FinishReason
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.outputs import ModelRunnerOutput
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


_EOS_IDS = (154820, 154827, 154829)


def _admission_with_three_eos_ids() -> RequestAdmission:
    return RequestAdmission(
        registry=RequestRegistry(),
        scheduler_submit=lambda requests: None,
        tokenizer_client=None,
        tokenizer=None,
        context_config=eSurgeContextConfig.coerce_config(None),
        reserve_tokens=0,
        max_model_len=64,
        tool_parser_class=None,
        reasoning_parser_class=None,
        sampling_params_callback=lambda: None,
        generation_config_dict={"eos_token_id": list(_EOS_IDS)},
        primary_eos_token_id=_EOS_IDS[0],
        eos_token_ids=list(_EOS_IDS),
        extra_stops=[],
        ignore_stop_strings_in_reasoning=False,
        on_activity=lambda: None,
        info=lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_every_eos_id_stops_unless_ignore_eos_and_caller_stop_ids_always_stop(ignore_eos):
    admission = _admission_with_three_eos_ids()
    # One request samples each EOS id; the last samples a stop id the caller passed.
    sampled = {f"eos{i}": token for i, token in enumerate(_EOS_IDS)} | {"caller-stop": 777}
    records = []
    for rid in sampled:
        params = admission.prepare_sampling_params_for_request(
            SamplingParams(max_tokens=8, ignore_eos=ignore_eos, stop_token_ids=[777]), request_id=rid, prompt="p"
        )
        records += admission.add_request(rid, "p", params, prompt_token_ids=[1, 2, 3, 4], defer_scheduler_enqueue=True)
    core = _core(dp=1)
    try:
        core.add_requests(records)
        step = core.schedule()
        assert step.num_scheduled_tokens == dict.fromkeys(sampled, 4)
        rids = list(sampled)
        outputs = core.commit(
            step,
            ModelRunnerOutput(
                req_ids=rids,
                req_id_to_index={rid: i for i, rid in enumerate(rids)},
                sampled_token_ids=[[sampled[rid]] for rid in rids],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        finish = {out.request_id: out.finish_reason for out in outputs[0].outputs}
        for i in range(len(_EOS_IDS)):
            assert finish[f"eos{i}"] == (None if ignore_eos else FinishReason.STOP)
            assert core.is_live(f"eos{i}") is ignore_eos
        assert finish["caller-stop"] == FinishReason.STOP
        assert not core.is_live("caller-stop")
    finally:
        core.shutdown()


@pytest.mark.parametrize("name", [Scheduler, EngineRequest])
def test_removed_names_raise_and_point_to_the_replacement(name):
    with pytest.raises(RuntimeError, match="EngineCore"):
        name()
