"""Comprehensive regression tests for eSurge runners, executors, VLM prefill & speculative runner review findings."""

from __future__ import annotations

import gc
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import HybridCache, KDACacheView, RaggedPagesCacheConfig, RecurrentCacheView
from easydel.inference.esurge.multimodal.cache import VisionEncoderCache
from easydel.inference.esurge.multimodal.types import MultiModalFeature, PlaceholderRange
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.runners import host_sync, vlm_prefill
from easydel.inference.esurge.runners import model_runner as mr_mod
from easydel.inference.esurge.runners.async_types import (
    AsyncPreResults,
    AsyncWindowResult,
    DeviceInputTokenHandoff,
    DeviceSpecWindowHandoff,
    spec_window_sources,
)
from easydel.inference.esurge.runners.execution_manager import ExecutionManager
from easydel.inference.esurge.runners.execution_types import BackboneOutputs, CacheGroupInputs
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from easydel.inference.esurge.runners.executors.compile_orchestrator import CompileOrchestrator
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.inference.esurge.runners.executors.pipeline_microbatch import PipelineMicrobatchExecutor
from easydel.inference.esurge.runners.executors.sampler_executor import SamplerExecutor
from easydel.inference.esurge.runners.model_runner import _AsyncExecutionHandle, eSurgeRunner
from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
from easydel.inference.esurge.runners.slot_pool import RecurrentSlotPool
from easydel.inference.esurge.runners.spec.interface import NullSpeculation
from easydel.inference.esurge.runners.spec.rejection import rejection_sample_from_uniforms
from easydel.inference.esurge.runners.spec.sampling import score_sampled_tokens
from easydel.inference.esurge.runners.spec.strategy import (
    DrafterSpeculation,
    _restore_recurrent_state_for_slots,
    snapshot_recurrent_state,
)
from easydel.inference.esurge.runners.spec.verify import (
    NUM_PARAM_FIELDS,
    NUM_ROW_FIELDS,
    PARAM_MIN_P,
    PARAM_TEMPERATURE,
    PARAM_TOP_K,
    PARAM_TOP_P,
    ROW_DRAFTS,
    ROW_GREEDY,
    ROW_LIVE,
    ROW_REAL,
    ROW_SEQ,
    ROW_START,
    filter_logits,
    pack_pooled_drafts,
    pack_rows,
    resolve_rows,
    widen_spec_out,
)
from easydel.inference.esurge.runners.state_sync import SequenceStateSync
from easydel.inference.esurge.runners.states import CachedRequestState
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper
from easydel.inference.esurge.runners.window_planner import WindowPlanner
from easydel.inference.esurge.scheduler.output import CachedRequestData, NewRequestData, SchedulerOutput
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.sampling_processors import SamplingConstraints, compact_sampling_constraints
from easydel.modules.deepseek_v41.processing_deepseek_v41 import compute_prefill_embeddings
from easydel.modules.dspark._drafter import DSparkDrafter
from jax.sharding import Mesh, NamedSharding, PartitionSpec

pytestmark = pytest.mark.cpu_ok


# =============================================================================
# Shared Helpers (VLM / State / Spec / Runner Stubs)
# =============================================================================


def _make_sync(
    *,
    dp_size: int = 1,
    max_num_reqs: int = 4,
    num_pages: int = 32,
    slot_indexed_state: bool = False,
    spec_decode_recurrent_candidates: bool = False,
    defer_row_clears: bool = False,
):
    metadata = SimpleNamespace(data_parallel_size=dp_size, num_pages=num_pages)
    buf = SequenceBuffer(max_num_reqs, 32, 32, 64, [4])
    pool = RecurrentSlotPool(metadata=metadata, max_num_reqs=max_num_reqs, slot_indexed_state=slot_indexed_state)
    requests: dict[str, CachedRequestState] = {}
    clears: list[list[int]] = []
    commit_cpu = np.zeros((2, max_num_reqs), dtype=np.int32)
    pending_commit: dict[str, int] = {}
    sync = SequenceStateSync(
        sequence_buffer=buf,
        requests=requests,
        slot_pool=pool,
        metadata=metadata,
        executor_manager=SimpleNamespace(clear_recurrent_slots=lambda slots: clears.append(list(slots))),
        max_model_len=32,
        spec_decode_recurrent_candidates=spec_decode_recurrent_candidates,
        spec_recurrent_commit_cpu=commit_cpu,
        pending_spec_recurrent_commit_by_req=pending_commit,
        defer_row_clears=defer_row_clears,
    )
    return sync, buf, requests, pool, clears, commit_cpu, pending_commit


class _MropeDeepStackModel:
    def __init__(self, *, support_video_embeds: bool = True, support_image_embeds: bool = True):
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(deepstack_visual_indexes=[0, 1], spatial_merge_size=2)
        )
        self.image_calls = 0
        self.video_calls = 0
        self.embed_calls = 0
        self.last_embed_kwargs: dict | None = None
        if support_image_embeds and support_video_embeds:
            self.compute_embedding = self._compute_embedding_both
        elif support_image_embeds:
            self.compute_embedding = self._compute_embedding_image_only
        else:
            self.compute_embedding = self._compute_embedding_none
        if not support_video_embeds:
            self.get_video_features = None

    def _compute_embedding_both(self, ids, *, image_embeds=None, video_embeds=None):
        raise AssertionError("should call compute_embedding_with_info")

    def _compute_embedding_image_only(self, ids, *, image_embeds=None):
        raise AssertionError("should call compute_embedding_with_info")

    def _compute_embedding_none(self, ids):
        raise AssertionError("should call compute_embedding_with_info")

    def get_image_features(self, pixels, image_grid_thw=None):
        self.image_calls += 1
        return ((pixels + 10.0,), (pixels + 20.0, pixels + 30.0))

    def get_video_features(self, pixels, video_grid_thw=None):
        self.video_calls += 1
        return ((pixels + 100.0,), (pixels + 200.0, pixels + 300.0))

    def compute_embedding_with_info(
        self,
        ids,
        *,
        pixel_values=None,
        pixel_values_videos=None,
        image_embeds=None,
        video_embeds=None,
        deepstack_image_embeds=None,
        deepstack_video_embeds=None,
        **kwargs,
    ):
        self.embed_calls += 1
        self.last_embed_kwargs = dict(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_embeds=image_embeds,
            video_embeds=video_embeds,
            deepstack_image_embeds=deepstack_image_embeds,
            deepstack_video_embeds=deepstack_video_embeds,
            **kwargs,
        )
        if image_embeds is None and pixel_values is not None:
            embeds_t, ds = self.get_image_features(pixel_values, kwargs.get("image_grid_thw"))
            image_embeds = embeds_t[0]
            deepstack_image_embeds = ds
        if video_embeds is None and pixel_values_videos is not None:
            video_embeds = pixel_values_videos + 100.0
            deepstack_video_embeds = (pixel_values_videos + 200.0, pixel_values_videos + 300.0)
        embeds = np.repeat(ids[..., None], 2, axis=-1).astype(np.float32)
        mask = (ids == 90) | (ids == 91)
        layers = [np.zeros((int(mask.sum()), 2), dtype=np.float32) for _ in range(2)]
        visual_ids = ids[mask]
        for token, rows, aux in (
            (90, image_embeds, deepstack_image_embeds),
            (91, video_embeds, deepstack_video_embeds),
        ):
            if rows is not None:
                embeds[ids == token] = rows
                for layer, res in zip(layers, aux, strict=True):
                    layer[visual_ids == token] = res
        pos = np.stack([np.arange(ids.shape[-1], dtype=np.int32)[None, :]] * 3)
        deltas = np.array([[-2]], dtype=np.int32)
        return embeds, SimpleNamespace(
            position_ids=pos,
            rope_deltas=deltas,
            visual_pos_masks=mask,
            deepstack_visual_embeds=layers,
        )


class _TokenTypeModel:
    def __init__(self):
        self.config = SimpleNamespace(vision_params=lambda: {"downsample_ratio": 3})
        self.mesh = nullcontext()
        self.dtype = np.float32
        self.base_model = self
        self.calls = 0

    def get_image_features(self, pixels, image_grid):
        self.calls += 1
        return np.full((1, 1, 2), pixels.flat[0], dtype=np.float32)

    def compute_embedding(self, ids):
        return np.repeat(ids[..., None], 2, axis=-1).astype(np.float32)

    def merge_image_embeddings(self, embeds, types, rows):
        result = embeds.copy()
        result[types == 1] = rows.reshape(-1, 2)
        return result

    def esurge_compute_prefill_embeddings(self, ids, features, encoder_cache_provider=None):
        return compute_prefill_embeddings(self, ids, features, encoder_cache_provider=encoder_cache_provider)


def _vlm_helper(monkeypatch, model, *, cache: bool = True):
    monkeypatch.setattr(vlm_prefill, "jnp", np)
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(vlm_prefill, "model_uses_mrope", lambda m: isinstance(m, _MropeDeepStackModel))
    monkeypatch.setattr(vlm_prefill, "model_needs_token_types", lambda m: isinstance(m, _TokenTypeModel))
    return VlmPrefillHelper(
        model_getter=lambda: model,
        metadata=SimpleNamespace(data_parallel_size=1),
        compile_vision_encoder=False,
        vision_patch_buckets=None,
        max_num_batched_tokens=64,
        max_num_tokens=64,
        vision_cache=VisionEncoderCache(capacity_mb=4) if cache else None,
    )


def _img_feat(value: float = 1.0, offset: int = 1):
    return MultiModalFeature.from_image(
        np.full((1, 2), value, dtype=np.float32),
        np.array([[1, 1, 1]], dtype=np.int32),
        PlaceholderRange(offset, 1, "image"),
    )


def _vid_feat(value: float = 2.0, offset: int = 3):
    return MultiModalFeature.from_video(
        np.full((1, 2), value, dtype=np.float32),
        np.array([[1, 1, 1]], dtype=np.int32),
        PlaceholderRange(offset, 1, "video"),
    )


def _make_spec_strategy(*, drafter=None, num_draft_tokens: int = 3, max_num_reqs: int = 4, draft_schedule=None):
    buf = SequenceBuffer(max_num_reqs, 64, 64, 32, [4])
    exec_mgr = SimpleNamespace(
        rng_key=jax.random.PRNGKey(0),
        kv_pages=None,
        last_spec_window_delta=None,
        overwrite_sampler_penalty_rows=lambda rows: None,
    )
    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_seqs=max_num_reqs,
        max_model_len=64,
        num_draft_tokens=num_draft_tokens,
        draft_schedule=draft_schedule,
        sequence_buffer=buf,
        requests={},
        executor_manager=exec_mgr,
        spec_decode_recurrent_candidates=False,
        spec_decode_recurrent_replay=False,
        async_scheduling=True,
        _spec_carry=None,
        _spec_recorded_this_step=False,
        record_spec_out=lambda spec_out, replace=False: None,
    )
    strategy = DrafterSpeculation(
        runner=runner,
        drafter=drafter,
        num_draft_tokens=num_draft_tokens,
        draft_schedule=draft_schedule,
    )
    return strategy, runner, buf, exec_mgr


# =============================================================================
# S1-C3: Multimodal Vision Prefill Sharing & Cached-Only Hydration
# =============================================================================


def test_s1_c3_shared_mm_features_n_gt_1_cached_model(monkeypatch):
    model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    feat = _img_feat(3.0, offset=1)
    pv = feat.pixel_values
    grid = feat.grid_thw

    req0 = CachedRequestState(
        "r-0", [7, 90, 8], SamplingParams(), None, ([1],), 0, [], pixel_values=pv, image_grid_thw=grid, mm_features=[feat]
    )
    req1 = CachedRequestState(
        "r-1", [7, 90, 8], SamplingParams(), None, ([2],), 0, [], pixel_values=pv, image_grid_thw=grid, mm_features=[feat]
    )

    h.precompute_prefill(req0)
    h.precompute_prefill(req1)
    assert model.image_calls == 1
    np.testing.assert_array_equal(req0.prefill_inputs_embeds, req1.prefill_inputs_embeds)
    np.testing.assert_array_equal(req0.prefill_position_ids, req1.prefill_position_ids)
    np.testing.assert_array_equal(req0.prefill_deepstack_visual_embeds[0], req1.prefill_deepstack_visual_embeds[0])


def test_s1_c3_shared_mm_features_n_gt_1_native_model_finish_does_not_break_sibling(monkeypatch):
    model = _MropeDeepStackModel(support_image_embeds=False, support_video_embeds=False)
    h = _vlm_helper(monkeypatch, model, cache=False)
    sync, _, requests, _, _, _, _ = _make_sync()
    feat = _img_feat(4.0, offset=1)
    pv = feat.pixel_values
    grid = feat.grid_thw

    admit = SchedulerOutput(
        [
            NewRequestData(
                "r-0", [7, 90, 8], SamplingParams(), ([1],), 0, pixel_values=pv, image_grid_thw=grid, mm_features=[feat]
            ),
            NewRequestData(
                "r-1", [7, 90, 8], SamplingParams(), ([2],), 0, pixel_values=pv, image_grid_thw=grid, mm_features=[feat]
            ),
        ],
        CachedRequestData.make_empty(),
        {"r-0": 3, "r-1": 3},
        6,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    h.precompute_prefill(requests["r-0"])
    assert requests["r-0"].pixel_values is pv
    assert feat.pixel_values is pv

    finish_0 = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"r-0"})
    sync.update_states(finish_0)
    assert feat.pixel_values is pv
    assert requests["r-1"].pixel_values is pv

    h.precompute_prefill(requests["r-1"])
    np.testing.assert_array_equal(requests["r-1"].prefill_inputs_embeds[:, 0], [7.0, 14.0, 8.0])

    req1 = requests["r-1"]
    finish_1 = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"r-1"})
    sync.update_states(finish_1)
    assert req1.pixel_values is None
    assert req1.mm_features == []
    # External caller's MultiModalFeature was never destructively mutated by terminal cleanup.
    assert feat.pixel_values is pv


def test_s1_c3_cached_only_sibling_hydrates_from_vision_cache(monkeypatch):
    model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    feat0 = _img_feat(5.0, offset=1)
    req0 = CachedRequestState(
        "r-0",
        [7, 90, 8],
        SamplingParams(),
        None,
        ([1],),
        0,
        [],
        pixel_values=feat0.pixel_values,
        image_grid_thw=feat0.grid_thw,
        mm_features=[feat0],
    )
    h.precompute_prefill(req0)

    feat1 = MultiModalFeature(mm_hash=feat0.mm_hash, modality="image", pixel_values=None, grid_thw=None)
    req1 = CachedRequestState("r-1", [7, 90, 8], SamplingParams(), None, ([2],), 0, [], mm_features=[feat1])
    h.precompute_prefill(req1)

    assert model.image_calls == 1
    np.testing.assert_array_equal(req1.prefill_inputs_embeds, req0.prefill_inputs_embeds)
    np.testing.assert_array_equal(req1.prefill_deepstack_visual_embeds[1], req0.prefill_deepstack_visual_embeds[1])
    np.testing.assert_array_equal(model.last_embed_kwargs["image_grid_thw"], [[1, 1, 1]])


def test_s1_c3_token_type_cached_only_sibling_hydrates_from_vision_cache(monkeypatch):
    model = _TokenTypeModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    monkeypatch.setattr(jax, "numpy", np)
    feat0 = MultiModalFeature.from_image(
        np.full((9, 3, 1, 1), 6.0, np.float32), np.array([[1, 3, 3]]), PlaceholderRange(1, 4)
    )
    req0 = CachedRequestState("r-0", [7, 99, 99, 99, 99], SamplingParams(), None, ([1],), 0, [], mm_features=[feat0])
    h.precompute_prefill(req0)

    feat1 = MultiModalFeature(
        mm_hash=feat0.mm_hash,
        modality="image",
        pixel_values=None,
        grid_thw=None,
        placeholder_range=PlaceholderRange(1, 4),
    )
    req1 = CachedRequestState("r-1", [7, 99, 99, 99, 99], SamplingParams(), None, ([2],), 0, [], mm_features=[feat1])
    h.precompute_prefill(req1)
    assert model.calls == 1
    np.testing.assert_array_equal(req1.prefill_inputs_embeds, req0.prefill_inputs_embeds)
    np.testing.assert_array_equal(req1.token_types, req0.token_types)


# =============================================================================
# S1-C4: VLM Preemption & Resume Re-Prefill
# =============================================================================


@pytest.mark.parametrize("mode", ["cached_mm", "raw_only_no_mm", "native_no_cache"])
def test_s1_c4_vlm_preemption_resume_reprefills_cleanly(monkeypatch, mode):
    if mode == "native_no_cache":
        model = _MropeDeepStackModel(support_image_embeds=False, support_video_embeds=False)
        h = _vlm_helper(monkeypatch, model, cache=False)
    else:
        model = _MropeDeepStackModel()
        h = _vlm_helper(monkeypatch, model, cache=(mode == "cached_mm"))

    sync, _, requests, _, _, _, _ = _make_sync()
    feat = _img_feat(7.0, offset=1)
    admit = SchedulerOutput(
        [
            NewRequestData(
                "r0",
                [5, 90, 6],
                SamplingParams(),
                ([1],),
                0,
                pixel_values=feat.pixel_values,
                image_grid_thw=feat.grid_thw,
                mm_features=[] if mode == "raw_only_no_mm" else [feat],
            )
        ],
        CachedRequestData.make_empty(),
        {"r0": 3},
        3,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    req = requests["r0"]
    h.precompute_prefill(req)
    orig_embeds = req.prefill_inputs_embeds.copy()
    orig_pos = req.prefill_position_ids.copy()
    orig_deltas = req.prefill_rope_deltas.copy()
    orig_ds = [a.copy() for a in req.prefill_deepstack_visual_embeds]

    decode_step = SchedulerOutput(
        [], CachedRequestData(["r0"], [False], [[11]], [([],)], [3]), {"r0": 1}, 1, {}, [0], set()
    )
    sync.update_states(decode_step)
    assert req.prefill_inputs_embeds is None
    assert req.prefill_rope_deltas is not None
    assert req.has_vision

    preempt = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"r0"})
    sync.update_states(preempt)
    assert req.has_vision

    resume = SchedulerOutput([], CachedRequestData(["r0"], [True], [[]], [([2],)], [0]), {"r0": 3}, 3, {}, [0], set())
    sync.update_states(resume)
    assert req.has_vision
    assert not req.vision_processed
    assert req.prefill_rope_deltas is None

    h.precompute_prefill(req)
    np.testing.assert_array_equal(req.prefill_inputs_embeds, orig_embeds)
    np.testing.assert_array_equal(req.prefill_position_ids, orig_pos)
    np.testing.assert_array_equal(req.prefill_rope_deltas, orig_deltas)
    for got, want in zip(req.prefill_deepstack_visual_embeds, orig_ds, strict=True):
        np.testing.assert_array_equal(got, want)


def test_s1_c4_preemption_resume_full_prompt_hit_preserves_rope_deltas(monkeypatch):
    model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    sync, _, requests, _, _, _, _ = _make_sync()
    feat = _img_feat(2.0, offset=1)
    admit = SchedulerOutput(
        [
            NewRequestData(
                "r0",
                [5, 90, 6],
                SamplingParams(),
                ([1],),
                0,
                pixel_values=feat.pixel_values,
                image_grid_thw=feat.grid_thw,
                mm_features=[feat],
            )
        ],
        CachedRequestData.make_empty(),
        {"r0": 3},
        3,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    req = requests["r0"]
    h.precompute_prefill(req)
    orig_deltas = req.prefill_rope_deltas.copy()
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [False], [[11]], [([],)], [3]), {"r0": 1}, 1, {}, [0], set())
    )
    sync.update_states(
        SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"r0"})
    )
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [True], [[]], [([2],)], [3]), {"r0": 1}, 1, {}, [0], set())
    )
    assert req.vision_processed
    np.testing.assert_array_equal(req.prefill_rope_deltas, orig_deltas)


def test_s1_c4_token_type_preemption_resume(monkeypatch):
    model = _TokenTypeModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    monkeypatch.setattr(jax, "numpy", np)
    sync, _, requests, _, _, _, _ = _make_sync()
    feat = MultiModalFeature.from_image(
        np.full((9, 3, 1, 1), 8.0, np.float32), np.array([[1, 3, 3]]), PlaceholderRange(1, 4)
    )
    sync.update_states(
        SchedulerOutput(
            [NewRequestData("r0", [7, 99, 99, 99, 99], SamplingParams(), ([1],), 0, mm_features=[feat])],
            CachedRequestData.make_empty(),
            {"r0": 5},
            5,
            {},
            [0],
            set(),
        )
    )
    req = requests["r0"]
    h.precompute_prefill(req)
    orig_embeds = req.prefill_inputs_embeds.copy()
    orig_types = req.token_types.copy()
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [False], [[12]], [([],)], [5]), {"r0": 1}, 1, {}, [0], set())
    )
    sync.update_states(
        SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"r0"})
    )
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [True], [[]], [([2],)], [0]), {"r0": 5}, 5, {}, [0], set())
    )
    assert req.has_vision and not req.vision_processed and req.token_types is None
    h.precompute_prefill(req)
    np.testing.assert_array_equal(req.prefill_inputs_embeds, orig_embeds)
    np.testing.assert_array_equal(req.token_types, orig_types)


# =============================================================================
# S1-C7: Multi-Window Async Device Token Handoff
# =============================================================================


def test_s1_c7_multi_window_async_handoff_preserved_across_all_windows(monkeypatch):
    """Window 0 using DeviceInputTokenHandoff must NOT clear pending_pre_async_results before window 1."""
    runner = object.__new__(eSurgeRunner)
    max_num_reqs = 4
    num_reqs_window = 2
    buf = SequenceBuffer(max_num_reqs, 32, 32, 64, [4])
    requests: dict[str, CachedRequestState] = {}
    for i in range(3):
        rid = f"r{i}"
        req = CachedRequestState(rid, [10 + i, 20 + i], SamplingParams(temperature=0.0), None, ([i + 1],), 2, [0])
        requests[rid] = req
        buf.add_request(req)
        buf.num_computed_tokens[i] = 2
        buf.num_tokens_no_spec[i] = 3
        buf.num_tokens[i] = 3
        buf.token_ids[i, 2] = 0

    prev_win0 = AsyncWindowResult(
        req_ids=["r0", "r1"],
        row_positions=[0, 1],
        sampled_token_ids=jnp.asarray([101, 102], dtype=jnp.int32),
        valid_mask=[True, True],
    )
    prev_win1 = AsyncWindowResult(
        req_ids=["r2"],
        row_positions=[0],
        sampled_token_ids=jnp.asarray([103, -1], dtype=jnp.int32),
        valid_mask=[True],
    )
    pre_results = AsyncPreResults(
        windows=[prev_win0, prev_win1],
        request_seq_lens=[
            (0, 0, requests["r0"], 2),
            (1, 1, requests["r1"], 2),
            (2, 2, requests["r2"], 2),
        ],
    )

    handoffs_seen: list[DeviceInputTokenHandoff | None] = []
    repair_calls = 0

    def _fake_execute(**kwargs):
        handoffs_seen.append(kwargs.get("device_token_handoff"))
        padded = int(kwargs["padded_num_reqs"])
        return (
            jnp.full((padded,), 7, dtype=jnp.int32),
            jnp.ones((padded,), dtype=jnp.bool_),
            kwargs["input_ids_buf"],
            kwargs["position_ids_buf"],
            None,
            None,
            {},
        )

    runner.sequence_buffer = buf
    runner.requests = requests
    runner.kv_cache_groups = []
    runner.metadata = SimpleNamespace(data_parallel_size=1, num_pages=32, version="v3", page_size=4)
    runner.max_num_reqs = max_num_reqs
    runner.num_reqs_max_model_len = num_reqs_window
    runner.max_model_len = 32
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=64))
    runner.drafter = None
    runner.num_draft_tokens = 0
    runner.draft_schedule = None
    runner.spec = NullSpeculation()
    runner.async_scheduling = True
    runner._pre_async_results = pre_results
    runner._allow_sparse_window_packing = False
    runner.slot_pool = RecurrentSlotPool(metadata=runner.metadata, max_num_reqs=max_num_reqs, slot_indexed_state=False)
    runner.spec_decode_recurrent_candidates = False
    runner.spec_decode_recurrent_replay = False
    runner.spec_decode_debug_max_traces = 0
    runner.enable_sampler_metrics = False
    runner.verbose = False
    runner.log_it = lambda *a, **k: None
    runner.perf = mr_mod.RunnerPerfTracker(history_maxlen=64, alpha=0.2)
    runner._spec_recorded_this_step = False
    runner._spec_carry = None
    runner.num_tokens_paddings = [2, 4]
    runner.active_num_seq_buckets = [2]
    runner.input_ids_buf = jnp.zeros((4,), dtype=jnp.int32)
    runner.position_ids_buf = jnp.zeros((4,), dtype=jnp.int32)
    runner._scheduled_full_cpu = np.zeros((num_reqs_window,), dtype=np.int32)
    runner._active_mask_full_cpu = np.zeros((num_reqs_window,), dtype=np.bool_)
    runner._req_num_tokens_cpu = np.zeros((num_reqs_window,), dtype=np.int32)
    runner._window_row_indices_cpu = np.zeros((num_reqs_window,), dtype=np.int32)
    runner._window_recurrent_slot_indices_cpu = np.zeros((num_reqs_window,), dtype=np.int32)
    runner._window_temperature_cpu = np.zeros((num_reqs_window,), dtype=np.float32)
    runner._window_top_p_cpu = np.ones((num_reqs_window,), dtype=np.float32)
    runner._window_top_k_cpu = np.zeros((num_reqs_window,), dtype=np.int32)
    runner._window_min_p_cpu = np.zeros((num_reqs_window,), dtype=np.float32)
    runner._window_frequency_penalties_cpu = np.zeros((num_reqs_window,), dtype=np.float32)
    runner._window_presence_penalties_cpu = np.zeros((num_reqs_window,), dtype=np.float32)
    runner._window_repetition_penalties_cpu = np.ones((num_reqs_window,), dtype=np.float32)
    runner._handoff_scalar_cache = {}
    runner._handoff_positions_cache = {}
    runner._handoff_zero_scalar = None
    runner._handoff_gather_programs = {}
    runner.window_planner = SimpleNamespace(
        get_token_bucket=lambda target, active_buckets, request_bucket=None: max(2, int(target)),
    )
    runner.executor_manager = SimpleNamespace(
        max_num_reqs=num_reqs_window,
        cache_group_configs=(),
        _empty_sharding=None,
        execute=_fake_execute,
        invalidate_sampler_penalty_state=lambda *a, **k: None,
    )
    runner.state_sync = SimpleNamespace(
        update_states=lambda so: False,
        reorder_decode_first=lambda so: None,
        reorder_decode_first_per_shard=lambda so, dp: None,
        modify_prev_results=lambda pr: None,
    )
    runner._plain_recurrent_row_sync_enabled = lambda: False
    runner._apply_recurrent_snapshot_restores = lambda so: None
    runner._apply_recurrent_snapshot_saves = lambda so: None
    runner._get_current_bucket = lambda n: 2
    runner.cache_group_inputs_for_execute = lambda **k: None
    runner._model_uses_vlm_inputs = lambda: False

    def _counted_repair(pr=None):
        nonlocal repair_calls
        repair_calls += 1

    runner._modify_prev_results = _counted_repair
    monkeypatch.setattr(mr_mod, "model_has_side_inputs", lambda m: False)

    sched = SchedulerOutput(
        [],
        CachedRequestData(["r0", "r1", "r2"], [False, False, False], [[], [], []], [([],), ([],), ([],)], [2, 2, 2]),
        {"r0": 1, "r1": 1, "r2": 1},
        3,
        {},
        [0],
        set(),
        async_scheduling=True,
    )
    out = runner._execute_model_impl(sched, return_async_output=True)
    assert isinstance(out, _AsyncExecutionHandle)
    assert len(handoffs_seen) == 2
    assert handoffs_seen[0] is not None, "Window 0 must receive device token handoff"
    assert handoffs_seen[1] is not None, "Window 1 must ALSO receive device token handoff (S1-C7)"
    assert repair_calls == 0
    # Verify window 1's handoff patches r2's token (103), NOT 0!
    assert int(np.asarray(handoffs_seen[1].token_ids)[0]) == 103


# =============================================================================
# S1-C9 & S1-M15: Overlapped Spec Verify Routing, pack_pooled_drafts & Plain Carry
# =============================================================================


def test_s1_c9_pack_pooled_drafts_drops_negative_rows():
    seed_tokens = jnp.asarray([11, 22], dtype=jnp.int32)
    drafts = jnp.asarray([[101, 102], [201, 202]], dtype=jnp.int32)
    output_rows = jnp.asarray([-1, 1], dtype=jnp.int32)
    packed = pack_pooled_drafts(seed_tokens, drafts, output_rows, num_rows=4, num_draft_tokens=2)
    packed_np = np.asarray(packed)
    assert packed_np.shape == (4, 6)
    # Row 3 (num_rows - 1) MUST NOT be corrupted by output_rows[0] == -1!
    assert np.all(packed_np[3] == -1)
    assert np.all(packed_np[0] == -1)
    assert np.all(packed_np[2] == -1)
    assert int(packed_np[1, 0]) == 22
    assert int(packed_np[1, 3]) == -1
    assert packed_np[1, 4:6].tolist() == [201, 202]


def test_s1_c9_record_plain_spec_out_slices_row_positions_and_maps_global_rows():
    runner = object.__new__(eSurgeRunner)
    buf = SequenceBuffer(6, 32, 32, 64, [4])
    for i, rid in enumerate(("pad0", "r_a", "pad2", "r_b")):
        buf.add_request(CachedRequestState(rid, [1, 2], SamplingParams(), None, ([i + 1],), 0, []))
    runner.sequence_buffer = buf
    runner.drafter = object()
    runner.async_scheduling = True
    runner.num_draft_tokens = 2
    runner.max_num_reqs = 6
    runner._spec_carry = None
    runner._spec_recorded_this_step = False

    # Sparse row_positions [2, 0] inside a 4-row window mapped to global rows 1 and 3.
    win = AsyncWindowResult(
        req_ids=["r_a", "r_b"],
        row_positions=[2, 0],
        sampled_token_ids=jnp.asarray([77, 99, 55, 88], dtype=jnp.int32),
        valid_mask=[True, True],
    )
    runner._record_plain_spec_out([win])
    assert runner._spec_carry is not None
    packed = np.asarray(runner._spec_carry.spec_out)
    assert packed.shape == (6, 2 * 2 + 2)
    # r_a is at global row 1, took sampled_token_ids[2] == 55
    assert int(packed[1, 0]) == 55
    # r_b is at global row 3, took sampled_token_ids[0] == 77
    assert int(packed[3, 0]) == 77
    # Unrelated rows stay -1
    assert np.all(packed[0] == -1)
    assert np.all(packed[2] == -1)
    assert np.all(packed[4] == -1)
    assert np.all(packed[5] == -1)


def test_s1_c9_s1_m15_lone_overlapped_async_verify_uses_spec_step_and_emits_device_tokens(monkeypatch):
    """A lone overlapped async verify row with `-1` host draft placeholders must route through begin_draft_in_verify."""
    spec_step_calls = []

    class _StubDrafter:
        def spec_step(self, **kwargs):
            spec_step_calls.append(kwargs)
            # Pool size is 4 (max_num_reqs); r0 is assigned a pool row — populate all pool rows.
            pool_size = int(kwargs["seed_pos_base"].shape[0])
            arg_mat = jnp.broadcast_to(jnp.asarray([[50, 51, 52]], dtype=jnp.int32), (pool_size, 3))
            accepted = jnp.full((pool_size,), 2, dtype=jnp.int32)
            seed_positions = jnp.full((pool_size,), 10, dtype=jnp.int32)
            spec_out = jnp.full((4, 6), -1, dtype=jnp.int32)
            spec_out = spec_out.at[1, 0].set(50).at[1, 1].set(51).at[1, 2].set(52).at[1, 3].set(2)
            spec_out = spec_out.at[1, 4].set(60).at[1, 5].set(61)
            return arg_mat, accepted, seed_positions, spec_out

        def reset(self, **kwargs):
            pass

    strategy, runner, buf, exec_mgr = _make_spec_strategy(drafter=_StubDrafter(), num_draft_tokens=2, max_num_reqs=4)
    # Put r0 at global row 1 while its window-local row_pos is 0!
    buf.add_request(CachedRequestState("dummy", [1, 2], SamplingParams(), None, ([1],), 0, []))
    req = CachedRequestState("r0", [1, 2, 3, 4], SamplingParams(temperature=0.0), None, ([2],), 6, [7, 8])
    buf.add_request(req)
    runner.requests["r0"] = req
    # Window-local row 0 had spec_delta = 1, global row 1 had spec_delta = 99 (trap for S1-C11!).
    exec_mgr.last_spec_window_delta = jnp.asarray([1, 99, 0, 0], dtype=jnp.int32)
    runner.can_batch_draft = lambda rs: True
    strategy.can_batch_draft = lambda rs: True
    strategy.can_draft_in_verify = lambda: True

    meta = SimpleNamespace(
        request_id="r0",
        row_pos=0,
        req_idx=1,
        start_pos=6,
        real_count=1,
        scheduled_draft_tokens=[-1, -1],
        buffer_draft_tokens=[-1, -1],
        target_local_indices=[0, 1],
        bonus_local_index=2,
    )
    gathered_hidden = jnp.zeros((3, 8), dtype=jnp.float32)
    handle = strategy.begin_draft_in_verify([(meta, 0, 3)], gathered_hidden=gathered_hidden, k_pad=2)
    assert handle is not None
    assert len(spec_step_calls) == 1
    results = strategy.finish_draft_in_verify(handle)
    assert "r0" in results
    res = results["r0"]
    # Device-resolved emission + drafts + known_len from seed_positions[0] (10) - spec_delta[row_pos=0] (1) + 2?
    # Wait: seed_pos = 10 -> known_len = seed_pos + 2 = 12 (seed_position is index of last hidden row = known_len - 2).
    assert res["emitted"] == [50, 51, 52]
    assert res["accepted"] == 2
    assert res["drafts"] == [60, 61]
    assert res["known_len"] == 12
    # Confirm spec_delta passed to drafter came from window-local row_pos=0 (value 1), NOT req_idx=1 (value 99)!
    collected_row = int(np.nonzero(np.asarray(spec_step_calls[0]["collected_mask"]))[0][0])
    passed_delta = np.asarray(spec_step_calls[0]["spec_delta"])
    assert int(passed_delta[collected_row]) == 1
    assert int(np.asarray(spec_step_calls[0]["seed_pos_base"])[collected_row]) == 6


# =============================================================================
# S1-C10: Per-Slot Recurrent Rollback Preserves Sibling Slots
# =============================================================================


def test_s1_c10_restore_recurrent_state_for_slots_preserves_siblings():
    conv0 = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    rec0 = jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32)
    pos0 = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    rec_view = RecurrentCacheView(conv_state=conv0, recurrent_state=rec0, positions=pos0, seqlen_offset=None, metadata=None)

    q0 = jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    k0 = jnp.asarray([[4.0], [5.0], [6.0]], dtype=jnp.float32)
    v0 = jnp.asarray([[7.0], [8.0], [9.0]], dtype=jnp.float32)
    kr0 = jnp.asarray([[100.0], [200.0], [300.0]], dtype=jnp.float32)
    kda_view = KDACacheView(
        q_conv_state=q0, k_conv_state=k0, v_conv_state=v0, recurrent_state=kr0, positions=pos0, metadata=None
    )
    pre_cache = HybridCache(views=[rec_view, kda_view])
    snap = snapshot_recurrent_state(pre_cache)
    assert snap is not None

    # Simulate forward pass mutating ALL 3 slots (0, 1, 2)
    live_rec_view = rec_view.replace(
        conv_state=conv0 + 100.0,
        recurrent_state=rec0 + 100.0,
        positions=pos0 + 10,
    )
    live_kda_view = kda_view.replace(
        q_conv_state=q0 + 50.0,
        k_conv_state=k0 + 50.0,
        v_conv_state=v0 + 50.0,
        recurrent_state=kr0 + 50.0,
        positions=pos0 + 10,
    )
    live_cache = HybridCache(views=[live_rec_view, live_kda_view])

    # Roll back ONLY slot 1; slots 0 and 2 MUST retain their live post-forward values!
    restored = _restore_recurrent_state_for_slots(live_cache, snap, [1])
    rv = restored.views[0]
    kv = restored.views[1]

    np.testing.assert_array_equal(np.asarray(rv.conv_state[1]), [3.0, 4.0])
    np.testing.assert_array_equal(np.asarray(rv.conv_state[0]), [101.0, 102.0])
    np.testing.assert_array_equal(np.asarray(rv.conv_state[2]), [105.0, 106.0])
    np.testing.assert_array_equal(np.asarray(rv.positions), [11, 2, 13])

    np.testing.assert_array_equal(np.asarray(kv.q_conv_state[1]), [2.0])
    np.testing.assert_array_equal(np.asarray(kv.q_conv_state[0]), [51.0])
    np.testing.assert_array_equal(np.asarray(kv.recurrent_state[1]), [200.0])
    np.testing.assert_array_equal(np.asarray(kv.recurrent_state[2]), [350.0])


# =============================================================================
# S1-H8: Lifecycle Cleanup & Weight Hot-Swap Invalidation
# =============================================================================


def test_s1_h8_dspark_update_and_release_target_model():
    drafter = object.__new__(DSparkDrafter)
    t1 = SimpleNamespace(tag="target_v1")
    t2 = SimpleNamespace(tag="target_v2")
    drafter.target_model = t1
    drafter._programs = ("cached_program",)
    drafter._last_window = {0: ("hidden", 0, 0, 2)}
    drafter._rings = (jnp.zeros((2, 4)),)
    drafter._capacity = 2
    drafter._owner = {0: "r0"}
    drafter._through = {0: 5}
    drafter._broken = {0}

    drafter.update_target_model(t2)
    assert drafter.target_model is t2
    assert drafter._programs is None
    assert drafter._last_window == {}

    drafter.release_target_model()
    assert drafter.target_model is None
    assert drafter._programs is None
    assert drafter._rings is None
    assert drafter._capacity == 0
    assert drafter._owner == {}
    assert drafter._through == {}
    assert drafter._broken == set()


def test_s1_h8_runner_lifecycle_hooks_clear_all_runner_and_spec_state():
    calls: list[str] = []
    runner = object.__new__(eSurgeRunner)
    runner.model = SimpleNamespace(tag="old_model")
    runner.metadata = SimpleNamespace(data_parallel_size=1, num_pages=16)
    runner.max_num_reqs = 4
    runner.requests = {"r0": object()}
    runner.sequence_buffer = SimpleNamespace(clear=lambda: calls.append("buf.clear"))
    runner.slot_pool = SimpleNamespace(reset=lambda: calls.append("slot_pool.reset"))
    runner._spec_carry = "stale_carry"
    runner._spec_recorded_this_step = True
    runner._pre_async_results = "stale_pre_async"
    runner._pending_spec_recurrent_commit_by_req = {"r0": 2}
    runner._spec_recurrent_commit_cpu = np.ones((2, 4), dtype=np.int32)
    runner._handoff_scalar_cache = {1: "x"}
    runner._handoff_positions_cache = {1: "y"}
    runner._handoff_zero_scalar = "z"
    runner.vlm = SimpleNamespace(
        clear_buffers=lambda: calls.append("vlm.clear_buffers"),
        reset_vision_cache=lambda: calls.append("vlm.reset_vision_cache"),
        reset_cpu_buffers=lambda: calls.append("vlm.reset_cpu_buffers"),
    )
    runner.spec = SimpleNamespace(
        update_target_model=lambda m: calls.append(f"spec.update:{m.tag}"),
        release_target_model=lambda: calls.append("spec.release"),
        on_reset=lambda: calls.append("spec.on_reset"),
    )
    runner._setup_variables = lambda: calls.append("runner._setup_variables")
    runner.executor_manager = SimpleNamespace(
        update_graphs=lambda **k: calls.append(f"exec.update_graphs:{k['model'].tag}"),
        release_model_state=lambda **k: calls.append("exec.release_model_state"),
        destroy_kv_cache=lambda: calls.append("exec.destroy_kv_cache"),
        drop_recurrent_snapshots=lambda: calls.append("exec.drop_recurrent_snapshots"),
        reset_recurrent_bases=lambda: calls.append("exec.reset_recurrent_bases"),
        clear_kv_cache=lambda: calls.append("exec.clear_kv_cache"),
        invalidate_sampler_penalty_state=lambda *a, **k: None,
    )

    new_model = SimpleNamespace(tag="new_model", graphdef="gd_v2")
    new_model.esurge_compatible_model = new_model
    runner.requests.clear()
    runner.update_model_weights(new_model)
    assert runner.model is new_model
    assert "exec.update_graphs:new_model" in calls
    assert "spec.update:new_model" in calls
    assert "vlm.reset_vision_cache" in calls
    assert runner._spec_carry is None
    assert runner._pre_async_results is None

    runner._spec_carry = "carry2"
    runner._pre_async_results = "pre2"
    runner._pending_spec_recurrent_commit_by_req["r1"] = 1
    runner.reset_state()
    assert "spec.on_reset" in calls
    assert "slot_pool.reset" in calls
    assert runner._spec_carry is None
    assert runner._pre_async_results is None
    assert runner._pending_spec_recurrent_commit_by_req == {}

    runner._spec_carry = "carry3"
    runner.destroy_kv_cache()
    assert "exec.destroy_kv_cache" in calls
    assert runner._spec_carry is None

    runner._spec_carry = "carry4"
    runner.release_model_state()
    assert "exec.release_model_state" in calls
    assert "spec.release" in calls
    assert runner._spec_carry is None


# =============================================================================
# S1-H11: DP-Local SequenceBuffer Compaction
# =============================================================================


def test_s1_h11_dp_local_compaction_on_one_and_zero_remaining_reqs():
    sync, buf, _, _, _, _, _ = _make_sync(dp_size=2, max_num_reqs=4, num_pages=32)
    admit = SchedulerOutput(
        [
            NewRequestData("s0-a", [1, 2], SamplingParams(), ([1],), 0, dp_rank=0),
            NewRequestData("s0-b", [3, 4], SamplingParams(), ([2],), 0, dp_rank=0),
            NewRequestData("s1-a", [5, 6], SamplingParams(), ([17],), 0, dp_rank=1),
            NewRequestData("s1-b", [7, 8], SamplingParams(), ([18],), 0, dp_rank=1),
        ],
        CachedRequestData.make_empty(),
        {"s0-a": 2, "s0-b": 2, "s1-a": 2, "s1-b": 2},
        8,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    assert buf.req_ids == ["s0-a", "s0-b", "s1-a", "s1-b"]

    finish3 = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"s0-a", "s0-b", "s1-a"})
    assert sync.update_states(finish3) is True
    assert buf.num_reqs == 1
    assert buf.req_id_to_index == {"s1-b": 2}
    assert buf.req_ids == [None, None, "s1-b"]

    finish_last = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"s1-b"})
    assert sync.update_states(finish_last) is True
    assert buf.num_reqs == 0
    assert buf.num_slots == 0
    assert buf.req_ids == []
    assert buf.req_output_token_ids == []


def test_s1_h11_dp_local_shard0_leading_hole_compacted_when_num_reqs_is_one():
    sync, buf, _, _, _, _, _ = _make_sync(dp_size=2, max_num_reqs=4, num_pages=32)
    admit = SchedulerOutput(
        [
            NewRequestData("s0-a", [1, 2], SamplingParams(), ([1],), 0, dp_rank=0),
            NewRequestData("s0-b", [3, 4], SamplingParams(), ([2],), 0, dp_rank=0),
        ],
        CachedRequestData.make_empty(),
        {"s0-a": 2, "s0-b": 2},
        4,
        {},
        [0],
        set(),
    )
    sync.update_states(admit)
    sync.update_states(SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"s0-a"}))
    assert buf.req_ids == ["s0-b"]
    assert buf.req_id_to_index == {"s0-b": 0}
    assert buf.num_slots == 1


# =============================================================================
# S1-H12: Mixed Cached-Only & Raw Modalities in VlmPrefillHelper
# =============================================================================


def test_s1_h12_mixed_cached_image_raw_video_with_no_vision_cache(monkeypatch):
    model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    img = _img_feat(3.0, offset=1)
    warm = CachedRequestState(
        "warm",
        [7, 90, 8],
        SamplingParams(),
        None,
        ([1],),
        0,
        [],
        pixel_values=img.pixel_values,
        image_grid_thw=img.grid_thw,
        mm_features=[img],
    )
    h.precompute_prefill(warm)
    assert img.pixel_values is None and img.has_cached_embeddings

    h.vision_cache = None
    vid = _vid_feat(6.0, offset=3)
    mixed = CachedRequestState(
        "mixed",
        [7, 90, 8, 91],
        SamplingParams(),
        None,
        ([2],),
        0,
        [],
        pixel_values_videos=vid.pixel_values,
        video_grid_thw=vid.grid_thw,
        mm_features=[img, vid],
    )
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], [7.0, 13.0, 8.0, 106.0])
    np.testing.assert_array_equal(mixed.prefill_deepstack_visual_embeds[0][:, 0], [23.0, 206.0])
    assert mixed.pixel_values_videos is not None


def test_s1_h12_mixed_cached_image_raw_video_model_lacks_video_embeds(monkeypatch):
    full_model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, full_model, cache=True)
    img = _img_feat(4.0, offset=1)
    warm = CachedRequestState(
        "warm",
        [7, 90, 8],
        SamplingParams(),
        None,
        ([1],),
        0,
        [],
        pixel_values=img.pixel_values,
        image_grid_thw=img.grid_thw,
        mm_features=[img],
    )
    h.precompute_prefill(warm)

    img_only_model = _MropeDeepStackModel(support_image_embeds=True, support_video_embeds=False)
    h.model_getter = lambda: img_only_model
    img._esurge_vision_cache_identity = h._vision_cache_identity()

    vid = _vid_feat(9.0, offset=3)
    mixed = CachedRequestState(
        "mixed",
        [7, 90, 8, 91],
        SamplingParams(),
        None,
        ([2],),
        0,
        [],
        pixel_values_videos=vid.pixel_values,
        video_grid_thw=vid.grid_thw,
        mm_features=[img, vid],
    )
    h.precompute_prefill(mixed)
    np.testing.assert_array_equal(mixed.prefill_inputs_embeds[:, 0], [7.0, 14.0, 8.0, 109.0])


def test_s1_h12_uncovered_cached_only_modality_raises_even_when_other_modality_is_raw(monkeypatch):
    model = _MropeDeepStackModel(support_image_embeds=False, support_video_embeds=False)
    h = _vlm_helper(monkeypatch, model, cache=False)
    img = MultiModalFeature(mm_hash="abc", modality="image", pixel_values=None, grid_thw=None)
    vid = _vid_feat(5.0, offset=3)
    bad = CachedRequestState(
        "bad",
        [7, 90, 8, 91],
        SamplingParams(),
        None,
        ([1],),
        0,
        [],
        pixel_values_videos=vid.pixel_values,
        video_grid_thw=vid.grid_thw,
        mm_features=[img, vid],
    )
    with pytest.raises(RuntimeError, match="needs raw image media"):
        h.precompute_prefill(bad)


# =============================================================================
# S1-H14 & S1-L2: Sampling Constraints in Speculative Eligibility, Verify & Scoring
# =============================================================================


def test_s1_h14_is_spec_request_excludes_constrained_requests():
    strategy, _runner, _buf, _mgr = _make_spec_strategy(drafter=object(), num_draft_tokens=2)

    plain = CachedRequestState("r0", [1, 2], SamplingParams(temperature=0.0), None, ([1],), 0, [])
    with_bias = CachedRequestState("r1", [1, 2], SamplingParams(temperature=0.0, logit_bias={5: 3.0}), None, ([1],), 0, [])
    with_allowed = CachedRequestState(
        "r2", [1, 2], SamplingParams(temperature=0.0, allowed_token_ids=[1, 2]), None, ([1],), 0, []
    )
    with_empty_allowed = CachedRequestState(
        "r3", [1, 2], SamplingParams(temperature=0.0, allowed_token_ids=[]), None, ([1],), 0, []
    )
    bad_params = SamplingParams(temperature=0.0, bad_words=["forbidden"])
    bad_params._bad_words_token_ids = [[3]]
    with_bad = CachedRequestState("r4", [1, 2], bad_params, None, ([1],), 0, [])

    assert strategy.is_spec_request(plain) is True
    assert strategy.is_spec_request(with_bias) is False
    assert strategy.is_spec_request(with_allowed) is False
    assert strategy.is_spec_request(with_empty_allowed) is False
    assert strategy.is_spec_request(with_bad) is False


def test_s1_l2_verify_resolve_rows_broadcasts_2d_constraints_over_3d_logits():
    p, k, v = 2, 2, 6
    # Raw logits prefer token 1 everywhere, but logit_bias strongly promotes token 4 and bad_words bans token 1!
    logits = jnp.zeros((p, k, v), dtype=jnp.float32).at[:, :, 1].set(5.0)
    drafts = jnp.asarray([[4, 4], [1, 1]], dtype=jnp.int32)
    sampled = jnp.asarray([4, 2], dtype=jnp.int32)
    rows = np.zeros((p, NUM_ROW_FIELDS), dtype=np.int32)
    rows[:, ROW_SEQ] = [0, 1]
    rows[:, ROW_START] = [2, 2]
    rows[:, ROW_REAL] = [1, 1]
    rows[:, ROW_DRAFTS] = [2, 2]
    rows[:, ROW_LIVE] = [1, 1]
    rows[:, ROW_GREEDY] = [1, 1]
    params = np.zeros((p, NUM_PARAM_FIELDS), dtype=np.float32)
    params[:, PARAM_TEMPERATURE] = 0.0
    params[:, PARAM_TOP_P] = 1.0
    params[:, PARAM_TOP_K] = 0.0
    params[:, PARAM_MIN_P] = 0.0

    # 2D [P, V] constraints against 3D [P, K, V] logits (previously raised shape broadcast error!)
    logit_bias = jnp.zeros((p, v), dtype=jnp.float32).at[0, 4].set(20.0)
    allowed_mask = jnp.ones((p, v), dtype=jnp.bool_).at[1, :].set(False).at[1, 3].set(True)
    bad_words = jnp.zeros((p, v), dtype=jnp.bool_).at[1, 1].set(True)

    res = resolve_rows(
        logits,
        drafts,
        sampled,
        jnp.asarray(rows),
        jnp.asarray(params),
        jax.random.PRNGKey(0),
        logit_bias=logit_bias,
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids=bad_words,
    )
    # Row 0: bias boosts token 4 -> both drafts [4, 4] accepted!
    assert int(res.accepted[0]) == 2
    assert np.asarray(res.emitted[0]).tolist() == [4, 4, 4]
    # Row 1: token 1 banned & only token 3 allowed -> draft 1 rejected, corrected to 3!
    assert int(res.accepted[1]) == 0
    assert int(res.emitted[1, 0]) == 3


# =============================================================================
# S1-H15: Preemption Cleanup & Self-Healing of Drafter Pool Rows
# =============================================================================


def test_s1_h15_preemption_releases_drafter_rows_and_per_request_state():
    released_in_drafter: list[str] = []

    class _TrackingDrafter:
        def release_request(self, rid: str) -> None:
            released_in_drafter.append(rid)

        def reset(self, **kwargs):
            pass

    strategy, _runner, buf, _mgr = _make_spec_strategy(drafter=_TrackingDrafter(), num_draft_tokens=2, max_num_reqs=4)
    buf.add_request(CachedRequestState("r0", [1, 2], SamplingParams(), None, ([1],), 0, []))
    buf.add_request(CachedRequestState("r1", [3, 4], SamplingParams(), None, ([2],), 0, []))

    row0 = strategy._stable_drafter_row("r0", 0)
    row1 = strategy._stable_drafter_row("r1", 1)
    assert row0 != row1

    strategy._mtp_persist_req_by_row[row0] = "r0"
    strategy._mtp_persist_base_pos_by_row[row0] = 0
    strategy._mtp_persist_last_seed_by_row[row0] = 5
    strategy._mtp_persist_written_by_row[row0] = 2
    strategy.draft_full_log_probs_by_req["r0"] = jnp.zeros((2, 8))
    strategy._backoff_by_req["r0"] = 3
    strategy._penalty_counts_stale.add("r0")
    strategy._whole_block_window["r0"] = (0, 0, 2)

    buf.remove_request("r0")
    preempt = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"r0"})
    strategy._release_finished_drafter_rows(preempt)

    assert "r0" in released_in_drafter
    assert "r0" not in strategy._mtp_row_by_req
    assert row0 in strategy._mtp_row_free
    assert row0 not in strategy._mtp_persist_req_by_row
    assert "r0" not in strategy.draft_full_log_probs_by_req
    assert "r0" not in strategy._backoff_by_req
    assert "r0" not in strategy._penalty_counts_stale
    assert "r0" not in strategy._whole_block_window

    # Now simulate r1 vanishing from sequence_buffer without a release event -> _stable_drafter_row self-heals!
    buf.remove_request("r1")
    for idx in range(4):
        rid = f"new_{idx}"
        buf.add_request(CachedRequestState(rid, [1, 2], SamplingParams(), None, ([idx + 1],), 0, []))
        assigned = strategy._stable_drafter_row(rid, idx)
        assert 0 <= assigned < 4
    assert "r1" not in strategy._mtp_row_by_req


def test_s1_h15_dspark_release_request_drops_stale_ring_owner():
    drafter = object.__new__(DSparkDrafter)
    drafter._owner = {0: "r0", 1: "r1"}
    drafter._through = {0: 12, 1: 7}
    drafter._last_window = {0: ("h", 0, 10, 3), 1: ("h", 3, 6, 2)}
    drafter._broken = {0}

    drafter.release_request("r0")
    assert 0 not in drafter._owner
    assert 0 not in drafter._through
    assert 0 not in drafter._last_window
    assert 0 not in drafter._broken
    assert drafter._owner[1] == "r1"
    assert drafter._through[1] == 7


# =============================================================================
# S1-H16: Dynamic-K Throttling to 0 Checkpoints Persisted MTP + Widens spec_out
# =============================================================================


def test_s1_h16_dynamic_k_zero_checkpoints_mtp_and_widens_spec_out():
    strategy, runner, buf, _mgr = _make_spec_strategy(drafter=object(), num_draft_tokens=3, max_num_reqs=4)
    strategy._mtp_persist_enabled = lambda: True
    req = CachedRequestState("r0", [1, 2, 3, 4], SamplingParams(temperature=0.0), None, ([1],), 0, [])
    buf.add_request(req)
    runner.requests["r0"] = req
    row = strategy._stable_drafter_row("r0", 0)
    strategy._mtp_persist_req_by_row[row] = "r0"
    strategy._mtp_persist_base_pos_by_row[row] = 0
    strategy._mtp_persist_last_seed_by_row[row] = 10
    strategy._mtp_persist_written_by_row[row] = 3

    # Throttle live_num_draft_tokens to 0 (e.g. high concurrency or verify budget == 0).
    strategy.live_num_draft_tokens = 0
    out = strategy.draft_next(
        req_id="r0",
        seed_token=42,
        seed_position=11,
        seed_hidden=jnp.zeros((8,), dtype=jnp.float32),
        req_state=req,
        row_pos=0,
    )
    assert out == []
    # Checkpointing advances last_seed to 11 and clears written to 0 so the next draft drops rejected slots!
    assert strategy._mtp_persist_last_seed_by_row[row] == 11
    assert strategy._mtp_persist_written_by_row[row] == 0

    # Widening: k_live=1 spec_out (width 2*1+2=4) widened to K_max=3 (width 2*3+2=8).
    small = jnp.asarray(
        [
            [10, 11, 1, 20],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=jnp.int32,
    )
    wide = np.asarray(widen_spec_out(small, num_draft_tokens=3))
    assert wide.shape == (4, 8)
    assert wide[0].tolist() == [10, 11, -1, -1, 1, 20, -1, -1]
    assert np.all(wide[1:] == -1)

    # pack_rows aligns emitted/drafts when emitted_cols < k + 1 and draft_cols < k
    rows = np.zeros((1, NUM_ROW_FIELDS), dtype=np.int32)
    rows[0, ROW_SEQ] = 2
    packed = np.asarray(
        pack_rows(
            jnp.asarray(rows),
            jnp.asarray([[5, 6]], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
            jnp.asarray([[9]], dtype=jnp.int32),
            num_rows=4,
            num_draft_tokens=3,
        )
    )
    assert packed.shape == (4, 8)
    assert packed[2].tolist() == [5, 6, -1, -1, 1, 9, -1, -1]

    # record_spec_out merges narrowed k_live=1 into existing K_max=3 carry without shape error
    r = object.__new__(eSurgeRunner)
    r.num_draft_tokens = 3
    r._spec_carry = None
    r._spec_recorded_this_step = False
    r.sequence_buffer = buf
    r.record_spec_out(wide)
    other = jnp.full((4, 4), -1, dtype=jnp.int32).at[1].set(jnp.asarray([30, -1, 0, 40], dtype=jnp.int32))
    r.record_spec_out(other)
    merged = np.asarray(r._spec_carry.spec_out)
    assert merged.shape == (4, 8)
    assert merged[0].tolist() == [10, 11, -1, -1, 1, 20, -1, -1]
    assert merged[1].tolist() == [30, -1, -1, -1, 0, 40, -1, -1]


# =============================================================================
# S1-H17: Joint-Vocab Sizing (`total_vocab_size` > `vocab_size`)
# =============================================================================


def test_s1_h17_sequence_buffer_accepts_joint_total_vocab_size():
    buf = SequenceBuffer(max_num_reqs=4, max_model_len=16, max_num_batched_tokens=16, vocab_size=500, page_sizes=[4])
    assert buf.vocab_size == 500
    req = CachedRequestState(
        "r0",
        [1, 2],
        SamplingParams(top_k=450, allowed_token_ids=[0, 499]),
        None,
        ([1],),
        0,
        [],
    )
    buf.add_request(req)
    assert "r0" in buf.top_k_reqs
    assert int(buf.top_k[0]) == 450
    assert buf.allowed_token_ids_mask is not None
    assert buf.allowed_token_ids_mask.shape == (4, 500)
    assert bool(buf.allowed_token_ids_mask[0, 499]) is False
    assert bool(buf.allowed_token_ids_mask[0, 10]) is True


def test_s1_h17_runner_resolved_vocab_size_and_sampling_processors():
    runner = object.__new__(eSurgeRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=320, total_vocab_size=512))
    assert runner._resolved_vocab_size() == 512

    sp = SamplingParams(logit_bias={499: 2.5}, allowed_token_ids=[10, 500], bad_words=["x"])
    sp._bad_words_token_ids = [[505]]
    req = CachedRequestState(
        "r0",
        [1, 2],
        sp,
        None,
        ([1],),
        0,
        [],
    )
    runner.requests = {"r0": req}
    constraints = runner._sampling_processors_for_window(
        ["r0"],
        np.asarray([[1, 2, 0, 0]], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )
    assert constraints is not None
    assert 499 in constraints.bias_tokens.tolist()
    assert 500 in constraints.allowed_tokens.tolist()
    assert 505 in constraints.banned_tokens.tolist()


# =============================================================================
# S1-H18: Logprob Scoring Constraints + Async Handle Post-Verify Override
# =============================================================================


def test_s1_h18_score_sampled_tokens_applies_sparse_constraints_and_async_handle_overrides_logprobs():
    # Raw logits: token 0 has logit 10, token 2 has logit 0.
    # Logit bias adds +20 to token 2 -> token 2 becomes the dominant probability!
    logits = jnp.asarray([[10.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    constraints = SamplingConstraints(
        bias_rows=np.asarray([0], dtype=np.int32),
        bias_tokens=np.asarray([2], dtype=np.int32),
        bias_values=np.asarray([20.0], dtype=np.float32),
        restricted_rows=np.asarray([], dtype=np.int32),
        allowed_rows=np.asarray([], dtype=np.int32),
        allowed_tokens=np.asarray([], dtype=np.int32),
        banned_rows=np.asarray([], dtype=np.int32),
        banned_tokens=np.asarray([], dtype=np.int32),
    )
    compacted = compact_sampling_constraints(np.asarray([0], dtype=np.int64), constraints)
    assert compacted is not None

    _ids_c, scores_c, ranks_c = score_sampled_tokens(
        logits,
        jnp.asarray([2], dtype=jnp.int32),
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        num_logprobs=1,
        sparse_constraints=compacted,
    )
    # With +20 bias on token 2, logit(2)=20 vs logit(0)=10 -> rank 0, logprob ~ 0.0!
    assert int(ranks_c[0]) == 0
    assert float(scores_c[0, 0]) > -0.01

    # Also test runner._score_output_row passing constraints through
    runner = object.__new__(eSurgeRunner)
    _ids_r, scores_r, ranks_r = runner._score_output_row(
        logits=logits,
        sampled_tokens=jnp.asarray([2], dtype=jnp.int32),
        history=np.zeros((4,), dtype=np.int32),
        seq_len=0,
        sampling_params=SamplingParams(temperature=1.0, logprobs=1),
        sampling_constraints=constraints,
        row_pos=0,
    )
    assert int(np.asarray(ranks_r)[0]) == 0
    assert float(np.asarray(scores_r)[0, 0]) > -0.01

    # _AsyncExecutionHandle must let host_token_logprobs supersede stale pre-verify window.token_logprobs!
    win = AsyncWindowResult(
        req_ids=["r0", "r1"],
        row_positions=[0, 1],
        sampled_token_ids=jnp.asarray([5, 6], dtype=jnp.int32),
        valid_mask=[True, True],
        token_logprobs=jnp.asarray([-9.5, -1.2], dtype=jnp.float32),
    )
    handle = _AsyncExecutionHandle(
        model_runner_output=ModelRunnerOutput(
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            req_id_to_row_index={"r0": 0, "r1": 1},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            finished_sending=None,
            finished_recving=None,
        ),
        windows=[win],
        host_rows={0: [10, 11]},
        host_token_logprobs={"r0": -0.25},
    )
    out = handle.get_output()
    assert out.sampled_token_ids == [[10, 11], [6]]
    assert out.token_logprobs is not None
    assert pytest.approx(out.token_logprobs["r0"], rel=1e-5) == -0.25
    assert pytest.approx(out.token_logprobs["r1"], rel=1e-5) == -1.2


# =============================================================================
# S1-M11: Vision Patch Buckets Multiply by Spatial Merge Unit
# =============================================================================


def test_s1_m11_vision_patch_buckets_multiply_by_merge_unit():
    model = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    h = VlmPrefillHelper(
        model_getter=lambda: model,
        metadata=SimpleNamespace(data_parallel_size=1),
        compile_vision_encoder=True,
        vision_patch_buckets=None,
        max_num_batched_tokens=64,
        max_num_tokens=64,
    )
    assert h.vision_patch_buckets_for_compile(max_num_batched_tokens=64) == [16, 32, 64, 128, 256]
    assert h.vision_patch_buckets_for_compile(max_num_batched_tokens=2048)[-1] == 8192


# =============================================================================
# S1-M13: _repair_placeholder Out-Of-Bounds Guard
# =============================================================================


def test_s1_m13_repair_placeholder_does_not_clobber_last_token_when_oob():
    buf = SequenceBuffer(max_num_reqs=4, max_model_len=32, max_num_batched_tokens=32, vocab_size=64, page_sizes=[4])
    req = CachedRequestState("r0", [1, 2, 3], SamplingParams(), None, ([1],), 3, [42])
    buf.add_request(req)
    buf.token_ids[0, 3] = 42
    buf.token_ids[0, 4] = 99
    requests = {"r0": req}

    SequenceStateSync._repair_placeholder(buf, requests, req, placeholder_idx=4, token=77, max_model_len=32)
    assert req.output_token_ids == [42]
    assert int(buf.token_ids[0, 3]) == 42
    assert int(buf.token_ids[0, 4]) == 99

    SequenceStateSync._repair_placeholder(buf, requests, req, placeholder_idx=2, token=88, max_model_len=32)
    assert req.output_token_ids == [42]
    assert int(buf.token_ids[0, 2]) == 3

    SequenceStateSync._repair_placeholder(buf, requests, req, placeholder_idx=3, token=55, max_model_len=32)
    assert req.output_token_ids == [55]
    assert int(buf.token_ids[0, 3]) == 55


# =============================================================================
# S1-M14: Speculative Rejection (-1e10), Sampling Params (0.0), bf16, Seed & Verify Budget
# =============================================================================


def test_s1_m14_rejection_never_accepts_masked_minus_1e10_target_token():
    # Target masked token 1 to -1e10 (e.g. top-k/top-p filter); draft ALSO had -1e10 on token 1 so d_lp <= t_lp.
    # Previously t_lp - d_lp == 0 -> exp(0) == 1.0 -> ALWAYS accepted an impossible masked token!
    target_lp = jnp.asarray([[[-1e10, -1e10, 0.0, -1e10], [-1e10, -1e10, 0.0, -1e10]]], dtype=jnp.float32)
    draft_lp = jnp.asarray([[[-1e10, -1e10, 0.0, -1e10]]], dtype=jnp.float32)
    draft_tokens = jnp.asarray([[1]], dtype=jnp.int32)
    accepted, emitted = rejection_sample_from_uniforms(
        target_lp,
        draft_lp,
        draft_tokens,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([[0.0]], dtype=jnp.float32),
        jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
    )
    assert int(accepted[0]) == 0
    assert int(emitted[0, 0]) == 2


def test_s1_m14_sampling_args_preserves_zero_and_seed_determinism_and_verify_budget():
    strategy, _runner, _buf, exec_mgr = _make_spec_strategy(drafter=object(), num_draft_tokens=4)

    # sampling_args must NOT replace explicit 0.0 with 1.0!
    req_zero = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0, top_p=0.0, top_k=0, min_p=0.0))
    temp, top_p, top_k, min_p = strategy.sampling_args(req_zero)
    assert temp == 0.0
    assert top_p == 0.0
    assert top_k == 0
    assert min_p == 0.0
    req_sampled = CachedRequestState(
        "r_sampled", [1, 2], SamplingParams(temperature=0.7, top_p=0.5, top_k=0, min_p=0.0), None, ([1],), 0, []
    )
    t_s, p_s, k_s, m_s = strategy.sampling_args(req_sampled)
    assert pytest.approx(float(t_s), rel=1e-5) == 0.7
    assert pytest.approx(float(p_s), rel=1e-5) == 0.5
    assert int(k_s) == 0
    assert float(m_s) == 0.0

    # Per-request deterministic RNG key must be invariant to engine rng_key mutations!
    req_seeded = CachedRequestState("r_seed", [1, 2, 3], SamplingParams(temperature=0.8, seed=1234567890123), None, ([1],), 0, [])
    k1 = strategy.request_rng_key(req_seeded, position=7, stream=1)
    exec_mgr.rng_key = jax.random.PRNGKey(99999)
    k2 = strategy.request_rng_key(req_seeded, position=7, stream=1)
    k3 = strategy.request_rng_key(req_seeded, position=8, stream=1)
    np.testing.assert_array_equal(np.asarray(k1), np.asarray(k2))
    assert not np.array_equal(np.asarray(k1), np.asarray(k3))

    # resolve_step_draft_budget honors scheduler_output.num_draft_tokens_next
    assert strategy.resolve_step_draft_budget(1, num_draft_tokens_next=2) == 2
    assert strategy.live_num_draft_tokens == 2
    assert strategy.resolve_step_draft_budget(1, num_draft_tokens_next=0) == 0
    assert strategy.resolve_step_draft_budget(1, num_draft_tokens_next=-1) == 4

    # bf16 filter_logits and filtered_log_probs run cleanly on bfloat16 logits and return float32
    bf16_logits = jnp.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=jnp.bfloat16)
    req_top = CachedRequestState("r_bf16", [1, 2], SamplingParams(temperature=0.7, top_k=2, top_p=0.9), None, ([1],), 0, [])
    lp = strategy.filtered_log_probs(bf16_logits, req_top)
    assert lp.dtype == jnp.float32
    fl = filter_logits(bf16_logits, jnp.float32(0.7), jnp.float32(0.9), jnp.int32(2), jnp.float32(0.0))
    assert fl.dtype == jnp.float32


# =============================================================================
# S1-L1: _measure_step_times Pruned Bucket Guard
# =============================================================================


def test_s1_l1_measure_step_times_skips_infeasible_pruned_buckets():
    executed: list[tuple[int, int]] = []
    runner = object.__new__(eSurgeRunner)
    runner.drafter = None
    runner.num_draft_tokens = 0
    runner.pipeline_plan = SimpleNamespace(is_enabled=False)
    runner.active_num_seq_buckets = [4, 8]
    runner._allow_sparse_window_packing = True
    runner.max_num_reqs = 8
    runner.metadata = SimpleNamespace()
    model_exec = SimpleNamespace(
        has_compiled=lambda nt, nr: (nt, nr) in {(4, 4), (8, 8)},
    )
    runner.executor_manager = SimpleNamespace(
        _model_executor=model_exec,
        kv_pages=None,
        rng_key=jax.random.PRNGKey(0),
        get_compile_configurations=lambda kv, rng, nt, mnr, pnr, meta: (None, None, None, (nt, pnr)),
        execute_model=lambda nt, pnr, inp: (executed.append((nt, pnr)), jnp.asarray(0))[1],
        _reset_kv_pages_after_compile_warmup=lambda force=False: None,
    )

    # num_tokens=2 < min(req_buckets)=4 -> infeasible & skipped; num_tokens=6 not compiled -> skipped!
    table = runner._measure_step_times([2, 4, 6, 8])
    assert table is not None
    assert sorted(table.keys()) == [4, 8]
    assert all(nt in (4, 8) for nt, _ in executed)

    # When NO bucket is compiled, returns None instead of empty dict!
    model_exec.has_compiled = lambda nt, nr: False
    assert runner._measure_step_times([2, 4]) is None


# =============================================================================
# S1-L3 & S1-L5: StateSync & SequenceBuffer / VlmPrefill Low-Severity Fixes
# =============================================================================


def test_s1_l3_finished_has_changes_page_ids_copy_and_spec_commit_clear():
    sync, buf, requests, _, _, commit_cpu, _ = _make_sync(
        slot_indexed_state=True, spec_decode_recurrent_candidates=True
    )
    orig_pages = ([1, 2],)
    admit = SchedulerOutput(
        [NewRequestData("r0", [1, 2, 3], SamplingParams(), orig_pages, 0)],
        CachedRequestData.make_empty(),
        {"r0": 3},
        3,
        {},
        [0],
        set(),
    )
    assert sync.update_states(admit) is True

    append_step = SchedulerOutput(
        [], CachedRequestData(["r0"], [False], [[]], [([3],)], [2]), {"r0": 1}, 1, {}, [0], set()
    )
    sync.update_states(append_step)
    assert orig_pages == ([1, 2],)
    assert requests["r0"].page_ids == ([1, 2, 3],)

    sync.update_states(
        SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], set(), preempted_req_ids={"r0"})
    )
    resumed_pages = ([5, 6],)
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [True], [[]], [resumed_pages], [0]), {"r0": 3}, 3, {}, [0], set())
    )
    sync.update_states(
        SchedulerOutput([], CachedRequestData(["r0"], [False], [[]], [([7],)], [2]), {"r0": 1}, 1, {}, [0], set())
    )
    assert resumed_pages == ([5, 6],)
    assert requests["r0"].page_ids == ([5, 6, 7],)

    req_idx = buf.req_id_to_index["r0"]
    commit_cpu[:, req_idx] = 9
    finish = SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"r0"})
    assert sync.update_states(finish) is True
    assert (commit_cpu[:, req_idx] == 0).all()


def test_s1_l5_grid_hash_normalization(monkeypatch):
    model = _MropeDeepStackModel()
    h = _vlm_helper(monkeypatch, model, cache=True)
    pixels = np.ones((2, 3), dtype=np.float32)
    calls = 0

    def encode():
        nonlocal calls
        calls += 1
        return (pixels + 1.0, ())

    grids = [
        np.array([[1, 2, 2]], dtype=np.int32),
        np.array([[1, 2, 2]], dtype=np.int64),
        np.array([1, 2, 2], dtype=np.int32),
        ((1, 2, 2),),
        [[1, 2, 2]],
        [1, 2, 2],
    ]
    for g in grids:
        h._encoded_vision_features(pixels, g, modality="image", encode=encode)
    assert calls == 1
    assert h.vision_cache.hits == len(grids) - 1


def test_s1_l5_small_min_p_included_in_min_p_reqs():
    buf = SequenceBuffer(max_num_reqs=4, max_model_len=16, max_num_batched_tokens=16, vocab_size=64, page_sizes=[4])
    r_tiny = CachedRequestState("r_tiny", [1, 2], SamplingParams(temperature=0.8, min_p=1e-6), None, ([1],), 0, [])
    r_edge = CachedRequestState("r_edge", [1, 2], SamplingParams(temperature=0.8, min_p=1e-5), None, ([2],), 0, [])
    r_zero = CachedRequestState("r_zero", [1, 2], SamplingParams(temperature=0.8, min_p=0.0), None, ([3],), 0, [])
    buf.add_request(r_tiny)
    buf.add_request(r_edge)
    buf.add_request(r_zero)
    assert "r_tiny" in buf.min_p_reqs
    assert "r_edge" in buf.min_p_reqs
    assert "r_zero" not in buf.min_p_reqs
    assert not buf.no_min_p
    assert "min_p" in buf.get_active_sampling_params(buf.req_id_to_index["r_tiny"])
    assert "min_p" not in buf.get_active_sampling_params(buf.req_id_to_index["r_zero"])


# =============================================================================
# Execution Manager, Executors, Host Sync & Window Planner Regressions
# =============================================================================


def _single_device_sharding():
    return jax.sharding.SingleDeviceSharding(jax.devices()[0])


def _cache_config(*, version="v3", page_size=16, dp=1, num_pages=64, max_num_reqs=4, max_model_len=64):
    return RaggedPagesCacheConfig(
        num_hidden_layers=2,
        max_model_length=max_model_len,
        num_kv_heads=1,
        k_headdim=16,
        v_headdim=16,
        page_size=page_size,
        num_pages=num_pages,
        max_num_pages_per_req=max_model_len // page_size,
        num_slices_per_kv_cache_update_page=1,
        max_num_tokens=16,
        max_num_reqs=max_num_reqs,
        data_parallel_size=dp,
        version=version,
    )


# ---------------------------------------------------------------------------
# S1-C5: MPMD weight-update desync + HBM leak
# ---------------------------------------------------------------------------
def test_s1_c5_on_graphs_updated_bumps_weight_generation_and_clears_caches():
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor.use_aot_forward = False
    executor.graphdef = object()
    executor._weight_generation = 0
    executor._cache = OrderedDict(
        [
            ((8, 4, "model_step", "mpmd"), "fused-mpmd"),
            ((8, 4, "model_step", "aot"), "spmd-entry"),
        ]
    )
    executor._pipeline_model_step_fn_cache = OrderedDict([((4, "mpmd:1"), "fn")])
    executor._pipeline_kv_carry_map_cache = {(8, ()): {0: {1: 0}}}
    executor._backbone_fn = lambda *a: a
    executor._backbone_fn_decode = lambda *a: a

    cleared = []
    seen_keys = []

    class _Runtime:
        def clear_prepare_cache(self):
            cleared.append(True)

        def dispatch(self, fn, *args, prepare_cache_key=None, runtime_static_argnums=None):
            seen_keys.append(prepare_cache_key)
            return "ok"

    executor._pipeline_runtime = _Runtime()
    meta = SimpleNamespace(input_ids_buf=np.zeros((8,), dtype=np.int32), version="v3", num_pages=4, page_size=16)

    executor._dispatch_pipeline_backbone("gs1", "go1", {}, meta)
    executor._dispatch_pipeline_model_step(lambda *a: None, "gs1", "go1", {}, meta, padded_num_reqs=4)

    # Same graphdef hot-swap: must bump weight generation & clear prepare cache, keep fused MPMD cache
    executor.on_graphs_updated(graphdef_changed=False)
    assert len(cleared) == 1
    assert (8, 4, "model_step", "mpmd") in executor._cache
    assert len(executor._pipeline_model_step_fn_cache) == 1

    executor._dispatch_pipeline_backbone("gs2", "go2", {}, meta)
    executor._dispatch_pipeline_model_step(lambda *a: None, "gs2", "go2", {}, meta, padded_num_reqs=4)
    assert seen_keys[0] != seen_keys[2]  # backbone prepare_cache_key changed despite identical id(graphdef)
    assert seen_keys[1] != seen_keys[3]  # model_step prepare_cache_key changed despite identical id(graphdef)

    # Graphdef change: must also drop _pipeline_model_step_fn_cache, carry map, and MPMD _cache entries
    executor.on_graphs_updated(graphdef_changed=True)
    assert len(cleared) == 2
    assert not executor._pipeline_model_step_fn_cache
    assert not executor._pipeline_kv_carry_map_cache
    assert (8, 4, "model_step", "mpmd") not in executor._cache
    assert (8, 4, "model_step", "aot") in executor._cache


# ---------------------------------------------------------------------------
# S1-C6 & S1-H19: PP microbatching gates + metadata refresh
# ---------------------------------------------------------------------------
def test_s1_c6_and_s1_h19_microbatch_gates_and_metadata_refresh():
    # Direct PipelineMicrobatchExecutor check for identity-layout HybridCache with recurrent views
    rec_view = RecurrentCacheView(
        conv_state=jnp.zeros((4, 2, 4), dtype=jnp.float32),
        recurrent_state=jnp.zeros((4, 2, 4, 4), dtype=jnp.float32),
        positions=jnp.zeros((4,), dtype=jnp.int32),
        metadata=None,
        layer_index=0,
    )
    hybrid = HybridCache(views=[rec_view])
    fake_exec = SimpleNamespace(
        pipeline_plan=SimpleNamespace(is_enabled=True, stage_meshes=(object(), object())),
        pipeline_runtime=object(),
        full_hidden_state_max_tokens=0,
    )
    fake_mgr = SimpleNamespace(_model_executor=fake_exec, kv_pages=hybrid)
    mb = PipelineMicrobatchExecutor(fake_mgr)
    dummy_kwargs = dict(
        num_tokens=4,
        scheduled_full_cpu=np.ones(4, dtype=np.int32),
        active_mask_full_cpu=np.ones(4, dtype=bool),
        window_row_indices_cpu=np.arange(4, dtype=np.int32),
        input_ids_buf=None,
        position_ids_buf=None,
        padded_num_reqs=4,
        req_num_tokens_full_cpu=np.full(4, 2, dtype=np.int32),
        token_ids_cpu=np.zeros((4, 8), dtype=np.int32),
        num_computed_tokens_cpu=np.ones(4, dtype=np.int32),
        temperature_cpu=np.ones(4, dtype=np.float32),
        top_p_cpu=np.ones(4, dtype=np.float32),
        top_k_cpu=np.zeros(4, dtype=np.int32),
        min_p_cpu=np.zeros(4, dtype=np.float32),
        frequency_penalties_cpu=np.zeros(4, dtype=np.float32),
        presence_penalties_cpu=np.zeros(4, dtype=np.float32),
        repetition_penalties_cpu=np.ones(4, dtype=np.float32),
        page_table_cpu=np.ones((4, 2), dtype=np.int32),
        mrope_position_ids_cpu=None,
        prefill_embeds_cpu=None,
        prefill_embeds_mask_cpu=None,
        visual_pos_masks_cpu=None,
        deepstack_visual_embeds_cpu=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        device_token_handoff=None,
        wait_for_outputs=False,
    )
    assert mb.try_execute(**dummy_kwargs) is None

    # Full hidden state required (drafter) must also skip microbatching
    fake_mgr.kv_pages = None
    fake_exec.full_hidden_state_max_tokens = 16
    assert mb.try_execute(**dummy_kwargs) is None


# ---------------------------------------------------------------------------
# S1-C8: Verify-contract recurrent slot bases across async spec_window_handoff
# ---------------------------------------------------------------------------
def _pack_spec_out(n_req: int, k: int, rows: dict[int, tuple[list[int], int, list[int]]]) -> np.ndarray:
    out = np.full((n_req, 2 * k + 2), -1, dtype=np.int32)
    for row, (emit, accepted, drafts) in rows.items():
        out[row, : len(emit)] = emit
        out[row, k + 1] = accepted
        out[row, k + 2 : k + 2 + len(drafts)] = drafts
    return out


def test_s1_c8_recurrent_slot_bases_corrected_after_overlapped_steps_and_cleared():
    sharding = _single_device_sharding()
    k = 3
    w = 1 + k
    n_req = 4
    n_tok = 16
    cfg = _cache_config(max_num_reqs=n_req, max_model_len=64, page_size=16)

    preparer = BatchMetadataPreparer(
        metadata=cfg,
        empty_sharding=sharding,
        max_num_tokens=n_tok,
        max_num_reqs=n_req,
        max_model_len=64,
        min_input_pad=1,
        enable_spec_recurrent_commit=True,
        slot_indexed_state=True,
        recurrent_slot_group_size=w,
        input_sharding=sharding,
    )

    model_exec = ModelStepExecutor.__new__(ModelStepExecutor)
    model_exec.metadata = cfg
    model_exec._metadata_version = "v3"
    model_exec._slot_indexed_state = True
    model_exec.recurrent_slot_group_size = w

    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.recurrent_slot_group_size = w
    mgr._recurrent_base_cpu = np.zeros((2, n_req), dtype=np.int32)
    mgr._recurrent_base_by_slot = np.zeros((n_req * w,), dtype=np.int64)
    mgr._pending_recurrent_base_deltas = {}
    mgr.kv_pages = None
    mgr.max_num_reqs = n_req
    mgr._batch_preparer = preparer

    pages = np.tile(np.arange(4, dtype=np.int32), (n_req, 1))
    slots = np.array([0, 4, 8, 12], dtype=np.int32)
    zeros_i32 = np.zeros(n_req, dtype=np.int32)
    zeros_f32 = np.zeros(n_req, dtype=np.float32)
    ones_f32 = np.ones(n_req, dtype=np.float32)

    def run_prep(sched, computed, handoff):
        skip_mask = None
        fed_mask = None
        if handoff is not None:
            src_host = np.asarray(handoff.sources, dtype=np.int32)
            fed_mask = src_host[0, :n_req] >= 0
            if handoff.rewrite_recurrent_base:
                skip_mask = fed_mask
        commit = mgr._recurrent_slot_bases(sched, computed, slots, skip_host_read_mask=skip_mask)
        meta, *_ = preparer.prepare_batch_metadata(
            num_tokens_static=n_tok,
            scheduled_full_cpu=sched,
            active_mask_full_cpu=sched > 0,
            input_ids_buf=None,
            position_ids_buf=None,
            token_ids_cpu=np.zeros((n_req, 64), dtype=np.int32),
            num_computed_tokens_cpu=computed,
            temperature_cpu=zeros_f32,
            top_p_cpu=ones_f32,
            top_k_cpu=zeros_i32,
            min_p_cpu=zeros_f32,
            frequency_penalties_cpu=zeros_f32,
            presence_penalties_cpu=zeros_f32,
            repetition_penalties_cpu=ones_f32,
            page_table_cpu=pages,
            padded_num_reqs_in=n_req,
            window_row_indices_cpu=np.arange(n_req, dtype=np.int32),
            recurrent_slot_indices_cpu=slots,
            spec_recurrent_commit_cpu=commit,
        )
        if handoff is not None:
            meta, delta = preparer.apply_spec_window_handoff(meta, handoff)
            mgr._record_pending_recurrent_base_deltas(
                scheduled_cpu=sched,
                slot_indices_cpu=slots,
                delta_dev=delta,
                fed_mask=fed_mask,
            )
        cm = model_exec._build_cache_metadata(meta, n_req)
        return cm

    # Step 0 (committed): req0 at computed=10, req1 at computed=20; both verify 1+3 tokens.
    # Slot bases after step 0: slot 0 -> 11, slot 4 -> 21.
    run_prep(np.array([4, 4, 0, 0], np.int32), np.array([10, 20, 0, 0], np.int32), None)
    assert list(mgr._recurrent_base_by_slot[[0, 4]]) == [11, 21]

    # Step 1 (overlapped): step 0 accepted a0=1 (delta=2), a1=0 (delta=3).
    # True computed at step 1: req0=12, req1=21. Optimistic computed: req0=14, req1=24.
    spec_out_0 = _pack_spec_out(n_req, k, {0: ([1, 2], 1, [10, 11, 12]), 1: ([5], 0, [20, 21, 22])})
    src_1 = spec_window_sources(["r0", "r1", None, None], {"r0": (0, k), "r1": (1, k)}, n_req)
    handoff_1 = DeviceSpecWindowHandoff.from_spec_out(
        spec_out=jnp.asarray(spec_out_0), num_draft_tokens=k, sources=src_1, rewrite_recurrent_base=True
    )
    with jax.transfer_guard_device_to_host("disallow"):
        cm1 = run_prep(np.array([4, 4, 0, 0], np.int32), np.array([14, 24, 0, 0], np.int32), handoff_1)
    # Step 1 must read slot 0 + 1 and slot 4 + 0!
    np.testing.assert_array_equal(np.asarray(cm1.recurrent_read_offset)[:2], [1, 0])

    # Step 2 (chained overlap): req0 is fed again (step 1 accepted a0=2 -> delta=1; true computed=12+1+2=15, opt=16);
    # req1 sits out step 2! Must STILL incur zero D2H for fed req0!
    spec_out_1 = _pack_spec_out(n_req, k, {0: ([10, 11, 12], 2, [30, 31, 32]), 1: ([20, 21, 22], 2, [40, 41, 42])})
    src_2 = spec_window_sources(["r0", None, None, None], {"r0": (0, k)}, n_req)
    handoff_2 = DeviceSpecWindowHandoff.from_spec_out(
        spec_out=jnp.asarray(spec_out_1), num_draft_tokens=k, sources=src_2, rewrite_recurrent_base=True
    )
    with jax.transfer_guard_device_to_host("disallow"):
        cm2 = run_prep(np.array([4, 0, 0, 0], np.int32), np.array([16, 0, 0, 0], np.int32), handoff_2)
    np.testing.assert_array_equal(np.asarray(cm2.recurrent_read_offset)[:1], [2])

    # Step 3 (committed, handoff=None):
    # - req0 ran step 2 at true_computed=15 -> slot 0 true base = 16; suppose step 2 accepted a=1 -> true_computed=17.
    #   Expected read_offset for req0 = 17 - 16 = 1!
    # - req1 last ran step 1 at true_computed=21 -> slot 4 true base = 22; step 1 accepted a=2 -> true_computed=24.
    #   Expected read_offset for req1 = 24 - 22 = 2!
    cm3 = run_prep(np.array([4, 4, 0, 0], np.int32), np.array([17, 24, 0, 0], np.int32), None)
    np.testing.assert_array_equal(np.asarray(cm3.recurrent_read_offset)[:2], [1, 2])
    assert not mgr._pending_recurrent_base_deltas

    # Verify clear_recurrent_slots zeroes bases AND drops pending deltas
    mgr._pending_recurrent_base_deltas[0] = (jnp.array([3, 0, 0, 0], dtype=jnp.int32), 0)
    mgr.clear_recurrent_slots([0, 1, 2, 3])
    assert int(mgr._recurrent_base_by_slot[0]) == 0
    assert 0 not in mgr._pending_recurrent_base_deltas

    # Verify reset_recurrent_bases clears all
    mgr._recurrent_base_by_slot[4] = 99
    mgr._pending_recurrent_base_deltas[4] = (jnp.array([0, 2, 0, 0], dtype=jnp.int32), 1)
    mgr.reset_recurrent_bases()
    assert np.all(mgr._recurrent_base_by_slot == 0)
    assert not mgr._pending_recurrent_base_deltas


# ---------------------------------------------------------------------------
# S1-C12: DP + VLM get_compile_configurations
# ---------------------------------------------------------------------------
def test_s1_c12_dp_vlm_compile_configurations_keep_vlm_fields(monkeypatch):
    sharding = _single_device_sharding()
    cfg = _cache_config(dp=2, max_num_reqs=4, max_model_len=32, page_size=16)

    text_cfg = SimpleNamespace(
        vocab_size=64,
        total_vocab_size=80,
        hidden_size=8,
        rope_scaling={"mrope_section": [2, 2, 2]},
    )
    vision_cfg = SimpleNamespace(deepstack_visual_indexes=[0, 1])
    model_cfg = SimpleNamespace(
        get_text_config=lambda: text_cfg,
        image_token_id=10,
        vision_config=vision_cfg,
    )
    model = SimpleNamespace(
        config=model_cfg,
        _task_type="image-text-to-text",
        dtype=jnp.float32,
        get_image_features=lambda *a, **k: None,
    )

    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.model = model
    mgr.metadata = cfg
    mgr.max_model_len = 32
    mgr.max_num_tokens = 16
    mgr.max_num_reqs = 4
    mgr._empty_sharding = sharding
    mgr.cache_group_configs = ()
    mgr.layer_to_cache_group = ()
    mgr.graphdef = None
    mgr.graphstate = None
    mgr.graphother = None
    mgr._batch_preparer = BatchMetadataPreparer(
        metadata=cfg,
        empty_sharding=sharding,
        max_num_tokens=16,
        max_num_reqs=4,
        max_model_len=32,
        min_input_pad=1,
        input_sharding=sharding,
    )

    _, _, _, inputs = mgr.get_compile_configurations(
        kv_pages=None,
        rng_key=jax.random.PRNGKey(0),
        num_tokens=8,
        max_num_reqs=4,
        padded_num_reqs=4,
        metadata=cfg,
    )
    bm = inputs.batch_metadata
    assert bm.prefill_embeds is not None and bm.prefill_embeds.shape == (8, 8)
    assert bm.prefill_embeds_mask is not None and bm.prefill_embeds_mask.shape == (8,)
    assert bm.mrope_position_ids is not None and bm.mrope_position_ids.shape == (3, 8)
    assert bm.visual_pos_masks is not None and bm.visual_pos_masks.shape == (8,)
    assert bm.deepstack_visual_embeds is not None and len(bm.deepstack_visual_embeds) == 2


# ---------------------------------------------------------------------------
# S1-H8: Sharding-only _pipeline_model_step_templates + saved shardings across release/update
# ---------------------------------------------------------------------------
def test_s1_h8_templates_do_not_pin_live_arrays_and_release_preserves_shardings():
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sh_rep = NamedSharding(mesh, PartitionSpec())

    class _ArrayHolder(dict):
        pass

    kv = {"k": jax.device_put(np.zeros((4, 2), dtype=np.float32), sh_rep)}
    gs = {"w": jax.device_put(np.ones((4, 2), dtype=np.float32), sh_rep)}
    go = {"b": jax.device_put(np.zeros((2,), dtype=np.float32), sh_rep)}

    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.mesh = mesh
    executor._pipeline_model_step_templates = (
        executor._resolve_sharding_structure(kv),
        executor._resolve_sharding_structure(gs),
        executor._resolve_sharding_structure(go),
    )

    # Confirm templates contain ONLY Shardings, never jax.Arrays
    for tmpl in executor._pipeline_model_step_templates:
        for leaf in jax.tree_util.tree_leaves(tmpl):
            assert isinstance(leaf, jax.sharding.Sharding)
            assert not isinstance(leaf, jax.Array)

    # Refreshing kv_pages_template must not retain the live array
    kv2 = _ArrayHolder(k=jax.device_put(np.ones((4, 2), dtype=np.float32), sh_rep))
    ref_kv2 = weakref.ref(kv2)
    executor.refresh_kv_pages_template(kv2)
    del kv2
    gc.collect()
    assert ref_kv2() is None

    # Release -> update_graphs preserves compiled shardings even if restored weights arrive with a different sharding
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.mesh = mesh
    mgr.model = SimpleNamespace(mesh=mesh)
    mgr.graphdef = object()
    mgr.graphstate = {"w": jax.device_put(np.ones((4, 2), dtype=np.float32), sh_rep)}
    mgr.graphother = {"b": jax.device_put(np.zeros((2,), dtype=np.float32), sh_rep)}
    mgr._recurrent_base_by_slot = np.ones((4,), dtype=np.int64)
    mgr._pending_recurrent_base_deltas = {}
    mgr._debug_baselines = {}
    mgr._pp_microbatch = None

    cleared_prepare = []
    runtime = SimpleNamespace(clear_prepare_cache=lambda: cleared_prepare.append(True))
    executor._pipeline_runtime = runtime
    executor.model = mgr.model
    executor.graphdef = mgr.graphdef
    executor._runtime_graphstate_arg = mgr.graphstate
    executor._runtime_graphother_arg = mgr.graphother
    executor._packed_source_leaves = ((), ())
    executor._lm_head_state = {"lm": np.ones((2,), dtype=np.float32)}
    executor._lm_head_graphdef = object()
    executor._lm_head_postprocess = lambda x: x
    executor._flat_state_args = False
    executor._weight_generation = 0
    executor._cache = OrderedDict()
    executor._pipeline_model_step_fn_cache = OrderedDict()
    executor._pipeline_kv_carry_map_cache = {}
    mgr._model_executor = executor
    mgr._sampler_executor = SimpleNamespace(model=mgr.model, clear_cache=lambda: None)

    mgr.release_model_state(clear_compiled_cache=False)
    assert mgr.graphstate is None
    assert executor._lm_head_state is None
    assert cleared_prepare
    assert mgr._compiled_graphstate_shardings is not None

    # Restore with plain unsharded NumPy arrays -> must land back on sh_rep!
    new_gs = {"w": np.full((4, 2), 7.0, dtype=np.float32)}
    new_go = {"b": np.full((2,), 3.0, dtype=np.float32)}
    mgr.model = SimpleNamespace(mesh=SimpleNamespace(is_mpmd=False))
    mgr.mesh = SimpleNamespace(is_mpmd=False)
    mgr.update_graphs(graphdef=object(), graphstate=new_gs, graphother=new_go)
    assert mgr.graphstate["w"].sharding == sh_rep
    assert mgr.graphother["b"].sharding == sh_rep
    np.testing.assert_array_equal(np.asarray(mgr.graphstate["w"]), 7.0)


# ---------------------------------------------------------------------------
# S1-H17: total_vocab_size resolution in SamplerExecutor & ExecutionManager
# ---------------------------------------------------------------------------
def test_s1_h17_sampler_executor_uses_total_vocab_size():
    sharding = _single_device_sharding()
    text_cfg = SimpleNamespace(vocab_size=32, total_vocab_size=48)
    model = SimpleNamespace(config=SimpleNamespace(get_text_config=lambda: text_cfg), dtype=jnp.float32)

    se = SamplerExecutor(model=model, max_model_len=16, empty_sharding=sharding, use_aot_forward=True)
    dummy_inputs = SimpleNamespace(
        rng_key=jax.random.PRNGKey(0),
        req_num_tokens_full=jnp.zeros((2,), dtype=jnp.int32),
    )
    se.compile(num_tokens=4, padded_num_reqs=2, inputs=dummy_inputs, metadata=None, greedy=True)
    fn = se.get_compiled(num_tokens=4, padded_num_reqs=2, greedy=True)
    # Run with total_vocab_size=48 logits (would crash AOT with vocab_size=32!)
    logits = jnp.zeros((2, 48), dtype=jnp.float32).at[:, 45].set(10.0)
    _, tokens, _, _ = fn(
        jnp.ones((6, 2), dtype=jnp.float32),
        jnp.array(
            [
                [0, 1],
                [1, 1],
                [2, 2],
                [0, 1],
                [1, 1],
                [0, 0],
                [2, 2],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            dtype=jnp.int32,
        ),
        jnp.array([2, 2], dtype=jnp.int32),
        logits,
        jax.random.PRNGKey(0),
        jnp.zeros((2, 48), dtype=jnp.uint32),
    )
    np.testing.assert_array_equal(np.asarray(tokens), [45, 45])


# ---------------------------------------------------------------------------
# S1-H21: PP OOM num_pages shrink syncs metadata + invalidates caches
# ---------------------------------------------------------------------------
def test_s1_h21_oom_shrink_updates_metadata_and_invalidates_caches():
    meta = _cache_config(num_pages=100, dp=2)
    other_meta = _cache_config(num_pages=100, dp=2)
    attempts = []

    class _OomModel:
        def init_operations_cache(self, **kwargs):
            n = int(kwargs["ragged_config"].num_pages)
            attempts.append(n)
            if len(attempts) == 1:
                raise RuntimeError("RESOURCE_EXHAUSTED: out of HBM")
            return {"kv": n}

    invalidated = []
    refreshed = []
    cleared_prepare = []

    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.model = _OomModel()
    mgr.metadata = meta
    mgr.max_num_reqs = 4
    mgr.max_model_len = 64
    mgr.recurrent_slot_group_size = 1
    mgr.speculative_recurrent_state_tokens = 0
    mgr.speculative_trace_tokens = 0
    mgr.pipeline_plan = SimpleNamespace(is_enabled=True)
    mgr._batch_preparer = SimpleNamespace(
        metadata=other_meta,
        invalidate_page_table_cache=lambda: invalidated.append(True),
    )
    mgr._model_executor = SimpleNamespace(
        metadata=other_meta,
        refresh_kv_pages_template=lambda c: refreshed.append(c),
        pipeline_runtime=SimpleNamespace(clear_prepare_cache=lambda: cleared_prepare.append(True)),
        _pipeline_kv_carry_map_cache={"stale": {}},
    )

    cache = mgr._init_operations_cache_with_retry(quantizer=None, masking_details=None)
    assert attempts == [100, 78]
    assert cache == {"kv": 78}
    assert meta.num_pages == 78
    assert other_meta.num_pages == 78
    assert meta.num_pages % 2 == 0
    assert invalidated == [True]
    assert refreshed == [{"kv": 78}]
    assert cleared_prepare == [True]
    assert not mgr._model_executor._pipeline_kv_carry_map_cache


# ---------------------------------------------------------------------------
# S1-M7: Exclude cached device page table from broadcast + recursive put_host_payload + recurrent warmup hit
# ---------------------------------------------------------------------------
def test_s1_m7_recursive_put_host_payload_and_broadcast_skips_device_pages(monkeypatch):
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sh = NamedSharding(mesh, PartitionSpec())
    monkeypatch.setattr(host_sync.jax, "process_count", lambda: 2)

    already_placed = jax.device_put(np.arange(4, dtype=np.int32), sh)
    nested = {
        "a": [np.array([1, 2], dtype=np.int32), None],
        "b": (np.array([3.0], dtype=np.float32), {"c": already_placed}),
    }
    placed = host_sync.put_host_payload(nested, sh, host_payloads_replicated=True)
    assert isinstance(placed["a"], list) and placed["a"][1] is None
    assert isinstance(placed["a"][0], jax.Array) and placed["a"][0].sharding == sh
    assert isinstance(placed["b"], tuple) and placed["b"][1]["c"] is already_placed

    # Verify _broadcast_host_payload skips already-device jax.Array leaves
    seen_broadcast = []
    from jax.experimental import multihost_utils

    monkeypatch.setattr(
        multihost_utils,
        "broadcast_one_to_all",
        lambda x: (seen_broadcast.append(x), x)[1],
    )
    preparer = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
    host_a = np.zeros((4,), dtype=np.int32)
    host_b = np.ones((4,), dtype=np.int32)
    dev_pt = jax.device_put(np.full((4, 2), 9, dtype=np.int32), sh)
    merged = preparer._broadcast_host_payload((host_a, dev_pt, host_b))
    assert len(seen_broadcast) == 1
    assert len(seen_broadcast[0]) == 2  # dev_pt excluded!
    assert merged[1] is dev_pt


def test_s1_m7_recurrent_slot_programs_hit_warmup_without_recompile():
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sh = NamedSharding(mesh, PartitionSpec())
    rec = RecurrentCacheView(
        conv_state=jax.device_put(np.ones((4, 2, 3), dtype=np.float32), sh),
        recurrent_state=jax.device_put(np.ones((4, 2, 3, 3), dtype=np.float32), sh),
        positions=jax.device_put(np.arange(4, dtype=np.int32), sh),
        metadata=None,
        layer_index=0,
    )
    mgr = ExecutionManager.__new__(ExecutionManager)
    mgr.kv_pages = HybridCache(views=[rec])
    mgr.max_num_reqs = 4
    mgr.host_payloads_replicated = True
    mgr.speculative_recurrent_state_tokens = 0
    mgr.recurrent_snapshot_slots = 4

    mgr._warm_recurrent_slot_programs()
    # Now exercise clear, permute, save, restore; verify values
    mgr.clear_recurrent_slots([1])
    np.testing.assert_array_equal(np.asarray(mgr.kv_pages.views[0].conv_state[1]), 0.0)
    np.testing.assert_array_equal(np.asarray(mgr.kv_pages.views[0].conv_state[0]), 1.0)
    mgr.save_recurrent_snapshots([0], [2])
    mgr.clear_recurrent_slots([0])
    np.testing.assert_array_equal(np.asarray(mgr.kv_pages.views[0].conv_state[0]), 0.0)
    mgr.restore_recurrent_snapshots([0], [2])
    np.testing.assert_array_equal(np.asarray(mgr.kv_pages.views[0].conv_state[0]), 1.0)
    mgr.permute_recurrent_slots(np.array([0, 2, -1, 3], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(mgr.kv_pages.views[0].conv_state[2]), 0.0)


# ---------------------------------------------------------------------------
# S1-M8: Persistent child preparers + get_pipeline_lm_head part_token_shapes
# ---------------------------------------------------------------------------
def test_s1_m8_child_preparer_persists_across_window_identity_changes():
    sharding = _single_device_sharding()
    cfg0 = _cache_config(page_size=16)
    cfg1 = _cache_config(page_size=8)
    preparer = BatchMetadataPreparer(
        metadata=cfg0,
        empty_sharding=sharding,
        max_num_tokens=16,
        max_num_reqs=4,
        max_model_len=64,
        min_input_pad=1,
        cache_group_configs=(cfg0, cfg1),
        layer_to_cache_group=(0, 1),
    )

    def prep_with_identity(identity, table1):
        return preparer.prepare_batch_metadata(
            num_tokens_static=4,
            scheduled_full_cpu=np.ones(4, dtype=np.int32),
            active_mask_full_cpu=np.ones(4, dtype=bool),
            input_ids_buf=None,
            position_ids_buf=None,
            token_ids_cpu=np.zeros((4, 64), dtype=np.int32),
            num_computed_tokens_cpu=np.ones(4, dtype=np.int32),
            temperature_cpu=np.ones(4, dtype=np.float32),
            top_p_cpu=np.ones(4, dtype=np.float32),
            top_k_cpu=np.zeros(4, dtype=np.int32),
            min_p_cpu=np.zeros(4, dtype=np.float32),
            frequency_penalties_cpu=np.zeros(4, dtype=np.float32),
            presence_penalties_cpu=np.zeros(4, dtype=np.float32),
            repetition_penalties_cpu=np.ones(4, dtype=np.float32),
            page_table_cpu=np.ones((4, 4), dtype=np.int32),
            padded_num_reqs_in=4,
            cache_group_inputs=(
                CacheGroupInputs(np.ones((4, 4), dtype=np.int32), page_table_version=1, window_identity=identity),
                CacheGroupInputs(table1, page_table_version=1, window_identity=identity),
            ),
        )

    t1 = np.arange(32, dtype=np.int32).reshape(4, 8)
    t2 = t1[::-1].copy()
    bm1, *_ = prep_with_identity((0, 1, 2, 3), t1)
    bm1_pages = np.asarray(bm1.cache_groups[1].pages_tables).copy()
    child0_before = preparer._cache_group_preparers[0][1]
    child1_before = preparer._cache_group_preparers[1][1]

    # Same page_table_version=1 but changed window_identity: child must be REUSED (not recreated) AND table refreshed!
    bm2, *_ = prep_with_identity((3, 2, 1, 0), t2)
    assert preparer._cache_group_preparers[0][1] is child0_before
    assert preparer._cache_group_preparers[1][1] is child1_before
    np.testing.assert_array_equal(bm1_pages, t1)
    np.testing.assert_array_equal(np.asarray(bm2.cache_groups[1].pages_tables), t2)


def test_s1_m8_get_pipeline_lm_head_warms_part_token_shapes():
    mesh = Mesh(np.asarray(jax.devices()), ("x",))

    class _Proj(spx.Module):
        in_features = 4
        out_features = 6

        def native_forward(self, x):
            return jnp.ones((x.shape[0], 6), dtype=x.dtype) * jnp.sum(x, axis=-1, keepdims=True)

    proj = _Proj()
    head_gd, head_st = spx.export(proj)

    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor.model = SimpleNamespace(
        mesh=mesh,
        dtype=jnp.float32,
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=4)),
        get_lm_head=lambda: proj,
    )
    executor._pipeline_lm_head_cache = OrderedDict()
    executor._lm_head_graphdef = head_gd
    executor._lm_head_state = head_st
    executor._lm_head_postprocess = None
    executor._lm_head_uses_tied_projection = False
    executor._lm_head_projection_is_complete = False
    executor._lm_head_tied_precision = None
    executor._lm_head_stage_mesh = lambda: mesh
    executor._lm_head_replicated_sharding = lambda: NamedSharding(mesh, PartitionSpec())
    executor._lm_head_hidden_sharding = lambda shape: NamedSharding(mesh, PartitionSpec())
    executor._place_lm_head_hidden = lambda x: x
    executor._place_lm_head_index = lambda x: x

    # Microbatch token shape is 8 while part_rows is (3, 2)
    fn = executor.get_pipeline_lm_head(padded_num_reqs=8, part_rows=(3, 2), part_token_shapes=(8, 8))
    logits = fn(
        None,
        None,
        (jnp.ones((8, 4), dtype=jnp.float32), jnp.ones((8, 4), dtype=jnp.float32)),
        (jnp.arange(3, dtype=jnp.int32), jnp.arange(2, dtype=jnp.int32)),
    )
    assert logits.shape == (8, 6)
    # Different token shapes must have distinct cache keys
    fn2 = executor.get_pipeline_lm_head(padded_num_reqs=8, part_rows=(3, 2), part_token_shapes=(4, 4))
    assert fn2 is not fn


# ---------------------------------------------------------------------------
# S1-M9: WindowPlanner.get_token_paddings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("min_tok", "max_tok", "gap", "expected"),
    [
        (16, 256, 0, [16, 32, 64, 128, 256]),
        (16, 300, 128, [16, 32, 64, 128, 256, 300]),
        (16, 30, 128, [16, 30]),
        (16, 8, 0, [8]),
        (16, 8, 128, [8]),
        (64, 300, 32, [64, 96, 128, 160, 192, 224, 256, 288, 300]),
        (16, 128, 128, [16, 32, 64, 128]),
    ],
)
def test_s1_m9_get_token_paddings_sorted_capped_and_non_empty(min_tok, max_tok, gap, expected):
    paddings = WindowPlanner.get_token_paddings(min_tok, max_tok, gap)
    assert paddings == expected
    assert paddings == sorted(set(paddings))
    assert paddings[-1] == max_tok
    assert all(p <= max_tok for p in paddings)


# ---------------------------------------------------------------------------
# S1-L1: min_input_pad > min_token_pad + MPMD pipeline_runtime=None
# ---------------------------------------------------------------------------
def test_s1_l1_feasible_pairs_and_mpmd_without_pipeline_runtime():
    # min_input_pad=16 > min_token_pad=8 must still include (8, 16) so token bucket 8 is compiled & measurable
    assert CompileOrchestrator._get_feasible_compile_pairs([8, 16, 32], [16, 32]) == [
        (8, 16),
        (16, 16),
        (32, 16),
        (32, 32),
    ]
    # Unchanged when min(reqs_padds) <= min(num_tokens_paddings)
    assert CompileOrchestrator._get_feasible_compile_pairs([8, 16, 32], [8, 16, 32, 64]) == [
        (8, 8),
        (16, 8),
        (16, 16),
        (32, 8),
        (32, 16),
        (32, 32),
    ]

    # MPMD executor with pipeline_runtime=None normalizes use_pipeline_runtime=True -> False
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor.use_aot_forward = False
    executor._pipeline_runtime = None
    executor._cache = OrderedDict()
    executor._backbone_cache = OrderedDict()
    executor._lm_head_cache = OrderedDict()

    direct_key = (8, "backbone", "mpmd", False)
    executor._backbone_cache[direct_key] = lambda *a: BackboneOutputs(kv_pages=a[2], hidden_states=jnp.zeros((8, 4)))
    executor._lm_head_cache[(4, "lm_head", "mpmd")] = lambda *a: jnp.zeros((4, 6))

    assert executor._backbone_cache_key(8, use_pipeline_runtime=True) == direct_key
    assert executor.has_backbone(8, use_pipeline_runtime=True)
    assert executor.get_backbone(num_tokens=8, use_pipeline_runtime=True) is executor._backbone_cache[direct_key]
    combined = executor.get_compiled(num_tokens=8, padded_num_reqs=4, use_pipeline_runtime=True)
    assert callable(combined)


def test_execution_manager_destroy_kv_cache_and_reset_recurrent_bases():
    mgr = ExecutionManager.__new__(ExecutionManager)
    kv_arr = jax.device_put(jnp.ones((4, 4), dtype=jnp.float32))
    pos_arr = jax.device_put(jnp.zeros((4,), dtype=jnp.int32))
    cache_obj = HybridCache(views=[RecurrentCacheView(conv_state=kv_arr, recurrent_state=None, positions=pos_arr)])
    cache_ref = weakref.ref(cache_obj)
    mgr.kv_pages = cache_obj
    del cache_obj, kv_arr, pos_arr
    mgr._recurrent_snapshot_arrays = {"k": [object()]}
    mgr._recurrent_snapshot_signature = ("sig",)
    mgr._recurrent_base_by_slot = np.asarray([10, 20, 30, 40], dtype=np.int64)
    mgr._pending_recurrent_base_deltas = {1: (jnp.asarray([5], dtype=jnp.int32), 0)}
    mgr.last_batch_metadata = object()
    mgr.last_spec_window_delta = object()

    events = []
    mgr._batch_preparer = SimpleNamespace(invalidate_page_table_cache=lambda: events.append("inv_pt"))
    mgr._model_executor = SimpleNamespace(
        refresh_kv_pages_template=lambda val: events.append(("refresh_tpl", val)),
        pipeline_runtime=SimpleNamespace(clear_prepare_cache=lambda: events.append("clear_pp")),
        _pipeline_kv_carry_map_cache={"a": 1},
    )

    mgr.destroy_kv_cache()
    gc.collect()
    assert mgr.kv_pages is None
    assert cache_ref() is None
    assert mgr._recurrent_snapshot_arrays is None
    assert mgr._recurrent_snapshot_signature is None
    assert np.all(mgr._recurrent_base_by_slot == 0)
    assert mgr._pending_recurrent_base_deltas == {}
    assert mgr.last_batch_metadata is None
    assert mgr.last_spec_window_delta is None
    assert mgr._model_executor._pipeline_kv_carry_map_cache == {}
    assert "inv_pt" in events
    assert ("refresh_tpl", None) in events
    assert "clear_pp" in events


def test_overlap_async_device_put_snapshots_reused_host_scratch_buffers():
    """Reused NumPy scratch buffers in BatchMetadataPreparer and SamplerRuntime must not alias in-flight device_put."""
    sharding = _single_device_sharding()
    cfg = _cache_config(max_num_reqs=4, max_model_len=32, page_size=16)
    preparer = BatchMetadataPreparer(
        metadata=cfg,
        empty_sharding=sharding,
        max_num_tokens=8,
        max_num_reqs=4,
        max_model_len=32,
        min_input_pad=1,
        input_sharding=sharding,
    )

    # Simulate an async slow computation consuming the transferred array while the host mutates the scratch buffer
    @jax.jit
    def _slow_read(x, payload):
        for _ in range(15):
            x = x @ x
        return payload + jnp.int32(x[0, 0] * 0)

    work = jnp.ones((128, 128), dtype=jnp.float32)
    scratch = np.zeros((4,), dtype=np.int32)
    _ = _slow_read(work, jax.device_put(scratch.copy(), sharding)).block_until_ready()

    outs = []
    for step_val in range(1, 5):
        scratch[:] = step_val
        (placed,) = preparer._device_put_host_payload((scratch,))
        outs.append((step_val, _slow_read(work, placed)))
    for expected, arr in outs:
        assert int(np.asarray(arr)[0]) == expected


def test_s1_c9_s1_h14_non_spec_constrained_row_on_drafter_runner_advances_num_tokens_under_async(monkeypatch):
    """Constrained non-spec rows on a drafter runner under async_scheduling must advance num_tokens & num_tokens_no_spec."""
    runner = object.__new__(eSurgeRunner)
    max_num_reqs = 4
    buf = SequenceBuffer(max_num_reqs, 32, 32, 64, [4])
    req = CachedRequestState(
        "r_constrained",
        [10, 11, 12],
        SamplingParams(temperature=0.0, logit_bias={5: 2.0}),
        None,
        ([1],),
        0,
        [],
    )
    requests = {"r_constrained": req}
    buf.add_request(req)
    assert int(buf.num_tokens[0]) == 3
    assert int(buf.num_tokens_no_spec[0]) == 3

    class _StubDrafter:
        supports_draft_in_verify = True

        def spec_step(self, **kwargs):
            raise AssertionError("non-spec constrained row must not invoke spec_step")

        def verify_draft_batched(self, **kwargs):
            raise AssertionError("non-spec constrained row must not invoke verify_draft_batched")

        def reset(self, **kwargs):
            pass

    runner.sequence_buffer = buf
    runner.requests = requests
    runner.kv_cache_groups = []
    runner.metadata = SimpleNamespace(data_parallel_size=1, num_pages=32, version="v3", page_size=4)
    runner.max_num_reqs = max_num_reqs
    runner.num_reqs_max_model_len = max_num_reqs
    runner.max_model_len = 32
    runner.model = SimpleNamespace(config=SimpleNamespace(vocab_size=64))
    runner.drafter = _StubDrafter()
    runner.num_draft_tokens = 2
    runner.draft_schedule = None
    runner.async_scheduling = True
    runner._pre_async_results = None
    runner._allow_sparse_window_packing = False
    runner.slot_pool = RecurrentSlotPool(metadata=runner.metadata, max_num_reqs=max_num_reqs, slot_indexed_state=False)
    runner.enable_sampler_metrics = False
    runner.verbose = False
    runner.log_it = lambda *a, **k: None
    runner.perf = mr_mod.RunnerPerfTracker(history_maxlen=64, alpha=0.2)
    runner._spec_recorded_this_step = False
    runner._spec_carry = None
    runner.num_tokens_paddings = [4]
    runner.active_num_seq_buckets = [4]
    runner.input_ids_buf = jnp.zeros((4,), dtype=jnp.int32)
    runner.position_ids_buf = jnp.zeros((4,), dtype=jnp.int32)
    runner._scheduled_full_cpu = np.zeros((max_num_reqs,), dtype=np.int32)
    runner._active_mask_full_cpu = np.zeros((max_num_reqs,), dtype=np.bool_)
    runner._req_num_tokens_cpu = np.zeros((max_num_reqs,), dtype=np.int32)
    runner._window_row_indices_cpu = np.zeros((max_num_reqs,), dtype=np.int32)
    runner._window_recurrent_slot_indices_cpu = np.zeros((max_num_reqs,), dtype=np.int32)
    runner._window_temperature_cpu = np.zeros((max_num_reqs,), dtype=np.float32)
    runner._window_top_p_cpu = np.ones((max_num_reqs,), dtype=np.float32)
    runner._window_top_k_cpu = np.zeros((max_num_reqs,), dtype=np.int32)
    runner._window_min_p_cpu = np.zeros((max_num_reqs,), dtype=np.float32)
    runner._window_frequency_penalties_cpu = np.zeros((max_num_reqs,), dtype=np.float32)
    runner._window_presence_penalties_cpu = np.zeros((max_num_reqs,), dtype=np.float32)
    runner._window_repetition_penalties_cpu = np.ones((max_num_reqs,), dtype=np.float32)
    runner._handoff_scalar_cache = {}
    runner._handoff_positions_cache = {}
    runner._handoff_zero_scalar = None
    runner._handoff_gather_programs = {}
    runner.window_planner = SimpleNamespace(
        get_token_bucket=lambda target, active_buckets, request_bucket=None: 4,
    )
    runner.executor_manager = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        cache_group_configs=(),
        _empty_sharding=None,
        rng_key=jax.random.PRNGKey(0),
        kv_pages=None,
        execute=lambda **kwargs: (
            jnp.asarray([42, -1, -1, -1], dtype=jnp.int32),
            jnp.asarray([True, False, False, False], dtype=jnp.bool_),
            kwargs["input_ids_buf"],
            kwargs["position_ids_buf"],
            jnp.zeros((4, 8), dtype=jnp.float32),
            None,
            {},
        ),
        invalidate_sampler_penalty_state=lambda *a, **k: None,
        overwrite_sampler_penalty_rows=lambda rows: None,
    )
    runner.spec = DrafterSpeculation(runner=runner, drafter=runner.drafter, num_draft_tokens=2)
    runner.spec_decode_recurrent_candidates = False
    runner.spec_decode_recurrent_replay = False
    runner.spec_decode_debug_max_traces = 0
    runner.state_sync = SimpleNamespace(
        update_states=lambda so: False,
        reorder_decode_first=lambda so: None,
        reorder_decode_first_per_shard=lambda so, dp: None,
        modify_prev_results=lambda pr: None,
    )
    runner._plain_recurrent_row_sync_enabled = lambda: False
    runner._apply_recurrent_snapshot_restores = lambda so: None
    runner._apply_recurrent_snapshot_saves = lambda so: None
    runner._get_current_bucket = lambda n: 4
    runner.cache_group_inputs_for_execute = lambda **k: None
    runner._model_uses_vlm_inputs = lambda: False
    monkeypatch.setattr(mr_mod, "model_has_side_inputs", lambda m: False)

    sched = SchedulerOutput(
        [],
        CachedRequestData(["r_constrained"], [False], [[]], [([],)], [0]),
        {"r_constrained": 3},
        3,
        {},
        [0],
        set(),
        async_scheduling=True,
        async_safe=False,
    )
    out = runner._execute_model_impl(sched, return_async_output=True)
    assert int(buf.token_ids[0, 3]) == 42
    assert int(buf.num_tokens_no_spec[0]) == 4
    assert int(buf.num_tokens[0]) == 4
    assert req.output_token_ids == [42]
    resolved = out.get_output() if hasattr(out, "get_output") else out
    assert resolved.sampled_token_ids == [[42]]

    # Also verify EASYDEL_DISABLE_BATCHED_EMIT=1 disables _can_defer_spec_step
    sched_defer = SchedulerOutput(
        [],
        CachedRequestData(["r_constrained"], [False], [[]], [([],)], [3]),
        {"r_constrained": 1},
        1,
        {},
        [0],
        set(),
        async_scheduling=True,
        async_safe=True,
    )
    runner._has_recurrent_layers = False
    monkeypatch.setenv("EASYDEL_DISABLE_BATCHED_EMIT", "0")
    assert runner._can_defer_spec_step(sched_defer) is True
    monkeypatch.setenv("EASYDEL_DISABLE_BATCHED_EMIT", "1")
    assert runner._can_defer_spec_step(sched_defer) is False


def test_s1_c10_s1_m7_mpmd_disjoint_stage_meshes_and_multihost_placement(monkeypatch):
    """Snapshot, slot-selective restore, and _gather_spec_delta work across disjoint MPMD stage meshes and multi-host."""
    import os
    import subprocess
    import sys

    # 1. Real 2-device MPMD disjoint stage mesh test (in subprocess with 2 virtual CPU devices)
    mpmd_script = """
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from easydel.caching import HybridCache, RecurrentCacheView
from easydel.inference.esurge.runners.spec.strategy import (
    _gather_spec_delta,
    _place_host_for_array,
    _restore_recurrent_state_for_slots,
    snapshot_recurrent_state,
)
devs = jax.devices()
assert len(devs) >= 2
m0 = Mesh(np.array([devs[0]]), ("tp",))
m1 = Mesh(np.array([devs[1]]), ("tp",))
sh0 = NamedSharding(m0, PartitionSpec())
sh1 = NamedSharding(m1, PartitionSpec())
v0 = RecurrentCacheView(
    conv_state=jax.device_put(np.arange(8, dtype=np.float32).reshape(4, 2), sh0),
    recurrent_state=jax.device_put(np.arange(4, dtype=np.float32).reshape(4, 1), sh0),
    positions=jax.device_put(np.array([1, 2, 3, 4], dtype=np.int32), sh0),
    seqlen_offset=None,
    metadata=None,
)
v1 = RecurrentCacheView(
    conv_state=jax.device_put(np.arange(100, 108, dtype=np.float32).reshape(4, 2), sh1),
    recurrent_state=jax.device_put(np.arange(100, 104, dtype=np.float32).reshape(4, 1), sh1),
    positions=jax.device_put(np.array([10, 20, 30, 40], dtype=np.int32), sh1),
    seqlen_offset=None,
    metadata=None,
)
pre = HybridCache(views=[v0, v1])
snap = snapshot_recurrent_state(pre)
assert snap is not None
live = HybridCache(views=[
    v0.replace(conv_state=v0.conv_state + 50.0, recurrent_state=v0.recurrent_state + 50.0, positions=v0.positions + 5),
    v1.replace(conv_state=v1.conv_state + 50.0, recurrent_state=v1.recurrent_state + 50.0, positions=v1.positions + 5),
])
restored = _restore_recurrent_state_for_slots(live, snap, [1])
assert restored.views[0].conv_state.sharding == sh0
assert restored.views[1].conv_state.sharding == sh1
np.testing.assert_array_equal(np.asarray(restored.views[0].conv_state[1]), [2.0, 3.0])
np.testing.assert_array_equal(np.asarray(restored.views[0].conv_state[0]), [50.0, 51.0])
np.testing.assert_array_equal(np.asarray(restored.views[1].conv_state[1]), [102.0, 103.0])
np.testing.assert_array_equal(np.asarray(restored.views[1].conv_state[2]), [154.0, 155.0])
delta_dev = jax.device_put(np.array([2, 0, 3, 1], dtype=np.int32), sh1)
idx_dev = _place_host_for_array(np.array([2, -1], dtype=np.int32), delta_dev)
assert idx_dev.sharding == sh1
gathered = _gather_spec_delta(delta_dev, idx_dev)
np.testing.assert_array_equal(np.asarray(gathered), [3, 0])
"""
    env = dict(
        os.environ,
        ENABLE_DISTRIBUTED_INIT="0",
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    proc = subprocess.run([sys.executable, "-c", mpmd_script], env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"MPMD 2-device test failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"

    # 2. Verify multi-host host_sync placement path in _merge_slot_arrays and begin_draft_in_verify
    from easydel.inference.esurge.runners.spec import strategy as strat_mod

    placed_calls: list[tuple[np.ndarray, jax.sharding.Sharding, bool | None]] = []
    orig_put = strat_mod.put_host_payload

    def _spy_put(values, shardings, *, host_payloads_replicated=None):
        placed_calls.append((np.asarray(values), shardings, host_payloads_replicated))
        return orig_put(values, shardings, host_payloads_replicated=host_payloads_replicated)

    monkeypatch.setattr(strat_mod, "put_host_payload", _spy_put)
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sh = NamedSharding(mesh, PartitionSpec())
    v = RecurrentCacheView(
        conv_state=jax.device_put(np.ones((4, 2), dtype=np.float32), sh),
        recurrent_state=None,
        positions=jax.device_put(np.arange(4, dtype=np.int32), sh),
        seqlen_offset=None,
        metadata=None,
    )
    snap = snapshot_recurrent_state(HybridCache(views=[v]))
    live = HybridCache(views=[v.replace(conv_state=v.conv_state + 9.0, positions=v.positions + 9)])
    _restore_recurrent_state_for_slots(live, snap, [2], host_payloads_replicated=True)
    assert len(placed_calls) == 1
    assert placed_calls[0][0].tolist() == [False, False, True, False]
    assert placed_calls[0][1] == sh
    assert placed_calls[0][2] is True


# =============================================================================
# Leak Audit #3: n > 1 VLM sibling finish/abort while sibling 1 is still waiting
# =============================================================================


def test_leak_3_vlm_n_gt_1_sibling_finish_while_sibling_waiting_does_not_crash(monkeypatch):
    from easydel.inference.esurge.config import eSurgeContextConfig
    from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
    from easydel.inference.esurge.engine.admission import RequestAdmission
    from easydel.inference.esurge.engine.core_driver import EngineCoreDriver
    from easydel.inference.esurge.engine.registry import RequestRegistry

    # Use max_num_seqs=1 so sibling 0 runs & finishes while sibling 1 is still queued in EngineCoreDriver._records!
    core = EngineCoreDriver(
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=4,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=["layer.0"],
            )
        ],
        max_num_seqs=1,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
    )
    try:
        registry = RequestRegistry()
        admission = RequestAdmission(
            registry=registry,
            scheduler_submit=core.add_requests,
            tokenizer_client=SimpleNamespace(tokenize=lambda rid, p: [7, 90, 8]),
            tokenizer=SimpleNamespace(eos_token_id=0, decode=lambda ids, skip_special_tokens=False: ""),
            context_config=eSurgeContextConfig.coerce_config(None),
            reserve_tokens=4,
            max_model_len=64,
            tool_parser_class=None,
            reasoning_parser_class=None,
            sampling_params_callback=lambda: None,
            generation_config_dict=None,
            primary_eos_token_id=0,
            eos_token_ids=[0],
            extra_stops=[],
            ignore_stop_strings_in_reasoning=False,
            on_activity=lambda: None,
            info=lambda *a, **kw: None,
        )

        # Native non-cached model (cannot rebuild from VisionEncoderCache — must have raw pixels!)
        model = _MropeDeepStackModel(support_image_embeds=False, support_video_embeds=False)
        helper = _vlm_helper(monkeypatch, model, cache=False)
        sync, _, requests, _, _, _, _ = _make_sync()

        feat = _img_feat(4.0, offset=1)
        pv = feat.pixel_values
        grid = feat.grid_thw
        admission.add_request(
            "vlm-n2",
            "describe",
            SamplingParams(n=2, max_tokens=4),
            prompt_token_ids=[7, 90, 8],
            pixel_values=pv,
            image_grid_thw=grid,
            mm_features=[feat],
        )

        # Step 1: only sibling 0 enters runner; sibling 1 stays in EngineCoreDriver._records (waiting).
        so1 = core.schedule()
        assert [r.req_id for r in so1.scheduled_new_reqs] == ["vlm-n2-0"]
        assert core.num_waiting == 1
        sync.update_states(so1)
        helper.precompute_prefill(requests["vlm-n2-0"])
        np.testing.assert_array_equal(requests["vlm-n2-0"].prefill_inputs_embeds[:, 0], [7.0, 14.0, 8.0])

        # Sibling 0 finishes (or aborts) BEFORE sibling 1 is ever scheduled!
        core.finish_requests(["vlm-n2-0"], 1)
        so2 = core.schedule()
        assert "vlm-n2-0" in so2.finished_req_ids
        assert [r.req_id for r in so2.scheduled_new_reqs] == ["vlm-n2-1"]
        sync.update_states(so2)

        # Sibling 1 must still have intact raw media and precompute_prefill cleanly without crashing!
        req1 = requests["vlm-n2-1"]
        assert req1.pixel_values is pv
        assert len(req1.mm_features) == 1
        assert req1.mm_features[0].pixel_values is pv
        assert req1.mm_features[0].grid_thw is grid
        helper.precompute_prefill(req1)
        np.testing.assert_array_equal(req1.prefill_inputs_embeds[:, 0], [7.0, 14.0, 8.0])

        # Also verify direct NewRequestData sharing the EXACT same feat object cannot break an unscheduled sibling.
        shared_feat = _img_feat(6.0, offset=1)
        sync2, _, _reqs2, _, _, _, _ = _make_sync()
        sync2.update_states(
            SchedulerOutput(
                [
                    NewRequestData(
                        "d-0",
                        [7, 90, 8],
                        SamplingParams(),
                        ([1],),
                        0,
                        pixel_values=shared_feat.pixel_values,
                        image_grid_thw=shared_feat.grid_thw,
                        mm_features=[shared_feat],
                    )
                ],
                CachedRequestData.make_empty(),
                {"d-0": 3},
                3,
                {},
                [0],
                set(),
            )
        )
        # Finish d-0 before d-1 is even seen by sync2.
        sync2.update_states(SchedulerOutput([], CachedRequestData.make_empty(), {}, 0, {}, [0], {"d-0"}))
        assert shared_feat.pixel_values is not None
        assert shared_feat.grid_thw is not None
    finally:
        core.shutdown()


# =============================================================================
# Leak Audit #4: eSurgeRunner._side_input_provider cleared on release/update/shutdown
# =============================================================================


def test_leak_4_side_input_provider_released_and_rebuilt_across_weight_updates(monkeypatch):
    import weakref

    from easydel.inference.esurge.runners import model_runner as mr_mod
    from easydel.inference.esurge.runners.model_runner import eSurgeRunner

    class _FakeEngramModel:
        def __init__(self, table_version: int):
            self.table_version = table_version
            self.graphdef = f"gd-{table_version}"

        @property
        def esurge_compatible_model(self):
            return self

    class _FakeEngramProvider:
        def __init__(self, model, tok_state):
            self.model = model
            self.table_version = model.table_version
            self.tok_state = tok_state

    monkeypatch.setattr(mr_mod, "build_side_input_provider", lambda model, state: _FakeEngramProvider(model, state))

    runner = eSurgeRunner.__new__(eSurgeRunner)
    m1 = _FakeEngramModel(1)
    m1_ref = weakref.ref(m1)
    runner.model = m1
    runner.requests = {}
    runner.sequence_buffer = SimpleNamespace(clear=lambda: None)
    runner.slot_pool = None
    runner._executor = None
    runner._side_input_tokenizer_state = "tok-v1"
    runner._side_input_provider = None
    runner.executor_manager = SimpleNamespace(
        release_model_state=lambda clear_compiled_cache=False: None,
        update_graphs=lambda **kwargs: None,
        drop_recurrent_snapshots=lambda: None,
        reset_recurrent_bases=lambda: None,
        shutdown=lambda: None,
        last_batch_metadata="stale-meta",
        last_spec_window_delta="stale-delta",
    )

    p1 = runner._get_side_input_provider()
    assert p1.table_version == 1
    p1_ref = weakref.ref(p1)

    # 1. release_model_state() must drop _side_input_provider so old Engram modules/tables are freed.
    runner.release_model_state()
    assert runner._side_input_provider is None
    assert runner.model is None
    del p1, m1
    import gc

    gc.collect()
    assert p1_ref() is None
    assert m1_ref() is None

    # 2. update_model_weights() resets _side_input_provider so it rebuilds against new weights/tables.
    m2 = _FakeEngramModel(2)
    runner.model = m2
    p2 = runner._get_side_input_provider()
    assert p2.table_version == 2

    m3 = _FakeEngramModel(3)
    runner._setup_variables = lambda: None
    runner.update_model_weights(model=m3, reset_state=False)
    assert runner._side_input_provider is None
    p3 = runner._get_side_input_provider()
    assert p3.table_version == 3
    assert p3 is not p2

    # 3. shutdown() also clears _side_input_provider.
    runner.shutdown()
    assert runner._side_input_provider is None


# =============================================================================
# Leak Audit #15: BatchMetadataPreparer & SamplerRuntime device HBM cache release
# =============================================================================


def test_leak_15_device_hbm_caches_released_on_destroy_kv_and_release_model_state():
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
    from easydel.inference.esurge.runners.executors.sampler_executor import SamplerRuntime

    # 1. BatchMetadataPreparer.clear_device_caches() clears own & nested child preparer device caches.
    parent = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
    child = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
    async_child = BatchMetadataPreparer.__new__(BatchMetadataPreparer)
    for p in (parent, child, async_child):
        p._cached_pages_tables_dev = jnp.ones((2, 2), dtype=jnp.int32)
        p._cached_page_table_version = 7
        p._cached_rows_to_copy = 2
        p._zero_dev_cache = {("prefill_embeds", (4, 8), "float32"): jnp.zeros((4, 8), dtype=jnp.float32)}
        p._payload_put_cache = {0: (b"x", (2,), "int32", jnp.zeros((2,), dtype=jnp.int32))}
        p._pending_transfer = (jnp.zeros((2,), dtype=jnp.int32),)
        p._pending_transfer_metadata = {"tag": 1}
        p._pending_group_result = ("group",)
        p._cache_group_preparers = {}
        p._async_group_preparer = None
    parent._cache_group_preparers = {0: ((1,), child)}
    parent._async_group_preparer = async_child

    # 2. SamplerRuntime releases [max_num_reqs, vocab_size] buffers and lazily recreates them.
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    sampler_rt = SamplerRuntime(
        sampler_executor=SimpleNamespace(),
        sampler_sharding=sharding,
        empty_sharding=sharding,
        max_num_reqs=4,
        vocab_size=32,
        min_input_pad=1,
    )
    assert sampler_rt._sampler_zero_token_counts is not None
    assert sampler_rt._sampler_token_counts is not None

    em = ExecutionManager.__new__(ExecutionManager)
    em.kv_pages = object()
    em.drop_recurrent_snapshots = lambda: None
    em.last_batch_metadata = object()
    em.last_spec_window_delta = object()
    em._batch_preparer = parent
    em.sampler_runtime = sampler_rt
    em._greedy_total_tokens_cache = {4: jnp.array(4, dtype=jnp.int32)}
    em._model_executor = SimpleNamespace(
        model=object(),
        graphdef=object(),
        clear_runtime_graph_args=lambda: None,
        refresh_kv_pages_template=lambda val: None,
        _pipeline_kv_carry_map_cache={"k": "v"},
        pipeline_runtime=SimpleNamespace(clear_prepare_cache=lambda: None),
    )
    em._sampler_executor = SimpleNamespace(model=object())
    em.mesh = None
    em.graphdef = None
    em.graphstate = None
    em.graphother = None

    em.destroy_kv_cache()
    for p in (parent, child, async_child):
        assert p._cached_pages_tables_dev is None
        assert p._zero_dev_cache == {}
        assert p._payload_put_cache == {}
        assert p._pending_transfer is None
        assert p._pending_transfer_metadata is None
        assert p._pending_group_result is None
    assert sampler_rt._sampler_token_counts is None
    assert sampler_rt._sampler_zero_token_counts is None
    assert em._greedy_total_tokens_cache == {}

    # Lazy recreation on demand works seamlessly after release.
    recreated = em._sampler_zero_token_counts
    assert recreated.shape == (4, 32)
    assert recreated.dtype == jnp.uint32

    # Repopulate and confirm release_model_state() also clears every cache!
    parent._zero_dev_cache[("deepstack", (2, 4), "float32")] = jnp.zeros((2, 4), dtype=jnp.float32)
    em._greedy_total_tokens_cache[8] = jnp.array(8, dtype=jnp.int32)
    em.release_model_state(clear_compiled_cache=False)
    assert parent._zero_dev_cache == {}
    assert sampler_rt._sampler_token_counts is None
    assert sampler_rt._sampler_zero_token_counts is None
    assert em._greedy_total_tokens_cache == {}


# =============================================================================
# Leak Audit #25: PipelineStageRuntime._logged_plan_keys & runner.reset_state()
# =============================================================================


def test_leak_25_pipeline_logged_plan_keys_and_runner_reset_state_clears_metadata():
    from easydel.inference.esurge.runners import pipeline_runtime as pr_mod
    from easydel.inference.esurge.runners.model_runner import eSurgeRunner
    from easydel.inference.esurge.runners.pipeline_runtime import PipelineStageRuntime

    # 1. PipelineStageRuntime._logged_plan_keys cleared on clear_prepare_cache()/shutdown() and bounded.
    rt = PipelineStageRuntime.__new__(PipelineStageRuntime)
    cleared = []
    rt._inline_executor = SimpleNamespace(
        clear_prepare_cache=lambda: cleared.append("inline_clear"),
        shutdown=lambda: cleared.append("inline_shutdown"),
        get_prepared_plan_summary=lambda key, **kw: {
            "arg_leaf_counts": [1],
            "num_stages": 1,
            "kv_leaves": 1,
            "metadata_leaves": 1,
            "dynamic_slots": [0],
            "dynamic_kv_slots": [0],
            "dynamic_metadata_slots": [0],
            "stage_slots": [0],
            "prev_slots": [0],
            "total_invars": [1],
            "donated_by_stage": [0],
            "kv_outputs_by_stage": [0],
        },
    )
    rt._wavefront_executor = SimpleNamespace(
        clear_prepare_cache=lambda: cleared.append("wave_clear"),
        shutdown=lambda: cleared.append("wave_shutdown"),
    )
    rt._logged_plan_keys = set()

    for idx in range(pr_mod._MAX_LOGGED_PLAN_KEYS + 20):
        rt._maybe_log_cached_plan("single", rt._inline_executor, ("gen", idx))
    assert len(rt._logged_plan_keys) <= pr_mod._MAX_LOGGED_PLAN_KEYS

    rt.clear_prepare_cache()
    assert rt._logged_plan_keys == set()

    rt._logged_plan_keys.add(("single", ("gen", 999)))
    rt.shutdown()
    assert rt._logged_plan_keys == set()

    # 2. eSurgeRunner.reset_state() clears executor_manager.last_batch_metadata / last_spec_window_delta.
    runner = eSurgeRunner.__new__(eSurgeRunner)
    runner.requests = {"r1": object()}
    runner.sequence_buffer = SimpleNamespace(clear=lambda: None)
    runner.slot_pool = None
    runner.executor_manager = SimpleNamespace(
        last_batch_metadata=object(),
        last_spec_window_delta=object(),
        drop_recurrent_snapshots=lambda: None,
        reset_recurrent_bases=lambda: None,
    )
    runner.reset_state()
    assert runner.executor_manager.last_batch_metadata is None
    assert runner.executor_manager.last_spec_window_delta is None
