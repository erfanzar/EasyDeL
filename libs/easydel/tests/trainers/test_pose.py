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

"""Unit tests for PoSE (Positional Skip-wisE) position augmentation.

The two worked examples (A: unpacked single doc; B: packed via seq_lens with a padded
tail) are the exact position oracles from the spec; the RNG is pinned via a scripted
generator so the split/gap are deterministic. Also covers the warmup schedule, no-op
safety cases, the input_pos==0 reset-detection path, and the freq-table validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers.pose import (
    PoSEConfig,
    _seq_lens_from_segment_ids,
    apply_pose_to_batch,
    apply_pose_to_positions,
    current_pose_probability,
    validate_pose_target_max_pos,
)


class _ScriptedRNG:
    """Returns pre-scripted draws so split/gap are deterministic in tests.

    ``random()`` (Bernoulli gate + warped gap) and ``integers(low, high)`` (split + uniform
    gap) are consumed from independent queues in call order — matching numpy.random.Generator.
    """

    def __init__(self, randoms=(), integers=()):
        self._r = list(randoms)
        self._i = list(integers)

    def random(self):
        return self._r.pop(0)

    def integers(self, low, high):
        v = self._i.pop(0)
        assert low <= v < high, f"scripted int {v} out of [{low},{high})"
        return v


def test_example_A_unpacked_single_doc():
    """Spec §2.3: tokens len 6, target 16, frac 0.5/0.5, split_idx=3, gap=2 → [0,1,2,5,6,7]."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=16, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    pos = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    rng = _ScriptedRNG(randoms=[0.0], integers=[2])  # Bernoulli fires; gap=2
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng)
    assert pos.tolist() == [[0, 1, 2, 5, 6, 7]]


def test_example_B_packed_seqlens_padded_tail():
    """Spec §2.4: seq_lens=[3,2,3], padded tail, gaps 1 then 0 → [0,2,3, 0,1, 2,3,4]."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=10, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    pos = np.array([[0, 1, 2, 0, 1, 2, 3, 4]], dtype=np.int64)
    tokens = np.array([[11, 12, 13, 14, 15, 0, 0, 0]], dtype=np.int64)
    labels = np.array([[21, 22, 23, 24, 25, -100, -100, -100]], dtype=np.int64)
    seq_lens = np.array([[3, 2, 3]], dtype=np.int64)
    seq_lens_num = np.array([3], dtype=np.int64)
    rng = _ScriptedRNG(randoms=[0.0, 0.0], integers=[1, 0])  # 2 docs fire; gaps 1,0
    apply_pose_to_positions(
        pos,
        config=cfg,
        step=0,
        rng=rng,
        tokens=tokens,
        labels=labels,
        pad_id=0,
        seq_lens=seq_lens,
        seq_lens_num=seq_lens_num,
    )
    assert pos.tolist() == [[0, 2, 3, 0, 1, 2, 3, 4]]


def test_warmup_schedule():
    """Spec §4: p_pose 0.10 → max 0.50 over 10 steps; linear; disabled→0; no-warmup→p_pose."""
    cfg = PoSEConfig(enabled=True, p_pose=0.10, max_p_pose=0.50, warmup_steps=10, target_max_pos=128)
    assert current_pose_probability(cfg, 0) == pytest.approx(0.10)
    assert current_pose_probability(cfg, 5) == pytest.approx(0.30)
    assert current_pose_probability(cfg, 10) == pytest.approx(0.50)
    assert current_pose_probability(cfg, 999) == pytest.approx(0.50)  # clamped
    assert current_pose_probability(PoSEConfig(enabled=False, p_pose=0.5), 5) == 0.0
    no_warmup = PoSEConfig(enabled=True, p_pose=0.3, warmup_steps=0, target_max_pos=128)
    assert current_pose_probability(no_warmup, 7) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "cfg_kwargs, pos",
    [
        (dict(enabled=False, p_pose=1.0, target_max_pos=16), [[0, 1, 2, 3]]),  # disabled
        (dict(enabled=True, p_pose=0.0, target_max_pos=16), [[0, 1, 2, 3]]),  # p<=0
        (dict(enabled=True, p_pose=1.0, target_max_pos=16), [[0]]),  # S<2
    ],
)
def test_noop_cases(cfg_kwargs, pos):
    cfg = PoSEConfig(**cfg_kwargs)
    arr = np.array(pos, dtype=np.int64)
    before = arr.copy()
    # p<=0 / disabled never draw; S<2 returns before drawing — empty scripted RNG is safe
    apply_pose_to_positions(arr, config=cfg, step=0, rng=_ScriptedRNG())
    assert arr.tolist() == before.tolist()


def test_noop_max_gap_le_zero():
    """target_max_pos too small ⇒ max_gap<=0 ⇒ positions untouched (Bernoulli draw consumed)."""
    cfg = PoSEConfig(enabled=True, p_pose=1.0, target_max_pos=4, min_chunk_frac=0.5, max_chunk_frac=0.5)
    pos = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)  # L=6, max_gap=4-0-6<0
    before = pos.copy()
    apply_pose_to_positions(pos, config=cfg, step=0, rng=_ScriptedRNG(randoms=[0.0]))
    assert pos.tolist() == before.tolist()


def test_segment_detection_input_pos_reset():
    """Fallback: packed docs detected via input_pos==0 resets (no seq_lens/cu_seqlens)."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=32, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    # two docs of len 4 each, positions reset at index 4
    pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int64)
    rng = _ScriptedRNG(randoms=[0.0, 0.0], integers=[3, 3])  # both docs fire, gap=3 each
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng)
    # doc1 (base0,split2,gap3): [0,1, 5,6]; doc2 (base0,split2,gap3): [0,1, 5,6]
    assert pos.tolist() == [[0, 1, 5, 6, 0, 1, 5, 6]]


def test_validate_target_max_pos():
    cfg = PoSEConfig(enabled=True, p_pose=0.5, target_max_pos=8192)
    validate_pose_target_max_pos(cfg, model_max_freq_pos=8192, who="model")  # equal is ok
    validate_pose_target_max_pos(cfg, model_max_freq_pos=262144, who="model")
    validate_pose_target_max_pos(cfg, model_max_freq_pos=None, who="model")  # unknown -> skip
    with pytest.raises(ValueError, match="target_max_pos"):
        validate_pose_target_max_pos(cfg, model_max_freq_pos=4096, who="model")
    validate_pose_target_max_pos(PoSEConfig(enabled=False, target_max_pos=999999), model_max_freq_pos=8, who="m")


def test_config_validation():
    with pytest.raises(ValueError, match="p_pose"):
        PoSEConfig(p_pose=1.5)
    with pytest.raises(ValueError, match="min_chunk_frac must be <="):
        PoSEConfig(min_chunk_frac=0.8, max_chunk_frac=0.2)
    with pytest.raises(ValueError, match="target_max_pos must be > 1"):
        PoSEConfig(enabled=True, p_pose=0.5, target_max_pos=1)
    assert PoSEConfig(p_pose=0.3).max_p_pose == 0.3  # defaults to p_pose


def test_positions_stay_below_target_real_rng():
    """Property: with a real RNG, every emitted position < target_max_pos and chunk1 unchanged."""
    cfg = PoSEConfig(enabled=True, p_pose=1.0, target_max_pos=128, bias_large_gap=True)
    rng = np.random.default_rng(0)
    for _ in range(50):
        pos = np.tile(np.arange(20, dtype=np.int64), (4, 1))  # 4 rows, single doc each, L=20
        apply_pose_to_positions(pos, config=cfg, step=0, rng=rng)
        assert pos.max() < cfg.target_max_pos
        assert (pos[:, 0] == 0).all()  # first token of each doc keeps base 0


def test_core_image_token_skip():
    """Text-only gate: a doc containing image_token_id is left untouched (no PoSE, no RNG draw)."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=32, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    pos = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.int64)
    input_ids = np.array([[10, 11, 12, 13, 10, 999, 12, 13]], dtype=np.int64)  # doc1 has image token 999
    seq_lens = np.array([[4, 4]], dtype=np.int64)
    seq_lens_num = np.array([2], dtype=np.int64)
    # doc0 (text) fires: Bernoulli=0.0, gap=5 (split=2 deterministic); doc1 (image) skipped → no draw
    rng = _ScriptedRNG(randoms=[0.0], integers=[5])
    apply_pose_to_positions(
        pos,
        config=cfg,
        step=0,
        rng=rng,
        tokens=input_ids,
        seq_lens=seq_lens,
        seq_lens_num=seq_lens_num,
        image_token_id=999,
    )
    assert pos.tolist() == [[0, 1, 7, 8, 0, 1, 2, 3]]  # doc0 skipped+gap; doc1 untouched


def test_seq_lens_from_segment_ids():
    seg = np.array([[0, 0, 0, 1, 1, 2, 2, 2]], dtype=np.int64)
    seq_lens, seq_lens_num = _seq_lens_from_segment_ids(seg)
    assert seq_lens.tolist() == [[3, 2, 3]]
    assert seq_lens_num.tolist() == [3]


def test_apply_pose_to_batch_text_only_3d_mrope():
    """Hook on a 3D mRoPE batch: text doc gets identical skip on all 3 axes; image doc untouched."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=64, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    input_ids = np.array([[10, 11, 12, 13, 10, 999, 12, 13]], dtype=np.int64)
    labels = np.array([[10, 11, 12, 13, 10, 999, 12, 13]], dtype=np.int64)
    segment_ids = np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int64)
    # 3D [3,B,S]: text doc0 = arange replicated on all axes; image doc1 = a non-arange grid
    grid = np.array([[5, 5, 5, 6], [5, 6, 7, 5], [5, 5, 6, 5]], dtype=np.int64)  # (3, 4) image-doc positions
    position_ids = np.zeros((3, 1, 8), dtype=np.int64)
    for ax in range(3):
        position_ids[ax, 0, :4] = np.arange(4)
        position_ids[ax, 0, 4:] = grid[ax]
    image_grid = position_ids[:, :, 4:].copy()  # snapshot image-doc positions
    batch = {"position_ids": position_ids, "input_ids": input_ids, "labels": labels, "segment_ids": segment_ids}
    apply_pose_to_batch(batch, config=cfg, step=0, seed=0, image_token_id=999, pad_id=0)
    out = batch["position_ids"]
    # image doc untouched on every axis
    assert np.array_equal(out[:, :, 4:], image_grid)
    # text doc: all 3 axes identical, first token base 0, all positions < target
    assert np.array_equal(out[0, 0, :4], out[1, 0, :4]) and np.array_equal(out[1, 0, :4], out[2, 0, :4])
    assert out[0, 0, 0] == 0
    assert out.max() < cfg.target_max_pos


def test_apply_pose_to_batch_noop_and_missing_positions():
    """Disabled is a no-op; enabled but missing position_ids fails loud (real misconfig)."""
    batch = {"position_ids": np.tile(np.arange(4), (1, 1)), "input_ids": np.zeros((1, 4), np.int64)}
    before = batch["position_ids"].copy()
    apply_pose_to_batch(batch, config=PoSEConfig(enabled=False), step=0, seed=0, image_token_id=None, pad_id=0)
    assert np.array_equal(batch["position_ids"], before)
    cfg = PoSEConfig(enabled=True, p_pose=1.0, target_max_pos=32)
    with pytest.raises(ValueError, match="position_ids"):
        apply_pose_to_batch(
            {"input_ids": np.zeros((1, 4), np.int64)}, config=cfg, step=0, seed=0, image_token_id=None, pad_id=0
        )


def test_random_split_branch_and_draw_order():
    """Gap A: default-frac (0.25/0.75) min<max RANDOM split branch + Bernoulli→split→gap order.

    L=8, frac 0.25/0.75 ⇒ min_chunk=ceil(2)=2, max_chunk=floor(6)=6 ⇒ split drawn from [2,6].
    Scripted integers consumed in order [split=4, gap=3]; if the code drew gap before split this
    would mis-order and fail.
    """
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=64, min_chunk_frac=0.25, max_chunk_frac=0.75, bias_large_gap=False
    )
    pos = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    rng = _ScriptedRNG(randoms=[0.0], integers=[4, 3])  # Bernoulli fires; split=4; gap=3
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng)
    # first=[0,1,2,3]; second_start=0+4+3=7; second=[7,8,9,10]
    assert pos.tolist() == [[0, 1, 2, 3, 7, 8, 9, 10]]


def test_cu_seqlens_segment_path():
    """Gap B: per-doc segmentation via cu_seqlens (+cu_seqlens_num)."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=32, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    pos = np.array([[0, 1, 2, 0, 1]], dtype=np.int64)
    cu_seqlens = np.array([[0, 3, 5, 0]], dtype=np.int64)  # offsets 0,3,5 (+ trailing pad slot)
    cu_seqlens_num = np.array([3], dtype=np.int64)
    rng = _ScriptedRNG(randoms=[0.0, 0.0], integers=[5, 0])  # 2 docs fire; gaps 5,0
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng, cu_seqlens=cu_seqlens, cu_seqlens_num=cu_seqlens_num)
    # seg(0,3): split=1,base=0,gap=5 -> [0,6,7]; seg(3,5): split=1,base=0,gap=0 -> [0,1]
    assert pos.tolist() == [[0, 6, 7, 0, 1]]


def test_bias_large_gap_warp():
    """Gap C: bias_large_gap=True warp g=1-(1-u)^2, gap=int(u'*(max_gap+1)), drawn via random()."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=16, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=True
    )
    pos = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    # random() order: [Bernoulli=0.0, gap_u=0.5]; u=1-(1-0.5)^2=0.75; max_gap=16-0-6=10; gap=int(0.75*11)=8
    rng = _ScriptedRNG(randoms=[0.0, 0.5], integers=[])  # frac 0.5/0.5 ⇒ split not drawn
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng)
    # split=3; second_start=0+3+8=11; second=[11,12,13]
    assert pos.tolist() == [[0, 1, 2, 11, 12, 13]]


@pytest.mark.parametrize("bad", [np.zeros(4, dtype=np.int64), np.zeros((1, 2, 3), dtype=np.int64)])
def test_rank_not_2_raises(bad):
    """Gap D: positions must be rank-2 [B,S]; rank-1 / rank-3 fail loud."""
    cfg = PoSEConfig(enabled=True, p_pose=1.0, target_max_pos=16)
    with pytest.raises(ValueError, match="rank-2"):
        apply_pose_to_positions(bad, config=cfg, step=0, rng=_ScriptedRNG())


def test_valid_end_lt2_row_and_short_trailing_segment_skipped():
    """Gap E: an all-pad row (valid_end<2) is untouched; an L<2 trailing segment is skipped (no RNG draw)."""
    cfg = PoSEConfig(
        enabled=True, p_pose=1.0, target_max_pos=32, min_chunk_frac=0.5, max_chunk_frac=0.5, bias_large_gap=False
    )
    # row 0: entirely padded tail -> valid_end=0 -> skipped; row 1: doc(len3) + trailing doc(len1)
    pos = np.array([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=np.int64)
    tokens = np.array([[0, 0, 0, 0], [11, 12, 13, 14]], dtype=np.int64)
    labels = np.array([[-100, -100, -100, -100], [21, 22, 23, 24]], dtype=np.int64)
    # only row1 seg(0,3) fires+draws (Bernoulli=0.0, gap=2); the L=1 trailing seg draws nothing
    rng = _ScriptedRNG(randoms=[0.0], integers=[2])
    apply_pose_to_positions(pos, config=cfg, step=0, rng=rng, tokens=tokens, labels=labels, pad_id=0)
    # row0 untouched; row1 seg(0,3): split=1,base=0,gap=2 -> [0,3,4]; trailing pos[3]=0 untouched
    assert pos.tolist() == [[0, 1, 2, 3], [0, 3, 4, 0]]
