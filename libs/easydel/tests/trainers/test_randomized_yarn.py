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
"""Randomized YaRN position augmentation: emitted position_ids, curriculum, determinism and refusals."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers.pose import PoSEConfig
from easydel.trainers.randomized_yarn import (
    RandomizedYaRNConfig,
    apply_randomized_yarn_to_batch,
    apply_randomized_yarn_to_positions,
    current_l_max,
    randomized_yarn_rng,
    sample_monotone_positions,
    validate_randomized_yarn_target,
)

pytestmark = pytest.mark.cpu_ok


def _cfg(**kwargs) -> RandomizedYaRNConfig:
    base = {"enabled": True, "l_max_schedule": [64], "target_max_pos": 64}
    base.update(kwargs)
    return RandomizedYaRNConfig(**base)


def _apply(positions, cfg, step=0, seed=0, **kwargs):
    batch = {"position_ids": positions, **kwargs}
    return apply_randomized_yarn_to_batch(batch, config=cfg, step=step, seed=seed, image_token_id=None, pad_id=0)


def test_positions_sorted_distinct_and_within_l_max():
    cfg = _cfg(l_max_schedule=[48], target_max_pos=64)
    out = _apply(np.tile(np.arange(16, dtype=np.int64), (8, 1)), cfg)["position_ids"]
    assert out.shape == (8, 16)
    assert (np.diff(out, axis=1) > 0).all()
    assert out.min() >= 0 and out.max() < 48
    assert not (out == np.arange(16)).all()


def test_run_length_keeps_runs_contiguous():
    rng = np.random.default_rng(3)
    for n in (16, 10):
        out = sample_monotone_positions(rng, n=n, l_max=64, run_length=4)
        assert out.shape == (n,) and (np.diff(out) > 0).all() and out.max() < 64
        for start in range(0, n, 4):
            run = out[start : start + 4]
            assert run[0] % 4 == 0
            assert (np.diff(run) == 1).all()


def test_run_length_through_the_batch_transform():
    cfg = _cfg(l_max_schedule=[256], target_max_pos=256, run_length=8)
    out = _apply(np.tile(np.arange(32, dtype=np.int64), (4, 1)), cfg, step=5)["position_ids"]
    runs = out.reshape(4, 4, 8)
    assert (np.diff(runs, axis=-1) == 1).all()
    assert (runs[..., 0] % 8 == 0).all()


def test_curriculum_schedule_phases():
    cfg = _cfg(l_max_schedule=[8, 16, 32], target_max_pos=32, curriculum_total_steps=30)
    assert [current_l_max(cfg, s) for s in (0, 9, 10, 19, 20, 29, 30, 10_000)] == [8, 8, 16, 16, 32, 32, 32, 32]
    assert current_l_max(_cfg(l_max_schedule=[8, 16, 32], target_max_pos=32), 0) == 32
    assert current_l_max(RandomizedYaRNConfig(), 0) == 0

    positions = np.tile(np.arange(6, dtype=np.int64), (64, 1))
    early = _apply(positions.copy(), cfg, step=0)["position_ids"]
    late = _apply(positions.copy(), cfg, step=25)["position_ids"]
    assert early.max() < 8
    assert late.max() >= 16 and late.max() < 32


def test_determinism_per_seed_and_step_without_cross_run_collisions():
    cfg = _cfg(l_max_schedule=[4096], target_max_pos=4096)
    positions = np.tile(np.arange(32, dtype=np.int64), (2, 1))

    def run(seed, step):
        return _apply(positions.copy(), cfg, step=step, seed=seed)["position_ids"]

    np.testing.assert_array_equal(run(7, 3), run(7, 3))
    assert not np.array_equal(run(7, 3), run(7, 4))
    # seed + step would hand (1, 0) and (0, 1) the same stream.
    assert not np.array_equal(run(1, 0), run(0, 1))
    a = randomized_yarn_rng(1, 0).integers(0, 2**31, 8)
    b = randomized_yarn_rng(0, 1).integers(0, 2**31, 8)
    assert not np.array_equal(a, b)


def test_packed_documents_each_keep_their_own_base_and_order():
    cfg = _cfg(l_max_schedule=[40], target_max_pos=64)
    positions = np.array([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    segment_ids = np.array([[0] * 6 + [1] * 8], dtype=np.int64)
    out = _apply(positions, cfg, segment_ids=segment_ids)["position_ids"][0]
    first, second = out[:6], out[6:]
    for doc in (first, second):
        assert (np.diff(doc) > 0).all() and doc.min() >= 0 and doc.max() < 40


def test_padded_tail_and_image_documents_untouched():
    cfg = _cfg(l_max_schedule=[64], target_max_pos=64)
    positions = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    tokens = np.array([[5, 6, 7, 8, 9, 0, 0, 0]], dtype=np.int64)
    labels = np.array([[5, 6, 7, 8, 9, -100, -100, -100]], dtype=np.int64)
    out = _apply(positions.copy(), cfg, input_ids=tokens, labels=labels)["position_ids"]
    np.testing.assert_array_equal(out[0, 5:], [5, 6, 7])
    assert (np.diff(out[0, :5]) > 0).all()

    image = np.array([[5, 999, 7, 8, 9, 10, 11, 12]], dtype=np.int64)
    batch = {"position_ids": positions.copy(), "input_ids": image}
    apply_randomized_yarn_to_batch(batch, config=cfg, step=0, seed=0, image_token_id=999, pad_id=None)
    np.testing.assert_array_equal(batch["position_ids"], positions)


def test_target_ceiling_caps_each_document_from_its_base():
    cfg = _cfg(l_max_schedule=[64], target_max_pos=110)
    positions = np.arange(100, 108, dtype=np.int64)[None, :]
    rng = np.random.default_rng(0)
    for _ in range(50):
        pos = positions.copy()
        apply_randomized_yarn_to_positions(pos, config=cfg, step=0, rng=rng)
        assert pos[0, 0] >= 100 and pos.max() < 110 and (np.diff(pos[0]) > 0).all()
    too_long = np.arange(100, 115, dtype=np.int64)[None, :]
    pos = too_long.copy()
    apply_randomized_yarn_to_positions(pos, config=cfg, step=0, rng=rng)
    np.testing.assert_array_equal(pos, too_long)


def test_mrope_axes_get_identical_draws_and_input_is_not_mutated():
    cfg = _cfg(l_max_schedule=[256], target_max_pos=256)
    base = np.tile(np.arange(12, dtype=np.int64), (3, 2, 1))
    base.flags.writeable = False
    out = _apply(base, cfg, step=2)["position_ids"]
    assert out.shape == (3, 2, 12)
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[1], out[2])
    np.testing.assert_array_equal(base, np.tile(np.arange(12), (3, 2, 1)))


def test_marginal_draw_is_uniform():
    rng = np.random.default_rng(11)
    counts = np.zeros(16)
    for _ in range(4000):
        np.add.at(counts, sample_monotone_positions(rng, n=4, l_max=16), 1)
    expected = 4000 * 4 / 16
    assert np.all(np.abs(counts - expected) < 0.1 * expected)


def test_noop_and_missing_positions():
    positions = np.tile(np.arange(8, dtype=np.int64), (1, 1))
    out = _apply(positions.copy(), RandomizedYaRNConfig())["position_ids"]
    np.testing.assert_array_equal(out, positions)
    out = _apply(positions.copy(), _cfg(p_apply=0.0))["position_ids"]
    np.testing.assert_array_equal(out, positions)
    with pytest.raises(ValueError, match="position_ids"):
        apply_randomized_yarn_to_batch(
            {"input_ids": positions}, config=_cfg(), step=0, seed=0, image_token_id=None, pad_id=0
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"l_max_schedule": []}, "non-empty l_max_schedule"),
        ({"l_max_schedule": [32, 16]}, "non-decreasing"),
        ({"l_max_schedule": [128], "target_max_pos": 64}, "target_max_pos must be >="),
        ({"run_length": 0}, "run_length"),
        ({"p_apply": 1.5}, "p_apply"),
        ({"curriculum_total_steps": -1}, "curriculum_total_steps"),
        ({"only_buckets": []}, "only_buckets"),
    ],
)
def test_config_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kwargs)


def test_rope_table_check_and_dict_roundtrip():
    cfg = _cfg(l_max_schedule=[64, 128], target_max_pos=256, curriculum_total_steps=10, run_length=4)
    with pytest.raises(ValueError, match="RoPE frequency-table"):
        validate_randomized_yarn_target(cfg, model_max_freq_pos=128, who="Tiny")
    validate_randomized_yarn_target(cfg, model_max_freq_pos=256, who="Tiny")
    assert RandomizedYaRNConfig.from_dict(cfg.to_dict()) == cfg


def test_training_arguments_refuse_pose_with_randomized_yarn():
    from easydel.trainers.training_configurations import TrainingArguments

    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingArguments(
            pose=PoSEConfig(enabled=True, p_pose=0.5, target_max_pos=128),
            randomized_yarn={"enabled": True, "l_max_schedule": [64], "target_max_pos": 128},
        )
    args = TrainingArguments(randomized_yarn={"enabled": True, "l_max_schedule": [64], "target_max_pos": 128})
    assert isinstance(args.randomized_yarn, RandomizedYaRNConfig) and args.randomized_yarn.enabled
