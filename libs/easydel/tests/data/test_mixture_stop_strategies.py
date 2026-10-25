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

"""Stop-strategy semantics of the mixers: ``error`` mode, per-epoch reshuffled restarts, seeked-resume
equivalence across wrap boundaries, weight validation, live draw accounting and dataset attribution."""

from __future__ import annotations

from itertools import islice

import pytest
from easydel.data.core.config import DatasetConfig, MixStageConfig, PipelineConfig, WeightSchedulePoint
from easydel.data.core.protocols import PipelineContext
from easydel.data.transforms.mixture import (
    DATASET_ID_FIELD,
    MixedShardedSource,
    MixStage,
    WeightScheduler,
    block_mixture_interleave,
)

from ._common import ListSource

pytestmark = pytest.mark.cpu_ok


def _source(name: str, n_shards: int, rows_per_shard: int, with_info: bool = True) -> ListSource:
    return ListSource(
        {f"{name}-s{i}": [{"src": name, "shard": i, "i": j} for j in range(rows_per_shard)] for i in range(n_shards)},
        with_info=with_info,
    )


def _mix(sources, **kw):
    kw.setdefault("weights", dict.fromkeys(sources, 1.0))
    kw.setdefault("seed", 11)
    return MixedShardedSource(sources=sources, **kw)


def _epochs_of(rows, source_name, rows_per_epoch):
    seq = [r["shard"] for r in rows if r["__source__"] == source_name]
    return [seq[k : k + rows_per_epoch] for k in range(0, len(seq) - rows_per_epoch + 1, rows_per_epoch)]


def _counts(rows):
    out: dict[str, int] = {}
    for r in rows:
        out[r["__source__"]] = out.get(r["__source__"], 0) + 1
    return out


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="stop_strategy"):
        _mix({"a": _source("a", 1, 4)}, stop_strategy="restrat")


def test_restart_preserves_exact_ratios_after_wrap():
    mix = _mix({"small": _source("small", 2, 5), "big": _source("big", 2, 500)}, block_size=10)
    rows = list(islice(mix.open_shard("mixed_shard_0"), 400))
    # 40 blocks x (5 small + 5 big): ratios hold exactly although "small" wrapped ~20 times.
    assert _counts(rows) == {"small": 200, "big": 200}


def test_error_strategy_raises_on_exhaustion():
    mix = _mix({"small": _source("small", 1, 6), "big": _source("big", 1, 500)}, block_size=10, stop_strategy="error")
    with pytest.raises(RuntimeError, match="'small' exhausted"):
        list(islice(mix.open_shard("mixed_shard_0"), 100))


def test_error_strategy_is_silent_while_no_source_exhausts():
    mix = _mix({"a": _source("a", 1, 50), "b": _source("b", 1, 50)}, block_size=10, stop_strategy="error")
    rows = list(islice(mix.open_shard("mixed_shard_0"), 100))
    assert _counts(rows) == {"a": 50, "b": 50}


def test_all_exhausted_sees_every_row_once_then_terminates():
    mix = _mix(
        {"small": _source("small", 1, 6), "big": _source("big", 1, 30)}, block_size=10, stop_strategy="all_exhausted"
    )
    rows = list(mix.open_shard("mixed_shard_0"))
    assert _counts(rows) == {"small": 6, "big": 30}
    assert sorted(r["i"] for r in rows if r["src"] == "big") == list(range(30))


def test_restart_default_replays_identical_shard_order():
    mix = _mix({"a": _source("a", 4, 3), "pad": _source("pad", 1, 10_000)}, block_size=8)
    epochs = _epochs_of(list(islice(mix.open_shard("mixed_shard_0"), 200)), "a", 12)
    assert len(epochs) >= 3
    assert epochs[1] == epochs[0] == epochs[2] == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3


def _reshuffled(seed=7, **kw):
    return _mix(
        {"a": _source("a", 12, 3), "pad": _source("pad", 1, 10_000)},
        block_size=8,
        reshuffle_on_restart=True,
        seed=seed,
        **kw,
    )


def test_reshuffle_on_restart_permutes_each_epoch_deterministically():
    rows = list(islice(_reshuffled().open_shard("mixed_shard_0"), 400))
    epochs = _epochs_of(rows, "a", 36)
    assert len(epochs) >= 3
    assert epochs[0] == [s for s in range(12) for _ in range(3)], "epoch 0 must keep canonical shard order"
    assert epochs[1] != epochs[0] and epochs[2] != epochs[1]  # 12 shards: P(identity permutation) ~ 2e-9
    assert sorted(epochs[1]) == sorted(epochs[0]) == sorted(epochs[2])  # nothing lost or doubled
    assert list(islice(_reshuffled().open_shard("mixed_shard_0"), 400)) == rows  # same seed, same stream
    assert list(islice(_reshuffled(seed=8).open_shard("mixed_shard_0"), 400)) != rows


@pytest.mark.parametrize("with_info", [True, False], ids=["metadata-seek", "sequential-skip"])
@pytest.mark.parametrize("split", [5, 36, 37, 100, 250])
def test_seeked_resume_is_exact_across_reshuffled_wraps(split, with_info):
    def build():
        return _mix(
            {"a": _source("a", 12, 3, with_info=with_info), "b": _source("b", 2, 50, with_info=with_info)},
            block_size=8,
            reshuffle_on_restart=True,
            seed=11,
        )

    full = list(islice(build().open_shard("mixed_shard_0"), 300))
    tail = list(islice(build().open_shard_at_row("mixed_shard_0", split), 300 - split))
    assert tail == full[split:]


@pytest.mark.parametrize("row", [0, 40, 73, 180])
def test_unseeded_reshuffle_resumes_exactly_on_a_new_mixer(row):
    """An unseeded mixer draws its seed once and records it in the resume marker, so reshuffled epochs replay."""
    sources = {"a": _source("a", 6, 4), "b": _source("b", 1, 500)}
    mixer = MixedShardedSource(sources, block_size=8, seed=None, reshuffle_on_restart=True)
    expected = list(islice(mixer.open_shard("mixed_shard_0"), row, row + 60))
    state = mixer.resume_state(row)
    fresh = MixedShardedSource(sources, block_size=8, seed=None, reshuffle_on_restart=True)
    assert list(islice(fresh.open_shard_from_state(state), 60)) == expected


def test_seeked_resume_with_empty_constituent_replays_the_fresh_stream():
    """An empty constituent forces sequential replay; the result still equals the fresh tail."""
    mix = _mix({"empty": ListSource({"e-s0": []}, with_info=True), "b": _source("b", 1, 30)}, block_size=10)
    full = list(islice(mix.open_shard("mixed_shard_0"), 40))
    assert list(islice(mix.open_shard_at_row("mixed_shard_0", 20), 20)) == full[20:]


def test_mixstage_forwards_options_and_keeps_seed_zero():
    config = PipelineConfig(datasets=[DatasetConfig(data_files="x.jsonl", name="a")])
    stage = MixStage(MixStageConfig(reshuffle_on_restart=True, attach_dataset_ids=True, seed=0, block_size=4))
    mixed = stage.process({"a": _source("a", 2, 4), "b": _source("b", 2, 4)}, PipelineContext(config=config, seed=777))[
        "mixed"
    ]
    ref = MixedShardedSource(
        {"a": _source("a", 2, 4), "b": _source("b", 2, 4)},
        block_size=4,
        seed=0,
        reshuffle_on_restart=True,
        attach_dataset_ids=True,
    )
    assert list(islice(mixed.open_shard("mixed_shard_0"), 40)) == list(islice(ref.open_shard("mixed_shard_0"), 40))


def test_mixstage_config_accepts_error_strategy():
    assert MixStageConfig(stop_strategy="error").stop_strategy == "error"


def test_scheduler_names_must_match_sources():
    sched = WeightScheduler([WeightSchedulePoint(step=0, weights={"x": 0.5, "y": 0.5})], interpolation="step")
    with pytest.raises(ValueError, match="dataset names"):
        _mix({"a": _source("a", 1, 4), "b": _source("b", 1, 4)}, weight_scheduler=sched)


def test_zero_weight_source_is_never_drawn_and_all_zero_raises():
    mix = _mix({"a": _source("a", 1, 50), "z": _source("z", 1, 50)}, weights={"a": 1.0, "z": 0.0}, block_size=10)
    assert all(r["__source__"] == "a" for r in islice(mix.open_shard("mixed_shard_0"), 100))
    with pytest.raises(ValueError, match="> 0"):
        _mix({"a": _source("a", 1, 5)}, weights={"a": 0.0})


def test_weight_scheduler_clamps_past_last_knot():
    sched = WeightScheduler(
        [
            WeightSchedulePoint(step=0, weights={"a": 0.9, "b": 0.1}),
            WeightSchedulePoint(step=100, weights={"a": 0.5, "b": 0.5}),
        ],
        interpolation="linear",
    )
    w = sched.get_weights(step=10_000_000)
    assert abs(w["a"] - 0.5) < 1e-9 and abs(w["b"] - 0.5) < 1e-9


def test_live_draw_counts_track_realized_mixture():
    mix = _mix({"a": _source("a", 1, 500), "b": _source("b", 1, 500)}, weights={"a": 3.0, "b": 1.0}, block_size=8)
    rows = list(islice(mix.open_shard("mixed_shard_0"), 400))
    assert mix.last_draw_counts == {"a": 300, "b": 100}
    assert sum(mix.last_draw_counts.values()) == len(rows)


def test_live_draw_counts_exact_across_wraps():
    mix = _mix({"a": _source("a", 1, 50), "b": _source("b", 1, 50)}, block_size=10)
    rows = list(islice(mix.open_shard("mixed_shard_0"), 160))  # each source wraps once
    assert mix.last_draw_counts == _counts(rows) == {"a": 80, "b": 80}
    assert mix.last_epochs == {"a": 1, "b": 1}


def test_checkpoint_replay_does_not_clobber_live_counts():
    mix = _mix({"a": _source("a", 1, 50), "b": _source("b", 1, 50)}, block_size=10)
    list(islice(mix.open_shard("mixed_shard_0"), 20))
    mix.resume_state(70)
    assert mix.last_draw_counts == {"a": 10, "b": 10}


def test_negative_weights_rejected_everywhere():
    with pytest.raises(ValueError, match="non-negative"):
        _mix({"a": _source("a", 1, 4), "b": _source("b", 1, 4)}, weights={"a": 1.5, "b": -0.5})
    with pytest.raises(ValueError, match="non-negative"):
        WeightSchedulePoint(step=0, weights={"a": 1.5, "b": -0.5})
    with pytest.raises(ValueError, match="non-negative"):
        block_mixture_interleave({"a": [{"v": 1}], "b": [{"v": 2}]}, weights={"a": 1.5, "b": -0.5}, block_size=4)


@pytest.mark.parametrize("strategy", ["restart", "error"])
def test_mixture_starvation_rejected_for_recycling_strategies(strategy):
    with pytest.raises(ValueError, match="starvation"):
        _mix(
            {"a": _source("a", 1, 40), "b": _source("b", 1, 40)},
            weights={"a": 1000.0, "b": 1.0},
            block_size=10,
            stop_strategy=strategy,
        )
    sched = WeightScheduler(
        [
            WeightSchedulePoint(step=0, weights={"a": 0.5, "b": 0.5}),
            WeightSchedulePoint(step=50, weights={"a": 0.99, "b": 0.01}),
        ]
    )
    with pytest.raises(ValueError, match="knot 50"):
        _mix(
            {"a": _source("a", 1, 40), "b": _source("b", 1, 40)},
            block_size=10,
            weight_scheduler=sched,
            stop_strategy=strategy,
        )


def test_tiny_weights_still_drain_under_all_exhausted():
    """Finite strategies redistribute over live sources, so a tiny weight is not starvation there."""
    mix = _mix(
        {"a": _source("a", 1, 3), "b": _source("b", 1, 2)},
        weights={"a": 1000.0, "b": 1.0},
        block_size=10,
        stop_strategy="all_exhausted",
    )
    assert _counts(list(mix.open_shard("mixed_shard_0"))) == {"a": 3, "b": 2}


def test_single_source_reshuffle_wraps_and_accounts_exactly():
    def build():
        return _mix({"a": _source("a", 12, 3)}, block_size=8, reshuffle_on_restart=True, seed=7)

    epochs = _epochs_of(list(islice(build().open_shard("mixed_shard_0"), 108)), "a", 36)
    assert len(epochs) == 3 and epochs[1] != epochs[0] and sorted(epochs[1]) == sorted(epochs[0])
    mix = build()
    list(islice(mix.open_shard("mixed_shard_0"), 40))
    assert mix.last_draw_counts == {"a": 40}
    assert mix.last_epochs == {"a": 1}


def test_attach_dataset_ids_on_fresh_and_resumed_streams():
    def build():
        return _mix({"x": _source("x", 1, 30), "y": _source("y", 1, 30)}, block_size=6, attach_dataset_ids=True)

    rows = list(islice(build().open_shard("mixed_shard_0"), 90))
    assert all(r[DATASET_ID_FIELD] == {"x": 0, "y": 1}[r["__source__"]] for r in rows)
    assert list(islice(build().open_shard_at_row("mixed_shard_0", 33), 57)) == rows[33:]
    plain = _mix({"x": _source("x", 1, 30), "y": _source("y", 1, 30)}, block_size=6)
    assert DATASET_ID_FIELD not in next(iter(plain.open_shard("mixed_shard_0")))


def _rows(name: str, n: int):
    return [{"src": name, "i": i} for i in range(n)]


def test_block_interleave_rejects_unknown_stop():
    with pytest.raises(ValueError, match="stop"):
        block_mixture_interleave({"a": _rows("a", 2)}, stop="first-exhausted")


def test_block_interleave_error_raises_on_exhaustion():
    mixed = block_mixture_interleave(
        {"short": _rows("short", 3), "long": _rows("long", 50)}, block_size=4, seed=0, stop="error"
    )
    with pytest.raises(RuntimeError, match="short"):
        list(islice(iter(mixed), 200))


def test_block_interleave_all_exhausted_recycles_then_terminates():
    """Regression: every non-restart value used to behave like first_exhausted."""
    mixed = block_mixture_interleave(
        {"short": _rows("short", 2), "long": _rows("long", 20)}, block_size=4, seed=0, stop="all_exhausted"
    )
    rows = list(iter(mixed))
    assert sorted(r["i"] for r in rows if r["src"] == "long") == list(range(20))  # long consumed exactly once
    shorts = [r for r in rows if r["src"] == "short"]
    assert len(shorts) > 2 and {r["i"] for r in shorts} == {0, 1}  # short recycled past its own exhaustion


def test_block_interleave_first_exhausted_stops_at_first():
    mixed = block_mixture_interleave(
        {"short": _rows("short", 2), "long": _rows("long", 20)}, block_size=4, seed=0, stop="first_exhausted"
    )
    rows = list(iter(mixed))
    assert sum(1 for r in rows if r["src"] == "short") == 2
    assert len(rows) < 10


def test_block_interleave_starvation_and_attribution():
    with pytest.raises(ValueError, match="starvation"):
        block_mixture_interleave(
            {"a": _rows("a", 40), "b": _rows("b", 40)}, weights={"a": 1000.0, "b": 1.0}, block_size=10
        )
    mixed = block_mixture_interleave(
        {"a": _rows("a", 40), "b": _rows("b", 40)}, block_size=10, seed=0, attach_dataset_ids=True
    )
    rows = list(islice(iter(mixed), 40))
    assert all(r[DATASET_ID_FIELD] == {"a": 0, "b": 1}[r["src"]] for r in rows)
