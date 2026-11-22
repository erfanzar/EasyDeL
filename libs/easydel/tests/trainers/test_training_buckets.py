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

"""Unit tests for training buckets (``easydel.trainers.buckets``).

The first part (rule math + config resolution) is pure-Python and needs no JAX.
The structural-equality and end-to-end tests construct a small model and are
skipped when JAX / a real model backend is unavailable.
"""

from __future__ import annotations

import copy
import itertools

import pytest
from easydel.trainers.buckets import (
    BucketRule,
    CallableBucketRule,
    CycleBucketRule,
    ModBucketRule,
    PatternBucketRule,
    PiecewiseBucketRule,
    StepThresholdRule,
    TrainingBucket,
    resolve_bucket_config,
)


class _FakeConfig:
    """Minimal stand-in for EasyDeLBaseConfig with a mutable attribute."""

    def __init__(self, attn_mechanism: str = "auto"):
        self.attn_mechanism = attn_mechanism


# #
# BucketRule math
# #


class TestModBucketRule:
    def test_mod_matches_every_n_steps(self):
        rule = ModBucketRule(mod=5, on_bucket=0, off_bucket=1)
        # step % 5 == 0 -> on_bucket (0); else off_bucket (1)
        assert [rule.select(s) for s in range(12)] == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]

    def test_offset_shifts_alignment(self):
        rule = ModBucketRule(mod=3, offset=1, on_bucket=2, off_bucket=0)
        # (step - 1) % 3 == 0 -> steps 1, 4, 7 ...
        assert rule.select(0) == 0
        assert rule.select(1) == 2
        assert rule.select(4) == 2

    def test_invalid_mod_raises(self):
        with pytest.raises(ValueError):
            ModBucketRule(mod=0)


class TestCycleBucketRule:
    def test_alternates_in_period_blocks(self):
        rule = CycleBucketRule(period=5, num_buckets=2)
        # 5 consecutive steps per bucket, wrapping: 0*5, 1*5, 0*5, ...
        assert [rule.select(s) for s in range(22)] == [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 2

    def test_three_bucket_cycle(self):
        rule = CycleBucketRule(period=2, num_buckets=3)
        assert [rule.select(s) for s in range(8)] == [0, 0, 1, 1, 2, 2, 0, 0]

    def test_roundtrip(self):
        rule = CycleBucketRule(period=5, num_buckets=2)
        restored = BucketRule.from_dict(rule.to_dict())
        assert isinstance(restored, CycleBucketRule)
        assert [restored.select(s) for s in range(12)] == [rule.select(s) for s in range(12)]

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            CycleBucketRule(period=0)
        with pytest.raises(ValueError):
            CycleBucketRule(period=5, num_buckets=0)


class TestStepThresholdRule:
    def test_threshold_boundaries(self):
        rule = StepThresholdRule(thresholds=[100, 500])
        assert rule.select(0) == 0
        assert rule.select(99) == 0
        assert rule.select(100) == 1
        assert rule.select(499) == 1
        assert rule.select(500) == 2
        assert rule.select(10000) == 2

    def test_unsorted_input_is_sorted(self):
        rule = StepThresholdRule(thresholds=[500, 100])
        assert rule.thresholds == [100, 500]

    def test_duplicate_thresholds_raise(self):
        with pytest.raises(ValueError):
            StepThresholdRule(thresholds=[100, 100, 500])

    def test_single_threshold(self):
        rule = StepThresholdRule(thresholds=[10])
        assert rule.select(9) == 0
        assert rule.select(10) == 1


class TestPiecewiseBucketRule:
    def test_segments_delegate_on_global_step(self):
        # mod-3 before step 10, mod-2 from step 10 on; sub-rules see the
        # global step, so the mod-2 segment's parity is global, not local.
        rule = PiecewiseBucketRule(
            boundaries=[10],
            rules=[
                ModBucketRule(mod=3, offset=0, on_bucket=1, off_bucket=0),
                ModBucketRule(mod=2, offset=0, on_bucket=1, off_bucket=0),
            ],
        )
        assert [rule.select(s) for s in range(9)] == [1, 0, 0, 1, 0, 0, 1, 0, 0]
        assert rule.select(9) == 1  # last step of segment 0: (9 - 0) % 3 == 0
        assert rule.select(10) == 1  # first step of segment 1: 10 % 2 == 0
        assert [rule.select(s) for s in range(10, 16)] == [1, 0, 1, 0, 1, 0]

    def test_boundary_belongs_to_later_segment(self):
        rule = PiecewiseBucketRule(
            boundaries=[5],
            rules=[
                ModBucketRule(mod=1, on_bucket=0, off_bucket=1),  # always 0
                ModBucketRule(mod=1, on_bucket=1, off_bucket=0),  # always 1
            ],
        )
        assert rule.select(4) == 0
        assert rule.select(5) == 1

    def test_cooldown_cadence_change(self):
        # The production shape: long-context bucket (1) every 11th step for
        # most of the run, then every 4th step during a 500-step cooldown.
        start, cooldown_start, end = 15_400, 17_900, 18_400
        rule = PiecewiseBucketRule(
            boundaries=[cooldown_start],
            rules=[
                ModBucketRule(mod=11, offset=(start + 10) % 11, on_bucket=1, off_bucket=0),
                ModBucketRule(mod=4, offset=(cooldown_start + 3) % 4, on_bucket=1, off_bucket=0),
            ],
        )
        pre = [s for s in range(start, cooldown_start) if rule.select(s) == 1]
        assert pre[0] == start + 10  # opens with ten short steps
        assert all(b - a == 11 for a, b in itertools.pairwise(pre))
        cool = [s for s in range(cooldown_start, end) if rule.select(s) == 1]
        assert cool[0] == cooldown_start + 3  # cooldown opens with three short steps
        assert all(b - a == 4 for a, b in itertools.pairwise(cool))
        assert len(cool) == 125

    def test_roundtrip(self):
        rule = PiecewiseBucketRule(
            boundaries=[100, 200],
            rules=[
                ModBucketRule(mod=5, on_bucket=1, off_bucket=0),
                CycleBucketRule(period=2, num_buckets=2),
                StepThresholdRule(thresholds=[300]),
            ],
        )
        d = rule.to_dict()
        assert d["kind"] == "piecewise"
        restored = BucketRule.from_dict(d)
        assert isinstance(restored, PiecewiseBucketRule)
        assert restored.to_dict() == d
        probe = [0, 5, 99, 100, 101, 199, 200, 299, 300, 10**6]
        assert [restored.select(s) for s in probe] == [rule.select(s) for s in probe]

    def test_rule_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            PiecewiseBucketRule(boundaries=[10], rules=[ModBucketRule(mod=2)])

    def test_non_increasing_boundaries_raise(self):
        rules = [ModBucketRule(mod=2), ModBucketRule(mod=3), ModBucketRule(mod=5)]
        with pytest.raises(ValueError):
            PiecewiseBucketRule(boundaries=[10, 10], rules=rules)
        with pytest.raises(ValueError):
            PiecewiseBucketRule(boundaries=[20, 10], rules=rules)

    def test_non_rule_entry_raises(self):
        with pytest.raises(TypeError):
            PiecewiseBucketRule(boundaries=[], rules=[lambda s: 0])


class TestPatternBucketRule:
    """Repeating explicit N-way ratios — what mod/cycle/piecewise cannot express."""

    def test_three_bucket_2_2_1_cadence(self):
        # The production shape: 2 x medium(0), 2 x short(1), 1 x long(2) over mod 5.
        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2])
        assert [rule.select(s) for s in range(12)] == [0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 0]

    def test_realized_ratio_is_exact_over_many_cycles(self):
        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2])
        # Anchored at a resumed step divisible by the modulus: phase is well defined.
        window = [rule.select(s) for s in range(3000, 3500)]
        assert window[:5] == [0, 0, 1, 1, 2]
        counts = {b: window.count(b) for b in (0, 1, 2)}
        assert counts == {0: 200, 1: 200, 2: 100}  # exactly 2:2:1 over 500 steps

    def test_offset_shifts_phase(self):
        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2], offset=1)
        # pattern[0] now lands on step 1, so step 0 sees the last phase (2).
        assert rule.select(0) == 2
        assert [rule.select(s) for s in range(1, 6)] == [0, 0, 1, 1, 2]

    def test_negative_steps_stay_in_range(self):
        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2])
        # Python modulo keeps negatives in range; a rule must never emit a bad index.
        assert all(0 <= rule.select(s) <= 2 for s in range(-10, 10))

    def test_roundtrip(self):
        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2], offset=3)
        d = rule.to_dict()
        assert d == {"kind": "pattern", "pattern": [0, 0, 1, 1, 2], "offset": 3}
        restored = BucketRule.from_dict(d)
        assert isinstance(restored, PatternBucketRule)
        assert restored.to_dict() == d
        probe = [0, 1, 2, 3, 4, 5, 3000, 3001, 6999, 10**6]
        assert [restored.select(s) for s in probe] == [rule.select(s) for s in probe]

    def test_json_serializable(self):
        # TrainingArguments.to_dict runs on every checkpoint save, so the dict
        # projection must survive json.dumps (CallableBucketRule does not).
        import json

        rule = PatternBucketRule(pattern=[0, 0, 1, 1, 2])
        assert json.loads(json.dumps(rule.to_dict())) == rule.to_dict()

    def test_empty_pattern_raises(self):
        with pytest.raises(ValueError):
            PatternBucketRule(pattern=[])

    def test_negative_bucket_index_raises(self):
        with pytest.raises(ValueError):
            PatternBucketRule(pattern=[0, -1])

    def test_existing_kinds_cannot_express_2_2_1(self):
        """Why this rule exists: the other kinds structurally cannot do 2:2:1."""
        target = [0, 0, 1, 1, 2] * 2
        # equal-width blocks only -> 2:2:2
        assert [CycleBucketRule(period=2, num_buckets=3).select(s) for s in range(10)] != target
        # two buckets only
        assert len({ModBucketRule(mod=5).select(s) for s in range(10)}) == 2
        # absolute step ranges, not repeating phases: sticks in the last segment
        piecewise = PiecewiseBucketRule(
            boundaries=[2, 4],
            rules=[
                ModBucketRule(mod=1, on_bucket=0, off_bucket=0),
                ModBucketRule(mod=1, on_bucket=1, off_bucket=1),
                ModBucketRule(mod=1, on_bucket=2, off_bucket=2),
            ],
        )
        assert [piecewise.select(s) for s in range(10)] == [0, 0, 1, 1, 2, 2, 2, 2, 2, 2]


class TestCallableBucketRule:
    def test_callable_select(self):
        rule = CallableBucketRule(fn=lambda s: 0 if s % 2 == 0 else 1)
        assert rule.select(0) == 0
        assert rule.select(1) == 1
        assert rule.select(2) == 0

    def test_not_serializable(self):
        rule = CallableBucketRule(fn=lambda s: 0)
        with pytest.raises(TypeError):
            rule.to_dict()


# #
# Serialization round-trips
# #


class TestRuleSerialization:
    def test_mod_rule_roundtrip(self):
        rule = ModBucketRule(mod=5, offset=0, on_bucket=0, off_bucket=1)
        d = rule.to_dict()
        assert d["kind"] == "mod"
        restored = BucketRule.from_dict(d)
        assert isinstance(restored, ModBucketRule)
        assert restored.to_dict() == d

    def test_step_rule_roundtrip(self):
        rule = StepThresholdRule(thresholds=[100, 500])
        d = rule.to_dict()
        assert d["kind"] == "step"
        restored = BucketRule.from_dict(d)
        assert isinstance(restored, StepThresholdRule)
        assert restored.thresholds == [100, 500]

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            BucketRule.from_dict({"kind": "nope"})


# #
# resolve_bucket_config
# #


class TestResolveBucketConfig:
    def test_none_inherits_base(self):
        base = _FakeConfig("auto")
        bucket = TrainingBucket(name="inherit", max_length=8, config=None)
        out = resolve_bucket_config(base, bucket)
        assert out is base
        assert out.attn_mechanism == "auto"

    def test_dict_overrides_via_setattr(self):
        base = _FakeConfig("auto")
        bucket = TrainingBucket(name="vanilla", max_length=8, config={"attn_mechanism": "vanilla"})
        out = resolve_bucket_config(base, bucket)
        assert out.attn_mechanism == "vanilla"
        # base is not mutated (deepcopy).
        assert base.attn_mechanism == "auto"

    def test_dict_unknown_key_raises(self):
        base = _FakeConfig("auto")
        bucket = TrainingBucket(name="bad", max_length=8, config={"does_not_exist": 1})
        with pytest.raises(AttributeError):
            resolve_bucket_config(base, bucket)

    def test_explicit_config_passes_through(self):
        other = _FakeConfig("blocksparse")
        base = _FakeConfig("auto")
        bucket = TrainingBucket(name="splash", max_length=8, config=other)
        out = resolve_bucket_config(base, bucket)
        assert out is other
        assert out.attn_mechanism == "blocksparse"

    def test_base_independent_after_override(self):
        """A dict override must not share mutable state with the base config."""
        base = _FakeConfig("auto")
        base_copy = copy.deepcopy(base)
        bucket = TrainingBucket(name="v", max_length=8, config={"attn_mechanism": "vanilla"})
        out = resolve_bucket_config(base, bucket)
        out.attn_mechanism = "changed"
        assert base.attn_mechanism == base_copy.attn_mechanism == "auto"


# #
# TrainingBucket defaults
# #


class TestTrainingBucket:
    def test_defaults_are_none(self):
        b = TrainingBucket(name="x", max_length=64)
        assert b.config is None
        assert b.gradient_accumulation_steps is None
        assert b.loss_config is None
        assert b.step_partition_spec is None

    def test_distillation_bucket_shape(self):
        # The motivating example: vanilla 8k vs blocksparse 131k.
        vanilla = TrainingBucket(name="vanilla_8k", max_length=8192, config={"attn_mechanism": "vanilla"})
        splash = TrainingBucket(name="splash_131k", max_length=131072, config={"attn_mechanism": "blocksparse"})
        assert vanilla.max_length == 8192
        assert splash.max_length == 131072


# #
# Structural-equality + integration (need JAX + a model backend)
# #


def _has_jax():
    try:
        import jax  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_jax(), reason="requires JAX")
class TestBucketGraphdefStructure:
    """Verify that two model variants differing only in attn_mechanism share an
    identical graphstate structure — the invariant the bucket design relies on.
    """

    def test_vanilla_and_blocksparse_share_param_structure(self):
        import jax
        import spectrax as spx
        from easydel import LlamaConfig
        from easydel.modules.llama.modeling_llama import LlamaForCausalLM

        # Force a CPU backend-friendly tiny config.
        base_cfg = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        base_cfg.attn_mechanism = "vanilla"
        alt_cfg = copy.deepcopy(base_cfg)
        alt_cfg.attn_mechanism = "blocksparse"

        base_model = LlamaForCausalLM.lazy_init(
            config=base_cfg,
            dtype=jax.numpy.float32,
            param_dtype=jax.numpy.float32,
            precision="high",
            rngs=spx.Rngs(0),
        )
        alt_model = LlamaForCausalLM.lazy_init(
            config=alt_cfg,
            dtype=jax.numpy.float32,
            param_dtype=jax.numpy.float32,
            precision="high",
            rngs=spx.Rngs(0),
        )
        _, gstate_base, _ = base_model.split_module()
        _, gstate_alt, _ = alt_model.split_module()
        assert jax.tree_util.tree_structure(gstate_base) == jax.tree_util.tree_structure(gstate_alt), (
            "vanilla and blocksparse variants must share graphstate structure"
        )


class _FakeShardedSource:
    """Minimal ShardedDataSource stand-in yielding one scripted first row."""

    def __init__(self, first_row: dict, *, wrapper_name: str | None = None):
        self._first_row = first_row
        self._wrapper_name = wrapper_name
        if wrapper_name is not None:
            # Emulate a heterogeneous wrapper (MixedShardedSource/ShuffledShardedSource)
            # by presenting that type name while delegating rows through ``_source``.
            self.__class__ = type(wrapper_name, (_FakeShardedSource,), {})
            self._source = _FakeShardedSource(first_row)

    @property
    def shard_names(self):
        return ["shard-0"]

    def open_shard(self, _name):
        return iter([self._first_row])


class _StubDistillTrainer:
    """Bare attribute holder to exercise DistillationTrainer's per-bucket transform
    resolution without constructing a full trainer (no model / tokenizer download)."""

    def __init__(self, *, max_length: int, user_data_collator: bool):
        from types import SimpleNamespace

        from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer

        self.processing_class = SimpleNamespace(chat_template=None)
        self.arguments = SimpleNamespace(
            max_length=max_length,
            dataset_text_field=None,
            assistant_only_loss=False,
            completion_only_loss=None,
            sequence_packing=False,
            truncation_mode="keep_start",
        )
        self._user_data_collator = user_data_collator
        # Bind the real methods under test (and their helpers) onto this stub so
        # internal ``self.`` calls resolve, without constructing a full trainer.
        self._source_requires_row_preprocessing = DistillationTrainer._source_requires_row_preprocessing
        for name in (
            "_source_is_pretokenized",
            "_build_sft_preprocess_transform",
            "_get_bucket_preprocess_transform",
        ):
            setattr(self, name, getattr(DistillationTrainer, name).__get__(self, type(self)))


class TestDistillationBucketPreprocess:
    """DistillationTrainer resolves the per-bucket SFT transform from the BUCKET
    source's own schema and sizes it to the bucket — not the primary train source.

    Regression: a pretokenized long-context primary previously gated the transform
    off globally, so a raw-text bucket reached its collator without ``input_ids``
    (KeyError); and when it did tokenize, it padded to the global ``max_length``.
    """

    def test_pretokenized_bucket_source_skips_transform(self):
        stub = _StubDistillTrainer(max_length=131072, user_data_collator=False)
        src = _FakeShardedSource({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        assert stub._source_is_pretokenized(src) is True
        assert stub._get_bucket_preprocess_transform(src, TrainingBucket(name="b", max_length=131072)) is None

    def test_raw_bucket_source_tokenizes_at_bucket_length_unpadded(self):
        from easydel.trainers.prompt_transforms import SFTPreprocessTransform

        stub = _StubDistillTrainer(max_length=131072, user_data_collator=False)
        raw = _FakeShardedSource({"messages": [{"role": "user", "content": "hi"}]})
        assert stub._source_is_pretokenized(raw) is False
        bucket = TrainingBucket(name="a", max_length=8192, data_collator=lambda rows: rows)
        transform = stub._get_bucket_preprocess_transform(raw, bucket)
        assert isinstance(transform, SFTPreprocessTransform)
        # sized to the BUCKET (8192), not the global 131072
        assert transform._max_length == 8192
        # a bucket with its own packing collator wants unpadded rows
        assert transform._padding is False

    def test_raw_bucket_without_collator_pads_to_bucket_length(self):
        stub = _StubDistillTrainer(max_length=131072, user_data_collator=False)
        raw = _FakeShardedSource({"messages": [{"role": "user", "content": "hi"}]})
        transform = stub._get_bucket_preprocess_transform(raw, TrainingBucket(name="a", max_length=4096))
        assert transform._max_length == 4096
        assert transform._padding == "max_length"

    def test_mixed_source_treated_as_raw(self):
        # A heterogeneous mixture (pretokenized VLM rows + raw text rows) must not
        # be judged pretokenized off its first row — the transform passes already
        # tokenized rows through and tokenizes the rest.
        stub = _StubDistillTrainer(max_length=131072, user_data_collator=False)
        mixed = _FakeShardedSource({"input_ids": [1, 2, 3]}, wrapper_name="MixedShardedSource")
        assert stub._source_is_pretokenized(mixed) is False
