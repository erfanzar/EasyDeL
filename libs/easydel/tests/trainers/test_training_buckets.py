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

import pytest
from easydel.trainers.buckets import (
    BucketRule,
    CallableBucketRule,
    CycleBucketRule,
    ModBucketRule,
    StepThresholdRule,
    TrainingBucket,
    resolve_bucket_config,
)


class _FakeConfig:
    """Minimal stand-in for EasyDeLBaseConfig with a mutable attribute."""

    def __init__(self, attn_mechanism: str = "auto"):
        self.attn_mechanism = attn_mechanism


# --------------------------------------------------------------------------- #
# BucketRule math
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Serialization round-trips
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# resolve_bucket_config
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# TrainingBucket defaults
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Structural-equality + integration (need JAX + a model backend)
# --------------------------------------------------------------------------- #


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
        assert jax.tree_util.tree_structure(gstate_base) == jax.tree_util.tree_structure(
            gstate_alt
        ), "vanilla and blocksparse variants must share graphstate structure"
