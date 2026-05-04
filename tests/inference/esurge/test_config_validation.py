# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for eSurge sectioned configuration validation."""

from __future__ import annotations

import pytest

from spectrax.common_types import NOT_GIVEN

from easydel.inference.esurge.config import eSurgeCacheRuntimeConfig, eSurgeRuntimeConfig


class TestRuntimeConfig:
    def test_valid_config(self):
        config = eSurgeRuntimeConfig.from_dict(
            min_input_pad=4,
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=8192,
        )

        assert config.max_num_seqs == 16
        assert config.min_input_pad == 4
        assert not hasattr(config, "pipeline_inference")

    def test_max_num_seqs_zero_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs must be positive"):
            eSurgeRuntimeConfig.from_dict(max_num_seqs=0, max_num_batched_tokens=2048, max_model_len=8192)

    def test_max_num_seqs_negative_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs must be positive"):
            eSurgeRuntimeConfig.from_dict(max_num_seqs=-1, max_num_batched_tokens=2048, max_model_len=8192)

    def test_min_input_pad_zero_raises(self):
        with pytest.raises(ValueError, match="min_input_pad must be positive"):
            eSurgeRuntimeConfig.from_dict(min_input_pad=0)

    def test_min_token_pad_zero_raises_when_specified(self):
        with pytest.raises(ValueError, match="min_token_pad must be positive"):
            eSurgeRuntimeConfig.from_dict(min_token_pad=0)

    def test_max_num_batched_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_num_batched_tokens must be positive"):
            eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_num_batched_tokens=0, max_model_len=8192)

    def test_max_num_batched_tokens_none_allowed(self):
        config = eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_num_batched_tokens=None, max_model_len=8192)

        assert config.max_num_batched_tokens is None

    def test_max_num_batched_tokens_not_given_default(self):
        config = eSurgeRuntimeConfig.from_dict()

        assert config.max_num_batched_tokens is NOT_GIVEN

    def test_max_model_len_zero_raises(self):
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_num_batched_tokens=2048, max_model_len=0)

    def test_long_prefill_defaults_to_none(self):
        config = eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=8192)

        assert config.long_prefill_token_threshold is None

    def test_long_prefill_explicit_value(self):
        config = eSurgeRuntimeConfig.from_dict(
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            max_model_len=8192,
            long_prefill_token_threshold=2048,
        )

        assert config.long_prefill_token_threshold == 2048

    def test_long_prefill_negative_raises(self):
        with pytest.raises(ValueError, match="long_prefill_token_threshold"):
            eSurgeRuntimeConfig.from_dict(
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                max_model_len=8192,
                long_prefill_token_threshold=-1,
            )

    def test_valid_buckets_are_preserved(self):
        config = eSurgeRuntimeConfig.from_dict(
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            max_model_len=8192,
            max_num_seq_buckets=(8, 16, 32, 64),
        )

        assert config.max_num_seq_buckets == (8, 16, 32, 64)

    def test_async_scheduling_default(self):
        config = eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_num_batched_tokens=2048, max_model_len=8192)

        assert config.async_scheduling is True

    def test_pp_microbatch_policy_accepts_auto_disable_and_positive_ints(self):
        auto_config = eSurgeRuntimeConfig.from_dict(pp_microbatch_count="auto", pp_microbatch_size="auto")
        disabled_config = eSurgeRuntimeConfig.from_dict(pp_microbatch_count=0)
        sized_config = eSurgeRuntimeConfig.from_dict(pp_microbatch_size="4")

        assert auto_config.pp_microbatch_count == "auto"
        assert disabled_config.pp_microbatch_count is None
        assert sized_config.pp_microbatch_size == 4

    def test_pp_microbatch_policy_rejects_negative_or_ambiguous_values(self):
        with pytest.raises(ValueError, match="pp_microbatch_count"):
            eSurgeRuntimeConfig.from_dict(pp_microbatch_count=-1)
        with pytest.raises(ValueError, match="Only one of pp_microbatch_count"):
            eSurgeRuntimeConfig.from_dict(pp_microbatch_count=4, pp_microbatch_size=2)


class TestCacheRuntimeConfig:
    def test_valid_config(self):
        config = eSurgeCacheRuntimeConfig.from_dict(
            max_cache_tokens=4096,
            page_size=128,
            enable_prefix_caching=True,
        )

        assert config.max_cache_tokens == 4096
        assert config.page_size == 128

    def test_page_size_zero_raises(self):
        with pytest.raises(ValueError, match="page_size must be positive"):
            eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=4096, page_size=0, enable_prefix_caching=True)

    def test_page_size_negative_raises(self):
        with pytest.raises(ValueError, match="page_size must be positive"):
            eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=4096, page_size=-1, enable_prefix_caching=True)

    def test_max_cache_tokens_none_allowed(self):
        config = eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=None, page_size=128, enable_prefix_caching=True)

        assert config.max_cache_tokens is None

    def test_max_cache_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_cache_tokens must be positive"):
            eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=0, page_size=128, enable_prefix_caching=True)

    def test_max_cache_tokens_negative_raises(self):
        with pytest.raises(ValueError, match="max_cache_tokens must be positive"):
            eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=-1, page_size=128, enable_prefix_caching=True)

    def test_hbm_utilization_range(self):
        with pytest.raises(ValueError, match="hbm_utilization"):
            eSurgeCacheRuntimeConfig.from_dict(hbm_utilization=0.0)
        with pytest.raises(ValueError, match="hbm_utilization"):
            eSurgeCacheRuntimeConfig.from_dict(hbm_utilization=1.1)

    def test_cache_capacity_margin_range(self):
        with pytest.raises(ValueError, match="cache_capacity_margin"):
            eSurgeCacheRuntimeConfig.from_dict(cache_capacity_margin=0.0)

    def test_prefix_caching_disabled(self):
        config = eSurgeCacheRuntimeConfig.from_dict(max_cache_tokens=4096, page_size=128, enable_prefix_caching=False)

        assert config.enable_prefix_caching is False
