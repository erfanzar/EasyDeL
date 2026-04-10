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

"""Tests for eSurge configuration validation."""

import pytest

from easydel.inference.esurge.config import (
    LONG_PREFILL_TRS,
    CacheConfig,
    SchedulerConfig,
)


class TestSchedulerConfig:
    def test_valid_config(self):
        config = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=8192,
        )
        assert config.max_num_seqs == 16
        assert config.policy == "fcfs"

    def test_max_num_seqs_zero_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs must be positive"):
            SchedulerConfig(max_num_seqs=0, max_num_batched_tokens=2048, max_model_len=8192)

    def test_max_num_seqs_negative_raises(self):
        with pytest.raises(ValueError, match="max_num_seqs must be positive"):
            SchedulerConfig(max_num_seqs=-1, max_num_batched_tokens=2048, max_model_len=8192)

    def test_max_num_batched_tokens_zero_raises(self):
        with pytest.raises(ValueError, match="max_num_batched_tokens must be positive"):
            SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=0, max_model_len=8192)

    def test_max_num_batched_tokens_none_allowed(self):
        config = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=None, max_model_len=8192)
        assert config.max_num_batched_tokens is None

    def test_max_model_len_zero_raises(self):
        with pytest.raises(ValueError, match="max_model_len must be positive"):
            SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=2048, max_model_len=0)

    def test_batched_tokens_exceeds_model_len_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=16384, max_model_len=8192)

    def test_long_prefill_defaults_to_batched_tokens(self):
        config = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=8192)
        assert config.long_prefill_token_threshold == 4096

    def test_long_prefill_defaults_to_constant_when_no_batched(self):
        config = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=None, max_model_len=8192)
        assert config.long_prefill_token_threshold == LONG_PREFILL_TRS

    def test_long_prefill_negative_raises(self):
        with pytest.raises(ValueError, match="long_prefill_token_threshold"):
            SchedulerConfig(
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                max_model_len=8192,
                long_prefill_token_threshold=-1,
            )

    def test_token_safety_margin_negative_raises(self):
        with pytest.raises(ValueError, match="token_safety_margin"):
            SchedulerConfig(
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                max_model_len=8192,
                token_safety_margin=-1,
            )

    def test_token_safety_margin_zero_allowed(self):
        config = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=8192,
            token_safety_margin=0,
        )
        assert config.token_safety_margin == 0

    def test_empty_buckets_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SchedulerConfig(
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                max_model_len=8192,
                max_num_seq_buckets=(),
            )

    def test_negative_bucket_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            SchedulerConfig(
                max_num_seqs=16,
                max_num_batched_tokens=2048,
                max_model_len=8192,
                max_num_seq_buckets=(8, 16, -1),
            )

    def test_valid_buckets(self):
        config = SchedulerConfig(
            max_num_seqs=64,
            max_num_batched_tokens=2048,
            max_model_len=8192,
            max_num_seq_buckets=(8, 16, 32, 64),
        )
        assert config.max_num_seq_buckets == (8, 16, 32, 64)

    def test_priority_policy(self):
        config = SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=8192,
            policy="priority",
        )
        assert config.policy == "priority"

    def test_async_scheduling_default(self):
        config = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=2048, max_model_len=8192)
        assert config.async_scheduling is True


class TestCacheConfig:
    def test_valid_config(self):
        config = CacheConfig(num_pages=1000, page_size=128, enable_prefix_caching=True)
        assert config.num_pages == 1000
        assert config.page_size == 128

    def test_page_size_zero_raises(self):
        with pytest.raises(ValueError, match="page_size must be positive"):
            CacheConfig(num_pages=1000, page_size=0, enable_prefix_caching=True)

    def test_page_size_negative_raises(self):
        with pytest.raises(ValueError, match="page_size must be positive"):
            CacheConfig(num_pages=1000, page_size=-1, enable_prefix_caching=True)

    def test_num_pages_none_allowed(self):
        config = CacheConfig(num_pages=None, page_size=128, enable_prefix_caching=True)
        assert config.num_pages is None

    def test_num_pages_zero_raises(self):
        with pytest.raises(ValueError, match="num_pages must be positive"):
            CacheConfig(num_pages=0, page_size=128, enable_prefix_caching=True)

    def test_num_pages_negative_raises(self):
        with pytest.raises(ValueError, match="num_pages must be positive"):
            CacheConfig(num_pages=-1, page_size=128, enable_prefix_caching=True)

    def test_prefix_caching_disabled(self):
        config = CacheConfig(num_pages=1000, page_size=128, enable_prefix_caching=False)
        assert config.enable_prefix_caching is False
