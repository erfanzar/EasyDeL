# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Regression tests: recurrent-state dtype is a ``HybridCacheConfig`` field (no env knob)."""

import jax.numpy as jnp
import pytest
from easydel.caching import HybridCacheConfig
from spectrax import PartitionAxis

pytestmark = pytest.mark.cpu_ok


def _config(**kwargs):
    return HybridCacheConfig.create(
        num_hidden_layers=2,
        partition_axis=PartitionAxis(),
        batch_size=1,
        sequence_length=8,
        num_key_value_heads=2,
        head_dim=4,
        d_inner=8,
        d_conv=2,
        d_state=4,
        layer_types=("linear_attention", "full_attention"),
        **kwargs,
    )


def test_recurrent_state_dtype_defaults_to_float32():
    assert _config().recurrent_state_dtype is None


def test_recurrent_state_dtype_accepts_dtype_objects_names_and_cache_sentinel():
    assert _config(recurrent_state_dtype=jnp.bfloat16).recurrent_state_dtype == "bfloat16"
    assert _config(recurrent_state_dtype="float16").recurrent_state_dtype == "float16"
    assert _config(recurrent_state_dtype="cache").recurrent_state_dtype == "cache"
