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

"""Comprehensive tests for elarge processing utilities."""

import jax
from jax import numpy as jnp

from easydel.infra.elarge.processing import coerce_dtype, coerce_precision


class TestCoerceDtype:
    def test_none_returns_float32(self):
        assert coerce_dtype(None) == jnp.float32

    def test_jnp_dtype_passthrough(self):
        assert coerce_dtype(jnp.float16) == jnp.float16
        assert coerce_dtype(jnp.bfloat16) == jnp.bfloat16
        assert coerce_dtype(jnp.float32) == jnp.float32

    def test_bf16_abbreviation(self):
        assert coerce_dtype("bf16") == jnp.bfloat16

    def test_bfloat16_full(self):
        assert coerce_dtype("bfloat16") == jnp.bfloat16

    def test_fp16(self):
        assert coerce_dtype("fp16") == jnp.float16

    def test_float16(self):
        assert coerce_dtype("float16") == jnp.float16

    def test_f16(self):
        assert coerce_dtype("f16") == jnp.float16

    def test_fp32(self):
        assert coerce_dtype("fp32") == jnp.float32

    def test_float32(self):
        assert coerce_dtype("float32") == jnp.float32

    def test_f32(self):
        assert coerce_dtype("f32") == jnp.float32

    def test_fp64(self):
        assert coerce_dtype("fp64") == jnp.float64

    def test_float64(self):
        assert coerce_dtype("float64") == jnp.float64

    def test_fp8_e4m3(self):
        assert coerce_dtype("fp8_e4m3") == jnp.float8_e4m3

    def test_nvfp8(self):
        assert coerce_dtype("nvfp8") == jnp.float8_e4m3

    def test_mxfp8(self):
        assert coerce_dtype("mxfp8") == jnp.float8_e5m2

    def test_unknown_returns_float32(self):
        assert coerce_dtype("unknown_dtype") == jnp.float32

    def test_case_insensitive(self):
        assert coerce_dtype("BF16") == jnp.bfloat16
        assert coerce_dtype("FP16") == jnp.float16


class TestCoercePrecision:
    def test_none_returns_none(self):
        assert coerce_precision(None) is None

    def test_precision_passthrough(self):
        assert coerce_precision(jax.lax.Precision.HIGH) == jax.lax.Precision.HIGH

    def test_default_string(self):
        assert coerce_precision("DEFAULT") == jax.lax.Precision.DEFAULT

    def test_high_string(self):
        assert coerce_precision("HIGH") == jax.lax.Precision.HIGH

    def test_highest_string(self):
        assert coerce_precision("HIGHEST") == jax.lax.Precision.HIGHEST

    def test_case_insensitive(self):
        assert coerce_precision("high") == jax.lax.Precision.HIGH
        assert coerce_precision("High") == jax.lax.Precision.HIGH

    def test_unknown_returns_default(self):
        assert coerce_precision("unknown") == jax.lax.Precision.DEFAULT
