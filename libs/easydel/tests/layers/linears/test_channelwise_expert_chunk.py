# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""``channelwise_quantize_array(expert_chunk=...)``: chunking over the expert axis is value-identical to one step.

Computation test (TPU): experts are independent along axis 0, so codes and scales must match bit for bit.
"""

import numpy as np
import pytest
from easydel.layers.linears._linear_quantized import channelwise_quantize_array
from jax import numpy as jnp


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("chunk", [1, 3, 5, 7, 64])  # 7 does not divide 7 experts evenly with 3/5; 64 > experts
def test_expert_chunking_matches_unchunked(bits, chunk):
    rng = np.random.default_rng(0)
    kernel = jnp.asarray(rng.standard_normal((7, 64, 48)).astype(np.float32) * 3.0, dtype=jnp.bfloat16)
    codes, scales, _ = channelwise_quantize_array(kernel, bits)
    c_codes, c_scales, _ = channelwise_quantize_array(kernel, bits, expert_chunk=chunk)
    assert c_codes.dtype == codes.dtype and c_codes.shape == codes.shape
    assert c_scales.shape == (7, 1, 48)
    np.testing.assert_array_equal(np.asarray(c_codes.astype(jnp.int8)), np.asarray(codes.astype(jnp.int8)))
    np.testing.assert_array_equal(np.asarray(c_scales), np.asarray(scales))


def test_chunk_is_ignored_for_dense_2d_kernels():
    kernel = jnp.asarray(np.random.default_rng(1).standard_normal((64, 48)).astype(np.float32))
    a = channelwise_quantize_array(kernel, 8)
    b = channelwise_quantize_array(kernel, 8, expert_chunk=2)
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
