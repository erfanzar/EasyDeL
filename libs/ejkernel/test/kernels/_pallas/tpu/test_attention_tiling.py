# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Forward/VJP regressions for the attention kernels' tiled softmax statistics."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.blocksparse_attention import blocksparse_attention
from ejkernel.kernels._pallas.tpu.flash_attention import flash_attention
from ejkernel.kernels._pallas.tpu.flash_mla import flash_mla
from ejkernel.kernels._xla.blocksparse_attention import blocksparse_attention as blocksparse_attention_xla
from ejkernel.kernels._xla.flash_attention import flash_attention as flash_attention_xla
from ejkernel.kernels._xla.flash_mla import flash_mla as flash_mla_xla


@pytest.mark.parametrize("operation", ["blocksparse_attention", "flash_attention", "flash_mla"])
def test_tiled_attention_forward_and_vjp_match_xla(operation):
    """Exercise multiple KV tiles and 256-wide outputs, including both backward kernels.

    Softmax statistics are broadcast across KV tiles; normalization and segment
    masks must hold in both the primal and its VJP.
    """
    keys = jax.random.split(jax.random.PRNGKey(2026), 5)
    seq_len, head_dim = 256, 256
    if operation == "flash_mla":
        kernel, reference = flash_mla, flash_mla_xla
        shapes = [(1, seq_len, 1, 128), (1, seq_len, 128), (128, 1, 128), (128, 1, head_dim)]
        inputs = tuple(
            jax.random.normal(key, shape, dtype=jnp.float32) for key, shape in zip(keys[:4], shapes, strict=True)
        )
        inputs = (*inputs[:2], inputs[2] / 128**0.5, inputs[3] / 128**0.5)
        kwargs = {"causal": True}
    else:
        if operation == "blocksparse_attention":
            kernel, reference = blocksparse_attention, blocksparse_attention_xla
            shape = (1, 1, seq_len, head_dim)
        else:
            kernel, reference = flash_attention, flash_attention_xla
            shape = (1, seq_len, 1, head_dim)
        inputs = tuple(jax.random.normal(key, shape, dtype=jnp.float32) for key in keys[:3])
        # A document boundary inside a tile exercises non-uniform mask metadata.
        segments = (jnp.arange(seq_len)[None, :] >= 96).astype(jnp.int32)
        kwargs = {"causal": True, "q_segment_ids": segments, "kv_segment_ids": segments}

    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(partial(kernel, **kwargs), *inputs)
        expected, reference_pullback = jax.vjp(partial(reference, **kwargs), *inputs)
        cotangent = jax.random.normal(keys[-1], output.shape, dtype=jnp.float32)
        gradients = pullback(cotangent)
        expected_gradients = reference_pullback(cotangent)

    assert output.shape == expected.shape
    assert output.dtype == inputs[0].dtype
    assert bool(jnp.isfinite(output).all())
    np.testing.assert_allclose(output, expected, rtol=1e-2, atol=1e-2)
    for gradient, expected_gradient, primal in zip(gradients, expected_gradients, inputs, strict=True):
        assert gradient.shape == primal.shape
        assert gradient.dtype == primal.dtype
        assert bool(jnp.isfinite(gradient).all())
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-2, atol=1e-2)
