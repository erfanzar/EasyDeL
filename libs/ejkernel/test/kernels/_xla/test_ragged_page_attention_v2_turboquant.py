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

"""Smoke tests for XLA ragged_page_attention_v2_turboquant."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ejkernel.kernels import Platform, kernel_registry
from ejkernel.kernels._xla import ragged_page_attention_v2_turboquant


def test_registry_and_export_for_ragged_page_attention_v2_turboquant():
    impl = kernel_registry.get("ragged_page_attention_v2_turboquant", platform=Platform.XLA)
    assert callable(impl)
    assert callable(ragged_page_attention_v2_turboquant)


@pytest.mark.parametrize("num_seqs", [1, jnp.array([1], dtype=jnp.int32)])
def test_ragged_page_attention_v2_turboquant_smoke_run(num_seqs):
    head_dim = 8
    qjl_dim = 8

    queries = jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, head_dim) / 10

    key_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    key_signs_pages = jnp.zeros((1, 2, 1, qjl_dim // 8), dtype=jnp.uint8)
    key_norms_pages = jnp.zeros((1, 2, 1, 2), dtype=jnp.bfloat16)
    value_indices_pages = jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8)
    value_norms_pages = jnp.zeros((1, 2, 1), dtype=jnp.bfloat16)

    context_lens = jnp.array([2], dtype=jnp.int32)
    block_tables = jnp.array([[0]], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 2], dtype=jnp.int32)

    rotation_matrix = jnp.eye(head_dim, dtype=jnp.float32)
    qjl_projection = jnp.eye(qjl_dim, head_dim, dtype=jnp.float32)
    key_codebook = jnp.linspace(-1.0, 1.0, 2 ** (4 - 1), dtype=jnp.float32)
    value_codebook = jnp.linspace(-1.0, 1.0, 2**4, dtype=jnp.float32)

    output = ragged_page_attention_v2_turboquant(
        queries,
        key_indices_pages,
        key_signs_pages,
        key_norms_pages,
        value_indices_pages,
        value_norms_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        rotation_matrix,
        qjl_projection,
        key_codebook,
        value_codebook,
        compute_dtype=jnp.float32,
        mask_value=-1e9,
        qjl_dim=qjl_dim,
    )

    assert output.shape == queries.shape
    assert output.dtype == queries.dtype
    assert jnp.isfinite(output).all()
