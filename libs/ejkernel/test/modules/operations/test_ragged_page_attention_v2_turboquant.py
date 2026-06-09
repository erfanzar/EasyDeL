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

"""Operation-level tests for ragged_page_attention_v2_turboquant."""

from __future__ import annotations

import jax.numpy as jnp

from ejkernel import modules
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    RaggedPageAttentionv2TurboQuant,
    RaggedPageAttentionv2TurboQuantConfig,
    ragged_page_attention_v2_turboquant,
)


def _make_inputs():
    head_dim = 8
    qjl_dim = 8

    return dict(
        queries=jnp.arange(16, dtype=jnp.bfloat16).reshape(2, 1, head_dim) / 10,
        key_indices_pages=jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8),
        key_signs_pages=jnp.zeros((1, 2, 1, qjl_dim // 8), dtype=jnp.uint8),
        key_norms_pages=jnp.zeros((1, 2, 1, 2), dtype=jnp.bfloat16),
        value_indices_pages=jnp.zeros((1, 2, 1, head_dim // 2), dtype=jnp.uint8),
        value_norms_pages=jnp.zeros((1, 2, 1), dtype=jnp.bfloat16),
        context_lens=jnp.array([2], dtype=jnp.int32),
        block_tables=jnp.array([[0]], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 2], dtype=jnp.int32),
        num_seqs=1,
        rotation_matrix=jnp.eye(head_dim, dtype=jnp.float32),
        qjl_projection=jnp.eye(qjl_dim, head_dim, dtype=jnp.float32),
        key_codebook=jnp.linspace(-1.0, 1.0, 2 ** (4 - 1), dtype=jnp.float32),
        value_codebook=jnp.linspace(-1.0, 1.0, 2**4, dtype=jnp.float32),
        qjl_dim=qjl_dim,
    )


def test_operation_is_exported_from_modules_init_files():
    """Verify TurboQuant v2 symbols are exported from public module packages."""
    assert hasattr(operations, "ragged_page_attention_v2_turboquant")
    assert hasattr(operations, "RaggedPageAttentionv2TurboQuant")
    assert hasattr(operations, "RaggedPageAttentionv2TurboQuantConfig")

    assert hasattr(modules, "ragged_page_attention_v2_turboquant")
    assert hasattr(modules, "RaggedPageAttentionv2TurboQuant")
    assert hasattr(modules, "RaggedPageAttentionv2TurboQuantConfig")


def test_ragged_page_attention_v2_turboquant_functional_api_runs_xla():
    """Verify the public functional API runs on XLA."""
    inputs = _make_inputs()
    output = ragged_page_attention_v2_turboquant(
        inputs["queries"],
        inputs["key_indices_pages"],
        inputs["key_signs_pages"],
        inputs["key_norms_pages"],
        inputs["value_indices_pages"],
        inputs["value_norms_pages"],
        inputs["context_lens"],
        inputs["block_tables"],
        inputs["query_start_loc"],
        inputs["num_seqs"],
        inputs["rotation_matrix"],
        inputs["qjl_projection"],
        inputs["key_codebook"],
        inputs["value_codebook"],
        qjl_dim=inputs["qjl_dim"],
        platform="xla",
    )

    assert output.shape == inputs["queries"].shape
    assert output.dtype == inputs["queries"].dtype
    assert jnp.isfinite(output).all()


def test_ragged_page_attention_v2_turboquant_class_api_runs_xla():
    """Verify the Kernel.run API runs on XLA."""
    inputs = _make_inputs()
    op = RaggedPageAttentionv2TurboQuant()
    cfg = RaggedPageAttentionv2TurboQuantConfig(
        platform="xla",
        num_kv_pages_per_block=1,
        num_queries_per_block=8,
    )

    output = op.run(
        inputs["queries"],
        inputs["key_indices_pages"],
        inputs["key_signs_pages"],
        inputs["key_norms_pages"],
        inputs["value_indices_pages"],
        inputs["value_norms_pages"],
        inputs["context_lens"],
        inputs["block_tables"],
        inputs["query_start_loc"],
        inputs["num_seqs"],
        inputs["rotation_matrix"],
        inputs["qjl_projection"],
        inputs["key_codebook"],
        inputs["value_codebook"],
        qjl_dim=inputs["qjl_dim"],
        cfg=cfg,
    )

    assert output.shape == inputs["queries"].shape
    assert output.dtype == inputs["queries"].dtype
    assert jnp.isfinite(output).all()
