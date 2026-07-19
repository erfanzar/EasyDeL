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

"""GQA head-grouping parity for BlockSparseAttn (splash) under shard_map.

Regression test for the production bug where a mesh whose query-head shard
factor does not divide the KV head count (e.g. 16 query heads / 2 KV heads
with tp=4) silently replicated the KV-head axis while the query heads stayed
sharded. The kernel's in-shard blocked GQA regrouping
(``local_h // (local_q_heads / local_kv_heads)``) then attends half of the
query heads to the wrong KV head — garbage output at every sequence length.
This corrupted the Qwen3.6-35B-A3B (16q/2kv, head_dim 256) teacher/student
forward on every SPLASH step of the 131k distillation runs.

The failure reproduces on CPU because the grouping happens at the
shard_map/spec level, not in the Pallas kernel: the XLA blocksparse impl uses
the same blocked regrouping.

Run:
    ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      uv run pytest libs/easydel/tests/operations/kernels/test_blocksparse_gqa_sharding.py
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr

from easydel.infra import EasyDeLBaseConfig
from easydel.operations._operation_impl import OperationMetadata
from easydel.operations.kernels.blocksparse_attention import BlockSparseAttn

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 8,
    reason="requires 8 (fake) devices: set XLA_FLAGS=--xla_force_host_platform_device_count=8",
)


def _reference_attention_bthd(q, k, v, softmax_scale, causal=True):
    """f32 dense reference with blocked GQA grouping (q head h -> kv head h // reps)."""
    qf = q.astype(jnp.float32).transpose(0, 2, 1, 3)
    kf = k.astype(jnp.float32).transpose(0, 2, 1, 3)
    vf = v.astype(jnp.float32).transpose(0, 2, 1, 3)
    reps = qf.shape[1] // kf.shape[1]
    kf = jnp.repeat(kf, reps, axis=1)
    vf = jnp.repeat(vf, reps, axis=1)
    logits = jnp.einsum("bhtd,bhkd->bhtk", qf * softmax_scale, kf)
    if causal:
        t, s = logits.shape[-2], logits.shape[-1]
        logits = jnp.where(jnp.tril(jnp.ones((t, s), dtype=bool)), logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhtk,bhkd->bhtd", weights, vf)
    return out.transpose(0, 2, 1, 3)


def _run_adapter(q_heads: int, kv_heads: int, axis_dims: tuple[int, ...], seq: int = 512, head_dim: int = 64):
    config = EasyDeLBaseConfig(sharding_axis_dims=axis_dims)
    metadata = OperationMetadata(runtime_dtype=jnp.bfloat16, base_config=config)
    op = BlockSparseAttn(metadata)

    kq, kk, kv = jr.split(jr.PRNGKey(0), 3)
    q = jr.normal(kq, (1, seq, q_heads, head_dim), dtype=jnp.bfloat16)
    k = jr.normal(kk, (1, seq, kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jr.normal(kv, (1, seq, kv_heads, head_dim), dtype=jnp.bfloat16)

    with metadata.mesh:
        out = op(query=q, key=k, value=v, causal=True).attention_outputs
    ref = _reference_attention_bthd(q, k, v, softmax_scale=head_dim**-0.5, causal=True)

    err = np.abs(np.asarray(out, dtype=np.float32) - np.asarray(ref, dtype=np.float32))
    return err.max(), err.mean()


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "axis_dims", "tag"),
    [
        # Production shape class: tp=4 does NOT divide 2 KV heads (the bug trigger).
        (16, 2, (1, 1, 2, 1, 4, 1), "gqa16:2-tp4"),
        # tp divides KV heads: was already correct; must stay correct.
        (16, 2, (1, 1, 4, 1, 2, 1), "gqa16:2-tp2"),
        # MHA control.
        (8, 8, (1, 1, 2, 1, 4, 1), "mha8:8-tp4"),
        # Single-device control.
        (16, 2, (1, 1, 1, 1, 1, 1), "gqa16:2-1dev"),
    ],
)
def test_blocksparse_gqa_grouping_parity(q_heads, kv_heads, axis_dims, tag):
    max_abs, mean_abs = _run_adapter(q_heads, kv_heads, axis_dims)
    # bf16 kernel vs f32 reference: worst observed healthy error is ~0.02 max /
    # ~4e-4 mean; the grouping bug produces ~3.4 max / ~0.08 mean.
    assert max_abs < 0.1, f"[{tag}] max abs error {max_abs} — wrong GQA head grouping under shard_map?"
    assert mean_abs < 5e-3, f"[{tag}] mean abs error {mean_abs} — wrong GQA head grouping under shard_map?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
