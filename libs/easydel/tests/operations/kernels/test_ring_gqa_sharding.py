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

"""GQA head-grouping parity for RingAttn under shard_map.

Sibling of ``test_blocksparse_gqa_sharding.py`` — same defect class:
a mesh whose query-head shard factor does not divide the KV head count
(16 q heads / 2 KV heads with tp=4) silently replicated the KV-head axis
while query heads stayed sharded, so the ring kernels' in-shard blocked GQA
regrouping attended query heads to the wrong KV head (silent corruption on
the Pallas TPU path). Additionally, the XLA ring reference did not support
GQA at all — it rebinds ``num_heads`` per operand and crashed with a reshape
TypeError for any ``num_kv_heads != num_q_heads``.

Pre-fix behavior on CPU: every GQA case below raises TypeError; post-fix all
cases match the dense f32 reference.

Run:
    ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
      uv run pytest libs/easydel/tests/operations/kernels/test_ring_gqa_sharding.py
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr

from easydel.infra import EasyDeLBaseConfig
from easydel.operations._operation_impl import OperationMetadata
from easydel.operations.kernels.ring_attention import RingAttn

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
    op = RingAttn(metadata)

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
        # Production shape class: tp=4 does NOT divide 2 KV heads (silent
        # corruption on Pallas; XLA reference crashed pre-fix).
        (16, 2, (1, 1, 2, 1, 4, 1), "gqa16:2-tp4"),
        # tp divides KV heads (the configuration ring was validated with).
        (16, 2, (1, 1, 4, 1, 2, 1), "gqa16:2-tp2"),
        # MHA control (worked pre-fix; must stay correct).
        (8, 8, (1, 1, 2, 1, 4, 1), "mha8:8-tp4"),
        # Single-device GQA (XLA reference crashed pre-fix).
        (16, 2, (1, 1, 1, 1, 1, 1), "gqa16:2-1dev"),
    ],
)
def test_ring_gqa_grouping_parity(q_heads, kv_heads, axis_dims, tag):
    max_abs, mean_abs = _run_adapter(q_heads, kv_heads, axis_dims)
    # bf16 kernel vs f32 reference: healthy error is ~0.007 max / ~1.5e-4 mean;
    # wrong GQA grouping produces O(1) max errors.
    assert max_abs < 0.1, f"[{tag}] max abs error {max_abs} — wrong GQA head grouping under shard_map?"
    assert mean_abs < 5e-3, f"[{tag}] mean abs error {mean_abs} — wrong GQA head grouping under shard_map?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
