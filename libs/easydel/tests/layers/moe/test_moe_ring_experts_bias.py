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

"""Regression: fused MoE ring-of-experts multi-shard output must be correct.

The ring-of-experts path (``BaseMoeModule._sparse_moe_call`` with
``use_ring_of_experts=True``) keeps the full ``tokens*k`` permuted buffer and
relies on the non-local *tail* rows being exactly zero: the ``unpermute`` +
expert-axis ``psum_scatter`` combine sums each row across expert shards, so only
the shard that OWNS a token's expert may carry a nonzero value.

Two ways the tail was NOT zero, both fixed by explicitly zeroing it in the ring
branch:

1. The path assumed the down grouped_matmul zeros rows past ``sum(local
   group_sizes)``. XLA:CPU honors that; **TPU does not** — the tail holds
   garbage, so the multi-shard output was ~373x too large (with NaNs at small
   magnitudes) even with zero bias. This is a TPU-only bug; the CPU fake-device
   mesh masks it because ragged_dot zeros the tail there.
2. ``wd_bias`` (down-proj bias) is added *after* the matmul — unlike wi/wu bias,
   wiped by the matmul's tail-zeroing — and its clamped out-of-bounds gather
   tainted the tail, so every token accumulated ~``(ep-1)`` bogus bias terms.

GPT-OSS is the only stacked-expert family carrying a ``wd_bias``. The ring path
uses ``all_gather``/``psum_scatter`` (never ``ragged_all_to_all``), so it lowers
on XLA:CPU and the nonzero-bias contamination is provable on the fake-device CPU
mesh; the garbage-tail (#1) reproduces on real TPU (see the isolation probe).
This test compares ep=1 (correct reference) vs ep=all-devices with a nonzero
bias, which fails on both platforms pre-fix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _build_gptoss_block(*, sharding_axis_dims, num_experts: int = 8, seed: int = 0):
    """Instantiate a GptOssMLP (fused MoE, carries a down-proj bias) under a mesh."""
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=num_experts,
        num_experts_per_tok=2,
        # Force the XLA (ragged_dot) grouped matmul so the test never depends on
        # Pallas TPU kernel tiling constraints.
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(use_ring_of_experts=True)
    return GptOssMLP(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(seed),
    )


def _set_nonzero_down_bias(block, num_experts: int, hidden_size: int) -> None:
    """Overwrite the down-proj bias with a fixed nonzero array.

    ``use_bias=True`` initialises the bias to zeros, which would make the tail
    contamination invisible (0 * anything == 0). A deterministic nonzero bias
    (identical across the compared blocks) is what surfaces the bug.
    """
    bias = jax.random.normal(jax.random.PRNGKey(7), (num_experts, hidden_size), dtype=jnp.float32)
    block.experts.down_proj.bias.value = bias


def test_ring_down_bias_matches_across_expert_sharding():
    """ep=1 (reference) and ep=all-devices ring outputs must agree.

    Expert parallelism is a pure sharding axis: with identical weights and input
    the math is independent of how experts are split across devices, so ep=1 and
    ep=N must produce the same per-token output. At ep=1 ``experts_per_shard ==
    n_routed_experts`` so the fix's local-row mask is all-True (a no-op) and the
    expert-axis combine is a no-op too — that output is the correct reference.

    Pre-fix the multi-shard ring path corrupts the output via a nonzero tail:
    on TPU the grouped matmul leaves garbage in the tail (diverges O(1) even at
    zero bias), and on any backend the after-matmul ``wd_bias`` gather taints the
    tail (~(ep-1) spurious bias terms/token, ~6.8 at this unit-scale bias on the
    8-fake-device CPU mesh). The explicit tail-zero fix drops it to a small
    cross-sharding residual (~1.6e-2 CPU / ~1.5e-5 TPU here — fp/bf16-ULP scale,
    not a correctness leak). The ``1e-1`` tolerance sits an order of magnitude
    above that residual and well below the contamination, failing loudly on any
    real leak. Runs on the fake-8-device CPU mesh and on a real >=2-chip TPU.
    """
    dc = jax.device_count()
    if dc < 2:
        pytest.skip("needs >=2 devices for a multi-shard expert mesh")
    if 8 % dc != 0:
        pytest.skip(f"num_experts=8 must be divisible by ep={dc}")

    num_experts, hidden_size = 8, 32

    ref = _build_gptoss_block(sharding_axis_dims=(1, -1, 1, 1, 1, 1), num_experts=num_experts, seed=123)
    shd = _build_gptoss_block(sharding_axis_dims=(1, 1, 1, -1, 1, 1), num_experts=num_experts, seed=123)
    _set_nonzero_down_bias(ref, num_experts, hidden_size)
    _set_nonzero_down_bias(shd, num_experts, hidden_size)

    # B divisible by dp (=dc on the ep=1 reference mesh).
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 4, hidden_size), dtype=jnp.float32)

    with ref.config.mesh:
        out_ref, _ = ref(x)
    with shd.config.mesh:
        out_shd, _ = shd(x)

    assert bool(jnp.all(jnp.isfinite(out_ref)))
    assert bool(jnp.all(jnp.isfinite(out_shd)))
    max_abs_diff = float(jnp.max(jnp.abs(out_ref - out_shd)))
    assert max_abs_diff < 1e-1, (
        "ring-of-experts multi-shard output diverges from the ep=1 reference: "
        f"ep=1 vs ep={dc} max_abs_diff={max_abs_diff:.3e} (expected < 1e-1). The "
        "non-local tail rows are nonzero (TPU garbage and/or clamped wd_bias), and "
        "the expert-axis psum_scatter sums them into every token."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
