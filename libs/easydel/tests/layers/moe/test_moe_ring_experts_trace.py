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

"""Regression: fused MoE ring-of-experts dispatch must trace under jit.

The ring-of-experts branch of ``BaseMoeModule._sparse_moe_call`` used to slice
the permuted token buffer down to the (data-dependent) number of tokens routed
to a shard's local experts::

    valid_token_count = jnp.sum(group_sizes)          # a tracer
    x = jax.lax.dynamic_slice_in_dim(x, 0, valid_token_count, axis=0)

The *size* argument of ``dynamic_slice_in_dim`` must be a static Python int, so
under jit this raised ``TracerIntegerConversionError`` at trace time and the
whole ``use_ring_of_experts=True`` path was unusable.

The fix drops the slice entirely and keeps the full static-length buffer: the
grouped matmul (``ragged_dot_general``) writes zeros for every row past
``sum(group_sizes)``, so the non-local tail contributes exactly zero and the
per-shard result is unchanged before the ``psum_scatter`` expert combine. These
tests assert (a) the ring path now traces and runs with finite output, (b) it
matches the default all-to-all path numerically on a single-shard expert mesh,
and (c) the default ``use_ring_of_experts=False`` path is unaffected.

Note: multi-shard ring dispatch uses ``all_gather``/``psum_scatter`` (never
``ragged_all_to_all``), so unlike the ep>1 all-to-all path it lowers on
XLA:CPU. Runtime numerics across many real expert shards still want TPU, but the
trace-time crash this file guards against is fully provable on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _build_block(*, sharding_axis_dims, use_ring_of_experts: bool, num_experts: int = 8, seed: int = 0):
    """Instantiate a Qwen3MoeSparseBlock (fused MoE) under the given mesh shape."""
    from easydel.modules.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseBlock
    from easydel.modules.qwen3_moe.qwen3_moe_configuration import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_method="fused_moe",
        # Force the XLA (ragged_dot) grouped matmul so the test never depends on
        # Pallas TPU kernel tiling constraints.
        moe_force_xla_gmm=True,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(use_ring_of_experts=use_ring_of_experts)
    return Qwen3MoeSparseBlock(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(seed),
    )


def test_ring_of_experts_traces_and_runs():
    """The ring-of-experts forward must trace (no TracerIntegerConversionError) and be finite.

    ``ep`` fills all visible devices; the expert mesh folds fsdp/sp into ep, so this
    exercises the multi-shard ring dispatch (all_gather + psum_scatter). ``num_experts``
    is 8 so it divides any device count that divides 8 (CI runs 8 fake CPU devices).
    """
    block = _build_block(sharding_axis_dims=(1, 1, 1, -1, 1, 1), use_ring_of_experts=True, num_experts=8)
    x = jnp.ones((2, 4, 32), dtype=jnp.float32)

    with block.config.mesh:
        out, router_logits = block(x)

    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out))), "ring-of-experts output has non-finite values"
    assert bool(jnp.all(jnp.isfinite(router_logits)))


def test_nonring_traces_and_runs():
    """Default all-to-all path (use_ring_of_experts=False) is unaffected and stays finite.

    Uses a single-shard expert mesh (dp fills all devices, ep=fsdp=sp=1) so the expert
    mesh has ep_size==1 and the ep>1 ragged_all_to_all (unsupported on XLA:CPU) is not
    reached.
    """
    dc = jax.device_count()
    block = _build_block(sharding_axis_dims=(1, -1, 1, 1, 1, 1), use_ring_of_experts=False)
    x = jnp.ones((dc, 4, 32), dtype=jnp.float32)

    with block.config.mesh:
        out, _ = block(x)

    assert out.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_ring_matches_nonring_on_single_shard():
    """On a single-shard expert mesh, ring dispatch must equal the all-to-all dispatch.

    With ep_size==1 the all_gather/psum_scatter collectives are no-ops, so the ring path
    and the default path perform the same permute -> grouped_matmul -> unpermute over an
    identical token set. With identical weights (same rng seed) and inputs the outputs
    must match bit-for-bit. This is the numerics-preservation guarantee of the fix: the
    removed dynamic slice was purely an (illegal) optimization, and ragged_dot's zeroed
    tail reproduces the same values.
    """
    dc = jax.device_count()
    single = (1, -1, 1, 1, 1, 1)  # dp fills devices -> expert mesh ep_size == 1

    ring = _build_block(sharding_axis_dims=single, use_ring_of_experts=True, seed=123)
    nonring = _build_block(sharding_axis_dims=single, use_ring_of_experts=False, seed=123)

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (dc, 4, 32), dtype=jnp.float32)

    with ring.config.mesh:
        out_ring, _ = ring(x)
    with nonring.config.mesh:
        out_nonring, _ = nonring(x)

    max_abs_diff = float(jnp.max(jnp.abs(out_ring - out_nonring)))
    assert max_abs_diff == 0.0, f"ring vs non-ring diverged on single-shard mesh: max_abs_diff={max_abs_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
