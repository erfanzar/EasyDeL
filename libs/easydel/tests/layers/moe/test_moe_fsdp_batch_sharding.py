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

"""Regression: fused MoE must shard the batch over fsdp on the SPMD stage mesh.

On SPMD meshes (pp=1, fsdp not ep-bound) ``_sparse_moe_call`` runs its
``shard_map`` over the 5-D stage mesh, where dp and fsdp are separate axes.
The in/out specs historically listed only ``dp`` on the batch dimension, so on
dp=1 / fsdp-dominant training meshes (e.g. v5p-2048 ``(1, 1, 64, 4, 4, 1)``)
the entire global batch was replicated into every shard. The permuted dispatch
buffer then materializes ``[B*S*k, H]`` per core — 137 GB at B=512, S=8192,
k=8, H=2048 — and TPU compilation fails the int32 allocation bound
(``jellyfish/bounds_check.cc: allocation_size_words <= int32_max``).

The folded 3-D expert mesh (taken when pp>1 or fsdp/sp are ep-bound with
size > 1) was never affected: ``_build_active_expert_mesh`` multiplies fsdp
into its dp axis before the specs are built.

A second latent bug is covered here as well: the ring-of-experts branch
reshaped its combined output with the *global* hidden size ``HD``. Under
tp>1 the earlier TP ``psum_scatter`` leaves each row at ``HD // tp`` features,
so the reshape misfolded hidden chunks into the batch dimension right before
the expert-axis combine, producing a wrong global output shape (observed
``(4, 4, 64)`` instead of ``(8, 4, 32)``) on any mesh with ring + tp>1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _require_fake_mesh():
    """Skip unless the fake 8-device CPU host mesh is available."""
    if jax.device_count() < 8:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def _build_gptoss_block(
    *,
    sharding_axis_dims,
    ring: bool,
    fsdp_is_ep_bound: bool = False,
    num_experts: int = 8,
    seed: int = 123,
):
    """Instantiate a GptOssMLP (fused MoE) under a mesh.

    Mirrors the scaffold of ``test_moe_ring_experts_bias``. ``ring`` selects
    ring-of-experts (the ep>1 variant that lowers on XLA:CPU; the non-ring
    ep>1 path uses ``jax.lax.ragged_all_to_all``, unsupported by the CPU
    thunk emitter). ``fsdp_is_ep_bound=False`` keeps fsdp as a batch axis so
    SPMD meshes take the 5-D stage-mesh path, matching production training
    configs where batch parallelism lives on fsdp.

    Args:
        sharding_axis_dims: 6-tuple of mesh dims in ``(pp,dp,fsdp,ep,tp,sp)``
            order.
        ring: Whether to enable ring-of-experts.
        fsdp_is_ep_bound: Whether fsdp folds into expert parallelism (True
            forces the folded 3-D expert-mesh path when fsdp > 1).
        num_experts: Number of routed experts.
        seed: Parameter init seed; identical seeds give identical weights
            across meshes.

    Returns:
        A ``GptOssMLP`` module built under the requested mesh.
    """
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
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        use_ring_of_experts=ring,
        fsdp_is_ep_bound=fsdp_is_ep_bound,
        sp_is_ep_bound=False,
    )
    return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _input(hidden_size: int = 32):
    """Deterministic (8, 4, hidden) test batch shared by all meshes."""
    return jax.random.normal(jax.random.PRNGKey(0), (8, 4, hidden_size), dtype=jnp.float32)


def test_stage_mesh_batch_axes_include_fsdp():
    """On the 5-D stage mesh the batch spec must name (dp, fsdp); folded mesh stays dp-only."""
    from easydel.layers.moe._communication_utils import DP, resolve_eformer_axis

    _require_fake_mesh()

    stage = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ring=True, fsdp_is_ep_bound=False)
    with stage.config.mesh:
        expert_mesh = stage._active_auto_expert_mesh(arr=None)
        assert set(expert_mesh.jax_mesh.axis_names) == {"dp", "fsdp", "ep", "tp", "sp"}, (
            f"expected the 5-D SPMD stage mesh, got {expert_mesh.jax_mesh.axis_names}"
        )
        dp_axis_name = resolve_eformer_axis(DP, stage.runtime_sharding_resolver)
        batch_axes = stage._moe_batch_axis_names(expert_mesh, dp_axis_name)
    assert batch_axes == ("dp", "fsdp"), (
        f"stage-mesh fused-MoE batch axes must be ('dp', 'fsdp'), got {batch_axes}: without fsdp the "
        "global batch is replicated into every shard and the [B*S*k, H] dispatch buffer overflows "
        "the TPU int32 allocation bound at production shapes"
    )

    folded = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ring=True, fsdp_is_ep_bound=True)
    with folded.config.mesh:
        folded_mesh = folded._active_auto_expert_mesh(arr=None)
        assert tuple(folded_mesh.jax_mesh.axis_names) == ("dp", "ep", "tp")
        dp_axis_name = resolve_eformer_axis(DP, folded.runtime_sharding_resolver)
        folded_axes = folded._moe_batch_axis_names(folded_mesh, dp_axis_name)
    assert folded_axes == ("dp",), (
        f"folded expert mesh already absorbs fsdp into its dp axis; batch axes must stay ('dp',), got {folded_axes}"
    )


@pytest.mark.parametrize(
    ("candidate_dims", "ring"),
    [
        pytest.param((1, 1, 4, 1, 2, 1), False, id="noring-fsdp4-tp2"),
        pytest.param((1, 1, 2, 2, 2, 1), True, id="ring-fsdp2-ep2-tp2"),
    ],
)
def test_fused_moe_local_batch_sharded_over_fsdp(candidate_dims, ring):
    """The per-shard token count inside the fused shard_map must scale as B/(dp*fsdp).

    ``permute`` invokes ``refine_inputs_hook`` on the flattened local tokens
    inside the shard_map body, so the hook observes the true per-shard shape
    at trace time. In ring mode the hook runs after the expert-axis
    all-gather, so the expected local row count carries an extra ep factor.
    """
    from easydel.layers.moe._communication_utils import MoeFusedHooks

    _require_fake_mesh()

    block = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring)
    seen: list[tuple[int, ...]] = []

    def spy(inputs_2d, weights, inputs_shape):
        seen.append(tuple(inputs_2d.shape))
        return inputs_2d

    block.moe_hooks = MoeFusedHooks(refine_inputs_hook=spy)
    x = _input()

    with block.config.mesh:
        out, _ = block(x)

    _pp, dp, fsdp, ep, _tp, _sp = candidate_dims
    batch, seq, hidden = x.shape
    expected_rows = (batch // (dp * fsdp)) * seq
    if ring:
        expected_rows *= ep  # hook runs after the ring all-gather over ep
    assert seen, "refine_inputs_hook was never invoked inside the fused MoE shard_map"
    assert set(seen) == {(expected_rows, hidden)}, (
        f"per-shard fused-MoE token buffer is {set(seen)}, expected {{({expected_rows}, {hidden})}}: "
        "the batch dimension is not sharded over (dp, fsdp) inside the shard_map, so dispatch "
        "buffers scale with the global batch"
    )
    assert out.shape == x.shape


@pytest.mark.parametrize(
    ("candidate_dims", "ring"),
    [
        pytest.param((1, 1, 4, 1, 2, 1), False, id="noring-fsdp4-tp2"),
        pytest.param((1, 1, 2, 2, 2, 1), True, id="ring-fsdp2-ep2-tp2"),
        pytest.param((1, 1, 2, 2, 1, 2), True, id="ring-fsdp2-ep2-sp2-tp1"),
    ],
)
def test_fsdp_batch_sharding_matches_dp_reference(candidate_dims, ring):
    """dp=8 reference vs fsdp-carried batch: identical weights, identical outputs.

    Batch parallelism is pure sharding: moving it from dp to fsdp must not
    change the math or the global output shape. The reference mesh (dp=all)
    exercises the historical known-good path. Non-ring ep>1 cannot run on
    XLA:CPU (``ragged_all_to_all`` is unsupported by the CPU thunk emitter),
    so the non-ring case covers ep=1 with fsdp>1/tp>1 and ring covers ep>1
    with tp in {1, 2}; the sp>1 case locks sp's replicated batch behavior.
    """
    _require_fake_mesh()

    ref = _build_gptoss_block(sharding_axis_dims=(1, -1, 1, 1, 1, 1), ring=ring)
    shd = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring)
    x = _input()

    with ref.config.mesh:
        out_ref, _ = ref(x)
    with shd.config.mesh:
        out_shd, _ = shd(x)

    assert out_ref.shape == x.shape
    assert out_shd.shape == out_ref.shape, f"global output shape drifted: {out_shd.shape} vs {out_ref.shape}"
    assert bool(jnp.all(jnp.isfinite(out_ref)))
    assert bool(jnp.all(jnp.isfinite(out_shd)))
    max_abs_diff = float(jnp.max(jnp.abs(out_ref - out_shd)))
    assert max_abs_diff < 1e-3, (
        f"fsdp-carried batch diverges from dp reference on {candidate_dims}: max_abs_diff={max_abs_diff:.3e}"
    )


def test_ring_tp_output_shape_on_folded_mesh():
    """Ring + tp>1 on the folded expert mesh must keep the global output shape.

    Regression for the ring-branch combine reshape, which used the global
    hidden size ``HD`` on rows already scattered to ``HD // tp`` features by
    the TP ``psum_scatter``; with tp=2 the misfold returned ``(4, 4, 64)``
    for an ``(8, 4, 32)`` input.
    """
    _require_fake_mesh()

    ref = _build_gptoss_block(sharding_axis_dims=(1, -1, 1, 1, 1, 1), ring=True)
    # fsdp_is_ep_bound=True folds fsdp into ep: 3-D expert mesh (dp=1, ep=4, tp=2).
    folded = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ring=True, fsdp_is_ep_bound=True)
    x = _input()

    with ref.config.mesh:
        out_ref, _ = ref(x)
    with folded.config.mesh:
        out_folded, _ = folded(x)

    assert out_folded.shape == out_ref.shape, (
        f"ring+tp>1 combine misfolded hidden into batch: {out_folded.shape} vs {out_ref.shape}"
    )
    max_abs_diff = float(jnp.max(jnp.abs(out_ref - out_folded)))
    assert max_abs_diff < 1e-3, f"ring+tp>1 folded-mesh output diverges: max_abs_diff={max_abs_diff:.3e}"


def test_fsdp_batch_grad_matches_dp_reference():
    """Gradients through the fused shard_map match across dp- and fsdp-carried batches."""
    _require_fake_mesh()

    ref = _build_gptoss_block(sharding_axis_dims=(1, -1, 1, 1, 1, 1), ring=True)
    shd = _build_gptoss_block(sharding_axis_dims=(1, 1, 2, 2, 2, 1), ring=True)
    x = _input()

    def loss(block, xin):
        out, _ = block(xin)
        return jnp.sum(out * out)

    with ref.config.mesh:
        grad_ref = jax.grad(lambda xin: loss(ref, xin))(x)
    with shd.config.mesh:
        grad_shd = jax.grad(lambda xin: loss(shd, xin))(x)

    assert grad_shd.shape == grad_ref.shape
    assert bool(jnp.all(jnp.isfinite(grad_ref)))
    assert bool(jnp.all(jnp.isfinite(grad_shd)))
    max_abs_diff = float(jnp.max(jnp.abs(grad_ref - grad_shd)))
    assert max_abs_diff < 1e-4, f"fsdp-carried batch gradient diverges from dp reference: max_abs_diff={max_abs_diff:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
