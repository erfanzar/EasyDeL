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

"""Fused-MoE batch-axis placement: resolver ordering and divisible-prefix trimming.

``moe_batch_axes_follow_resolver`` names the fused-MoE ``shard_map`` batch axes in the order the runtime
resolver pins the residual stream (``HiddenStateSharding``). With dp and fsdp both > 1 the default
``(dp, fsdp)`` order maps each global row to a different device than the resolver's ``(fsdp, dp)``, so
GSPMD moves whole shards with collective-permutes at the block boundary; the reordered names remove them
without changing any value.

Small inference batches (a one-request decode step) that cannot divide the full batch group keep the
longest divisible prefix of the batch axes instead of replicating over all of them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from spectrax import apply_logical_sharding, common_types

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
DP2_FSDP2 = (1, 2, 2, 1, 1, 1)
SEQ, HIDDEN = 4, 32


def _require_four_devices():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices for dp2 x fsdp2")


def _build(dims, *, follow_resolver: bool = False, seed: int = 123):
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        moe_method="fused_moe",
        moe_force_xla_gmm=True,
        sharding_axis_dims=dims,
        sharding_axis_names=AXIS_NAMES,
        scan_layers=False,
    )
    config.add_basic_configurations(
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_batch_axes_follow_resolver=follow_resolver,
    )
    return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _input(batch: int):
    return jax.random.normal(jax.random.PRNGKey(0), (batch, SEQ, HIDDEN), dtype=jnp.float32)


def _pinned_forward(block):
    """The decoder layer's view: the residual pinned to ``HiddenStateSharding`` around the block."""
    resolver = block.config.runtime_sharding_resolver

    def fwd(x):
        x = apply_logical_sharding(x, dynamic_axes=common_types.HiddenStateSharding, partition_manager=resolver)
        out, _ = block(x)
        return apply_logical_sharding(out, dynamic_axes=common_types.HiddenStateSharding, partition_manager=resolver)

    return fwd


def _collective_permutes(block, x) -> int:
    with block.config.mesh:
        hlo = jax.jit(_pinned_forward(block)).lower(x).compile().as_text()
    return sum(1 for line in hlo.splitlines() if " collective-permute(" in line or " collective-permute-start(" in line)


def _values_and_grads(block, x):
    fwd = _pinned_forward(block)

    def loss_x(xin):
        return jnp.sum(fwd(xin) ** 2)

    def loss_module(module):
        out, _ = module(x)
        return jnp.sum(out**2)

    with block.config.mesh:
        out = np.asarray(jax.device_get(jax.jit(fwd)(x)))
        gx = np.asarray(jax.device_get(jax.jit(jax.grad(loss_x))(x)))
        grads = spx.grad(loss_module)(block)
    gw = np.asarray(jax.device_get(grads["parameters"]["experts"]["gate_proj"]["weight"]))
    return out, gx, gw


def test_resolver_order_matches_residual_batch_spec():
    _require_four_devices()
    block = _build(DP2_FSDP2)
    spec = block.config.runtime_sharding_resolver.resolve(
        dynamic_axes=common_types.HiddenStateSharding, shape=(8, SEQ, HIDDEN)
    )
    assert tuple(spec[0]) == ("fsdp", "dp"), f"fixture assumes the residual batch is (fsdp, dp), got {spec}"


def test_resolver_order_removes_boundary_permutes_and_keeps_values():
    _require_four_devices()
    x = _input(8)
    default = _build(DP2_FSDP2)
    ordered = _build(DP2_FSDP2, follow_resolver=True)
    assert default.config.moe_batch_axes_follow_resolver is False
    assert ordered.config.moe_batch_axes_follow_resolver is True

    n_default, n_ordered = _collective_permutes(default, x), _collective_permutes(ordered, x)
    print(f"collective-permutes: default={n_default} resolver-order={n_ordered}")
    assert n_default > 0, f"fixture inert: default order compiled {n_default} collective-permutes"
    assert n_ordered == 0, n_ordered

    out_d, gx_d, gw_d = _values_and_grads(default, x)
    out_o, gx_o, gw_o = _values_and_grads(ordered, x)
    assert np.isfinite(out_o).all() and np.isfinite(gw_o).all()
    np.testing.assert_allclose(out_o, out_d, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gx_o, gx_d, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gw_o, gw_d, rtol=1e-6, atol=1e-6)


def test_dp1_mesh_has_no_boundary_permutes_in_either_order():
    """Control: with one dp shard both orders name the same device for every row."""
    _require_four_devices()
    x = _input(8)
    for follow in (False, True):
        block = _build((1, 1, 4, 1, 1, 1), follow_resolver=follow)
        assert _collective_permutes(block, x) == 0, follow


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("follow", [False, True])
def test_small_inference_batch_keeps_divisible_prefix_and_matches_reference(batch, follow):
    from easydel.layers.moe._communication_utils import MoeFusedHooks
    from easydel.utils.inference_mode import set_inference_mode

    _require_four_devices()
    block = _build(DP2_FSDP2, follow_resolver=follow)
    x_full = _input(8)
    with block.config.mesh, set_inference_mode():
        reference = np.asarray(jax.device_get(block(x_full)[0]))

    seen: list[tuple[int, ...]] = []

    def spy(inputs_2d, weights, inputs_shape):
        seen.append(tuple(inputs_2d.shape))
        return inputs_2d

    block.moe_hooks = MoeFusedHooks(refine_inputs_hook=spy)
    x = x_full[:batch]
    with block.config.mesh, set_inference_mode():
        out = np.asarray(jax.device_get(block(x)[0]))

    assert out.shape == (batch, SEQ, HIDDEN)
    np.testing.assert_allclose(out, reference[:batch], rtol=1e-5, atol=1e-5)
    # batch 2 divides dp=2 (fsdp dropped): one row per shard; batch 1 divides nothing: replicated.
    expected_rows = (batch // 2 if batch % 2 == 0 else batch) * SEQ
    assert set(seen) == {(expected_rows, HIDDEN)}, seen


def test_small_training_batch_still_errors():
    """Outside inference an indivisible batch keeps the loud shard_map error."""
    _require_four_devices()
    block = _build(DP2_FSDP2)
    with block.config.mesh, pytest.raises(Exception):  # noqa: B017
        block(_input(2))
