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

"""Regression: fused MoE on the folded expert mesh must use its expert-axis size.

When fsdp or sp is bound into expert parallelism (``fsdp_is_ep_bound`` /
``sp_is_ep_bound``, both default True) with size > 1, ``_sparse_moe_call`` runs
its ``shard_map`` on the folded ``(dp, ep, tp)`` mesh whose expert axis spans
``ep * fsdp * sp`` devices. The body used the model mesh's physical ``ep`` size
instead, so each shard received ``E / (ep*fsdp*sp)`` experts while grouping
tokens for ``E / ep``. The grouped matmul then rejected the shapes
(``expected rhs group dimension size to be 8, got 2``) on every such layout,
including the default fsdp-bound training mesh.

The non-ring path dispatches with ``ragged_all_to_all``, which XLA:CPU cannot
lower, so those cases require an accelerator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _block(dims, *, fsdp_is_ep_bound, sp_is_ep_bound, ring=False):
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = GptOssConfig(
        hidden_size=32,
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
        use_ring_of_experts=ring, fsdp_is_ep_bound=fsdp_is_ep_bound, sp_is_ep_bound=sp_is_ep_bound
    )
    return GptOssMLP(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(123),
    )


def _output_and_gradients(block, x):
    graphdef, state = spx.export(block)
    weights = jnp.linspace(-1.0, 1.0, x.size, dtype=jnp.float32).reshape(x.shape)

    def loss(x, state):
        out, _ = spx.bind(graphdef, state)(x)
        return jnp.sum(out * weights), out

    with block.config.mesh:
        (_, out), (dx, dstate) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(x, state)
    return jax.device_get((out, dx, dstate.flatten()))


@pytest.mark.parametrize(
    ("dims", "plain_dims", "fsdp_bound", "sp_bound", "ring"),
    [
        pytest.param((1, 1, 4, 1, 1, 1), (1, 1, 1, 4, 1, 1), True, True, False, id="fsdp4-bound"),
        pytest.param((1, 1, 2, 2, 1, 1), (1, 1, 1, 4, 1, 1), True, True, False, id="fsdp2-ep2-bound"),
        pytest.param((1, 1, 1, 2, 1, 2), (1, 1, 1, 4, 1, 1), True, True, False, id="ep2-sp2-bound"),
        pytest.param((1, 1, 1, 1, 2, 2), (1, 1, 1, 2, 2, 1), False, True, False, id="tp2-sp2-bound"),
        pytest.param((1, 1, 2, 2, 1, 1), (1, 1, 1, 4, 1, 1), True, True, True, id="ring-fsdp2-ep2-bound"),
        pytest.param((1, 1, 1, 1, 2, 2), (1, 1, 1, 2, 2, 1), False, True, True, id="ring-tp2-sp2-bound"),
    ],
)
def test_folded_expert_mesh_matches_plain_expert_parallel_layout(dims, plain_dims, fsdp_bound, sp_bound, ring):
    """A folded expert axis of size N must compute what a physical ``ep=N`` axis does.

    The plain layout places the same number of expert shards on the physical
    ``ep`` axis (with the same tp), which never takes the folded path.
    """
    if jax.device_count() < 4:
        pytest.skip("needs four devices")
    if not ring and jax.default_backend() == "cpu":
        pytest.skip("ragged_all_to_all is not implemented by XLA:CPU")
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 4, 32), dtype=jnp.float32)
    plain = _block(plain_dims, fsdp_is_ep_bound=fsdp_bound, sp_is_ep_bound=sp_bound, ring=ring)
    folded = _block(dims, fsdp_is_ep_bound=fsdp_bound, sp_is_ep_bound=sp_bound, ring=ring)
    with folded.config.mesh:
        expert_mesh = folded._active_auto_expert_mesh(arr=None)
        group = folded._expert_axis_group(expert_mesh)
    # Bound axes stay on the stage mesh and join the expert axis as a compound name (one abstract mesh per program).
    assert isinstance(group, tuple) and group[0] == "ep", f"case must use a compound expert axis, got {group}"
    assert expert_mesh.jax_mesh.abstract_mesh.axis_names == folded.config.mesh.abstract_mesh.axis_names
    with plain.config.mesh:
        plain_mesh = plain._active_auto_expert_mesh(arr=None)
    sizes = dict(expert_mesh.jax_mesh.shape)
    assert int(np.prod([sizes[name] for name in group])) == dict(plain_mesh.jax_mesh.shape)["ep"]

    want_out, want_dx, want_dstate = _output_and_gradients(plain, x)
    got_out, got_dx, got_dstate = _output_and_gradients(folded, x)
    np.testing.assert_allclose(got_out, want_out, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_dx, want_dx, rtol=1e-6, atol=1e-7)
    assert got_dstate.keys() == want_dstate.keys()
    for key, want in want_dstate.items():
        np.testing.assert_allclose(got_dstate[key], want, rtol=1e-6, atol=1e-7, err_msg=key)


@pytest.mark.parametrize("dims", [(1, 1, 4, 1, 1, 1), (1, 1, 2, 2, 1, 1)], ids=["fsdp4-bound", "fsdp2-ep2-bound"])
def test_bound_expert_axis_compiles_with_host_callback(dims):
    """Params as jit arguments + a host callback in the same program on an fsdp-bound mesh.

    The fused MoE used to run on a separate folded ``(dp, ep, tp)`` mesh; JAX rejects two abstract meshes in a program
    holding a host callback ("mesh should be the same across the entire program"). The output must also match the
    callback-free program.
    """
    if jax.device_count() < 4:
        pytest.skip("needs four devices")
    if jax.default_backend() == "cpu":
        pytest.skip("ragged_all_to_all is not implemented by XLA:CPU")
    block = _block(dims, fsdp_is_ep_bound=True, sp_is_ep_bound=True)
    graphdef, state = spx.export(block)
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 4, 32), dtype=jnp.float32)

    def run(state, x, with_callback):
        if with_callback:
            x = x + jax.pure_callback(lambda v: np.zeros_like(np.asarray(v)), jax.ShapeDtypeStruct(x.shape, x.dtype), x)
        out, _ = spx.bind(graphdef, state)(x)
        return out + x

    with block.config.mesh:
        placed = jax.device_put(x, jax.sharding.NamedSharding(block.config.mesh, jax.sharding.PartitionSpec("dp")))
        plain = np.asarray(jax.jit(run, static_argnums=2)(state, placed, False))
        with_cb = np.asarray(jax.jit(run, static_argnums=2)(state, placed, True))
    assert np.isfinite(plain).all()
    np.testing.assert_allclose(with_cb, plain, rtol=0, atol=0)
