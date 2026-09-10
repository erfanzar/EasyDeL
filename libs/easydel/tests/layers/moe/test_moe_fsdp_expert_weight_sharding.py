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

"""FSDP-sharded MoE expert weights with gather-at-use (``moe_fsdp_shard_expert_weights``).

On ep=1 training meshes the ``(EP, None, TP)`` at-rest expert-weight layout
degenerates to TP-only sharding: a v5p-2048 memory probe (dp4/fsdp64/ep1/tp4,
qwen3.6-35B-A3B student + teacher) measured 30.06G of live arrays at
``EasyDeLState.init_tx`` entry, dominated by 80x bf16 (256, 2048, 1024)
gate_up shards (268 MB each) and 80x bf16 (256, 512, 2048) down shards
(134 MB each) — ~15 GB/model of expert weights resident per device. Optimizer
init (f32 mu + f32 nu over those params) then failed allocation.

``moe_fsdp_shard_expert_weights=True`` (ZeRO-3 style) shards the expert
(leading) dimension over fsdp at rest — ``P(('ep','fsdp'), None, 'tp')`` —
and all-gathers the weights over fsdp inside the fused shard_map right before
use. The all_gather's transpose is a psum_scatter over fsdp, so expert-weight
gradients leave the shard_map already sharded like the parameters, and
``init_tx``'s param-sharding mirroring shards the optimizer state the same
way. These tests pin:

* the at-rest sharding of expert kernels/biases includes fsdp when the flag
  is on, and is unchanged when off (default no-op),
* forward outputs and expert-weight gradients match the flag-off dispatch on
  the ep=1 all-to-all-free mesh and the ring fsdp2/ep2/tp2 mesh,
* gradients keep the fsdp sharding (they land pre-scattered),
* optimizer states created via ``EasyDeLState.init_tx`` mirror the fsdp
  sharding,
* non-divisible expert counts fall back safely (no wrong math, no crash).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")
TOL = 1e-5


def _require_fake_mesh():
    """Skip unless the fake 8-device CPU host mesh is available."""
    if jax.device_count() < 8:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")


def _make_config(*, sharding_axis_dims, ring: bool, fsdp_shard: bool, num_experts: int = 8):
    """Build a GptOssConfig with the fused MoE + the flag under test."""
    from easydel.modules.gpt_oss.gpt_oss_configuration import GptOssConfig

    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=64,
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
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_fsdp_shard_expert_weights=fsdp_shard,
    )
    return config


def _build_gptoss_block(*, sharding_axis_dims, ring: bool, fsdp_shard: bool, num_experts: int = 8, seed: int = 123):
    """Instantiate a GptOssMLP under the parameter-init sharding context.

    Bare layer construction bypasses the EasyDeL base-module ``__init__``
    wrapper, so the context (which installs the ambient
    ``moe_expert_param_layout_scope`` and parameter placement) is entered
    explicitly — the same path every real model constructor takes.
    """
    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssMLP

    config = _make_config(
        sharding_axis_dims=sharding_axis_dims, ring=ring, fsdp_shard=fsdp_shard, num_experts=num_experts
    )
    with _parameter_init_sharding_context(config):
        return GptOssMLP(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _input(hidden_size: int = 32):
    """Deterministic (8, 4, hidden) test batch shared by all meshes."""
    return jax.random.normal(jax.random.PRNGKey(0), (8, 4, hidden_size), dtype=jnp.float32)


def _dim0_axes(array) -> tuple[str, ...]:
    """Return the mesh axes sharding dim 0 of ``array`` as a flat tuple."""
    spec = getattr(array.sharding, "spec", None)
    if spec is None or len(spec) == 0 or spec[0] is None:
        return ()
    entry = spec[0]
    return tuple(entry) if isinstance(entry, tuple) else (entry,)


def test_rest_sharding_includes_fsdp_only_when_enabled():
    """Expert kernels/biases at rest: P(('ep','fsdp'),...) on, P('ep',...) off."""
    _require_fake_mesh()

    on = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=True)
    off = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=False)

    for name in ("gate_proj", "up_proj", "down_proj"):
        w_on = getattr(on.experts, name).weight.value
        w_off = getattr(off.experts, name).weight.value
        assert "fsdp" in _dim0_axes(w_on), (
            f"{name}.weight must shard its expert dim over fsdp when "
            f"moe_fsdp_shard_expert_weights=True, got {w_on.sharding}"
        )
        assert "fsdp" not in _dim0_axes(w_off), (
            f"{name}.weight must keep the historical layout when the flag is off, got {w_off.sharding}"
        )
        b_on = getattr(on.experts, name).bias.value
        assert "fsdp" in _dim0_axes(b_on), f"{name}.bias must follow the expert-dim sharding, got {b_on.sharding}"


@pytest.mark.parametrize(
    ("candidate_dims", "ring"),
    [
        pytest.param((1, 1, 4, 1, 2, 1), False, id="noring-fsdp4-tp2-ep1"),
        pytest.param((1, 1, 2, 2, 2, 1), True, id="ring-fsdp2-ep2-tp2"),
    ],
)
def test_gather_at_use_matches_replicated_weights(candidate_dims, ring):
    """Flag-on forward and expert-weight grads must match flag-off exactly.

    The gather reconstructs the identical ep-local expert block from the
    fsdp shards, so the per-shard math is unchanged; the gradient path adds
    the all_gather transpose (psum_scatter over fsdp), which must reproduce
    the replicated-weight gradient values while landing them fsdp-sharded.
    """
    _require_fake_mesh()

    off = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring, fsdp_shard=False)
    on = _build_gptoss_block(sharding_axis_dims=candidate_dims, ring=ring, fsdp_shard=True)
    x = _input()

    with off.config.mesh:
        out_off, _ = off(x)
    with on.config.mesh:
        out_on, _ = on(x)

    assert out_on.shape == out_off.shape == x.shape
    fwd_diff = float(jnp.max(jnp.abs(out_on - out_off)))
    assert fwd_diff < TOL, f"gather-at-use forward diverges on {candidate_dims}: {fwd_diff:.3e}"

    def loss(module):
        out, _ = module(x)
        return jnp.sum(out * out)

    with off.config.mesh:
        grads_off = spx.grad(loss)(off)
    with on.config.mesh:
        grads_on = spx.grad(loss)(on)

    for name in ("gate_proj", "up_proj", "down_proj"):
        g_on = grads_on["parameters"]["experts"][name]["weight"]
        g_off = grads_off["parameters"]["experts"][name]["weight"]
        grad_diff = float(jnp.max(jnp.abs(g_on - g_off)))
        assert grad_diff < TOL, f"{name} weight grad diverges on {candidate_dims}: {grad_diff:.3e}"
        assert "fsdp" in _dim0_axes(g_on), (
            f"{name} weight grad must leave the shard_map fsdp-sharded (psum_scatter transpose), got {g_on.sharding}"
        )


def test_flag_off_is_bit_identical_to_knob_absent():
    """Flag-off must equal a config that predates the knob, bitwise."""
    _require_fake_mesh()

    explicit_off = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=False)
    legacy = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=False)
    delattr(legacy.config, "moe_fsdp_shard_expert_weights")
    x = _input()

    with explicit_off.config.mesh:
        out_off, _ = explicit_off(x)
    with legacy.config.mesh:
        out_legacy, _ = legacy(x)

    assert bool(jnp.array_equal(out_off, out_legacy))


def test_non_divisible_expert_count_falls_back_safely():
    """E=6 with ep*fsdp=4: the runtime gather must disable itself, math intact.

    6 % 4 != 0, so the weight in-specs cannot carry (ep, fsdp); the module
    logs a warning and keeps the fsdp-replicated in-specs (any at-rest fsdp
    sharding is gathered at the shard_map boundary instead). Output must
    match the flag-off block bit-for-bit wherever specs sanitize away.
    """
    _require_fake_mesh()

    off = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=False, num_experts=6)
    on = _build_gptoss_block(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=True, num_experts=6)
    x = _input()

    with off.config.mesh:
        out_off, _ = off(x)
    with on.config.mesh:
        out_on, _ = on(x)

    assert out_on.shape == out_off.shape
    diff = float(jnp.max(jnp.abs(out_on - out_off)))
    assert diff < TOL, f"non-divisible fallback changed the math: {diff:.3e}"


def test_optimizer_state_mirrors_fsdp_sharding():
    """EasyDeLState.init_tx must shard mu/nu like the fsdp-sharded params.

    This is the production payoff: on the v5p mesh the optimizer state over
    15 GB/model of fsdp-replicated expert weights is what failed allocation;
    mirroring the ('ep','fsdp') param sharding divides it by the fsdp world
    size.
    """
    import optax
    from easydel.infra.base_state import EasyDeLState
    from easydel.modules.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    _require_fake_mesh()

    config = _make_config(sharding_axis_dims=(1, 1, 4, 1, 2, 1), ring=False, fsdp_shard=True)
    with config.mesh:
        model = GptOssForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))

    weight = model.model.layers[0].mlp.experts.gate_proj.weight.value
    assert "fsdp" in _dim0_axes(weight), f"model-level init must fsdp-shard expert weights, got {weight.sharding}"

    state = EasyDeLState.create(model=model).init_tx(optax.adam(1e-3))
    found = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        key = jax.tree_util.keystr(path)
        if "gate_proj" in key and "weight" in key and hasattr(leaf, "sharding"):
            found += 1
            assert "fsdp" in _dim0_axes(leaf), f"optimizer slot {key} lost the fsdp sharding: {leaf.sharding}"
    assert found >= 2, "expected at least adam mu and nu slots for the gate_proj weight"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
