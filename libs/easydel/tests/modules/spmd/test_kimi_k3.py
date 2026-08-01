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

"""Kimi K3 text-backbone building blocks.

K3 reuses the Kimi-Linear tower (MLA full-attention layers interleaved with KDA
linear-attention layers) and adds three things on top: the SITU gated
activation, LatentMoE routed experts, and Attention Residuals. These tests cover
the first two plus the config surface; the numbers they assert come from the
published ``moonshotai/Kimi-K3`` ``config.json``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.kimi_linear import KimiLinearConfig
from easydel.modules.kimi_linear.modeling_kimi_linear import (
    KimiSparseMoeBlock,
    resolve_gated_activation,
    situ_and_mul,
)
from jax import numpy as jnp

# Layer geometry of the released 2.8T checkpoint.
K3_NUM_LAYERS = 93
K3_FULL_ATTENTION_LAYERS = 24
K3_ATTN_RES_BLOCK = 12
K3_SITU_BETA = 4.0
K3_SITU_LINEAR_BETA = 25.0


def _k3_like_config(**overrides) -> KimiLinearConfig:
    """A small config carrying K3's distinguishing fields."""
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=K3_SITU_BETA,
        activation_situ_linear_beta=K3_SITU_LINEAR_BETA,
        attn_res_block_size=K3_ATTN_RES_BLOCK,
        # The fused path dispatches through `ragged-all-to-all`, which XLA:CPU
        # cannot lower, so these blocks would only be executable on accelerators.
        # The dense path exercises the same routing and expert widths.
        moe_method="dense_moe",
    )
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


def test_full_attention_layers_alone_determine_the_hybrid_split():
    """K3 ships only ``full_attn_layers``; the KDA complement must be derived.

    The published config lists 24 one-indexed full-attention layers out of 93
    and no ``kda_layers`` key at all, so a config that requires both cannot load
    the real checkpoint.
    """
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=K3_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        linear_attn_config={
            "full_attn_layers": [*range(4, 93, 4), 93],
            "head_dim": 128,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
        },
    )
    layer_types = list(config.layer_types)

    assert len(layer_types) == K3_NUM_LAYERS
    assert layer_types.count("full_attention") == K3_FULL_ATTENTION_LAYERS
    assert layer_types.count("kda_linear_attention") == K3_NUM_LAYERS - K3_FULL_ATTENTION_LAYERS
    # One-indexed: layer 4 in the config is index 3 here.
    assert layer_types[3] == "full_attention"
    assert layer_types[0] == "kda_linear_attention"


@pytest.mark.parametrize("linear_beta", [K3_SITU_LINEAR_BETA, None])
def test_situ_matches_its_reference_definition(linear_beta):
    """SITU is ``beta*tanh(g/beta)*sigmoid(g)`` times an optionally squashed ``up``."""
    rng = np.random.default_rng(0)
    gate = rng.normal(size=(4, 16)).astype(np.float32)
    up = rng.normal(size=(4, 16)).astype(np.float32)

    expected = K3_SITU_BETA * np.tanh(gate / K3_SITU_BETA) * (1.0 / (1.0 + np.exp(-gate)))
    expected = expected * (linear_beta * np.tanh(up / linear_beta) if linear_beta is not None else up)

    got = np.asarray(situ_and_mul(jnp.asarray(gate), jnp.asarray(up), beta=K3_SITU_BETA, linear_beta=linear_beta))

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_situ_transforms_the_linear_branch_too():
    """A unary activation composed with a plain multiply cannot express SITU.

    ``linear_beta`` squashes ``up`` as well as gating it, so dropping that term
    silently changes the layer's output -- guard the distinction.
    """
    gate = jnp.linspace(-3.0, 3.0, 32).reshape(2, 16)
    up = jnp.full((2, 16), 50.0)  # large enough that the tanh squash bites

    with_squash = situ_and_mul(gate, up, beta=K3_SITU_BETA, linear_beta=K3_SITU_LINEAR_BETA)
    without_squash = situ_and_mul(gate, up, beta=K3_SITU_BETA, linear_beta=None)

    assert not np.allclose(np.asarray(with_squash), np.asarray(without_squash))


def test_resolve_gated_activation_falls_back_to_plain_gating():
    """Non-SITU activations keep the usual ``act_fn(gate) * up`` behaviour."""
    config = _k3_like_config(hidden_act="silu")
    activation = resolve_gated_activation(config)
    gate = jnp.linspace(-2.0, 2.0, 16).reshape(2, 8)
    up = jnp.ones((2, 8))

    np.testing.assert_allclose(np.asarray(activation(gate, up)), np.asarray(jax.nn.silu(gate) * up), rtol=1e-6)


def test_latent_moe_runs_experts_in_the_latent_width():
    """Routed experts live at ``routed_expert_hidden_size``, the router does not.

    K3 routes on the full 7168-wide hidden state while its experts operate in a
    3584-wide latent space, so the expert kernels must be built at the narrow
    width and the gate at the wide one.
    """
    config = _k3_like_config()
    with config.mesh:
        block = KimiSparseMoeBlock(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        hidden = jax.random.normal(jax.random.PRNGKey(0), (1, 4, config.hidden_size), dtype=jnp.float32)
        out, router_logits = block(hidden)

    assert block.use_latent_moe
    assert block.moe_hidden_size == config.routed_expert_hidden_size
    # Experts consume the latent width; the router still sees the full width.
    assert block.experts.gate_up_proj.weight.value.shape[1] == config.routed_expert_hidden_size
    assert block.gate.weight.value.shape[0] == config.hidden_size
    assert out.shape == hidden.shape
    assert bool(jnp.all(jnp.isfinite(out)))
    # The block returns per-selected-expert routing weights, so the trailing
    # axis is top-k rather than the full expert count.
    assert router_logits.shape[-1] == config.num_experts_per_token


def test_latent_moe_is_off_without_the_expert_width():
    """Without ``routed_expert_hidden_size`` the block stays full-width."""
    config = _k3_like_config(routed_expert_hidden_size=None, latent_moe_use_norm=False)
    with config.mesh:
        block = KimiSparseMoeBlock(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        out, _ = block(jnp.ones((1, 4, config.hidden_size), dtype=jnp.float32))

    assert not block.use_latent_moe
    assert block.moe_hidden_size == config.hidden_size
    assert block.experts.gate_up_proj.weight.value.shape[1] == config.hidden_size
    assert out.shape == (1, 4, config.hidden_size)
