# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""ExpertTensor MoE block: the inference combine (local unpermute + psum) matches the training combine
(ragged all-to-all) and an independent dense per-expert reference, single-pass and chunk-scanned.

TPU only (four devices): XLA:CPU has no ragged-all-to-all lowering for the training-path reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpSparseMoeBlock
from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig
from easydel.utils.inference_mode import set_inference_mode

TP = 4
EXPERTS = 8
TOP_K = 2
HIDDEN = 32
INTER = 32


def _build(chunk):
    config = Qwen4ExpTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=16,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, 1, 1, TP, 1))
    block = Qwen4ExpSparseMoeBlock(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(5)
    )
    for name in ("gate_up_proj", "down_proj"):
        projection = getattr(block.experts, name)
        projection.weight.value = projection.weight.value * 5
    block.config.use_expert_tensor_mode = True
    block.config.moe_chunk_size = chunk
    return block


def _routed(block, hidden):
    return block.moe_call(
        hidden_state=hidden,
        gate_layer=block.gate,
        expert_layer=block.experts,
        gate_up_kernel=block.experts.gate_up_proj.kernel_view(),
        wd_kernel=block.experts.down_proj.kernel_view(),
        act_fn=block.experts.act_fn,
    )[0]


def _reference(block, hidden):
    """Dense per-expert SwiGLU with softmax top-k routing (renormalized), in float64."""
    x = np.asarray(hidden, np.float64).reshape(-1, HIDDEN)
    logits = np.asarray(block.gate(jnp.asarray(x, jnp.float32)), np.float64)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    selected = np.argsort(-probs, axis=-1)[:, :TOP_K]
    weights = np.take_along_axis(probs, selected, axis=-1)
    weights /= weights.sum(-1, keepdims=True)
    gate_up = np.asarray(block.experts.gate_up_proj.kernel_view(), np.float64)  # [E, H, 2I]
    down = np.asarray(block.experts.down_proj.kernel_view(), np.float64)  # [E, I, H]
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for slot in range(TOP_K):
            e = selected[t, slot]
            gu = x[t] @ gate_up[e]
            gate, up = gu[:INTER], gu[INTER:]
            out[t] += weights[t, slot] * ((gate / (1.0 + np.exp(-gate)) * up) @ down[e])
    return out.reshape(np.asarray(hidden).shape)


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() != TP, reason="four TPU devices required")
@pytest.mark.parametrize("chunk", [None, 4])
def test_expert_tensor_inference_combine_matches_training_combine_and_reference(chunk):
    block = _build(chunk)
    hidden = jnp.asarray(np.random.default_rng(11).normal(size=(1, 8, HIDDEN)) * 3, jnp.float32)
    with spx.use_mesh(block.config.mesh):
        training = np.asarray(jax.jit(lambda x: _routed(block, x))(hidden))
        with set_inference_mode():
            inference = np.asarray(jax.jit(lambda x: _routed(block, x))(hidden))
    reference = _reference(block, hidden)
    assert float(np.linalg.norm(reference)) > 0.1
    assert np.isfinite(inference).all()
    np.testing.assert_allclose(training, reference, rtol=2e-4, atol=2e-4, err_msg="training combine")
    np.testing.assert_allclose(inference, reference, rtol=2e-4, atol=2e-4, err_msg="inference combine")
    np.testing.assert_allclose(inference, training, rtol=1e-5, atol=1e-5)


def _dense_reference(block, hidden):
    """Differentiable dense per-expert SwiGLU with softmax top-k routing (renormalized), at HIGHEST precision."""
    hp = jax.lax.Precision.HIGHEST
    x = hidden.reshape(-1, HIDDEN)
    probs = jax.nn.softmax(block.gate(x).astype(jnp.float32), axis=-1)
    weights, selected = jax.lax.top_k(probs, TOP_K)
    weights = weights / weights.sum(-1, keepdims=True)
    gate_up = block.experts.gate_up_proj.kernel_view()
    down = block.experts.down_proj.kernel_view()
    projected = jnp.einsum("th,ehi->tei", x, gate_up, precision=hp)
    expert_out = jnp.einsum(
        "tei,eih->teh", jax.nn.silu(projected[..., :INTER]) * projected[..., INTER:], down, precision=hp
    )
    dense = jnp.zeros((x.shape[0], EXPERTS), jnp.float32).at[jnp.arange(x.shape[0])[:, None], selected].set(weights)
    return jnp.einsum("teh,te->th", expert_out, dense, precision=hp).reshape(hidden.shape)


@pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() != TP, reason="four TPU devices required")
@pytest.mark.parametrize("chunk", [None, 4])
def test_expert_tensor_psum_combine_gradients_match_ragged_combine_and_reference(chunk):
    """Autodiff through the psum combine gives the same input, router and expert-weight gradients."""
    block = _build(chunk)
    rng = np.random.default_rng(11)
    hidden = jnp.asarray(rng.normal(size=(1, 8, HIDDEN)) * 3, jnp.float32)
    cotangent = jnp.asarray(rng.normal(size=hidden.shape), jnp.float32)
    graphdef, state = spx.export(block)

    def grads(fn):
        def loss(params, x):
            return jnp.sum(fn(spx.bind(graphdef, params), x) * cotangent)

        return jax.jit(jax.grad(loss, argnums=(0, 1)))(state, hidden)

    with spx.use_mesh(block.config.mesh):
        ragged = grads(_routed)
        with set_inference_mode():
            psum = grads(_routed)
        reference = grads(_dense_reference)

    def leaves(tree):
        return {
            jax.tree_util.keystr(path): np.asarray(value)
            for path, value in jax.tree_util.tree_leaves_with_path(tree)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) and value.size
        }

    ref_leaves, ragged_leaves, psum_leaves = leaves(reference), leaves(ragged), leaves(psum)
    checked = 0
    for key, ref in ref_leaves.items():
        scale = float(np.abs(ref).max())
        if scale == 0.0:
            continue  # the shared expert is outside moe_call: zero gradient on every path
        checked += 1
        np.testing.assert_allclose(psum_leaves[key], ref, rtol=0, atol=2e-5 * scale, err_msg=f"psum {key}")
        np.testing.assert_allclose(ragged_leaves[key], ref, rtol=0, atol=2e-5 * scale, err_msg=f"ragged {key}")
    # hidden-state gradient + router + expert gate_up + expert down
    assert checked >= 4, sorted(ref_leaves)
