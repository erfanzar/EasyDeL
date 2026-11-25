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

"""Tests for GLM-5.4-style Loop Transformer conversion on Qwen3Next / Qwen3.5 (Qwen3.8-27B)."""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.utils import AttnMaskType
from easydel.modules.qwen3_next.modeling_qwen3_next import compute_loop_gate_stats, materialize_missing_loop_state


def _make_tiny_qwen3_5_text_config(
    *,
    num_hidden_layers: int = 12,
    loop_start_layer: int | None = None,
    loop_end_layer: int | None = None,
    num_loop_iterations: int = 1,
    loop_gate_type: str = "scalar",
    loop_gate_init_biases: tuple[float, ...] | None = None,
    loop_use_iteration_embedding: bool = False,
    loop_kv_cache_mode: str = "separate",
    loop_kv_sliding_window: int = 128,
    loop_kv_overlap: bool = True,
) -> ed.Qwen3_5TextConfig:
    return ed.Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_attention_separate_proj=True,
        attn_output_gate=True,
        mtp_num_hidden_layers=0,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        scan_layers=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        loop_start_layer=loop_start_layer,
        loop_end_layer=loop_end_layer,
        num_loop_iterations=num_loop_iterations,
        loop_gate_type=loop_gate_type,
        loop_gate_init_biases=loop_gate_init_biases,
        loop_use_iteration_embedding=loop_use_iteration_embedding,
        loop_kv_cache_mode=loop_kv_cache_mode,
        loop_kv_sliding_window=loop_kv_sliding_window,
        loop_kv_overlap=loop_kv_overlap,
    )


def _make_tiny_qwen3_5_vlm_config(**text_kwargs) -> ed.Qwen3_5Config:
    text_cfg = _make_tiny_qwen3_5_text_config(**text_kwargs)
    text_cfg.rope_scaling = {
        "rope_type": "default",
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    }
    vision_cfg = ed.Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=text_cfg.hidden_size,
        num_position_embeddings=64,
        deepstack_visual_indexes=[],
    )
    return ed.Qwen3_5Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        video_token_id=252,
        vision_start_token_id=253,
        vision_end_token_id=254,
        image_token_id=255,
    )


@pytest.mark.cpu_ok
def test_qwen38_27b_loop_schedule_and_validation():
    """Verify exact Qwen3.8-27B 64 -> 88 virtual schedule (Option A: [20, 44) x 2)."""
    cfg = ed.Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        loop_start_layer=20,
        loop_end_layer=44,
        num_loop_iterations=2,
    )
    assert cfg.is_looped is True
    assert cfg.num_cache_layers == 88
    assert cfg.resolved_loop_gate_init_biases == (-5.0, 5.0)

    schedule = cfg.loop_layer_schedule
    assert len(schedule) == 88
    assert schedule[:20] == tuple((i, 0) for i in range(0, 20))
    assert schedule[20:44] == tuple((i, 0) for i in range(20, 44))
    assert schedule[44:68] == tuple((i, 1) for i in range(20, 44))
    assert schedule[68:] == tuple((i, 0) for i in range(44, 64))

    eff_types = cfg.effective_layer_types
    assert eff_types is not None and len(eff_types) == 88
    assert eff_types.count("linear_attention") == 66  # 48 base + 18 looped pass-2
    assert eff_types.count("full_attention") == 22  # 16 base + 6 looped pass-2

    # Invalid loop configs raise clean errors
    with pytest.raises(ValueError, match="requires both loop_start_layer and loop_end_layer"):
        ed.Qwen3_5TextConfig(num_hidden_layers=12, num_loop_iterations=2)
    with pytest.raises(ValueError, match="Invalid loop range"):
        ed.Qwen3_5TextConfig(num_hidden_layers=12, loop_start_layer=8, loop_end_layer=4, num_loop_iterations=2)
    with pytest.raises(ValueError, match="loop_gate_type"):
        ed.Qwen3_5TextConfig(
            num_hidden_layers=12, loop_start_layer=4, loop_end_layer=8, num_loop_iterations=2, loop_gate_type="bad"
        )
    with pytest.raises(ValueError, match="loop_gate_init_biases length"):
        ed.Qwen3_5TextConfig(
            num_hidden_layers=12,
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_gate_init_biases=(-5.0,),
        )


@pytest.mark.cpu_ok
def test_materialize_missing_loop_state_preserves_trained_and_inits_missing():
    """Verify missing loop gate leaves get exact identity init while saved leaves are untouched."""
    cfg = _make_tiny_qwen3_5_vlm_config(
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-4.0, 6.0),
        loop_use_iteration_embedding=True,
    )

    class _FakeLeaf:
        def __init__(self, shape):
            self.shape = shape

    base_key = ("parameters", "model", "language_model", "layers", "0", "input_layernorm", "weight")
    g0_w = ("parameters", "model", "language_model", "loop_gates", "0", "proj", "weight")
    g0_b = ("parameters", "model", "language_model", "loop_gates", "0", "proj", "bias")
    g1_b = ("parameters", "model", "language_model", "loop_gates", "1", "proj", "bias")
    g0_n = ("parameters", "model", "language_model", "loop_gates", "0", "x_norm", "weight")
    iter_k = ("parameters", "model", "language_model", "loop_iter_embeddings")

    expected = {
        base_key: _FakeLeaf((64,)),
        g0_w: _FakeLeaf((128, 1)),
        g0_b: _FakeLeaf((1,)),
        g1_b: _FakeLeaf((1,)),
        g0_n: _FakeLeaf((64,)),
        iter_k: _FakeLeaf((2, 64)),
    }

    # Case 1: pure checkpoint (no loop leaves saved)
    state = {base_key: np.ones((64,), dtype=np.float32)}
    upgraded = materialize_missing_loop_state(cfg, dict(state), expected)
    assert set(upgraded.keys()) == set(expected.keys())
    np.testing.assert_allclose(upgraded[g0_w], 0.0)
    np.testing.assert_allclose(upgraded[g0_b], -4.0)
    np.testing.assert_allclose(upgraded[g1_b], 6.0)
    np.testing.assert_allclose(upgraded[g0_n], 0.0)
    np.testing.assert_allclose(upgraded[iter_k], 0.0)

    # Case 2: already-trained looped checkpoint preserves saved gate values
    trained = dict(upgraded)
    trained[g0_b] = np.array([0.75], dtype=np.float32)
    preserved = materialize_missing_loop_state(cfg, dict(trained), expected)
    np.testing.assert_allclose(preserved[g0_b], 0.75)


def test_loop_step0_identity_parity_and_gradients():
    """Verify step-0 function preservation and finite pooled gradients across both loop passes."""
    base_cfg = _make_tiny_qwen3_5_text_config(num_hidden_layers=12)
    exact_loop_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-20.0, 20.0),
    )
    default_loop_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_type="channel",
        loop_use_iteration_embedding=True,
    )

    base_model = ed.Qwen3_5ForCausalLM(base_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    exact_loop = ed.Qwen3_5ForCausalLM(exact_loop_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))

    # Copy base weights into looped model (loop_gates keep identity init)
    _, base_gstate, _ = base_model.split_module()
    exact_gdef, exact_gstate, exact_gother = exact_loop.split_module()
    exact_loop = exact_loop.merge_module(exact_gdef, exact_gstate.overlay(base_gstate), exact_gother)

    input_ids = jnp.asarray([[1, 5, 19, 42, 7, 13, 88, 3]], dtype=jnp.int32)
    base_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(base_model, input_ids)
    loop_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(exact_loop, input_ids)
    max_diff = float(jnp.max(jnp.abs(base_logits - loop_logits)))
    assert max_diff < 1e-4, f"Expected near-exact step-0 parity with (-20, +20) gate init, got max_diff={max_diff}"

    # Verify gradients flow through BOTH Pass 1 and Pass 2 (channel gate + iter embeddings)
    chan_model = ed.Qwen3_5ForCausalLM(default_loop_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(1))
    graphdef, params, rest = chan_model.split_module()

    @jax.jit
    def _val_and_grad(p):
        def loss_fn(p_in):
            m = chan_model.merge_module(graphdef, p_in, rest)
            logits = m(input_ids=input_ids).logits
            return jnp.mean(logits[:, :-1] ** 2)

        return jax.value_and_grad(loss_fn)(p)

    loss, grads = _val_and_grad(params)
    assert jnp.isfinite(loss)
    grad_flat = {str(path): val for path, val in grads.flatten().items()}
    for gate_idx in ("0", "1"):
        g_w = next(v for k, v in grad_flat.items() if k.endswith(f"loop_gates.{gate_idx}.proj.weight"))
        g_b = next(v for k, v in grad_flat.items() if k.endswith(f"loop_gates.{gate_idx}.proj.bias"))
        assert jnp.all(jnp.isfinite(g_w)) and float(jnp.linalg.norm(g_w)) > 0.0
        assert jnp.all(jnp.isfinite(g_b)) and float(jnp.linalg.norm(g_b)) > 0.0
    g_iter = next(v for k, v in grad_flat.items() if k.endswith("loop_iter_embeddings"))
    assert jnp.all(jnp.isfinite(g_iter)) and float(jnp.linalg.norm(g_iter)) > 0.0


def test_loop_hybrid_cache_prefill_decode_equivalence():
    """Verify 16-slot virtual HybridCache (12 physical + 4 looped) matches full forward during decode."""
    cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-0.3, 0.7),  # non-trivial Pass 1 + Pass 2 writes!
    )
    model = ed.Qwen3_5ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))

    cache_info = model.get_operations_cache_info()
    assert len(cache_info.layers) == 16
    assert [layer.layer_index for layer in cache_info.layers] == list(range(16))

    cache = model.init_cache(batch_size=1, max_length=32)
    assert len(cache.views) == 16

    tokens = jnp.asarray([[4, 17, 29, 8, 53, 12, 91, 6]], dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        full_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(model, tokens)

        # Prefill first 5 tokens, then decode remaining 3 tokens one by one
        out = spx.jit(lambda m, ids, c: m(input_ids=ids, past_key_values=c))(model, tokens[:, :5], cache)
        cache = out.past_key_values
        np.testing.assert_allclose(np.asarray(out.logits), np.asarray(full_logits[:, :5]), rtol=1e-3, atol=1e-3)

        step_fn = spx.jit(lambda m, ids, pos, c: m(input_ids=ids, position_ids=pos, past_key_values=c))
        for pos in range(5, tokens.shape[1]):
            step_out = step_fn(
                model,
                tokens[:, pos : pos + 1],
                jnp.asarray([[pos]], dtype=jnp.int32),
                cache,
            )
            cache = step_out.past_key_values
            np.testing.assert_allclose(
                np.asarray(step_out.logits[:, 0]),
                np.asarray(full_logits[:, pos]),
                rtol=1e-2,
                atol=1e-2,
            )


def test_unlooped_vlm_checkpoint_reloads_into_looped_vlm_roundtrip(tmp_path):
    """Save an unlooped Qwen3_5ForConditionalGeneration, reload it looped, verify parity & re-save."""
    base_cfg = _make_tiny_qwen3_5_vlm_config(num_hidden_layers=12)
    base_model = ed.Qwen3_5ForConditionalGeneration(
        base_cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(3),
    )
    base_dir = tmp_path / "base_vlm"
    base_model.save_pretrained(str(base_dir))

    # Reload the untouched pure checkpoint as both unlooped and looped VLMs (strict mismatch_allowed=False!)
    reloaded_base = ed.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        mismatch_allowed=False,
        config_kwargs={"attn_dtype": jnp.float32, "kvdtype": jnp.float32},
    )
    looped = ed.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        mismatch_allowed=False,
        config_kwargs={
            "attn_dtype": jnp.float32,
            "kvdtype": jnp.float32,
            "loop_start_layer": 4,
            "loop_end_layer": 8,
            "num_loop_iterations": 2,
            "loop_gate_init_biases": (-20.0, 20.0),
        },
    )
    assert looped.config.text_config.is_looped is True
    assert looped.config.text_config.num_cache_layers == 16

    input_ids = jnp.asarray([[2, 11, 23, 37, 51, 9]], dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        base_logits = np.asarray(spx.jit(lambda m, ids: m(input_ids=ids).logits)(reloaded_base, input_ids))
        loop_logits = np.asarray(spx.jit(lambda m, ids: m(input_ids=ids).logits)(looped, input_ids))
    assert float(np.max(np.abs(base_logits - loop_logits))) < 1e-4

    # Mutate gate bias to simulate training, save looped checkpoint, reload & verify trained gate survived
    looped.model.language_model.loop_gates[0].proj.bias.value = jnp.asarray([-0.25], dtype=jnp.float32)
    looped_dir = tmp_path / "looped_vlm"
    looped.save_pretrained(str(looped_dir))

    reloaded = ed.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(looped_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        mismatch_allowed=False,
    )
    assert reloaded.config.text_config.is_looped is True
    np.testing.assert_allclose(
        np.asarray(reloaded.model.language_model.loop_gates[0].proj.bias.value),
        np.asarray([-0.25], dtype=np.float32),
        atol=1e-6,
    )


@pytest.mark.cpu_ok
def test_hybrid_sliding_window_mask_details_and_validation():
    """Verify Qwen3.8-27B hybrid_sliding_window mask_details maps all 88 slots without Coda collision."""
    cfg = ed.Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        loop_start_layer=20,
        loop_end_layer=44,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=128,
        loop_kv_overlap=True,
    )
    details = cfg.get_mask_details()
    assert isinstance(details, dict)
    assert len(details) == 88

    sliding_slots = sorted(k for k, v in details.items() if v.mask_type == AttnMaskType.SLIDING)
    full_slots = sorted(k for k, v in details.items() if v.mask_type == AttnMaskType.FULL)
    linear_slots = sorted(k for k, v in details.items() if v.mask_type == AttnMaskType.LINEAR)

    # Pass 2 full-attention slots ONLY: 47, 51, 55, 59, 63, 67
    assert sliding_slots == [47, 51, 55, 59, 63, 67]
    assert all(details[k].size == 128 for k in sliding_slots)
    # Prelude (3..19) + Pass 1 (23..43) + Coda (71..87) stay FULL (no Coda collision with Pass 2!)
    assert full_slots == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 71, 75, 79, 83, 87]
    assert len(linear_slots) == 66

    # Separate mode leaves get_mask_details() as None (zero regression)
    sep_cfg = _make_tiny_qwen3_5_text_config(
        loop_start_layer=4, loop_end_layer=8, num_loop_iterations=2, loop_kv_cache_mode="separate"
    )
    assert sep_cfg.get_mask_details() is None

    # Invalid KV cache config raises ValueError
    with pytest.raises(ValueError, match="loop_kv_cache_mode"):
        _make_tiny_qwen3_5_text_config(
            loop_start_layer=4, loop_end_layer=8, num_loop_iterations=2, loop_kv_cache_mode="bad_mode"
        )
    with pytest.raises(ValueError, match="loop_kv_sliding_window"):
        _make_tiny_qwen3_5_text_config(
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_kv_cache_mode="hybrid_sliding_window",
            loop_kv_sliding_window=0,
        )


def test_hybrid_sliding_window_cache_shape_parity_and_gradients():
    """Verify GLM-5.4 hybrid sliding-window KV cache: ring-buffer sizing, prefill/decode parity, and gradients."""
    window = 4
    base_cfg = _make_tiny_qwen3_5_text_config(num_hidden_layers=12)
    disjoint_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-20.0, 20.0),
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=window,
        loop_kv_overlap=False,
    )
    overlap_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-0.4, 0.8),
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=window,
        loop_kv_overlap=True,
    )

    # 1. Exact step-0 identity parity with overlap=False even for seq_len (10) > sliding_window (4)
    base_model = ed.Qwen3_5ForCausalLM(base_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
    disjoint_model = ed.Qwen3_5ForCausalLM(disjoint_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
    _, base_gstate, _ = base_model.split_module()
    d_gdef, d_gstate, d_gother = disjoint_model.split_module()
    disjoint_model = disjoint_model.merge_module(d_gdef, d_gstate.overlay(base_gstate), d_gother)

    tokens = jnp.asarray([[3, 14, 27, 8, 55, 19, 42, 6, 71, 22]], dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        base_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(base_model, tokens)
        disjoint_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(disjoint_model, tokens)
        max_diff = float(jnp.max(jnp.abs(base_logits - disjoint_logits)))
        assert max_diff < 1e-4, f"Expected disjoint hybrid_sliding_window step-0 parity < 1e-4, got {max_diff}"

        # 2. Verify Pass-2 cache view allocates ONLY min(W, max_length) == 4 while Pass-1 & Coda stay full (32)
        for label, model in [
            ("disjoint", disjoint_model),
            (
                "overlap",
                ed.Qwen3_5ForCausalLM(overlap_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(13)),
            ),
        ]:
            cache = model.init_cache(batch_size=1, max_length=32)
            assert len(cache.views) == 16
            assert cache.views[3].key.shape[1] == 32, f"{label}: Prelude full-attn slot 3 must be full length"
            assert cache.views[7].key.shape[1] == 32, f"{label}: Pass-1 full-attn slot 7 must be full length"
            assert cache.views[11].key.shape[1] == window, (
                f"{label}: Pass-2 full-attn slot 11 must be sliding window {window}"
            )
            assert cache.views[15].key.shape[1] == 32, f"{label}: Coda full-attn slot 15 must be full length"

            full_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(model, tokens)
            # Prefill 6 tokens (> window=4 so sliding cache already wraps!) then decode 4 tokens one-by-one
            prefill_out = spx.jit(lambda m, ids, c: m(input_ids=ids, past_key_values=c))(model, tokens[:, :6], cache)
            cache = prefill_out.past_key_values
            np.testing.assert_allclose(
                np.asarray(prefill_out.logits),
                np.asarray(full_logits[:, :6]),
                rtol=1e-3,
                atol=2e-3,
            )
            step_fn = spx.jit(lambda m, ids, pos, c: m(input_ids=ids, position_ids=pos, past_key_values=c))
            for pos in range(6, tokens.shape[1]):
                step_out = step_fn(
                    model,
                    tokens[:, pos : pos + 1],
                    jnp.asarray([[pos]], dtype=jnp.int32),
                    cache,
                )
                cache = step_out.past_key_values
                np.testing.assert_allclose(
                    np.asarray(step_out.logits[:, 0]),
                    np.asarray(full_logits[:, pos]),
                    rtol=1e-2,
                    atol=1e-2,
                )

    # 3. Finite non-zero gradients through dual-stream LSE attention (overlap=True)
    overlap_model = ed.Qwen3_5ForCausalLM(overlap_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(17))
    gdef, params, rest = overlap_model.split_module()

    @jax.jit
    def _val_and_grad_overlap(p):
        def loss_fn(p_in):
            m = overlap_model.merge_module(gdef, p_in, rest)
            return jnp.mean(m(input_ids=tokens).logits ** 2)

        return jax.value_and_grad(loss_fn)(p)

    loss, grads = _val_and_grad_overlap(params)
    assert jnp.isfinite(loss)
    grad_flat = {str(path): val for path, val in grads.flatten().items()}
    for k, v in grad_flat.items():
        assert jnp.all(jnp.isfinite(v)), f"Non-finite gradient for {k}"
    qkv_grad = next(v for k, v in grad_flat.items() if "layers.7.self_attn.qkv_proj" in k)
    assert float(jnp.linalg.norm(qkv_grad)) > 0.0


def test_compute_loop_gate_stats_telemetry():
    """Verify compute_loop_gate_stats works on CausalLM and VLM under both KV cache modes."""
    tokens = jnp.asarray([[2, 9, 17, 33, 45, 11, 7, 29]], dtype=jnp.int32)
    for kv_mode in ("separate", "hybrid_sliding_window"):
        cfg = _make_tiny_qwen3_5_vlm_config(
            num_hidden_layers=12,
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_gate_init_biases=(-1.0, 1.5),
            loop_kv_cache_mode=kv_mode,
            loop_kv_sliding_window=4,
        )
        vlm = ed.Qwen3_5ForConditionalGeneration(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(21))
        stats = compute_loop_gate_stats(vlm, tokens)
        assert len(stats["per_pass"]) == 2
        for entry in stats["per_pass"]:
            assert 0.0 <= entry["gate_min"] <= entry["gate_mean"] <= entry["gate_max"] <= 1.0
            assert np.isfinite(entry["norm_ratio"]) and entry["norm_ratio"] > 0.0
            assert np.isfinite(entry["delta_ratio"]) and entry["delta_ratio"] >= 0.0
        assert stats["mean_nll"] is not None and np.isfinite(stats["mean_nll"])
        assert stats["pass1_loss_corr"] is not None and -1.0001 <= stats["pass1_loss_corr"] <= 1.0001


def test_differential_gate_lr_and_no_weight_decay_optimizer():
    """Verify param_lr_multipliers + no_weight_decay_patterns scales loop gates without touching trunk WD."""
    import optax
    from easydel.trainers.training_configurations import TrainingArguments

    cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_type="channel",
        loop_gate_init_biases=(-3.0, 3.0),
        loop_use_iteration_embedding=True,
    )
    model = ed.Qwen3_5ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(41))
    _, params, _ = model.split_module()

    base_args = TrainingArguments(
        optimizer="adamw",
        scheduler="none",
        learning_rate=1e-4,
        weight_decay=0.1,
        clip_grad=None,
        max_training_steps=10,
    )
    diff_args = TrainingArguments(
        optimizer="adamw",
        scheduler="none",
        learning_rate=1e-4,
        weight_decay=0.1,
        clip_grad=None,
        max_training_steps=10,
        param_lr_multipliers={"loop_gates": 10.0, "loop_iter_embeddings": 10.0},
        no_weight_decay_patterns=["loop_gates", "loop_iter_embeddings"],
    )
    tx_base, _ = base_args.get_optimizer_and_scheduler(10)
    tx_diff, _ = diff_args.get_optimizer_and_scheduler(10)

    # Zero gradients on non-zero params isolate weight decay; constant grads isolate LR multiplier.
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    ones_grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 0.01), params)

    st_base_0 = tx_base.init(params)
    st_diff_0 = tx_diff.init(params)
    upd_base_wd, _ = tx_base.update(zero_grads, st_base_0, params)
    upd_diff_wd, _ = tx_diff.update(zero_grads, st_diff_0, params)

    flat_p = {str(k): np.asarray(v) for k, v in params.flatten().items()}
    flat_wd_base = {str(k): np.asarray(v) for k, v in upd_base_wd.flatten().items()}
    flat_wd_diff = {str(k): np.asarray(v) for k, v in upd_diff_wd.flatten().items()}

    # Gate bias is non-zero (-3.0): must get exact 0 weight-decay update under no_weight_decay_patterns!
    gate_bias_key = next(k for k in flat_p if k.endswith("loop_gates.0.proj.bias"))
    trunk_key = next(k for k in flat_p if "layers.0." in k and flat_p[k].ndim >= 2)
    assert np.max(np.abs(flat_wd_diff[gate_bias_key])) == 0.0
    assert np.max(np.abs(flat_wd_base[gate_bias_key])) > 0.0
    np.testing.assert_allclose(flat_wd_diff[trunk_key], flat_wd_base[trunk_key], rtol=1e-6, atol=1e-6)

    # Pure Adam step with wd=0 on gate_w (zero-init weight) vs trunk_key: gate update must be exact 10x!
    upd_base_g, _ = tx_base.update(ones_grads, st_base_0, params)
    upd_diff_g, _ = tx_diff.update(ones_grads, st_diff_0, params)
    flat_g_base = {str(k): np.asarray(v) for k, v in upd_base_g.flatten().items()}
    flat_g_diff = {str(k): np.asarray(v) for k, v in upd_diff_g.flatten().items()}

    gate_w_key = next(k for k in flat_p if k.endswith("loop_gates.0.proj.weight"))
    iter_key = next(k for k in flat_p if k.endswith("loop_iter_embeddings"))
    np.testing.assert_allclose(flat_g_diff[gate_w_key], 10.0 * flat_g_base[gate_w_key], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(flat_g_diff[iter_key], 10.0 * flat_g_base[iter_key], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(flat_g_diff[trunk_key], flat_g_base[trunk_key], rtol=1e-6, atol=1e-6)
    # Zero extra optimizer state overhead (last chain state is EmptyState).
    assert isinstance(st_diff_0[-1], optax.EmptyState)


def test_compute_loop_gate_stats_knockout_identity_and_active():
    """Verify Loop-Knockout deltas are ~0 at identity init and finite/non-zero when Pass-1 gate is open."""
    tokens = jnp.asarray([[2, 9, 17, 33, 45, 11, 7, 29]], dtype=jnp.int32)
    asst_mask = jnp.asarray([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=jnp.bool_)

    # 1. Step-0 exact identity (-20, +20): knocking out Pass 1 must leave NLL unchanged (~0 delta).
    id_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_init_biases=(-20.0, 20.0),
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
        loop_kv_overlap=False,
    )
    id_model = ed.Qwen3_5ForCausalLM(id_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(51))
    id_stats = compute_loop_gate_stats(id_model, tokens, assistant_mask=asst_mask, include_knockout=True)
    ko_id = id_stats["knockout"]
    assert abs(ko_id["gate_zero"]["all_delta_nll"]) < 1e-3
    assert abs(ko_id["single_pass"]["all_delta_nll"]) < 1e-3
    assert abs(ko_id["gate_zero"]["asst_delta_nll"]) < 1e-3
    assert abs(ko_id["single_pass"]["asst_delta_nll"]) < 1e-3

    # 2. Active channel gate + iteration embeddings: knockout produces finite non-zero deltas.
    active_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_gate_type="channel",
        loop_gate_init_biases=(0.0, 1.5),
        loop_use_iteration_embedding=True,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
        loop_kv_overlap=True,
    )
    act_model = ed.Qwen3_5ForCausalLM(active_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(53))
    act_stats = compute_loop_gate_stats(act_model, tokens, assistant_mask=asst_mask, include_knockout=True)
    ko_act = act_stats["knockout"]
    assert np.isfinite(ko_act["gate_zero"]["asst_delta_nll"]) and abs(ko_act["gate_zero"]["asst_delta_nll"]) > 1e-4
    assert np.isfinite(ko_act["single_pass"]["asst_delta_nll"]) and abs(ko_act["single_pass"]["asst_delta_nll"]) > 1e-4

    # 3. K=3 active channel gate + iteration embeddings: per-pass loss_corr and gate_zero across K=3.
    k3_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=3,
        loop_gate_type="channel",
        loop_gate_init_biases=(-2.0, -2.0, 2.0),
        loop_use_iteration_embedding=True,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
        loop_kv_overlap=True,
    )
    k3_model = ed.Qwen3_5ForCausalLM(k3_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(57))
    k3_stats = compute_loop_gate_stats(k3_model, tokens, assistant_mask=asst_mask, include_knockout=True)
    assert len(k3_stats["per_pass"]) == 3
    for p_entry in k3_stats["per_pass"]:
        assert p_entry["loss_corr"] is not None and -1.0001 <= p_entry["loss_corr"] <= 1.0001
    assert np.isfinite(k3_stats["knockout"]["gate_zero"]["asst_delta_nll"])
    assert np.isfinite(k3_stats["knockout"]["single_pass"]["asst_delta_nll"])


def test_upgraded_loop_state_sharded_on_mesh_and_elarge_teacher_isolation(tmp_path):
    """Verify missing loop leaves land on the TPU mesh (not cpu:0) and teacher_base_config stays unlooped."""
    from easydel.infra.elarge import eLargeModel

    base_dir = tmp_path / "base_vlm"
    base_cfg = _make_tiny_qwen3_5_vlm_config(num_hidden_layers=12)
    base_model = ed.Qwen3_5ForConditionalGeneration(
        base_cfg,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        rngs=spx.Rngs(61),
    )
    base_model.save_pretrained(str(base_dir))

    n_dev = len(jax.devices())
    tp = 2 if n_dev >= 4 else 1
    axis_dims = [1, 1, -1, 1, tp, 1]

    for gate_type in ("scalar", "channel"):
        elm = eLargeModel(
            {
                "model": {"name_or_path": str(base_dir), "task": "image-text-to-text"},
                "teacher_model": {"name_or_path": str(base_dir), "task": "image-text-to-text"},
                "loader": {"dtype": "bf16", "param_dtype": "bf16"},
                "sharding": {"axis_dims": axis_dims, "auto_shard_model": True},
                "base_config": {
                    "values": {
                        "loop_start_layer": 4,
                        "loop_end_layer": 8,
                        "num_loop_iterations": 2,
                        "loop_gate_type": gate_type,
                        "loop_gate_init_biases": [-3.0, 3.0],
                        "loop_use_iteration_embedding": True,
                        "loop_kv_cache_mode": "hybrid_sliding_window",
                        "loop_kv_sliding_window": 4,
                        "loop_kv_overlap": True,
                    }
                },
                "teacher_base_config": {"values": {}},
            }
        )
        student = elm.build_model()
        teacher = elm.build_teacher_model()

        assert student.config.text_config.is_looped is True
        assert student.config.text_config.loop_gate_type == gate_type
        assert student.config.text_config.loop_use_iteration_embedding is True
        assert student.config.text_config.resolved_loop_gate_init_biases == (-3.0, 3.0)
        assert teacher is not None and teacher.config.text_config.is_looped is False
        assert teacher.model.language_model.loop_gates is None

        mesh_devices = set(student.mesh.jax_mesh.devices.flat)
        _, params, _ = student.split_module()
        loop_leaves = {
            str(k): v
            for k, v in params.flatten().items()
            if "loop_gates" in str(k) or "loop_iter_embeddings" in str(k)
        }
        assert len(loop_leaves) == 9  # 2 gates x (x_norm, delta_norm, proj.weight, proj.bias) + loop_iter_embeddings
        for k, arr in loop_leaves.items():
            assert arr.dtype == jnp.bfloat16, f"Expected bf16 for {k}, got {arr.dtype}"
            assert set(arr.sharding.device_set).issubset(mesh_devices), (
                f"Upgraded leaf {k} landed off-mesh on {arr.sharding.device_set}"
            )
            if k.endswith("loop_gates.0.proj.bias"):
                np.testing.assert_allclose(np.asarray(arr, dtype=np.float32), -3.0, atol=1e-2)
            elif k.endswith("loop_gates.1.proj.bias"):
                np.testing.assert_allclose(np.asarray(arr, dtype=np.float32), 3.0, atol=1e-2)
            elif k.endswith("loop_iter_embeddings"):
                assert arr.shape == (2, 64)
                np.testing.assert_allclose(np.asarray(arr, dtype=np.float32), 0.0, atol=1e-6)
        # Forward pass runs cleanly on the TPU mesh without device mismatch!
        toks = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
        out_s = student(input_ids=toks).logits
        out_t = teacher(input_ids=toks).logits
        assert jnp.all(jnp.isfinite(out_s)) and jnp.all(jnp.isfinite(out_t))


def test_dual_stream_chunked_segment_fast_path_matches_dense_reference():
    """Verify EasyDeL DualStreamAttn registered op matches unchunked dense reference under XLA."""
    from easydel.operations import OperationMetadata, OperationRegistry
    from ejkernel.modules import DualStreamAttentionConfig
    from ejkernel.types import MaskInfo

    rng = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    bsz, seq_len, n_q, n_kv, head_dim = 2, 14, 4, 2, 16
    q = jax.random.normal(k1, (bsz, seq_len, n_q, head_dim), dtype=jnp.float32)
    kg = jax.random.normal(k2, (bsz, seq_len, n_kv, head_dim), dtype=jnp.float32)
    vg = jax.random.normal(k3, (bsz, seq_len, n_kv, head_dim), dtype=jnp.float32)
    kl = jax.random.normal(k4, (bsz, seq_len, n_kv, head_dim), dtype=jnp.float32)
    vl = jax.random.normal(k5, (bsz, seq_len, n_kv, head_dim), dtype=jnp.float32)

    # Packed 2-document sequence + trailing padding (-1)
    seg_ids = jnp.asarray(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, -1],
        ],
        dtype=jnp.int32,
    )
    mi_seg = MaskInfo.from_segments(q_segment_ids=seg_ids, kv_segment_ids=seg_ids)
    mi_dense = MaskInfo.from_segments(q_segment_ids=seg_ids, kv_segment_ids=seg_ids)
    _ = mi_dense.get_or_compute_attention_mask(dtype=jnp.bool_)

    tiny_cfg = _make_tiny_qwen3_5_text_config(num_hidden_layers=4)
    meta_ref = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=tiny_cfg,
        operation_configs={"dual_stream_attention": DualStreamAttentionConfig(q_chunk_size=64, kv_chunk_size=64)},
    )
    meta_fast = OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        base_config=tiny_cfg,
        operation_configs={"dual_stream_attention": DualStreamAttentionConfig(q_chunk_size=4, kv_chunk_size=4)},
    )
    op_ref = OperationRegistry.create("dual_stream_attention", meta_ref)
    op_fast = OperationRegistry.create("dual_stream_attention", meta_fast)

    with jax.default_matmul_precision("highest"):
        for overlap in (True, False):
            ref = op_ref(
                query=q,
                k_global=kg,
                v_global=vg,
                k_local=kl,
                v_local=vl,
                global_mask_info=mi_dense,
                local_mask_info=mi_dense,
                softmax_scale=head_dim**-0.5,
                sliding_window=4,
                overlap=overlap,
                platform="xla",
            ).attention_outputs
            fast = op_fast(
                query=q,
                k_global=kg,
                v_global=vg,
                k_local=kl,
                v_local=vl,
                global_mask_info=mi_seg,
                local_mask_info=mi_seg,
                softmax_scale=head_dim**-0.5,
                sliding_window=4,
                overlap=overlap,
                platform="xla",
            ).attention_outputs
            np.testing.assert_allclose(np.asarray(fast), np.asarray(ref), rtol=1e-5, atol=1e-5)


def test_segment_aware_depthwise_conv_matches_reference():
    """Verify optimized slice-based _segment_aware_depthwise_conv matches exact causal masked sum."""
    cfg = _make_tiny_qwen3_5_text_config(num_hidden_layers=4)
    layer = ed.Qwen3_5ForCausalLM(cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(5)).model.layers[0]
    rng = jax.random.PRNGKey(9)
    x = jax.random.normal(rng, (2, 12, layer.linear_attn.conv_dim), dtype=jnp.float32)
    seg = jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, -1, -1], [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, -1]], dtype=jnp.int32)

    got = layer.linear_attn._segment_aware_depthwise_conv(x, seg)

    # Exact reference loop over positions
    w = np.asarray(layer.linear_attn.conv1d.weight.value[:, 0, :], dtype=np.float32)  # [K, C]
    x_np = np.asarray(x, dtype=np.float32)
    seg_np = np.asarray(seg)
    k_size = w.shape[0]
    ref = np.zeros_like(x_np)
    for b in range(x_np.shape[0]):
        for t in range(x_np.shape[1]):
            for j in range(k_size):
                src_t = t - (k_size - 1 - j)
                if src_t >= 0 and seg_np[b, src_t] == seg_np[b, t]:
                    ref[b, t] += x_np[b, src_t] * w[j]
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-5)


def test_pallas_dual_stream_splash_attention_fwd_bwd_parity():
    """Verify EasyDeL DualStreamAttn TPU Pallas forward + custom-VJP backward match XLA fallback."""
    from easydel.operations import OperationMetadata
    from easydel.operations.kernels import DualStreamAttn
    from ejkernel.types import MaskInfo

    if jax.default_backend() != "tpu":
        pytest.skip("Pallas SplashAttention dual-stream kernel is TPU-only")

    rng = jax.random.PRNGKey(101)
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
    bsz, seq_len, n_q, n_kv, head_dim = 1, 256, 4, 2, 128
    q = jax.random.normal(k1, (bsz, seq_len, n_q, head_dim), dtype=jnp.bfloat16) * 0.2
    kg = jax.random.normal(k2, (bsz, seq_len, n_kv, head_dim), dtype=jnp.bfloat16) * 0.2
    vg = jax.random.normal(k3, (bsz, seq_len, n_kv, head_dim), dtype=jnp.bfloat16) * 0.2
    kl = jax.random.normal(k4, (bsz, seq_len, n_kv, head_dim), dtype=jnp.bfloat16) * 0.2
    vl = jax.random.normal(k5, (bsz, seq_len, n_kv, head_dim), dtype=jnp.bfloat16) * 0.2
    dout = jax.random.normal(k6, (bsz, seq_len, n_q, head_dim), dtype=jnp.bfloat16) * 0.1

    # Packed 2-document sequence + trailing padding (-1)
    seg = jnp.concatenate(
        [
            jnp.zeros((1, 140), dtype=jnp.int32),
            jnp.ones((1, 100), dtype=jnp.int32),
            jnp.full((1, 16), -1, dtype=jnp.int32),
        ],
        axis=1,
    )
    scale = head_dim**-0.5
    tiny_cfg = _make_tiny_qwen3_5_text_config(num_hidden_layers=4)
    op = DualStreamAttn(
        OperationMetadata(runtime_dtype=jnp.bfloat16, runtime_softmax_dtype=jnp.float32, base_config=tiny_cfg)
    )

    for overlap in (True, False):

        @jax.jit
        def _ref_vg(q_, kg_, vg_, kl_, vl_, _ov=overlap):
            def _loss(q_i, kg_i, vg_i, kl_i, vl_i):
                mi_local = MaskInfo.from_segments(q_segment_ids=seg, kv_segment_ids=seg)
                o = op(
                    query=q_i,
                    k_global=kg_i,
                    v_global=vg_i,
                    k_local=kl_i,
                    v_local=vl_i,
                    global_mask_info=mi_local,
                    local_mask_info=mi_local,
                    softmax_scale=scale,
                    sliding_window=64,
                    overlap=_ov,
                    platform="xla",
                ).attention_outputs
                return jnp.sum(o.astype(jnp.float32) * dout.astype(jnp.float32)), o

            (_loss_val, out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
                q_, kg_, vg_, kl_, vl_
            )
            return out, grads

        @jax.jit
        def _pallas_vg(q_, kg_, vg_, kl_, vl_, _ov=overlap):
            def _loss(q_i, kg_i, vg_i, kl_i, vl_i):
                mi_local = MaskInfo.from_segments(q_segment_ids=seg, kv_segment_ids=seg)
                o = op(
                    query=q_i,
                    k_global=kg_i,
                    v_global=vg_i,
                    k_local=kl_i,
                    v_local=vl_i,
                    global_mask_info=mi_local,
                    local_mask_info=mi_local,
                    softmax_scale=scale,
                    sliding_window=64,
                    overlap=_ov,
                    platform="pallas",
                ).attention_outputs
                return jnp.sum(o.astype(jnp.float32) * dout.astype(jnp.float32)), o

            (_loss_val, out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(
                q_, kg_, vg_, kl_, vl_
            )
            return out, grads

        ref_o, ref_g = _ref_vg(q, kg, vg, kl, vl)
        pal_o, pal_g = _pallas_vg(q, kg, vg, kl, vl)
        np.testing.assert_allclose(
            np.asarray(pal_o, dtype=np.float32),
            np.asarray(ref_o, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
            err_msg=f"Forward mismatch for overlap={overlap}",
        )
        for name, pg, rg in zip(("dq", "dk_g", "dv_g", "dk_l", "dv_l"), pal_g, ref_g, strict=True):
            assert jnp.all(jnp.isfinite(pg)), f"Non-finite Pallas grad {name} (overlap={overlap})"
            np.testing.assert_allclose(
                np.asarray(pg, dtype=np.float32),
                np.asarray(rg, dtype=np.float32),
                rtol=3e-2,
                atol=3e-2,
                err_msg=f"Backward mismatch for {name} (overlap={overlap})",
            )


def test_loop_scanned_block_scan_matches_unrolled_fwd_and_grads():
    """Verify scan_layers=True + block_scan=True matches unrolled looped forward and gradients."""
    tokens = jnp.asarray([[4, 17, 29, 8, 53, 12, 91, 6, 33, 19]], dtype=jnp.int32)
    for kv_mode in ("separate", "hybrid_sliding_window"):
        unrolled_cfg = _make_tiny_qwen3_5_text_config(
            num_hidden_layers=12,
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_gate_init_biases=(-0.4, 0.8),
            loop_kv_cache_mode=kv_mode,
            loop_kv_sliding_window=4,
        )
        scanned_cfg = _make_tiny_qwen3_5_text_config(
            num_hidden_layers=12,
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_gate_init_biases=(-0.4, 0.8),
            loop_kv_cache_mode=kv_mode,
            loop_kv_sliding_window=4,
        )
        scanned_cfg.scan_layers = True
        scanned_cfg.layer_scan_options = {"text": {"block_scan": True}}

        m_unrolled = ed.Qwen3_5ForCausalLM(unrolled_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(31))
        m_scanned = ed.Qwen3_5ForCausalLM(scanned_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(31))
        _, u_state, _ = m_unrolled.split_module()
        s_gdef, _, s_rest = m_scanned.split_module()
        m_scanned = m_scanned.merge_module(s_gdef, u_state, s_rest)

        u_gdef, u_p, u_rest = m_unrolled.split_module()
        s_gdef, s_p, s_rest = m_scanned.split_module()

        @jax.jit
        def _vg_u(p, _m=m_unrolled, _gd=u_gdef, _r=u_rest):
            def _f(p_in):
                logits = _m.merge_module(_gd, p_in, _r)(input_ids=tokens).logits
                return jnp.mean(logits**2), logits

            return jax.value_and_grad(_f, has_aux=True)(p)

        @jax.jit
        def _vg_s(p, _m=m_scanned, _gd=s_gdef, _r=s_rest):
            def _f(p_in):
                logits = _m.merge_module(_gd, p_in, _r)(input_ids=tokens).logits
                return jnp.mean(logits**2), logits

            return jax.value_and_grad(_f, has_aux=True)(p)

        with jax.default_matmul_precision("highest"):
            (_, u_logits), u_grads = _vg_u(u_p)
            (_, s_logits), s_grads = _vg_s(s_p)
        np.testing.assert_allclose(np.asarray(s_logits), np.asarray(u_logits), rtol=1e-4, atol=1e-4)
        uf = u_grads.flatten()
        sf = s_grads.flatten()
        for k in uf:
            np.testing.assert_allclose(np.asarray(sf[k]), np.asarray(uf[k]), rtol=1e-3, atol=1e-3, err_msg=f"{kv_mode}: {k}")


@pytest.mark.cpu_ok
def test_dual_stream_ragged_page_attn_registry_isolation_and_esurge_specs():
    """Verify DualStreamRaggedPageAttn registration, zero private ejkernel imports, and eSurge cache specs."""
    from pathlib import Path

    from easydel.caching import RaggedPagesCacheView
    from easydel.inference.esurge.core.interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
        create_kv_cache_specs_from_config,
    )
    from easydel.operations import OperationMetadata, OperationRegistry
    from easydel.operations.kernels import DualStreamRaggedPageAttn
    from easydel.operations.requirements import MetadataField

    tiny_cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=12,
        loop_start_layer=4,
        loop_end_layer=8,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
    )
    meta = OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, base_config=tiny_cfg)
    op = OperationRegistry.create("dual_stream_ragged_page_attention", meta)
    assert isinstance(op, DualStreamRaggedPageAttn)
    assert op.get_impl_name() == "dual_stream_ragged_page_attention"

    reqs = op.get_requirements()
    assert reqs.cache.cache_view_class is RaggedPagesCacheView
    assert MetadataField.PAGES_TABLES in reqs.metadata.required
    assert MetadataField.REQUEST_DISTRIBUTION in reqs.metadata.required

    # Layering hygiene: EasyDeL adapter must never import private ejkernel.kernels._* or Pallas directly.
    adapter_src = (
        Path(__file__).resolve().parents[2] / "easydel" / "operations" / "kernels" / "dual_stream_ragged_page_attention.py"
    ).read_text()
    for forbidden in ("ejkernel.kernels.", "_pallas", "_xla", "_triton", "pallas_call", "jax.experimental.pallas"):
        assert forbidden not in adapter_src, f"Forbidden private/backend import {forbidden!r} in EasyDeL adapter"

    # eSurge cache-group builder automatically separates Group 0 (FULL + LINEAR) from Group 1 (SLIDING Pass-2).
    full_27b_cfg = ed.Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        loop_start_layer=20,
        loop_end_layer=44,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=128,
        loop_kv_overlap=True,
    )
    specs = create_kv_cache_specs_from_config(
        config=full_27b_cfg,
        page_size=128,
        num_kv_heads=4,
        head_size=256,
        dtype=jnp.bfloat16,
    )
    assert len(specs) == 2
    full_groups = [g for g in specs if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    sliding_groups = [g for g in specs if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
    assert len(full_groups) == 1 and len(sliding_groups) == 1
    assert sliding_groups[0].kv_cache_spec.sliding_window == 128
    assert sliding_groups[0].layer_names == [f"layer.{idx}" for idx in (47, 51, 55, 59, 63, 67)]
    assert len(full_groups[0].layer_names) == 82


def test_dual_stream_ragged_page_attn_adapter_matches_dense_with_eviction():
    """Verify EasyDeL DualStreamRaggedPageAttn matches DualStreamAttn across evicted sliding pages."""
    from easydel.caching import RaggedPagesCacheConfig, RaggedPagesCacheView, RaggedPagesMetadata
    from easydel.operations import OperationMetadata, OperationRegistry
    from ejkernel.types import MaskInfo

    rng = jax.random.PRNGKey(2026)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    num_q_heads, num_kv_heads, head_dim = 4, 2, 128
    page_size = 4
    window = 4
    total_len = 10  # 3 pages: [0..3], [4..7], [8..9]; at pos >= 8, local page 0 is evicted!
    scale = head_dim**-0.5

    q_full = jax.random.normal(k1, (1, total_len, num_q_heads, head_dim), dtype=jnp.float32) * 0.25
    kg_full = jax.random.normal(k2, (1, total_len, num_kv_heads, head_dim), dtype=jnp.float32) * 0.25
    vg_full = jax.random.normal(k3, (1, total_len, num_kv_heads, head_dim), dtype=jnp.float32) * 0.25
    kl_full = jax.random.normal(k4, (1, total_len, num_kv_heads, head_dim), dtype=jnp.float32) * 0.25
    vl_full = jax.random.normal(k5, (1, total_len, num_kv_heads, head_dim), dtype=jnp.float32) * 0.25

    cfg = _make_tiny_qwen3_5_text_config(
        num_hidden_layers=4,
        loop_start_layer=0,
        loop_end_layer=4,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=window,
    )
    cfg.num_attention_heads = num_q_heads
    cfg.num_key_value_heads = num_kv_heads
    cfg.head_dim = head_dim
    meta = OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, base_config=cfg)
    dense_op = OperationRegistry.create("dual_stream_attention", meta)
    rpa_v3_op = OperationRegistry.create("ragged_page_attention_v3", meta)
    ds_rpa_op = OperationRegistry.create("dual_stream_ragged_page_attention", meta)

    cache_cfg = RaggedPagesCacheConfig(
        num_hidden_layers=4,
        max_model_length=32,
        num_kv_heads=num_kv_heads,
        logical_num_kv_heads=num_kv_heads,
        k_headdim=head_dim,
        v_headdim=head_dim,
        page_size=page_size,
        num_pages=8,
        max_num_pages_per_req=8,
        version="v3",
        _kvdtype_str="fp32",
    )

    mi_full = MaskInfo.from_segments(
        q_segment_ids=jnp.zeros((1, total_len), dtype=jnp.int32),
        kv_segment_ids=jnp.zeros((1, total_len), dtype=jnp.int32),
    )

    for overlap in (True, False):
        dense_out = dense_op(
            query=q_full,
            k_global=kg_full,
            v_global=vg_full,
            k_local=kl_full,
            v_local=vl_full,
            global_mask_info=mi_full,
            local_mask_info=mi_full,
            softmax_scale=scale,
            sliding_window=window,
            overlap=overlap,
            platform="xla",
        ).attention_outputs

        with cfg.mesh:
            g_view = RaggedPagesCacheView.init(
                config=cache_cfg,
                layer_index=0,
                mesh=cfg.mesh,
                runtime_sharding_resolver=cfg.runtime_sharding_resolver,
            )
            l_view = RaggedPagesCacheView.init(
                config=cache_cfg,
                layer_index=1,
                mesh=cfg.mesh,
                runtime_sharding_resolver=cfg.runtime_sharding_resolver,
            )

        # Step 1: Prefill 6 tokens [0..5] into global & local views
        # Step 2..5: Decode tokens 6, 7, 8, 9 one by one; at token 8 & 9 evict local page 0!
        steps = [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        step_outputs = []
        for s_start, s_end in steps:
            q_len = s_end - s_start
            g_len = s_end
            g_bt = jnp.asarray([[1, 3, 5, 0]], dtype=jnp.int32)
            # Evict local page 0 (phys page 2) once s_start >= 8 (all tokens in page 0 are >= 5 steps away > W=4)
            if s_start >= 8:
                l_len = s_end - page_size
                l_bt = jnp.asarray([[4, 6, 0, 0]], dtype=jnp.int32)
            else:
                l_len = s_end
                l_bt = jnp.asarray([[2, 4, 6, 0]], dtype=jnp.int32)

            qsl = jnp.asarray([0, q_len], dtype=jnp.int32)
            dist = jnp.asarray([1, 1, 1] if q_len == 1 else [0, 0, 1], dtype=jnp.int32)
            g_meta = RaggedPagesMetadata(
                pages_tables=g_bt,
                context_lens=jnp.asarray([g_len], dtype=jnp.int32),
                query_start_loc=qsl,
                num_seqs=jnp.asarray([1], dtype=jnp.int32),
                request_distribution=dist,
                page_size=page_size,
                version="v3",
            )
            l_meta = RaggedPagesMetadata(
                pages_tables=l_bt,
                context_lens=jnp.asarray([l_len], dtype=jnp.int32),
                query_start_loc=qsl,
                num_seqs=jnp.asarray([1], dtype=jnp.int32),
                request_distribution=dist,
                page_size=page_size,
                version="v3",
            )
            grouped_meta = g_meta.replace(cache_groups=(g_meta, l_meta), layer_to_cache_group=(0, 1))

            # Pass 1 populates global view via RaggedPageAttnV3
            g_res = rpa_v3_op(
                query=q_full[:, s_start:s_end],
                key=kg_full[:, s_start:s_end],
                value=vg_full[:, s_start:s_end],
                cache_view=g_view,
                cache_metadata=grouped_meta.for_layer(0),
                softmax_scale=scale,
            )
            g_view = g_res.cache_view
            g_pages_before = np.asarray(g_view.kv_pages, dtype=np.float32).copy()

            # Pass 2 runs DualStreamRaggedPageAttn reading g_view (read-only) + updating l_view
            ds_res = ds_rpa_op(
                query=q_full[:, s_start:s_end],
                key=kl_full[:, s_start:s_end],
                value=vl_full[:, s_start:s_end],
                global_cache_view=g_view,
                local_cache_view=l_view,
                global_cache_metadata=grouped_meta,
                local_cache_metadata=grouped_meta,
                softmax_scale=scale,
                sliding_window=window,
                overlap=overlap,
            )
            l_view = ds_res.cache_view
            np.testing.assert_array_equal(np.asarray(g_view.kv_pages, dtype=np.float32), g_pages_before)
            step_outputs.append(np.asarray(ds_res.attention_outputs, dtype=np.float32))

        got = np.concatenate(step_outputs, axis=1)
        np.testing.assert_allclose(
            got,
            np.asarray(dense_out, dtype=np.float32),
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"DualStreamRaggedPageAttn mismatch vs DualStreamAttn (overlap={overlap})",
        )


def test_loop_hybrid_sliding_window_ragged_paged_prefill_decode_equivalence():
    """End-to-end looped Qwen3_5ForCausalLM with RaggedPagesCacheView + RecurrentCacheView matches cacheless forward."""
    from easydel.caching import (
        HybridCache,
        RaggedPagesCacheConfig,
        RaggedPagesCacheView,
        RaggedPagesMetadata,
        RecurrentCacheView,
    )

    window = 4
    page_size = 4
    tokens = jnp.asarray([[3, 14, 27, 8, 55, 19, 42, 6, 71, 22]], dtype=jnp.int32)
    total_len = int(tokens.shape[1])

    for overlap in (False, True):
        base_kwargs = dict(
            num_hidden_layers=12,
            loop_start_layer=4,
            loop_end_layer=8,
            num_loop_iterations=2,
            loop_gate_init_biases=(-0.4, 0.8),
            loop_kv_cache_mode="hybrid_sliding_window",
            loop_kv_sliding_window=window,
            loop_kv_overlap=overlap,
        )
        ref_cfg = _make_tiny_qwen3_5_text_config(**base_kwargs)
        ref_cfg.head_dim = 128
        ref_cfg.attn_mechanism = "vanilla"

        paged_cfg = _make_tiny_qwen3_5_text_config(**base_kwargs)
        paged_cfg.head_dim = 128
        paged_cfg.attn_mechanism = "ragged_page_attention_v3"

        ref_model = ed.Qwen3_5ForCausalLM(ref_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(29))
        paged_model = ed.Qwen3_5ForCausalLM(paged_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(29))
        _, ref_state, _ = ref_model.split_module()
        p_gdef, p_state, p_other = paged_model.split_module()
        paged_model = paged_model.merge_module(p_gdef, p_state.overlay(ref_state), p_other)

        ragged_cfg = RaggedPagesCacheConfig(
            num_hidden_layers=paged_cfg.num_cache_layers,
            max_model_length=32,
            num_kv_heads=paged_cfg.num_key_value_heads,
            logical_num_kv_heads=paged_cfg.num_key_value_heads,
            k_headdim=paged_cfg.head_dim,
            v_headdim=paged_cfg.head_dim,
            page_size=page_size,
            num_pages=8,
            max_num_pages_per_req=8,
            version="v3",
            _kvdtype_str="fp32",
        )
        cache = paged_model.init_operations_cache(
            batch_size=1,
            max_length=32,
            page_size=page_size,
            dtype=jnp.float32,
            ragged_config=ragged_cfg,
        )
        assert isinstance(cache, HybridCache)
        assert len(cache.views) == 16
        # Full-attention slots (Prelude 3, Pass-1 7, Pass-2 11, Coda 15) must be RaggedPagesCacheView
        for slot in (3, 7, 11, 15):
            assert isinstance(cache.views[slot], RaggedPagesCacheView)
        for slot in (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14):
            assert isinstance(cache.views[slot], RecurrentCacheView)

        # Group 0 owns all FULL + LINEAR slots; Group 1 owns Pass-2 SLIDING slot (11).
        layer_to_group = tuple(1 if idx == 11 else 0 for idx in range(16))

        with jax.default_matmul_precision("highest"):
            ref_logits = np.asarray(spx.jit(lambda m, ids: m(input_ids=ids).logits)(ref_model, tokens))

            # Prefill 6 tokens (> window=4) + decode remaining 4 tokens one-by-one with Group-1 page eviction at pos >= 8
            steps = [(0, 6), (6, 7), (7, 8), (8, 9), (9, total_len)]
            for s_start, s_end in steps:
                q_len = s_end - s_start
                g_bt = jnp.asarray([[1, 2, 3, 0]], dtype=jnp.int32)
                if s_start >= 8:
                    l_len = s_end - page_size
                    l_bt = jnp.asarray([[5, 6, 0, 0]], dtype=jnp.int32)
                else:
                    l_len = s_end
                    l_bt = jnp.asarray([[4, 5, 6, 0]], dtype=jnp.int32)

                qsl = jnp.asarray([0, q_len], dtype=jnp.int32)
                dist = jnp.asarray([1, 1, 1] if q_len == 1 else [0, 0, 1], dtype=jnp.int32)
                pos_1d = jnp.arange(s_start, s_end, dtype=jnp.int32)

                g_meta = RaggedPagesMetadata(
                    pages_tables=g_bt,
                    context_lens=jnp.asarray([s_end], dtype=jnp.int32),
                    query_start_loc=qsl,
                    position_ids=pos_1d,
                    num_seqs=jnp.asarray([1], dtype=jnp.int32),
                    request_distribution=dist,
                    page_size=page_size,
                    version="v3",
                )
                l_meta = RaggedPagesMetadata(
                    pages_tables=l_bt,
                    context_lens=jnp.asarray([l_len], dtype=jnp.int32),
                    query_start_loc=qsl,
                    position_ids=pos_1d,
                    num_seqs=jnp.asarray([1], dtype=jnp.int32),
                    request_distribution=dist,
                    page_size=page_size,
                    version="v3",
                )
                grouped_meta = g_meta.replace(cache_groups=(g_meta, l_meta), layer_to_cache_group=layer_to_group)

                out = spx.jit(
                    lambda m, ids, pos, c, cm: m(
                        input_ids=ids,
                        position_ids=pos,
                        past_key_values=c,
                        cache_metadata=cm,
                    )
                )(paged_model, tokens[:, s_start:s_end], pos_1d[None, :], cache, grouped_meta)
                cache = out.past_key_values
                np.testing.assert_allclose(
                    np.asarray(out.logits, dtype=np.float32),
                    ref_logits[:, s_start:s_end],
                    rtol=1e-2,
                    atol=1e-2,
                    err_msg=f"Looped paged Qwen3.5 logits mismatch at step [{s_start}:{s_end}) (overlap={overlap})",
                )
