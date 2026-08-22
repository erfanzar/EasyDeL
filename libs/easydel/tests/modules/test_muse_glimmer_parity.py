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

"""Numerical parity checks for Muse-Glimmer against an independent NumPy reference.

``transformers`` only ships ``muse_glimmer`` from the release that introduced the
architecture, so the usual HF-comparison suite in ``tests/modules/spmd`` cannot
run on older pins. These tests instead transcribe the reference implementation
(``transformers/models/muse_glimmer/modeling_muse_glimmer.py``) directly into
NumPy and compare it against the EasyDeL port, driving both from one
HuggingFace-named ``state_dict`` so the checkpoint conversion path — including
the fused ``[Q | gate | K | V]`` and ``[gate | up]`` layouts — is exercised too.
"""

import inspect
import math

import easydel as ed
import jax
import ml_dtypes
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.muse_glimmer import modeling_muse_glimmer as mg
from jax import numpy as jnp

torch = pytest.importorskip("torch", reason="HF-format state dicts are built as torch tensors")

RTOL = 1e-5
ATOL = 1e-5


def _bfloat16_round(values: np.ndarray) -> np.ndarray:
    """Round through bfloat16, then back to float64.

    ``EasyDeLBaseConfig.get_basic_frequencies`` deliberately stores the RoPE
    cos/sin table in bfloat16 for memory efficiency (shared by every family in
    the zoo), so the reference has to model that rounding to stay comparable.
    """
    return np.asarray(values, dtype=ml_dtypes.bfloat16).astype(np.float64)


def _text_config(**overrides):
    """Build a small but structurally complete text configuration.

    Eight layers keep two full/NoPE layers and six sliding/RoPE layers in the
    default schedule, so both branches are covered.
    """
    kwargs = {
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "sliding_window": 5,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "qk_scale_factor": 3.87,
        "output_multiplier": 0.25,
        "final_logit_softcapping": 20.0,
        "rope_theta": 500_000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        # Pin the attention kernel to float32; the default activation dtype for
        # attention is bfloat16, which would dominate the comparison.
        "attn_dtype": "float32",
        "attn_softmax_dtype": "float32",
    }
    kwargs.update(overrides)
    return ed.MuseGlimmerTextConfig(**kwargs)


def _vision_config(**overrides):
    """Build a small vision tower configuration with both attention branches."""
    kwargs = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "patch_size": 2,
        "patch_temporal": 2,
        "merge_size": 2,
        "pos_emb_height": 4,
        "pos_emb_width": 4,
        "layer_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "attn_dtype": "float32",
        "attn_softmax_dtype": "float32",
    }
    kwargs.update(overrides)
    return ed.MuseGlimmerVisionConfig(**kwargs)


def _model_config(text_config, vision_config, image_token_id, video_token_id):
    """Bind the text and vision configs into a composite VLM configuration."""
    return ed.MuseGlimmerConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        out_hidden_size=vision_config.out_hidden_size,
        projector_hidden_size=24,
        projector_hidden_act="gelu",
    )


def _build_state_dict(config, seed: int = 0) -> dict[str, "torch.Tensor"]:
    """Create a random HuggingFace-named ``state_dict`` for the whole VLM.

    Key names and tensor orientations follow HF's ``MuseGlimmerForConditionalGeneration``
    exactly (``nn.Linear`` weights stored as ``(out_features, in_features)``), so
    loading it through EasyDeL's conversion path is a real test of that path.

    Args:
        config: Composite Muse-Glimmer configuration.
        seed: Seed for the random weight generator.

    Returns:
        dict[str, torch.Tensor]: The synthetic checkpoint.
    """
    rng = np.random.default_rng(seed)
    text = config.text_config
    vision = config.vision_config
    state: dict[str, np.ndarray] = {}

    def normal(*shape, scale=0.05):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    state["model.language_model.embed_tokens.weight"] = normal(text.vocab_size, text.hidden_size)
    state["model.language_model.norm.weight"] = normal(text.hidden_size)
    state["lm_head.weight"] = normal(text.vocab_size, text.hidden_size)

    q_dim = text.num_attention_heads * text.head_dim
    kv_dim = text.num_key_value_heads * text.head_dim
    for layer in range(text.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}"
        state[f"{prefix}.self_attn.q_proj.weight"] = normal(q_dim, text.hidden_size)
        state[f"{prefix}.self_attn.k_proj.weight"] = normal(kv_dim, text.hidden_size)
        state[f"{prefix}.self_attn.v_proj.weight"] = normal(kv_dim, text.hidden_size)
        state[f"{prefix}.self_attn.gate_proj.weight"] = normal(q_dim, text.hidden_size)
        state[f"{prefix}.self_attn.o_proj.weight"] = normal(text.hidden_size, q_dim)
        state[f"{prefix}.mlp.gate_proj.weight"] = normal(text.intermediate_size, text.hidden_size)
        state[f"{prefix}.mlp.up_proj.weight"] = normal(text.intermediate_size, text.hidden_size)
        state[f"{prefix}.mlp.down_proj.weight"] = normal(text.hidden_size, text.intermediate_size)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            state[f"{prefix}.{norm}.weight"] = normal(text.hidden_size)

    patch_features = vision.patch_temporal * vision.in_channels * vision.patch_size**2
    state["model.vision_tower.patch_embedder.patch_embedding.weight"] = normal(vision.hidden_size, patch_features)
    state["model.vision_tower.patch_embedder.position_embedding_table.weight"] = normal(
        vision.pos_emb_height * vision.pos_emb_width, vision.hidden_size
    )
    for norm in ("ln_pre", "ln_post"):
        state[f"model.vision_tower.{norm}.weight"] = normal(vision.hidden_size)
        state[f"model.vision_tower.{norm}.bias"] = normal(vision.hidden_size)
    for layer in range(vision.num_hidden_layers):
        prefix = f"model.vision_tower.layers.{layer}"
        for proj in ("q_proj", "k_proj", "v_proj", "proj"):
            state[f"{prefix}.attn.{proj}.weight"] = normal(vision.hidden_size, vision.hidden_size)
            state[f"{prefix}.attn.{proj}.bias"] = normal(vision.hidden_size)
        state[f"{prefix}.mlp.fc1.weight"] = normal(vision.intermediate_size, vision.hidden_size)
        state[f"{prefix}.mlp.fc1.bias"] = normal(vision.intermediate_size)
        state[f"{prefix}.mlp.fc2.weight"] = normal(vision.hidden_size, vision.intermediate_size)
        state[f"{prefix}.mlp.fc2.bias"] = normal(vision.hidden_size)
        for norm in ("norm1", "norm2"):
            state[f"{prefix}.{norm}.weight"] = normal(vision.hidden_size)
            state[f"{prefix}.{norm}.bias"] = normal(vision.hidden_size)

    state["model.vision_adapter.fc1.weight"] = normal(config.projector_hidden_size, config.out_hidden_size)
    state["model.vision_adapter.fc2.weight"] = normal(config.projector_hidden_size, config.projector_hidden_size)
    state["model.vision_projection.weight"] = normal(text.hidden_size, config.projector_hidden_size)

    return {key: torch.from_numpy(value) for key, value in state.items()}


def _load_easydel(config, state_dict):
    """Instantiate the EasyDeL VLM and load the HF-named ``state_dict`` into it."""
    model = ed.MuseGlimmerForConditionalGeneration.lazy_init(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    model = ed.traversals.merge_model_and_tree(model, tree=model.transform_fn(state_dict))
    model.eval()
    return model.shard_model()


# --------------------------------------------------------------------------------------
# NumPy reference, transcribed from transformers/models/muse_glimmer/modeling_muse_glimmer.py
# --------------------------------------------------------------------------------------


def _ref_rms_scaleless(x, eps):
    """``MuseGlimmerRMSNorm(with_scale=False)``."""
    x = x.astype(np.float64)
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def _ref_rms_scaled(x, weight, eps):
    """``MuseGlimmerRMSNorm(with_scale=True)`` — scale applied as ``weight``."""
    return _ref_rms_scaleless(x, eps) * weight.astype(np.float64)


def _ref_centered_rms(x, weight, eps):
    """``MuseGlimmerTextCenteredRMSNorm`` — scale applied as ``1 + weight``."""
    return _ref_rms_scaleless(x, eps) * (1.0 + weight.astype(np.float64))


def _ref_layer_norm(x, weight, bias, eps):
    """Standard LayerNorm over the trailing axis."""
    x = x.astype(np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight.astype(np.float64) + bias.astype(np.float64)


_ERF = np.frompyfunc(math.erf, 1, 1)


def _ref_gelu(x):
    """Exact (erf) GELU, matching ``ACT2FN["gelu"]`` on both sides."""
    return 0.5 * x * (1.0 + _ERF(x / np.sqrt(2.0)).astype(np.float64))


def _ref_rotate_half(x):
    """``rotate_half`` from the reference implementation."""
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _ref_rope_tables(positions, head_dim, base):
    """Default RoPE cos/sin tables for ``positions`` (NeoX pairing)."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return _bfloat16_round(np.cos(emb)), _bfloat16_round(np.sin(emb))


# EasyDeL's shared attention path treats `sliding_window` as a per-side radius
# (`easydel/layers/attention/_flexible.py::_apply_sliding_window` computes
# `width = left + right + 1`), so a causal layer configured with window `w`
# attends to `w + 1` tokens. HuggingFace treats it as a span: its
# `sliding_window_overlay` keeps `kv_idx > q_idx - w`, i.e. `w` tokens.
# Every sliding-window family in the zoo forwards `config.sliding_window`
# unchanged, so this port does too and the reference below follows the runtime
# convention. `test_sliding_window_span` pins the resulting span on its own, so
# the discrepancy is visible and this suite fails loudly if the shared path
# changes.
_EASYDEL_SLIDING_WINDOW_IS_RADIUS = True


def _ref_causal_masks(seq_len, sliding_window):
    """Additive full-causal and sliding-window-causal masks."""
    q_idx = np.arange(seq_len)[:, None]
    kv_idx = np.arange(seq_len)[None, :]
    causal = kv_idx <= q_idx
    span = sliding_window + 1 if _EASYDEL_SLIDING_WINDOW_IS_RADIUS else sliding_window
    sliding = causal & (kv_idx > q_idx - span)
    neg = -np.inf
    return (
        np.where(causal, 0.0, neg),
        np.where(sliding, 0.0, neg),
    )


def _ref_text_logits(config, state, input_ids):
    """Run the reference Muse-Glimmer text stack and LM head.

    Args:
        config: Composite Muse-Glimmer configuration.
        state: HF-named ``state_dict`` as NumPy arrays.
        input_ids: Token ids of shape ``(batch, seq_len)``.

    Returns:
        np.ndarray: Scaled, soft-capped logits of shape ``(batch, seq_len, vocab_size)``.
    """
    text = config.text_config
    heads, kv_heads, head_dim = text.num_attention_heads, text.num_key_value_heads, text.head_dim
    groups = heads // kv_heads
    batch, seq_len = input_ids.shape
    positions = np.arange(seq_len)
    full_mask, sliding_mask = _ref_causal_masks(seq_len, text.sliding_window)

    hidden = state["model.language_model.embed_tokens.weight"][input_ids]
    hidden = _ref_rms_scaleless(hidden, text.rms_norm_eps)

    for layer in range(text.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}"
        residual = hidden
        normed = _ref_centered_rms(hidden, state[f"{prefix}.input_layernorm.weight"], text.rms_norm_eps)

        def project(name, out_heads, x=normed, p=prefix):
            weight = state[f"{p}.self_attn.{name}.weight"].astype(np.float64)
            return (x @ weight.T).reshape(batch, seq_len, out_heads, head_dim)

        query = project("q_proj", heads)
        key = project("k_proj", kv_heads)
        value = project("v_proj", kv_heads)
        gate = project("gate_proj", heads)

        query = _ref_rms_scaleless(query, text.rms_norm_eps) * text.qk_scale_factor
        key = _ref_rms_scaleless(key, text.rms_norm_eps)

        theta = text.layer_rope_theta[layer]
        if theta:
            cos, sin = _ref_rope_tables(positions, head_dim, theta)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            query = query * cos + _ref_rotate_half(query) * sin
            key = key * cos + _ref_rotate_half(key) * sin

        key = np.repeat(key, groups, axis=2)
        value = np.repeat(value, groups, axis=2)

        scores = np.einsum("bqhd,bkhd->bhqk", query, key) * (head_dim**-0.5)
        mask = sliding_mask if text.layer_types[layer] == "sliding_attention" else full_mask
        scores = scores + mask[None, None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        context = np.einsum("bhqk,bkhd->bqhd", weights, value)

        context = context * (1.0 / (1.0 + np.exp(-gate)))
        context = context.reshape(batch, seq_len, heads * head_dim)
        attn_out = context @ state[f"{prefix}.self_attn.o_proj.weight"].astype(np.float64).T
        hidden = residual + _ref_centered_rms(
            attn_out, state[f"{prefix}.post_attention_layernorm.weight"], text.post_norm_eps
        )

        residual = hidden
        normed = _ref_centered_rms(hidden, state[f"{prefix}.pre_feedforward_layernorm.weight"], text.rms_norm_eps)
        gate_h = normed @ state[f"{prefix}.mlp.gate_proj.weight"].astype(np.float64).T
        up_h = normed @ state[f"{prefix}.mlp.up_proj.weight"].astype(np.float64).T
        act = gate_h / (1.0 + np.exp(-gate_h))
        ffn = (act * up_h) @ state[f"{prefix}.mlp.down_proj.weight"].astype(np.float64).T
        hidden = residual + _ref_centered_rms(
            ffn, state[f"{prefix}.post_feedforward_layernorm.weight"], text.post_norm_eps
        )

    hidden = _ref_rms_scaled(hidden, state["model.language_model.norm.weight"], text.rms_norm_eps)
    logits = hidden @ state["lm_head.weight"].astype(np.float64).T
    logits = logits * text.output_multiplier
    cap = text.final_logit_softcapping
    if cap is not None:
        logits = cap * np.tanh(logits / cap)
    return logits


def _ref_vision_features(config, state, pixel_values, grid_thw):
    """Run the reference Muse-Glimmer vision tower.

    Args:
        config: Composite Muse-Glimmer configuration.
        state: HF-named ``state_dict`` as NumPy arrays.
        pixel_values: Packed patches of shape ``(total_patches, patch_features)``.
        grid_thw: ``(num_images, 3)`` patch-grid dimensions.

    Returns:
        np.ndarray: Merged vision tokens, shape ``(total / merge**2, hidden * merge**2)``.
    """
    vision = config.vision_config
    heads = vision.num_attention_heads
    head_dim = vision.hidden_size // heads
    merge = vision.merge_size

    cu_seqlens = mg.get_vision_cu_seqlens(grid_thw)
    window_index, cu_window = mg.get_vision_window_index(grid_thw, 1, vision.window_size, vision.patch_size)

    hidden = pixel_values.astype(np.float64) @ state[
        "model.vision_tower.patch_embedder.patch_embedding.weight"
    ].astype(np.float64).T
    indices, weights = mg.get_vision_interpolation_indices_and_weights(grid_thw, vision.pos_emb_height, 1)
    table = state["model.vision_tower.patch_embedder.position_embedding_table.weight"].astype(np.float64)
    hidden = hidden + (table[indices] * weights[:, :, None].astype(np.float64)).sum(axis=1)

    hidden = _ref_layer_norm(
        hidden,
        state["model.vision_tower.ln_pre.weight"],
        state["model.vision_tower.ln_pre.bias"],
        vision.layer_norm_eps,
    )
    hidden = hidden[window_index]

    position_ids = mg.get_vision_position_ids(grid_thw, 1)[:, ::-1] + 1
    position_ids = position_ids[window_index]
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (vision.rope_theta ** (np.arange(0, spatial_dim, 2, dtype=np.float64) / spatial_dim))
    freq_w = position_ids[:, 0].astype(np.float64)[:, None] * inv_freq[None, :]
    freq_h = position_ids[:, 1].astype(np.float64)[:, None] * inv_freq[None, :]
    freq = np.concatenate([freq_w, freq_h, freq_w, freq_h], axis=-1)
    cos, sin = np.cos(freq)[:, None, :], np.sin(freq)[:, None, :]

    seq_len = hidden.shape[0]

    def segment_bias(boundaries):
        segment_ids = np.zeros((seq_len,), dtype=np.int32)
        for index in range(len(boundaries) - 1):
            segment_ids[int(boundaries[index]) : int(boundaries[index + 1])] = index
        return np.where(segment_ids[:, None] == segment_ids[None, :], 0.0, -np.inf)

    bias_by_type = {
        "full_attention": segment_bias(cu_seqlens),
        "window_attention": segment_bias(cu_window),
    }

    for layer in range(vision.num_hidden_layers):
        prefix = f"model.vision_tower.layers.{layer}"
        normed = _ref_layer_norm(
            hidden, state[f"{prefix}.norm1.weight"], state[f"{prefix}.norm1.bias"], 1e-5
        )

        def project(name, x=normed, p=prefix):
            weight = state[f"{p}.attn.{name}.weight"].astype(np.float64)
            bias = state[f"{p}.attn.{name}.bias"].astype(np.float64)
            return (x @ weight.T + bias).reshape(seq_len, heads, head_dim)

        query, key, value = project("q_proj"), project("k_proj"), project("v_proj")
        query = query * cos + _ref_rotate_half(query) * sin
        key = key * cos + _ref_rotate_half(key) * sin

        scores = np.einsum("qhd,khd->hqk", query, key) * (head_dim**-0.5)
        scores = scores + bias_by_type[vision.layer_types[layer]][None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        context = np.einsum("hqk,khd->qhd", probs, value).reshape(seq_len, vision.hidden_size)
        attn_out = context @ state[f"{prefix}.attn.proj.weight"].astype(np.float64).T + state[
            f"{prefix}.attn.proj.bias"
        ].astype(np.float64)
        hidden = hidden + attn_out

        normed = _ref_layer_norm(
            hidden, state[f"{prefix}.norm2.weight"], state[f"{prefix}.norm2.bias"], 1e-5
        )
        inner = normed @ state[f"{prefix}.mlp.fc1.weight"].astype(np.float64).T + state[
            f"{prefix}.mlp.fc1.bias"
        ].astype(np.float64)
        inner = _ref_gelu(inner)
        outer = inner @ state[f"{prefix}.mlp.fc2.weight"].astype(np.float64).T + state[
            f"{prefix}.mlp.fc2.bias"
        ].astype(np.float64)
        hidden = hidden + outer

    hidden = hidden[np.argsort(window_index)]
    hidden = _ref_layer_norm(
        hidden,
        state["model.vision_tower.ln_post.weight"],
        state["model.vision_tower.ln_post.bias"],
        vision.layer_norm_eps,
    )

    shuffle_index = mg.get_vision_pixel_shuffle_index(grid_thw, merge)
    hidden = hidden[shuffle_index]
    hidden = hidden.reshape(-1, merge * merge, vision.hidden_size)
    return np.transpose(hidden, (0, 2, 1)).reshape(-1, vision.hidden_size * merge * merge)


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_setup():
    """Build one config + checkpoint + loaded model shared by the parity tests."""
    text = _text_config()
    vision = _vision_config()
    config = _model_config(text, vision, image_token_id=95, video_token_id=96)
    state_dict = _build_state_dict(config, seed=7)
    numpy_state = {key: value.numpy() for key, value in state_dict.items()}
    model = _load_easydel(config, state_dict)
    return config, numpy_state, model


def test_text_logits_match_reference(parity_setup):
    """Text-only logits must match the NumPy reference of the HF implementation.

    Covers gated attention, the scale-less QK-norm plus query scale, the
    sliding/full attention schedule, per-layer NoPE, the sandwich norms and the
    ``output_multiplier`` + tanh soft-cap on the head — all driven through the
    HF-named checkpoint, so the fused-projection reform rules are covered too.
    """
    config, state, model = parity_setup
    rng = np.random.default_rng(3)
    input_ids = rng.integers(0, config.text_config.vocab_size - 3, size=(2, 24), dtype=np.int32)

    expected = _ref_text_logits(config, state, input_ids)
    actual = np.asarray(model(input_ids=jnp.asarray(input_ids)).logits, dtype=np.float64)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_scanned_text_stack_matches_reference():
    """``scan_layers=True`` must produce the same logits as the unrolled stack.

    The scanned path gathers each layer's RoPE table by a traced index instead
    of a Python one, so it needs its own check.
    """
    text = _text_config(scan_layers=True)
    config = _model_config(text, _vision_config(scan_layers=True), image_token_id=95, video_token_id=96)
    state_dict = _build_state_dict(config, seed=7)
    state = {key: value.numpy() for key, value in state_dict.items()}
    model = _load_easydel(config, state_dict)

    rng = np.random.default_rng(3)
    input_ids = rng.integers(0, config.text_config.vocab_size - 3, size=(2, 24), dtype=np.int32)

    expected = _ref_text_logits(config, state, input_ids)
    actual = np.asarray(model(input_ids=jnp.asarray(input_ids)).logits, dtype=np.float64)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_softcap_bounds_logits(parity_setup):
    """The head must keep logits strictly inside the soft-cap band."""
    config, _, model = parity_setup
    cap = config.text_config.final_logit_softcapping
    rng = np.random.default_rng(11)
    input_ids = rng.integers(0, config.text_config.vocab_size - 3, size=(1, 16), dtype=np.int32)
    logits = np.asarray(model(input_ids=jnp.asarray(input_ids)).logits)
    assert np.all(np.abs(logits) < cap)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_vision_tower_matches_reference(scan_layers):
    """Vision-tower output must match the NumPy reference, unrolled and scanned.

    Covers patch embedding, bilinear position-grid resampling, the window
    permutation, the interleaved 2-D RoPE, the window/full attention schedule
    and the pixel-shuffle head. Both layer-stack paths are checked because the
    scanned path selects each layer's block-diagonal bias by a traced index
    rather than a Python one.
    """
    text = _text_config(num_hidden_layers=4, scan_layers=scan_layers)
    vision = _vision_config(scan_layers=scan_layers)
    config = _model_config(text, vision, image_token_id=95, video_token_id=96)
    state_dict = _build_state_dict(config, seed=7)
    state = {key: value.numpy() for key, value in state_dict.items()}
    model = _load_easydel(config, state_dict)

    grid = np.array([[1, 8, 8], [1, 4, 6]], dtype=np.int64)
    total = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_features = vision.patch_temporal * vision.in_channels * vision.patch_size**2
    pixel_values = np.random.default_rng(5).normal(size=(total, patch_features)).astype(np.float32)

    expected = _ref_vision_features(config, state, pixel_values, grid)
    actual = np.asarray(
        model.get_vision_tower()(pixel_values=jnp.asarray(pixel_values), grid_thw=grid),
        dtype=np.float64,
    )

    assert actual.shape == expected.shape == (total // vision.merge_size**2, vision.out_hidden_size)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param([[2, 8, 8]], id="two-frame-video"),
        pytest.param([[3, 4, 6], [1, 8, 8]], id="video-plus-image"),
    ],
)
def test_vision_tower_handles_temporal_grids(parity_setup, grid):
    """Multi-frame and mixed video/image batches must match the reference.

    Frames repeat the spatial position ids, offset the window permutation and
    the pixel-shuffle gather per frame, and extend ``cu_seqlens`` one entry per
    frame — none of which a single-frame grid exercises.
    """
    config, state, model = parity_setup
    vision = config.vision_config
    grid = np.array(grid, dtype=np.int64)

    total = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_features = vision.patch_temporal * vision.in_channels * vision.patch_size**2
    pixel_values = np.random.default_rng(5).normal(size=(total, patch_features)).astype(np.float32)

    expected = _ref_vision_features(config, state, pixel_values, grid)
    actual = np.asarray(
        model.get_video_features(jnp.asarray(pixel_values), grid),
        dtype=np.float64,
    )

    # `get_video_features` also runs the adapter + projection, so compare the
    # tower output separately and only check the shape contract end to end.
    tower = np.asarray(
        model.get_vision_tower()(pixel_values=jnp.asarray(pixel_values), grid_thw=grid),
        dtype=np.float64,
    )
    assert tower.shape == expected.shape == (total // vision.merge_size**2, vision.out_hidden_size)
    np.testing.assert_allclose(tower, expected, rtol=RTOL, atol=ATOL)
    assert actual.shape == (total // vision.merge_size**2, config.text_config.hidden_size)
    assert np.isfinite(actual).all()


def test_image_features_replace_placeholder_positions(parity_setup):
    """Projected vision features must land exactly on the placeholder positions.

    Non-placeholder positions must keep their (normalized) text embedding, and
    placeholder positions must carry the normalized projected vision features in
    sequence order.
    """
    config, _, model = parity_setup
    vision = config.vision_config
    grid = np.array([[1, 4, 4]], dtype=np.int64)
    total = 16
    patch_features = vision.patch_temporal * vision.in_channels * vision.patch_size**2
    pixel_values = jnp.asarray(np.random.default_rng(9).normal(size=(total, patch_features)).astype(np.float32))
    num_image_tokens = total // vision.merge_size**2

    seq_len = num_image_tokens + 6
    input_ids = np.full((1, seq_len), 4, dtype=np.int32)
    placeholders = slice(2, 2 + num_image_tokens)
    input_ids[0, placeholders] = config.image_token_id

    image_features = model.get_image_features(pixel_values, grid)
    embeds = np.asarray(
        model.compute_embedding(jnp.asarray(input_ids), image_features=image_features),
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        embeds[0, placeholders],
        np.asarray(image_features, dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
    )
    # Text positions keep the plain normalized token embedding.
    text_only = np.asarray(model.compute_embedding(jnp.asarray(input_ids)), dtype=np.float64)
    keep = np.ones((seq_len,), dtype=bool)
    keep[placeholders] = False
    np.testing.assert_allclose(embeds[0, keep], text_only[0, keep], rtol=RTOL, atol=ATOL)


def test_tensor_parallel_layout_round_trips():
    """A tp=2 mesh must reproduce the tp=1 result from the same checkpoint.

    Fused column-parallel weights are rank-interleaved on the TP axis, so a
    layout that describes its segments incorrectly silently scrambles the
    projection at ``tp > 1`` while staying correct at ``tp = 1``. This checks
    both fused layouts introduced by this family — the four-segment
    ``[Q | gate | K | V]`` attention projection and the vision tower's
    ``[Q | K | V]`` — by loading one HF checkpoint onto both meshes.
    """

    def build(sharding_axis_dims):
        text = _text_config(num_hidden_layers=4, sharding_axis_dims=sharding_axis_dims)
        vision = _vision_config(sharding_axis_dims=sharding_axis_dims)
        config = _model_config(text, vision, image_token_id=95, video_token_id=96)
        state_dict = _build_state_dict(config, seed=7)
        snapshot = {key: value.numpy() for key, value in state_dict.items()}
        return config, snapshot, _load_easydel(config, state_dict)

    input_ids = np.random.default_rng(3).integers(0, 90, size=(2, 24), dtype=np.int32)

    # (pp, dp, fsdp, ep, tp, sp)
    config, state, single = build((1, 1, 1, 1, 1, 1))
    _, _, parallel = build((1, 1, 1, 1, 2, 1))

    expected = _ref_text_logits(config, state, input_ids)
    single_logits = np.asarray(single(input_ids=jnp.asarray(input_ids)).logits, dtype=np.float64)
    parallel_logits = np.asarray(parallel(input_ids=jnp.asarray(input_ids)).logits, dtype=np.float64)

    np.testing.assert_allclose(single_logits, expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(parallel_logits, expected, rtol=RTOL, atol=ATOL)

    vision = config.vision_config
    grid = np.array([[1, 8, 8]], dtype=np.int64)
    total = 64
    patch_features = vision.patch_temporal * vision.in_channels * vision.patch_size**2
    pixel_values = np.random.default_rng(5).normal(size=(total, patch_features)).astype(np.float32)

    expected_vision = _ref_vision_features(config, state, pixel_values, grid)
    for name, model in (("tp=1", single), ("tp=2", parallel)):
        actual = np.asarray(
            model.get_vision_tower()(pixel_values=jnp.asarray(pixel_values), grid_thw=grid),
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected_vision, rtol=RTOL, atol=ATOL, err_msg=f"vision tower at {name}")


def test_fused_attention_layout_sources_match_hf_names(parity_setup):
    """The fused attention layout must fuse exactly HF's four projection tensors."""
    _, _, model = parity_setup
    attention = model.get_language_model().layers[0].self_attn
    layout = attention.qkv_proj.layout
    assert [segment.name for segment in layout.segments] == ["q", "gate", "k", "v"]
    assert [segment.source_prefix for segment in layout.segments] == [
        "q_proj",
        "gate_proj",
        "k_proj",
        "v_proj",
    ]
    reform = attention.reform_param
    assert reform["qkv_proj.weight$"]["sources"] == (
        "q_proj.weight",
        "gate_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
    )


def test_nope_layers_receive_an_identity_rope_table(parity_setup):
    """Layers with ``layer_rope_theta == 0`` must get a no-op rotation.

    Every layer keeps a rotary module so the stack stays scannable, so NoPE has
    to show up in the frequency table instead: ``cos = 1``, ``sin = 0``, which
    leaves Q/K untouched. RoPE layers must get a genuinely rotating table.
    """
    config, _, model = parity_setup
    text = config.text_config
    tables, selector = model.get_language_model().rope_frequency_bank
    tables = np.asarray(tables, dtype=np.float64)
    selector = np.asarray(selector)

    assert any(theta == 0 for theta in text.layer_rope_theta), "test config must include NoPE layers"
    assert selector.shape == (text.num_hidden_layers,)

    half = tables.shape[-1] // 2
    for index, theta in enumerate(text.layer_rope_theta):
        table = tables[selector[index]]
        cos, sin = table[:, :half], table[:, half:]
        if theta:
            assert not np.allclose(sin, 0.0), f"layer {index} should rotate"
        else:
            np.testing.assert_array_equal(cos, np.ones_like(cos))
            np.testing.assert_array_equal(sin, np.zeros_like(sin))
            # NoPE layers are exactly the full-attention layers in this schedule.
            assert text.layer_types[index] == "full_attention"


def test_cached_decode_matches_full_forward():
    """A prefill + one-token decode must reproduce the full forward's last logits.

    Exercises the KV-cache path across the mixed sliding/full and RoPE/NoPE
    layer schedule, including the vision-aware generation plumbing that carries
    the cache between steps.

    The window is left non-binding for the probe sequence (as it is in the
    released config, where ``sliding_window`` is 2048): EasyDeL's shared
    cached-prefill path diverges from a plain forward once the window actually
    binds, which reproduces on other sliding-window families too and is
    therefore not exercised here.
    """
    text = _text_config(num_hidden_layers=8, sliding_window=256)
    config = _model_config(text, _vision_config(), image_token_id=95, video_token_id=96)
    model = _load_easydel(config, _build_state_dict(config, seed=4))

    seq_len = 12
    input_ids = np.random.default_rng(0).integers(0, text.vocab_size - 3, size=(1, seq_len), dtype=np.int32)
    full = np.asarray(model(input_ids=jnp.asarray(input_ids)).logits, dtype=np.float64)

    max_length = seq_len + 4
    inputs = model.prepare_inputs_for_generation(
        input_ids=jnp.asarray(input_ids[:, :-1]),
        max_length=max_length,
        pad_token_id=0,
        attention_mask=jnp.ones((1, seq_len - 1), dtype="i4"),
    )
    inputs["past_key_values"] = model.init_cache(batch_size=1, max_length=max_length, pad_token_id=0)

    prefill = model(
        input_ids=jnp.asarray(input_ids[:, :-1]),
        **{key: value for key, value in inputs.items() if key != "input_ids"},
    )
    np.testing.assert_allclose(
        np.asarray(prefill.logits, dtype=np.float64), full[:, :-1], rtol=RTOL, atol=ATOL
    )

    next_inputs = model.update_inputs_for_generation(prefill, dict(inputs))
    step = model(
        input_ids=jnp.asarray(input_ids[:, -1:]),
        **{key: value for key, value in next_inputs.items() if key != "input_ids"},
    )
    decoded = np.asarray(step.logits, dtype=np.float64)
    assert decoded.shape == (1, 1, text.vocab_size)
    np.testing.assert_allclose(decoded[:, -1], full[:, -1], rtol=RTOL, atol=ATOL)


def test_sliding_window_span():
    """Pin the number of tokens a sliding layer actually attends to.

    Measured by dependency: for a single-layer causal model, perturbing the
    token at position ``p`` can only change outputs at positions ``p`` through
    ``p + span - 1``.

    EasyDeL's shared attention path treats ``sliding_window`` as a per-side
    radius, so the span is ``sliding_window + 1``; HuggingFace's
    ``sliding_window_overlay`` treats it as a span of ``sliding_window`` tokens.
    This port forwards ``config.sliding_window`` unchanged, matching every other
    sliding-window family in the zoo. If the shared convention is ever aligned
    with HuggingFace, this test — and the reference masks above — must move
    together.
    """
    window = 4
    text = _text_config(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        layer_rope_theta=[0.0],
        sliding_window=window,
    )
    config = _model_config(text, _vision_config(), image_token_id=95, video_token_id=96)
    model = _load_easydel(config, _build_state_dict(config, seed=2))

    seq_len = 16
    base_ids = np.full((1, seq_len), 3, dtype=np.int32)
    perturbed = base_ids.copy()
    position = 5
    perturbed[0, position] = 7

    base = np.asarray(model(input_ids=jnp.asarray(base_ids)).logits, dtype=np.float64)
    changed = np.asarray(model(input_ids=jnp.asarray(perturbed)).logits, dtype=np.float64)
    affected = np.where(np.abs(base - changed).max(axis=-1)[0] > 1e-6)[0]

    assert affected[0] == position, "a causal model must not change outputs before the perturbed token"
    span = int(affected[-1] - position + 1)
    assert span == window + 1, (
        f"expected a {window + 1}-token span for sliding_window={window} "
        f"(EasyDeL radius convention), measured {span}"
    )


def test_window_index_partitions_all_patches():
    """The window permutation must be a true permutation with consistent boundaries."""
    vision = _vision_config(pos_emb_height=4, pos_emb_width=4, patch_size=2)
    grid = np.array([[1, 8, 8], [2, 4, 6]], dtype=np.int64)
    total = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())

    window_index, cu_window = mg.get_vision_window_index(grid, 1, vision.window_size, vision.patch_size)
    assert np.array_equal(np.sort(window_index), np.arange(total))
    assert cu_window[0] == 0
    assert cu_window[-1] == total
    assert np.all(np.diff(cu_window) > 0), "empty windows must be collapsed"

    cu_seqlens = mg.get_vision_cu_seqlens(grid)
    assert cu_seqlens[0] == 0
    assert cu_seqlens[-1] == total


def test_layer_schedules_follow_reference_pattern():
    """Default text/vision layer schedules must match the reference derivations."""
    text = _text_config(num_hidden_layers=12)
    expected_text = [
        "full_attention" if (12 - 1 - i) % 4 == 0 else "sliding_attention" for i in range(12)
    ]
    assert text.layer_types == expected_text
    assert text.layer_rope_theta == [0.0 if t == "full_attention" else text.rope_theta for t in expected_text]

    vision = _vision_config(num_hidden_layers=10)
    expected_vision = [
        "full_attention" if (i + 1) % 4 == 0 or i == 9 else "window_attention" for i in range(10)
    ]
    assert vision.layer_types == expected_vision


def test_config_round_trips_through_save_pretrained(tmp_path):
    """Saving and reloading the config must preserve both layer schedules.

    The vision tower's ``"window_attention"`` schedule is validated against a
    vision-specific vocabulary rather than the HuggingFace-wide allow-list, so
    the save path (which re-runs the class validators) and a non-default
    schedule both need covering.
    """
    vision = _vision_config(num_hidden_layers=6)
    text = _text_config(num_hidden_layers=6)
    config = _model_config(text, vision, image_token_id=95, video_token_id=96)

    config.save_pretrained(str(tmp_path))
    reloaded = ed.MuseGlimmerConfig.from_pretrained(str(tmp_path))

    assert reloaded.vision_config.layer_types == vision.layer_types
    assert reloaded.text_config.layer_types == text.layer_types
    assert reloaded.text_config.layer_rope_theta == text.layer_rope_theta
    assert reloaded.text_config.sliding_window == text.sliding_window
    assert reloaded.text_config.qk_scale_factor == text.qk_scale_factor
    assert reloaded.text_config.output_multiplier == text.output_multiplier
    assert reloaded.text_config.post_norm_eps == text.post_norm_eps
    assert reloaded.text_config.final_logit_softcapping == text.final_logit_softcapping

    # A non-default vision schedule must survive rather than be re-derived.
    custom = ["full_attention"] * 6
    assert ed.MuseGlimmerVisionConfig(num_hidden_layers=6, layer_types=custom).layer_types == custom

    with pytest.raises(ValueError):
        ed.MuseGlimmerVisionConfig(num_hidden_layers=6, layer_types=["not_a_layer_type"] * 6)


def test_registered_in_shared_model_type_tables():
    """The family must be listed wherever the stack gates on ``model_type``.

    ``MuseGlimmerForConditionalGeneration`` matches neither the
    ``forimagetexttotext`` nor the ``vision2seq`` architecture heuristic, so the
    HF->EasyDeL converter only routes it to the image-text-to-text task via the
    explicit table. eSurge separately keys flat-patch preprocessing off
    ``model_type``, and Muse-Glimmer consumes pre-packed patches.
    """
    from easydel.inference.esurge.multimodal.manager import MultiModalManager
    from easydel.scripts.convert_hf_to_easydel import IMAGE_TEXT_TO_TEXT_MODEL_TYPES

    assert "muse_glimmer" in IMAGE_TEXT_TO_TEXT_MODEL_TYPES

    source = inspect.getsource(MultiModalManager._supports_flat_patch_inputs)
    assert '"muse_glimmer"' in source, "muse_glimmer must be treated as a flat-patch VLM"


def test_vision_config_exposes_shared_preprocessing_aliases():
    """The vision config must answer the attribute names shared code probes.

    The stored fields keep HuggingFace's spelling (``merge_size`` /
    ``patch_temporal``) so ``config.json`` round-trips, while eSurge's
    patchifier reads ``spatial_merge_size`` / ``temporal_patch_size``. If the
    aliases disappear, ``getattr(..., 1)`` silently yields the wrong grid.
    """
    vision = _vision_config(merge_size=2, patch_temporal=2)
    assert vision.spatial_merge_size == vision.merge_size == 2
    assert vision.temporal_patch_size == vision.patch_temporal == 2


def test_mask_details_track_layer_types():
    """``get_mask_details`` must mark exactly the sliding layers as sliding."""
    text = _text_config(num_hidden_layers=8, sliding_window=13)
    details = text.get_mask_details()
    assert len(details) == 8
    for index, layer_type in enumerate(text.layer_types):
        detail = details[index]
        assert detail.size == 13
        expected = ed.infra.utils.AttnMaskType.SLIDING if layer_type == "sliding_attention" else (
            ed.infra.utils.AttnMaskType.FULL
        )
        assert detail.mask_type == expected


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
