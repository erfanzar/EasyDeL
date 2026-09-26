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

"""Loading a saved HF checkpoint into EasyDeL must reproduce HF's logits.

Goes through the real loading paths (config.json, safetensors with per-expert
keys, reform/fusion rules): ``from_pretrained(from_torch=True)`` in both torch
load modes and the ``huggingface_to_easydel_sequential`` converter followed by
a native load. Compares with a relative-L2 bound (correct loads land near 1e-3
on TPU): the family testers' ``atol=0.125`` is loose enough to hide a wrong
RoPE pairing (~3%) or router combine weights taken from the bias-corrected
scores (~1%). Expert widths are chosen with ``2 * moe_intermediate != hidden``
so a transposed expert weight fails on shape instead of passing silently.
"""

import json

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers
from easydel.modules.kimi_linear.modeling_kimi_linear import KimiLinearModel
from easydel.operations.kernels.kda import fused_kda_gate, fused_kda_gate_per_channel

LOAD_KWARGS = dict(
    dtype=jnp.float32,
    param_dtype=jnp.float32,
    precision=jax.lax.Precision.HIGHEST,
    sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    auto_shard_model=True,
)


def _perturbed(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.add_(0.02 * torch.randn_like(p))
        # Routers keep their selection bias in a buffer; a zero bias would hide
        # combine weights taken from the biased scores.
        for name, b in model.named_buffers():
            if "e_score_correction_bias" in name:
                b.copy_(torch.randn_like(b))
    return model.float().eval()


def _hf_logits(hf_model, ids):
    with torch.no_grad():
        return hf_model(input_ids=torch.from_numpy(ids), use_cache=False).logits.float().numpy()


def _easydel_logits(model, ids):
    with model.mesh:
        return np.asarray(model(input_ids=jnp.asarray(ids, jnp.int32)).logits, np.float32)


def _assert_close(got, want):
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 5e-3, f"logits rel-L2 {rel:.3e}"
    assert np.mean(got.argmax(-1) == want.argmax(-1)) > 0.97


def _load(hf_model, tmp_path, load_mode, **config_kwargs):
    """Save ``hf_model`` and load it back into EasyDeL through ``load_mode``."""
    source = tmp_path / "hf"
    hf_model.save_pretrained(source, safe_serialization=True)
    config_kwargs = ed.EasyDeLBaseConfigDict(attn_mechanism="vanilla", attn_dtype=jnp.float32, **config_kwargs)
    if load_mode != "sequential":
        return ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(source),
            from_torch=True,
            torch_load_mode=load_mode,
            config_kwargs=config_kwargs,
            **LOAD_KWARGS,
        )
    native = tmp_path / "easydel"
    _, module_class = ed.get_modules_by_type(hf_model.config.model_type, ed.TaskType.CAUSAL_LM)
    module_class.huggingface_to_easydel_sequential(
        pretrained_model_name_or_path=str(source),
        save_directory=str(native),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        verbose=False,
    )
    return ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(native), config_kwargs=config_kwargs, **LOAD_KWARGS
    )


def _deepseek_v3(rope_interleave):
    config = transformers.DeepseekV3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        rope_interleave=rope_interleave,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return _perturbed(transformers.DeepseekV3ForCausalLM(config))


def _ids(hf_model):
    return np.random.default_rng(0).integers(3, hf_model.config.vocab_size, size=(2, 32))


@pytest.mark.parametrize("rope_interleave", [True, False])
@pytest.mark.parametrize("load_mode", ["full", "streaming", "sequential"])
def test_deepseek_v3_matches_hf(tmp_path, rope_interleave, load_mode):
    hf_model = _deepseek_v3(rope_interleave)
    ids = _ids(hf_model)
    _assert_close(_easydel_logits(_load(hf_model, tmp_path, load_mode), ids), _hf_logits(hf_model, ids))


@pytest.mark.parametrize("load_mode", ["full", "streaming", "sequential"])
def test_qwen3_moe_matches_hf(tmp_path, load_mode):
    config = transformers.Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    hf_model = _perturbed(transformers.Qwen3MoeForCausalLM(config))
    ids = _ids(hf_model)
    _assert_close(_easydel_logits(_load(hf_model, tmp_path, load_mode), ids), _hf_logits(hf_model, ids))


def test_native_deepseek_checkpoint_without_rope_interleave_keeps_split_half(tmp_path):
    """A native config.json saved before ``rope_interleave`` existed loads with split-half RoPE."""
    hf_model = _deepseek_v3(rope_interleave=False)
    ids = _ids(hf_model)
    want = _hf_logits(hf_model, ids)
    _load(hf_model, tmp_path, "sequential")

    config_path = tmp_path / "easydel" / "config.json"
    saved = json.loads(config_path.read_text())
    assert saved["rope_interleave"] is False  # new native saves record the key
    del saved["rope_interleave"]
    config_path.write_text(json.dumps(saved))

    legacy = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path / "easydel"),
        config_kwargs=ed.EasyDeLBaseConfigDict(attn_mechanism="vanilla", attn_dtype=jnp.float32),
        **LOAD_KWARGS,
    )
    assert legacy.config.rope_interleave is False
    _assert_close(_easydel_logits(legacy, ids), want)

    overridden = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path / "easydel"),
        config_kwargs=ed.EasyDeLBaseConfigDict(attn_mechanism="vanilla", rope_interleave=True),
        **LOAD_KWARGS,
    )
    assert overridden.config.rope_interleave is True


def test_legacy_per_head_kda_gate_leaves_are_widened():
    """Per-head ``f_b_proj``/``dt_bias`` from older Kimi-Linear saves widen to the same per-channel decay."""
    heads, head_dim, rank = 2, 4, 3
    key_fb = ("parameters", "model", "layers", 0, "self_attn", "f_b_proj", "weight")
    key_dt = ("parameters", "model", "layers", 0, "self_attn", "dt_bias")
    rng = np.random.default_rng(0)
    old_fb = rng.normal(size=(rank, heads)).astype(np.float32)
    old_dt = rng.normal(size=(heads,)).astype(np.float32)
    expected = {
        key_fb: jax.ShapeDtypeStruct((rank, heads * head_dim), jnp.float32),
        key_dt: jax.ShapeDtypeStruct((heads * head_dim,), jnp.float32),
    }
    state = KimiLinearModel._upgrade_legacy_native_state(None, {key_fb: old_fb, key_dt: old_dt}, expected)
    assert state[key_fb].shape == expected[key_fb].shape and state[key_dt].shape == expected[key_dt].shape

    x = rng.normal(size=(1, 5, rank)).astype(np.float32)
    a_log = rng.normal(size=(heads,)).astype(np.float32)
    old_decay = fused_kda_gate(x @ old_fb, a_log, old_dt)  # (1, 5, heads)
    new_decay = fused_kda_gate_per_channel(x @ np.asarray(state[key_fb]), a_log, state[key_dt], lower_bound=None)
    np.testing.assert_allclose(
        np.asarray(new_decay), np.repeat(np.asarray(old_decay)[..., None], head_dim, -1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("load_mode", ["full", "streaming", "sequential"])
def test_mixtral_matches_hf(tmp_path, load_mode):
    """Mixtral checkpoints keep ``block_sparse_moe.experts.<i>.w1/w3/w2`` (renamed + merged on load)."""
    config = transformers.MixtralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=8,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    hf_model = _perturbed(transformers.MixtralForCausalLM(config))
    ids = _ids(hf_model)
    _assert_close(_easydel_logits(_load(hf_model, tmp_path, load_mode), ids), _hf_logits(hf_model, ids))


@pytest.mark.parametrize("load_mode", ["full", "streaming", "sequential"])
def test_gpt2_matches_hf(tmp_path, load_mode):
    """GPT-2 keeps dropout at p=0.1 in its config: a loaded model must be in inference mode."""
    config = transformers.GPT2Config(vocab_size=256, n_positions=128, n_embd=64, n_layer=2, n_head=4)
    hf_model = _perturbed(transformers.GPT2LMHeadModel(config))
    ids = _ids(hf_model)
    model = _load(hf_model, tmp_path, load_mode)
    assert not model._spx_training
    _assert_close(_easydel_logits(model, ids), _hf_logits(hf_model, ids))


def test_trainer_switches_loaded_models_to_training_mode(tmp_path):
    from easydel.trainers.base_trainer import BaseTrainer

    hf_model = _perturbed(
        transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=64, n_embd=32, n_layer=1, n_head=2))
    )
    state = _load(hf_model, tmp_path, "full").to_state()
    assert not state.model._spx_training
    trained = BaseTrainer._in_training_mode(state)
    assert trained.model._spx_training and trained.model.transformer.dropout._spx_training
    assert trained.graphstate is state.graphstate


def _gpt_oss():
    config = transformers.GptOssConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=8,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    return _perturbed(transformers.GptOssForCausalLM(config))


def _qwen3_moe():
    config = transformers.Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return _perturbed(transformers.Qwen3MoeForCausalLM(config))


@pytest.mark.parametrize("moe_method", ["fused_moe", "standard_moe", "dense_moe"])
@pytest.mark.parametrize("build", [_gpt_oss, _qwen3_moe], ids=["gpt_oss", "qwen3_moe"])
def test_moe_block_matches_hf_in_float32(tmp_path, build, moe_method):
    """An f32 MoE block reproduces HF at f32 accuracy on every dispatch path.

    Logit-level parity hides this: the expert projections used to leave the
    grouped matmul as bf16 (and GPT-OSS's block ran at the bf16 default dtype),
    a ~4e-3 per-block error, and the standard path combined experts with raw
    router logits instead of softmax scores.
    """
    hf_model = build()
    model = _load(hf_model, tmp_path, "streaming", moe_method=moe_method)
    x = np.random.default_rng(1).standard_normal((1, 16, hf_model.config.hidden_size)).astype(np.float32)
    with torch.no_grad():
        want = hf_model.model.layers[0].mlp(torch.from_numpy(x))
        want = (want[0] if isinstance(want, tuple) else want).numpy()
    with model.mesh:
        got = model.model.layers[0].mlp(jnp.asarray(x))
        got = np.asarray(got[0] if isinstance(got, tuple) else got)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
