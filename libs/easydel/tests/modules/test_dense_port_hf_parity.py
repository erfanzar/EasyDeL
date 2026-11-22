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

"""Dense-family ports must reproduce HF logits from a saved HF checkpoint.

Each case builds a tiny HF model, perturbs every parameter (norm scales and
biases by a large amount, since default init -- ones / zeros -- hides norm and
bias bugs), saves it, loads it back through ``from_pretrained(from_torch=True)``
(full and streaming) or the sequential converter, and compares logits with a
relative-L2 bound. The configs are chosen so each previously-broken code path is
exercised: fused QKV packings (Falcon grouped / per-head, GPT-NeoX per-head),
mean-centred norms and interleaved RoPE (Cohere), per-head QK norms (StableLM,
Phi), attention soft-capping (Gemma2), bidirectional attention (EmbeddingGemma),
``use_sliding_window=False`` (Qwen2/Qwen3), NoPE gating (EXAONE-4), ALiBi scaling
(Falcon-RW), FFN width (OPT), LM-head bias (GPT-J), tied heads (GPT-2) and
``clip_qkv`` (MPT). Sliding windows are kept wider than the prompt so the
window-size convention does not enter the comparison.
"""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers

LOAD_KWARGS = dict(
    dtype=jnp.float32,
    param_dtype=jnp.float32,
    precision=jax.lax.Precision.HIGHEST,
    sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    auto_shard_model=True,
)
SEQ_LEN = 32


def _perturbed(model, matrix_scale=0.02, vector_scale=0.3):
    """Perturb every float parameter; 1-D params (norm scales, biases) by a lot."""
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.add_((vector_scale if p.ndim == 1 else matrix_scale) * torch.randn_like(p))
    return model.float().eval()


def _eager(config):
    config._attn_implementation = "eager"
    return config


def _hf_cohere(use_qk_norm):
    config = transformers.CohereConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_qk_norm=use_qk_norm,
        logit_scale=1.0,
    )
    return _perturbed(transformers.CohereForCausalLM(_eager(config)))


def _hf_gemma2():
    config = transformers.Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=128,
        attn_logit_softcapping=1.0,
        final_logit_softcapping=30.0,
    )
    model = _perturbed(transformers.Gemma2ForCausalLM(_eager(config)))
    with torch.no_grad():  # large scores, so the soft cap actually bites
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(8.0)
            layer.self_attn.k_proj.weight.mul_(8.0)
    return model


def _hf_gemma3_bidirectional():
    config = transformers.Gemma3TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
        use_bidirectional_attention=True,
    )
    return _perturbed(transformers.Gemma3ForCausalLM(_eager(config)))


def _hf_qwen2():
    config = transformers.Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_sliding_window=False,
        sliding_window=4,
        max_window_layers=1,
        tie_word_embeddings=False,
    )
    return _perturbed(transformers.Qwen2ForCausalLM(_eager(config)))


def _hf_qwen3():
    config = transformers.Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        use_sliding_window=False,
        sliding_window=4,
        max_window_layers=1,
        tie_word_embeddings=False,
    )
    return _perturbed(transformers.Qwen3ForCausalLM(_eager(config)))


def _hf_stablelm():
    config = transformers.StableLmConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        qk_layernorm=True,
        use_qkv_bias=True,
        partial_rotary_factor=0.25,
    )
    return _perturbed(transformers.StableLmForCausalLM(_eager(config)))


def _hf_falcon(variant):
    common = dict(vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4)
    if variant == "new_decoder_gqa":  # Falcon-40B/180B: grouped per-KV-head packing
        config = transformers.FalconConfig(
            **common,
            num_kv_heads=2,
            new_decoder_architecture=True,
            parallel_attn=True,
            bias=False,
        )
    elif variant == "classic_mha_alibi":  # Falcon-RW: per-head packing + ALiBi
        config = transformers.FalconConfig(
            **common,
            new_decoder_architecture=False,
            multi_query=False,
            parallel_attn=False,
            alibi=True,
            bias=True,
        )
    else:  # Falcon-7B: MQA with ``num_kv_heads`` omitted
        config = transformers.FalconConfig(
            **common,
            new_decoder_architecture=False,
            multi_query=True,
            parallel_attn=True,
            bias=False,
        )
    if variant != "classic_mha_alibi":
        return _perturbed(transformers.FalconForCausalLM(_eager(config)))
    # HF's eager ALiBi path adds the bias twice (once to the scores, once more via the
    # alibi-filled mask); SDPA matches the original Falcon-RW ``(QK^T + alibi) / sqrt(d)``.
    config._attn_implementation = "sdpa"
    model = _perturbed(transformers.FalconForCausalLM(config))
    with torch.no_grad():  # amplify the attention branch so the ALiBi scale is visible in the logits
        for layer in model.transformer.h:
            layer.self_attention.dense.weight.mul_(10.0)
    return model


def _hf_gpt_neox():
    config = transformers.GPTNeoXConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    return _perturbed(transformers.GPTNeoXForCausalLM(_eager(config)))


def _hf_gptj():
    config = transformers.GPTJConfig(vocab_size=256, n_embd=64, n_layer=2, n_head=4, rotary_dim=8, n_positions=128)
    return _perturbed(transformers.GPTJForCausalLM(_eager(config)))


def _hf_opt():
    config = transformers.OPTConfig(
        vocab_size=256,
        hidden_size=64,
        ffn_dim=128,
        word_embed_proj_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
    )
    return _perturbed(transformers.OPTForCausalLM(_eager(config)))


def _hf_exaone4():
    config = transformers.Exaone4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    return _perturbed(transformers.Exaone4ForCausalLM(_eager(config)))


def _hf_gpt2():
    config = transformers.GPT2Config(vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_positions=128)
    return _perturbed(transformers.GPT2LMHeadModel(_eager(config)))


def _hf_phi():
    config = transformers.PhiConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        qk_layernorm=True,
        partial_rotary_factor=0.5,
    )
    return _perturbed(transformers.PhiForCausalLM(_eager(config)))


def _hf_mpt():
    config = transformers.MptConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        expansion_ratio=2,
        max_seq_len=128,
        attn_config={"clip_qkv": 0.05},
    )
    return _perturbed(transformers.MptForCausalLM(_eager(config)))


def _hf_seed_oss():
    config = transformers.SeedOssConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    return _perturbed(transformers.SeedOssForCausalLM(_eager(config)))


CASES = {
    "cohere": lambda: _hf_cohere(use_qk_norm=False),
    "cohere_qk_norm": lambda: _hf_cohere(use_qk_norm=True),
    "gemma2_attn_softcap": _hf_gemma2,
    "gemma3_bidirectional": _hf_gemma3_bidirectional,
    "qwen2_no_sliding": _hf_qwen2,
    "qwen3_no_sliding": _hf_qwen3,
    "stablelm_qk_norm_qkv_bias": _hf_stablelm,
    "falcon_new_decoder_gqa": lambda: _hf_falcon("new_decoder_gqa"),
    "falcon_classic_mha_alibi": lambda: _hf_falcon("classic_mha_alibi"),
    "falcon_classic_mqa": lambda: _hf_falcon("classic_mqa"),
    "gpt_neox": _hf_gpt_neox,
    "gptj_lm_head_bias": _hf_gptj,
    "opt_ffn_dim": _hf_opt,
    "exaone4_no_sliding_rope": _hf_exaone4,
    "gpt2_tied_head": _hf_gpt2,
    "phi_qk_layernorm": _hf_phi,
    "mpt_clip_qkv": _hf_mpt,
    "seed_oss_positions": _hf_seed_oss,
}


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


def _load(hf_model, tmp_path, load_mode):
    """Save ``hf_model`` and load it back into EasyDeL through ``load_mode``."""
    source = tmp_path / "hf"
    hf_model.save_pretrained(source, safe_serialization=True)
    config_kwargs = ed.EasyDeLBaseConfigDict(attn_mechanism="vanilla", attn_dtype=jnp.float32)
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


@pytest.mark.parametrize("load_mode", ["full", "streaming", "sequential"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_dense_port_matches_hf(tmp_path, case, load_mode):
    hf_model = CASES[case]()
    ids = np.random.default_rng(0).integers(3, hf_model.config.vocab_size, size=(2, SEQ_LEN))
    _assert_close(_easydel_logits(_load(hf_model, tmp_path, load_mode), ids), _hf_logits(hf_model, ids))


@pytest.mark.parametrize("case", ["falcon_new_decoder_gqa", "falcon_classic_mha_alibi", "gpt_neox"])
def test_fused_qkv_export_round_trips(tmp_path, case):
    """Export must undo the fused-QKV reorder: to_torch reproduces the HF state dict."""
    hf_model = CASES[case]()
    model = _load(hf_model, tmp_path, "full")
    exported = model.to_torch()
    want = hf_model.state_dict()
    got = exported.state_dict()
    for key, tensor in want.items():
        if "query_key_value" in key:
            torch.testing.assert_close(got[key].float(), tensor.float(), rtol=0, atol=1e-6)


def test_config_sliding_schedules_match_hf():
    """``use_sliding_window=False`` disables the window (Qwen2/Qwen3); the MoE schedules follow HF."""
    common = dict(num_hidden_layers=4, hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
    for ed_cls, hf_cls in [(ed.Qwen2Config, transformers.Qwen2Config), (ed.Qwen3Config, transformers.Qwen3Config)]:
        for use_sliding_window in (False, True):
            kwargs = dict(common, use_sliding_window=use_sliding_window, sliding_window=8, max_window_layers=2)
            ed_config, hf_config = ed_cls(**kwargs), hf_cls(**kwargs)
            assert ed_config.sliding_window == hf_config.sliding_window
            assert list(ed_config.layer_types) == list(hf_config.layer_types)

    moe = dict(common, num_experts=4, num_experts_per_tok=2, sliding_window=8, max_window_layers=3)
    for use_sliding_window in (False, True):
        ed_config = ed.Qwen2MoeConfig(**moe, use_sliding_window=use_sliding_window)
        hf_config = transformers.Qwen2MoeConfig(**moe, use_sliding_window=use_sliding_window)
        assert list(ed_config.layer_types) == list(hf_config.layer_types)
        ed3 = ed.Qwen3MoeConfig(**moe, use_sliding_window=use_sliding_window)
        expected = "sliding_attention" if use_sliding_window else "full_attention"
        assert list(ed3.layer_types) == [expected] * common["num_hidden_layers"]


def test_config_defaults_match_hf():
    """head_dim / tie / KV-head defaults that HF resolves when a hub config omits the key."""
    assert ed.MistralConfig(hidden_size=64, num_attention_heads=4).head_dim == 16
    assert ed.SeedOssConfig(hidden_size=64, num_attention_heads=4).head_dim == 128
    assert ed.GPT2Config().tie_word_embeddings is True
    assert ed.MptConfig().tie_word_embeddings is True
    assert ed.Gemma2Config().attn_logit_softcapping == transformers.Gemma2Config().attn_logit_softcapping
    falcon_7b = ed.FalconConfig(num_attention_heads=71, multi_query=True, new_decoder_architecture=False)
    assert falcon_7b.num_key_value_heads == 1
    falcon_40b = ed.FalconConfig(
        hidden_size=8192, num_attention_heads=128, num_kv_heads=8, new_decoder_architecture=True
    )
    assert falcon_40b.num_key_value_heads == 8


def test_internlm2_grouped_wqkv_reform_matches_remote_code():
    """The ``wqkv`` rule turns the remote code's grouped rows into the runtime ``[Q | K | V]`` split."""
    heads, kv_heads, head_dim, hidden = 4, 2, 16, 64
    groups = heads // kv_heads
    config = ed.InternLM2Config(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        bias=True,
    )
    model = ed.InternLM2ForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=ed.Rngs(0))
    rules = model._get_reform_param()
    weight_rule = next(rule for key, rule in rules.items() if key.endswith("wqkv.weight$"))
    bias_rule = next(rule for key, rule in rules.items() if key.endswith("wqkv.bias$"))

    torch.manual_seed(0)
    weight = torch.randn((heads + 2 * kv_heads) * head_dim, hidden)
    bias = torch.randn((heads + 2 * kv_heads) * head_dim)
    x = torch.randn(3, hidden)

    # Remote code: rearrange(qkv, "b q (h gs d) -> b q h gs d", gs=2 + groups)
    ref = (x @ weight.T + bias).reshape(3, kv_heads, groups + 2, head_dim)
    ref_q = ref[:, :, :groups].reshape(3, heads * head_dim)
    ref_k = ref[:, :, -2].reshape(3, kv_heads * head_dim)
    ref_v = ref[:, :, -1].reshape(3, kv_heads * head_dim)

    ed_weight = weight_rule["splits"][0]["spliter"](weight)  # EasyDeL layout: [in, out]
    ed_bias = bias_rule["splits"][0]["spliter"](bias)
    q, k, v = torch.split(x @ ed_weight + ed_bias, [heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], -1)
    torch.testing.assert_close(q, ref_q)
    torch.testing.assert_close(k, ref_k)
    torch.testing.assert_close(v, ref_v)

    torch.testing.assert_close(weight_rule["inverse_spliter"](torch, ed_weight), weight, rtol=0, atol=0)
    torch.testing.assert_close(bias_rule["inverse_spliter"](torch, ed_bias), bias, rtol=0, atol=0)


def test_falcon_returns_hidden_states_like_hf(tmp_path):
    """``output_hidden_states`` appends the final normed state as one more tuple entry."""
    hf_model = CASES["falcon_new_decoder_gqa"]()
    ids = np.random.default_rng(0).integers(3, hf_model.config.vocab_size, size=(2, SEQ_LEN))
    model = _load(hf_model, tmp_path, "streaming")
    with model.mesh:
        got = model(input_ids=jnp.asarray(ids, jnp.int32), output_hidden_states=True).hidden_states
    with torch.no_grad():
        want = hf_model(input_ids=torch.from_numpy(ids), output_hidden_states=True, use_cache=False).hidden_states
    assert len(got) == len(want)
    np.testing.assert_allclose(np.asarray(got[-1], np.float32), want[-1].numpy(), rtol=5e-3, atol=5e-3)
