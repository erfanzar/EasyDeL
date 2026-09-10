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

"""End-to-end HF-checkpoint conversion test for GLM-5-Next.

Builds a tiny EasyDeL model, writes a fake HF-format checkpoint that mirrors
the published `zai-org/GLM-5.3-Flash` layout (composite `model_type:
"glm5_next"` config with nested `text_config`, flat ``hc_*`` mHC parameter
names, individually-stored MoE experts), runs the real sequential converter
over it, loads the TensorStore result back, and checks the loaded model
reproduces the source model's logits.

Run: ``pytest tests/modules/spmd/test_glm5_next_hf_checkpoint.py`` (CPU trio).
"""

import json

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from safetensors.torch import save_file

# 1-D parameters kept verbatim on export (no transpose, no rename).
_KEEP_AS_IS_SUFFIXES = (
    ".A_log",
    ".dt_bias",
    ".scale",
    ".base",
    ".bias",
    "index_kpool_compress_ape",
    "index_kpool_compress_gate",
    "e_score_correction_bias",
    "o_norm.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_a_layernorm.weight",
    "kv_a_layernorm.weight",
    "k_norm.weight",
    "norm.weight",
)


def _tiny_kwargs():
    return dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=256,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        mlp_layer_types=["dense", "sparse", "dense", "sparse"],
        attention_bias=False,
        attention_dropout=0.0,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        index_kpool_always_select_tail=True,
        layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        swiglu_limit=10.0,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        hc_mult=2,
        hc_eps=1e-6,
        hc_sinkhorn_iters=5,
    )


def _easydel_leaf_to_hf(key: str, array: jax.Array) -> dict[str, np.ndarray]:
    """Convert one EasyDeL parameter leaf into its HF-layout tensor(s).

    Args:
        key: Dotted EasyDeL parameter path (``parameters.`` prefix stripped).
        array: The EasyDeL-side values.

    Returns:
        Mapping of HF key -> torch-layout numpy array. Consolidated MoE expert
        stacks expand into one entry per expert (matching how published HF
        checkpoints store them).
    """
    arr = np.asarray(array)

    # mHC: nested EasyDeL sub-module params -> HF flat layer params. All mHC
    # tensors are raw Parameters: HF orientation matches EasyDeL's, so `fn`
    # (2-D [mix, hc*hidden]) passes through untransposed, like base/scale.
    for ed_name, hf_name in (
        ("attn_hc.fn", "hc_attn_fn"),
        ("attn_hc.base", "hc_attn_base"),
        ("attn_hc.scale", "hc_attn_scale"),
        ("ffn_hc.fn", "hc_ffn_fn"),
        ("ffn_hc.base", "hc_ffn_base"),
        ("ffn_hc.scale", "hc_ffn_scale"),
    ):
        if key.endswith(ed_name):
            return {key[: -len(ed_name)] + hf_name: arr}

    # Forget gate: nested under self_attn.forget_gate; HF keeps it flat.
    if ".forget_gate." in key:
        key = key.replace(".forget_gate.", ".")
        if arr.ndim == 2:
            arr = arr.T
        return {key: arr}

    # Depthwise conv kernels: flax (d, 1, C) -> torch (C, 1, d).
    if key.endswith("_conv1d.weight"):
        return {key: arr.transpose(2, 1, 0)}

    # Embeddings keep their layout.
    if key.endswith("embed_tokens.weight"):
        return {key: arr}

    # Consolidated MoE expert stacks [E, in, out] (down: [E, M, H]) expand to
    # per-expert torch [out, in] tensors under mlp.experts.<i>.<name>.weight.
    if ".experts." in key and arr.ndim == 3:
        base, proj, name = key.rsplit(".", 2)
        return {f"{base}.{i}.{proj}.{name}": arr[i].T for i in range(arr.shape[0])}

    # Everything else: standard torch [out, in] orientation for 2-D weights;
    # 1-D params pass through.
    if arr.ndim == 2 and not any(key.endswith(s) for s in _KEEP_AS_IS_SUFFIXES):
        arr = arr.T
    return {key: arr}


@pytest.fixture(scope="module")
def converted_checkpoint(tmp_path_factory):
    """Tiny model -> fake HF checkpoint dir -> sequential conversion dir."""
    torch = pytest.importorskip("torch")
    config = ed.Glm5NextTextConfig(**_tiny_kwargs())
    # XLA:CPU cannot lower the default ejkernel grouped-matmul path on this
    # host; force the XLA gmm + fsdp batch-sharding recipe (no expert a2a).
    config.moe_force_xla_gmm = True
    config.add_basic_configurations(
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        fsdp_is_ep_bound=False,
        use_sharding_constraint=False,
    )
    model = ed.Glm5NextForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    _, params = spx.export(model)

    hf_state: dict[str, "torch.Tensor"] = {}
    for path, leaf in jax.tree.leaves_with_path(params):
        tup = tuple(str(k.key) for k in path)
        key = ".".join(tup)
        if key.startswith("parameters."):
            key = key[len("parameters.") :]
        if key.startswith("rng."):
            continue
        for hf_key, arr in _easydel_leaf_to_hf(key, leaf).items():
            # The published composite checkpoint nests the language model under
            # model.language_model. (lm_head stays top-level). This exercises
            # the _checkpoint_key_normalizer end-to-end.
            if hf_key.startswith("model."):
                hf_key = "model.language_model." + hf_key[len("model.") :]
            hf_state[hf_key] = torch.from_numpy(np.ascontiguousarray(arr.copy()))

    # Vision-tower noise: the normalizer must drop these (runtime is text-only).
    hf_state["model.visual.blocks.0.attn.proj.weight"] = torch.zeros(4, 4)
    hf_state["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 4, 2, 2)

    source_dir = tmp_path_factory.mktemp("hf_source")
    keys = sorted(hf_state)
    shard_a = {k: hf_state[k] for k in keys[::2]}
    shard_b = {k: hf_state[k] for k in keys[1::2]}
    save_file(shard_a, str(source_dir / "model-00001-of-00002.safetensors"))
    save_file(shard_b, str(source_dir / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard_a}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard_b})
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (source_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    composite_config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "torch_dtype": "bfloat16",
        "text_config": {
            **_tiny_kwargs(),
            "model_type": "glm5_next_text",
            "moe_force_xla_gmm": True,
        },
        "vision_config": {
            "model_type": "glm5_next_vision",
            "depth": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "image_size": 28,
            "patch_size": 14,
            "out_hidden_size": 32,
        },
        "image_token_id": 4,
        "video_token_id": 5,
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (source_dir / "config.json").write_text(json.dumps(composite_config))

    out_dir = tmp_path_factory.mktemp("easydel_out")
    ed.Glm5NextCompositeCausalLM.huggingface_to_easydel_sequential(
        pretrained_model_name_or_path=str(source_dir),
        save_directory=str(out_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        torch_streaming_cache="temp",
        verbose=False,
    )
    return model, out_dir


def test_checkpoint_files_written(converted_checkpoint):
    """The converter produced the standard EasyDeL checkpoint layout."""
    _, out_dir = converted_checkpoint
    for name in ("config.json", "tensorstore_index.json"):
        assert (out_dir / name).exists(), f"missing {name}"
    index = json.loads((out_dir / "tensorstore_index.json").read_text())
    entries = index["prefixes"]["model"]
    assert entries, "tensorstore index has no entries"
    assert all(e["dtype"] == "float32" for e in entries)


def test_roundtrip_logits_match(converted_checkpoint):
    """Loading the converted checkpoint reproduces the source model's logits."""
    source, out_dir = converted_checkpoint
    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(out_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        config_kwargs={"moe_force_xla_gmm": True, "fsdp_is_ep_bound": False},
    )
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 64, size=(8, 8)), dtype="i4")
    src_out = source(input_ids=ids)
    loaded_out = loaded(input_ids=ids)
    assert loaded_out.logits.shape == src_out.logits.shape
    np.testing.assert_allclose(
        np.asarray(loaded_out.logits),
        np.asarray(src_out.logits),
        atol=1e-3,
        rtol=1e-3,
    )
