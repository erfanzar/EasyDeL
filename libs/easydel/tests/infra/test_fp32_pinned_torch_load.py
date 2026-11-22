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

"""A half-precision HF load of an fp32-pinned family keeps only the pinned leaves in float32 (TPU)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

BIAS = "e_score_correction_bias"


@pytest.fixture(scope="module")
def glm4_moe_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Glm4MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        vocab_size=256,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.Glm4MoeForCausalLM(config).to(torch.float32)
    bias = {}
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith(BIAS):
                # 1 + 2^-12 steps are below bf16 resolution, so a bf16 round trip would erase them.
                values = 1.0 + torch.arange(buffer.numel(), dtype=torch.float32) * 2.0**-12
                buffer.copy_(values)
                bias[name] = values.numpy()
    assert bias
    path = tmp_path_factory.mktemp("tiny_glm4_moe")
    model.save_pretrained(path)
    return str(path), bias


def test_bf16_load_keeps_pinned_bias_exact_and_streams_the_rest_in_bf16(glm4_moe_checkpoint, monkeypatch):
    import easydel as ed
    from easydel.infra.mixins.bridge import EasyBridgeMixin

    path, bias = glm4_moe_checkpoint
    torch_dtypes: dict[str, str] = {}
    original = EasyBridgeMixin._load_full_torch_checkpoint.__func__

    def recording_loader(cls, *args, **kwargs):
        out = original(cls, *args, **kwargs)
        torch_dtypes.update({key: str(value.dtype) for key, value in out[2].items()})
        return out

    monkeypatch.setattr(EasyBridgeMixin, "_load_full_torch_checkpoint", classmethod(recording_loader))
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        auto_shard_model=True,
    )

    pinned = {key for key in torch_dtypes if key.endswith(BIAS)}
    assert pinned
    assert {torch_dtypes[key] for key in pinned} == {"torch.float32"}
    assert {dtype for key, dtype in torch_dtypes.items() if key not in pinned} == {"torch.bfloat16"}

    leaves = jax.tree_util.tree_flatten_with_path(model.split_module()[1])[0]
    loaded = [np.asarray(leaf) for p, leaf in leaves if BIAS in jax.tree_util.keystr(p)]
    others = {str(leaf.dtype) for p, leaf in leaves if BIAS not in jax.tree_util.keystr(p)}
    assert others == {"bfloat16"}
    assert len(loaded) == len(bias)
    expected = sorted(np.asarray(v, np.float32).tobytes() for v in bias.values())
    assert all(arr.dtype == np.float32 for arr in loaded)
    assert sorted(arr.reshape(-1).tobytes() for arr in loaded) == expected
