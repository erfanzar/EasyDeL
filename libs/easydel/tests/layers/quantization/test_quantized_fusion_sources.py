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

"""A ``reform_param`` fusion must decode its sources before concatenating them.

The streaming converter claims a fusion source (HF ``gate_proj``/``up_proj``
feeding EasyDeL's fused ``gate_up_proj``) in its own branch, which returns
before the branch that decodes pre-quantized tensors. On a quantized
checkpoint that made the fusion concatenate *packed codes*, and because the
synthetic fused key has no scale sibling of its own the result was then
dropped instead of raising: DeepSeek-V4-Flash converted to 1242 tensors with
every shared-expert ``gate_up_proj`` simply absent, and nothing said so.

These drive the real ``huggingface_to_easydel_sequential`` against a tiny
block-fp8 Llama, because the property only exists at that seam -- the
non-streaming converter reaches fusion through a different mechanism and was
never affected.
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType
from easydel.layers.quantization.checkpoint._codecs import decode_scaled_elements

pytest.importorskip("safetensors")

HIDDEN = 128
INTERMEDIATE = 128
BLOCK = (128, 128)
LAYERS = 1
VOCAB = 64


def _fp8(shape, rng):
    """A block-fp8 weight plus its scale, as a checkpoint ships them."""
    weight = torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(torch.float8_e4m3fn)
    scale = torch.from_numpy(
        (rng.random((shape[0] // BLOCK[0], shape[1] // BLOCK[1])).astype(np.float32) + 0.5),
    )
    return weight, scale


def _write_checkpoint(directory, rng, mlp_names=("gate_proj", "up_proj", "down_proj")):
    """A one-layer Llama whose MLP gate/up are block-fp8 quantized.

    ``mlp_names`` renames the three MLP tensors so the same fixture can also
    stand in for a vendor-layout checkpoint.
    """
    from safetensors.torch import save_file

    gate_name, up_name, down_name = mlp_names

    plain = lambda *shape: torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(torch.bfloat16)  # noqa: E731

    gate_w, gate_s = _fp8((INTERMEDIATE, HIDDEN), rng)
    up_w, up_s = _fp8((INTERMEDIATE, HIDDEN), rng)

    tensors = {
        "model.embed_tokens.weight": plain(VOCAB, HIDDEN),
        "model.norm.weight": plain(HIDDEN),
        "lm_head.weight": plain(VOCAB, HIDDEN),
        "model.layers.0.input_layernorm.weight": plain(HIDDEN),
        "model.layers.0.post_attention_layernorm.weight": plain(HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": plain(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": plain(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": plain(HIDDEN, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": plain(HIDDEN, HIDDEN),
        f"model.layers.0.mlp.{down_name}.weight": plain(HIDDEN, INTERMEDIATE),
        f"model.layers.0.mlp.{gate_name}.weight": gate_w,
        f"model.layers.0.mlp.{gate_name}.weight_scale_inv": gate_s,
        f"model.layers.0.mlp.{up_name}.weight": up_w,
        f"model.layers.0.mlp.{up_name}.weight_scale_inv": up_s,
    }
    save_file(tensors, str(directory / "model.safetensors"))

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": list(BLOCK),
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    return gate_w, gate_s, up_w, up_s


def _read_converted(out_dir, rel_path):
    """Read one array back out of the written TensorStore checkpoint."""
    import tensorstore as ts
    from jax.experimental.array_serialization import serialization as jax_ser

    spec = jax_ser.get_tensorstore_spec(str(out_dir / rel_path))
    return np.asarray(ts.open(ts.Spec(spec), open=True).result().read().result())


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Convert the fixture checkpoint once; every case reads the result."""
    rng = np.random.default_rng(0)
    src = tmp_path_factory.mktemp("src")
    out = tmp_path_factory.mktemp("out")
    sources = _write_checkpoint(src, rng)

    class _Base(EasyDeLBaseModule):
        _model_task = TaskType.CAUSAL_LM

    _Base.huggingface_to_easydel_sequential(
        pretrained_model_name_or_path=str(src),
        save_directory=str(out),
        verbose=False,
    )
    index = json.loads((out / "tensorstore_index.json").read_text())
    return out, {e["path"]: e for e in index["prefixes"]["model"]}, sources


FUSED_PATH = "model/model/layers/0/mlp/gate_up_proj/weight"


def test_fused_target_is_written_at_all(converted):
    """The regression itself: the fused parameter went missing, silently.

    It is not enough that conversion "succeeded" -- it reported writing every
    tensor it wrote, and the absent one was never counted.
    """
    _, entries, _ = converted
    assert FUSED_PATH in entries, (
        f"{FUSED_PATH} absent from the converted checkpoint; present paths under that layer: "
        f"{sorted(p for p in entries if 'layers/0/mlp' in p)}"
    )


def test_fused_target_holds_decoded_values_not_packed_codes(converted):
    """Values must match a dense fusion of the decoded sources.

    Packed fp8 codes reinterpreted as numbers keep the right shape and a
    plausible magnitude, so shape and finiteness checks both pass on the
    broken output. Only comparing against the decoded reference catches it.
    """
    out, _entries, (gate_w, gate_s, up_w, up_s) = converted
    got = _read_converted(out, FUSED_PATH)

    def _dense(weight, scale):
        return np.asarray(
            decode_scaled_elements(
                jnp.asarray(weight.to(torch.float32).numpy()),
                jnp.asarray(scale.numpy()),
                transpose=False,
                dtype=jnp.float32,
                block_shape=BLOCK,
            )
        )

    # EasyDeL stores [in, out]; HF ships [out, in], and the fusion concatenates
    # gate then up along the output axis.
    expected = np.concatenate([_dense(gate_w, gate_s), _dense(up_w, up_s)], axis=0).T

    assert got.shape == expected.shape == (HIDDEN, 2 * INTERMEDIATE)
    np.testing.assert_allclose(got.astype(np.float32), expected, rtol=2e-2, atol=2e-2)


def test_scale_siblings_do_not_become_parameters(converted):
    """The scale tensors are consumed by the decode, not written as params."""
    _, entries, _ = converted
    assert not [p for p in entries if "scale" in p]


# --- vendor-layout checkpoints -------------------------------------------------
#
# A family that publishes weights under its own names declares a
# `_checkpoint_key_normalizer` mapping one raw checkpoint key to its EasyDeL
# parameter name, or `None` for a tensor the runtime does not own. Fusion
# targets are built from ALREADY-normalized source names, so passing one back
# through that normalizer asks it about a key no checkpoint contains -- it
# answers `None`, and the converter drops the tensor as unowned.


def _vendor_normalizer(key: str) -> str | None:
    """A minimal stand-in for DeepSeek-V4's normalizer.

    Renames the vendor's MLP names and reports anything outside its table as
    unowned, which is what makes a re-normalized fusion target vanish.
    """
    if ".mlp.w1." in key:
        return key.replace(".mlp.w1.", ".mlp.gate_proj.")
    if ".mlp.w3." in key:
        return key.replace(".mlp.w3.", ".mlp.up_proj.")
    if ".mlp.w2." in key:
        return key.replace(".mlp.w2.", ".mlp.down_proj.")
    if ".mlp." in key:
        return None
    return key


@pytest.fixture(scope="module")
def converted_vendor(tmp_path_factory):
    """Convert a checkpoint whose MLP tensors use vendor names."""
    import easydel as ed

    rng = np.random.default_rng(1)
    src = tmp_path_factory.mktemp("vendor_src")
    out = tmp_path_factory.mktemp("vendor_out")
    _write_checkpoint(src, rng, mlp_names=("w1", "w3", "w2"))

    class _Base(EasyDeLBaseModule):
        _model_task = TaskType.CAUSAL_LM

    module_cls = ed.LlamaForCausalLM
    had = "_checkpoint_key_normalizer" in vars(module_cls)
    module_cls._checkpoint_key_normalizer = staticmethod(_vendor_normalizer)
    try:
        _Base.huggingface_to_easydel_sequential(
            pretrained_model_name_or_path=str(src),
            save_directory=str(out),
            verbose=False,
        )
    finally:
        if not had:
            del module_cls._checkpoint_key_normalizer
    index = json.loads((out / "tensorstore_index.json").read_text())
    return {e["path"]: e for e in index["prefixes"]["model"]}


def test_fusion_target_survives_a_vendor_key_normalizer(converted_vendor):
    """The DeepSeek-V4 failure, reduced.

    ``gate_up_proj`` is not a checkpoint key, so a vendor normalizer calls it
    unowned. Applying one to a fusion target therefore deletes the parameter --
    and because "unowned" is a normal, expected answer, nothing is logged.
    """
    assert FUSED_PATH in converted_vendor, (
        "fused gate_up_proj was dropped as 'unowned' after its already-normalized "
        f"key was normalized again; wrote: {sorted(converted_vendor)}"
    )
