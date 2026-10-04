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

"""Load-time dequantization of DeepSeek-V4.1's native fp8 / fp4 checkpoint layout (computation test).

V4.1 ships ``quant_method=fp8`` with ``weight_block_size=[32, 32]``, ``scale_fmt=ue8m0`` and ``expert_dtype=fp4``
(``ref/config.json``). Routed experts are int8-packed e2m1 nibble pairs with e8m0 scales per 32 inputs
(``ref/convert.py:13-34``: low nibble = element 2k, high nibble = 2k+1); dense weights are e4m3 with one e8m0 scale per
32x32 block; engram tables carry one scale per row and 32 columns (``ref/model.py:297-322``). The expected values here
are computed in NumPy from those definitions, independent of EasyDeL's codecs.
"""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from easydel.infra.mixins.bridge import _QUANT_CONSUMED, _build_quant_decoder  # noqa: E402

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
V41_QUANT = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4",
}


def _e8m0(rng, shape):
    codes = rng.integers(122, 132, size=shape).astype(np.uint8)
    return torch.from_numpy(codes).view(torch.float8_e8m0fnu), np.exp2(codes.astype(np.float32) - 127.0)


def _fp8_block(rng, out_dim, in_dim, block=32):
    weight = torch.from_numpy(rng.normal(0, 2, size=(out_dim, in_dim)).astype(np.float32)).to(torch.float8_e4m3fn)
    scale_t, scale = _e8m0(rng, (-(-out_dim // block), -(-in_dim // block)))
    expanded = np.repeat(np.repeat(scale, block, axis=0), block, axis=1)[:out_dim, :in_dim]
    return weight, scale_t, weight.float().numpy() * expanded


def _fp4_expert(rng, out_dim, in_dim):
    packed = rng.integers(0, 256, size=(out_dim, in_dim // 2)).astype(np.uint8)
    scale_t, scale = _e8m0(rng, (out_dim, in_dim // 32))
    values = np.empty((out_dim, in_dim), np.float32)
    values[:, 0::2] = FP4_TABLE[packed & 0x0F]
    values[:, 1::2] = FP4_TABLE[packed >> 4]
    return torch.from_numpy(packed).view(torch.int8), scale_t, values * np.repeat(scale, 32, axis=1)


def _engram_rows(rng, rows, dim):
    weight = torch.from_numpy(rng.normal(0, 1, size=(rows, dim)).astype(np.float32)).to(torch.float8_e4m3fn)
    scale_t, scale = _e8m0(rng, (rows, dim // 32))
    return weight, scale_t, weight.float().numpy() * np.repeat(scale, 32, axis=1)


def test_v41_native_quantized_tensors_decode_to_reference_values():
    rng = np.random.default_rng(0)
    cases = {
        "layers.2.attn.wq_a": _fp8_block(rng, 64, 96),
        "layers.5.attn.wo_b": _fp8_block(rng, 40, 72),  # partial trailing 32-blocks on both axes
        "layers.0.ffn.experts.3.w1": _fp4_expert(rng, 64, 96),
        "layers.0.ffn.shared_experts.w1": _fp8_block(rng, 64, 96),
        "layers.1.engram.embed": _engram_rows(rng, 50, 64),
    }
    tensors = {}
    for stem, (weight, scale, _) in cases.items():
        tensors[f"{stem}.weight"] = weight
        tensors[f"{stem}.scale"] = scale
    plain = torch.from_numpy(rng.normal(size=(32, 64)).astype(np.float32))
    tensors["layers.2.attn.compressor.wkv.weight"] = plain

    decode = _build_quant_decoder(types.SimpleNamespace(quantization_config=dict(V41_QUANT)), list(tensors))
    assert decode is not None
    fetch = tensors.get
    for stem, (_, _, expected) in cases.items():
        assert decode(f"{stem}.scale", tensors[f"{stem}.scale"], fetch) is _QUANT_CONSUMED
        got = decode(f"{stem}.weight", tensors[f"{stem}.weight"], fetch)
        got = got.float().numpy() if hasattr(got, "float") else np.asarray(got, np.float32)
        assert got.shape == expected.shape, (stem, got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0, err_msg=stem)
    # Unscaled tensors (bf16/fp32 compressor, indexer wk, head) pass through untouched.
    assert decode("layers.2.attn.compressor.wkv.weight", plain, fetch) is None


def test_fp4_decode_is_sensitive_to_nibble_order():
    """Negative control: swapping the nibble order must change the decoded expert."""
    rng = np.random.default_rng(1)
    weight, scale, expected = _fp4_expert(rng, 32, 64)
    stem = "layers.0.ffn.experts.0.w2"
    tensors = {f"{stem}.weight": weight, f"{stem}.scale": scale}
    decode = _build_quant_decoder(types.SimpleNamespace(quantization_config=dict(V41_QUANT)), list(tensors))
    got = decode(f"{stem}.weight", weight, tensors.get).float().numpy()
    swapped = expected.copy()
    swapped[:, 0::2], swapped[:, 1::2] = expected[:, 1::2], expected[:, 0::2]
    assert not np.allclose(got, swapped)
    np.testing.assert_array_equal(got, expected)
