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

"""DeepSeek-V4's published quantization is MXFP4 + block-fp8, not a new format.

Read off a real shard of ``deepseek-ai/DeepSeek-V4-Flash-0731``:

* experts (97% of the model) -- ``w1.weight`` ``[2048, 2048]`` ``int8`` with
  ``w1.scale`` ``[2048, 128]`` ``float8_e8m0fnu``. The int8 packs two E2M1
  codes per byte, so the real shape is ``[2048, 4096]`` and 4096/128 gives a
  32-element scale block: exactly MXFP4.
* dense/attention -- ``float8_e4m3fn`` elements with ``float8_e8m0fnu`` scales
  on a 128x128 grid, matching ``weight_block_size: [128, 128]``.

``config.json``'s ``expert_dtype: "fp4"`` and ``scale_fmt: "ue8m0"`` look like
a bespoke scheme and are not one; these cases pin the two existing codecs
against DeepSeek's own arithmetic so nobody writes a third.
"""

import typing

import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.quantization.checkpoint._codecs import decode_mxfp4, decode_scaled_elements

# The E2M1 value table, verbatim from the repo's `inference/convert.py`.
FP4_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

E8M0_BIAS = 127


def _e8m0(exponents: np.ndarray) -> np.ndarray:
    """Materialize E8M0 codes as the powers of two they denote."""
    return np.exp2(exponents.astype(np.float32) - E8M0_BIAS)


def _deepseek_unpack_fp4(packed: np.ndarray) -> np.ndarray:
    """Unpack two E2M1 codes per byte, low nibble first (convert.py's order)."""
    low, high = packed & 0x0F, (packed >> 4) & 0x0F
    return np.stack([FP4_TABLE[low], FP4_TABLE[high]], axis=-1).reshape(packed.shape[0], -1)


def test_e2m1_nibble_order_and_table_match_deepseek():
    """Every one of the 256 byte values must unpack identically.

    A swapped nibble order still produces plausible weights -- correct shape,
    correct magnitudes -- so it would survive shape and finiteness checks and
    only show up as a degraded model.
    """
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    reference = _deepseek_unpack_fp4(packed)
    got = np.asarray(
        decode_mxfp4(
            jnp.asarray(packed),
            jnp.zeros((16, packed.shape[1] * 2 // 32), jnp.uint8) + E8M0_BIAS,
            transpose=False,
            dtype=jnp.float32,
        )
    )
    np.testing.assert_array_equal(got, reference)


def test_mxfp4_expert_decode_matches_deepseek_arithmetic():
    """Expert layout: [out, in/2] int8 codes + [out, in/32] E8M0 scales."""
    rng = np.random.default_rng(0)
    out_dim, in_dim, block = 64, 256, 32
    packed = rng.integers(0, 256, size=(out_dim, in_dim // 2), dtype=np.uint8)
    exps = rng.integers(E8M0_BIAS - 4, E8M0_BIAS + 4, size=(out_dim, in_dim // block)).astype(np.uint8)

    reference = _deepseek_unpack_fp4(packed) * np.repeat(_e8m0(exps), block, axis=1)
    got = np.asarray(decode_mxfp4(jnp.asarray(packed), jnp.asarray(exps), transpose=False, dtype=jnp.float32))

    assert got.shape == (out_dim, in_dim)
    np.testing.assert_array_equal(got, reference)


@pytest.mark.parametrize("block", [(128, 128), (64, 64)])
def test_block_fp8_dense_decode_matches_deepseek_arithmetic(block):
    """Dense layout: E4M3 elements over a 2-D block-scale grid."""
    rng = np.random.default_rng(1)
    out_dim, in_dim = 4 * block[0], 3 * block[1]
    weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    scale = _e8m0(rng.integers(E8M0_BIAS - 2, E8M0_BIAS + 2, size=(out_dim // block[0], in_dim // block[1])))

    reference = weight * np.kron(scale, np.ones(block, np.float32))
    got = np.asarray(
        decode_scaled_elements(
            jnp.asarray(weight), jnp.asarray(scale), transpose=False, dtype=jnp.float32, block_shape=block
        )
    )
    np.testing.assert_array_equal(got, reference)


def test_expert_scale_block_is_32_for_the_published_shapes():
    """Guards the inference that makes the expert path MXFP4 rather than novel.

    ``w1.weight`` is ``[2048, 2048]`` int8 and ``w1.scale`` is ``[2048, 128]``:
    two codes per byte gives in_dim 4096, and 4096 // 128 == 32.
    """
    packed_cols, scale_cols = 2048, 128
    assert (packed_cols * 2) // scale_cols == 32


class TestDeepseekV4MixedAdapter:
    """V4 declares one `quant_method` but ships two schemes.

    `config.json` says `quant_method: "fp8"` with `weight_block_size:
    [128,128]`, which describes only the dense tensors; a separate top-level
    `expert_dtype: "fp4"` switches the routed experts -- 97% of the
    parameters -- to MXFP4. Handing the whole checkpoint to the plain fp8
    adapter is silently wrong rather than an error, because packed E2M1
    nibbles are valid fp8 bit patterns: it produces plausible garbage at half
    the true input width.
    """

    QUANT_CONFIG: typing.ClassVar[dict] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "expert_dtype": "fp4",
    }

    @staticmethod
    def _adapter():
        from easydel.layers.quantization.checkpoint import get_adapter

        return get_adapter("deepseek_v4_mixed")

    def test_experts_resolve_to_four_bit_and_dense_to_block_fp8(self):
        adapter = self._adapter()
        parsed = adapter.parse_config(self.QUANT_CONFIG)

        assert parsed.default.quant_method == "fp8"
        assert parsed.default.block_shape == (128, 128)
        assert adapter.target_spec(parsed.default).bits == 8

        pattern, expert_source = parsed.overrides[0]
        assert "experts" in pattern
        assert expert_source.quant_method == "mxfp4"
        assert expert_source.group_size == 32
        # The W4 of A16W4: experts must stay 4-bit, or the model is 530 GiB
        # of bf16 against 383 GiB of HBM.
        assert adapter.target_spec(expert_source, expert_dim=True).bits == 4

    def test_scale_suffix_matches_what_deepseek_ships(self):
        """V4 ships a bare ``scale``; the delegates name it ``weight_scale*``.

        A reform rule only fires when every declared source key is present, so
        an unrenamed suffix makes the fusion silently never run.
        """
        adapter = self._adapter()
        parsed = adapter.parse_config(self.QUANT_CONFIG)
        for source in (parsed.default, parsed.overrides[0][1]):
            suffixes = adapter.source_suffixes(source)
            assert "scale" in suffixes
            assert not any(s.startswith("weight_scale") for s in suffixes)

    def test_without_expert_dtype_it_is_plain_fp8(self):
        """No expert override declared -> behave exactly like the fp8 adapter."""
        adapter = self._adapter()
        config = {k: v for k, v in self.QUANT_CONFIG.items() if k != "expert_dtype"}
        parsed = adapter.parse_config(config)
        assert parsed.overrides == ()
        assert adapter.target_spec(parsed.default).bits == 8

    def test_shared_experts_are_not_treated_as_routed(self):
        """`shared_experts.` contains "experts." but is block-fp8, not fp4.

        Only the ROUTED experts carry `expert_dtype`; the shared expert is
        stored like the rest of the dense path. A pattern that matches both
        sends the shared expert to the MXFP4 codec, which fails on dtype --
        that is exactly how this was found, mid-conversion.
        """
        import re

        pattern = self._adapter().expert_pattern
        routed = [
            "layers.0.ffn.experts.0.w1.weight",
            "model.layers.9.mlp.experts.17.gate_proj.weight",
        ]
        not_routed = [
            "layers.0.ffn.shared_experts.w1.weight",
            "model.layers.9.mlp.shared_experts.gate_up_proj.weight",
            "layers.0.attn.wq_a.weight",
        ]
        for key in routed:
            assert re.search(pattern, key), f"{key} should match the routed-expert pattern"
        for key in not_routed:
            assert not re.search(pattern, key), f"{key} must NOT match the routed-expert pattern"
