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

"""Vision-tower parity regressions against the torch references.

* ``torch_bicubic_resize`` must equal ``F.interpolate(mode="bicubic",
  align_corners=False)`` (A=-0.75, border-clamped taps, no antialias).
  ``jax.image.resize`` uses the A=-0.5 Keys kernel and antialiases when
  downsampling, so SigLIP / InternVL / MoonViT (Kimi-VL) position-embedding
  interpolation drifted from HF.
* SigLIP ``interpolate`` called ``.unsqueeze`` on a JAX parameter (crash).
* SigLIP's pooling-head ``in_proj_weight`` was declared ``(3E, E)`` although the
  converter transposes it to ``(E, 3E)`` and ``forward`` splits the last axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
F = torch.nn.functional


def _torch_resize(x_hwc: np.ndarray, height: int, width: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x_hwc)).permute(2, 0, 1)[None]
    out = F.interpolate(t, size=(height, width), mode="bicubic", align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


@pytest.mark.parametrize(
    ("in_hw", "out_hw"),
    [((8, 8), (5, 11)), ((6, 9), (23, 4)), ((16, 16), (9, 13)), ((4, 4), (4, 4)), ((3, 5), (1, 7))],
)
def test_torch_bicubic_resize_matches_torch(in_hw, out_hw):
    from easydel.modules._base import torch_bicubic_resize

    rng = np.random.default_rng(0)
    x = rng.standard_normal((*in_hw, 5)).astype(np.float32)
    expected = _torch_resize(x, *out_hw)
    actual = np.asarray(torch_bicubic_resize(jnp.asarray(x), *out_hw))
    assert actual.shape == (*out_hw, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_kimi_vl_pos_emb_matches_moonvit_reference():
    """MoonViT ``Learnable2DInterpPosEmb`` resizes with torch bicubic, no antialias."""
    from easydel.modules.kimi_vl.modeling_kimi_vl import Learnable2DInterpPosEmb

    height, width, dim = 4, 5, 6
    module = Learnable2DInterpPosEmb(height, width, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    rng = np.random.default_rng(1)
    hf_weight = rng.standard_normal((height, width, dim)).astype(np.float32)  # HF (height, width, dim)
    module.weight.value = jnp.asarray(hf_weight.transpose(2, 1, 0))  # converter's 3-D permute(2, 1, 0)

    grid_hws = np.array([[3, 7], [4, 5], [2, 2], [9, 3]])
    total = int((grid_hws[:, 0] * grid_hws[:, 1]).sum())
    x = rng.standard_normal((total, dim)).astype(np.float32)

    # Remote-code reference (modeling_kimi_vl.Learnable2DInterpPosEmb.forward).
    ref_weight = torch.from_numpy(hf_weight)
    pos = [
        F.interpolate(ref_weight.permute(2, 0, 1)[None], size=(int(h), int(w)), mode="bicubic")[0]
        .permute(1, 2, 0)
        .flatten(end_dim=1)
        for h, w in grid_hws
    ]
    expected = x + torch.cat(pos).numpy()

    actual = np.asarray(module(jnp.asarray(x), grid_hws))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_siglip_interpolate_pos_encoding_matches_hf():
    import easydel as ed
    from easydel.modules.siglip.modeling_siglip import SiglipVisionEmbeddings
    from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings as HFSiglipVisionEmbeddings

    kwargs = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=16,
        patch_size=4,
    )
    hf = HFSiglipVisionEmbeddings(transformers.SiglipVisionConfig(**kwargs)).eval()
    ours = SiglipVisionEmbeddings(
        ed.SiglipVisionConfig(**kwargs, sharding_axis_dims=(1, 1, 1, 1, 1, 1)),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    ours.patch_embedding.weight.value = jnp.asarray(hf.patch_embedding.weight.detach().numpy().transpose(2, 3, 1, 0))
    ours.patch_embedding.bias.value = jnp.asarray(hf.patch_embedding.bias.detach().numpy())
    ours.position_embedding.weight.value = jnp.asarray(hf.position_embedding.weight.detach().numpy())

    rng = np.random.default_rng(2)
    for height, width in [(24, 32), (12, 20), (8, 8)]:
        pixels = rng.standard_normal((2, 3, height, width)).astype(np.float32)
        with torch.no_grad():
            expected = hf(torch.from_numpy(pixels), interpolate_pos_encoding=True).numpy()
        actual = np.asarray(ours(jnp.asarray(pixels), interpolate_pos_encoding=True))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_siglip_pooling_attention_matches_torch_mha():
    """``in_proj_weight`` is held as the transpose of torch's ``(3E, E)`` tensor."""
    from easydel.modules.siglip.modeling_siglip import MultiheadAttention

    embed_dim, num_heads = 12, 3
    ours = MultiheadAttention(embed_dim, num_heads, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert tuple(ours.in_proj_weight.value.shape) == (embed_dim, 3 * embed_dim)

    ref = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).eval()
    with torch.no_grad():
        ref.in_proj_bias.normal_()
        ref.out_proj.bias.normal_()
    # Mirror the HF->EasyDeL converter: every 2-D ``*weight`` leaf is transposed.
    ours.in_proj_weight.value = jnp.asarray(ref.in_proj_weight.detach().numpy().T)
    ours.in_proj_bias.value = jnp.asarray(ref.in_proj_bias.detach().numpy())
    ours.out_proj.weight.value = jnp.asarray(ref.out_proj.weight.detach().numpy().T)
    ours.out_proj.bias.value = jnp.asarray(ref.out_proj.bias.detach().numpy())

    rng = np.random.default_rng(3)
    probe = rng.standard_normal((2, 1, embed_dim)).astype(np.float32)
    hidden = rng.standard_normal((2, 7, embed_dim)).astype(np.float32)
    with torch.no_grad():
        expected = ref(torch.from_numpy(probe), torch.from_numpy(hidden), torch.from_numpy(hidden))[0].numpy()
    with jax.default_matmul_precision("highest"):
        actual = np.asarray(ours(jnp.asarray(probe), jnp.asarray(hidden), jnp.asarray(hidden)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
