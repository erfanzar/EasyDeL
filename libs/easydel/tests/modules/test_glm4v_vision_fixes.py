# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""GLM-4V vision-tower parity with HF (shared by glm4v, glm46v and glm4v_moe).

Pins four divergences from ``transformers.models.glm4v``:

* the vision MLP is ``hidden_size -> out_hidden_size`` (``Glm4VisionMlp``), not
  ``intermediate_size`` (that width belongs to the patch merger);
* the learned position grid is resampled with
  ``grid_sample(mode="bicubic", align_corners=False, padding_mode="border")``,
  not bilinear ``linspace`` interpolation;
* the patch merger uses ``torch.nn.LayerNorm`` (eps 1e-5) and exact GELU;
* video frames are spelled with ``<|image|>`` tokens, so video features are
  scattered at ``image_token_id`` (HF ``get_placeholder_mask``).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

torch = pytest.importorskip("torch")
F = torch.nn.functional
hf_glm4v = pytest.importorskip("transformers.models.glm4v.modeling_glm4v")
hf_glm4v_config = pytest.importorskip("transformers.models.glm4v.configuration_glm4v")
hf_vision_utils = pytest.importorskip("transformers.vision_utils")

from easydel.modules.glm4v.glm4v_configuration import Glm4vVisionConfig  # noqa: E402
from easydel.modules.glm4v.modeling_glm4v import (  # noqa: E402
    Glm4vModel,
    Glm4vVisionMLP,
    Glm4vVisionModel,
    Glm4vVisionPatchMerger,
    bicubic_border_resample_matrix,
)


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    """Compare against fp32 torch at full precision (TPU's default f32 matmul is a single bf16 pass)."""
    with jax.default_matmul_precision("highest"):
        yield


def _torch_grid_sample_reference(grid_2d: np.ndarray, h: int, w: int) -> np.ndarray:
    hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    h_coords = torch.tensor(hh.reshape(-1), dtype=torch.float32)
    w_coords = torch.tensor(ww.reshape(-1), dtype=torch.float32)
    grid = torch.stack((((w_coords + 0.5) / w) * 2 - 1, ((h_coords + 0.5) / h) * 2 - 1), dim=-1)[None, :, None]
    source = torch.tensor(grid_2d).permute(2, 0, 1)[None]
    out = F.grid_sample(source, grid, mode="bicubic", align_corners=False, padding_mode="border")
    return out.squeeze(0).squeeze(-1).permute(1, 0).numpy().reshape(h, w, -1)


@pytest.mark.parametrize(("side", "h", "w"), [(6, 4, 10), (6, 6, 6), (5, 12, 3), (24, 16, 32), (24, 98, 40), (4, 1, 2)])
def test_bicubic_border_matrix_matches_torch_grid_sample(side, h, w):
    rng = np.random.default_rng(0)
    grid_2d = rng.standard_normal((side, side, 5)).astype(np.float32)
    rows = bicubic_border_resample_matrix(h, side)
    cols = bicubic_border_resample_matrix(w, side)
    ours = np.einsum("hs,std,wt->hwd", rows, grid_2d, cols)
    np.testing.assert_allclose(ours, _torch_grid_sample_reference(grid_2d, h, w), rtol=1e-5, atol=1e-5)


def test_pos_embed_interpolation_matches_hf_embeddings():
    hidden, side, merge = 8, 6, 2
    hf_cfg = hf_glm4v_config.Glm4vVisionConfig(
        hidden_size=hidden, image_size=side * 14, patch_size=14, spatial_merge_size=merge
    )
    torch.manual_seed(0)
    hf_embeddings = hf_glm4v.Glm4vVisionEmbeddings(hf_cfg)
    torch.nn.init.normal_(hf_embeddings.position_embedding.weight)

    grid_thw = torch.tensor([[1, 4, 8], [1, 12, 6], [1, 2, 2]])
    position_ids = hf_vision_utils.get_vision_position_ids(grid_thw, merge)
    seqlens = grid_thw[:, 1] * grid_thw[:, 2]
    with torch.no_grad():
        expected = hf_embeddings(
            torch.zeros(int(seqlens.sum()), hidden), seqlens, grid_thw, position_ids[:, 0], position_ids[:, 1]
        ).numpy()

    weight = hf_embeddings.position_embedding.weight.detach().numpy()
    stub = SimpleNamespace(
        spatial_merge_size=merge,
        num_grid_per_side=side,
        pos_embed=SimpleNamespace(weight=SimpleNamespace(value=jnp.asarray(weight))),
    )
    ours = Glm4vVisionModel.fast_pos_embed_interpolate(stub, jnp.asarray(grid_thw.numpy()))
    np.testing.assert_allclose(np.asarray(ours), expected, rtol=1e-5, atol=1e-5)


def test_vision_mlp_width_is_out_hidden_size():
    config = Glm4vVisionConfig(hidden_size=16, out_hidden_size=24, intermediate_size=40, num_heads=2, depth=1)
    mlp = Glm4vVisionMLP(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert tuple(mlp.gate_up_proj.weight.value.shape) == (16, 2 * 24)
    assert tuple(mlp.down_proj.weight.value.shape) == (24, 16)

    hf_mlp = hf_glm4v.Glm4VisionMlp(SimpleNamespace(hidden_size=16, out_hidden_size=24, hidden_act="silu"))
    assert tuple(hf_mlp.gate_proj.weight.shape) == (24, 16)


def test_patch_merger_matches_hf():
    dim, context = 8, 12
    torch.manual_seed(0)
    hf_merger = hf_glm4v.Glm4vVisionPatchMerger(dim=dim, context_dim=context, hidden_act="silu")
    with torch.no_grad():
        hf_merger.post_projection_norm.weight.normal_()
        hf_merger.post_projection_norm.bias.normal_()

    merger = Glm4vVisionPatchMerger(
        dim=dim, context_dim=context, hidden_act="silu", dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    assert merger.norm.epsilon == pytest.approx(1e-5)

    def _t(linear):
        return jnp.asarray(linear.weight.detach().numpy().T)

    merger.proj.weight.value = _t(hf_merger.proj)
    merger.norm.weight.value = jnp.asarray(hf_merger.post_projection_norm.weight.detach().numpy())
    merger.norm.bias.value = jnp.asarray(hf_merger.post_projection_norm.bias.detach().numpy())
    merger.gate_up_proj.weight.value = jnp.concatenate([_t(hf_merger.gate_proj), _t(hf_merger.up_proj)], axis=1)
    merger.down_proj.weight.value = _t(hf_merger.down_proj)

    x = np.random.default_rng(1).standard_normal((5, dim)).astype(np.float32) * 3.0
    with torch.no_grad():
        expected = hf_merger(torch.tensor(x)).numpy()
    np.testing.assert_allclose(np.asarray(merger(jnp.asarray(x))), expected, rtol=1e-4, atol=1e-5)


def test_video_features_merge_at_image_token_positions():
    image_token_id, video_token_id, hidden = 5, 6, 4
    stub = SimpleNamespace(config=SimpleNamespace(image_token_id=image_token_id, video_token_id=video_token_id))
    # HF processor layout for one 2-frame video: each frame is <boi> <image>*n <eoi> <timestamp>.
    input_ids = jnp.asarray([[1, 3, image_token_id, image_token_id, 4, 9, 3, image_token_id, image_token_id, 4, 9]])
    inputs_embeds = jnp.zeros((1, input_ids.shape[1], hidden), dtype=jnp.float32)
    video_embeds = jnp.arange(4 * hidden, dtype=jnp.float32).reshape(4, hidden) + 1.0

    merged = Glm4vModel.compute_embedding(stub, input_ids, inputs_embeds=inputs_embeds, video_embeds=video_embeds)
    positions = np.flatnonzero(np.asarray(input_ids[0]) == image_token_id)
    np.testing.assert_array_equal(np.asarray(merged[0, positions]), np.asarray(video_embeds))
    others = np.setdiff1d(np.arange(input_ids.shape[1]), positions)
    assert not np.asarray(merged[0, others]).any()
