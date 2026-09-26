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

"""Regression tests for Gemma3/Gemma4 vision-path parity with HF.

* Gemma4 vision tower drops padded pooled rows like HF ``hidden_states[pooler_mask]``
  (static-shape: valid rows first in HF order, zeroed padding rows trailing), so a
  batch whose *first* image is smaller than the padded patch budget still feeds the
  sequential placeholder merge the right soft tokens.
* Pooling scale + standardize run in float32 (HF ``Gemma4VisionPooler``).
* ``use_clipped_linears`` clamps inputs/outputs with the per-projection
  ``input_min``/``input_max``/``output_min``/``output_max`` checkpoint buffers.
* Gemma3/Gemma4 multiply embeddings by ``sqrt(hidden)`` rounded to the weight dtype.
"""

import copy

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import transformers
from easydel.modules.gemma3.modeling_gemma3 import _gemma3_embed_scale

try:
    from tests.modules.test_utils.model_factory import create_ed_model, setup_config
except ImportError:
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        create_ed_model,
        setup_config,
    )

PATCH_SIZE = 2
GRID = 4  # padded patch budget: GRID x GRID patches
POOL = 2


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    """Compare against fp32 torch at full precision (TPU's default f32 matmul is a single bf16 pass)."""
    with jax.default_matmul_precision("highest"):
        yield


def _vision_config(use_clipped_linears: bool):
    return ed.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        patch_size=PATCH_SIZE,
        pooling_kernel_size=POOL,
        position_embedding_size=16,
        standardize=True,
        use_clipped_linears=use_clipped_linears,
    )


def _perturb_hf(hf_model):
    """Random weights + non-trivial standardize/clip buffers (distinct per projection)."""
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in hf_model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
        for name, buf in hf_model.named_buffers():
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "std_bias":
                buf.copy_(0.5 * torch.randn(buf.shape, generator=gen))
            elif leaf == "std_scale":
                buf.copy_(1.0 + 0.2 * torch.randn(buf.shape, generator=gen))
            elif leaf in ("input_min", "output_min"):
                buf.fill_(-(0.3 + torch.rand((), generator=gen).item()))
            elif leaf in ("input_max", "output_max"):
                buf.fill_(0.3 + torch.rand((), generator=gen).item())
    return hf_model.float().eval()


def _variable_size_batch():
    """Image 0 is 4x2 patches (padded to 16), image 1 is the full 4x4 grid."""
    rng = np.random.default_rng(0)
    num_patches = GRID * GRID
    pixel_values = rng.random((2, num_patches, 3 * PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    positions = -np.ones((2, num_patches, 2), dtype=np.int64)
    ys, xs = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    full = np.stack((xs, ys), axis=-1).reshape(-1, 2)
    small = full[full[:, 1] < 2]
    positions[0, : len(small)] = small
    positions[1] = full
    pixel_values[0, len(small) :] = 0.0
    return pixel_values, positions


@pytest.mark.parametrize("use_clipped_linears", [False, True])
def test_gemma4_vision_matches_hf_with_padding(small_model_config, use_clipped_linears):
    local_cfg = dict(small_model_config)
    local_cfg["attn_dtype"] = jnp.float32
    local_cfg["attn_softmax_dtype"] = jnp.float32
    config = setup_config(_vision_config(use_clipped_linears), local_cfg)

    hf_config = copy.deepcopy(config)
    hf_config._attn_implementation = "eager"
    hf_model = _perturb_hf(transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel(hf_config))
    if use_clipped_linears:
        assert any(n.endswith("q_proj.input_min") for n, _ in hf_model.named_buffers())

    pixel_values, positions = _variable_size_batch()
    with torch.no_grad():
        want = (
            hf_model(pixel_values=torch.from_numpy(pixel_values), pixel_position_ids=torch.from_numpy(positions))
            .last_hidden_state.float()
            .numpy()
        )
    # 2 valid pooled rows for the 4x2 image + 4 for the 4x4 image.
    assert want.shape == (6, 32)

    with config.mesh:
        ed_model = create_ed_model(
            module_name="gemma4_vision",
            task=ed.TaskType.BASE_VISION,
            config=config,
            small_model_config=local_cfg,
            hf_model=hf_model,
        )
        got = np.asarray(
            ed_model(
                pixel_values=jnp.asarray(pixel_values),
                pixel_position_ids=jnp.asarray(positions, dtype=jnp.int32),
            ).last_hidden_state,
            dtype=np.float32,
        )

    assert got.shape == (2 * (GRID // POOL) ** 2, 32)
    np.testing.assert_allclose(got[: want.shape[0]], want, rtol=2e-3, atol=2e-3)
    assert not np.any(got[want.shape[0] :]), "padded pooled rows must trail as zeros"


def test_gemma4_vision_clip_buffers_are_loaded(small_model_config):
    """Each unfused clipped projection gets its own scalar bounds from the checkpoint."""
    local_cfg = dict(small_model_config)
    config = setup_config(_vision_config(True), local_cfg)
    hf_config = copy.deepcopy(config)
    hf_config._attn_implementation = "eager"
    hf_model = _perturb_hf(transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel(hf_config))
    with config.mesh:
        ed_model = create_ed_model(
            module_name="gemma4_vision",
            task=ed.TaskType.BASE_VISION,
            config=config,
            small_model_config=local_cfg,
            hf_model=hf_model,
        )
    hf_attn = hf_model.encoder.layers[1].self_attn
    ed_attn = ed_model.encoder.layers[1].self_attn
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for bound in ("input_min", "input_max", "output_min", "output_max"):
            np.testing.assert_allclose(
                float(getattr(getattr(ed_attn, proj), bound).value),
                getattr(getattr(hf_attn, proj), bound).item(),
                rtol=1e-6,
            )


@pytest.mark.parametrize("hidden_size", [640, 1152, 2560, 3840, 5376])
def test_gemma3_embed_scale_rounds_to_weight_dtype(hidden_size):
    """HF: ``embed * embed_scale.to(weight.dtype)`` -- e.g. sqrt(2560) -> 50.5 for bf16 weights."""

    class _Embed:
        param_dtype = jnp.bfloat16

    inputs = jnp.zeros((1, 1, hidden_size), dtype=jnp.float32)
    scale = _gemma3_embed_scale(_Embed(), hidden_size, inputs)
    expected = torch.tensor(hidden_size**0.5).to(torch.bfloat16).float().item()
    assert scale.dtype == jnp.float32
    assert float(scale) == expected

    class _Fp32Embed:
        param_dtype = jnp.float32

    scale_fp32 = _gemma3_embed_scale(_Fp32Embed(), hidden_size, inputs)
    assert float(scale_fp32) == pytest.approx(torch.tensor(hidden_size**0.5).item(), rel=1e-7)
