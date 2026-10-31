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

"""Qwen3-VL vision tower: deepstack + ``scan_layers`` regression.

The tower collects deepstack features into a Python list keyed by the
traced layer index, which is incompatible with a compiled ``lax.scan``
layer loop. The forward must therefore fall back to the unrolled path
whenever ``deepstack_visual_indexes`` is non-empty — previously a
``TracerBoolConversionError`` crashed any ``scan_layers=True`` tower
with deepstack configured (the standard Qwen3-VL setup).
"""

import numpy as np
import spectrax as spx
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VisionTransformerPretrainedModel
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLVisionConfig
from jax import numpy as jnp


def _tiny_config(**overrides):
    kwargs = {
        "depth": 3,
        "hidden_size": 32,
        "num_heads": 4,
        "intermediate_size": 64,
        "patch_size": 4,
        "temporal_patch_size": 1,
        "spatial_merge_size": 2,
        "num_position_embeddings": 64,
        "deepstack_visual_indexes": [1],
        "out_hidden_size": 16,
        "scan_layers": True,
    }
    kwargs.update(overrides)
    return Qwen3VLVisionConfig(**kwargs)


def _forward(config):
    model = Qwen3VisionTransformerPretrainedModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    rng = np.random.default_rng(0)
    # one image, t=1 h=4 w=4 -> 16 patches of 1*3*4*4 values
    pixels = rng.standard_normal(16 * 48).astype(np.float32)
    grid = np.asarray([[1, 4, 4]], dtype=np.int32)
    return model(pixels, grid_thw=grid, max_grid_size=8)


def test_vision_tower_deepstack_with_scan_layers_runs():
    """scan_layers=True with deepstack configured must not crash and must
    return one feature per deepstack index with the merged shape."""
    config = _tiny_config()
    merged, deepstack = _forward(config)
    assert merged.shape == (4, 16)  # 16 patches / merge**2, out_hidden_size
    assert len(deepstack) == len(config.deepstack_visual_indexes)
    assert deepstack[0].shape == (4, 16)


def test_vision_tower_deepstack_matches_unscanned_reference():
    """The unrolled fallback must produce the same outputs as scan_layers=False."""
    scanned, scanned_ds = _forward(_tiny_config(scan_layers=True))
    plain, plain_ds = _forward(_tiny_config(scan_layers=False))
    np.testing.assert_allclose(np.asarray(scanned), np.asarray(plain), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(scanned_ds[0]), np.asarray(plain_ds[0]), rtol=1e-5, atol=1e-5)


def test_vision_tower_scan_layers_without_deepstack_still_scans():
    """scan_layers=True without deepstack keeps working (compiled loop path)."""
    config = _tiny_config(deepstack_visual_indexes=[])
    merged, deepstack = _forward(config)
    assert merged.shape == (4, 16)
    assert deepstack == []


def _tiny_text_model():
    from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig

    mrope_section = [2, 3, 3]
    head_dim = 2 * sum(mrope_section)
    config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=4 * head_dim,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=head_dim,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": mrope_section,
            "mrope_interleaved": True,
            "rope_theta": 5_000_000.0,
        },
        scan_layers=True,
    )
    return Qwen3VLTextModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_text_stack_deepstack_with_scan_layers_matches_unrolled():
    """The text stack must not run DeepStack injection under lax.scan (r11 S10-2).

    With ``scan_layers=True`` and no cache/output flags, a non-empty DeepStack list used to hit
    ``idx < len(...)`` on a traced scan index (TracerBoolConversionError).
    """
    model = _tiny_text_model()
    hidden = model.config.hidden_size
    batch, seq = 2, 8
    rng = np.random.default_rng(1)
    embeds = jnp.asarray(rng.standard_normal((batch, seq, hidden)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, None, :], (3, batch, seq))
    visual_mask = jnp.zeros((batch, seq), bool).at[:, 2:5].set(True)
    num_visual = int(visual_mask.sum())
    deepstack = [jnp.asarray(rng.standard_normal((num_visual, hidden)), jnp.float32) for _ in range(2)]

    def run(ds, **kwargs):
        out = model(
            inputs_embeds=embeds,
            position_ids=positions,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=ds,
            **kwargs,
        )
        return np.asarray(out.last_hidden_state, np.float32)

    scanned = run(deepstack)  # default trace=False: used to crash
    unrolled = run(deepstack, trace=True)
    np.testing.assert_allclose(scanned, unrolled, rtol=1e-5, atol=1e-5)
    plain = run(None)
    assert np.abs(scanned - plain).max() > 1e-2  # negative control: DeepStack changes the output
    # An empty list is a no-op and keeps the compiled scan path.
    np.testing.assert_allclose(run([]), plain, rtol=1e-5, atol=1e-5)
