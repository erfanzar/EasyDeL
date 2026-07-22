"""The VLM vision-feature JITs must not capture model weights as constants.

Regression test for a compile/cache poisoning bug: ``VlmPrefillHelper``'s
image/video feature closures were jitted over ``model`` directly, baking the
whole vision tower into the compiled program as captured constants. On real
VLMs that made every vision-bucket executable gigabytes large — slow to
compile, and its persistent compilation-cache entries exceeded protobuf's 2 GB
parse limit, crashing (SIGILL) any later process that read the cache.

The fix threads the weights as a ``spx.State`` jit INPUT (export outside, bind
inside). These tests assert, on a tiny Qwen3.5 VLM:

* the jitted path is output-identical to calling ``model.get_image_features``
  eagerly, and
* the lowered program contains no large dense constants (weights arrive as
  arguments, not literals).
"""

from __future__ import annotations

import os
import re
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import numpy as np
import pytest
import spectrax as spx
from jax import numpy as jnp

import easydel as ed
from easydel.inference.esurge.runners.vlm_prefill import VlmPrefillHelper

_HIDDEN = 64


def make_tiny_vlm() -> ed.Qwen3_5ForConditionalGeneration:
    text_config = ed.Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=_HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    vision_config = ed.Qwen3_5VisionConfig(
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=_HIDDEN,
        num_position_embeddings=512,
        deepstack_visual_indexes=[],
    )
    config = ed.Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        video_token_id=252,
        vision_start_token_id=253,
        vision_end_token_id=254,
        image_token_id=255,
    )
    return ed.Qwen3_5ForConditionalGeneration(
        config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32
    )


def _pixels_and_grid(model):
    vc = model.config.vision_config
    grid = (1, 8, 8)
    raw_patches = grid[0] * grid[1] * grid[2]
    patch_features = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    key = jax.random.PRNGKey(3)
    pixels = jax.random.normal(key, (raw_patches, patch_features), dtype=jnp.float32)
    return pixels, (grid,), 8


def _helper_stub(model) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_getter=lambda: model,
        image_features_jit=None,
        video_features_jit=None,
    )


def _max_constant_elements(mlir_text: str) -> int:
    """Largest dense-constant tensor (in elements) in a lowered MLIR module."""
    worst = 0
    for line in mlir_text.splitlines():
        if "constant" not in line:
            continue
        for shape in re.findall(r"tensor<([0-9x]+)x[a-z]", line):
            elems = 1
            for d in shape.split("x"):
                if d:
                    elems *= int(d)
            worst = max(worst, elems)
    return worst


def test_image_features_jit_matches_eager():
    model = make_tiny_vlm()
    stub = _helper_stub(model)
    fn = VlmPrefillHelper.get_image_features_jit(stub)
    pixels, grid_thw, max_grid = _pixels_and_grid(model)

    got_embeds, got_deepstack = fn(pixels, image_grid_thw=grid_thw, image_max_grid_size=max_grid)
    want_tuple, want_deepstack = model.get_image_features(pixels, grid_thw, max_grid)
    want = jnp.concatenate(want_tuple, axis=0)

    # jit-vs-eager float reassociation on CPU: observed max |diff| ~4.5e-5 in f32.
    np.testing.assert_allclose(np.asarray(got_embeds), np.asarray(want), rtol=1e-3, atol=2e-4)
    assert not got_deepstack and not want_deepstack, "no deepstack expected with deepstack_visual_indexes=[]"


def test_image_features_jit_captures_no_weight_constants():
    model = make_tiny_vlm()
    stub = _helper_stub(model)
    fn = VlmPrefillHelper.get_image_features_jit(stub)
    pixels, grid_thw, max_grid = _pixels_and_grid(model)

    _, state = spx.export(model)
    lowered = fn._inner_jit.lower(state, pixels, image_grid_thw=grid_thw, image_max_grid_size=max_grid)
    worst = _max_constant_elements(lowered.as_text())
    # Even this tiny model's single largest vision weight (128x256 mlp = 32k
    # elements) would blow this bound if weights were captured; position/index
    # constants stay far below it.
    assert worst < 10_000, f"lowered vision program embeds a {worst}-element constant (weight capture regressed)"


def test_image_features_jit_is_reused_and_stable():
    """The wrapper is built once and repeated calls give identical outputs."""
    model = make_tiny_vlm()
    stub = _helper_stub(model)
    fn1 = VlmPrefillHelper.get_image_features_jit(stub)
    fn2 = VlmPrefillHelper.get_image_features_jit(stub)
    assert fn1 is fn2
    pixels, grid_thw, max_grid = _pixels_and_grid(model)
    a, _ = fn1(pixels, image_grid_thw=grid_thw, image_max_grid_size=max_grid)
    b, _ = fn1(pixels, image_grid_thw=grid_thw, image_max_grid_size=max_grid)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
