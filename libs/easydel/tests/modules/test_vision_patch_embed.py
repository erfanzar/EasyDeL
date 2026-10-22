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

"""API and numerical regressions for the extracted vision patch embedding.

Only the explicitly marked API tests may be run without an accelerator.
"""

import inspect

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.embeddings._vision import VisionPatchEmbed
from easydel.modules.minimax_m3_vl.modeling_minimax_m3_vl import Qwen2VLPatchEmbed as MiniMaxPatchEmbed
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLPatchEmbed
from jax import numpy as jnp
from spectrax import nn


@pytest.mark.cpu_ok
def test_legacy_imports_are_shared_class_aliases():
    """Existing public imports remain usable without a cross-family dependency."""
    assert Qwen2VLPatchEmbed is VisionPatchEmbed
    assert MiniMaxPatchEmbed is VisionPatchEmbed


@pytest.mark.cpu_ok
def test_constructor_retains_parameter_order_defaults_and_keyword_only_rngs():
    """The move preserves positional arguments, dtype defaults and RNG requirements."""
    parameters = inspect.signature(VisionPatchEmbed.__init__).parameters
    assert tuple(parameters) == (
        "self",
        "patch_size",
        "temporal_patch_size",
        "in_channels",
        "embed_dim",
        "precision",
        "dtype",
        "param_dtype",
        "partition_manager",
        "rngs",
    )
    defaults = {name: parameter.default for name, parameter in parameters.items()}
    assert defaults == {
        "self": inspect.Parameter.empty,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "embed_dim": 1152,
        "precision": None,
        "dtype": jnp.bfloat16,
        "param_dtype": jnp.bfloat16,
        "partition_manager": None,
        "rngs": inspect.Parameter.empty,
    }
    assert parameters["rngs"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("temporal_patch_size", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_patch_order_projection_dtype_and_parameter_paths(temporal_patch_size, dtype):
    """Compare the public forward against independent channel-first NumPy contraction."""
    if jax.default_backend() not in {"tpu", "gpu", "cuda", "rocm"}:
        pytest.skip("Numerical patch embedding regression requires an accelerator")

    patch_size, in_channels, embed_dim, num_patches = 2, 3, 4, 3
    kwargs = dict(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        precision=jax.lax.Precision.HIGHEST,
        dtype=dtype,
        param_dtype=dtype,
    )
    patch = VisionPatchEmbed(**kwargs, rngs=spx.Rngs(17))
    reference_conv = nn.Conv(
        in_channels=in_channels,
        out_channels=embed_dim,
        kernel_size=(temporal_patch_size, patch_size, patch_size),
        stride=(temporal_patch_size, patch_size, patch_size),
        use_bias=False,
        dtype=dtype,
        param_dtype=dtype,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(17),
    )
    # The move must retain the original Conv initializer, RNG sequence and storage dtype.
    np.testing.assert_array_equal(np.asarray(patch.proj.weight.value), np.asarray(reference_conv.weight.value))
    assert patch.proj.weight.value.dtype == dtype

    pixels = np.arange(num_patches * in_channels * temporal_patch_size * patch_size**2, dtype=np.float32)
    pixels = (pixels % 11 - 5).reshape(num_patches, -1)
    weight = np.arange(temporal_patch_size * patch_size**2 * in_channels * embed_dim, dtype=np.float32)
    weight = ((weight % 7 - 3) / 8).reshape(temporal_patch_size, patch_size, patch_size, in_channels, embed_dim)
    patch.proj.weight.value = jnp.asarray(weight, dtype=dtype)
    channel_first = pixels.reshape(num_patches, in_channels, temporal_patch_size, patch_size, patch_size)
    expected = np.einsum("ncthw,thwce->ne", channel_first, weight)
    actual = patch(jnp.asarray(pixels))
    assert actual.shape == (num_patches, embed_dim)
    assert actual.dtype == dtype
    tolerance = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(np.asarray(actual).astype(np.float32), expected, rtol=tolerance, atol=tolerance)

    # These are the unchanged child names used by the two owning vision towers.
    class VisionTowers(spx.Module):
        def __init__(self):
            self.patch_embed = patch
            self.embeddings = VisionPatchEmbed(**kwargs, rngs=spx.Rngs(18))

    _, state = spx.export(VisionTowers())
    parameters = {path: leaf for collection, path, leaf in state.items() if collection == "parameters"}
    assert set(parameters) == {"patch_embed.proj.weight", "embeddings.proj.weight"}
    for leaf in parameters.values():
        value = leaf.value if hasattr(leaf, "value") else leaf
        assert value.shape == (temporal_patch_size, patch_size, patch_size, in_channels, embed_dim)
        assert value.dtype == dtype
