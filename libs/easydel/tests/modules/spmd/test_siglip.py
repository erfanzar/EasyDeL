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

"""Tests for SigLIP model."""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import transformers
from easydel.modules.siglip.modeling_siglip import SiglipVisionEmbeddings

try:
    from tests.modules.test_utils import BaseModuleTester
except ImportError:
    from tests.modules.test_utils import BaseModuleTester  # pyright: ignore[reportImplicitRelativeImport]


class TestSigLIP:
    """Test suite for SigLIP model."""

    @pytest.fixture
    def siglip_vision_config(self, small_model_config):
        """Create SigLIP vision config."""
        return ed.SiglipVisionConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=384,
            patch_size=14,
        )

    @pytest.fixture
    def siglip_text_config(self, small_model_config):
        """Create SigLIP text config."""
        return ed.SiglipTextConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,  # Must be >= sequence_length (128)
        )

    @pytest.fixture
    def siglip_config(self, siglip_vision_config, siglip_text_config):
        """Create SigLIP config."""
        return ed.SiglipConfig(
            vision_config=siglip_vision_config,
            text_config=siglip_text_config,
        )

    def test_vision_model(self, siglip_vision_config, small_model_config):
        """Test SiglipVisionModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="siglip_vision_model",
            hf_class=transformers.SiglipVisionModel,
            task=ed.TaskType.BASE_VISION,  # Registered as BASE_VISION, not BASE_MODULE
            config=siglip_vision_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SigLIP vision BASE_VISION failed: {result.error_message or result.comparison.details}"

    def test_text_model(self, siglip_text_config, small_model_config):
        """Test SiglipTextModel."""
        tester = BaseModuleTester()
        result = tester.run(
            module_name="siglip_text_model",
            hf_class=transformers.SiglipTextModel,
            task=ed.TaskType.BASE_MODULE,
            config=siglip_text_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"SigLIP text BASE_MODULE failed: {result.error_message or result.comparison.details}"


@pytest.mark.parametrize("ambient", [None, "bfloat16"])
def test_vision_embeddings_highest_precision_matches_numpy(ambient):
    """Patch projection honors model precision independently of the global default."""
    config = ed.SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=8,
        patch_size=4,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    embeddings = SiglipVisionEmbeddings(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    rng = np.random.default_rng(37)
    pixels = rng.normal(size=(2, 3, 8, 8)).astype(np.float32)
    kernel = rng.normal(size=(4, 4, 3, 8)).astype(np.float32)
    bias = rng.normal(size=(8,)).astype(np.float32)
    positions = rng.normal(size=(4, 8)).astype(np.float32)
    embeddings.patch_embedding.weight.value = jnp.asarray(kernel)
    embeddings.patch_embedding.bias.value = jnp.asarray(bias)
    embeddings.position_embedding.weight.value = jnp.asarray(positions)

    # Explicit patch extraction and NumPy contraction, not a second JAX conv.
    expected = np.empty((2, 4, 8), dtype=np.float64)
    for row in range(2):
        for col in range(2):
            patch = pixels[:, :, 4 * row : 4 * (row + 1), 4 * col : 4 * (col + 1)]
            expected[:, 2 * row + col] = (
                np.einsum("bchw,hwco->bo", patch.astype(np.float64), kernel.astype(np.float64))
                + bias
                + positions[2 * row + col]
            )
    with jax.default_matmul_precision(ambient):
        actual = spx.jit(lambda module, inputs: module(inputs))(embeddings, jnp.asarray(pixels))
    assert actual.shape == (2, 4, 8)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
