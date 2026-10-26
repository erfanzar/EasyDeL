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

"""Public dense KDA cached continuation and packed reset parity (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import KDACacheConfig, KDACacheView
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextLinearAttention
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp

from .test_glm_dsa_cached_sparse import _hidden
from .test_glm_dsa_forward_contracts import _config


@pytest.mark.parametrize("packed_boundary", [False, True])
def test_glm5_dense_kda_cached_multitoken_continuation_matches_full_prefill(packed_boundary):
    config = _config(
        "glm5_next",
        layer_types=["linear_attention"],
        linear_num_heads=2,
        linear_head_dim=4,
        linear_conv_kernel_dim=3,
    )
    with config.mesh, jax.default_matmul_precision("highest"):
        layer = Glm5NextLinearAttention(
            config,
            layer_idx=0,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        layer.eval()
        cache_config = KDACacheConfig.create(
            num_hidden_layers=1,
            partition_axis=config.partition_axis,
            batch_size=1,
            key_dim=8,
            value_dim=8,
            d_conv=3,
            recurrent_state_shape=(2, 4, 4),
        )
        empty = KDACacheView.init(cache_config, layer_index=0, dtype=jnp.float32)
        hidden = _hidden(1, 10)
        segments = jnp.asarray([[0] * 6 + [1 if packed_boundary else 0] * 4], jnp.int32)
        full = layer(hidden, MaskInfo.from_segments(segments))
        full_cached = layer(hidden, MaskInfo.from_segments(segments), empty)
        np.testing.assert_allclose(
            np.asarray(full_cached.attention_output), np.asarray(full.attention_output), rtol=2e-5, atol=2e-6
        )
        step = jax.jit(lambda cache, x, seg: layer(x, MaskInfo.from_segments(seg), cache))
        view = empty
        begin = 0
        # The second multi-token append continues the first segment and, in
        # the packed case, also crosses a later document boundary. Decode and
        # another multi-token append must then carry the new segment's state.
        for size in (4, 3, 1, 2):
            result = step(view, hidden[:, begin : begin + size], segments[:, begin : begin + size])
            np.testing.assert_allclose(
                np.asarray(result.attention_output),
                np.asarray(full.attention_output[:, begin : begin + size]),
                rtol=2e-5,
                atol=2e-6,
            )
            view = result.cache_view
            begin += size
        for name in ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state"):
            np.testing.assert_allclose(
                np.asarray(getattr(view, name)),
                np.asarray(getattr(full_cached.cache_view, name)),
                rtol=2e-5,
                atol=2e-6,
            )
        if packed_boundary:
            independent = layer(hidden[:, 6:], MaskInfo.from_segments(jnp.zeros((1, 4), jnp.int32)))
            np.testing.assert_allclose(
                np.asarray(full.attention_output[:, 6:]),
                np.asarray(independent.attention_output),
                rtol=2e-5,
                atol=2e-6,
            )
