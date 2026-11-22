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

"""Public GPT-2 attention dispatch, precision and NumPy parity (accelerator only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.layers.attention import AttentionMechanisms
from easydel.modules.gpt2.gpt2_configuration import GPT2Config
from easydel.modules.gpt2.modeling_gpt2 import GPT2Attention
from jax import lax
from jax import numpy as jnp
from spectrax.common_types import MODE_TRAIN


@pytest.mark.parametrize("mechanism", ["vanilla", "sdpa", "flash_attn2", "ring", "auto"])
@pytest.mark.parametrize("decode_mechanism", [None, "sdpa", "flash_attn2", "ring"])
@pytest.mark.parametrize("upcast", [False, True])
def test_registered_attention_metadata_preserves_upcast_and_dispatch(monkeypatch, mechanism, decode_mechanism, upcast):
    # Resolve auto predictably, then construct and run the real public layer.
    # The registered operations themselves are never replaced.
    monkeypatch.setattr(
        "easydel.layers.attention._flexible.get_optimal_config",
        lambda: (AttentionMechanisms.FLASH_ATTN2, jnp.float16),
    )
    config = GPT2Config(
        n_embd=8,
        n_head=2,
        n_layer=3,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        attn_mechanism=mechanism,
        decode_attn_mechanism=decode_mechanism,
        attn_dtype=jnp.float16,
        attn_softmax_dtype=jnp.float32,
        reorder_and_upcast_attn=upcast,
        scale_attn_weights=False,
        scale_attn_by_inverse_layer_idx=True,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        layer = GPT2Attention(
            config,
            layer_idx=2,
            dtype=jnp.float16,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            causal=False,
            rngs=spx.Rngs(0),
        )
        hidden = jnp.asarray(np.arange(24).reshape(1, 3, 8) / 4.0, dtype=jnp.float16)
        actual = layer(hidden, None, jnp.arange(3)[None, :], MODE_TRAIN)
    performer = layer.attention_performer
    expected_dtype = jnp.float32 if upcast else jnp.float16
    expected_mechanism = "flash_attn2" if mechanism == "auto" else mechanism
    impl_names = performer.impl.get_impl_name()
    assert expected_mechanism in (impl_names if isinstance(impl_names, tuple) else (impl_names,))
    assert performer.impl.get_impl_metadata().runtime_dtype == expected_dtype
    assert performer.impl.get_impl_metadata().runtime_softmax_dtype == jnp.float32
    assert performer.metadata.runtime_dtype == expected_dtype
    assert performer.softmax_scale == pytest.approx(1.0 / 3.0)
    if decode_mechanism is None:
        assert performer.impl_decode is None
    else:
        decode_names = performer.impl_decode.get_impl_name()
        assert decode_mechanism in (decode_names if isinstance(decode_names, tuple) else (decode_names,))
        assert performer.impl_decode.get_impl_metadata().runtime_dtype == expected_dtype
        assert performer.impl_decode.get_impl_metadata().runtime_softmax_dtype == jnp.float32
    # Independent reference includes fp16 projection rounding, fp32 softmax,
    # inverse-layer scaling, and conversion back to the activation dtype.
    qkv = (np.asarray(hidden) @ np.asarray(layer.c_attn.weight.value, dtype=np.float16).T).astype(np.float16)
    qkv = (qkv + np.asarray(layer.c_attn.bias.value, dtype=np.float16)).astype(np.float16)
    query, key, value = (x.reshape(1, 3, 2, 4).astype(np.float32) for x in np.split(qkv, 3, axis=-1))
    scores = np.einsum("bqhd,bkhd->bhqk", query, key) / 3.0
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    attended = np.einsum("bhqk,bkhd->bqhd", probabilities, value).reshape(1, 3, 8).astype(np.float16)
    expected = attended @ np.asarray(layer.c_proj.weight.value, dtype=np.float16).T
    expected += np.asarray(layer.c_proj.bias.value, dtype=np.float16)
    np.testing.assert_allclose(np.asarray(actual.attention_output), expected, rtol=2e-3, atol=2e-4)
    assert actual.attention_output.dtype == jnp.float16
    assert config.attn_dtype == jnp.float16
    assert config.attn_softmax_dtype == jnp.float32
    assert config.decode_attn_mechanism == decode_mechanism
