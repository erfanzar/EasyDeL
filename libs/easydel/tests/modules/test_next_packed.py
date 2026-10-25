"""Rank-major packed linear attention must preserve independent per-rank recurrence."""

from types import SimpleNamespace

import numpy as np
import pytest
import spectrax as spx
from easydel.caching import RecurrentCacheView
from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextLinearAttention
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize("seq_len", [1, 4])
def test_rank_major_unified_fallback_matches_independent_rank_updates(seq_len):
    config = Qwen3NextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        use_ragged_gdr=False,
        attn_mechanism="vanilla",
    )
    model = Qwen3NextLinearAttention(
        config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(0)
    )
    states = jnp.asarray(np.random.default_rng(0).normal(size=(2, seq_len, 16)), dtype=jnp.float32)

    def cache(rows):
        return RecurrentCacheView(
            conv_state=jnp.zeros((rows, 24, 4), dtype=jnp.float32),
            recurrent_state=jnp.zeros((rows, 2, 4, 4), dtype=jnp.float32),
            positions=None,
        )

    rank_metadata = SimpleNamespace(
        query_start_loc=jnp.array([[0, seq_len], [0, seq_len]], dtype=jnp.int32),
        num_seqs=jnp.array(2, dtype=jnp.int32),
    )
    actual = model(states, mask_info=None, cache_view=cache(2), cache_metadata=rank_metadata)
    for row in range(2):
        metadata = SimpleNamespace(
            query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
            num_seqs=jnp.array(1, dtype=jnp.int32),
        )
        separate = model(states[row : row + 1], mask_info=None, cache_view=cache(1), cache_metadata=metadata)
        np.testing.assert_allclose(
            np.asarray(actual.attention_output[row : row + 1]),
            np.asarray(separate.attention_output),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(actual.cache_view.recurrent_state[row : row + 1]),
            np.asarray(separate.cache_view.recurrent_state),
            rtol=2e-4,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            np.asarray(actual.cache_view.conv_state[row : row + 1]),
            np.asarray(separate.cache_view.conv_state),
            rtol=2e-4,
            atol=2e-4,
        )


@pytest.mark.parametrize("seq_len", [1, 2, 3])
def test_short_segmented_sequence_matches_unsegmented_causal_convolution(seq_len):
    config = Qwen3NextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        use_ragged_gdr=False,
        attn_mechanism="vanilla",
    )
    model = Qwen3NextLinearAttention(
        config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision="highest",
        rngs=spx.Rngs(11),
    )
    states = jnp.asarray(np.random.default_rng(12).normal(size=(2, seq_len, 16)), dtype=jnp.float32)
    segments = MaskInfo.from_segments(jnp.zeros((2, seq_len), dtype=jnp.int32))
    # The unsegmented path uses the backend causal convolution, not the shifted
    # tap implementation, and is an independent reference for one document.
    expected = model(states, mask_info=None).attention_output
    actual = model(states, mask_info=segments).attention_output
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-4)
