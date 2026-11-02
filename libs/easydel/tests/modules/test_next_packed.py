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


def test_packed_rows_follow_recurrent_state_indices():
    """eSurge names every row's pool slot (``recurrent_state_indices``) when a row window can start past slot 0.

    The step on a 4-slot pool with indices equals the same step on a pool holding only those slots in row order (no
    indices), and the other slots stay bit-identical.
    """
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
        config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(21)
    )
    rng = np.random.default_rng(22)
    conv_pool = rng.normal(size=(4, 24, 4)).astype(np.float32)
    rec_pool = (0.1 * rng.normal(size=(4, 2, 4, 4))).astype(np.float32)
    slots = [3, 1]
    tokens = jnp.asarray(rng.normal(size=(1, 4, 16)), dtype=jnp.float32)
    states = jnp.zeros((1, 8, 16), dtype=jnp.float32).at[:, :4].set(tokens)
    # Row 0: one decode token continuing its request; row 1: a 3-token continuation.
    metadata = SimpleNamespace(
        query_start_loc=jnp.array([0, 1, 4], dtype=jnp.int32),
        num_seqs=jnp.array(2, dtype=jnp.int32),
        context_lens=jnp.array([9, 7], dtype=jnp.int32),
    )
    indexed = SimpleNamespace(**vars(metadata), recurrent_state_indices=jnp.array([slots], dtype=jnp.int32))

    def view(conv, rec):
        return RecurrentCacheView(conv_state=jnp.asarray(conv), recurrent_state=jnp.asarray(rec), positions=None)

    want = model(states, mask_info=None, cache_view=view(conv_pool[slots], rec_pool[slots]), cache_metadata=metadata)
    got = model(states, mask_info=None, cache_view=view(conv_pool, rec_pool), cache_metadata=indexed)
    np.testing.assert_allclose(np.asarray(got.attention_output), np.asarray(want.attention_output), rtol=1e-5, atol=1e-6)
    for name, pool in (("conv_state", conv_pool), ("recurrent_state", rec_pool)):
        new = np.asarray(getattr(got.cache_view, name))
        np.testing.assert_allclose(
            new[slots], np.asarray(getattr(want.cache_view, name)), rtol=1e-5, atol=1e-6, err_msg=name
        )
        np.testing.assert_array_equal(new[[0, 2]], pool[[0, 2]], err_msg=name)
