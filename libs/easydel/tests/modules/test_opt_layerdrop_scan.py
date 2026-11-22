"""OPT LayerDrop state/output regressions; accelerator-only, never CPU."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.opt.modeling_opt import OPTDecoder
from easydel.modules.opt.opt_configuration import OPTConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def _decoder(*, scan_layers, dropout, attention_dropout, explicit_dropout_stream):
    """Build identical stacks with either declared or initially lazy dropout RNGs."""
    config = OPTConfig(
        vocab_size=16,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=8,
        num_attention_heads=2,
        max_position_embeddings=8,
        attn_mechanism="vanilla",
        gradient_checkpointing="nothing_saveable",
        scan_layers=scan_layers,
        dropout=dropout,
        attention_dropout=attention_dropout,
        layerdrop=0.5,
    )
    rngs = spx.Rngs(0, dropout=17) if explicit_dropout_stream else spx.Rngs(0)
    decoder = OPTDecoder(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    decoder.train()
    return decoder


def _explicit_layerdrop(decoder, ids, positions):
    """Independent Python residual selection, without OPTDecoder.forward or scan."""
    hidden = decoder.dropout_layer(decoder.embed_tokens(ids) + decoder.embed_positions(positions))
    mask = MaskInfo.dynamic_init(input_ids=ids, inputs_embeds=hidden)
    decisions = []
    for layer in decoder.layers:
        updated = layer(hidden, mask_info=mask, mode=common_types.MODE_TRAIN).hidden_states
        drop = jax.random.uniform(decoder.rngs.key("dropout")) < decoder.config.layerdrop
        decisions.append(bool(drop))
        # A dropped block is the identity, not an inverted-dropout rescaling.
        hidden = hidden if bool(drop) else updated
    return decoder.final_layer_norm(hidden), decisions


def _dropout_counter(decoder):
    """Read the packed stream's high/low counter words without consuming a key."""
    return np.asarray(decoder.rngs.stream("dropout").value)[-2:]


@pytest.mark.parametrize("use_jit", [False, True])
@pytest.mark.parametrize("explicit_dropout_stream", [False, True])
@pytest.mark.parametrize("dropout,attention_dropout", [(0.0, 0.0), (0.2, 0.15)])
def test_opt_fractional_layerdrop_scan_matches_explicit_iteration(
    use_jit, explicit_dropout_stream, dropout, attention_dropout
):
    """Scan draws independently per layer and preserves RNG advances across calls/JIT."""
    options = dict(
        dropout=dropout,
        attention_dropout=attention_dropout,
        explicit_dropout_stream=explicit_dropout_stream,
    )
    scanned = _decoder(scan_layers=True, **options)
    traced = _decoder(scan_layers=False, **options)
    reference = _decoder(scan_layers=False, **options)
    ids = jnp.array([[2, 3, 4]])
    positions = jnp.array([[0, 1, 2]])

    def run(decoder, tokens, position_ids):
        return decoder(tokens, position_ids=position_ids).last_hidden_state

    run_scan = spx.jit(run, mutable="rng") if use_jit else run
    draws_per_call = 8 * (1 + 2 * bool(dropout) + bool(attention_dropout)) + bool(dropout)
    observed_decisions = []
    for call in range(1, 4):
        actual = run_scan(scanned, ids, positions)
        python_output = run(traced, ids, positions)
        expected, decisions = _explicit_layerdrop(reference, ids, positions)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(python_output, expected, rtol=1e-5, atol=1e-5)
        for decoder in (scanned, traced, reference):
            np.testing.assert_array_equal(_dropout_counter(decoder), [0, call * draws_per_call])
        observed_decisions.append(decisions)
    assert any(any(row) and not all(row) for row in observed_decisions)
    assert any(row != observed_decisions[0] for row in observed_decisions[1:])

    # Evaluation never draws for LayerDrop (or either ordinary dropout path).
    scanned.eval()
    counter_before = _dropout_counter(scanned).copy()
    eval_output = run_scan(scanned, ids, positions)
    np.testing.assert_array_equal(run_scan(scanned, ids, positions), eval_output)
    np.testing.assert_array_equal(_dropout_counter(scanned), counter_before)
