"""Hardware-only tests for configurable gate ordering and normalization."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.mamba2 import Mamba2Config
from easydel.modules.mamba2.modeling_mamba2 import Mamba2Mixer, MambaRMSNormGated
from jax import numpy as jnp


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("rms_norm", [False, True])
@pytest.mark.parametrize("with_gate", [False, True])
def test_gated_norm_matches_independent_reference(norm_before_gate, rms_norm, with_gate):
    values = np.array([[[0.2, -1.3, 2.1, 0.7]]], dtype=np.float32)
    gate = np.array([[[0.1, 1.4, -0.7, 2.0]]], dtype=np.float32)
    activated_gate = gate / (1 + np.exp(-gate))
    norm = MambaRMSNormGated(4, eps=1e-5, norm_before_gate=norm_before_gate, rms_norm=rms_norm)
    norm.weight.value = jnp.array([0.5, 2.0, -1.0, 0.8])
    expected = values.copy()
    if with_gate and not norm_before_gate:
        expected *= activated_gate
    if not rms_norm:
        expected -= expected.mean(axis=-1, keepdims=True)
    expected /= np.sqrt(np.mean(expected**2, axis=-1, keepdims=True) + 1e-5)
    expected *= np.array([0.5, 2.0, -1.0, 0.8], dtype=np.float32)
    if with_gate and norm_before_gate:
        expected *= activated_gate
    output = norm(jnp.asarray(values), jnp.asarray(gate) if with_gate else None)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("rms_norm", [False, True])
def test_mixer_legacy_config_preserves_hf_output_normalization(norm_before_gate, rms_norm):
    config = Mamba2Config(
        hidden_size=4,
        expand=1,
        num_heads=2,
        head_dim=2,
        n_groups=1,
        state_size=2,
        num_hidden_layers=1,
        conv_kernel=2,
        norm_before_gate=norm_before_gate,
        rms_norm=rms_norm,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    with config.mesh:
        mixer = Mamba2Mixer(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    values = np.array([[[0.5, -2.0, 1.3, 0.8]]], dtype=np.float32)
    gate = np.array([[[1.2, -0.7, 0.9, 0.3]]], dtype=np.float32)
    activated_gate = gate / (1 + np.exp(-gate))
    # Legacy mixer flags do not alter HF checkpoint normalization: gate first,
    # then RMSNorm. Standalone configurable norm coverage remains above.
    expected = values * activated_gate
    expected /= np.sqrt(np.mean(expected**2, axis=-1, keepdims=True) + config.layer_norm_epsilon)
    # Gate activation and centered variance use float32 device elementary
    # functions. Near-zero normalized components need an absolute ulp budget,
    # just as in the independent primitive reference above (TPU error < 9e-7).
    np.testing.assert_allclose(
        np.asarray(mixer.norm(jnp.asarray(values), jnp.asarray(gate))), expected, rtol=1e-5, atol=1e-6
    )
