"""Hardware-only regression coverage for Mamba parameter initialization."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.mamba import MambaConfig
from easydel.modules.mamba.modeling_mamba import MambaMixer
from easydel.modules.mamba2 import Mamba2Config
from easydel.modules.mamba2.modeling_mamba2 import Mamba2Mixer
from jax import numpy as jnp


def _config(family, **kwargs):
    options = dict(
        hidden_size=32,
        state_size=2,
        expand=2,
        num_hidden_layers=1,
        conv_kernel=2,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        **kwargs,
    )
    if family == "mamba":
        return MambaConfig(**options), MambaMixer
    return Mamba2Config(num_heads=32, head_dim=2, n_groups=1, chunk_size=4, **options), Mamba2Mixer


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_initial_time_steps_have_log_uniform_support(family):
    config, mixer_cls = _config(family)
    with config.mesh:
        mixer = mixer_cls(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))
    bias = mixer.dt_proj.bias.value if family == "mamba" else mixer.dt_bias.value
    dt = np.asarray(jax.nn.softplus(bias))
    assert np.all(dt >= config.time_step_min * (1 - 1e-5))
    assert np.all(dt <= config.time_step_max * (1 + 1e-5))
    log_fraction = np.log(dt / config.time_step_min) / np.log(config.time_step_max / config.time_step_min)
    assert 0.3 < log_fraction.mean() < 0.7
    assert np.count_nonzero(log_fraction < 0.2) > 0
    assert np.count_nonzero(log_fraction > 0.8) > 0


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_residual_projection_initialization_scales_with_depth(family):
    config, mixer_cls = _config(family, rescale_prenorm_residual=True)
    with config.mesh:
        shallow = mixer_cls(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
    config.num_hidden_layers = 4
    with config.mesh:
        deep = mixer_cls(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
    np.testing.assert_allclose(np.asarray(deep.out_proj.weight.value), np.asarray(shallow.out_proj.weight.value) / 2)
