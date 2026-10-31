"""FCM numerical regressions; run on an accelerator, not on CPU."""

import jax
import numpy as np
from easydel.infra.base_config import EasyDeLBaseConfig


def test_get_fcm_mask_samples_reproducible_boolean_mask_with_visible_first_column():
    config = EasyDeLBaseConfig(fcm_max_ratio=0.8, fcm_min_ratio=0.2)
    key = jax.random.PRNGKey(0)
    mask = config.get_fcm_mask(2, 6, deterministic=False, rng=key)
    repeated = config.get_fcm_mask(2, 6, deterministic=False, rng=key)
    assert mask.shape == (2, 1, 6, 6)
    assert mask.dtype == bool
    np.testing.assert_array_equal(np.asarray(mask), np.asarray(repeated))
    assert np.asarray(mask)[:, :, :, 0].all()
    assert not np.asarray(mask)[:, :, :, 1:].all()


def test_documented_shape_and_fixed_ratio_api():
    config = EasyDeLBaseConfig()
    key = jax.random.PRNGKey(0)
    visible = config.get_fcm_mask(key=key, shape=(2, 1, 4, 4), fcm_ratio=0.0)
    hidden = config.get_fcm_mask(key=key, shape=(2, 1, 4, 4), fcm_ratio=1.0)
    assert np.asarray(visible).all()
    expected = np.zeros((2, 1, 4, 4), dtype=bool)
    expected[:, :, :, 0] = True
    np.testing.assert_array_equal(np.asarray(hidden), expected)
    assert config.get_fcm_mask(key=key, shape=(2, 1, 4, 4), fcm_ratio=0.1, deterministic=True) is None
