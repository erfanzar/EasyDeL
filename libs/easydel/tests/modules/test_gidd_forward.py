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

"""GIDD diffusion LM forward: regression for four total-breakage bugs.

The family previously could not run at all: (1) a raw boolean array was
passed in the attention performer's ``mask_info`` slot (the noise-combined
mask now rides ``init_bias`` with ``mask_info=None`` — attention ops only
materialize ``init_bias`` when no mask_info is given); (2) the
input_ids/inputs_embeds XOR guard was inverted and rejected every valid
call; (3) ``nn.relu`` referenced a nonexistent spectrax attribute; (4) the
task head lacked ``compute_lm_logits``.

Note: ``head_init_scale`` defaults to 0.0 (zero-init output head, by
design), so noise-mask sensitivity must be asserted on hidden states, not
logits.
"""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gidd import GiddConfig
from easydel.modules.gidd.modeling_gidd import GiddForDiffusionLM
from jax import numpy as jnp


@pytest.fixture(scope="module")
def model():
    cfg = GiddConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    return GiddForDiffusionLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def _inputs():
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 64, (2, 16)), jnp.int32)
    noise = jnp.asarray(np.random.default_rng(1).integers(0, 2, (2, 16)).astype(bool))
    return ids, noise


def test_forward_runs_and_is_finite(model):
    ids, noise = _inputs()
    out = model(input_ids=ids, noise_mask=noise)
    assert out.logits.shape == (2, 16, 64)
    assert bool(jnp.isfinite(out.logits).all())
    assert bool(jnp.isfinite(out.last_hidden_state).all())


def test_noise_mask_changes_attention(model):
    ids, noise = _inputs()
    with_noise = model(input_ids=ids, noise_mask=noise).last_hidden_state
    without = model(input_ids=ids, noise_mask=jnp.zeros_like(noise)).last_hidden_state
    assert float(jnp.abs(with_noise - without).max()) > 1e-4


def test_input_guard_rejects_invalid_combos(model):
    ids, noise = _inputs()
    with pytest.raises(ValueError):
        model(noise_mask=noise)  # neither input_ids nor inputs_embeds
    embeds = jnp.zeros((2, 16, 64), jnp.float32)
    with pytest.raises(ValueError):
        model(input_ids=ids, inputs_embeds=embeds, noise_mask=noise)  # both
