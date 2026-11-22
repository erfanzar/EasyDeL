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

"""Split PP LM-head logit post-processing resolution.

The split MPMD LM-head jits only hold the exported projection, so they cannot call
``model.compute_lm_logits``. A model that overrides ``compute_lm_logits`` exposes its
extra math through ``get_lm_logit_postprocess()``; inference must use that ordered
contract rather than guessing transformations from configuration attributes.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import (
    _project_stage_lm_head,
    resolve_lm_logit_postprocess,
)
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_model_hook_is_authoritative_over_config_conventions():
    def marker(logits):
        return logits + 7

    model = SimpleNamespace(
        get_lm_logit_postprocess=lambda: marker,
        logit_scale=3.0,
        config=SimpleNamespace(final_logit_softcapping=5.0),
    )
    assert resolve_lm_logit_postprocess(model)(3) == 10


@pytest.mark.cpu_ok
def test_hook_may_declare_no_postprocessing():
    model = SimpleNamespace(get_lm_logit_postprocess=lambda: None, config=SimpleNamespace(final_logit_softcapping=5.0))
    assert resolve_lm_logit_postprocess(model) is None


@pytest.mark.cpu_ok
def test_no_postprocessing_configured_resolves_to_none():
    assert resolve_lm_logit_postprocess(SimpleNamespace(config=SimpleNamespace())) is None


@pytest.mark.cpu_ok
def test_config_attributes_do_not_invent_an_unadvertised_transform():
    # BaseTaskModule exposes the complete ordered hook and forbids inferring
    # transforms from config (including an identity hook).
    # A duck-typed model without that hook has no declared transform;
    # its incidental field names cannot establish clip/scale/softcap ordering.
    model = SimpleNamespace(
        _logit_cap_feature=SimpleNamespace(cap_value=2.0),
        logit_scale=2.0,
        output_multiplier_scale=3.0,
        base_model=SimpleNamespace(lm_head_multiplier=4.0),
        config=SimpleNamespace(final_logit_softcapping=5.0),
    )
    assert resolve_lm_logit_postprocess(model) is None


class _Projection(spx.Module):
    def native_forward(self, hidden):
        return hidden * 2

    def attend(self, hidden):
        return hidden * 3


@pytest.mark.parametrize("tied, scale", [(False, 2), (True, 3)])
def test_stage_projection_applies_authoritative_model_postprocess(tied, scale):
    graphdef, state = spx.export(_Projection())
    hidden = jnp.asarray([[-20.0, 0.0, 40.0]], dtype=jnp.float32)
    model = SimpleNamespace(get_lm_logit_postprocess=lambda: lambda x: 30 * jnp.tanh(x / 30))
    logits = _project_stage_lm_head(
        graphdef,
        state,
        hidden,
        uses_tied_projection=tied,
        postprocess=resolve_lm_logit_postprocess(model),
        scope="test",
    )
    expected = 30 * np.tanh(np.asarray(hidden) * scale / 30)
    np.testing.assert_allclose(np.asarray(logits), expected, rtol=1e-5, atol=1e-5)
