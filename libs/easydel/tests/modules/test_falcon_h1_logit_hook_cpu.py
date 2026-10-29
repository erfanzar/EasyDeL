"""Non-computation checks for Falcon-H1's trace-safe public logit hook."""

from types import SimpleNamespace

import pytest
from easydel.modules.falcon_h1 import FalconH1ForCausalLM


@pytest.mark.cpu_ok
def test_falcon_h1_logit_hook_snapshots_multiplier_without_capturing_live_model():
    # The hook only needs scalar settings, not initialized model parameters.
    model = object.__new__(FalconH1ForCausalLM)
    object.__setattr__(model, "model", SimpleNamespace(lm_head_multiplier=2.5))
    object.__setattr__(model, "_logit_cap_feature", None)
    postprocess = model.get_lm_logit_postprocess()
    assert postprocess(4.0) == 10.0
    assert postprocess(-4.0) == -10.0
    model.model.lm_head_multiplier = 7.0
    # A compiled/split head must retain its captured settings, not reach back
    # into a mutable module tree while being traced.
    assert postprocess(4.0) == 10.0
    assert model.get_lm_logit_postprocess()(4.0) == 28.0
    assert all(
        cell.cell_contents is not model and cell.cell_contents is not model.model for cell in postprocess.__closure__
    )
