"""Parameter-free Grok split-head hook contracts (no array computation)."""

import gc
import weakref

import pytest
from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.grok_1 import Grok1ForCausalLM


def _model_metadata(scale, clip_cap):
    """Construct only metadata without allocating parameters or device arrays."""
    model = object.__new__(Grok1ForCausalLM)
    object.__setattr__(model, "output_multiplier_scale", scale)
    object.__setattr__(model, "_logit_cap_feature", None if clip_cap is None else LogitCapFeature(clip_cap))
    return model


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_split_head_captures_output_scale_snapshot(scale):
    model = _model_metadata(scale, None)
    postprocess = resolve_lm_logit_postprocess(model)
    assert postprocess(4.0) == 4.0 * scale
    assert postprocess(-4.0) == -4.0 * scale
    object.__setattr__(model, "output_multiplier_scale", 3.0)
    assert postprocess(4.0) == 4.0 * scale
    assert model.get_lm_logit_postprocess()(4.0) == 12.0


@pytest.mark.cpu_ok
@pytest.mark.parametrize("clip_cap", [None, 0.25])
def test_split_head_postprocess_does_not_retain_live_model(clip_cap):
    model = _model_metadata(0.5, clip_cap)
    reference = weakref.ref(model)
    postprocess = resolve_lm_logit_postprocess(model)
    del model
    gc.collect()
    assert reference() is None
    assert callable(postprocess)
