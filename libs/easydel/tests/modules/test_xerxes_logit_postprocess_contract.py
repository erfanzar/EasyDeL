"""CPU-safe capability and model-lifetime checks for Xerxes split heads."""

import gc
import weakref

import pytest
from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.xerxes.modeling_xerxes import PostCross, XerxesForCausalLM
from spectrax import nn

pytestmark = pytest.mark.cpu_ok


def _uninitialized_model(postprocess, clip_cap):
    """Prepare parameter-free model metadata without creating any device arrays."""
    model = object.__new__(XerxesForCausalLM)
    object.__setattr__(model, "post_pross", postprocess)
    object.__setattr__(model, "_logit_cap_feature", None if clip_cap is None else LogitCapFeature(clip_cap))
    return model


def test_identity_head_declares_no_logit_postprocessing():
    model = _uninitialized_model(nn.Identity(), None)
    assert model.get_lm_logit_postprocess() is None
    assert resolve_lm_logit_postprocess(model) is None


@pytest.mark.parametrize("postprocess_type, clip_cap", [(PostCross, None), (PostCross, 12.0), (nn.Identity, 12.0)])
def test_split_postprocessing_does_not_keep_full_model_alive(postprocess_type, clip_cap):
    model = _uninitialized_model(postprocess_type(), clip_cap)
    reference = weakref.ref(model)
    postprocess = resolve_lm_logit_postprocess(model)
    assert callable(postprocess)
    del model
    gc.collect()
    # A stage-local postprocessor must not retain the owning model's weights.
    assert reference() is None
    assert callable(postprocess)
