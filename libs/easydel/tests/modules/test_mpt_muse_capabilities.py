"""CPU-safe capability and logit-transform discovery regressions (no array computation)."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess
from easydel.modules._base._base_task_module import BaseTaskModule
from easydel.modules.mosaic_mpt.modeling_mosaic import MptForCausalLM
from easydel.modules.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration
from easydel.modules.muse_glimmer.muse_glimmer_configuration import MuseGlimmerConfig


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scale,expected", [(None, 8.0), (0.0, 0.0), (2.5, 20.0), ("inv_sqrt_d_model", 2.0)])
def test_mpt_exported_logit_transform_resolves_documented_scale(scale, expected):
    class ScalarMpt(MptForCausalLM):
        @property
        def config(self):
            return SimpleNamespace(logit_scale=scale, d_model=16)

    model = object.__new__(ScalarMpt)
    object.__setattr__(model, "_logit_cap_feature", None)
    postprocess = model.get_lm_logit_postprocess()
    actual = 8.0 if postprocess is None else postprocess(8.0)
    assert actual == expected


@pytest.mark.cpu_ok
def test_muse_config_declares_flat_patch_layout_without_family_name_lookup():
    config = MuseGlimmerConfig()
    config.model_type = "custom_muse_alias"
    manager = SimpleNamespace(model=SimpleNamespace(config=config))
    assert MultiModalManager._supports_flat_patch_inputs(manager) is True


def _head_contract(multiplier, softcap, clip_cap=None):
    text_config = SimpleNamespace(output_multiplier=multiplier, final_logit_softcapping=softcap)
    model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: text_config),
        _logit_cap_feature=None if clip_cap is None else SimpleNamespace(cap_value=clip_cap),
    )
    model.get_base_lm_logit_postprocess = lambda: BaseTaskModule.get_base_lm_logit_postprocess(model)
    model.get_lm_logit_postprocess = lambda: MuseGlimmerForConditionalGeneration.get_lm_logit_postprocess(model)
    return model


@pytest.mark.cpu_ok
def test_identity_muse_head_has_no_split_postprocessing():
    assert resolve_lm_logit_postprocess(_head_contract(1.0, None)) is None


@pytest.mark.cpu_ok
@pytest.mark.parametrize("multiplier,softcap,clip_cap", [(2.5, None, None), (2.5, 4.0, 3.0), (1.0, None, 3.0)])
def test_muse_split_head_transform_captures_settings_not_model(multiplier, softcap, clip_cap):
    postprocess = resolve_lm_logit_postprocess(_head_contract(multiplier, softcap, clip_cap))
    assert callable(postprocess)

    def assert_scalar_capture_only(callback):
        for cell in callback.__closure__ or ():
            captured = cell.cell_contents
            if callable(captured):
                assert getattr(captured, "__self__", None) is None
                assert_scalar_capture_only(captured)
            else:
                assert captured is None or isinstance(captured, (int, float))

    assert_scalar_capture_only(postprocess)


@pytest.mark.cpu_ok
def test_muse_split_head_preserves_custom_clipping_with_identity_scaling():
    class CustomMuse(MuseGlimmerForConditionalGeneration):
        @property
        def config(self):
            text_config = SimpleNamespace(output_multiplier=1.0, final_logit_softcapping=None)
            return SimpleNamespace(get_text_config=lambda: text_config)

        def apply_logit_cap(self, logits):
            # Pure Python lets this regression exercise the override without
            # constructing parameters or computing with JAX arrays.
            return max(-0.5, min(logits, 0.5))

    model = object.__new__(CustomMuse)
    object.__setattr__(model, "_logit_cap_feature", None)
    postprocess = resolve_lm_logit_postprocess(model)
    assert callable(postprocess)
    assert postprocess(2.0) == 0.5
    assert postprocess(-2.0) == -0.5
    assert postprocess(0.25) == 0.25
