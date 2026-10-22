"""Host-only VLM projection compatibility regressions (no JAX dispatch)."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.modules._base import BaseCausalLMModule, BaseConditionalGenerationModule, BaseVisionLanguageModule
from easydel.modules._base._base_task_module import BaseTaskModule
from easydel.modules._base._features import LogitCapFeature
from easydel.modules._base._vlm_features import MultiDimensionalRoPEFeature, VisionEncoderFeature

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("wrapper", [BaseCausalLMModule, BaseConditionalGenerationModule])
@pytest.mark.parametrize("constructor_tie,config_tie", [(True, False), (False, True), (False, False)])
def test_inherited_projection_and_trace_safe_projection_resolve_the_same_weights(wrapper, constructor_tie, config_tie):
    class Head:
        def __call__(self, hidden, *, w=None):
            return hidden, w

        native_forward = __call__

    class Harness:
        apply_lm_head = wrapper.apply_lm_head
        make_lm_head_fn = EasyDeLBaseModule.make_lm_head_fn
        _tie_embeddings_enabled = EasyDeLBaseModule._tie_embeddings_enabled
        config = SimpleNamespace(tie_word_embeddings=config_tie)
        _tie_embeddings_feature = SimpleNamespace(tie=constructor_tie)

        def get_embedding(self):
            return SimpleNamespace(weight=SimpleNamespace(value=SimpleNamespace(T=embedding_weight)))

        def get_lm_head(self):
            return head

    embedding_weight, hidden, head = object(), object(), Head()
    model = Harness()
    expected = (hidden, embedding_weight if constructor_tie or config_tie else None)
    assert model.apply_lm_head(hidden) == expected
    assert model.make_lm_head_fn()(hidden) == expected


def test_vision_selection_extension_preserves_layer_and_strategy_overrides():
    states = (np.arange(12).reshape(1, 3, 4), np.arange(12, 24).reshape(1, 3, 4))
    model = SimpleNamespace(_vision_encoder_feature=VisionEncoderFeature(-1, "default"))
    np.testing.assert_array_equal(BaseVisionLanguageModule._select_vision_features(model, states), states[-1][:, 1:])
    np.testing.assert_array_equal(
        BaseVisionLanguageModule._select_vision_features(model, states, feature_layer=0, select_strategy="full"),
        states[0],
    )
    np.testing.assert_array_equal(
        BaseVisionLanguageModule._select_vision_features(model, states, select_strategy="pooled"), states[-1][:, :1]
    )


def test_public_image_extraction_consumes_constructor_and_call_selection_settings(monkeypatch):
    class HostVLM(BaseVisionLanguageModule):
        _auto_register = False

        @property
        def base_model(self):
            return self.backbone

        def _clear_dynamic_language_stage_hints(self):
            pass

        def get_image_features(self, pixel_values, **kwargs):
            encoded = self.get_vision_tower()(pixel_values, output_hidden_states=True)
            return self.get_projector()(self._select_vision_features(encoded.hidden_states, **kwargs))

    states = (np.arange(12).reshape(1, 3, 4), np.arange(12, 24).reshape(1, 3, 4))
    backbone = SimpleNamespace(
        vision_tower=lambda *args, **kwargs: SimpleNamespace(hidden_states=states),
        multi_modal_projector=lambda features: features + 100,
    )

    def host_parent_init(self, *, base_model, **kwargs):
        self.backbone = base_model

    monkeypatch.setattr(BaseConditionalGenerationModule, "__init__", host_parent_init)
    model = HostVLM(
        config=SimpleNamespace(),
        base_model=backbone,
        rngs=None,
        vision_feature_layer=0,
        vision_feature_select_strategy="full",
    )
    np.testing.assert_array_equal(model.get_image_features(object()), states[0] + 100)
    np.testing.assert_array_equal(
        model.get_image_features(object(), feature_layer=1, select_strategy="default"), states[1][:, 1:] + 100
    )
    np.testing.assert_array_equal(
        model.get_image_features(object(), feature_layer=1, select_strategy="pooled"), states[1][:, :1] + 100
    )


def test_mrope_feature_configures_frequency_allocation_without_changing_layout_flags():
    rotary = SimpleNamespace(rotary_dim=16, mrope_section=(2, 3, 3), mrope_interleaved=False, repetition_style=True)
    feature = MultiDimensionalRoPEFeature(mrope_section=(4, 2, 2))
    feature.configure_rotary(rotary)
    assert rotary.mrope_section == (4, 2, 2)
    assert rotary.mrope_interleaved is False
    assert rotary.repetition_style is True
    smaller_rotary = SimpleNamespace(rotary_dim=8)
    feature.configure_rotary(smaller_rotary)
    assert smaller_rotary.mrope_section == (2, 1, 1)
    with pytest.raises(ValueError, match="incompatible"):
        feature.configure_rotary(SimpleNamespace(rotary_dim=6))


@pytest.mark.parametrize("section", [(0, 0, 0), (-1, 2, 3), (1, 2), (1.0, 2, 3)])
def test_invalid_mrope_allocation_is_rejected(section):
    with pytest.raises(ValueError, match="mrope_section"):
        MultiDimensionalRoPEFeature(mrope_section=section)


def test_vlm_constructor_integrates_custom_mrope_with_family_rotary_modules(monkeypatch):
    import spectrax as spx
    from easydel.layers.rotary import MultiModalRotaryEmbedding

    class HostRotary(MultiModalRotaryEmbedding):
        def __init__(self):
            # Configuration-only rotary fixture: no inverse frequencies/arrays.
            self.rotary_dim = 8
            self.mrope_section = (2, 1, 1)
            self.mrope_interleaved = False
            self.repetition_style = True

    class Backbone(spx.Module):
        def __init__(self):
            self.rotary = HostRotary()

    class HostVLM(BaseVisionLanguageModule):
        _auto_register = False
        _uses_mrope = True

        @property
        def base_model(self):
            return self.backbone

        def get_image_features(self, pixel_values, **kwargs):
            return pixel_values

        def _clear_dynamic_language_stage_hints(self):
            pass

    def host_parent_init(self, *, base_model, **kwargs):
        self.backbone = base_model

    monkeypatch.setattr(BaseConditionalGenerationModule, "__init__", host_parent_init)
    default = HostVLM(config=SimpleNamespace(), base_model=Backbone(), rngs=None)
    custom = HostVLM(config=SimpleNamespace(), base_model=Backbone(), rngs=None, mrope_section=(1, 2, 1))
    assert default.base_model.rotary.mrope_section == (2, 1, 1)
    assert custom.base_model.rotary.mrope_section == (1, 2, 1)
    assert custom.base_model.rotary.mrope_interleaved is False
    assert custom.base_model.rotary.repetition_style is True


def test_base_clipping_stage_has_no_model_or_array_capture_and_preserves_identity():
    model = SimpleNamespace(_logit_cap_feature=None)
    assert BaseTaskModule.get_base_lm_logit_postprocess(model) is None
    model._logit_cap_feature = LogitCapFeature(3.0)
    clip = BaseTaskModule.get_base_lm_logit_postprocess(model)
    assert clip is not None
    assert tuple(cell.cell_contents for cell in clip.__closure__) == (3.0,)


@pytest.mark.parametrize("cap", [None, 3.0])
def test_authoritative_base_postprocess_declares_identity_or_captures_only_cap(cap):
    from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess

    class HostTask:
        # Exercise the public contract without constructing parameters or
        # evaluating a JAX operation; numerical parity has separate real-model
        # coverage in test_split_head_compute.py.
        get_base_lm_logit_postprocess = BaseTaskModule.get_base_lm_logit_postprocess
        get_lm_logit_postprocess = BaseTaskModule.get_lm_logit_postprocess

        def __init__(self):
            self._logit_cap_feature = None if cap is None else LogitCapFeature(cap)
            # Unrelated conventions must not override the model's contract.
            self.config = SimpleNamespace(final_logit_softcapping=5.0)
            self.logit_scale = 7.0
            self.parameters = object()

    model = HostTask()
    model_ref = weakref.ref(model)
    captured = model.get_lm_logit_postprocess()
    resolved = resolve_lm_logit_postprocess(model)
    if cap is None:
        assert captured is None
        assert resolved is None
    else:
        assert captured is not None
        assert resolved is not None
        assert tuple(cell.cell_contents for cell in captured.__closure__) == (cap,)
        assert tuple(cell.cell_contents for cell in resolved.__closure__) == (cap,)
        # The compiled callable owns a snapshot, not mutable model features.
        model._logit_cap_feature = LogitCapFeature(9.0)
        updated = model.get_lm_logit_postprocess()
        assert tuple(cell.cell_contents for cell in updated.__closure__) == (9.0,)
        assert tuple(cell.cell_contents for cell in captured.__closure__) == (cap,)
    del model
    assert model_ref() is None


def test_split_resolver_does_not_infer_transforms_for_models_without_hooks():
    from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess

    model = SimpleNamespace(
        logit_scale=7.0,
        config=SimpleNamespace(final_logit_softcapping=5.0, logit_cap=3.0, output_multiplier_scale=2.0),
    )
    assert resolve_lm_logit_postprocess(model) is None


@pytest.mark.parametrize("override_kind", ["method", "static", "instance"])
@pytest.mark.parametrize("feature_cap", [None, 0.25])
def test_custom_logit_cap_matches_full_trace_safe_and_split_contracts(monkeypatch, override_kind, feature_cap):
    from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess
    from easydel.modules._base import _base_task_module

    class Head:
        def __call__(self, hidden, *, w=None):
            return hidden * 2.0

        native_forward = __call__

    class HostTask(BaseCausalLMModule):
        _auto_register = False

        def __init__(self):
            # No parameter construction or JAX dispatch. Exercise the inherited
            # public projection methods with a NumPy-only host head instead.
            self._logit_cap_feature = None if feature_cap is None else LogitCapFeature(feature_cap)
            self.custom_cap = 3.0
            self.custom_offset = 0.5

        def get_lm_head(self):
            return Head()

        def _tie_embeddings_enabled(self):
            return False

        def _resolve_lmhead_chunksize(self, seq_len=None):
            return None

        def _constrain_lm_logits(self, logits):
            return logits

        def apply_logit_cap(self, logits):
            return np.clip(logits, -self.custom_cap, self.custom_cap) + self.custom_offset

    if override_kind == "static":
        HostTask.apply_logit_cap = staticmethod(lambda logits: np.clip(logits, -3.0, 3.0) + 0.5)
    model = HostTask()
    if override_kind == "instance":
        model.apply_logit_cap = lambda logits: np.clip(logits, -3.0, 3.0) + 0.5
    monkeypatch.setattr(_base_task_module, "checkpoint_name", lambda value, name: value)
    hidden = np.array([[-2.0, -0.25, 0.0, 0.25, 2.0]])
    expected = np.array([[-2.5, 0.0, 0.5, 1.0, 3.5]])
    np.testing.assert_array_equal(model.compute_lm_logits(hidden), expected)
    np.testing.assert_array_equal(model.make_lm_head_fn()(hidden), expected)
    postprocess = resolve_lm_logit_postprocess(model)
    assert postprocess is not None
    np.testing.assert_array_equal(postprocess(model.apply_lm_head(hidden)), expected)
    trace_projection = model.make_lm_head_fn()
    model_ref = weakref.ref(model)
    # Snapshots must neither follow later mutations nor keep the backbone alive.
    model.custom_cap = 100.0
    model.custom_offset = 50.0
    del model
    gc.collect()  # SpectraX's host-side bookkeeping can contain ordinary cycles.
    assert model_ref() is None
    np.testing.assert_array_equal(postprocess(hidden * 2.0), expected)
    np.testing.assert_array_equal(trace_projection(hidden), expected)
