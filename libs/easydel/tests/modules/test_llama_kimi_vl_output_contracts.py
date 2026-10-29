"""Array-free public output, shape and composite-head contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules.kimi_vl.modeling_kimi_vl import KimiVLForConditionalGeneration
from easydel.modules.llama import VisionLlamaConfig, VisionLlamaForCausalLM
from easydel.modules.llama import modeling_vision_llama as vision
from easydel.modules.llama4 import Llama4VisionConfig
from easydel.modules.llama4 import modeling_llama4 as llama4
from easydel.modules.mamba.modeling_mamba import MambaForCausalLM
from easydel.modules.mamba2.modeling_mamba2 import Mamba2ForCausalLM

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("model_class", [MambaForCausalLM, Mamba2ForCausalLM])
def test_recurrent_public_output_preserves_packed_metadata_and_final_states(model_class):
    hidden, mask = object(), object()
    child = SimpleNamespace(last_hidden_state=hidden, hidden_states=None, cache=None, cache_params=None)

    def backbone(**kwargs):
        # The public wrapper must not discard the document-boundary carrier.
        assert kwargs["mask_info"] is mask
        return child

    model = SimpleNamespace(backbone=backbone)
    output = model_class.forward(model, inputs_embeds=object(), mask_info=mask, apply_lm_head=False)
    assert output.logits is None
    assert output.last_hidden_state is hidden
    assert output.hidden_states is None


def test_kimi_public_headless_output_preserves_decoder_final_states():
    hidden = object()
    model = SimpleNamespace(language_model=lambda **kwargs: SimpleNamespace(logits=None, last_hidden_state=hidden))
    output = KimiVLForConditionalGeneration.forward(model, inputs_embeds=object(), apply_lm_head=False)
    assert output.logits is None
    assert output.last_hidden_state is hidden


class _ShapeLinear:
    def __init__(self, in_features, out_features, **kwargs):
        self.in_features, self.out_features = in_features, out_features

    def __call__(self, hidden):
        assert hidden.shape[-1] == self.in_features
        return np.zeros((*hidden.shape[:-1], self.out_features), dtype=hidden.dtype)


@pytest.mark.parametrize("custom_widths", [False, True])
def test_llama4_vision_adapter_connected_widths(monkeypatch, custom_widths):
    monkeypatch.setattr(llama4, "ColumnParallelLinear", _ShapeLinear)
    options = dict(projector_input_dim=11, projector_output_dim=13) if custom_widths else {}
    config = Llama4VisionConfig(**options)
    adapter = llama4.Llama4VisionMLP2(config, rngs=None)
    # Spatial shuffle produces hidden_size / ratio**2 channels, independent
    # of the encoder's FFN intermediate width.
    channels = round(config.hidden_size / config.pixel_shuffle_ratio**2)
    adapter.activation_fn = lambda values: values
    output = adapter.forward(np.zeros((1, 1, channels), dtype=np.float32))
    assert output.shape == (1, 1, config.vision_output_dim)
    assert adapter.fc1.out_features == config.projector_input_dim
    assert adapter.fc2.out_features == config.projector_output_dim


def test_llama4_valid_checkpoint_shapes_remain_unchanged(monkeypatch):
    monkeypatch.setattr(llama4, "ColumnParallelLinear", _ShapeLinear)
    config = Llama4VisionConfig(
        hidden_size=1408,
        intermediate_size=5632,
        projector_input_dim=7680,
        projector_output_dim=7680,
        vision_output_dim=7680,
    )
    adapter = llama4.Llama4VisionMLP2(config, rngs=None)
    assert (adapter.fc1.in_features, adapter.fc1.out_features) == (5632, 7680)
    assert (adapter.fc2.in_features, adapter.fc2.out_features) == (7680, 7680)


@pytest.mark.parametrize("options", [dict(vision_output_dim=17), dict(pixel_shuffle_ratio=0)])
def test_llama4_rejects_disconnected_vision_config(options):
    with pytest.raises(ValueError):
        Llama4VisionConfig(**options)


class _Head:
    def __init__(self, weight):
        self.weight = SimpleNamespace(value=np.asarray(weight, dtype=np.float32))

    def native_forward(self, hidden, *, w=None):
        return hidden @ (self.weight.value if w is None else w)


@pytest.mark.parametrize("text_tied", [False, True])
@pytest.mark.parametrize("vision_tied", [False, True])
@pytest.mark.parametrize("cap_kind", ["clip", "double"])
def test_joint_head_full_trace_and_export_apply_postprocess_once(monkeypatch, text_tied, vision_tied, cap_kind):
    from easydel.modules._base import _base_task_module as task

    monkeypatch.setattr(task, "checkpoint_name", lambda value, name: value)
    monkeypatch.setattr(
        vision,
        "jnp",
        SimpleNamespace(
            concatenate=np.concatenate,
            dot=lambda x, w, precision=None: x @ w,
        ),
    )

    class HostVision(VisionLlamaForCausalLM):
        _auto_register = False

        def apply_logit_cap(self, logits):
            return np.clip(logits, -1, 1) if self.cap_kind == "clip" else logits * 2

        def _constrain_lm_logits(self, logits):
            return logits

    model = HostVision.__new__(HostVision)
    model.config = VisionLlamaConfig(
        vocab_size=2, vision_vocab_size=2, eos_token_id=None, tie_vision_embeddings=vision_tied
    )
    model.config.lmhead_chunksize = None
    model.dtype, model.precision, model.cap_kind = np.float32, None, cap_kind
    model.get_lm_head = lambda: _Head([[3, -3], [2, -2]])
    model.get_embedding = lambda: _Head([[1, 2], [3, 4]])
    model._tie_embeddings_enabled = lambda: text_tied
    model.model = SimpleNamespace(vision_embed_tokens=_Head([[5, 6], [7, 8]]))
    model.vision_lm_head = _Head([[9, -9], [10, -10]])
    hidden = np.asarray([[1, 2]], dtype=np.float32)
    text_weight = model.get_embedding().weight.value.T if text_tied else model.get_lm_head().weight.value
    vision_weight = model.model.vision_embed_tokens.weight.value.T if vision_tied else model.vision_lm_head.weight.value
    raw = np.concatenate((hidden @ text_weight, hidden @ vision_weight), axis=-1)
    expected = np.clip(raw, -1, 1) if cap_kind == "clip" else raw * 2
    np.testing.assert_array_equal(model.apply_lm_head(hidden), raw)
    np.testing.assert_array_equal(model.compute_lm_logits(hidden), expected)
    np.testing.assert_array_equal(model.make_lm_head_fn()(hidden), expected)
    projection = model.get_lm_head_projection()
    np.testing.assert_array_equal(projection.forward(hidden), expected)
    assert not hasattr(projection, "model")


def test_llama4_multimodal_output_preserves_router_and_headless_states():
    hidden, routers, aux = object(), object(), object()
    model = SimpleNamespace(
        config=SimpleNamespace(output_attentions=False, output_hidden_states=False),
        language_model=lambda **kwargs: SimpleNamespace(
            logits=None,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            last_hidden_state=hidden,
            router_logits=routers,
            aux_loss=aux,
        ),
    )
    output = llama4.Llama4ForConditionalGeneration.forward(model, inputs_embeds=object())
    assert output.last_hidden_state is hidden
    assert output.router_logits is routers
    assert output.aux_loss is aux
