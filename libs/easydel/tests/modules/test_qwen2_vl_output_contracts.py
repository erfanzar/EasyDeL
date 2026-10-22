"""Configuration/output-protocol regressions without JAX numerical execution."""

from contextlib import nullcontext
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.modeling_outputs import BaseModelOutput, MoeModelOutput
from easydel.infra.utils import AttnMaskType
from easydel.modules.qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig
from easydel.modules.qwen2_vl import modeling_qwen2_vl as qwen2
from easydel.modules.qwen3 import modeling_qwen3 as qwen3
from easydel.modules.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel

pytestmark = pytest.mark.cpu_ok


def test_multisegment_vision_attention_abstract_trace():
    # eval_shape executes no numerical program, including parameter initialization.
    config = qwen2.Qwen2VLVisionConfig(depth=1, embed_dim=16, num_heads=2, sharding_axis_dims=(1, 1, 1, 1, 1, 1))

    def forward(hidden, boundaries, cos, sin):
        attention = qwen2.Qwen2VLVisionAttention(
            config,
            layer_idx=0,
            dim=16,
            num_heads=2,
            dtype=np.float32,
            param_dtype=np.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        return attention(hidden, boundaries, (cos, sin))

    result = jax.eval_shape(
        forward,
        jax.ShapeDtypeStruct((6, 16), np.float32),
        jax.ShapeDtypeStruct((3,), np.int32),
        jax.ShapeDtypeStruct((6, 8), np.float32),
        jax.ShapeDtypeStruct((6, 8), np.float32),
    )
    assert result.shape == (6, 16)
    assert result.dtype == np.float32


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_packed_automatic_mtp_abstract_trace(moe, vision):
    from ejkernel.types import MaskInfo

    from .test_mtp_outputs import _model

    def forward(ids, segments, positions):
        model = _model(moe, vision)
        output = model(input_ids=ids, mask_info=MaskInfo.from_segments(segments), position_ids=positions)
        return output.mtp_logits, output.mtp_loss

    tokens = jax.ShapeDtypeStruct((1, 6), np.int32)
    logits, loss = jax.eval_shape(forward, tokens, tokens, tokens)
    assert logits.shape == (1, 6, 32)
    assert loss.shape == ()
    assert logits.dtype == loss.dtype == np.float32


@pytest.mark.parametrize("schedule", [None, ["sliding_attention", "full_attention"]])
def test_nested_qwen2_vl_mask_metadata_and_cache_specs(schedule):
    from easydel.inference.esurge.core.interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
        create_kv_cache_specs_from_config,
    )

    config = Qwen2VLConfig(
        text_config=dict(
            num_hidden_layers=2,
            use_sliding_window=True,
            sliding_window=4,
            max_window_layers=1,
            layer_types=schedule,
        )
    )
    types = schedule or ["full_attention", "sliding_attention"]
    details = config.get_mask_details()
    assert [details[i].mask_type for i in range(2)] == [AttnMaskType.from_hf(t) for t in types]
    specs = create_kv_cache_specs_from_config(config, 16, 2, 4, np.float32)
    assert len(specs) == 2
    for spec, kind in zip(specs, types, strict=True):
        assert isinstance(spec.kv_cache_spec, SlidingWindowSpec if kind == "sliding_attention" else FullAttentionSpec)
    assert len(Qwen2VLConfig().get_mask_details()) == 80


@pytest.mark.parametrize("schedule", [None, ["sliding_attention", "full_attention"]])
def test_attention_window_agrees_with_public_layer_schedule(monkeypatch, schedule):
    config = Qwen2VLTextConfig(
        num_hidden_layers=2,
        use_sliding_window=True,
        sliding_window=4,
        max_window_layers=1,
        layer_types=schedule,
    )

    # Inspect the resolved structural attention setting without initializing weights.
    def init(self, **kwargs):
        self.sliding_window = kwargs["sliding_window"]

    monkeypatch.setattr(qwen2.UnifiedAttention, "__init__", init)
    for i, kind in enumerate(config.layer_types):
        attention = qwen2.Qwen2VLAttention(config, layer_idx=i, rngs=None)
        assert attention.sliding_window == (4 if kind == "sliding_attention" else None)


@pytest.mark.parametrize("explicit", [None, False, True])
def test_qwen3_config_selected_output_histories_and_explicit_overrides(monkeypatch, explicit):
    hidden = SimpleNamespace(shape=(1, 3, 4))
    attention = object()

    def block(**kwargs):
        return SimpleNamespace(hidden_states=kwargs["hidden_states"], cache_view=None, attention_weight=attention)

    class Layers:
        def __len__(self):
            return 1

        def scan(self, fn, carry, **kwargs):
            return fn(block, carry)

    model = SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=16, output_attentions=True, output_hidden_states=True, runtime_sharding_resolver=None
        ),
        layers=Layers(),
        frequencies=None,
        norm=lambda h: h,
        _layer_scan_trace=lambda *args, **kwargs: True,
        _layer_stage_context=lambda *args, **kwargs: nullcontext(),
        _layer_cache_view_at=lambda *args, **kwargs: None,
        _layer_cache_view_update=lambda cv, *args, **kwargs: cv,
        _mark_layer_stage_boundary=lambda h, *args, **kwargs: h,
    )
    monkeypatch.setattr(
        qwen3, "MaskInfo", SimpleNamespace(dynamic_init=lambda **kwargs: SimpleNamespace(q_position_ids=object()))
    )
    monkeypatch.setattr(qwen3, "apply_logical_sharding", lambda h, **kwargs: h)
    monkeypatch.setattr(qwen3, "checkpoint_name", lambda h, name: h)
    output = qwen3.Qwen3Model.forward(
        model, inputs_embeds=hidden, output_attentions=explicit, output_hidden_states=explicit
    )
    if explicit is False:
        assert output.hidden_states is output.attentions is None
    else:
        assert output.hidden_states == (hidden, hidden)
        assert output.attentions == (attention,)


@pytest.mark.parametrize("enabled", [False, True, None])
def test_multimodal_moe_preserves_decoder_router_outputs(enabled):
    router = (object(),)
    hidden = object()
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(rope_scaling=None)),
        language_model=lambda **kwargs: MoeModelOutput(
            last_hidden_state=hidden,
            router_logits=router if kwargs["output_router_logits"] is not False else None,
        ),
    )
    output = Qwen3_5MoeModel.forward(
        model, inputs_embeds=hidden, position_ids=SimpleNamespace(ndim=2), output_router_logits=enabled
    )
    assert output.last_hidden_state is hidden
    assert output.router_logits is (None if enabled is False else router)


def test_embeddings_only_qwen2_vl_can_derive_text_positions(monkeypatch):
    positions = np.array([[0, 0, 1]], dtype=np.int32)
    hidden = object()
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            text_config=SimpleNamespace(use_cache=False),
        ),
        language_model=lambda **kwargs: BaseModelOutput(last_hidden_state=kwargs["position_ids"]),
    )
    monkeypatch.setattr(
        qwen2, "MaskInfo", SimpleNamespace(dynamic_init=lambda **kwargs: SimpleNamespace(q_position_ids=positions))
    )
    monkeypatch.setattr(qwen2, "jnp", np)
    output = qwen2.Qwen2VLModel.forward(model, inputs_embeds=hidden)
    np.testing.assert_array_equal(output.last_hidden_state, np.broadcast_to(positions[None], (3, 1, 3)))
