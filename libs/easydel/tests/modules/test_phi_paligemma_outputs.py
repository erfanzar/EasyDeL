"""Public module output and prefix-visibility regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.paligemma.modeling_paligemma import PaliGemmaModel
from easydel.modules.paligemma.paligemma_configuration import PaliGemmaConfig
from easydel.modules.phi.modeling_phi import PhiForCausalLM
from easydel.modules.phi.phi_configuration import PhiConfig
from easydel.modules.phi3.modeling_phi3 import Phi3ForCausalLM
from easydel.modules.phi3.phi3_configuration import Phi3Config
from easydel.modules.qwen2.modeling_qwen import Qwen2ForCausalLM
from easydel.modules.qwen2.qwen_configuration import Qwen2Config
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def set_single_device_mesh(config):
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    config.set_model_mesh(mesh)
    return config


@pytest.mark.parametrize(
    "config_class,model_class",
    [(PhiConfig, PhiForCausalLM), (Phi3Config, Phi3ForCausalLM), (Qwen2Config, Qwen2ForCausalLM)],
)
def test_task_models_resolve_config_output_defaults(config_class, model_class):
    config = set_single_device_mesh(
        config_class(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            attn_mechanism="vanilla",
            output_attentions=True,
            output_hidden_states=True,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
    )
    model = model_class(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    outputs = model(input_ids=ids)
    assert len(outputs.hidden_states) == 2
    assert len(outputs.attentions) == 1
    assert outputs.hidden_states[-1].shape == (1, 3, 16)
    assert outputs.attentions[0].shape == (1, 2, 3, 3)
    suppressed = model(input_ids=ids, output_hidden_states=False, output_attentions=False)
    assert suppressed.hidden_states is None
    assert suppressed.attentions is None


def paligemma_model():
    common = dict(attn_mechanism="vanilla", sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    config = PaliGemmaConfig(
        text_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=32,
            max_position_embeddings=16,
            **common,
        ),
        vision_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
            vision_use_head=False,
            **common,
        ),
        projection_dim=16,
        hidden_size=16,
        vocab_size=32,
        image_token_index=31,
        **common,
    )
    for subconfig in (config, config.text_config, config.vision_config):
        set_single_device_mesh(subconfig)
    model = PaliGemmaModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(1)
    )
    model.eval()
    return model


@pytest.mark.parametrize("visibility_kind", ["pairwise", "directed", "segments", "padding"])
def test_paligemma_prefix_relaxes_only_causal_visibility(visibility_kind):
    model = paligemma_model()
    if visibility_kind == "pairwise":
        visibility = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], bool)
        mask = MaskInfo.from_attention_mask(jnp.asarray(visibility[None, None]))
    elif visibility_kind == "directed":
        visibility = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], bool)
        mask = MaskInfo.from_attention_mask(jnp.asarray(visibility[None, None]))
    elif visibility_kind == "segments":
        visibility = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], bool)
        mask = MaskInfo.from_segments(jnp.array([[0, 0, 1, 1]], jnp.int32))
    else:
        visibility = np.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]], bool)
        mask = MaskInfo.from_attention_mask(jnp.array([[1, 1, 0, 1]], jnp.bool_))
    # All genuine prefix pairs are bidirectional. The last token is a suffix.
    types = np.array([0, 0, 0, 1])
    allowed = visibility & (np.tri(4, dtype=bool) | ((types[:, None] == 0) & (types[None, :] == 0)))
    outputs = model(
        inputs_embeds=jnp.arange(64, dtype=jnp.float32).reshape(1, 4, 16) / 64,
        mask_info=mask,
        position_ids=jnp.arange(1, 5)[None],
        token_type_ids=jnp.asarray(types[None]),
        output_attentions=True,
    )
    weights = np.asarray(outputs.attentions[0])[0]
    # An all-masked padded query is backend-dependent; genuine query rows must
    # preserve every forbidden edge and assign mass to every allowed edge.
    for query in np.flatnonzero(allowed.any(-1)):
        np.testing.assert_array_equal(weights[:, query, ~allowed[query]], 0)
        assert np.all(weights[:, query, allowed[query]] > 0)
