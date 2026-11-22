"""Muse-Glimmer full vs exported split LM-head parity (accelerator only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import (
    _project_stage_lm_head,
    resolve_lm_logit_postprocess,
)
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration
from easydel.modules.muse_glimmer.muse_glimmer_configuration import MuseGlimmerConfig
from jax import numpy as jnp


@pytest.mark.parametrize("multiplier", [1.0, 2.5])
@pytest.mark.parametrize("softcap", [None, 0.25])
@pytest.mark.parametrize("clip_cap", [None, 0.1])
@pytest.mark.parametrize("custom_clip", [False, True])
def test_muse_full_and_exported_split_head_match(multiplier, softcap, clip_cap, custom_clip):
    config = MuseGlimmerConfig(
        text_config={
            "vocab_size": 16,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 16,
            "output_multiplier": multiplier,
            "final_logit_softcapping": softcap,
            "attn_mechanism": "vanilla",
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 2,
            "patch_temporal": 1,
            "merge_size": 1,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "attn_mechanism": "vanilla",
        },
        out_hidden_size=16,
        projector_hidden_size=16,
        attn_mechanism="vanilla",
    )

    class CustomMuse(MuseGlimmerForConditionalGeneration):
        def apply_logit_cap(self, logits):
            return jnp.clip(logits + 0.2, -0.3, 0.3)

    model_class = CustomMuse if custom_clip else MuseGlimmerForConditionalGeneration
    model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    if clip_cap is not None:
        model._logit_cap_feature = LogitCapFeature(clip_cap)
    hidden = jnp.arange(48, dtype=jnp.float32).reshape(3, 16) / 8 - 3
    raw_logits = np.asarray(model.apply_lm_head(hidden))
    expected = np.clip(raw_logits, -clip_cap, clip_cap) if clip_cap is not None else raw_logits
    if custom_clip:
        expected = np.clip(raw_logits + 0.2, -0.3, 0.3)
    expected = expected * multiplier
    if softcap is not None:
        expected = softcap * np.tanh(expected / softcap)

    full_logits = model.compute_lm_logits(hidden)
    postprocess = resolve_lm_logit_postprocess(model)
    if multiplier == 1.0 and softcap is None and clip_cap is None and not custom_clip:
        assert postprocess is None
    else:
        assert callable(postprocess)
    graphdef, state = spx.export(model.lm_head)
    split_logits = _project_stage_lm_head(
        graphdef,
        state,
        hidden,
        uses_tied_projection=False,
        postprocess=postprocess,
        scope="muse_parity",
    )
    np.testing.assert_allclose(full_logits, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(split_logits, expected, rtol=1e-5, atol=1e-6)
    assert split_logits.dtype == full_logits.dtype
