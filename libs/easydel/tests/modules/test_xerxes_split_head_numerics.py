"""Hardware-only full/split Xerxes LM-head parity against independent NumPy math."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import (
    _project_stage_lm_head,
    resolve_lm_logit_postprocess,
)
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.xerxes.modeling_xerxes import XerxesForCausalLM
from easydel.modules.xerxes.xerxes_configuration import XerxesConfig
from jax import numpy as jnp


@pytest.mark.parametrize("xe_kvnorm, xe_moe", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("clip_cap", [None, 12.0])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_full_and_split_heads_match_ordered_reference(xe_kvnorm, xe_moe, tied, clip_cap, chunk_size):
    config = XerxesConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        xe_kvnorm=xe_kvnorm,
        xe_moe=xe_moe,
        tie_word_embeddings=tied,
        lmhead_chunksize=chunk_size,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    # Tied Embed.attend follows ambient precision, while the linear LM head
    # accepts an explicit precision. Pin both paths for the NumPy reference;
    # default TPU f32 dots round operands to bf16 before accumulation.
    with config.mesh, jax.default_matmul_precision("highest"):
        model = XerxesForCausalLM(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )
        model.eval()
        # Exercise the inherited clipping feature as well as Xerxes' own cap.
        model._logit_cap_feature = None if clip_cap is None else LogitCapFeature(clip_cap)
        hidden = jnp.asarray(np.linspace(-2000, 2000, 40, dtype=np.float32).reshape(1, 5, 8))
        projection = model.get_embedding() if tied else model.get_lm_head()
        graphdef, state = spx.export(projection)
        split = _project_stage_lm_head(
            graphdef,
            state,
            hidden.reshape(-1, 8),
            uses_tied_projection=tied,
            postprocess=resolve_lm_logit_postprocess(model),
            scope="xerxes_parity",
        ).reshape(1, 5, 16)
        full = model.compute_lm_logits(hidden)
        weight = np.asarray(projection.weight.value)
        expected = np.asarray(hidden) @ (weight.T if tied else weight)
        if clip_cap is not None:
            expected = np.clip(expected, -clip_cap, clip_cap)
        if not (xe_kvnorm and not xe_moe):
            expected = np.tanh(expected / 30.0) * 30.0
        np.testing.assert_allclose(np.asarray(full), expected, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(np.asarray(split), expected, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(np.asarray(split), np.asarray(full), rtol=2e-4, atol=2e-4)
        assert split.dtype == full.dtype == jnp.float32
        if xe_kvnorm and not xe_moe and clip_cap is None:
            assert model.get_lm_logit_postprocess() is None
        else:
            assert callable(model.get_lm_logit_postprocess())
