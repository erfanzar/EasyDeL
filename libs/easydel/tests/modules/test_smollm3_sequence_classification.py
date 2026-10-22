"""SmolLM3 classification output regression; accelerator execution required."""

import numpy as np
import spectrax as spx
from easydel.modules.smollm3.modeling_smollm3 import SmolLM3ForSequenceClassification
from easydel.modules.smollm3.smollm3_configuration import SmolLM3Config
from jax import numpy as jnp


def test_smollm3_sequence_classification_outputs_bias_free_label_logits():
    config = SmolLM3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        pad_token_id=0,
        num_labels=3,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = SmolLM3ForSequenceClassification(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    input_ids = jnp.array([[1, 2, 3], [4, 5, 0]], dtype=jnp.int32)
    outputs = model(input_ids=input_ids, attention_mask=input_ids != 0)

    assert outputs.logits.shape == (2, 3)
    assert outputs.logits.dtype == jnp.float32
    assert np.isfinite(np.asarray(outputs.logits)).all()
    assert model.score.bias is None
    assert model.score.weight.value.shape == (16, 3)
