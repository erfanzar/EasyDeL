# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence classification must pool the masked last token (accelerator tests)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.mistral import MistralConfig, MistralForSequenceClassification
from easydel.modules.mixtral import MixtralConfig, MixtralForSequenceClassification
from jax import numpy as jnp


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [(MistralConfig, MistralForSequenceClassification), (MixtralConfig, MixtralForSequenceClassification)],
)
@pytest.mark.parametrize("embeddings_only", [False, True])
def test_classification_pools_attention_mask_without_pad_id(config_class, model_class, embeddings_only):
    extra = (
        dict(num_local_experts=2, num_experts_per_tok=1, initialization_of_moe=True)
        if config_class is MixtralConfig
        else {}
    )
    config = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        pad_token_id=None,
        num_labels=2,
        **extra,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    with config.mesh:
        model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        ids = jnp.asarray([[3, 4, 5, 6], [7, 8, 9, 10]])
        mask = jnp.asarray([[1, 1, 0, 0], [0, 0, 1, 1]])
        inputs = {"inputs_embeds": model.base_model.get_embedding()(ids)} if embeddings_only else {"input_ids": ids}
        outputs = model(**inputs, attention_mask=mask)
        hidden = model.base_model(**inputs, attention_mask=mask).last_hidden_state
        token_logits = np.asarray(model.score(hidden))
    # Explicit expected positions: right padding in row 0, left padding in row 1.
    expected = np.stack((token_logits[0, 1], token_logits[1, 3]))
    np.testing.assert_allclose(np.asarray(outputs.logits), expected, rtol=1e-5, atol=1e-6)
