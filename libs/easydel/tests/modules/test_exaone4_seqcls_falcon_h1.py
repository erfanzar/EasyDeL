# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Exaone4 sequence classification and Falcon-H1 regression tests (computation; run on TPU/GPU only)."""

import easydel as ed
import jax
import numpy as np
import spectrax as spx
from jax import numpy as jnp


def test_exaone4_sequence_classification_constructs_and_runs():
    config = ed.Exaone4Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=None,
        num_labels=3,
    )
    model = ed.Exaone4ForSequenceClassification(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    out = model(input_ids=jnp.full((2, 4), 3, dtype=jnp.int32))
    assert out.logits.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(jax.device_get(out.logits))))
