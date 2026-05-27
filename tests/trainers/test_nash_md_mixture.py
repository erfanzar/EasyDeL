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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from easydel.trainers.nash_md_trainer.nash_md_trainer import GeometricMixtureLogitsProcessor


class _FakeRefModel:
    def __init__(self, logits):
        self.logits = logits
        self.last_attention_mask = None

    def __call__(self, *, input_ids, attention_mask):
        del input_ids
        self.last_attention_mask = attention_mask
        return SimpleNamespace(logits=self.logits)


def test_nash_md_geometric_mixture_processor_mixes_policy_and_ref_logits():
    policy_scores = jnp.array([[1.0, 2.0, -1.0]], dtype=jnp.float32)
    ref_logits = jnp.array(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    processor = GeometricMixtureLogitsProcessor(_FakeRefModel(ref_logits), mixture_coef=0.25, pad_token_id=0)
    input_ids = jnp.array([[0, 5, 6, 0]], dtype=jnp.int32)

    actual = processor(input_ids, policy_scores, cur_len=jnp.asarray(3))

    expected_logits = 0.25 * ref_logits[:, 2, :] + 0.75 * policy_scores
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(jax.nn.log_softmax(expected_logits, axis=-1)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(processor.ref_model.last_attention_mask),
        np.asarray(jnp.array([[0, 1, 1, 0]], dtype=jnp.int32)),
    )
