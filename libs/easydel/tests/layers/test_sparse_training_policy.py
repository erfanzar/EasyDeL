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

"""Sparse-training policy contracts use only Python metadata, not arrays."""

import pytest
from easydel.layers.indexer import AUTO_SPARSE_MIN_SEQ, use_sparse_training_attention

pytestmark = pytest.mark.cpu_ok


def _eligible_options(**overrides):
    return dict(
        dict(
            topk_indices=object(),
            cache_view=None,
            mask_info=object(),
            output_attentions=False,
            sliding_window=None,
            softmax_aux=None,
        ),
        **overrides,
    )


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("backend", ["cpu", "gpu", "tpu"])
def test_explicit_config_overrides_device_and_sequence_threshold(flag, backend):
    assert use_sparse_training_attention(**_eligible_options(), sparse_training=flag, seq_len=1, backend=backend) is flag


@pytest.mark.parametrize("flag", [None, True, False])
@pytest.mark.parametrize(
    "unsupported",
    [
        {"topk_indices": None},
        {"cache_view": object()},
        {"mask_info": None},
        {"output_attentions": True},
        {"sliding_window": 128},
        {"softmax_aux": object()},
    ],
)
def test_unsupported_features_remain_dense_even_when_sparse_is_forced(flag, unsupported):
    assert not use_sparse_training_attention(
        **_eligible_options(**unsupported), sparse_training=flag, seq_len=32768, backend="tpu"
    )


@pytest.mark.parametrize("backend", ["cpu", "gpu", "tpu"])
@pytest.mark.parametrize("seq_len", [None, 0, 8191, 8192, 16384])
def test_auto_mode_preserves_tpu_crossover(backend, seq_len):
    # The literal checks preservation of the existing external config policy.
    assert AUTO_SPARSE_MIN_SEQ == 8192
    expected = backend == "tpu" and seq_len is not None and seq_len >= 8192
    assert use_sparse_training_attention(**_eligible_options(), seq_len=seq_len, backend=backend) is expected


def test_auto_mode_can_resolve_the_active_backend_without_layer_internals(monkeypatch):
    import jax

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert use_sparse_training_attention(**_eligible_options(), seq_len=8192)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert not use_sparse_training_attention(**_eligible_options(), seq_len=8192)
