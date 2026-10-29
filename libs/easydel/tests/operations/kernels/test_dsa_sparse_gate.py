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

"""Selection logic of ``use_sparse_training_attention`` (forced on / off / auto by length)."""

from types import SimpleNamespace

import pytest
from easydel.operations.kernels import dsa_sparse_attention as dsa

pytestmark = pytest.mark.cpu_ok


def _layer(flag, sinks=None):
    return SimpleNamespace(config=SimpleNamespace(dsa_sparse_training=flag), _softmax_aux=lambda: sinks)


def _use(layer, seq_len, **over):
    kw = dict(topk_indices=object(), cache_view=None, mask_info=object(), output_attentions=False, sliding_window=None)
    kw.update(over)
    return dsa.use_sparse_training_attention(layer, seq_len=seq_len, **kw)


def test_forced_flags_override_length():
    assert _use(_layer(True), 16) is True
    assert _use(_layer(False), 1 << 20) is False


@pytest.mark.parametrize("backend", ["tpu", "cpu"])
def test_auto_uses_threshold_on_tpu_only(monkeypatch, backend):
    monkeypatch.setattr(dsa.jax, "default_backend", lambda: backend)
    short = _use(_layer(None), dsa.AUTO_SPARSE_MIN_SEQ - 1)
    long = _use(_layer(None), dsa.AUTO_SPARSE_MIN_SEQ)
    assert short is False
    assert long is (backend == "tpu")


@pytest.mark.parametrize(
    "override",
    [
        {"topk_indices": None},
        {"cache_view": object()},
        {"mask_info": None},
        {"output_attentions": True},
        {"sliding_window": 128},
    ],
)
def test_ineligible_never_sparse_even_when_forced(override):
    assert _use(_layer(True), 1 << 16, **override) is False


def test_sinks_disable_sparse():
    assert _use(_layer(True, sinks=object()), 1 << 16) is False


def _metadata(query, key, mesh_shape=None):
    mesh = SimpleNamespace(shape=mesh_shape or {"dp": 1, "fsdp": 2, "tp": 2, "sp": 1})
    rules = SimpleNamespace(query=query, key=key)
    return SimpleNamespace(mesh=mesh, get_shardings=lambda *a, **k: rules)


def test_bthd_specs_accepts_matching_batch_and_head_axes():
    spec = dsa._bthd_specs(_metadata(("fsdp", None, "tp", None), ("fsdp", None, "tp", None)))
    assert spec is not None
    q_spec, kv_spec, batch_axes = spec
    assert tuple(q_spec) == ("fsdp", None, "tp", None) and tuple(kv_spec) == tuple(q_spec)
    assert batch_axes == "fsdp"


def test_bthd_specs_rejects_mismatched_head_axes():
    # q sharded over heads while kv is replicated would misalign the GQA head map inside shard_map.
    assert dsa._bthd_specs(_metadata(("fsdp", None, "tp", None), ("fsdp", None, None, None))) is None


def test_bthd_specs_rejects_sequence_sharding():
    meta = _metadata(("fsdp", "sp", "tp", None), ("fsdp", "sp", "tp", None), {"fsdp": 2, "tp": 2, "sp": 2})
    assert dsa._bthd_specs(meta) is None
