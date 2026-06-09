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

import numpy as np
from spectrax import common_types as ct

import easydel.operations.kernels.ragged_page_attention as ragged_page_mod


class _FakePartitionManager:
    def resolve(self, *, axes, **_kwargs):
        return tuple(axes)


def test_ragged_page_v3_replicates_kv_specs_when_cache_kv_axis_is_unsharded(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_ragged_page_attention_v3(*args, **kwargs):
        captured["in_specs"] = kwargs.get("in_specs")
        captured["out_specs"] = kwargs.get("out_specs")
        return np.zeros_like(args[0]), args[3]

    monkeypatch.setattr(ragged_page_mod, "ragged_page_attention_v3", _fake_ragged_page_attention_v3)
    monkeypatch.setattr(ragged_page_mod.jax, "default_backend", lambda: "cpu")

    metadata = SimpleNamespace(
        partition_manager=_FakePartitionManager(),
        mesh=SimpleNamespace(shape={"tp": 4}),
        get_operation_config=lambda *_args, **_kwargs: None,
    )
    op = ragged_page_mod.RaggedPageAttnV3(metadata)

    cache_view = SimpleNamespace(
        metadata=SimpleNamespace(data_parallel_size=1, kv_head_shards=1),
        kv_pages=np.zeros((8, 16, 1, 2, 256), dtype=np.float32),
    )
    cache_metadata = SimpleNamespace(
        context_lens=np.asarray([4], dtype=np.int32),
        pages_tables=np.zeros((1, 1), dtype=np.int32),
        query_start_loc=np.asarray([0, 4], dtype=np.int32),
        request_distribution=np.asarray([0, 0, 1], dtype=np.int32),
    )

    op.forward_v3(
        query=np.zeros((4, 8, 256), dtype=np.float32),
        key=np.zeros((4, 1, 256), dtype=np.float32),
        value=np.zeros((4, 1, 256), dtype=np.float32),
        cache_view=cache_view,
        cache_metadata=cache_metadata,
    )

    in_specs = captured["in_specs"]
    assert in_specs is not None
    assert in_specs[1][1] == ct.EMPTY
    assert in_specs[2][1] == ct.EMPTY
    assert in_specs[3][2] == ct.EMPTY
