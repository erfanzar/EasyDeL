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

"""Llama4 text attention schedule exposed to the paged serving cache."""

import jax.numpy as jnp
import pytest
from easydel.inference.esurge.core.interface import FullAttentionSpec, create_kv_cache_specs_from_config
from easydel.infra.utils import AttnMaskType
from easydel.modules.llama4.llama4_configuration import Llama4TextConfig
from transformers import Llama4TextConfig as HFLlama4TextConfig


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"no_rope_layer_interval": 3},
        {"no_rope_layers": [0, 1, 0, 1, 0, 1]},
        {
            "no_rope_layers": [1, 0, 1, 0, 1, 0],
            "layer_types": ["full_attention", "chunked_attention"] * 3,
        },
    ],
)
def test_mask_details_match_hf_layer_schedule(config_kwargs):
    """Explicit layer types override the rotary schedule for mask and cache allocation."""
    kwargs = {"num_hidden_layers": 6, "attention_chunk_size": 16, **config_kwargs}
    config = Llama4TextConfig(**kwargs)
    hf_config = HFLlama4TextConfig(**kwargs)
    assert config.no_rope_layers == hf_config.no_rope_layers
    assert config.layer_types == hf_config.layer_types
    details = config.get_mask_details()
    assert set(details) == set(range(6))
    for idx, layer_type in enumerate(hf_config.layer_types):
        expected_chunked = layer_type == "chunked_attention"
        assert details[idx].mask_type == (AttnMaskType.CHUNK if expected_chunked else AttnMaskType.FULL)
        assert details[idx].chunks == (16 if expected_chunked else None)


@pytest.mark.cpu_ok
def test_chunked_layers_stay_in_full_attention_cache_group():
    """Chunked layers must not get a page-freeing cache group the ragged kernel cannot honour.

    The ragged paged-attention kernels take no chunk size and the runner feeds the FULL group's
    page table to every layer, so a ``ChunkedLocalAttentionSpec`` group would free pages of
    earlier chunks that the kernel still reads.
    """
    config = Llama4TextConfig(num_hidden_layers=6, attention_chunk_size=16, no_rope_layers=[0, 1, 0, 1, 0, 1])
    assert any(d.mask_type == AttnMaskType.CHUNK for d in config.get_mask_details().values())
    groups = create_kv_cache_specs_from_config(
        config=config, page_size=16, num_kv_heads=2, head_size=8, dtype=jnp.bfloat16
    )
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, FullAttentionSpec)
    assert sorted(groups[0].layer_names) == sorted(f"layer.{i}" for i in range(6))


def test_dense_cached_multi_token_masks_use_absolute_positions():
    """Cached continuation queries sit at ``indexes + 0..Q-1``, not ``0..Q-1`` (r10 S10-1).

    Six tokens are cached, two arrive, chunk size 4, cache width 16. Query 6 must see keys 0..6
    (full layers) and 4..6 (chunked layers); query 7 sees 0..7 and 4..7. Nothing past the query.
    """
    from types import SimpleNamespace

    import numpy as np
    from easydel.modules.llama4.modeling_llama4 import _llama4_layer_mask_infos
    from ejkernel.types import MaskInfo

    cache_width, chunk = 16, 4
    view = SimpleNamespace(indexes=jnp.asarray([6], dtype=jnp.int32), key=jnp.zeros((1, cache_width, 1, 1), jnp.float32))
    mask_info = MaskInfo.dynamic_init(input_ids=jnp.ones((1, 2), jnp.int32))
    full, chunked = _llama4_layer_mask_infos(mask_info, SimpleNamespace(views=[view]), chunk)
    full_mask = np.asarray(full.get_or_compute_attention_mask(dtype=jnp.bool_))[0, 0]
    chunked_mask = np.asarray(chunked.get_or_compute_attention_mask(dtype=jnp.bool_))[0, 0]
    assert full_mask.shape == (2, cache_width)

    kv = np.arange(cache_width)
    for row, pos in enumerate((6, 7)):
        np.testing.assert_array_equal(full_mask[row], kv <= pos)
        np.testing.assert_array_equal(chunked_mask[row], (kv <= pos) & (kv // chunk == pos // chunk))


def test_dense_cached_full_window_mask_rows_are_not_offset_twice():
    """A full-window mask already numbers its rows by absolute slot (r11 S10-1).

    Six tokens cached, one decoded with a width-16 attention mask: row 6 (the one the cache slices
    out) must keep keys 0..6 on full layers and 4..6 on chunked layers, not be shifted to slot 12.
    """
    from types import SimpleNamespace

    import numpy as np
    from easydel.modules.llama4.modeling_llama4 import _llama4_layer_mask_infos
    from ejkernel.types import MaskInfo

    cache_width, chunk = 16, 4
    view = SimpleNamespace(indexes=jnp.asarray([6], dtype=jnp.int32), key=jnp.zeros((1, cache_width, 1, 1), jnp.float32))
    window = jnp.asarray([[1] * 7 + [0] * 9], dtype=jnp.int32)
    mask_info = MaskInfo.dynamic_init(input_ids=jnp.ones((1, 1), jnp.int32), attention_mask=window)
    full, chunked = _llama4_layer_mask_infos(mask_info, SimpleNamespace(views=[view]), chunk, query_len=1)
    full_mask = np.asarray(full.get_or_compute_attention_mask(dtype=jnp.bool_))[0, 0]
    chunked_mask = np.asarray(chunked.get_or_compute_attention_mask(dtype=jnp.bool_))[0, 0]
    assert full_mask.shape[-2] == cache_width, full_mask.shape  # full-window rows

    kv = np.arange(cache_width)
    np.testing.assert_array_equal(full_mask[6], kv <= 6)
    np.testing.assert_array_equal(chunked_mask[6], (kv >= 4) & (kv <= 6))
