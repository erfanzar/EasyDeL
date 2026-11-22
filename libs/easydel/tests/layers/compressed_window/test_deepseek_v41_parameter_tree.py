# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""DeepseekV41ForCausalLM keeps its checkpoint leaf paths after its blocks moved into ``easydel.layers``."""

import jax
import spectrax as spx
from easydel.layers.compressed_window import GatedPoolCompressor
from easydel.layers.engram import GatedEngram
from easydel.layers.indexer import SharedKeyIndexer
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Compressor, DeepseekV41Indexer
from jax import numpy as jnp

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def test_engram_case_parameter_tree_matches_the_harness_listing():
    config = DeepseekV41Config(**ref.case_config_kwargs("engram"))
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        _, state = spx.export(model)
    actual = {}
    for collection, path, leaf in state.items():
        array = _leaf_array(leaf)
        if collection == "parameters" and isinstance(array, jax.Array) and array.size:
            actual[path] = tuple(array.shape)
    expected = ref.easydel_param_shapes(config)
    assert any(".engram." in name for name in expected), "the engram case must exercise engram leaves"
    assert sorted(set(expected) - set(actual)) == []
    assert sorted(set(actual) - set(expected)) == []
    assert actual == expected
    # engram layers are the generic layer itself, found and ordered by the model mixin
    engrams = model.engram_modules()
    assert engrams and all(type(m) is GatedEngram for m in engrams)
    assert [m.layer_hash_index for m in engrams] == list(range(len(config.engram_layer_ids)))
    assert [m.table_rows for m in engrams] == config.engram_table_rows()


def test_v41_blocks_are_thin_bindings_of_the_generic_layers():
    assert issubclass(DeepseekV41Compressor, GatedPoolCompressor)
    assert issubclass(DeepseekV41Indexer, SharedKeyIndexer)
