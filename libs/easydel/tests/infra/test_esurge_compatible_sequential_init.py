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

"""Regression tests for rebinding sequential_init-built models onto a fresh graphdef.

``sequential_init`` reassigns each submodule's ``rngs`` (``module.rngs =
rng.fork(1)[0]``), so a model built that way exports an ``rng`` collection whose
stream paths differ from a freshly-constructed module's -- it grows per-submodule
streams and drops the shared top-level ``rngs.parameters`` stream. Rebinding such
a model onto a freshly-built graph definition (``esurge_compatible_model`` /
``update_module`` / ``new_graphdef`` + ``merge_module``) previously raised
``KeyError: 'parameters'`` inside ``spx.bind`` because the graphdef referenced an
RNG stream the exported state no longer carried.
"""

import jax.numpy as jnp
import spectrax as spx

import easydel as ed

_CFG = dict(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    max_position_embeddings=64,
    attn_mechanism="vanilla",
)


def _config():
    return ed.LlamaConfig(**_CFG)


def _constructor_model():
    return ed.LlamaForCausalLM(
        config=_config(),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        rngs=spx.Rngs(0),
    )


def _sequential_init_model():
    return ed.LlamaForCausalLM.sequential_init(
        config=_config(),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        rngs=spx.Rngs(0),
    )


def _param_values(model):
    return {path: (v.value if hasattr(v, "value") else v) for _c, path, v in model.graphstate.items()}


def _assert_params_preserved(before, after):
    assert set(before) == set(after)
    for key in before:
        assert jnp.array_equal(before[key], after[key]), f"parameter {key} changed during rebind"


def test_esurge_compatible_model_from_constructor_still_works():
    """The constructor path must keep working (the reconciliation is a no-op here)."""
    model = _constructor_model()
    before = _param_values(model)

    compat = model.esurge_compatible_model

    assert compat.config.attn_mechanism == "ragged_page_attention_v3"
    # A clean re-export proves the rebound module is structurally valid.
    spx.export(compat)
    _assert_params_preserved(before, _param_values(compat))


def test_esurge_compatible_model_from_sequential_init():
    """sequential_init models must convert instead of raising KeyError: 'parameters'."""
    model = _sequential_init_model()
    before = _param_values(model)

    compat = model.esurge_compatible_model

    assert compat.config.attn_mechanism == "ragged_page_attention_v3"
    _, state = spx.export(compat)
    assert "parameters" in state.collections()
    _assert_params_preserved(before, _param_values(compat))


def test_esurge_compatible_model_from_sequential_init_and_shard():
    """sequential_init + shard_model (the standard preload path) must also convert."""
    model = _sequential_init_model().shard_model()
    before = _param_values(model)

    compat = model.esurge_compatible_model

    assert compat.config.attn_mechanism == "ragged_page_attention_v3"
    _assert_params_preserved(before, _param_values(compat))


def test_update_module_from_sequential_init():
    """update_module uses the same merge_module(fresh_graphdef, ...) path and must not raise."""
    model = _sequential_init_model()
    before = _param_values(model)

    updated = model.update_module(attn_mechanism="ragged_page_attention_v3")

    assert updated.config.attn_mechanism == "ragged_page_attention_v3"
    _assert_params_preserved(before, _param_values(updated))


def test_merge_module_does_not_drop_present_rng_streams():
    """The reconciliation must never overwrite RNG streams the state already carries."""
    model = _constructor_model()
    graphdef = model.new_graphdef(attn_mechanism="ragged_page_attention_v3", recursive_update=True)
    graphother = model.graphother
    rebound = model.merge_module(graphdef, model.graphstate, graphother)

    _, rebound_state = spx.export(rebound)
    _, source_state = spx.export(model)
    # Existing rng streams are preserved verbatim (overlay lets the real state win).
    for collection, path, value in source_state.items():
        if collection != "rng":
            continue
        rebound_value = rebound_state.get("rng", path, None)
        if rebound_value is None:
            continue
        src = value.value if hasattr(value, "value") else value
        dst = rebound_value.value if hasattr(rebound_value, "value") else rebound_value
        assert jnp.array_equal(src, dst)
