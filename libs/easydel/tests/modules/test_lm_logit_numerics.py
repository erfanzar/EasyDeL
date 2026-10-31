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

"""Full, trace-safe and split-stage projection parity (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor, resolve_lm_logit_postprocess
from easydel.layers import Embed
from easydel.modules.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from easydel.modules.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from easydel.modules.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from easydel.modules.gidd.modeling_gidd import GiddForDiffusionLM
from jax import numpy as jnp

from .test_lm_logit_contract import HEADS, _head


@pytest.fixture(autouse=True)
def _full_precision_matmuls():
    # The independent NumPy reference uses fp32 products; TPU's default may
    # round fp32 operands to bf16. Match the reference, not a looser tolerance.
    with jax.default_matmul_precision("highest"):
        yield


class _Projection(spx.Module):
    def __init__(self, weight):
        self.weight = spx.Parameter(jnp.asarray(weight))

    def native_forward(self, hidden, *, w=None):
        return hidden @ (self.weight.value if w is None else w)

    forward = native_forward


class _Embedding(spx.Module):
    def __init__(self, weight):
        self.weight = spx.Parameter(jnp.asarray(weight))
        self.dtype = self.weight.value.dtype

    @staticmethod
    def promote_dtype(values, *, dtype):
        return tuple(jnp.asarray(value, dtype=dtype) for value in values)

    def attend(self, hidden):
        return hidden @ self.weight.value.T


def _numerical_head(head_class, *, tied, soft_cap, clip):
    head = _head(head_class, tied=tied, soft_cap=soft_cap, clip=clip)
    # Deliberately distinct matrices catch accidentally selecting the wrong projection.
    own_weight = np.arange(-7, 8, dtype=np.float32).reshape(3, 5) / 3.0
    embed_weight = np.arange(15, 0, -1, dtype=np.float32).reshape(5, 3) / 5.0
    projection, embedding = _Projection(own_weight), _Embedding(embed_weight)
    object.__setattr__(head, "lm_head", projection)
    object.__setattr__(head, "get_lm_head", lambda: projection)
    object.__setattr__(head, "get_embedding", lambda: embedding)
    object.__setattr__(head, "_resolve_lmhead_chunksize", lambda seq_len: None)
    object.__setattr__(head, "_constrain_lm_logits", lambda logits: logits)
    selected_weight = embed_weight.T if tied else own_weight
    return head, selected_weight


def _expected(head_class, logits, *, soft_cap, clip):
    if head_class is GiddForDiffusionLM:
        return logits
    if head_class in (Gemma2ForCausalLM, Gemma3ForCausalLM) and clip is not None:
        logits = np.clip(logits, -clip, clip)
    if soft_cap is not None:
        logits = soft_cap * np.tanh(logits / soft_cap)
    if head_class in (Gemma3ForConditionalGeneration, Gemma4ForConditionalGeneration) and clip is not None:
        logits = np.clip(logits, -clip, clip)
    return logits


@pytest.mark.parametrize("head_class", HEADS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("soft_cap", [None, 2.0], ids=["no-soft-cap", "soft-cap"])
@pytest.mark.parametrize("clip", [None, 0.7], ids=["no-base-clip", "base-clip"])
def test_full_trace_safe_and_raw_projection_hooks_match_independent_reference(head_class, tied, soft_cap, clip):
    head, weight = _numerical_head(head_class, tied=tied, soft_cap=soft_cap, clip=clip)
    hidden = np.asarray([[1.0, -2.0, 3.0], [-3.0, 2.0, 1.0], [0.2, 0.4, -0.1]], np.float32)
    expected = _expected(head_class, hidden @ weight, soft_cap=soft_cap, clip=clip)
    hidden = jnp.asarray(hidden)
    full = head.compute_lm_logits(hidden)
    trace_safe = jax.jit(head.make_lm_head_fn(vocab_shard_stage=0))(hidden)
    raw = hidden @ jnp.asarray(weight)
    postprocess = resolve_lm_logit_postprocess(head)
    split = raw if postprocess is None else jax.jit(postprocess)(raw)
    for output in (full, trace_safe, split):
        np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-5, atol=2e-6)
        assert output.dtype == full.dtype


@pytest.mark.parametrize("soft_cap", [None, 2.0], ids=["no-soft-cap", "soft-cap"])
def test_gemma4_tied_head_explicit_precision_overrides_ambient_default(soft_cap):
    head, weight = _numerical_head(Gemma4ForCausalLM, tied=True, soft_cap=soft_cap, clip=None)
    object.__setattr__(head, "precision", jax.lax.Precision.HIGHEST)
    hidden_np = np.asarray([[0.2137, -1.3219, 2.7341], [1.1337, -0.7319, -0.3341]], np.float32)
    expected = _expected(Gemma4ForCausalLM, hidden_np @ weight, soft_cap=soft_cap, clip=None)
    embedding = Embed(
        5,
        3,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        embedding_init=lambda key, shape, dtype: jnp.asarray(weight.T, dtype=dtype),
        rngs=spx.Rngs(0),
    )
    object.__setattr__(head, "get_embedding", lambda: embedding)
    with jax.default_matmul_precision("default"):
        hidden = jnp.asarray(hidden_np)
        full = head.apply_lm_head(hidden)
        project = head.make_lm_head_fn()
        traced = jax.jit(project)(hidden)
        # The factory must also remain read-only and correct in a nested trace.
        scanned = jax.jit(lambda states: jax.lax.map(project, states))(hidden)
        executor = _split_executor(head, tied=True)
        executor.compile_lm_head(padded_num_reqs=2, graphdef=None, graphstate=None, graphother=None, inputs=None)
        split = executor.get_lm_head(padded_num_reqs=2)(None, None, hidden)
        pipeline_fn = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
        pipeline = pipeline_fn(None, None, (hidden[:1], hidden[1:]), (jnp.asarray([0]), jnp.asarray([0])))
    for actual in (full, traced, scanned, split, pipeline):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-6)
        assert actual.dtype == jnp.float32


def _split_executor(head, *, tied):
    # Test the actual two split-stage executables without building a backbone or
    # distributed runtime. Only placement/graph-refresh plumbing is replaced.
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    head.config.get_text_config().hidden_size = 3
    executor.model = SimpleNamespace(config=head.config, dtype=jnp.float32, mesh=mesh)
    executor.mesh = mesh
    executor._lm_head_cache = {}
    executor._pipeline_lm_head_cache = {}
    projection_factory = getattr(head, "get_lm_head_projection", None)
    executor._lm_head_projection_is_complete = callable(projection_factory)
    if executor._lm_head_projection_is_complete:
        projection = projection_factory()
        executor._lm_head_postprocess = None
    else:
        projection = head.get_embedding() if tied else head.get_lm_head()
        executor._lm_head_postprocess = resolve_lm_logit_postprocess(head)
    executor._lm_head_graphdef, executor._lm_head_state = spx.export(projection)
    executor._lm_head_uses_tied_projection = tied
    executor._compile_mode = lambda: "mpmd"
    executor._uses_mpmd_mesh = lambda: True
    executor._graph_call_args = lambda state, other: (state, other)
    executor.refresh_lm_head_state = lambda **kwargs: None
    executor._lm_head_stage_mesh = lambda: mesh
    executor._lm_head_hidden_sharding = lambda shape: None
    executor._lm_head_replicated_sharding = lambda: None
    executor._place_lm_head_hidden = lambda hidden: hidden
    executor._place_lm_head_index = lambda indices: indices
    executor._cache_store = lambda cache, key, value: cache.__setitem__(key, value)
    executor._cache_lookup = lambda cache, key: cache[key]
    return executor


@pytest.mark.parametrize("head_class", HEADS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("soft_cap", [None, 2.0], ids=["no-soft-cap", "soft-cap"])
def test_both_split_entry_points_match_full_logits_with_base_clipping(head_class, tied, soft_cap):
    head, weight = _numerical_head(head_class, tied=tied, soft_cap=soft_cap, clip=0.7)
    hidden_np = np.asarray([[1.0, -2.0, 3.0], [-3.0, 2.0, 1.0], [0.2, 0.4, -0.1]], np.float32)
    expected = _expected(head_class, hidden_np @ weight, soft_cap=soft_cap, clip=0.7)
    hidden = jnp.asarray(hidden_np)
    executor = _split_executor(head, tied=tied)
    executor.compile_lm_head(padded_num_reqs=3, graphdef=None, graphstate=None, graphother=None, inputs=None)
    regular = executor.get_lm_head(padded_num_reqs=3)(None, None, hidden)
    pipeline_fn = executor.get_pipeline_lm_head(padded_num_reqs=3, part_rows=(2, 1))
    # Gather reversed rows from the first part, then the single row of the second.
    pipeline = pipeline_fn(None, None, (hidden[:2], hidden[2:]), (jnp.asarray([1, 0]), jnp.asarray([0])))
    full = head.compute_lm_logits(hidden)
    np.testing.assert_allclose(np.asarray(full), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(regular), np.asarray(full), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(pipeline), np.asarray(full)[[1, 0, 2]], rtol=2e-5, atol=2e-6)
