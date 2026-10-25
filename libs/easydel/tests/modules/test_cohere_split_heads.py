"""NumPy-only ordering regressions through real model and split-head control flow.

Only initialization, device placement, graph binding and compilation boundaries
are substituted. These tests do not execute JAX arrays or validate hardware.
"""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.executors import model_executor as executor_module
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.infra.factory import registry
from easydel.modules._base import _base_task_module as task_module
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
from easydel.modules.cohere2.modeling_cohere2 import Cohere2ForCausalLM

pytestmark = pytest.mark.cpu_ok


class _Projection:
    def __init__(self, weight):
        self.weight = SimpleNamespace(value=weight)

    def __call__(self, hidden, *, w=None):
        return hidden @ (self.weight.value if w is None else w)

    native_forward = __call__

    def attend(self, hidden):
        return hidden @ self.weight.value.T


class _HostInitialization:
    """Replace parameter allocation, keeping all production logit methods intact."""

    # Subclasses are created at collection time, before monkeypatch fixtures can
    # restore registry state. Opt out of the production auto-registration hook.
    _auto_register = False

    def __init__(self, tied, cap, scale):
        self.config = SimpleNamespace(
            tie_word_embeddings=tied, lmhead_chunksize=None, get_text_config=lambda: SimpleNamespace(hidden_size=2)
        )
        self.dtype = np.float32
        self.logit_scale = scale
        self._logit_cap_feature = (
            None if cap is None else SimpleNamespace(cap_value=cap, apply=lambda logits: np.clip(logits, -cap, cap))
        )
        self.embedding = _Projection(np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        self.head = _Projection(np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32))

    @property
    def mesh(self):
        return nullcontext()

    def get_embedding(self):
        return self.embedding

    def get_lm_head(self):
        return self.head

    def _constrain_lm_logits(self, logits):
        return logits


class _HostCohere(_HostInitialization, CohereForCausalLM):
    pass


class _HostCohere2(_HostInitialization, Cohere2ForCausalLM):
    pass


@pytest.mark.parametrize("model_class", [CohereForCausalLM, Cohere2ForCausalLM])
def test_host_subclasses_preserve_real_family_factory_registration(model_class):
    registration = registry.get_module_registration(model_class._task_type, model_class._model_type)
    assert registration.module is model_class


@pytest.mark.parametrize("family", ["cohere", "cohere2"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("cap", [None, 2.0])
def test_full_trace_safe_and_both_split_heads_preserve_family_order(monkeypatch, family, tied, cap):
    # Replace all device/compiler boundaries before constructing the metadata-only model.
    monkeypatch.setattr(task_module, "jnp", np)
    monkeypatch.setattr(task_module, "checkpoint_name", lambda value, name: value)
    monkeypatch.setattr(executor_module, "jnp", np)
    monkeypatch.setattr(executor_module.jax, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(executor_module.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(executor_module.spx, "bind", lambda graphdef, state: graphdef)
    monkeypatch.setattr(executor_module.spx, "export", lambda projection: (projection, object()))
    monkeypatch.setattr(executor_module.spx, "extract_sharding_structure", lambda *args, **kwargs: None)
    model = (_HostCohere if family == "cohere" else _HostCohere2)(tied, cap, 3.0)
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.model = model
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor._lm_head_cache = OrderedDict()
    executor._pipeline_lm_head_cache = OrderedDict()
    executor._lm_head_uses_tied_projection = tied
    executor._embedding_stage_mesh = nullcontext
    executor._final_stage_mesh = nullcontext
    executor._lm_head_hidden_sharding = lambda shape: None
    executor._lm_head_replicated_sharding = lambda: None
    executor._stage_local_lm_head_state = lambda state: state
    executor._place_lm_head_hidden = lambda hidden: hidden
    executor._place_lm_head_index = lambda index: index
    executor._graph_call_args = lambda state, other: (state, other)
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.compile_lm_head(padded_num_reqs=2, graphdef=model, graphstate=state, graphother=None, inputs=None)

    # Saturated and unsaturated logits distinguish ordering, missing scaling and
    # duplicate clipping/scaling; positive-only examples cannot catch all three.
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    raw = hidden @ (model.embedding.weight.value.T if tied else model.head.weight.value)
    if family == "cohere":
        expected = (np.clip(raw, -cap, cap) if cap is not None else raw) * 3.0
    else:
        expected = np.clip(raw * 3.0, -cap, cap) if cap is not None else raw * 3.0
    np.testing.assert_allclose(model.compute_lm_logits(hidden), expected)
    np.testing.assert_allclose(model.make_lm_head_fn(vocab_shard_stage=0)(hidden), expected)
    np.testing.assert_allclose(model.get_lm_logit_postprocess()(raw), expected)
    np.testing.assert_allclose(executor.get_lm_head(padded_num_reqs=2)(None, None, hidden), expected)
    from_parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    np.testing.assert_allclose(
        from_parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0]))), expected
    )

    # A split-head hook must not capture the live model/parameter tree: later
    # scalar mutation must not change the captured compile-time semantics.
    postprocess = model.get_lm_logit_postprocess()
    model.logit_scale = 99.0
    if cap is not None:
        model._logit_cap_feature.cap_value = 99.0
    np.testing.assert_allclose(postprocess(raw), expected)
    assert all(cell.cell_contents is not model for cell in postprocess.__closure__)


@pytest.mark.parametrize("cap", [None, 2.0])
def test_cohere2_optional_scale_keeps_clipping(monkeypatch, cap):
    monkeypatch.setattr(task_module, "jnp", np)
    monkeypatch.setattr(task_module, "checkpoint_name", lambda value, name: value)
    model = _HostCohere2(False, cap, None)
    hidden = np.asarray([[2, -3]], dtype=np.float32)
    raw = hidden @ model.head.weight.value
    expected = np.clip(raw, -cap, cap) if cap is not None else raw
    postprocess = model.get_lm_logit_postprocess()
    np.testing.assert_array_equal(raw if postprocess is None else postprocess(raw), expected)
    np.testing.assert_array_equal(model.compute_lm_logits(hidden), expected)
    np.testing.assert_array_equal(model.make_lm_head_fn()(hidden), expected)
