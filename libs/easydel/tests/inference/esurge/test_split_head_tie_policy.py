"""Host-only split-head orchestration parity for constructor/config tying.

The real projection policy and both serving entry points run with NumPy-only
projection modules and identity compiler/binding adapters. No JAX arrays run;
accelerator compilation and physical placement remain separate validation.
"""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.executors import model_executor as executor_module
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.infra.base_module import EasyDeLBaseModule

pytestmark = pytest.mark.cpu_ok


class _Projection:
    def __init__(self, weight):
        self.weight = SimpleNamespace(value=weight)

    def __call__(self, hidden, *, w=None):
        return hidden @ (self.weight.value if w is None else w)

    def native_forward(self, hidden):
        return self(hidden)

    def attend(self, hidden):
        return hidden @ self.weight.value.T


class _HostModel:
    _tie_embeddings_enabled = EasyDeLBaseModule._tie_embeddings_enabled
    _lm_head_kernel_spec = EasyDeLBaseModule._lm_head_kernel_spec
    _tied_lm_head_weight = EasyDeLBaseModule._tied_lm_head_weight
    apply_lm_head = EasyDeLBaseModule.apply_lm_head

    def __init__(self, flags, constructor_tie):
        self.config = SimpleNamespace(**flags, get_text_config=lambda: SimpleNamespace(hidden_size=2))
        self._tie_embeddings_feature = SimpleNamespace(tie=constructor_tie)
        self.mesh = nullcontext()
        self.dtype = np.float32
        self.embedding = _Projection(np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        self.head = _Projection(np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32))

    def get_embedding(self):
        return self.embedding

    def get_lm_head(self):
        return self.head

    def get_lm_logit_postprocess(self):
        return None

    def compute_lm_logits(self, hidden):
        return self.apply_lm_head(hidden)


@pytest.mark.parametrize(
    "flags, constructor_tie, expected_tied",
    [
        ({"tie_word_embeddings": False}, True, True),
        ({"tie_word_embeddings": False, "share_input_output_layers": True}, False, False),
        ({"tie_word_embeddings": False, "use_lm_head": True}, False, False),
        ({"use_lm_head": True, "share_input_output_layers": False}, False, True),
        ({"tie_word_embeddings": True}, False, True),
    ],
)
def test_full_and_both_split_heads_share_live_constructor_aware_policy(
    monkeypatch, flags, constructor_tie, expected_tied
):
    model = _HostModel(flags, constructor_tie)
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    # A graph refresh may replace constructor features independently of the old
    # model object kept for shape metadata. Export and placement must use live policy.
    executor.model = _HostModel({"tie_word_embeddings": not expected_tied}, False)
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor._lm_head_cache = OrderedDict()
    executor._pipeline_lm_head_cache = OrderedDict(
        {(2, (1, 1), "pipeline_lm_head", "mpmd"): lambda *args: np.full((2, 3), -999)}
    )
    executor._backbone_cache = OrderedDict(stale_backbone=object())
    executor._pipeline_model_step_fn_cache = OrderedDict(stale_fused=object())
    executor._cache = OrderedDict(stale_combined=object())
    executor._pipeline_kv_carry_map_cache = {}
    executor._lm_head_hidden_sharding_cache = {(2, 2): object()}
    executor._pipeline_runtime = None
    executor._lm_head_uses_tied_projection = executor._uses_tied_lm_head()
    executor._embedding_stage_mesh = lambda: embedding_stage
    executor._final_stage_mesh = lambda: final_stage
    executor._lm_head_hidden_sharding = lambda shape: None
    executor._lm_head_replicated_sharding = lambda: None
    executor._stage_local_lm_head_state = lambda state: state
    executor._place_lm_head_hidden = lambda hidden: hidden
    executor._place_lm_head_index = lambda index: index
    executor._graph_call_args = lambda state, other: (state, other)
    embedding_stage, final_stage = nullcontext(), nullcontext()
    monkeypatch.setattr(executor_module, "jnp", np)
    monkeypatch.setattr(executor_module.jax, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(executor_module.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(executor_module.spx, "bind", lambda graphdef, state: graphdef)
    monkeypatch.setattr(executor_module.spx, "export", lambda projection: (projection, object()))
    monkeypatch.setattr(executor_module.spx, "extract_sharding_structure", lambda *args, **kwargs: None)
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.compile_lm_head(padded_num_reqs=2, graphdef=model, graphstate=state, graphother=None, inputs=None)
    hidden = np.asarray([[2, 3], [4, 5]], dtype=np.float32)
    full = model.compute_lm_logits(hidden)
    expected = hidden @ (model.embedding.weight.value.T if expected_tied else model.head.weight.value)
    np.testing.assert_array_equal(full, expected)
    split = executor.get_lm_head(padded_num_reqs=2)(None, None, hidden)
    np.testing.assert_array_equal(split, expected)
    from_parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    np.testing.assert_array_equal(
        from_parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0]))), expected
    )
    assert executor._lm_head_stage_mesh() is (embedding_stage if expected_tied else final_stage)


@pytest.mark.parametrize("answer", [True, False])
def test_model_resolver_is_authoritative_over_all_config_flags(answer):
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=not answer, share_input_output_layers=not answer),
        _tie_embeddings_enabled=lambda: answer,
    )
    assert executor._uses_tied_lm_head() is answer


def test_legacy_model_without_resolver_retains_nested_config_support():
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(tie_word_embeddings=True))
    )
    assert executor._uses_tied_lm_head()
