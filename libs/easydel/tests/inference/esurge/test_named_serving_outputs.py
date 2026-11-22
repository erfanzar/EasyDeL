"""Serving builders consume the always-named model output and its updated cache.

EasyDeL forwards always return their named output type, so serving calls the
model directly and never passes HF-style ``return_dict``/``use_cache`` switches.
These are host-only orchestration tests. Compilation, model binding and numerical
operations are replaced by ordinary Python/NumPy equivalents; no JAX arrays run.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors import model_executor as executor_module
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor

pytestmark = pytest.mark.cpu_ok


class _NamedModel(spx.Module):
    def __init__(self):
        self.mesh = nullcontext()
        self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=2))

    def forward(self, input_ids, past_key_values, **kwargs):
        # Serving must not inject deprecated output switches into the forward.
        assert "return_dict" not in kwargs
        assert "use_cache" not in kwargs
        hidden = input_ids[..., None].repeat(2, axis=-1)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=past_key_values + "/updated")

    def compute_lm_logits(self, hidden):
        return hidden * 3


@pytest.mark.parametrize("builder", ["backbone", "pipeline", "spmd"])
def test_every_serving_builder_consumes_named_outputs_and_updated_cache(monkeypatch, builder):
    model = _NamedModel()
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.model = model
    executor.mesh = SimpleNamespace(is_mpmd=False)
    executor.metadata = SimpleNamespace(get_max_num_seqs=lambda: 2)
    executor.max_num_reqs = 2
    executor._empty_sharding = None
    executor.mpmd_scheduler = None
    executor._metadata_sharding_template = lambda: None
    executor._lm_head_hidden_sharding = lambda shape: None
    executor._lm_head_replicated_sharding = lambda: None
    executor._graph_arg_sharding = lambda state, **kwargs: None
    executor._flat_state_args = False
    executor._graphstate_treedef = executor._graphother_treedef = None
    executor._graphstate_packer = executor._graphother_packer = None
    executor.main_hidden_enabled = False
    executor.full_hidden_state_max_tokens = 0
    executor._build_cache_metadata = lambda metadata, rows: None
    executor._prepare_model_call = lambda *args, **kwargs: (
        {"input_ids": np.asarray([[4, 5]])},
        None,
        {},
    )
    executor._flatten_model_hidden = lambda hidden: hidden.squeeze(0)
    monkeypatch.setattr(executor_module.jax, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(executor_module.spx, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(executor_module.spx, "bind", lambda *args: model)
    monkeypatch.setattr(executor_module.spx, "extract_sharding_structure", lambda *args, **kwargs: None)
    graphstate = SimpleNamespace(merge=lambda other, **kwargs: None)
    metadata = SimpleNamespace(logits_indices=np.asarray([1, 0]))
    templates = dict(kv_pages_template=None, graphstate_template=None, graphother_template=None)
    if builder == "backbone":
        fn = executor._build_backbone_fn(**templates)
        output = fn(None, graphstate, None, "cache", metadata)
        np.testing.assert_array_equal(output.hidden_states, [[4, 4], [5, 5]])
    elif builder == "pipeline":
        fn = executor._build_pipeline_model_step_fn(**templates, padded_num_reqs=2, graphdef=None)
        output = fn(graphstate, None, "cache", metadata)
        np.testing.assert_array_equal(output.logits, [[15, 15], [12, 12]])
    else:
        fn = executor._build_model_step_fn(**templates)
        output = fn(None, graphstate, None, "cache", metadata, 2)
        np.testing.assert_array_equal(output.logits, [[15, 15], [12, 12]])
    assert output.kv_pages == "cache/updated"
