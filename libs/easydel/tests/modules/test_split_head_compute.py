"""Real-model numerical split-head regressions (accelerator only).

The executor harness selects its real stage-local compilation entry points on a
single resolved stage mesh. Projection export, binding, JIT and postprocessing
are not mocked. This checks numerical parity, not physical multi-stage placement.
"""

from collections import OrderedDict

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.infra.sharding import resolve_stage_mesh
from easydel.modules._base import BaseCausalLMModule
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.arctic import ArcticConfig, ArcticModel
from easydel.modules.cohere import CohereConfig, CohereForCausalLM
from easydel.modules.cohere2 import Cohere2Config, Cohere2ForCausalLM
from easydel.modules.opt import OPTConfig, OPTForCausalLM
from jax import numpy as jnp


def _stage_executor(model):
    """Use the real compiled split-head paths without compiling a backbone."""
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.model = model
    executor.mesh = model.mesh
    executor._flat_state_args = False
    executor._lm_head_cache = OrderedDict()
    executor._pipeline_lm_head_cache = OrderedDict()
    executor._lm_head_hidden_sharding_cache = {}
    executor._lm_head_graphdef = None
    executor._lm_head_state = None
    executor._lm_head_uses_tied_projection = executor._uses_tied_lm_head()
    # Route both public entry points through their stage-local branches. There
    # is intentionally no PP scheduler or emulated inter-device transport here.
    stage_mesh = resolve_stage_mesh(model.mesh, stage=(0, 1))
    executor._uses_mpmd_mesh = lambda: True
    executor._lm_head_stage_mesh = lambda: stage_mesh
    return executor


def _project(executor, model, hidden, entrypoint):
    graphdef, state = spx.export(model)
    executor.compile_lm_head(
        padded_num_reqs=hidden.shape[0],
        graphdef=graphdef,
        graphstate=state,
        graphother=spx.State(),
        inputs=None,
    )
    if entrypoint == "compile_lm_head":
        return executor.get_lm_head(padded_num_reqs=hidden.shape[0])(state, None, hidden)
    head = executor.get_pipeline_lm_head(padded_num_reqs=hidden.shape[0], part_rows=(1, 1))
    # Reverse the first part's row order to exercise gather, then concatenate
    # one row from each part; this must reproduce the original hidden rows.
    return head(state, None, (hidden[::-1], hidden[1:]), (jnp.array([1]), jnp.array([0])))


def _config_values():
    return dict(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )


@pytest.mark.parametrize("entrypoint", ["compile_lm_head", "get_pipeline_lm_head"])
@pytest.mark.parametrize("constructor_tie", [False, True])
def test_real_base_full_and_split_heads_honor_constructor_tying(entrypoint, constructor_tie):
    config = ArcticConfig(**_config_values(), num_local_experts=2, num_experts_per_tok=1, moe_layer_frequency=2)
    model = BaseCausalLMModule(
        config=config,
        base_model_class=ArcticModel,
        tie_word_embeddings=constructor_tie,
        logit_cap=0.25,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(1),
    )
    model.get_embedding().weight.value = jnp.full_like(model.get_embedding().weight.value, 0.5)
    model.get_lm_head().weight.value = jnp.full_like(model.get_lm_head().weight.value, -0.5)
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 8
    expected = np.full((2, 16), 0.25 if constructor_tie else -0.25, dtype=np.float32)
    np.testing.assert_allclose(model.compute_lm_logits(hidden), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _project(_stage_executor(model), model, hidden, entrypoint), expected, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("entrypoint", ["compile_lm_head", "get_pipeline_lm_head", "make_lm_head_fn"])
@pytest.mark.parametrize("family", ["cohere", "cohere2"])
@pytest.mark.parametrize("clip_cap", [None, 0.25])
def test_real_scaled_family_full_and_split_heads_preserve_clipping_order(entrypoint, family, clip_cap):
    config_class, model_class = {
        "cohere": (CohereConfig, CohereForCausalLM),
        "cohere2": (Cohere2Config, Cohere2ForCausalLM),
    }[family]
    model = model_class(
        config_class(**_config_values(), logit_scale=2.0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(2),
    )
    # These concrete constructors inherit the base clipping feature but do not
    # accept its optional constructor argument. Configure that existing feature
    # explicitly so the regression cannot pass with clipping silently disabled.
    model._logit_cap_feature = None if clip_cap is None else LogitCapFeature(clip_cap)
    model.get_lm_head().weight.value = jnp.full_like(model.get_lm_head().weight.value, 0.5)
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 8
    raw = np.asarray(hidden) @ np.full((8, 16), 0.5, dtype=np.float32)
    if family == "cohere":
        expected = (raw if clip_cap is None else np.clip(raw, -clip_cap, clip_cap)) * 2.0
    else:
        expected = raw * 2.0
        if clip_cap is not None:
            expected = np.clip(expected, -clip_cap, clip_cap)
    np.testing.assert_allclose(model.compute_lm_logits(hidden), expected, rtol=1e-5, atol=1e-6)
    actual = (
        model.make_lm_head_fn()(hidden)
        if entrypoint == "make_lm_head_fn"
        else _project(_stage_executor(model), model, hidden, entrypoint)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("entrypoint", ["compile_lm_head", "get_pipeline_lm_head"])
@pytest.mark.parametrize("tied", [False, True])
def test_projected_opt_split_head_infers_backbone_output_width(entrypoint, tied):
    """Both warmups infer projected width without a caller's hidden_dim override."""
    config = OPTConfig(
        vocab_size=16,
        hidden_size=16,
        word_embed_proj_dim=8,
        ffn_dim=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        tie_word_embeddings=tied,
    )
    model = OPTForCausalLM(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(3),
    )
    model.eval()
    ids = jnp.array([[1, 2, 3, 4]])
    hidden = model(input_ids=ids, apply_lm_head=False).last_hidden_state.reshape(4, 8)
    assert config.hidden_size == 16
    kernel = model.get_embedding().weight.value.T if tied else model.get_lm_head().weight.value
    expected = np.asarray(hidden) @ np.asarray(kernel)
    np.testing.assert_allclose(model(input_ids=ids).logits.reshape(4, 16), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.make_lm_head_fn()(hidden), expected, rtol=1e-5, atol=1e-6)

    executor = _stage_executor(model)
    graphdef, state = spx.export(model)
    if entrypoint == "compile_lm_head":
        # Deliberately omit hidden_dim, just like CompileOrchestrator does.
        executor.compile_lm_head(
            padded_num_reqs=4,
            graphdef=graphdef,
            graphstate=state,
            graphother=spx.State(),
            inputs=None,
        )
        actual = executor.get_lm_head(padded_num_reqs=4)(state, None, hidden)
    else:
        # Initialize the exported head directly so this regression reaches the
        # pipeline warmup even if compile_lm_head's independent warmup is broken.
        executor.refresh_lm_head_state(graphdef=graphdef, graphstate=state, graphother=spx.State())
        head = executor.get_pipeline_lm_head(padded_num_reqs=4, part_rows=(4, 4))
        actual = head(
            state,
            None,
            (hidden[::-1], hidden),
            (jnp.array([3, 2]), jnp.array([2, 3])),
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
