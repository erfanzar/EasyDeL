"""Real family logit methods through both split entry points, using NumPy.

Only parameter allocation, JAX/compiler/binding and physical-placement boundaries
are replaced. This is host orchestration evidence, not compiled hardware parity.
"""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners.executors import model_executor as execution
from easydel.infra.factory import registry
from easydel.modules._base import _base_task_module as task
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
from easydel.modules.cohere2.modeling_cohere2 import Cohere2ForCausalLM
from easydel.modules.gemma2 import modeling_gemma2
from easydel.modules.gemma4 import modeling_gemma4
from easydel.modules.llama import modeling_vision_llama
from easydel.modules.llama.llama_configuration import VisionLlamaConfig

pytestmark = pytest.mark.cpu_ok


class _Projection:
    # Mirrors the real Embed surface used by tied heads (dtype + promote_dtype).
    dtype = np.float32

    def __init__(self, weight):
        self.weight = SimpleNamespace(value=np.asarray(weight, dtype=np.float32))

    def promote_dtype(self, arrays, dtype=None):
        return tuple(np.asarray(array, dtype=dtype or self.dtype) for array in arrays)

    def __call__(self, hidden, *, w=None):
        return hidden @ (self.weight.value if w is None else w)

    native_forward = __call__

    def attend(self, hidden, precision=None):
        del precision  # NumPy host double: precision has no effect on float32 matmul.
        return hidden @ self.weight.value.T


class _HostInitialization:
    # These host-only subclasses must never replace real model registrations
    # merely because pytest imports this module during collection.
    _auto_register = False

    def __init__(self, tied, cap, softcap):
        self.config = SimpleNamespace(
            tie_word_embeddings=tied,
            final_logit_softcapping=softcap,
            lmhead_chunksize=None,
            get_text_config=lambda: SimpleNamespace(hidden_size=2),
        )
        self.dtype = np.float32
        # Every real EasyDeL module carries a matmul precision; model-owned split
        # projections (e.g. Gemma4.get_lm_head_projection) read it.
        self.precision = None
        self.logit_scale = 3.0
        self._logit_cap_feature = SimpleNamespace(cap_value=cap, apply=lambda logits: np.clip(logits, -cap, cap))
        self.embedding = _Projection([[1, 2], [3, 4], [5, 6]])
        self.head = _Projection([[7, 8, 9], [10, 11, 12]])

    @property
    def mesh(self):
        return nullcontext()

    def get_embedding(self):
        return self.embedding

    def get_lm_head(self):
        return self.head

    def _constrain_lm_logits(self, logits):
        return logits


class _Cohere(_HostInitialization, CohereForCausalLM):
    pass


class _Cohere2(_HostInitialization, Cohere2ForCausalLM):
    pass


class _Gemma2(_HostInitialization, modeling_gemma2.Gemma2ForCausalLM):
    pass


class _Gemma4(_HostInitialization, modeling_gemma4.Gemma4ForCausalLM):
    pass


class _HostJnp:
    """NumPy stand-in for ``jnp`` that accepts JAX-only keyword arguments.

    Gemma4's tied head forwards its matmul ``precision`` to ``jnp.dot``; NumPy's
    float32 host matmul has no precision setting, so the keyword is dropped.
    """

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def dot(left, right, precision=None):
        del precision
        return np.dot(left, right)


def _executor(monkeypatch, model, tied):
    monkeypatch.setattr(task, "jnp", np)
    monkeypatch.setattr(task, "checkpoint_name", lambda value, name: value)
    monkeypatch.setattr(modeling_gemma2, "jnp", np)
    monkeypatch.setattr(modeling_gemma4, "jnp", _HostJnp())
    monkeypatch.setattr(execution, "jnp", np)
    monkeypatch.setattr(execution.jax.nn, "tanh", np.tanh)
    monkeypatch.setattr(execution.jax, "jit", lambda **kwargs: lambda fn: fn)
    monkeypatch.setattr(execution.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(execution.spx, "bind", lambda graphdef, state: graphdef)
    monkeypatch.setattr(execution.spx, "export", lambda projection: (projection, object()))
    monkeypatch.setattr(execution.spx, "extract_sharding_structure", lambda *args, **kwargs: None)
    executor = execution.ModelStepExecutor.__new__(execution.ModelStepExecutor)
    executor.model = model
    executor.mesh = SimpleNamespace(is_mpmd=True)
    executor._lm_head_cache = OrderedDict()
    executor._pipeline_lm_head_cache = OrderedDict()
    executor._backbone_cache = OrderedDict()
    executor._pipeline_model_step_fn_cache = OrderedDict()
    executor._cache = OrderedDict()
    executor._pipeline_kv_carry_map_cache = {}
    executor._lm_head_hidden_sharding_cache = {}
    executor._pipeline_runtime = None
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
    return executor


@pytest.mark.parametrize("tied", [False, True])
def test_sharded_stage_head_warmup_and_retained_calls_use_positional_jit_arguments(monkeypatch, tied):
    model = _Cohere(tied, 2.0, None)
    executor = _executor(monkeypatch, model, tied)

    def jit(**options):
        def decorate(fn):
            def call(*args, **kwargs):
                # Match the real JAX calling contract, without compiling or
                # allocating arrays in this host-only regression.
                if "in_shardings" in options and kwargs:
                    raise ValueError("pjit does not support kwargs when in_shardings is specified")
                return fn(*args, **kwargs)

            return call

        return decorate

    monkeypatch.setattr(execution.jax, "jit", jit)
    monkeypatch.setattr(execution.jax, "device_put", lambda value, sharding: value)
    executor._lm_head_hidden_sharding = lambda shape: object()
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.compile_lm_head(padded_num_reqs=3, graphdef=model, graphstate=state, graphother=None, inputs=None)
    hidden = np.asarray([[0.01, -0.02], [2, -3], [-1, 0.5]], dtype=np.float32)
    single = executor.get_lm_head(padded_num_reqs=3)
    parts = executor.get_pipeline_lm_head(padded_num_reqs=3, part_rows=(1, 2))
    expected = model.compute_lm_logits(hidden)
    np.testing.assert_allclose(single(None, None, hidden), expected, rtol=1e-6)
    np.testing.assert_allclose(
        parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0, 1]))), expected, rtol=1e-6
    )
    model.logit_scale = 7.0
    executor.refresh_lm_head_state(graphdef=model, graphstate=state, graphother=None)
    expected = model.compute_lm_logits(hidden)
    np.testing.assert_allclose(single(None, None, hidden), expected, rtol=1e-6)
    np.testing.assert_allclose(
        parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0, 1]))), expected, rtol=1e-6
    )


def _both_split(executor, hidden, expected):
    np.testing.assert_allclose(executor.get_lm_head(padded_num_reqs=2)(None, None, hidden), expected, rtol=1e-6)
    parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    np.testing.assert_allclose(
        parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0]))), expected, rtol=1e-6
    )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("logical_metadata", [False, True])
def test_split_head_warmups_infer_projected_backbone_width(monkeypatch, tied, logical_metadata):
    model = _Cohere(tied, 2.0, None)
    # A projected backbone emits width 2 despite transformer width 4. The
    # selected projection, not the transformer's config, owns the input width.
    model.config.get_text_config = lambda: SimpleNamespace(hidden_size=4)
    if logical_metadata:
        if tied:
            model.embedding.features = 2
        else:
            model.head.in_features = 2
    executor = _executor(monkeypatch, model, tied)
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    raw = hidden @ (model.embedding.weight.value.T if tied else model.head.weight.value)
    expected = np.clip(raw, -2.0, 2.0) * 3.0
    _both_split(executor, hidden, expected)


@pytest.mark.parametrize("tied", [False, True])
def test_explicit_split_head_width_override_remains_authoritative(monkeypatch, tied):
    model = _Cohere(tied, 2.0, None)
    executor = _executor(monkeypatch, model, tied)
    # Deliberately conflicting metadata demonstrates that the existing public
    # hidden_dim override still controls the warmup, rather than being ignored.
    if tied:
        model.embedding.features = 7
    else:
        model.head.in_features = 7
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.compile_lm_head(
        padded_num_reqs=3, graphdef=model, graphstate=state, graphother=None, inputs=None, hidden_dim=2
    )
    hidden = np.asarray([[0.01, -0.02], [2, -3], [-1, 0.5]], dtype=np.float32)
    raw = hidden @ (model.embedding.weight.value.T if tied else model.head.weight.value)
    np.testing.assert_allclose(
        executor.get_lm_head(padded_num_reqs=3)(None, None, hidden), np.clip(raw, -2.0, 2.0) * 3.0, rtol=1e-6
    )


@pytest.mark.parametrize("family", ["cohere", "cohere2", "gemma2", "gemma4"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("softcap", [None, 1.5])
def test_real_full_logits_and_both_split_heads_preserve_order(monkeypatch, family, tied, softcap):
    model = {"cohere": _Cohere, "cohere2": _Cohere2, "gemma2": _Gemma2, "gemma4": _Gemma4}[family](tied, 2.0, softcap)
    executor = _executor(monkeypatch, model, tied)
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    raw = hidden @ (model.embedding.weight.value.T if tied else model.head.weight.value)
    if family == "cohere":
        expected = np.clip(raw, -2.0, 2.0) * 3.0
    elif family == "cohere2":
        expected = np.clip(raw * 3.0, -2.0, 2.0)
    else:
        expected = np.clip(raw, -2.0, 2.0) if family == "gemma2" else raw
        if softcap is not None:
            expected = softcap * np.tanh(expected / softcap)
    np.testing.assert_allclose(model.compute_lm_logits(hidden), expected, rtol=1e-6)
    _both_split(executor, hidden, expected)


@pytest.mark.parametrize("family", ["cohere", "cohere2", "gemma2", "gemma4"])
@pytest.mark.parametrize("tied", [False, True])
def test_successful_refresh_updates_retained_single_and_parts_heads(monkeypatch, family, tied):
    cls = {"cohere": _Cohere, "cohere2": _Cohere2, "gemma2": _Gemma2, "gemma4": _Gemma4}[family]
    model = cls(tied, 2.0, 1.5)
    executor = _executor(monkeypatch, model, tied)
    single = executor.get_lm_head(padded_num_reqs=2)
    parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    previous = single(None, None, hidden).copy()
    model.logit_scale = 7.0
    model._logit_cap_feature = SimpleNamespace(cap_value=0.5, apply=lambda logits: np.clip(logits, -0.5, 0.5))
    model.config.final_logit_softcapping = 0.75
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.refresh_lm_head_state(graphdef=model, graphstate=state, graphother=None)
    expected = model.compute_lm_logits(hidden)
    assert not np.allclose(previous, expected)
    np.testing.assert_allclose(single(None, None, hidden), expected, rtol=1e-6)
    np.testing.assert_allclose(
        parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0]))), expected, rtol=1e-6
    )
    _both_split(executor, hidden, expected)


class _HostVisionLlama(_HostInitialization, modeling_vision_llama.VisionLlamaForCausalLM):
    # Exercise the missing-capability boundary even after the owning family
    # eventually adds its complete factory. Actual family projection methods
    # below are not replaced with a synthetic concatenation implementation.
    get_lm_head_projection = None

    def __init__(self, tied, vision_tied):
        super().__init__(tied, 2.0, None)
        self.config = VisionLlamaConfig(
            vocab_size=3,
            vision_vocab_size=2,
            hidden_size=2,
            intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            tie_word_embeddings=tied,
            tie_vision_embeddings=vision_tied,
        )
        self._logit_cap_feature = None
        self.precision = None
        self.model = SimpleNamespace(vision_embed_tokens=_Projection([[1, 3], [2, 4]]))
        self.vision_lm_head = _Projection([[5, 7], [6, 8]])


def _host_vision_math(monkeypatch):
    monkeypatch.setattr(
        modeling_vision_llama,
        "jnp",
        SimpleNamespace(concatenate=np.concatenate, dot=lambda left, right, precision=None: left @ right),
    )


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("vision_tied", [False, True])
def test_actual_joint_vocabulary_cannot_silently_export_only_text(monkeypatch, tied, vision_tied):
    _host_vision_math(monkeypatch)
    model = _HostVisionLlama(tied, vision_tied)
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    # Parameter allocation/compiler boundaries are host adapters. Exercise the
    # actual family make_lm_head_fn/apply_lm_head/compute_lm_logits contracts.
    monkeypatch.setattr(task, "checkpoint_name", lambda value, name: value)
    text_weight = model.embedding.weight.value.T if tied else model.head.weight.value
    vision_weight = model.model.vision_embed_tokens.weight.value.T if vision_tied else model.vision_lm_head.weight.value
    expected = np.concatenate((hidden @ text_weight, hidden @ vision_weight), axis=-1)
    np.testing.assert_array_equal(model.compute_lm_logits(hidden), expected)
    assert expected.shape[-1] == model.config.total_vocab_size == 5
    with pytest.raises(ValueError, match=r"5 joint-vocabulary logits.*split projection exposes 3"):
        _executor(monkeypatch, model, tied)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("vision_tied", [False, True])
def test_incomplete_joint_head_refresh_invalidates_both_split_entry_points(monkeypatch, tied, vision_tied):
    _host_vision_math(monkeypatch)
    executor = _executor(monkeypatch, _Composite(tied, 2.0, None), tied)
    old_single = executor.get_lm_head(padded_num_reqs=2)
    old_parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    with pytest.raises(ValueError, match="get_lm_head_projection"):
        executor.refresh_lm_head_state(graphdef=_HostVisionLlama(tied, vision_tied), graphstate=state, graphother=None)
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    # Neither a new lookup nor a retained executable may serve the old head
    # after the live model failed the complete-vocabulary export contract.
    with pytest.raises(KeyError):
        executor.get_lm_head(padded_num_reqs=2)
    with pytest.raises(ValueError, match="state was not initialized"):
        executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    with pytest.raises(ValueError, match="state was not initialized"):
        old_single(None, None, hidden)
    with pytest.raises(ValueError, match="state was not initialized"):
        old_parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0])))


@pytest.mark.parametrize(
    "model_class",
    [
        CohereForCausalLM,
        Cohere2ForCausalLM,
        modeling_gemma2.Gemma2ForCausalLM,
        modeling_gemma4.Gemma4ForCausalLM,
        modeling_vision_llama.VisionLlamaForCausalLM,
    ],
)
def test_host_fixtures_preserve_real_family_factory_registration(model_class):
    registration = registry.get_module_registration(model_class._task_type, model_class._model_type)
    assert registration.module is model_class


class _CompleteProjection:
    def __init__(self, model):
        self.text = model.embedding if model.config.tie_word_embeddings else model.head
        self.vision = model.vision
        self.tied = model.config.tie_word_embeddings
        self.scale = model.logit_scale
        self.cap = model._logit_cap_feature.cap_value

    def __call__(self, hidden):
        text = self.text.attend(hidden) if self.tied else self.text(hidden)
        # Nontrivial ordered transforms on one vocabulary, a second independent
        # parameterized projection on the other: neither is scalar recovery.
        text = np.clip(text * self.scale, -self.cap, self.cap)
        return np.concatenate((text, self.vision(hidden)), axis=-1)


class _Composite(_HostInitialization):
    def __init__(self, tied, cap, softcap):
        super().__init__(tied, cap, softcap)
        self.config.hidden_size = 2
        self.config.total_vocab_size = 5
        self.config.get_text_config = lambda: self.config
        self.vision = _Projection([[1, 3], [2, 4]])

    def get_lm_head_projection(self):
        return _CompleteProjection(self)

    def get_lm_logit_postprocess(self):
        # Must not be called in complete-projection mode or applied twice.
        raise AssertionError("The complete projection owns postprocessing")

    def compute_lm_logits(self, hidden):
        return self.get_lm_head_projection()(hidden)


@pytest.mark.parametrize("tied", [False, True])
def test_model_owned_complete_projection_includes_auxiliary_vocabulary(monkeypatch, tied):
    model = _Composite(tied, 2.0, None)
    executor = _executor(monkeypatch, model, tied)
    hidden = np.asarray([[0.01, -0.02], [2, -3]], dtype=np.float32)
    expected = model.compute_lm_logits(hidden)
    assert expected.shape == (2, 5)
    _both_split(executor, hidden, expected)
    single = executor.get_lm_head(padded_num_reqs=2)
    parts = executor.get_pipeline_lm_head(padded_num_reqs=2, part_rows=(1, 1))
    model.logit_scale = 7.0
    model._logit_cap_feature.cap_value = 0.5
    state = SimpleNamespace(merge=lambda other, **kwargs: None)
    executor.refresh_lm_head_state(graphdef=model, graphstate=state, graphother=None)
    refreshed = model.compute_lm_logits(hidden)
    assert not np.allclose(expected, refreshed)
    np.testing.assert_allclose(single(None, None, hidden), refreshed, rtol=1e-6)
    np.testing.assert_allclose(
        parts(None, None, (hidden[:1], hidden[1:]), (np.asarray([0]), np.asarray([0]))), refreshed, rtol=1e-6
    )
    _both_split(executor, hidden, refreshed)
