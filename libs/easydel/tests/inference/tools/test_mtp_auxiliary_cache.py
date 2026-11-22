"""Inline MTP preserves model-declared indexer storage and per-request ownership."""

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCacheConfig, TransformerCacheView
from easydel.inference.speculative import Qwen3_5MTPDrafter, _scatter_view_row, _slice_view_row
from easydel.modules.glm5_next.glm5_next_configuration import Glm5NextTextConfig
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextForCausalLM
from easydel.modules.glm5_next.modeling_glm5_next_mtp import Glm5NextMTPHead
from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
from jax import numpy as jnp
from jax.sharding import Mesh


def _drafter(spec_head, omit_width=False, monkeypatch=None):
    config_cls = Glm5NextTextConfig if spec_head else GlmMoeDsaConfig
    model_cls = Glm5NextForCausalLM if spec_head else GlmMoeDsaForCausalLM
    config = config_cls(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0 if spec_head else 2,
        v_head_dim=4,
        index_head_dim=3,
        max_position_embeddings=16,
        mtp_num_hidden_layers=1,
    )
    # Allocation reads metadata only; no parameters or accelerator arrays are needed.
    model = model_cls.__new__(model_cls)
    model.config = config
    config.set_model_mesh(Mesh(np.array(jax.devices()[:1]), ("tp",)))
    model.dtype = jnp.bfloat16
    model.mtp = None
    if spec_head:
        head = Glm5NextMTPHead.__new__(Glm5NextMTPHead)
        head.config = config
        head.max_cache_length = 12
        model.mtp = head
        if omit_width:
            spec = head.cache_spec()
            spec.pop("auxiliary_dim")
            monkeypatch.setattr(Glm5NextMTPHead, "cache_spec", lambda self: spec)
    elif omit_width:
        original = model.create_transformer_cache_config
        monkeypatch.setattr(
            GlmMoeDsaForCausalLM,
            "create_transformer_cache_config",
            lambda self, **kwargs: original(**kwargs).replace(auxiliary_dim=None),
        )
    drafter = Qwen3_5MTPDrafter.__new__(Qwen3_5MTPDrafter)
    drafter.model = model
    drafter.uses_mtp_cache = True
    drafter._mtp_cache_max_length = 8
    return drafter


@pytest.mark.cpu_ok
@pytest.mark.parametrize("spec_head", [False, True])
@pytest.mark.parametrize("omit_width", [False, True])
def test_inline_mtp_allocates_declared_auxiliary_shape_without_numerical_execution(spec_head, omit_width, monkeypatch):
    drafter = _drafter(spec_head, omit_width, monkeypatch)

    def allocate():
        drafter.reset(batch_size=4)
        return drafter._mtp_cache

    cache = jax.eval_shape(allocate)
    assert len(cache.views) == 1
    view = cache.views[0]
    assert view.key.shape[:2] == view.value.shape[:2] == (4, 8)
    assert view.key.dtype == view.value.dtype == jnp.bfloat16
    if omit_width:
        assert view.auxiliary_state is None
        assert view.metadata.auxiliary_dim is None
    else:
        width = 7 if spec_head else 3
        assert view.auxiliary_state.shape == (4, 8, width)
        assert view.auxiliary_state.dtype == jnp.float32
        assert view.metadata.auxiliary_dim == width
    if spec_head:
        assert view.key.shape[2:] == (4, 4)
        assert view.value.shape[2:] == (4, 4)


def _view(key, auxiliary):
    return TransformerCacheView(
        key=key,
        value=key,
        indexes=jnp.zeros((4,), jnp.int32),
        starts=jnp.zeros((4,), jnp.int32),
        metadata=TransformerCacheConfig.create(
            batch_size=4,
            sequence_length=8,
            num_hidden_layers=1,
            num_heads=1,
            head_dim=1,
            auxiliary_dim=None if auxiliary is None else auxiliary.shape[-1],
        ),
        auxiliary_state=auxiliary,
        maximum_sequence_length=8,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("full_state,updated_state", [(False, False), (False, True), (True, False), (True, True)])
def test_auxiliary_slice_scatter_optional_state_abstract_shapes(full_state, updated_state):
    def update(key, auxiliary, row):
        view = _view(key, auxiliary if full_state else None)
        selected = _slice_view_row(view, row)
        assert selected.auxiliary_state is None or selected.auxiliary_state.shape == (1, 8, 3)
        selected = selected.replace(auxiliary_state=jnp.zeros((1, 8, 3), jnp.float32) if updated_state else None)
        return _scatter_view_row(view, selected, row)

    result = jax.eval_shape(
        update,
        jax.ShapeDtypeStruct((4, 8, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((4, 8, 3), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    if full_state or updated_state:
        assert result.auxiliary_state.shape == (4, 8, 3)
        assert result.auxiliary_state.dtype == jnp.float32
    else:
        assert result.auxiliary_state is None


@pytest.mark.parametrize("row", [0, 1, 2, 3])
@pytest.mark.parametrize("full_state,updated_state", [(False, False), (False, True), (True, False), (True, True)])
def test_auxiliary_slice_scatter_preserves_other_requests(row, full_state, updated_state):
    key = jnp.arange(32, dtype=jnp.float32).reshape(4, 8, 1, 1)
    auxiliary = jnp.arange(96, dtype=jnp.float32).reshape(4, 8, 3) if full_state else None
    original = _view(key, auxiliary)
    selected = jax.jit(_slice_view_row)(original, jnp.array(row, jnp.int32))
    if full_state:
        np.testing.assert_array_equal(selected.auxiliary_state, np.asarray(auxiliary)[row : row + 1])
    selected = selected.replace(auxiliary_state=jnp.full((1, 8, 3), 101, jnp.float32) if updated_state else None)
    result = jax.jit(_scatter_view_row)(original, selected, jnp.array(row, jnp.int32))
    np.testing.assert_array_equal(result.key, key)
    np.testing.assert_array_equal(result.value, key)
    if full_state or updated_state:
        expected = np.asarray(auxiliary).copy() if full_state else np.zeros((4, 8, 3), np.float32)
        if updated_state:
            expected[row] = 101
        np.testing.assert_array_equal(result.auxiliary_state, expected)
    else:
        assert result.auxiliary_state is None
