from contextlib import nullcontext
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from easydel.caching import KDACacheView, RaggedPagesCacheView
from easydel.infra.mixins.generation import (
    EasyGenerationMixin,
    _is_cache_view_subclass,
    _is_standard_ragged_view_class,
    _resolve_recurrent_cache_dtype,
)

pytestmark = pytest.mark.cpu_ok


def test_recurrent_cache_dtype_defaults_to_kv_dtype():
    config = SimpleNamespace()
    assert _resolve_recurrent_cache_dtype(config, jnp.bfloat16) == jnp.dtype(jnp.bfloat16)


def test_recurrent_cache_dtype_defaults_to_mamba_state_dtype():
    config = SimpleNamespace(mamba_ssm_dtype="float32")
    assert _resolve_recurrent_cache_dtype(config, jnp.bfloat16) == jnp.dtype(jnp.float32)


def test_recurrent_cache_dtype_can_differ_from_fp8_attention_kv():
    config = SimpleNamespace(recurrent_cache_dtype="bfloat16")
    assert _resolve_recurrent_cache_dtype(config, jnp.float8_e4m3fn) == jnp.dtype(jnp.bfloat16)


def test_recurrent_cache_dtype_rejects_unknown_name():
    config = SimpleNamespace(recurrent_cache_dtype="not_a_dtype")
    try:
        _resolve_recurrent_cache_dtype(config, jnp.bfloat16)
    except ValueError as exc:
        assert "not_a_dtype" in str(exc)
    else:
        raise AssertionError("expected unsupported dtype to raise")


def test_custom_ragged_page_view_is_recognized():
    class CustomRaggedPagesCacheView(RaggedPagesCacheView):
        pass

    assert _is_cache_view_subclass(CustomRaggedPagesCacheView, RaggedPagesCacheView)
    assert _is_standard_ragged_view_class(CustomRaggedPagesCacheView)
    assert not _is_cache_view_subclass(object(), RaggedPagesCacheView)


def test_kda_cache_uses_independent_recurrent_dtype(monkeypatch):
    text_config = SimpleNamespace(
        num_hidden_layers=1,
        mesh=nullcontext(),
        recurrent_cache_dtype="float32",
        runtime_sharding_resolver=None,
        get_kv_shared_layer_mapping=lambda: {},
        get_mask_details=lambda: None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: text_config),
        mesh=nullcontext(),
        init_operations_cache_config=lambda **kwargs: [object()],
        get_operations_cache_view=lambda: {0: KDACacheView},
        get_operations_cache_view_by_slot=lambda: {},
        _pipeline_stage_count=lambda: 1,
        _layer_stage_context=lambda idx, total_layers: nullcontext(),
        num_draft_cache_layers=lambda: 0,
    )
    sentinel = object()
    seen = {}

    def fake_init(cls, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(KDACacheView, "init", classmethod(fake_init))
    cache = EasyGenerationMixin.init_operations_cache(
        model,
        batch_size=1,
        max_length=8,
        dtype=jnp.bfloat16,
    )
    assert cache.views == [sentinel]
    assert seen["dtype"] == jnp.dtype(jnp.float32)
