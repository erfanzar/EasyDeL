from types import SimpleNamespace

import jax.numpy as jnp
from easydel.caching import RaggedPagesCacheView
from easydel.infra.mixins.generation import (
    _is_cache_view_subclass,
    _is_standard_ragged_view_class,
    _resolve_recurrent_cache_dtype,
)


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
