"""Cache compatibility follows declared views of the active operation, not registry spelling."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.caching import RaggedPagesCacheView, TransformerCacheView
from easydel.layers.attention import FlexibleAttentionModule, _flexible
from easydel.operations import AttentionOutput
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


class _Output:
    def astype(self, dtype):
        return self


class _Operation:
    def __init__(self, view_class):
        self.view_class = view_class
        self.metadata = SimpleNamespace(runtime_dtype=None)
        self.output = _Output()

    def get_impl_name(self):
        return "custom_attention_backend"

    def get_instance_requirements(self, mode):
        return SimpleNamespace(cache=SimpleNamespace(cache_view_class=self.view_class))

    def __call__(self, **kwargs):
        return AttentionOutput(attention_outputs=self.output, cache_view=kwargs["cache_view"])


def _module(prefill_view, decode_view=None):
    module = object.__new__(FlexibleAttentionModule)
    for name, value in {
        "config": SimpleNamespace(output_attentions=False),
        "impl": _Operation(prefill_view),
        "impl_decode": _Operation(decode_view) if decode_view else None,
        "deterministic": True,
        "softmax_scale": 1.0,
        "dropout_prob": 0.0,
    }.items():
        object.__setattr__(module, name, value)
    return module


@pytest.mark.parametrize("probability", [-0.1, 1.0, float("nan")])
def test_invalid_active_dropout_probability_is_rejected(probability):
    module = _module(TransformerCacheView)
    with pytest.raises(ValueError, match="dropout_prob must be"):
        module(None, None, None, mode=None, deterministic=False, dropout_prob=probability)


def test_active_dropout_requires_a_key():
    module = _module(TransformerCacheView)
    with pytest.raises(ValueError, match="dropout_rng must be provided"):
        module(None, None, None, mode=None, deterministic=False, dropout_prob=0.5)


def test_custom_backend_accepts_declared_ragged_cache(monkeypatch):
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    module = _module(RaggedPagesCacheView)
    cache = object.__new__(RaggedPagesCacheView)
    result = module(None, None, None, mode=None, cache_view=cache)
    assert result.cache_view is cache
    assert result.attention_outputs is module.impl.output


def test_mismatched_declared_cache_view_is_rejected():
    module = _module(TransformerCacheView)
    with pytest.raises(ValueError, match="requires TransformerCacheView"):
        module(None, None, None, mode=None, cache_view=object.__new__(RaggedPagesCacheView))


def test_decode_validates_decode_requirements_not_prefill(monkeypatch):
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    module = _module(TransformerCacheView, RaggedPagesCacheView)
    cache = object.__new__(RaggedPagesCacheView)
    result = module(None, None, None, mode=common_types.MODE_DECODE, cache_view=cache)
    assert result.attention_outputs is module.impl_decode.output
    module = _module(RaggedPagesCacheView, TransformerCacheView)
    with pytest.raises(ValueError, match="requires TransformerCacheView"):
        module(None, None, None, mode=common_types.MODE_DECODE, cache_view=cache)
