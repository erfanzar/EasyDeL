"""Array-free dispatch contracts: a host reference backend exposes stochastic public outputs.

These tests exercise routing and training/RNG policy, not JAX kernel numerics. Real-backend
numerical coverage lives in test_flexible_dropout.py and must run on an accelerator separately.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.attention import FlexibleAttentionModule, _flexible
from easydel.operations import AttentionOutput
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


class _HostAttention:
    """Minimal independent uniform-attention backend with optional inverted dropout."""

    metadata = SimpleNamespace(runtime_dtype=np.float32)

    def __init__(self, name, supports_dropout):
        self.name = name
        self.supports_dropout = supports_dropout

    def get_impl_name(self):
        return self.name

    def __call__(
        self,
        *,
        value,
        dropout_prob,
        dropout_rng,
        deterministic,
        cache_view,
        mask_info=None,
        bias=None,
        init_bias=None,
        normalize_output=True,
        **kwargs,
    ):
        if bias is None and mask_info is None and init_bias is not None:
            bias = init_bias()
        logits = np.zeros((1, 1, 8, 8), dtype=np.float32) if bias is None else np.broadcast_to(bias, (1, 1, 8, 8))
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        if mask_info is not None:
            weights = np.where(mask_info.attention_mask, weights, 0)
        if normalize_output:
            denominator = weights.sum(-1, keepdims=True)
            weights /= np.where(denominator, denominator, 1)
        if self.supports_dropout and not deterministic and dropout_prob:
            keep = np.random.default_rng(dropout_rng).random(weights.shape) >= dropout_prob
            weights = weights * keep / (1 - dropout_prob)
        output = np.einsum("bhqk,bkhd->bqhd", weights, value)
        return AttentionOutput(attention_outputs=output, attention_weights=weights, cache_view=cache_view)


class _HostRngs:
    def __init__(self):
        self.next = 10

    def key(self, name):
        assert name == "dropout"
        self.next += 1
        return self.next


def _module(monkeypatch, name, *, rngs=None, training=True):
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    monkeypatch.setattr(_flexible.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        _flexible.OperationRegistry,
        "create",
        lambda impl_name, **kwargs: _HostAttention(impl_name, supports_dropout=impl_name == "vanilla"),
    )
    module = object.__new__(FlexibleAttentionModule)
    for attr, value in {
        "config": SimpleNamespace(output_attentions=True),
        "metadata": _HostAttention.metadata,
        "impl": _HostAttention(name, supports_dropout=name == "vanilla"),
        "impl_decode": None,
        "_requires_cache": False,
        "_spx_training": training,
        "deterministic": None,
        "dropout_prob": 0.5,
        "softmax_scale": 1.0,
        "rngs": rngs,
    }.items():
        object.__setattr__(module, attr, value)
    return module


def _call(module, *, mode=None, **kwargs):
    q = np.zeros((1, 8, 1, 2), dtype=np.float32)
    v = np.arange(16, dtype=np.float32).reshape(1, 8, 1, 2)
    return module(q, q, v, mode=mode, causal=False, **kwargs)


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "cudnn", "cuda_flash_attn2", "flash_attn2", "blocksparse"])
def test_active_dropout_survives_backend_selection(monkeypatch, backend):
    module = _module(monkeypatch, backend)
    baseline = _call(module)
    dropped = _call(module, deterministic=False, dropout_rng=42)
    repeated = _call(module, deterministic=False, dropout_rng=42)
    np.testing.assert_array_equal(baseline.attention_outputs, np.broadcast_to([7, 8], (1, 8, 1, 2)))
    np.testing.assert_array_equal(dropped.attention_outputs, repeated.attention_outputs)
    assert not np.array_equal(dropped.attention_outputs, baseline.attention_outputs)
    assert set(np.unique(dropped.attention_weights)) == {0.0, 0.25}


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "flash_attn2"])
def test_unnormalized_dropout_preserves_stabilized_numerator(monkeypatch, backend):
    module = _module(monkeypatch, backend)
    dropped = _call(module, deterministic=False, dropout_rng=42, normalize_output=False)
    repeated = _call(module, deterministic=False, dropout_rng=42, normalize_output=False)
    assert set(np.unique(dropped.attention_weights)) == {0.0, 2.0}
    values = np.arange(16, dtype=np.float32).reshape(1, 8, 1, 2)
    np.testing.assert_array_equal(
        dropped.attention_outputs, np.einsum("bhqk,bkhd->bqhd", dropped.attention_weights, values)
    )
    np.testing.assert_array_equal(dropped.attention_outputs, repeated.attention_outputs)


def test_training_mode_uses_advancing_constructor_rng_without_explicit_determinism(monkeypatch):
    module = _module(monkeypatch, "sdpa", rngs=_HostRngs())
    first = _call(module, mode=common_types.MODE_TRAIN)
    second = _call(module, mode=common_types.MODE_TRAIN)
    assert not np.array_equal(first.attention_outputs, second.attention_outputs)
    assert (first.attention_weights == 0).any()


@pytest.mark.parametrize("mode", [None, common_types.MODE_PREFILL, common_types.MODE_DECODE])
def test_nontraining_modes_do_not_consume_dropout_rng(monkeypatch, mode):
    module = _module(monkeypatch, "sdpa", rngs=_HostRngs())
    cache = object.__new__(_flexible.TransformerCacheView) if mode == common_types.MODE_DECODE else None
    result = _call(module, mode=mode, cache_view=cache)
    np.testing.assert_array_equal(result.attention_outputs, np.broadcast_to([7, 8], (1, 8, 1, 2)))
    assert module.rngs.next == 10


def test_eval_and_explicit_determinism_override_training_without_consuming_rng(monkeypatch):
    module = _module(monkeypatch, "sdpa", rngs=_HostRngs(), training=False)
    evaluated = _call(module, mode=common_types.MODE_TRAIN)
    object.__setattr__(module, "_spx_training", True)
    forced = _call(module, mode=common_types.MODE_TRAIN, deterministic=True)
    np.testing.assert_array_equal(evaluated.attention_outputs, forced.attention_outputs)
    assert module.rngs.next == 10
    forced_random = _call(module, mode=common_types.MODE_PREFILL, deterministic=False)
    assert not np.array_equal(forced_random.attention_outputs, forced.attention_outputs)


def test_explicit_key_takes_precedence_and_zero_override_never_consumes_rng(monkeypatch):
    module = _module(monkeypatch, "sdpa", rngs=_HostRngs())
    first = _call(module, mode=common_types.MODE_TRAIN, dropout_rng=42)
    second = _call(module, mode=common_types.MODE_TRAIN, dropout_rng=42)
    np.testing.assert_array_equal(first.attention_outputs, second.attention_outputs)
    zero = _call(module, mode=common_types.MODE_TRAIN, dropout_prob=0)
    np.testing.assert_array_equal(zero.attention_outputs, np.broadcast_to([7, 8], (1, 8, 1, 2)))
    assert module.rngs.next == 10


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "flash_attn2"])
@pytest.mark.parametrize("lengths", [np.array([4]), np.array([0, 4])])
def test_varlen_dropout_preserves_visibility_in_simulated_multihost_dispatch(monkeypatch, backend, lengths):
    module = _module(monkeypatch, backend)
    # NumPy substitutes ONLY this module's array namespace. No JAX arrays/backend are initialized;
    # hardware names below are routing inputs, not actual devices. Real JAX parity is a separate test.
    monkeypatch.setattr(_flexible, "jnp", np)
    monkeypatch.setattr(_flexible.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(_flexible.jax, "process_count", lambda: 2)
    result = _call(module, deterministic=False, dropout_rng=42, cum_seqlens_q=lengths, cum_seqlens_k=lengths)
    weights = result.attention_weights
    np.testing.assert_array_equal(weights[..., 4:, :], 0)
    np.testing.assert_array_equal(weights[..., :4, 4:], 0)
    assert set(np.unique(weights[..., :4, :4])) == {0.0, 0.5}
    v = np.arange(16, dtype=np.float32).reshape(1, 8, 1, 2)
    np.testing.assert_array_equal(result.attention_outputs, np.einsum("bhqk,bkhd->bqhd", weights, v))


def test_packed_dropout_keeps_sequences_separate_and_respects_causal_window(monkeypatch):
    module = _module(monkeypatch, "flash_attn2")
    monkeypatch.setattr(_flexible, "jnp", np)
    q = np.zeros((1, 8, 1, 2), dtype=np.float32)
    v = np.arange(16, dtype=np.float32).reshape(1, 8, 1, 2)
    result = module(
        q,
        q,
        v,
        mode=None,
        deterministic=False,
        dropout_rng=42,
        cum_seqlens_q=np.array([0, 4, 8]),
        cum_seqlens_k=np.array([0, 4, 8]),
        causal=True,
        sliding_window=2,
    )
    weights = result.attention_weights[0, 0]
    np.testing.assert_array_equal(weights[:4, 4:], 0)
    np.testing.assert_array_equal(weights[4:, :4], 0)
    np.testing.assert_array_equal(np.triu(weights, 1), 0)
    np.testing.assert_array_equal(np.tril(weights, -2), 0)
    assert (np.diag(weights) > 0).any()


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "flash_attn2"])
def test_varlen_dropout_preserves_lazy_additive_bias(monkeypatch, backend):
    module = _module(monkeypatch, backend)
    monkeypatch.setattr(_flexible, "jnp", np)
    bias = np.broadcast_to(np.arange(8, dtype=np.float32), (1, 1, 8, 8))
    lengths = np.array([0, 4, 8])
    calls = []

    def init_bias():
        calls.append(True)
        return bias

    arguments = dict(deterministic=False, dropout_rng=42, cum_seqlens_q=lengths, cum_seqlens_k=lengths)
    lazy = _call(module, init_bias=init_bias, **arguments)
    explicit = _call(module, bias=bias, **arguments)
    uniform = _call(module, **arguments)
    np.testing.assert_array_equal(lazy.attention_outputs, explicit.attention_outputs)
    np.testing.assert_array_equal(lazy.attention_weights, explicit.attention_weights)
    assert not np.array_equal(lazy.attention_outputs, uniform.attention_outputs)
    assert calls == [True]
    # An explicit bias takes precedence; the unused callback must remain lazy.
    _call(module, bias=bias, init_bias=init_bias, **arguments)
    assert calls == [True]


def test_dense_cache_with_baked_chunk_mask_retains_existing_backend_support(monkeypatch):
    from easydel.infra.utils import AttnMaskDetail, AttnMaskType

    module = _module(monkeypatch, "vanilla")
    view = object.__new__(_flexible.TransformerCacheView)
    view.masking_details = AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=4)
    positions = np.arange(8)
    visibility = (positions[:, None] // 4 == positions[None, :] // 4)[None, None]
    mask_info = _flexible.MaskInfo(_attention_mask=visibility)
    result = _call(module, mask_info=mask_info, cache_view=view)
    expected = np.broadcast_to([3, 4], (1, 8, 1, 2)).copy()
    expected[:, 4:] = [11, 12]
    np.testing.assert_array_equal(result.attention_outputs, expected)
