# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.layers.attention._flexible import AttentionMechanisms, FlexibleAttentionModule, get_optimal_config
from easydel.operations import AttentionOutput
from easydel.operations.kernels.flash_attention import FlashAttn


def test_get_optimal_config_falls_back_to_vanilla_on_multihost_tpu_v3(monkeypatch):
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.process_count", lambda: 2)
    monkeypatch.setattr("easydel.layers.attention._flexible.tpu_version_check", lambda version: version == "v3")
    warnings: list[str] = []
    monkeypatch.setattr(
        "easydel.layers.attention._flexible.logger.warning_once",
        lambda message: warnings.append(message),
    )

    mechanism, dtype = get_optimal_config()

    assert mechanism == AttentionMechanisms.VANILLA
    assert dtype == jnp.bfloat16
    assert len(warnings) == 1


def test_flash_attention_falls_back_to_sdpa_on_multihost_tpu(monkeypatch):
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.process_count", lambda: 2)
    warnings: list[str] = []
    monkeypatch.setattr(
        "easydel.operations.kernels.flash_attention.logger.warning_once",
        lambda message: warnings.append(message),
    )

    expected = SimpleNamespace(attention_outputs="sdpa")

    class _SdpaStub:
        @staticmethod
        def get_unsupported_fallback_features(**kwargs):
            return ()

        def __init__(self, metadata):
            self.metadata = metadata

        def __call__(self, **kwargs):
            return expected

    monkeypatch.setattr("easydel.operations.kernels.flash_attention.ScaledDotProductAttn", _SdpaStub)

    attn = FlashAttn(metadata=object())
    result = attn.forward_native(
        query=np.zeros((1, 1, 1, 1), dtype=np.float32),
        key=np.zeros((1, 1, 1, 1), dtype=np.float32),
        value=np.zeros((1, 1, 1, 1), dtype=np.float32),
        cum_seqlens_q=np.asarray([0, 1], dtype=np.int32),
        cum_seqlens_k=np.asarray([0, 1], dtype=np.int32),
    )

    assert result is expected
    assert len(warnings) == 1


def test_flash_attention_multihost_fixed_length_falls_back_to_vanilla(monkeypatch):
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.process_count", lambda: 2)
    warnings: list[str] = []
    monkeypatch.setattr(
        "easydel.operations.kernels.flash_attention.logger.warning_once",
        lambda message: warnings.append(message),
    )

    expected = SimpleNamespace(attention_outputs="vanilla")

    class _VanillaStub:
        def __init__(self, metadata):
            self.metadata = metadata

        def __call__(self, **kwargs):
            return expected

    class _SdpaStub:
        @staticmethod
        def get_unsupported_fallback_features(**kwargs):
            return ("softmax_aux",)

        def __init__(self, metadata):
            self.metadata = metadata

        def __call__(self, **kwargs):
            raise AssertionError("fixed-length multihost fallback should not route through SDPA")

    monkeypatch.setattr("easydel.operations.kernels.flash_attention.VanillaAttn", _VanillaStub)
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.ScaledDotProductAttn", _SdpaStub)

    attn = FlashAttn(metadata=object())
    result = attn.forward_native(
        query=np.zeros((1, 1, 1, 1), dtype=np.float32),
        key=np.zeros((1, 1, 1, 1), dtype=np.float32),
        value=np.zeros((1, 1, 1, 1), dtype=np.float32),
        softmax_aux=np.zeros((1,), dtype=np.float32),
        logits_soft_cap=1.0,
        dropout_prob=0.1,
    )

    assert result is expected
    assert len(warnings) == 1


def test_flexible_attention_routes_multihost_varlen_vanilla_to_sdpa(monkeypatch):
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.process_count", lambda: 2)
    warnings: list[str] = []
    monkeypatch.setattr(
        "easydel.layers.attention._flexible.logger.warning_once",
        lambda message: warnings.append(message),
    )

    class _MeshCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _VanillaStub:
        def __init__(self):
            self.metadata = SimpleNamespace(runtime_dtype=jnp.float32)
            self.calls = 0

        def get_impl_name(self):
            return AttentionMechanisms.VANILLA

        def __call__(self, **kwargs):
            self.calls += 1
            return AttentionOutput(attention_outputs=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32))

    sdpa_calls: list[dict[str, object]] = []

    class _SdpaStub:
        @staticmethod
        def get_unsupported_fallback_features(**kwargs):
            return ()

        def __init__(self, metadata):
            self.metadata = metadata

        def __call__(self, **kwargs):
            sdpa_calls.append(kwargs)
            return AttentionOutput(attention_outputs=jnp.ones((1, 1, 1, 1), dtype=jnp.float32))

    monkeypatch.setattr("easydel.layers.attention._flexible.ScaledDotProductAttn", _SdpaStub)

    module = object.__new__(FlexibleAttentionModule)
    object.__setattr__(module, "config", SimpleNamespace(mesh=_MeshCtx()))
    object.__setattr__(module, "metadata", SimpleNamespace(runtime_dtype=jnp.float32))
    object.__setattr__(module, "impl", _VanillaStub())
    object.__setattr__(module, "impl_decode", None)
    object.__setattr__(module, "deterministic", True)
    object.__setattr__(module, "softmax_scale", 1.0)

    result = FlexibleAttentionModule.forward(
        module,
        query_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
        key_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
        value_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
        mode=None,
        cum_seqlens_q=jnp.asarray([0, 1], dtype=jnp.int32),
        cum_seqlens_k=jnp.asarray([0, 1], dtype=jnp.int32),
    )

    assert module.impl.calls == 0
    assert len(sdpa_calls) == 1
    assert np.array_equal(np.asarray(result.attention_outputs), np.ones((1, 1, 1, 1), dtype=np.float32))
    assert len(warnings) == 1


def test_flash_attention_multihost_varlen_fallback_rejects_unsupported_sdpa_features(monkeypatch):
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.process_count", lambda: 2)

    attn = FlashAttn(metadata=object())

    with pytest.raises(ValueError, match="softmax_aux, logits_soft_cap"):
        attn.forward_native(
            query=np.zeros((1, 1, 1, 1), dtype=np.float32),
            key=np.zeros((1, 1, 1, 1), dtype=np.float32),
            value=np.zeros((1, 1, 1, 1), dtype=np.float32),
            cum_seqlens_q=np.asarray([0, 1], dtype=np.int32),
            cum_seqlens_k=np.asarray([0, 1], dtype=np.int32),
            softmax_aux=np.zeros((1,), dtype=np.float32),
            logits_soft_cap=1.0,
        )


def test_flash_attention_multihost_varlen_mla_fallback_rejects_vanilla(monkeypatch):
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.operations.kernels.flash_attention.jax.process_count", lambda: 2)

    attn = FlashAttn(metadata=object())

    with pytest.raises(ValueError, match="cannot fall back to VANILLA"):
        attn.forward_native(
            query=np.zeros((1, 1, 1, 2), dtype=np.float32),
            key=np.zeros((1, 1, 1, 1), dtype=np.float32),
            value=np.zeros((1, 1, 1, 1), dtype=np.float32),
            cum_seqlens_q=np.asarray([0, 1], dtype=np.int32),
            cum_seqlens_k=np.asarray([0, 1], dtype=np.int32),
        )


def test_flexible_attention_multihost_varlen_reroute_rejects_unsupported_sdpa_features(monkeypatch):
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.process_count", lambda: 2)

    class _MeshCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _VanillaStub:
        def __init__(self):
            self.metadata = SimpleNamespace(runtime_dtype=jnp.float32)
            self.calls = 0

        def get_impl_name(self):
            return AttentionMechanisms.VANILLA

        def __call__(self, **kwargs):
            self.calls += 1
            return AttentionOutput(attention_outputs=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32))

    module = object.__new__(FlexibleAttentionModule)
    object.__setattr__(module, "config", SimpleNamespace(mesh=_MeshCtx()))
    object.__setattr__(module, "metadata", SimpleNamespace(runtime_dtype=jnp.float32))
    object.__setattr__(module, "impl", _VanillaStub())
    object.__setattr__(module, "impl_decode", None)
    object.__setattr__(module, "deterministic", True)
    object.__setattr__(module, "softmax_scale", 1.0)

    with pytest.raises(ValueError, match="softmax_aux, logits_soft_cap"):
        FlexibleAttentionModule.forward(
            module,
            query_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
            key_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
            value_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
            mode=None,
            cum_seqlens_q=jnp.asarray([0, 1], dtype=jnp.int32),
            cum_seqlens_k=jnp.asarray([0, 1], dtype=jnp.int32),
            softmax_aux=jnp.ones((1,), dtype=jnp.float32),
            logits_soft_cap=1.0,
        )

    assert module.impl.calls == 0


def test_flexible_attention_multihost_varlen_mla_requires_explicit_failure(monkeypatch):
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.default_backend", lambda: "tpu")
    monkeypatch.setattr("easydel.layers.attention._flexible.jax.process_count", lambda: 2)

    class _MeshCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _VanillaStub:
        def __init__(self):
            self.metadata = SimpleNamespace(runtime_dtype=jnp.float32)
            self.calls = 0

        def get_impl_name(self):
            return AttentionMechanisms.VANILLA

        def __call__(self, **kwargs):
            self.calls += 1
            return AttentionOutput(attention_outputs=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32))

    module = object.__new__(FlexibleAttentionModule)
    object.__setattr__(module, "config", SimpleNamespace(mesh=_MeshCtx()))
    object.__setattr__(module, "metadata", SimpleNamespace(runtime_dtype=jnp.float32))
    object.__setattr__(module, "impl", _VanillaStub())
    object.__setattr__(module, "impl_decode", None)
    object.__setattr__(module, "deterministic", True)
    object.__setattr__(module, "softmax_scale", 1.0)

    with pytest.raises(ValueError, match="head dimensions differ"):
        FlexibleAttentionModule.forward(
            module,
            query_states=jnp.ones((1, 1, 1, 2), dtype=jnp.float32),
            key_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
            value_states=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
            mode=None,
            cum_seqlens_q=jnp.asarray([0, 1], dtype=jnp.int32),
            cum_seqlens_k=jnp.asarray([0, 1], dtype=jnp.int32),
        )

    assert module.impl.calls == 0
