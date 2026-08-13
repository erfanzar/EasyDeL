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

"""FusedMlpOp adapter: registry wiring, requirements, and kernel parity."""

import inspect
import re

import jax
import numpy as np
import pytest
from easydel.infra import EasyDeLBaseConfig
from easydel.operations import FusedMlpOp, OperationMetadata, OperationRegistry
from ejkernel.modules import fused_mlp as ejkernel_fused_mlp
from ejkernel.modules.operations.configs import FusedMlpConfig
from jax import numpy as jnp


def _metadata(**overrides) -> OperationMetadata:
    kwargs = dict(
        runtime_dtype=jnp.bfloat16,
        runtime_softmax_dtype=jnp.float32,
        base_config=EasyDeLBaseConfig(),
    )
    kwargs.update(overrides)
    return OperationMetadata(**kwargs)


def _dense_inputs(seed=0, batch=2, seq=4, k=64, i=128):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(batch, seq, k)), jnp.bfloat16)
    w_gate = jnp.asarray(rng.normal(size=(k, i)), jnp.bfloat16)
    w_up = jnp.asarray(rng.normal(size=(k, i)), jnp.bfloat16)
    w_down = jnp.asarray(rng.normal(size=(i, k)), jnp.bfloat16)
    return x, w_gate, w_up, w_down


def _quantize_channelwise(rng, k, n):
    w = rng.normal(size=(k, n)).astype(np.float32)
    scale = np.abs(w).max(axis=0, keepdims=True) / 127.0
    codes = np.clip(np.round(w / scale), -127, 127)
    return jnp.asarray(codes, jnp.int8), jnp.asarray(scale, jnp.float32)


class TestRegistryWiring:
    def test_registered_and_creatable_by_name(self):
        op = OperationRegistry.create("fused_mlp", _metadata())
        assert isinstance(op, FusedMlpOp)

    def test_impl_name_matches_config_key(self):
        assert FusedMlpOp.get_impl_name() == "fused_mlp"

    def test_requirements_declare_cache_free_operation(self):
        reqs = FusedMlpOp.get_requirements()
        assert reqs.cache.requires_cache is False

    def test_metadata_config_plumbing_reaches_adapter(self):
        mlp_cfg = FusedMlpConfig(platform="auto", backend="any", tile_i=2048, prefill_threshold=128)
        cfg = EasyDeLBaseConfig(operation_configs={"fused_mlp": mlp_cfg})
        metadata = _metadata(base_config=cfg)
        assert metadata.get_operation_config("fused_mlp") is mlp_cfg


class TestAdapterStructure:
    """The adapter must import ejkernel only through public module surfaces."""

    def test_no_private_ejkernel_or_pallas_imports(self):
        import easydel.operations.kernels.fused_mlp as adapter_module

        source = inspect.getsource(adapter_module)
        forbidden = r"ejkernel\.kernels\.|_pallas|_xla|_triton|pallas_call|jax\.experimental\.pallas|\bpl\."
        assert re.search(forbidden, source) is None

    def test_public_ejkernel_import_present(self):
        import easydel.operations.kernels.fused_mlp as adapter_module

        source = inspect.getsource(adapter_module)
        assert "from ejkernel.modules import fused_mlp" in source


class TestForwardParity:
    def test_dense_matches_ejkernel_operation(self):
        x, w_gate, w_up, w_down = _dense_inputs()
        op = OperationRegistry.create("fused_mlp", _metadata())
        got = op(x, w_gate, w_up, w_down)
        want = ejkernel_fused_mlp(x.reshape(-1, x.shape[-1]), w_gate, w_up, w_down).reshape(x.shape)
        assert got.shape == x.shape
        assert got.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def test_int8_matches_ejkernel_operation(self):
        rng = np.random.default_rng(1)
        k, i = 64, 128
        x = jnp.asarray(rng.normal(size=(3, 5, k)), jnp.bfloat16)
        gq, gs = _quantize_channelwise(rng, k, i)
        uq, us = _quantize_channelwise(rng, k, i)
        dq, ds = _quantize_channelwise(rng, i, k)

        op = OperationRegistry.create("fused_mlp", _metadata())
        got = op(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds)
        want = ejkernel_fused_mlp(x.reshape(-1, k), gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds).reshape(
            x.shape
        )
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def test_fused_gate_up_layout_roundtrip(self):
        x, w_gate, w_up, w_down = _dense_inputs(seed=2)
        op = OperationRegistry.create("fused_mlp", _metadata())
        fused = jnp.concatenate([w_gate, w_up], axis=-1)
        got = op(x, gate_up=fused, w_down=w_down)
        want = op(x, w_gate, w_up, w_down)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def test_runtime_dtype_applied_to_activations(self):
        x, w_gate, w_up, w_down = _dense_inputs(seed=3)
        op = OperationRegistry.create("fused_mlp", _metadata(runtime_dtype=jnp.float32))
        out = op(x, w_gate.astype(jnp.float32), w_up.astype(jnp.float32), w_down.astype(jnp.float32))
        assert out.dtype == jnp.float32

    def test_frozen_int8_gradients_flow_through_adapter(self):
        rng = np.random.default_rng(4)
        k, i = 64, 128
        x = jnp.asarray(rng.normal(size=(2, 3, k)), jnp.bfloat16)
        gq, gs = _quantize_channelwise(rng, k, i)
        uq, us = _quantize_channelwise(rng, k, i)
        dq, ds = _quantize_channelwise(rng, i, k)
        op = OperationRegistry.create("fused_mlp", _metadata())

        def loss(x):
            out = op(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        dx = jax.grad(loss)(x)
        assert dx.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(dx.astype(jnp.float32))))
        assert float(jnp.abs(dx.astype(jnp.float32)).max()) > 0.0


class TestConfigPlumbing:
    def test_operation_config_overrides_defaults(self):
        """A FusedMlpConfig on the base config must reach the ejkernel call."""
        x, w_gate, w_up, w_down = _dense_inputs(seed=5)
        mlp_cfg = FusedMlpConfig(platform="auto", backend="any", tile_i=2048, prefill_threshold=8)
        cfg = EasyDeLBaseConfig(operation_configs={"fused_mlp": mlp_cfg})
        op = OperationRegistry.create("fused_mlp", _metadata(base_config=cfg))
        # Dense path ignores both knobs numerically; this asserts the plumbing
        # does not error and output is unchanged (knobs only alter integer and
        # packed formats' internal scheduling, never dense math).
        got = op(x, w_gate, w_up, w_down)
        want = op.forward_native(x, w_gate, w_up, w_down)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def _spy_ejkernel_call(self, monkeypatch):
        """Wrap the adapter's ejkernel call, capturing the kwargs it sends.

        The platform opt-out only changes numerics on int4-MXU TPUs (the
        Pallas path is unreachable on CPU), so CPU tests must assert the
        argument contract of the public ejkernel API instead.
        """
        import easydel.operations.kernels.fused_mlp as adapter_module

        captured = {}
        real = adapter_module.ejkernel_fused_mlp

        def spy(x2d, *args, **kwargs):
            captured.update(kwargs)
            return real(x2d, *args, **kwargs)

        monkeypatch.setattr(adapter_module, "ejkernel_fused_mlp", spy)
        return captured

    def test_config_platform_reaches_ejkernel_call(self, monkeypatch):
        """FusedMlpConfig(platform='xla') must plumb through as the opt-out."""
        captured = self._spy_ejkernel_call(monkeypatch)
        x, w_gate, w_up, w_down = _dense_inputs(seed=7)
        mlp_cfg = FusedMlpConfig(platform="xla", backend="any")
        cfg = EasyDeLBaseConfig(operation_configs={"fused_mlp": mlp_cfg})
        op = OperationRegistry.create("fused_mlp", _metadata(base_config=cfg))
        got = op(x, w_gate, w_up, w_down)
        assert captured["platform"] == "xla"
        want = ejkernel_fused_mlp(x.reshape(-1, x.shape[-1]), w_gate, w_up, w_down, platform="xla").reshape(x.shape)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))

    def test_force_native_runtime_forces_xla_platform(self, monkeypatch):
        """FORCE_NATIVE_RUNTIME=1 must hard-override any requested platform."""
        monkeypatch.setenv("FORCE_NATIVE_RUNTIME", "1")
        captured = self._spy_ejkernel_call(monkeypatch)
        x, w_gate, w_up, w_down = _dense_inputs(seed=8)
        op = OperationRegistry.create("fused_mlp", _metadata())
        op(x, w_gate, w_up, w_down, platform="pallas")
        assert captured["platform"] == "xla"

    def test_explicit_kwarg_beats_operation_config(self):
        rng = np.random.default_rng(6)
        k, i = 64, 128
        x = jnp.asarray(rng.normal(size=(2, 3, k)), jnp.bfloat16)
        gq, gs = _quantize_channelwise(rng, k, i)
        uq, us = _quantize_channelwise(rng, k, i)
        dq, ds = _quantize_channelwise(rng, i, k)

        mlp_cfg = FusedMlpConfig(platform="auto", backend="any", prefill_threshold=1)
        cfg = EasyDeLBaseConfig(operation_configs={"fused_mlp": mlp_cfg})
        op = OperationRegistry.create("fused_mlp", _metadata(base_config=cfg))

        # cfg says prefill_threshold=1 (int-dot path for these 6 tokens with
        # quantize_activations); the explicit kwarg forces the fused-upcast
        # path instead, and the two paths differ numerically.
        int_dot = op(x, gq, uq, dq, gate_scale=gs, up_scale=us, down_scale=ds, quantize_activations=True)
        fused_upcast = op(
            x,
            gq,
            uq,
            dq,
            gate_scale=gs,
            up_scale=us,
            down_scale=ds,
            quantize_activations=True,
            prefill_threshold=10_000,
        )
        assert not np.array_equal(np.asarray(int_dot, np.float32), np.asarray(fused_upcast, np.float32))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
