# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Module-level tests for GatedDeltaRule operation."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import pytest

from ejkernel import modules
from ejkernel.kernels._xla.gated_delta_rule import gated_delta_rule as gated_delta_rule_xla
from ejkernel.modules import operations
from ejkernel.modules.operations import (
    GatedDeltaRule,
    GatedDeltaRuleConfig,
    gated_delta_rule,
    gdr_attention,
)

from ._utils import assert_allclose, device_platform


def test_gdr_is_exported_from_modules_init_files():
    assert hasattr(operations, "gated_delta_rule")
    assert hasattr(operations, "GatedDeltaRule")
    assert hasattr(operations, "GatedDeltaRuleConfig")
    assert hasattr(operations, "gdr_attention")

    assert hasattr(modules, "gated_delta_rule")
    assert hasattr(modules, "GatedDeltaRule")
    assert hasattr(modules, "GatedDeltaRuleConfig")
    assert hasattr(modules, "gdr_attention")


def test_gdr_attention_alias():
    assert gdr_attention is gated_delta_rule


def test_gdr_config_serialization():
    cfg = GatedDeltaRuleConfig(chunk_size=128, platform="xla", backend="any")
    d = cfg.to_dict()
    assert d["chunk_size"] == 128
    assert d["platform"] == "xla"

    cfg2 = GatedDeltaRuleConfig.from_dict(d)
    assert cfg2.chunk_size == cfg.chunk_size
    assert cfg2.platform == cfg.platform

    j = cfg.to_json()
    cfg3 = GatedDeltaRuleConfig.from_json(j)
    assert cfg3.chunk_size == cfg.chunk_size


def _make_inputs(batch=1, seq_len=8, heads=2, qk_dim=8, v_dim=8, dtype=jnp.bfloat16, seed=0):
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, kb, kd = jax.random.split(key, 5)

    q = jax.random.normal(kq, (batch, seq_len, heads, qk_dim), dtype=jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (batch, seq_len, heads, qk_dim), dtype=jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (batch, seq_len, heads, v_dim), dtype=jnp.float32).astype(dtype)
    beta = jax.nn.sigmoid(jax.random.normal(kb, (batch, seq_len, heads), dtype=jnp.float32)).astype(dtype)
    decay = (jax.random.normal(kd, (batch, seq_len, heads), dtype=jnp.float32) * -0.01).astype(dtype)

    return q, k, v, beta, decay


def test_gdr_chunked_and_recurrent_match_and_return_state():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out_chunked, state = gated_delta_rule(q, k, v, beta, decay, return_state=True, use_chunked=True, platform="xla")
    out_recurrent = gated_delta_rule(q, k, v, beta, decay, return_state=False, use_chunked=False, platform="xla")

    assert out_chunked.shape == (batch, seq_len, heads, v_dim)
    assert out_recurrent.shape == (batch, seq_len, heads, v_dim)
    assert state.shape == (batch, heads, qk_dim, v_dim)

    assert_allclose(out_chunked, out_recurrent, atol=0.25)


def test_gdr_single_step_with_state():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    _, state = gated_delta_rule(q, k, v, beta, decay, return_state=True, use_chunked=True, platform="xla")

    q1, k1, v1, beta1, decay1 = _make_inputs(batch, 1, heads, qk_dim, v_dim, seed=1)

    out_next, state_next = gated_delta_rule(q1, k1, v1, beta1, decay1, state, return_state=True, platform="xla")
    assert out_next.shape == (batch, 1, heads, v_dim)
    assert state_next.shape == (batch, heads, qk_dim, v_dim)


def test_gdr_single_step_function_fast_path_bypasses_executor(monkeypatch):
    batch, heads, qk_dim, v_dim = 1, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, 1, heads, qk_dim, v_dim, dtype=jnp.bfloat16, seed=21)
    init = jax.random.normal(
        jax.random.PRNGKey(22),
        (batch, heads, qk_dim, v_dim),
        dtype=jnp.bfloat16,
    )
    cfg = GatedDeltaRuleConfig(chunk_size=8, platform="xla", backend="any")
    module_impl = importlib.import_module("ejkernel.modules.operations.gated_delta_rule")

    def _fail_executor(*args, **kwargs):
        raise AssertionError("single-step inference should bypass the executor")

    monkeypatch.setattr(module_impl, "_executor", _fail_executor)

    out_fast, state_fast = module_impl.gated_delta_rule(
        q,
        k,
        v,
        beta,
        decay,
        init,
        return_state=True,
        cfg=cfg,
    )
    out_xla, state_xla = gated_delta_rule_xla(
        q,
        k,
        v,
        beta,
        decay,
        initial_state=init,
        use_chunked=False,
    )

    assert_allclose(out_fast, out_xla, atol=0.01)
    assert_allclose(state_fast, state_xla, atol=0.03)


def test_gdr_class_api():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    op = GatedDeltaRule()
    cfg = GatedDeltaRuleConfig(chunk_size=8, platform="xla")

    out = op.run(q, k, v, beta, decay, use_chunked=True, cfg=cfg)
    assert out.shape == (batch, seq_len, heads, v_dim)


def test_gdr_no_decay():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, _ = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out = gated_delta_rule(q, k, v, beta, None, platform="xla")
    assert out.shape == (batch, seq_len, heads, v_dim)
    assert jnp.all(jnp.isfinite(out))


def test_gdr_backward():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.float32)

    def loss_fn(q, k, v, beta, decay):
        out = gated_delta_rule(q, k, v, beta, decay, use_chunked=True, platform="xla")
        return jnp.sum(out)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    for i, g in enumerate(grads):
        assert g.shape == [q, k, v, beta, decay][i].shape, f"grad {i} shape mismatch"
        assert jnp.all(jnp.isfinite(g)), f"grad {i} has non-finite values"


def test_gdr_use_qk_l2norm_toggle():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim)

    out_norm = gated_delta_rule(q, k, v, beta, decay, use_qk_l2norm=True, platform="xla")
    out_nonorm = gated_delta_rule(q, k, v, beta, decay, use_qk_l2norm=False, platform="xla")

    assert out_norm.shape == out_nonorm.shape
    # Results should differ when L2 norm is toggled
    assert not jnp.allclose(out_norm.astype(jnp.float32), out_nonorm.astype(jnp.float32), atol=1e-3)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_gdr_pallas_matches_xla_and_auto_dispatches_on_tpu():
    batch, seq_len, heads, qk_dim, v_dim = 1, 8, 2, 8, 8
    q, k, v, beta, decay = _make_inputs(batch, seq_len, heads, qk_dim, v_dim, dtype=jnp.float32, seed=7)
    cfg_xla = GatedDeltaRuleConfig(chunk_size=8, platform="xla", backend="any")
    cfg_pallas = GatedDeltaRuleConfig(chunk_size=8, platform="pallas", backend="any")
    cfg_auto = GatedDeltaRuleConfig(chunk_size=8, platform="auto", backend="any")

    out_xla, state_xla = gated_delta_rule(q, k, v, beta, decay, return_state=True, use_chunked=True, cfg=cfg_xla)
    out_pallas, state_pallas = gated_delta_rule(
        q, k, v, beta, decay, return_state=True, use_chunked=True, cfg=cfg_pallas
    )
    out_auto, state_auto = gated_delta_rule(q, k, v, beta, decay, return_state=True, use_chunked=True, cfg=cfg_auto)

    assert_allclose(out_pallas, out_xla, atol=0.25)
    assert_allclose(state_pallas, state_xla, atol=0.25)
    assert_allclose(out_auto, out_pallas, atol=0.0)
    assert_allclose(state_auto, state_pallas, atol=0.0)
