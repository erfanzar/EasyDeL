from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from ejkernel.modules.operations import ragged_decode_attention

from ._utils import assert_allclose, dense_attention_reference, device_platform, rand_qkv


def test_ragged_decode_attention_matches_dense_reference_and_supports_query_rank_4():
    key = jax.random.PRNGKey(0)
    batch, seq_len, q_heads, kv_heads, head_dim = 3, 16, 4, 2, 32

    q3, k, v = rand_qkv(key, batch=batch, q_len=1, kv_len=seq_len, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim)
    q3 = q3[:, 0, :, :]

    sequence_start = jnp.array([0, 2, 0], dtype=jnp.int32)
    sequence_end = jnp.array([8, 12, 16], dtype=jnp.int32)
    sliding_window = (6, 0)
    logits_soft_cap = 10.0
    softmax_aux = jnp.array([0.1, -0.2], dtype=jnp.bfloat16)

    out3 = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=head_dim**-0.5,
        platform="xla",
    )
    out4 = ragged_decode_attention(
        q3[:, None, :, :],
        k,
        v,
        sequence_start,
        sequence_end,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=head_dim**-0.5,
        platform="xla",
    )

    assert out3.shape == (batch, q_heads, head_dim)
    assert out4.shape == (batch, 1, q_heads, head_dim)
    assert_allclose(out4[:, 0], out3, atol=0.0)

    q_pos = sequence_end - 1
    kv_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    left, right = sliding_window
    mask = (kv_pos >= sequence_start[:, None]) & (kv_pos < sequence_end[:, None])
    mask = mask & (kv_pos >= (q_pos[:, None] - left)) & (kv_pos <= (q_pos[:, None] + right))
    mask = mask[:, None, None, :]

    ref_out, _ = dense_attention_reference(
        q3[:, None, :, :],
        k,
        v,
        attention_mask=mask,
        softmax_scale=head_dim**-0.5,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
    assert_allclose(out3, ref_out[:, 0], atol=0.25)


@pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only automatic backend routing")
@pytest.mark.parametrize("feature", ["sliding_window", "logits_soft_cap", "softmax_aux"])
def test_ragged_decode_attention_auto_supports_optional_features_on_gpu(feature):
    q3, k, v = rand_qkv(jax.random.PRNGKey(2), batch=2, q_len=1, kv_len=32, q_heads=4, kv_heads=2, head_dim=32)
    q3 = q3[:, 0]
    starts = jnp.array([0, 0], dtype=jnp.int32)
    ends = jnp.array([16, 32], dtype=jnp.int32)
    options = {
        "sliding_window": (8, 0),
        "logits_soft_cap": 10.0,
        "softmax_aux": jnp.array([0.1], dtype=jnp.float32),
    }
    selected = {feature: options[feature]}
    aux = selected.pop("softmax_aux", None)  # softmax_aux is positional-only in the public API
    actual = ragged_decode_attention(q3, k, v, starts, ends, aux, platform="auto", **selected)

    # Independent dense reference (not the XLA kernel): mask by [start, end) and the optional window.
    kv_len = k.shape[1]
    q_pos = ends - 1
    kv_pos = jnp.arange(kv_len, dtype=jnp.int32)[None, :]
    mask = (kv_pos >= starts[:, None]) & (kv_pos < ends[:, None])
    if feature == "sliding_window":
        left, right = options["sliding_window"]
        mask = mask & (kv_pos >= (q_pos[:, None] - left)) & (kv_pos <= (q_pos[:, None] + right))
    expected, _ = dense_attention_reference(
        q3[:, None, :, :],
        k,
        v,
        attention_mask=mask[:, None, None, :],
        softmax_scale=q3.shape[-1] ** -0.5,
        logits_soft_cap=selected.get("logits_soft_cap"),
        softmax_aux=aux,
    )
    assert_allclose(actual, expected[:, 0], atol=0.25)


_OPS_MODULE = "ejkernel.modules.operations.ragged_decode_attention"


@pytest.mark.cpu_ok
@pytest.mark.parametrize("triton_registered", [True, False])
@pytest.mark.parametrize("feature", [None, "sliding_window", "logits_soft_cap", "softmax_aux"])
def test_ragged_decode_attention_gpu_auto_platform_selection(monkeypatch, feature, triton_registered):
    """On GPU, auto + optional features picks Triton when registered, else XLA; no features stays auto."""
    import importlib
    from types import SimpleNamespace

    from ejkernel.kernels._registry import Backend, Platform

    mod = importlib.import_module(_OPS_MODULE)
    specs = [SimpleNamespace(platform=Platform.PALLAS, backend=Backend.GPU)]
    if triton_registered:
        specs.append(SimpleNamespace(platform=Platform.TRITON, backend=Backend.GPU))
    captured = {}

    def fake_executor(_op, **kwargs):
        captured.update(kwargs)
        return kwargs["query"]

    monkeypatch.setattr(mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(mod, "kernel_registry", SimpleNamespace(list_implementations=lambda _name: specs))
    monkeypatch.setattr(mod, "_ragged_decode_attention_executor", fake_executor)

    import numpy as np

    q = np.zeros((1, 2, 4), dtype=np.float32)
    kv = np.zeros((1, 8, 2, 4), dtype=np.float32)
    starts = np.zeros((1,), dtype=np.int32)
    ends = np.full((1,), 8, dtype=np.int32)
    kwargs = {}
    aux = None  # softmax_aux is positional-only in the public API
    if feature == "sliding_window":
        kwargs["sliding_window"] = (4, 0)
    elif feature == "logits_soft_cap":
        kwargs["logits_soft_cap"] = 10.0
    elif feature == "softmax_aux":
        aux = np.zeros((1,), dtype=np.float32)
    mod.ragged_decode_attention(q, kv, kv, starts, ends, aux, **kwargs)

    if feature is None:
        assert captured["platform"] is None
    else:
        assert captured["platform"] == ("triton" if triton_registered else "xla")


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ragged_decode_attention_pallas_matches_xla_on_tpu():
    key = jax.random.PRNGKey(1)
    batch, seq_len, q_heads, kv_heads, head_dim = 2, 32, 4, 2, 32
    q3, k, v = rand_qkv(key, batch=batch, q_len=1, kv_len=seq_len, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim)
    q3 = q3[:, 0, :, :]
    sequence_start = jnp.zeros((batch,), dtype=jnp.int32)
    sequence_end = jnp.array([16, 32], dtype=jnp.int32)

    out_xla = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        None,
        platform="xla",
    )
    out_pallas = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        None,
        platform="pallas",
    )

    assert_allclose(out_pallas, out_xla, atol=0.35)
