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

"""Public dense-attention precision, gradient, and configuration contracts."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.modules import Attention, attention
from ejkernel.modules.operations.configs import AttentionConfig
from ejkernel.ops import Invocation
from ejkernel.types import MaskInfo


def _inputs():
    # Multiple nontrivial rows keep the TPU contractions on matrix hardware;
    # a single-row/vector fixture can hide DEFAULT vs HIGHEST differences.
    rng = np.random.default_rng(90210)
    return tuple(
        jnp.asarray(rng.normal(size=shape).astype(np.float32))
        for shape in ((2, 32, 4, 64), (2, 48, 2, 64), (2, 48, 2, 32))
    )


def _reference(q, k, v, *, precision, bias=None, mask=None, causal=False, soft_cap=None):
    """Independent BHSD matmul reference, expanding KV heads rather than Q."""
    repeats = q.shape[2] // k.shape[2]
    q = q.astype(jnp.float32).transpose(0, 2, 1, 3)
    k = jnp.repeat(k.astype(jnp.float32), repeats, axis=2).transpose(0, 2, 3, 1)
    v = jnp.repeat(v.astype(jnp.float32), repeats, axis=2).transpose(0, 2, 1, 3)
    logits = jnp.matmul(q * 0.125, k, precision=precision)
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    if causal:
        visible = jnp.arange(k.shape[-1])[None, :] <= jnp.arange(q.shape[-2])[:, None]
        logits = jnp.where(visible, logits, jnp.finfo(jnp.float32).min)
    if bias is not None:
        logits = logits + bias
    elif mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
    unnormalized = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    weights = unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)
    out = jnp.matmul(weights, v, precision=precision).transpose(0, 2, 1, 3)
    return out, weights


def _assert_outputs(actual, expected, *, atol=2e-6, rtol=2e-5):
    for result, reference in zip(actual, expected, strict=True):
        assert result.shape == reference.shape
        assert result.dtype == jnp.float32
        assert np.isfinite(np.asarray(result)).all()
        np.testing.assert_allclose(result, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("feature", ["plain", "mask", "bias_causal_soft_cap"])
@pytest.mark.parametrize("precision", [jax.lax.Precision.HIGHEST, (jax.lax.Precision.HIGHEST,) * 2])
def test_public_attention_highest_outputs_weights_and_gradients(feature, precision):
    q, k, v = _inputs()
    bias = None
    mask = None
    kwargs = {}
    ref_kwargs = {}
    if feature == "mask":
        mask = jnp.broadcast_to(jnp.arange(48)[None, None, None, :] % 3 != 1, (2, 1, 32, 48))
        kwargs["mask_info"] = MaskInfo.from_attention_mask(mask)
        ref_kwargs["mask"] = mask
    elif feature == "bias_causal_soft_cap":
        bias = jnp.asarray(np.random.default_rng(7).normal(size=(2, 4, 32, 48)), dtype=jnp.float32)
        kwargs.update(causal=True, logits_soft_cap=3.0)
        ref_kwargs.update(bias=bias, causal=True, soft_cap=3.0)

    def actual(q, k, v):
        return attention(q, k, v, bias, dtype=jnp.float32, softmax_dtype=jnp.float32, precision=precision, **kwargs)

    reference = partial(_reference, precision=jax.lax.Precision.HIGHEST, **ref_kwargs)
    # Explicit HIGHEST must override a lower ambient policy, including under jit.
    with jax.default_matmul_precision("bfloat16"):
        _assert_outputs(jax.jit(actual)(q, k, v), jax.jit(reference)(q, k, v))

        def objective(fn, q, k, v):
            out, weights = fn(q, k, v)
            # Both returned tensors contribute nontrivial cotangents.
            return jnp.sum(jnp.sin(out)) + 0.7 * jnp.sum(jnp.square(weights))

        actual_grads = jax.jit(jax.grad(partial(objective, actual), argnums=(0, 1, 2)))(q, k, v)
        reference_grads = jax.jit(jax.grad(partial(objective, reference), argnums=(0, 1, 2)))(q, k, v)
    _assert_outputs(actual_grads, reference_grads, atol=3e-5, rtol=5e-5)


def test_public_attention_none_inherits_ambient_but_explicit_default_does_not():
    # None continues automatic dispatch. On GPU it may select TileLang, whose
    # legacy semantics do not expose JAX's ambient matmul precision.
    if jax.default_backend() == "gpu":
        pytest.skip("Ambient precision is an XLA contract; automatic GPU dispatch may use TileLang")
    q, k, v = _inputs()
    results = {}
    for ambient in ("bfloat16", "float32"):
        with jax.default_matmul_precision(ambient):
            # Fresh functions intentionally trace independently in each context.
            omitted = jax.jit(lambda q, k, v: attention(q, k, v, dtype=jnp.float32))(q, k, v)
            inherited = jax.jit(lambda q, k, v: attention(q, k, v, dtype=jnp.float32, precision=None))(q, k, v)
            explicit = jax.jit(
                lambda q, k, v: attention(q, k, v, dtype=jnp.float32, precision=jax.lax.Precision.DEFAULT)
            )(q, k, v)
            inherited_ref = jax.jit(partial(_reference, precision=None))(q, k, v)
            explicit_ref = jax.jit(partial(_reference, precision=jax.lax.Precision.DEFAULT))(q, k, v)
        _assert_outputs(omitted, inherited)
        _assert_outputs(inherited, inherited_ref)
        _assert_outputs(explicit, explicit_ref)
        results[ambient] = inherited, explicit

    _assert_outputs(results["bfloat16"][1], results["float32"][1])
    if jax.default_backend() == "tpu":
        # Assert the fixture really distinguishes the policies on physical TPU,
        # separately for QK (weights) and the final output (QK plus AV).
        for inherited, explicit in zip(*results["float32"], strict=True):
            assert np.max(np.abs(np.asarray(inherited) - np.asarray(explicit))) > 1e-5


def test_public_attention_highest_controls_value_contraction():
    q, k, v = _inputs()
    # Zero logits give exactly uniform 1/128 probabilities, independent of QK
    # precision. Only the AV contraction can cause a policy-dependent result.
    q = jnp.zeros_like(q)
    k = jnp.zeros((2, 128, 2, 64), dtype=jnp.float32)
    v = jnp.asarray(np.random.default_rng(16).normal(size=(2, 128, 2, 32)), dtype=jnp.float32)
    with jax.default_matmul_precision("bfloat16"):
        actual = jax.jit(lambda q, k, v: attention(q, k, v, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST))(
            q, k, v
        )
    expected_out = np.broadcast_to(np.repeat(np.asarray(v).mean(axis=1, keepdims=True), 2, axis=2), (2, 32, 4, 32))
    expected_weights = np.full((2, 4, 32, 128), 1 / 128, dtype=np.float32)
    _assert_outputs(actual, (expected_out, expected_weights))


@pytest.mark.parametrize("precision", [jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST])
def test_explicit_precision_config_candidates_are_xla_and_have_distinct_cache_keys(precision):
    q, k, v = _inputs()
    kernel = Attention()
    kwargs = dict(query=q, key=k, value=v)
    inv = Invocation(op_id="attention", args=(), kwargs={**kwargs, "precision": precision})
    assert kernel.heuristic_cfg(inv).platform == "xla"
    for candidates in (kernel.candidate_cfgs(inv), kernel.candidate_cfgs_gpu(inv), kernel.candidate_cfgs_tpu(inv)):
        assert candidates
        assert all(cfg.platform == "xla" for cfg in candidates)
    keys = {
        Invocation(op_id="attention", args=(), kwargs={**kwargs, "precision": p}).make_key(kernel.key_builder)
        for p in (None, jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)
    }
    assert len(keys) == 3


@pytest.mark.parametrize("platform", ["tilelang", "pallas", "triton"])
def test_explicit_precision_rejects_unsupported_manual_platform(platform):
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="Explicit attention precision requires"):
        Attention().run(q, k, v, cfg=AttentionConfig(platform=platform), precision=jax.lax.Precision.DEFAULT)
