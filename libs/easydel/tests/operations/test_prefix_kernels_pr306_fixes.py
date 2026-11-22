"""Regression tests for PR #306 EasyDeL prefix-shared attention & RingAttn fixes.

Covers:
- Finding #27: FlexibleAttentionModule.forward / AttentionModule.concatenate deferred
  prefix-shared fork handling of additive bias (ALiBi / MPT), call-time logits_soft_cap
  (Grok-1), sliding_window, causal=False, normalize_output=False, softmax_aux, active
  dropout, and output_attentions=True.
- Finding #28: FlexibleAttentionModule forwarding raw precision (None / explicit) to
  RingAttn and RingPagedAttn so ambient ``jax.default_matmul_precision("highest")``
  is honoured on model calls.
"""

from __future__ import annotations

import easydel as ed
import easydel.caching.shared_prefix as shared_prefix
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import TransformerCache
from easydel.caching.shared_prefix import PrefixCaptureView, PrefixForkView
from easydel.layers.attention._flexible import AttentionModule, FlexibleAttentionModule
from easydel.modules.mosaic_mpt.mosaic_configuration import MptAttentionConfig, MptConfig
from ejkernel.modules.operations.configs import RingAttentionConfig
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from spectrax import common_types


def _dot_precisions(node):
    """Yield the ``precision`` parameter of every ``dot_general`` in a (nested) jaxpr."""
    if hasattr(node, "eqns"):
        for equation in node.eqns:
            if equation.primitive.name == "dot_general":
                yield equation.params["precision"]
            yield from _dot_precisions(equation.params)
    elif hasattr(node, "jaxpr"):
        yield from _dot_precisions(node.jaxpr)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _dot_precisions(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _dot_precisions(value)


def _fork_inputs(
    *,
    prompt_batch: int = 2,
    num_generations: int = 2,
    prefix_len: int = 6,
    completion_len: int = 5,
    num_q_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    fork_batch = prompt_batch * num_generations
    prefix_key = jnp.asarray(rng.standard_normal((prompt_batch, prefix_len, num_kv_heads, head_dim)), jnp.float32)
    prefix_value = jnp.asarray(rng.standard_normal((prompt_batch, prefix_len, num_kv_heads, head_dim)), jnp.float32)
    prefix_seg = np.zeros((prompt_batch, prefix_len), dtype=np.int32)
    prefix_seg[0, :2] = -1  # left-padded prompt

    query = jnp.asarray(rng.standard_normal((fork_batch, completion_len, num_q_heads, head_dim)), jnp.float32)
    key = jnp.asarray(rng.standard_normal((fork_batch, completion_len, num_kv_heads, head_dim)), jnp.float32)
    value = jnp.asarray(rng.standard_normal((fork_batch, completion_len, num_kv_heads, head_dim)), jnp.float32)

    q_seg = np.zeros((fork_batch, completion_len), dtype=np.int32)
    q_seg[1, -2:] = -1  # right-padded completion
    q_seg[3, -1:] = -1
    mask_info = MaskInfo.from_segments(q_segment_ids=jnp.asarray(q_seg))

    deferred_view = PrefixForkView(
        prefix_key=prefix_key,
        prefix_value=prefix_value,
        prefix_segment_ids=jnp.asarray(prefix_seg),
        num_generations=num_generations,
        deferred_attention=True,
    )
    eager_view = deferred_view.replace(deferred_attention=False)
    return query, key, value, mask_info, deferred_view, eager_view


def _make_performer(attn_mechanism: str = "vanilla", dropout_prob: float = 0.0, **cfg_kwargs):
    cfg_defaults = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
    )
    cfg_defaults.update(cfg_kwargs)
    cfg = ed.LlamaConfig(**cfg_defaults)
    performer = FlexibleAttentionModule(
        base_config=cfg,
        softmax_scale=16**-0.5,
        dropout_prob=dropout_prob,
        rngs=spx.Rngs(0),
    )
    return cfg, performer


# -------------------------------------------------------------------------------------- Finding #27


@pytest.mark.parametrize("attn_mechanism", ["vanilla", "sdpa"])
def test_deferred_prefix_fork_with_additive_bias_matches_eager(attn_mechanism):
    """Additive bias passed to FlexibleAttentionModule.forward must not be dropped by grouped fork."""
    cfg, performer = _make_performer(attn_mechanism)
    query, key, value, mask_info, deferred_view, eager_view = _fork_inputs(seed=11)
    rng = np.random.default_rng(12)
    kv_len = deferred_view.prefix_key.shape[1] + key.shape[1]
    additive_bias = jnp.asarray(
        rng.standard_normal((query.shape[0], query.shape[2], query.shape[1], kv_len)),
        jnp.float32,
    )

    def run_deferred(q, k, v, b):
        out = performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            bias=b,
            mask_info=mask_info,
            cache_view=deferred_view,
            causal=True,
            deterministic=True,
        ).attention_outputs
        return jnp.sum(out * out), out

    def run_eager(q, k, v, b):
        full_k, full_v, full_mask, view = eager_view.concatenate_to_cache(key=k, value=v, mask_info=mask_info)
        out = performer.forward(
            query_states=q,
            key_states=full_k,
            value_states=full_v,
            mode=common_types.MODE_TRAIN,
            bias=b,
            mask_info=full_mask,
            cache_view=view,
            causal=True,
            deterministic=True,
        ).attention_outputs
        return jnp.sum(out * out), out

    with jax.default_matmul_precision("highest"), cfg.mesh:
        (_, out_def), grads_def = jax.value_and_grad(run_deferred, argnums=(0, 1, 2, 3), has_aux=True)(
            query, key, value, additive_bias
        )
        (_, out_ref), grads_ref = jax.value_and_grad(run_eager, argnums=(0, 1, 2, 3), has_aux=True)(
            query, key, value, additive_bias
        )
        out_no_bias = performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            bias=None,
            mask_info=mask_info,
            cache_view=deferred_view,
            causal=True,
            deterministic=True,
        ).attention_outputs

    # Verify bias genuinely affected the output (was not silently dropped) and matches eager parity.
    assert float(jnp.max(jnp.abs(out_def - out_no_bias))) > 1e-2
    np.testing.assert_allclose(np.asarray(out_def), np.asarray(out_ref), rtol=1e-4, atol=1e-5)
    for g_def, g_ref in zip(grads_def, grads_ref, strict=True):
        np.testing.assert_allclose(np.asarray(g_def), np.asarray(g_ref), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("attn_mechanism", ["vanilla", "sdpa"])
def test_mpt_alibi_prefix_shared_logps_match_repeated_prompt_forward(attn_mechanism):
    """MptForCausalLM (ALiBi) capture+fork attention must match repeated-prompt forward."""
    cfg = MptConfig(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        expansion_ratio=2,
        max_seq_len=64,
        attn_config=MptAttentionConfig(alibi=True, alibi_bias_max=8),
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
    )
    rng = np.random.default_rng(7)
    prompts, prompt_len, completion_len, num_generations = 2, 6, 5, 2
    prompt_mask = np.ones((prompts, prompt_len), np.int32)
    prompt_mask[0, :2] = 0
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 60, (prompts, prompt_len)), 0).astype(np.int32)
    rows = prompts * num_generations
    completion_mask = np.ones((rows, completion_len), np.int32)
    completion_mask[1, -2:] = 0
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 60, (rows, completion_len)), 0).astype(np.int32)

    with jax.default_matmul_precision("highest"), cfg.mesh:
        model = ed.MptForCausalLM(
            config=cfg,
            rngs=spx.Rngs(0),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
        )
        # Capture prompt KV once.
        cap_out = model(
            input_ids=jnp.asarray(prompt_ids),
            attention_mask=jnp.asarray(prompt_mask),
            past_key_values=TransformerCache(views=[PrefixCaptureView() for _ in range(cfg.n_layers)]),
            mode=common_types.MODE_TRAIN,
        )
        prefix_seg = jnp.where(jnp.asarray(prompt_mask) > 0, 0, -1).astype(jnp.int32)
        fork_views = [
            PrefixForkView(
                prefix_key=v.key,
                prefix_value=v.value,
                prefix_segment_ids=prefix_seg,
                num_generations=num_generations,
            )
            for v in cap_out.past_key_values.views
        ]
        # Fork completions against captured prefix (ALiBi bias must be materialized & applied).
        fork_out = model(
            input_ids=jnp.asarray(completion_ids),
            attention_mask=jnp.asarray(completion_mask),
            past_key_values=TransformerCache(views=fork_views),
            mode=common_types.MODE_TRAIN,
        )
        # Repeated-prompt reference.
        full_ids = jnp.asarray(np.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1))
        full_mask = jnp.asarray(np.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1))
        ref_out = model(input_ids=full_ids, attention_mask=full_mask, mode=common_types.MODE_TRAIN)

    valid = completion_mask > 0
    shared_logits = np.asarray(fork_out.logits)[valid]
    ref_logits = np.asarray(ref_out.logits[:, prompt_len:, :])[valid]
    err = float(np.max(np.abs(shared_logits - ref_logits)))
    assert err <= 1e-4, f"MptForCausalLM ({attn_mechanism}) ALiBi prefix-shared mismatch: {err}"


@pytest.mark.parametrize(
    "case_name,extra_kwargs",
    [
        ("call_time_logits_soft_cap", {"logits_soft_cap": 2.0}),
        ("sliding_window", {"sliding_window": 3}),
        ("non_causal", {"causal": False}),
        ("unnormalized", {"normalize_output": False}),
        ("sinks", {"softmax_aux": jnp.asarray([0.5, -0.3, 0.2, 0.8], jnp.float32)}),
        ("dropout", {"deterministic": False, "dropout_prob": 0.5, "dropout_rng": jax.random.PRNGKey(42)}),
        ("output_attentions", {"output_attentions": True}),
    ],
)
def test_deferred_prefix_fork_unsupported_features_fall_back_to_eager_parity(case_name, extra_kwargs):
    """Every feature not supported by grouped_fork_attention must materialize and match eager vanilla."""
    cfg, performer = _make_performer("vanilla")
    query, key, value, mask_info, deferred_view, eager_view = _fork_inputs(seed=21)

    common_kwargs = dict(
        mode=common_types.MODE_TRAIN,
        causal=True,
        deterministic=True,
    )
    common_kwargs.update(extra_kwargs)

    with jax.default_matmul_precision("highest"), cfg.mesh:
        out_def = performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mask_info=mask_info,
            cache_view=deferred_view,
            **common_kwargs,
        )
        full_k, full_v, full_mask, mat_view = eager_view.concatenate_to_cache(
            key=key, value=value, mask_info=mask_info
        )
        out_ref = performer.forward(
            query_states=query,
            key_states=full_k,
            value_states=full_v,
            mask_info=full_mask,
            cache_view=mat_view,
            **common_kwargs,
        )
        out_plain = performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            mask_info=mask_info,
            cache_view=deferred_view,
            causal=True,
            deterministic=True,
        )

    np.testing.assert_allclose(
        np.asarray(out_def.attention_outputs),
        np.asarray(out_ref.attention_outputs),
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"Mismatch for {case_name}",
    )
    if case_name == "output_attentions":
        assert out_def.attention_weights is not None
        np.testing.assert_allclose(
            np.asarray(out_def.attention_weights),
            np.asarray(out_ref.attention_weights),
            rtol=1e-4,
            atol=1e-5,
        )
    else:
        assert float(jnp.max(jnp.abs(out_def.attention_outputs - out_plain.attention_outputs))) > 1e-4, (
            f"Feature {case_name} had no effect vs plain grouped fork"
        )


def test_standard_deferred_prefix_fork_still_avoids_copy(monkeypatch):
    """Standard causal vanilla fork with no extra features must still stay on grouped_fork_attention."""
    cfg, performer = _make_performer("vanilla")
    query, key, value, mask_info, deferred_view, _ = _fork_inputs(seed=33)

    def forbid_copy(self, **kwargs):
        raise AssertionError("concatenate_to_cache must not run on plain grouped fork")

    monkeypatch.setattr(shared_prefix.PrefixForkView, "concatenate_to_cache", forbid_copy)
    with jax.default_matmul_precision("highest"), cfg.mesh:
        out = performer.forward(
            query_states=query,
            key_states=key,
            value_states=value,
            mode=common_types.MODE_TRAIN,
            mask_info=mask_info,
            cache_view=deferred_view,
            causal=True,
            deterministic=True,
        )
    assert out.attention_outputs.shape == query.shape
    assert jnp.all(jnp.isfinite(out.attention_outputs))


@pytest.mark.parametrize("reason", ["alibi", "non_causal"])
def test_concatenate_does_not_defer_for_alibi_or_non_causal(reason):
    """AttentionModule.concatenate must materialize full P+C KV for ALiBi or non-causal layers."""
    cfg, performer = _make_performer("vanilla")
    query, key, value, mask_info, _, eager_view = _fork_inputs(seed=44)

    class _DummyAttn(AttentionModule):
        def __init__(self, config, attention_performer, attention_type):
            super().__init__(config)
            self.attention_performer = attention_performer
            self.attention_type = attention_type
            self.dtype = jnp.float32

    layer = _DummyAttn(cfg, performer, "alibi" if reason == "alibi" else "standard")
    k_out, v_out, m_out, _, view_out, _ = layer.concatenate(
        query=query,
        key=key,
        value=value,
        mask_info=mask_info,
        cache_view=eager_view,
        causal=(reason != "non_causal"),
    )
    expected_kv_len = eager_view.prefix_key.shape[1] + key.shape[1]
    assert view_out.deferred_attention is False
    assert k_out.shape[1] == expected_kv_len
    assert v_out.shape[1] == expected_kv_len
    assert m_out is not None and m_out.kv_positions is not None and m_out.kv_positions.shape[1] == expected_kv_len


# -------------------------------------------------------------------------------------- Finding #28


def _ring_flexible_module(attn_mechanism: str = "ring"):
    axis_dims = (1, 1, 1, 2, 2, 1) if jax.device_count() >= 4 else (1, 1, 1, 1, 1, 1)
    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sharding_axis_dims=axis_dims,
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
        operation_configs={"ring": RingAttentionConfig(platform="xla", backend="any")},
    )
    return cfg, FlexibleAttentionModule(base_config=cfg, softmax_scale=16**-0.5, dropout_prob=0.0)


@pytest.mark.parametrize("attn_mechanism", ["ring", "ring_paged"])
@pytest.mark.parametrize(
    "ambient,expected",
    [
        (None, lax.Precision.DEFAULT),
        ("highest", lax.Precision.HIGHEST),
        ("float32", lax.Precision.HIGHEST),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("gradient", [False, True])
def test_flexible_attention_ring_honours_ambient_matmul_precision(
    attn_mechanism, ambient, expected, with_bias, gradient
):
    """FlexibleAttentionModule(ring/ring_paged) must preserve precision=None so ambient precision applies."""
    cfg, performer = _ring_flexible_module(attn_mechanism)
    descriptors = [
        jax.ShapeDtypeStruct((1, 8, 4, 16), jnp.float32),
        jax.ShapeDtypeStruct((1, 8, 2, 16), jnp.float32),
        jax.ShapeDtypeStruct((1, 8, 2, 16), jnp.float32),
    ]

    def forward(q, k, v):
        bias = jnp.zeros((1, 1, 8, 8), dtype=jnp.float32) if with_bias else None
        out = performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            bias=bias,
            causal=True,
            deterministic=True,
        ).attention_outputs
        return out.sum() if gradient else out

    fn = jax.grad(forward, argnums=(0, 1, 2)) if gradient else forward
    with cfg.mesh:
        if ambient is None:
            program = jax.make_jaxpr(fn)(*descriptors)
        else:
            with jax.default_matmul_precision(ambient):
                program = jax.make_jaxpr(fn)(*descriptors)

    precisions = list(_dot_precisions(program))
    assert precisions, "Expected dot_general equations in ring jaxpr"
    assert all(p == (expected, expected) for p in precisions), set(map(str, precisions))


@pytest.mark.parametrize(
    "ambient,explicit,expected",
    [
        ("highest", lax.Precision.DEFAULT, lax.Precision.DEFAULT),
        (None, lax.Precision.HIGHEST, lax.Precision.HIGHEST),
    ],
)
def test_flexible_attention_ring_explicit_precision_overrides_ambient(ambient, explicit, expected):
    """An explicit precision passed to FlexibleAttentionModule.forward must still override ambient."""
    cfg, performer = _ring_flexible_module("ring")
    descriptors = [
        jax.ShapeDtypeStruct((1, 8, 4, 16), jnp.float32),
        jax.ShapeDtypeStruct((1, 8, 2, 16), jnp.float32),
        jax.ShapeDtypeStruct((1, 8, 2, 16), jnp.float32),
    ]

    def forward(q, k, v):
        return performer.forward(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=common_types.MODE_TRAIN,
            causal=True,
            deterministic=True,
            precision=explicit,
        ).attention_outputs

    with cfg.mesh:
        if ambient is None:
            program = jax.make_jaxpr(forward)(*descriptors)
        else:
            with jax.default_matmul_precision(ambient):
                program = jax.make_jaxpr(forward)(*descriptors)

    precisions = list(_dot_precisions(program))
    assert precisions
    assert all(p == (expected, expected) for p in precisions), set(map(str, precisions))


def test_flexible_attention_ring_executes_with_vanilla_parity_on_mesh():
    """End-to-end execution of FlexibleAttentionModule(ring) under highest ambient precision matches vanilla."""
    cfg_ring, ring_performer = _ring_flexible_module("ring")
    cfg_van, van_performer = _make_performer(
        "vanilla",
        sharding_axis_dims=(1, 1, 1, 2, 2, 1) if jax.device_count() >= 4 else (1, 1, 1, 1, 1, 1),
    )
    rng = np.random.default_rng(99)
    q = jnp.asarray(rng.standard_normal((1, 8, 4, 16)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((1, 8, 2, 16)), jnp.float32)
    v = jnp.asarray(rng.standard_normal((1, 8, 2, 16)), jnp.float32)

    with jax.default_matmul_precision("highest"):
        with cfg_ring.mesh:
            out_ring = ring_performer.forward(
                query_states=q,
                key_states=k,
                value_states=v,
                mode=common_types.MODE_TRAIN,
                causal=True,
                deterministic=True,
            ).attention_outputs
        with cfg_van.mesh:
            out_van = van_performer.forward(
                query_states=q,
                key_states=k,
                value_states=v,
                mode=common_types.MODE_TRAIN,
                causal=True,
                deterministic=True,
            ).attention_outputs

    np.testing.assert_allclose(np.asarray(out_ring), np.asarray(out_van), rtol=1e-4, atol=1e-5)
