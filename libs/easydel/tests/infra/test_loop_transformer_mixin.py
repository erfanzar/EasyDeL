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

"""Tests for generic Loop Transformer infrastructure across EasyDeL model families."""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.infra import (
    EasyDeLBaseConfig,
    LoopTransformerMixin,
    compute_loop_gate_stats,
    materialize_missing_loop_state,
)
from easydel.infra.utils import AttnMaskType
from easydel.layers import LoopGate


def _make_config(
    family: str,
    *,
    num_hidden_layers: int = 6,
    head_dim: int = 16,
    attn_mechanism: str = "vanilla",
    scan_layers: bool = False,
    gradient_checkpointing: str = "",
    loop_start_layer: int | None = None,
    loop_end_layer: int | None = None,
    num_loop_iterations: int = 1,
    loop_gate_type: str = "scalar",
    loop_gate_init_biases: tuple[float, ...] | list[float] | None = None,
    loop_use_iteration_embedding: bool = False,
    loop_kv_cache_mode: str = "separate",
    loop_kv_sliding_window: int = 128,
    loop_kv_overlap: bool = True,
):
    common = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=128,
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        scan_layers=scan_layers,
        gradient_checkpointing=gradient_checkpointing,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        loop_start_layer=loop_start_layer,
        loop_end_layer=loop_end_layer,
        num_loop_iterations=num_loop_iterations,
        loop_gate_type=loop_gate_type,
        loop_gate_init_biases=loop_gate_init_biases,
        loop_use_iteration_embedding=loop_use_iteration_embedding,
        loop_kv_cache_mode=loop_kv_cache_mode,
        loop_kv_sliding_window=loop_kv_sliding_window,
        loop_kv_overlap=loop_kv_overlap,
    )
    if family == "qwen3":
        return ed.Qwen3Config(**common)
    if family == "llama":
        return ed.LlamaConfig(**common)
    raise ValueError(f"Unsupported test family: {family}")


def _make_causal_lm(family: str, config, *, seed: int = 0, dtype=jnp.float32, param_dtype=jnp.float32):
    cls = ed.Qwen3ForCausalLM if family == "qwen3" else ed.LlamaForCausalLM
    return cls(config, dtype=dtype, param_dtype=param_dtype, rngs=spx.Rngs(seed))


@pytest.mark.cpu_ok
def test_base_config_loop_defaults_validation_and_mask_details():
    """Verify default-off base-config loop fields, validation, schedules, and mask details."""
    base = EasyDeLBaseConfig()
    assert base.is_looped is False
    assert base.num_loop_iterations == 1
    assert base.loop_start_layer is None
    assert base.loop_end_layer is None
    assert base.resolved_loop_gate_init_biases == ()
    assert issubclass(ed.Qwen3Model, LoopTransformerMixin)
    assert issubclass(ed.LlamaModel, LoopTransformerMixin)

    for family in ("qwen3", "llama"):
        plain = _make_config(family, num_hidden_layers=6)
        assert plain.is_looped is False
        assert plain.num_cache_layers == 6
        assert plain.loop_layer_schedule == tuple((i, 0) for i in range(6))

        # K=3 looped config in hybrid_sliding_window mode: [0,1] + [2,3]*3 + [4,5] = 10 virtual slots
        looped = _make_config(
            family,
            num_hidden_layers=6,
            loop_start_layer=2,
            loop_end_layer=4,
            num_loop_iterations=3,
            loop_gate_type="channel",
            loop_use_iteration_embedding=True,
            loop_kv_cache_mode="hybrid_sliding_window",
            loop_kv_sliding_window=32,
        )
        assert looped.is_looped is True
        assert looped.num_cache_layers == 10
        assert looped.resolved_loop_gate_init_biases == (-5.0, -5.0, 5.0)
        assert looped.loop_layer_schedule == (
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (2, 2),
            (3, 2),
            (4, 0),
            (5, 0),
        )
        details = looped.get_mask_details()
        assert details is not None and len(details) == 10
        for slot in (0, 1, 2, 3, 8, 9):
            assert details[slot].mask_type == AttnMaskType.FULL
        for slot in (4, 5, 6, 7):
            assert details[slot].mask_type == AttnMaskType.SLIDING
            assert details[slot].size == 32

    # Validation edge cases on generic configs
    with pytest.raises(ValueError, match="num_loop_iterations must be >= 1"):
        ed.LlamaConfig(num_hidden_layers=6, num_loop_iterations=0)
    with pytest.raises(ValueError, match="requires both loop_start_layer and loop_end_layer"):
        ed.Qwen3Config(num_hidden_layers=6, num_loop_iterations=2)
    with pytest.raises(ValueError, match="Invalid loop range"):
        ed.LlamaConfig(num_hidden_layers=6, loop_start_layer=4, loop_end_layer=8, num_loop_iterations=2)
    with pytest.raises(ValueError, match="loop_gate_type"):
        ed.Qwen3Config(
            num_hidden_layers=6, loop_start_layer=2, loop_end_layer=4, num_loop_iterations=2, loop_gate_type="bad"
        )
    with pytest.raises(ValueError, match="loop_kv_cache_mode"):
        ed.LlamaConfig(
            num_hidden_layers=6, loop_start_layer=2, loop_end_layer=4, num_loop_iterations=2, loop_kv_cache_mode="bad"
        )
    with pytest.raises(ValueError, match="loop_gate_init_biases length"):
        ed.Qwen3Config(
            num_hidden_layers=6,
            loop_start_layer=2,
            loop_end_layer=4,
            num_loop_iterations=3,
            loop_gate_init_biases=(-5.0, 5.0),
        )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_esurge_kv_cache_specs_for_looped_generic_models(family):
    """Verify eSurge splits looped Qwen3/Llama hybrid KV slots into FullAttention + SlidingWindow groups."""
    from easydel.inference.esurge.core.interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
        create_kv_cache_specs_from_config,
    )

    hybrid_cfg = _make_config(
        family,
        num_hidden_layers=6,
        head_dim=128,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=3,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=64,
    )
    specs = create_kv_cache_specs_from_config(
        config=hybrid_cfg,
        page_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=jnp.bfloat16,
    )
    assert len(specs) == 2
    full_groups = [g for g in specs if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    sliding_groups = [g for g in specs if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
    assert len(full_groups) == 1 and len(sliding_groups) == 1
    assert sliding_groups[0].kv_cache_spec.sliding_window == 64
    assert sliding_groups[0].layer_names == ["layer.4", "layer.5", "layer.6", "layer.7"]
    assert full_groups[0].layer_names == ["layer.0", "layer.1", "layer.2", "layer.3", "layer.8", "layer.9"]

    sep_cfg = _make_config(
        family,
        num_hidden_layers=6,
        head_dim=128,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=2,
        loop_kv_cache_mode="separate",
    )
    sep_specs = create_kv_cache_specs_from_config(
        config=sep_cfg,
        page_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=jnp.bfloat16,
    )
    assert len(sep_specs) == 1
    assert isinstance(sep_specs[0].kv_cache_spec, FullAttentionSpec)
    assert len(sep_specs[0].layer_names) == 8


@pytest.mark.cpu_ok
def test_materialize_missing_loop_state_generic():
    """Verify generic LoopTransformerMixin state upgrade populates missing gates and keeps saved ones."""
    cfg = _make_config(
        "llama",
        num_hidden_layers=6,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=3,
        loop_gate_type="channel",
        loop_gate_init_biases=(-3.0, -2.5, 3.5),
        loop_use_iteration_embedding=True,
    )

    class _Leaf:
        def __init__(self, shape):
            self.shape = shape

    base_k = ("parameters", "model", "layers", "0", "input_layernorm", "weight")
    g0_w = ("parameters", "model", "loop_gates", "0", "proj", "weight")
    g0_b = ("parameters", "model", "loop_gates", "0", "proj", "bias")
    g1_b = ("parameters", "model", "loop_gates", "1", "proj", "bias")
    g2_b = ("parameters", "model", "loop_gates", "2", "proj", "bias")
    g0_n = ("parameters", "model", "loop_gates", "0", "x_norm", "weight")
    iter_k = ("parameters", "model", "loop_iter_embeddings")

    expected = {
        base_k: _Leaf((64,)),
        g0_w: _Leaf((128, 64)),
        g0_b: _Leaf((64,)),
        g1_b: _Leaf((64,)),
        g2_b: _Leaf((64,)),
        g0_n: _Leaf((64,)),
        iter_k: _Leaf((3, 64)),
    }
    upgraded = materialize_missing_loop_state(cfg, {base_k: np.ones((64,), dtype=np.float32)}, expected)
    assert set(upgraded.keys()) == set(expected.keys())
    np.testing.assert_allclose(upgraded[g0_w], 0.0)
    np.testing.assert_allclose(upgraded[g0_b], -3.0)
    np.testing.assert_allclose(upgraded[g1_b], -2.5)
    np.testing.assert_allclose(upgraded[g2_b], 3.5)
    np.testing.assert_allclose(upgraded[g0_n], 0.0)
    np.testing.assert_allclose(upgraded[iter_k], 0.0)

    trained = dict(upgraded)
    trained[g1_b] = np.full((64,), 0.42, dtype=np.float32)
    preserved = materialize_missing_loop_state(cfg, dict(trained), expected)
    np.testing.assert_allclose(preserved[g1_b], 0.42)


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("num_loops", [2, 3])
def test_generic_loop_step0_identity_and_gradients(family, num_loops):
    """Verify step-0 identity parity and finite non-zero gradients on Qwen3 and Llama for K=2,3."""
    base_cfg = _make_config(family, num_hidden_layers=6)
    id_biases = tuple([-20.0] * (num_loops - 1) + [20.0])
    id_cfg = _make_config(
        family,
        num_hidden_layers=6,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=num_loops,
        loop_gate_init_biases=id_biases,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
        loop_kv_overlap=False,
    )

    base_model = _make_causal_lm(family, base_cfg, seed=0)
    loop_model = _make_causal_lm(family, id_cfg, seed=0)
    assert loop_model.model.loop_gates is not None
    assert len(loop_model.model.loop_gates) == num_loops
    assert isinstance(loop_model.model.loop_gates[0], LoopGate)

    _, base_state, _ = base_model.split_module()
    l_gdef, l_state, l_other = loop_model.split_module()
    loop_model = loop_model.merge_module(l_gdef, l_state.overlay(base_state), l_other)

    tokens = jnp.asarray([[1, 7, 19, 42, 5, 13, 88, 3]], dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        base_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(base_model, tokens)
        loop_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(loop_model, tokens)
    max_diff = float(jnp.max(jnp.abs(base_logits - loop_logits)))
    assert max_diff < 1e-4, f"{family} K={num_loops} step-0 parity failed: max_diff={max_diff}"

    # Channel gates + iteration embeddings receive finite non-zero gradients across all passes.
    act_biases = tuple([-2.0] * (num_loops - 1) + [2.0])
    act_cfg = _make_config(
        family,
        num_hidden_layers=6,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=num_loops,
        loop_gate_type="channel",
        loop_gate_init_biases=act_biases,
        loop_use_iteration_embedding=True,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
        loop_kv_overlap=True,
    )
    act_model = _make_causal_lm(family, act_cfg, seed=11)
    gdef, params, rest = act_model.split_module()

    @jax.jit
    def _vg(p):
        def _loss(p_in):
            logits = act_model.merge_module(gdef, p_in, rest)(input_ids=tokens).logits
            return jnp.mean(logits**2)

        return jax.value_and_grad(_loss)(p)

    loss, grads = _vg(params)
    assert np.isfinite(float(loss))
    flat_g = {str(k).removeprefix("parameters/"): v for k, v in grads.flatten().items()}
    for it in range(num_loops):
        w_key = f"model.loop_gates.{it}.proj.weight"
        b_key = f"model.loop_gates.{it}.proj.bias"
        assert w_key in flat_g and b_key in flat_g
        assert jnp.all(jnp.isfinite(flat_g[w_key])) and float(jnp.linalg.norm(flat_g[w_key])) > 0.0
        assert jnp.all(jnp.isfinite(flat_g[b_key])) and float(jnp.linalg.norm(flat_g[b_key])) > 0.0
    iter_key = "model.loop_iter_embeddings"
    assert iter_key in flat_g
    assert jnp.all(jnp.isfinite(flat_g[iter_key])) and float(jnp.linalg.norm(flat_g[iter_key])) > 0.0


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize(
    "kv_mode,overlap", [("separate", True), ("hybrid_sliding_window", True), ("hybrid_sliding_window", False)]
)
def test_generic_loop_scan_layers_and_remat_parity(family, kv_mode, overlap):
    """Verify scan_layers=True + auto_remat matches unrolled forward & grads on Qwen3 and Llama."""
    tokens = jnp.asarray([[4, 17, 29, 8, 53, 12, 91, 6, 33, 19]], dtype=jnp.int32)
    shared = dict(
        num_hidden_layers=6,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=3,
        loop_gate_type="channel",
        loop_gate_init_biases=(-0.5, -0.2, 0.8),
        loop_use_iteration_embedding=True,
        loop_kv_cache_mode=kv_mode,
        loop_kv_sliding_window=4,
        loop_kv_overlap=overlap,
    )
    u_cfg = _make_config(family, scan_layers=False, gradient_checkpointing="", **shared)
    s_cfg = _make_config(family, scan_layers=True, gradient_checkpointing="nothing_saveable", **shared)

    m_u = _make_causal_lm(family, u_cfg, seed=21)
    m_s = _make_causal_lm(family, s_cfg, seed=21)
    # Looped Llama must keep self.layers unstacked so slice-wise scan works cleanly.
    assert len(m_s.model.layers) == 6

    _, u_state, _ = m_u.split_module()
    s_gdef, _, s_rest = m_s.split_module()
    m_s = m_s.merge_module(s_gdef, u_state, s_rest)

    u_gdef, u_p, u_rest = m_u.split_module()
    s_gdef, s_p, s_rest = m_s.split_module()

    @jax.jit
    def _vg_u(p):
        def _f(p_in):
            logits = m_u.merge_module(u_gdef, p_in, u_rest)(input_ids=tokens).logits
            return jnp.mean(logits**2), logits

        return jax.value_and_grad(_f, has_aux=True)(p)

    @jax.jit
    def _vg_s(p):
        def _f(p_in):
            logits = m_s.merge_module(s_gdef, p_in, s_rest)(input_ids=tokens).logits
            return jnp.mean(logits**2), logits

        return jax.value_and_grad(_f, has_aux=True)(p)

    with jax.default_matmul_precision("highest"):
        (_, u_logits), u_grads = _vg_u(u_p)
        (_, s_logits), s_grads = _vg_s(s_p)

    np.testing.assert_allclose(np.asarray(s_logits), np.asarray(u_logits), rtol=1e-4, atol=1e-4)
    uf = u_grads.flatten()
    sf = s_grads.flatten()
    for k in uf:
        np.testing.assert_allclose(
            np.asarray(sf[k]),
            np.asarray(uf[k]),
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"{family} {kv_mode} overlap={overlap}: {k}",
        )


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("kv_mode", ["separate", "hybrid_sliding_window"])
def test_generic_loop_transformer_cache_prefill_decode_parity(family, kv_mode):
    """Verify TransformerCache prefill + single-token decode matches cacheless forward on Qwen3 and Llama."""
    window = 4
    tokens = jnp.asarray([[3, 14, 27, 8, 55, 19, 42, 6, 71, 22]], dtype=jnp.int32)
    cfg = _make_config(
        family,
        num_hidden_layers=6,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=2,
        loop_gate_type="channel",
        loop_gate_init_biases=(-0.4, 0.8),
        loop_use_iteration_embedding=True,
        loop_kv_cache_mode=kv_mode,
        loop_kv_sliding_window=window,
        loop_kv_overlap=True,
    )
    model = _make_causal_lm(family, cfg, seed=31)
    cache = model.init_cache(batch_size=1, max_length=16)
    assert len(cache.views) == 8
    if kv_mode == "hybrid_sliding_window":
        # Pass-2 core slots (4, 5) must allocate a fixed W-token ring buffer instead of max_length=16.
        for slot in (4, 5):
            assert cache.views[slot].key.shape[1] == window
        for slot in (0, 1, 2, 3, 6, 7):
            assert cache.views[slot].key.shape[1] == 16

    with jax.default_matmul_precision("highest"):
        ref_logits = np.asarray(spx.jit(lambda m, ids: m(input_ids=ids).logits)(model, tokens))

        # Prefill 6 tokens (> window=4), then decode tokens 6..9 one by one.
        steps = [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        for s_start, s_end in steps:
            pos = jnp.arange(s_start, s_end, dtype=jnp.int32)[None, :]
            out = spx.jit(lambda m, ids, p, c: m(input_ids=ids, position_ids=p, past_key_values=c))(
                model, tokens[:, s_start:s_end], pos, cache
            )
            cache = out.past_key_values
            np.testing.assert_allclose(
                np.asarray(out.logits, dtype=np.float32),
                ref_logits[:, s_start:s_end],
                rtol=2e-3,
                atol=2e-3,
                err_msg=f"{family} {kv_mode} cache mismatch at [{s_start}:{s_end})",
            )


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("overlap", [False, True])
def test_generic_loop_ragged_pages_dual_stream_parity(family, overlap):
    """Verify eSurge RaggedPagesCache dual-stream prefill + decode with page eviction on Qwen3 and Llama."""
    from easydel.caching import (
        HybridCache,
        RaggedPagesCache,
        RaggedPagesCacheConfig,
        RaggedPagesCacheView,
        RaggedPagesMetadata,
    )

    window = 4
    page_size = 4
    tokens = jnp.asarray([[3, 14, 27, 8, 55, 19, 42, 6, 71, 22]], dtype=jnp.int32)
    total_len = int(tokens.shape[1])

    base_kwargs = dict(
        num_hidden_layers=6,
        head_dim=128,
        loop_start_layer=2,
        loop_end_layer=4,
        num_loop_iterations=2,
        loop_gate_type="channel",
        loop_gate_init_biases=(-0.4, 0.8),
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=window,
        loop_kv_overlap=overlap,
    )
    ref_cfg = _make_config(family, attn_mechanism="vanilla", **base_kwargs)
    paged_cfg = _make_config(family, attn_mechanism="ragged_page_attention_v3", **base_kwargs)

    ref_model = _make_causal_lm(family, ref_cfg, seed=41)
    paged_model = _make_causal_lm(family, paged_cfg, seed=41)
    _, ref_state, _ = ref_model.split_module()
    p_gdef, p_state, p_other = paged_model.split_module()
    paged_model = paged_model.merge_module(p_gdef, p_state.overlay(ref_state), p_other)

    ragged_cfg = RaggedPagesCacheConfig(
        num_hidden_layers=paged_cfg.num_cache_layers,
        max_model_length=32,
        num_kv_heads=paged_cfg.num_key_value_heads,
        logical_num_kv_heads=paged_cfg.num_key_value_heads,
        k_headdim=paged_cfg.head_dim,
        v_headdim=paged_cfg.head_dim,
        page_size=page_size,
        num_pages=8,
        max_num_pages_per_req=8,
        version="v3",
        _kvdtype_str="fp32",
    )
    cache = paged_model.init_operations_cache(
        batch_size=1,
        max_length=32,
        page_size=page_size,
        dtype=jnp.float32,
        ragged_config=ragged_cfg,
    )
    assert isinstance(cache, (HybridCache, RaggedPagesCache))
    assert len(cache.views) == 8
    for view in cache.views:
        assert isinstance(view, RaggedPagesCacheView)

    # Group 0 owns FULL slots (0, 1, 2, 3, 6, 7); Group 1 owns Pass-2 SLIDING slots (4, 5).
    layer_to_group = (0, 0, 0, 0, 1, 1, 0, 0)

    with jax.default_matmul_precision("highest"):
        ref_logits = np.asarray(spx.jit(lambda m, ids: m(input_ids=ids).logits)(ref_model, tokens))

        steps = [(0, 6), (6, 7), (7, 8), (8, 9), (9, total_len)]
        for s_start, s_end in steps:
            q_len = s_end - s_start
            g_bt = jnp.asarray([[1, 2, 3, 0]], dtype=jnp.int32)
            if s_start >= 8:
                l_len = s_end - page_size
                l_bt = jnp.asarray([[5, 6, 0, 0]], dtype=jnp.int32)
            else:
                l_len = s_end
                l_bt = jnp.asarray([[4, 5, 6, 0]], dtype=jnp.int32)

            qsl = jnp.asarray([0, q_len], dtype=jnp.int32)
            dist = jnp.asarray([1, 1, 1] if q_len == 1 else [0, 0, 1], dtype=jnp.int32)
            pos_1d = jnp.arange(s_start, s_end, dtype=jnp.int32)

            g_meta = RaggedPagesMetadata(
                pages_tables=g_bt,
                context_lens=jnp.asarray([s_end], dtype=jnp.int32),
                query_start_loc=qsl,
                position_ids=pos_1d,
                num_seqs=jnp.asarray([1], dtype=jnp.int32),
                request_distribution=dist,
                page_size=page_size,
                version="v3",
            )
            l_meta = RaggedPagesMetadata(
                pages_tables=l_bt,
                context_lens=jnp.asarray([l_len], dtype=jnp.int32),
                query_start_loc=qsl,
                position_ids=pos_1d,
                num_seqs=jnp.asarray([1], dtype=jnp.int32),
                request_distribution=dist,
                page_size=page_size,
                version="v3",
            )
            grouped_meta = g_meta.replace(cache_groups=(g_meta, l_meta), layer_to_cache_group=layer_to_group)

            out = spx.jit(
                lambda m, ids, pos, c, cm: m(
                    input_ids=ids,
                    position_ids=pos,
                    past_key_values=c,
                    cache_metadata=cm,
                )
            )(paged_model, tokens[:, s_start:s_end], pos_1d[None, :], cache, grouped_meta)
            cache = out.past_key_values
            np.testing.assert_allclose(
                np.asarray(out.logits, dtype=np.float32),
                ref_logits[:, s_start:s_end],
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"{family} paged dual-stream mismatch at [{s_start}:{s_end}) overlap={overlap}",
            )


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_generic_zero_conversion_checkpoint_roundtrip_and_telemetry(family, tmp_path):
    """Save unlooped Qwen3/Llama -> load looped via from_pretrained (zero conversion) -> verify parity & knockout."""
    ckpt_dir = tmp_path / f"{family}_base"
    base_cfg = _make_config(family, num_hidden_layers=6)
    base_model = _make_causal_lm(family, base_cfg, seed=51)
    base_model.save_pretrained(str(ckpt_dir))

    cls = ed.Qwen3ForCausalLM if family == "qwen3" else ed.LlamaForCausalLM
    looped = cls.from_pretrained(
        str(ckpt_dir),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        mismatch_allowed=False,
        config_kwargs={
            "attn_mechanism": "vanilla",
            "attn_dtype": jnp.float32,
            "kvdtype": jnp.float32,
            "loop_start_layer": 2,
            "loop_end_layer": 4,
            "num_loop_iterations": 3,
            "loop_gate_type": "channel",
            "loop_gate_init_biases": [-20.0, -20.0, 20.0],
            "loop_use_iteration_embedding": True,
            "loop_kv_cache_mode": "hybrid_sliding_window",
            "loop_kv_sliding_window": 4,
            "loop_kv_overlap": False,
        },
    )
    assert looped.config.is_looped is True
    assert looped.config.num_loop_iterations == 3
    assert looped.model.loop_gates is not None and len(looped.model.loop_gates) == 3

    tokens = jnp.asarray([[1, 8, 22, 47, 11, 63, 9, 31]], dtype=jnp.int32)
    asst_mask = jnp.asarray([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=jnp.bool_)
    with jax.default_matmul_precision("highest"):
        base_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(base_model, tokens)
        loop_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(looped, tokens)
    np.testing.assert_allclose(np.asarray(loop_logits), np.asarray(base_logits), rtol=1e-4, atol=1e-4)

    # Generic compute_loop_gate_stats + knockout works on Qwen3 & Llama!
    stats = compute_loop_gate_stats(looped, tokens, assistant_mask=asst_mask, include_knockout=True)
    assert len(stats["per_pass"]) == 3
    for p_entry in stats["per_pass"]:
        assert p_entry["loss_corr"] is not None and -1.0001 <= p_entry["loss_corr"] <= 1.0001
    ko = stats["knockout"]
    assert abs(ko["gate_zero"]["all_delta_nll"]) < 1e-3
    assert abs(ko["single_pass"]["all_delta_nll"]) < 1e-3
    assert abs(ko["gate_zero"]["asst_delta_nll"]) < 1e-3
    assert abs(ko["single_pass"]["asst_delta_nll"]) < 1e-3


@pytest.mark.parametrize(
    "family,dual_stream,extra_kwargs",
    [
        ("mistral", True, {"sliding_window": None}),
        (
            "qwen3_moe",
            True,
            {
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 64,
                "decoder_sparse_step": 1,
                "sliding_window": None,
            },
        ),
        (
            "deepseek_v3",
            False,
            {
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "moe_intermediate_size": 64,
                "first_k_dense_replace": 1,
                "n_group": 1,
                "topk_group": 1,
                "q_lora_rank": 32,
                "kv_lora_rank": 32,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 16,
                "v_head_dim": 16,
            },
        ),
        ("gemma3_text", True, {"sliding_window": 32, "query_pre_attn_scalar": 16}),
        (
            "gpt2",
            True,
            {
                "n_embd": 64,
                "n_layer": 4,
                "n_head": 4,
                "n_positions": 64,
                "n_ctx": 64,
                "attn_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
            },
        ),
        ("gpt_neox", True, {"rotary_pct": 0.5}),
        (
            "mamba2",
            False,
            {
                "intermediate_size": 64,
                "num_heads": 4,
                "head_dim": 16,
                "state_size": 16,
                "conv_kernel": 4,
                "n_groups": 1,
                "chunk_size": 16,
                "expand": 1,
            },
        ),
    ],
)
def test_diverse_model_families_loop_parity_and_grads(family, dual_stream, extra_kwargs):
    """Verify looped step-0 identity parity and active gradients across diverse EasyDeL archetypes."""
    import inspect

    from easydel.infra.factory import TaskType

    cfg_cls, model_cls = ed.get_modules_by_type(family, TaskType.CAUSAL_LM)
    base_kw = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 64,
        "attn_mechanism": "vanilla",
        "attn_dtype": jnp.float32,
        "kvdtype": jnp.float32,
        "scan_layers": False,
        "sharding_axis_dims": (1, 1, 1, 1, 1, 1),
        "moe_method": "standard_moe",
        **extra_kwargs,
    }
    sig = inspect.signature(cfg_cls.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _mk(**loop_kw):
        raw = {**base_kw, **loop_kw}
        if not has_var_kw:
            raw = {k: v for k, v in raw.items() if k in sig.parameters}
        return cfg_cls(**raw)

    base_m = model_cls(
        _mk(), dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    loop_m = model_cls(
        _mk(
            loop_start_layer=1,
            loop_end_layer=3,
            num_loop_iterations=2,
            loop_gate_init_biases=(-20.0, 20.0),
            loop_kv_cache_mode="hybrid_sliding_window" if dual_stream else "separate",
            loop_kv_sliding_window=4,
            loop_kv_overlap=False,
        ),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    _, base_state, _ = base_m.split_module()
    l_gdef, l_state, l_other = loop_m.split_module()
    loop_m = loop_m.merge_module(l_gdef, l_state.overlay(base_state), l_other)
    base_m.eval()
    loop_m.eval()

    tokens = jnp.asarray([[1, 5, 19, 42, 7, 13, 27, 3]], dtype=jnp.int32)
    with jax.default_matmul_precision("highest"):
        base_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(base_m, tokens)
        loop_logits = spx.jit(lambda m, ids: m(input_ids=ids).logits)(loop_m, tokens)
    assert float(jnp.max(jnp.abs(base_logits - loop_logits))) < 1e-3

    act_m = model_cls(
        _mk(
            loop_start_layer=1,
            loop_end_layer=3,
            num_loop_iterations=2,
            loop_gate_type="channel",
            loop_gate_init_biases=(-1.0, 1.0),
            loop_use_iteration_embedding=True,
            loop_kv_cache_mode="hybrid_sliding_window" if dual_stream else "separate",
            loop_kv_sliding_window=4,
            loop_kv_overlap=True,
        ),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(3),
    )
    act_m.eval()
    gdef, params, rest = act_m.split_module()

    @jax.jit
    def _vg(p):
        def _loss(p_in):
            return jnp.mean(act_m.merge_module(gdef, p_in, rest)(input_ids=tokens).logits ** 2)

        return jax.value_and_grad(_loss)(p)

    loss, grads = _vg(params)
    assert np.isfinite(float(loss))
    flat_g = {str(k): v for k, v in grads.flatten().items()}
    gate_keys = [k for k in flat_g if "loop_gates.0.proj.weight" in k or "loop_gates.1.proj.weight" in k]
    assert len(gate_keys) == 2
    for k in gate_keys:
        assert float(jnp.linalg.norm(flat_g[k])) > 0.0


def test_dual_stream_guards_attention_sinks_and_logit_softcap():
    """Verify non-paged hybrid_sliding_window rejects softmax_aux (gpt_oss) and logits_soft_cap (gemma2) loudly."""
    tokens = jnp.asarray([[1, 5, 19, 42]], dtype=jnp.int32)

    gpt_oss_cfg = ed.GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention"] * 4,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        moe_method="standard_moe",
        loop_start_layer=1,
        loop_end_layer=3,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
    )
    gpt_oss = ed.GptOssForCausalLM(gpt_oss_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    with pytest.raises(NotImplementedError, match="softmax_aux"):
        gpt_oss(input_ids=tokens)

    gemma2_cfg = ed.Gemma2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=32,
        query_pre_attn_scalar=16,
        attn_logit_softcapping=50.0,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        loop_start_layer=1,
        loop_end_layer=3,
        num_loop_iterations=2,
        loop_kv_cache_mode="hybrid_sliding_window",
        loop_kv_sliding_window=4,
    )
    gemma2 = ed.Gemma2ForCausalLM(gemma2_cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        gemma2(input_ids=tokens)
