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

"""Accelerator tests for single-mesh ``dp`` SPMD lowering, deferred ``dp`` reduction, and fused MoE mHC write VJP."""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.sharding import (
    active_dp_spmd_axis,
    dp_spmd_vmap_scope,
    strip_axis_from_spec,
    strip_compound_axis_from_spec,
)
from easydel.layers.indexer._primitives import segment_pool_layout
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextHyperConnection, Glm5NextTextMoE
from easydel.trainers.dp_sharding import (
    can_use_dp_spmd_vmap,
    dp_spmd_step_gradients,
    reduce_dp_weighted,
    resolve_graphstate_specs,
)
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as Ps


@pytest.mark.cpu_ok
def test_dp_spmd_scope_and_spec_stripping():
    assert active_dp_spmd_axis() is None
    with dp_spmd_vmap_scope("dp"):
        assert active_dp_spmd_axis() == "dp"
        assert strip_compound_axis_from_spec(Ps(("dp", "fsdp"), None), "dp") == Ps("fsdp", None)
        assert strip_compound_axis_from_spec(Ps(("fsdp", "dp"), "tp"), "dp") == Ps("fsdp", "tp")
        assert strip_compound_axis_from_spec(Ps("dp", None), "dp") == Ps("dp", None)
        assert strip_axis_from_spec(Ps("dp", None), "dp") == Ps(None, None)
        assert strip_axis_from_spec(Ps(("dp", "fsdp"), None, "tp"), "dp") == Ps("fsdp", None, "tp")
    assert active_dp_spmd_axis() is None


def test_segment_pool_gather_custom_vjp():
    bsz, seq_len, kpool, dim = 2, 64, 4, 16
    seg = jnp.asarray(
        [
            [0] * 20 + [1] * 28 + [-1] * 16,
            [0] * 13 + [1] * 19 + [2] * 32,
        ],
        dtype=jnp.int32,
    )
    layout = segment_pool_layout(seg, kpool, max_documents=8)
    x = jax.random.normal(jax.random.key(0), (bsz, seq_len, dim), dtype=jnp.float32)
    w = jax.random.normal(jax.random.key(1), (bsz, layout.order.shape[1], dim), dtype=jnp.float32)

    def loss_custom(inp):
        return jnp.sum(layout.gather(inp) * w)

    def loss_ref(inp):
        idx = jnp.maximum(layout.order, 0)
        gathered = jnp.take_along_axis(inp, idx[..., None], axis=1)
        gathered = jnp.where((layout.order >= 0)[..., None], gathered, 0.0)
        return jnp.sum(gathered * w)

    v_c, g_c = jax.value_and_grad(loss_custom)(x)
    v_r, g_r = jax.value_and_grad(loss_ref)(x)
    assert float(jnp.max(jnp.abs(v_c - v_r))) < 1e-5
    assert float(jnp.max(jnp.abs(g_c - g_r))) < 1e-5


def _small_glm5_cfg(sharding_axis_dims: tuple[int, int, int, int, int, int]) -> ed.Glm5NextTextConfig:
    return ed.Glm5NextTextConfig(
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=512,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_group=4,
        topk_group=2,
        num_hidden_layers=2,
        first_k_dense_replace=0,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["sparse", "sparse"],
        indexer_types=["full", "full"],
        num_attention_heads=16,
        num_key_value_heads=16,
        kv_lora_rank=256,
        q_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        v_head_dim=128,
        index_n_heads=8,
        index_head_dim=128,
        index_kpool=4,
        index_topk=64,
        linear_attn_config={
            "kda_layers": [0],
            "full_attn_layers": [1],
            "num_heads": 16,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "d_conv": 4,
            "expand_k": 1.0,
            "expand_v": 1.0,
        },
        vocab_size=1024,
        eos_token_id=[1],
        pad_token_id=0,
        sharding_axis_dims=sharding_axis_dims,
        scan_layers=True,
        kda_chunk_impl="kda_chunk",
        dsa_pool_training=True,
        dsa_pool_kernel="gather",
        fsdp_is_ep_bound=False,
        moe_fsdp_shard_expert_weights=True,
        moe_chunk_size=128,
        moe_defer_tp_reduce=True,
        gradient_checkpointing="nothing_saveable",
        mask_max_position_embeddings=256,
        freq_max_position_embeddings=256,
        precompute_masks=False,
    )


def test_glm5_next_fused_post_write_remat_dce_and_grad_parity():
    if jax.local_device_count() < 4:
        pytest.skip("Requires at least 4 accelerator devices")

    cfg = _small_glm5_cfg((1, 1, 2, 1, 2, 1))
    jm = cfg.mesh.jax_mesh
    streams = jax.device_put(
        jax.random.normal(jax.random.key(2), (2, 256, cfg.hc_mult, cfg.hidden_size), dtype=jnp.float32),
        NamedSharding(jm, Ps("fsdp", None, None, "tp")),
    )

    class _Block(ed.EasyDeLBaseModule):
        def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.float32, precision=None, *, rngs):
            super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            self.moe = Glm5NextTextMoE(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            self.hc = Glm5NextHyperConnection(
                config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            )

        def __call__(self, hidden_streams, *, fuse: bool):
            @jax.checkpoint
            def _rematted(hs):
                post, comb, collapsed = self.hc(hs)
                gate = self.hc.fused_post_write_gate(hs, post, comb) if fuse else None
                ffn_out, _ = self.moe(collapsed, post_write_gate=gate)
                return self.hc.residual_write(hs, ffn_out, post, comb)

            return _rematted(hidden_streams)

    with cfg.mesh:
        blk = _Block(cfg, rngs=spx.Rngs(3)).shard_model()
        blk_state = blk.to_state()

    def make_step(fuse: bool):
        @jax.jit
        def _step(st, hs):
            def _loss(gs, x):
                m = st.merge(gs)
                return jnp.mean(m(x, fuse=fuse) ** 2)

            return jax.value_and_grad(_loss, argnums=(0, 1))(st.graphstate, hs)

        return _step

    with cfg.mesh:
        f_unfused = make_step(False)
        f_fused = make_step(True)
        hlo_unfused = f_unfused.lower(blk_state, streams).as_text("hlo")
        hlo_fused = f_fused.lower(blk_state, streams).as_text("hlo")
        v_u, (g_gs_u, g_hs_u) = f_unfused(blk_state, streams)
        v_f, (g_gs_f, g_hs_f) = f_fused(blk_state, streams)

    assert hlo_fused.count(" while(") < hlo_unfused.count(" while(")
    assert float(jnp.abs(v_u - v_f)) < 1e-5
    assert float(jnp.max(jnp.abs(g_hs_u - g_hs_f))) < 1e-5
    for gu, gf in zip(jax.tree.leaves(g_gs_u), jax.tree.leaves(g_gs_f), strict=True):
        assert float(jnp.max(jnp.abs(gu - gf))) < 1e-4


def test_qwen4_exp_fused_post_write_remat_dce_and_grad_parity():
    if jax.local_device_count() < 4:
        pytest.skip("Requires at least 4 accelerator devices")

    from easydel.layers.residual import GatedResidual, inject_streams
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpSparseMoeBlock
    from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig

    cfg = Qwen4ExpTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        hc_count=4,
        hc_lowrank=32,
        sharding_axis_dims=(1, 1, 2, 1, 2, 1),
        moe_method="fused_moe",
        moe_chunk_size=128,
        moe_defer_tp_reduce=True,
        fsdp_is_ep_bound=False,
    )
    jm = cfg.mesh.jax_mesh
    streams = jax.device_put(
        jax.random.normal(jax.random.key(7), (2, 256, cfg.hc_count * cfg.hidden_size), dtype=jnp.float32),
        NamedSharding(jm, Ps("fsdp", None, "tp")),
    )

    class _Block(ed.EasyDeLBaseModule):
        def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.float32, precision=None, *, rngs):
            super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            self.moe = Qwen4ExpSparseMoeBlock(
                config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
            )
            self.hc = GatedResidual(
                hidden_size=config.hidden_size,
                hc_count=config.hc_count,
                hc_lowrank=config.hc_lowrank,
                eps=config.rms_norm_eps,
                use_combine=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        def __call__(self, hidden_streams, *, fuse: bool):
            @jax.checkpoint
            def _rematted(hs):
                mixed, hyper_input, injection = self.hc(hs)
                gate = self.hc.fused_post_write_gate(hyper_input, injection, mesh=self.config.mesh) if fuse else None
                ffn_out, _ = self.moe(mixed, post_write_gate=gate)
                return inject_streams(hyper_input, ffn_out, injection)

            return _rematted(hidden_streams)

    with cfg.mesh:
        blk = _Block(cfg, rngs=spx.Rngs(8)).shard_model()
        blk_state = blk.to_state()

    def make_step(fuse: bool):
        @jax.jit
        def _step(st, hs):
            def _loss(gs, x):
                m = st.merge(gs)
                return jnp.mean(m(x, fuse=fuse) ** 2)

            return jax.value_and_grad(_loss, argnums=(0, 1))(st.graphstate, hs)

        return _step

    with cfg.mesh:
        f_unfused = make_step(False)
        f_fused = make_step(True)
        hlo_unfused = f_unfused.lower(blk_state, streams).as_text("hlo")
        hlo_fused = f_fused.lower(blk_state, streams).as_text("hlo")
        v_u, (g_gs_u, g_hs_u) = f_unfused(blk_state, streams)
        v_f, (g_gs_f, g_hs_f) = f_fused(blk_state, streams)

    assert hlo_fused.count(" while(") < hlo_unfused.count(" while(")
    assert float(jnp.abs(v_u - v_f)) < 1e-5
    assert float(jnp.max(jnp.abs(g_hs_u - g_hs_f))) < 1e-5
    for gu, gf in zip(jax.tree.leaves(g_gs_u), jax.tree.leaves(g_gs_f), strict=True):
        assert float(jnp.max(jnp.abs(gu - gf))) < 1e-4


@pytest.mark.parametrize("variant", ["v4_moe", "v4_hash", "v41_vl"])
def test_deepseek_v4_and_v41_fused_post_write_remat_dce_and_grad_parity(variant):
    if jax.local_device_count() < 4:
        pytest.skip("Requires at least 4 accelerator devices")

    from easydel.layers.residual import ManifoldHyperConnection, ManifoldHyperConnectionConfig, manifold_residual_write
    from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
    from easydel.modules.deepseek_v41.deepseek_v41_configuration import DeepseekV41Config
    from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DeepseekV41SparseMoeBlock

    if variant in ("v4_moe", "v4_hash"):
        cfg = DeepseekV4Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            hc_mult=4,
            hc_sinkhorn_iters=4,
            mlp_layer_types=["hash_moe" if variant == "v4_hash" else "moe"],
            sharding_axis_dims=(1, 1, 2, 1, 2, 1),
            moe_method="fused_moe",
            moe_chunk_size=128,
            moe_defer_tp_reduce=True,
            fsdp_is_ep_bound=False,
        )
    else:
        cfg = DeepseekV41Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            dspark_target_layer_ids=[],
            candidate_source_layer_id=-1,
            num_attention_heads=4,
            num_key_value_heads=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            hc_mult=4,
            hc_sinkhorn_iters=4,
            sharding_axis_dims=(1, 1, 2, 1, 2, 1),
            moe_method="fused_moe",
            moe_chunk_size=128,
            moe_defer_tp_reduce=True,
            fsdp_is_ep_bound=False,
        )

    jm = cfg.mesh.jax_mesh
    streams = jax.device_put(
        jax.random.normal(jax.random.key(4), (2, 256, cfg.hc_mult, cfg.hidden_size), dtype=jnp.float32),
        NamedSharding(jm, Ps("fsdp", None, None, "tp")),
    )
    input_ids = (
        jax.device_put(
            jax.random.randint(jax.random.key(6), (2, 256), 0, cfg.vocab_size, dtype=jnp.int32),
            NamedSharding(jm, Ps("fsdp", None)),
        )
        if variant == "v4_hash"
        else None
    )
    image_mask = (
        jax.device_put(
            (jax.numpy.arange(256)[None, :] % 4 == 0).repeat(2, axis=0),
            NamedSharding(jm, Ps("fsdp", None)),
        )
        if variant == "v41_vl"
        else None
    )

    class _Block(ed.EasyDeLBaseModule):
        def __init__(self, config, dtype=jnp.float32, param_dtype=jnp.float32, precision=None, *, rngs):
            super().__init__(config=config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs)
            if variant in ("v4_moe", "v4_hash"):
                self.moe = DeepseekV4SparseMoeBlock(
                    config, layer_idx=0, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
                )
            else:
                self.moe = DeepseekV41SparseMoeBlock(
                    config,
                    layer_idx=0,
                    image_router_bias=True,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
            self.hc = ManifoldHyperConnection(
                ManifoldHyperConnectionConfig.from_model_config(config),
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                mesh_source=config,
            )

        def __call__(self, hidden_streams, *, fuse: bool):
            @jax.checkpoint
            def _rematted(hs):
                post, comb, collapsed = self.hc(hs)
                gate = self.hc.fused_post_write_gate(hs, post, comb) if fuse else None
                if image_mask is not None:
                    ffn_out, _ = self.moe(collapsed, image_mask=image_mask, post_write_gate=gate)
                else:
                    ffn_out, _ = self.moe(collapsed, input_ids=input_ids, post_write_gate=gate)
                return manifold_residual_write(hs, ffn_out, post, comb, precision=self.precision)

            return _rematted(hidden_streams)

    with cfg.mesh:
        blk = _Block(cfg, rngs=spx.Rngs(5)).shard_model()
        blk_state = blk.to_state()

    def make_step(fuse: bool):
        @jax.jit
        def _step(st, hs):
            def _loss(gs, x):
                m = st.merge(gs)
                return jnp.mean(m(x, fuse=fuse) ** 2)

            return jax.value_and_grad(_loss, argnums=(0, 1))(st.graphstate, hs)

        return _step

    with cfg.mesh:
        f_unfused = make_step(False)
        f_fused = make_step(True)
        hlo_unfused = f_unfused.lower(blk_state, streams).as_text("hlo")
        hlo_fused = f_fused.lower(blk_state, streams).as_text("hlo")
        v_u, (g_gs_u, g_hs_u) = f_unfused(blk_state, streams)
        v_f, (g_gs_f, g_hs_f) = f_fused(blk_state, streams)

    assert hlo_fused.count(" while(") < hlo_unfused.count(" while(")
    assert float(jnp.abs(v_u - v_f)) < 1e-5
    assert float(jnp.max(jnp.abs(g_hs_u - g_hs_f))) < 1e-5
    for gu, gf in zip(jax.tree.leaves(g_gs_u), jax.tree.leaves(g_gs_f), strict=True):
        assert float(jnp.max(jnp.abs(gu - gf))) < 1e-4


def test_glm5_next_dp_spmd_step_gradients_and_deferred_reduce():
    if jax.local_device_count() < 4:
        pytest.skip("Requires at least 4 accelerator devices")

    cfg = _small_glm5_cfg((1, 2, 2, 1, 1, 1))
    with cfg.mesh:
        model = ed.Glm5NextTextModel(cfg, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0)).shard_model()
        state = model.to_state()

    jm = cfg.mesh.jax_mesh
    ps_batch = Ps(("dp", "fsdp"))
    assert can_use_dp_spmd_vmap(state=state, batch_size=4, minibatch_size=4, partition_spec=ps_batch)

    batch = {
        "input_ids": jax.device_put(
            jax.random.randint(jax.random.key(0), (4, 256), 0, 1024, dtype=jnp.int32),
            NamedSharding(jm, Ps(("dp", "fsdp"), None)),
        ),
        "segment_ids": jax.device_put(
            jnp.where(jnp.arange(256)[None, :] < 128, 0, 1).repeat(4, axis=0).astype(jnp.int32),
            NamedSharding(jm, Ps(("dp", "fsdp"), None)),
        ),
        "position_ids": jax.device_put(
            (jnp.arange(256)[None, :] % 128).repeat(4, axis=0).astype(jnp.int32),
            NamedSharding(jm, Ps(("dp", "fsdp"), None)),
        ),
        "mass": jax.device_put(
            jnp.asarray([100.0, 100.0, 300.0, 300.0], dtype=jnp.float32),
            NamedSharding(jm, Ps(("dp", "fsdp"))),
        ),
    }

    def slice_grad_fn(s_st, _e_st, b_slice):
        w = jnp.sum(b_slice["mass"])

        def loss_fn(gs):
            m = s_st.replace(graphstate=gs).model
            out = m(
                input_ids=b_slice["input_ids"],
                segment_ids=b_slice["segment_ids"],
                position_ids=b_slice["position_ids"],
            )
            loss = jnp.mean(out.last_hidden_state.astype(jnp.float32) ** 2)
            return loss, LossMetrics(loss=loss, weight_sum=w)

        (_, met), grads = jax.value_and_grad(loss_fn, has_aux=True)(s_st.graphstate)
        return grads, met, w

    gs_specs = resolve_graphstate_specs(state, cfg.mesh)

    with cfg.mesh:
        grads_inline, met_inline, tw_inline = jax.jit(
            lambda s, b: dp_spmd_step_gradients(
                student_state=s,
                batch=b,
                batch_size=4,
                minibatch_size=4,
                partition_spec=ps_batch,
                slice_grad_and_weight_fn=slice_grad_fn,
                defer_dp_reduce=False,
            )
        )(state, batch)

        grads_dp, met_dp, w_dp = jax.jit(
            lambda s, b: dp_spmd_step_gradients(
                student_state=s,
                batch=b,
                batch_size=4,
                minibatch_size=4,
                partition_spec=ps_batch,
                slice_grad_and_weight_fn=slice_grad_fn,
                defer_dp_reduce=True,
            )
        )(state, batch)

        grads_def, met_def, tw_def = jax.jit(
            lambda g, m, w: reduce_dp_weighted(g, m, w, gs_specs, jm, dp_axis="dp", normalize_grads=True)
        )(grads_dp, met_dp, w_dp)

    assert float(jnp.abs(tw_inline - 800.0)) == 0.0
    assert float(jnp.abs(tw_def - 800.0)) == 0.0
    assert float(jnp.abs(met_inline.loss - met_def.loss)) < 1e-6
    for a, b in zip(jax.tree.leaves(grads_inline), jax.tree.leaves(grads_def), strict=True):
        assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) < 1e-5


def test_can_use_dp_spmd_vmap_rejects_pp_and_mpmd_meshes():
    if jax.local_device_count() < 4:
        pytest.skip("Requires at least 4 accelerator devices")

    class _DummyModel:
        def __init__(self, mesh):
            self.mesh = mesh
            self.config = type("_Cfg", (), {"mesh": mesh})()

    class _DummyState:
        def __init__(self, mesh):
            self.model = _DummyModel(mesh)

    # pp > 1 MPMD mesh (pp=2, dp=2) must never enter dp_spmd_vmap
    mpmd_pp2 = spx.create_mesh((2, 2, 1, 1, 1, 1), ("pp", "dp", "fsdp", "ep", "tp", "sp"), mpmd_axis="pp")
    assert not can_use_dp_spmd_vmap(
        state=_DummyState(mpmd_pp2), batch_size=4, minibatch_size=4, partition_spec=Ps("dp")
    )

    # Degenerate pp == 1 with mpmd_axis="pp" is non-MPMD in SpectraX (is_mpmd=False), so pure SPMD applies
    mpmd_pp1 = spx.create_mesh((1, 2, 2, 1, 1, 1), ("pp", "dp", "fsdp", "ep", "tp", "sp"), mpmd_axis="pp")
    assert not mpmd_pp1.is_mpmd
    assert can_use_dp_spmd_vmap(
        state=_DummyState(mpmd_pp1), batch_size=4, minibatch_size=4, partition_spec=Ps(("dp", "fsdp"))
    )

    # Custom MPMD axis name ("stage" > 1) must also be rejected via is_mpmd_mesh
    mpmd_custom = spx.create_mesh((2, 2), ("stage", "dp"), mpmd_axis="stage")
    assert not can_use_dp_spmd_vmap(
        state=_DummyState(mpmd_custom), batch_size=4, minibatch_size=4, partition_spec=Ps("dp")
    )


def test_all_shared_expert_and_hyper_moe_models_follow_modular_contract():
    """Verify zero modularity gaps across all MoE models with shared experts or hyper-connections."""
    import ast
    from pathlib import Path

    modules_root = Path(ed.__file__).resolve().parent / "modules"
    missing_shared: list[str] = []
    missing_post_write: list[str] = []

    for path in sorted(modules_root.glob("**/modeling_*.py")):
        src = path.read_text()
        if "BaseMoeModule" not in src:
            continue
        tree = ast.parse(src, filename=str(path))
        rel = str(path.relative_to(modules_root))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [ast.unparse(b) for b in node.bases]
            if not any("BaseMoeModule" in b or "SparseMoeBlock" in b for b in bases):
                continue
            cls_src = ast.get_source_segment(src, node) or ""
            if "self.moe_call(" not in cls_src:
                continue
            has_shared_attr = "self.shared_expert" in cls_src
            if has_shared_attr and "shared_expert=" not in cls_src:
                missing_shared.append(f"{rel}:{node.name}")
        if ("ManifoldHyperConnection" in src or "GatedResidual" in src) and "self.moe_call(" in src:
            if "fused_post_write_gate(" not in src or "post_write_gate=" not in src:
                missing_post_write.append(rel)

    assert not missing_shared, f"BaseMoeModule classes with shared experts not passed via shared_expert=: {missing_shared}"
    assert not missing_post_write, f"Hyper-connection MoE modules missing fused_post_write_gate wiring: {missing_post_write}"
