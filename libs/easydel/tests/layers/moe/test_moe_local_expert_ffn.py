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

"""Expert tensor mode: fsdp-sharded expert weights and the experts-over-tp local expert FFN.

* In expert tensor mode (experts on tp), ``moe_fsdp_shard_expert_weights`` shards the expert weights' second
  (feature) dim over fsdp at rest and all-gathers that dim inside the fused MoE. Outputs and gradients match the
  replicated layout, and gradients stay fsdp-sharded.
* ``moe_local_expert_ffn`` runs each tp shard's experts through ejkernel's ``moe_local_ffn`` and reduce-scatters
  the partial sums over tp. Outputs and every gradient (input, router, experts) match expert tensor mode's regular
  dispatch on the same weights and routing; in bfloat16 at kernel-sized shapes the Pallas TPU kernels run inside the
  model.
* ``moe_local_expert_ffn`` is inert (bit-identical) outside expert tensor mode.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

F32, BF16 = jnp.float32, jnp.bfloat16
DIMS = (1, 1, 2, 1, 2, 1)  # fsdp2 x tp2: experts over tp, tokens over fsdp


def _dims():
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices for fsdp2 x tp2")
    return DIMS


def _config(*, hidden, inter, experts, top_k, expert_tensor, local=False, fsdp_shard=False, force_xla=True, **extra):
    config = ed.Glm5NextTextConfig(
        vocab_size=128, hidden_size=hidden, intermediate_size=64, moe_intermediate_size=inter, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, n_routed_experts=experts, num_experts_per_tok=top_k,
        n_shared_experts=0, kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=0, qk_nope_head_dim=8, v_head_dim=8,
        index_topk=8, index_head_dim=8, index_n_heads=2, index_kpool=4, layer_types=["linear_attention"],
        mlp_layer_types=["sparse"], linear_head_dim=8, linear_num_heads=2, linear_conv_kernel_dim=4, hc_mult=2,
        sharding_axis_dims=_dims(),
    )  # fmt: skip
    config.add_basic_configurations(
        use_expert_tensor_mode=expert_tensor,
        fsdp_is_ep_bound=False,
        sp_is_ep_bound=False,
        moe_force_xla_gmm=force_xla,
        moe_local_expert_ffn=local,
        moe_fsdp_shard_expert_weights=fsdp_shard,
        **extra,
    )
    return config


def _block(config, dtype, seed=0):
    from easydel.infra.base_module import _parameter_init_sharding_context
    from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    precision = jax.lax.Precision.HIGHEST if dtype == F32 else None
    with _parameter_init_sharding_context(config):
        return Glm5NextTextMoE(config=config, dtype=dtype, param_dtype=dtype, precision=precision, rngs=spx.Rngs(seed))


def _inputs(shape, dtype, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return jax.random.normal(k1, shape, F32).astype(dtype), jax.random.normal(k2, shape, F32)


def _run(block, x, ct):
    """Output, input gradient and every parameter gradient (path -> array) of ``sum(out * ct)``."""
    gdef, state = spx.export(block)

    def loss(xin, st):
        out, _ = spx.bind(gdef, st)(xin)
        return jnp.sum(out.astype(F32) * ct), out

    with block.config.mesh, jax.default_matmul_precision("highest" if x.dtype == F32 else "default"):
        (_, out), (gx, gstate) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True, allow_int=True))(
            x, state
        )
    grads = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(gstate)[0]
        if jnp.issubdtype(leaf.dtype, jnp.inexact)
    }
    return out, gx, grads


def _rel(a, b):
    a = np.asarray(jnp.asarray(a, F32), np.float64)
    b = np.asarray(jnp.asarray(b, F32), np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _dim_axes(array, dim):
    spec = getattr(array.sharding, "spec", None)
    if spec is None or len(spec) <= dim or spec[dim] is None:
        return ()
    entry = spec[dim]
    return tuple(entry) if isinstance(entry, tuple) else (entry,)


def _expert_leaf(grads, name):
    (key,) = [k for k in grads if name in k and "weight" in k]
    return grads[key]


def _assert_same(got, want, *, out_tol, grad_tol):
    (out_g, gx_g, grads_g), (out_w, gx_w, grads_w) = got, want
    assert out_g.shape == out_w.shape
    assert np.isfinite(np.asarray(jnp.asarray(out_g, F32))).all()
    assert _rel(out_g, out_w) < out_tol, ("output", _rel(out_g, out_w))
    assert _rel(gx_g, gx_w) < grad_tol, ("d input", _rel(gx_g, gx_w))
    assert grads_g.keys() == grads_w.keys()
    for key in grads_w:
        assert np.isfinite(np.asarray(jnp.asarray(grads_g[key], F32))).all(), key
        if float(jnp.linalg.norm(jnp.asarray(grads_w[key], F32))) == 0.0:
            assert float(jnp.linalg.norm(jnp.asarray(grads_g[key], F32))) == 0.0, key
            continue
        assert _rel(grads_g[key], grads_w[key]) < grad_tol, (key, _rel(grads_g[key], grads_w[key]))


SMALL = dict(hidden=32, inter=16, experts=4, top_k=2)


def test_expert_tensor_fsdp_weights_are_sharded_and_match_replicated():
    on = _block(_config(expert_tensor=True, fsdp_shard=True, **SMALL), F32)
    off = _block(_config(expert_tensor=True, fsdp_shard=False, **SMALL), F32)
    for name in ("gate_proj", "up_proj", "down_proj"):
        w_on = getattr(on.experts, name).weight.value
        w_off = getattr(off.experts, name).weight.value
        assert _dim_axes(w_on, 0) == ("tp",) and _dim_axes(w_on, 1) == ("fsdp",), (name, w_on.sharding)
        assert _dim_axes(w_off, 0) == ("tp",) and _dim_axes(w_off, 1) == (), (name, w_off.sharding)
    x, ct = _inputs((4, 16, SMALL["hidden"]), F32)
    got, want = _run(on, x, ct), _run(off, x, ct)
    _assert_same(got, want, out_tol=1e-5, grad_tol=1e-5)
    for name in ("gate_proj", "up_proj", "down_proj"):
        g = _expert_leaf(got[2], name)
        assert "fsdp" in _dim_axes(g, 1), f"{name} grad must stay fsdp-sharded on dim 1, got {g.sharding}"


@pytest.mark.parametrize("fsdp_shard", [False, True])
def test_local_expert_ffn_matches_regular_dispatch_fp32(fsdp_shard):
    # 64 tokens per shard in 32-token chunks and 16-row windows: several chunks and windows per call.
    ref = _block(_config(expert_tensor=True, fsdp_shard=fsdp_shard, **SMALL), F32)
    loc = _block(
        _config(
            expert_tensor=True,
            local=True,
            fsdp_shard=fsdp_shard,
            moe_chunk_size=32,
            moe_local_ffn_window_rows=16,
            **SMALL,
        ),
        F32,
    )
    x, ct = _inputs((4, 32, SMALL["hidden"]), F32, seed=1)
    got, want = _run(loc, x, ct), _run(ref, x, ct)
    _assert_same(got, want, out_tol=1e-5, grad_tol=1e-5)
    # The local path reduce-scatters its partial sums over tp; the regular expert-tensor dispatch returns full rows.
    assert _dim_axes(got[0], 2) == ("tp",), got[0].sharding
    assert _dim_axes(want[0], 2) == (), want[0].sharding


def test_local_expert_ffn_runs_the_pallas_kernels_in_bf16():
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernels")
    big = dict(hidden=2048, inter=256, experts=8, top_k=2)
    ref = _block(_config(expert_tensor=True, fsdp_shard=True, **big), BF16)
    loc = _block(
        _config(expert_tensor=True, local=True, fsdp_shard=True, force_xla=False, moe_local_ffn_window_rows=1024, **big),
        BF16,
    )
    x, ct = _inputs((2, 1024, big["hidden"]), BF16, seed=2)
    with loc.config.mesh:
        hlo = jax.jit(lambda xin: loc(xin)[0]).lower(x).compile().as_text()
    assert "moe_row_gather_sum" in hlo, "the Pallas gather-sum kernel must run inside the model"
    got, want = _run(loc, x, ct), _run(ref, x, ct)
    _assert_same(got, want, out_tol=2e-2, grad_tol=3e-2)


def test_dispatch_knobs_that_do_not_apply_on_the_local_path_change_nothing():
    """``moe_row_gather_kernel`` / ``moe_defer_tp_reduce`` have nothing to act on in the local FFN: same results."""
    x, ct = _inputs((4, 16, SMALL["hidden"]), F32, seed=4)
    plain = _run(_block(_config(expert_tensor=True, local=True, moe_chunk_size=32, **SMALL), F32), x, ct)
    knobs = _run(
        _block(
            _config(
                expert_tensor=True, local=True, moe_chunk_size=32, moe_row_gather_kernel=True,
                moe_defer_tp_reduce=True, **SMALL,
            ),
            F32,
        ),
        x,
        ct,
    )  # fmt: skip
    np.testing.assert_array_equal(np.asarray(knobs[0]), np.asarray(plain[0]))
    np.testing.assert_array_equal(np.asarray(knobs[1]), np.asarray(plain[1]))
    for key in plain[2]:
        np.testing.assert_array_equal(np.asarray(knobs[2][key]), np.asarray(plain[2][key]))


@pytest.mark.cpu_ok
@pytest.mark.parametrize("bad", [0, -3, 1.5, True, "16"])
def test_window_rows_is_validated_by_both_config_entry_points(bad):
    with pytest.raises((TypeError, ValueError), match="moe_local_ffn_window_rows"):
        ed.LlamaConfig(hidden_size=64, num_attention_heads=4, moe_local_ffn_window_rows=bad)
    config = ed.LlamaConfig(hidden_size=64, num_attention_heads=4)
    with pytest.raises((TypeError, ValueError), match="moe_local_ffn_window_rows"):
        config.add_basic_configurations(moe_local_ffn_window_rows=bad)
    config.add_basic_configurations(moe_local_ffn_window_rows=2048)
    assert config.moe_local_ffn_window_rows == 2048


def test_local_expert_ffn_is_inert_outside_expert_tensor_mode():
    x, ct = _inputs((4, 16, SMALL["hidden"]), F32, seed=3)
    on = _run(_block(_config(expert_tensor=False, local=True, **SMALL), F32), x, ct)
    off = _run(_block(_config(expert_tensor=False, local=False, **SMALL), F32), x, ct)
    np.testing.assert_array_equal(np.asarray(on[0]), np.asarray(off[0]))
    np.testing.assert_array_equal(np.asarray(on[1]), np.asarray(off[1]))
    for key in off[2]:
        np.testing.assert_array_equal(np.asarray(on[2][key]), np.asarray(off[2][key]))
