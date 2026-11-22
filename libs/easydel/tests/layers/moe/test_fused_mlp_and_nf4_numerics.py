"""Fused MLP plan selection and NF4 numerics against independent references; accelerator only."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.linears import ColumnParallelLinear, ColumnParallelMoELinear, RowParallelMoELinear
from easydel.layers.linears._utils import nf4_qmm_jax
from easydel.layers.mlp import GatedMLP
from easydel.layers.moe import record_expert_load
from easydel.modules.arctic import ArcticConfig
from easydel.modules.arctic.modeling_arctic import ArcticMoeBlock
from jax import numpy as jnp


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_fused_enabled_mlp_honors_projection_training_quantization(projection):
    # Replicated projections and >=512 local rows admit the dense fused plan.
    config = ArcticConfig(
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        sharding_axis_dims=(1, jax.device_count(), 1, 1, 1, 1),
        sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    config.use_fused_mlp = True
    config.gradient_checkpointing = "nothing_saveable"
    mlp = GatedMLP(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(1))
    rng = np.random.default_rng(22)
    for proj in [mlp.gate_up_proj, mlp.down_proj]:
        proj.weight.value = jnp.asarray(rng.normal(size=proj.weight.value.shape) / 2, dtype=jnp.bfloat16)
    spx.quantization.quantize_model(
        mlp,
        spx.quantization.QuantProvider(
            [
                spx.quantization.QuantRule(
                    module_path=projection,
                    op_names=("dot_general",),
                    weight_qtype="int8",
                    weight_calibration_method="fixed,127.5",
                )
            ]
        ),
    )
    # Fixed symmetric range gives scale=1: independent integer rounding, not
    # another call through the production rule resolver or linear forward.
    wf = np.asarray(mlp.gate_up_proj.weight.value).astype(np.float32)
    wd = np.asarray(mlp.down_proj.weight.value).astype(np.float32)
    if projection == "gate_up_proj":
        wf = np.rint(wf)
    else:
        wd = np.rint(wd)
    wf, wd = jnp.asarray(wf, jnp.bfloat16), jnp.asarray(wd, jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(jax.device_count(), 512, 128)) / 4, dtype=jnp.bfloat16)

    def reference(z):
        gu = jnp.einsum("bth,hm->btm", z, wf, precision=jax.lax.Precision.HIGHEST)
        gate, up = jnp.split(gu, 2, axis=-1)
        return jnp.einsum("btm,mh->bth", jax.nn.silu(gate) * up, wd, precision=jax.lax.Precision.HIGHEST)

    with config.mesh, jax.default_matmul_precision("highest"):
        actual, expected = mlp(x), reference(x)
        actual_grad = jax.grad(lambda z: mlp(z).astype(jnp.float32).sum())(x)
        expected_grad = jax.grad(lambda z: reference(z).astype(jnp.float32).sum())(x)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), rtol=0.02, atol=0.02)
    np.testing.assert_allclose(actual_grad.astype(jnp.float32), expected_grad.astype(jnp.float32), rtol=0.02, atol=0.04)


@pytest.mark.parametrize("linear_class", [ColumnParallelMoELinear, RowParallelMoELinear])
def test_biased_ragged_projection_accepts_dynamic_group_sizes_under_jit(linear_class):
    layer = linear_class(2, 4, 8, dtype=jnp.float32, rngs=spx.Rngs(2))
    layer.bias.value = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 8
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 12
    run = jax.jit(lambda a, groups: layer(a, groups))
    for groups in [(1, 2), (0, 3), (3, 0)]:
        expected = np.concatenate(
            [
                np.asarray(rows) @ np.asarray(layer.weight.value[e]) + np.asarray(layer.bias.value[e])
                for e, rows in enumerate(np.split(np.asarray(x), [groups[0]]))
            ],
            axis=0,
        )
        np.testing.assert_allclose(run(x, jnp.asarray(groups, jnp.int32)), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("blocks,packed_width", [(1, 3), (2, 32), (5, 7)])
def test_standalone_nf4_matmul_handles_block_units_and_partial_tiles(blocks, packed_width):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    packed = rng.integers(0, 256, size=(7, blocks, packed_width), dtype=np.uint8)
    scales = rng.uniform(0.2, 1.0, size=(7, blocks)).astype(np.float32)
    # Independent high-/low-nibble unpacking and the documented NF4 polynomial.
    codes = np.empty((7, blocks, 2 * packed_width), dtype=np.float32)
    codes[..., 0::2] = packed // 16
    codes[..., 1::2] = packed % 16
    values = np.polyval(
        [
            1.82943132356953e-5,
            -0.00068587779130373,
            0.0100420261313669,
            -0.0722703570217226,
            0.346075459755188,
            -0.994166218659335,
        ],
        codes,
    )
    expected = x @ (values * scales[..., None]).reshape(7, -1)
    actual = nf4_qmm_jax(
        jnp.asarray(x), jnp.asarray(packed), jnp.asarray(scales), BK=3, BM=2, BNQL=2, compute_dtype=jnp.float32
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def _block(method="fused_moe", ep_size=1, ring=False):
    devices = jax.device_count()
    config = ArcticConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_layer_frequency=1,
        max_position_embeddings=16,
        moe_method=method,
        sharding_axis_dims=(1, 1, devices // ep_size, ep_size, 1, 1),
        sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
    )
    config.add_basic_configurations(use_ring_of_experts=ring, fsdp_is_ep_bound=False, sp_is_ep_bound=False)
    config.moe_force_xla_gmm = True
    block = ArcticMoeBlock(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(3),
        layer_idx=0,
    )
    return config, block


@pytest.mark.parametrize("method", ["standard_moe", "dense_moe", "fused_moe"])
@pytest.mark.parametrize("input_scaling", [False, True])
def test_moe_methods_preserve_distinct_router_stream_and_input_scaling(method, input_scaling):
    config, block = _block(method)
    gate = ColumnParallelLinear(
        12, 4, use_bias=False, dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(9)
    )
    rng = np.random.default_rng(10)
    gate.weight.value = jnp.asarray(rng.normal(size=(12, 4)).astype(np.float32) / 3)
    wi = rng.normal(size=(4, 8, 16)).astype(np.float32) / 4
    wu = rng.normal(size=(4, 8, 16)).astype(np.float32) / 4
    wd = rng.normal(size=(4, 16, 8)).astype(np.float32) / 4
    block.experts.gate_up_proj.weight.value = jnp.asarray(np.concatenate([wi, wu], axis=-1))
    block.experts.w2.weight.value = jnp.asarray(wd)
    x = rng.normal(size=(2 * jax.device_count(), 2, 8)).astype(np.float32)
    router = rng.normal(size=(*x.shape[:2], 12)).astype(np.float32)
    hooks = block.moe_hooks.replace(
        before_gate=lambda z: z + 0.25,
        refine_weights_hook=lambda w: w / w.sum(axis=-1, keepdims=True),
    )
    if input_scaling:
        hooks = hooks.replace(
            scale_replicated_inputs=lambda z, w: z * w[:, None],
            output_weights_hook=jnp.ones_like,
        )
    with config.mesh, jax.default_matmul_precision("highest"):
        actual, logits = block.moe_call(
            hidden_state=jnp.asarray(x),
            gate_hidden_state=jnp.asarray(router),
            gate_layer=gate,
            expert_layer=block.experts,
            gate_up_kernel=block.experts.gate_up_proj.kernel_view(),
            wd_kernel=block.experts.w2.kernel_view(),
            hooks=hooks,
            act_fn=jax.nn.silu,
        )
    expected_logits = (router.reshape(-1, 12) + 0.25) @ np.asarray(gate.weight.value)
    probs = np.exp(expected_logits - expected_logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    indices = np.argsort(-probs, axis=-1)[:, :2]
    weights = np.take_along_axis(probs, indices, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = np.zeros_like(x.reshape(-1, 8))
    for token, row in enumerate(x.reshape(-1, 8)):
        for slot, expert in enumerate(indices[token]):
            w = weights[token, slot]
            expert_input = row * w if input_scaling else row
            a, b = expert_input @ wi[expert], expert_input @ wu[expert]
            expert_output = ((a / (1 + np.exp(-a))) * b) @ wd[expert]
            expected[token] += expert_output if input_scaling else expert_output * w
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual, expected.reshape(x.shape), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("ring", [False, True])
def test_fused_multilayer_recording_keeps_layer_ids_and_sums_real_calls(ring):
    config, block = _block(ep_size=2, ring=ring)
    block.layer_idx = 3
    second = ArcticMoeBlock(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(4), layer_idx=7)
    x = jnp.ones((2 * jax.device_count(), 2, 8), dtype=jnp.float32)
    with config.mesh, record_expert_load(ep_size=2) as recorder:
        for _ in range(2):
            for model_layer in [block, second]:
                model_layer(x)
        jax.effects_barrier()
    assert set(recorder.stack()) == {3, 7}
    assert recorder.summary()["n_layers"] == 2
    exported = recorder.to_arrays()["logical_count"]
    assert exported.shape == (1, 2, 4)
    for layer, counts in zip([3, 7], exported[0], strict=True):
        assert counts.sum() == 2 * x.shape[0] * x.shape[1] * 2
        np.testing.assert_array_equal(counts, recorder.stack()[layer])
