"""Prefix-shared GRPO on hybrid qwen3_next: TP meshes, bf16, projection variants and the default Pallas kernel.

Measured on TPU v5p (4 chips); every tolerance is about 4x the measured maximum.
"""

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.caching import TransformerCache
from easydel.caching.shared_prefix import PrefixCaptureView, PrefixRecurrentCaptureView
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.group_relative_policy_optimization._fn import (
    get_per_token_logps,
    get_per_token_logps_prefix_shared,
    grpo_step,
)
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import common_types

PROMPTS, D_CONV = 2, 4
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention"]
MESHES = {"tp2": (1, 1, 1, 1, 2, 1), "tp4": (1, 1, 1, 1, 4, 1)}
_XLA_GDR = {"gated_delta_rule": GatedDeltaRuleConfig(platform="xla", backend="any", chunk_size=16)}

# fp32 with the XLA gated-delta kernel (HIGHEST matmuls): log-probs <= 4.8e-7 on tp1/tp2/tp4 and both projection
# variants; SGD updates <= 2.4e-7 (tp4), 1.3e-7 (tp2), 1.2e-7 (split projections); loss difference 0.
LOGPS_ATOL = 2e-6
UPDATE_ATOL = 1e-6
needs_4_devices = pytest.mark.skipif(jax.device_count() < 4, reason="needs a mesh of at least 4 devices")


def _config(axis_dims=(1, 1, 1, 1, 1, 1), heads=(4, 2, 2, 4), gdr="xla", dtype=jnp.float32, max_positions=64, **kw):
    return ed.Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=heads[0],
        num_key_value_heads=heads[1],
        head_dim=16,
        layer_types=list(LAYER_TYPES),
        linear_conv_kernel_dim=D_CONV,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=heads[2],
        linear_num_value_heads=heads[3],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        mlp_only_layers=list(range(len(LAYER_TYPES))),
        max_position_embeddings=max_positions,
        sharding_axis_dims=axis_dims,
        attn_mechanism="vanilla",
        attn_dtype=dtype,
        **({"operation_configs": _XLA_GDR} if gdr == "xla" else {}),
        **kw,
    )


def _model(cfg, dtype=jnp.float32, precision=lax.Precision.HIGHEST):
    with cfg.mesh:
        model = ed.Qwen3NextForCausalLM(
            config=cfg, rngs=spx.Rngs(0), dtype=dtype, param_dtype=dtype, precision=precision
        )
        return model.shard_model()


def _batch(num_generations, prompt_len=6, completion_len=5):
    rng = np.random.default_rng(num_generations)
    prompt_mask = np.ones((PROMPTS, prompt_len), np.int32)
    prompt_mask[0, :4] = 0  # ragged prompt; at prompt_len 6 it is shorter than d_conv
    prompt_mask[1, :1] = 0
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 120, (PROMPTS, prompt_len)), 0).astype(np.int32)
    rows = PROMPTS * num_generations
    lengths = rng.integers(1, completion_len + 1, rows)
    lengths[0] = completion_len
    completion_mask = (np.arange(completion_len)[None, :] < lengths[:, None]).astype(np.int32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 120, (rows, completion_len)), 0).astype(np.int32)
    advantages = rng.standard_normal(rows).astype(np.float32)
    return prompt_ids, prompt_mask, completion_ids, completion_mask, advantages


def _logps_error(model, cfg, batch, num_generations, highest=True):
    """Max |shared - repeated| over valid completion tokens, both computed in one jitted program."""
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = batch
    input_ids = np.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
    attention_mask = np.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)
    gdef, state = spx.export(model)

    @jax.jit
    def both(st):
        m = spx.bind(gdef, st)
        shared = get_per_token_logps_prefix_shared(
            m,
            jnp.asarray(prompt_ids),
            jnp.asarray(prompt_mask),
            jnp.asarray(completion_ids),
            jnp.asarray(completion_mask),
            num_generations,
        )
        reference = get_per_token_logps(m, jnp.asarray(input_ids), jnp.asarray(attention_mask), prompt_ids.shape[1])
        return shared, reference

    with jax.default_matmul_precision("highest" if highest else "default"), cfg.mesh:
        shared, reference = both(state)
    shared, reference = np.asarray(shared, np.float32), np.asarray(reference, np.float32)
    valid = completion_mask > 0
    per_position = np.where(valid, np.abs(shared - reference), 0.0).max(axis=0)
    print("MEASURE per-position", " ".join(f"{e:.1e}" for e in per_position))
    return float(np.max(np.abs(shared[valid] - reference[valid]))), reference


def _grpo_updates(model, cfg, batch, num_generations):
    prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = batch
    inputs = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(advantages),
    }
    out = {}
    with jax.default_matmul_precision("highest"), cfg.mesh:
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        for shared in (False, True):
            new_state, metrics = grpo_step(
                state,
                inputs,
                num_generations,
                0.0,
                partition_spec=PartitionSpec(),
                is_training=True,
                loss_type="dapo",
                prefix_shared_forward=shared,
            )
            leaves = jax.tree_util.tree_leaves(new_state.graphstate)
            out[shared] = (
                float(metrics.loss),
                [b - np.asarray(a) for b, a in zip(before, leaves, strict=True)],
                [getattr(x.sharding, "spec", None) for x in leaves],
            )
    return out


def _assert_updates_match(out, tag):
    (loss_r, upd_r, _), (loss_s, upd_s, specs) = out[False], out[True]
    upd_err = max(float(np.max(np.abs(s - r))) for s, r in zip(upd_s, upd_r, strict=True))
    print(f"MEASURE {tag}: loss {abs(loss_s - loss_r):.3e} upd {upd_err:.3e}")
    assert any(np.max(np.abs(r)) > 0 for r in upd_r)
    np.testing.assert_allclose(loss_s, loss_r, rtol=1e-6, atol=1e-7)
    for s, r in zip(upd_s, upd_r, strict=True):
        np.testing.assert_allclose(s, r, rtol=0, atol=UPDATE_ATOL)
    return specs


# ------------------------------------------------------------------------------------------------ tensor parallel
@needs_4_devices
@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_hybrid_shared_logps_and_captured_state_sharding_on_tp_mesh(mesh_name):
    cfg = _config(MESHES[mesh_name], heads=(8, 4, 4, 8))
    model = _model(cfg)
    for num_generations in (1, 4):
        error, _ = _logps_error(model, cfg, _batch(num_generations), num_generations)
        print(f"MEASURE hybrid-logps {mesh_name} G{num_generations}: {error:.3e}")
        assert error <= LOGPS_ATOL, error

    prompt_ids, prompt_mask, *_ = _batch(2)
    gdef, state = spx.export(model)

    @jax.jit
    def capture(st):
        views = [PrefixRecurrentCaptureView() if t == "linear_attention" else PrefixCaptureView() for t in LAYER_TYPES]
        out = spx.bind(gdef, st)(
            input_ids=jnp.asarray(prompt_ids),
            attention_mask=jnp.asarray(prompt_mask),
            past_key_values=TransformerCache(views=views),
            mode=common_types.MODE_TRAIN,
            apply_lm_head=False,
        )
        return [
            (v.conv_state, v.recurrent_state)
            for v in out.past_key_values.views
            if isinstance(v, PrefixRecurrentCaptureView)
        ]

    with jax.default_matmul_precision("highest"), cfg.mesh:
        states = capture(state)
    tp = dict(zip(cfg.mesh.axis_names, cfg.mesh.devices.shape, strict=True))["tp"]
    assert len(states) == LAYER_TYPES.count("linear_attention")
    for conv, recurrent in states:
        print(f"MEASURE hybrid-capture {mesh_name}: conv {conv.sharding.spec} recurrent {recurrent.sharding.spec}")
        # conv window [B, conv_dim, d_conv] keeps its channel dim, recurrent [B, value_heads, k, v] its head dim, on tp.
        assert conv.sharding.shard_shape(conv.shape)[1] == conv.shape[1] // tp, conv.sharding
        assert recurrent.sharding.shard_shape(recurrent.shape)[1] == recurrent.shape[1] // tp, recurrent.sharding


@needs_4_devices
@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_hybrid_grpo_step_loss_and_updates_match_on_tp_mesh(mesh_name):
    cfg = _config(MESHES[mesh_name], heads=(8, 4, 4, 8))
    specs = _assert_updates_match(_grpo_updates(_model(cfg), cfg, _batch(2), 2), f"hybrid-grpo {mesh_name}")
    # The updated parameters stay tensor-parallel sharded through the shared path (29 of 33 leaves measured).
    assert sum("tp" in str(spec) for spec in specs) > len(specs) // 2, specs


# ------------------------------------------------------------------------------------------- projection variants
@pytest.mark.parametrize("variant", ["separate_proj", "merged_split_proj"])
def test_hybrid_projection_variants_match_repeated_prompt_forward(variant):
    """``linear_attention_separate_proj`` (in_proj_qkv/z/b/a) and the merged ``in_proj_qkvz`` split layout."""
    flags = {
        "linear_attention_separate_proj": variant == "separate_proj",
        "linear_attention_merged_split_proj": variant == "merged_split_proj",
    }
    cfg = _config(**flags)
    model = _model(cfg)
    layer = model.model.layers[0].linear_attn
    assert layer.uses_split_proj == (variant == "separate_proj")
    assert layer.uses_merged_split_proj == (variant == "merged_split_proj")
    for num_generations in (1, 3):
        error, _ = _logps_error(model, cfg, _batch(num_generations), num_generations)
        print(f"MEASURE hybrid-logps {variant} G{num_generations}: {error:.3e}")
        assert error <= LOGPS_ATOL, error
    if variant == "separate_proj":
        _assert_updates_match(_grpo_updates(model, cfg, _batch(2), 2), f"hybrid-grpo {variant}")


# ------------------------------------------------------------------------------------------------------- bf16
# bf16 params and activations, XLA kernel, default matmul precision. Measured shared vs repeated (both bf16):
# 8.9e-4 (G=2) and 2.2e-3 (G=4) with |logp| up to 5.3; the same weights' bf16 vs fp32 repeated forward differ
# by 3.0e-3, so the shared path adds no more error than bf16 itself.
BF16_LOGPS_ATOL = 8e-3


@pytest.mark.parametrize("num_generations", [2, 4])
def test_hybrid_bf16_shared_logps_match_repeated_prompt_forward(num_generations):
    cfg32 = _config()
    model32 = _model(cfg32)
    cfg16 = _config(dtype=jnp.bfloat16)
    model16 = _model(cfg16, dtype=jnp.bfloat16, precision=None)
    gdef16, state16 = spx.export(model16)
    leaves16, treedef = jax.tree_util.tree_flatten(state16)
    leaves32 = jax.tree_util.tree_leaves(spx.export(model32)[1])
    # Same weights as the fp32 model, rounded to bf16, so the bf16-vs-fp32 gap below is pure precision.
    model16 = spx.bind(
        gdef16,
        jax.tree_util.tree_unflatten(treedef, [a.astype(b.dtype) for a, b in zip(leaves32, leaves16, strict=True)]),
    )
    batch = _batch(num_generations)
    error16, reference16 = _logps_error(model16, cfg16, batch, num_generations, highest=False)
    _, reference32 = _logps_error(model32, cfg32, batch, num_generations)
    valid = batch[3] > 0
    precision_gap = float(np.max(np.abs(reference16[valid] - reference32[valid])))
    print(f"MEASURE hybrid-bf16 G{num_generations}: shared-vs-repeated {error16:.3e} bf16-vs-fp32 {precision_gap:.3e}")
    assert error16 <= BF16_LOGPS_ATOL, error16


# ------------------------------------------------------------------------------------- default TPU Pallas kernel
# The default TPU gated-delta kernel runs its matmuls at DEFAULT precision (bf16 passes). The fork pass restarts the
# chunked recurrence from the injected state, so its chunks are aligned differently from the repeated forward's and
# rounding differs: kernel-level split-vs-full output gaps are 1e-4..2.2e-4 (|out| ~0.06), exactly 0 when the split
# falls on a chunk boundary, and <= 9e-8 with the kernel's matmuls forced to HIGHEST. Measured model log-prob gaps:
# 6.3e-5 (tp4) and 2.6e-4 (single device); the bound is ~4x the larger.
PALLAS_LOGPS_ATOL = 1e-3


@pytest.mark.parametrize("prompt_len,completion_len", [(6, 5), (130, 8)])
def test_hybrid_shared_logps_on_default_pallas_kernel(prompt_len, completion_len):
    if jax.default_backend() != "tpu":
        pytest.skip("the Pallas gated-delta kernel is the default only on TPU")
    cfg = _config(gdr=None, max_positions=256)
    error, _ = _logps_error(_model(cfg), cfg, _batch(2, prompt_len, completion_len), 2)
    print(f"MEASURE hybrid-logps pallas P{prompt_len} C{completion_len}: {error:.3e}")
    assert error <= PALLAS_LOGPS_ATOL, error
