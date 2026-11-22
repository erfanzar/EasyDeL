"""Prefix-shared GRPO forward on a real tensor-parallel mesh: parity with the repeated-prompt forward and sharding."""

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.caching import TransformerCache
from easydel.caching.shared_prefix import PrefixCaptureView
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.group_relative_policy_optimization._fn import (
    get_per_token_logps,
    get_per_token_logps_prefix_shared,
    grpo_step,
)
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import common_types

if jax.device_count() < 4:
    pytest.skip("needs a mesh of at least 4 devices", allow_module_level=True)

PROMPTS, PROMPT_LEN, COMPLETION_LEN, LAYERS = 2, 6, 5, 2
MESHES = {"tp4": (1, 1, 1, 1, 4, 1), "fsdp2_tp2": (1, 1, 2, 1, 2, 1)}
MECHANISMS = ["vanilla", "sdpa"]


def _config(attn_mechanism, axis_dims):
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        mlp_bias=False,
        sharding_axis_dims=axis_dims,
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
    )


def _model(cfg):
    with cfg.mesh:
        model = ed.LlamaForCausalLM(
            config=cfg,
            rngs=spx.Rngs(0),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        return model.shard_model()


def _batch(num_generations):
    rng = np.random.default_rng(num_generations)
    prompt_mask = np.ones((PROMPTS, PROMPT_LEN), np.int32)
    prompt_mask[0, :2] = 0  # left-padded ragged prompt
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 120, (PROMPTS, PROMPT_LEN)), 0).astype(np.int32)
    rows = PROMPTS * num_generations
    lengths = rng.integers(1, COMPLETION_LEN + 1, rows)
    completion_mask = (np.arange(COMPLETION_LEN)[None, :] < lengths[:, None]).astype(np.int32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 120, (rows, COMPLETION_LEN)), 0).astype(np.int32)
    advantages = rng.standard_normal(rows).astype(np.float32)
    return prompt_ids, prompt_mask, completion_ids, completion_mask, advantages


def _assert_tp_sharded_weights(model):
    spec = model.get_lm_head().weight.value.sharding.spec
    assert "tp" in str(spec), spec


@pytest.mark.parametrize("attn_mechanism", MECHANISMS)
@pytest.mark.parametrize("num_generations", [1, 4])
@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_shared_per_token_logps_match_repeated_forward_on_tp_mesh(mesh_name, num_generations, attn_mechanism):
    cfg = _config(attn_mechanism, MESHES[mesh_name])
    model = _model(cfg)
    _assert_tp_sharded_weights(model)
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = _batch(num_generations)
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
        reference = get_per_token_logps(m, jnp.asarray(input_ids), jnp.asarray(attention_mask), PROMPT_LEN)
        return shared, reference

    with jax.default_matmul_precision("highest"), cfg.mesh:
        shared, reference = both(state)
    valid = completion_mask > 0
    error = float(np.max(np.abs(np.asarray(shared)[valid] - np.asarray(reference)[valid])))
    print(f"MEASURE grpo-logps {mesh_name} {attn_mechanism} G{num_generations}: {error:.3e} spec {shared.sharding.spec}")
    assert shared.shape == reference.shape == (PROMPTS * num_generations, COMPLETION_LEN)
    # Measured max 4.8e-7 on tp4 (G=4), 0 elsewhere; ~4x headroom.
    assert error <= 2e-6, error


@pytest.mark.parametrize("attn_mechanism", MECHANISMS)
@pytest.mark.parametrize("num_generations", [1, 4])
@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_grpo_step_loss_and_update_match_repeated_forward_on_tp_mesh(mesh_name, num_generations, attn_mechanism):
    cfg = _config(attn_mechanism, MESHES[mesh_name])
    prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = _batch(num_generations)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(advantages),
    }
    model = _model(cfg)
    with jax.default_matmul_precision("highest"), cfg.mesh:
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        out = {}
        for shared in (False, True):
            new_state, metrics = grpo_step(
                state,
                batch,
                num_generations,
                0.0,
                partition_spec=PartitionSpec(),
                is_training=True,
                loss_type="dapo",
                prefix_shared_forward=shared,
            )
            leaves = jax.tree_util.tree_leaves(new_state.graphstate)
            out[shared] = (float(metrics.loss), [b - np.asarray(a) for b, a in zip(before, leaves, strict=True)])
            if shared:
                specs = [x.sharding.spec for x in leaves]
    (loss_r, upd_r), (loss_s, upd_s) = out[False], out[True]
    upd_err = max(float(np.max(np.abs(s - r))) for s, r in zip(upd_s, upd_r, strict=True))
    print(
        f"MEASURE grpo-step {mesh_name} {attn_mechanism} G{num_generations}: loss {abs(loss_s - loss_r):.3e} "
        f"upd {upd_err:.3e} max|upd| {max(float(np.max(np.abs(r))) for r in upd_r):.3e}"
    )
    # The updated parameters stay tensor-parallel sharded (no silent replication through the shared path).
    assert any("tp" in str(spec) for spec in specs), specs
    assert any(np.max(np.abs(r)) > 0 for r in upd_r)
    # Measured: loss diff 0, update diff <= 6.0e-8 (max |update| ~0.18-0.29); ~4x headroom.
    np.testing.assert_allclose(loss_s, loss_r, rtol=1e-6, atol=1e-7)
    for s, r in zip(upd_s, upd_r, strict=True):
        np.testing.assert_allclose(s, r, rtol=0, atol=2.5e-7)


@pytest.mark.parametrize("attn_mechanism", MECHANISMS)
@pytest.mark.parametrize("mesh_name", list(MESHES))
def test_captured_prefix_kv_stays_head_sharded(mesh_name, attn_mechanism):
    """The captured prefix K/V keep their kv-head dim on ``tp`` inside a jitted run (not fully replicated)."""
    cfg = _config(attn_mechanism, MESHES[mesh_name])
    model = _model(cfg)
    prompt_ids, prompt_mask, *_ = _batch(2)
    gdef, state = spx.export(model)

    @jax.jit
    def capture(st):
        out = spx.bind(gdef, st)(
            input_ids=jnp.asarray(prompt_ids),
            attention_mask=jnp.asarray(prompt_mask),
            past_key_values=TransformerCache(views=[PrefixCaptureView() for _ in range(LAYERS)]),
            mode=common_types.MODE_TRAIN,
            apply_lm_head=False,
        )
        return [(view.key, view.value) for view in out.past_key_values.views]

    with jax.default_matmul_precision("highest"), cfg.mesh:
        views = capture(state)
    tp = dict(zip(cfg.mesh.axis_names, cfg.mesh.devices.shape, strict=True))["tp"]
    for key, value in views:
        print(f"MEASURE grpo-kv {mesh_name} {attn_mechanism}: key {key.sharding.spec} value {value.sharding.spec}")
        assert key.shape == (PROMPTS, PROMPT_LEN, 4, 16)
        for arr in (key, value):
            shard_shape = arr.sharding.shard_shape(arr.shape)
            assert shard_shape[2] == arr.shape[2] // tp, (arr.sharding, shard_shape)
