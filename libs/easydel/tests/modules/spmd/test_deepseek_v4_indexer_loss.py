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

"""CSA indexer warm-up: dense-attention targets, isolated gradients and task-head plumbing.

Numerical tests require an accelerator. The default keeps the existing LM-loss
STE; enabling warm-up replaces that STE with the detached-input KL objective.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.layers.rotary import labeled_rope_cos_sin
from easydel.modules.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

COEF = 0.5
SEQ = 16


def _make_config(coef=COEF, layer_types=None):
    layers = layer_types or [
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    config = DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=len(layers),
        num_attention_heads=2,
        head_dim=16,
        q_lora_rank=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        layer_types=layers,
        mlp_layer_types=["moe"] * len(layers),
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        max_position_embeddings=64,
        sliding_window=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        o_groups=1,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        partial_rotary_factor=0.25,
        initializer_range=0.12,
        indexer_aux_loss_coef=coef,
        # Single-device mesh, as in the existing DeepSeek-V4 numeric tests: these
        # checks cover loss math, and batch-1 inputs cannot shard the MoE dispatch
        # over a multi-device fsdp axis.
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        fsdp_is_ep_bound=False,
        dspark_target_layer_ids=[0],
    )
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _build(coef=COEF, layer_types=None):
    config = _make_config(coef, layer_types)
    with config.mesh:
        model = DeepseekV4ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(7),
        )
        model.eval()
    return model


def _ids():
    return jnp.asarray(np.random.default_rng(3).integers(0, 64, size=(1, SEQ)), dtype=jnp.int32)


def _arrays(state):
    return {
        path: np.asarray(leaf.value if hasattr(leaf, "value") else leaf)
        for collection, path, leaf in state.items()
        if collection == "parameters"
    }


def _aux_fn(state, ids):
    def loss(params):
        return state.merge(params)(input_ids=ids, apply_lm_head=False).indexer_aux_loss

    return loss


def test_config_default_roundtrip_and_nonnegative_validation():
    """The opt-in coefficient survives config serialization; negative/NaN values are rejected."""
    default = DeepseekV4Config()
    assert default.indexer_aux_loss_coef == 0.0
    config = DeepseekV4Config(indexer_aux_loss_coef=0.25)
    assert DeepseekV4Config.from_dict(config.to_dict()).indexer_aux_loss_coef == 0.25
    for invalid in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="indexer_aux_loss_coef must be >= 0"):
            DeepseekV4Config(indexer_aux_loss_coef=invalid)


def test_indexer_params_get_finite_nonzero_gradients_and_nothing_else_does():
    model, ids = _build(), _ids()
    with model.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        loss, grads = jax.value_and_grad(_aux_fn(state, ids))(state.graphstate)
        # Warm-up disables the indexer's main-LM STE, not only its input gradient.
        lm_grads = jax.grad(lambda p: jnp.mean(state.merge(p)(input_ids=ids).logits ** 2))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    named, main = _arrays(grads), _arrays(lm_grads)
    expected = set(_arrays(state.graphstate))
    assert set(named) == expected
    indexer_names = {n for n in expected if ".indexer." in n}
    assert indexer_names
    for name, grad in named.items():
        assert np.isfinite(grad).all(), name
        if name in indexer_names:
            assert np.any(grad != 0), f"all-zero indexer gradient: {name}"
            assert np.all(main[name] == 0), f"warm-up retained an LM gradient: {name}"
        else:
            assert np.all(grad == 0), f"auxiliary gradient leaked to {name}"
    assert any(np.any(g != 0) for n, g in main.items() if n not in indexer_names)


def test_aux_loss_decreases_updating_only_indexer_params():
    model, ids = _build(), _ids()
    with model.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        before = _arrays(state.graphstate)

        @jax.jit
        def step(state):
            loss, grads = jax.value_and_grad(_aux_fn(state, ids))(state.graphstate)
            return loss, state.apply_gradients(grads=grads)

        losses = []
        for _ in range(10):
            loss, state = step(state)
            losses.append(float(loss))
    assert np.isfinite(losses).all(), losses
    assert losses[-1] < 0.99 * losses[0], losses
    after = _arrays(state.graphstate)
    changed = {n for n in before if not np.array_equal(before[n], after[n])}
    assert changed
    assert all(".indexer." in n for n in changed), sorted(changed)


def test_coef_zero_logits_are_identical_and_aux_scales_in_both_head_paths():
    off, on, ids = _build(0.0), _build(), _ids()
    for hooked in (False, True):
        with off.mesh:
            baseline = off(input_ids=ids, output_dspark_hidden=hooked)
        with on.mesh:
            warmup = on(input_ids=ids, output_dspark_hidden=hooked)
        assert baseline.indexer_aux_loss is None and baseline.aux_loss is None
        assert np.isfinite(float(warmup.indexer_aux_loss)) and float(warmup.indexer_aux_loss) > 0.0
        np.testing.assert_array_equal(np.asarray(baseline.logits), np.asarray(warmup.logits))
        np.testing.assert_allclose(float(warmup.aux_loss), COEF * float(warmup.indexer_aux_loss), rtol=1e-6)
        if hooked:
            np.testing.assert_array_equal(np.asarray(baseline.main_hidden), np.asarray(warmup.main_hidden))
    # Coefficient zero retains the pre-existing LM-loss STE training signal.
    with off.mesh:
        state = EasyDeLState.create(model=off)
        grads = jax.grad(lambda p: jnp.mean(state.merge(p)(input_ids=ids).logits ** 2))(state.graphstate)
    assert any(np.any(g != 0) for n, g in _arrays(grads).items() if ".indexer." in n)


def _numpy_kl(q, kv, entries, scores, sinks, scaling, window, valid, positions, rate):
    """Independent dense fp64 attention reference, over ALL causal entries (not top-k)."""
    q, kv, entries, scores, sinks = (np.asarray(x, dtype=np.float64) for x in (q, kv, entries, scores, sinks))
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[1]
    losses = []
    for b in range(batch):
        for s in range(seq):
            reach = min(num_entries, (int(positions[b, s]) + 1) // rate)
            if reach == 0:
                continue
            keys = np.arange(seq)
            keys = keys[(keys <= s) & (keys > s - window) & valid[b]]
            logits = np.concatenate(
                [q[b, :, s] @ kv[b, keys].T * scaling, q[b, :, s] @ entries[b, :reach].T * scaling, sinks[:, None]],
                axis=-1,
            )
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            target = probs[:, len(keys) : len(keys) + reach].sum(axis=0)
            target /= target.sum()
            pred = scores[b, s, :reach]
            log_pred = pred - np.log(np.exp(pred - pred.max()).sum()) - pred.max()
            losses.append(np.sum(target * (np.log(target) - log_pred)))
    assert heads > 1 and num_entries > 2  # The head sum and a non-top-k target must be exercised.
    return np.mean(losses)


def test_aux_matches_dense_attention_reference_and_sums_only_csa_layers():
    model, ids = _build(), _ids()
    positions = jnp.arange(SEQ, dtype=jnp.int32)[None]
    with model.mesh:
        # Different sink masses per head make omitting the token window or sinks observable in the head sum.
        for layer in model.model.layers:
            layer.self_attn.sinks.value = jnp.asarray([-1.0, 3.0], dtype=jnp.float32)
        output = model(input_ids=ids, output_hidden_states=True)
        expected = 0.0
        for i, layer in enumerate(model.model.layers):
            if layer.attention_type != "compressed_sparse_attention":
                continue
            _, _, collapsed = layer.attn_hc(output.hidden_states[i])
            hidden = layer.input_layernorm(collapsed)
            attn = layer.self_attn
            cos, sin = labeled_rope_cos_sin(model.config, attn.rope_label, positions, hidden.dtype)
            qres, q = attn.project_query(hidden, cos, sin)
            kv = attn.project_kv(hidden, cos, sin)
            entries, _ = attn.compressor(hidden, qres, positions)
            _, raw_scores = attn.compressor.indexer(hidden, qres, positions, return_scores=True)
            expected += _numpy_kl(
                q,
                kv,
                entries,
                raw_scores,
                attn.sinks.value,
                attn.scaling,
                attn.sliding_window,
                np.ones((1, SEQ), dtype=bool),
                np.asarray(positions),
                attn.compressor.compress_rate,
            )
    np.testing.assert_allclose(float(output.indexer_aux_loss), expected, rtol=2e-4, atol=2e-6)


def test_attention_target_respects_token_padding_and_full_causal_score_domain():
    model = _build(layer_types=["compressed_sparse_attention"])
    positions = jnp.arange(SEQ, dtype=jnp.int32)[None]
    hidden = jnp.asarray(np.random.default_rng(15).normal(size=(1, SEQ, 32)), dtype=jnp.float32)
    valid = np.ones((1, SEQ), dtype=bool)
    valid[:, 2::3] = False
    qidx, kidx = np.arange(SEQ)[:, None], np.arange(SEQ)[None]
    allowed = (kidx <= qidx) & (kidx > qidx - model.config.sliding_window)
    bias = jnp.asarray(np.where(allowed[None, None] & valid[:, None, None], 0.0, np.finfo(np.float32).min))
    with model.mesh:
        attn = model.model.layers[0].self_attn
        attn.sinks.value = jnp.asarray([-1.0, 3.0], dtype=jnp.float32)
        cos, sin = labeled_rope_cos_sin(model.config, "compress", positions, hidden.dtype)
        embeddings = {"compress": (cos, sin)}
        _, _, aux = attn(hidden, embeddings, positions, bias, return_indexer_aux_loss=True)
        qres, q = attn.project_query(hidden, cos, sin)
        kv = attn.project_kv(hidden, cos, sin)
        entries, _, scores = attn.compressor(hidden, qres, positions, return_scores=True)
        expected = _numpy_kl(
            q,
            kv,
            entries,
            scores,
            attn.sinks.value,
            attn.scaling,
            attn.sliding_window,
            valid,
            np.asarray(positions),
            attn.compressor.compress_rate,
        )
    reach = (np.asarray(positions) + 1) // attn.compressor.compress_rate
    reachable = np.arange(entries.shape[1])[None, None] < reach[..., None]
    np.testing.assert_array_equal(np.isfinite(np.asarray(scores)), reachable)
    assert entries.shape[1] > model.config.index_topk
    np.testing.assert_allclose(float(aux), expected, rtol=2e-4, atol=2e-6)


def test_aux_absent_without_csa_entries_and_on_cached_prefill_decode():
    model, ids = _build(), _ids()
    with model.mesh:
        short = model(input_ids=ids[:, :1])
        assert short.indexer_aux_loss is None and short.aux_loss is None
        cache = model.init_cache(batch_size=1, max_length=32)
        prefill = model(input_ids=ids[:, :8], past_key_values=cache)
        assert prefill.indexer_aux_loss is None and prefill.aux_loss is None
        decode = model(input_ids=ids[:, 8:9], past_key_values=prefill.past_key_values)
        assert decode.indexer_aux_loss is None and decode.aux_loss is None
    for layer_type in ("heavily_compressed_attention", "sliding_attention"):
        no_indexer = _build(layer_types=[layer_type])
        with no_indexer.mesh:
            output = no_indexer(input_ids=ids)
        assert output.indexer_aux_loss is None and output.aux_loss is None
