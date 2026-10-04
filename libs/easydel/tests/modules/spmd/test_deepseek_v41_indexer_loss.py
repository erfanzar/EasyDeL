# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4.1 indexer auxiliary objective (``indexer_aux_loss_coef``).

``KL(stop_grad(dense head-summed attention over compressed entries) || softmax(indexer scores))`` summed over index
source layers. Checks: the loss against a float64 NumPy reference, finite non-zero gradients on every indexer parameter
(and exactly zero elsewhere), the loss decreasing under optax steps, and the default (coef 0) forward unchanged.
"""

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.layers.indexer import IndexerScoreInputs, shared_key_scores
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import indexer_distillation_loss

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

COEF = 0.5
MASK = -1e30


def _reference_kl(q, kv, entries, sinks, bias, scores, scaling) -> float:
    batch, heads, seq, _ = q.shape
    num_entries = entries.shape[1]
    total, count = 0.0, 0
    for b in range(batch):
        for s in range(seq):
            reach = np.isfinite(scores[b, s])
            if not reach.any():
                continue
            summed = np.zeros(num_entries)
            for h in range(heads):
                window = (q[b, h, s] @ kv[b].T) * scaling + bias[b, 0, s]
                entry = np.where(reach, (q[b, h, s] @ entries[b].T) * scaling, -np.inf)
                logits = np.concatenate([window, entry, [sinks[h]]])
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                summed += probs[seq : seq + num_entries]
            target = summed / summed.sum()
            row = np.where(reach, scores[b, s], -np.inf)
            log_pred = row - row.max()
            log_pred = log_pred - np.log(np.exp(log_pred).sum())
            total += sum(target[t] * (np.log(target[t]) - log_pred[t]) for t in range(num_entries) if target[t] > 0)
            count += 1
    return total / count


@pytest.mark.parametrize("block_q", [64, 4, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_distillation_loss_matches_numpy_reference(block_q, padded):
    """The query-tiled, window-banded loss equals a dense float64 reference (several tilings, padded keys)."""
    rng = np.random.default_rng(0)
    batch, heads, seq, dim, ratio, window = 2, 3, 6, 4, 2, 3
    num_entries = seq // ratio
    q = rng.normal(size=(batch, heads, seq, dim))
    kv = rng.normal(size=(batch, seq, dim))
    entries = rng.normal(size=(batch, num_entries, dim))
    sinks = rng.normal(size=(heads,))
    positions = np.arange(seq)
    valid = np.ones((batch, seq), bool)
    if padded:
        valid[0, 2] = False
        valid[1, 4] = False
    allowed = (positions[None, :] <= positions[:, None]) & (positions[None, :] > positions[:, None] - window)
    allowed = allowed[None] & valid[:, None, :]
    bias = np.where(allowed, 0.0, MASK)[:, None]
    reach = np.arange(num_entries)[None, :] < ((positions + 1) // ratio)[:, None]  # query 0 reaches nothing
    scores = np.where(reach[None], rng.normal(size=(batch, seq, num_entries)), -np.inf)
    scaling = dim**-0.5
    expected = _reference_kl(q, kv, entries, sinks, bias, scores, scaling)
    assert expected > 1e-3

    def loss(q_, scores_):
        return indexer_distillation_loss(
            q_,
            jnp.asarray(kv, jnp.float32),
            jnp.asarray(entries, jnp.float32),
            jnp.asarray(sinks, jnp.float32),
            jnp.asarray(valid),
            scores_,
            window=window,
            scaling=scaling,
            precision=jax.lax.Precision.HIGHEST,
            block_q=block_q,
        )

    q32, scores32 = jnp.asarray(q, jnp.float32), jnp.asarray(scores, jnp.float32)
    np.testing.assert_allclose(float(loss(q32, scores32)), expected, rtol=1e-4)
    grad_q, grad_scores = jax.grad(loss, argnums=(0, 1))(q32, scores32)
    assert not np.any(np.asarray(grad_q))  # attention side is stop-gradient
    grad_scores = np.asarray(grad_scores)
    assert np.isfinite(grad_scores).all() and np.any(grad_scores != 0)
    assert not np.any(grad_scores[~np.broadcast_to(reach[None], grad_scores.shape)])


@pytest.mark.parametrize("block_q", [64, 4, 3])
def test_lazy_score_inputs_match_dense_scores(block_q):
    """Per-tile scores from IndexerScoreInputs give the same loss and indexer gradients as dense scores."""
    rng = np.random.default_rng(1)
    batch, heads, seq, dim, ratio, window = 2, 3, 10, 4, 2, 3
    ih, idim = 2, 5
    num_entries = seq // ratio
    q = jnp.asarray(rng.normal(size=(batch, heads, seq, dim)), jnp.float32)
    kv = jnp.asarray(rng.normal(size=(batch, seq, dim)), jnp.float32)
    entries = jnp.asarray(rng.normal(size=(batch, num_entries, dim)), jnp.float32)
    sinks = jnp.asarray(rng.normal(size=(heads,)), jnp.float32)
    valid = jnp.ones((batch, seq), bool).at[0, 3].set(False)
    iq = jnp.asarray(rng.normal(size=(batch, seq, ih, idim)), jnp.float32)
    iw = jnp.asarray(rng.normal(size=(batch, seq, ih)), jnp.float32)
    ik = jnp.asarray(rng.normal(size=(batch, num_entries, idim)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None], (batch, seq))
    hi = jax.lax.Precision.HIGHEST
    kwargs = dict(window=window, scaling=dim**-0.5, precision=hi, block_q=block_q)

    def lazy(a, b, c):
        inputs = IndexerScoreInputs(a, b, c, positions, ratio, hi)
        return indexer_distillation_loss(q, kv, entries, sinks, valid, inputs, **kwargs)

    def dense(a, b, c):
        scores = shared_key_scores(a, b, c, positions, ratio, hi)
        return indexer_distillation_loss(q, kv, entries, sinks, valid, scores, **kwargs)

    lv, lg = jax.value_and_grad(lazy, argnums=(0, 1, 2))(iq, iw, ik)
    dv, dg = jax.value_and_grad(dense, argnums=(0, 1, 2))(iq, iw, ik)
    assert float(dv) > 1e-3
    np.testing.assert_allclose(float(lv), float(dv), rtol=1e-5)
    for a, b in zip(lg, dg, strict=True):
        assert np.isfinite(np.asarray(a)).all() and np.any(np.asarray(b) != 0)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


def _make_config(coef: float) -> DeepseekV41Config:
    config = DeepseekV41Config(**ref.case_config_kwargs("plain"), indexer_aux_loss_coef=coef)
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


@functools.cache
def _params_and_ids():
    config = _make_config(0.0)
    return ref.random_easydel_params(config, ref.CASE_SEEDS["plain"]), ref.case_input_ids(config, "plain")


def _build_easydel(config: DeepseekV41Config, params: dict[str, np.ndarray]):
    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        graphdef, state = spx.export(model)
        for collection, path, leaf in list(state.items()):
            if path not in params:
                continue
            old = _leaf_array(leaf)
            new = jnp.asarray(params[path], dtype=jnp.float32)
            sharding = getattr(old, "sharding", None)
            if sharding is not None:
                new = jax.device_put(new, sharding)
            state.set(collection, path, new)
        model = spx.bind(graphdef, state)
        model.eval()
    return model


def _key_part(key) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def _named_leaves(tree, known: set[str]) -> dict[str, tp.Any]:
    """``{easydel parameter name: leaf}`` of a graphstate-shaped pytree; leaves that are not parameters are skipped."""
    pairs: list[tuple[list[str], tp.Any]] = []
    if hasattr(tree, "items"):
        try:
            pairs = [([collection, *path.split(".")], leaf) for collection, path, leaf in tree.items()]
        except (TypeError, ValueError):
            pairs = []
    if not pairs:
        flat, _ = jax.tree_util.tree_flatten_with_path(tree)
        pairs = [([_key_part(k) for k in path], leaf) for path, leaf in flat]
    named: dict[str, tp.Any] = {}
    for parts, leaf in pairs:
        leaf = _leaf_array(leaf)
        if not hasattr(leaf, "size") or not leaf.size:
            continue
        if parts and parts[-1] == "value":
            parts = parts[:-1]
        match = next((".".join(parts[i:]) for i in range(len(parts)) if ".".join(parts[i:]) in known), None)
        if match is not None:
            named[match] = leaf
    return named


def _aux_loss_fn(state, ids):
    def loss_fn(graphstate):
        return state.merge(graphstate)(input_ids=ids).indexer_aux_loss

    return loss_fn


def test_indexer_params_get_finite_nonzero_gradients_and_nothing_else_does():
    params, ids_np = _params_and_ids()
    config = _make_config(COEF)
    model = _build_easydel(config, params)
    ids = jnp.asarray(ids_np, dtype=jnp.int32)
    known = set(params)
    indexer_names = {n for n in known if ".indexer." in n}
    assert indexer_names
    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        loss, grads = jax.value_and_grad(_aux_loss_fn(state, ids))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0
    grads = {name: np.asarray(g) for name, g in _named_leaves(grads, known).items()}
    assert indexer_names <= set(grads), sorted(indexer_names - set(grads))
    for name in sorted(indexer_names):
        assert np.isfinite(grads[name]).all(), name
        assert np.any(grads[name] != 0), f"{name}: all-zero indexer gradient"
    leaked = sorted(n for n, g in grads.items() if n not in indexer_names and np.any(g != 0))
    assert not leaked, f"the auxiliary loss reached non-indexer parameters: {leaked[:8]}"


def test_aux_loss_decreases_updating_only_indexer_params():
    params, ids_np = _params_and_ids()
    config = _make_config(COEF)
    model = _build_easydel(config, params)
    ids = jnp.asarray(ids_np, dtype=jnp.int32)
    known = set(params)
    before = {n: np.asarray(v) for n, v in _named_leaves(spx.export(model)[1], known).items()}

    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)

        @jax.jit
        def step(state):
            loss, grads = jax.value_and_grad(_aux_loss_fn(state, ids))(state.graphstate)
            return loss, state.apply_gradients(grads=grads)

        losses = []
        for _ in range(20):
            loss, state = step(state)
            losses.append(float(loss))
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.99 * losses[0], f"indexer loss did not decrease: {losses[0]:.5f} -> {losses[-1]:.5f}"
    after = {n: np.asarray(v) for n, v in _named_leaves(state.graphstate, known).items()}
    changed = {n for n in before if n in after and not np.array_equal(before[n], after[n])}
    assert changed, "no parameter changed"
    assert all(".indexer." in n for n in changed), sorted(n for n in changed if ".indexer." not in n)[:8]


def test_coef_zero_forward_is_unchanged_and_aux_loss_scales_with_coef():
    params, ids_np = _params_and_ids()
    ids = jnp.asarray(ids_np, dtype=jnp.int32)
    off_config, on_config = _make_config(0.0), _make_config(COEF)
    off_model, on_model = _build_easydel(off_config, params), _build_easydel(on_config, params)
    with off_config.mesh:
        off = off_model(input_ids=ids)
    with on_config.mesh:
        on = on_model(input_ids=ids)
    assert off.indexer_aux_loss is None and off.aux_loss is None
    assert on.indexer_aux_loss is not None and float(on.indexer_aux_loss) > 0
    np.testing.assert_allclose(float(on.aux_loss), COEF * float(on.indexer_aux_loss), rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(off.logits), np.asarray(on.logits))
    labels = ids[:, 1:]
    ce = [
        float(optax.softmax_cross_entropy_with_integer_labels(o.logits[:, :-1].astype(jnp.float32), labels).mean())
        for o in (off, on)
    ]
    assert ce[0] == ce[1]
