# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4.1 indexer auxiliary objective (``indexer_aux_loss_coef``), model level.

``KL(stop_grad(dense head-summed attention over compressed entries) || softmax(indexer scores))`` summed over index
source layers. The loss math itself (float64 NumPy parity, lazy scores, optional window / sink) is covered by
``tests/operations/kernels/test_indexer_distill.py``. Checks here: finite non-zero gradients on every indexer
parameter (and exactly zero elsewhere), the loss decreasing under optax steps, and the default (coef 0) forward
unchanged.
"""

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import optax
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

COEF = 0.5


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
