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

"""DeepSeek-V4.1 numerical parity against DeepSeek's own reference ``model.py`` (torch oracle, fp32).

The oracle (``tests/modules/_deepseek_v41_reference.py``) is the official reference implementation with pure-torch
stand-ins for its tilelang kernels. EasyDeL's random-initialised weights are replaced by the oracle's seeded draws
(sharp score distributions), copied into BOTH models, and the logits of all positions are compared.

Tolerance: both sides are fp32; the TPU model runs with ``precision=HIGHEST`` so matmuls are not bf16 passes. The
remaining differences are summation order and ``exp``/``rsqrt`` implementations (~1e-5 on logits of std ~1), so
``rtol=atol=1e-3`` leaves two orders of margin while a wrong rope / ownership / selection rule moves logits by >1e-2
(see the negative controls). Top-k selections are protected by a margin precondition measured on the reference.

Not emulated by EasyDeL (and therefore disabled in the oracle): fp8 / fp4 activation fake-quantization.
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
from easydel.layers.engram import EngramHasher
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

RTOL = ATOL = 1e-3
MIN_TOPK_GAP = 1e-3
GRAD_REL_TOL = 5e-3


class Oracle(tp.NamedTuple):
    """Everything computed once per case on the torch side."""

    config: DeepseekV41Config
    params: dict[str, np.ndarray]
    ids: np.ndarray
    hashes: np.ndarray | None
    ref_model: tp.Any
    logits: np.ndarray
    records: list


def _make_config(case: str) -> DeepseekV41Config:
    config = DeepseekV41Config(**ref.case_config_kwargs(case))
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


@functools.cache
def _oracle(case: str) -> Oracle:
    config = _make_config(case)
    params = ref.random_easydel_params(config, ref.CASE_SEEDS[case])
    ids = ref.case_input_ids(config, case)
    hashes = None
    if config.engram_layer_ids:
        hashes = EngramHasher.from_config(config, np.arange(ref.VOCAB, dtype=np.int64)).hash(ids)
    ref_model = ref.build_reference(config)
    ref.load_easydel_params(ref_model, params)
    records: list = []
    logits = ref.reference_logits(ref_model, ids, hashes, capture=records).numpy()
    return Oracle(config, params, ids, hashes, ref_model, logits, records)


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _build_easydel(config: DeepseekV41Config, params: dict[str, np.ndarray]):
    """fp32 / HIGHEST EasyDeL model whose every parameter is overwritten with ``params``."""
    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        graphdef, state = spx.export(model)
        written, unknown = set(), []
        for collection, path, leaf in list(state.items()):
            if path not in params:
                # Only trainable parameters must map to the oracle; RNG / buffer collections have no reference twin.
                if collection == "parameters" and isinstance(_leaf_array(leaf), jax.Array) and _leaf_array(leaf).size:
                    unknown.append(f"{collection}:{path}")
                continue
            old = _leaf_array(leaf)
            assert tuple(old.shape) == params[path].shape, (path, old.shape, params[path].shape)
            new = jnp.asarray(params[path], dtype=jnp.float32)
            sharding = getattr(old, "sharding", None)
            if sharding is not None:
                new = jax.device_put(new, sharding)
            state.set(collection, path, new)
            written.add(path)
        missing = sorted(set(params) - written)
        assert not missing, f"EasyDeL model lacks parameters the oracle expects: {missing[:8]}"
        assert not unknown, f"EasyDeL parameters unknown to the oracle mapping: {unknown[:8]}"
        model = spx.bind(graphdef, state)
        model.eval()
    return model


def _easydel_logits(model, config, ids, hashes) -> np.ndarray:
    with config.mesh:
        out = model(
            input_ids=jnp.asarray(ids, dtype=jnp.int32),
            engram_hash_ids=None if hashes is None else jnp.asarray(hashes, dtype=jnp.int32),
        )
        return np.asarray(out.logits)


@pytest.mark.parametrize("case", ref.CASES)
def test_selection_preconditions(case):
    """The comparison is only meaningful if real, tie-free top-k selection happens in the reference."""
    oracle = _oracle(case)
    assert ref.check_selection_schedule(oracle.config, oracle.ids.shape[1]) == []
    margins = ref.topk_margins(oracle.records)
    assert {"index", "candidate_block", "router"} <= set(margins), margins
    for kind in ("index", "candidate_block"):
        assert margins[kind]["selecting_rows"] > 0, f"{kind} never discarded an entry: {margins[kind]}"
    for kind, stats in margins.items():
        assert stats["min_gap"] > MIN_TOPK_GAP, f"{kind} top-k has a near tie: {stats}"
    config = oracle.config
    cover = config.candidate_topk_blocks * config.candidate_block_size - config.candidate_block_size + 1
    assert cover >= config.index_topk


@pytest.mark.parametrize("case", ref.CASES)
def test_logits_parity_with_reference(case):
    """Plain / engram / yarn-compress-rope: all-position fp32 logits match DeepSeek's reference."""
    oracle = _oracle(case)
    assert oracle.logits.std() > 0.3
    model = _build_easydel(oracle.config, oracle.params)
    logits = _easydel_logits(model, oracle.config, oracle.ids, oracle.hashes)
    assert logits.shape == oracle.logits.shape == (ref.CASE_BATCH, ref.CASE_SEQ_LEN[case], ref.VOCAB)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits, oracle.logits, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.4.self_attn.compressor.kv_proj.weight",  # ratio-1 KV source shared by layers 5-7
        "model.layers.1.self_attn.compressor.gate_proj.weight",  # ratio-2 KV source shared by layers 2-3
        "model.layers.3.self_attn.indexer.weights_proj.weight",  # index source scoring layer 1's keys
    ],
)
def test_negative_control_parity_breaks(name):
    """Perturbing one parameter in the EasyDeL copy only must break parity: the path is really exercised."""
    oracle = _oracle("plain")
    rng = np.random.default_rng(7)
    params = dict(oracle.params)
    params[name] = params[name] + 0.5 * params[name].std() * rng.standard_normal(params[name].shape).astype(np.float32)
    model = _build_easydel(oracle.config, params)
    logits = _easydel_logits(model, oracle.config, oracle.ids, oracle.hashes)
    gap = float(np.max(np.abs(logits - oracle.logits)))
    assert gap > 1e-2, f"perturbing {name} moved logits by only {gap:.3e}; that path is not exercised"


def _key_part(key) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    return str(key)


def _named_leaves(tree, known: set[str]) -> dict[str, jax.Array]:
    """``{easydel parameter name: leaf}`` of a graphstate-shaped pytree (gradients)."""
    pairs: list[tuple[list[str], tp.Any]] = []
    if hasattr(tree, "items"):
        try:
            pairs = [([collection, *path.split(".")], leaf) for collection, path, leaf in tree.items()]
        except (TypeError, ValueError):
            pairs = []
    if not pairs:
        flat, _ = jax.tree_util.tree_flatten_with_path(tree)
        pairs = [([_key_part(k) for k in path], leaf) for path, leaf in flat]
    named: dict[str, jax.Array] = {}
    unknown = []
    for parts, leaf in pairs:
        leaf = _leaf_array(leaf)
        if not hasattr(leaf, "size") or not leaf.size:
            continue
        if parts and parts[-1] == "value":
            parts = parts[:-1]
        match = next((".".join(parts[i:]) for i in range(len(parts)) if ".".join(parts[i:]) in known), None)
        if match is None:
            unknown.append(".".join(parts))
        else:
            named[match] = leaf
    assert not unknown, f"gradient leaves unknown to the oracle mapping: {unknown[:8]}"
    return named


def test_training_step_gradients_against_reference_autograd():
    """One CE step: finite grads, non-zero grads where expected, and torch-autograd parity (engram case)."""
    oracle = _oracle("engram")
    config = oracle.config
    model = _build_easydel(config, oracle.params)
    ids = jnp.asarray(oracle.ids, dtype=jnp.int32)
    hashes = jnp.asarray(oracle.hashes, dtype=jnp.int32)

    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)

        def loss_fn(graphstate):
            logits = state.merge(graphstate)(input_ids=ids, engram_hash_ids=hashes).logits.astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:]).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.graphstate)
        grads = _named_leaves(grads, set(oracle.params))
        grads = {name: np.asarray(g) for name, g in grads.items()}

    assert np.isfinite(float(loss))
    assert all(np.isfinite(g).all() for g in grads.values()), "non-finite gradient"
    # The router correction bias is selection-only and the indexer is a hard top-k: neither receives LM gradient.
    exempt = lambda n: n.endswith("e_score_correction_bias") or ".indexer." in n  # noqa: E731
    zero = sorted(n for n, g in grads.items() if not exempt(n) and not np.any(g != 0))
    assert not zero, f"parameters with an all-zero gradient: {zero[:10]}"
    assert set(oracle.params) - {n for n in oracle.params if exempt(n)} <= set(grads)

    ref_loss, ref_grads = ref.reference_loss_and_grads(oracle.ref_model, oracle.ids, oracle.hashes)
    np.testing.assert_allclose(float(loss), ref_loss, rtol=RTOL, atol=ATOL)
    worst: dict[str, float] = {}
    for name, grad in grads.items():
        if exempt(name):
            continue
        for ref_name, value in ref.easydel_to_reference(name, grad):
            expected = ref_grads.get(ref_name)
            if expected is None:
                # torch leaves `.grad` unset on modules that never ran (an expert no token routed to): that is a
                # zero gradient, so EasyDeL's slice for it must be exactly zero too.
                assert not np.any(np.asarray(value) != 0), f"{ref_name}: torch has no gradient, EasyDeL is nonzero"
                continue
            scale = max(float(np.abs(expected).max()), 1e-6)
            worst[name] = max(worst.get(name, 0.0), float(np.abs(value - expected).max()) / scale)
    bad = {n: e for n, e in worst.items() if e > GRAD_REL_TOL}
    assert not bad, f"gradients disagree with torch autograd (max|d|/max|ref|): {dict(sorted(bad.items())[:10])}"


@pytest.mark.parametrize(
    ("case", "prefill_len"),
    [
        # Odd prefill: ratio-2 sources carry an open group into decode; decode closes groups mid-stream.
        ("plain", 13),
        # Prefill shorter than the window and the candidate blocks: entries are emitted by decode steps.
        ("plain", 3),
        # Engram layers on the cached path (hash ids of position t depend only on tokens <= t).
        ("engram", 14),
    ],
)
def test_cached_decode_matches_stateless(case, prefill_len):
    """Prefill + one-token cached decode must reproduce the stateless forward at every decoded position.

    The stateless forward is itself pinned to DeepSeek's reference by ``test_logits_parity_with_reference``. The
    decode path exercises a different implementation: per-row ring writes, group close in the source compressor,
    the indexer over stored keys in a longer padded buffer, and attention over gathered selected entries.
    """
    oracle = _oracle(case)
    config = oracle.config
    model = _build_easydel(config, oracle.params)
    ids = jnp.asarray(oracle.ids, dtype=jnp.int32)
    hashes = None if oracle.hashes is None else jnp.asarray(oracle.hashes, dtype=jnp.int32)
    batch, total = ids.shape

    def hash_slice(start: int, stop: int):
        return None if hashes is None else hashes[:, start:stop]

    graphdef, state = spx.export(model)

    # One jitted step: compiles once for the prefill shape and once for all single-token decode steps
    # (eager per-token calls retrace the whole stack every step).
    @jax.jit
    def cached_step(state, input_ids, hash_ids, cache):
        out = spx.bind(graphdef, state)(input_ids=input_ids, engram_hash_ids=hash_ids, past_key_values=cache)
        return out.logits[:, -1], out.past_key_values

    with config.mesh:
        stateless = np.asarray(model(input_ids=ids, engram_hash_ids=hashes).logits)
        cache = model.init_cache(batch_size=batch, max_length=total + 8)
        last, cache = cached_step(state, ids[:, :prefill_len], hash_slice(0, prefill_len), cache)
        steps = [np.asarray(last)]
        for pos in range(prefill_len, total):
            last, cache = cached_step(state, ids[:, pos : pos + 1], hash_slice(pos, pos + 1), cache)
            steps.append(np.asarray(last))

    worst = max(float(np.abs(stateless[:, prefill_len - 1 + k] - step).max()) for k, step in enumerate(steps))
    assert worst < 2e-4, f"cached decode diverged from the stateless forward: max|d|={worst:.3e}"
    for idx, view in enumerate(cache.views):
        assert bool(jnp.all(view.cache_position == total))
        # State lives only where the reference keeps it: compressed KV on KV sources, index keys on key owners.
        assert (view.compressed_kv is not None) == config.is_kv_source(idx)
        assert (view.index_k is not None) == config.owns_index_keys(idx)
