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

"""DeepSeek-V4.1 vision-language parity against DeepSeek's own reference ``model.py`` + ``vision.py`` (torch, fp32).

The oracle (``tests/modules/_deepseek_v41_reference.py``) is the official reference with pure-torch stand-ins for its
tilelang kernels and the REAL ``ref/vision.py``. One image (patch grid 5x4: both sides need aligner padding to a
multiple of 3) sits in batch row 0; row 1 is text only. The seeded parameter draw is copied into both models and the
logits of all positions are compared (``image_mask = token_types >= 0`` selects ``bias_vl`` in every router and
``~image_mask`` is the engram mask, as in the reference ``Transformer.forward``).

Tolerance: fp32 with ``precision=HIGHEST`` on both sides (~1e-5 summation noise), so ``rtol=atol=1e-3`` leaves two orders
of margin, while the negative controls move the image-row logits by orders of magnitude more than that. Top-k selections
are protected by a margin precondition measured on the reference.
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
from easydel.modules.deepseek_v41 import (
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41ForConditionalGeneration,
)
from easydel.modules.deepseek_v41._engram_hash import EngramHasher

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

RTOL = ATOL = 1e-3
MIN_TOPK_GAP = 1e-3
VISION_PREFIXES = ("model.vision.", "model.aligner.", "model.image_")


class Oracle(tp.NamedTuple):
    """Everything computed once on the torch side."""

    config: DeepseekV41Config
    params: dict[str, np.ndarray]
    inputs: ref.VLInputs
    hashes: np.ndarray
    ref_model: tp.Any
    logits: np.ndarray
    records: list


def _finalize(config: DeepseekV41Config) -> DeepseekV41Config:
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


def _make_config() -> DeepseekV41Config:
    return _finalize(DeepseekV41Config(**ref.vl_config_kwargs()))


def _make_text_config() -> DeepseekV41Config:
    return _finalize(DeepseekV41Config(**ref.case_config_kwargs("engram")))


@functools.cache
def _oracle() -> Oracle:
    config = _make_config()
    assert config.vision_enabled
    params = ref.random_easydel_params(config, ref.VL_SEED)
    inputs = ref.vl_inputs(config)
    # Image-span positions are dead in the n-gram hash (reference: ``engram_mask = ~image_mask``).
    hasher = EngramHasher.from_config(config, np.arange(ref.VOCAB, dtype=np.int64))
    hashes = hasher.hash(inputs.ids, inputs.token_types >= 0)
    ref_model = ref.build_reference(config)
    ref.load_easydel_params(ref_model, params)
    records: list = []
    logits = ref.reference_logits(
        ref_model,
        inputs.ids,
        hashes,
        capture=records,
        token_types=inputs.token_types,
        images=ref.vl_reference_images(inputs),
    ).numpy()
    return Oracle(config, params, inputs, hashes, ref_model, logits, records)


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _build_easydel(config: DeepseekV41Config, params: dict[str, np.ndarray], model_cls=None):
    """fp32 / HIGHEST EasyDeL model whose every parameter is overwritten with ``params``."""
    model_cls = model_cls or DeepseekV41ForConditionalGeneration
    with config.mesh:
        model = model_cls.sequential_init(
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


def _vl_kwargs(oracle: Oracle, length: int | None = None) -> dict[str, tp.Any]:
    """Full VL call arguments (first ``length`` positions; the whole image span must fit)."""
    sl = slice(None, length)
    inputs = oracle.inputs
    return dict(
        input_ids=jnp.asarray(inputs.ids[:, sl], dtype=jnp.int32),
        engram_hash_ids=jnp.asarray(oracle.hashes[:, sl], dtype=jnp.int32),
        token_types=jnp.asarray(inputs.token_types[:, sl], dtype=jnp.int32),
        pixel_values=jnp.asarray(inputs.patches)[None],  # [N=1, P, 3, patch, patch]
        image_grid=inputs.grid,
    )


def _easydel_logits(model, oracle: Oracle) -> np.ndarray:
    with oracle.config.mesh:
        return np.asarray(model(**_vl_kwargs(oracle)).logits)


def test_selection_preconditions():
    """The comparison is only meaningful if real, tie-free top-k selection happens in the reference."""
    oracle = _oracle()
    assert ref.check_selection_schedule(oracle.config, ref.VL_SEQ_LEN) == []
    margins = ref.topk_margins(oracle.records)
    assert {"index", "candidate_block", "router"} <= set(margins), margins
    for kind in ("index", "candidate_block"):
        assert margins[kind]["selecting_rows"] > 0, f"{kind} never discarded an entry: {margins[kind]}"
    for kind, stats in margins.items():
        assert stats["min_gap"] > MIN_TOPK_GAP, f"{kind} top-k has a near tie: {stats}"
    types = oracle.inputs.token_types
    assert (types[0] >= 0).sum() == 8 and (types[1] >= 0).sum() == 0
    # grid 5x4 -> 2x2 LLM grid after padding: both sides are padded by the aligner
    assert oracle.inputs.grid[0] % 3 != 0 and oracle.inputs.grid[1] % 3 != 0


def test_logits_parity_with_reference():
    """All-position fp32 logits with an image (bias_vl routing, masked engram, merged span) match the reference."""
    oracle = _oracle()
    assert oracle.logits.std() > 0.3
    model = _build_easydel(oracle.config, oracle.params)
    logits = _easydel_logits(model, oracle)
    assert logits.shape == oracle.logits.shape == (ref.CASE_BATCH, ref.VL_SEQ_LEN, ref.VOCAB)
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits, oracle.logits, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "name",
    [
        "model.vision.blocks.0.attn.wqkv.weight",  # ViT attention (2D RoPE path)
        "model.vision.blocks.1.mlp.w2.weight",  # ViT MLP
        "model.aligner.w1.weight",  # 3x3 unfold aligner
    ],
)
def test_negative_control_vision_weight(name):
    """Perturbing one vision weight must move the image row and leave the text-only row at parity."""
    oracle = _oracle()
    rng = np.random.default_rng(7)
    params = dict(oracle.params)
    params[name] = params[name] + 0.5 * params[name].std() * rng.standard_normal(params[name].shape).astype(np.float32)
    logits = _easydel_logits(_build_easydel(oracle.config, params), oracle)
    image_gap = float(np.max(np.abs(logits[0] - oracle.logits[0])))
    assert image_gap > 1e-2, f"perturbing {name} moved the image row by only {image_gap:.3e}"
    np.testing.assert_allclose(logits[1], oracle.logits[1], rtol=RTOL, atol=ATOL)


def test_negative_control_bias_vl_only_acts_inside_image_spans():
    """Perturbing every ``bias_vl`` moves the image row, never the text-only row (bias applies only to image tokens)."""
    oracle = _oracle()
    names = sorted(n for n in oracle.params if n.endswith("gate.bias_vl"))
    assert len(names) == oracle.config.num_hidden_layers
    rng = np.random.default_rng(11)
    params = dict(oracle.params)
    for name in names:
        params[name] = params[name] + 2.0 * rng.standard_normal(params[name].shape).astype(np.float32)
    logits = _easydel_logits(_build_easydel(oracle.config, params), oracle)
    image_gap = float(np.max(np.abs(logits[0] - oracle.logits[0])))
    assert image_gap > 1e-2, f"perturbing bias_vl moved the image row by only {image_gap:.3e}"
    np.testing.assert_allclose(logits[1], oracle.logits[1], rtol=RTOL, atol=ATOL)
    # Positions before the span cannot see any image token: they must be untouched as well.
    start = ref.VL_IMAGE_START
    np.testing.assert_allclose(logits[0, :start], oracle.logits[0, :start], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("with_vision_config", [False, True])
def test_vision_disabled_causal_lm_param_tree_is_unchanged(with_vision_config):
    """``DeepseekV41ForCausalLM`` declares exactly the text parameters: no ``bias_vl``, vision, aligner or image_*."""
    text_config = _make_text_config()
    expected = ref.easydel_param_shapes(text_config)
    config = _make_config() if with_vision_config else text_config
    with config.mesh:
        model = DeepseekV41ForCausalLM.sequential_init(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        _, state = spx.export(model)
    shapes = {}
    for collection, path, leaf in state.items():
        arr = _leaf_array(leaf)
        if collection == "parameters" and isinstance(arr, jax.Array) and arr.size:
            shapes[path] = tuple(arr.shape)
    assert shapes == expected
    assert not [n for n in shapes if "bias_vl" in n or "vision" in n or "aligner" in n or "image_" in n]


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


def test_training_step_gradients_flow_through_vision():
    """One CE step over the text targets: finite grads everywhere and non-zero grads on every vision-side parameter."""
    oracle = _oracle()
    config = oracle.config
    model = _build_easydel(config, oracle.params)
    call = _vl_kwargs(oracle)
    ids = call["input_ids"]
    target_weight = (call["token_types"][:, 1:] < 0).astype(jnp.float32)  # predict text tokens only

    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)

        def loss_fn(graphstate):
            logits = state.merge(graphstate)(**call).logits.astype(jnp.float32)
            per_token = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:])
            return jnp.sum(per_token * target_weight) / jnp.sum(target_weight)

        loss, grads = jax.value_and_grad(loss_fn)(state.graphstate)
        grads = {name: np.asarray(g) for name, g in _named_leaves(grads, set(oracle.params)).items()}

    assert np.isfinite(float(loss))
    assert all(np.isfinite(g).all() for g in grads.values()), "non-finite gradient"
    vision_names = sorted(n for n in oracle.params if n.startswith(VISION_PREFIXES))
    assert vision_names, "the VL parameter table lists no vision parameters"
    assert set(vision_names) <= set(grads), sorted(set(vision_names) - set(grads))[:8]
    zero = [n for n in vision_names if not np.any(grads[n] != 0)]
    assert not zero, f"vision-side parameters with an all-zero gradient: {zero[:10]}"


def test_cached_prefill_with_image_and_decode_match_stateless():
    """Prefill (image span inside the prefill) + three one-token decode steps reproduce the stateless forward."""
    oracle = _oracle()
    config = oracle.config
    model = _build_easydel(config, oracle.params)
    prefill_len, decode_steps = 14, 3
    total = prefill_len + decode_steps
    call = _vl_kwargs(oracle, total)
    ids, hashes, token_types = call["input_ids"], call["engram_hash_ids"], call["token_types"]
    grid, pixel_values = call["image_grid"], call["pixel_values"]
    batch = ids.shape[0]
    assert int(np.flatnonzero(oracle.inputs.token_types[0] >= 0).max()) < prefill_len  # whole span in the prefill

    graphdef, state = spx.export(model)

    @jax.jit
    def cached_step(state, input_ids, hash_ids, types, pixels, cache):
        out = spx.bind(graphdef, state)(
            input_ids=input_ids,
            engram_hash_ids=hash_ids,
            token_types=types,
            pixel_values=pixels,
            image_grid=grid,
            past_key_values=cache,
        )
        return out.logits[:, -1], out.past_key_values

    with config.mesh:
        stateless = np.asarray(model(**call).logits)
        cache = model.init_cache(batch_size=batch, max_length=total + 8)
        last, cache = cached_step(
            state, ids[:, :prefill_len], hashes[:, :prefill_len], token_types[:, :prefill_len], pixel_values, cache
        )
        steps = [np.asarray(last)]
        for pos in range(prefill_len, total):  # decode: text tokens only, no types / images
            last, cache = cached_step(state, ids[:, pos : pos + 1], hashes[:, pos : pos + 1], None, None, cache)
            steps.append(np.asarray(last))

    assert len(steps) == decode_steps + 1
    worst = max(float(np.abs(stateless[:, prefill_len - 1 + k] - step).max()) for k, step in enumerate(steps))
    assert worst < 2e-4, f"cached image prefill + decode diverged from the stateless forward: max|d|={worst:.3e}"
    for view in cache.views:
        assert bool(jnp.all(view.cache_position == total))
