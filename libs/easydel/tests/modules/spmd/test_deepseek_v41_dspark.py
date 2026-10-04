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

"""DeepSeek-V4.1 DSpark draft model parity against DeepSeek's reference ``Transformer.forward_spec`` (fp32).

The oracle is the official reference ``model.py`` built with ``dspark_block_size > 0`` (pure-torch kernel stand-ins, see
``tests/modules/_deepseek_v41_reference.py``), temperature 0 (greedy). The same random parameters are copied into both
sides (each reference ``mtp.*`` tensor exactly once), the backbone embedding / LM head are shared, and the main hidden
states are random inputs. A prefill call seeds the window rings, then two decode steps (each advancing the position by
one, so step 2 reads the ring written by step 1) are compared: greedy ids exactly, logits and confidence within
``rtol=atol=1e-3``. The TPU model runs with ``precision=HIGHEST``. Router top-k and the draft arg-max are protected by
margin preconditions measured on the reference.

Not emulated by EasyDeL (and disabled in the oracle): fp8 activation fake-quantization of the window K/V.
"""

import functools
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41DSpark

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

RTOL = ATOL = 1e-3
MIN_GAP = 1e-3
LAST_STAGE = 1  # dspark_config_kwargs builds 2 stages


class Oracle(tp.NamedTuple):
    """Everything computed once per prefill length on the torch side."""

    config: DeepseekV41Config
    backbone: dict[str, np.ndarray]
    dspark: dict[str, np.ndarray]
    inputs: ref.DSparkInputs
    outputs: list[ref.DSparkOutputs]
    records: list


def _make_config() -> DeepseekV41Config:
    config = DeepseekV41Config(**ref.dspark_config_kwargs())
    config.moe_force_xla_gmm = True
    config.attach_custom_arguments()
    return config


@functools.cache
def _oracle(prefill_len: int) -> Oracle:
    config = _make_config()
    backbone = ref.random_easydel_params(config, ref.CASE_SEEDS["plain"])
    dspark = ref.random_dspark_params(config, ref.DSPARK_SEED)
    ref_model = ref.build_reference(config, dspark=True)
    ref.load_easydel_params(ref_model, backbone)  # tied embed / head and the (unused) backbone layers
    ref.load_dspark_params(ref_model, dspark)  # every mtp.* tensor exactly once
    inputs = ref.dspark_inputs(config, prefill_len)
    records: list = []
    outputs = ref.reference_dspark(ref_model, inputs, capture=records)
    return Oracle(config, backbone, dspark, inputs, outputs, records)


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _build_easydel(config: DeepseekV41Config, params: dict[str, np.ndarray]):
    """fp32 / HIGHEST draft model whose every parameter is overwritten with ``params``."""
    with config.mesh:
        model = DeepseekV41DSpark(
            config,
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
        assert not missing, f"DSpark model lacks parameters the oracle expects: {missing[:8]}"
        assert not unknown, f"DSpark parameters unknown to the oracle mapping: {unknown[:8]}"
        return spx.bind(graphdef, state)


class Run(tp.NamedTuple):
    outputs: list[ref.DSparkOutputs]
    traces: int
    final_position: np.ndarray


def _run_easydel(config: DeepseekV41Config, model, oracle: Oracle) -> Run:
    """Prefill, then one jitted ``draft`` per decode step; the shared embedding / LM head come from the backbone."""
    embed_table = jnp.asarray(oracle.backbone["model.embed_tokens.weight"])
    head_kernel = jnp.asarray(oracle.backbone["lm_head.weight"])

    def embed(ids):
        return embed_table[ids]

    def lm_head(hidden):
        return jnp.einsum("...d,dv->...v", hidden.astype(jnp.float32), head_kernel, precision=jax.lax.Precision.HIGHEST)

    graphdef, state = spx.export(model)
    traces: list[int] = []

    @jax.jit
    def step(state, tokens, hidden, draft_state):
        traces.append(1)  # runs at trace time only
        return spx.bind(graphdef, state).draft(tokens, hidden, draft_state, embed, lm_head)

    inputs = oracle.inputs
    outputs: list[ref.DSparkOutputs] = []
    with config.mesh:
        draft_state = model.prefill(jnp.asarray(inputs.prefill_hidden))
        for tokens, hidden in zip(inputs.tokens, inputs.step_hidden, strict=True):
            ids, logits, confidence, draft_state = step(
                state, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(hidden), draft_state
            )
            outputs.append(
                ref.DSparkOutputs(np.asarray(ids).astype(np.int64), np.asarray(logits), np.asarray(confidence))
            )
    return Run(outputs, len(traces), np.asarray(draft_state.position))


@pytest.mark.parametrize("prefill_len", ref.DSPARK_PREFILL_LENS)
def test_draft_parity_with_reference(prefill_len):
    """Greedy ids / logits / confidence of two decode steps match the reference (ring partly valid and wrapped)."""
    oracle = _oracle(prefill_len)
    config = oracle.config
    block = config.dspark_block_size
    margins = ref.topk_margins(oracle.records)
    assert margins["router"]["min_gap"] > MIN_GAP, margins
    for out in oracle.outputs:
        assert out.logits.std() > 0.3
        top2 = np.sort(out.logits, axis=-1)[..., -2:]
        assert float((top2[..., 1] - top2[..., 0]).min()) > MIN_GAP, "near-tie in the greedy draft arg-max"

    model = _build_easydel(config, {**oracle.dspark})
    run = _run_easydel(config, model, oracle)
    assert len(run.outputs) == ref.DSPARK_NUM_STEPS
    for step, (got, want) in enumerate(zip(run.outputs, oracle.outputs, strict=True)):
        assert got.ids.shape == want.ids.shape == (ref.CASE_BATCH, block + 1)
        assert got.logits.shape == want.logits.shape == (ref.CASE_BATCH, block, ref.VOCAB)
        assert got.confidence.shape == want.confidence.shape == (ref.CASE_BATCH, block)
        assert np.isfinite(got.logits).all() and np.isfinite(got.confidence).all()
        np.testing.assert_array_equal(got.ids, want.ids, err_msg=f"step {step}")
        np.testing.assert_allclose(got.logits, want.logits, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")
        np.testing.assert_allclose(got.confidence, want.confidence, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")


@pytest.mark.parametrize(
    "name",
    [
        f"stages.{LAST_STAGE}.markov_head.head.weight",  # token-conditioned logit bias
        f"stages.{LAST_STAGE}.markov_head.embed.weight",
        "stages.0.main_proj.weight",  # main hidden states -> window K/V of every stage
    ],
)
def test_negative_control_logits_break(name):
    """Perturbing one parameter in the EasyDeL copy only must move the draft logits: the path is really exercised."""
    oracle = _oracle(ref.DSPARK_PREFILL_LENS[0])
    rng = np.random.default_rng(7)
    params = dict(oracle.dspark)
    params[name] = params[name] + 0.5 * params[name].std() * rng.standard_normal(params[name].shape).astype(np.float32)
    model = _build_easydel(oracle.config, params)
    run = _run_easydel(oracle.config, model, oracle)
    gap = max(float(np.abs(g.logits - w.logits).max()) for g, w in zip(run.outputs, oracle.outputs, strict=True))
    assert gap > 1e-2, f"perturbing {name} moved the logits by only {gap:.3e}; that path is not exercised"


def test_negative_control_confidence_head_only_moves_confidence():
    """The confidence head is a pure read-out: perturbing it changes confidence and nothing else."""
    oracle = _oracle(ref.DSPARK_PREFILL_LENS[0])
    name = f"stages.{LAST_STAGE}.confidence_head.proj.weight"
    rng = np.random.default_rng(11)
    params = dict(oracle.dspark)
    params[name] = params[name] + 0.5 * params[name].std() * rng.standard_normal(params[name].shape).astype(np.float32)
    model = _build_easydel(oracle.config, params)
    run = _run_easydel(oracle.config, model, oracle)
    conf_gap = max(
        float(np.abs(g.confidence - w.confidence).max()) for g, w in zip(run.outputs, oracle.outputs, strict=True)
    )
    assert conf_gap > 1e-2, f"confidence moved by only {conf_gap:.3e}"
    for got, want in zip(run.outputs, oracle.outputs, strict=True):
        np.testing.assert_array_equal(got.ids, want.ids)
        np.testing.assert_allclose(got.logits, want.logits, rtol=RTOL, atol=ATOL)


def test_draft_step_compiles_once_across_decode_steps():
    """One jitted draft step serves every decode step: the carried state keeps shape and dtype."""
    prefill_len = ref.DSPARK_PREFILL_LENS[0]
    oracle = _oracle(prefill_len)
    model = _build_easydel(oracle.config, {**oracle.dspark})
    run = _run_easydel(oracle.config, model, oracle)
    assert len(run.outputs) == 2
    assert run.traces == 1, f"draft step was traced {run.traces} times"
    assert run.outputs[0].ids.shape == (ref.CASE_BATCH, oracle.config.dspark_block_size + 1)
    np.testing.assert_array_equal(run.final_position, np.full((ref.CASE_BATCH,), prefill_len + 2))
