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

"""DeepSeek-V4 DSpark draft model parity against DeepSeek's V4-Flash-0731 reference ``Transformer.forward_spec``.

The oracle is the released ``inference/model.py`` (copied to ``.xerxes/projects/mtp-sweep/ref-v4-0731``) with the
pure-torch kernel stand-ins of ``tests/modules/_deepseek_v41_reference.py`` (fp8/fp4 activation fake-quant = identity,
fp32 weights), built with ``temperature=0`` (greedy). Random parameters are copied into both sides (every reference
``mtp.*`` tensor exactly once); the main hidden states are random inputs, so the draft model is checked on its own. A
prefill call seeds the window rings, then two decode steps are compared: greedy ids exactly, logits and confidence
within ``rtol=atol=1e-3`` (EasyDeL at ``precision=HIGHEST``). Router top-k and the draft arg-max are protected by margin
preconditions measured on the reference.

Also covered: the backbone hook ``DeepseekV4Model.forward(output_dspark_hidden=True)`` returns the stream mean of each
target layer's OUTPUT and leaves the default path unchanged.
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys
import types
import typing as tp

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import torch
import torch.nn.functional as F
from easydel.modules.deepseek_v4.modeling_deepseek_v4_dspark import DeepseekV4DSpark

try:
    from tests.modules import _deepseek_v41_reference as v41ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as v41ref  # pyright: ignore[reportMissingImports]

REF_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / ".xerxes"
    / "projects"
    / "mtp-sweep"
    / "ref-v4-0731"
    / "inference_model.py"
)
pytestmark = pytest.mark.skipif(not REF_PATH.exists(), reason=f"V4-0731 reference not found at {REF_PATH}")

VOCAB = 256
BATCH = 2
STAGES = 3
BLOCK = 3
WINDOW = 4
MAX_SEQ = 64
NUM_LAYERS = 2
TARGETS = (0, 1)
PREFILL_LENS = (2, 6)  # 2 < window (partially valid ring), 6 > window (ring wraps at seeding)
NUM_STEPS = 2
PARAM_SEED = 11
RTOL = ATOL = 1e-3
MIN_GAP = 1e-3


def _make_config() -> ed.DeepseekV4Config:
    cfg = ed.DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=512,
        layer_types=["sliding_attention"] * NUM_LAYERS,
        mlp_layer_types=["moe"] * NUM_LAYERS,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        swiglu_limit=10.0,
        sliding_window=WINDOW,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        initializer_range=0.2,
        tie_word_embeddings=False,
        dspark_block_size=BLOCK,
        dspark_target_layer_ids=list(TARGETS),
        dspark_noise_token_id=5,
        dspark_markov_rank=8,
    )
    cfg.moe_force_xla_gmm = True
    if hasattr(cfg, "attach_custom_arguments"):
        cfg.attach_custom_arguments()
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# Reference module
# ----------------------------------------------------------------------------------------------------------------------


@functools.cache
def _ref_module() -> types.ModuleType:
    """Import the V4-0731 ``inference/model.py`` with the pure-torch ``kernel`` stand-ins."""
    if not hasattr(torch, "float4_e2m1fn_x2"):  # referenced (never matched) by ``linear()`` on older torch
        torch.float4_e2m1fn_x2 = "float4_e2m1fn_x2"
    saved = sys.modules.get("kernel")
    previous_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        kernel = types.ModuleType("kernel")
        for fn in (
            v41ref.sparse_attn,
            v41ref.hc_split_sinkhorn,
            v41ref.act_quant,
            v41ref.fp4_act_quant,
            v41ref.fp8_gemm,
            v41ref.fp4_gemm,
        ):
            setattr(kernel, fn.__name__, fn)
        sys.modules["kernel"] = kernel
        spec = importlib.util.spec_from_file_location("_deepseek_v4_0731_reference", REF_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_flag
        if saved is None:
            sys.modules.pop("kernel", None)
        else:
            sys.modules["kernel"] = saved
    return module


def _build_reference(cfg):
    module = _ref_module()
    args = module.ModelArgs(
        max_batch_size=BATCH,
        max_seq_len=MAX_SEQ,
        temperature=0.0,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        scale_dtype="fp32",
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=NUM_LAYERS,
        n_hash_layers=0,
        n_mtp_layers=STAGES,
        n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=1,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=float(cfg.routed_scaling_factor),
        swiglu_limit=float(cfg.swiglu_limit),
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=float(cfg.rms_norm_eps),
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=(0,) * (NUM_LAYERS + STAGES),
        original_seq_len=0,
        rope_theta=float(cfg.rope_theta),
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=float(cfg.hc_eps),
        dspark_block_size=BLOCK,
        dspark_noise_token_id=int(cfg.dspark_noise_token_id),
        dspark_target_layer_ids=TARGETS,
        dspark_markov_rank=int(cfg.dspark_markov_rank),
    )
    model = module.Transformer(args)
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------------------------------
# Parameters: EasyDeL names -> reference mtp.N.* tensors
# ----------------------------------------------------------------------------------------------------------------------

_DIRECT = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "attn_hc.fn": "hc_attn_fn",
    "attn_hc.base": "hc_attn_base",
    "attn_hc.scale": "hc_attn_scale",
    "ffn_hc.fn": "hc_ffn_fn",
    "ffn_hc.base": "hc_ffn_base",
    "ffn_hc.scale": "hc_ffn_scale",
    "self_attn.q_a_norm.weight": "attn.q_norm.weight",
    "self_attn.kv_norm.weight": "attn.kv_norm.weight",
    "self_attn.sinks": "attn.attn_sink",
    "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
    "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight",
    "hc_head.hc_fn": "hc_head_fn",
    "hc_head.hc_base": "hc_head_base",
    "hc_head.hc_scale": "hc_head_scale",
    "markov_head.markov_w1.weight": "markov_head.markov_w1.weight",
}
_TRANSPOSED = {
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_proj.weight": "attn.wkv.weight",
    "self_attn.o_a_proj.weight": "attn.wo_a.weight",
    "self_attn.o_b_proj.weight": "attn.wo_b.weight",
    "mlp.gate.weight": "ffn.gate.weight",
    "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    "main_proj.weight": "main_proj.weight",
    "markov_head.markov_w2.weight": "markov_head.markov_w2.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}
_EXPERT_LEAF = {
    "mlp.experts.gate_proj.weight": "w1",
    "mlp.experts.up_proj.weight": "w3",
    "mlp.experts.down_proj.weight": "w2",
}


def _to_reference(name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    stage_str, rest = name[len("stages.") :].split(".", 1)
    prefix = f"mtp.{int(stage_str)}."
    if rest in _DIRECT:
        return [(prefix + _DIRECT[rest], array)]
    if rest in _TRANSPOSED:
        return [(prefix + _TRANSPOSED[rest], array.T)]
    if rest == "mlp.shared_experts.gate_up_proj.weight":  # [D, 2M]: gate columns first (tp=1), then up
        half = array.shape[1] // 2
        return [
            (prefix + "ffn.shared_experts.w1.weight", array[:, :half].T),
            (prefix + "ffn.shared_experts.w3.weight", array[:, half:].T),
        ]
    if rest in _EXPERT_LEAF:  # [E, in, out] -> per-expert [out, in]
        leaf = _EXPERT_LEAF[rest]
        return [(f"{prefix}ffn.experts.{e}.{leaf}.weight", array[e].T) for e in range(array.shape[0])]
    raise KeyError(f"unmapped DSpark parameter {name!r}")


def _param_shapes(cfg) -> dict[str, tuple[int, ...]]:
    with cfg.mesh:
        model = spx.eval_shape(lambda: DeepseekV4DSpark(cfg, rngs=spx.Rngs(0), num_stages=STAGES))
    _graphdef, state = spx.export(model)
    return {path: tuple(leaf.shape) for _collection, path, leaf in state.items()}


def _draw(rng: np.random.Generator, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name.endswith("hc_head.hc_fn"):
        return (rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
    if name.endswith("hc_head.hc_base"):
        return (0.5 * rng.standard_normal(shape)).astype(np.float32)
    if name.endswith("hc_head.hc_scale"):
        return (1.0 + 0.2 * rng.standard_normal(shape)).astype(np.float32)
    if name.endswith("markov_w1.weight"):
        return rng.standard_normal(shape).astype(np.float32)
    return v41ref._draw(rng, name, shape)


def _random_params(cfg, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dspark = {name: _draw(rng, name, shape) for name, shape in sorted(_param_shapes(cfg).items())}
    backbone = {
        "embed": rng.standard_normal((cfg.vocab_size, cfg.hidden_size)).astype(np.float32),
        "head": (rng.standard_normal((cfg.hidden_size, cfg.vocab_size)) / np.sqrt(cfg.hidden_size)).astype(np.float32),
    }
    return dspark, backbone


def _load_reference(ref_model, dspark: dict[str, np.ndarray], backbone: dict[str, np.ndarray]) -> None:
    params = dict(ref_model.named_parameters())
    written: dict[str, int] = {}
    pairs = [("embed.weight", backbone["embed"]), ("head.weight", backbone["head"].T)]
    for name in sorted(dspark):
        pairs.extend(_to_reference(name, dspark[name]))
    for ref_name, value in pairs:
        target = params[ref_name]
        value = np.ascontiguousarray(value, dtype=np.float32)
        assert tuple(target.shape) == value.shape, (ref_name, tuple(target.shape), value.shape)
        with torch.no_grad():
            target.copy_(torch.from_numpy(value))
        written[ref_name] = written.get(ref_name, 0) + 1
    expected = {n for n in params if n.startswith("mtp.")}
    missing = sorted(expected - set(written))
    repeated = sorted(n for n, count in written.items() if count != 1)
    assert expected and not missing and not repeated, (missing[:8], repeated[:8])


# ----------------------------------------------------------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------------------------------------------------------


class Inputs(tp.NamedTuple):
    prefill_hidden: np.ndarray
    tokens: np.ndarray
    step_hidden: np.ndarray


class Outputs(tp.NamedTuple):
    ids: np.ndarray
    logits: np.ndarray
    confidence: np.ndarray


class Oracle(tp.NamedTuple):
    config: tp.Any
    dspark: dict[str, np.ndarray]
    backbone: dict[str, np.ndarray]
    inputs: Inputs
    outputs: list[Outputs]
    router_gap: float


def _inputs(cfg, prefill_len: int) -> Inputs:
    rng = np.random.default_rng(5000 + prefill_len)
    width = len(TARGETS) * cfg.hidden_size
    return Inputs(
        rng.standard_normal((BATCH, prefill_len, width)).astype(np.float32),
        rng.integers(0, cfg.vocab_size, size=(NUM_STEPS, BATCH), dtype=np.int64),
        rng.standard_normal((NUM_STEPS, BATCH, 1, width)).astype(np.float32),
    )


@functools.cache
def _oracle(prefill_len: int) -> Oracle:
    cfg = _make_config()
    dspark, backbone = _random_params(cfg, PARAM_SEED)
    ref_model = _build_reference(cfg)
    _load_reference(ref_model, dspark, backbone)
    inputs = _inputs(cfg, prefill_len)
    module = _ref_module()
    gaps: list[float] = []
    original_gate = module.Gate.forward

    def gate_forward(self, x, input_ids=None):  # record the learned top-k margin, then run the reference gate
        scores = F.softplus(F.linear(x.float(), self.weight.float())).sqrt()
        if self.bias is not None:
            scores = scores + self.bias
        top = scores.topk(self.topk + 1, dim=-1).values
        gaps.append(float((top[:, self.topk - 1] - top[:, self.topk]).min()))
        return original_gate(self, x, input_ids)

    outputs: list[Outputs] = []
    module.Gate.forward = gate_forward
    try:
        with torch.no_grad():
            seeded = ref_model.forward_spec(
                torch.zeros(BATCH, dtype=torch.long), torch.from_numpy(inputs.prefill_hidden), 0
            )
            assert seeded is None, "the prefill call only seeds the window rings"
            start = prefill_len
            for tokens, hidden in zip(inputs.tokens, inputs.step_hidden, strict=True):
                ids, logits, confidence = ref_model.forward_spec(
                    torch.as_tensor(tokens, dtype=torch.long), torch.from_numpy(np.ascontiguousarray(hidden)), start
                )
                outputs.append(
                    Outputs(ids.numpy().astype(np.int64).copy(), logits.numpy().copy(), confidence.numpy().copy())
                )
                start += 1
    finally:
        module.Gate.forward = original_gate
    assert cfg.scoring_func == "sqrtsoftplus" and gaps
    return Oracle(cfg, dspark, backbone, inputs, outputs, min(gaps))


# ----------------------------------------------------------------------------------------------------------------------
# EasyDeL side
# ----------------------------------------------------------------------------------------------------------------------


def _leaf_array(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _build_easydel(cfg, params: dict[str, np.ndarray]) -> DeepseekV4DSpark:
    with cfg.mesh:
        model = DeepseekV4DSpark(
            cfg,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
            num_stages=STAGES,
        )
        graphdef, state = spx.export(model)
        # Every leaf replicated on the config mesh: leaves built without sharding metadata would otherwise stay on
        # one device and the jitted draft step would see incompatible device sets.
        mesh = getattr(cfg.mesh, "jax_mesh", cfg.mesh)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        written = set()
        for collection, path, leaf in list(state.items()):
            old = _leaf_array(leaf)
            if path not in params:
                if isinstance(old, jax.Array):
                    state.set(collection, path, jax.device_put(old, replicated))
                continue
            assert tuple(old.shape) == params[path].shape, (path, old.shape, params[path].shape)
            new = jax.device_put(jnp.asarray(params[path], dtype=jnp.float32), replicated)
            state.set(collection, path, new)
            written.add(path)
        assert written == set(params), sorted(set(params) - written)[:8]
        return spx.bind(graphdef, state)


def _run_easydel(cfg, model: DeepseekV4DSpark, oracle: Oracle) -> list[Outputs]:
    embed_table = jnp.asarray(oracle.backbone["embed"])
    head_kernel = jnp.asarray(oracle.backbone["head"])

    def embed(ids):
        return embed_table[ids]

    def lm_head(hidden):
        return jnp.einsum("...d,dv->...v", hidden.astype(jnp.float32), head_kernel, precision=jax.lax.Precision.HIGHEST)

    graphdef, state = spx.export(model)

    @jax.jit
    def step(state, tokens, hidden, draft_state):
        return spx.bind(graphdef, state).draft(tokens, hidden, draft_state, embed, lm_head)

    outputs: list[Outputs] = []
    with cfg.mesh:
        draft_state = model.prefill(jnp.asarray(oracle.inputs.prefill_hidden))
        for tokens, hidden in zip(oracle.inputs.tokens, oracle.inputs.step_hidden, strict=True):
            ids, logits, confidence, draft_state = step(
                state, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(hidden), draft_state
            )
            outputs.append(Outputs(np.asarray(ids).astype(np.int64), np.asarray(logits), np.asarray(confidence)))
    assert int(np.asarray(draft_state.position)[0]) == oracle.inputs.prefill_hidden.shape[1] + NUM_STEPS
    return outputs


# ----------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prefill_len", PREFILL_LENS)
def test_draft_parity_with_reference(prefill_len):
    """Greedy ids / logits / confidence of two decode steps match the V4-0731 reference."""
    oracle = _oracle(prefill_len)
    assert oracle.router_gap > MIN_GAP, f"router top-k near-tie in the reference: {oracle.router_gap:.3e}"
    for out in oracle.outputs:
        assert out.logits.std() > 0.3
        top2 = np.sort(out.logits, axis=-1)[..., -2:]
        assert float((top2[..., 1] - top2[..., 0]).min()) > MIN_GAP, "near-tie in the greedy draft arg-max"
    got_all = _run_easydel(oracle.config, _build_easydel(oracle.config, oracle.dspark), oracle)
    for step, (got, want) in enumerate(zip(got_all, oracle.outputs, strict=True)):
        assert got.ids.shape == want.ids.shape == (BATCH, BLOCK + 1)
        assert got.logits.shape == want.logits.shape == (BATCH, BLOCK, VOCAB)
        np.testing.assert_array_equal(got.ids, want.ids, err_msg=f"step {step}")
        np.testing.assert_allclose(got.logits, want.logits, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")
        np.testing.assert_allclose(got.confidence, want.confidence, rtol=RTOL, atol=ATOL, err_msg=f"step {step}")


@pytest.mark.parametrize(
    "name",
    [
        f"stages.{STAGES - 1}.hc_head.hc_fn",  # V4-only learned stream collapse of the last stage
        f"stages.{STAGES - 1}.markov_head.markov_w2.weight",
        "stages.0.main_proj.weight",
    ],
)
def test_negative_control_parameter_moves_logits(name):
    """Perturbing one parameter in the EasyDeL copy only must move the draft logits: the path is exercised."""
    oracle = _oracle(PREFILL_LENS[0])
    rng = np.random.default_rng(7)
    params = dict(oracle.dspark)
    params[name] = params[name] + 0.5 * params[name].std() * rng.standard_normal(params[name].shape).astype(np.float32)
    got = _run_easydel(oracle.config, _build_easydel(oracle.config, params), oracle)
    gap = max(float(np.abs(g.logits - w.logits).max()) for g, w in zip(got, oracle.outputs, strict=True))
    assert gap > 1e-2, f"perturbing {name} moved the logits by only {gap:.3e}"


def test_negative_control_query_head_norm_is_required():
    """Dropping V4's per-head query RMS norm (the V4.1 behavior) breaks parity: the V4-specific path is exercised."""
    oracle = _oracle(PREFILL_LENS[1])
    model = _build_easydel(oracle.config, oracle.dspark)
    for stage in model.stages:
        stage.self_attn.q_head_norm_eps = None
    got = _run_easydel(oracle.config, model, oracle)
    gap = max(float(np.abs(g.logits - w.logits).max()) for g, w in zip(got, oracle.outputs, strict=True))
    assert gap > 1e-2, f"disabling the query head norm moved the logits by only {gap:.3e}"


def _tiny_backbone():
    cfg = ed.DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=512,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        initializer_range=0.2,
        tie_word_embeddings=False,
        dspark_block_size=BLOCK,
        dspark_target_layer_ids=[0, 2],
        dspark_noise_token_id=5,
        dspark_markov_rank=8,
    )
    cfg.moe_force_xla_gmm = True
    return ed.DeepseekV4ForCausalLM(
        config=cfg,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )


def test_backbone_main_hidden_is_target_layer_output_mean():
    """``output_dspark_hidden`` returns the stream mean AFTER each target layer; the default path is unchanged."""
    model = _tiny_backbone()
    ids = jnp.asarray(np.random.default_rng(1).integers(0, VOCAB, size=(1, 12)), dtype=jnp.int32)
    with model.mesh:
        plain = model(input_ids=ids, output_hidden_states=True)
        hooked = model(input_ids=ids, output_hidden_states=True, output_dspark_hidden=True)
    assert getattr(plain, "main_hidden", None) is None
    np.testing.assert_array_equal(np.asarray(hooked.logits), np.asarray(plain.logits))
    # ``hidden_states[i]`` is the stream INPUT of layer i, i.e. the output of layer i - 1.
    want = np.concatenate([np.asarray(plain.hidden_states[i + 1]).mean(axis=2) for i in (0, 2)], axis=-1)
    got = np.asarray(hooked.main_hidden)
    assert got.shape == (1, 12, 2 * 64)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    # Negative control: the INPUT of the target layers (V4.1's capture point) is a different tensor.
    before = np.concatenate([np.asarray(plain.hidden_states[i]).mean(axis=2) for i in (0, 2)], axis=-1)
    assert float(np.abs(got - before).max()) > 1e-2
