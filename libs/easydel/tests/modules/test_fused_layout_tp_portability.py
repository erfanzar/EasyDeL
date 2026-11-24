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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native checkpoints must be tensor-parallel portable.

Fused projections (qkv_proj / gate_up_proj) are TP-interleaved in memory.
A checkpoint saved under one tensor-parallel size and loaded under another
must still produce identical logits — historically it did NOT (the on-disk
layout silently depended on the save mesh), which scrambled Q/K/V and
gate/up for every fused layer while names and shapes still matched. This is
the regression test for that bug (the qwen3.6-27b tp=4 -> tp∈{1,2} incident).

Run on a CPU box as::

    ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \\
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
    pytest libs/easydel/tests/modules/test_fused_layout_tp_portability.py

The env vars must be set in the shell: tests/modules/conftest.py imports
easydel (and thus initializes jax) before this module's setdefault lines run.
"""

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import easydel as ed
import jax
import numpy as np
import optax
import pytest
from jax import numpy as jnp

if jax.device_count() < 4:
    pytest.skip(
        "needs >=4 devices for the tp=4 cases — set "
        "XLA_FLAGS=--xla_force_host_platform_device_count=8 (with JAX_PLATFORMS=cpu) "
        "in the shell before pytest starts",
        allow_module_level=True,
    )

IDS = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.int32)
MODEL_MODULES_ROOT = Path(__file__).resolve().parents[2] / "easydel" / "modules"


def _module_location(path: Path, line: int) -> str:
    return f"{path.relative_to(MODEL_MODULES_ROOT.parent)}:{line}"


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _is_dense_fused_target(target: ast.expr) -> bool:
    return _target_name(target) in {"gate_up_proj", "qkv_proj"}


def _is_column_parallel_linear_call(value: ast.expr) -> bool:
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "ColumnParallelLinear"


def _is_gate_up_call(value: ast.expr) -> bool:
    return _contains_projection_call(value, {"gate_up_proj"})


def _is_qkv_call(value: ast.expr) -> bool:
    return _contains_projection_call(value, {"qkv_proj", "query_key_value_projection"})


def _contains_projection_call(value: ast.expr, names: set[str]) -> bool:
    for node in ast.walk(value):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in names:
            return True
    return False


def _is_jnp_split(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "split"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "jnp"
    )


def test_dense_fused_projection_linears_carry_fused_layout():
    missing = []
    for path in MODEL_MODULES_ROOT.glob("*/modeling_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not any(_is_dense_fused_target(target) for target in node.targets):
                continue
            if not _is_column_parallel_linear_call(node.value):
                continue
            if not any(keyword.arg == "layout" for keyword in node.value.keywords):
                missing.append(_module_location(path, node.lineno))

    assert not missing, "dense fused qkv/gate_up linears must carry a fused layout:\n" + "\n".join(missing)


def test_registered_model_registry_is_covered_by_static_fused_audit():
    import easydel.modules  # noqa: F401
    from easydel.infra.factory import registry

    audited_files = {path.resolve() for path in MODEL_MODULES_ROOT.glob("*/modeling_*.py")}
    unique_model_types = sorted({model_type for task in registry.task_registry.values() for model_type in task})
    assert len(unique_model_types) >= 87, "registry import did not expose the full model surface"

    missing = []
    for task_type, registrations in registry.task_registry.items():
        for model_type, registration in registrations.items():
            module_file = Path(inspect.getfile(registration.module)).resolve()
            if module_file.name.startswith("modeling_") and MODEL_MODULES_ROOT.resolve() in module_file.parents:
                if module_file not in audited_files:
                    missing.append(f"{task_type.value}:{model_type}:{module_file}")

    assert not missing, (
        "registered model implementation files must be included in the fused-layout audit:\n"
        + "\n".join(sorted(missing))
    )


def test_gate_up_projection_outputs_use_layout_aware_split():
    direct_splits = []
    for path in MODEL_MODULES_ROOT.glob("*/modeling_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            gate_up_outputs = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or not _is_gate_up_call(node.value):
                    continue
                gate_up_outputs.update(target.id for target in node.targets if isinstance(target, ast.Name))
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or not _is_jnp_split(node.value):
                    continue
                if (
                    node.value.args
                    and isinstance(node.value.args[0], ast.Name)
                    and node.value.args[0].id in gate_up_outputs
                ):
                    direct_splits.append(_module_location(path, node.lineno))

    assert not direct_splits, "fused gate_up_proj outputs must use split_fused_gate_up_projection:\n" + "\n".join(
        direct_splits
    )


def _name_is_used_as_jnp_split_arg(value: ast.expr, names: set[str]) -> bool:
    return (
        _is_jnp_split(value) and bool(value.args) and isinstance(value.args[0], ast.Name) and value.args[0].id in names
    )


def _contains_direct_name_subscript(value: ast.expr, names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in names
        for node in ast.walk(value)
    )


def _contains_direct_name_reshape(value: ast.expr, names: set[str]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reshape"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in names
        for node in ast.walk(value)
    )


def test_qkv_projection_outputs_use_layout_aware_split():
    direct_ops = []
    for path in MODEL_MODULES_ROOT.glob("*/modeling_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            qkv_outputs = set()
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign) or not _is_qkv_call(node.value):
                    continue
                qkv_outputs.update(target.id for target in node.targets if isinstance(target, ast.Name))
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign):
                    continue
                if (
                    _name_is_used_as_jnp_split_arg(node.value, qkv_outputs)
                    or _contains_direct_name_subscript(node.value, qkv_outputs)
                    or _contains_direct_name_reshape(node.value, qkv_outputs)
                ):
                    direct_ops.append(_module_location(path, node.lineno))

    assert not direct_ops, "fused qkv_proj outputs must use layout-aware split helpers:\n" + "\n".join(direct_ops)


def _config(tp: int) -> "ed.LlamaConfig":
    cfg = ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        mlp_bias=False,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return cfg


def _logits(model) -> np.ndarray:
    import spectrax  # noqa: F401

    with model.mesh:
        out = model(input_ids=jnp.asarray(IDS), attention_mask=jnp.ones_like(jnp.asarray(IDS)))
    return np.asarray(jax.device_get(out.logits)).astype(np.float32)


def _state_logits(state) -> np.ndarray:
    return _logits(state.model)


def _key_to_string(key) -> str:
    return ".".join(str(part) for part in key) if isinstance(key, tuple) else str(key)


def _path_matches_fused_param(path: str, module_path: str) -> bool:
    markers = []
    for suffix in (".weight", ".bias"):
        for prefix in (module_path, f"parameters.{module_path}"):
            marker = f"{prefix}{suffix}"
            markers.extend((marker, f"{marker}.value"))
    return any(path.endswith(marker) for marker in markers)


def _canonical_fused_parameter_leaves(model):
    import spectrax as spx
    from easydel.layers.layouts import canonicalize_fused_state, fused_layout_param_specs
    from easydel.utils.traversals import flatten_dict

    specs = fused_layout_param_specs(model)
    state = spx.export(model)[1].raw()
    flat = flatten_dict(state)
    flat_arrays = {key: value.value if hasattr(value, "value") else value for key, value in flat.items()}
    canonical = canonicalize_fused_state(model, dict(flat_arrays), model.config)
    leaves = {}
    for path, value in canonical.items():
        key = _key_to_string(path)
        if any(_path_matches_fused_param(key, module_path) for module_path, _ in specs):
            leaves[key] = np.asarray(jax.device_get(value)).astype(np.float32)
    assert leaves, "test model should expose fused parameter leaves"
    return leaves


def _runtime_fused_parameter_leaves(model):
    import spectrax as spx
    from easydel.layers.layouts import fused_layout_param_specs
    from easydel.utils.traversals import flatten_dict

    specs = fused_layout_param_specs(model)
    flat = flatten_dict(spx.export(model)[1].raw())
    leaves = {}
    for path, value in flat.items():
        key = _key_to_string(path)
        if any(_path_matches_fused_param(key, module_path) for module_path, _ in specs):
            array = value.value if hasattr(value, "value") else value
            leaves[key] = array
    assert leaves, "test model should expose fused parameter leaves"
    return leaves


def _loss_for_graphstate(state, graphstate):
    model = state.merge(graphstate)
    with model.mesh:
        out = model(input_ids=jnp.asarray(IDS), attention_mask=jnp.ones_like(jnp.asarray(IDS)))
        logits = out.logits[:, :-1, :]
        labels = jnp.asarray(IDS)[:, 1:]
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def _train_step(state):
    loss, grads = jax.value_and_grad(lambda graphstate: _loss_for_graphstate(state, graphstate))(state.graphstate)
    return state.apply_gradients(grads=grads), float(jax.device_get(loss))


def _train_state(family: str, tp: int, *, precision=None):
    model = FAMILIES[family](tp, **({} if precision is None else {"precision": precision}))
    tx = optax.adamw(learning_rate=1e-3)
    state = ed.EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    state, _ = _train_step(state)
    state, _ = _train_step(state)
    return state, tx


def _llama(tp, *, precision=None):
    import spectrax as spx

    return ed.LlamaForCausalLM(
        config=_config(tp), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=precision
    )


def _phi3(tp):
    """Natively-fused qkv_proj + fused gate_up (fuse_qkv_projection=True path)."""
    import spectrax as spx

    cfg = ed.Phi3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.Phi3ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _minimax(tp):
    """Linear-attention qkv_proj + fused MoE gate_up experts."""
    import spectrax as spx

    cfg = ed.MiniMaxConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        block_size=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=["linear_attention"],
        moe_method="standard_moe",
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.MiniMaxForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _openelm(tp):
    """Per-layer dense qkv_proj with non-uniform OpenELM attention metadata."""
    import spectrax as spx

    cfg = ed.OpenELMConfig(
        vocab_size=128,
        max_context_length=64,
        num_transformer_layers=2,
        model_dim=64,
        head_dim=16,
        qkv_multipliers=1.0,
        num_gqa_groups=2,
        ffn_multipliers=1.0,
        ffn_dim_divisor=64,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.OpenELMForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _qwen3_moe(tp, *, precision=None):
    """Fused MoE expert gate_up (ColumnParallelMoELinear) + dense fused layers."""
    import spectrax as spx

    cfg = ed.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        decoder_sparse_step=1,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        # fused_moe's expert all-to-all (ragged-all-to-all) is unimplemented on
        # XLA:CPU; the dispatch engine is irrelevant here — the fused
        # gate_up_proj weights under test are identical either way.
        moe_method="standard_moe",
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.Qwen3MoeForCausalLM(
        config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=precision
    )


def _qwen3_next(tp, *, precision=None):
    """GDR linear attention (packed qkvz/ba) with routed and shared experts."""
    import spectrax as spx

    cfg = ed.Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        moe_method="standard_moe",  # see _qwen3_moe
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.Qwen3NextForCausalLM(
        config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=precision
    )


def _qwen3_next_dense(tp):
    """Dense-MLP Qwen3-Next path used by non-MoE checkpoint round-trip tests."""
    import spectrax as spx

    cfg = ed.Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        mlp_only_layers=[0, 1],
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.Qwen3NextForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


FAMILIES = {
    "llama": _llama,
    "phi3": _phi3,
    "minimax": _minimax,
    "openelm": _openelm,
    "qwen3_moe": _qwen3_moe,
    "qwen3_next": _qwen3_next,
}
TRAIN_STATE_FAMILIES = {"llama": _llama, "qwen3_moe": _qwen3_moe}


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("load_tp", [1, 2, 4])
def test_fused_checkpoint_is_tp_portable(tmp_path, load_tp, family):
    save_tp = 2
    # This structural round-trip compares different TPU contraction shapes.
    # Request the same explicit precision at both endpoints for Qwen3-Next/MoE:
    # float32 storage alone permits reduced-precision products, and the loader
    # normalizes None to DEFAULT instead of inheriting an ambient override.
    precision_kwargs = {"precision": jax.lax.Precision.HIGHEST} if family in {"qwen3_next", "qwen3_moe"} else {}
    model = FAMILIES[family](save_tp, **precision_kwargs)
    reference = _logits(model)
    reference_fused = _canonical_fused_parameter_leaves(model)
    model.save_pretrained(str(tmp_path / "ckpt"))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path / "ckpt"),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
        **precision_kwargs,
    )
    # Check layout integrity before arithmetic parity to distinguish corruption
    # from topology-dependent floating-point execution.
    restored_fused = _canonical_fused_parameter_leaves(loaded)
    assert restored_fused.keys() == reference_fused.keys()
    for path, expected in reference_fused.items():
        fused_err = float(np.max(np.abs(expected - restored_fused[path])))
        assert fused_err < 1e-6, (
            f"[{family}] fused parameter leaf {path} changed after save_tp={save_tp} -> load_tp={load_tp} "
            f"(max|Δ|={fused_err})"
        )

    restored = _logits(loaded)
    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance(family), (
        f"[{family}] save_tp={save_tp} -> load_tp={load_tp} changed the model (max|Δlogits|={err})"
    )


@pytest.mark.parametrize("save_tp", [1, 2, 4])
@pytest.mark.parametrize("load_tp", [1, 2, 4])
def test_fused_checkpoint_save_load_tp_matrix(tmp_path, save_tp, load_tp):
    """The full workflow: weights are saved verbatim in the save mesh's
    interleave with that tp recorded in config.json; any other mesh
    re-interleaves on load and overwrites the recorded tp with its own."""
    import json as _json

    # Compare checkpoint layouts, not TPU DEFAULT's shape-dependent reduced
    # products. Both ends request fp32 products; the global policy is untouched.
    model = _llama(save_tp, precision=jax.lax.Precision.HIGHEST)
    canonical_before = _canonical_fused_parameter_leaves(model)
    reference = _logits(model)
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))

    config_payload = _json.loads((ckpt / "config.json").read_text())
    assert config_payload.get("fused_param_tp") == save_tp

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
    )
    canonical_after = _canonical_fused_parameter_leaves(loaded)
    assert canonical_before.keys() == canonical_after.keys()
    for key in canonical_before:
        np.testing.assert_array_equal(canonical_before[key], canonical_after[key], err_msg=key)
    restored = _logits(loaded)
    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance("llama"), (
        f"save_tp={save_tp} -> load_tp={load_tp} changed the model (max|Δlogits|={err})"
    )
    # the live config is overwritten to describe the layout now in memory
    assert getattr(loaded.config, "fused_param_tp", None) == load_tp


def test_fused_tp_survives_lost_marker(tmp_path):
    """The tp is stamped into config.json, so a checkpoint whose sidecar
    marker is lost (partial copy, failed best-effort write) still
    re-interleaves correctly instead of being misread as legacy."""
    save_tp, load_tp = 2, 4
    model = _llama(save_tp)
    reference = _logits(model)
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))

    (ckpt / "fused_param_layout.json").unlink()
    from easydel.layers.layouts import read_fused_checkpoint_tp

    assert read_fused_checkpoint_tp(ckpt) == save_tp

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
    )
    restored = _logits(loaded)
    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance("llama"), (
        f"markerless checkpoint was not re-interleaved via the config.json tp (max|Δlogits|={err})"
    )


def test_fused_tp_backcompat_canonical_format(tmp_path):
    """Checkpoints written by the previous format (canonical contiguous
    weights + `fused_param_layout: "canonical"`) must keep loading: canonical
    is exactly tp=1."""
    import json as _json

    model = _llama(1)  # tp=1 save: weights on disk are canonical contiguous
    reference = _logits(model)
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))

    # rewrite marker + config.json into the previous format
    (ckpt / "fused_param_layout.json").write_text(_json.dumps({"fused_param_layout": "canonical"}))
    config_payload = _json.loads((ckpt / "config.json").read_text())
    config_payload.pop("fused_param_tp", None)
    config_payload["fused_param_layout"] = "canonical"
    (ckpt / "config.json").write_text(_json.dumps(config_payload))

    from easydel.layers.layouts import read_fused_checkpoint_tp

    assert read_fused_checkpoint_tp(ckpt) == 1

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        auto_shard_model=True,
    )
    restored = _logits(loaded)
    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance("llama"), (
        f"previous-format canonical checkpoint failed to load at tp=2 (max|Δlogits|={err})"
    )
    assert getattr(loaded.config, "fused_param_tp", None) == 2
    assert "fused_param_layout" not in loaded.config.__dict__


def test_unstamped_checkpoint_assumed_canonical(tmp_path):
    """A checkpoint with no fused-tp record anywhere (no marker, no config
    field) loads as canonical tp=1 and re-interleaves for the live mesh.
    The previous verbatim default silently scrambled tp=1 conversions loaded
    under tp>1 (production distill read KL ~ ln(vocab) from exactly this)."""
    import json as _json

    model = _llama(1)  # tp=1 save: weights on disk are canonical contiguous
    reference = _logits(model)
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))

    (ckpt / "fused_param_layout.json").unlink()
    config_payload = _json.loads((ckpt / "config.json").read_text())
    config_payload.pop("fused_param_tp", None)
    config_payload.pop("fused_param_layout", None)
    (ckpt / "config.json").write_text(_json.dumps(config_payload))

    from easydel.layers.layouts import read_fused_checkpoint_tp

    assert read_fused_checkpoint_tp(ckpt) is None

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        auto_shard_model=True,
    )
    restored = _logits(loaded)
    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance("llama"), (
        f"unstamped canonical checkpoint failed to load at tp=2 (max|Δlogits|={err})"
    )
    assert getattr(loaded.config, "fused_param_tp", None) == 2


def test_stamp_and_strip_fused_tp_config_recurse_into_sub_configs():
    from easydel.layers.layouts import stamp_fused_tp_config, strip_fused_tp_config

    cfg = ed.LlamaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1)
    sub = ed.LlamaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1)
    cfg.text_config = sub
    cfg.fused_param_layout = "canonical"  # retired tag must be dropped

    stamp_fused_tp_config(cfg, 4)
    assert cfg.fused_param_tp == 4
    assert sub.fused_param_tp == 4
    assert "fused_param_layout" not in cfg.__dict__

    strip_fused_tp_config(cfg)
    assert "fused_param_tp" not in cfg.__dict__
    assert "fused_param_tp" not in sub.__dict__


def test_fused_params_registry_matches_walker():
    from easydel.layers.layouts import fused_layout_param_specs

    model = _llama(tp=2)
    registry = model.fused_params
    assert registry, "llama exposes fused qkv/gate_up projections"
    assert registry == dict(fused_layout_param_specs(model))
    for path, sizes in registry.items():
        assert isinstance(path, str)
        assert all(isinstance(s, int) and s > 0 for s in sizes)
    assert any(path.endswith("gate_up_proj") for path in registry)
    assert any(path.endswith("qkv_proj") for path in registry)


def test_to_torch_refuses_mismatched_fused_tp():
    model = _llama(tp=2)
    model.config.fused_param_tp = 4  # weights claim tp=4 interleave, mesh says 2
    with pytest.raises(RuntimeError, match="fused_param_tp"):
        model.to_torch()


def test_to_torch_after_cross_tp_load_matches(tmp_path):
    """The full user workflow, export leg: save at tp=2, re-load at tp=4
    (retp re-interleaves), then to_torch from the tp=4 mesh. The HF tensors
    must be bit-identical to exporting the original tp=2 model — i.e. the
    exporter de-interleaves with the LOADED tp, not some stale one."""
    pytest.importorskip("torch")
    import torch

    model = _llama(2)
    hf_reference = model.to_torch().state_dict()
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 4, 1),
        auto_shard_model=True,
    )
    assert getattr(loaded.config, "fused_param_tp", None) == 4
    hf_exported = loaded.to_torch().state_dict()

    assert hf_exported.keys() == hf_reference.keys()
    for key, reference in hf_reference.items():
        assert torch.equal(reference, hf_exported[key]), (
            f"{key} differs between tp=2 export and export after tp=4 re-load; "
            "the exporter did not de-interleave with the loaded tp"
        )


def test_prefused_hf_checkpoint_roundtrips_at_tp2():
    """Phi-3-style HF checkpoints store qkv_proj/gate_up_proj pre-fused
    (contiguous). At tp>1 the loader must interleave them and the exporter
    must de-interleave back; at tp=1 both transforms are the identity, so
    only a tp>1 run catches this corruption class (same shapes, scrambled
    columns)."""
    pytest.importorskip("torch")
    import torch

    reference = _phi3(1)
    # Phi3's default pad_token_id (32000) exceeds the tiny test vocab; the HF
    # torch constructor asserts padding_idx < num_embeddings.
    reference.config.pad_token_id = 0
    reference_logits = _logits(reference)
    hf_reference = reference.to_torch()
    hf_state = hf_reference.state_dict()

    loaded = _phi3(2)
    loaded.config.pad_token_id = 0
    loaded = ed.traversals.merge_model_and_tree(loaded, tree=loaded.transform_fn(hf_state))
    loaded.eval()
    err = float(np.max(np.abs(reference_logits - _logits(loaded))))
    assert err < _model_logit_tolerance("phi3"), (
        f"pre-fused HF checkpoint loaded at tp=2 diverged (max|Δlogits|={err}); "
        "fused qkv/gate_up were not TP-interleaved on load"
    )

    hf_exported = loaded.to_torch().state_dict()
    fused_keys = [k for k in hf_state if "qkv_proj" in k or "gate_up_proj" in k]
    assert fused_keys, "phi3 HF state dict should contain pre-fused projections"
    for key in fused_keys:
        assert torch.equal(hf_state[key], hf_exported[key]), (
            f"{key} changed across the tp=2 import/export round-trip; "
            "the exporter did not de-interleave the pre-fused tensor"
        )


def test_prefused_hf_checkpoint_tp2_save_tp4_load_exports_to_torch(tmp_path):
    """Exact torch -> EasyDeL tp2 save -> EasyDeL tp4 load -> torch workflow."""
    pytest.importorskip("torch")
    import torch

    reference = _phi3(1)
    reference.config.pad_token_id = 0
    hf_state = reference.to_torch().state_dict()

    loaded_tp2 = _phi3(2)
    loaded_tp2.config.pad_token_id = 0
    loaded_tp2 = ed.traversals.merge_model_and_tree(loaded_tp2, tree=loaded_tp2.transform_fn(hf_state))
    loaded_tp2.eval()
    ckpt = tmp_path / "ckpt"
    loaded_tp2.save_pretrained(str(ckpt))

    loaded_tp4 = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 4, 1),
        auto_shard_model=True,
    )
    assert getattr(loaded_tp4.config, "fused_param_tp", None) == 4
    hf_exported = loaded_tp4.to_torch().state_dict()

    assert hf_exported.keys() == hf_state.keys()
    for key, reference_tensor in hf_state.items():
        assert torch.equal(reference_tensor, hf_exported[key]), (
            f"{key} changed across torch -> EasyDeL tp2 save -> EasyDeL tp4 load -> torch"
        )


def test_qwen3_next_dense_torch_tp2_save_tp4_load_exports_to_torch(tmp_path):
    """Dense Qwen3-Next torch -> EasyDeL tp2 save -> EasyDeL tp4 load -> torch."""
    pytest.importorskip("torch")
    import torch

    reference = _qwen3_next_dense(1)
    hf_state = reference.to_torch().state_dict()
    hf_reference = {key: value.detach().clone() for key, value in hf_state.items()}

    loaded_tp2 = _qwen3_next_dense(2)
    loaded_tp2 = ed.traversals.merge_model_and_tree(loaded_tp2, tree=loaded_tp2.transform_fn(dict(hf_state)))
    loaded_tp2.eval()
    ckpt = tmp_path / "ckpt"
    loaded_tp2.save_pretrained(str(ckpt))

    loaded_tp4 = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 4, 1),
        auto_shard_model=True,
    )
    assert getattr(loaded_tp4.config, "fused_param_tp", None) == 4
    hf_exported = loaded_tp4.to_torch().state_dict()

    assert hf_exported.keys() == hf_reference.keys()
    for key, reference_tensor in hf_reference.items():
        assert torch.equal(reference_tensor, hf_exported[key]), (
            f"{key} changed across dense Qwen3-Next torch -> EasyDeL tp2 save -> EasyDeL tp4 load -> torch"
        )


def _canonical_fused_optimizer_leaves(state):
    from easydel.layers.layouts import canonicalize_fused_optimizer_state, fused_layout_param_specs
    from easydel.utils.traversals import flatten_tree

    model = state.model
    specs = fused_layout_param_specs(model)
    canonical = canonicalize_fused_optimizer_state(model, state.opt_state, model.config)
    flat = flatten_tree(canonical, sep=".")
    leaves = {}
    for path, value in flat.items():
        if any(_path_matches_fused_param(path, module_path) for module_path, _ in specs):
            leaves[path] = np.asarray(jax.device_get(value)).astype(np.float32)
    assert leaves, "test model should expose fused optimizer leaves"
    return leaves


def _train_state_logit_tolerance(family: str) -> float:
    # Trained Qwen3-MoE checkpoints have a slightly higher same-TP CPU numeric
    # floor than the init-only portability matrix above. Keep optimizer moments
    # strict below; this tolerance only gates logits/loss smoothness.
    return 1e-3 if family == "qwen3_moe" else 5e-4


def _model_logit_tolerance(family: str) -> float:
    return 3e-3 if family == "minimax" else 5e-4


def test_parameter_value_leaf_paths_are_layout_portable():
    from easydel.layers.layouts import canonicalize_fused_state, runtimeize_fused_state

    model = _llama(tp=2)
    runtime_leaves = _runtime_fused_parameter_leaves(model)
    canonical_leaves = _canonical_fused_parameter_leaves(model)

    value_key_state = {f"{path}.value": value for path, value in runtime_leaves.items()}
    canonical = canonicalize_fused_state(model, dict(value_key_state), model.config)
    for path, expected in canonical_leaves.items():
        np.testing.assert_allclose(np.asarray(jax.device_get(canonical[f"{path}.value"])), expected)

    restored = runtimeize_fused_state(model, canonical, model.config)
    for path, expected in runtime_leaves.items():
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored[f"{path}.value"])),
            np.asarray(jax.device_get(expected)),
        )


def _custom_optimizer_state(model):
    runtime_leaves = _runtime_fused_parameter_leaves(model)
    canonical_leaves = _canonical_fused_parameter_leaves(model)
    first_runtime_leaf = next(iter(runtime_leaves.values()))
    unrelated = jnp.arange(first_runtime_leaf.size, dtype=first_runtime_leaf.dtype).reshape(first_runtime_leaf.shape)

    custom_optimizer_state = {
        "slots_by_param": {
            path: {
                "ema": value,
                "history": [value * 2],
            }
            for path, value in runtime_leaves.items()
        },
        "flat_alias_slots": {f"custom_optimizer.{path}.velocity": value + 1 for path, value in runtime_leaves.items()},
        "value_wrapped_slots": {
            f"{path}.value": {
                "ema": value + 3,
                "history": [value * 4],
            }
            for path, value in runtime_leaves.items()
        },
        "flat_alias_value_slots": {
            f"custom_optimizer.{path}.value.velocity": value + 5 for path, value in runtime_leaves.items()
        },
        "unrelated_same_shape": unrelated,
    }
    return custom_optimizer_state, canonical_leaves, runtime_leaves, unrelated


def _assert_custom_optimizer_is_canonical(opt_state, expected_canonical, unrelated):
    for path, expected in expected_canonical.items():
        np.testing.assert_allclose(np.asarray(jax.device_get(opt_state["slots_by_param"][path]["ema"])), expected)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(opt_state["slots_by_param"][path]["history"][0])),
            expected * 2,
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(opt_state["flat_alias_slots"][f"custom_optimizer.{path}.velocity"])),
            expected + 1,
        )
        value_path = f"{path}.value"
        np.testing.assert_allclose(
            np.asarray(jax.device_get(opt_state["value_wrapped_slots"][value_path]["ema"])),
            expected + 3,
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(opt_state["value_wrapped_slots"][value_path]["history"][0])),
            expected * 4,
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(opt_state["flat_alias_value_slots"][f"custom_optimizer.{value_path}.velocity"])),
            expected + 5,
        )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(opt_state["unrelated_same_shape"])),
        np.asarray(jax.device_get(unrelated)),
    )


def test_custom_optimizer_slots_are_layout_portable():
    from easydel.layers.layouts import canonicalize_fused_optimizer_state, runtimeize_fused_optimizer_state

    model = _llama(tp=2)
    custom_optimizer_state, canonical_leaves, runtime_leaves, unrelated = _custom_optimizer_state(model)

    canonical = canonicalize_fused_optimizer_state(model, custom_optimizer_state, model.config)
    _assert_custom_optimizer_is_canonical(canonical, canonical_leaves, unrelated)

    restored = runtimeize_fused_optimizer_state(model, canonical, model.config)
    for path, expected in runtime_leaves.items():
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["slots_by_param"][path]["ema"])),
            np.asarray(jax.device_get(expected)),
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["slots_by_param"][path]["history"][0])),
            np.asarray(jax.device_get(expected * 2)),
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["flat_alias_slots"][f"custom_optimizer.{path}.velocity"])),
            np.asarray(jax.device_get(expected + 1)),
        )
        value_path = f"{path}.value"
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["value_wrapped_slots"][value_path]["ema"])),
            np.asarray(jax.device_get(expected + 3)),
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["value_wrapped_slots"][value_path]["history"][0])),
            np.asarray(jax.device_get(expected * 4)),
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored["flat_alias_value_slots"][f"custom_optimizer.{value_path}.velocity"])),
            np.asarray(jax.device_get(expected + 5)),
        )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(restored["unrelated_same_shape"])),
        np.asarray(jax.device_get(unrelated)),
    )


def test_custom_optimizer_state_checkpoint_is_tp_portable(tmp_path):
    from easydel.layers.layouts import canonicalize_fused_optimizer_state, read_fused_checkpoint_tp

    save_model = _llama(tp=2)
    custom_optimizer_state, canonical_leaves, _, unrelated = _custom_optimizer_state(save_model)
    state = ed.EasyDeLState.create(model=save_model, tx=None, opt_state=custom_optimizer_state)

    ckpt = tmp_path / "custom-optimizer-ckpt"
    state.save_state(ckpt, float_dtype=None, save_optimizer=True)
    assert read_fused_checkpoint_tp(ckpt) == 2

    loaded = ed.EasyDeLState.load_state(
        load_directory=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 4, 1),
        auto_shard_model=True,
        tx_template=None,
    )
    assert loaded.opt_state is not None
    canonical = canonicalize_fused_optimizer_state(loaded.model, loaded.opt_state, loaded.model.config)
    _assert_custom_optimizer_is_canonical(canonical, canonical_leaves, unrelated)


def _train_state_next_logit_tolerance(family: str) -> float:
    return 2.5e-3 if family == "qwen3_moe" else _train_state_logit_tolerance(family)


@pytest.mark.parametrize("family", list(TRAIN_STATE_FAMILIES))
@pytest.mark.parametrize("load_tp", [1, 2, 4])
def test_fused_train_state_checkpoint_is_tp_portable(tmp_path, load_tp, family):
    from easydel.layers.layouts import read_fused_checkpoint_tp

    save_tp = 2
    # Keep the trained-MoE comparison on the same explicit precision policy at
    # both ends, rather than measuring TPU DEFAULT's shape-dependent products.
    precision = jax.lax.Precision.HIGHEST if family == "qwen3_moe" else None
    state, tx = _train_state(family, save_tp, precision=precision)
    reference_logits = _state_logits(state)
    reference_params = _canonical_fused_parameter_leaves(state.model)
    reference_opt = _canonical_fused_optimizer_leaves(state)

    ckpt = tmp_path / "trainer-ckpt"
    state.save_state(ckpt, float_dtype=None, save_optimizer=True)
    assert read_fused_checkpoint_tp(ckpt) == save_tp

    loaded = ed.EasyDeLState.load_state(
        load_directory=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=precision,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
        tx_template=tx,
    ).replace(tx=tx)

    restored_params = _canonical_fused_parameter_leaves(loaded.model)
    assert restored_params.keys() == reference_params.keys()
    for path, expected in reference_params.items():
        np.testing.assert_array_equal(restored_params[path], expected, err_msg=path)
    restored_opt = _canonical_fused_optimizer_leaves(loaded)
    assert restored_opt.keys() == reference_opt.keys()
    for path, expected in reference_opt.items():
        err = float(np.max(np.abs(expected - restored_opt[path])))
        assert err < 1e-6, (
            f"[{family}] optimizer leaf {path} changed after canonical/runtime round-trip "
            f"save_tp={save_tp} -> load_tp={load_tp} (max|Δ|={err})"
        )

    restored_logits = _state_logits(loaded)
    logits_err = float(np.max(np.abs(reference_logits - restored_logits)))
    logit_tol = _train_state_logit_tolerance(family)
    direction = f"[{family}] train-state save_tp={save_tp} -> load_tp={load_tp}"
    msg = f"{direction} changed logits (max|Δlogits|={logits_err})"
    assert logits_err < logit_tol, msg

    reference_next, reference_loss = _train_step(state)
    loaded_next, loaded_loss = _train_step(loaded)
    assert np.isfinite(loaded_loss)
    assert abs(reference_loss - loaded_loss) < logit_tol

    next_logits_err = float(np.max(np.abs(_state_logits(reference_next) - _state_logits(loaded_next))))
    assert next_logits_err < _train_state_next_logit_tolerance(family), (
        f"[{family}] one-step resume diverged after save_tp={save_tp} -> load_tp={load_tp} "
        f"(max|Δlogits|={next_logits_err})"
    )


def _qwen3_next_linear_variant(
    tp: int,
    *,
    separate_proj: bool = False,
    merged_split_proj: bool = False,
    num_k_heads: int = 8,
    num_v_heads: int = 8,
    head_dim: int = 8,
    hidden_size: int = 64,
    intermediate_size: int = 128,
):
    import spectrax as spx

    cfg = ed.Qwen3NextConfig(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=head_dim,
        max_position_embeddings=64,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        linear_attention_separate_proj=separate_proj,
        linear_attention_merged_split_proj=merged_split_proj,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
        mlp_only_layers=[0, 1],
    )
    cfg.sharding_axis_dims = (1, 1, -1, 1, tp, 1)
    return ed.Qwen3NextForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


@pytest.mark.parametrize(
    ("save_tp", "load_tp"),
    [(1, 2), (1, 4), (1, 8), (2, 1), (2, 4), (2, 8), (4, 1), (4, 2), (4, 8), (8, 1), (8, 2), (8, 4)],
)
def test_qwen3_next_merged_split_proj_tp_portability(tmp_path, save_tp, load_tp):
    """Merged-split Qwen3-Next (2D in_proj_qkvz/in_proj_ba + 3D conv1d) is portable across TP=1/2/4/8."""
    if jax.device_count() < max(save_tp, load_tp):
        pytest.skip(f"needs >= {max(save_tp, load_tp)} devices")

    from easydel.layers.layouts import fused_layout_param_specs, read_fused_checkpoint_tp

    save_model = _qwen3_next_linear_variant(save_tp, merged_split_proj=True)
    spec_paths = {path for path, _ in fused_layout_param_specs(save_model)}
    assert any(p.endswith(".conv1d") for p in spec_paths)
    assert any(p.endswith(".in_proj_qkvz") for p in spec_paths)
    assert any(p.endswith(".in_proj_ba") for p in spec_paths)

    ref_logits = _logits(save_model)
    ref_canon = _canonical_fused_parameter_leaves(save_model)
    assert any(p.endswith(".conv1d.weight") and arr.ndim == 3 for p, arr in ref_canon.items())

    ckpt = tmp_path / f"merged-tp{save_tp}-to-tp{load_tp}"
    save_model.save_pretrained(str(ckpt))
    assert read_fused_checkpoint_tp(ckpt) == save_tp

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
    )
    assert getattr(loaded.config, "fused_param_tp", None) == load_tp

    loaded_canon = _canonical_fused_parameter_leaves(loaded)
    assert loaded_canon.keys() == ref_canon.keys()
    for path, expected in ref_canon.items():
        np.testing.assert_array_equal(loaded_canon[path], expected, err_msg=path)

    logits_err = float(np.max(np.abs(ref_logits - _logits(loaded))))
    assert logits_err < 5e-4, f"merged_split_proj tp={save_tp}->{load_tp} changed logits (max|Δ|={logits_err})"


@pytest.mark.parametrize("load_tp", [1, 4, 8])
def test_qwen3_next_merged_split_proj_optimizer_state_tp_portability(tmp_path, load_tp):
    """3D conv1d and 2D in_proj_qkvz/in_proj_ba optimizer moments stay aligned across TP=2 -> 1/4/8."""
    if jax.device_count() < load_tp:
        pytest.skip(f"needs >= {load_tp} devices")

    save_model = _qwen3_next_linear_variant(2, merged_split_proj=True)
    tx = optax.adam(1e-3)
    state = ed.EasyDeLState.create(model=save_model, tx=tx, init_opt_state=True)
    state, _ = _train_step(state)
    ref_logits = _state_logits(state)
    ref_opt = _canonical_fused_optimizer_leaves(state)
    assert any(".conv1d.weight" in p for p in ref_opt)

    ckpt = tmp_path / f"merged-opt-tp2-to-tp{load_tp}"
    state.save_state(ckpt, float_dtype=None, save_optimizer=True)

    loaded = ed.EasyDeLState.load_state(
        load_directory=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
        tx_template=tx,
    ).replace(tx=tx)

    loaded_opt = _canonical_fused_optimizer_leaves(loaded)
    assert loaded_opt.keys() == ref_opt.keys()
    for path, expected in ref_opt.items():
        assert float(np.max(np.abs(expected - loaded_opt[path]))) < 1e-6, path
    assert float(np.max(np.abs(ref_logits - _state_logits(loaded)))) < 5e-4


@pytest.mark.parametrize(("save_tp", "load_tp"), [(1, 2), (2, 2), (2, 4), (2, 1), (4, 2)])
def test_qwen3_next_separate_to_merged_cross_mode_tp_portability(tmp_path, save_tp, load_tp):
    """Saved separate_proj=True at any TP loads cleanly into merged_split_proj=True at any TP (incl. same TP>1)."""
    save_model = _qwen3_next_linear_variant(save_tp, separate_proj=True)
    ref_logits = _logits(save_model)

    ckpt = tmp_path / f"sep-tp{save_tp}-to-merged-tp{load_tp}"
    save_model.save_pretrained(str(ckpt))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        config_kwargs={"linear_attention_separate_proj": False, "linear_attention_merged_split_proj": True},
        auto_shard_model=True,
    )
    assert loaded.model.layers[0].linear_attn.uses_merged_split_proj is True
    logits_err = float(np.max(np.abs(ref_logits - _logits(loaded))))
    assert logits_err < 5e-4, f"separate(tp={save_tp})->merged(tp={load_tp}) changed logits (max|Δ|={logits_err})"


@pytest.mark.parametrize("load_tp", [1, 2, 4])
def test_qwen3_next_merged_to_separate_cross_mode_tp_portability(tmp_path, load_tp):
    """Saved merged_split_proj=True at TP=2 unfuses cleanly into separate_proj=True at TP=1/2/4."""
    save_model = _qwen3_next_linear_variant(2, merged_split_proj=True)
    ref_logits = _logits(save_model)

    ckpt = tmp_path / f"merged-tp2-to-sep-tp{load_tp}"
    save_model.save_pretrained(str(ckpt))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        config_kwargs={"linear_attention_merged_split_proj": False, "linear_attention_separate_proj": True},
        auto_shard_model=True,
    )
    assert loaded.model.layers[0].linear_attn.uses_split_proj is True
    logits_err = float(np.max(np.abs(ref_logits - _logits(loaded))))
    assert logits_err < 5e-4, f"merged(tp=2)->separate(tp={load_tp}) changed logits (max|Δ|={logits_err})"


def test_qwen3_next_merged_to_can_merge_tp_fallback_unfusion(tmp_path):
    """Saved merged_split_proj=True at TP=2 unfuses cleanly when loaded at TP=8 where num_k_heads=4 < TP."""
    if jax.device_count() < 8:
        pytest.skip("needs >= 8 devices")

    save_model = _qwen3_next_linear_variant(2, merged_split_proj=True, num_k_heads=4, num_v_heads=4)
    ref_logits = _logits(save_model)

    ckpt = tmp_path / "merged-tp2-to-fallback-tp8"
    save_model.save_pretrained(str(ckpt))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 8, 1),
        auto_shard_model=True,
    )
    assert loaded.model.layers[0].linear_attn.uses_merged_split_proj is False
    assert loaded.model.layers[0].linear_attn.uses_split_proj is True
    logits_err = float(np.max(np.abs(ref_logits - _logits(loaded))))
    assert logits_err < 5e-4, f"merged(tp=2)->fallback(tp=8) changed logits (max|Δ|={logits_err})"


def test_qwen3_next_merged_vs_default_packed_mismatch_raises(tmp_path):
    """Loading merged_split_proj checkpoint as default packed (or vice versa) raises ValueError instead of corrupting."""
    merged = _qwen3_next_linear_variant(2, merged_split_proj=True)
    merged_ckpt = tmp_path / "merged-ckpt"
    merged.save_pretrained(str(merged_ckpt))
    with pytest.raises(ValueError, match="linear_attention_merged_split_proj"):
        ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(merged_ckpt),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            sharding_axis_dims=(1, 1, -1, 1, 2, 1),
            config_kwargs={"linear_attention_merged_split_proj": False, "linear_attention_separate_proj": False},
            auto_shard_model=True,
        )

    packed = _qwen3_next_linear_variant(2, merged_split_proj=False, separate_proj=False)
    packed_ckpt = tmp_path / "packed-ckpt"
    packed.save_pretrained(str(packed_ckpt))
    with pytest.raises(ValueError, match="linear_attention_merged_split_proj"):
        ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(packed_ckpt),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            sharding_axis_dims=(1, 1, -1, 1, 2, 1),
            config_kwargs={"linear_attention_merged_split_proj": True, "linear_attention_separate_proj": False},
            auto_shard_model=True,
        )


def test_quantized_fused_cross_tp_portability_and_indivisible_guard(tmp_path):
    """Quantized fused checkpoints transform word/group-scaled leaves across TP or fail loudly when indivisible."""
    from easydel.layers.layouts import retp_fused_state
    from easydel.layers.quantization import QuantizationConfig, QuantizationType

    # 1. Channelwise quantized merged-split Qwen3-Next: TP=2 -> TP=1 and TP=4.
    cw_cfg = QuantizationConfig.for_matmul("w8a16")
    cw_model = _qwen3_next_linear_variant(2, merged_split_proj=True).quantize(
        quantization_config=cw_cfg, apply_quantization=True, verbose=False
    )
    cw_ref_logits = _logits(cw_model)
    cw_ckpt = tmp_path / "cw-merged-tp2"
    cw_model.save_pretrained(str(cw_ckpt))

    for load_tp in (1, 4):
        cw_loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=str(cw_ckpt),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
            quantization_config=cw_cfg,
            auto_shard_model=True,
        )
        assert float(np.max(np.abs(cw_ref_logits - _logits(cw_loaded)))) < 5e-4

    # 2. Packed grouped (NF4, group_size=16) Llama: qkv/gate_up segments all divide tp*16 (and tp*8).
    nf4_cfg = QuantizationConfig(dtype=QuantizationType.NF4, group_size=16)
    nf4_model = _llama(tp=2).quantize(quantization_config=nf4_cfg, apply_quantization=True, verbose=False)
    nf4_ref_logits = _logits(nf4_model)
    nf4_ckpt = tmp_path / "nf4-llama-tp2"
    nf4_model.save_pretrained(str(nf4_ckpt))

    nf4_tp1 = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(nf4_ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        quantization_config=nf4_cfg,
        auto_shard_model=True,
    )
    assert float(np.max(np.abs(nf4_ref_logits - _logits(nf4_tp1)))) < 5e-4

    # 3. Indivisible packed quantized scales (segments=(32, 32), stride=64 -> shape[-1]=1) must raise ValueError.
    class _DummyFused:
        def __init__(self):
            self.fused_params = {"proj": (32, 32)}

    indivisible_state = {
        ("parameters", "proj", "quant_kernel"): jnp.zeros((16, 8), dtype=jnp.uint32),
        ("parameters", "proj", "quant_scales"): jnp.ones((16, 1), dtype=jnp.float32),
    }
    with pytest.raises(ValueError, match="Cannot re-interleave packed quantized fused leaf"):
        retp_fused_state(_DummyFused(), indivisible_state, saved_tp=1, target_tp=2)

    # 4. Forced-layout TPU shard spec resolution for tiny grouped-quantized projections (e.g. in_proj_ba [H, 64],
    # group_size=64 -> scales [H, 1]) on TP=4 must replicate kernel & scales instead of returning mismatched specs.
    import spectrax as spx
    from easydel.layers.linears._linear_quantized import ColumnParallelLinearQuantized, RowParallelLinearQuantized

    ba_layer = ColumnParallelLinearQuantized(
        in_features=64,
        out_features=64,
        use_bias=False,
        config=QuantizationConfig(dtype=QuantizationType.INT8, group_size=64),
        rngs=spx.Rngs(0),
    )
    cfg_tp4 = _config(tp=4)
    resolved = ba_layer._resolve_shard_specs(
        cfg_tp4.mesh,
        jnp.zeros((2, 64), dtype=jnp.float32),
        ba_layer.quant_kernel.value,
        ba_layer.quant_scales.value,
        ba_layer.quant_biases.value,
        is_tpu=True,
    )
    assert resolved is not None
    _, kernel_spec, scale_spec, bias_spec, _, tp_axis_name = resolved
    assert kernel_spec == scale_spec
    assert bias_spec == kernel_spec
    assert tp_axis_name is None

    # 5. RowParallelLinearQuantized with in_features indivisible by TP=4 (both kernel & scales sanitize to
    # replicated) must ALSO clear tp_axis_name=None so _local_matmul does not psum replicated outputs by tp_size.
    row_indivisible = RowParallelLinearQuantized(
        in_features=62,
        out_features=64,
        use_bias=False,
        config=QuantizationConfig(dtype=QuantizationType.INT8, group_size=64),
        rngs=spx.Rngs(0),
    )
    resolved_row = row_indivisible._resolve_shard_specs(
        cfg_tp4.mesh,
        jnp.zeros((2, 62), dtype=jnp.float32),
        row_indivisible.quant_kernel.value,
        row_indivisible.quant_scales.value,
        row_indivisible.quant_biases.value,
        is_tpu=True,
    )
    assert resolved_row is not None
    _, r_kspec, r_sspec, r_bspec, _, r_tp = resolved_row
    assert r_kspec == jax.sharding.PartitionSpec(None, None)
    assert r_sspec == jax.sharding.PartitionSpec(None, None)
    assert r_bspec == jax.sharding.PartitionSpec(None, None)
    assert r_tp is None

    # 6. ColumnParallelLinearQuantized where BOTH kernel and scales sanitize to replicated (out=16, bits=4 ->
    # kernel (H, 2), scales (H, 1) on TP=4) must also clear tp_axis_name=None.
    col_both_rep = ColumnParallelLinearQuantized(
        in_features=64,
        out_features=16,
        use_bias=False,
        config=QuantizationConfig(dtype=QuantizationType.NF4, group_size=16),
        rngs=spx.Rngs(0),
    )
    resolved_col = col_both_rep._resolve_shard_specs(
        cfg_tp4.mesh,
        jnp.zeros((2, 64), dtype=jnp.float32),
        col_both_rep.quant_kernel.value,
        col_both_rep.quant_scales.value,
        col_both_rep.quant_biases.value,
        is_tpu=True,
    )
    assert resolved_col is not None
    _, c_kspec, c_sspec, _, _, c_tp = resolved_col
    assert c_kspec == jax.sharding.PartitionSpec(None, None)
    assert c_sspec == jax.sharding.PartitionSpec(None, None)
    assert c_tp is None


def test_axis_policy_and_config_guards(tmp_path):
    """axis_policy={'hidden_state_axis': None} survives base_config, AutoConfig, AutoShardAndGather, and from_pretrained."""
    import json as _json

    from easydel.layers.layouts._canonical import _read_saved_linear_attention_flags
    from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig, AutoShardAndGatherFunctions
    from easydel.modules.qwen3_5_moe.qwen3_5_moe_configuration import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

    # Qwen3.5-MoE must allow linear_attention_merged_split_proj=True without colliding with separate_proj,
    # and must reject passing both flags as True simultaneously.
    q35_moe_cfg = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_attention_merged_split_proj=True,
    )
    assert q35_moe_cfg.linear_attention_merged_split_proj is True
    assert q35_moe_cfg.linear_attention_separate_proj is False

    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_attention_separate_proj=True,
            linear_attention_merged_split_proj=True,
        )

    # Composite VLM config: single-flag override via AutoEasyDeLConfig.from_pretrained must update text_config
    # even when saved text_config has linear_attention_separate_proj=True, and _read_saved_linear_attention_flags
    # must prefer text_config over conflicting top-level flags.
    vlm_dir = tmp_path / "qwen3_5_moe_vlm_cfg"
    vlm_cfg = Qwen3_5MoeConfig(
        text_config=Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_attention_separate_proj=True,
        )
    )
    vlm_cfg.save_pretrained(str(vlm_dir))
    vlm_loaded_cfg = AutoEasyDeLConfig.from_pretrained(
        str(vlm_dir),
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        linear_attention_merged_split_proj=True,
    )
    assert vlm_loaded_cfg.text_config.linear_attention_merged_split_proj is True
    assert vlm_loaded_cfg.text_config.linear_attention_separate_proj is False

    # Conflicting top-level flag must not shadow text_config in _read_saved_linear_attention_flags.
    vlm_json = _json.loads((vlm_dir / "config.json").read_text())
    vlm_json["linear_attention_merged_split_proj"] = False
    vlm_json["text_config"]["linear_attention_merged_split_proj"] = True
    vlm_json["text_config"]["linear_attention_separate_proj"] = False
    (vlm_dir / "config.json").write_text(_json.dumps(vlm_json))
    saved_merged, saved_sep, found = _read_saved_linear_attention_flags(vlm_dir)
    assert (saved_merged, saved_sep, found) == (True, False, True)

    # add_basic_configurations with partition_axis=None + axis_policy must NOT wipe axis_policy.
    cfg = _config(tp=2)
    cfg.add_basic_configurations(axis_policy={"hidden_state_axis": None}, partition_axis=None)
    assert cfg.partition_axis is not None
    assert cfg.partition_axis.hidden_state_axis is None
    assert cfg.axis_policy.hidden_state_axis is None

    # Save a native checkpoint and verify AutoEasyDeLConfig, AutoShardAndGatherFunctions, and from_pretrained.
    model = _llama(tp=2)
    ckpt = tmp_path / "axis-policy-ckpt"
    model.save_pretrained(str(ckpt))

    auto_cfg = AutoEasyDeLConfig.from_pretrained(
        str(ckpt),
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        axis_policy={"hidden_state_axis": None},
    )
    assert auto_cfg.partition_axis.hidden_state_axis is None
    assert auto_cfg.axis_policy.hidden_state_axis is None

    shard_fns, _ = AutoShardAndGatherFunctions.from_pretrained(
        str(ckpt),
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        config_kwargs={
            "axis_policy": {"hidden_state_axis": None},
            "sharding_axis_dims": (1, 1, -1, 1, 2, 1),
            "from_torch": False,
            "trust_remote_code": False,
            "model_task": ed.TaskType.CAUSAL_LM,
        },
    )
    assert shard_fns

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        config_kwargs={
            "axis_policy": {"hidden_state_axis": None},
            "sharding_axis_dims": (1, 1, -1, 1, 2, 1),
            "platform": "jax",
        },
        auto_shard_model=True,
    )
    assert loaded.config.partition_axis.hidden_state_axis is None
    assert loaded.config.axis_policy.hidden_state_axis is None

    # Explicit partition_axis + config_kwargs={"partition_axis": ...} must not raise duplicate-kwarg TypeError.
    loaded_override = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        partition_axis=ed.PartitionAxis(),
        config_kwargs={"partition_axis": ed.PartitionAxis(hidden_state_axis=None)},
        auto_shard_model=True,
    )
    assert loaded_override.config.partition_axis.hidden_state_axis is None

    # HF from_torch=True must also preserve axis_policy in config_kwargs and stamp fused_param_tp.
    pytest.importorskip("torch")
    hf_dir = tmp_path / "hf-axis-policy-ckpt"
    model.to_torch().save_pretrained(str(hf_dir))
    loaded_hf = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(hf_dir),
        from_torch=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        config_kwargs={"axis_policy": {"hidden_state_axis": None}},
        auto_shard_model=True,
    )
    assert loaded_hf.config.partition_axis.hidden_state_axis is None
    assert loaded_hf.config.axis_policy.hidden_state_axis is None
    assert getattr(loaded_hf.config, "fused_param_tp", None) == 2


def test_single_flag_linear_attention_override_and_optimizer_fallback_rejection(tmp_path):
    """Single-flag linear-attention config_kwargs work both ways; cross-mode fallback rejects stale optimizer state."""
    import json as _json

    # 1. Saved separate_proj=True -> single-flag config_kwargs={"linear_attention_merged_split_proj": True}
    sep_model = _qwen3_next_linear_variant(2, separate_proj=True)
    sep_ref = _logits(sep_model)
    sep_ckpt = tmp_path / "single-flag-sep"
    sep_model.save_pretrained(str(sep_ckpt))

    loaded_merged = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(sep_ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        config_kwargs={"linear_attention_merged_split_proj": True},
        auto_shard_model=True,
    )
    assert loaded_merged.model.layers[0].linear_attn.uses_merged_split_proj is True
    assert float(np.max(np.abs(sep_ref - _logits(loaded_merged)))) < 5e-4

    # 2. Saved merged_split_proj=True -> single-flag config_kwargs={"linear_attention_separate_proj": True}
    merged_ckpt = tmp_path / "single-flag-merged"
    loaded_merged.save_pretrained(str(merged_ckpt))
    loaded_sep = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(merged_ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        config_kwargs={"linear_attention_separate_proj": True},
        auto_shard_model=True,
    )
    assert loaded_sep.model.layers[0].linear_attn.uses_split_proj is True
    assert float(np.max(np.abs(sep_ref - _logits(loaded_sep)))) < 5e-4

    # 3. Unstamped legacy optimizer checkpoint assumes canonical tp=1 and re-interleaves for live tp=2.
    state_tp1, tx = _train_state("llama", 1)
    ref_opt_tp1 = _canonical_fused_optimizer_leaves(state_tp1)
    unstamped_ckpt = tmp_path / "unstamped-opt-ckpt"
    state_tp1.save_state(unstamped_ckpt, float_dtype=None, save_optimizer=True)
    (unstamped_ckpt / "fused_param_layout.json").unlink()
    cfg_payload = _json.loads((unstamped_ckpt / "config.json").read_text())
    cfg_payload.pop("fused_param_tp", None)
    cfg_payload.pop("fused_param_layout", None)
    (unstamped_ckpt / "config.json").write_text(_json.dumps(cfg_payload))

    reloaded_opt = ed.EasyDeLState.load_state(
        load_directory=str(unstamped_ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        auto_shard_model=True,
        tx_template=tx,
    )
    reloaded_opt_leaves = _canonical_fused_optimizer_leaves(reloaded_opt)
    assert reloaded_opt_leaves.keys() == ref_opt_tp1.keys()
    for path, expected in ref_opt_tp1.items():
        assert float(np.max(np.abs(expected - reloaded_opt_leaves[path]))) < 1e-6, path

    # 4. Cross-mode can_merge_tp fallback rejects raw merged-split optimizer state with ValueError.
    if jax.device_count() >= 8:
        merged_opt_model = _qwen3_next_linear_variant(2, merged_split_proj=True, num_k_heads=4, num_v_heads=4)
        opt_tx = optax.adam(1e-3)
        merged_state = ed.EasyDeLState.create(model=merged_opt_model, tx=opt_tx, init_opt_state=True)
        merged_state, _ = _train_step(merged_state)
        merged_opt_ckpt = tmp_path / "merged-opt-fallback"
        merged_state.save_state(merged_opt_ckpt, float_dtype=None, save_optimizer=True)
        with pytest.raises(ValueError, match="linear_attention_merged_split_proj"):
            ed.EasyDeLState.load_state(
                load_directory=str(merged_opt_ckpt),
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                sharding_axis_dims=(1, 1, -1, 1, 8, 1),
                auto_shard_model=True,
                tx_template=None,
                allow_optimizer_restore_failure=False,
            )
        fallback_state = ed.EasyDeLState.load_state(
            load_directory=str(merged_opt_ckpt),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            sharding_axis_dims=(1, 1, -1, 1, 8, 1),
            auto_shard_model=True,
            tx_template=None,
        )
        assert fallback_state.opt_state is None


def test_single_token_rows_invalidates_jit_key_and_tp_sharding_helpers():
    """single_token_rows & decode_mode_specs invalidate jax.jit keys; with_tp_last_axis_sharding respects scopes."""
    from easydel.infra.sharding import (
        decode_mode_specs,
        dp_spmd_vmap_scope,
        inference_mode_forces_decode,
        inference_mode_single_token_rows,
        single_token_rows,
    )
    from easydel.layers.layouts._runtime import with_tp_last_axis_sharding

    trace_count = 0

    @jax.jit
    def _branch(x):
        nonlocal trace_count
        trace_count += 1
        return x + 1.0 if inference_mode_single_token_rows() else x - 1.0

    x = jnp.zeros((2,), dtype=jnp.float32)
    out_false = _branch(x)
    with single_token_rows(True):
        out_true = _branch(x)
    out_false_again = _branch(x)

    assert trace_count == 2
    np.testing.assert_array_equal(np.asarray(out_false), np.array([-1.0, -1.0]))
    np.testing.assert_array_equal(np.asarray(out_true), np.array([1.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(out_false_again), np.array([-1.0, -1.0]))

    decode_trace_count = 0

    @jax.jit
    def _decode_branch(y):
        nonlocal decode_trace_count
        decode_trace_count += 1
        return y + 2.0 if inference_mode_forces_decode() else y - 2.0

    dec_false = _decode_branch(x)
    with decode_mode_specs(True):
        dec_true = _decode_branch(x)
    dec_false_again = _decode_branch(x)

    assert decode_trace_count == 2
    np.testing.assert_array_equal(np.asarray(dec_false), np.array([-2.0, -2.0]))
    np.testing.assert_array_equal(np.asarray(dec_true), np.array([2.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(dec_false_again), np.array([-2.0, -2.0]))

    # with_tp_last_axis_sharding inside dp_spmd_vmap_scope("dp") when dp > fsdp must not re-shard on "dp",
    # and must not emit duplicate "tp" axes when an input already carries "tp" on an earlier dimension.
    if jax.device_count() >= 8:
        cfg = _config(tp=2)
        cfg.sharding_axis_dims = (1, 4, 1, 1, 2, 1)
        mesh = cfg.mesh
        with mesh:

            @jax.jit
            def _vmapped(inp):
                with dp_spmd_vmap_scope("dp"):
                    return jax.vmap(
                        lambda y: with_tp_last_axis_sharding(y, cfg),
                        in_axes=0,
                        out_axes=0,
                        spmd_axis_name="dp",
                    )(inp)

            arr = jnp.ones((4, 4, 2, 8), dtype=jnp.float32)
            res = _vmapped(arr)
            assert res.shape == arr.shape

            dup_tp_arr = jax.device_put(
                jnp.ones((2, 4, 8), dtype=jnp.float32),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp", None, None)),
            )
            dedup_res = with_tp_last_axis_sharding(dup_tp_arr, cfg)
            assert dedup_res.sharding.spec == jax.sharding.PartitionSpec(None, None, "tp")

        # decode_mode_specs with sp > 1 must use decode_query_sequence_axis (None) instead of "sp"
        # and re-trace on a shared jitted callable.
        cfg_sp = _config(tp=2)
        cfg_sp.sharding_axis_dims = (1, 1, 1, 1, 2, 4)
        with cfg_sp.mesh:

            @jax.jit
            def _constrain(inp):
                return with_tp_last_axis_sharding(inp, cfg_sp)

            arr_dec = jnp.ones((1, 4, 8), dtype=jnp.float32)
            res_prefill = _constrain(arr_dec)
            assert res_prefill.sharding.spec == jax.sharding.PartitionSpec(None, "sp", "tp")
            with decode_mode_specs(True):
                res_dec = _constrain(arr_dec)
            assert res_dec.sharding.spec == jax.sharding.PartitionSpec(None, None, "tp")
