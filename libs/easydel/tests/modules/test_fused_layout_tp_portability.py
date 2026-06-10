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


def _train_state(family: str, tp: int):
    model = FAMILIES[family](tp)
    tx = optax.adamw(learning_rate=1e-3)
    state = ed.EasyDeLState.create(model=model, tx=tx, init_opt_state=True)
    state, _ = _train_step(state)
    state, _ = _train_step(state)
    return state, tx


def _llama(tp):
    import spectrax as spx

    return ed.LlamaForCausalLM(config=_config(tp), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


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


def _qwen3_moe(tp):
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
    return ed.Qwen3MoeForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _qwen3_next(tp):
    """GDR linear attention (packed qkvz/ba) + gated full attention."""
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
    model = FAMILIES[family](save_tp)
    reference = _logits(model)
    reference_fused = _canonical_fused_parameter_leaves(model)
    model.save_pretrained(str(tmp_path / "ckpt"))

    loaded = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(tmp_path / "ckpt"),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
    )
    restored = _logits(loaded)

    err = float(np.max(np.abs(reference - restored)))
    assert err < _model_logit_tolerance(family), (
        f"[{family}] save_tp={save_tp} -> load_tp={load_tp} changed the model (max|Δlogits|={err}); "
        "fused projections were not layout-normalized across meshes"
    )
    restored_fused = _canonical_fused_parameter_leaves(loaded)
    assert restored_fused.keys() == reference_fused.keys()
    for path, expected in reference_fused.items():
        fused_err = float(np.max(np.abs(expected - restored_fused[path])))
        assert fused_err < 1e-6, (
            f"[{family}] fused parameter leaf {path} changed after save_tp={save_tp} -> load_tp={load_tp} "
            f"(max|Δ|={fused_err})"
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
    from easydel.layers.layouts import canonicalize_fused_optimizer_state, read_fused_layout_marker

    save_model = _llama(tp=2)
    custom_optimizer_state, canonical_leaves, _, unrelated = _custom_optimizer_state(save_model)
    state = ed.EasyDeLState.create(model=save_model, tx=None, opt_state=custom_optimizer_state)

    ckpt = tmp_path / "custom-optimizer-ckpt"
    state.save_state(ckpt, float_dtype=None, save_optimizer=True)
    assert read_fused_layout_marker(ckpt) == "canonical"

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
    from easydel.layers.layouts import read_fused_layout_marker

    save_tp = 2
    state, tx = _train_state(family, save_tp)
    reference_logits = _state_logits(state)
    reference_opt = _canonical_fused_optimizer_leaves(state)

    ckpt = tmp_path / "trainer-ckpt"
    state.save_state(ckpt, float_dtype=None, save_optimizer=True)
    assert read_fused_layout_marker(ckpt) == "canonical"

    loaded = ed.EasyDeLState.load_state(
        load_directory=str(ckpt),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, load_tp, 1),
        auto_shard_model=True,
        tx_template=tx,
    ).replace(tx=tx)

    restored_logits = _state_logits(loaded)
    logits_err = float(np.max(np.abs(reference_logits - restored_logits)))
    logit_tol = _train_state_logit_tolerance(family)
    direction = f"[{family}] train-state save_tp={save_tp} -> load_tp={load_tp}"
    msg = f"{direction} changed logits (max|Δlogits|={logits_err})"
    assert logits_err < logit_tol, msg

    restored_opt = _canonical_fused_optimizer_leaves(loaded)
    assert restored_opt.keys() == reference_opt.keys()
    for path, expected in reference_opt.items():
        err = float(np.max(np.abs(expected - restored_opt[path])))
        assert err < 1e-6, (
            f"[{family}] optimizer leaf {path} changed after canonical/runtime round-trip "
            f"save_tp={save_tp} -> load_tp={load_tp} (max|Δ|={err})"
        )

    reference_next, reference_loss = _train_step(state)
    loaded_next, loaded_loss = _train_step(loaded)
    assert np.isfinite(loaded_loss)
    assert abs(reference_loss - loaded_loss) < logit_tol

    next_logits_err = float(np.max(np.abs(_state_logits(reference_next) - _state_logits(loaded_next))))
    assert next_logits_err < _train_state_next_logit_tolerance(family), (
        f"[{family}] one-step resume diverged after save_tp={save_tp} -> load_tp={load_tp} "
        f"(max|Δlogits|={next_logits_err})"
    )
