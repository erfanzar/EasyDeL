"""Array-free public API tests for parameter merging, config rebuilds, topology overrides and state load/shard."""

import json
from types import SimpleNamespace

import pytest
import spectrax as spx
from easydel.infra import base_state
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState


class _ScalarModel(EasyDeLBaseModule):
    """Real graph/state round trips with Python scalars, not array computation."""

    def __init__(self):
        self.weight = spx.Parameter(7)
        self.child = spx.Module()
        self.child.weight = spx.Parameter(11)
        self.child.buffer = spx.Buffer(13)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("nested", [False, True])
def test_merge_parameter_values_round_trips_real_state_without_mutating_source(nested):
    model = _ScalarModel()
    replacements = (
        {"parameters": {"weight": 17, "child": {"weight": 19}}}
        if nested
        else {("parameters", "weight"): 17, ("parameters", "child", "weight"): 19}
    )
    rebuilt = model.merge_parameter_values(replacements)
    assert rebuilt.weight.value == 17
    assert rebuilt.child.weight.value == 19
    assert rebuilt.child.buffer.value == 13
    assert model.weight.value == 7
    assert model.child.weight.value == 11
    assert isinstance(rebuilt.split_parameters(), spx.State)


@pytest.mark.cpu_ok
def test_merge_parameter_values_rejects_unknown_keys_atomically_and_respects_selector():
    model = _ScalarModel()
    with pytest.raises(KeyError, match="not found"):
        model.merge_parameter_values({("parameters", "weight"): 17, ("parameters", "missing"): 19})
    assert model.weight.value == 7
    rebuilt = model.merge_parameter_values({"buffers": {"child": {"buffer": 23}}}, selector="buffers")
    assert rebuilt.child.buffer.value == 23
    assert rebuilt.child.weight.value == 11
    assert model.child.buffer.value == 13


@pytest.mark.cpu_ok
@pytest.mark.parametrize("style", ["flat", "nested", "dotted"])
def test_merge_parameter_values_supports_legacy_paths_and_parameter_values_output(style):
    model = _ScalarModel()
    if style == "flat":
        replacements = {("weight",): 17, ("child", "weight"): 19}
    elif style == "nested":
        replacements = {"weight": 17, "child": {"weight": 19}}
    else:
        replacements = model.parameter_values()
        assert replacements == {"child.weight": 11, "weight": 7}
        replacements.update({"weight": 17, "child.weight": 19})
    rebuilt = model.merge_parameter_values(replacements)
    assert rebuilt.weight.value == 17
    assert rebuilt.child.weight.value == 19
    assert rebuilt.child.buffer.value == 13
    assert model.weight.value == 7


class _RebuildProbe:
    """Stub only reconstruction; exercise the public configuration-update APIs."""

    _apply_recursive_config_updates = staticmethod(EasyDeLBaseModule._apply_recursive_config_updates)
    new_graphdef = EasyDeLBaseModule.new_graphdef
    update_module = EasyDeLBaseModule.update_module
    graphstate = None
    graphother = None

    def __init__(self, config):
        self.config = config

    def _lazy_init_rebuilt_module(self, config):
        return SimpleNamespace(graphdef=config)

    def merge_module(self, graphdef, graphstate, graphother):
        return SimpleNamespace(config=graphdef)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("api", ["new_graphdef", "update_module"])
def test_recursive_public_rebuild_retargets_real_multilevel_omni_configs(api):
    from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import Qwen3OmniMoeConfig

    config = Qwen3OmniMoeConfig()
    thinker_text = config.thinker_config.text_config
    talker_text = config.talker_config.text_config
    assert thinker_text.attn_mechanism == "vanilla"
    assert talker_text.attn_mechanism == "vanilla"
    model = _RebuildProbe(config)
    result = getattr(model, api)(recursive_update=True, attn_mechanism="unified_attention")
    updated = result if api == "new_graphdef" else result.config
    assert updated.attn_mechanism == "unified_attention"
    assert updated.thinker_config.attn_mechanism == "unified_attention"
    assert updated.thinker_config.text_config.attn_mechanism == "unified_attention"
    assert updated.talker_config.text_config.attn_mechanism == "unified_attention"
    expected_original = "vanilla" if api == "new_graphdef" else "unified_attention"
    assert thinker_text.attn_mechanism == expected_original
    assert talker_text.attn_mechanism == expected_original


@pytest.mark.cpu_ok
def test_recursive_public_rebuild_handles_shared_children_and_cycles():
    root = EasyDeLBaseConfig()
    child = EasyDeLBaseConfig()
    grandchild = EasyDeLBaseConfig()
    root.child = child
    root.shared = grandchild
    child.child = grandchild
    grandchild.parent = root
    updated = _RebuildProbe(root).new_graphdef(recursive_update=True, attn_mechanism="unified_attention")
    assert updated.child.child is updated.shared
    assert updated.shared.parent is updated
    assert updated.child.attn_mechanism == "unified_attention"
    assert updated.shared.attn_mechanism == "unified_attention"
    assert grandchild.attn_mechanism == "vanilla"


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "accessor,setter",
    [("mesh", "set_model_mesh"), ("explicit_mesh", "set_explicit_mesh"), ("manual_mesh", "set_manual_mesh")],
)
@pytest.mark.parametrize("api", ["add_basic_configurations", "new_graphdef", "update_module"])
def test_topology_overrides_rebuild_only_auto_mesh_caches(accessor, setter, api):
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1), backend="cpu")
    original = getattr(config, accessor)
    axes = ("pp", "batch", "fsdp", "ep", "tp", "sp")

    def update(target):
        if api == "add_basic_configurations":
            target.add_basic_configurations(sharding_axis_names=axes)
            return target
        result = getattr(_RebuildProbe(target), api)(sharding_axis_names=axes)
        return result if api == "new_graphdef" else result.config

    updated = update(config)
    rebuilt = getattr(updated, accessor)
    assert rebuilt is not original
    assert rebuilt.axis_names == axes
    if api == "new_graphdef":
        assert getattr(config, accessor) is original
    assert not any(key.endswith("_auto_created") for key in updated.to_dict())

    pinned = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1), backend="cpu")
    getattr(pinned, setter)(original)
    assert getattr(update(pinned), accessor) is original


@pytest.mark.cpu_ok
def test_recursive_topology_overrides_invalidate_propagated_child_meshes():
    root = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1), backend="cpu")
    child = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1), backend="cpu")
    root.sub_configs = {"child": EasyDeLBaseConfig}
    root.child = child
    old = root.mesh
    assert child.mesh is old
    axes = ("pp", "batch", "fsdp", "ep", "tp", "sp")
    updated = _RebuildProbe(root).new_graphdef(recursive_update=True, sharding_axis_names=axes)
    # Access the child first: it must not depend on root.mesh to refresh it.
    assert updated.child.mesh.axis_names == axes
    assert updated.child.mesh is not old
    assert updated.mesh.axis_names == axes
    assert root.mesh is old
    assert child.mesh is old


@pytest.mark.cpu_ok
@pytest.mark.parametrize("mode", ["default", "required", "best_effort", "skip"])
@pytest.mark.parametrize("error", [OSError("corrupt optimizer checkpoint"), ValueError("optimizer shape mismatch")])
def test_public_load_state_is_best_effort_unless_strict_restore_is_requested(tmp_path, monkeypatch, mode, error):
    from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig

    (tmp_path / "metadata.json").write_text(json.dumps({"step": 11, "has_optimizer_state": True}))
    config = SimpleNamespace(architectures=["ReviewModel"])
    monkeypatch.setattr(AutoEasyDeLConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(AutoEasyDeLConfig, "bind_model_task", lambda *args: "review")
    model = object()
    monkeypatch.setattr(EasyDeLBaseModule, "from_pretrained", classmethod(lambda cls, **kwargs: model))
    # Avoid creating even a scalar JAX array; the checkpoint orchestration is
    # under test, not array conversion or the model serialization backend.
    monkeypatch.setattr(base_state, "jnp", SimpleNamespace(asarray=lambda value, dtype: value, int32="int32"))

    class _RestoreState:
        opt_state = None

        def __init__(self, step):
            self.step = step

        def load_optimizer(self, **kwargs):
            if mode == "skip":
                pytest.fail("explicit weights-only loading attempted optimizer restore")
            raise error

    monkeypatch.setattr(EasyDeLState, "create", classmethod(lambda cls, step, model: _RestoreState(step)))
    kwargs = {"skip_optimizer_state": mode == "skip"}
    if mode in ("required", "best_effort"):
        kwargs["allow_optimizer_restore_failure"] = mode == "best_effort"
    if mode == "required":
        with pytest.raises(type(error)) as caught:
            EasyDeLState.load_state(tmp_path, auto_shard_model=False, **kwargs)
        assert caught.value is error
    else:
        state = EasyDeLState.load_state(tmp_path, auto_shard_model=False, **kwargs)
        assert state.step == 11
        assert state.opt_state is None


@pytest.mark.cpu_ok
def test_public_shard_state_forwards_override_to_parameters_buffers_and_step(monkeypatch):
    from dataclasses import dataclass, replace

    mesh = object()
    default_mesh = object()
    replicated = object()
    monkeypatch.setattr(base_state, "replicated_named_sharding", lambda value: replicated if value is mesh else None)
    monkeypatch.setattr(base_state, "jnp", SimpleNamespace(asarray=lambda value, dtype: value, int32="int32"))
    monkeypatch.setattr(
        base_state, "jax", SimpleNamespace(Array=type(None), device_put=lambda value, target: (value, target))
    )

    class _Model:
        def _get_mesh(self, unused):
            return default_mesh

        def apply_sharding_for_tree(self, tree, *, mesh=None):
            return (tree, mesh)

    @dataclass
    class _State:
        step: int = 9
        graphstate: str = "parameters"
        graphother: str = "buffers"
        opt_state: object = None
        tx: object = None
        model: object = None

        def replace(self, **kwargs):
            return replace(self, **kwargs)

    result = EasyDeLState.shard_state(_State(model=_Model()), mesh=mesh)
    assert result.graphstate == ("parameters", mesh)
    assert result.graphother == ("buffers", mesh)
    assert result.step == (9, replicated)
