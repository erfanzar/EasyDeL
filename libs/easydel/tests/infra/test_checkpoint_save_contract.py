"""CPU-only checkpoint layout, discovery metadata and strict export regressions."""

import json
from types import SimpleNamespace

import jax
import pytest
from easydel.infra.mixins import bridge
from eformer.paths import ePath


class _Config:
    def save_pretrained(self, directory):
        ePath(directory).mkdir(parents=True, exist_ok=True)
        (ePath(directory) / "config.json").write_text(json.dumps({"architectures": self.architectures}))


class _TextOnlyModel(bridge.EasyBridgeMixin):
    """A non-numerical model so the real Checkpointer can be exercised on CPU."""

    config = _Config()
    mesh = None
    generation_config = None

    def can_generate(self):
        return False

    def _model_card(self, *args):
        return "# Checkpoint fixture"

    def assert_parameters_materialized(self, **kwargs):
        if not self.materialized:
            raise ValueError("unmaterialized parameter")

    def __init__(self, materialized=True):
        self.materialized = materialized


@pytest.fixture
def text_model(monkeypatch):
    state = SimpleNamespace(raw=lambda: {"parameters": {"label": "trained"}})
    monkeypatch.setattr(bridge.spx, "export", lambda model: (None, state))
    model = _TextOnlyModel()
    # Device metadata only: no JAX arrays or numerical computation.
    model.mesh = jax.sharding.Mesh([jax.devices("cpu")[0]], ("single",))
    return model


@pytest.mark.cpu_ok
@pytest.mark.parametrize("already_versioned", [False, True])
def test_save_step_produces_complete_versioned_checkpoint(tmp_path, text_model, already_versioned):
    root = tmp_path / "run-17" if already_versioned else tmp_path
    text_model.save_pretrained(root, step=17)
    checkpoint = tmp_path / "run-17"
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "README.md").is_file()
    # A pure-Python state has a structure sidecar but no TensorStore array index.
    assert (checkpoint / "model_structure.json").is_file()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["step"] == 17
    assert not (checkpoint / "run-17").exists()
    structure = json.loads((checkpoint / "model_structure.json").read_text())
    assert structure["extras"]["step"] == 17
    assert structure["array_keys"] == []


@pytest.mark.cpu_ok
@pytest.mark.parametrize("subdirectory", ["arbitrary-state-name", "arbitrary-state-name/_resume_model"])
def test_save_step_can_preserve_caller_owned_checkpoint_layout(tmp_path, text_model, subdirectory):
    root = tmp_path / subdirectory
    text_model.save_pretrained(root, step=23, versioned=False)
    assert (root / "config.json").is_file()
    assert json.loads((root / "metadata.json").read_text())["step"] == 23
    assert not (root / "run-23").exists()
    restored, extras = bridge.Checkpointer(base_path=str(root), save_interval=None, step_policies=[]).load_pytree(
        mesh=text_model.mesh, prefix="model", discover_latest=False, load_treedef=True
    )
    assert restored == {"parameters": {"label": "trained"}}
    assert extras["step"] == 23


@pytest.mark.cpu_ok
@pytest.mark.parametrize("merged_lora", [False, True])
def test_state_owned_save_keeps_model_and_resume_artifacts_at_declared_paths(
    tmp_path, text_model, monkeypatch, merged_lora
):
    """Exercise the real state save caller without parameter arrays or a live optimizer.

    This regression also verifies the unmerged checkpoint's structure can be read
    from the exact directory used by state resume (without latest-step discovery).
    """
    from easydel.infra import base_state

    text_model.lora_is_enabled = merged_lora
    monkeypatch.setattr(base_state, "deepcopy_model", lambda model: _TextOnlyModel())
    monkeypatch.setattr(_TextOnlyModel, "unwrap_lora_to_layers", lambda self, **kwargs: None, raising=False)
    state = SimpleNamespace(model=text_model, step=17, opt_state=None)
    root = tmp_path / "arbitrary-state-name"
    base_state.EasyDeLState.save_state(state, root, save_optimizer=False, merge_lora_before_save=merged_lora)

    model_directories = [root, root / "_resume_model"] if merged_lora else [root]
    missing = [
        str(directory / name)
        for directory in model_directories
        for name in ("config.json", "model_structure.json")
        if not (directory / name).is_file()
    ]
    assert not missing, f"state save misplaced model artifacts: {missing}"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["step"] == 17
    assert metadata["has_resume_model"] is merged_lora
    assert not metadata["has_optimizer_state"]
    resume_directory = model_directories[-1]
    restored, extras = bridge.Checkpointer(
        base_path=str(resume_directory), save_interval=None, step_policies=[]
    ).load_pytree(mesh=text_model.mesh, prefix="model", discover_latest=False, load_treedef=True)
    assert restored == {"parameters": {"label": "trained"}}
    assert extras["step"] == 17
    assert text_model.lora_is_enabled is merged_lora
    assert all(not (directory / "run-17").exists() for directory in model_directories)


@pytest.mark.cpu_ok
def test_strict_save_rejects_unmaterialized_parameters_before_writing(tmp_path):
    with pytest.raises(ValueError, match="unmaterialized"):
        _TextOnlyModel(materialized=False).save_pretrained(tmp_path / "missing", mismatch_allowed=False)
    assert not (tmp_path / "missing").exists()


@pytest.mark.cpu_ok
def test_strict_push_rejects_unmaterialized_parameters_before_external_write(monkeypatch):
    # An unexpected remote call fails the test without contacting the Hub.
    def forbid_remote(*args, **kwargs):
        pytest.fail("strict export must reject incomplete parameters before creating a Hub repo")

    monkeypatch.setattr(bridge, "create_repo", forbid_remote)
    with pytest.raises(ValueError, match="unmaterialized"):
        _TextOnlyModel(materialized=False).push_to_hub("fixture/model", mismatch_allowed=False)
