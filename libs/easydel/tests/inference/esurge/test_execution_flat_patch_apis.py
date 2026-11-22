"""Host-only coverage for the public execution and flat-patch contracts."""

import importlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.multimodal.manager import MultiModalManager
from easydel.inference.esurge.runners.execution_manager import ExecutionManager

from .test_mm_manager import _image

pytestmark = pytest.mark.cpu_ok


class _Executor:
    """Host projection stand-in with separate compile-time and runtime state."""

    def __init__(self):
        self.main_hidden_enabled = False
        self.model = None
        self.graphdef = None
        self.runtime_state = (2, 3)
        self.compiled = {}
        self.fail_compile = False

    def cache_keys(self):
        return tuple(self.compiled)

    def get_lm_head(self, *, padded_num_reqs):
        return self.compiled[padded_num_reqs]

    def compile_lm_head(self, *, padded_num_reqs, graphdef, graphstate, graphother, inputs, hidden_dim, dtype):
        if self.fail_compile:
            raise RuntimeError("compilation unavailable")

        def project(state, other, rows):
            if rows.shape != (padded_num_reqs, hidden_dim) or rows.dtype != dtype:
                raise ValueError("unexpected projection signature")
            return rows * state + other

        self.compiled[padded_num_reqs] = project

    def runtime_graph_args(self):
        return self.runtime_state

    def clear_runtime_graph_args(self):
        self.runtime_state = (None, None)

    def clear_cache(self):
        self.compiled.clear()


class _Reference:
    """Weak-referenceable stand-in for a live model or state tree."""


def _manager():
    manager = object.__new__(ExecutionManager)
    manager.model = _Reference()
    manager.graphdef = _Reference()
    manager.graphstate = _Reference()
    manager.graphother = _Reference()
    manager._model_executor = _Executor()
    manager._model_executor.model = manager.model
    manager._model_executor.graphdef = manager.graphdef
    manager._sampler_executor = SimpleNamespace(model=manager.model, clear_cache=lambda: None)
    manager._debug_baselines = {}
    manager._pp_microbatch = None
    return manager


def test_projection_handles_uncached_buckets_and_uses_live_runtime_state():
    manager = _manager()
    rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(manager.project_hidden_rows(rows), rows * 2 + 3)
    # A cached executable must observe runtime state updates, not captured templates.
    manager._model_executor.runtime_state = (5, 7)
    manager._model_executor.fail_compile = True
    np.testing.assert_array_equal(manager.project_hidden_rows(rows), rows * 5 + 7)
    with pytest.raises(RuntimeError, match="compilation unavailable"):
        manager.project_hidden_rows(rows[:2])


def test_projection_rejects_non_matrix_inputs():
    manager = _manager()
    with pytest.raises(ValueError, match="shape"):
        manager.project_hidden_rows(np.zeros((2, 3, 4), dtype=np.float32))


def test_main_hidden_output_is_configured_before_compiling():
    manager = _manager()
    assert not manager.main_hidden_enabled
    manager.set_main_hidden_enabled()
    assert manager.main_hidden_enabled
    manager.set_main_hidden_enabled(False)
    assert not manager.main_hidden_enabled
    manager.project_hidden_rows(np.ones((1, 4), dtype=np.float32))
    manager.set_main_hidden_enabled(False)
    with pytest.raises(RuntimeError, match="before compiling"):
        manager.set_main_hidden_enabled(True)
    assert not manager.main_hidden_enabled


@pytest.mark.parametrize("clear_compiled_cache", [False, True])
def test_release_drops_live_references_and_prevents_projection(clear_compiled_cache):
    manager = _manager()
    model = weakref.ref(manager.model)
    graphdef = weakref.ref(manager.graphdef)
    graphstate = weakref.ref(manager.graphstate)
    graphother = weakref.ref(manager.graphother)
    manager._model_executor.runtime_state = (manager.graphstate, manager.graphother)
    manager.release_model_state(clear_compiled_cache=clear_compiled_cache)
    assert model() is graphdef() is graphstate() is graphother() is None
    with pytest.raises(RuntimeError, match="Restore model state"):
        manager.project_hidden_rows(np.ones((1, 4), dtype=np.float32))
    manager.release_model_state(clear_compiled_cache=clear_compiled_cache)


@pytest.mark.parametrize("owner", ["model", "config"])
def test_flat_patch_capability_enables_public_image_preprocessing(owner):
    config = SimpleNamespace(
        model_type="new_flat_patch_family",
        vision_config=SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1),
    )
    model = SimpleNamespace(config=config)
    (model if owner == "model" else config).supports_flat_patch_inputs = True
    manager = MultiModalManager(model=model, resolution_buckets=[(28, 28)])
    pixels, grid = manager.process_images([_image(size=(28, 28))])
    assert pixels.shape == (4, 3 * 2 * 14 * 14)
    np.testing.assert_array_equal(grid, [[1, 2, 2]])


@pytest.mark.parametrize(
    ("family", "config_name"),
    [
        ("glm4v", "Glm4vConfig"),
        ("glm4v_moe", "Glm4vMoeConfig"),
        ("glm46v", "Glm46VConfig"),
        ("muse_glimmer", "MuseGlimmerConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("qwen3_vl", "Qwen3VLConfig"),
        ("qwen3_vl_moe", "Qwen3VLMoeConfig"),
    ],
)
def test_legacy_configs_preserve_flat_patch_preprocessing_without_family_routing(family, config_name):
    module = importlib.import_module(f"easydel.modules.{family}.{family}_configuration")
    config = getattr(module, config_name)()
    # Use the actual family's declaration, but deliberately change its identity and
    # shrink only the vision geometry so the image protocol needs no Pillow.
    config.model_type = "renamed_custom_family"
    config.vision_config = SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1)
    manager = MultiModalManager(model=SimpleNamespace(config=config), resolution_buckets=[(28, 28)])
    pixels, grid = manager.process_images([_image(size=(28, 28))])
    assert pixels.shape == (4, 3 * 2 * 14 * 14)
    np.testing.assert_array_equal(grid, [[1, 2, 2]])


@pytest.mark.parametrize(
    "model_type", ["glm4v", "glm4v_moe", "glm46v", "muse_glimmer", "qwen2_vl", "qwen3_vl", "qwen3_vl_moe"]
)
def test_family_name_alone_does_not_declare_patch_layout(model_type):
    config = SimpleNamespace(
        model_type=model_type,
        vision_config=SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1),
    )
    manager = MultiModalManager(model=SimpleNamespace(config=config), resolution_buckets=[(28, 28)])
    with pytest.raises(ValueError, match="flat-patch"):
        manager.process_images([_image(size=(28, 28))])


@pytest.mark.parametrize("declaration", ["supports_flat_patch_inputs", "esurge_flat_patch_inputs"])
def test_explicit_opt_out_overrides_config_capability(declaration):
    config = SimpleNamespace(
        model_type="qwen3_vl",
        vision_config=SimpleNamespace(patch_size=14, temporal_patch_size=2, spatial_merge_size=1),
        supports_flat_patch_inputs=True,
    )
    model = SimpleNamespace(config=config)
    setattr(model, declaration, False)
    manager = MultiModalManager(model=model, resolution_buckets=[(28, 28)])
    with pytest.raises(ValueError, match="flat-patch"):
        manager.process_images([_image(size=(28, 28))])
