"""BEMA auxiliary-checkpoint protocol and accelerator-only resume parity."""

import importlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.trainers.bema_trainer.bema_config import BEMACallback
from easydel.trainers.bema_trainer.bema_trainer import BEMADPOTrainer
from easydel.trainers.direct_preference_optimization_trainer import DPOTrainer
from eformer.paths import ePath
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass
class StateFixture:
    graphstate: object
    graphother: object
    model: object

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def policy_save(self, state, save_directory=None, *args, **kwargs):
    ePath(save_directory).mkdir(parents=True, exist_ok=True)
    return str(save_directory)


@pytest.mark.cpu_ok
def test_bema_auxiliary_storage_roundtrip_and_incomplete_rejection(tmp_path, monkeypatch):
    module = importlib.import_module("easydel.trainers.bema_trainer.bema_trainer")

    class MetadataStorage:
        """Host-only storage adapter: real trainer manifest/trees, no device arrays."""

        def __init__(self, base_path, **kwargs):
            self.path = ePath(base_path)

        def save_pytree(self, tree, prefix):
            (self.path / f"{prefix}.json").write_text(json.dumps(tree))

        def load_pytree(self, *, prefix, **kwargs):
            return json.loads((self.path / f"{prefix}.json").read_text()), {}

    monkeypatch.setattr(DPOTrainer, "_save_state", policy_save)
    monkeypatch.setattr(module, "Checkpointer", MetadataStorage)
    monkeypatch.setattr(module.jax, "eval_shape", lambda fn: fn())
    model = SimpleNamespace(mesh=None, resolve_shardings_regex=lambda: [])
    trainer = object.__new__(BEMADPOTrainer)
    trainer.bema_callback = BEMACallback(update_freq=1, ref_model_update_freq=2, update_ref_model=True)
    trainer.reference_state = StateFixture(spx.State({"params": {"weight": 11.0}}), spx.State({}), model)
    for name, value in [("theta0", 1.0), ("ema", 2.0), ("bema", 3.0)]:
        setattr(trainer.bema_callback, f"_{name}_graphstate", spx.State({"params": {"weight": value}}))
    path = trainer._save_state(object(), str(tmp_path / "checkpoint"))
    restored = object.__new__(BEMADPOTrainer)
    restored.bema_callback = BEMACallback()
    restored.reference_state = StateFixture(spx.State({"params": {"weight": -1.0}}), spx.State({}), model)
    restored._load_auxiliary_states(path)
    assert restored.reference_state.graphstate.raw() == {"params": {"weight": 11.0}}
    for name, value in [("theta0", 1.0), ("ema", 2.0), ("bema", 3.0)]:
        assert getattr(restored.bema_callback, f"_{name}_graphstate").raw() == {"params": {"weight": value}}
    assert restored.bema_callback.update_freq == 1
    assert restored.bema_callback.ref_model_update_freq == 2
    manifest_path = ePath(path) / "bema" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        restored._load_auxiliary_states(path)
    # Legacy policy-only checkpoints (no auxiliary state) still load.
    restored._load_auxiliary_states(str(tmp_path / "legacy"))


def test_bema_checkpoint_resume_matches_uninterrupted_reference_updates(tmp_path, monkeypatch):
    """Real TensorStore roundtrip and independent smoothing reference; accelerator only."""
    monkeypatch.setattr(DPOTrainer, "_save_state", policy_save)
    mesh = Mesh(np.asarray(jax.devices()), ("fsdp",))
    sharding = NamedSharding(mesh, PartitionSpec())
    model = SimpleNamespace(mesh=mesh, resolve_shardings_regex=lambda: [(".*", sharding)])

    def state(value):
        weights = jax.device_put(jnp.full((4,), value, dtype=jnp.float32), sharding)
        return StateFixture(spx.State({"params": {"weight": weights}}), spx.State({}), model)

    def make_trainer():
        trainer = object.__new__(BEMADPOTrainer)
        trainer.bema_callback = BEMACallback(update_freq=1, ref_model_update_freq=2, update_ref_model=True)
        trainer.reference_state = state(-1.0)
        return trainer

    uninterrupted = make_trainer()
    for step, value in enumerate([0.0, 2.0]):
        uninterrupted.reference_state = uninterrupted.bema_callback.update_reference_state(
            state(value),
            uninterrupted.reference_state,
            step,
        )
    path = uninterrupted._save_state(state(2.0), str(tmp_path / "checkpoint"))
    resumed = make_trainer()
    resumed._load_auxiliary_states(path)
    # Step 1 did not refresh the reference: restoring only BEMA weights would be wrong.
    np.testing.assert_array_equal(resumed.reference_state.graphstate.raw()["params"]["weight"], np.zeros(4))
    ema = 2.0 * 11.0**-0.5
    for step, value in [(2, 3.0), (3, 4.0), (4, 5.0)]:
        beta, alpha = (10.0 + step) ** -0.5, (10.0 + step) ** -0.2
        ema = (1.0 - beta) * ema + beta * value
        expected_bema = ema + alpha * value
        for trainer in [uninterrupted, resumed]:
            trainer.reference_state = trainer.bema_callback.update_reference_state(
                state(value), trainer.reference_state, step
            )
        actual = resumed.bema_callback._bema_graphstate.raw()["params"]["weight"]
        np.testing.assert_allclose(actual, np.full(4, expected_bema), rtol=1e-5)
        np.testing.assert_allclose(
            resumed.reference_state.graphstate.raw()["params"]["weight"],
            uninterrupted.reference_state.graphstate.raw()["params"]["weight"],
            rtol=1e-6,
        )
