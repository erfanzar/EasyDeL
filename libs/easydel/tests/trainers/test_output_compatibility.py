"""Both documented output import paths share a schema without breaking positional callers."""

from dataclasses import fields

import jax
import pytest
from easydel.trainers.trainer.modeling_output import TrainerOutput as LegacyTrainerOutput
from easydel.trainers.trainer_protocol import TrainerOutput

pytestmark = pytest.mark.cpu_ok


def test_legacy_output_is_a_canonical_output_with_identical_fields():
    output = LegacyTrainerOutput("state", None, "manager", "shard", "last", "checkpoint")
    assert isinstance(output, TrainerOutput)
    assert tuple(field.name for field in fields(output)) == tuple(field.name for field in fields(TrainerOutput))
    assert output.state == "state"
    assert output.checkpoint_manager == "manager"
    assert output.shard_fns == "shard"
    assert output.last_save_file_name == "last"
    assert output.checkpoint_path == "checkpoint"
    # PyTree reconstruction calls constructors by keyword and must retain every field.
    leaves, structure = jax.tree_util.tree_flatten(output)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert restored == output


def test_canonical_output_preserves_existing_positional_arguments():
    output = TrainerOutput("state", None, "last", "checkpoint")
    assert output.last_save_file_name == "last"
    assert output.checkpoint_path == "checkpoint"
    assert output.checkpoint_manager is None
    assert output.shard_fns is None
