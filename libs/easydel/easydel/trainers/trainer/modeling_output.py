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

"""Training output data structures.

This module defines the output types returned by training operations,
including the final state and associated metadata from training runs.
"""

from __future__ import annotations

import collections.abc
import typing as tp

import spectrax as spx
from eformer.pytree import auto_pytree

from ..trainer_protocol import TrainerOutput as BaseTrainerOutput

if tp.TYPE_CHECKING:
    from easydel.infra.base_state import EasyDeLState
else:
    EasyDeLState = tp.Any
CallFN: tp.TypeAlias = tp.Any | collections.abc.Mapping[str, tp.Callable] | dict[str, tp.Callable]


@auto_pytree
class TrainerOutput(BaseTrainerOutput):
    """Backward-compatible constructor for the canonical training output schema.

    Contains the final model state and metadata from the training process,
    including checkpointing information and utility functions for state manipulation.

    Attributes:
        state: The final EasyDeLState after training completion.
        mesh: The JAX sharding mesh used during training, if any.
        checkpoint_manager: Manager object for handling model checkpoints.
        shard_fns: Functions for sharding parameters across devices.
        last_save_file_name: Name of the most recently saved checkpoint file.
        checkpoint_path: Full path to the checkpoint directory.
    """

    def __init__(
        self,
        state: EasyDeLState,
        mesh: spx.SpxMesh | None,
        checkpoint_manager: tp.Any,
        shard_fns: CallFN | None = None,
        last_save_file_name: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """Preserve the legacy positional ordering while sharing the output fields.

        Args:
            state: Final model state.
            mesh: Training mesh, if any.
            checkpoint_manager: Manager for handling model checkpoints.
            shard_fns: Optional parameter-sharding functions.
            last_save_file_name: Most recently saved checkpoint file.
            checkpoint_path: Checkpoint directory path.
        """
        super().__init__(
            state=state,
            mesh=mesh,
            last_save_file_name=last_save_file_name,
            checkpoint_path=checkpoint_path,
            checkpoint_manager=checkpoint_manager,
            shard_fns=shard_fns,
        )
