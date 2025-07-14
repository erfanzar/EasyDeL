# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
from __future__ import annotations

import typing as tp

import jax
from eformer.pytree import auto_pytree

if tp.TYPE_CHECKING:
    from easydel.infra.base_state import EasyDeLState
else:
    EasyDeLState = tp.Any
CallFN: tp.TypeAlias = tp.Any | tp.Mapping[str, tp.Callable] | dict[str, tp.Callable]


@auto_pytree
class TrainerOutput:
    state: EasyDeLState
    mesh: jax.sharding.Mesh | None
    checkpoint_manager: tp.Any
    gather_fns: CallFN | None = None
    shard_fns: CallFN | None = None
    last_save_file_name: str | None = None
    checkpoint_path: str | None = None
