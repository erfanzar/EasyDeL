# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""JAX mesh creation and management utilities.

This submodule provides functions for creating and managing JAX device meshes
for distributed computation. It handles complex scenarios including multi-host,
multi-slice TPU setups, and various parallelism configurations.

Key Functions:
    create_mesh: Create a JAX mesh with customizable axis dimensions and names.
    create_cpu_mesh: Create a mesh using CPU devices (useful for testing).
    parse_mesh_from_string: Create a mesh from a string configuration.
    force_cpu: Context manager to force CPU execution.
    cpu_context: Context manager combining CPU mesh and CPU execution.

Classes:
    MeshPartitionHelper: Helper for analyzing and applying partition strategies.

Supported Parallelism Strategies:
    - dp (Data Parallelism): Replicates model, splits data batches
    - fsdp (Fully Sharded Data Parallelism): Shards both model and data
    - tp (Tensor Parallelism): Splits tensors across devices
    - sp (Sequence Parallelism): Splits along sequence dimension
    - ep (Expert Parallelism): For mixture-of-experts models

Example:
    >>> from eformer.escale.mesh import create_mesh, cpu_context
    >>> # Create a mesh for 8 devices with data and model parallelism
    >>> mesh = create_mesh(axis_dims=(2, 4), axis_names=('dp', 'tp'))
    >>> # Or use CPU for testing
    >>> with cpu_context() as mesh:
    ...     # Run distributed code on CPU
    ...     pass
"""

from .creation import cpu_context, create_cpu_mesh, create_mesh, force_cpu, parse_mesh_from_string
from .mesh_helpers import MeshPartitionHelper

__all__ = ("MeshPartitionHelper", "cpu_context", "create_cpu_mesh", "create_mesh", "force_cpu", "parse_mesh_from_string")
