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

"""
Dataclass-based argument parsing utilities.

This module provides tools for parsing command-line arguments directly into
Python dataclass instances, combining the type safety of dataclasses with
the flexibility of argparse.

Classes:
    DataClassArgumentParser: ArgumentParser subclass that generates arguments
        from dataclass type hints.

Functions:
    Argu: Helper function for creating dataclass fields with argument metadata.

Type Aliases:
    DataClass: Type alias for dataclass instances.
    DataClassType: Type alias for dataclass types.

Example:
    >>> from dataclasses import dataclass
    >>> from eformer.aparser import DataClassArgumentParser, Argu
    >>>
    >>> @dataclass
    >>> class TrainingConfig:
    ...     learning_rate: float = Argu(default=1e-4, help="Learning rate")
    ...     batch_size: int = Argu(default=32, help="Batch size")
    ...     num_epochs: int = Argu(default=10, help="Number of epochs")
    >>>
    >>> parser = DataClassArgumentParser(TrainingConfig)
    >>> config, = parser.parse_args_into_dataclasses()
    >>> print(config.learning_rate)
    0.0001

Features:
    - Automatic argument generation from dataclass fields
    - Support for boolean flags with --no-{flag} variants
    - Literal and Enum type support with choices
    - JSON and YAML configuration file loading
    - Argument aliases for convenience
"""

from ._aparser import (
    ArgumentDefaultsHelpFormatter,
    DataClass,
    DataClassArgumentParser,
    DataClassType,
)

__all__ = (
    "ArgumentDefaultsHelpFormatter",
    "DataClass",
    "DataClassArgumentParser",
    "DataClassType",
)
