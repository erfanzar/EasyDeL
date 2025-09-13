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

"""EasyDeL Large Model Infrastructure.

This module provides the core infrastructure for handling large-scale language models
in EasyDeL. It includes configuration management, model building utilities, and
optimization strategies for efficient training and inference.

Key Components:
    - eLargeModel: Main class for managing large language model configurations
    - builders: Utility functions for constructing model architectures
    - normalizer: Input/output normalization utilities
    - trainer_types: Training-specific type definitions and utilities
    - types: Core type definitions for the module
    - utils: Helper functions and utilities
"""

from .elarge_model import eLargeModel

__all__ = ("eLargeModel",)
