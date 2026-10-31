# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Utility functions and helpers for ejkernel.ops.

This module provides various utilities for device fingerprinting, serialization,
metadata extraction, data carriers, and other supporting functionality.

Classes:
    FwdParams: Forward pass parameters for kernel configuration
    BwdParams: Backward pass parameters for kernel configuration

Fingerprinting Functions:
    device_fingerprint: Generate stable device identifier
    device_kind: Get device type (gpu, cpu, tpu)
    get_device_platform: Extract platform identifier from device
    sharding_fingerprint: Extract sharding information from arrays
    abstractify: Convert arrays to shape/dtype specs for hashing
    stable_json: Deterministic JSON serialization
    short_hash: Generate short hash from object
    default_key_builder_with_sharding: Generate cache key with sharding info

Metadata Functions:
    extract_labels_from_hlo_text: Find operation labels in HLO text
    find_labels_in_lowered: Extract labels from lowered JAX computation
    label: Generate operation label
    labels_to_configs: Map labels to configurations

Serialization Functions:
    to_json: Serialize object to JSON string
    from_json: Deserialize object from JSON string
"""

from .datacarrier import BwdParams, FwdParams
from .fingerprint import (
    abstractify,
    default_key_builder_with_sharding,
    device_fingerprint,
    device_kind,
    get_device_platform,
    sharding_fingerprint,
    short_hash,
    stable_json,
)
from .meta import extract_labels_from_hlo_text, find_labels_in_lowered, label, labels_to_configs
from .serialize import from_json, to_json

__all__ = (
    "BwdParams",
    "FwdParams",
    "abstractify",
    "default_key_builder_with_sharding",
    "device_fingerprint",
    "device_kind",
    "extract_labels_from_hlo_text",
    "find_labels_in_lowered",
    "from_json",
    "get_device_platform",
    "label",
    "labels_to_configs",
    "sharding_fingerprint",
    "short_hash",
    "stable_json",
    "to_json",
)
