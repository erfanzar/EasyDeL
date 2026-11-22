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

"""Tests for DNS service discovery rank assignment (pure Python, no array computation)."""

from __future__ import annotations

import pytest
from easydel.inference.esurge.distributed.discovery import DiscoveryResult, resolve_service_hosts


@pytest.mark.cpu_ok
def test_rank_to_host_maps_rank_index_to_sorted_host():
    result = DiscoveryResult(hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"])

    assert result.rank_to_host == {0: "10.0.0.1", 1: "10.0.0.2", 2: "10.0.0.3"}
    assert result.world_size == 3


@pytest.mark.cpu_ok
def test_resolve_service_hosts_literal_ip_and_world_size_mismatch():
    result = resolve_service_hosts("127.0.0.1", world_size=1)
    assert result.rank_to_host[0] == "127.0.0.1"

    with pytest.raises(ValueError, match="world size mismatch"):
        resolve_service_hosts("127.0.0.1", world_size=2)
