# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Array-free serialization and semantic-axis regression contracts."""

import copy
import pickle

import pytest
from spectrax import PartitionAxis
from spectrax import common_types as ct

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_not_given_pickle_preserves_singleton(protocol):
    restored = pickle.loads(pickle.dumps({"sentinel": ct.NOT_GIVEN, "replicated": None}, protocol=protocol))
    assert restored["sentinel"] is ct.NOT_GIVEN
    assert restored["replicated"] is None


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_not_given_copy_preserves_singleton(copier):
    assert copier(ct.NOT_GIVEN) is ct.NOT_GIVEN


@pytest.fixture
def isolated_axis_registry(monkeypatch):
    """Keep all registrations, including failing cases, local to each test."""
    monkeypatch.setattr(PartitionAxis, "_REGISTERED_SEMANTIC_MAP", dict(PartitionAxis._REGISTERED_SEMANTIC_MAP))
    monkeypatch.setattr(PartitionAxis, "_REGISTERED_GENERATION_MAP", dict(PartitionAxis._REGISTERED_GENERATION_MAP))


@pytest.mark.parametrize("mode", [ct.MODE_TRAIN, ct.MODE_PREFILL, ct.MODE_DECODE, ct.MODE_INSERT])
def test_registered_replication_and_aliases(isolated_axis_registry, mode):
    PartitionAxis.register("__R5B_REPLICATED__", None)
    PartitionAxis.register("__R5B_ALIAS__", "__R5B_REPLICATED__")
    PartitionAxis.register("__R5B_FUSED__", ("__R5B_REPLICATED__", ct.TP))
    paxis = PartitionAxis()
    assert paxis.resolve_axis(["__R5B_REPLICATED__", "__R5B_ALIAS__"], mode) == [None, None]
    assert paxis.resolve_axis([["__R5B_REPLICATED__", ct.TP]], mode) == [[None, "tp"]]
    assert paxis.resolve_axis(["__R5B_FUSED__"], mode) == [(None, "tp")]
    with pytest.raises(ValueError, match="Unknown semantic axis"):
        paxis.resolve_axis(["__R5B_UNKNOWN__"], mode)
    with pytest.raises(ValueError, match="Unknown semantic axis"):
        paxis.resolve_axis([["__R5B_UNKNOWN__", ct.TP]], mode)


@pytest.mark.parametrize("mode", [ct.MODE_TRAIN, ct.MODE_PREFILL, ct.MODE_DECODE, ct.MODE_INSERT])
def test_replicated_registration_retains_generation_override(isolated_axis_registry, mode):
    PartitionAxis.register("__R5B_REPLICATED__", None, generation_axis_rule=ct.TP)
    PartitionAxis.register("__R5B_OVERRIDE__", ct.DP, generation_axis_rule=None)
    expected = ["tp", None] if mode in ct.GENERATION_MODES else [None, "dp"]
    paxis = PartitionAxis()
    assert paxis.resolve_axis(["__R5B_REPLICATED__", "__R5B_OVERRIDE__"], mode) == expected
    assert paxis.resolve_axis([["__R5B_REPLICATED__", "__R5B_OVERRIDE__"]], mode) == [expected]


def test_semantic_alias_cycles_still_raise(isolated_axis_registry):
    PartitionAxis.register("__R5B_FIRST__", "__R5B_SECOND__")
    PartitionAxis.register("__R5B_SECOND__", "__R5B_FIRST__")
    with pytest.raises(ValueError, match="Cyclic semantic axis"):
        PartitionAxis().resolve_axis(["__R5B_FIRST__"], ct.MODE_TRAIN)
