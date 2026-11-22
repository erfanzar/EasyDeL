"""Host-only state transitions for documented model / sharding utility APIs."""

from types import SimpleNamespace

import pytest
from easydel.layers.attention import AttentionModule
from easydel.layers.engram import EngramModelMixin

pytestmark = pytest.mark.cpu_ok


class _HostTable:
    uses_host_table = False

    def place_table_on_host(self):
        self.uses_host_table = True


class _Model(EngramModelMixin):
    def __init__(self):
        self.tables = (_HostTable(), _HostTable())

    def engram_modules(self):
        return self.tables


def test_model_host_placement_updates_every_table_idempotently():
    model = _Model()
    assert not model.engram_uses_host_tables()
    assert model.place_engram_tables_on_host() == 2
    assert model.engram_uses_host_tables()
    assert model.place_engram_tables_on_host() == 2
    assert model.engram_uses_host_tables()


def test_existing_tensor_sharding_needs_no_config_or_fallback_mesh():
    module = object.__new__(AttentionModule)
    tensor = SimpleNamespace(sharding=SimpleNamespace(spec=("batch", None, "heads", None)))
    assert module.get_sharding_safely(tensor) == ("batch", None, "heads", None)
