import re

import jax
from eformer.common_types import Replicated
from flax import nnx as nn
from jax.sharding import Mesh

from easydel.infra.base_module import EasyDeLBaseModule


class _DummyConfig:
    partition_manager = None
    mesh = Mesh(jax.devices()[:1], ("dp",))


class _DummyLayer(nn.Module):
    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        return {"kernel": Replicated}


class _DummyModel(EasyDeLBaseModule):
    def __init__(self, config: _DummyConfig, *, rngs: nn.Rngs):
        super().__init__(
            config=config,
            dtype=None,
            param_dtype=None,
            precision=None,
            rngs=rngs,
        )
        self.layers = nn.List([_DummyLayer() for _ in range(13)])


def test_resolve_shardings_compacts_layer_indices():
    model = _DummyModel(config=_DummyConfig(), rngs=nn.Rngs(0))
    rules = model.resolve_shardings_automatically()

    assert rules, "Expected sharding rules to be generated."
    assert rules[-1][0] == ".*"

    path = "layers/5/kernel"
    matching = [(pat, spec) for pat, spec in rules if re.match(pat, path)]
    assert matching, "Expected a regex rule matching layers/5/kernel."
    assert matching[0][1] == Replicated

    exact_rule = "^layers/5/kernel$"
    assert all(pat != exact_rule for pat, _ in rules), "Expected compacted regex, not an exact rule."
