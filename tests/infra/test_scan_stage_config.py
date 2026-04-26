import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from jax.sharding import PartitionSpec
from spectrax import nn

from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.elarge.processing import materialize_base_config


class _Block(spx.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.weight = spx.Parameter(jnp.asarray(scale, dtype=jnp.float32))

    def forward(self, x):
        return x + self.weight.value


def _build_staged_layers(num_layers: int) -> nn.ModuleList:
    layers = nn.ModuleList([])
    for layer_idx in range(num_layers):
        with spx.assign_stage(total=num_layers, current=layer_idx):
            layers.append(_Block(float(layer_idx)))
    return layers


def test_base_config_scan_layers_and_removed_fields():
    cfg = EasyDeLBaseConfig(
        scan_layers=True,
        pipeline_virtual_stages=2,
        pipeline_stage_layout="loop",
        hardware_abstraction=True,
        pallas_m_block_size=32,
        pallas_k_block_size=32,
        pallas_n_block_size=32,
    )

    assert cfg.scan_layers is True
    assert cfg.pipeline_virtual_stages == 2
    assert cfg.pipeline_stage_layout == "loop"
    assert cfg.to_dict()["pipeline_stage_layout"] == "loop"
    for key in ("hardware_abstraction", "pallas_m_block_size", "pallas_k_block_size", "pallas_n_block_size"):
        assert not hasattr(cfg, key)
        assert key not in cfg.to_dict()


def test_elarge_materialize_drops_removed_base_config_keys():
    base = materialize_base_config(
        {
            "model": {"name_or_path": "dummy/model"},
            "base_config": {
                "values": {
                    "scan_layers": True,
                    "hardware_abstraction": True,
                    "pallas_m_block_size": 64,
                }
            },
        }
    )

    assert base["scan_layers"] is True
    assert "hardware_abstraction" not in base
    assert "pallas_m_block_size" not in base


def test_inline_assign_stage_sets_pipeline_metadata():
    layers = _build_staged_layers(4)

    assignments = [var.metadata.get("pipeline_stage") for _path, var in spx.iter_variables(layers)]
    assert assignments == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_base_config_mesh_is_spxmesh_with_pp_axis():
    cfg = EasyDeLBaseConfig()

    assert isinstance(cfg.mesh, spx.SpxMesh)
    assert cfg.mesh.mpmd_axis == "pp"
    assert cfg.mesh.mpmd_mesh is not None
    assert cfg.pipeline_stage_layout == "loop"


def test_pipeline_stage_layout_is_explicit_not_env(monkeypatch):
    if jax.device_count() < 2:
        pytest.skip("requires at least two devices for a pp>1 mesh")

    monkeypatch.setenv("SPX_VIRTUAL_STAGE_LAYOUT", "contiguous")
    import easydel as ed

    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        scan_layers=True,
        sharding_axis_dims=(2, 1, -1, 1, 1, 1),
        pipeline_virtual_stages=2,
        pipeline_stage_layout="interleaved",
    )
    model = ed.LlamaForCausalLM(cfg, rngs=ed.Rngs(0))

    assignments = [
        var.resolved_stage_index(cfg.mesh)
        for path, var in spx.iter_variables(model.model.layers)
        if path.endswith("input_layernorm.weight")
    ]

    assert assignments == [0, 1, 0, 1]


def test_stage_assignment_resolves_against_config_pp_mesh():
    if jax.device_count() < 2:
        pytest.skip("requires at least two devices for a pp>1 mesh")

    cfg = EasyDeLBaseConfig(sharding_axis_dims=(2, 1, -1, 1, 1, 1))
    layers = _build_staged_layers(4)

    resolved = [var.resolved_stage_index(cfg.mesh) for _path, var in spx.iter_variables(layers)]

    assert resolved == [0, 0, 1, 1]


def test_direct_sxstage_iter_inserts_marker():
    jaxpr = jax.make_jaxpr(lambda x: spx.sxstage_iter(x, stage=0))(jnp.ones((2,)))

    assert any(eqn.primitive.name == "sxstage_iter" for eqn in jaxpr.jaxpr.eqns)


def test_modulelist_scan_trace_path_matches_scan():
    layers = _build_staged_layers(3)
    x = jnp.asarray(0.0, dtype=jnp.float32)

    scanned = layers.scan(lambda layer, carry: layer(carry), x)
    traced = layers.scan(lambda layer, carry: layer(carry), x, trace=True)

    assert jnp.allclose(scanned, traced)


def test_llama_empty_cache_prefill_uses_real_scan_when_enabled(monkeypatch):
    from spectrax.core.containers import ModuleList, StackedModuleList

    import easydel as ed

    calls = []
    original_module_scan = ModuleList.scan
    original_stacked_scan = StackedModuleList.scan

    def wrapped_scan(self, fn, init_carry, *, trace=False, unroll=None):
        calls.append(trace)
        original = original_stacked_scan if isinstance(self, StackedModuleList) else original_module_scan
        return original(self, fn, init_carry, trace=trace, unroll=unroll)

    monkeypatch.setattr(ModuleList, "scan", wrapped_scan)
    monkeypatch.setattr(StackedModuleList, "scan", wrapped_scan)

    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        scan_layers=True,
    )
    model = ed.LlamaForCausalLM(cfg, rngs=ed.Rngs(0))

    out = model.model(input_ids=jnp.ones((1, 4), dtype=jnp.int32)).last_hidden_state

    assert out.shape == (1, 4, 32)
    assert calls == [False]


def test_llama_pp_mesh_forces_trace_path_for_stage_markers(monkeypatch):
    if jax.device_count() < 2:
        pytest.skip("requires at least two devices for a pp>1 mesh")

    del monkeypatch
    from spectrax.core.containers import ModuleList

    import easydel as ed

    cfg = ed.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
        scan_layers=True,
        sharding_axis_dims=(2, 1, -1, 1, 1, 1),
    )
    model = ed.LlamaForCausalLM(cfg, rngs=ed.Rngs(0))

    assignments = [
        var.resolved_stage_index(cfg.mesh)
        for path, var in spx.iter_variables(model.model.layers)
        if path.endswith("input_layernorm.weight")
    ]
    jaxpr = jax.make_jaxpr(lambda x: model.model._mark_layer_stage_boundary(x, 0, layers=model.model.layers))(
        jnp.ones((1, 4, 32), dtype=jnp.bfloat16)
    )

    assert isinstance(model.model.layers, ModuleList)
    assert assignments == [0, 1]
    assert model.model._layer_scan_trace(False) is True
    markers = [eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "sxstage_iter"]
    assert markers
    assert markers[0].params["sharding"] == PartitionSpec(None, "sp", "tp")
