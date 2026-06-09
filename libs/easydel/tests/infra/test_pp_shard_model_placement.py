from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

import easydel as ed

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires at least 4 devices for pp4 placement")
def test_llama_shard_model_places_pp_variables_on_stage_submeshes():
    mesh = spx.create_mesh(
        axis_dims=(4, 1, 1, 1, 1, 1),
        axis_names=AXIS_NAMES,
        mpmd_axis="pp",
    )
    config = ed.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8,
        use_cache=False,
        scan_layers=True,
        lmhead_chunksize=4,
        sharding_axis_dims=(4, 1, 1, 1, 1, 1),
        sharding_axis_names=AXIS_NAMES,
    )
    config.set_model_mesh(mesh)

    with mesh:
        model = ed.LlamaForCausalLM(
            config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=ed.Rngs(0),
        ).shard_model(mesh=mesh)

    mpmd_mesh = mesh.mpmd_mesh
    expected_devices = {stage: set(mpmd_mesh.submesh(stage).devices.flat) for stage in range(mpmd_mesh.mpmd_dim)}
    expected_layers = {
        0: {0, 1},
        1: {2, 3},
        2: {4, 5},
        3: {6, 7},
    }
    layers_by_stage = {stage: set() for stage in range(mpmd_mesh.mpmd_dim)}
    specials_by_stage = {stage: set() for stage in range(mpmd_mesh.mpmd_dim)}
    placement_errors = []
    unexpected_unassigned = []

    for path, variable in spx.iter_variables(model):
        value = getattr(variable, "value", None)
        if not isinstance(value, jax.Array):
            continue

        stage = variable.resolved_stage_index(mesh)
        if stage is None:
            if not path.startswith("rngs."):
                unexpected_unassigned.append(path)
            continue

        actual_devices = set(value.devices())
        if actual_devices != expected_devices[stage]:
            placement_errors.append(path)

        if ".layers." in path:
            parts = path.split(".")
            layer_pos = parts.index("layers") + 1
            layers_by_stage[stage].add(int(parts[layer_pos]))
        elif "embed_tokens" in path or "lm_head" in path or path.endswith("norm.weight"):
            specials_by_stage[stage].add(path)

    assert placement_errors == []
    assert unexpected_unassigned == []
    assert layers_by_stage == expected_layers
    assert "model.embed_tokens.weight" in specials_by_stage[0]
    assert "lm_head.weight" in specials_by_stage[3]
    assert "model.norm.weight" in specials_by_stage[3]
