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

"""Tests for ``EasyDeLBaseModule.resolve_shardings_automatically``.

The contract under test:
  1. Returns a tuple of ``(slash-form regex, jax.sharding.NamedSharding)`` pairs.
  2. Each layer index is a *literal* in its regex -- no ``\\d+`` collapse.
     Per-stage placements under PP can differ between layers; collapsing
     would lose that information.
  3. The pipeline axis (``"pp"``) never appears in any returned spec.
     ``spx.extract_sharding_structure`` strips it; ``named_sharding_for_variable``
     resolves to a stage-local submesh and the spec only references intra-stage
     axes (``fsdp``, ``tp``, ...).
  4. Under MPMD (pp>1), each variable's NamedSharding lives on its
     pipeline stage's submesh, NOT the full mesh.
  5. Under SPMD (pp=1), every NamedSharding lives on the full mesh.
"""

from __future__ import annotations

import collections
import re

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

import easydel as ed

AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def _tiny_llama_config(*, sharding_axis_dims, num_hidden_layers=8):
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=8,
        use_cache=False,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=AXIS_NAMES,
    )


def _device_set(named_sharding) -> tuple[int, ...]:
    return tuple(sorted(int(d.id) for d in named_sharding.mesh.devices.flatten()))


def _spec_axes(spec) -> set[str]:
    """Flatten a PartitionSpec into the set of axis names it references."""
    out: set[str] = set()
    for entry in spec:
        if entry is None:
            continue
        if isinstance(entry, str):
            out.add(entry)
        else:
            out.update(a for a in entry if a)
    return out


def _layer_index_from_path(path: str) -> int | None:
    m = re.search(r"/layers/(\d+)/", path)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Common invariants applied to every config


def _assert_common_invariants(rules):
    assert len(rules) > 0, "expected at least one rule"

    # 1. Shape of every entry: (str, NamedSharding)
    for pat, ns in rules:
        assert isinstance(pat, str), f"pattern not str: {pat!r}"
        assert isinstance(ns, jax.sharding.NamedSharding), f"value not NamedSharding: {ns!r}"

    # 2. No \d+ collapse anywhere -- per-layer regex must pin the literal index.
    for pat, _ in rules:
        assert r"\d+" not in pat, f"per-layer regex must use literal index, got: {pat}"

    # 3. ``pp`` axis must never appear in any returned spec.
    for pat, ns in rules:
        assert "pp" not in _spec_axes(ns.spec), f"{pat}: spec={ns.spec} still references pp"


# ---------------------------------------------------------------------------
# Per-mesh assertions


def _assert_each_layer_on_single_submesh(rules):
    """Under PP, all leaves of the same layer must land on the same submesh."""
    layer_to_devsets: dict[int, set[tuple[int, ...]]] = collections.defaultdict(set)
    for pat, ns in rules:
        idx = _layer_index_from_path(pat)
        if idx is None:
            continue
        layer_to_devsets[idx].add(_device_set(ns))
    bad = {idx: ds for idx, ds in layer_to_devsets.items() if len(ds) > 1}
    assert not bad, f"layers landing on multiple submeshes: {bad}"


def _assert_all_on_full_mesh(rules, n_devices):
    full = tuple(range(n_devices))
    for pat, ns in rules:
        assert _device_set(ns) == full, f"{pat}: expected full mesh {full}, got {_device_set(ns)}"


# ---------------------------------------------------------------------------
# Tests


def test_returns_tuple_of_str_and_named_sharding_spmd():
    """SPMD case: return shape contract + every leaf on the full mesh."""
    config = _tiny_llama_config(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    assert config.mesh.is_mpmd is False
    assert config.mesh.mpmd_axis is None

    model = ed.LlamaForCausalLM(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    rules = model.resolve_shardings_automatically()

    _assert_common_invariants(rules)
    _assert_all_on_full_mesh(rules, n_devices=jax.device_count())


@pytest.mark.skipif(jax.device_count() < 4, reason="requires >=4 devices for pp=4 placement")
def test_pp4_each_variable_lives_on_its_stage_submesh():
    """PP=4 case: every NamedSharding bound to a single-chip stage submesh."""
    config = _tiny_llama_config(sharding_axis_dims=(-1, 1, 1, 1, 1, 1))
    assert config.mesh.is_mpmd is True
    assert config.mesh.mpmd_axis == "pp"
    assert int(config.mesh.shape["pp"]) == jax.device_count()

    model = ed.LlamaForCausalLM(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    rules = model.resolve_shardings_automatically()

    _assert_common_invariants(rules)
    _assert_each_layer_on_single_submesh(rules)

    # Every layer's submesh must be exactly one device under pp=N (no other parallelism).
    for pat, ns in rules:
        if _layer_index_from_path(pat) is None:
            continue
        devs = _device_set(ns)
        assert len(devs) == 1, f"{pat}: PP-only stage submesh should be one device, got {devs}"

    # Every chip should own at least one layer (no unused stages).
    chips_used = {_device_set(ns)[0] for pat, ns in rules if _layer_index_from_path(pat) is not None}
    assert chips_used == set(range(jax.device_count())), f"unused stages: {chips_used}"


@pytest.mark.skipif(jax.device_count() < 4, reason="requires >=4 devices for pp=2,fsdp=2")
def test_pp2_fsdp2_layers_on_two_chip_submeshes():
    """Mixed PP=2 + FSDP=2: each layer on a 2-chip submesh, fsdp axis still in spec."""
    config = _tiny_llama_config(sharding_axis_dims=(2, 1, 2, 1, 1, 1))
    assert config.mesh.is_mpmd is True
    assert int(config.mesh.shape["pp"]) == 2
    assert int(config.mesh.shape["fsdp"]) == 2

    model = ed.LlamaForCausalLM(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    rules = model.resolve_shardings_automatically()

    _assert_common_invariants(rules)
    _assert_each_layer_on_single_submesh(rules)

    # Each layer's submesh covers exactly 2 chips (the FSDP dim within its stage).
    for pat, ns in rules:
        if _layer_index_from_path(pat) is None:
            continue
        assert len(_device_set(ns)) == 2, f"{pat}: PP=2,FSDP=2 stage submesh should be 2 chips, got {_device_set(ns)}"

    # Should observe fsdp axis surviving in some specs.
    fsdp_seen = any("fsdp" in _spec_axes(ns.spec) for _, ns in rules)
    assert fsdp_seen, "expected at least one spec to still reference 'fsdp' under FSDP=2"


@pytest.mark.skipif(jax.device_count() < 4, reason="requires >=4 devices for two-stage PP")
def test_per_layer_rules_can_differ_between_stages():
    """Two different layers on different stages must produce DIFFERENT NamedShardings."""
    config = _tiny_llama_config(sharding_axis_dims=(-1, 1, 1, 1, 1, 1))
    model = ed.LlamaForCausalLM(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    rules = model.resolve_shardings_automatically()

    # Pick two layers known to land on different stages (loop schedule:
    # 0->stage0, 4->stage1 for an 8-layer / pp=4 setup with V=2 virtual stages).
    by_layer = collections.defaultdict(list)
    for pat, ns in rules:
        idx = _layer_index_from_path(pat)
        if idx is not None:
            by_layer[idx].append((pat, ns))

    # Find any pair of layers whose first-leaf submesh differs.
    devsets = {idx: _device_set(items[0][1]) for idx, items in by_layer.items()}
    distinct_devsets = set(devsets.values())
    assert len(distinct_devsets) >= 2, (
        f"PP=4 should produce >=2 distinct stage submeshes across layers, got {distinct_devsets}"
    )


def test_no_callers_get_partitionspec_objects():
    """Belt-and-suspenders: the *value* of every rule must NOT be a PartitionSpec.

    Old shape returned ``(regex, PartitionSpec)``; new shape returns
    ``(regex, NamedSharding)``.  Anyone unpacking with the old assumption
    would silently get the wrong type -- this regression test makes that
    impossible.
    """
    config = _tiny_llama_config(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    model = ed.LlamaForCausalLM(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    rules = model.resolve_shardings_automatically()

    for pat, value in rules:
        assert not isinstance(value, jax.sharding.PartitionSpec), (
            f"{pat}: rule value must be NamedSharding, not PartitionSpec"
        )
