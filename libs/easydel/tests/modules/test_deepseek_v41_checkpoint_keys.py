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

"""Check the DeepSeek-V4.1 native checkpoint key mapper against the REAL released key list.

The released ``model.safetensors.index.json`` (96,085 keys) is the ground truth. The assertions compare the mapper's
output with independently stated expectations (the unowned-key rules, the config's ownership schedule and the module
attribute names of ``modeling_deepseek_v41.py`` / ``modeling_deepseek_v4.py``), never with the mapper itself.
Pure Python: no array computation.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest
from easydel.modules.deepseek_v41 import DeepseekV41Config
from easydel.modules.deepseek_v41._native_checkpoint import native_key_to_easydel_v41

pytestmark = pytest.mark.cpu_ok

# tests/modules/<this file>: parents[0]=modules, [1]=tests, [2]=libs/easydel, [3]=libs, [4]=repository root.
_INDEX_PATH = Path(__file__).resolve().parents[4] / ".xerxes/projects/deepseek-v41/ref/model.safetensors.index.json"
_TOTAL_KEYS = 96_085
_UNOWNED_ROOTS = frozenset({"image_start", "image_end", "image_newline"})
_UNOWNED_PREFIXES = ("mtp.", "vision.", "aligner.")
_LAYER_NAME = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT_NAME = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(.+)$")
_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
# The ~100 GB fp8 engram tables are filled by the dedicated row-chunked loader (``_engram_loader``), so the key mapper
# deliberately maps them to None: they must never reach the streaming conversion / fp32 quant decoder.
_ENGRAM_TABLE = re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")


def _is_unowned(key: str) -> bool:
    """Independent statement of which released keys the text model does not own (or loads out of band)."""
    return (
        key.startswith(_UNOWNED_PREFIXES)
        or key in _UNOWNED_ROOTS
        or key.endswith(".ffn.gate.bias_vl")
        or _ENGRAM_TABLE.match(key) is not None
    )


@pytest.fixture(scope="module")
def keys() -> list[str]:
    if not _INDEX_PATH.is_file():
        pytest.skip(f"released DeepSeek-V4.1 key index not found at {_INDEX_PATH}")
    with _INDEX_PATH.open() as f:
        return list(json.load(f)["weight_map"])


@pytest.fixture(scope="module")
def mapping(keys: list[str]) -> dict[str, str | None]:
    return {k: native_key_to_easydel_v41(k) for k in keys}


@pytest.fixture(scope="module")
def mapped(mapping: dict[str, str | None]) -> dict[str, str]:
    return {k: v for k, v in mapping.items() if v is not None}


@pytest.fixture(scope="module")
def config() -> DeepseekV41Config:
    return DeepseekV41Config()


@pytest.fixture(scope="module")
def by_layer(mapped: dict[str, str], config: DeepseekV41Config) -> dict[int, set[str]]:
    """Mapped per-layer names (layer-relative), with per-expert indices collapsed."""
    layers: dict[int, set[str]] = collections.defaultdict(set)
    for name in mapped.values():
        m = _LAYER_NAME.match(name)
        if m is None:
            continue
        rest = m.group(2)
        em = _EXPERT_NAME.match(name)
        if em is not None:
            rest = f"mlp.experts.{em.group(3)}"
        layers[int(m.group(1))].add(rest)
    return layers


def test_index_has_the_released_key_count(keys: list[str]) -> None:
    assert len(keys) == _TOTAL_KEYS


def test_unmapped_keys_are_exactly_the_unowned_ones(mapping: dict[str, str | None]) -> None:
    bad_type = [k for k, v in mapping.items() if v is not None and not (isinstance(v, str) and v)]
    assert not bad_type, f"mapper returned a non-string/empty value for {bad_type[:10]}"

    none_keys = {k for k, v in mapping.items() if v is None}
    expected = {k for k in mapping if _is_unowned(k)}
    unexpected_none = sorted(none_keys - expected)
    unexpectedly_mapped = sorted(expected - none_keys)
    assert not unexpected_none, f"{len(unexpected_none)} unowned-by-accident keys, e.g. {unexpected_none[:10]}"
    assert not unexpectedly_mapped, f"{len(unexpectedly_mapped)} keys should be unowned, e.g. {unexpectedly_mapped[:10]}"
    # every category really occurs in the release (guards against an empty/ trivial comparison)
    for prefix in _UNOWNED_PREFIXES:
        assert any(k.startswith(prefix) for k in none_keys), prefix
    assert _UNOWNED_ROOTS <= none_keys
    assert any(k.endswith(".ffn.gate.bias_vl") and k.startswith("layers.") for k in none_keys)


def test_no_two_source_keys_collide(mapped: dict[str, str]) -> None:
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for key, name in mapped.items():
        owners[name].append(key)
    collisions = {name: srcs for name, srcs in owners.items() if len(srcs) > 1}
    assert not collisions, f"{len(collisions)} colliding names, e.g. {dict(list(collisions.items())[:3])}"


def test_every_scale_name_has_a_matching_weight(mapped: dict[str, str]) -> None:
    names = set(mapped.values())
    # ``attn_hc.scale`` / ``ffn_hc.scale`` are the bare mHC ``scale`` parameters (ManifoldHyperConnection.scale, shape
    # (3,)), not dequant scales, so they have no ``.weight`` sibling by design.
    hc_scales = {n for n in names if re.fullmatch(r"model\.layers\.\d+\.(attn|ffn)_hc\.scale", n)}
    assert len(hc_scales) == 2 * 40
    scales = sorted(n for n in names if n.endswith(".scale") and n not in hc_scales)
    assert scales, "the release carries fp8/fp4 dequant scales"
    orphans = [n for n in scales if n[: -len("scale")] + "weight" not in names]
    assert not orphans, f"{len(orphans)} scale names without a .weight sibling, e.g. {orphans[:10]}"


def test_backbone_layers_cover_the_config_range(by_layer: dict[int, set[str]], config: DeepseekV41Config) -> None:
    assert config.num_hidden_layers == 40
    assert config.num_nextn_predict_layers == 3
    assert set(by_layer) == set(range(config.num_hidden_layers))


def test_ownership_structure_matches_the_config_schedule(
    by_layer: dict[int, set[str]], config: DeepseekV41Config
) -> None:
    errors: list[str] = []
    engram_ids = set(config.engram_layer_ids)
    for i in range(config.num_hidden_layers):
        names = by_layer[i]

        def has(prefix: str, names: set[str] = names) -> bool:
            return any(n.startswith(prefix) for n in names)

        checks = {
            "compressor": (has("self_attn.compressor."), config.is_kv_source(i)),
            "compressor.gate_proj": (
                "self_attn.compressor.gate_proj.weight" in names,
                config.is_kv_source(i) and config.layer_compress_ratio(i) > 1,
            ),
            "indexer.q_b_proj": ("self_attn.indexer.q_b_proj.weight" in names, config.is_index_source(i)),
            "indexer.weights_proj": ("self_attn.indexer.weights_proj.weight" in names, config.is_index_source(i)),
            "indexer.wk": ("self_attn.indexer.wk.weight" in names, config.owns_index_keys(i)),
            "indexer.k_norm": ("self_attn.indexer.k_norm.weight" in names, config.owns_index_keys(i)),
            "engram": (has("engram."), i in engram_ids),
        }
        errors += [
            f"layer {i} {what}: real keys={got} config={want}" for what, (got, want) in checks.items() if got != want
        ]
    assert not errors, "\n".join(errors)


def test_every_layer_has_all_routed_experts(mapped: dict[str, str], config: DeepseekV41Config) -> None:
    experts: dict[tuple[int, str, str], set[int]] = collections.defaultdict(set)
    for name in mapped.values():
        m = _EXPERT_NAME.match(name)
        if m is None:
            continue
        layer = int(re.search(r"layers\.(\d+)\.", m.group(1)).group(1))
        proj, _, suffix = m.group(3).partition(".")
        experts[(layer, proj, suffix)].add(int(m.group(2)))
    full = set(range(config.n_routed_experts))
    errors = []
    for i in range(config.num_hidden_layers):
        for proj in _EXPERT_PROJS:
            for suffix in ("weight", "scale"):
                got = experts.get((i, proj, suffix), set())
                if got != full:
                    errors.append(f"layer {i} {proj}.{suffix}: {len(got)} distinct experts, expected {len(full)}")
    assert not errors, "\n".join(errors[:20])


def _expected_layer_parameters(config: DeepseekV41Config, i: int) -> set[str]:
    """Per-layer parameter names from the module attribute structure (see modeling_deepseek_v41.py / _v4.py).

    The shared expert (``DeepseekV4MLP``) holds one fused ``gate_up_proj`` plus ``down_proj``. The mapper emits the
    unfused ``shared_experts.gate_proj`` / ``up_proj`` names for native ``w1`` / ``w3``; those are the HF-side names that
    ``DeepseekV4MLP.reform_param`` (``gate_up_proj.build_reform_param``) fuses into ``gate_up_proj``. They are therefore
    asserted against that reform convention rather than against a direct module leaf. ``down_proj`` is a direct leaf.
    """
    names = {
        # DeepseekV41Attention
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_norm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_proj.weight",
        "self_attn.kv_norm.weight",
        "self_attn.o_a_proj.weight",
        "self_attn.o_b_proj.weight",
        "self_attn.sinks",
        # DeepseekV41DecoderLayer
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        # DeepseekV4SparseMoeBlock: router, stacked experts (index collapsed), shared expert (reform convention)
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "mlp.experts.gate_proj.weight",
        "mlp.experts.up_proj.weight",
        "mlp.experts.down_proj.weight",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
    }
    for hc in ("attn_hc", "ffn_hc"):
        names |= {f"{hc}.fn", f"{hc}.base", f"{hc}.scale"}
    if config.is_kv_source(i):  # DeepseekV41Compressor
        names |= {"self_attn.compressor.kv_proj.weight", "self_attn.compressor.kv_norm.weight"}
        if config.layer_compress_ratio(i) > 1:
            names.add("self_attn.compressor.gate_proj.weight")
    if config.is_index_source(i):  # DeepseekV41Indexer
        names |= {"self_attn.indexer.q_b_proj.weight", "self_attn.indexer.weights_proj.weight"}
        if config.owns_index_keys(i):
            names |= {"self_attn.indexer.wk.weight", "self_attn.indexer.k_norm.weight"}
    if i in config.engram_layer_ids:  # DeepseekV41Engram
        # ``engram.embed.{weight,scale}`` exist on the module but are loaded by ``_engram_loader`` (mapped to None).
        names |= {"engram.kv_proj.weight", "engram.q_weight", "engram.k_weight"}
    return names


def test_mapped_weight_names_exist_in_the_module_structure(
    by_layer: dict[int, set[str]], config: DeepseekV41Config
) -> None:
    errors = []
    for i in range(config.num_hidden_layers):
        got = {n for n in by_layer[i] if not n.endswith(".scale") or n.startswith(("attn_hc.", "ffn_hc."))}
        expected = _expected_layer_parameters(config, i)
        if got != expected:
            errors.append(f"layer {i}: unexpected={sorted(got - expected)} missing={sorted(expected - got)}")
    assert not errors, "\n".join(errors[:10])


def test_model_level_names(mapped: dict[str, str]) -> None:
    """Names outside ``model.layers.N``: embedding, final norm and head (no learned ``hc_head`` in V4.1)."""
    top = {n for n in mapped.values() if _LAYER_NAME.match(n) is None}
    assert top == {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
