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

"""Check the DSpark ``mtp.*`` key normalizer against the REAL released DeepSeek-V4.1 key list.

The released ``model.safetensors.index.json`` is the ground truth. Expected names are stated explicitly below (the
module tree of ``DeepseekV41DSpark``: ``stages.N.{self_attn, mlp, input_layernorm, ...}``) and are never derived from
the normalizer itself. The names are also compared with the parameter table the TPU parity test builds the EasyDeL
model against. Pure Python: no array computation.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import pytest
from easydel.modules.deepseek_v41 import DeepseekV41Config
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import (
    DEEPSEEK_V41_KEY_RULES,
    DEEPSEEK_V41_MTP_KEY_RULES,
    DEEPSEEK_V41_VL_KEY_RULES,
)

try:
    from tests.modules import _deepseek_v41_reference as ref
except ImportError:  # pragma: no cover
    from modules import _deepseek_v41_reference as ref  # pyright: ignore[reportMissingImports]

pytestmark = pytest.mark.cpu_ok

_INDEX_PATH = Path(__file__).resolve().parents[4] / ".xerxes/projects/deepseek-v41/ref/model.safetensors.index.json"
_NUM_MTP_KEYS = 2401
_STAGES = 3
_EXPERTS = 128
_STAGE_NAME = re.compile(r"^stages\.(\d+)\.(.+)$")
_EXPERT_NAME = re.compile(r"^mlp\.experts\.(\d+)\.(.+)$")
_QUANTIZED_LINEARS = (
    "self_attn.q_a_proj",
    "self_attn.q_b_proj",
    "self_attn.kv_proj",
    "self_attn.o_a_proj",
    "self_attn.o_b_proj",
    "mlp.shared_experts.gate_proj",
    "mlp.shared_experts.up_proj",
    "mlp.shared_experts.down_proj",
    "mlp.experts.gate_proj",
    "mlp.experts.up_proj",
    "mlp.experts.down_proj",
)


@pytest.fixture(scope="module")
def mtp_keys() -> list[str]:
    if not _INDEX_PATH.is_file():
        pytest.skip(f"released DeepSeek-V4.1 key index not found at {_INDEX_PATH}")
    with _INDEX_PATH.open() as f:
        return [k for k in json.load(f)["weight_map"] if k.startswith("mtp.")]


@pytest.fixture(scope="module")
def mapping(mtp_keys: list[str]) -> dict[str, str | None]:
    return {k: DEEPSEEK_V41_MTP_KEY_RULES(k) for k in mtp_keys}


@pytest.fixture(scope="module")
def mapped(mapping: dict[str, str | None]) -> dict[str, str]:
    return {k: v for k, v in mapping.items() if v is not None}


@pytest.fixture(scope="module")
def config() -> DeepseekV41Config:
    return DeepseekV41Config()


def _expected_stage_names(stage: int) -> set[str]:
    """Per-stage parameter names (per-expert index collapsed) stated from the module structure."""
    names: set[str] = set()
    for module in _QUANTIZED_LINEARS:  # fp8 / fp4 weights carry a dequant ``.scale`` in the checkpoint
        names |= {f"{module}.weight", f"{module}.scale"}
    names |= {
        "self_attn.q_a_norm.weight",
        "self_attn.kv_norm.weight",
        "self_attn.sinks",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
    }
    for hc in ("attn_hc", "ffn_hc"):
        names |= {f"{hc}.fn", f"{hc}.base", f"{hc}.scale"}
    if stage == 0:
        names |= {"main_proj.weight", "main_proj.scale", "main_norm.weight"}
    if stage == _STAGES - 1:
        names |= {
            "norm.weight",
            "markov_head.embed.weight",
            "markov_head.head.weight",
            "confidence_head.proj.weight",
        }
    return names


def test_index_has_the_released_mtp_key_count(mtp_keys: list[str]) -> None:
    assert len(mtp_keys) == _NUM_MTP_KEYS
    assert {int(k.split(".")[1]) for k in mtp_keys} == set(range(_STAGES))


def test_config_matches_the_release(config: DeepseekV41Config) -> None:
    assert config.num_nextn_predict_layers == _STAGES
    assert config.dspark_n_routed_experts == _EXPERTS


def test_unmapped_keys_are_exactly_the_image_router_biases(mapping: dict[str, str | None]) -> None:
    none_keys = {k for k, v in mapping.items() if v is None}
    assert none_keys == {f"mtp.{i}.ffn.gate.bias_vl" for i in range(_STAGES)}
    # the tied ``mtp.N.embed`` / ``mtp.N.head`` are not in the release at all (convert.py skips them)
    assert not [k for k in mapping if re.match(r"^mtp\.\d+\.(embed|head)\.", k)]


def test_tied_embed_and_head_are_unowned() -> None:
    for key in ("mtp.0.embed.weight", "mtp.2.head.weight"):
        assert DEEPSEEK_V41_MTP_KEY_RULES(key) is None


def test_no_two_source_keys_collide(mapped: dict[str, str]) -> None:
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for key, name in mapped.items():
        owners[name].append(key)
    collisions = {name: srcs for name, srcs in owners.items() if len(srcs) > 1}
    assert not collisions, f"{len(collisions)} colliding names, e.g. {dict(list(collisions.items())[:3])}"
    assert len(mapped) == _NUM_MTP_KEYS - _STAGES


def test_mapped_names_match_the_module_tree(mapped: dict[str, str]) -> None:
    by_stage: dict[int, set[str]] = collections.defaultdict(set)
    experts: dict[tuple[int, str], set[int]] = collections.defaultdict(set)
    for name in mapped.values():
        m = _STAGE_NAME.match(name)
        assert m is not None, f"{name!r} is not under stages.N"
        stage, rest = int(m.group(1)), m.group(2)
        em = _EXPERT_NAME.match(rest)
        if em is not None:
            experts[(stage, em.group(2))].add(int(em.group(1)))
            rest = f"mlp.experts.{em.group(2)}"
        by_stage[stage].add(rest)
    assert set(by_stage) == set(range(_STAGES))
    for stage in range(_STAGES):
        got, want = by_stage[stage], _expected_stage_names(stage)
        assert got == want, f"stage {stage}: unexpected={sorted(got - want)} missing={sorted(want - got)}"
    for (stage, leaf), indices in experts.items():
        assert indices == set(range(_EXPERTS)), f"stage {stage} {leaf}: {len(indices)} experts"


def test_spot_checked_names(mapped: dict[str, str]) -> None:
    expected = {
        "mtp.0.attn.wq_a.weight": "stages.0.self_attn.q_a_proj.weight",
        "mtp.1.attn.attn_sink": "stages.1.self_attn.sinks",
        "mtp.2.attn.q_norm.weight": "stages.2.self_attn.q_a_norm.weight",
        "mtp.0.attn.wo_a.weight": "stages.0.self_attn.o_a_proj.weight",
        "mtp.1.attn_norm.weight": "stages.1.input_layernorm.weight",
        "mtp.1.ffn_norm.weight": "stages.1.post_attention_layernorm.weight",
        "mtp.0.hc_attn_fn": "stages.0.attn_hc.fn",
        "mtp.2.hc_ffn_scale": "stages.2.ffn_hc.scale",
        "mtp.0.ffn.gate.bias": "stages.0.mlp.gate.e_score_correction_bias",
        "mtp.0.ffn.experts.17.w1.weight": "stages.0.mlp.experts.17.gate_proj.weight",
        "mtp.2.ffn.experts.127.w2.scale": "stages.2.mlp.experts.127.down_proj.scale",
        "mtp.1.ffn.shared_experts.w3.weight": "stages.1.mlp.shared_experts.up_proj.weight",
        "mtp.0.main_proj.weight": "stages.0.main_proj.weight",
        "mtp.0.main_norm.weight": "stages.0.main_norm.weight",
        "mtp.2.norm.weight": "stages.2.norm.weight",
        "mtp.2.markov_head.embed.weight": "stages.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight": "stages.2.markov_head.head.weight",
        "mtp.2.confidence_head.proj.weight": "stages.2.confidence_head.proj.weight",
    }
    for key, name in expected.items():
        assert mapped[key] == name, key


def test_names_match_the_parity_test_parameter_table(mapped: dict[str, str], config: DeepseekV41Config) -> None:
    """The EasyDeL model the TPU test builds has exactly the parameters the real checkpoint provides.

    The table keeps the shared expert fused (``gate_up_proj``, as ``DeepseekV4MLP`` declares it) and has no dequant
    scales, so the checkpoint names are folded the same way before comparing.
    """
    folded: set[str] = set()
    for name in mapped.values():
        name = re.sub(r"\.experts\.\d+\.", ".experts.", name)
        if name.endswith(".scale") and not re.search(r"(attn|ffn)_hc\.scale$", name):
            continue  # fp8 / fp4 dequant scale
        name = re.sub(r"shared_experts\.(gate|up)_proj\.", "shared_experts.gate_up_proj.", name)
        folded.add(name)
    assert folded == set(ref.dspark_param_shapes(config))


def test_normalizer_is_idempotent_and_leaves_other_keys_alone() -> None:
    assert DEEPSEEK_V41_MTP_KEY_RULES("stages.1.mlp.gate.weight") == "stages.1.mlp.gate.weight"
    assert DEEPSEEK_V41_MTP_KEY_RULES("layers.3.attn.wq_a.weight") is None
    assert DEEPSEEK_V41_MTP_KEY_RULES("embed.weight") is None
    assert DEEPSEEK_V41_MTP_KEY_RULES("mtp.0.ffn.gate.bias_vl") is None


def test_existing_normalizers_still_leave_mtp_unowned(mtp_keys: list[str]) -> None:
    assert all(DEEPSEEK_V41_KEY_RULES(k) is None for k in mtp_keys)
    assert all(DEEPSEEK_V41_VL_KEY_RULES(k) is None for k in mtp_keys)
