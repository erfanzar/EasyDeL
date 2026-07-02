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

"""Guards against GDR configs re-introducing the divergent dtype flags.

``use_input_dtype_phase1_outputs`` / ``use_input_dtype_state`` are honored by
the forward-only (no-grad) TPU/Pallas core but ignored by the custom-VJP
training core, so any framework-selected config carrying ``True`` makes a
frozen model's forward diverge from the same weights' grad-path forward
(the distillation-KL inflation bug). Framework selection has two entry
points: fresh autotune candidates and reloaded persistent-cache entries —
both must come back with the flags off.
"""

from __future__ import annotations

from types import SimpleNamespace

from ejkernel.modules.operations.configs import GatedDeltaRuleConfig
from ejkernel.modules.operations.gated_delta_rule import GatedDeltaRule, _sanitize_persistent_gdr_cfg
from ejkernel.ops.config.persistent import PersistentCache


def _all_candidates():
    op = GatedDeltaRule()
    inv = SimpleNamespace(kwargs={})
    yield from op.candidate_cfgs_tpu(inv)
    yield from op.candidate_cfgs_gpu(inv)
    yield from op.candidate_cfgs(inv)


def test_autotune_candidates_never_carry_divergent_dtype_flags():
    count = 0
    for cfg in _all_candidates():
        assert cfg.use_input_dtype_phase1_outputs is False, cfg
        assert cfg.use_input_dtype_state is False, cfg
        count += 1
    assert count > 0


def test_persistent_cache_sanitizes_stale_dtype_flags(tmp_path):
    cache = PersistentCache(
        "gated_delta_rule",
        path=str(tmp_path / "gated_delta_rule.json"),
        loader=_sanitize_persistent_gdr_cfg,
        cfg_type=GatedDeltaRuleConfig,
    )
    # entry persisted before the defaults were flipped: flags stored as True
    stale = GatedDeltaRuleConfig(
        platform="pallas",
        backend="tpu",
        chunk_size=256,
        use_input_dtype_phase1_outputs=True,
        use_input_dtype_state=True,
    )
    cache.put("tpu-v5p", "gated_delta_rule", "callkey", stale)
    loaded = cache.get("tpu-v5p", "gated_delta_rule", "callkey")
    assert isinstance(loaded, GatedDeltaRuleConfig)
    assert loaded.chunk_size == 256
    assert loaded.platform == "pallas"
    assert loaded.use_input_dtype_phase1_outputs is False
    assert loaded.use_input_dtype_state is False

    # entry from an even older schema with a since-removed key still loads
    cache.put(
        "tpu-v5p",
        "gated_delta_rule",
        "legacy",
        {"chunk_size": 128, "use_input_dtype_phase1_outputs": True, "legacy_knob": 1},
    )
    legacy = cache.get("tpu-v5p", "gated_delta_rule", "legacy")
    assert isinstance(legacy, GatedDeltaRuleConfig)
    assert legacy.chunk_size == 128
    assert legacy.use_input_dtype_phase1_outputs is False
