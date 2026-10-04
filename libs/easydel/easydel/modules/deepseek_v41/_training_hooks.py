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

"""DeepSeek-V4.1 implementation of the trainer's optional ``host_batch_transform`` model hook.

With host-resident fp8 engram tables, a batch carrying ``engram_hash_ids`` would make the compiled train step look the
rows up through ``jax.pure_callback`` mid-program. The hook gathers those rows on the host while the batch is prepared
and replaces ``engram_hash_ids`` with ``engram_rows`` / ``engram_scales`` (the same inputs eSurge serving passes), so
the train step does no table lookup. Device-table models keep ``engram_hash_ids`` unchanged.
"""

from __future__ import annotations

import typing as tp

import numpy as np

from ._esurge_hooks import _engram_modules, _host_gathered


def host_batch_transform(model: tp.Any, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Swap ``engram_hash_ids [B, S, L, cols]`` for host-gathered ``engram_rows`` / ``engram_scales``.

    Args:
        model: The trained V4.1 model (concrete or abstract; only module attributes are read).
        batch: Purified host batch.

    Returns:
        The batch with ``engram_rows`` (uint8 ``[B, S, L, cols, head_dim]``) and ``engram_scales`` (uint8
        ``[B, S, L, cols, head_dim // 32]``) in place of ``engram_hash_ids`` when every engram layer keeps its fp8
        table in host RAM; otherwise the batch unchanged.
    """
    hash_ids = batch.get("engram_hash_ids")
    if hash_ids is None or "engram_rows" in batch or not _host_gathered(model):
        return batch
    ids = np.asarray(hash_ids)
    modules = _engram_modules(model)  # ordered by layer_hash_index, the forward's engram axis order
    if ids.ndim != 4 or ids.shape[2] != len(modules):
        raise ValueError(f"engram_hash_ids must be [batch, seq, {len(modules)}, cols]; got shape {tuple(ids.shape)}.")
    gathered = [module.gather_host_rows(ids[:, :, layer]) for layer, module in enumerate(modules)]
    out = {key: value for key, value in batch.items() if key != "engram_hash_ids"}
    out["engram_rows"] = np.stack([rows for rows, _ in gathered], axis=2)
    out["engram_scales"] = np.stack([scales for _, scales in gathered], axis=2)
    return out


def register_training_hooks() -> None:
    """Attach the trainer host-batch hook to the registered V4.1 model classes."""
    from .modeling_deepseek_v41 import DeepseekV41ForCausalLM, DeepseekV41ForConditionalGeneration

    for cls in (DeepseekV41ForCausalLM, DeepseekV41ForConditionalGeneration):
        cls.host_batch_transform = staticmethod(host_batch_transform)
