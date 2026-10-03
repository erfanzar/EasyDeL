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

"""EasyDeL -> HF export keeps the LM head's tying (computation test: accelerator only).

Regression for two export bugs found under transformers 5.18:

- ``load_state_dict(assign=True)`` replaced ``lm_head.weight`` with its own tensor, untying it from the
  embedding, so a tied MPT export computed logits from a stale head (max|dlogits| ~9).
- MPT aliases ``tie_word_embeddings`` as ``use_lm_head``; the exported config only carried the alias and fell
  back to HF's tied default, so an untied model exported as tied.
"""

import easydel as ed
import numpy as np
import pytest
from jax import numpy as jnp

torch = pytest.importorskip("torch")

EXPORT_ATOL = 5e-2


@pytest.mark.parametrize("tied", [True, False])
def test_mpt_export_preserves_head_tying(tied):
    import spectrax as spx

    config_cls, module_cls = ed.get_modules_by_type("mpt", ed.TaskType.CAUSAL_LM)
    cfg = config_cls(d_model=64, n_heads=4, n_layers=2, max_seq_len=128, vocab_size=256, use_lm_head=tied)
    cfg.scan_layers = False
    model = module_cls(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = np.random.default_rng(0).integers(0, 256, size=(1, 12)).astype("int32")
    ed_logits = np.asarray(model(input_ids=jnp.asarray(ids)).logits, dtype=np.float32)

    hf = model.to_torch()
    hf.eval()
    assert bool(hf.config.tie_word_embeddings) is tied
    shares_storage = hf.lm_head.weight.data_ptr() == hf.transformer.wte.weight.data_ptr()
    assert shares_storage is tied
    with torch.no_grad():
        hf_logits = hf(input_ids=torch.tensor(ids, dtype=torch.long)).logits.float().numpy()
    assert float(np.max(np.abs(ed_logits - hf_logits))) < EXPORT_ATOL
