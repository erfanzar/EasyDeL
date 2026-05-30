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

"""Tests for MTP (Multi-Token-Prediction) knowledge distillation.

Covers:
* the model exposing ``mtp_logits`` on its output when the MTP head is active,
* the soft MTP-KD loss math (KL >= 0, perfect-match -> 0, shift-by-2 alignment,
  masking),
* the ``DistillationConfig`` MTP knobs + validation.

Run directly: ``python tests/trainers/test_mtp_distillation.py`` (CPU or TPU).
"""

import os
import sys
import time
import traceback

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import easydel  # noqa: E402,F401
import easydel.trainers  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import spectrax as spx  # noqa: E402

from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402
from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig  # noqa: E402
from easydel.trainers import DistillationConfig  # noqa: E402
from easydel.trainers.distillation_trainer._fn import (  # noqa: E402
    mtp_chain_distillation_loss,
    mtp_distillation_loss,
)

_RESULTS = []


def test(name):
    def deco(fn):
        def wrapper():
            t0 = time.time()
            try:
                fn()
                _RESULTS.append(("PASS", name))
                print(f"  PASS [{(time.time() - t0) * 1000:8.1f} ms] {name}")
            except Exception:
                tb = traceback.format_exc()
                _RESULTS.append(("FAIL", name))
                print(f"  FAIL              {name}\n    -> {tb.strip().splitlines()[-1]}")
                for line in tb.strip().splitlines()[-10:]:
                    print(f"       {line}")

        return wrapper

    return deco


def _cfg(mtp_layers=1, mtp_coef=0.3, hidden=64, vocab=128):
    return Qwen3_5TextConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=hidden // 2,
        max_position_embeddings=64,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        mtp_loss_coef=mtp_coef,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )


@test("model exposes outputs.mtp_logits when the MTP head is active")
def test_model_exposes_mtp_logits():
    b, s, v = 2, 16, 128
    ids = jax.random.randint(jax.random.PRNGKey(1), (b, s), 0, v)
    am = jnp.ones((b, s), dtype="i4")
    model = Qwen3_5ForCausalLM(config=_cfg(1, 0.3), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    out = model(input_ids=ids, attention_mask=am)
    assert out.mtp_logits is not None, "mtp_logits should be exposed when MTP active"
    assert tuple(out.mtp_logits.shape) == (b, s, v), f"bad shape {out.mtp_logits.shape}"
    assert out.aux_loss is not None, "aux_loss (self-supervised MTP CE) should be present"


@test("model exposes no mtp_logits when MTP disabled")
def test_model_no_mtp_logits_when_disabled():
    b, s, v = 2, 16, 128
    ids = jax.random.randint(jax.random.PRNGKey(1), (b, s), 0, v)
    model = Qwen3_5ForCausalLM(config=_cfg(0, 0.0), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    out = model(input_ids=ids, attention_mask=jnp.ones((b, s), dtype="i4"))
    assert getattr(out, "mtp_logits", None) is None, "mtp_logits should be None when MTP disabled"


@test("mtp_distillation_loss: KL >= 0 and finite")
def test_mtp_kd_nonnegative():
    b, s, v = 2, 16, 128
    teacher = jax.random.normal(jax.random.PRNGKey(3), (b, s, v)) * 2.0
    student = jax.random.normal(jax.random.PRNGKey(4), (b, s, v)) * 2.0
    loss = mtp_distillation_loss(student, teacher, attention_mask=jnp.ones((b, s)), temperature=1.0)
    assert jnp.isfinite(loss), "loss must be finite"
    assert float(loss) >= -1e-5, f"KL must be >= 0, got {float(loss)}"


@test("mtp_distillation_loss: perfect match (student=teacher shifted by 1) -> KL ~ 0")
def test_mtp_kd_perfect_match():
    # student MTP[:, t] predicts x_{t+2}; teacher[:, t+1] is that same conditional.
    b, s, v = 2, 16, 128
    teacher = jax.random.normal(jax.random.PRNGKey(3), (b, s, v)) * 2.0
    student = jnp.concatenate([teacher[:, 1:], teacher[:, -1:]], axis=1)  # shift left by 1
    loss = mtp_distillation_loss(student, teacher, attention_mask=jnp.ones((b, s)), temperature=1.0)
    assert abs(float(loss)) < 1e-3, f"perfect-match KL should be ~0, got {float(loss)}"


@test("mtp_distillation_loss: completion mask restricts the loss window")
def test_mtp_kd_mask():
    b, s, v = 2, 16, 128
    teacher = jax.random.normal(jax.random.PRNGKey(3), (b, s, v)) * 2.0
    student = jax.random.normal(jax.random.PRNGKey(4), (b, s, v)) * 2.0
    cmask = jnp.zeros((b, s), dtype=jnp.float32).at[:, 5:10].set(1.0)
    loss = mtp_distillation_loss(student, teacher, loss_mask=cmask, temperature=2.0)
    assert jnp.isfinite(loss), "masked loss must be finite"


@test("compute_mtp_chain: shape + step-1 matches the single-step path")
def test_mtp_chain_shape_and_consistency():
    b, s, v, k = 2, 32, 128, 6
    ids = jax.random.randint(jax.random.PRNGKey(1), (b, s), 0, v)
    am = jnp.ones((b, s), dtype="i4")
    model = Qwen3_5ForCausalLM(config=_cfg(1, 0.3), rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    out = model(input_ids=ids, attention_mask=am)
    chain = model.compute_mtp_chain(out, ids, k, attention_mask=am)
    assert tuple(chain.shape) == (k, b, s, v), f"bad chain shape {chain.shape}"
    # step 1 of the chain must equal the depth-1 single-step logits exactly
    assert jnp.allclose(chain[0], out.mtp_logits, atol=1e-4), "chain[0] != single-step mtp_logits"


@test("mtp_chain_distillation_loss: per-step alignment (perfect chain -> KL ~ 0)")
def test_mtp_chain_perfect_alignment():
    b, s, v, k = 2, 32, 128, 6
    teacher = jax.random.normal(jax.random.PRNGKey(9), (b, s, v)) * 2.0
    # step j (k=j+1) target = teacher shifted left by k -> a perfect student equals that
    perfect = jnp.stack(
        [jnp.concatenate([teacher[:, kk:], jnp.broadcast_to(teacher[:, -1:], (b, kk, v))], axis=1)[:, :s]
         for kk in range(1, k + 1)],
        axis=0,
    )
    mean_kd, per_step = mtp_chain_distillation_loss(perfect, teacher, attention_mask=jnp.ones((b, s)), temperature=1.0)
    assert abs(float(mean_kd)) < 1e-3, f"perfect-chain mean KD should be ~0, got {float(mean_kd)}"
    assert max(float(x) for x in per_step) < 1e-3, "some per-step KD not ~0 on a perfect chain"
    assert len(per_step) == k


@test("DistillationConfig: MTP knobs + validation")
def test_config_validation():
    c = DistillationConfig(mtp_distillation=True, mtp_kd_weight=0.5, mtp_draft_tokens=6, save_directory="/tmp/_mtp_cfg")
    assert c.mtp_distillation is True and c.mtp_kd_weight == 0.5 and c.mtp_draft_tokens == 6
    # incompatible with logits_chunk_size
    raised = False
    try:
        DistillationConfig(mtp_distillation=True, logits_chunk_size=128, save_directory="/tmp/_mtp_cfg")
    except ValueError:
        raised = True
    assert raised, "mtp_distillation + logits_chunk_size must raise"
    # negative weight
    raised = False
    try:
        DistillationConfig(mtp_distillation=True, mtp_kd_weight=-1.0, save_directory="/tmp/_mtp_cfg")
    except ValueError:
        raised = True
    assert raised, "negative mtp_kd_weight must raise"
    # draft>1 requires mtp_distillation
    raised = False
    try:
        DistillationConfig(mtp_draft_tokens=6, save_directory="/tmp/_mtp_cfg")
    except ValueError:
        raised = True
    assert raised, "mtp_draft_tokens>1 without mtp_distillation must raise"


ALL_TESTS = [
    test_model_exposes_mtp_logits,
    test_model_no_mtp_logits_when_disabled,
    test_mtp_kd_nonnegative,
    test_mtp_kd_perfect_match,
    test_mtp_kd_mask,
    test_mtp_chain_shape_and_consistency,
    test_mtp_chain_perfect_alignment,
    test_config_validation,
]


if __name__ == "__main__":
    print("=" * 80)
    print(f"Running {len(ALL_TESTS)} MTP-distillation tests")
    print("=" * 80)
    for t in ALL_TESTS:
        t()
    n_pass = sum(1 for s, _ in _RESULTS if s == "PASS")
    n_fail = sum(1 for s, _ in _RESULTS if s == "FAIL")
    print("=" * 80)
    print(f"Summary: {n_pass} passed, {n_fail} failed (total: {len(_RESULTS)})")
    sys.exit(0 if n_fail == 0 else 1)
