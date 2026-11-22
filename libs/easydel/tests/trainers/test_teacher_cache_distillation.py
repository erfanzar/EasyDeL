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
"""Offline teacher cache in the distillation step: cached top-k + tail KL equals the live dense top-k KL.

A tiny Llama teacher is dumped into a real on-disk cache (``TeacherCacheWriter`` + ``write_manifest``), the
trainer's host-side join fills the batch from it, and ``distillation_step(teacher_from_cache=True,
teacher_state=None)`` is compared with the live-teacher ``loss_top_k`` / ``loss_add_tail`` step. Runs on TPU.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.data.teacher_cache import TeacherCacheReader, TeacherCacheWriter, teacher_topk_from_logits, write_manifest
from easydel.trainers.distillation_trainer import _fn as distill_fn
from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer
from jax import numpy as jnp

VOCAB, SEQ, BATCH, K = 64, 16, 2, 8


def _tiny_state(seed: int):
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    model = LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))
    return model.to_state(trainable_selector="parameters")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("teacher_cache"))
    teacher, student = _tiny_state(1), _tiny_state(2)
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, VOCAB, size=(BATCH, SEQ)).astype(np.int32)
    attention = np.ones((BATCH, SEQ), np.int32)
    with teacher.model.mesh:
        logits = teacher.model(input_ids=jnp.asarray(input_ids), attention_mask=jnp.asarray(attention)).logits
        idx, logp, tail = (np.asarray(x) for x in teacher_topk_from_logits(logits, K))
    writer = TeacherCacheWriter(root, host_index=0, k=K)
    for r in range(BATCH):
        writer.append(input_ids[r], np.ones(SEQ, np.uint8), idx[r], logp[r], tail[r])
    writer.close()
    write_manifest(root, meta={"teacher_uri": "tiny", "tokenizer": "tok", "k": K, "segment_align": 1}, host_count=1)
    return root, teacher, student, input_ids, attention


def _step(student, batch, teacher_state, **kwargs):
    with student.model.mesh:
        return distill_fn.distillation_step(
            student_state=student,
            batch={k: jnp.asarray(v) for k, v in batch.items()},
            teacher_state=teacher_state,
            is_training=False,
            temperature=1.0,
            alpha=1.0,
            **kwargs,
        )


@pytest.mark.parametrize("topk", [0, 4])
def test_cached_topk_tail_kl_matches_live_dense_topk_kl(setup, topk):
    root, teacher, student, input_ids, attention = setup
    trainer = object.__new__(DistillationTrainer)
    trainer._teacher_cache_reader = TeacherCacheReader(
        root, expect={"teacher_uri": "tiny", "tokenizer": "tok", "segment_align": 1}
    )
    infos: dict = {}
    batch = trainer._join_teacher_cache({"input_ids": input_ids.copy(), "attention_mask": attention.copy()}, infos)
    assert infos["teacher_cache_documents"] == float(BATCH)
    assert batch["teacher_topk_idx"].shape == (BATCH, SEQ, K)

    cached = _step(
        student,
        batch,
        None,
        logits_chunk_size=8,
        teacher_from_cache=True,
        teacher_cache_topk=topk,
        teacher_cache_add_tail=True,
    )
    dense = _step(
        student,
        {"input_ids": input_ids, "attention_mask": attention},
        teacher,
        loss_top_k=topk or K,
        loss_add_tail=True,
    )
    cached_loss, dense_loss = float(cached.loss), float(dense.loss)
    assert np.isfinite(cached_loss) and cached_loss > 0.0
    # The cache stores log-probs in float16; the remaining gap is that rounding.
    np.testing.assert_allclose(cached_loss, dense_loss, rtol=3e-3, atol=3e-3)


def test_cached_step_without_cache_tensors_raises(setup):
    _, _, student, input_ids, attention = setup
    with pytest.raises(ValueError, match="no cached teacher top-k"):
        _step(
            student,
            {"input_ids": input_ids, "attention_mask": attention},
            None,
            logits_chunk_size=8,
            teacher_from_cache=True,
        )


def test_reader_contract_mismatch_and_config_refusals(setup, tmp_path):
    from easydel.data.teacher_cache import TeacherCacheMismatch

    root = setup[0]
    with pytest.raises(TeacherCacheMismatch, match="teacher_uri"):
        TeacherCacheReader(root, expect={"teacher_uri": "other"})
    base = dict(save_directory=str(tmp_path), use_wandb=False, teacher_from_cache=True, temperature=1.0)
    # logits_chunk_size=0 means "no chunking"; the cached KL cannot run there (only None was rejected before).
    for bad in (dict(logits_chunk_size=0, teacher_cache_path=root), dict(logits_chunk_size=8)):
        with pytest.raises(ValueError, match="teacher_from_cache=True` refused"):
            DistillationConfig(**base, **bad)
    with pytest.raises(ValueError, match=r"temperature must be 1\.0"):
        DistillationConfig(**{**base, "temperature": 2.0}, logits_chunk_size=8, teacher_cache_path=root)
    ok = DistillationConfig(**base, logits_chunk_size=8, teacher_cache_path=root)
    assert ok.teacher_from_cache and ok.teacher_cache_add_tail


def test_jax_backend_is_an_accelerator():
    assert jax.default_backend() in ("tpu", "gpu")
