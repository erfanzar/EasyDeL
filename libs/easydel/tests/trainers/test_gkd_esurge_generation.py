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

"""GKD on-policy and sequence-KD generation through eSurge on a tiny model (accelerator test)."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.trainers._shared import copy_frozen_policy
from easydel.trainers.generalized_knowledge_distillation_trainer.gkd_config import GKDConfig
from easydel.trainers.generalized_knowledge_distillation_trainer.gkd_trainer import GKDTrainer

from tests.inference.esurge.runtime_pass._tiny_runtime_common import load_test_tokenizer
from tests.trainers.test_esurge_token_id_rollout import _tiny_llama

MAX_NEW_TOKENS = 6
PROMPT_WIDTH = 10


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tokenizer = load_test_tokenizer()
    student = _tiny_llama(tokenizer).to_state()
    trainer = object.__new__(GKDTrainer)
    trainer.arguments = GKDConfig(
        save_directory=str(tmp_path_factory.mktemp("gkd")),
        max_length=32,
        total_batch_size=2,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.9,
        esurge_max_num_seqs=2,
        esurge_page_size=16,
        esurge_max_cache_tokens=1024,
        esurge_max_num_batched_tokens=32,
        esurge_silent_mode=True,
        esurge_async_scheduling=False,
        esurge_overlap_execution=False,
    )
    trainer.processing_class = tokenizer
    trainer.pad_token_id = tokenizer.pad_token_id
    trainer._warned_missing_prompt = False
    return trainer, student, tokenizer


def _batch(tokenizer, lengths):
    rng = np.random.default_rng(0)
    ids = np.full((len(lengths), PROMPT_WIDTH), tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, PROMPT_WIDTH - length :] = rng.integers(100, 5000, size=length)
        mask[row, PROMPT_WIDTH - length :] = 1
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "prompt_input_ids": ids,
        "prompt_attention_mask": mask,
        "labels": np.full_like(ids, -100),
    }


@pytest.mark.parametrize("source", ["student", "teacher"])
def test_generated_batch_masks_prompt_and_padding_and_bounds_completion(setup, source):
    trainer, student, tokenizer = setup
    batch = _batch(tokenizer, [PROMPT_WIDTH, 6])

    new_batch, info = trainer._apply_generation(student, batch, source=source)

    ids = np.asarray(new_batch["input_ids"])
    labels = np.asarray(new_batch["labels"])
    attention = np.asarray(new_batch["attention_mask"])
    completion = np.asarray(new_batch["completion_mask"])
    np.testing.assert_array_equal(ids[:, :PROMPT_WIDTH], batch["input_ids"])
    assert ids.shape[1] <= PROMPT_WIDTH + MAX_NEW_TOKENS
    assert (labels[:, :PROMPT_WIDTH] == -100).all()
    assert (labels[attention == 0] == -100).all()
    assert (completion[:, :PROMPT_WIDTH] == 0).all()
    np.testing.assert_array_equal(labels[completion == 1], ids[completion == 1])
    lengths = completion.sum(axis=-1)
    assert (lengths >= 1).all() and (lengths <= MAX_NEW_TOKENS).all()
    assert info[f"gkd_completion_length_{source}"] == pytest.approx(float(lengths.mean()))


def test_self_distillation_teacher_shares_the_student_engine_and_a_separate_teacher_does_not(setup):
    trainer, student, tokenizer = setup
    batch = _batch(tokenizer, [PROMPT_WIDTH, 8])
    frozen = copy_frozen_policy(student)
    separate = _tiny_llama(tokenizer).to_state()

    trainer._apply_generation(student, batch, source="student")
    trainer._apply_generation(frozen, batch, source="teacher")
    student_engines = {entry["cache_key"] for entry in student.model.list_esurge_engines()}
    assert {entry["cache_key"] for entry in frozen.model.list_esurge_engines()} == student_engines

    trainer._apply_generation(separate, batch, source="teacher")
    separate_engines = {entry["cache_key"] for entry in separate.model.list_esurge_engines()}
    assert separate_engines and not separate_engines & student_engines
