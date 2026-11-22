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

"""Comprehensive regression tests for Preference Optimization, Distillation & Speculative Decoding findings (PKD-C1..PKD-L3)."""

from __future__ import annotations

import functools
import gc
import inspect
import math
import os
import tempfile
import warnings
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from datasets import Dataset
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig
from easydel.modules.dspark import DSparkConfig, DSparkModel
from easydel.modules.eagle3 import Eagle3Config, Eagle3Model
from easydel.modules.llama.llama_configuration import LlamaConfig
from easydel.modules.llama.modeling_llama import LlamaForCausalLM
from easydel.trainers import _logprob_utils
from easydel.trainers._shared import esurge_generation
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.binary_classifier_optimization_trainer import _fn as bco_fn
from easydel.trainers.binary_classifier_optimization_trainer.bco_config import BCOConfig
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from easydel.trainers.buckets import PatternBucketRule, TrainingBucket
from easydel.trainers.contrastive_preference_optimization_trainer import _fn as cpo_fn
from easydel.trainers.contrastive_preference_optimization_trainer.cpo_config import CPOConfig
from easydel.trainers.direct_preference_optimization_trainer import DPOTrainer
from easydel.trainers.direct_preference_optimization_trainer import _fn as dpo_fn
from easydel.trainers.direct_preference_optimization_trainer._fn import concatenated_forward as dpo_concatenated_forward
from easydel.trainers.direct_preference_optimization_trainer.dpo_config import DPOConfig
from easydel.trainers.distillation_trainer import (
    DistillationConfig,
    DistillationTrainer,
)
from easydel.trainers.distillation_trainer import _fn as kd_fn
from easydel.trainers.distillation_trainer.loss_groups import LOSS_GROUP_IDS_KEY, loss_supervision_mask
from easydel.trainers.generalized_knowledge_distillation_trainer import GKDConfig, GKDTrainer
from easydel.trainers.generalized_knowledge_distillation_trainer import _fn as gkd_fn
from easydel.trainers.gold_trainer import GOLDConfig, GOLDTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOConfig
from easydel.trainers.kto_trainer import _fn as kto_fn
from easydel.trainers.kto_trainer.kto_config import KTOConfig
from easydel.trainers.kto_trainer.kto_trainer import KTOTrainer, _EncoderDecoderKTOPreprocessTransform
from easydel.trainers.minillm_trainer import MiniLLMConfig, MiniLLMTrainer
from easydel.trainers.nash_md_trainer import NashMDConfig, NashMDTrainer
from easydel.trainers.nash_md_trainer._fn import (
    _nash_md_scheduled_loss_cache_key,
    nash_md_step,
)
from easydel.trainers.nash_md_trainer.nash_md_trainer import GeometricMixtureLogitsProcessor
from easydel.trainers.odds_ratio_preference_optimization_trainer import _fn as orpo_fn
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_config import ORPOConfig
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_trainer import ORPOTrainer
from easydel.trainers.on_policy_distillation_trainer import (
    OnPolicyDistillationConfig,
    OnPolicyDistillationTrainer,
)
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from easydel.trainers.sdft_trainer import SDFTConfig, SDFTTrainer, SelfDistillationConfig
from easydel.trainers.self_distillation_policy_optimization import SDPOConfig
from easydel.trainers.seq_kd_trainer import SeqKDConfig, SeqKDTrainer
from easydel.trainers.sparse_distillation_trainer import (
    SparseDistillationConfig,
    SparseDistillationTrainer,
)
from easydel.trainers.sparse_distillation_trainer._fn import (
    partial_kl_distillation_loss,
    sparse_distillation_step,
)
from easydel.trainers.speculative_decoding_trainer import (
    SpeculativeDecodingConfig,
    SpeculativeDecodingTrainer,
)
from easydel.trainers.speculative_decoding_trainer import _fn as spec_fn
from easydel.trainers.ssd_trainer import SSDConfig, SSDTrainer
from easydel.trainers.tpo_trainer import _fn as tpo_fn
from easydel.trainers.tpo_trainer.tpo_config import TPOConfig
from easydel.trainers.tpo_trainer.tpo_preprocess import DataCollatorForTriplePreferenceGrain, TPOPreprocessTransform
from easydel.trainers.tpo_trainer.tpo_trainer import TPOTrainer
from easydel.trainers.trainer.trainer import Trainer
from easydel.trainers.training_utils import ScheduledStepCall, get_scheduled_loss_adapter
from easydel.trainers.xpo_trainer import XPOConfig, XPOTrainer
from easydel.trainers.xpo_trainer._fn import (
    _compute_pair_logps,
    _prepare_xpo_scheduled_batch,
    xpo_step,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

pytestmark = pytest.mark.cpu_ok


# ===========================================================================
# Group C: Distillation, GKD, GOLD, Speculative Decoding (PKD-C4, C5, C8, H4, H8, H10, H12, M2, M4, L2, L3)
# ===========================================================================

class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1
    pad_token = "<eos>"
    eos_token = "<eos>"
    padding_side = "right"
    chat_template = None

    def __call__(self, text, **kwargs):
        del kwargs
        if isinstance(text, str):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}
        return {"input_ids": [[1, 2, 3, 0]] * len(text), "attention_mask": [[1, 1, 1, 0]] * len(text)}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)

    def batch_decode(self, batch_ids, skip_special_tokens=False):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in batch_ids]


def _make_tiny_llama(seed: int = 0, *, vocab_size: int = 32, hidden_size: int = 16, num_layers: int = 2):
    cfg = ed.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        max_position_embeddings=32,
        mlp_bias=False,
        tie_word_embeddings=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="sdpa",
        attn_dtype=jnp.float32,
    )
    model = ed.LlamaForCausalLM(config=cfg, rngs=spx.Rngs(seed), dtype=jnp.float32, param_dtype=jnp.float32)
    return cfg, model


# =============================================================================
# PKD-C4 / PKD-M4 / PKD-L2: GKD offline labels==-100, assistant_masks, bf16, GA
# =============================================================================


def test_pkd_c4_gkd_gjsd_intersects_labels_minus_100_and_keeps_fp32():
    """PKD-C4 + PKD-M4: _ejkernel_gjsd_loss must intersect mask with (labels != -100) and stay float32."""
    # Prompt positions [0, 1] have huge divergence; completion positions [2, 3] match identically.
    student = jnp.array(
        [[[50.0, -50.0, 0.0, 0.0], [-50.0, 50.0, 0.0, 0.0], [1.0, 2.0, -1.0, 0.5], [0.2, -0.3, 1.5, 0.1]]],
        dtype=jnp.bfloat16,
    )
    teacher = jnp.array(
        [[[-50.0, 50.0, 0.0, 0.0], [50.0, -50.0, 0.0, 0.0], [1.0, 2.0, -1.0, 0.5], [0.2, -0.3, 1.5, 0.1]]],
        dtype=jnp.bfloat16,
    )
    attention_mask = jnp.ones((1, 4), dtype=jnp.int32)
    labels = jnp.array([[-100, -100, 2, 3]], dtype=jnp.int32)

    loss = gkd_fn.generalized_jsd_loss(
        student_logits=student,
        teacher_logits=teacher,
        labels=labels,
        mask=attention_mask,
        beta=0.5,
        temperature=1.0,
    )
    assert loss.dtype == jnp.float32
    assert float(loss) < 1e-5


def test_pkd_c4_gkd_step_respects_assistant_masks_aux_z_loss_and_weighted_ga():
    """PKD-C4 + PKD-M4: gkd_step uses assistant_masks, labels==-100, z_loss, aux_loss, and weighted GA."""
    cfg, student = _make_tiny_llama(0)
    _, teacher = _make_tiny_llama(1)
    rng = np.random.default_rng(7)
    input_ids = jnp.asarray(rng.integers(1, 32, size=(4, 8)), dtype=jnp.int32)
    attention_mask = jnp.ones((4, 8), dtype=jnp.int32)
    # Unequal active token counts across microbatches: rows 0-1 have 6 active tokens, rows 2-3 have 2.
    assistant_masks = jnp.array(
        [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.where(assistant_masks == 1, input_ids, -100)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "assistant_masks": assistant_masks,
        "labels": labels,
    }
    with cfg.mesh:
        student_state = EasyDeLState.create(model=student, tx=optax.sgd(0.0), init_opt_state=True)
        teacher_state = teacher.to_state()
        lr_fn = optax.constant_schedule(0.0)
        loss_cfg = LossConfig(z_loss=0.05)

        _, ga1 = gkd_fn.gkd_step(
            student_state, batch, teacher_state, loss_cfg, lr_fn, None, 1, True, 0.5, 1.0, None
        )
        _, ga2 = gkd_fn.gkd_step(
            student_state, batch, teacher_state, loss_cfg, lr_fn, None, 2, True, 0.5, 1.0, None
        )
    assert ga1.z_loss is not None and float(ga1.z_loss) > 0.0
    np.testing.assert_allclose(float(ga2.loss), float(ga1.loss), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(ga2.z_loss), float(ga1.z_loss), rtol=1e-4, atol=1e-5)


def test_pkd_c4_gkd_preprocess_invokes_assistant_supervision_and_strips_sideband(tmp_path):
    """PKD-C4: GKDTrainer._preprocess_batch_input derives completion_mask from labels/-100 and strips sideband."""
    _, student = _make_tiny_llama(0)
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 2, "attention_mask": [[1, 1, 1, 1]] * 2, "labels": [[-100, -100, 3, 4]] * 2}
    )
    args = GKDConfig(
        save_directory=str(tmp_path / "gkd_prep"),
        max_length=8,
        total_batch_size=2,
        lmbda=0.0,
        max_new_tokens=2,
        use_wandb=False,
        do_last_save=False,
    )
    trainer = GKDTrainer(arguments=args, processing_class=_TinyTokenizer(), model=student, train_dataset=ds)
    raw = {
        "input_ids": np.array([[1, 2, 3, 4]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int32),
        "labels": np.array([[-100, -100, 3, 4]], dtype=np.int32),
        "_assistant_only": np.array([1], dtype=np.int32),
        "_dataset_id": np.array([0], dtype=np.int32),
    }
    processed_eval, _ = trainer._preprocess_batch_input(trainer.model_state, dict(raw), is_train=False)
    assert "_assistant_only" not in processed_eval
    assert "_dataset_id" not in processed_eval
    np.testing.assert_array_equal(np.asarray(processed_eval["completion_mask"]), [[0, 0, 1, 1]])


# =============================================================================
# PKD-C5 & PKD-H10: GKD rollout masks with pad==eos, truncation, stale keys, static prompts
# =============================================================================


def test_pkd_c5_gkd_build_batch_preserves_eos_when_pad_equals_eos_and_handles_truncation():
    """PKD-C5: _build_batch_from_sequences uses rollout masks when pad==eos, drops stale keys, left-pad position_ids."""
    trainer = object.__new__(GKDTrainer)
    trainer.arguments = GKDConfig(max_length=6, max_new_tokens=3, temperature=1.0)
    trainer.pad_token_id = 0  # pad_token_id == eos_token_id == 0

    # Prompt len 4 (row0: [pad=0, 5, 6, eos=0], row1: [pad=0, pad=0, 7, 8]);
    # Completion len 4 (row0: [9, eos=0, pad=0, pad=0], row1: [10, 11, 12, eos=0]).
    # Total len = 8 > max_length=6 -> left-truncated by start=2.
    sequences = jnp.array(
        [
            [0, 5, 6, 0, 9, 0, 0, 0],
            [0, 0, 7, 8, 10, 11, 12, 0],
        ],
        dtype=jnp.int32,
    )
    prompt_mask = jnp.array([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=jnp.int32)
    completion_mask = jnp.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=jnp.int32)

    orig_batch = {
        "input_ids": jnp.zeros((2, 4), dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "labels": jnp.zeros((2, 4), dtype=jnp.int32),
        "position_ids": jnp.zeros((2, 4), dtype=jnp.int32),
        "token_type_ids": jnp.ones((2, 4), dtype=jnp.int32),
        "segment_ids": jnp.ones((2, 4), dtype=jnp.int32),
    }

    out = trainer._build_batch_from_sequences(
        orig_batch,
        sequences,
        prompt_seq_len=4,
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
    )
    # Sliced [:, 2:] -> seq len 6
    np.testing.assert_array_equal(np.asarray(out["input_ids"]), [[6, 0, 9, 0, 0, 0], [7, 8, 10, 11, 12, 0]])
    # EOS at col 1 (prompt) and col 3 (row0 comp) and col 5 (row1 comp) MUST stay 1!
    np.testing.assert_array_equal(np.asarray(out["attention_mask"]), [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(np.asarray(out["completion_mask"]), [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
    np.testing.assert_array_equal(
        np.asarray(out["labels"]),
        [[-100, -100, 9, 0, -100, -100], [-100, -100, 10, 11, 12, 0]],
    )
    assert "token_type_ids" not in out
    assert "segment_ids" not in out
    # Left-pad-aware position_ids check with explicit leading padding:
    trainer.arguments.max_length = 8
    out_untrunc = trainer._build_batch_from_sequences(
        orig_batch, sequences, prompt_seq_len=4, prompt_mask=prompt_mask, completion_mask=completion_mask
    )
    np.testing.assert_array_equal(
        np.asarray(out_untrunc["position_ids"]),
        [[0, 0, 1, 2, 3, 4, 0, 0], [0, 0, 0, 1, 2, 3, 4, 5]],
    )


def test_pkd_h10_gkd_seq_kd_lmbda_no_wasted_rollout_and_static_contiguous_prompts():
    """PKD-H10: seq_kd + lmbda does not waste teacher rollout; _extract_prompts is static & contiguous."""
    trainer = object.__new__(GKDTrainer)
    trainer.arguments = GKDConfig(max_length=12, max_new_tokens=4, temperature=1.0, lmbda=1.0, seq_kd=True)
    trainer.pad_token_id = 0
    trainer.lmbda = 1.0
    trainer.seq_kd = True
    trainer._on_policy_rng = SimpleNamespace(random=lambda: 0.0)
    trainer.teacher_state = SimpleNamespace(tag="teacher")
    trainer._user_data_collator = False
    trainer.data_collator = None

    # 1. Contiguous multi-turn prefix + static prompt width across batches with different prompt lengths.
    batch_a = {
        "input_ids": jnp.array([[10, 11, 20, 21, 30, 31, 40, 41]], dtype=jnp.int32),
        "attention_mask": jnp.ones((1, 8), dtype=jnp.int32),
        "completion_mask": jnp.array([[0, 0, 1, 1, 0, 0, 1, 1]], dtype=jnp.int32),
    }
    batch_b = {
        "input_ids": jnp.array([[1, 2, 3, 4, 5, 20, 21, 22]], dtype=jnp.int32),
        "attention_mask": jnp.ones((1, 8), dtype=jnp.int32),
        "completion_mask": jnp.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=jnp.int32),
    }
    p_ids_a, p_mask_a, p_len_a = trainer._extract_prompts(batch_a)
    p_ids_b, p_mask_b, p_len_b = trainer._extract_prompts(batch_b)
    # Static width = max_length - max_new_tokens = 8 for BOTH batches.
    assert p_ids_a.shape == (1, 8)
    assert p_ids_b.shape == (1, 8)
    # Multi-turn batch_a must keep ONLY contiguous prefix [10, 11], NOT [10, 11, 30, 31].
    assert int(p_len_a[0]) == 2
    np.testing.assert_array_equal(np.asarray(p_ids_a[0, -2:]), [10, 11])
    np.testing.assert_array_equal(np.asarray(p_mask_a[0]), [0, 0, 0, 0, 0, 0, 1, 1])
    assert int(p_len_b[0]) == 5
    np.testing.assert_array_equal(np.asarray(p_ids_b[0, -5:]), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(p_mask_b[0]), [0, 0, 0, 1, 1, 1, 1, 1])

    # 2. With seq_kd=True & lmbda=1.0, student sampling runs and teacher rollout is NOT wasted.
    sources_called: list[str] = []

    def _fake_apply_gen(gen_state, b, *, source):
        del gen_state
        sources_called.append(source)
        return dict(b), {f"gkd_completion_length_{source}": 2.0}

    trainer._apply_generation = _fake_apply_gen
    trainer._apply_assistant_supervision = lambda b, infos: b
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b
    trainer._preprocess_batch_input(SimpleNamespace(tag="student"), dict(batch_a), is_train=True)
    assert sources_called == ["student"]


def test_pkd_l2_m4_gkd_config_validation_and_packing_disabled(tmp_path):
    """PKD-L2 + PKD-M4: GKDConfig rejects negative lmbda and non-positive temperature; GKDTrainer disables packing."""
    with pytest.raises(ValueError, match="lmbda"):
        GKDConfig(lmbda=-0.1)
    with pytest.raises(ValueError, match="temperature"):
        GKDConfig(temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        GKDConfig(temperature=-1.0)
    assert GKDConfig(lmbda=0.0).lmbda is None

    _, student = _make_tiny_llama(0)
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 2, "attention_mask": [[1, 1, 1, 1]] * 2})
    args = GKDConfig(
        save_directory=str(tmp_path / "gkd_pack"),
        max_length=4,
        total_batch_size=2,
        lmbda=0.0,
        packing=True,
        use_wandb=False,
        do_last_save=False,
    )
    trainer = GKDTrainer(arguments=args, processing_class=_TinyTokenizer(), model=student, train_dataset=ds)
    assert trainer.arguments.packing is False

    with pytest.raises(ValueError, match="model"):
        GKDTrainer(arguments=args, processing_class=_TinyTokenizer(), model=None, train_dataset=ds)


# =============================================================================
# PKD-C8 / PKD-H12 / PKD-M4 / PKD-L3: Speculative decoding (EAGLE3 + DSpark)
# =============================================================================


def test_pkd_c8_eagle3_speculative_step_passes_return_logits_and_trains():
    """PKD-C8 (#1): Eagle3Model works in speculative_decoding_step with default ttt_length and chain=None."""
    target_cfg, target = _make_tiny_llama(seed=1, vocab_size=32, hidden_size=16, num_layers=5)
    eagle_cfg = Eagle3Config(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        target_layer_ids=(0, 1, 2, 3, 4),
        ttt_length=3,
        step_loss_decay=0.8,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    drafter = Eagle3Model(config=eagle_cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    batch = {
        "input_ids": jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 8), dtype=jnp.int32),
    }
    with target_cfg.mesh:
        drafter_state = EasyDeLState.create(model=drafter, tx=optax.sgd(1e-2), init_opt_state=True)
        target_state = target.to_state()
        lr_fn = optax.constant_schedule(1e-2)

        # Default draft_chain_method="auto" + num_draft_tokens=1 -> promotes to ttt_length=3 and runs compute_mtp_chain.
        new_state, metrics = spec_fn.speculative_decoding_step(
            drafter_state=drafter_state,
            batch=batch,
            target_state=target_state,
            learning_rate_fn=lr_fn,
            is_training=True,
            target_output_hidden_states=True,
        )
        assert jnp.isfinite(metrics.loss)
        assert int(metrics.other_metrics["draft_steps"]) == 3
        assert int(new_state.step) == 1

        # draft_chain_method=None -> extracts draft_logits from Eagle3ForwardOutput (not bare hidden_states).
        eval_metrics = spec_fn.speculative_decoding_step(
            drafter_state=new_state,
            batch=batch,
            target_state=target_state,
            is_training=False,
            draft_chain_method=None,
            target_output_hidden_states=True,
        )
        assert jnp.isfinite(eval_metrics.loss)


def test_pkd_c8_dspark_default_selector_and_l1_confidence_loss(tmp_path):
    """PKD-C8 (#2 & #3): Default SpeculativeDecodingConfig freezes DSpark vocab IO & forwards target logits for L1/conf."""
    target_cfg, target = _make_tiny_llama(seed=1, vocab_size=32, hidden_size=16, num_layers=2)
    dspark_cfg = DSparkConfig(
        vocab_size=32,
        hidden_size=16,
        target_hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        target_layer_ids=(-1, 0),
        block_size=3,
        num_anchors=2,
        mask_token_id=31,
        markov_rank=4,
        freeze_embeddings_and_lm_head=True,
        l1_loss_alpha=0.5,
        confidence_head_alpha=0.25,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    drafter = DSparkModel(config=dspark_cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]] * 2, "attention_mask": [[1] * 8] * 2})

    # Default SpeculativeDecodingConfig has trainable_selector="parameters" and pass_target_outputs=False!
    args = SpeculativeDecodingConfig(
        save_directory=str(tmp_path / "dspark_raw"),
        max_length=8,
        total_batch_size=2,
        target_output_hidden_states=True,
        use_wandb=False,
        do_last_save=False,
    )
    trainer_raw = SpeculativeDecodingTrainer(
        arguments=args,
        processing_class=_TinyTokenizer(),
        drafter_model=drafter,
        target_model=target,
        train_dataset=ds,
    )
    trainable_paths = {p for _, p in trainer_raw.model_state.graphstate.paths()}
    assert not any("embed_tokens" in p or "lm_head" in p for p in trainable_paths)

    # Pre-split EasyDeLState must also resolve drafter_module.trainable_selector cleanly.
    pre_split = drafter.to_state()
    trainer_state = SpeculativeDecodingTrainer(
        arguments=SpeculativeDecodingConfig(
            save_directory=str(tmp_path / "dspark_state"),
            max_length=8,
            total_batch_size=2,
            target_output_hidden_states=True,
            use_wandb=False,
            do_last_save=False,
        ),
        processing_class=_TinyTokenizer(),
        drafter_model=pre_split,
        target_model=target.to_state(),
        train_dataset=ds,
    )
    assert {p for _, p in trainer_state.model_state.graphstate.paths()} == trainable_paths

    # L1 + confidence loss succeeds with default pass_target_outputs=False!
    batch = {
        "input_ids": jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 8), dtype=jnp.int32),
    }
    with target_cfg.mesh:
        metrics = spec_fn.speculative_decoding_step(
            drafter_state=trainer_raw.model_state,
            batch=batch,
            target_state=trainer_raw.target_state,
            is_training=False,
            target_output_hidden_states=True,
            pass_target_outputs=False,
        )
    assert jnp.isfinite(metrics.loss)
    assert "draft_accept_rate" in metrics.other_metrics
    assert "tau_probabilistic" in metrics.other_metrics
    assert float(metrics.other_metrics["deepspec_block_l1_loss"]) >= 0.0
    assert float(metrics.other_metrics["deepspec_block_confidence_loss"]) >= 0.0


def test_pkd_h12_m4_l3_speculative_block_metrics_call_inspection_segments_decay_and_inf():
    """PKD-H12 + PKD-M4 + PKD-L3: DFlash metrics, forward inspection, chain None fallback, segments, decay, -inf."""
    # 1. aligned_target_logits=None (DFlash / CE-only DSpark) must NOT emit fake draft_accept_rate=0 / tau=1.
    block_logits = jnp.zeros((1, 2, 2, 4), dtype=jnp.float32)
    outputs = SimpleNamespace(
        block_logits=block_logits,
        target_ids=jnp.zeros((1, 2, 2), dtype=jnp.int32),
        eval_mask=jnp.ones((1, 2, 2), dtype=jnp.bool_),
        block_keep_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        aligned_target_logits=None,
    )
    ce_cfg = SimpleNamespace(ce_loss_alpha=1.0, l1_loss_alpha=0.0, confidence_head_alpha=0.0, loss_decay_gamma=None)
    _, metrics = spec_fn._deepspec_block_loss(outputs, ce_cfg)
    assert "draft_accept_rate" not in metrics.other_metrics
    assert "tau_probabilistic" not in metrics.other_metrics

    # 2. _module_accepts_block_loss_kwargs must inspect `forward`, not `__call__`.
    class _NonDeepSpecWithBlockSize:
        config = SimpleNamespace(block_size=4)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids, attention_mask=None):
            return input_ids

    assert spec_fn._module_accepts_block_loss_kwargs(_NonDeepSpecWithBlockSize(), None) is False

    # 3. _call_chain_method forwards mask_info/position_ids, and _resolve_draft_logits handles None return.
    seen_kwargs: dict = {}

    class _ChainReturningNone:
        def compute_mtp_chain(self, base_outputs, input_ids, n_steps, attention_mask=None, mask_info=None, position_ids=None):
            del base_outputs, input_ids, n_steps
            seen_kwargs["attention_mask"] = attention_mask
            seen_kwargs["mask_info"] = mask_info
            seen_kwargs["position_ids"] = position_ids
            return None

    target_logits = jnp.zeros((1, 4, 8), dtype=jnp.float32)
    drafter_out = SimpleNamespace(logits=target_logits)
    resolved = spec_fn._resolve_draft_logits(
        _ChainReturningNone(),
        drafter_out,
        {
            "input_ids": jnp.ones((1, 4), dtype=jnp.int32),
            "attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "mask_info": "mi",
            "position_ids": jnp.arange(4)[None, :],
        },
        target_logits=target_logits,
        draft_logits_attr=None,
        draft_chain_method="auto",
        num_draft_tokens=2,
    )
    assert seen_kwargs["mask_info"] == "mi"
    assert seen_kwargs["position_ids"] is not None
    assert resolved.shape == (1, 1, 4, 8)

    # 4. Packed segment_ids prevent cross-document leakage for target_offset=0 and target_offset=1.
    seg_ids = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
    # Place huge mismatch at cross-document boundary positions:
    # For offset=0: pos 2 predicts pos 3 (doc0 -> doc1).
    # For offset=1: pos 1 predicts pos 3 (doc0 -> doc1) and pos 2 predicts pos 4 (doc0 -> doc1).
    t_logits = jnp.zeros((1, 6, 4), dtype=jnp.float32)
    d_logits = jnp.zeros((1, 6, 4), dtype=jnp.float32)
    d_logits = d_logits.at[0, 1].set(jnp.array([20.0, -20.0, 0.0, 0.0]))
    d_logits = d_logits.at[0, 2].set(jnp.array([20.0, -20.0, 0.0, 0.0]))
    # For offset=1, both pos 1 and pos 2 cross the doc0->doc1 boundary, so KL must be 0.0!
    comp_off1 = spec_fn._aligned_step_components(
        d_logits,
        t_logits,
        labels=None,
        input_ids=jnp.zeros((1, 6), dtype=jnp.int32),
        attention_mask=jnp.ones((1, 6), dtype=jnp.int32),
        loss_mask=None,
        temperature=1.0,
        target_offset=1,
        segment_ids=seg_ids,
    )
    assert float(comp_off1["weight_sum"]) == 2.0  # only pos 0 (doc0: 0->2) and pos 3 (doc1: 3->5)
    assert abs(float(comp_off1["kl_num"])) < 1e-6

    # 5. step_loss_decay=1.0 is normalized (no Kx jump vs step_loss_decay=None on uniform valid steps).
    rng = np.random.default_rng(11)
    d_chain = jnp.asarray(rng.normal(size=(3, 2, 10, 6)), dtype=jnp.float32)
    t_seq = jnp.asarray(rng.normal(size=(2, 10, 6)), dtype=jnp.float32)
    ids = jnp.zeros((2, 10), dtype=jnp.int32)
    loss_decay1, _ = spec_fn.speculative_draft_loss(
        d_chain, t_seq, input_ids=ids, num_draft_tokens=3, step_loss_decay=1.0
    )
    loss_none, _ = spec_fn.speculative_draft_loss(
        d_chain, t_seq, input_ids=ids, num_draft_tokens=3, step_loss_decay=None
    )
    # Not a 3x jump: close to 1x (exact equality when per-step counts match; within 15% when trailing differs slightly).
    assert 0.8 <= float(loss_decay1) / float(loss_none) <= 1.25

    # 6. PKD-L3: Masked -inf vocab logits produce finite loss & finite gradients (no 0 * -inf = NaN).
    t_inf = jnp.array([[[1.0, 2.0, -jnp.inf, -jnp.inf], [0.5, -0.5, -jnp.inf, -jnp.inf]]], dtype=jnp.float32)
    d_inf = jnp.array([[[0.2, 1.5, -jnp.inf, -jnp.inf], [0.1, 0.3, -jnp.inf, -jnp.inf]]], dtype=jnp.float32)

    def _kl_fn(d):
        c = spec_fn._aligned_step_components(
            d, t_inf, labels=None, input_ids=jnp.zeros((1, 2), dtype=jnp.int32),
            attention_mask=jnp.ones((1, 2), dtype=jnp.int32), loss_mask=None, temperature=1.0, target_offset=0
        )
        return c["kl_num"] / c["weight_sum"]

    val, grad = jax.value_and_grad(_kl_fn)(d_inf)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))

    # 7. PKD-M4: Cached target hidden states bypass target forward when require_logits=False.
    called = []

    class _FailTarget:
        config = SimpleNamespace(num_hidden_layers=2)

        def __call__(self, *args, **kwargs):
            called.append(True)
            raise AssertionError("target forward should not run when cached hidden states are provided")

    bundle = spec_fn._target_forward_bundle_from_module(
        _FailTarget(),
        {"input_ids": jnp.ones((1, 4), dtype=jnp.int32), "_target_hidden_states": (jnp.ones((1, 4, 8)), jnp.ones((1, 4, 8)))},
        target_logits_attr="logits",
        target_forward_method=None,
        target_output_hidden_states=True,
        pass_target_outputs=False,
        require_logits=False,
        target_layer_ids=(0,),
    )
    assert not called
    assert "hidden_states" in bundle


def test_pkd_m4_speculative_ga2_weighted_loss_grads_and_pooled_metrics():
    """PKD-M4: speculative_decoding_step GA=2 matches GA=1 on loss, grads, AND pooled metrics."""
    cfg, drafter = _make_tiny_llama(0)
    _, target = _make_tiny_llama(1)
    rng = np.random.default_rng(0)
    completion_mask = np.ones((4, 8), dtype=np.int32)
    completion_mask[2:, :-2] = 0
    batch = {
        "input_ids": jnp.asarray(rng.integers(0, 32, (4, 8)), dtype=jnp.int32),
        "attention_mask": jnp.ones((4, 8), dtype=jnp.int32),
        "completion_mask": jnp.asarray(completion_mask),
    }
    with cfg.mesh:
        ds = EasyDeLState.create(model=drafter, tx=optax.sgd(0.0), init_opt_state=True)
        ts = target.to_state()
        lr_fn = optax.constant_schedule(0.0)
        _, ga1 = spec_fn.speculative_decoding_step(ds, batch, ts, None, lr_fn, None, 1, True)
        _, ga2 = spec_fn.speculative_decoding_step(ds, batch, ts, None, lr_fn, None, 2, True)
    np.testing.assert_allclose(float(ga2.loss), float(ga1.loss), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        float(ga2.other_metrics["draft_accept_rate"]),
        float(ga1.other_metrics["draft_accept_rate"]),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(float(ga2.accuracy), float(ga1.accuracy), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(ga2.weight_sum), 18.0)


# =============================================================================
# PKD-H4: eval_config_overrides + eval_state_shardings + pre-split teacher/target selector
# =============================================================================


@pytest.mark.parametrize("family", ["distillation", "gkd", "spec"])
def test_pkd_h4_eval_config_overrides_with_custom_presplit_auxiliary_selector(tmp_path, family):
    """PKD-H4: eval_config_overrides succeeds with eval_state_shardings & non-default pre-split teacher/target."""
    _, student = _make_tiny_llama(seed=0)
    _, aux_model = _make_tiny_llama(seed=1)
    # Pre-split the auxiliary teacher/target with a non-default subset selector (e.g. only lm_head)!
    custom_selector = spx.select().at_path("*lm_head*")
    aux_state = aux_model.to_state(trainable_selector=custom_selector)
    assert len(list(aux_state.graphstate.paths())) < len(list(aux_model.to_state().graphstate.paths()))

    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3, 4]] * 2, "attention_mask": [[1, 1, 1, 1]] * 2, "labels": [[-100, -100, 3, 4]] * 2}
    )
    common = dict(
        save_directory=str(tmp_path / f"eval_ovr_{family}"),
        max_length=4,
        total_batch_size=2,
        eval_batch_size=2,
        eval_config_overrides={"attn_mechanism": "vanilla"},
        use_wandb=False,
        do_last_save=False,
        weight_distribution_log_steps=0,
    )
    if family == "distillation":
        trainer = DistillationTrainer(
            arguments=DistillationConfig(**common),
            processing_class=_TinyTokenizer(),
            student_model=student,
            teacher_model=aux_state,
            train_dataset=ds,
            eval_dataset=ds,
        )
    elif family == "gkd":
        trainer = GKDTrainer(
            arguments=GKDConfig(**common, lmbda=0.0, max_new_tokens=1),
            processing_class=_TinyTokenizer(),
            model=student,
            teacher_model=aux_state,
            train_dataset=ds,
            eval_dataset=ds,
        )
    else:
        trainer = SpeculativeDecodingTrainer(
            arguments=SpeculativeDecodingConfig(**common),
            processing_class=_TinyTokenizer(),
            drafter_model=student,
            target_model=aux_state,
            train_dataset=ds,
            eval_dataset=ds,
        )

    eval_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    eval_aux = trainer._eval_shared_fn_extra_args[0]
    assert str(eval_state.model.config.attn_mechanism) == "vanilla"
    assert str(eval_aux.model.config.attn_mechanism) == "vanilla"
    batch = {
        "input_ids": jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "labels": jnp.array([[-100, -100, 3, 4], [-100, -100, 3, 4]], dtype=jnp.int32),
    }
    metrics = trainer.sharded_evaluation_step_function(
        eval_state, batch, *trainer._eval_shared_fn_extra_args, *trainer._eval_shared_fn_static_args
    )
    assert jnp.isfinite(metrics.loss)


# =============================================================================
# PKD-H8 / PKD-M2 / PKD-M4 / PKD-L3: Distillation & GOLD fixes
# =============================================================================


def test_pkd_h8_mtp_and_chain_distillation_masking_short_seq_labels_and_fp32():
    """PKD-H8: mtp_distillation_loss and mtp_chain_distillation_loss mask tail, survive S<2, honor -100, keep fp32."""
    # 1. With attention_mask=None, loss_mask=None, trailing positions MUST be masked out!
    teacher = jnp.array(
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]]],
        dtype=jnp.bfloat16,
    )
    student = jnp.concatenate([teacher[:, 1:], teacher[:, -1:]], axis=1)
    # Corrupt trailing 2 student MTP positions (no valid t+2 target!).
    student_bad_tail = student.at[:, -2:].set(jnp.array([100.0, -100.0, 0.0], dtype=jnp.bfloat16))
    loss = kd_fn.mtp_distillation_loss(student_bad_tail, teacher, temperature=1.0)
    assert loss.dtype == jnp.float32
    assert float(loss) < 1e-4

    # 2. Chain loss with no masks masks trailing k+1 positions and keeps float32!
    chain_step1 = student_bad_tail
    chain_step2 = jnp.concatenate([teacher[:, 2:], jnp.broadcast_to(teacher[:, -1:], (1, 2, 3))], axis=1)
    chain_step2 = chain_step2.at[:, -3:].set(jnp.array([100.0, -100.0, 0.0], dtype=jnp.bfloat16))
    chain = jnp.stack([chain_step1, chain_step2], axis=0)
    chain_loss, per_step = kd_fn.mtp_chain_distillation_loss(chain, teacher, temperature=1.0)
    assert chain_loss.dtype == jnp.float32
    assert all(p.dtype == jnp.float32 for p in per_step)
    assert float(chain_loss) < 1e-4

    # 3. S < 2 does NOT crash!
    short_t = jnp.ones((2, 1, 4), dtype=jnp.bfloat16)
    short_s = jnp.zeros((2, 1, 4), dtype=jnp.bfloat16)
    short_loss = kd_fn.mtp_distillation_loss(short_s, short_t, temperature=1.0)
    assert jnp.isfinite(short_loss) and float(short_loss) == 0.0

    # 4. labels == -100 are respected!
    # Position t=0 predicts token t+2=2; set labels[0, 2] = -100 and corrupt student[:, 0].
    labels = jnp.array([[1, 1, -100, 1, 1]], dtype=jnp.int32)
    student_bad_pos0 = student.at[:, 0].set(jnp.array([100.0, -100.0, 0.0], dtype=jnp.bfloat16))
    masked_loss = kd_fn.mtp_distillation_loss(student_bad_pos0, teacher, labels=labels, temperature=1.0)
    assert float(masked_loss) < 1e-4


def test_pkd_m2_m4_l3_gold_and_distillation_chunked_parity_zloss_groups_and_hygiene(tmp_path):
    """PKD-M2 + PKD-M4 + PKD-L3: GOLD hash/cache, beta+topk refusal, chunked parity, non-chunked z_loss & groups."""
    # 1. GOLDConfig hashability
    gold_cfg = GOLDConfig(save_directory=str(tmp_path / "gold"), use_wandb=False)
    assert isinstance(hash(gold_cfg), int)

    # 2. GOLDTrainer with teacher_from_cache=True does NOT clone a self-teacher (which would raise ValueError).
    gold_cache_cfg = GOLDConfig(
        save_directory=str(tmp_path / "gold_cache"),
        teacher_from_cache=True,
        teacher_cache_path=str(tmp_path / "missing_cache"),
        logits_chunk_size=4,
        temperature=1.0,
        use_wandb=False,
    )
    _, student = _make_tiny_llama(0)
    with pytest.raises(Exception) as exc_info:
        GOLDTrainer(arguments=gold_cache_cfg, model=student, teacher_model=None, processing_class=_TinyTokenizer())
    # Should fail trying to open missing_cache (TeacherCacheReader), NOT on "teacher_from_cache=True and teacher_model were both given"!
    assert "were both given" not in str(exc_info.value)

    # 3. beta + loss_top_k is genuinely honored (not silently ignored!) and matches full-vocab GJSD at k == V.
    cfg_beta_topk = DistillationConfig(
        save_directory=str(tmp_path / "beta_topk"), beta=0.3, loss_top_k=4, loss_add_tail=True, use_wandb=False
    )
    assert cfg_beta_topk.beta == 0.3 and cfg_beta_topk.loss_top_k == 4

    rng = np.random.default_rng(19)
    s = jnp.asarray(rng.normal(size=(2, 6, 8)), dtype=jnp.float32)
    t = jnp.asarray(rng.normal(size=(2, 6, 8)), dtype=jnp.float32)
    am = jnp.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
    labels = jnp.array([[-100, 1, 2, -100, 4, -100], [-100, -100, 3, 5, -100, -100]], dtype=jnp.int32)
    group_ids = jnp.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=jnp.int32)

    topk_fwd, _ = kd_fn.distillation_loss(s, t, attention_mask=am, alpha=1.0, temperature=1.5, loss_top_k=3, loss_add_tail=True)
    topk_jsd, _ = kd_fn.distillation_loss(
        s, t, attention_mask=am, alpha=1.0, temperature=1.5, beta=0.3, loss_top_k=3, loss_add_tail=True
    )
    assert abs(float(topk_jsd) - float(topk_fwd)) > 1e-3  # beta is NOT silently ignored!
    full_jsd, _ = kd_fn.distillation_loss(s, t, attention_mask=am, alpha=1.0, temperature=1.5, beta=0.3)
    topk_full_jsd, _ = kd_fn.distillation_loss(s, t, attention_mask=am, alpha=1.0, temperature=1.5, beta=0.3, loss_top_k=8)
    np.testing.assert_allclose(float(topk_full_jsd), float(full_jsd), rtol=1e-4, atol=1e-5)

    # 4. Chunked parity for beta=0.3, cakld_gamma=0.6, loss_top_k=3+tail, and beta+loss_top_k+tail!
    for extra in (
        dict(beta=0.3),
        dict(beta=1.0),
        dict(cakld_gamma=0.6),
        dict(loss_top_k=3, loss_add_tail=True),
        dict(beta=0.3, loss_top_k=3, loss_add_tail=True),
    ):
        dense_loss, dense_m = kd_fn.distillation_loss(
            s, t, attention_mask=am, labels=labels, use_hard_labels=True, temperature=1.5, alpha=0.8,
            metric_group_ids=group_ids, num_metric_groups=2, **extra
        )
        chunk_loss, chunk_m = kd_fn.chunked_distillation_loss(
            s, t, lambda x: x, lambda x: x, attention_mask=am, labels=labels, use_hard_labels=True,
            temperature=1.5, alpha=0.8, chunk_size=2, metric_group_ids=group_ids, num_metric_groups=2, **extra
        )
        np.testing.assert_allclose(float(chunk_loss), float(dense_loss), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(chunk_m["group_kl_ratio"]), np.asarray(dense_m["group_kl_ratio"]), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(chunk_m["group_mass_ratio"]), np.asarray(dense_m["group_mass_ratio"]), rtol=1e-4, atol=1e-5
        )

    # 5. Non-chunked z_loss ignores labels == -100 (matches chunked z_loss!).
    class _FixedLM:
        mesh = None
        config = SimpleNamespace(partition_axis=None)

        def __init__(self, logits):
            self._logits = logits
            self.seen_keys: set[str] = set()

        def __call__(self, input_ids, attention_mask=None, apply_lm_head=True, **kwargs):
            self.seen_keys.update(kwargs.keys())
            return SimpleNamespace(logits=self._logits, last_hidden_state=self._logits, aux_loss=None)

        def make_lm_head_fn(self):
            return lambda h: h

    s_mod = _FixedLM(s)
    t_mod = _FixedLM(t)
    s_st = SimpleNamespace(model=s_mod, graphstate={}, step=jnp.asarray(0), merge=lambda _: s_mod)
    t_st = SimpleNamespace(model=t_mod, graphstate={}, merge=lambda _: t_mod)
    batch_z = {
        "input_ids": jnp.ones((2, 6), dtype=jnp.int32),
        "attention_mask": am,
        "labels": labels,
        "_teacher_logits": t,  # also checks PKD-L3: _teacher_* keys must NOT leak to **kwargs student!
    }
    eval_dense = kd_fn.distillation_step(
        s_st, batch_z, t_st, loss_config=LossConfig(z_loss=0.1), is_training=False, temperature=1.0, alpha=1.0
    )
    assert "_teacher_logits" not in s_mod.seen_keys
    batch_chunk = {"input_ids": jnp.ones((2, 6), dtype=jnp.int32), "attention_mask": am, "labels": labels}
    eval_chunk = kd_fn.distillation_step(
        s_st, batch_chunk, t_st, loss_config=LossConfig(z_loss=0.1), is_training=False, temperature=1.0, alpha=1.0,
        logits_chunk_size=2,
    )
    np.testing.assert_allclose(float(eval_dense.z_loss), float(eval_chunk.z_loss), rtol=1e-5, atol=1e-5)

    # 6. _attach_loss_groups skips cleanly on unattributed eval batch (is_train=False) while raising on train.
    dt = object.__new__(DistillationTrainer)
    dt.arguments = SimpleNamespace(loss_group_weights=[0.5, 0.5], loss_group_dataset_map=None, loss_group_max_multiplier=None)
    unattrib = {"input_ids": np.ones((2, 4), np.int32), "attention_mask": np.ones((2, 4), np.int32)}
    eval_out = dt._attach_loss_groups(dict(unattrib), {}, None, None, is_train=False)
    assert LOSS_GROUP_IDS_KEY not in eval_out
    with pytest.raises(ValueError, match="dataset attribution"):
        dt._attach_loss_groups(dict(unattrib), {}, None, None, is_train=True)

    # 7. PKD-L3: _normalize_attention uses jnp.abs in float32; cached temp!=1.0 raises; explicit None mask falls back.
    signed_attn = jnp.array([[1.0, -1.0, 2.0]], dtype=jnp.bfloat16)
    normed = kd_fn._normalize_attention(signed_attn)
    assert jnp.all(jnp.isfinite(normed))
    np.testing.assert_allclose(np.asarray(jnp.sum(jnp.abs(normed.astype(jnp.float32)), axis=-1)), [1.0], atol=1e-2)

    with pytest.raises(ValueError, match=r"temperature == 1\.0"):
        kd_fn.chunked_distillation_loss(
            s, None, lambda x: x, None, temperature=2.0, chunk_size=2,
            teacher_topk_idx=jnp.zeros((2, 6, 2), dtype=jnp.int32),
            teacher_topk_logp=jnp.zeros((2, 6, 2), dtype=jnp.float32),
        )

    sup = loss_supervision_mask(
        {"input_ids": np.ones((1, 4), np.int32), "completion_mask": None, "attention_mask": np.array([[1, 1, 0, 0]])},
        shift_tokens=False,
    )
    np.testing.assert_array_equal(sup, [[1.0, 1.0, 0.0, 0.0]])


def test_pkd_m4_distillation_bucket_teacher_overrides_and_mtp_freeze_ordering(tmp_path):
    """PKD-M4: mtp_loss_coef freezes before bucket/eval copies, and buckets get per-bucket teacher graphdefs."""
    _, student = _make_tiny_llama(seed=0)
    _, teacher = _make_tiny_llama(seed=1)
    student.config.mtp_loss_coef = 0.5  # simulate checkpoint with active MTP head

    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]] * 2, "attention_mask": [[1, 1, 1, 1]] * 2})
    args = DistillationConfig(
        save_directory=str(tmp_path / "distill_buckets"),
        max_length=4,
        total_batch_size=2,
        mtp_distillation=False,
        eval_config_overrides={"attn_mechanism": "vanilla"},
        use_wandb=False,
        do_last_save=False,
        weight_distribution_log_steps=0,
    )
    buckets = [
        TrainingBucket(name="b0", max_length=4, total_batch_size=2, config={"attn_mechanism": "vanilla"}),
    ]
    trainer = DistillationTrainer(
        arguments=args,
        processing_class=_TinyTokenizer(),
        student_model=student,
        teacher_model=teacher,
        train_dataset=ds,
        eval_dataset=ds,
        buckets=buckets,
        bucket_rule=PatternBucketRule(pattern=[0]),
    )
    # Frozen before eval_config_overrides and buckets deep-copied config!
    eval_state = trainer._eval_state_for_compiled_step(trainer.model_state)
    assert float(trainer.model.config.mtp_loss_coef) == 0.0
    assert float(eval_state.model.config.mtp_loss_coef) == 0.0
    assert len(trainer._bucket_teacher_states) == 1
    assert str(trainer._bucket_teacher_states[0].model.config.attn_mechanism) == "vanilla"
    assert str(trainer.teacher_state.model.config.attn_mechanism) == "sdpa"

# ===========================================================================
# Group A: Offline Preference Optimization & Logprob Utils (PKD-C1, C3, H2, H3, H4, H5, H6, H7, H10, H16, M1-M4, L1, L2)
# ===========================================================================

def _single_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:1]), ("dp",))


class _TinyCharTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        ids = [ord(ch) % 20 + 3 for ch in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _CausalLMWithBackbone:
    """Mimics ``LlamaForCausalLM`` exposing a headless ``.model`` backbone."""

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.model = SimpleNamespace(
            __call__=lambda **kwargs: SimpleNamespace(logits=None, last_hidden_state=None),
        )
        self.received_kwargs: list[dict] = []

    def __call__(self, input_ids, attention_mask, **kwargs):
        self.received_kwargs.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        base = jnp.arange(self.vocab_size, dtype=jnp.float32) * 0.25
        logits = jnp.broadcast_to(base, (*input_ids.shape, self.vocab_size))
        return SimpleNamespace(logits=logits, aux_loss=None)


class _SimpleParamLM(spx.Module):
    """Minimal differentiable SpectraX causal LM whose logits depend on token history."""

    def __init__(self, vocab_size: int = 8, mesh: Mesh | None = None, aux_loss=None):
        self.vocab_size = vocab_size
        self.mesh = mesh
        self._aux_loss = aux_loss
        self.embed = spx.Parameter(
            jnp.sin(jnp.arange(vocab_size * vocab_size, dtype=jnp.float32).reshape(vocab_size, vocab_size) * 1.3 + 0.2)
        )
        self.bias = spx.Parameter(jnp.linspace(-0.2, 0.3, vocab_size, dtype=jnp.float32))

    def _get_mesh(self, mesh=None):
        return mesh if mesh is not None else self.mesh

    def __call__(self, input_ids, attention_mask, **kwargs):
        del kwargs
        tok_embeds = self.embed.value[input_ids]
        masked = tok_embeds * attention_mask[..., None].astype(jnp.float32)
        prefix_ctx = jnp.cumsum(masked, axis=1) / jnp.maximum(jnp.cumsum(attention_mask, axis=1)[..., None], 1.0)
        logits = prefix_ctx + self.bias.value
        return SimpleNamespace(logits=logits, aux_loss=self._aux_loss)


# ---------------------------------------------------------------------------
# PKD-C1: ORPO concatenated_forward must not unwrap ForCausalLM.model
# ---------------------------------------------------------------------------
def test_pkd_c1_orpo_does_not_unwrap_causal_lm_backbone():
    lm = _CausalLMWithBackbone(vocab_size=6)
    batch = {
        "chosen_input_ids": jnp.array([[1, 2, 3, 4]]),
        "chosen_attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
        "chosen_labels": jnp.array([[-100, -100, 3, 4]]),
        "rejected_input_ids": jnp.array([[1, 2, 5, 4]]),
        "rejected_attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
        "rejected_labels": jnp.array([[-100, -100, 5, 4]]),
    }
    chosen_lp, rejected_lp, _, _, nll, acc = orpo_fn.concatenated_forward(
        lm,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert len(lm.received_kwargs) == 1
    assert np.isfinite(np.asarray(chosen_lp)).all()
    assert np.isfinite(np.asarray(rejected_lp)).all()
    assert np.isfinite(float(nll))
    assert 0.0 <= float(acc) <= 1.0
    # Non-callable state wrapper (e.g. SimpleNamespace(model=lm)) still unwraps to lm.
    chosen_lp_ns, rejected_lp_ns, _, _, nll_ns, acc_ns = orpo_fn.concatenated_forward(
        SimpleNamespace(model=lm),
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    assert len(lm.received_kwargs) == 2
    assert np.allclose(np.asarray(chosen_lp_ns), np.asarray(chosen_lp))
    assert np.allclose(np.asarray(rejected_lp_ns), np.asarray(rejected_lp))
    assert np.isclose(float(nll_ns), float(nll))
    assert np.isclose(float(acc_ns), float(acc))


# ---------------------------------------------------------------------------
# PKD-C3: KTO _build_kl_batch genuinely mismatches decoder-only prompt & completion
# ---------------------------------------------------------------------------
def test_pkd_c3_kto_build_kl_batch_mismatches_decoder_only_pairs():
    batch = {
        "prompt_input_ids": jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.array([[0, 1, 1], [1, 1, 1]], dtype=jnp.int32),
        # Row 0: prompt [1, 2], completion [6, 7, 5]
        # Row 1: prompt [3, 4, 5], completion [2]
        "completion_input_ids": jnp.array([[1, 2, 6, 7, 5, 0], [3, 4, 5, 2, 0, 0]], dtype=jnp.int32),
        "completion_attention_mask": jnp.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=jnp.int32),
        "completion_labels": jnp.array([[-100, -100, 6, 7, 5, -100], [-100, -100, -100, 2, -100, -100]], dtype=jnp.int32),
    }

    kl_batch = kto_fn._build_kl_batch(
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )

    # Row 0 must pair prompt_0 ([1, 2]) with completion_1 ([2])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_input_ids"][0]), [1, 2, 2, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_attention_mask"][0]), [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_labels"][0]), [-100, -100, 2, -100, -100, -100])

    # Row 1 must pair prompt_1 ([3, 4, 5]) with completion_0 ([6, 7, 5])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_input_ids"][1]), [3, 4, 5, 6, 7, 5])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_attention_mask"][1]), [1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(np.asarray(kl_batch["completion_labels"][1]), [-100, -100, -100, 6, 7, 5])

    # Context-dependent model must produce DIFFERENT logps on kl_batch than roll(matched_logps, 1)
    model = _SimpleParamLM(vocab_size=8)
    matched = bco_fn.concatenated_forward(
        model,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )["completion_logps"]
    mismatched = bco_fn.concatenated_forward(
        model,
        kl_batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )["completion_logps"]
    assert not np.allclose(np.asarray(mismatched), np.asarray(jnp.roll(matched, shift=1, axis=0)))


# ---------------------------------------------------------------------------
# PKD-H2: Numerical stability in DPO, CPO, ORPO, and _logprob_utils
# ---------------------------------------------------------------------------
def test_pkd_h2_dpo_discopop_and_forward_kl_finite_on_extreme_logratios():
    def discopop_sum(chosen_lp):
        return dpo_fn.compute_dpo_losses(
            chosen_lp,
            jnp.array([-1.0, -2.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            beta=1.0,
            label_smoothing=0.0,
            loss_type="discopop",
            discopop_tau=0.05,
        ).sum()

    extreme_chosen = jnp.array([-200.0, -500.0], dtype=jnp.float32)
    val, grad = jax.value_and_grad(discopop_sum)(extreme_chosen)
    assert np.isfinite(float(val))
    assert np.isfinite(np.asarray(grad)).all()

    def forward_kl_sum(chosen_lp):
        return dpo_fn.compute_dpo_losses(
            chosen_lp,
            jnp.array([-250.0, -300.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            beta=0.1,
            label_smoothing=0.0,
            loss_type="sigmoid",
            f_divergence_type="forward_kl",
        ).sum()

    val_fkl, grad_fkl = jax.value_and_grad(forward_kl_sum)(extreme_chosen)
    assert np.isfinite(float(val_fkl))
    assert np.isfinite(np.asarray(grad_fkl)).all()


@pytest.mark.parametrize("alpha", [0.1, 1e-6, -0.2])
def test_pkd_h2_cpo_alphapo_finite_on_extreme_logps(alpha):
    chosen = jnp.array([-150.0, -2000.0], dtype=jnp.float32)
    rejected = jnp.array([-160.0, -2010.0], dtype=jnp.float32)

    def loss_sum(c):
        losses, _, _ = cpo_fn.cpo_loss(
            c,
            rejected,
            beta=0.1,
            label_smoothing=0.0,
            loss_type="simpo",
            simpo_gamma=0.5,
            alpha=alpha,
        )
        return losses.sum()

    val, grad = jax.value_and_grad(loss_sum)(chosen)
    assert np.isfinite(float(val))
    assert np.isfinite(np.asarray(grad)).all()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pkd_h2_orpo_log1mexp_and_odds_ratio_finite_at_zero_logp(dtype):
    def or_loss_sum(chosen_lp, rejected_lp):
        losses, *_ = orpo_fn.odds_ratio_loss(0.1, chosen_lp, rejected_lp)
        return losses.sum()

    zero = jnp.array([0.0, 0.0], dtype=dtype)
    mixed = jnp.array([0.0, -2.5], dtype=dtype)

    v_both, (gc_both, gr_both) = jax.value_and_grad(or_loss_sum, argnums=(0, 1))(zero, zero)
    assert np.isfinite(float(v_both))
    assert np.isfinite(np.asarray(gc_both)).all()
    assert np.isfinite(np.asarray(gr_both)).all()

    v_mix, (gc_mix, gr_mix) = jax.value_and_grad(or_loss_sum, argnums=(0, 1))(zero, mixed)
    assert np.isfinite(float(v_mix))
    assert np.isfinite(np.asarray(gc_mix)).all()
    assert np.isfinite(np.asarray(gr_mix)).all()


@pytest.mark.parametrize("chunk_size", [None, 2, 3])
def test_pkd_h2_logprob_utils_entropy_masked_neginf_and_large_offset(chunk_size):
    # Masked -inf vocab entries must give finite entropy AND finite gradients.
    base_logits = jnp.array([[[1.5, -jnp.inf, 0.5, -jnp.inf, -0.2]]], dtype=jnp.float32)
    targets = jnp.array([[0]], dtype=jnp.int32)

    def entropy_fn(x):
        _, ent = _logprob_utils.compute_token_logps_and_entropies_chunked(
            x, targets, return_entropy=True, chunk_size=chunk_size
        )
        return ent.sum()

    ent_val, ent_grad = jax.value_and_grad(entropy_fn)(base_logits)
    finite_logits = np.array([1.5, 0.5, -0.2], dtype=np.float64)
    probs = np.exp(finite_logits - np.max(finite_logits))
    probs /= probs.sum()
    expected_ent = float(-np.sum(probs * np.log(probs)))
    np.testing.assert_allclose(float(ent_val), expected_ent, rtol=1e-5, atol=1e-5)
    assert np.isfinite(np.asarray(ent_grad)).all()

    # Large constant offset must preserve shift-invariant entropy precision.
    shifted_logits = jnp.array([[[10000.0 + 1.5, 10000.0 + 0.5, 10000.0 - 0.2, 10000.0 - 1.0, 10000.0 + 0.0]]])
    unshifted_logits = jnp.array([[[1.5, 0.5, -0.2, -1.0, 0.0]]])
    _, ent_shifted = _logprob_utils.compute_token_logps_and_entropies_chunked(
        shifted_logits, targets, return_entropy=True, chunk_size=chunk_size
    )
    _, ent_unshifted = _logprob_utils.compute_token_logps_and_entropies_chunked(
        unshifted_logits, targets, return_entropy=True, chunk_size=None
    )
    np.testing.assert_allclose(np.asarray(ent_shifted), np.asarray(ent_unshifted), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# PKD-H3: Decoder-only jnp.roll wrap-around guard in DPO, CPO, and TPO
# ---------------------------------------------------------------------------
def test_pkd_h3_decoder_only_roll_does_not_wrap_token_zero():
    # Logits differ dramatically at the final position so wrapping token 0 into position L-1
    # would immediately shift summed logps and lengths.
    class PositionSpecificModel:
        def __call__(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            bsz, seq_len = input_ids.shape
            logits = jnp.zeros((bsz, seq_len, 6), dtype=jnp.float32)
            # Make position L-1 strongly penalize token 1 (which sits at index 0 after keep_end).
            logits = logits.at[:, -1, 1].set(-50.0)
            return SimpleNamespace(logits=logits)

    model = PositionSpecificModel()
    # Prompt length 1, completion length 3, max_length=3 with keep_end -> prompt dropped,
    # so loss_mask[:, 0] == 1 BEFORE shift! Only positions 1 and 2 (2 tokens, NOT 3) are causally scoreable.
    batch = {
        "prompt_input_ids": jnp.array([[5]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.array([[1]], dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[1, 2, 3]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.array([[1, 1, 1]], dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[1, 4, 2]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.array([[1, 1, 1]], dtype=jnp.int32),
        "reference_input_ids": jnp.array([[1, 3, 4]], dtype=jnp.int32),
        "reference_attention_mask": jnp.array([[1, 1, 1]], dtype=jnp.int32),
    }
    expected_logp = -2.0 * math.log(6.0)

    dpo_out = dpo_fn.concatenated_forward(
        model,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=3,
        truncation_mode="keep_end",
    )
    assert float(dpo_out["chosen_lengths"][0]) == pytest.approx(2.0)
    assert float(dpo_out["chosen_logps"][0]) == pytest.approx(expected_logp, abs=1e-5)

    cpo_out = cpo_fn.concatenated_forward(
        model,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=3,
        truncation_mode="keep_end",
    )
    assert float(cpo_out["chosen_lengths"][0]) == pytest.approx(2.0)
    assert float(cpo_out["chosen_logps"][0]) == pytest.approx(expected_logp, abs=1e-5)

    tpo_out = tpo_fn.tpo_concatenated_forward(
        model,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=3,
        truncation_mode="keep_end",
    )
    assert float(tpo_out["chosen_lengths"][0]) == pytest.approx(2.0)
    assert float(tpo_out["chosen_logps"][0]) == pytest.approx(expected_logp, abs=1e-5)


# ---------------------------------------------------------------------------
# PKD-H4: eval_state_shardings & auxiliary eval_config_overrides across ORPO/BCO/KTO/TPO
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DummyState:
    graphdef: str
    graphstate: object = None
    model: object = None

    @property
    def shardings(self):
        return self

    def replace(self, **changes):
        return replace(self, **changes)


@pytest.mark.parametrize("trainer_cls,has_ref", [(ORPOTrainer, False), (BCOTrainer, True), (KTOTrainer, True), (TPOTrainer, False)])
def test_pkd_h4_eval_state_shardings_and_aux_reference_graphdef(monkeypatch, trainer_cls, has_ref):
    import importlib

    module = importlib.import_module(trainer_cls.__module__)
    captured_eval_in_shardings = {}

    def fake_compile_step(fn, *, in_shardings, **kwargs):
        del kwargs
        if fn.__name__ in ("evaluation_step", "tpo_evaluation_step") or (
            len(in_shardings) >= 2 and in_shardings[0].graphdef == "eval-policy"
        ):
            captured_eval_in_shardings["shardings"] = in_shardings
        return SimpleNamespace(fn=fn, in_shardings=in_shardings)

    monkeypatch.setattr(module, "replicated_named_sharding", lambda mesh: "replicated")
    monkeypatch.setattr(module, "resolve_straight_through_emulator", lambda **kwargs: None)
    monkeypatch.setattr(module, "compile_trainer_auxiliary", lambda fn, **kwargs: fn)
    monkeypatch.setattr(module, "compile_trainer_step", fake_compile_step)

    trainer = object.__new__(trainer_cls)
    trainer._model = SimpleNamespace(mesh=None, output_router_logits=False)
    trainer.padding_value = 0
    trainer.scheduler = None
    trainer.calculate_kl = False
    trainer.aux_loss_enabled = False
    trainer.aux_loss_coef = 0.0
    trainer._runtime_trace = lambda *args, **kwargs: None
    trainer.state_shardings = _DummyState("train-policy")
    trainer.model_state = _DummyState("train-policy", model=trainer._model)
    trainer._eval_graphdef = "eval-policy"
    trainer.arguments = SimpleNamespace(
        quantization_mode=None,
        quantization_group_size=64,
        quantization_bits=8,
        tensor_straight_through=False,
        straight_through_emulator=None,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=16,
        truncation_mode="keep_end",
        aux_loss_enabled=False,
        loss_type="sigmoid",
        tpo_alpha=1.0,
        tpo_l_gamma=0.5,
        logprob_vocab_chunk_size=None,
        beta=0.1,
        label_smoothing=0.0,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_config=None,
        step_partition_spec=None,
        gradient_accumulation_steps=1,
        mpmd_scheduler=None,
        ensure_checkpoint_path=lambda: None,
        get_streaming_checkpointer=lambda: None,
    )
    if has_ref:
        trainer.reference_state = _DummyState("train-ref", model=SimpleNamespace())
        trainer._eval_reference_graphdef = "eval-ref"

    trainer.configure_functions()
    eval_in = captured_eval_in_shardings["shardings"]
    assert eval_in[0].graphdef == "eval-policy"
    if has_ref:
        assert eval_in[2].graphdef == "eval-ref"
        assert trainer._eval_shared_fn_extra_args[0].graphdef == "eval-ref"
        assert trainer._train_shared_fn_extra_args[0].graphdef == "train-ref"


# ---------------------------------------------------------------------------
# PKD-H5: out_shardings=empty_sharding (not 1-tuple) + static_argnames + DPO precompute JIT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("trainer_cls", [DPOTrainer, TPOTrainer])
def test_pkd_h5_auxiliary_forward_out_shardings_not_tuple_and_static_argnames(monkeypatch, trainer_cls):
    import importlib

    module = importlib.import_module(trainer_cls.__module__)
    aux_calls = []

    def fake_compile_aux(fn, **kwargs):
        aux_calls.append(kwargs)
        return fn

    monkeypatch.setattr(module, "replicated_named_sharding", lambda mesh: "replicated_sharding")
    monkeypatch.setattr(module, "resolve_straight_through_emulator", lambda **kwargs: None)
    monkeypatch.setattr(module, "compile_trainer_auxiliary", fake_compile_aux)
    monkeypatch.setattr(module, "compile_trainer_step", lambda fn, **kwargs: SimpleNamespace())

    trainer = object.__new__(trainer_cls)
    trainer._model = SimpleNamespace(mesh=None)
    trainer.padding_value = 0
    trainer.scheduler = None
    trainer._runtime_trace = lambda *args, **kwargs: None
    trainer.state_shardings = _DummyState("train")
    trainer.model_state = trainer.state_shardings
    trainer._eval_graphdef = None
    trainer.reference_state = None
    trainer.arguments = SimpleNamespace(
        quantization_mode=None,
        quantization_group_size=64,
        quantization_bits=8,
        tensor_straight_through=False,
        straight_through_emulator=None,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        max_length=16,
        truncation_mode="keep_end",
        aux_loss_enabled=False,
        loss_type="sigmoid",
        loss_weights=None,
        f_divergence_type="reverse_kl",
        f_alpha_divergence_coef=0.5,
        use_weighting=False,
        discopop_tau=0.05,
        ld_alpha=None,
        rpo_alpha=None,
        reference_free=True,
        force_use_ref_model=False,
        tpo_alpha=1.0,
        tpo_l_gamma=0.5,
        logprob_vocab_chunk_size=None,
        beta=0.1,
        label_smoothing=0.0,
        loss_config=None,
        step_partition_spec=None,
        gradient_accumulation_steps=1,
        mpmd_scheduler=None,
        ensure_checkpoint_path=lambda: None,
        get_streaming_checkpointer=lambda: None,
    )
    trainer.configure_functions()
    assert len(aux_calls) == 1
    assert aux_calls[0]["out_shardings"] == "replicated_sharding"
    assert not isinstance(aux_calls[0]["out_shardings"], tuple)
    assert "logprob_vocab_chunk_size" in aux_calls[0]["static_argnames"]
    if trainer_cls is DPOTrainer:
        assert "ld_alpha" in aux_calls[0]["static_argnames"]


def test_pkd_h5_jitted_dict_forward_runs_on_mesh():
    mesh = _single_device_mesh()
    replicated = NamedSharding(mesh, PartitionSpec())
    model = _SimpleParamLM(vocab_size=6, mesh=mesh)
    batch = {
        "prompt_input_ids": jnp.array([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 2]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
    }

    trainer = object.__new__(DPOTrainer)
    trainer.padding_value = 0
    trainer.arguments = DPOConfig(max_length=8, max_prompt_length=4)
    trainer.arguments.is_encoder_decoder = False
    jitted_fn = trainer._compile_concatenated_forward(mesh=mesh, empty_sharding=replicated)
    out = jitted_fn(model, batch=batch)
    assert isinstance(out, dict)
    assert "chosen_logps" in out and "rejected_logps" in out
    assert "mean_chosen_logits" not in out

    # Also verify _precompute_reference_log_probs_for_split JIT-compiles concatenated_forward before iterating
    pre_tr = object.__new__(DPOTrainer)
    pre_tr._model = model
    pre_tr.model_state = EasyDeLState.create(model=model)
    pre_tr.reference_state = EasyDeLState.create(model=model)
    pre_tr.padding_value = 0
    pre_tr.arguments = DPOConfig(max_length=8, max_prompt_length=4, precompute_ref_log_probs=True)
    pre_tr.arguments.is_encoder_decoder = False
    pre_tr.concatenated_forward = None
    pre_tr.dataset_train = Dataset.from_dict(
        {
            "prompt_input_ids": [[1, 2]],
            "prompt_attention_mask": [[1, 1]],
            "chosen_input_ids": [[3, 4]],
            "chosen_attention_mask": [[1, 1]],
            "rejected_input_ids": [[5, 2]],
            "rejected_attention_mask": [[1, 1]],
        }
    )
    pre_tr._train_source = pre_tr._to_sharded_source(pre_tr.dataset_train)
    pre_tr.data_collator = lambda rows: {k: np.asarray([r[k] for r in rows], dtype=np.int32) for k in rows[0]}
    compile_calls = {"count": 0}
    orig_compile = DPOTrainer._compile_concatenated_forward

    def spy_compile(self, **kwargs):
        compile_calls["count"] += 1
        return orig_compile(self, **kwargs)

    pre_tr._compile_concatenated_forward = spy_compile.__get__(pre_tr, DPOTrainer)
    ok = pre_tr._precompute_reference_log_probs_for_split(
        dataset_attr="dataset_train", source_attr="_train_source", batch_size=1, is_train=True, desc="test"
    )
    assert ok is True
    assert compile_calls["count"] == 1
    assert pre_tr.concatenated_forward is not None
    assert "ref_chosen_logps" in pre_tr.dataset_train.column_names


# ---------------------------------------------------------------------------
# PKD-H6: aux_loss=None guard across DPO, BCO, KTO, TPO
# ---------------------------------------------------------------------------
def test_pkd_h6_aux_loss_none_does_not_raise_type_error():
    mesh = _single_device_mesh()
    model = _SimpleParamLM(vocab_size=6, mesh=mesh, aux_loss=None)
    state = EasyDeLState.create(model=model, tx=optax.sgd(1e-3), init_opt_state=True)

    pref_batch = {
        "prompt_input_ids": jnp.array([[1, 2], [2, 3]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4], [4, 5]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 1], [1, 2]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "reference_input_ids": jnp.array([[3, 5], [4, 1]], dtype=jnp.int32),
        "reference_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
    }

    dpo_out = dpo_fn.concatenated_forward(
        model, pref_batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0, aux_loss_enabled=True
    )
    assert "aux_loss" not in dpo_out
    _, dpo_metrics = dpo_fn.training_step(
        state,
        pref_batch,
        state,
        learning_rate_fn=lambda _: 1e-3,
        concatenated_forward=lambda m, b, **kw: {**dpo_fn.concatenated_forward(m, b, False, -100, 0, aux_loss_enabled=True), "aux_loss": None},
        reference_free=True,
        partition_spec=PartitionSpec(),
    )
    assert np.isfinite(float(dpo_metrics.loss))

    tpo_out = tpo_fn.tpo_concatenated_forward(
        model, pref_batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0, aux_loss_enabled=True
    )
    assert "aux_loss" not in tpo_out
    _, tpo_metrics = tpo_fn.tpo_training_step(
        state,
        pref_batch,
        learning_rate_fn=lambda _: 1e-3,
        concatenated_forward=lambda m, b: {**tpo_fn.tpo_concatenated_forward(m, b, False, -100, 0, aux_loss_enabled=True), "aux_loss": None},
        partition_spec=PartitionSpec(),
    )
    assert np.isfinite(float(tpo_metrics.loss))

    unpaired_batch = {
        "prompt_input_ids": jnp.array([[1, 2], [2, 3]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_input_ids": jnp.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=jnp.int32),
        "completion_attention_mask": jnp.ones((2, 4), dtype=jnp.int32),
        "completion_labels": jnp.array([[-100, -100, 3, 4], [-100, -100, 4, 5]], dtype=jnp.int32),
        "label": jnp.array([True, False]),
    }
    bco_out = bco_fn.concatenated_forward(
        model, unpaired_batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0, aux_loss_enabled=True
    )
    assert "aux_loss" not in bco_out

    def kto_fwd_with_none_aux(m, b):
        return {**bco_fn.concatenated_forward(m, b, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0, aux_loss_enabled=True), "aux_loss": None}

    _, kto_metrics = kto_fn.training_step(
        state,
        unpaired_batch,
        state,
        learning_rate_fn=lambda _: 1e-3,
        forward_fn=kto_fwd_with_none_aux,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=True,
        aux_loss_coef=0.1,
        partition_spec=PartitionSpec(),
    )
    assert np.isfinite(float(kto_metrics.loss))
    kto_eval = kto_fn.evaluation_step(
        state,
        unpaired_batch,
        state,
        forward_fn=kto_fwd_with_none_aux,
        beta=0.1,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=True,
        aux_loss_coef=0.1,
        partition_spec=PartitionSpec(),
    )
    assert np.isfinite(float(kto_eval.loss))


# ---------------------------------------------------------------------------
# PKD-H7: ORPO concatenated_inputs unequal lengths, np.ndarray, VLM keys, max_length
# ---------------------------------------------------------------------------
def test_pkd_h7_orpo_unequal_lengths_numpy_vlm_and_max_length():
    lm = _CausalLMWithBackbone(vocab_size=8)
    batch = {
        "chosen_input_ids": np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32),
        "chosen_attention_mask": np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32),
        "chosen_labels": np.array([[-100, -100, 3, 4, 5, 6]], dtype=np.int32),
        "rejected_input_ids": np.array([[1, 2, 5]], dtype=np.int32),
        "rejected_attention_mask": np.array([[1, 1, 1]], dtype=np.int32),
        "rejected_labels": np.array([[-100, -100, 5]], dtype=np.int32),
        "pixel_values": np.ones((1, 2, 2), dtype=np.float32),
        "pixel_attention_mask": np.ones((1, 2), dtype=np.int32),
        "image_sizes": np.array([[2, 2]], dtype=np.int32),
    }
    concat = orpo_fn.concatenated_inputs(
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=7,
        max_length=5,
    )
    assert concat["concatenated_input_ids"].shape == (2, 5)
    assert concat["concatenated_attention_mask"].shape == (2, 5)
    assert concat["concatenated_labels"].shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(concat["concatenated_input_ids"][1]), [1, 2, 5, 7, 7])
    np.testing.assert_array_equal(np.asarray(concat["concatenated_attention_mask"][1]), [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(concat["concatenated_labels"][1]), [-100, -100, 5, -100, -100])
    assert concat["pixel_values"].shape == (2, 2, 2)
    assert concat["pixel_attention_mask"].shape == (2, 2)
    assert concat["image_sizes"].shape == (2, 2)

    orpo_fn.concatenated_forward(
        lm,
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=7,
        max_length=5,
    )
    assert "pixel_values" in lm.received_kwargs[0]
    assert lm.received_kwargs[0]["input_ids"].shape == (2, 5)


# ---------------------------------------------------------------------------
# PKD-H10: BCO UDM pop embedding_* + _vectorize_prompt; DPO microbatched reference_forward
# ---------------------------------------------------------------------------
def test_pkd_h10_bco_udm_vectorize_and_pop_embedding_keys():
    seen = {}

    def emb_fn(input_ids, attention_mask):
        seen["input_ids"] = input_ids.copy()
        seen["dtype"] = input_ids.dtype
        return np.ones((*input_ids.shape, 3), dtype=np.float32)

    # VLM-style processor without top-level pad_token_id + embedding tokenizer with pad_token_id=None
    vlm_processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=2))
    emb_tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=9)

    trainer = object.__new__(BCOTrainer)
    trainer.processing_class = vlm_processor
    trainer.embedding_tokenizer = emb_tokenizer
    trainer.embedding_func = emb_fn
    trainer.running = bco_fn.RunningMoments()
    trainer.clf_weights = (np.zeros(3, dtype=np.float32), 0.0)
    trainer.arguments = SimpleNamespace(min_density_ratio=0.5, max_density_ratio=10.0)
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b

    # Token ID 2 appears as a REAL token (attention_mask=1) and must NOT be overwritten;
    # padded slot (attention_mask=0) must be replaced by emb_tokenizer's fallback id (9) with int dtype.
    emb = trainer._vectorize_prompt(
        np.array([[2, 5, 2]], dtype=np.int32),
        np.array([[1, 1, 0]], dtype=np.int32),
    )
    assert emb.shape == (1, 3)
    assert seen["dtype"] == np.int32
    np.testing.assert_array_equal(seen["input_ids"], [[2, 5, 9]])

    raw_batch = {
        "prompt_input_ids": np.array([[1, 2]], dtype=np.int32),
        "label": np.array([False]),
        "embedding_input_ids": np.array([[2, 5, 2]], dtype=np.int32),
        "embedding_attention_mask": np.array([[1, 1, 0]], dtype=np.int32),
    }
    processed, info = trainer._preprocess_batch_input(None, raw_batch, is_train=True)
    assert "embedding_input_ids" not in processed
    assert "embedding_attention_mask" not in processed
    assert "udm_weights" in processed
    assert "udm_ratio_mean" in info
    # Original caller dict is untouched
    assert "embedding_input_ids" in raw_batch


def test_pkd_h10_dpo_microbatches_reference_forward_under_ga():
    mesh = _single_device_mesh()
    policy_model = _SimpleParamLM(vocab_size=6, mesh=mesh)
    ref_model = _SimpleParamLM(vocab_size=6, mesh=mesh)
    policy_state = EasyDeLState.create(model=policy_model, tx=optax.sgd(1e-2), init_opt_state=True)
    ref_state = EasyDeLState.create(model=ref_model)

    batch = {
        "prompt_input_ids": jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4], [4, 5], [5, 1], [1, 2]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 1], [1, 2], [2, 3], [3, 4]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((4, 2), dtype=jnp.int32),
    }

    seen_batch_sizes = []

    def tracking_forward(model, call_batch, **kwargs):
        seen_batch_sizes.append(call_batch["prompt_input_ids"].shape[0])
        return dpo_fn.concatenated_forward(
            model,
            call_batch,
            is_encoder_decoder=False,
            label_pad_token_id=-100,
            padding_value=0,
            ld_alpha=kwargs.get("ld_alpha"),
        )

    state_ga1, metrics_ga1 = dpo_fn.training_step(
        policy_state,
        batch,
        ref_state,
        learning_rate_fn=lambda _: 1e-2,
        concatenated_forward=tracking_forward,
        gradient_accumulation_steps=1,
        partition_spec=PartitionSpec(),
    )
    seen_batch_sizes.clear()
    state_ga2, metrics_ga2 = dpo_fn.training_step(
        policy_state,
        batch,
        ref_state,
        learning_rate_fn=lambda _: 1e-2,
        concatenated_forward=tracking_forward,
        gradient_accumulation_steps=2,
        partition_spec=PartitionSpec(),
    )
    # Every forward call under GA=2 (including reference_forward!) must see minibatch_size=2, never 4.
    assert seen_batch_sizes and all(sz == 2 for sz in seen_batch_sizes)
    np.testing.assert_allclose(float(metrics_ga2.loss), float(metrics_ga1.loss), rtol=1e-5, atol=1e-5)
    for l1, l2 in zip(
        jax.tree_util.tree_leaves(state_ga1.graphstate),
        jax.tree_util.tree_leaves(state_ga2.graphstate),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(l2), np.asarray(l1), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# PKD-H16: KTO enc-dec strip / precompute / eval transform; TPO VLM & no ref clone; DPO projection module
# ---------------------------------------------------------------------------
def test_pkd_h16_kto_encdec_strip_eval_transform_and_no_dead_kl_precompute():
    tok = _TinyCharTokenizer()
    encdec_tf = _EncoderDecoderKTOPreprocessTransform(tokenizer=tok, max_prompt_length=8, max_completion_length=8)
    row = encdec_tf({"prompt": "ab", "completion": "cd", "label": True})
    # Leading prompt prefix must be stripped from completion_* on encoder-decoder
    assert len(row["prompt_input_ids"]) == 2
    assert row["completion_labels"] == row["completion_input_ids"]
    assert -100 not in row["completion_labels"]

    # Pretokenized train + raw eval must NOT call eval_source.transform(None)
    trainer = object.__new__(KTOTrainer)
    trainer.processing_class = tok
    trainer.arguments = KTOConfig(is_encoder_decoder=False)
    pretok_ds = Dataset.from_dict({"prompt_input_ids": [[1, 2]], "completion_input_ids": [[1, 2, 3]], "label": [True]})
    raw_ds = Dataset.from_dict({"prompt": ["ab"], "completion": ["cd"], "label": [True]})
    trainer._train_source = trainer._to_sharded_source(pretok_ds)
    assert trainer._is_pretokenized() is True
    eval_src = trainer._build_source_from_dataset(raw_ds)
    eval_rows = [r for s in eval_src.shard_names for r in eval_src.open_shard(s)]
    assert "prompt_input_ids" in eval_rows[0]

    # Pre-existing reference_logps must NOT crash with duplicate column or run dead KL pass
    ds_with_ref = Dataset.from_dict({"prompt_input_ids": [[1, 2]], "reference_logps": [-0.5], "label": [True]})
    trainer.dataset_train = ds_with_ref
    trainer._train_source = trainer._to_sharded_source(ds_with_ref)
    trainer.calculate_kl = True
    assert trainer._precompute_reference_log_probs_for_split(
        dataset_attr="dataset_train", source_attr="_train_source", batch_size=1, is_train=True, desc="test"
    )


def test_pkd_h16_tpo_vlm_keys_and_dpo_wrapper_lmhead_resolution():
    batch = {
        "prompt_input_ids": jnp.array([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 2]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "reference_input_ids": jnp.array([[4, 3]], dtype=jnp.int32),
        "reference_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "pixel_values": jnp.ones((1, 2, 2), dtype=jnp.float32),
        "pixel_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "image_sizes": jnp.array([[2, 2]], dtype=jnp.int32),
    }
    cat3 = tpo_fn._tpo_concatenated_inputs(batch, padding_value=0, include_reference=True)
    assert cat3["pixel_values"].shape == (3, 2, 2)
    assert cat3["pixel_attention_mask"].shape == (3, 2)
    assert cat3["image_sizes"].shape == (3, 2)
    cat2 = tpo_fn._tpo_concatenated_inputs(batch, padding_value=0, include_reference=False)
    assert cat2["pixel_values"].shape == (2, 2, 2)

    # DPO wrapper model whose compute_lm_logits lives on model.model
    class InnerProjection:
        config = SimpleNamespace(lmhead_chunksize=2)

        def compute_lm_logits(self, h):
            return h

    class WrapperModel:
        def __init__(self):
            self.model = InnerProjection()

        def __call__(self, input_ids, attention_mask, apply_lm_head=True, **kwargs):
            del attention_mask, kwargs
            h = jnp.broadcast_to(jnp.arange(6, dtype=jnp.float32) * 0.1, (*input_ids.shape, 6))
            return SimpleNamespace(logits=h if apply_lm_head else None, last_hidden_state=h)

    wrapper = WrapperModel()
    out = dpo_fn.concatenated_forward(
        wrapper, batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0
    )
    assert np.isfinite(np.asarray(out["chosen_logps"])).all()


# ---------------------------------------------------------------------------
# PKD-M1: TPO STE + ScheduledLossAdapter; KTO vocab_shard_stage
# ---------------------------------------------------------------------------
def test_pkd_m1_tpo_ste_and_scheduled_loss_adapter_and_kto_vocab_shard_stage(monkeypatch):
    mesh = _single_device_mesh()
    model = _SimpleParamLM(vocab_size=6, mesh=mesh)
    state = EasyDeLState.create(model=model, tx=optax.sgd(1e-2), init_opt_state=True)
    ste_called = {"count": 0}

    def ste(tree):
        ste_called["count"] += 1
        return jax.tree_util.tree_map(lambda x: x + jax.lax.stop_gradient(0.1 - x), tree)

    batch = {
        "prompt_input_ids": jnp.array([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 1]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
    }

    tpo_fn.tpo_training_step(
        state,
        batch,
        learning_rate_fn=lambda _: 1e-2,
        concatenated_forward=lambda m, b, **kw: tpo_fn.tpo_concatenated_forward(m, b, False, -100, 0, **kw),
        partition_spec=PartitionSpec(),
        straight_through_emulator=ste,
    )
    assert ste_called["count"] == 1

    adapter = get_scheduled_loss_adapter(tpo_fn.tpo_training_step)
    assert adapter is not None and adapter.name == "tpo"

    # Verify KTO scheduled loss forwards vocab_shard_stage
    seen_stage = {}

    def fake_kto_fwd(module, call_batch, vocab_shard_stage=None):
        del module, call_batch
        seen_stage["stage"] = vocab_shard_stage
        return {"completion_logps": jnp.array([-1.0, -2.0], dtype=jnp.float32)}

    monkeypatch.setattr(kto_fn, "bind_scheduled_module", lambda call, tree: tree)
    monkeypatch.setattr(kto_fn, "constrain_scheduled_batch", lambda module, b, spec: b)
    monkeypatch.setattr(kto_fn, "_scheduled_terminal_stage_rank", lambda module, sched: 3)
    fake_call = SimpleNamespace(
        schedule=None,
        get={
            "forward_fn": fake_kto_fwd,
            "beta": 0.1,
            "desirable_weight": 1.0,
            "undesirable_weight": 1.0,
            "loss_type": "kto",
            "calculate_kl": False,
            "aux_loss_coef": 0.0,
            "partition_spec": None,
        }.get,
    )
    sched_loss = kto_fn._make_kto_scheduled_loss(fake_call)
    val = sched_loss(
        {},
        {"label": jnp.array([True, False]), "reference_logps": jnp.array([-1.0, -2.0], dtype=jnp.float32)},
    )
    assert seen_stage["stage"] == 3
    assert np.isfinite(float(val))


# ---------------------------------------------------------------------------
# PKD-M2 & PKD-M4: Config validation, auto-detection, no ref clone, FLOPs
# ---------------------------------------------------------------------------
def test_pkd_m2_and_m4_configs_autodetect_no_ref_clone_and_flops(monkeypatch):
    # DPOConfig edge cases
    cfg = DPOConfig(max_prompt_length=None, max_length=512)
    assert cfg.max_completion_length is None
    with pytest.raises(ValueError, match="at least one"):
        DPOConfig(loss_type=())
    with pytest.raises(ValueError, match="beta"):
        DPOConfig(beta=0.0)
    with pytest.raises(ValueError, match=r"Label smoothing must be greater than 0\.0"):
        DPOConfig(loss_type="exo_pair", label_smoothing=-0.1)
    with pytest.raises(ValueError, match=r"Label smoothing must be greater than 0\.0"):
        DPOConfig(loss_type="exo_pair", label_smoothing=1.0)
    with pytest.raises(ValueError, match="beta"):
        CPOConfig(beta=-0.1)
    with pytest.raises(ValueError, match="beta"):
        TPOConfig(beta=0.0)
    assert isinstance(hash(TPOConfig()), int)

    # Auto-detect is_encoder_decoder + skip reference_state clone when reference_free=True
    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)

    class _EncDecLM(_SimpleParamLM):
        def __init__(self):
            super().__init__(vocab_size=6, mesh=_single_device_mesh())
            self.config = SimpleNamespace(is_encoder_decoder=True)

    real_encdec_state = EasyDeLState.create(model=_EncDecLM())
    monkeypatch.setattr(
        "easydel.trainers.direct_preference_optimization_trainer.dpo_trainer.copy_frozen_policy",
        lambda _: (_ for _ in ()).throw(AssertionError("should not clone reference_state when reference_free")),
    )
    tok = _TinyCharTokenizer()

    dpo_tr = DPOTrainer(
        arguments=DPOConfig(reference_free=True, force_use_ref_model=False, disable_dropout=False),
        model=real_encdec_state,
        processing_class=tok,
    )
    assert dpo_tr.is_encoder_decoder is True
    assert dpo_tr.arguments.is_encoder_decoder is True
    assert dpo_tr.reference_state is None

    tpo_tr = TPOTrainer(
        arguments=TPOConfig(disable_dropout=False),
        model=real_encdec_state,
        processing_class=tok,
    )
    assert tpo_tr.reference_state is None

    orpo_tr = ORPOTrainer(
        arguments=ORPOConfig(disable_dropout=False),
        model=real_encdec_state,
        processing_class=tok,
    )
    assert orpo_tr.is_encoder_decoder is True
    assert orpo_tr.arguments.is_encoder_decoder is True

    # KTOTrainer max_prompt_length=None
    kto_tr = KTOTrainer(
        arguments=KTOConfig(max_prompt_length=None, max_length=128, disable_dropout=False),
        model=real_encdec_state,
        reference_model=real_encdec_state,
        processing_class=tok,
        train_dataset=Dataset.from_dict({"prompt_input_ids": [[1, 2]], "label": [True]}),
    )
    assert kto_tr.arguments.max_prompt_length is None

    # PKD-M4: DPO frozen ref FLOPs uses include_backward=False and _extra_backward_flops_per_token=0.0
    from easydel.trainers.direct_preference_optimization_trainer import dpo_trainer as dpo_mod

    flop_calls = []

    class _FlopModel:
        mesh = None

        def flops_per_token(self, *, include_loss, include_backward):
            flop_calls.append((include_loss, include_backward))
            return 123.0

    monkeypatch.setattr(dpo_mod, "replicated_named_sharding", lambda mesh: None)
    monkeypatch.setattr(dpo_mod, "resolve_straight_through_emulator", lambda **kwargs: None)
    monkeypatch.setattr(dpo_mod, "compile_trainer_auxiliary", lambda fn, **kwargs: fn)
    monkeypatch.setattr(dpo_mod, "compile_trainer_step", lambda fn, **kwargs: SimpleNamespace())
    monkeypatch.setattr(DPOConfig, "get_streaming_checkpointer", lambda self: None)
    monkeypatch.setattr(DPOConfig, "ensure_checkpoint_path", lambda self: None)

    flop_tr = object.__new__(DPOTrainer)
    flop_tr._model = _FlopModel()
    flop_tr.padding_value = 0
    flop_tr.scheduler = None
    flop_tr._runtime_trace = lambda *args, **kwargs: None
    flop_tr.state_shardings = _DummyState("train")
    flop_tr.model_state = flop_tr.state_shardings
    flop_tr._eval_graphdef = None
    flop_tr.reference_state = _DummyState("ref", model=_FlopModel())
    flop_tr.arguments = DPOConfig(reference_free=False)
    flop_tr.configure_functions()
    assert flop_calls == [(True, False)]
    assert flop_tr._extra_forward_flops_per_token == 123.0
    assert flop_tr._extra_backward_flops_per_token == 0.0


# ---------------------------------------------------------------------------
# PKD-M3, PKD-L1, PKD-L2: WPO, mean_logits/V, f-div shifts, CPO, BCO, KTO GA KL, TPO Grain/EOS/headless
# ---------------------------------------------------------------------------
def test_pkd_m3_dpo_wpo_f_div_and_mean_logits():
    # WPO log_denom matches float64 reference in bfloat16 and with vocab chunking
    logits = jnp.array([[[2.0, -1.0, 4.5, 0.5, -3.0, 1.2]]], dtype=jnp.bfloat16)
    f64 = np.asarray(logits, dtype=np.float64)
    p = np.exp(f64 - np.max(f64, axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    expected_log_denom = np.log(np.sum(p * p, axis=-1))
    np.testing.assert_allclose(np.asarray(dpo_fn._compute_wpo_log_denom(logits, None)), expected_log_denom, atol=1e-4)
    np.testing.assert_allclose(np.asarray(dpo_fn._compute_wpo_log_denom(logits, 2)), expected_log_denom, atol=1e-4)

    # All f-divergences are centered at 0 when logratio == 0
    zeros = jnp.zeros((2,), dtype=jnp.float32)
    for f_div in ("reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"):
        c_s, r_s = dpo_fn._compute_f_divergence_scores(zeros, zeros, f_divergence_type=f_div, f_alpha_divergence_coef=0.5)
        np.testing.assert_allclose(np.asarray(c_s), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(r_s), 0.0, atol=1e-6)

    # apo_zero, apo_down, aot_pair respect f_divergence_type != reverse_kl
    c = jnp.array([-0.5, -1.5], dtype=jnp.float32)
    r = jnp.array([-1.2, -0.8], dtype=jnp.float32)
    rc = jnp.array([-0.9, -1.1], dtype=jnp.float32)
    rr = jnp.array([-1.0, -1.3], dtype=jnp.float32)
    for lt in ("aot_pair", "apo_zero", "apo_down"):
        l_rev = dpo_fn.compute_dpo_losses(c, r, rc, rr, beta=0.2, label_smoothing=0.0, loss_type=lt, f_divergence_type="reverse_kl")
        l_fwd = dpo_fn.compute_dpo_losses(c, r, rc, rr, beta=0.2, label_smoothing=0.0, loss_type=lt, f_divergence_type="forward_kl")
        assert not np.allclose(np.asarray(l_rev), np.asarray(l_fwd))

    # mean_chosen_logits divides by vocab_size V (not raw sum across V)
    class ConstLogits:
        def __call__(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            return SimpleNamespace(logits=jnp.full((*input_ids.shape, 8), 3.0, dtype=jnp.float32))

    batch = {
        "prompt_input_ids": jnp.array([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[5, 2]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
    }
    out = dpo_fn.concatenated_forward(ConstLogits(), batch, False, -100, 0, return_mean_logits=True)
    assert float(out["mean_chosen_logits"]) == pytest.approx(3.0, abs=1e-5)
    assert float(out["mean_rejected_logits"]) == pytest.approx(3.0, abs=1e-5)

    # PKD-L2: get_loss_function rejects dead "kto" and forwards f_divergence_type & loss_weights
    with pytest.raises(ValueError, match="not valid"):
        dpo_fn.get_loss_function("kto", beta=0.1, label_smoothing=0.0)
    fn_fkl = dpo_fn.get_loss_function(("sigmoid", "hinge"), beta=0.2, label_smoothing=0.0)
    res_fn = fn_fkl(c, r, rc, rr, 0.2, 0.0, loss_weights=(0.3, 0.7), f_divergence_type="forward_kl")
    res_direct = dpo_fn.compute_dpo_losses(
        c, r, rc, rr, beta=0.2, label_smoothing=0.0, loss_type=("sigmoid", "hinge"), loss_weights=(0.3, 0.7), f_divergence_type="forward_kl"
    )
    np.testing.assert_allclose(np.asarray(res_fn), np.asarray(res_direct), atol=1e-6)


def test_pkd_m3_cpo_unclamped_zero_token_lengths_and_eval_partition_spec():
    class UniformModel:
        mesh = _single_device_mesh()

        def __call__(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            return SimpleNamespace(logits=jnp.zeros((*input_ids.shape, 4), dtype=jnp.float32))

    # Row 0 has 2 valid chosen tokens; Row 1 has 0 valid chosen tokens (all masked).
    batch = {
        "prompt_input_ids": jnp.array([[1, 2], [1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 2], [0, 0]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.array([[1, 1], [0, 0]], dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[2, 1], [2, 1]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
    }
    out = cpo_fn.concatenated_forward(UniformModel(), batch, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0)
    np.testing.assert_array_equal(np.asarray(out["chosen_lengths"]), [2.0, 0.0])
    # _policy_nll_loss must divide by 2.0 (true tokens), NOT 3.0!
    nll = cpo_fn._policy_nll_loss(out["chosen_logps_raw"], out["chosen_lengths"])
    assert float(nll) == pytest.approx(math.log(4.0), rel=1e-5)

    # evaluation_step runs cleanly with partition_spec applied
    state = EasyDeLState.create(model=_SimpleParamLM(vocab_size=4, mesh=_single_device_mesh()))
    eval_metrics = cpo_fn.evaluation_step(
        state,
        batch,
        concatenated_forward_fn=lambda m, b: cpo_fn.concatenated_forward(m, b, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0),
        beta=0.1,
        label_smoothing=0.0,
        loss_type="sigmoid",
        cpo_alpha=1.0,
        simpo_gamma=0.5,
        alpha=0.0,
        partition_spec=PartitionSpec(),
    )
    assert np.isfinite(float(eval_metrics.loss))


def test_pkd_m3_and_l2_bco_moments_guard_reward_dilution_and_eval_scalars(monkeypatch):
    rm = bco_fn.RunningMoments()
    rm.update_from_moments(2.0, 1.0, 4.0)
    snap = rm.as_dict()
    rm.update_from_moments(float("nan"), 1.0, 4.0)
    rm.update_from_moments(1.0, float("inf"), 4.0)
    rm.update_from_moments(1.0, 1.0, float("inf"))
    assert rm.as_dict() == snap

    # Verify BCOTrainer.on_step_end batches all 3 reward moments into ONE jax.device_get call
    device_get_calls = {"count": 0}
    orig_device_get = jax.device_get

    def spy_device_get(x):
        device_get_calls["count"] += 1
        return orig_device_get(x)

    monkeypatch.setattr(jax, "device_get", spy_device_get)
    bco_tr = object.__new__(BCOTrainer)
    bco_tr.running = bco_fn.RunningMoments()
    bco_tr.arguments = SimpleNamespace(gradient_accumulation_steps=1)
    step_metrics = SimpleNamespace(
        other_metrics={
            "reward_moments/sum": jnp.array(8.0, dtype=jnp.float32),
            "reward_moments/sumsq": jnp.array(20.0, dtype=jnp.float32),
            "reward_moments/count": jnp.array(4.0, dtype=jnp.float32),
        }
    )
    bco_tr.on_step_end(None, step_metrics, None)
    assert device_get_calls["count"] == 1
    assert bco_tr.running.mean == pytest.approx(2.0, abs=1e-5)
    assert bco_tr.running.var == pytest.approx(1.0, abs=1e-5)
    monkeypatch.setattr(jax, "device_get", orig_device_get)

    mesh = _single_device_mesh()
    state = EasyDeLState.create(model=_SimpleParamLM(vocab_size=6, mesh=mesh))
    batch = {
        "prompt_input_ids": jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "completion_input_ids": jnp.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 1]], dtype=jnp.int32),
        "completion_attention_mask": jnp.ones((4, 3), dtype=jnp.int32),
        "completion_labels": jnp.array([[-100, -100, 3], [-100, -100, 4], [-100, -100, 5], [-100, -100, 1]], dtype=jnp.int32),
        "label": jnp.array([True, False, False, False]),  # 1 chosen, 3 rejected
        "reference_logps": jnp.array([-10.0, -10.0, -10.0, -10.0], dtype=jnp.float32),
    }

    def fixed_fwd(m, b):
        del m, b
        return {"completion_logps": jnp.array([-6.0, -8.0, -8.0, -8.0], dtype=jnp.float32), "mean_completion_logits": jnp.array(0.0)}

    eval_metrics = bco_fn.evaluation_step(state, batch, state, fixed_fwd, beta=0.5)
    # True chosen reward = 0.5 * (-6 - (-10)) = +2.0 (NOT diluted by 1/4 -> 0.5!)
    # True rejected reward = 0.5 * (-8 - (-10)) = +1.0 (NOT diluted by 3/4 -> 0.75!)
    assert float(eval_metrics.chosen_rewards) == pytest.approx(2.0, abs=1e-5)
    assert float(eval_metrics.rejected_rewards) == pytest.approx(1.0, abs=1e-5)
    # PKD-L2: all eval other_metrics must be 0-D scalars (no 1D mask/chosen, mask/rejected)
    assert "mask/chosen" not in eval_metrics.other_metrics
    assert "mask/rejected" not in eval_metrics.other_metrics
    for k, v in eval_metrics.other_metrics.items():
        assert np.asarray(v).ndim == 0, f"BCO eval metric {k} is not 0-D"


def test_pkd_m3_kto_reward_dilution_and_no_ga_kl_jensen_bias():
    mesh = _single_device_mesh()
    state = EasyDeLState.create(model=_SimpleParamLM(vocab_size=6, mesh=mesh), tx=optax.sgd(1e-2), init_opt_state=True)
    ref_state = EasyDeLState.create(model=_SimpleParamLM(vocab_size=6, mesh=mesh))
    # Microbatch 0 has policy_kl - ref_kl = +2.0, Microbatch 1 has -4.0.
    # Full-batch mean is -1.0 -> max(-1.0, 0.0) == 0.0!
    # Per-microbatch mean-of-max would be (2.0 + 0.0) / 2 = +1.0 (Jensen bias!).
    batch = {
        "prompt_input_ids": jnp.array([[1], [2], [3], [4]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((4, 1), dtype=jnp.int32),
        "completion_input_ids": jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=jnp.int32),
        "completion_attention_mask": jnp.ones((4, 2), dtype=jnp.int32),
        "completion_labels": jnp.array([[-100, 2], [-100, 3], [-100, 4], [-100, 5]], dtype=jnp.int32),
        "label": jnp.array([True, False, True, False]),
        "reference_logps": jnp.array([-5.0, -5.0, -5.0, -5.0], dtype=jnp.float32),
        "_policy_kl_logps": jnp.array([2.0, 2.0, -4.0, -4.0], dtype=jnp.float32),
        "_reference_kl_logps": jnp.zeros((4,), dtype=jnp.float32),
    }

    def fwd(m, b):
        return bco_fn.concatenated_forward(m, b, is_encoder_decoder=False, label_pad_token_id=-100, padding_value=0)

    _, m_ga1 = kto_fn.training_step(
        state,
        batch,
        ref_state,
        learning_rate_fn=lambda _: 1e-2,
        forward_fn=fwd,
        beta=0.5,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=True,
        aux_loss_coef=0.0,
        gradient_accumulation_steps=1,
        partition_spec=PartitionSpec(),
    )
    _, m_ga2 = kto_fn.training_step(
        state,
        batch,
        ref_state,
        learning_rate_fn=lambda _: 1e-2,
        forward_fn=fwd,
        beta=0.5,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=True,
        aux_loss_coef=0.0,
        gradient_accumulation_steps=2,
        partition_spec=PartitionSpec(),
    )
    assert float(m_ga1.other_metrics["kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(m_ga2.other_metrics["kl"]) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(float(m_ga2.loss), float(m_ga1.loss), rtol=1e-5, atol=1e-5)

    # Verify KTO eval chosen_rewards is undiluted class mean
    eval_m = kto_fn.evaluation_step(
        state,
        {**batch, "label": jnp.array([True, False, False, False])},
        ref_state,
        forward_fn=lambda m, b: {"completion_logps": jnp.array([-1.0, -3.0, -3.0, -3.0], dtype=jnp.float32)},
        beta=0.5,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        loss_type="kto",
        calculate_kl=False,
        aux_loss_coef=0.0,
        partition_spec=PartitionSpec(),
    )
    # 0.5 * (-1 - (-5)) = +2.0 (NOT diluted by 1/4)
    assert float(eval_m.chosen_rewards) == pytest.approx(2.0, abs=1e-5)
    assert float(eval_m.rejected_rewards) == pytest.approx(1.0, abs=1e-5)


def test_pkd_m3_tpo_list_eos_numpy_grain_collator_and_build_preprocess_transform():
    class ListEosTokenizer(_TinyCharTokenizer):
        def __init__(self):
            self.eos_token_id = [2, 99]

    tf = TPOPreprocessTransform(tokenizer=ListEosTokenizer(), max_prompt_length=8, max_completion_length=8)
    row = tf({"prompt": "hi", "chosen": "ok", "rejected": "no", "reference": "yes"})
    assert all(isinstance(tok_id, int) for tok_id in row["chosen_input_ids"])
    assert row["chosen_input_ids"][-1] == 2

    collator = DataCollatorForTriplePreferenceGrain(max_prompt_length=8, max_completion_length=8, pad_token_id=0)
    collated = collator(row)
    assert isinstance(collated["reference_input_ids"], np.ndarray)
    assert isinstance(collated["reference_attention_mask"], np.ndarray)
    assert not isinstance(collated["reference_input_ids"], jax.Array)

    trainer = object.__new__(TPOTrainer)
    trainer.processing_class = ListEosTokenizer()
    trainer.arguments = TPOConfig(max_prompt_length=8, max_completion_length=8)
    assert isinstance(trainer._build_preprocess_transform(), TPOPreprocessTransform)


def test_pkd_m3_tpo_headless_lmhead_chunksize_matches_dense():
    class ChunkedModel:
        def __init__(self, chunk_size):
            self.config = SimpleNamespace(lmhead_chunksize=chunk_size)

        def __call__(self, input_ids, attention_mask, apply_lm_head=True, **kwargs):
            del attention_mask, kwargs
            h = jnp.broadcast_to(jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, -0.5]), (*input_ids.shape, 6))
            return SimpleNamespace(logits=h if apply_lm_head else None, last_hidden_state=h)

        def compute_lm_logits(self, hidden):
            return hidden

    batch = {
        "prompt_input_ids": jnp.array([[1, 2]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((1, 2), dtype=jnp.int32),
        "chosen_input_ids": jnp.array([[3, 4, 5]], dtype=jnp.int32),
        "chosen_attention_mask": jnp.ones((1, 3), dtype=jnp.int32),
        "rejected_input_ids": jnp.array([[2, 1, 0]], dtype=jnp.int32),
        "rejected_attention_mask": jnp.array([[1, 1, 0]], dtype=jnp.int32),
        "reference_input_ids": jnp.array([[4, 5, 1]], dtype=jnp.int32),
        "reference_attention_mask": jnp.ones((1, 3), dtype=jnp.int32),
    }
    dense = tpo_fn.tpo_concatenated_forward(ChunkedModel(None), batch, False, -100, 0)
    headless = tpo_fn.tpo_concatenated_forward(ChunkedModel(2), batch, False, -100, 0, logprob_vocab_chunk_size=2)
    for key in ("chosen_logps", "rejected_logps", "chosen_lengths", "rejected_lengths", "reference_nll"):
        np.testing.assert_allclose(np.asarray(headless[key]), np.asarray(dense[key]), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# PKD-L1 & PKD-L2: _logprob_utils getattr + chunked masking; ORPO eval metrics; DPOConfig rpo_alpha docs
# ---------------------------------------------------------------------------
def test_pkd_l1_logprob_utils_getattr_and_chunked_masked_inf():
    # Model whose config has no lmhead_chunksize attribute must return None (not AttributeError)
    no_attr_model = SimpleNamespace(config=SimpleNamespace(), compute_lm_logits=lambda h: h)
    assert _logprob_utils.resolve_lmhead_chunksize(no_attr_model) is None

    # Masked-out positions with -inf logits must NOT poison logit_sums via (-inf * 0 = NaN)
    class MaskedInfModel:
        def compute_lm_logits(self, h):
            return h

    hidden = jnp.array([[[1.0, 2.0, 3.0, 4.0], [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf]]], dtype=jnp.float32)
    labels = jnp.array([[2, 0]], dtype=jnp.int32)
    loss_mask = jnp.array([[True, False]])
    logp_sums, logit_sums, counts = _logprob_utils.compute_sequence_scores_from_hidden_states(
        MaskedInfModel(),
        hidden,
        labels,
        loss_mask,
        token_chunk_size=1,
        vocab_chunk_size=2,
    )
    assert np.isfinite(np.asarray(logp_sums)).all()
    assert float(logit_sums[0]) == pytest.approx(10.0, abs=1e-5)
    assert float(counts[0]) == pytest.approx(1.0)


def test_pkd_l2_orpo_eval_does_not_double_prefix_and_dpo_rpo_docs():
    assert "does not currently implement" not in (DPOConfig.__doc__ or "")
    mesh = _single_device_mesh()
    state = EasyDeLState.create(model=_SimpleParamLM(vocab_size=6, mesh=mesh))
    batch = {
        "chosen_input_ids": jnp.array([[1, 2, 3, 4]]),
        "chosen_attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
        "chosen_labels": jnp.array([[-100, -100, 3, 4]]),
        "rejected_input_ids": jnp.array([[1, 2, 5, 4]]),
        "rejected_attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
        "rejected_labels": jnp.array([[-100, -100, 5, 4]]),
    }
    metrics = orpo_fn.orpo_step(
        state,
        batch,
        concatenated_forward=lambda s, b: orpo_fn.concatenated_forward(s, b, False, -100, 0),
        mode="eval",
        partition_spec=PartitionSpec(),
    )
    assert "nll_loss" in metrics.other_metrics
    assert not any(k.startswith("eval_") for k in metrics.other_metrics)

# ===========================================================================
# Group B: Online Preference & On-Policy/RL-Adjacent KD (PKD-C2, C6, C7, C9, H1, H4, H9-H11, H13-H16, M1-M4, L2, L3)
# ===========================================================================

def _tiny_llama(seed: int = 0, *, vocab_size: int = 32, attn_mechanism: str = "vanilla") -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=32,
        attn_mechanism=attn_mechanism,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    return LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))


def _word_tokenizer(words: list[str], *, padding_side: str = "left") -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    for token in words:
        if token not in vocab:
            vocab[token] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        padding_side=padding_side,
    )


# ---------------------------------------------------------------------------
# PKD-C2, PKD-H11, PKD-H16, PKD-M2, PKD-M3, PKD-L2: OnlineDPO
# ---------------------------------------------------------------------------


def test_online_dpo_batch_does_not_double_prompts_in_dpo_concatenated_forward(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    cfg = OnlineDPOConfig(max_prompt_length=3, max_completion_length=2, max_length=5)
    trainer = OnlineDPOTrainer(cfg, model=None, reward_funcs=(lambda **kwargs: [1.0, 0.0],))
    trainer.arguments = cfg
    trainer._eos_token_id = [99]

    prompt_ids = jnp.array([[10, 11, 12]], dtype=jnp.int32)
    prompt_mask = jnp.array([[1, 1, 1]], dtype=jnp.int32)
    completion_ids = jnp.array([[20, 21], [22, 23]], dtype=jnp.int32)
    completion_mask = jnp.array([[1, 1], [1, 1]], dtype=jnp.int32)
    rewards = jnp.array([2.0, 0.5], dtype=jnp.float32)

    dpo_batch, _ = trainer._build_online_dpo_batch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rewards=rewards,
    )
    # Completion-only tensors emitted to match inherited DPO concatenated_forward.
    np.testing.assert_array_equal(np.asarray(dpo_batch["chosen_input_ids"]), [[20, 21]])
    np.testing.assert_array_equal(np.asarray(dpo_batch["rejected_input_ids"]), [[22, 23]])
    np.testing.assert_array_equal(np.asarray(dpo_batch["chosen_attention_mask"]), [[1, 1]])
    np.testing.assert_array_equal(np.asarray(dpo_batch["rejected_attention_mask"]), [[1, 1]])

    seen_input_ids: list[np.ndarray] = []

    class RecordingLM:
        config = SimpleNamespace(lmhead_chunksize=None)

        def __call__(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            seen_input_ids.append(np.asarray(jax.device_get(input_ids)))
            # Position-dependent logits so prompt vs completion scoring is distinguishable.
            vocab = 30
            logits = jnp.zeros((*input_ids.shape, vocab), dtype=jnp.float32)
            # Make position 2 predict token 20 strongly, position 3 predict token 21 strongly.
            logits = logits.at[:, 2, 20].set(3.0).at[:, 3, 21].set(4.0)
            return SimpleNamespace(logits=logits)

    output = dpo_concatenated_forward(
        RecordingLM(),
        dpo_batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        max_length=5,
        truncation_mode="keep_start",
    )
    assert len(seen_input_ids) == 1
    # Exact single prompt + completion, never [prompt, prompt, completion].
    np.testing.assert_array_equal(
        seen_input_ids[0],
        [[10, 11, 12, 20, 21], [10, 11, 12, 22, 23]],
    )
    assert float(output["chosen_logps"][0]) > float(output["rejected_logps"][0])


def test_online_dpo_eos_penalty_masks_pad_equals_eos_and_nan_never_beats_negative(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    cfg = OnlineDPOConfig(missing_eos_penalty=5.0)
    # Also verifies PKD-M2 tuple reward_funcs & PKD-L2 removal of dead _bound_reward_forward.
    assert not hasattr(OnlineDPOTrainer, "_bound_reward_forward")
    trainer = OnlineDPOTrainer(
        cfg,
        model=None,
        reward_funcs=(lambda **kwargs: [None, -1.0, 2.0, 2.0],),
    )
    trainer.arguments = cfg
    trainer._eos_token_id = [99]

    rewards = trainer._score_online_dpo_rewards(
        prompts=["p0", "p0", "p1", "p1"],
        completions=["a", "b", "c", "d"],
        completion_ids=jnp.array([[7, 99], [8, 99], [9, 99], [10, 99]], dtype=jnp.int32),
        batch={"prompt": ["p0", "p1"]},
    )
    assert np.isnan(float(rewards[0])), "All-NaN completion must stay NaN, not become 0.0."
    assert float(rewards[1]) == pytest.approx(-1.0)

    # Pair 0: candidate 0 is NaN, candidate 1 is -1.0 -> valid candidate 1 must win!
    # Pair 1: candidate 2 has EOS only in masked pad slot (mask=0) -> penalized; candidate 3 has real EOS (mask=1) -> wins!
    prompt_ids = jnp.array([[1], [2]], dtype=jnp.int32)
    prompt_mask = jnp.ones_like(prompt_ids)
    completion_ids = jnp.array([[7, 99], [8, 99], [9, 99], [10, 99]], dtype=jnp.int32)
    completion_mask = jnp.array([[1, 1], [1, 1], [1, 0], [1, 1]], dtype=jnp.int32)

    dpo_batch, metrics = trainer._build_online_dpo_batch(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        rewards=rewards,
    )
    np.testing.assert_array_equal(np.asarray(dpo_batch["chosen_input_ids"]), [[8, 99], [10, 99]])
    np.testing.assert_array_equal(np.asarray(dpo_batch["rejected_input_ids"]), [[7, 99], [9, 99]])
    # Pair 0 rejected had mask=[1,1] (EOS present); pair 1 rejected had mask=[1,0] (no real EOS) -> mean 0.5.
    assert metrics["online_dpo/chosen_contains_eos"] == pytest.approx(1.0)
    assert metrics["online_dpo/rejected_contains_eos"] == pytest.approx(0.5)


def test_online_dpo_respects_custom_collator_and_grain_single_dict(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    custom_collator = lambda rows: {"custom": rows}  # noqa: E731
    trainer_custom = OnlineDPOTrainer(
        OnlineDPOConfig(),
        model=None,
        reward_funcs=lambda **kwargs: [0.0],
        data_collator=custom_collator,
    )
    assert trainer_custom.create_grain_collect_function(16) is custom_collator
    assert trainer_custom.create_tfds_collect_function(16) is custom_collator

    trainer_default = OnlineDPOTrainer(
        OnlineDPOConfig(),
        model=None,
        reward_funcs=lambda **kwargs: [0.0],
    )
    grain_collate = trainer_default.create_grain_collect_function(16)
    single_row = {"prompt": "hello", "answer": "world"}
    assert grain_collate(single_row) == single_row
    batched_rows = [{"prompt": "a", "answer": "1"}, {"prompt": "b", "answer": "2"}]
    assert grain_collate(batched_rows) == {"prompt": ["a", "b"], "answer": ["1", "2"]}


# ---------------------------------------------------------------------------
# PKD-C6: XPO & Nash-MD multi-step schedules never read donated model_state.step
# ---------------------------------------------------------------------------


def test_xpo_and_nash_md_schedules_survive_donated_model_state_step():
    @jax.jit(donate_argnums=(0,))
    def _donate_step(step_arr):
        return step_arr + 1

    initial_step = jnp.asarray(0, dtype=jnp.int32)
    live_step_1 = _donate_step(initial_step)
    live_step_1.block_until_ready()
    # Confirm initial_step truly is donated/deleted so reading it would crash.
    with pytest.raises(RuntimeError):
        jax.device_get(initial_step)

    for cls, schedule_attr in [(XPOTrainer, "_alpha_schedule"), (NashMDTrainer, "_mixture_schedule")]:
        trainer = object.__new__(cls)
        trainer.arguments = SimpleNamespace(num_train_epochs=2)
        trainer.max_training_steps = 4
        trainer.model_state = SimpleNamespace(step=initial_step)  # Donated!
        trainer._beta_schedule = [0.1, 0.5]
        setattr(trainer, schedule_attr, [0.2, 0.8])
        trainer._current_step = 0

        assert trainer._current_beta_value() == pytest.approx(0.1)
        assert trainer._schedule_value(getattr(trainer, schedule_attr), 0.0) == pytest.approx(0.2)

        # Advance via live state at epoch 1 (step=2) without touching donated model_state.step.
        trainer._sync_current_step(SimpleNamespace(step=jnp.asarray(2, dtype=jnp.int32)))
        assert trainer._current_beta_value() == pytest.approx(0.5)
        assert trainer._schedule_value(getattr(trainer, schedule_attr), 0.0) == pytest.approx(0.8)

        # Also test on_step_end updating host-side step counter.
        trainer.arguments.sync_ref_model = False
        trainer.ref_state = None
        trainer.on_step_end(SimpleNamespace(step=jnp.asarray(3, dtype=jnp.int32)), {}, step=3)
        assert trainer._current_step == 3


# ---------------------------------------------------------------------------
# PKD-C7: Nash-MD causal reward-to-go KL gradient check for T >= 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheduled", [False, True])
def test_nash_md_multitoken_trajectory_kl_gradient_matches_exact_kl(scheduled):
    """Verify Nash-MD includes cross-token (s > t) trajectory KL gradient terms for T=2."""
    mesh = _single_device_mesh()
    # Autoregressive 2-token policy over binary vocab {0, 1}:
    # theta[0]: logits for y_0
    # theta[1]: logits for y_1 | y_0 = 0 (matches ref -> zero suffix KL)
    # theta[2]: logits for y_1 | y_0 = 1 (far from ref -> large suffix KL)
    theta0 = jnp.array([[0.3, -0.2], [0.5, -0.5], [2.0, -2.0]], dtype=jnp.float32)
    ref_theta = jnp.array([[0.1, 0.2], [0.5, -0.5], [-2.0, 2.0]], dtype=jnp.float32)
    beta = 0.4

    trajectories = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
    prompt_ids = jnp.zeros((4, 1), dtype=jnp.int32)
    prompt_mask = jnp.ones((4, 1), dtype=jnp.int32)
    completion_mask = jnp.ones((4, 2), dtype=jnp.int32)

    class TwoStepARModel:
        def __init__(self, theta):
            self.mesh = mesh
            self.theta = theta

        def __call__(self, input_ids, **kwargs):
            del kwargs
            # input_ids is [B, 3]: [prompt, y_0, y_1]
            # logits[:, 0] predicts y_0 using theta[0]
            # logits[:, 1] predicts y_1 using theta[1] if y_0==0 else theta[2]
            y0 = input_ids[:, 1]
            logits_0 = jnp.broadcast_to(self.theta[0], (input_ids.shape[0], 2))
            logits_1 = jnp.where(y0[:, None] == 0, self.theta[1][None, :], self.theta[2][None, :])
            logits_2 = jnp.zeros_like(logits_0)
            return SimpleNamespace(logits=jnp.stack([logits_0, logits_1, logits_2], axis=1))

    class State:
        graphdef = None
        graphother = None

        def __init__(self, theta):
            self.graphstate = theta
            self.model = TwoStepARModel(theta)

        def merge(self, tree):
            return TwoStepARModel(tree)

    def _traj_logps(theta):
        lp0 = jax.nn.log_softmax(theta[0])[trajectories[:, 0]]
        lp1_0 = jax.nn.log_softmax(theta[1])[trajectories[:, 1]]
        lp1_1 = jax.nn.log_softmax(theta[2])[trajectories[:, 1]]
        lp1 = jnp.where(trajectories[:, 0] == 0, lp1_0, lp1_1)
        return jnp.stack([lp0, lp1], axis=1)

    ref_token_logps = _traj_logps(ref_theta)

    # Exact closed-form beta * KL(pi(y_0, y_1) || pi_ref(y_0, y_1)).
    def exact_kl(theta):
        pol_lp = _traj_logps(theta)
        traj_lp = pol_lp.sum(axis=1)
        ref_traj_lp = ref_token_logps.sum(axis=1)
        probs = jnp.exp(traj_lp)
        return beta * jnp.sum(probs * (traj_lp - ref_traj_lp))

    expected_grad = jax.grad(exact_kl)(theta0)

    # Weighted average of per-trajectory Nash-MD gradients under p(y_0, y_1; theta0).
    probs_const = jax.lax.stop_gradient(jnp.exp(_traj_logps(theta0).sum(axis=1)))

    def single_traj_loss(theta, idx):
        batch = {
            "prompt_ids": prompt_ids[idx : idx + 1],
            "prompt_mask": prompt_mask[idx : idx + 1],
            "completion_ids": trajectories[idx : idx + 1],
            "completion_mask": completion_mask[idx : idx + 1],
            "ref_token_logps": ref_token_logps[idx : idx + 1],
            "probabilities": jnp.array([0.5], dtype=jnp.float32),
        }
        state = State(theta)
        args = (state, batch, beta, None, None, None, PartitionSpec(), 1, False)
        if not scheduled:
            return nash_md_step(*args).loss
        bound = inspect.signature(nash_md_step).bind(*args)
        bound.apply_defaults()
        call = ScheduledStepCall(nash_md_step, state, batch, args, {}, bound.arguments, None)
        return get_scheduled_loss_adapter(nash_md_step).make_loss(call)(theta, batch)

    actual_grad = sum(
        probs_const[i] * jax.grad(single_traj_loss)(theta0, i)
        for i in range(4)
    )
    np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(expected_grad), rtol=1e-4, atol=1e-5)

    # Confirm the cross-token s=1 -> t=0 gradient component is genuinely non-zero.
    t0_single_step_only = jax.grad(
        lambda t: beta * jnp.sum(jnp.exp(jax.nn.log_softmax(t[0])) * (jax.nn.log_softmax(t[0]) - jax.nn.log_softmax(ref_theta[0])))
    )(theta0)[0]
    assert np.max(np.abs(np.asarray(actual_grad[0] - t0_single_step_only))) > 0.05


# ---------------------------------------------------------------------------
# PKD-C9: Rollout completion_mask intersection across 4 trainers
# ---------------------------------------------------------------------------


def test_rollout_completion_mask_intersected_for_opd_seqkd_ssd_and_sparse():
    prompt_ids = jnp.array([[3, 4]], dtype=jnp.int32)
    prompt_mask = jnp.array([[1, 1]], dtype=jnp.int32)
    # No EOS token in completion_ids (e.g. stopped by stop string); last 2 tokens are PAD.
    completion_ids = jnp.array([[5, 6, 0, 0]], dtype=jnp.int32)
    rollout_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    fake_results = GenerationResults(
        generation_results=["a b"],
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        sequences=jnp.concatenate([prompt_ids, completion_ids], axis=1),
        completion_ids=completion_ids,
        completion_mask=rollout_mask,
        decoded_prompts=["p"],
        completion_prompts=["p"],
        text=["this is a sufficiently long completion"],
        raw_text=["this is a sufficiently long completion"],
        reasoning=[None],
        tool_calls=[None],
    )
    raw_batch = {"input_ids": prompt_ids, "attention_mask": prompt_mask}

    # 1. OnPolicyDistillationTrainer
    opd = object.__new__(OnPolicyDistillationTrainer)
    opd.arguments = SimpleNamespace(generate_with_teacher=False)
    opd._eos_token_id = [99]
    opd._apply_user_data_collator = lambda b: b
    opd._purify_batch = lambda b: b
    opd._all_gather = lambda x: x
    opd._log_training_generations_to_wandb = lambda **kw: None
    opd.rollout = lambda **kw: fake_results
    opd_batch, _ = opd._preprocess_batch_input(SimpleNamespace(), raw_batch, is_train=True)
    np.testing.assert_array_equal(np.asarray(opd_batch["completion_mask"]), [[0, 0, 1, 1, 0, 0]])

    # 2. SeqKDTrainer (local teacher_state path)
    seqkd = object.__new__(SeqKDTrainer)
    seqkd.arguments = SimpleNamespace()
    seqkd.teacher_fn = None
    seqkd.teacher_state = SimpleNamespace()
    seqkd._eos_token_id = [99]
    seqkd._apply_user_data_collator = lambda b: b
    seqkd._purify_batch = lambda b: b
    seqkd._all_gather = lambda x: x
    seqkd._log_training_generations_to_wandb = lambda **kw: None
    seqkd.rollout = lambda **kw: fake_results
    seqkd_batch, _ = seqkd._preprocess_batch_input(SimpleNamespace(), raw_batch, is_train=True)
    np.testing.assert_array_equal(np.asarray(seqkd_batch["attention_mask"]), [[1, 1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(np.asarray(seqkd_batch["labels"]), [[-100, -100, 5, 6, -100, -100]])

    # 3. SSDTrainer (also checks PKD-H15 _all_gather calls)
    gathered_count = 0

    def _counting_gather(x):
        nonlocal gathered_count
        gathered_count += 1
        return x

    ssd = object.__new__(SSDTrainer)
    ssd.arguments = SimpleNamespace(filter_empty=True)
    ssd._eos_token_id = [99]
    ssd._pad_token_id = 0
    ssd.steps_per_generation = 1
    ssd.num_iterations = 1
    ssd._apply_user_data_collator = lambda b: b
    ssd._purify_batch = lambda b: b
    ssd._all_gather = _counting_gather
    ssd.rollout = lambda **kw: fake_results
    ssd_batch, _ = ssd._preprocess_batch_input(SimpleNamespace(), raw_batch, is_train=True)
    np.testing.assert_array_equal(np.asarray(ssd_batch["completion_mask"]), [[1, 1, 0, 0]])
    assert gathered_count == 4

    # 4. SparseDistillationTrainer
    sparse = object.__new__(SparseDistillationTrainer)
    sparse.arguments = SimpleNamespace(top_k_teacher=2)
    sparse.teacher_fn = lambda ids, mask, prompts: (
        jnp.zeros((*ids.shape, 2), dtype=jnp.int32),
        jnp.zeros((*ids.shape, 2), dtype=jnp.float32),
    )
    sparse.teacher_state = None
    sparse.processing_class = SimpleNamespace(batch_decode=lambda arr, **kw: ["p"] * len(arr))
    sparse._eos_token_id = [99]
    sparse._apply_user_data_collator = lambda b: b
    sparse._purify_batch = lambda b: b
    sparse._all_gather = lambda x: x
    sparse._log_training_generations_to_wandb = lambda **kw: None
    sparse.rollout = lambda **kw: fake_results
    sparse_batch, _ = sparse._preprocess_batch_input(SimpleNamespace(), raw_batch, is_train=True)
    np.testing.assert_array_equal(np.asarray(sparse_batch["completion_mask"]), [[0, 0, 1, 1, 0, 0]])


# ---------------------------------------------------------------------------
# PKD-H1, PKD-M2, PKD-L3: Config sampling preservation, max_sequence_length, hashability, alpha
# ---------------------------------------------------------------------------


def test_distillation_configs_preserve_sampling_fold_max_seq_len_and_defaults():
    sparse_cfg = SparseDistillationConfig(top_p=0.82, top_k=17)
    assert sparse_cfg.top_p == pytest.approx(0.82)
    assert sparse_cfg.top_k == 17
    assert sparse_cfg.generation_top_p == pytest.approx(0.82)
    assert sparse_cfg.generation_top_k == 17
    assert sparse_cfg.alpha == pytest.approx(1.0)

    for cfg_cls in (SparseDistillationConfig, SeqKDConfig, OnPolicyDistillationConfig):
        cfg_none_k = cfg_cls(top_k=None)
        assert cfg_none_k.generation_top_k == 0
        sampling = esurge_generation.resolve_sampling(cfg_none_k)
        assert sampling.top_k == 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cfg_legacy = cfg_cls(max_sequence_length=384, max_prompt_length=128)
        assert cfg_legacy.max_length == 384
        assert cfg_legacy.max_completion_length == 256

    assert OnPolicyDistillationConfig().alpha == pytest.approx(1.0)
    assert MiniLLMConfig().gamma == pytest.approx(1.0)

    # PKD-M2: hashability and XPOConfig.loss_type validation
    assert isinstance(hash(SDFTConfig()), int)
    assert isinstance(hash(SelfDistillationConfig()), int)
    assert isinstance(hash(SSDConfig()), int)
    assert isinstance(hash(OnlineDPOConfig()), int)
    with pytest.raises(ValueError, match="Unsupported XPO"):
        XPOConfig(loss_type="invalid_loss")  # type: ignore[arg-type]

    # PKD-H15: SSDConfig inherits GRPOConfig (beta=0.0, no SDPO max_feedback_length)
    ssd_cfg = SSDConfig()
    assert isinstance(ssd_cfg, GRPOConfig)
    assert not isinstance(ssd_cfg, SDPOConfig)
    assert ssd_cfg.beta == pytest.approx(0.0)
    assert not hasattr(ssd_cfg, "max_feedback_length")


# ---------------------------------------------------------------------------
# PKD-H4: eval_config_overrides + eval_state_shardings across all 5 trainers
# ---------------------------------------------------------------------------


def test_eval_config_overrides_and_eval_state_shardings_across_five_trainers():
    from datasets import Dataset

    tok = _word_tokenizer(["hello", "world", "foo", "bar"])
    prompt_ds = Dataset.from_dict({"prompt": ["hello world", "foo bar"]})
    overrides = {"attn_mechanism": "sdpa"}

    # Use a non-default selector on an explicit teacher/ref state to also test
    # live-graphstate selector derivation in _configure_eval_config_override!
    def _only_lm_head_selector(v, path):
        del v
        return "lm_head" in path

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. XPOTrainer
        xpo_policy = _tiny_llama(0, attn_mechanism="vanilla")
        xpo_ref = _tiny_llama(1, attn_mechanism="vanilla")
        xpo_ref.eval()
        xpo_ref_state = xpo_ref.to_state(trainable_selector=_only_lm_head_selector)
        xpo_trainer = XPOTrainer(
            arguments=XPOConfig(
                max_training_steps=1,
                total_batch_size=1,
                max_prompt_length=4,
                max_completion_length=4,
                eval_config_overrides=overrides,
                use_wandb=False,
                save_directory=os.path.join(tmpdir, "xpo"),
            ),
            model=xpo_policy,
            reference_model=xpo_ref_state,
            reward_funcs=(lambda **kw: [0.0] * len(kw.get("completions", [])),),
            train_dataset=prompt_ds,
            eval_dataset=prompt_ds,
            processing_class=tok,
        )
        assert xpo_trainer._eval_graphdef is not None
        assert xpo_trainer._eval_ref_graphdef is not None
        xpo_eval_batch = {
            "prompt_ids": jnp.ones((1, 4), dtype=jnp.int32),
            "prompt_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "policy_completion_ids": jnp.full((1, 4), 2, dtype=jnp.int32),
            "policy_completion_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "ref_completion_ids": jnp.full((1, 4), 3, dtype=jnp.int32),
            "ref_completion_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "chosen_mask": jnp.array([True]),
            "beta": jnp.array([0.1], dtype=jnp.float32),
            "alpha": jnp.array([1e-5], dtype=jnp.float32),
            "loss_type": jnp.array([0.0], dtype=jnp.float32),
        }
        xpo_eval_metrics = xpo_trainer.sharded_evaluation_step_function(
            xpo_trainer._eval_state_for_compiled_step(xpo_trainer.model_state),
            xpo_eval_batch,
            *xpo_trainer._eval_shared_fn_extra_args,
            *xpo_trainer._eval_shared_fn_static_args,
        )
        assert np.isfinite(float(xpo_eval_metrics.loss))

        # 2. NashMDTrainer (also exercises PKD-H16 dynamic prompt_length & PKD-M2 tuple reward_funcs)
        nash_policy = _tiny_llama(2, attn_mechanism="vanilla")
        nash_ref = _tiny_llama(3, attn_mechanism="vanilla")
        nash_ref.eval()
        nash_ref_state = nash_ref.to_state(trainable_selector=_only_lm_head_selector)
        nash_trainer = NashMDTrainer(
            arguments=NashMDConfig(
                max_training_steps=1,
                total_batch_size=1,
                max_prompt_length=6,
                max_completion_length=4,
                eval_config_overrides=overrides,
                use_wandb=False,
                save_directory=os.path.join(tmpdir, "nash"),
            ),
            model=nash_policy,
            reference_model=nash_ref_state,
            reward_funcs=(lambda **kw: [0.0] * len(kw.get("completions", [])),),
            train_dataset=prompt_ds,
            eval_dataset=prompt_ds,
            processing_class=tok,
        )
        assert nash_trainer._eval_graphdef is not None
        assert nash_trainer._eval_ref_graphdef is not None
        # Dynamic prompt_length=3 != max_prompt_length=6 (both keyword and positional)!
        ref_logps = nash_trainer.eval_compute_refmodel_logps(
            nash_trainer.ref_state.graphstate,
            nash_trainer.ref_state.graphother,
            jnp.ones((1, 7), dtype=jnp.int32),
            jnp.ones((1, 7), dtype=jnp.int32),
            prompt_length=3,
        )
        assert ref_logps.shape == (1, 4)
        ref_logps_pos = nash_trainer.eval_compute_refmodel_logps(
            nash_trainer.ref_state.graphstate,
            nash_trainer.ref_state.graphother,
            jnp.ones((1, 7), dtype=jnp.int32),
            jnp.ones((1, 7), dtype=jnp.int32),
            3,
        )
        np.testing.assert_allclose(np.asarray(ref_logps), np.asarray(ref_logps_pos), atol=1e-6)
        nash_eval_batch = {
            "prompt_ids": jnp.ones((1, 3), dtype=jnp.int32),
            "prompt_mask": jnp.ones((1, 3), dtype=jnp.int32),
            "completion_ids": jnp.full((1, 4), 2, dtype=jnp.int32),
            "completion_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "ref_token_logps": ref_logps,
            "probabilities": jnp.array([0.6], dtype=jnp.float32),
            "beta": jnp.array([0.1], dtype=jnp.float32),
        }
        nash_eval_metrics = nash_trainer.sharded_evaluation_step_function(
            nash_trainer._eval_state_for_compiled_step(nash_trainer.model_state),
            nash_eval_batch,
            *nash_trainer._eval_shared_fn_extra_args,
            *nash_trainer._eval_shared_fn_static_args,
        )
        assert np.isfinite(float(nash_eval_metrics.loss))

        # 3. OnPolicyDistillationTrainer (also checks PKD-M4 FLOPs & non-student teacher selector)
        opd_student = _tiny_llama(4, attn_mechanism="vanilla")
        opd_teacher = _tiny_llama(5, attn_mechanism="vanilla")
        opd_trainer = OnPolicyDistillationTrainer(
            arguments=OnPolicyDistillationConfig(
                max_training_steps=1,
                total_batch_size=1,
                max_prompt_length=4,
                max_completion_length=4,
                eval_config_overrides=overrides,
                trainable_selector=_only_lm_head_selector,
                use_wandb=False,
                save_directory=os.path.join(tmpdir, "opd"),
            ),
            processing_class=tok,
            student_model=opd_student,
            teacher_model=opd_teacher,
            train_dataset=prompt_ds,
            eval_dataset=prompt_ds,
        )
        assert opd_trainer._eval_graphdef is not None
        assert opd_trainer._eval_teacher_graphdef is not None
        assert opd_trainer._extra_backward_flops_per_token == 0.0
        assert opd_trainer._extra_forward_flops_per_token > 0.0
        # Teacher must NOT inherit student's lm_head-only trainable_selector!
        teacher_param_count = len(jax.tree_util.tree_leaves(opd_trainer.teacher_state.graphstate))
        student_param_count = len(jax.tree_util.tree_leaves(opd_trainer.model_state.graphstate))
        assert teacher_param_count > student_param_count
        opd_eval_batch = {
            "input_ids": jnp.ones((1, 8), dtype=jnp.int32),
            "attention_mask": jnp.ones((1, 8), dtype=jnp.int32),
            "completion_mask": jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32),
        }
        opd_eval_metrics = opd_trainer.sharded_evaluation_step_function(
            opd_trainer._eval_state_for_compiled_step(opd_trainer.model_state),
            opd_eval_batch,
            *opd_trainer._eval_shared_fn_extra_args,
            *opd_trainer._eval_shared_fn_static_args,
        )
        assert np.isfinite(float(opd_eval_metrics.loss))

        # 4. SparseDistillationTrainer (also checks PKD-H9 JIT teacher scorer + PKD-M4 FLOPs)
        sparse_student = _tiny_llama(6, attn_mechanism="vanilla")
        sparse_teacher = _tiny_llama(7, attn_mechanism="vanilla")
        sparse_trainer = SparseDistillationTrainer(
            arguments=SparseDistillationConfig(
                max_training_steps=1,
                total_batch_size=1,
                max_prompt_length=4,
                max_completion_length=4,
                top_k_teacher=4,
                eval_config_overrides=overrides,
                trainable_selector=_only_lm_head_selector,
                use_wandb=False,
                save_directory=os.path.join(tmpdir, "sparse"),
            ),
            processing_class=tok,
            student_model=sparse_student,
            teacher_model=sparse_teacher,
            train_dataset=prompt_ds,
            eval_dataset=prompt_ds,
        )
        assert sparse_trainer._eval_graphdef is not None
        assert sparse_trainer._eval_teacher_graphdef is not None
        assert sparse_trainer._extra_backward_flops_per_token == 0.0
        assert sparse_trainer._compute_teacher_topk_eval is not sparse_trainer._compute_teacher_topk_train
        top_k_idx, top_k_lp = sparse_trainer._compute_teacher_topk_eval(
            sparse_trainer.teacher_state.graphstate,
            sparse_trainer.teacher_state.graphother,
            jnp.ones((1, 8), dtype=jnp.int32),
            jnp.ones((1, 8), dtype=jnp.int32),
        )
        assert top_k_idx.shape == (1, 8, 4)
        assert top_k_lp.shape == (1, 8, 4)
        sparse_eval_batch = {
            "input_ids": jnp.ones((1, 8), dtype=jnp.int32),
            "attention_mask": jnp.ones((1, 8), dtype=jnp.int32),
            "completion_mask": jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32),
            "teacher_top_k_indices": top_k_idx,
            "teacher_top_k_logprobs": top_k_lp,
        }
        sparse_eval_metrics = sparse_trainer.sharded_evaluation_step_function(
            sparse_trainer._eval_state_for_compiled_step(sparse_trainer.model_state),
            sparse_eval_batch,
            *sparse_trainer._eval_shared_fn_static_args,
        )
        assert np.isfinite(float(sparse_eval_metrics.loss))

        # 5. SSDTrainer
        ssd_policy = _tiny_llama(8, attn_mechanism="vanilla")
        ssd_trainer = SSDTrainer(
            arguments=SSDConfig(
                max_training_steps=1,
                total_batch_size=1,
                max_prompt_length=4,
                max_completion_length=4,
                eval_config_overrides=overrides,
                use_wandb=False,
                save_directory=os.path.join(tmpdir, "ssd"),
            ),
            model=ssd_policy,
            train_dataset=prompt_ds,
            eval_dataset=prompt_ds,
            processing_class=tok,
        )
        assert ssd_trainer._eval_graphdef is not None
        assert ssd_trainer.ref_state is None
        ssd_eval_batch = {
            "prompt_ids": jnp.ones((1, 4), dtype=jnp.int32),
            "prompt_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "completion_ids": jnp.full((1, 4), 2, dtype=jnp.int32),
            "completion_mask": jnp.ones((1, 4), dtype=jnp.int32),
            "num_items_in_batch": jnp.asarray(4, dtype=jnp.int32),
        }
        ssd_eval_metrics = ssd_trainer.sharded_evaluation_step_function(
            ssd_trainer._eval_state_for_compiled_step(ssd_trainer.model_state),
            ssd_eval_batch,
            *ssd_trainer._eval_shared_fn_static_args,
        )
        assert np.isfinite(float(ssd_eval_metrics.loss))


# ---------------------------------------------------------------------------
# PKD-H9 & PKD-M4: SparseDistillation bf16 true partial KL, stop_grad, aux/z-loss, GA
# ---------------------------------------------------------------------------


def test_sparse_distillation_bf16_true_partial_kl_zero_when_identical_and_stop_grad():
    # K = 4 == V with non-degenerate distribution in bfloat16.
    rng = jax.random.PRNGKey(0)
    logits_f32 = jax.random.normal(rng, (2, 200, 4), dtype=jnp.float32)
    logits_bf16 = logits_f32.astype(jnp.bfloat16)
    teacher_log_probs = jax.nn.log_softmax(logits_bf16.astype(jnp.float32), axis=-1)
    top_k_lp, top_k_idx = jax.lax.top_k(teacher_log_probs, 4)
    # > 256 active tokens (398 tokens) to verify float32 normalizer without bf16 saturation.
    loss_mask = jnp.ones((2, 200), dtype=jnp.int32)

    loss, metrics = partial_kl_distillation_loss(
        student_logits=logits_bf16,
        teacher_top_k_indices=top_k_idx,
        teacher_top_k_logprobs=top_k_lp,
        loss_mask=loss_mask,
        temperature=2.0,
        alpha=0.0,  # Pure KL (labels=None) must NOT be zeroed out by alpha=0.0!
    )
    assert loss.dtype == jnp.float32
    assert metrics["kl_loss"].dtype == jnp.float32
    assert float(loss) == pytest.approx(0.0, abs=1e-5)
    assert float(metrics["kl_loss"]) == pytest.approx(0.0, abs=1e-5)

    # Verify stop_gradient on teacher_top_k_logprobs.
    grad_teacher = jax.grad(
        lambda t_lp: partial_kl_distillation_loss(
            student_logits=logits_bf16 + 0.5,
            teacher_top_k_indices=top_k_idx,
            teacher_top_k_logprobs=t_lp,
            loss_mask=loss_mask,
            temperature=1.0,
            alpha=1.0,
        )[0]
    )(top_k_lp)
    np.testing.assert_array_equal(np.asarray(grad_teacher), 0.0)


def test_sparse_distillation_step_aux_loss_z_loss_and_ga_microbatch_weighting():
    mesh = _single_device_mesh()

    class AuxModel(spx.Module):
        def __init__(self):
            self.bias = spx.Parameter(jnp.array([0.2, -0.3, 0.5], dtype=jnp.float32))
            self.mesh = mesh

        def _get_mesh(self, mesh=None):
            return self.mesh if mesh is None else mesh

        def forward(self, input_ids, attention_mask):
            del attention_mask
            # Different logits per row via input_ids so uneven microbatches diverge without weighting.
            base = input_ids[..., None].astype(jnp.float32) * jnp.array([0.4, -0.2, 0.1])
            logits = base + self.bias.value[None, None, :]
            aux = jnp.mean(self.bias.value**2) * 0.5
            return SimpleNamespace(logits=logits, aux_loss=aux)

    state = EasyDeLState.create(model=AuxModel()).init_tx(optax.sgd(0.1))
    # Row 0 has 1 active shifted token; row 1 has 3 active shifted tokens (imbalanced GA microbatches!).
    batch = {
        "input_ids": jnp.array([[1, 2, 0, 0], [2, 1, 2, 1]], dtype=jnp.int32),
        "attention_mask": jnp.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=jnp.int32),
        "completion_mask": jnp.array([[0, 1, 0, 0], [0, 1, 1, 1]], dtype=jnp.int32),
        "teacher_top_k_indices": jnp.broadcast_to(jnp.array([0, 1], dtype=jnp.int32), (2, 4, 2)),
        "teacher_top_k_logprobs": jnp.broadcast_to(jnp.array([-0.2, -1.7], dtype=jnp.float32), (2, 4, 2)),
    }
    loss_cfg = LossConfig(z_loss=0.1)

    eval_metrics = sparse_distillation_step(
        state,
        batch,
        loss_config=loss_cfg,
        partition_spec=PartitionSpec(),
        gradient_accumulation_steps=1,
        is_training=False,
        temperature=1.0,
        alpha=0.5,
    )
    assert "z_loss" in eval_metrics.other_metrics and float(eval_metrics.other_metrics["z_loss"]) > 0.0
    assert "aux_loss" in eval_metrics.other_metrics and float(eval_metrics.other_metrics["aux_loss"]) > 0.0
    assert float(eval_metrics.loss) == pytest.approx(
        float(eval_metrics.other_metrics["kl_loss"])
        + float(eval_metrics.other_metrics["z_loss"])
        + float(eval_metrics.other_metrics["aux_loss"]),
        abs=1e-5,
    )

    # Compare full-batch (GA=1) vs token-weighted GA=2 gradients on pure KL (without per-row aux).
    class PureModel(spx.Module):
        def __init__(self):
            self.bias = spx.Parameter(jnp.array([0.2, -0.3, 0.5], dtype=jnp.float32))
            self.mesh = mesh

        def _get_mesh(self, mesh=None):
            return self.mesh if mesh is None else mesh

        def forward(self, input_ids, attention_mask):
            del attention_mask
            base = input_ids[..., None].astype(jnp.float32) * jnp.array([0.4, -0.2, 0.1])
            return SimpleNamespace(logits=base + self.bias.value[None, None, :])

    state_ga1 = EasyDeLState.create(model=PureModel()).init_tx(optax.sgd(0.1))
    state_ga2 = EasyDeLState.create(model=PureModel()).init_tx(optax.sgd(0.1))
    next_ga1, _ = sparse_distillation_step(
        state_ga1,
        batch,
        loss_config=loss_cfg,
        partition_spec=PartitionSpec(),
        gradient_accumulation_steps=1,
        is_training=True,
        temperature=1.0,
        alpha=1.0,
    )
    next_ga2, _ = sparse_distillation_step(
        state_ga2,
        batch,
        loss_config=loss_cfg,
        partition_spec=PartitionSpec(),
        gradient_accumulation_steps=2,
        is_training=True,
        temperature=1.0,
        alpha=1.0,
    )
    np.testing.assert_allclose(
        np.asarray(next_ga1.graphstate["parameters"]["bias"]),
        np.asarray(next_ga2.graphstate["parameters"]["bias"]),
        rtol=1e-5,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# PKD-H10 & PKD-H11 & PKD-L2: Nash-MD cache keys, prompt_mask EOS, dual-KV mixture, penalty
# ---------------------------------------------------------------------------


def test_nash_md_beta_cache_key_stable_and_kv_mixture_matches_processor():
    # 1. Cache key stability for fresh jnp arrays and batch-carried beta.
    dummy_state = SimpleNamespace(graphdef=object(), graphother=object())
    base_batch = {"prompt_ids": jnp.ones((1, 2), dtype=jnp.int32)}
    call_a = ScheduledStepCall(
        nash_md_step,
        dummy_state,
        base_batch,
        (),
        {},
        {"beta": jnp.asarray(0.25, dtype=jnp.float32), "logprob_vocab_chunk_size": None, "partition_spec": None},
        None,
    )
    call_b = ScheduledStepCall(
        nash_md_step,
        dummy_state,
        base_batch,
        (),
        {},
        {"beta": jnp.asarray(0.25, dtype=jnp.float32), "logprob_vocab_chunk_size": None, "partition_spec": None},
        None,
    )
    assert _nash_md_scheduled_loss_cache_key(call_a) == _nash_md_scheduled_loss_cache_key(call_b)

    batch_beta_1 = {**base_batch, "beta": jnp.array([0.1], dtype=jnp.float32)}
    batch_beta_2 = {**base_batch, "beta": jnp.array([0.9], dtype=jnp.float32)}
    call_c = ScheduledStepCall(
        nash_md_step, dummy_state, batch_beta_1, (), {}, {"beta": 0.1, "logprob_vocab_chunk_size": None}, None
    )
    call_d = ScheduledStepCall(
        nash_md_step, dummy_state, batch_beta_2, (), {}, {"beta": 0.9, "logprob_vocab_chunk_size": None}, None
    )
    assert _nash_md_scheduled_loss_cache_key(call_c) == _nash_md_scheduled_loss_cache_key(call_d)

    # 2. GeometricMixtureLogitsProcessor with prompt_mask preserves prompt EOS when pad == eos.
    captured_masks: list[np.ndarray] = []

    class FakeRef:
        def __call__(self, *, input_ids, attention_mask):
            captured_masks.append(np.asarray(jax.device_get(attention_mask)))
            return SimpleNamespace(logits=jnp.zeros((*input_ids.shape, 8), dtype=jnp.float32))

    # pad_token_id == eos_token_id == 2; index 0 is PAD (mask=0), index 1 is real EOS token 2 (mask=1)!
    prompt_mask = jnp.array([[0, 1, 1]], dtype=jnp.int32)
    proc = GeometricMixtureLogitsProcessor(FakeRef(), mixture_coef=0.5, pad_token_id=2, prompt_mask=prompt_mask)
    proc(jnp.array([[2, 2, 5, 6, 2]], dtype=jnp.int32), jnp.zeros((1, 8)), cur_len=jnp.asarray(4))
    np.testing.assert_array_equal(captured_masks[0], [[0, 1, 1, 1, 0]])

    # 3. Dual-KV cached mixture generator on real tiny Llama models matches greedy GeometricMixtureLogitsProcessor
    #    and reuses one compiled executable across different mixture_coef values.
    tok = _word_tokenizer(["a", "b", "c", "d"])
    policy_model = _tiny_llama(10, vocab_size=16)
    ref_model = _tiny_llama(11, vocab_size=16)
    policy_model.eval()
    ref_model.eval()
    policy_state = policy_model.to_state()
    ref_state = ref_model.to_state()

    trainer = object.__new__(NashMDTrainer)
    trainer.arguments = NashMDConfig(
        max_prompt_length=3,
        max_completion_length=3,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    trainer.arguments.generation_do_sample = False
    trainer.processing_class = tok
    trainer.tokenizer = tok
    trainer._pad_token_id = 0
    trainer._eos_token_id = [1]
    trainer._mixture_generate_cache = {}
    trainer._mixture_prng_key = None

    assert trainer._can_use_cached_kv_mixture(policy_state, ref_state)
    p_ids = jnp.array([[0, 3, 4]], dtype=jnp.int32)
    p_mask = jnp.array([[0, 1, 1]], dtype=jnp.int32)

    res1 = trainer._generate_mixture_rollout(
        policy_state=policy_state,
        ref_state=ref_state,
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        mixture_coef=0.3,
    )
    assert len(trainer._mixture_generate_cache) == 1
    # Changing mixture_coef must hit the exact same cached compiled function!
    res2 = trainer._generate_mixture_rollout(
        policy_state=policy_state,
        ref_state=ref_state,
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        mixture_coef=0.7,
    )
    assert len(trainer._mixture_generate_cache) == 1
    assert res1.completion_ids.shape == (1, 3)
    assert res2.completion_ids.shape == (1, 3)

    # Verify first greedy generated token matches direct 0.3 * ref_logits + 0.7 * policy_logits.
    with policy_model.mesh:
        p_logits0 = policy_state.model(input_ids=p_ids, attention_mask=p_mask).logits[:, -1]
        r_logits0 = ref_state.model(input_ids=p_ids, attention_mask=p_mask).logits[:, -1]
    expected_first_tok = int(jnp.argmax(0.3 * r_logits0 + 0.7 * p_logits0, axis=-1)[0])
    assert int(res1.completion_ids[0, 0]) == expected_first_tok


def test_nash_md_missing_eos_penalty_masks_pad_eos_and_logs_mixture_time():
    trainer = object.__new__(NashMDTrainer)
    trainer.arguments = SimpleNamespace(num_train_epochs=1)
    trainer.max_training_steps = 1
    trainer._current_step = 0
    trainer._beta_schedule = 0.2
    trainer._mixture_schedule = 1.0  # Use ref_state via rollout
    trainer.missing_eos_penalty = 3.0
    trainer.processing_class = SimpleNamespace(
        eos_token_id=99,
        batch_decode=lambda arr, **kw: ["txt"] * len(arr),
    )
    trainer._pad_token_id = 99
    trainer._eos_token_id = [99]
    trainer.ref_state = SimpleNamespace(graphstate={}, graphother={})
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x

    prompt_ids = jnp.array([[1], [2]], dtype=jnp.int32)
    prompt_mask = jnp.ones_like(prompt_ids)
    # Row 0: EOS 99 appears ONLY in masked pad position (mask=0) -> must be penalized!
    # Row 1: EOS 99 appears in active position (mask=1) -> must NOT be penalized!
    policy_comp_ids = jnp.array([[7, 99], [7, 99]], dtype=jnp.int32)
    policy_comp_mask = jnp.array([[1, 0], [1, 1]], dtype=jnp.int32)
    mix_comp_ids = jnp.array([[8, 99], [8, 99]], dtype=jnp.int32)
    mix_comp_mask = jnp.array([[1, 1], [1, 0]], dtype=jnp.int32)

    rollout_calls = iter(
        [
            GenerationResults(
                generation_results=["a", "b"],
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                sequences=jnp.concatenate([prompt_ids, policy_comp_ids], axis=1),
                completion_ids=policy_comp_ids,
                completion_mask=policy_comp_mask,
                decoded_prompts=["p1", "p2"],
                completion_prompts=["p1", "p2"],
                text=["a", "b"],
                raw_text=["a", "b"],
                reasoning=[None, None],
                tool_calls=[None, None],
            ),
            GenerationResults(
                generation_results=["c", "d"],
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                sequences=jnp.concatenate([prompt_ids, mix_comp_ids], axis=1),
                completion_ids=mix_comp_ids,
                completion_mask=mix_comp_mask,
                decoded_prompts=["p1", "p2"],
                completion_prompts=["p1", "p2"],
                text=["c", "d"],
                raw_text=["c", "d"],
                reasoning=[None, None],
                tool_calls=[None, None],
            ),
        ]
    )
    trainer.rollout = lambda **kw: next(rollout_calls)
    trainer.compute_refmodel_logps = lambda gs, go, ids, mask, prompt_length=None: jnp.zeros((2, 2), dtype=jnp.float32)
    trainer._score_rewards = lambda *a, **kw: (jnp.array([1.0, 1.0], dtype=jnp.float32), {})

    wandb_logs: list[dict] = []
    trainer._log_training_generations_to_wandb = lambda **kw: wandb_logs.append(kw)

    processed, metrics = trainer._preprocess_batch_input(
        SimpleNamespace(step=jnp.asarray(0)),
        {"input_ids": prompt_ids, "attention_mask": prompt_mask},
        is_train=True,
    )
    # Row 0: policy penalized (1 - 3 = -2), mix not penalized (1) -> diff = -3 -> prob < 0.5
    # Row 1: policy not penalized (1), mix penalized (1 - 3 = -2) -> diff = +3 -> prob > 0.5
    probs = np.asarray(processed["probabilities"])
    assert probs[0] == pytest.approx(float(jax.nn.sigmoid(-3.0)), abs=1e-5)
    assert probs[1] == pytest.approx(float(jax.nn.sigmoid(3.0)), abs=1e-5)
    assert len(wandb_logs) == 2
    assert wandb_logs[1]["source"] == "mixture"
    assert wandb_logs[1]["generation_time"] == metrics["mixture_generation_time"]
    assert isinstance(trainer._train_shared_fn_extra_args[0], float)


# ---------------------------------------------------------------------------
# PKD-M1 & PKD-M3: XPO symmetric KL, true chosen/rejected rewards, unequal widths, microbatched ref
# ---------------------------------------------------------------------------


def test_xpo_symmetric_kl_unequal_widths_true_rewards_and_scheduled_vocab_stage():
    mesh = _single_device_mesh()

    class FixedPolicyModel(spx.Module):
        def __init__(self, logits_vec):
            self.mesh = mesh
            self.logits_vec = spx.Parameter(jnp.asarray(logits_vec, dtype=jnp.float32))

        def forward(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            vec = self.logits_vec.value
            return SimpleNamespace(logits=jnp.broadcast_to(vec, (*input_ids.shape, vec.shape[0])))

    pol_logits = jnp.array([2.0, 0.0, -1.0], dtype=jnp.float32)
    ref_logits = jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32)
    pol_state = EasyDeLState.create(model=FixedPolicyModel(pol_logits))
    ref_state = EasyDeLState.create(model=FixedPolicyModel(ref_logits))

    # Unequal pair widths: policy has width 2, ref has width 3!
    p_logps, r_logps = _compute_pair_logps(
        pol_state.model,
        prompt_ids=jnp.array([[0], [0]], dtype=jnp.int32),
        prompt_mask=jnp.ones((2, 1), dtype=jnp.int32),
        policy_completion_ids=jnp.array([[0, 0], [0, 1]], dtype=jnp.int32),
        policy_completion_mask=jnp.ones((2, 2), dtype=jnp.int32),
        ref_completion_ids=jnp.array([[2, 2, 2], [2, 1, 0]], dtype=jnp.int32),
        ref_completion_mask=jnp.ones((2, 3), dtype=jnp.int32),
        logprob_vocab_chunk_size=None,
    )
    assert p_logps.shape == (2, 2)
    assert r_logps.shape == (2, 3)

    # Policy sample is token 0 (high prob under pol, low under ref) -> kl_policy > 0
    # Ref sample is token 2 (low prob under pol, high under ref) -> kl_ref < 0
    # Symmetric KL = (kl_policy - kl_ref) / 2 must be strictly POSITIVE (old formula canceled to 0!).
    batch = {
        "prompt_ids": jnp.array([[0], [0]], dtype=jnp.int32),
        "prompt_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "policy_completion_ids": jnp.array([[0], [0]], dtype=jnp.int32),
        "policy_completion_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "ref_completion_ids": jnp.array([[2], [2]], dtype=jnp.int32),
        "ref_completion_mask": jnp.ones((2, 1), dtype=jnp.int32),
        "chosen_mask": jnp.array([True, False]),
        "beta": jnp.array([0.1, 0.1], dtype=jnp.float32),
        "alpha": jnp.array([0.0, 0.0], dtype=jnp.float32),
        "loss_type": jnp.array([0.0, 0.0], dtype=jnp.float32),
    }
    eval_metrics = xpo_step(
        pol_state,
        batch,
        ref_state,
        None,
        None,
        None,
        PartitionSpec(),
        1,
        False,
    )
    expected_kl_pol = float(jax.nn.log_softmax(pol_logits)[0] - jax.nn.log_softmax(ref_logits)[0])
    expected_kl_ref = float(jax.nn.log_softmax(pol_logits)[2] - jax.nn.log_softmax(ref_logits)[2])
    expected_sym_kl = (expected_kl_pol - expected_kl_ref) / 2.0
    assert expected_sym_kl > 2.0
    assert float(eval_metrics.other_metrics["kl"]) == pytest.approx(expected_sym_kl, abs=1e-5)

    # Verify microbatched scheduled reference preparation splits batch across microbatches=2.
    call = ScheduledStepCall(
        xpo_step,
        pol_state,
        batch,
        (),
        {},
        {
            "reference_state": ref_state,
            "partition_spec": PartitionSpec(),
            "logprob_vocab_chunk_size": None,
            "gradient_accumulation_steps": 2,
        },
        SimpleNamespace(microbatches=2),
    )
    prepared = _prepare_xpo_scheduled_batch(call)
    assert "_ref_on_policy" in prepared and "_ref_on_ref" in prepared
    assert prepared["_ref_on_policy"].shape == (2, 1)
    assert prepared["_ref_on_ref"].shape == (2, 1)

    # Verify XPOTrainer logs true rewards/chosen >= rewards/rejected and passes completion_ids to W&B.
    trainer = object.__new__(XPOTrainer)
    trainer.arguments = SimpleNamespace(missing_eos_penalty=None)
    trainer._beta_schedule = 0.1
    trainer._alpha_schedule = 1e-5
    trainer.loss_type_id = 0
    trainer._current_step = 0
    trainer.ref_state = object()
    trainer.reward_funcs = []
    trainer.reward_processing_classes = None
    trainer.processing_class = SimpleNamespace(eos_token_id=99)
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda v: v
    wandb_calls: list[dict] = []
    trainer._log_training_generations_to_wandb = lambda **kw: wandb_calls.append(kw)
    p_ids = jnp.array([[1], [2]], dtype=jnp.int32)
    p_mask = jnp.ones_like(p_ids)
    pol_ids = jnp.array([[5], [6]], dtype=jnp.int32)
    ref_ids = jnp.array([[7], [8]], dtype=jnp.int32)
    trainer.rollout = lambda **kw: SimpleNamespace(
        sequences=ref_ids if kw["state"] is trainer.ref_state else pol_ids,
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        completion_ids=ref_ids if kw["state"] is trainer.ref_state else pol_ids,
        completion_mask=jnp.ones((2, 1), dtype=jnp.int32),
        text=None,
        raw_text=None,
        reasoning=None,
        tool_calls=None,
    )
    # Policy scores [-1.0, 2.0], Ref scores [3.0, 0.0] -> chosen=[3.0, 2.0] (mean 2.5), rejected=[-1.0, 0.0] (mean -0.5)
    scores_iter = iter([jnp.array([-1.0, 2.0]), jnp.array([3.0, 0.0])])
    trainer._score_rewards = lambda *a, **kw: (next(scores_iter), {})
    _, xpo_metrics = trainer._preprocess_batch_input(
        SimpleNamespace(step=jnp.asarray(0)),
        {"input_ids": p_ids, "attention_mask": p_mask},
        is_train=True,
    )
    assert xpo_metrics["rewards/chosen"] == pytest.approx(2.5)
    assert xpo_metrics["rewards/rejected"] == pytest.approx(-0.5)
    assert len(wandb_calls) == 2
    assert wandb_calls[0]["completion_ids"] is not None
    assert wandb_calls[1]["completion_ids"] is not None

    # PKD-M1: Verify vocab_shard_stage parameter exists and is forwarded in both XPO & Nash-MD scheduled losses.
    from easydel.trainers.nash_md_trainer import _fn as nash_fn
    from easydel.trainers.xpo_trainer import _fn as xpo_fn

    assert "vocab_shard_stage" in inspect.signature(xpo_fn._compute_logps).parameters
    assert "vocab_shard_stage" in inspect.signature(xpo_fn._compute_pair_logps).parameters
    assert "vocab_shard_stage" in inspect.signature(nash_fn._compute_policy_logps).parameters
    assert "vocab_shard_stage=" in inspect.getsource(xpo_fn._make_xpo_scheduled_loss)
    assert "vocab_shard_stage=" in inspect.getsource(nash_fn._make_nash_md_scheduled_loss)


# ---------------------------------------------------------------------------
# PKD-H13 & PKD-M4: MiniLLM buffering, length_normalization, logit-level kd_temperature, masked metric
# ---------------------------------------------------------------------------


def test_minillm_length_norm_logit_temperature_buffering_and_masked_metric():
    # 1. length_normalization decoupled from gamma <= 0
    adv_norm = MiniLLMTrainer._compute_reverse_kl_advantage(
        student_logps=jnp.zeros((1, 4), dtype=jnp.float32),
        teacher_logps=jnp.array([[2.0, 4.0, 100.0, 100.0]], dtype=jnp.float32),
        completion_mask=jnp.array([[1, 1, 0, 0]], dtype=jnp.int32),
        gamma=0.0,
        length_normalization=True,
    )
    np.testing.assert_allclose(np.asarray(adv_norm), [[1.0, 2.0, 0.0, 0.0]], atol=1e-6)

    # 2. Logit-level kd_temperature (log_softmax(z / T), NOT log_softmax(z) / T)
    mesh = _single_device_mesh()
    teacher_logits_vec = jnp.array([3.0, 1.0, -2.0], dtype=jnp.float32)
    student_logits_vec = jnp.array([0.0, 2.0, -1.0], dtype=jnp.float32)

    class FixedLogitModel:
        def __init__(self, vec):
            self.mesh = mesh
            self.vec = vec

        def __call__(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            return SimpleNamespace(logits=jnp.broadcast_to(self.vec, (*input_ids.shape, self.vec.shape[0])))

    trainer = object.__new__(MiniLLMTrainer)
    trainer.arguments = SimpleNamespace(
        num_generations=1,
        num_generations_eval=1,
        mask_truncated_completions=False,
        max_prompt_length=1,
        max_length=5,
        esurge_importance_sampling_correction=False,
        logprob_vocab_chunk_size=None,
        advantage_estimator="group_relative",
        reward_clip_range=None,
        rkl_advantage=True,
        gamma=0.0,
        length_normalization=False,
        kd_temperature=2.0,
        single_step_decomposition=False,
    )
    trainer.steps_per_generation = 1
    trainer.num_iterations = 1
    trainer.ref_state = None
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.scale_rewards = "none"
    trainer.multi_objective_aggregation = "sum_then_normalize"
    trainer.reward_weights = jnp.ones((1,))
    trainer.reward_funcs = [lambda **kw: [0.0]]
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["reward"]
    trainer._pad_token_id = 0
    trainer._eos_token_id = [99]
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda v: v
    trainer._tool_call_loop_enabled = lambda: False
    trainer._run_environment_feedback = lambda **kw: None
    trainer._maybe_log_grpo_completions = lambda **kw: None
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer.teacher_state = SimpleNamespace(model=FixedLogitModel(teacher_logits_vec))

    prompt = jnp.ones((1, 1), dtype=jnp.int32)
    comp = jnp.array([[0, 0, 0, 0]], dtype=jnp.int32)
    mask = jnp.array([[1, 0, 0, 0]], dtype=jnp.int32)  # 1 active token + 3 pad tokens!
    trainer.rollout = lambda **kw: GenerationResults(
        generation_results=["x"],
        prompt_ids=prompt,
        prompt_mask=jnp.ones_like(prompt),
        sequences=jnp.concatenate([prompt, comp], axis=1),
        completion_ids=comp,
        completion_mask=mask,
        decoded_prompts=["p"],
        completion_prompts=["p"],
        text=["x"],
        raw_text=["x"],
    )
    student_state = SimpleNamespace(model=FixedLogitModel(student_logits_vec), step=jnp.asarray(0))
    batch_out, metrics_out = trainer._preprocess_batch_input(
        student_state,
        {"input_ids": prompt, "attention_mask": jnp.ones_like(prompt)},
        is_train=True,
    )
    expected_rkl = float(
        jax.nn.log_softmax(teacher_logits_vec / 2.0)[0] - jax.nn.log_softmax(student_logits_vec / 2.0)[0]
    )
    wrong_post_hoc_rkl = float(
        (jax.nn.log_softmax(teacher_logits_vec)[0] - jax.nn.log_softmax(student_logits_vec)[0]) / 2.0
    )
    assert abs(expected_rkl - wrong_post_hoc_rkl) > 0.05
    assert float(batch_out["advantages"][0, 0]) == pytest.approx(expected_rkl, abs=1e-5)
    # Masked denominator metric: active mean is expected_rkl (NOT expected_rkl / 4)!
    assert metrics_out["minillm/rkl_advantage"] == pytest.approx(expected_rkl, abs=1e-5)

    # 3. Multi-iteration buffering (num_iterations=3): caches rkl_advantage and reuses old_per_token_logps!
    trainer.arguments.kd_temperature = 1.0
    trainer.num_iterations = 3
    trainer._buffered_grpo_batch = None
    trainer._buffered_grpo_remaining = 0

    score_calls: list[str] = []

    def _recording_compute_state_logps(model_state, ids, msk, kwargs, *, prompt_length, logprob_vocab_chunk_size):
        del ids, msk, kwargs, prompt_length, logprob_vocab_chunk_size
        score_calls.append(model_state.tag)
        return jnp.full((1, 4), model_state.const_logp, dtype=jnp.float32)

    trainer.compute_state_logps = _recording_compute_state_logps
    trainer.teacher_state = SimpleNamespace(tag="teacher", const_logp=-0.2, model=SimpleNamespace())
    s0 = SimpleNamespace(tag="student_step0", const_logp=-1.0, model=SimpleNamespace(), step=jnp.asarray(0))
    s1 = SimpleNamespace(tag="student_step1", const_logp=-5.0, model=SimpleNamespace(), step=jnp.asarray(1))
    s2 = SimpleNamespace(tag="student_step2", const_logp=-9.0, model=SimpleNamespace(), step=jnp.asarray(2))

    b0, m0 = trainer._preprocess_batch_input(s0, {"input_ids": prompt, "attention_mask": jnp.ones_like(prompt)}, True)
    # GRPO scored student_step0 once for old_per_token_logps; MiniLLM scored teacher once and REUSED old_per_token_logps!
    assert score_calls == ["student_step0", "teacher"]
    assert m0["generation_reused"] == 0

    b1, m1 = trainer._preprocess_batch_input(s1, {"input_ids": prompt, "attention_mask": jnp.ones_like(prompt)}, True)
    b2, m2 = trainer._preprocess_batch_input(s2, {"input_ids": prompt, "attention_mask": jnp.ones_like(prompt)}, True)
    # Buffered steps 1 and 2 must NOT score teacher or updated students s1/s2!
    assert score_calls == ["student_step0", "teacher"]
    assert m1["generation_reused"] == 1
    assert m2["generation_reused"] == 1
    np.testing.assert_allclose(np.asarray(b1["advantages"]), np.asarray(b0["advantages"]))
    np.testing.assert_allclose(np.asarray(b2["advantages"]), np.asarray(b0["advantages"]))


# ---------------------------------------------------------------------------
# PKD-H14: SDFT direct teacher prefix, positive rewards, token skip, teacher-gen W&B logging
# ---------------------------------------------------------------------------


def test_sdft_direct_teacher_prefix_positive_reward_token_skip_and_teacher_gen_logging():
    tok = _word_tokenizer(
        ["Context:", "secret", "Prompt:", "question", "answer", "extra", "unsuccessful", "earlier", "attempt"]
    )
    trainer = object.__new__(SDFTTrainer)
    trainer.arguments = SDFTConfig(
        max_prompt_length=4,
        max_completion_length=4,
        max_feedback_length=8,
        teacher_prompt_template="Context: {privileged_context} Prompt: {prompt}",
        generate_from_teacher=True,
        num_loss_tokens_to_skip=1,
        success_reward_threshold=0.0,
        beta=0.0,
        mask_truncated_completions=False,
    )
    trainer.processing_class = tok
    trainer.tokenizer = tok
    trainer._pad_token_id = tok.pad_token_id
    trainer._eos_token_id = [tok.eos_token_id]
    trainer._effective_feedback_length = 8
    trainer.teacher_prompt_length = 12
    trainer.num_generations = 1
    trainer.train_is_conversational = False
    trainer.eval_is_conversational = False
    trainer.reward_funcs = [lambda **kw: [1.0]]  # Positive reward > threshold (0.0)!
    trainer.reward_processing_classes = [None]
    trainer.reward_func_names = ["pos_reward"]
    trainer.reward_weights = jnp.ones((1,), dtype=jnp.float32)
    trainer.feedback_func = trainer._sdft_template_feedback
    trainer._sdft_user_feedback_func = None
    trainer._sdft_privileged_context = None
    trainer._sdft_active_generation_state = None
    trainer._sdft_rendered_teacher_prefixes = None
    trainer._sdft_last_rollout_completion_mask = None
    trainer._sdft_in_preprocess = False
    trainer._sdft_is_train = True
    trainer._apply_user_data_collator = lambda b: b
    trainer._extract_reward_batch_sidechannels = lambda b: {}
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x

    ref_state = SimpleNamespace(tag="ref_teacher", step=jnp.asarray(0))
    live_student_state = SimpleNamespace(tag="live_student", step=jnp.asarray(7))
    trainer.ref_state = ref_state

    prompt_ids = jnp.array([[0, 0, 0, tok.convert_tokens_to_ids("question")]], dtype=jnp.int32)
    prompt_mask = jnp.array([[0, 0, 0, 1]], dtype=jnp.int32)
    comp_ids = jnp.array(
        [[tok.convert_tokens_to_ids("answer"), tok.convert_tokens_to_ids("extra"), 0, 0]],
        dtype=jnp.int32,
    )
    comp_mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)

    rollout_states: list[object] = []
    wandb_records: list[dict] = []

    from easydel.trainers.self_distillation_policy_optimization import SDPOTrainer

    # Bind parent rollout & wandb logging cleanly on a lightweight subclass-safe stub.
    orig_rollout = SDPOTrainer.rollout
    orig_wandb = SDPOTrainer._log_training_generations_to_wandb
    try:
        SDPOTrainer.rollout = lambda self, *a, **kw: (
            rollout_states.append(kw.get("state")),
            GenerationResults(
                generation_results=["answer extra"],
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                sequences=jnp.concatenate([prompt_ids, comp_ids], axis=1),
                completion_ids=comp_ids,
                completion_mask=comp_mask,
                decoded_prompts=["question"],
                completion_prompts=["question"],
                text=["answer extra"],
                raw_text=["answer extra"],
                reasoning=[None],
                tool_calls=[None],
            ),
        )[1]
        SDPOTrainer._log_training_generations_to_wandb = lambda self, *a, **kw: wandb_records.append(kw)

        out_batch, _ = trainer._preprocess_batch_input(
            live_student_state,
            {"input_ids": prompt_ids, "attention_mask": prompt_mask, "privileged_context": ["secret"]},
            is_train=True,
        )
    finally:
        SDPOTrainer.rollout = orig_rollout
        SDPOTrainer._log_training_generations_to_wandb = orig_wandb

    # 1. Rollout used ref_state, W&B logged live_student_state (step=7, NOT 0) and source="teacher"!
    assert rollout_states == [ref_state]
    assert len(wandb_records) == 1
    assert wandb_records[0]["state"] is live_student_state
    assert int(wandb_records[0]["state"].step) == 7
    assert wandb_records[0]["source"] == "teacher"

    # 2. Teacher prefix rendered directly without failure framing, without duplicated prompt, and retained on reward=1.0!
    teacher_prefix_ids = np.asarray(out_batch["teacher_ids"])[:, : trainer.teacher_prompt_length]
    teacher_prefix_mask = np.asarray(out_batch["teacher_mask"])[:, : trainer.teacher_prompt_length]
    # Left-padded so active tokens sit right-aligned immediately before completion_ids!
    assert teacher_prefix_mask[0, -1] == 1
    decoded_prefix = tok.decode(teacher_prefix_ids[0], skip_special_tokens=True)
    assert decoded_prefix == "Context: secret Prompt: question"
    assert "unsuccessful" not in decoded_prefix
    assert decoded_prefix.count("question") == 1

    # 3. num_items_in_batch accounts for num_loss_tokens_to_skip=1 (2 active tokens - 1 skipped = 1)
    #    AND rollout completion_mask was respected (2 trailing PAD tokens masked out).
    np.testing.assert_array_equal(np.asarray(out_batch["completion_mask"]), [[1, 1, 0, 0]])
    assert int(out_batch["num_items_in_batch"]) == 1


# ---------------------------------------------------------------------------
# PKD-H15 & PKD-L3: SeqKD teacher_fn right-padding + EOS & eval W&B guard
# ---------------------------------------------------------------------------


def test_seq_kd_teacher_fn_right_pads_appends_eos_and_skips_eval_wandb():
    tok = _word_tokenizer(["hello", "world", "prompt"], padding_side="left")
    assert tok.padding_side == "left"

    trainer = object.__new__(SeqKDTrainer)
    trainer.arguments = SeqKDConfig(max_prompt_length=3, max_completion_length=4)
    trainer.processing_class = tok
    trainer.teacher_fn = lambda prompts: ["hello world" for _ in prompts]
    trainer.teacher_state = None
    trainer._eos_token_id = [tok.eos_token_id]
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x
    wandb_calls: list[dict] = []
    trainer._log_training_generations_to_wandb = lambda **kw: wandb_calls.append(kw)

    prompt_ids = jnp.array([[0, 0, tok.convert_tokens_to_ids("prompt")]], dtype=jnp.int32)
    prompt_mask = jnp.array([[0, 0, 1]], dtype=jnp.int32)

    batch_eval, _ = trainer._preprocess_batch_input(
        SimpleNamespace(),
        {"input_ids": prompt_ids, "attention_mask": prompt_mask},
        is_train=False,
    )
    # Eval rollouts must NOT log to training_generations W&B table!
    assert wandb_calls == []

    # Prompt is left-padded [0, 0, prompt], completion is RIGHT-padded [hello, world, EOS, 0] (no PAD gap!).
    h_id = tok.convert_tokens_to_ids("hello")
    w_id = tok.convert_tokens_to_ids("world")
    eos_id = tok.eos_token_id
    p_id = tok.convert_tokens_to_ids("prompt")
    np.testing.assert_array_equal(np.asarray(batch_eval["input_ids"]), [[0, 0, p_id, h_id, w_id, eos_id, 0]])
    np.testing.assert_array_equal(np.asarray(batch_eval["attention_mask"]), [[0, 0, 1, 1, 1, 1, 0]])
    np.testing.assert_array_equal(np.asarray(batch_eval["labels"]), [[-100, -100, -100, h_id, w_id, eos_id, -100]])
    assert tok.padding_side == "left"


def test_pkd_h13_minillm_headless_lmhead_prepare_inputs_and_compute_lm_logits_fallback():
    """PKD-H13: MiniLLM headless temperature scoring honors prepare_lm_head_inputs and compute_lm_logits fallback."""
    from easydel.trainers.minillm_trainer.minillm_trainer import _compute_logps_with_temperature

    vocab_size = 7
    hidden_size = 4
    embed = jnp.sin(jnp.arange(vocab_size * hidden_size, dtype=jnp.float32).reshape(vocab_size, hidden_size) * 0.7)
    lm_weight = jnp.cos(
        jnp.arange(hidden_size * vocab_size, dtype=jnp.float32).reshape(hidden_size, vocab_size) * 0.5
    )

    class _HeadlessComputeOnlyLM:
        def __init__(self, chunk_size: int | None = 2):
            self.config = SimpleNamespace(lmhead_chunksize=chunk_size)
            self.prepare_calls = 0

        def prepare_lm_head_inputs(self, hidden_states):
            self.prepare_calls += 1
            return hidden_states * 1.5 + 0.25

        def compute_lm_logits(self, hidden_states):
            return hidden_states @ lm_weight

        def __call__(self, input_ids, attention_mask=None, apply_lm_head=True, **kwargs):
            del attention_mask, kwargs
            h = embed[input_ids]
            if not apply_lm_head:
                return SimpleNamespace(logits=None, last_hidden_state=h)
            logits = self.compute_lm_logits(self.prepare_lm_head_inputs(h))
            return SimpleNamespace(logits=logits, last_hidden_state=h)

    class _HeadlessMakeFnLM(_HeadlessComputeOnlyLM):
        def __init__(self, chunk_size: int | None = 2):
            super().__init__(chunk_size=chunk_size)
            self.make_fn_calls = 0

        def make_lm_head_fn(self, vocab_shard_stage=None):
            del vocab_shard_stage
            self.make_fn_calls += 1
            return self.compute_lm_logits

    input_ids = jnp.asarray([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    prompt_length = 2
    temperature = 2.5

    dense_model = _HeadlessComputeOnlyLM(chunk_size=None)
    dense_logps = _compute_logps_with_temperature(
        dense_model,
        input_ids,
        attention_mask,
        prompt_length=prompt_length,
        logprob_vocab_chunk_size=3,
        temperature=temperature,
    )

    # 1. Model exposing compute_lm_logits + prepare_lm_head_inputs (no make_lm_head_fn).
    compute_only = _HeadlessComputeOnlyLM(chunk_size=2)
    headless_compute_only = _compute_logps_with_temperature(
        compute_only,
        input_ids,
        attention_mask,
        prompt_length=prompt_length,
        logprob_vocab_chunk_size=3,
        temperature=temperature,
    )
    assert compute_only.prepare_calls > 0
    np.testing.assert_allclose(np.asarray(headless_compute_only), np.asarray(dense_logps), rtol=1e-5, atol=1e-5)

    # 2. Model exposing make_lm_head_fn + prepare_lm_head_inputs.
    make_fn_model = _HeadlessMakeFnLM(chunk_size=2)
    headless_make_fn = _compute_logps_with_temperature(
        make_fn_model,
        input_ids,
        attention_mask,
        prompt_length=prompt_length,
        logprob_vocab_chunk_size=3,
        temperature=temperature,
    )
    assert make_fn_model.prepare_calls > 0
    assert make_fn_model.make_fn_calls == 1
    np.testing.assert_allclose(np.asarray(headless_make_fn), np.asarray(dense_logps), rtol=1e-5, atol=1e-5)

    # 3. Missing compute_lm_logits raises clear TypeError; empty seq_len works cleanly.
    with pytest.raises(TypeError, match="compute_lm_logits"):
        _logprob_utils.compute_per_token_logps_and_entropies_from_hidden_states(
            model=SimpleNamespace(),
            hidden_states=jnp.zeros((2, 3, hidden_size), dtype=jnp.float32),
            targets=jnp.zeros((2, 3), dtype=jnp.int32),
            token_chunk_size=2,
            vocab_chunk_size=3,
            return_entropy=False,
            temperature=temperature,
        )
    empty_logps, empty_ent = _logprob_utils.compute_per_token_logps_and_entropies_from_hidden_states(
        model=compute_only,
        hidden_states=jnp.zeros((2, 0, hidden_size), dtype=jnp.float32),
        targets=jnp.zeros((2, 0), dtype=jnp.int32),
        token_chunk_size=2,
        vocab_chunk_size=3,
        return_entropy=False,
        temperature=temperature,
    )
    assert empty_logps.shape == (2, 0)
    assert empty_ent is None


# ===========================================================================
# Leak & Post-Fix Audit Regressions (#1, #11, #12, #13, #14, #22, #23, #24, #31, #32)
# ===========================================================================


class _TrackedDict(dict):
    """Weak-referenceable dict subclass for proving Step-0 batch release."""


class _TrackedSentinel:
    """Weak-referenceable payload for proving Step-0 opt_state / state release."""


def _closure_reachable_ids(fn) -> set[int]:
    """Return object IDs transitively reachable from a function's closure cells."""
    visited: set[int] = set()
    stack: list[object] = []
    for cell in fn.__closure__ or ():
        try:
            stack.append(cell.cell_contents)
        except ValueError:
            pass
    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        if isinstance(obj, (type, int, float, str, bytes, bool, type(None))):
            continue
        # Avoid traversing global module dicts or code globals.
        if inspect.ismodule(obj):
            continue
        if inspect.isfunction(obj):
            for c in obj.__closure__ or ():
                try:
                    stack.append(c.cell_contents)
                except ValueError:
                    pass
            continue
        for ref in gc.get_referents(obj):
            if not inspect.ismodule(ref) and not isinstance(ref, type):
                stack.append(ref)
    return visited


def test_leak_1_all_11_pref_kd_spec_scheduled_losses_do_not_pin_step0_call_or_state():
    """#1: All 11 Pref/KD/Spec MPMD scheduled_loss closures must never pin Step-0 call, state, opt_state, or batch."""
    _, model = _make_tiny_llama(seed=1, vocab_size=16, hidden_size=16, num_layers=1)
    _, teacher_model = _make_tiny_llama(seed=2, vocab_size=16, hidden_size=16, num_layers=1)
    teacher_state = teacher_model.to_state()

    def _dummy_forward(m, b, **kwargs):
        del m, b, kwargs
        return {
            "chosen_logps": jnp.zeros((2,), dtype=jnp.float32),
            "rejected_logps": jnp.zeros((2,), dtype=jnp.float32),
            "chosen_logps_raw": jnp.zeros((2,), dtype=jnp.float32),
            "chosen_lengths": jnp.ones((2,), dtype=jnp.float32),
            "completion_logps": jnp.zeros((2,), dtype=jnp.float32),
            "mean_completion_logits": jnp.array(0.0, dtype=jnp.float32),
        }

    cases: list[tuple[str, object, dict[str, object], dict[str, object]]] = [
        ("dpo", dpo_fn.training_step, {"concatenated_forward": _dummy_forward, "beta": 0.1}, {}),
        ("cpo", cpo_fn.training_step, {"concatenated_forward_fn": _dummy_forward, "beta": 0.1}, {}),
        ("orpo", orpo_fn.orpo_training_step, {"concatenated_forward": _dummy_forward, "beta": 0.1}, {}),
        ("kto", kto_fn.training_step, {"forward_fn": _dummy_forward, "beta": 0.1}, {}),
        (
            "bco",
            bco_fn.training_step,
            {"concatenated_forward_fn": _dummy_forward, "beta": 0.1, "gradient_accumulation_steps": 1},
            {"label": jnp.array([True, False])},
        ),
        ("tpo", tpo_fn.tpo_training_step, {"concatenated_forward": _dummy_forward, "beta": 0.01}, {}),
        ("nash_md", nash_md_step, {"beta": 0.1}, {}),
        ("xpo", xpo_step, {}, {}),
        ("distillation", kd_fn.distillation_step, {"teacher_state": teacher_state}, {}),
        ("gkd", gkd_fn.gkd_step, {"teacher_state": teacher_state}, {}),
        ("speculative_decoding", spec_fn.speculative_decoding_step, {"target_state": teacher_state}, {}),
    ]

    assert len(cases) == 11
    for name, step_fn, bound_args, batch_extra in cases:
        adapter = get_scheduled_loss_adapter(step_fn)
        assert adapter is not None, f"Missing scheduled loss adapter for {name}"

        opt_sentinel = _TrackedSentinel()
        batch_sentinel = _TrackedSentinel()
        step0_batch = _TrackedDict({"_sentinel": batch_sentinel, **batch_extra})
        step0_state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)
        step0_state = step0_state.replace(opt_state=(step0_state.opt_state, opt_sentinel))
        step0_call = ScheduledStepCall(
            step_fn=step_fn,
            state=step0_state,
            batch=step0_batch,
            args=(),
            kwargs={},
            bound_arguments=dict(bound_args),
            schedule=None,
        )

        call_ref = weakref.ref(step0_call)
        batch_ref = weakref.ref(step0_batch)
        batch_sentinel_ref = weakref.ref(batch_sentinel)
        opt_sentinel_ref = weakref.ref(opt_sentinel)
        forbidden_ids = {
            id(step0_call),
            id(step0_state),
            id(step0_state.graphstate),
            id(step0_state.opt_state),
            id(step0_batch),
            id(opt_sentinel),
            id(batch_sentinel),
        }

        scheduled_loss = adapter.make_loss(step0_call)
        reachable = _closure_reachable_ids(scheduled_loss)
        leaked = forbidden_ids & reachable
        assert not leaked, f"Adapter {name!r} scheduled_loss closure pinned Step-0 objects: {leaked}"

        del step0_call, step0_state, step0_batch, opt_sentinel, batch_sentinel
        gc.collect()

        assert call_ref() is None, f"Adapter {name!r} pinned Step-0 ScheduledStepCall"
        assert batch_ref() is None, f"Adapter {name!r} pinned Step-0 batch dict"
        assert batch_sentinel_ref() is None, f"Adapter {name!r} pinned Step-0 batch payload"
        assert opt_sentinel_ref() is None, f"Adapter {name!r} pinned Step-0 opt_state"


def test_leak_11_nash_md_mixture_cache_lru_identity_release_and_compiled_fallback():
    """#11: NashMDTrainer caps mixture JIT cache as an identity-checked LRU, clears on teardown, and caches fallback."""
    tok = _word_tokenizer(["a", "b", "c", "d"])
    _, policy_model = _make_tiny_llama(seed=21, vocab_size=16, hidden_size=16, num_layers=1)
    _, ref_model = _make_tiny_llama(seed=22, vocab_size=16, hidden_size=16, num_layers=1)
    policy_model.eval()
    ref_model.eval()
    policy_state = policy_model.to_state()
    ref_state = ref_model.to_state()

    trainer = object.__new__(NashMDTrainer)
    trainer.arguments = NashMDConfig(
        max_prompt_length=3,
        max_completion_length=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )
    trainer.arguments.generation_do_sample = False
    trainer.processing_class = tok
    trainer.tokenizer = tok
    trainer.model_state = policy_state
    trainer.ref_state = ref_state
    trainer._pad_token_id = 0
    trainer._eos_token_id = [1]
    trainer._mixture_generate_cache = {}
    trainer._mixture_prng_key = None
    trainer._MIXTURE_GENERATE_CACHE_MAXSIZE = 2

    # 1. LRU cap evicts oldest entry and refreshes recently hit entries.
    gd_a, gd_b, gd_c = object(), object(), object()
    fn_a, fn_b, fn_c = object(), object(), object()
    trainer._put_cached_mixture_fn(("k", 1), policy_graphdef=gd_a, ref_graphdef=gd_a, compiled=fn_a)
    trainer._put_cached_mixture_fn(("k", 2), policy_graphdef=gd_b, ref_graphdef=gd_b, compiled=fn_b)
    # Touch key 1 so key 2 becomes LRU.
    assert trainer._get_cached_mixture_fn(("k", 1), policy_graphdef=gd_a, ref_graphdef=gd_a) is fn_a
    trainer._put_cached_mixture_fn(("k", 3), policy_graphdef=gd_c, ref_graphdef=gd_c, compiled=fn_c)
    assert len(trainer._mixture_generate_cache) == 2
    assert ("k", 1) in trainer._mixture_generate_cache
    assert ("k", 3) in trainer._mixture_generate_cache
    assert ("k", 2) not in trainer._mixture_generate_cache

    # 2. Identity mismatch rejects recycled-id cache collision and removes stale entry.
    assert trainer._get_cached_mixture_fn(("k", 1), policy_graphdef=object(), ref_graphdef=gd_a) is None
    assert ("k", 1) not in trainer._mixture_generate_cache

    # 3. Mid-step release_generation_runtime(state=policy_state, clear_esurge_compiled_cache=False)
    #    preserves _mixture_generate_cache, whereas state=None or clear_esurge_compiled_cache=True clears it.
    trainer.release_generation_runtime(state=policy_state, clear_esurge_compiled_cache=False)
    assert len(trainer._mixture_generate_cache) == 1
    trainer.release_generation_runtime(state=None, clear_esurge_compiled_cache=False)
    assert len(trainer._mixture_generate_cache) == 0

    trainer._put_cached_mixture_fn(("k", 4), policy_graphdef=gd_a, ref_graphdef=gd_a, compiled=fn_a)
    trainer.release_generation_runtime(state=policy_state, clear_esurge_compiled_cache=True)
    assert len(trainer._mixture_generate_cache) == 0

    # 4. Compiled fallback path reuses ONE compiled executable across different mixture_coef & prompt_mask
    #    and matches direct geometric logit mixing on real tiny Llama models.
    trainer._MIXTURE_GENERATE_CACHE_MAXSIZE = 4
    assert trainer._can_use_cached_fallback_mixture(policy_state, ref_state)
    p_ids = jnp.array([[0, 3, 4]], dtype=jnp.int32)
    p_mask_1 = jnp.array([[0, 1, 1]], dtype=jnp.int32)
    p_mask_2 = jnp.array([[1, 1, 1]], dtype=jnp.int32)

    fb1 = trainer._generate_mixture_rollout(
        policy_state=policy_state,
        ref_state=ref_state,
        prompt_ids=p_ids,
        prompt_mask=p_mask_1,
        mixture_coef=0.25,
        use_kv_cache=False,
    )
    assert len(trainer._mixture_generate_cache) == 1
    fb2 = trainer._generate_mixture_rollout(
        policy_state=policy_state,
        ref_state=ref_state,
        prompt_ids=p_ids,
        prompt_mask=p_mask_2,
        mixture_coef=0.8,
        use_kv_cache=False,
    )
    assert len(trainer._mixture_generate_cache) == 1
    assert fb1.completion_ids.shape == (1, 2)
    assert fb2.completion_ids.shape == (1, 2)

    with policy_model.mesh:
        p_logits0 = policy_state.model(input_ids=p_ids, attention_mask=p_mask_1).logits[:, -1]
        r_logits0 = ref_state.model(input_ids=p_ids, attention_mask=p_mask_1).logits[:, -1]
    expected_tok = int(jnp.argmax(0.25 * r_logits0 + 0.75 * p_logits0, axis=-1)[0])
    assert int(fb1.completion_ids[0, 0]) == expected_tok


def test_leak_12_kto_and_bco_forward_fn_uses_partial_without_capturing_self():
    """#12: KTOTrainer and BCOTrainer bind concatenated_forward via functools.partial (never capturing self)."""
    tok = _word_tokenizer(["a", "b", "c", "d"])
    ds = Dataset.from_dict(
        {
            "prompt": ["a", "b"],
            "completion": ["c", "d"],
            "label": [True, False],
        }
    )

    _, bco_model = _make_tiny_llama(seed=31, vocab_size=16, hidden_size=16, num_layers=1)
    bco_trainer = BCOTrainer(
        arguments=BCOConfig(
            max_length=8,
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=bco_model,
        processing_class=tok,
        train_dataset=ds,
    )
    bco_fwd = bco_trainer._train_shared_fn_static_args[1]
    assert isinstance(bco_fwd, functools.partial)
    assert bco_fwd.func is bco_fn.concatenated_forward
    assert bco_trainer._eval_shared_fn_static_args[0] is bco_fwd
    bco_ref = weakref.ref(bco_trainer)
    del bco_trainer
    gc.collect()
    assert bco_ref() is None, "Holding BCO forward_fn must not keep BCOTrainer alive"

    _, kto_model = _make_tiny_llama(seed=32, vocab_size=16, hidden_size=16, num_layers=1)
    kto_trainer = KTOTrainer(
        arguments=KTOConfig(
            max_length=8,
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=kto_model,
        processing_class=tok,
        train_dataset=ds,
    )
    kto_fwd = kto_trainer._train_shared_fn_static_args[1]
    assert isinstance(kto_fwd, functools.partial)
    assert kto_fwd.func is bco_fn.concatenated_forward
    assert kto_trainer._eval_shared_fn_static_args[0] is kto_fwd
    kto_ref = weakref.ref(kto_trainer)
    del kto_trainer
    gc.collect()
    assert kto_ref() is None, "Holding KTO forward_fn must not keep KTOTrainer alive"


def test_leak_13_bco_kto_concatenated_forward_drops_completion_logits_and_uses_chunked_sum(monkeypatch):
    """#13: BCO/KTO concatenated_forward omits full [B, L, V] completion_logits and uses sum_logits_vocab_chunked."""
    rng = np.random.default_rng(13)
    raw_logits = jnp.asarray(rng.standard_normal((2, 5, 11)).astype(np.float32))

    class _MockLM:
        config = SimpleNamespace(lmhead_chunksize=None)

        def __call__(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, **kwargs):
            del input_ids, attention_mask, decoder_input_ids, labels, kwargs
            return SimpleNamespace(logits=raw_logits, aux_loss=None)

    chunk_calls: list[int | None] = []
    orig_sum = bco_fn.sum_logits_vocab_chunked

    def _spy_sum(logits, *, chunk_size):
        chunk_calls.append(chunk_size)
        return orig_sum(logits, chunk_size=chunk_size)

    monkeypatch.setattr(bco_fn, "sum_logits_vocab_chunked", _spy_sum)

    batch = {
        "prompt_input_ids": jnp.array([[1, 2], [2, 3]], dtype=jnp.int32),
        "prompt_attention_mask": jnp.ones((2, 2), dtype=jnp.int32),
        "completion_input_ids": jnp.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=jnp.int32),
        "completion_attention_mask": jnp.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=jnp.int32),
        "completion_labels": jnp.array([[-100, -100, 3, 4, 5], [-100, -100, 4, 5, -100]], dtype=jnp.int32),
    }

    causal_out = bco_fn.concatenated_forward(
        _MockLM(),
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=4,
    )
    assert "completion_logits" not in causal_out
    assert chunk_calls == [4]
    shifted_mask = batch["completion_labels"][:, 1:] != -100
    expected_causal_mean = (
        jnp.where(shifted_mask, raw_logits[:, :-1, :].sum(axis=-1), 0.0).sum() / shifted_mask.sum()
    )
    np.testing.assert_allclose(np.asarray(causal_out["mean_completion_logits"]), np.asarray(expected_causal_mean), atol=1e-5)

    enc_dec_out = bco_fn.concatenated_forward(
        _MockLM(),
        batch,
        is_encoder_decoder=True,
        label_pad_token_id=-100,
        padding_value=0,
        logprob_vocab_chunk_size=3,
    )
    assert "completion_logits" not in enc_dec_out
    assert chunk_calls == [4, 3]
    ed_mask = batch["completion_attention_mask"].astype(bool) & (batch["completion_labels"] != -100)
    expected_ed_mean = jnp.where(ed_mask, raw_logits.sum(axis=-1), 0.0).sum() / ed_mask.sum()
    np.testing.assert_allclose(np.asarray(enc_dec_out["mean_completion_logits"]), np.asarray(expected_ed_mean), atol=1e-5)


def test_leak_14_frozen_reward_model_states_drop_optimizer_state():
    """#14: Frozen reward-model EasyDeLStates in OnlineDPO/XPO/NashMD/MiniLLM/SDFT drop tx and opt_state."""
    tok = _word_tokenizer(["a", "b", "c", "d"])
    prompt_ds = Dataset.from_dict({"prompt": ["a", "b"]})

    _, policy_model = _make_tiny_llama(seed=41, vocab_size=16, hidden_size=16, num_layers=1)
    _, rm_model = _make_tiny_llama(seed=42, vocab_size=16, hidden_size=16, num_layers=1)
    rm_state_with_opt = EasyDeLState.create(model=rm_model, tx=optax.adam(1e-3), init_opt_state=True)
    assert rm_state_with_opt.tx is not None and rm_state_with_opt.opt_state is not None

    odpo = OnlineDPOTrainer(
        arguments=OnlineDPOConfig(
            max_length=8,
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=policy_model,
        reward_funcs=rm_state_with_opt,
        processing_class=tok,
        reward_processing_classes=tok,
        train_dataset=prompt_ds,
    )
    assert isinstance(odpo.reward_funcs[0], EasyDeLState)
    assert odpo.reward_funcs[0].tx is None
    assert odpo.reward_funcs[0].opt_state is None

    xpo = XPOTrainer(
        arguments=XPOConfig(
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=policy_model,
        reward_funcs=rm_state_with_opt,
        processing_class=tok,
        reward_processing_classes=tok,
        train_dataset=prompt_ds,
    )
    assert isinstance(xpo.reward_funcs[0], EasyDeLState)
    assert xpo.reward_funcs[0].tx is None
    assert xpo.reward_funcs[0].opt_state is None

    nash = NashMDTrainer(
        arguments=NashMDConfig(
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=policy_model,
        reward_funcs=rm_state_with_opt,
        processing_class=tok,
        reward_processing_classes=tok,
        train_dataset=prompt_ds,
    )
    assert isinstance(nash.reward_funcs[0], EasyDeLState)
    assert nash.reward_funcs[0].tx is None
    assert nash.reward_funcs[0].opt_state is None

    minillm = MiniLLMTrainer(
        arguments=MiniLLMConfig(
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=policy_model,
        teacher_model=rm_model,
        reward_funcs=rm_state_with_opt,
        processing_class=tok,
        reward_processing_classes=tok,
        train_dataset=prompt_ds,
    )
    assert isinstance(minillm.reward_funcs[0], EasyDeLState)
    assert minillm.reward_funcs[0].tx is None
    assert minillm.reward_funcs[0].opt_state is None

    sdft = SDFTTrainer(
        arguments=SDFTConfig(
            max_prompt_length=4,
            max_completion_length=4,
            total_batch_size=2,
            num_train_epochs=1,
            use_wandb=False,
        ),
        model=policy_model,
        reward_funcs=rm_state_with_opt,
        processing_class=tok,
        reward_processing_classes=tok,
        train_dataset=prompt_ds,
    )
    assert isinstance(sdft.reward_funcs[0], EasyDeLState)
    assert sdft.reward_funcs[0].tx is None
    assert sdft.reward_funcs[0].opt_state is None


def test_leak_22_precompute_ref_log_probs_releases_unused_reference_state(monkeypatch):
    """#22: DPO, BCO, and KTO (not calculate_kl) clear self.reference_state after precomputing all active splits."""
    monkeypatch.setattr(Trainer, "configure_dataloaders", lambda self: "configured")

    # 1. DPO: releases reference_state when precomputed & sync_ref_model=False & no BEMA ref update.
    dpo = object.__new__(DPOTrainer)
    dpo.arguments = DPOConfig(precompute_ref_log_probs=True, sync_ref_model=False)
    dpo.dataset_train = object()
    dpo.dataset_eval = object()
    dpo._train_source = object()
    dpo._eval_source = object()
    dpo._precomputed_train_ref_log_probs = False
    dpo._precomputed_eval_ref_log_probs = False
    dpo.reference_state = object()
    dpo._eval_reference_graphdef = object()
    monkeypatch.setattr(DPOTrainer, "training_batch_size", property(lambda self: 2))
    monkeypatch.setattr(DPOTrainer, "evaluation_batch_size", property(lambda self: 2))
    monkeypatch.setattr(DPOTrainer, "_source_has_reference_logps", staticmethod(lambda src: False))
    monkeypatch.setattr(DPOTrainer, "_precompute_reference_log_probs_for_split", lambda self, **kw: True)
    assert DPOTrainer.configure_dataloaders(dpo) == "configured"
    assert dpo.reference_state is None
    assert dpo._eval_reference_graphdef is None

    # DPO with sync_ref_model=True keeps reference_state.
    dpo_sync = object.__new__(DPOTrainer)
    dpo_sync.arguments = DPOConfig(precompute_ref_log_probs=True, sync_ref_model=True)
    dpo_sync.dataset_train = object()
    dpo_sync.dataset_eval = None
    dpo_sync._train_source = object()
    dpo_sync._precomputed_train_ref_log_probs = False
    dpo_sync._precomputed_eval_ref_log_probs = False
    ref_keep = object()
    dpo_sync.reference_state = ref_keep
    assert DPOTrainer.configure_dataloaders(dpo_sync) == "configured"
    assert dpo_sync.reference_state is ref_keep

    # DPO subclass with BEMA update_ref_model=True keeps reference_state.
    dpo_bema = object.__new__(DPOTrainer)
    dpo_bema.arguments = DPOConfig(precompute_ref_log_probs=True, sync_ref_model=False)
    dpo_bema.bema_callback = SimpleNamespace(update_ref_model=True)
    dpo_bema.dataset_train = object()
    dpo_bema.dataset_eval = None
    dpo_bema._train_source = object()
    dpo_bema._precomputed_train_ref_log_probs = False
    dpo_bema._precomputed_eval_ref_log_probs = False
    dpo_bema.reference_state = ref_keep
    assert DPOTrainer.configure_dataloaders(dpo_bema) == "configured"
    assert dpo_bema.reference_state is ref_keep

    # 2. BCO: releases reference_state when all active splits precompute; keeps if a split fails.
    monkeypatch.setattr(BCOTrainer, "training_batch_size", property(lambda self: 2))
    monkeypatch.setattr(BCOTrainer, "evaluation_batch_size", property(lambda self: 2))
    monkeypatch.setattr(BCOTrainer, "_source_has_reference_logps", staticmethod(lambda src: False))
    monkeypatch.setattr(BCOTrainer, "_precompute_reference_log_probs_for_split", lambda self, **kw: True)
    bco = object.__new__(BCOTrainer)
    bco.arguments = BCOConfig(precompute_ref_log_probs=True)
    bco.dataset_train = object()
    bco.dataset_eval = None
    bco._train_source = object()
    bco._precomputed_train_ref_log_probs = False
    bco._precomputed_eval_ref_log_probs = False
    bco.reference_state = object()
    bco._eval_reference_graphdef = object()
    assert BCOTrainer.configure_dataloaders(bco) == "configured"
    assert bco.reference_state is None
    assert bco._eval_reference_graphdef is None

    monkeypatch.setattr(
        BCOTrainer,
        "_precompute_reference_log_probs_for_split",
        lambda self, **kw: bool(kw["is_train"]),
    )
    bco_partial = object.__new__(BCOTrainer)
    bco_partial.arguments = BCOConfig(precompute_ref_log_probs=True)
    bco_partial.dataset_train = object()
    bco_partial.dataset_eval = object()
    bco_partial._train_source = object()
    bco_partial._eval_source = object()
    bco_partial._precomputed_train_ref_log_probs = False
    bco_partial._precomputed_eval_ref_log_probs = False
    bco_partial.reference_state = ref_keep
    assert BCOTrainer.configure_dataloaders(bco_partial) == "configured"
    assert bco_partial.reference_state is ref_keep

    # 3. KTO: releases reference_state when calculate_kl=False; keeps when calculate_kl=True.
    monkeypatch.setattr(KTOTrainer, "training_batch_size", property(lambda self: 2))
    monkeypatch.setattr(KTOTrainer, "evaluation_batch_size", property(lambda self: 2))
    monkeypatch.setattr(KTOTrainer, "_source_has_reference_logps", staticmethod(lambda src: False))
    monkeypatch.setattr(KTOTrainer, "_precompute_reference_log_probs_for_split", lambda self, **kw: True)
    kto_no_kl = object.__new__(KTOTrainer)
    kto_no_kl.arguments = KTOConfig(precompute_ref_log_probs=True, loss_type="apo_zero_unpaired")
    kto_no_kl.calculate_kl = False
    kto_no_kl.dataset_train = object()
    kto_no_kl.dataset_eval = None
    kto_no_kl._train_source = object()
    kto_no_kl.reference_state = object()
    kto_no_kl._eval_reference_graphdef = object()
    assert KTOTrainer.configure_dataloaders(kto_no_kl) == "configured"
    assert kto_no_kl.reference_state is None
    assert kto_no_kl._eval_reference_graphdef is None

    kto_kl = object.__new__(KTOTrainer)
    kto_kl.arguments = KTOConfig(precompute_ref_log_probs=True, loss_type="kto")
    kto_kl.calculate_kl = True
    kto_kl.dataset_train = object()
    kto_kl.dataset_eval = None
    kto_kl._train_source = object()
    kto_kl.reference_state = ref_keep
    assert KTOTrainer.configure_dataloaders(kto_kl) == "configured"
    assert kto_kl.reference_state is ref_keep


def test_leak_23_explicit_ref_teacher_shares_compatible_esurge_scope_and_releases_on_teardown():
    """#23: Compatible explicit ref/teacher shares policy esurge_cache_scope_key; incompatible releases on teardown."""
    tok = _word_tokenizer(["a", "b", "c", "d"])
    prompt_ds = Dataset.from_dict({"prompt": ["a", "b"]})
    text_ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [2, 3, 4, 1]], "attention_mask": [[1, 1, 1, 1]] * 2})

    _, policy_model = _make_tiny_llama(seed=51, vocab_size=16, hidden_size=16, num_layers=1)
    _, same_aux_model = _make_tiny_llama(seed=52, vocab_size=16, hidden_size=16, num_layers=1)
    _, diff_aux_model = _make_tiny_llama(seed=53, vocab_size=16, hidden_size=32, num_layers=1)

    def _zero_reward(prompts, completions, **kwargs):
        del prompts, kwargs
        return [0.0] * len(completions)

    # Compatible explicit reference/teacher models share policy esurge_cache_scope_key.
    xpo_same = XPOTrainer(
        arguments=XPOConfig(max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False),
        model=policy_model,
        reference_model=same_aux_model,
        reward_funcs=_zero_reward,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert xpo_same.ref_state.esurge_cache_scope_key == xpo_same.model_state.esurge_cache_scope_key

    nash_same = NashMDTrainer(
        arguments=NashMDConfig(max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False),
        model=policy_model,
        reference_model=same_aux_model,
        reward_funcs=_zero_reward,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert nash_same.ref_state.esurge_cache_scope_key == nash_same.model_state.esurge_cache_scope_key

    gkd_same = GKDTrainer(
        arguments=GKDConfig(max_length=8, total_batch_size=2, num_train_epochs=1, use_wandb=False),
        model=policy_model,
        teacher_model=same_aux_model,
        processing_class=tok,
        train_dataset=text_ds,
    )
    assert gkd_same.teacher_state.esurge_cache_scope_key == gkd_same.model_state.esurge_cache_scope_key

    opd_same = OnPolicyDistillationTrainer(
        arguments=OnPolicyDistillationConfig(
            max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False
        ),
        student_model=policy_model,
        teacher_model=same_aux_model,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert opd_same.teacher_state.esurge_cache_scope_key == opd_same.model_state.esurge_cache_scope_key

    seqkd_same = SeqKDTrainer(
        arguments=SeqKDConfig(
            max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False
        ),
        student_model=policy_model,
        teacher_model=same_aux_model,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert seqkd_same.teacher_state.esurge_cache_scope_key == seqkd_same.model_state.esurge_cache_scope_key

    sparse_same = SparseDistillationTrainer(
        arguments=SparseDistillationConfig(
            max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False
        ),
        student_model=policy_model,
        teacher_model=same_aux_model,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert sparse_same.teacher_state.esurge_cache_scope_key == sparse_same.model_state.esurge_cache_scope_key

    # Incompatible teacher model keeps a DISTINCT esurge_cache_scope_key AND is released on teardown!
    opd_diff = OnPolicyDistillationTrainer(
        arguments=OnPolicyDistillationConfig(
            max_prompt_length=4, max_completion_length=4, total_batch_size=2, num_train_epochs=1, use_wandb=False
        ),
        student_model=policy_model,
        teacher_model=diff_aux_model,
        processing_class=tok,
        train_dataset=prompt_ds,
    )
    assert opd_diff.teacher_state.esurge_cache_scope_key != opd_diff.model_state.esurge_cache_scope_key

    paused_scopes: list[tuple[str, bool, bool]] = []
    orig_pause = LlamaForCausalLM.pause_esurge

    def _spy_pause(self_mod, engine_id=None, *, release_model_state=False, clear_compiled_cache=False):
        del engine_id
        paused_scopes.append((self_mod._esurge_cache_scope(), bool(release_model_state), bool(clear_compiled_cache)))

    LlamaForCausalLM.pause_esurge = _spy_pause  # type: ignore[method-assign]
    try:
        opd_diff.release_generation_runtime(clear_esurge_compiled_cache=True)
    finally:
        LlamaForCausalLM.pause_esurge = orig_pause  # type: ignore[method-assign]

    assert paused_scopes == [
        (opd_diff.model_state.esurge_cache_scope_key, True, True),
        (opd_diff.teacher_state.esurge_cache_scope_key, True, True),
    ]


def test_leak_24_bco_scheduled_loss_applies_straight_through_emulator_once():
    """#24: BCO scheduled MPMD loss must apply straight_through_emulator once, not twice."""
    _, model = _make_tiny_llama(seed=61, vocab_size=16, hidden_size=16, num_layers=1)
    state = EasyDeLState.create(model=model, tx=optax.sgd(1e-3), init_opt_state=True)
    ste_calls = {"count": 0}

    def _counting_ste(tree):
        ste_calls["count"] += 1
        return tree

    def _stub_forward(module, batch, vocab_shard_stage=None):
        del module, batch, vocab_shard_stage
        return {
            "completion_logps": jnp.array([-1.0, -2.0], dtype=jnp.float32),
            "mean_completion_logits": jnp.array(0.0, dtype=jnp.float32),
        }

    batch = {
        "label": jnp.array([True, False]),
        "reference_logps": jnp.array([-1.2, -1.8], dtype=jnp.float32),
        "running_mean": jnp.array(0.0, dtype=jnp.float32),
    }
    call = ScheduledStepCall(
        step_fn=bco_fn.training_step,
        state=state,
        batch=batch,
        args=(),
        kwargs={},
        bound_arguments={
            "concatenated_forward_fn": _stub_forward,
            "beta": 0.1,
            "partition_spec": PartitionSpec(),
            "gradient_accumulation_steps": 1,
            "straight_through_emulator": _counting_ste,
        },
        schedule=None,
    )
    adapter = get_scheduled_loss_adapter(bco_fn.training_step)
    assert adapter is not None
    loss_fn = adapter.make_loss(call)
    loss, aux = loss_fn(state.graphstate, batch)
    assert jnp.isfinite(loss)
    assert "reward_moments/sum" in aux
    assert ste_calls["count"] == 1


def test_leak_31_bco_density_ratio_classifier_bounds_majority_class_ram():
    """#31: BCOTrainer._train_density_ratio_classifier bounds majority-class storage via reservoir sampling."""
    trainer = object.__new__(BCOTrainer)
    trainer.arguments = BCOConfig(prompt_sample_size=8)
    trainer._rng = np.random.default_rng(0)
    trainer.clf_weights = None

    class _ImbalancedSource:
        shard_names = ("shard-0",)

        def open_shard(self, shard_name):
            del shard_name
            # 5,000 desirable rows followed by 5 undesirable rows (worst-case sorted imbalance).
            for i in range(5000):
                yield {
                    "embedding_input_ids": [1, (i % 7) + 2],
                    "embedding_attention_mask": [1, 1],
                    "label": True,
                }
            for i in range(5):
                yield {
                    "embedding_input_ids": [9, (i % 7) + 10],
                    "embedding_attention_mask": [1, 1],
                    "label": False,
                }

    vectorize_calls = {"count": 0}
    fit_shapes: list[tuple[int, ...]] = []

    def _fake_vectorize(input_ids, attention_mask):
        del attention_mask
        vectorize_calls["count"] += 1
        arr = np.asarray(input_ids, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    orig_fit = BCOTrainer._fit_logistic_regression

    def _spy_fit(self_t, embeddings, labels, **kwargs):
        fit_shapes.append(embeddings.shape)
        return orig_fit(self_t, embeddings, labels, **kwargs)

    trainer._train_source = _ImbalancedSource()
    trainer._vectorize_prompt = _fake_vectorize
    trainer._fit_logistic_regression = _spy_fit.__get__(trainer, BCOTrainer)

    trainer._train_density_ratio_classifier()

    assert trainer.clf_weights is not None
    # Reservoir cap: at most 8 desirable + 5 undesirable = 13 stacked embeddings (never 5,005!).
    assert fit_shapes == [(13, 2)]
    # Reservoir sampling skips vectorizing rejected majority draws (O(k log N) << 5,005).
    assert vectorize_calls["count"] < 250


def test_leak_32_xpo_eval_rollout_uses_eval_ref_graphdef():
    """#32: XPOTrainer._preprocess_batch_input wires _eval_ref_graphdef into eval rollouts (and OPD eval teacher)."""
    sentinel_train_gd = object()
    sentinel_eval_gd = object()

    @dataclass
    class _MiniRefState:
        graphdef: object

        def replace(self, **changes):
            return replace(self, **changes)

    trainer = object.__new__(XPOTrainer)
    trainer.arguments = SimpleNamespace(max_length=16, missing_eos_penalty=None, num_train_epochs=1)
    trainer.max_training_steps = 1
    trainer._current_step = 0
    trainer._beta_schedule = 0.2
    trainer._alpha_schedule = 0.1
    trainer.loss_type_id = 0
    trainer.ref_state = _MiniRefState(graphdef=sentinel_train_gd)
    trainer._eval_ref_graphdef = sentinel_eval_gd
    trainer.reward_funcs = [lambda prompts, completions, **kw: [1.0] * len(completions)]
    trainer.reward_processing_classes = [None]
    trainer.processing_class = _TinyTokenizer()
    trainer._apply_user_data_collator = lambda b: b
    trainer._purify_batch = lambda b: b
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None

    seen_states: list[object] = []

    def _fake_rollout(*, input_ids, attention_mask, state, **kwargs):
        del kwargs
        seen_states.append(state)
        comp = jnp.array([[2, 3]], dtype=jnp.int32)
        cmask = jnp.array([[1, 1]], dtype=jnp.int32)
        return GenerationResults(
            generation_results=["ok"],
            prompt_ids=input_ids,
            prompt_mask=attention_mask,
            sequences=jnp.concatenate([input_ids, comp], axis=1),
            completion_ids=comp,
            completion_mask=cmask,
            decoded_prompts=["p"],
            completion_prompts=["p"],
            text=["ok"],
            reasoning=[None],
            tool_calls=[None],
            raw_text=["ok"],
        )

    trainer.rollout = _fake_rollout
    policy_state = SimpleNamespace(step=jnp.asarray(0, dtype=jnp.int32))
    batch = {"input_ids": jnp.array([[1, 2]], dtype=jnp.int32), "attention_mask": jnp.array([[1, 1]], dtype=jnp.int32)}

    # Eval rollout must pass ref_state with _eval_ref_graphdef.
    trainer._preprocess_batch_input(policy_state, batch, is_train=False)
    assert len(seen_states) == 2
    assert seen_states[1].graphdef is sentinel_eval_gd

    # Train rollout must keep train graphdef.
    seen_states.clear()
    trainer._preprocess_batch_input(policy_state, batch, is_train=True)
    assert len(seen_states) == 2
    assert seen_states[1].graphdef is sentinel_train_gd

    # Also verify OnPolicyDistillationTrainer uses _eval_teacher_graphdef during eval teacher rollouts.
    opd = object.__new__(OnPolicyDistillationTrainer)
    opd.arguments = SimpleNamespace(generate_with_teacher=True)
    opd.teacher_state = _MiniRefState(graphdef=sentinel_train_gd)
    opd._eval_teacher_graphdef = sentinel_eval_gd
    opd._apply_user_data_collator = lambda b: b
    opd._purify_batch = lambda b: b
    opd._make_attn_mask = lambda ids: jnp.ones_like(ids, dtype=jnp.int32)
    opd._all_gather = lambda x: x
    opd._log_training_generations_to_wandb = lambda **kw: None
    opd_seen: list[object] = []

    def _opd_rollout(*, input_ids, attention_mask, state, **kwargs):
        del kwargs
        opd_seen.append(state)
        comp = jnp.array([[2, 3]], dtype=jnp.int32)
        cmask = jnp.array([[1, 1]], dtype=jnp.int32)
        return GenerationResults(
            generation_results=["ok"],
            prompt_ids=input_ids,
            prompt_mask=attention_mask,
            sequences=jnp.concatenate([input_ids, comp], axis=1),
            completion_ids=comp,
            completion_mask=cmask,
            decoded_prompts=["p"],
            completion_prompts=["p"],
            text=["ok"],
            reasoning=[None],
            tool_calls=[None],
            raw_text=["ok"],
        )

    opd.rollout = _opd_rollout
    opd._preprocess_batch_input(policy_state, batch, is_train=False)
    assert opd_seen[-1].graphdef is sentinel_eval_gd
    opd._preprocess_batch_input(policy_state, batch, is_train=True)
    assert opd_seen[-1].graphdef is sentinel_train_gd
