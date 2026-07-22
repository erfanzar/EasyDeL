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

"""MTP-drafter self-distillation: train ONLY the built-in MTP head for deep speculative drafts.

Improves speculative-decoding acceptance at large draft depths (``--num-spec 8``
in eSurge) by distilling the model's own next-token distributions into its
Multi-Token-Prediction head, recursively applied ``mtp_draft_tokens`` steps
ahead (FastMTP-style teacher-forced chain). This is *self*-distillation:

    * teacher  = the same frozen checkpoint (no MTP head needed — its ordinary
      next-token logits shifted by ``k`` are exactly the step-``k`` target);
    * student  = the same checkpoint with ONLY the ``mtp.*`` parameters
      trainable (``trainable_selector`` selects parameters whose path contains
      ``"mtp"``); the trunk, embeddings, and LM head stay bitwise frozen, so
      the main model's outputs are untouched — only draft quality changes.

The loss seen by the MTP head is
``mtp_kd_weight * mean_k KL(teacher_{t+k} || draft_k) + mtp_loss_coef * MTP-CE``
(the self-supervised depth-1 CE rides along via the model's ``aux_loss``).
The main-token KD term is identically zero here (teacher == student trunk)
and contributes no gradient.

Acceptance is measured by ``scripts/bench_spec_ab.py`` (tok/step at K=8), not
by the loss curve.

Usage:
    python examples/post_training/mtp_drafter_distill.py \
        --model /dev/shm/easydel_ckpts/qwen36_27b_spx \
        --tokenizer Qwen/qwen3.6-27b \
        --output_dir /dev/shm/easydel_ckpts/mtp_kd_run

    # smoke: a few steps end-to-end including the final checkpoint save
    python examples/post_training/mtp_drafter_distill.py --max_steps 12 \
        --output_dir /dev/shm/easydel_ckpts/mtp_kd_smoke
"""

from __future__ import annotations

import os

# Route compile/kernel caches to /dev/shm (root disk is tight; repeated TPU
# compiles otherwise fill ~/.cache and ENOSPC the host).
os.environ.setdefault("EJKERNEL_PERSISTENT_CACHE_DIR", "/dev/shm/ejkernel-cache")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax-comp-cache")
os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from dataclasses import dataclass, field

import easydel as ed
import spectrax as spx
from datasets import load_dataset
from easydel.infra.elarge import eLargeModel
from eformer.aparser import DataClassArgumentParser


@dataclass
class MTPDrafterArgs:
    """Command-line arguments for MTP-drafter self-distillation."""

    model: str = field(
        default="/dev/shm/easydel_ckpts/qwen36_27b_spx",
        metadata={"help": "Checkpoint that carries the MTP head (student AND teacher; self-distillation)."},
    )
    tokenizer: str = field(
        default="Qwen/qwen3.6-27b",
        metadata={"help": "Tokenizer repo id or path matching the checkpoint vocab."},
    )
    dataset: str = field(
        default="trl-lib/ultrafeedback_binarized",
        metadata={"help": "Chat dataset; the 'chosen' messages are chat-templated as training text."},
    )
    output_dir: str = field(
        default="/dev/shm/easydel_ckpts/mtp_kd_run",
        metadata={"help": "Checkpoint directory (final save only by default; watch /dev/shm space)."},
    )
    max_steps: int = field(default=3000, metadata={"help": "Optimizer steps."})
    batch_size: int = field(default=16, metadata={"help": "Total batch size across devices."})
    max_length: int = field(default=1024, metadata={"help": "Max sequence length (chat rows padded/truncated)."})
    learning_rate: float = field(default=5e-5, metadata={"help": "LR for the MTP-head parameters."})
    warmup_steps: int = field(default=100, metadata={"help": "LR warmup steps."})
    mtp_draft_tokens: int = field(
        default=8,
        metadata={"help": "Recursive draft depth K distilled through the head (match serving --num-spec)."},
    )
    mtp_kd_weight: float = field(default=1.0, metadata={"help": "Weight on the mean-over-depth soft KD term."})
    temperature: float = field(
        default=1.0,
        metadata={"help": "KD softmax temperature. 1.0 matches the greedy/verify distribution directly."},
    )
    tp: int = field(default=4, metadata={"help": "Tensor parallel degree (4 = the serving layout of this ckpt)."})
    save_steps: int | None = field(
        default=None,
        metadata={"help": "Periodic save interval. None = save only at the end (one ~41GB full-model write)."},
    )
    wandb: bool = field(default=False, metadata={"help": "Enable Weights & Biases logging."})


def format_chat(example):
    """Extract the chosen chat messages from ultrafeedback_binarized."""
    return {"messages": example["chosen"]}


def main():
    parser = DataClassArgumentParser(MTPDrafterArgs, description="MTP-drafter self-distillation with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    # Train exactly the MTP head: parameters whose canonical path contains "mtp"
    # (mtp.fc / mtp.layers.0.* / mtp.norm / mtp.pre_fc_norm_*). Everything else
    # (trunk, embeddings, shared LM head) lands in graphother: no optimizer
    # state, no gradients, bitwise identical in the exported checkpoint.
    mtp_only = spx.as_selector("parameters") & spx.path_contains("mtp")

    elm = eLargeModel(
        {
            "model": {
                "name_or_path": args.model,
                "tokenizer": args.tokenizer,
                "task": "auto-bind",
            },
            # Self-distillation: the frozen teacher is the same checkpoint. Its
            # next-token logits at t+k are the step-k targets for the draft chain.
            "teacher_model": {
                "name_or_path": args.model,
                "tokenizer": args.tokenizer,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
            },
            "sharding": {
                "axis_dims": (1, 1, -1, 1, args.tp, 1),
                "axis_names": ("pp", "dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            # Training-runtime attention override only (the checkpoint bakes the
            # eSurge decode mechanism ragged_page_attention_v3, which is not a
            # training path). Deliberately NO gradient_checkpointing override:
            # the trunk is frozen (no backward -> no residuals to save) and a
            # remat-wrapped MTP layer in the SAVED config breaks the serving
            # drafter (jax.checkpoint rejects its positional string `mode` arg).
            # These runtime values are restored to the base checkpoint's before
            # the final save so the artifact serves exactly like the base.
            "base_config": {
                "values": {
                    "attn_mechanism": ed.AttentionMechanisms.AUTO,
                    "attn_dtype": "bf16",
                }
            },
            "trainer": {
                "trainer_type": "distillation",
                "save_directory": args.output_dir,
                "num_train_epochs": 1,
                "max_training_steps": args.max_steps,
                "total_batch_size": args.batch_size,
                "gradient_accumulation_steps": 1,
                "learning_rate": args.learning_rate,
                "scheduler": "cosine",
                "warmup_steps": args.warmup_steps,
                "clip_grad": 1.0,
                "max_length": args.max_length,
                "trainable_selector": mtp_only,
                # Pure soft-KD main term (identically zero for self-distillation;
                # the trained signal is the MTP chain KD + the model's own MTP CE).
                "alpha": 1.0,
                "temperature": args.temperature,
                "mtp_distillation": True,
                "mtp_kd_weight": args.mtp_kd_weight,
                "mtp_draft_tokens": args.mtp_draft_tokens,
                # Save policy: the full model is ~41 GB on /dev/shm — exactly one
                # save, done manually after training so the artifact's config is
                # restored to the base checkpoint's serving values first
                # (do_last_save would snapshot the training-runtime config).
                "save_steps": args.save_steps,
                "save_total_limit": 1,
                "save_optimizer_state": False,
                "do_last_save": False,
                "log_steps": 1,
                "report_steps": 10,
                "progress_bar_type": "json",
                "use_wandb": args.wandb,
            },
        }
    )

    train_dataset = (
        load_dataset(args.dataset, split="train")
        .shuffle(seed=42)
        .map(format_chat, remove_columns=["chosen", "rejected", "score_chosen", "score_rejected"])
    )

    # fp32 master weights for the trainable head: bf16 params with a small LR
    # round sub-ulp Adam updates to zero (ulp(1.0)=7.8e-3 on the norm scales,
    # ulp(0.02)=1.5e-4 on projections, vs lr=5e-5 deltas). Linear/norm layers
    # cast params to the bf16 compute dtype at call time, so this changes only
    # the optimizer accumulation; the final save casts back to param_dtype bf16.
    import jax.numpy as jnp

    student = elm.build_model()
    n_upcast = 0
    for _path, var in spx.iter_variables(student, select=mtp_only):
        var.value = var.value.astype(jnp.float32)
        n_upcast += 1
    print(f"[mtp-drafter-distill] upcast {n_upcast} mtp param leaves to float32 (master weights)", flush=True)

    trainer = elm.build_trainer(train_dataset=train_dataset)

    # Hard guard: the frozen-base contract. If the selector regresses, the run
    # would silently train 27B params (and OOM the optimizer) — fail fast instead.
    trainable_paths = list(trainer.model_state.graphstate.flatten().keys())
    non_mtp = [p for p in trainable_paths if "mtp" not in p]
    if not trainable_paths or non_mtp:
        raise RuntimeError(
            f"trainable_selector mismatch: {len(trainable_paths)} trainable leaves, "
            f"non-mtp={non_mtp[:5]} — expected only mtp.* parameters."
        )
    print(f"[mtp-drafter-distill] trainable leaves: {len(trainable_paths)} (all mtp.*)", flush=True)

    output = trainer.train()

    # Restore the training-runtime attention overrides to the BASE checkpoint's
    # serving values before the one final save. The saved config.json must be a
    # drop-in for eSurge: `attn_mechanism=auto` and a remat policy would change
    # (or break — see gradient_checkpointing note above) the serving runtime.
    import json

    base_cfg = json.loads((__import__("pathlib").Path(args.model) / "config.json").read_text())
    state = output.state
    pairs = [(state.model.config, base_cfg)]
    if getattr(state.model.config, "text_config", None) is not None and isinstance(base_cfg.get("text_config"), dict):
        pairs.append((state.model.config.text_config, base_cfg["text_config"]))
    for cfg, src in pairs:
        for key in ("attn_mechanism", "attn_dtype", "gradient_checkpointing"):
            if key in src:
                setattr(cfg, key, src[key])
    saved_dir = trainer._save_state(state)
    print(f"[mtp-drafter-distill] final checkpoint: {saved_dir}", flush=True)


if __name__ == "__main__":
    main()
