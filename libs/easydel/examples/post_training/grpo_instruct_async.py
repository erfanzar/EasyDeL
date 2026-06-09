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

"""Async-GRPO instruction-following RL with EasyDeL eLarge.

Reinforcement-learns instruction following on a prompt-only dataset using the
**Async** GRPO trainer (``trainer_type="async_grpo"``): rollout generation runs
in a background eSurge worker one step ahead of the optimizer, so the
accelerators never idle waiting on sampling.

Reward signal — fully self-contained, no second model:
    Two ``RewardProtocol`` verifiers score each completion per example.
      - ``InstructionConstraintReward``: parses *verifiable* constraints stated
        in the prompt (word/sentence/paragraph counts, JSON, casing, bullet
        lists, ending phrase, keyword inclusion, placeholders, highlights, ...)
        and returns the fraction satisfied — IFEval-style, but read straight
        from the instruction text so it needs no dataset constraint metadata.
        Returns ``None`` (neutral) when a prompt has no checkable constraint.
      - ``ResponseQualityReward``: a light regularizer that penalizes empty or
        degenerate (repetitive) generations.
    GRPO turns the per-completion rewards into group-relative advantages.

Model:
    Loaded directly from the GCS checkpoint ``gs://uscentral2stuff/
    qwen3.6-27b-bf16`` via tensorstore (no gcsfuse needed). EasyDeL reads the
    checkpoint's ``config.json`` ``model_type`` and binds the matching
    registered architecture — the checkpoint must declare a model_type EasyDeL
    knows (e.g. ``qwen3_5`` / ``qwen3_moe``); there is no ``qwen3_6`` module.

    Tokenizer: weights and tokenizer load separately.  HF ``AutoTokenizer``
    cannot read ``gs://`` paths, and EasyDeL checkpoints are often saved without
    tokenizer files, so pass ``--tokenizer`` pointing at an HF repo id (or a
    local / ``gs://`` dir that contains ``tokenizer.json``) whose vocab matches
    the checkpoint.

Sharding:
    6-axis mesh ``(pp, dp, fsdp, ep, tp, sp)``. Pure FSDP by default
    (``fsdp=-1`` auto-fills the device count); eSurge generation uses the
    ``fsdp`` axis as its data-parallel axis. Bump ``--tp`` only if a single host
    cannot hold a 27B bf16 shard.

Usage:
    python examples/post_training/grpo_instruct_async.py --tokenizer <hf-repo-or-dir>
    python examples/post_training/grpo_instruct_async.py --dataset allenai/RLVR-IFeval --tokenizer <hf-repo>
    python examples/post_training/grpo_instruct_async.py --tp 4 --num_generations 16 --tokenizer <hf-repo>
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser

import easydel as ed
from easydel import RewardProtocol
from easydel.infra.elarge import eLargeModel

# Reduction used by every RewardProtocol below. "mean" reproduces the classic
# GRPO group baseline (advantage = reward - group mean); all rewards passed
# together must agree on this value.
_REDUCTION = "mean"


@dataclass
class GRPOInstructArgs:
    """Command-line arguments for async-GRPO instruction-following training."""

    model: str = field(
        default="gs://uscentral2stuff/qwen3.6-27b-bf16",
        metadata={"help": "Policy model: GCS checkpoint, local dir, or HF repo id."},
    )
    tokenizer: str | None = field(
        default=None,
        metadata={
            "help": "Tokenizer source: HF repo id, local dir, or gs:// dir with tokenizer files. "
            "Defaults to --model (only works if that path actually has tokenizer files / is an HF repo). "
            "EasyDeL checkpoints saved without a tokenizer need this set explicitly."
        },
    )
    dataset: str = field(
        default="allenai/RLVR-IFeval",
        metadata={
            "help": "Instruction dataset (prompt-only). Should contain prompts that state verifiable constraints."
        },
    )
    dataset_split: str = field(default="train", metadata={"help": "Dataset split."})
    output_dir: str = field(default="outputs/grpo-instruct-async", metadata={"help": "Checkpoint directory."})
    max_steps: int = field(default=500, metadata={"help": "Maximum optimizer steps."})
    batch_size: int = field(default=8, metadata={"help": "Total prompt batch size across devices."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    num_generations: int = field(default=8, metadata={"help": "Completions sampled per prompt (GRPO group size)."})
    max_length: int = field(default=2048, metadata={"help": "Max total sequence length (prompt + completion)."})
    max_prompt_length: int = field(default=1024, metadata={"help": "Max prompt length in tokens."})
    tp: int = field(
        default=1,
        metadata={
            "help": "Tensor-parallel degree (keep 1 for pure FSDP; raise for a 27B host that can't hold the shard)."
        },
    )
    esurge_hbm_utilization: float | None = field(
        default=None,
        metadata={
            "help": "Fraction of HBM eSurge may use for its KV cache (e.g. 0.2 when co-locating a large model on few chips). "
            "Left None by default: with both this and --esurge_max_cache_tokens unset, the trainer auto-computes the "
            "KV-cache token budget as max_length x batch_size x num_generations instead of a blanket HBM fraction."
        },
    )
    esurge_page_size: int = field(
        default=64,
        metadata={
            "help": "eSurge KV page size. Lower it (e.g. 16/32) for large-head_dim models whose paged-attention kernel overflows the 16MB TPU VMEM."
        },
    )
    esurge_max_num_seqs: int = field(
        default=16,
        metadata={
            "help": "Max concurrent sequences (cache slots) in eSurge. Lower it (e.g. 8) for hybrid/linear-attention models whose recurrent-state kernel tile (slots x heads x d x d) overflows TPU VMEM."
        },
    )
    esurge_max_cache_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Hard ceiling on total eSurge KV-cache tokens. KV pages are sized to min(hbm_utilization-derived, this), so a generous hbm_utilization can't build (e.g.) a 1M-token cache when you only want 64k for the whole run. None keeps pure hbm_utilization sizing."
        },
    )
    request_timeout: float = field(
        default=1800.0,
        metadata={
            "help": "Seconds the async-GRPO loop waits on a rollout future. The FIRST rollout pays one-time eSurge compilation for a large model; raise this (e.g. 3600) so it doesn't time out before kernels are cached."
        },
    )
    learning_rate: float = field(default=1e-6, metadata={"help": "Optimizer learning rate."})
    wandb: bool = field(default=True, metadata={"help": "Enable Weights & Biases logging."})
    wandb_entity: str | None = field(default=None, metadata={"help": "W&B entity (team or username)."})


def _to_text(value: object) -> str:
    """Flatten a completion/prompt (str or chat message list) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in value)
    return str(value)


def _instruction_text(prompt: object) -> str:
    """Return the user-facing instruction text from a prompt (messages or str)."""
    if isinstance(prompt, list):
        users = [m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
        return "\n".join(users) if users else _to_text(prompt)
    return _to_text(prompt)


def _is_json(text: str) -> bool:
    """True if `text` (optionally fenced in ```...```) parses as JSON."""
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        json.loads(stripped)
        return True
    except (ValueError, TypeError):
        return False


def evaluate_constraints(instruction: str, response: str) -> list[bool]:
    """Detect verifiable constraints in `instruction` and check `response`.

    Each constraint that is *detected* in the instruction contributes one
    satisfied/violated boolean. Constraints that are not mentioned are not
    scored, so the result list is empty when the prompt has nothing checkable.
    """
    instr = instruction.lower()
    results: list[bool] = []

    def check(detected: object, satisfied: object) -> None:
        if detected:
            results.append(bool(satisfied))

    words = response.split()
    n_words = len(words)
    n_sent = len([s for s in re.split(r"[.!?]+", response) if s.strip()])
    n_para = len([p for p in re.split(r"\n\s*\n", response.strip()) if p.strip()])
    has_alpha = any(c.isalpha() for c in response)

    m = re.search(r"at least (\d+)\s+words", instr)
    check(m, m and n_words >= int(m.group(1)))
    m = re.search(r"(?:no more than|at most|less than|fewer than)\s+(\d+)\s+words", instr)
    check(m, m and n_words <= int(m.group(1)))
    m = re.search(r"exactly (\d+)\s+words", instr)
    check(m, m and n_words == int(m.group(1)))
    m = re.search(r"at least (\d+)\s+sentences", instr)
    check(m, m and n_sent >= int(m.group(1)))
    m = re.search(r"exactly (\d+)\s+sentences", instr)
    check(m, m and n_sent == int(m.group(1)))
    m = re.search(r"(\d+)\s+paragraphs", instr)
    check(m, m and n_para == int(m.group(1)))

    if re.search(r"\bjson\b", instr) and re.search(r"format|valid|output|respond|wrap|object", instr):
        results.append(_is_json(response))

    if re.search(r"all lowercase|in lowercase|lowercase letters only", instr):
        results.append(has_alpha and response == response.lower())
    if re.search(r"all uppercase|all capital letters|in all caps|capital letters only", instr):
        results.append(has_alpha and response == response.upper())

    if re.search(r"(?:no|without|refrain from using|avoid)\s+(?:any\s+)?commas", instr):
        results.append("," not in response)

    bullets = [ln for ln in response.splitlines() if re.match(r"\s*[-*•]\s+", ln)]
    m = re.search(r"(\d+)\s+bullet points", instr)
    if m:
        check(m, len(bullets) == int(m.group(1)))
    elif re.search(r"bullet points|bulleted list", instr):
        results.append(len(bullets) >= 1)
    if re.search(r"numbered list", instr):
        numbered = [ln for ln in response.splitlines() if re.match(r"\s*\d+[.)]\s+", ln)]
        results.append(len(numbered) >= 2)

    m = re.search(r"(?:end|finish)[^\"'“”‘’]*?(?:phrase|words?|sentence)[:\s]*[\"'“‘](.+?)[\"'”’]", instr)
    if m:
        results.append(response.strip().lower().endswith(m.group(1).strip().lower()))

    m = re.search(r"include the (?:word|keyword|phrase|term)s?\s+[\"'“‘]?([a-z0-9 \-]+?)[\"'”’.]?(?:\s|$)", instr)
    if m:
        results.append(m.group(1).strip().lower() in response.lower())

    if re.search(r"double angular brackets|wrap.*title|title.*<<", instr) or "<<" in instruction:
        results.append(bool(re.search(r"<<.+?>>", response)))

    m = re.search(r"(?:at least\s+)?(\d+)\s+placeholders", instr)
    if m:
        results.append(len(re.findall(r"\[[^\]\n]+\]", response)) >= int(m.group(1)))

    m = re.search(r"(?:at least\s+)?(\d+)\s+highlighted sections", instr)
    if m:
        results.append(len(re.findall(r"\*[^*\n]+\*", response)) >= int(m.group(1)))

    if re.search(r"\bpostscript\b|add a (?:p\.?\s?s\.?|postscript)", instr):
        results.append(bool(re.search(r"p\.?\s?s\.?", response.lower())))

    return results


class InstructionConstraintReward(RewardProtocol):
    """Fraction of verifiable instruction constraints the completion satisfies.

    Reads the constraints from the instruction text itself (IFEval-style), so it
    works on any instruction dataset without constraint metadata. Returns
    ``None`` for prompts with no detectable constraint, which GRPO treats as
    NaN and excludes from aggregation rather than diluting the signal.
    """

    reduction = _REDUCTION
    name = "constraint_following"

    def compute(self, *, prompt=None, completion=None, completion_text=None, **kwargs) -> float | None:
        instruction = _instruction_text(prompt)
        response = completion_text if completion_text is not None else _to_text(completion)
        checks = evaluate_constraints(instruction, response)
        if not checks:
            return None
        return sum(checks) / len(checks)


class ResponseQualityReward(RewardProtocol):
    """Penalize empty, truncated, or degenerate (repetitive) completions.

    Combines three per-completion signals surfaced by generation:
      - ``0`` for an empty response;
      - ``0`` when ``truncated`` is True (generation hit the length cap, i.e.
        ``finish_reason == "length"``) — an unfinished answer shouldn't be
        rewarded;
      - otherwise ``1 - repeated_4gram_fraction`` (clamped to ``[0, 1]``) to
        discourage filler/looping text.

    Acts as a light regularizer against reward hacking.
    """

    reduction = _REDUCTION
    name = "response_quality"

    def compute(self, *, completion=None, completion_text=None, truncated=None, **kwargs) -> float:
        response = completion_text if completion_text is not None else _to_text(completion)
        if not response.strip():
            return 0.0
        if truncated:  # cut off at the length cap rather than finishing cleanly
            return 0.0
        words = response.split()
        if len(words) < 8:
            return 1.0
        grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        repeated_fraction = 1.0 - (len(set(grams)) / len(grams))
        return max(0.0, 1.0 - repeated_fraction)


def prepare_dataset(dataset_name: str, split: str):
    """Load an instruction dataset and normalize every row to ``{"prompt": messages}``.

    Defensively handles the common schemas: a ``messages`` chat list, a
    ``prompt`` field (string or chat list), or an ``instruction`` / ``question``
    string. Only the prompt is needed — the constraint reward reads the
    instruction text directly from it.
    """
    ds = load_dataset(dataset_name, split=split)

    def _instruction(example: dict) -> str:
        for key in ("messages", "prompt", "conversations"):
            value = example.get(key)
            if isinstance(value, list):
                return _instruction_text(value)
            if isinstance(value, str):
                return value
        for key in ("instruction", "question", "query", "text"):
            if isinstance(example.get(key), str):
                return example[key]
        return ""

    def _format(example: dict) -> dict:
        return {"prompt": [{"role": "user", "content": _instruction(example)}]}

    ds = ds.map(_format, remove_columns=ds.column_names)
    return ds.filter(lambda ex: bool(ex["prompt"][0]["content"].strip()))


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
)


def resolve_tokenizer(path: str) -> str:
    """Return a path HF ``AutoTokenizer`` can load.

    HF can read local dirs and Hub repo ids directly, but not ``gs://`` (or
    other remote) URIs.  For a remote dir, the present tokenizer files are
    copied to a local temp dir (via ``ePath``) and that dir is returned.
    """
    if "://" not in path:  # local dir or HF repo id -> usable as-is
        return path

    import tempfile

    from eformer.paths import ePath

    src = ePath(path)
    dst = tempfile.mkdtemp(prefix="ed_tokenizer_")
    copied = []
    for name in _TOKENIZER_FILES:
        remote = src / name
        try:
            if remote.exists():
                (ePath(dst) / name).write_bytes(remote.read_bytes())
                copied.append(name)
        except Exception:
            continue
    if not copied:
        raise FileNotFoundError(
            f"No tokenizer files found under {path!r}. Point --tokenizer at an HF repo id "
            "or a directory that contains tokenizer.json / tokenizer_config.json."
        )
    return dst


def _prune_none_esurge_sizing(cfg: dict) -> dict:
    """Drop None KV-cache sizing knobs so they fall through to defaults.

    When both ``hbm_utilization`` and ``max_cache_tokens`` are left unset, the
    trainer auto-computes a KV-cache token budget from the rollout shape. Removing
    the None entries (rather than carrying them) keeps the trainer-section
    ``normalize_trainer_config`` setdefault path clean (so the dataclass/eLarge
    None default survives into the auto-compute branch). The standalone eSurge
    section is not used during ``train()`` and ``normalize_esurge_sections``
    already drops None-valued keys, so this is also belt-and-suspenders there.
    """
    for section, keys in (
        ("esurge", ("hbm_utilization", "max_cache_tokens")),
        ("trainer", ("esurge_hbm_utilization", "esurge_max_cache_tokens")),
    ):
        sec = cfg.get(section)
        if isinstance(sec, dict):
            for key in keys:
                if sec.get(key) is None:
                    sec.pop(key, None)
    return cfg


def main():
    parser = DataClassArgumentParser(GRPOInstructArgs, description="Async-GRPO instruction following with EasyDeL")
    (args,) = parser.parse_args_into_dataclasses()

    max_completion_length = args.max_length - args.max_prompt_length
    tokenizer_path = resolve_tokenizer(args.tokenizer or args.model)

    elm = eLargeModel(
        _prune_none_esurge_sizing(
            {
                "model": {
                    "name_or_path": args.model,
                    "tokenizer": tokenizer_path,
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
                "base_config": {
                    "values": {
                        "freq_max_position_embeddings": args.max_length,
                        "mask_max_position_embeddings": args.max_length,
                        "attn_mechanism": ed.AttentionMechanisms.AUTO,
                        "attn_dtype": "bf16",
                        "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
                    }
                },
                "esurge": {
                    "max_model_len": args.max_length,
                    "max_num_seqs": args.esurge_max_num_seqs,
                    "page_size": args.esurge_page_size,
                    "hbm_utilization": args.esurge_hbm_utilization,
                    "max_cache_tokens": args.esurge_max_cache_tokens,
                    "enable_prefix_caching": True,
                    "max_num_batched_tokens": 2048,
                    "use_aot_forward": True,
                    "data_parallelism_axis": "fsdp",
                    "verbose": True,
                },
                "trainer": {
                    "trainer_type": "async_grpo",
                    "save_directory": args.output_dir,
                    "num_train_epochs": 1,
                    "max_training_steps": args.max_steps,
                    "total_batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": 0.03,
                    "max_length": args.max_length,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": max_completion_length,
                    "beta": 0.0,  # no KL term -> reference model is aliased (not deep-copied), freeing ~13.6 GB/chip
                    "loss_type": "dapo",
                    "lmhead_chunksize": 2048,
                    "logprob_vocab_chunk_size": 2048,
                    "tx_mu_dtype": "bfloat16",
                    "num_return_sequences": args.num_generations,
                    "scale_rewards": "group",
                    # Reward weights live on each RewardProtocol (ResponseQualityReward(weight=0.3)
                    # below); no parallel reward_weights list needed.
                    "max_inflight_tasks": 32,
                    "max_staleness": 1,
                    "weight_sync_steps": 1,
                    "request_timeout": args.request_timeout,
                    "esurge_page_size": args.esurge_page_size,
                    "esurge_hbm_utilization": args.esurge_hbm_utilization,
                    "esurge_max_num_seqs": args.esurge_max_num_seqs,
                    "esurge_max_cache_tokens": args.esurge_max_cache_tokens,
                    "esurge_max_num_batched_tokens": 2048,
                    "esurge_enable_prefix_caching": True,
                    "esurge_data_parallelism_axis": "fsdp",
                    "generation_temperature": 0.9,
                    "generation_top_p": 0.95,
                    "generation_top_k": 50,
                    "generation_do_sample": True,
                    "log_steps": 1,
                    "save_steps": 100,
                    "save_total_limit": 2,
                    "log_completions": True,
                    "use_wandb": args.wandb,
                    "wandb_entity": args.wandb_entity,
                    "generation_interval": 50,
                    "generation_prompts": [
                        "Write a short product description for noise-cancelling headphones. Respond with at least 3 bullet points and finish your response with the exact phrase 'Order yours today!'",
                        "Explain photosynthesis in exactly 2 sentences, in all lowercase, without using any commas.",
                        "Give me a JSON object with keys 'title' and 'summary' describing the water cycle.",
                    ],
                },
            }
        )
    )

    train_dataset = prepare_dataset(args.dataset, args.dataset_split)

    elm.train(
        train_dataset=train_dataset,
        reward_funcs=[InstructionConstraintReward(), ResponseQualityReward(weight=0.3)],
    )


if __name__ == "__main__":
    main()
