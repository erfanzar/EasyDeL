# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
"""
How to use

Convert a Hugging Face checkpoint to an EasyDeL checkpoint (recommended: sequential, no push yet):

  python scripts/convert_hf_to_easydel.py \\
    --source lmsys/gpt-oss-120b-bf16 \\
    --out /mnt/gcs/easydel/gpt-oss-120b \\
    --repo-id EasyDeL/gpt-oss-120b \\
    --convert-mode sequential \\
    --no-push-to-hub \\
    --torch-streaming-cache temp \\
    --torch-streaming-tmp-dir /tmp/hf-shards \\
    --enable-hf-transfer \\
    --token $HF_TOKEN

Push later by re-running with `--push-to-hub` (or omitting `--no-push-to-hub`).

Sharding

`--sharding-axis-dims` and `--sharding-axis-names` define the 5D mesh as:
  dp,fsdp,ep,tp,sp

Use -1 in axis dims to auto-infer that axis from available devices.
Example (single host; auto-choose FSDP):
  --sharding-axis-dims 1,-1,1,1,1

Disk usage tips

- Prefer `--torch-streaming-cache temp` to avoid filling the HF cache with full shards.
- If you do use HF cache, redirect it with `--cache-dir` (or set HF_HOME/HF_HUB_CACHE).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import jax.numpy as jnp
from eformer.aparser import DataClassArgumentParser

try:
    from eformer.loggings import get_logger  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import logging

    def get_logger(name: str):  # type: ignore[no-redef]
        return logging.getLogger(name)


logger = get_logger("Convertor")


def _parse_dtype(value: str):
    v = value.strip().lower()
    mapping = {
        "bf16": jnp.bfloat16,
        "bfloat16": jnp.bfloat16,
        "fp16": jnp.float16,
        "float16": jnp.float16,
        "f16": jnp.float16,
        "fp32": jnp.float32,
        "float32": jnp.float32,
        "f32": jnp.float32,
    }
    if v not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}. Use one of: {', '.join(sorted(mapping))}")
    return mapping[v]


def _parse_int_list(value: str, *, expected_len: int | None = None) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    try:
        ints = tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"Expected comma-separated ints, got {value!r}") from e
    if expected_len is not None and len(ints) != expected_len:
        raise ValueError(f"Expected {expected_len} ints, got {len(ints)}: {value!r}")
    return ints


def _parse_str_list(value: str, *, expected_len: int | None = None) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in value.split(",") if p.strip() != "")
    if expected_len is not None and len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} items, got {len(parts)}: {value!r}")
    return parts


TaskType = Literal[
    "auto",
    "causal_lm",
    "seq2seq",
    "speech_seq2seq",
    "image_text_to_text",
    "zero_shot_image_classification",
    "sequence_classification",
    "diffusion_lm",
    "base",
    "vision",
    "any_to_any",
]
ConvertMode = Literal["sequential", "from_pretrained"]
TorchStreamingCache = Literal["hf_cache", "temp"]


@dataclass
class ConvertArgs:
    source: str = field(metadata={"help": "HF repo id (e.g. meta-llama/Llama-3.1-8B) or local path"})
    out: str = field(metadata={"help": "Output directory (local path; GCSFuse mount works)"})

    repo_id: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={"help": "Optional: HF repo id to push to (e.g. EasyDeL/my-easydel)"},
    )
    push_to_hub: bool = field(
        default=True,
        metadata={"help": "When --repo-id is set, push the converted folder to the HF Hub."},
    )

    task: TaskType = field(
        default="auto",
        metadata={"help": "Model task (controls which AutoEasyDeLModel* class is used)"},
    )
    convert_mode: ConvertMode = field(
        default="sequential",
        metadata={
            "help": "sequential streams shards and writes a TensorStore checkpoint without loading full params; "
            "from_pretrained loads the model then saves via model.save_pretrained()"
        },
    )

    torch_streaming_cache: TorchStreamingCache = field(
        default="hf_cache",
        metadata={
            "help": "Where to store streamed shard downloads: hf_cache keeps files in HF cache; "
            "temp downloads one shard at a time"
        },
    )
    torch_streaming_tmp_dir: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={
            "help": "Optional parent directory for temp shard downloads (only used with --torch-streaming-cache temp)"
        },
    )
    tensorstore_chunk_bytes: int = field(
        default=2_147_483_648,
        metadata={"help": "Max target chunk size when writing TensorStore chunks (default: 2GiB)"},
    )

    dtype: str = field(default="bf16", metadata={"help": "Compute dtype (bf16|fp16|fp32)"})
    param_dtype: str = field(default="bf16", metadata={"help": "Param dtype (bf16|fp16|fp32)"})

    sharding_axis_dims: str = field(
        default="1,-1,1,1,1",
        metadata={"help": "5D mesh dims: dp,fsdp,ep,tp,sp (use -1 for auto), e.g. 1,-1,1,1,1"},
    )
    sharding_axis_names: str = field(
        default="dp,fsdp,ep,tp,sp",
        metadata={"help": "5 axis names: dp,fsdp,ep,tp,sp"},
    )
    auto_shard_model: bool = field(default=True, metadata={"help": "Enable/disable automatic sharding"})

    cache_dir: Optional[str] = field(  # noqa: UP045
        default=None, metadata={"help": "HF cache directory (point this at your GCSFuse mount)"}
    )
    revision: Optional[str] = field(default=None, metadata={"help": "HF revision/branch/tag/commit"})  # noqa: UP045
    token: Optional[str] = field(  # noqa: UP045
        default=None,
        metadata={"help": "HF token (or rely on HF_TOKEN env / `huggingface-cli login`)"},
    )
    local_files_only: bool = field(default=False, metadata={"help": "Do not try to download from HF Hub"})
    force_download: bool = field(default=False, metadata={"help": "Force re-download from HF Hub"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Pass trust_remote_code=True to HF loaders"})
    enable_hf_transfer: bool = field(
        default=False,
        metadata={"help": "Enable hf_transfer accelerated HF downloads (requires `pip install hf_transfer`)"},
    )


def main(argv: list[str] | None = None) -> None:
    parser = DataClassArgumentParser(
        ConvertArgs,
        description="Download/convert a HuggingFace PyTorch checkpoint to EasyDeL and optionally push to HF Hub.",
    )
    (args,) = parser.parse_args_into_dataclasses(args=argv, look_for_args_file=False)

    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            import hf_transfer  # noqa: F401
        except Exception:
            logger.warn("Warning: `hf_transfer` is not installed. Run: pip install -U hf_transfer")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep this import after arg parsing so `--help` is fast.
    from transformers import AutoConfig

    import easydel as ed

    dtype = _parse_dtype(args.dtype)
    param_dtype = _parse_dtype(args.param_dtype)
    sharding_axis_dims = _parse_int_list(args.sharding_axis_dims, expected_len=5)
    sharding_axis_names = _parse_str_list(args.sharding_axis_names, expected_len=5)

    hf_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "token": args.token,
        "local_files_only": args.local_files_only,
        "force_download": args.force_download,
        "trust_remote_code": args.trust_remote_code,
    }
    hf_kwargs = {k: v for k, v in hf_kwargs.items() if v is not None}

    config = AutoConfig.from_pretrained(args.source, **hf_kwargs)

    def infer_task_from_config():
        architectures = [str(a) for a in (getattr(config, "architectures", None) or [])]
        joined = " ".join(architectures).lower()
        model_type = str(getattr(config, "model_type", "") or "").lower()

        if model_type in {"clip", "siglip"}:
            return "zero_shot_image_classification"
        if "forspeechseq2seq" in joined or model_type in {"whisper", "speech_to_text", "speech-to-text"}:
            return "speech_seq2seq"
        if (
            "forimagetexttotext" in joined
            or "vision2seq" in joined
            or model_type
            in {
                "llava",
                "idefics",
                "idefics2",
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen2vl",
                "qwen2_5vl",
                "paligemma",
            }
        ):
            return "image_text_to_text"
        if (
            getattr(config, "is_encoder_decoder", False)
            or "forconditionalgeneration" in joined
            or "forseq2seqlm" in joined
        ):
            return "seq2seq"
        if "forzeroshotimageclassification" in joined or "zeroshotimageclassification" in joined:
            return "zero_shot_image_classification"
        if "forsequenceclassification" in joined or "sequenceclassification" in joined:
            return "sequence_classification"
        if "fordiffusionlm" in joined or model_type.endswith("diffusion"):
            return "diffusion_lm"
        if "forcausallm" in joined or "causallm" in joined:
            return "causal_lm"
        return "causal_lm"

    task = infer_task_from_config() if args.task == "auto" else args.task
    logger.info(f"Task: {task}")

    task_to_cls = {
        "causal_lm": ed.AutoEasyDeLModelForCausalLM,
        "seq2seq": ed.AutoEasyDeLModelForSeq2SeqLM,
        "speech_seq2seq": ed.AutoEasyDeLModelForSpeechSeq2Seq,
        "image_text_to_text": ed.AutoEasyDeLModelForImageTextToText,
        "zero_shot_image_classification": ed.AutoEasyDeLModelForZeroShotImageClassification,
        "sequence_classification": ed.AutoEasyDeLModelForSequenceClassification,
        "diffusion_lm": ed.AutoEasyDeLModelForDiffusionLM,
        "base": ed.AutoEasyDeLModel,
        "vision": ed.AutoEasyDeLVisionModel,
        "any_to_any": ed.AutoEasyDeLAnyToAnyModel,
    }

    model_cls = task_to_cls[task]

    # Save tokenizer/processor too so `save_pretrained(..., push_to_hub=True)` uploads a complete repo.
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.source, **hf_kwargs)
        processor.save_pretrained(str(out_dir))
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.source, **hf_kwargs)
        tokenizer.save_pretrained(str(out_dir))
    except Exception:
        pass
    try:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(args.source, **hf_kwargs)
        image_processor.save_pretrained(str(out_dir))
    except Exception:
        pass
    try:
        from transformers import AutoFeatureExtractor

        feature_extractor = AutoFeatureExtractor.from_pretrained(args.source, **hf_kwargs)
        feature_extractor.save_pretrained(str(out_dir))
    except Exception:
        pass

    if args.convert_mode == "sequential":
        from easydel.infra.base_module import EasyDeLBaseModule

        class Base(EasyDeLBaseModule):
            _model_task = model_cls.model_task

        Base.huggingface_to_easydel_sequential(
            pretrained_model_name_or_path=args.source,
            save_directory=str(out_dir),
            output_repo_id=args.repo_id,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            torch_streaming_cache=args.torch_streaming_cache,
            torch_streaming_tmp_dir=args.torch_streaming_tmp_dir,
            tensorstore_chunk_bytes=args.tensorstore_chunk_bytes,
            verbose=True,
            **hf_kwargs,
        )
        if args.repo_id and args.push_to_hub:
            from huggingface_hub import HfApi

            api = HfApi(token=args.token)
            api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, token=args.token)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Adding EasyDeL Checkpoints",
            )
    else:
        model = model_cls.from_pretrained(
            args.source,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding_axis_dims=sharding_axis_dims,
            sharding_axis_names=sharding_axis_names,
            auto_shard_model=args.auto_shard_model,
            from_torch=True,
            torch_streaming_cache=args.torch_streaming_cache,
            torch_streaming_tmp_dir=args.torch_streaming_tmp_dir,
            **hf_kwargs,
        )
        if args.repo_id and args.push_to_hub:
            model.save_pretrained(
                str(out_dir),
                push_to_hub=True,
                repo_id=args.repo_id,
                token=args.token if args.token is not None else True,
            )
        else:
            model.save_pretrained(str(out_dir))

    logger.info(f"Done. Saved to: {out_dir}")
    if args.repo_id and args.push_to_hub:
        logger.info(f"Pushed to: {args.repo_id}")


if __name__ == "__main__":
    main()
