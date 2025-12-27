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
from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
from eformer.loggings import get_logger

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
        raise argparse.ArgumentTypeError(f"Unsupported dtype {value!r}. Use one of: {', '.join(sorted(mapping))}")
    return mapping[v]


def _parse_int_list(value: str, *, expected_len: int | None = None) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    try:
        ints = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected comma-separated ints, got {value!r}") from e
    if expected_len is not None and len(ints) != expected_len:
        raise argparse.ArgumentTypeError(f"Expected {expected_len} ints, got {len(ints)}: {value!r}")
    return ints


def _parse_str_list(value: str, *, expected_len: int | None = None) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in value.split(",") if p.strip() != "")
    if expected_len is not None and len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"Expected {expected_len} items, got {len(parts)}: {value!r}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/convert a HuggingFace PyTorch checkpoint to EasyDeL and optionally push to HF Hub."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="HF repo id (e.g. meta-llama/Llama-3.1-8B) or local path",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (local path; GCSFuse mount works)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional: HF repo id to push to (e.g. username/my-easydel)",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=(
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
        ),
        help="Model task (controls which AutoEasyDeLModel* class is used)",
    )
    parser.add_argument(
        "--convert-mode",
        choices=("sequential", "from_pretrained"),
        default="sequential",
        help="sequential streams shards and writes a TensorStore checkpoint without loading full params; "
        "from_pretrained loads the model then saves via model.save_pretrained()",
    )

    parser.add_argument(
        "--torch-streaming-cache",
        choices=("hf_cache", "temp"),
        default="hf_cache",
        help="Where to store streamed shard downloads: hf_cache keeps files in HF cache; temp downloads one shard at a time",
    )
    parser.add_argument(
        "--torch-streaming-tmp-dir",
        default=None,
        help="Optional parent directory for temp shard downloads (only used with --torch-streaming-cache temp)",
    )
    parser.add_argument(
        "--tensorstore-chunk-bytes",
        type=int,
        default=2_147_483_648,
        help="Max target chunk size when writing TensorStore chunks (default: 2GiB)",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Compute dtype (bf16|fp16|fp32)",
    )
    parser.add_argument(
        "--param-dtype",
        default="bf16",
        help="Param dtype (bf16|fp16|fp32)",
    )

    parser.add_argument(
        "--sharding-axis-dims",
        type=lambda s: _parse_int_list(s, expected_len=5),
        default="1,-1,1,1,1",
        help="5D mesh dims: dp,fsdp,ep,tp,sp (use -1 for auto), e.g. 1,-1,1,1,1",
    )
    parser.add_argument(
        "--sharding-axis-names",
        type=lambda s: _parse_str_list(s, expected_len=5),
        default="dp,fsdp,ep,tp,sp",
        help="5 axis names: dp,fsdp,ep,tp,sp",
    )
    parser.add_argument(
        "--auto-shard-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable automatic sharding",
    )

    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache directory (point this at your GCSFuse mount)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF revision/branch/tag/commit",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (or rely on HF_TOKEN env / `huggingface-cli login`)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not try to download from HF Hub",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download from HF Hub",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable hf_transfer accelerated HF downloads (requires `pip install hf_transfer`)",
    )

    args = parser.parse_args()

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
            dtype=dtype,
            param_dtype=param_dtype,
            sharding_axis_dims=args.sharding_axis_dims,
            sharding_axis_names=args.sharding_axis_names,
            torch_streaming_cache=args.torch_streaming_cache,
            torch_streaming_tmp_dir=args.torch_streaming_tmp_dir,
            tensorstore_chunk_bytes=args.tensorstore_chunk_bytes,
            verbose=True,
            **hf_kwargs,
        )
        if args.repo_id:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, token=args.token)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=args.repo_id,
                repo_type="model",
                token=args.token,
                commit_message="Add EasyDeL checkpoint",
            )
    else:
        model = model_cls.from_pretrained(
            args.source,
            dtype=dtype,
            param_dtype=param_dtype,
            sharding_axis_dims=args.sharding_axis_dims,
            sharding_axis_names=args.sharding_axis_names,
            auto_shard_model=args.auto_shard_model,
            from_torch=True,
            torch_streaming_cache=args.torch_streaming_cache,
            torch_streaming_tmp_dir=args.torch_streaming_tmp_dir,
            **hf_kwargs,
        )
        if args.repo_id:
            model.save_pretrained(
                str(out_dir),
                push_to_hub=True,
                repo_id=args.repo_id,
                token=args.token if args.token is not None else True,
            )
        else:
            model.save_pretrained(str(out_dir))

    logger.info(f"Done. Saved to: {out_dir}")
    if args.repo_id:
        logger.info(f"Pushed to: {args.repo_id}")


if __name__ == "__main__":
    main()
