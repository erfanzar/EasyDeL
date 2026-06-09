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

"""Synthetic serving benchmark for eLarge + eSurge.

This follows the same broad benchmark shape used by vLLM and SGLang for
synthetic/random serving workloads:

- synthetic prompts instead of a real dataset
- all requests submitted together
- fixed output length
- ``ignore_eos=True`` by default
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

import easydel as ed
from easydel.infra.elarge.processing import make_serializable, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic serving benchmark with eLarge + eSurge on Qwen/Qwen3-8B."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name or local path.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer override. Defaults to --model.")
    parser.add_argument("--num-prompts", type=int, default=256, help="Total number of synthetic prompts.")
    parser.add_argument("--parallel", type=int, default=256, help="Target concurrency, matching vLLM/SGLang naming.")
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Target synthetic prompt length in tokens. Defaults to max_model_len - output_len - reserve_tokens.",
    )
    parser.add_argument("--output-len", type=int, default=256, help="Target generation length in tokens.")
    parser.add_argument(
        "--input-len-range-ratio",
        type=float,
        default=0.0,
        help="Uniformly vary per-request input lengths within +/- this ratio.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt generation.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Do not force ignore_eos. By default this benchmark ignores EOS like synthetic serving benchmarks.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--output-path", type=Path, default=Path("tmp-files/qwen3_8b_elarge_serving_benchmark.json"))
    parser.add_argument("--max-model-len", type=int, default=8192, help="eSurge max context length.")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Explicit eSurge max concurrent sequences. Defaults to --parallel.",
    )
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=None,
        help="Optional eSurge reserve_tokens override. Defaults to eSurge's behavior: max_num_seqs.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="eSurge max batched tokens.")
    parser.add_argument("--hbm-utilization", type=float, default=0.85, help="eSurge HBM utilization.")
    parser.add_argument("--page-size", type=int, default=128, help="eSurge KV page size.")
    parser.add_argument(
        "--axis-dims",
        type=int,
        nargs=5,
        default=(1, 1, 1, -1, 1),
        metavar=("DP", "FSDP", "EP", "TP", "SP"),
        help="EasyDeL 5D sharding axis dims.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for model/tokenizer.")
    parser.add_argument("--dont-verbose", action="store_true", help="Disable verbose loader/eSurge logging.")
    args = parser.parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be > 0.")
    if args.parallel <= 0:
        raise ValueError("--parallel must be > 0.")
    if args.output_len <= 0:
        raise ValueError("--output-len must be > 0.")
    if args.input_len is not None and args.input_len <= 0:
        raise ValueError("--input-len must be > 0 when provided.")
    if args.reserve_tokens is not None and args.reserve_tokens < 0:
        raise ValueError("--reserve-tokens must be >= 0 when provided.")
    if args.input_len_range_ratio < 0.0 or args.input_len_range_ratio >= 1.0:
        raise ValueError("--input-len-range-ratio must be in [0.0, 1.0).")

    return args


def build_elarge_model(args: argparse.Namespace) -> ed.eLargeModel:
    tokenizer_name = args.tokenizer or args.model
    max_num_seqs = args.max_num_seqs or args.parallel
    return ed.eLargeModel(
        {
            "model": {
                "name_or_path": args.model,
                "tokenizer": tokenizer_name,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
                "verbose": not bool(args.dont_verbose),
                "trust_remote_code": bool(args.trust_remote_code),
            },
            "sharding": {
                "axis_dims": tuple(args.axis_dims),
                "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            "base_config": {
                "values": {
                    "freq_max_position_embeddings": args.max_model_len,
                    "mask_max_position_embeddings": args.max_model_len,
                    "attn_mechanism": ed.AttentionMechanisms.AUTO,
                    "attn_dtype": "bf16",
                    "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NONE,
                    "moe_method": ed.MoEMethods.FUSED_MOE,
                }
            },
            "esurge": {
                "runtime": {
                    "max_model_len": args.max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "min_input_pad": 8,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "use_aot_forward": True,
                    "runner_verbose": False,
                },
                "cache": {
                    "hbm_utilization": args.hbm_utilization,
                    "page_size": args.page_size,
                    "enable_prefix_caching": False,
                    "data_parallelism_axis": "fsdp",
                },
                "context": {
                    "reserve_tokens": args.reserve_tokens,
                },
            },
        }
    )


def get_text_tokenizer(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    return tokenizer or processor


def count_prompt_tokens(prompt_token_ids: list[int] | list[list[int]]) -> int:
    if not prompt_token_ids:
        return 0
    first = prompt_token_ids[0]
    if isinstance(first, list):
        return sum(len(segment) for segment in prompt_token_ids)
    return len(prompt_token_ids)


def sample_length(target: int, ratio: float, rng: random.Random) -> int:
    if ratio <= 0.0:
        return target
    low = max(1, int(target * (1.0 - ratio)))
    high = max(low, int(target * (1.0 + ratio)))
    return rng.randint(low, high)


def select_repeat_unit(tokenizer: PreTrainedTokenizerBase) -> str:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for candidate in (" a", "a", " hello", "."):
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if token_ids and not any(token_id in special_ids for token_id in token_ids):
            return candidate
    return "a"


def make_prompt_from_repeated_text(tokenizer: PreTrainedTokenizerBase, repeat_unit: str, token_count: int) -> str:
    def encoded_len(multiplier: int) -> int:
        text = repeat_unit * multiplier
        return len(tokenizer.encode(text, add_special_tokens=False))

    low = 1
    high = 1
    while encoded_len(high) < token_count:
        high *= 2

    while low < high:
        mid = (low + high) // 2
        if encoded_len(mid) >= token_count:
            high = mid
        else:
            low = mid + 1

    prompt = repeat_unit * high
    if not prompt.strip():
        prompt = ("a " * max(token_count, 1)).strip()
    return prompt


def build_synthetic_prompts(tokenizer: Any, args: argparse.Namespace, reserve_tokens: int) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    input_len = args.input_len
    if input_len is None:
        input_len = args.max_model_len - args.output_len - reserve_tokens
    if input_len <= 0:
        raise ValueError("input_len must leave room for output_len and reserve_tokens within max_model_len.")
    if input_len + args.output_len + reserve_tokens > args.max_model_len:
        raise ValueError("input_len + output_len + reserve_tokens must be <= max_model_len.")

    prompts = []
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        repeat_unit = select_repeat_unit(tokenizer)
        for prompt_index in range(args.num_prompts):
            target_len = sample_length(input_len, args.input_len_range_ratio, rng)
            prompt_text = make_prompt_from_repeated_text(tokenizer, repeat_unit, target_len)
            prompts.append(
                {
                    "prompt_index": prompt_index,
                    "target_input_len": target_len,
                    "prompt": prompt_text,
                }
            )
        return prompts

    for prompt_index in range(args.num_prompts):
        target_len = sample_length(input_len, args.input_len_range_ratio, rng)
        prompt_text = "a" * max(target_len, 1)
        prompts.append(
            {
                "prompt_index": prompt_index,
                "target_input_len": target_len,
                "prompt": prompt_text,
            }
        )
    return prompts


def run_generation(engine: Any, prompts: list[dict[str, Any]], sampling_params: ed.SamplingParams) -> dict[str, Any]:
    prompt_texts = [item["prompt"] for item in prompts]
    started_at = time.perf_counter()
    outputs = engine.generate(prompt_texts, sampling_params=sampling_params, use_tqdm=True)
    latency = time.perf_counter() - started_at

    prompt_tokens = [count_prompt_tokens(output.prompt_token_ids) for output in outputs]
    output_tokens = [int(output.num_generated_tokens or 0) for output in outputs]
    request_latencies = [float(output.processing_time or 0.0) for output in outputs]
    ttfts = [float(output.first_token_time) for output in outputs if output.first_token_time is not None]

    total_prompt_tokens = sum(prompt_tokens)
    total_output_tokens = sum(output_tokens)
    return {
        "latency_seconds": latency,
        "request_throughput_rps": len(outputs) / latency if latency > 0 else 0.0,
        "input_throughput_tps": total_prompt_tokens / latency if latency > 0 else 0.0,
        "output_throughput_tps": total_output_tokens / latency if latency > 0 else 0.0,
        "num_requests": len(outputs),
        "num_prompt_tokens": total_prompt_tokens,
        "num_output_tokens": total_output_tokens,
        "avg_request_latency_seconds": sum(request_latencies) / len(request_latencies) if request_latencies else 0.0,
        "avg_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "finish_reasons": dict(
            Counter((output.outputs[0].finish_reason if output.outputs else "unknown") for output in outputs)
        ),
        "requests": [
            {
                "request_index": prompts[index]["prompt_index"],
                "target_input_len": prompts[index]["target_input_len"],
                "actual_input_tokens": prompt_tokens[index],
                "output_tokens": output_tokens[index],
                "latency_seconds": request_latencies[index],
                "ttft_seconds": outputs[index].first_token_time,
                "finish_reason": outputs[index].outputs[0].finish_reason if outputs[index].outputs else None,
            }
            for index in range(len(outputs))
        ],
    }


def get_runner_perf(engine: Any) -> dict[str, Any]:
    runner = getattr(engine, "runner", None)
    if runner is None:
        return {}
    return {
        "agg_tps": getattr(runner, "_perf_last_agg_tps", None),
        "req_tps": getattr(runner, "_perf_last_req_tps", None),
        "ema_tps": getattr(runner, "_perf_tps_ema", None),
        "step_total_time_seconds": getattr(runner, "_perf_last_total_time", None),
        "step_total_tokens": getattr(runner, "_perf_last_total_tokens", None),
    }


def summarize_runner_history(engine: Any, expected_batch_size: int) -> dict[str, Any]:
    runner = getattr(engine, "runner", None)
    if runner is None:
        return {}

    perf_history = list(getattr(runner, "_perf_history", ()))
    if not perf_history:
        return {}

    all_tps = [float(sample.agg_tps) for sample in perf_history if float(sample.agg_tps) > 0.0]
    saturated_tps = [
        float(sample.agg_tps)
        for sample in perf_history
        if float(sample.agg_tps) > 0.0 and int(sample.num_scheduled_reqs) >= expected_batch_size
    ]
    steady_decode_tps = [
        float(sample.agg_tps)
        for sample in perf_history
        if float(sample.agg_tps) > 0.0
        and int(sample.num_scheduled_reqs) == expected_batch_size
        and int(sample.total_tokens) == int(sample.num_scheduled_reqs)
        and int(sample.num_new) == 0
        and int(sample.num_finished) == 0
    ]

    def _summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "last": values[-1],
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "all_steps": _summary(all_tps),
        "saturated_steps": _summary(saturated_tps),
        "steady_decode_steps": _summary(steady_decode_tps),
        "num_runner_steps": len(perf_history),
    }


def print_result(
    result: dict[str, Any],
    runner_perf: dict[str, Any],
    runner_history: dict[str, Any],
    engine_metrics: dict[str, Any] | None,
) -> None:
    decode_steps = runner_history.get("steady_decode_steps") or {}
    saturated_steps = runner_history.get("saturated_steps") or {}
    if decode_steps:
        print(f"Runner steady decode agg TPS mean: {decode_steps['mean']:.3f} token/s")
        print(f"Runner steady decode agg TPS max: {decode_steps['max']:.3f} token/s")
    elif saturated_steps:
        print(f"Runner saturated agg TPS mean: {saturated_steps['mean']:.3f} token/s")
        print(f"Runner saturated agg TPS max: {saturated_steps['max']:.3f} token/s")
    if runner_perf.get("agg_tps") is not None:
        print(f"Runner last-step agg TPS: {runner_perf['agg_tps']:.3f} token/s")
    if runner_perf.get("ema_tps") is not None:
        print(f"Runner EMA TPS: {runner_perf['ema_tps']:.3f} token/s")
    if runner_perf.get("req_tps") is not None:
        print(f"Runner last-step req TPS: {runner_perf['req_tps']:.3f} token/s/request")
    print(f"End-to-end wall latency: {result['latency_seconds']:.3f} s")
    print(f"End-to-end wall request throughput: {result['request_throughput_rps']:.3f} req/s")
    print(f"End-to-end wall input throughput: {result['input_throughput_tps']:.3f} token/s")
    print(f"End-to-end wall output throughput: {result['output_throughput_tps']:.3f} token/s")
    print(f"Avg request latency: {result['avg_request_latency_seconds']:.3f} s")
    print(f"Avg TTFT: {result['avg_ttft_seconds']:.3f} s")
    if engine_metrics and "error" not in engine_metrics:
        print(f"Completed-request avg throughput: {engine_metrics['average_throughput']:.3f} token/s")
        print(f"Completed-request avg latency: {engine_metrics['average_latency']:.3f} s")
        print(f"Completed-request avg TTFT: {engine_metrics['average_ttft']:.3f} s")


def main() -> None:
    args = parse_args()
    elm = build_elarge_model(args)
    engine = elm.build_esurge()

    try:
        engine.start_monitoring(enable_prometheus=False, enable_console=False, start_grafana=False)
        processor = getattr(engine, "processor", None) or getattr(engine, "tokenizer", None)
        tokenizer = get_text_tokenizer(processor)
        prompts = build_synthetic_prompts(tokenizer, args, reserve_tokens=int(engine.reserve_tokens))

        sampling_params = ed.SamplingParams(
            max_tokens=args.output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            ignore_eos=not bool(args.disable_ignore_eos),
        )

        result = run_generation(engine=engine, prompts=prompts, sampling_params=sampling_params)
        runner_perf = get_runner_perf(engine)
        runner_history = summarize_runner_history(
            engine=engine,
            expected_batch_size=min(args.num_prompts, args.max_num_seqs or args.parallel),
        )
        engine_metrics = engine.get_metrics_summary()
    finally:
        engine.terminate()

    results = {
        "method": "synthetic_serving_benchmark",
        "sources": {
            "shape": "aligned with vLLM/SGLang synthetic serving benchmarks",
            "synthetic_prompts": True,
            "ignore_eos": not bool(args.disable_ignore_eos),
        },
        "model": {
            "model_name_or_path": args.model,
            "tokenizer": args.tokenizer or args.model,
        },
        "benchmark": {
            "num_prompts": args.num_prompts,
            "parallel": args.parallel,
            "input_len": (
                args.input_len
                if args.input_len is not None
                else args.max_model_len - args.output_len - int(engine.reserve_tokens)
            ),
            "output_len": args.output_len,
            "input_len_range_ratio": args.input_len_range_ratio,
            "seed": args.seed,
            "ignore_eos": not bool(args.disable_ignore_eos),
            "reserve_tokens": int(engine.reserve_tokens),
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.output_len,
        },
        "esurge": {
            "runtime": {
                "max_num_seqs": args.max_num_seqs or args.parallel,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
            "cache": {
                "hbm_utilization": args.hbm_utilization,
                "page_size": args.page_size,
                "enable_prefix_caching": False,
            },
            "context": {
                "reserve_tokens": int(engine.reserve_tokens),
            },
            "axis_dims": tuple(args.axis_dims),
        },
        "result": result,
        "runner_perf": runner_perf,
        "runner_history": runner_history,
        "engine_metrics": engine_metrics,
    }

    print_result(result, runner_perf, runner_history, engine_metrics)
    serialized = make_serializable(results)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.output_path, json.dumps(serialized, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()
