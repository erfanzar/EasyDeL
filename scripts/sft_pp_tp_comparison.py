#!/usr/bin/env python
"""Compare EasyDeL SFT on a dummy 3B model: PP=2/TP=2 vs TP=4.

This script is meant to run on CPU with virtual devices (set
``XLA_FLAGS=--xla_force_host_platform_device_count=4``) or on a 4+ accelerator
slice.  It builds a Llama-style "dummy 3B" model from a config, generates a
small synthetic pretokenized dataset, and runs a few SFT training steps under
two mesh configurations:

* ``PP=2 + TP=2``: ``sharding_axis_dims=(2, 1, 1, 1, 2, 1)`` with
  ``pipeline_stage_regions=True`` and a ``Std1F1B`` pipeline schedule.
* ``TP=4``: ``sharding_axis_dims=(1, 1, 1, 1, 4, 1)`` with no pipeline
  scheduler (pure SPMD tensor parallelism).

The script reports per-step loss, wall time, and peak host memory.  On CPU the
absolute step times are not representative of accelerator training, but the two
configs can still be compared for loss parity and memory footprint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import threading
import time
import typing as tp
from pathlib import Path

import jax
import numpy as np
import psutil
import spectrax as spx
from datasets import Dataset
from jax import numpy as jnp

import easydel as ed

logger = logging.getLogger(__name__)

_AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


@dataclasses.dataclass
class _RunResult:
    name: str
    sharding_axis_dims: tuple[int, ...]
    schedule_name: str | None
    losses: list[float]
    step_times: list[float]
    compile_time: float
    peak_rss_mb: float
    error: str | None = None


class _FakeTokenizer:
    """Minimal tokenizer stand-in for pretokenized SFT data."""

    def __init__(self, vocab_size: int, pad_token_id: int = 0, eos_token_id: int = 1):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.pad_token = f"<pad:{pad_token_id}>"
        self.eos_token = f"<eos:{eos_token_id}>"


def _memory_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


class _PeakMemorySampler:
    """Sample peak host RSS in a background thread during a run."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, _memory_mb())
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)
        self.peak_mb = max(self.peak_mb, _memory_mb())
        return self.peak_mb


def _make_dummy_3b_config(
    sharding_axis_dims: tuple[int, ...],
    pipeline_stage_regions: bool,
    max_position_embeddings: int,
    *,
    scheduler_kind: str = "std1f1b",
    virtual_stages: int = 1,
    attn_mechanism: str = "vanilla",
    gradient_checkpointing: ed.EasyDeLGradientCheckPointers = ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    lmhead_chunksize: int | None = None,
    stage_layout: str = "loop",
) -> ed.LlamaConfig:
    """Build a Llama-style config sized close to 3B params (vocab=1024)."""
    if scheduler_kind == "kimik2":
        pipeline_stage_layout = stage_layout
        pipeline_virtual_stages = max(1, virtual_stages)
    else:
        pipeline_stage_layout = "contiguous"
        pipeline_virtual_stages = 1
    return ed.LlamaConfig(
        vocab_size=1024,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=28,
        num_attention_heads=24,
        num_key_value_heads=8,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        gradient_checkpointing=gradient_checkpointing,
        hidden_act="silu",
        scan_layers=False,
        # Pipeline-parallel controls
        pipeline_stage_regions=pipeline_stage_regions,
        pipeline_stage_layout=pipeline_stage_layout,
        pipeline_virtual_stages=pipeline_virtual_stages,
        # Mesh / sharding
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=_AXIS_NAMES,
        partition_axis=ed.PartitionAxis(
            batch_axis=("dp", "fsdp"),
            hidden_state_axis="tp",
            head_axis="tp",
            kv_head_axis="tp",
            sequence_axis="sp",
            expert_axis="ep",
        ),
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.bfloat16,
        attn_softmax_dtype=jnp.float32,
        use_sharding_constraint=False,
        backend=None,
        platform=None,
        lmhead_chunksize=lmhead_chunksize,
    )


def _make_small_config(
    sharding_axis_dims: tuple[int, ...],
    pipeline_stage_regions: bool,
    max_position_embeddings: int,
    *,
    scheduler_kind: str = "std1f1b",
    virtual_stages: int = 1,
    attn_mechanism: str = "vanilla",
    gradient_checkpointing: ed.EasyDeLGradientCheckPointers = ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
) -> ed.LlamaConfig:
    """Tiny proxy config for fast smoke tests."""
    if scheduler_kind == "kimik2":
        pipeline_stage_layout = "loop"
        pipeline_virtual_stages = max(1, virtual_stages)
    else:
        pipeline_stage_layout = "contiguous"
        pipeline_virtual_stages = 1
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=False,
        gradient_checkpointing=gradient_checkpointing,
        hidden_act="silu",
        scan_layers=False,
        pipeline_stage_regions=pipeline_stage_regions,
        pipeline_stage_layout=pipeline_stage_layout,
        pipeline_virtual_stages=pipeline_virtual_stages,
        sharding_axis_dims=sharding_axis_dims,
        sharding_axis_names=_AXIS_NAMES,
        partition_axis=ed.PartitionAxis(
            batch_axis=("dp", "fsdp"),
            hidden_state_axis="tp",
            head_axis="tp",
            kv_head_axis="tp",
            sequence_axis="sp",
            expert_axis="ep",
        ),
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.bfloat16,
        attn_softmax_dtype=jnp.float32,
        use_sharding_constraint=False,
        backend=None,
        platform=None,
    )


def _make_synthetic_dataset(
    *,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 42,
) -> Dataset:
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=np.int32)
    attention_mask = np.ones((num_samples, seq_len), dtype=np.int32)
    # Shift input_ids by one for causal LM labels (keep last token ignored).
    labels = np.roll(input_ids, shift=-1, axis=1)
    labels[:, -1] = -100
    return Dataset.from_dict(
        {
            "input_ids": [row for row in input_ids],
            "attention_mask": [row for row in attention_mask],
            "labels": [row for row in labels],
        }
    )


def _build_model(config: ed.LlamaConfig, dtype: jnp.dtype = jnp.float32) -> ed.EasyDeLBaseModule:
    with config.mesh:
        model = ed.AutoEasyDeLModelForCausalLM.from_config(
            config=config,
            dtype=dtype,
            param_dtype=dtype,
            precision=jax.lax.Precision.DEFAULT,
            rngs=spx.Rngs(42),
        )
        model = model.shard_model()
    return model


def _run_experiment(
    *,
    name: str,
    sharding_axis_dims: tuple[int, ...],
    schedule: spx.Schedule | None,
    microbatches: int,
    num_steps: int,
    seq_len: int,
    total_batch_size: int,
    dummy_3b: bool,
    dtype: jnp.dtype,
    scheduler_kind: str = "std1f1b",
    virtual_stages: int = 1,
    attn_mechanism: str = "vanilla",
    gradient_checkpointing: ed.EasyDeLGradientCheckPointers = ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    lmhead_chunksize: int | None = None,
    gradient_accumulation_steps: int = 1,
    stage_layout: str = "loop",
) -> _RunResult:
    """Build model + trainer and run a short SFT loop, collecting metrics."""
    logger.info("\n=== Starting run: %s ===", name)
    logger.info("sharding_axis_dims=%s schedule=%s", sharding_axis_dims, schedule)

    pipeline_stage_regions = sharding_axis_dims[0] > 1
    config_factory = _make_dummy_3b_config if dummy_3b else _make_small_config
    config = config_factory(
        sharding_axis_dims=sharding_axis_dims,
        pipeline_stage_regions=pipeline_stage_regions,
        max_position_embeddings=seq_len,
        scheduler_kind=scheduler_kind,
        virtual_stages=virtual_stages,
        attn_mechanism=attn_mechanism,
        gradient_checkpointing=gradient_checkpointing,
        lmhead_chunksize=lmhead_chunksize,
        stage_layout=stage_layout,
    )

    vocab_size = config.vocab_size
    dataset = _make_synthetic_dataset(
        num_samples=max(num_steps * total_batch_size * gradient_accumulation_steps, 64),
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=42,
    )
    tokenizer = _FakeTokenizer(vocab_size=vocab_size)

    args = ed.SFTConfig(
        model_name=f"dummy-sft-{name}",
        save_directory=f"tmp-files/sft-pp-tp-comparison/{name}",
        num_train_epochs=1,
        total_batch_size=total_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_training_steps=num_steps,
        log_steps=1,
        learning_rate=1e-5,
        max_length=seq_len,
        dataset_text_field=None,
        packing=False,
        loss_type="nll",
        mpmd_scheduler=schedule,
        use_wandb=False,
        do_last_save=False,
        save_optimizer_state=False,
        save_steps=None,
        progress_bar_type="json",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    rss_before = _memory_mb()
    sampler = _PeakMemorySampler()
    t0 = time.perf_counter()
    sampler.start()

    losses: list[float] = []
    step_times: list[float] = []
    _captured_records: list[dict[str, tp.Any]] = []

    try:
        model = _build_model(config, dtype=dtype)
        trainer = ed.SFTTrainer(
            arguments=args,
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        # Capture per-step metrics by wrapping the trainer's log_metrics hook.
        _original_log_metrics = trainer.log_metrics

        def _capturing_log_metrics(metrics, pbar, step, mode="train", **kwargs):
            if mode == "train":
                _captured_records.append(dict(metrics))
            return _original_log_metrics(metrics, pbar, step, mode, **kwargs)

        trainer.log_metrics = _capturing_log_metrics
        try:
            trainer.train()
        finally:
            trainer.log_metrics = _original_log_metrics
        compile_time = time.perf_counter() - t0
        peak_rss_mb = sampler.stop() - rss_before
    except Exception as exc:
        sampler.stop()
        logger.exception("Run %s failed", name)
        return _RunResult(
            name=name,
            sharding_axis_dims=sharding_axis_dims,
            schedule_name=type(schedule).__name__ if schedule is not None else None,
            losses=[],
            step_times=[],
            compile_time=0.0,
            peak_rss_mb=_memory_mb() - rss_before,
            error=f"{type(exc).__name__}: {exc}",
        )

    logger.info("Captured %d training metric records.", len(_captured_records))
    for record in _captured_records:
        loss_val = record.get("train/loss")
        if loss_val is None:
            loss_val = record.get("loss")
        if loss_val is not None:
            losses.append(float(loss_val))
        step_time = record.get("performance/train_step_time") or record.get("performance/execution_time")
        if step_time is not None:
            step_times.append(float(step_time))
    losses = losses[-num_steps:]
    step_times = step_times[-num_steps:]

    # Fallback: if the trainer didn't surface per-step metrics, time one extra
    # call by re-running a single step manually (only for the smoke-test path).
    if not losses and num_steps > 0:
        losses = [float("nan")] * num_steps
        step_times = [compile_time / max(num_steps, 1)] * num_steps

    return _RunResult(
        name=name,
        sharding_axis_dims=sharding_axis_dims,
        schedule_name=type(schedule).__name__ if schedule is not None else None,
        losses=losses,
        step_times=step_times,
        compile_time=compile_time,
        peak_rss_mb=peak_rss_mb,
        error=None,
    )


def _print_results(results: list[_RunResult]) -> None:
    print("\n" + "=" * 80)
    print("SFT PP=2/TP=2 vs TP=4 comparison results")
    print("=" * 80)
    for r in results:
        print(f"\nRun: {r.name}")
        print(f"  sharding_axis_dims: {r.sharding_axis_dims}")
        print(f"  schedule:           {r.schedule_name}")
        if r.error:
            print(f"  ERROR:              {r.error}")
            continue
        print(f"  compile+init time:  {r.compile_time:.2f}s")
        print(f"  peak RSS delta:     {r.peak_rss_mb:.1f} MB")
        print(f"  losses:             {r.losses}")
        print(f"  step times:         {[f'{t:.2f}s' for t in r.step_times]}")
        if r.step_times:
            print(f"  mean step time:     {sum(r.step_times) / len(r.step_times):.2f}s")
    print("\n" + "=" * 80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dummy-3b",
        action="store_true",
        help="Use the full dummy-3B Llama config (slow on CPU).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of training steps to run per configuration.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for synthetic data.",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=4,
        help="Global batch size (split into microbatches for PP).",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=4,
        help="Pipeline microbatch count for the PP=2/TP=2 run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("tmp-files/sft-pp-tp-comparison/results.json"),
        help="Where to write the JSON metrics file.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Compute/parameter dtype (bfloat16 may not work on CPU).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["std1f1b", "kimik2"],
        default="std1f1b",
        help="Pipeline scheduler to use for the PP=2/TP=2 run.",
    )
    parser.add_argument(
        "--virtual-stages",
        type=int,
        default=2,
        help="Virtual pipeline stages per physical stage (used by KimiK2).",
    )
    parser.add_argument(
        "--attn-mechanism",
        type=str,
        default="vanilla",
        help="Attention kernel (vanilla, flash_attn2, ring, etc.).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=[
            "nothing_saveable",
            "checkpoint_dots",
            "checkpoint_dots_with_no_batch_dims",
            "save_dot_with_no_batch_dims",
            "save_dot_except_mlp",
            "save_dot_except_mlp_wi",
            "save_vanilla",
            "save_vanilla_with_static",
            "save_mlp_full",
            "save_attention_full",
            "save_attention_output",
            "save_attention_query_key_value",
        ],
        default="nothing_saveable",
        help="Gradient checkpointing policy.",
    )
    parser.add_argument(
        "--lmhead-chunksize",
        type=int,
        default=None,
        help="Chunk size for the LM-head projection (sequence dimension).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = total_batch_size * grad_accum).",
    )
    parser.add_argument(
        "--stage-layout",
        choices=["loop", "contiguous"],
        default="loop",
        help="Virtual stage layout for KimiK2 (loop or contiguous).",
    )
    parser.add_argument(
        "--pp-degree",
        type=int,
        choices=[2, 4],
        default=2,
        help="Pipeline parallelism degree (2 -> PP=2/TP=2, 4 -> PP=4/TP=1).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    if args.dummy_3b and args.dtype == "bfloat16" and jax.default_backend() == "cpu":
        logger.warning("bfloat16 is not supported by XLA CPU; forcing float32.")
        dtype = jnp.float32

    _GC_NAME_MAP = {
        "nothing_saveable": "NOTHING_SAVEABLE",
        "checkpoint_dots": "CHECKPOINT_DOTS",
        "checkpoint_dots_with_no_batch_dims": "CHECKPOINT_DOTS_WITH_NO_BATCH_DMIS",
        "save_dot_with_no_batch_dims": "SAVE_DOT_WITH_NO_BATCH_DMIS",
        "save_dot_except_mlp": "SAVE_DOT_EXCEPT_MLP",
        "save_dot_except_mlp_wi": "SAVE_DOT_EXCEPT_MLP_WI",
        "save_vanilla": "SAVE_VANILLA",
        "save_vanilla_with_static": "SAVE_VANILLA_WITH_STATIC",
        "save_mlp_full": "SAVE_MLP_FULL",
        "save_attention_full": "SAVE_ATTENTION_FULL",
        "save_attention_output": "SAVE_ATTENTION_OUTPUT",
        "save_attention_query_key_value": "SAVE_ATTENTION_QUERY_KEY_VALUE",
    }
    gc_policy = getattr(ed.EasyDeLGradientCheckPointers, _GC_NAME_MAP[args.gradient_checkpointing])

    if args.scheduler == "kimik2":
        pp_schedule = spx.KimiK2(
            microbatches=args.microbatches,
            virtual_stages=args.virtual_stages,
            stage_layout="loop",
        )
    else:
        pp_schedule = spx.Std1F1B(microbatches=args.microbatches)

    # PP + TP
    if args.pp_degree == 4:
        pp_sharding = (4, 1, 1, 1, 1, 1)
        pp_name = "pp4_tp1"
    else:
        pp_sharding = (2, 1, 1, 1, 2, 1)
        pp_name = "pp2_tp2"
    pp_tp_result = _run_experiment(
        name=pp_name,
        sharding_axis_dims=pp_sharding,
        schedule=pp_schedule,
        microbatches=args.microbatches,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        total_batch_size=args.total_batch_size,
        dummy_3b=args.dummy_3b,
        dtype=dtype,
        scheduler_kind=args.scheduler,
        virtual_stages=args.virtual_stages,
        attn_mechanism=args.attn_mechanism,
        gradient_checkpointing=gc_policy,
        lmhead_chunksize=args.lmhead_chunksize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage_layout=args.stage_layout,
    )

    # Pure TP=4
    tp_only_result = _run_experiment(
        name="tp4",
        sharding_axis_dims=(1, 1, 1, 1, 4, 1),
        schedule=None,
        microbatches=1,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        total_batch_size=args.total_batch_size,
        dummy_3b=args.dummy_3b,
        dtype=dtype,
        scheduler_kind=args.scheduler,
        virtual_stages=args.virtual_stages,
        attn_mechanism=args.attn_mechanism,
        gradient_checkpointing=gc_policy,
        lmhead_chunksize=args.lmhead_chunksize,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage_layout=args.stage_layout,
    )

    results = [pp_tp_result, tp_only_result]
    _print_results(results)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "sharding_axis_dims": r.sharding_axis_dims,
                    "schedule_name": r.schedule_name,
                    "losses": r.losses,
                    "step_times": r.step_times,
                    "compile_time": r.compile_time,
                    "peak_rss_mb": r.peak_rss_mb,
                    "error": r.error,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote results to %s", args.output_json)
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
