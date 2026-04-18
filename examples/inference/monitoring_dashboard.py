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

"""eSurge live monitoring with Prometheus + Grafana.

Shows how to enable the built-in monitoring stack so you can observe
engine TPS, TTFT, active requests, cache page utilisation, batch size,
and more — all in real time through a Grafana dashboard.

Prerequisites
-------------
Before running this example you **must** install Prometheus and Grafana.
EasyDeL ships a helper script that does both in one shot:

    sudo bash scripts/install_grafana.sh --start

This installs the ``prometheus`` binary and the ``grafana-server`` package,
then starts Grafana on http://localhost:3000 (default login: admin / admin).

Python dependencies (``prometheus-client`` and ``rich``) are also required:

    pip install prometheus-client rich

How it works
------------
1. ``prometheus_client`` exposes engine metrics on an HTTP endpoint (default
   port 11184).
2. eSurge auto-starts a **Prometheus server** (port 9090) that scrapes
   that endpoint every 2 s.
3. eSurge detects the running Grafana instance and provisions an
   "eSurge Prometheus" data source + an "eSurge Engine Overview"
   dashboard via the Grafana HTTP API.
4. Open http://localhost:3000 → Dashboards → **eSurge Engine Overview**.

The dashboard panels include:
- Tokens / sec (stat + time-series)
- Running / Waiting requests
- Batch size
- Request latency (p50 / p99)
- Time to first token (p50 / p99)
- Schedule duration
- Cache pages (total, used, hit rate)
- Model execution time
- Token generation rate

All panels auto-refresh every 5 s.
"""

from __future__ import annotations

import easydel as ed


def main():
    model_id = "Qwen/Qwen3-8B"
    max_length = 8192
    max_concurrent_decodes = 32

    elm = ed.eLargeModel(
        {
            "model": {
                "name_or_path": model_id,
                "tokenizer": model_id,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
                "verbose": True,
            },
            "sharding": {
                "axis_dims": (1, 1, 1, -1, 1),
                "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            "base_config": {
                "values": {
                    "freq_max_position_embeddings": max_length,
                    "mask_max_position_embeddings": max_length,
                    "attn_mechanism": ed.AttentionMechanisms.AUTO,
                    "attn_dtype": "bf16",
                    "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NONE,
                    "moe_method": ed.MoEMethods.FUSED_MOE,
                }
            },
            "esurge": {
                "max_model_len": max_length,
                "max_num_seqs": max_concurrent_decodes,
                "min_input_pad": 8,
                "hbm_utilization": 0.9,
                "page_size": 128,
                "enable_prefix_caching": True,
                "verbose": True,
                "max_num_batched_tokens": 2048,
                "use_aot_forward": True,
                "data_parallelism_axis": "fsdp",
                "runner_verbose": False,
            },
        }
    )

    esurge = elm.build_esurge()

    # ── Start monitoring ──
    # This single call:
    #   1. Starts the prometheus_client exporter   (port 11184)
    #   2. Starts a Prometheus server              (port 9090)
    #   3. Provisions Grafana datasource+dashboard (port 3000)
    #   4. Optionally starts a Rich console monitor
    esurge.start_monitoring(
        enable_prometheus=True,  # expose metrics for scraping
        enable_console=False,  # rich terminal dashboard (non-blocking)
        start_grafana=True,  # auto-provision Grafana
    )

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
        "Summarize the plot of Romeo and Juliet.",
    ]

    for prompt in prompts:
        print(f"\n>> {prompt[:60]}...")
        for out in esurge.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=ed.SamplingParams(max_tokens=512),
            stream=True,
        ):
            if out.delta_text is not None:
                print(out.delta_text, end="")
        print(f"\n   TPS: {out.tokens_per_second:.1f}")

    # ── Print a metrics snapshot ──
    summary = esurge.get_metrics_summary()
    print("\n--- Metrics Summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Monitoring stays active until the engine is terminated.
    # To stop monitoring explicitly:
    #   esurge.stop_monitoring()
    esurge.terminate()


if __name__ == "__main__":
    main()
