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

"""Example of running an eSurge API server with OpenAI compatibility."""

import argparse

from easydel.inference.esurge import eSurge
from easydel.inference.esurge.config import (
    eSurgeCacheRuntimeConfig,
    eSurgeDistributedConfig,
    eSurgeRuntimeConfig,
)
from easydel.inference.esurge.server import eSurgeApiServer
from easydel.inference.openai_api_modules import FunctionCallFormat


def main():
    parser = argparse.ArgumentParser(description="Run eSurge API Server")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")
    parser.add_argument("--max-model-len", type=int, default=512, help="Maximum model length")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="Maximum number of sequences")
    parser.add_argument("--hbm-utilization", type=float, default=0.4, help="HBM utilization factor")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=11559, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--enable-function-calling", action="store_true", help="Enable function calling support")
    parser.add_argument("--require-api-key", action="store_true", help="Require API keys for every request")
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="Pre-provisioned API key (can be passed multiple times)",
    )
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    parser.add_argument(
        "--coordination",
        type=str,
        default="replicated",
        choices=["replicated", "zmq"],
        help="Multi-host step coordination ('zmq' for single-ingress serving on pods)",
    )
    parser.add_argument(
        "--distributed-service-name",
        type=str,
        default=None,
        help="DNS service name resolving all distributed hosts",
    )
    parser.add_argument(
        "--distributed-leader-addr",
        type=str,
        default=None,
        help="Leader host/IP for the coordination plane (defaults to the JAX coordinator host)",
    )
    parser.add_argument(
        "--distributed-control-port",
        type=int,
        default=19666,
        help="Distributed control-plane port",
    )
    parser.add_argument(
        "--distributed-auth-token",
        type=str,
        default=None,
        help="Shared auth token for distributed control-plane RPC",
    )

    args = parser.parse_args()

    print(f"Loading eSurge model: {args.model}")
    print("Configuration:")
    print(f"  Max model length: {args.max_model_len}")
    print(f"  Max sequences: {args.max_num_seqs}")
    print(f"  HBM utilization: {args.hbm_utilization}")
    print(f"  Function calling: {args.enable_function_calling}")

    # Initialize eSurge engine
    engine = eSurge(
        model=args.model,
        runtime=eSurgeRuntimeConfig.from_dict(
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            overlap_execution=False,
        ),
        cache=eSurgeCacheRuntimeConfig.from_dict(
            hbm_utilization=args.hbm_utilization,
            enable_prefix_caching=True,
        ),
        distributed=eSurgeDistributedConfig.from_dict(
            coordination=args.coordination,
            distributed_service_name=args.distributed_service_name,
            distributed_leader_addr=args.distributed_leader_addr,
            distributed_control_port=args.distributed_control_port,
            distributed_auth_token=args.distributed_auth_token,
        ),
    )

    if engine.distributed_role == "worker":
        raise RuntimeError("This script runs the public API server and must be launched on the leader rank.")

    print(f"\nStarting eSurge API server on http://{args.host}:{args.port}")
    print("OpenAI-compatible endpoints:")
    print(f"  - Chat: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - Completions: http://{args.host}:{args.port}/v1/completions")
    print(f"  - Models: http://{args.host}:{args.port}/v1/models")
    print(f"  - Health: http://{args.host}:{args.port}/health")
    print(f"  - Metrics: http://{args.host}:{args.port}/metrics")
    if args.enable_function_calling:
        print(f"  - Tools: http://{args.host}:{args.port}/v1/tools")
    print("\n" + "=" * 60)
    print("Server is starting...")
    print("=" * 60 + "\n")

    engine.start_monitoring()

    server = eSurgeApiServer(
        esurge_map={args.model: engine},
        oai_like_processor=True,
        enable_function_calling=args.enable_function_calling,
        default_function_format=FunctionCallFormat.OPENAI,
        require_api_key=args.require_api_key,
        api_keys=args.api_key,
    )

    if args.require_api_key and not args.api_key:
        generated_key = server.generate_api_key(label="cli")
        print(f"[API] Generated key for testing: {generated_key}")

    server.run(host=args.host, port=args.port, workers=args.workers, log_level=args.log_level)


if __name__ == "__main__":
    main()
