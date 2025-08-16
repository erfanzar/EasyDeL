"""Example of running an eSurge API server with OpenAI compatibility."""

import argparse

from easydel.inference.esurge import eSurge
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
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")

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
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        hbm_utilization=args.hbm_utilization,
        enable_prefix_caching=True,
    )

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
    )

    server.run(host=args.host, port=args.port, workers=args.workers, log_level=args.log_level)


if __name__ == "__main__":
    main()
