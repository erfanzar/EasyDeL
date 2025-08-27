#!/usr/bin/env python3
"""Example demonstrating eSurge metrics collection and logging."""

import asyncio

from easydel.inference.esurge import initialize_metrics, log_metrics_summary


async def main():
    """Main example function demonstrating metrics usage."""

    # Initialize metrics collection with file logging
    metrics_collector = initialize_metrics(
        log_file="esurge_metrics.log",
        log_interval=5.0,  # Log summary every 5 seconds
        history_size=1000,
        enable_detailed_logging=True,
    )

    print("Initialized eSurge metrics collector")
    print("Logging to: esurge_metrics.log")

    # Initialize eSurge engine (you would use your actual model here)
    # For this example, we'll simulate the metrics without running actual inference
    # engine = eSurge(
    #     model="microsoft/DialoGPT-medium",
    #     max_model_len=1024,
    #     max_num_seqs=8,
    # )

    # Simulate some metrics collection
    print("\nSimulating metrics collection...")

    # Simulate request processing
    for i in range(10):
        request_id = f"req_{i}"
        prompt_tokens = 20 + i * 5

        # Start request
        metrics_collector.start_request(request_id, prompt_tokens)
        print(f"Started request {request_id}")

        # Simulate some processing time
        await asyncio.sleep(0.1)

        # Record first token
        metrics_collector.record_first_token(request_id)

        # Simulate generating tokens
        for _ in range(5):
            await asyncio.sleep(0.02)
            metrics_collector.add_generated_tokens(request_id, 1)

        # Complete request
        metrics_collector.complete_request(request_id, finish_reason="stop")
        print(f"Completed request {request_id}")

        # Record some scheduler and runner metrics
        metrics_collector.record_scheduler_metrics(
            num_waiting=max(0, 10 - i),
            num_running=min(i + 1, 8),
            num_scheduled_tokens=prompt_tokens + 5,
            batch_size=min(i + 1, 8),
            schedule_time=0.001,
        )

        metrics_collector.record_runner_metrics(
            execution_time=0.05,
            batch_size=min(i + 1, 8),
            num_tokens=prompt_tokens + 5,
        )

        # Periodic metrics logging
        if i % 3 == 0:
            log_metrics_summary()

    # Get system metrics summary
    system_metrics = metrics_collector.get_system_metrics(window_seconds=30.0)
    print("\nSystem Metrics Summary:")
    print(f"  Total completed requests: {system_metrics.total_requests_completed}")
    print(f"  Total failed requests: {system_metrics.total_requests_failed}")
    print(f"  Total tokens generated: {system_metrics.total_tokens_generated}")
    print(f"  Average latency: {system_metrics.average_latency:.3f}s")
    print(f"  Average TTFT: {system_metrics.average_ttft:.3f}s")
    print(f"  Average throughput: {system_metrics.average_throughput:.1f} tokens/s")
    print(f"  Requests per second: {system_metrics.requests_per_second:.1f}")

    # Export detailed metrics to JSON
    metrics_collector.export_metrics("detailed_metrics.json")
    print("\nDetailed metrics exported to: detailed_metrics.json")

    # Force a final summary log
    log_metrics_summary()

    print("\nMetrics collection example completed!")


if __name__ == "__main__":
    asyncio.run(main())
