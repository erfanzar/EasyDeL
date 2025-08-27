#!/usr/bin/env python3
"""Complete eSurge monitoring demonstration with all features."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from easydel.inference.esurge import initialize_metrics
from easydel.inference.esurge.dashboard import create_dashboard
from easydel.inference.esurge.monitoring import start_console_monitor, start_monitoring_server, stop_monitoring


def simulate_inference_workload():
    """Simulate an inference workload for demonstration."""
    print("🔄 Starting simulated inference workload...")

    # Initialize metrics
    metrics_collector = initialize_metrics(
        log_file="esurge_demo_metrics.log",
        log_interval=2.0,
        history_size=100,
        enable_detailed_logging=True,
    )

    # Simulate processing requests
    for i in range(50):
        request_id = f"demo_req_{i:03d}"
        prompt_tokens = 15 + (i % 20)  # Varying prompt sizes

        # Start request
        metrics_collector.start_request(request_id, prompt_tokens)

        # Simulate processing time
        processing_time = 0.05 + (i % 10) * 0.01  # Varying processing times
        time.sleep(processing_time)

        # Record first token
        metrics_collector.record_first_token(request_id)

        # Simulate token generation
        generated_tokens = 5 + (i % 15)  # Varying output lengths
        for _ in range(generated_tokens):
            time.sleep(0.002)  # Small delay per token
            metrics_collector.add_generated_tokens(request_id, 1)

        # Complete request
        finish_reason = "stop" if i % 20 != 19 else "length"
        error = None if i % 25 != 24 else "timeout"

        metrics_collector.complete_request(request_id, finish_reason=finish_reason, error=error)

        # Record scheduler metrics
        waiting_requests = max(0, 10 - (i // 5))
        running_requests = min(i + 1, 8)

        metrics_collector.record_scheduler_metrics(
            num_waiting=waiting_requests,
            num_running=running_requests,
            num_scheduled_tokens=prompt_tokens + generated_tokens,
            num_preempted=1 if i % 30 == 29 else 0,
            batch_size=running_requests,
            schedule_time=0.001 + (i % 5) * 0.0005,
        )

        # Record runner metrics
        metrics_collector.record_runner_metrics(
            execution_time=processing_time,
            batch_size=running_requests,
            num_tokens=prompt_tokens + generated_tokens,
        )

        # Record cache metrics
        total_pages = 1000
        used_pages = min(50 + i * 2, total_pages)
        cache_hit_rate = 0.75 + (i % 10) * 0.02

        metrics_collector.record_cache_metrics(
            total_pages=total_pages,
            used_pages=used_pages,
            cache_hit_rate=cache_hit_rate,
        )

        # Brief pause between requests
        time.sleep(0.1)

    print("✅ Simulated workload completed!")


async def run_monitoring_demo():
    """Run the complete monitoring demonstration."""
    print("🚀 eSurge Monitoring Demo")
    print("=" * 50)

    # Start Prometheus monitoring server
    print("📊 Starting Prometheus metrics server...")
    start_monitoring_server(prometheus_port=8000, update_interval=1.0)

    # Start web dashboard in background
    print("🌐 Starting web dashboard...")
    dashboard = create_dashboard(host="localhost", port=8080)

    def run_dashboard():
        dashboard.run(log_level="warning")

    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    # Wait a moment for servers to start
    await asyncio.sleep(2)

    print("\n📍 Services started:")
    print("   • Prometheus metrics: http://localhost:8000/metrics")
    print("   • Web dashboard: http://localhost:8080")
    print("   • Console monitor: Starting below...")
    print()

    # Start workload simulation in background
    with ThreadPoolExecutor() as executor:
        workload_future = executor.submit(simulate_inference_workload)

        # Start console monitor (this will block until Ctrl+C)
        try:
            print("🖥️  Starting console monitor (Press Ctrl+C to stop)...")
            start_console_monitor(refresh_rate=1.0)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring demo...")

        # Wait for workload to complete
        workload_future.result()

    # Stop all monitoring
    stop_monitoring()
    print("✅ Demo completed!")


def run_console_only_demo():
    """Run console-only monitoring demo."""
    print("🖥️  eSurge Console Monitor Demo")
    print("=" * 40)

    # Initialize metrics
    initialize_metrics(
        log_file="console_demo_metrics.log",
        log_interval=1.0,
        enable_detailed_logging=True,
    )

    # Start workload in background
    with ThreadPoolExecutor() as executor:
        executor.submit(simulate_inference_workload)

        # Start console monitor
        try:
            start_console_monitor(refresh_rate=0.5)
        except KeyboardInterrupt:
            print("\n🛑 Console demo stopped!")


def run_prometheus_only_demo():
    """Run Prometheus-only monitoring demo."""
    print("📊 eSurge Prometheus Demo")
    print("=" * 30)

    # Initialize metrics
    initialize_metrics(
        log_file="prometheus_demo_metrics.log",
        log_interval=1.0,
        enable_detailed_logging=True,
    )

    # Start Prometheus server
    start_monitoring_server(prometheus_port=8000)

    print("📍 Prometheus metrics available at: http://localhost:8000/metrics")
    print("🔄 Running workload simulation...")

    # Run workload
    simulate_inference_workload()

    print("✅ Prometheus demo completed!")
    print("📊 Check metrics at: http://localhost:8000/metrics")


async def run_web_dashboard_demo():
    """Run web dashboard only demo."""
    print("🌐 eSurge Web Dashboard Demo")
    print("=" * 35)

    # Initialize metrics
    initialize_metrics(
        log_file="dashboard_demo_metrics.log",
        log_interval=1.0,
        enable_detailed_logging=True,
    )

    # Start dashboard
    dashboard = create_dashboard(host="localhost", port=8080)

    print("🌐 Web dashboard starting at: http://localhost:8080")
    print("🔄 Workload will start in 3 seconds...")

    # Start workload in background after brief delay
    async def delayed_workload():
        await asyncio.sleep(3)
        with ThreadPoolExecutor() as executor:
            executor.submit(simulate_inference_workload)

    # Run dashboard and workload concurrently
    await asyncio.gather(asyncio.create_task(delayed_workload()), asyncio.to_thread(dashboard.run, log_level="warning"))


def main():
    """Main demo function with menu."""
    print("🚀 eSurge Monitoring System Demo")
    print("=" * 40)
    print("Choose a demo to run:")
    print("1. 🎯 Complete demo (Prometheus + Web Dashboard + Console)")
    print("2. 🖥️  Console monitor only")
    print("3. 📊 Prometheus metrics only")
    print("4. 🌐 Web dashboard only")
    print("5. ❌ Exit")

    while True:
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            print("\n🎯 Running complete monitoring demo...")
            asyncio.run(run_monitoring_demo())
            break
        elif choice == "2":
            print("\n🖥️  Running console monitor demo...")
            run_console_only_demo()
            break
        elif choice == "3":
            print("\n📊 Running Prometheus demo...")
            run_prometheus_only_demo()
            break
        elif choice == "4":
            print("\n🌐 Running web dashboard demo...")
            asyncio.run(run_web_dashboard_demo())
            break
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback

        traceback.print_exc()
