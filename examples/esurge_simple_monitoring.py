#!/usr/bin/env python3
"""Simple example of eSurge with built-in monitoring."""

import asyncio
import time


def basic_monitoring_example():
    """Basic monitoring example with eSurge."""
    print("🚀 eSurge Simple Monitoring Example")
    print("=" * 40)

    # Initialize eSurge engine
    print("🤖 Initializing eSurge engine...")

    # NOTE: Replace with your actual model
    # For this example, we'll show the API without loading a real model
    """
    engine = eSurge(
        model="microsoft/DialoGPT-medium",
        max_model_len=1024,
        max_num_seqs=8,
    )
    """

    # Simulated engine for demo purposes
    class MockeSurge:
        def start_monitoring(self, **kwargs):
            print("🚀 Starting eSurge monitoring services...")
            print("📊 Metrics collection initialized")
            print("📈 Prometheus metrics: http://localhost:8000/metrics")
            print("🌐 Web dashboard: http://localhost:8080")
            print("🩺 Health check: http://localhost:8080/health")
            print("\n✅ Monitoring services started successfully!")
            print("📊 Metrics will be automatically collected during inference")
            print("🌐 Open http://localhost:8080 to view real-time metrics")
            return {
                "dashboard": "http://localhost:8080",
                "prometheus": "http://localhost:8000/metrics",
                "health": "http://localhost:8080/health",
                "api": "http://localhost:8080/api/metrics",
            }

        def get_metrics_summary(self):
            return {
                "requests_per_second": 2.5,
                "average_latency": 0.15,
                "average_ttft": 0.03,
                "average_throughput": 45.2,
                "total_completed": 10,
                "total_failed": 0,
                "total_tokens": 452,
                "active_requests": 0,
                "queue_size": 0,
                "running_requests": 0,
            }

        def stop_monitoring(self):
            print("🛑 Stopping eSurge monitoring services...")
            print("📈 Prometheus server stopped")
            print("🌐 Dashboard server will stop with process")
            print("✅ Monitoring services stopped")

        @property
        def monitoring_active(self):
            return True

    engine = MockeSurge()

    print("✅ eSurge engine initialized")
    print()

    # Start monitoring with simple one-liner
    print("📊 Starting monitoring with default settings...")
    urls = engine.start_monitoring()

    print("\n📍 Monitoring URLs:")
    for service, url in urls.items():
        print(f"   • {service.title()}: {url}")

    print("\n🔄 Simulating some inference work...")

    # Simulate some work
    time.sleep(2)

    # Check monitoring status
    print(f"\n📊 Monitoring active: {engine.monitoring_active}")

    # Get metrics summary
    metrics = engine.get_metrics_summary()
    print("\n📈 Current metrics summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.2f}")
        else:
            print(f"   • {key}: {value}")

    print("\n⏳ Monitoring services running... Press Ctrl+C to stop")

    try:
        # Keep running to let user see the dashboard
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        engine.stop_monitoring()


def advanced_monitoring_example():
    """Advanced monitoring example with custom settings."""
    print("🚀 eSurge Advanced Monitoring Example")
    print("=" * 45)

    # NOTE: Replace with your actual model initialization
    """
    engine = eSurge(
        model="your-model-name",
        max_model_len=2048,
        max_num_seqs=16,
    )
    """

    class MockeSurge:
        def start_monitoring(self, **kwargs):
            print("🚀 Starting eSurge monitoring services...")

            # Show the configured options
            print(f"📊 Dashboard port: {kwargs.get('dashboard_port', 8080)}")
            print(f"📈 Prometheus port: {kwargs.get('prometheus_port', 8000)}")
            print(f"🌐 Host: {kwargs.get('dashboard_host', 'localhost')}")
            print(f"📝 Log file: {kwargs.get('log_file', 'None')}")
            print(f"⏱️ Log interval: {kwargs.get('log_interval', 10.0)}s")
            print(f"📊 Enable Prometheus: {kwargs.get('enable_prometheus', True)}")
            print(f"🌐 Enable Dashboard: {kwargs.get('enable_dashboard', True)}")
            print(f"🖥️ Enable Console: {kwargs.get('enable_console', False)}")

            host = kwargs.get("dashboard_host", "localhost")
            dash_port = kwargs.get("dashboard_port", 8080)
            prom_port = kwargs.get("prometheus_port", 8000)
            return {
                "dashboard": f"http://{host}:{dash_port}",
                "prometheus": f"http://{host}:{prom_port}/metrics",
                "health": f"http://{host}:{dash_port}/health",
            }

        def stop_monitoring(self):
            print("🛑 Monitoring stopped")

    engine = MockeSurge()

    print("🔧 Starting monitoring with custom configuration...")

    # Advanced monitoring configuration
    urls = engine.start_monitoring(
        dashboard_port=8090,  # Custom dashboard port
        prometheus_port=8010,  # Custom Prometheus port
        dashboard_host="0.0.0.0",  # Listen on all interfaces
        enable_prometheus=True,  # Enable Prometheus metrics
        enable_dashboard=True,  # Enable web dashboard
        enable_console=False,  # Disable console monitor
        log_file="my_esurge_metrics.log",  # Custom log file
        log_interval=5.0,  # Log every 5 seconds
        history_size=2000,  # Keep more history
        enable_detailed_logging=True,  # Detailed logs
    )

    print("\n📍 Custom monitoring URLs:")
    for service, url in urls.items():
        print(f"   • {service.title()}: {url}")

    print("\n✅ Advanced monitoring configuration complete!")
    print("🔧 Try different combinations of settings for your needs")

    # Stop monitoring
    engine.stop_monitoring()


async def production_monitoring_example():
    """Production-ready monitoring example."""
    print("🏭 eSurge Production Monitoring Example")
    print("=" * 45)

    # NOTE: This would be your actual production setup
    """
    engine = eSurge(
        model="your-production-model",
        max_model_len=4096,
        max_num_seqs=32,
        dtype=jnp.float16,  # Memory efficient
    )
    """

    class MockeSurge:
        def __init__(self):
            self._monitoring_active = False

        def start_monitoring(self, **kwargs):
            self._monitoring_active = True
            return {"dashboard": "http://localhost:8080", "prometheus": "http://localhost:8000/metrics"}

        def generate(self, prompts, sampling_params=None):
            # Simulate generation
            await asyncio.sleep(0.1)
            return [f"Generated response for: {prompt[:20]}..." for prompt in prompts]

        def get_metrics_summary(self):
            return {
                "requests_per_second": 15.3,
                "average_latency": 0.08,
                "average_throughput": 234.5,
                "total_completed": 1523,
                "total_failed": 3,
            }

        @property
        def monitoring_active(self):
            return self._monitoring_active

        def stop_monitoring(self):
            self._monitoring_active = False

    engine = MockeSurge()

    print("🚀 Production monitoring setup...")

    # Production monitoring configuration
    urls = engine.start_monitoring(
        dashboard_port=8080,
        prometheus_port=8000,
        dashboard_host="0.0.0.0",  # Accept external connections
        enable_prometheus=True,  # Essential for production
        enable_dashboard=True,  # Web monitoring
        enable_console=False,  # No console in production
        log_file="production_metrics.log",  # Persistent logging
        log_interval=30.0,  # Less frequent logging
        history_size=5000,  # More history for analysis
        enable_detailed_logging=False,  # Reduce log volume
    )

    print("✅ Production monitoring started")
    print(f"📊 Metrics: {urls['prometheus']}")
    print(f"🌐 Dashboard: {urls['dashboard']}")

    # Simulate production workload
    print("\n🔄 Simulating production workload...")

    # Example prompts for testing (not used in simulation)
    # test_prompts = [
    #     "What is machine learning?",
    #     "Explain quantum computing",
    #     "How do neural networks work?",
    #     "What is artificial intelligence?",
    # ]
    # sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    for i in range(3):
        print(f"🔄 Processing batch {i+1}/3...")

        # This would be real inference in production
        # results = engine.generate(test_prompts, sampling_params)
        await asyncio.sleep(0.5)  # Simulate work

        # Check metrics
        if engine.monitoring_active:
            metrics = engine.get_metrics_summary()
            print(
                f"   📊 RPS: {metrics['requests_per_second']:.1f}, "
                f"Throughput: {metrics['average_throughput']:.1f} tok/s"
            )

    print("\n✅ Production workload simulation complete")
    print(f"📈 Check {urls['dashboard']} for detailed metrics")

    engine.stop_monitoring()


def main():
    """Main example selector."""
    print("🚀 eSurge Monitoring Examples")
    print("=" * 35)
    print("Choose an example:")
    print("1. 📊 Basic monitoring (simple setup)")
    print("2. 🔧 Advanced monitoring (custom config)")
    print("3. 🏭 Production monitoring (async example)")
    print("4. ❌ Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        basic_monitoring_example()
    elif choice == "2":
        advanced_monitoring_example()
    elif choice == "3":
        asyncio.run(production_monitoring_example())
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Example interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
